"""Surrogate model tests (LayoutSurrogate, SurrogateTrainer, SurrogateManager)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import numpy as np

from core import Layout, Position, Shortcut
from evolution import PermutationSampling, StructuralGenomeSanitizer, SwapMutation
from evolution.acceptance import (
    _dynamic_mouse_layer_report,
    _layer7_access_report,
)
from evolution.custom_ga import _select_feasibility_first_scalar
from evolution.surrogate import LayoutSurrogate, SurrogateManager, SurrogateTrainer


class TestSurrogate(unittest.TestCase):
    def test_forward(self):
        surrogate = LayoutSurrogate(n_positions=10, n_shortcuts=20, n_factors=3, hidden_dim=32)
        layouts = np.full((5, 10), -1, dtype=np.int32)
        layouts[:, :5] = np.arange(5)

        import torch
        x = torch.tensor(layouts, dtype=torch.long)
        out = surrogate(x)
        self.assertEqual(out.shape, (5, 3))

    def test_train_predict(self):
        surrogate = LayoutSurrogate(n_positions=10, n_shortcuts=20, n_factors=3, hidden_dim=32)
        trainer = SurrogateTrainer(surrogate, device="cpu")

        layouts = np.full((50, 10), -1, dtype=np.int32)
        layouts[:, :5] = np.random.randint(0, 20, size=(50, 5))
        scores = np.random.randn(50, 3).astype(np.float32)

        trainer.train(layouts, scores, epochs=10, batch_size=16)
        pred = trainer.predict(layouts[:10])
        self.assertEqual(pred.shape, (10, 3))

    def test_forward_with_constraints(self):
        surrogate = LayoutSurrogate(
            n_positions=10, n_shortcuts=20, n_factors=3, n_constraints=4, hidden_dim=32
        )
        layouts = np.full((5, 10), -1, dtype=np.int32)
        layouts[:, :5] = np.arange(5)

        import torch
        x = torch.tensor(layouts, dtype=torch.long)
        out = surrogate(x)
        self.assertEqual(out.shape, (5, 7))

    def test_train_predict_with_constraints(self):
        surrogate = LayoutSurrogate(
            n_positions=10, n_shortcuts=20, n_factors=3, n_constraints=4, hidden_dim=32
        )
        trainer = SurrogateTrainer(surrogate, device="cpu")

        layouts = np.full((50, 10), -1, dtype=np.int32)
        layouts[:, :5] = np.random.randint(0, 20, size=(50, 5))
        objectives = np.random.randn(50, 3).astype(np.float32)
        constraints = np.random.rand(50, 4).astype(np.float32)
        combined = np.concatenate([objectives, constraints], axis=1)

        trainer.train(layouts, combined, epochs=10, batch_size=16)
        pred = trainer.predict(layouts[:10])
        self.assertEqual(pred.shape, (10, 7))
        pred_obj = pred[:, :3]
        pred_cv = pred[:, 3:]
        self.assertEqual(pred_obj.shape, (10, 3))
        self.assertEqual(pred_cv.shape, (10, 4))

    def test_surrogate_manager_combines_constraints(self):
        surrogate = LayoutSurrogate(
            n_positions=10, n_shortcuts=20, n_factors=3, n_constraints=4, hidden_dim=32
        )
        trainer = SurrogateTrainer(surrogate, device="cpu")
        manager = SurrogateManager(surrogate, trainer)

        layouts = np.full((20, 10), -1, dtype=np.int32)
        layouts[:, :5] = np.random.randint(0, 20, size=(20, 5))
        objectives = np.random.randn(20, 3).astype(np.float32)
        constraints = np.random.rand(20, 4).astype(np.float32)

        manager.add_exact_evaluations(layouts, objectives, constraints)
        self.assertEqual(len(manager.exact_cache), 20)
        cached_scores = np.array([entry[1] for entry in manager.exact_cache])
        self.assertEqual(cached_scores.shape, (20, 7))

    def test_feasibility_first_selection(self):
        scalar = np.array([1.0, 0.5, 2.0, 0.1, 3.0], dtype=np.float32)
        cv = np.array([
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)
        survivors = _select_feasibility_first_scalar(scalar, cv, 3)
        # Feasible indices are 1, 2, 4. Top 3 must be exactly those.
        self.assertEqual(set(survivors.tolist()), {1, 2, 4})

    def test_sampling_excludes_frozen_assigned_shortcuts(self):
        seed = np.array([0, -1, 1, -1], dtype=np.int32)
        frozen = np.array([True, False, True, False])

        class Problem:
            n_var = 4

        sampler = PermutationSampling(
            n_shortcuts=4,
            frozen_mask=frozen,
            seed_genome=seed,
            inject_seed=True,
        )
        X = sampler._do(Problem(), 20)
        mutable = np.where(~frozen)[0]
        self.assertFalse(np.any(np.isin(X[:, mutable], [0, 1])))

        dirty = np.array([[0, 0, 1, 1]], dtype=np.int32)
        sanitizer = StructuralGenomeSanitizer(n_shortcuts=4, frozen_mask=frozen, seed_genome=seed)
        cleaned = sanitizer._do(Problem(), dirty.copy())
        self.assertEqual(cleaned[0, 0], 0)
        self.assertEqual(cleaned[0, 2], 1)
        self.assertEqual(cleaned[0, 1], -1)
        self.assertEqual(cleaned[0, 3], -1)

    def test_group_mutation_moves_arrow_group_as_unit(self):
        random.seed(100)
        positions = tuple(
            Position(i, 1, float(i % 6), float(i // 6), "left", 1, 1.0)
            for i in range(12)
        )
        shortcuts = tuple([
            Shortcut(0, "LeftArrow", "Left", "Nav", 1.0, base_key="LeftArrow"),
            Shortcut(1, "UpArrow", "Up", "Nav", 1.0, base_key="UpArrow"),
            Shortcut(2, "DownArrow", "Down", "Nav", 1.0, base_key="DownArrow"),
            Shortcut(3, "RightArrow", "Right", "Nav", 1.0, base_key="RightArrow"),
            *[Shortcut(i, f"K{i}", "", "App", 1.0) for i in range(4, 12)],
        ])
        genome = np.arange(12, dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(12, dtype=np.bool_))
        mutation = SwapMutation(prob=0.0, frozen_mask=layout.frozen_mask, layout=layout, group_move_prob=1.0)
        X = mutation._do(None, genome.reshape(1, -1).copy())
        moved = X[0]
        arrow_positions = [int(np.where(moved == sid)[0][0]) for sid in range(4)]
        self.assertEqual(len(set(arrow_positions)), 4)
        self.assertNotEqual(set(arrow_positions), {0, 1, 2, 3})
        self.assertTrue(set(arrow_positions).isdisjoint({0, 1, 2, 3}))
        self.assertFalse(any(int(moved[pos]) in {0, 1, 2, 3} for pos in range(4)))

    def test_group_mutation_moves_completion_group_as_unit(self):
        random.seed(200)
        # 5x4 grid (x=0..4, y=0..3), layer 1 — 20 positions.
        # Completion sids 0-4 start at positions 10-14 (y=2, x=0-4).
        # Valid anchors at y=0 (ax=2,3,4) are positions like {0,1,2,5,15}
        # which don't overlap {10,11,12,13,14}, so the mutation can move the group.
        positions = tuple(
            Position(i, 1, float(i % 5), float(i // 5), "left", 1, 1.0)
            for i in range(20)
        )
        bases = [
            "Dash and Underscore", "Equals and Plus", "Grave Accent and Tilde",
            "Right Brace", "Backslash and Pipe",
        ]
        shortcuts = tuple([
            *(Shortcut(i, bases[i], "", "Raw", 1.0, base_key=bases[i]) for i in range(5)),
            *[Shortcut(i, f"K{i}", "", "App", 1.0) for i in range(5, 20)],
        ])
        # Place completion sids 0-4 at positions 10-14; other sids at 0-9 and 15-19.
        genome = np.array(list(range(5, 15)) + list(range(5)) + list(range(15, 20)), dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(20, dtype=np.bool_))
        mutation = SwapMutation(prob=0.0, frozen_mask=layout.frozen_mask, layout=layout, group_move_prob=1.0)
        initial_completion = set(int(np.where(genome == sid)[0][0]) for sid in range(5))
        moved_any = False
        for _ in range(30):
            X = mutation._do(None, genome.reshape(1, -1).copy())
            moved = X[0]
            completion_positions = set(int(np.where(moved == sid)[0][0]) for sid in range(5))
            self.assertEqual(len(completion_positions), 5)
            if completion_positions != initial_completion:
                moved_any = True
                self.assertTrue(completion_positions.isdisjoint(initial_completion))
                self.assertFalse(any(int(moved[pos]) in set(range(5)) for pos in initial_completion))
        self.assertTrue(moved_any)

    def test_group_members_never_moved_by_individual_mutations(self):
        """Individual swap/random_assign/bulk_assign must not scatter group members."""
        random.seed(203)
        positions = tuple(
            Position(i, 1, float(i % 5), float(i // 5), "left", 1, 1.0)
            for i in range(20)
        )
        bases = [
            "Dash and Underscore", "Equals and Plus", "Grave Accent and Tilde",
            "Right Brace", "Backslash and Pipe",
        ]
        shortcuts = tuple([
            *(Shortcut(i, bases[i], "", "Raw", 1.0, base_key=bases[i]) for i in range(5)),
            *[Shortcut(i, f"K{i}", "", "App", 1.0) for i in range(5, 20)],
        ])
        genome = np.arange(20, dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(20, dtype=np.bool_))
        mutation = SwapMutation(
            prob=1.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            group_overwrite_prob=0.0,   # disable group-move
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            random_assign_prob=0.5,
            bulk_assign_prob=0.5,
            optional_arrow_drop_prob=0.0,
        )
        # Not constructor kwargs (matches other hardcoded-default operators);
        # disable directly so this individual-mutation invariants check isn't
        # affected by the raw-completion-cluster proposal, which is a
        # sanctioned coordinated group move in its own right (like
        # group_overwrite_prob), not one of the individual mutations under test here.
        mutation.raw_completion_cluster_prob = 0.0
        mutation.arrow_cluster_prob = 0.0
        group_sids = set(range(5))
        for _ in range(100):
            X = mutation._do(None, genome.reshape(1, -1).copy())
            moved = X[0]
            # Group sids must still be at their original positions (0-4)
            for sid in group_sids:
                positions_of_sid = set(int(idx) for idx in np.where(moved == sid)[0])
                self.assertTrue(positions_of_sid.issubset({0, 1, 2, 3, 4}),
                    f"sid {sid} moved to unexpected positions {positions_of_sid}")

    def test_mouse_workflow_mutation_proposes_acceptance_visible_layer(self):
        random.seed(300)
        positions = tuple([
            Position(0, 0, 3.0, 4.0, "left", 0, 0.1, is_thumb=True),
            Position(1, 0, 4.0, 4.0, "left", 0, 0.1, is_thumb=True),
            Position(2, 0, 5.0, 4.0, "left", 0, 0.1, is_thumb=True),
            Position(3, 0, 0.0, 0.0, "left", 1, 1.0),
            *(Position(i, 1, float(7 + (i - 4)), 1.0, "right", 1, 0.2) for i in range(4, 10)),
            *(Position(i, 1, float(i - 10), 2.0, "left", 1, 1.0) for i in range(10, 14)),
        ])
        shortcuts = tuple([
            Shortcut(0, "MB1", "Left click", "Mouse", 10.0, category="mouse", base_key="MB1"),
            Shortcut(1, "MB2", "Right click", "Mouse", 9.0, category="mouse", base_key="MB2"),
            Shortcut(2, "MB3", "Middle click", "Mouse", 7.0, category="mouse", base_key="MB3"),
            Shortcut(3, "MB4", "Back", "Mouse", 6.0, category="mouse", base_key="MB4"),
            Shortcut(4, "MB5", "Forward", "Mouse", 6.0, category="mouse", base_key="MB5"),
            Shortcut(5, "@scroll:L1:hold", "Scroll Mode Layer 1", "Layer Access", 15.0, category="layer_access", base_key="Scroll_L1", is_layer_access=True, access_target_layer=1, access_is_momentary=True),
            Shortcut(6, "@access:L1:hold", "Momentary Layer 1", "Layer Access", 12.0, category="layer_access", base_key="L1", is_layer_access=True, access_target_layer=1, access_is_momentary=True),
            Shortcut(7, "@access:L1:toggle", "Toggle Layer 1", "Layer Access", 12.0, category="layer_access", base_key="L1", is_layer_access=True, access_target_layer=1, access_is_momentary=False),
            Shortcut(8, "@access:L7:hold", "Momentary Layer 7", "Layer Access", 12.0, category="layer_access", base_key="L7", is_layer_access=True, access_target_layer=7, access_is_momentary=True),
            Shortcut(9, "@access:L7:toggle", "Toggle Layer 7", "Layer Access", 12.0, category="layer_access", base_key="L7", is_layer_access=True, access_target_layer=7, access_is_momentary=False),
            *(Shortcut(i, f"K{i}", "", "App", 1.0, base_key=f"K{i}") for i in range(10, 14)),
        ])
        genome = np.arange(14, dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(14, dtype=np.bool_))
        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            group_overwrite_prob=0.0,
            mouse_workflow_prob=1.0,
            l7_access_prob=0.0,
        )
        moved = mutation._do(None, genome.reshape(1, -1).copy())[0]
        report = _dynamic_mouse_layer_report(layout.clone_with(genome=moved))
        self.assertTrue(report["acceptance_pass"], report)

    def test_mouse_workflow_mutation_assigns_by_effort_priority(self):
        """Freshly-proposed mouse layer must put Scroll/MB2/MB1 on the
        lowest-effort candidate slots, not an arbitrary position-index order."""
        import evolution
        random.seed(301)
        positions = tuple([
            Position(0, 0, 3.0, 4.0, "left", 0, 0.1, is_thumb=True),
            Position(1, 0, 4.0, 4.0, "left", 0, 0.1, is_thumb=True),
            Position(2, 0, 5.0, 4.0, "left", 0, 0.1, is_thumb=True),
            Position(3, 0, 0.0, 0.0, "left", 1, 1.0),
            # 6 right-non-thumb candidate slots with distinct, deliberately
            # scrambled efforts (position index order != effort order). x=7/8
            # are deliberately given the two highest efforts so they're never
            # "best" and don't collide with the separate x7/x8 scroll-comfort
            # rule under test elsewhere.
            Position(4, 1, 7.0, 1.0, "right", 1, 1.75),
            Position(5, 1, 9.0, 1.0, "right", 1, 0.0),
            Position(6, 1, 8.0, 1.0, "right", 1, 1.0),
            Position(7, 1, 10.0, 1.0, "right", 1, 1.25),
            Position(8, 1, 11.0, 1.0, "right", 1, 0.5),
            Position(9, 1, 12.0, 1.0, "right", 1, 0.75),
            *(Position(i, 1, float(i - 10), 2.0, "left", 1, 1.0) for i in range(10, 14)),
        ])
        shortcuts = tuple([
            Shortcut(0, "MB1", "Left click", "Mouse", 10.0, category="mouse", base_key="MB1"),
            Shortcut(1, "MB2", "Right click", "Mouse", 9.0, category="mouse", base_key="MB2"),
            Shortcut(2, "MB3", "Middle click", "Mouse", 7.0, category="mouse", base_key="MB3"),
            Shortcut(3, "MB4", "Back", "Mouse", 6.0, category="mouse", base_key="MB4"),
            Shortcut(4, "MB5", "Forward", "Mouse", 6.0, category="mouse", base_key="MB5"),
            Shortcut(5, "@scroll:L1:hold", "Scroll Mode Layer 1", "Layer Access", 15.0, category="layer_access", base_key="Scroll_L1", is_layer_access=True, access_target_layer=1, access_is_momentary=True),
            Shortcut(6, "@access:L1:hold", "Momentary Layer 1", "Layer Access", 12.0, category="layer_access", base_key="L1", is_layer_access=True, access_target_layer=1, access_is_momentary=True),
            Shortcut(7, "@access:L1:toggle", "Toggle Layer 1", "Layer Access", 12.0, category="layer_access", base_key="L1", is_layer_access=True, access_target_layer=1, access_is_momentary=False),
            Shortcut(8, "@access:L7:hold", "Momentary Layer 7", "Layer Access", 12.0, category="layer_access", base_key="L7", is_layer_access=True, access_target_layer=7, access_is_momentary=True),
            Shortcut(9, "@access:L7:toggle", "Toggle Layer 7", "Layer Access", 12.0, category="layer_access", base_key="L7", is_layer_access=True, access_target_layer=7, access_is_momentary=False),
            *(Shortcut(i, f"K{i}", "", "App", 1.0, base_key=f"K{i}") for i in range(10, 14)),
        ])
        genome = np.arange(14, dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(14, dtype=np.bool_))

        for use_numba in (True, False):
            orig = evolution.NUMBA_AVAILABLE
            evolution.NUMBA_AVAILABLE = use_numba
            try:
                mutation = SwapMutation(
                    prob=0.0,
                    frozen_mask=layout.frozen_mask,
                    layout=layout,
                    group_overwrite_prob=0.0,
                    mouse_workflow_prob=1.0,
                    l7_access_prob=0.0,
                )
                moved = mutation._do(None, genome.reshape(1, -1).copy())[0]
            finally:
                evolution.NUMBA_AVAILABLE = orig

            pos_of = {sid: int(idx) for idx, sid in enumerate(moved) if 0 <= sid < 6}
            effort_of = {i: positions[i].effort for i in range(4, 10)}
            # Position 5 (effort 0.0) is the single lowest-effort slot: Scroll (sid 5)
            # must win it, matching Scroll's dominant effort-priority weight.
            self.assertEqual(pos_of.get(5), 5, f"[numba={use_numba}] Scroll should occupy the lowest-effort slot")
            # MB2 (sid 1) must land on a lower-effort slot than MB1 (sid 0).
            self.assertLess(
                effort_of[pos_of[1]], effort_of[pos_of[0]],
                f"[numba={use_numba}] MB2 should have lower effort than MB1",
            )
            # MB1 must land on a lower-effort slot than MB3/MB4/MB5.
            for lower_sid in (2, 3, 4):
                self.assertLess(
                    effort_of[pos_of[0]], effort_of[pos_of[lower_sid]],
                    f"[numba={use_numba}] MB1 should have lower effort than sid {lower_sid}",
                )

    def test_mouse_workflow_mutation_never_proposes_scroll_on_uncomfortable_x(self):
        """Regression test: when the lowest-effort candidate slot happens to
        be at x=7 or x=8 (uncomfortable for momentary Scroll per policy),
        Scroll must be moved to a different slot instead of landing there —
        this is a final-acceptance-reject condition, not a soft preference."""
        import evolution
        random.seed(301)
        positions = tuple([
            Position(0, 0, 3.0, 4.0, "left", 0, 0.1, is_thumb=True),
            Position(1, 0, 4.0, 4.0, "left", 0, 0.1, is_thumb=True),
            Position(2, 0, 5.0, 4.0, "left", 0, 0.1, is_thumb=True),
            Position(3, 0, 0.0, 0.0, "left", 1, 1.0),
            # The single lowest-effort slot (0.0) sits at x=8, deliberately
            # colliding with the uncomfortable-x rule.
            Position(4, 1, 7.0, 1.0, "right", 1, 1.75),
            Position(5, 1, 8.0, 1.0, "right", 1, 0.0),
            Position(6, 1, 9.0, 1.0, "right", 1, 1.0),
            Position(7, 1, 10.0, 1.0, "right", 1, 1.25),
            Position(8, 1, 11.0, 1.0, "right", 1, 0.5),
            Position(9, 1, 12.0, 1.0, "right", 1, 0.75),
            *(Position(i, 1, float(i - 10), 2.0, "left", 1, 1.0) for i in range(10, 14)),
        ])
        shortcuts = tuple([
            Shortcut(0, "MB1", "Left click", "Mouse", 10.0, category="mouse", base_key="MB1"),
            Shortcut(1, "MB2", "Right click", "Mouse", 9.0, category="mouse", base_key="MB2"),
            Shortcut(2, "MB3", "Middle click", "Mouse", 7.0, category="mouse", base_key="MB3"),
            Shortcut(3, "MB4", "Back", "Mouse", 6.0, category="mouse", base_key="MB4"),
            Shortcut(4, "MB5", "Forward", "Mouse", 6.0, category="mouse", base_key="MB5"),
            Shortcut(5, "@scroll:L1:hold", "Scroll Mode Layer 1", "Layer Access", 15.0, category="layer_access", base_key="Scroll_L1", is_layer_access=True, access_target_layer=1, access_is_momentary=True),
            Shortcut(6, "@access:L1:hold", "Momentary Layer 1", "Layer Access", 12.0, category="layer_access", base_key="L1", is_layer_access=True, access_target_layer=1, access_is_momentary=True),
            Shortcut(7, "@access:L1:toggle", "Toggle Layer 1", "Layer Access", 12.0, category="layer_access", base_key="L1", is_layer_access=True, access_target_layer=1, access_is_momentary=False),
            Shortcut(8, "@access:L7:hold", "Momentary Layer 7", "Layer Access", 12.0, category="layer_access", base_key="L7", is_layer_access=True, access_target_layer=7, access_is_momentary=True),
            Shortcut(9, "@access:L7:toggle", "Toggle Layer 7", "Layer Access", 12.0, category="layer_access", base_key="L7", is_layer_access=True, access_target_layer=7, access_is_momentary=False),
            *(Shortcut(i, f"K{i}", "", "App", 1.0, base_key=f"K{i}") for i in range(10, 14)),
        ])
        genome = np.arange(14, dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(14, dtype=np.bool_))

        for use_numba in (True, False):
            orig = evolution.NUMBA_AVAILABLE
            evolution.NUMBA_AVAILABLE = use_numba
            try:
                mutation = SwapMutation(
                    prob=0.0,
                    frozen_mask=layout.frozen_mask,
                    layout=layout,
                    group_overwrite_prob=0.0,
                    mouse_workflow_prob=1.0,
                    l7_access_prob=0.0,
                )
                moved = mutation._do(None, genome.reshape(1, -1).copy())[0]
            finally:
                evolution.NUMBA_AVAILABLE = orig

            scroll_pos = int(np.where(moved == 5)[0][0])
            scroll_x = positions[scroll_pos].x
            self.assertNotIn(
                scroll_x, (7.0, 8.0),
                f"[numba={use_numba}] Scroll must never be proposed at x=7/8 (uncomfortable, "
                f"final-acceptance reject), landed at x={scroll_x}",
            )

    def test_l7_access_mutation_proposes_hold_and_toggle(self):
        random.seed(400)
        positions = tuple(
            Position(i, 0 if i < 4 else 1, float(i), 0.0, "left", 0 if i < 4 else 1, 0.5, is_thumb=i < 4)
            for i in range(8)
        )
        shortcuts = tuple([
            Shortcut(0, "@access:L7:hold", "Momentary Layer 7", "Layer Access", 12.0, category="layer_access", base_key="L7", is_layer_access=True, access_target_layer=7, access_is_momentary=True),
            Shortcut(1, "@access:L7:toggle", "Toggle Layer 7", "Layer Access", 12.0, category="layer_access", base_key="L7", is_layer_access=True, access_target_layer=7, access_is_momentary=False),
            *(Shortcut(i, f"K{i}", "", "App", 1.0, base_key=f"K{i}") for i in range(2, 8)),
        ])
        genome = np.arange(8, dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(8, dtype=np.bool_))
        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            group_overwrite_prob=0.0,
            mouse_workflow_prob=0.0,
            l7_access_prob=1.0,
        )
        moved = mutation._do(None, genome.reshape(1, -1).copy())[0]
        report = _layer7_access_report(layout.clone_with(genome=moved))
        self.assertTrue(report["acceptance_pass"], report)

    def test_random_reassign_preserves_last_important_shortcut_copy(self):
        random.seed(500)
        positions = tuple(Position(i, 1, float(i), 0.0, "left", 1, 0.5) for i in range(4))
        shortcuts = tuple([
            Shortcut(0, "Critical", "", "App", 6.0, base_key="Critical"),
            Shortcut(1, "Weak", "", "App", 1.0, base_key="Weak"),
            Shortcut(2, "Other", "", "App", 1.0, base_key="Other"),
            Shortcut(3, "Spare", "", "App", 1.0, base_key="Spare"),
        ])
        genome = np.array([0, 1, 1, 2], dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(4, dtype=np.bool_))
        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            group_overwrite_prob=0.0,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            random_assign_prob=1.0,
        )
        for _ in range(50):
            moved = mutation._do(None, genome.reshape(1, -1).copy())[0]
            self.assertIn(0, set(int(sid) for sid in moved))

    def test_bulk_reassign_preserves_last_important_shortcut_copy(self):
        random.seed(501)
        positions = tuple(Position(i, 1, float(i), 0.0, "left", 1, 0.5) for i in range(12))
        shortcuts = tuple([
            Shortcut(0, "Critical", "", "App", 6.0, base_key="Critical"),
            *(Shortcut(i, f"Weak{i}", "", "App", 1.0, base_key=f"Weak{i}") for i in range(1, 12)),
        ])
        genome = np.array([0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(12, dtype=np.bool_))
        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            group_overwrite_prob=0.0,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            random_assign_prob=0.0,
            bulk_assign_prob=1.0,
        )
        changed_any = False
        for _ in range(50):
            moved = mutation._do(None, genome.reshape(1, -1).copy())[0]
            self.assertIn(0, set(int(sid) for sid in moved))
            changed_any = changed_any or not np.array_equal(moved, genome)
        self.assertTrue(changed_any)

    def test_optional_arrow_drop_mutation_removes_mutable_raw_arrows(self):
        random.seed(502)
        positions = tuple(Position(i, 1, float(i), 0.0, "left", 1, 0.5) for i in range(8))
        shortcuts = tuple([
            Shortcut(0, "Critical", "", "App", 6.0, base_key="Critical"),
            Shortcut(1, "LeftArrow", "", "Raw", 4.0, base_key="LeftArrow"),
            Shortcut(2, "UpArrow", "", "Raw", 4.0, base_key="UpArrow"),
            Shortcut(3, "DownArrow", "", "Raw", 4.0, base_key="DownArrow"),
            Shortcut(4, "RightArrow", "", "Raw", 4.0, base_key="RightArrow"),
            Shortcut(5, "WeakA", "", "App", 1.0, base_key="WeakA"),
            Shortcut(6, "WeakB", "", "App", 1.0, base_key="WeakB"),
            Shortcut(7, "WeakC", "", "App", 1.0, base_key="WeakC"),
        ])
        genome = np.arange(8, dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(8, dtype=np.bool_))
        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            group_overwrite_prob=0.0,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            random_assign_prob=0.0,
            bulk_assign_prob=0.0,
            optional_arrow_drop_prob=1.0,
        )
        moved = mutation._do(None, genome.reshape(1, -1).copy())[0]
        self.assertIn(0, set(int(sid) for sid in moved))
        self.assertFalse(any(int(sid) in {1, 2, 3, 4} for sid in moved))


if __name__ == "__main__":
    unittest.main()
