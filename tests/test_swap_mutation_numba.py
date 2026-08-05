"""Swap mutation (Numba) tests."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import numpy as np

from core import Layout, Position, Shortcut
from evolution import SwapMutation


class TestSwapMutationNumba(unittest.TestCase):
    def _build_simple_mutation_layout(self):
        """Layout with frozen positions, a group, and mutable positions."""
        positions = tuple(
            Position(i, 1, float(i % 6), float(i // 6), "left" if i < 6 else "right", 1, 1.0)
            for i in range(12)
        )
        shortcuts = tuple([
            Shortcut(0, "LeftArrow", "Left", "Nav", 1.0, base_key="LeftArrow"),
            Shortcut(1, "UpArrow", "Up", "Nav", 1.0, base_key="UpArrow"),
            Shortcut(2, "DownArrow", "Down", "Nav", 1.0, base_key="DownArrow"),
            Shortcut(3, "RightArrow", "Right", "Nav", 1.0, base_key="RightArrow"),
            Shortcut(4, "@access:L2:hold", "L2 hold", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=2, access_is_momentary=True),
            Shortcut(5, "@access:L2:toggle", "L2 toggle", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=2, access_is_momentary=False),
            Shortcut(6, "@access:L0:toggle", "L0 return", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=0, access_is_momentary=False),
            *[Shortcut(i, f"K{i}", "", "App", 1.0) for i in range(7, 12)],
        ])
        frozen = np.array([False] * 12, dtype=np.bool_)
        frozen[0] = True
        genome = np.arange(12, dtype=np.int32)
        layout = Layout(genome, positions, shortcuts, frozen)
        return layout

    def test_effort_swap_can_fix_same_effort_mouse_button_misplacement(self):
        # Regression test: MB1 and MB3 can both sit at effort=0.0 while MB3
        # squats on MB1's ideal slot (a real pattern observed in an actual
        # checkpoint). Before this fix, _effort_swap's cost was pure
        # effort*importance, so two effort-tied mouse buttons produced zero
        # gradient and this misplacement could never be corrected -- no
        # matter how many generations ran. Now cost also includes each
        # button's ideal-position penalty, and lower_mask has an equal-effort
        # fallback for mouse buttons, so this swap should become reachable.
        positions = tuple(
            Position(i, 3, float(8 + i), 2.0, "right", 1, 0.0) for i in range(4)
        ) + (Position(4, 3, 20.0, 5.0, "right", 4, 1.0),)
        shortcuts = tuple([
            Shortcut(0, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(1, "MB2", "Click", "Mouse", 15.0, "mouse"),
            Shortcut(2, "MB3", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(3, "Ctrl+A", "Select All", "Editor", 5.0, "editing"),
        ])
        frozen = np.zeros(len(positions), dtype=np.bool_)
        # idx0=x8 (MB1's ideal) holds MB3; idx1=x9 (MB2's ideal) holds MB1;
        # idx2=x10 holds MB2; idx3=x11 (MB3's ideal) stays empty; idx4 is an
        # unrelated filler position so the duplicate-candidate pool isn't empty.
        base_genome = np.array([2, 0, 1, -1, 3], dtype=np.int32)
        layout = Layout(base_genome.copy(), positions, shortcuts, frozen)

        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.0,
            effort_swap_prob=1.0,
            smart_duplicate_prob=0.0,
        )

        fixed = False
        for _ in range(300):
            genome = base_genome.copy()
            if mutation._effort_swap(genome) and genome[0] == 0:
                fixed = True
                break
        self.assertTrue(fixed, "effort_swap never moved MB1 onto its ideal slot despite an effort tie with MB3")

    def test_smart_duplicate_pool_excludes_mouse_buttons(self):
        """Generic duplicate spawning must not scatter mouse buttons across layers."""
        positions = tuple(
            Position(i, 1, float(i % 6), float(i // 6), "right", 1, 1.0)
            for i in range(12)
        )
        shortcuts = tuple([
            Shortcut(0, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(1, "MB2", "Click", "Mouse", 18.0, "mouse"),
            Shortcut(2, "MB3", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(3, "MB4", "Click", "Mouse", 7.0, "mouse"),
            Shortcut(4, "MB5", "Click", "Mouse", 7.0, "mouse"),
            Shortcut(5, "Ctrl+A", "Select All", "Editor", 9.0, "editing"),
            Shortcut(6, "Ctrl+B", "Bold", "Editor", 5.0, "editing"),
        ])
        frozen = np.zeros(len(positions), dtype=np.bool_)
        layout = Layout(
            np.array([0, 1, 2, 3, 4, 5, 6, -1, -1, -1, -1, -1], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.0,
            effort_swap_prob=0.0,
            smart_duplicate_prob=1.0,
        )
        self.assertFalse(set(range(5)) & set(mutation._dup_candidate_arr.tolist()))
        self.assertIn(5, mutation._dup_candidate_arr.tolist())

    def test_numba_kernel_preserves_invariants(self):
        """Numba-accelerated mutations must not touch frozen positions or scatter groups."""
        layout = self._build_simple_mutation_layout()
        mutation = SwapMutation(
            prob=0.5,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.5,
            effort_swap_prob=0.5,
            smart_duplicate_prob=0.5,
        )
        # Not a constructor kwarg (matches existing hardcoded-default operators
        # like momentary_reuse_repair_prob); disable directly so this
        # invariants check isn't affected by arrow clustering, which isn't
        # under test here.
        mutation.arrow_cluster_prob = 0.0
        np.random.seed(42)
        random.seed(42)
        pop = np.tile(layout.genome.astype(np.int32), (200, 1))
        # introduce some empty slots and duplicates
        pop[pop == 11] = -1
        pop[:, 4] = 4
        out = mutation._do(None, pop.copy())

        # Frozen position 0 must be unchanged.
        self.assertTrue(np.all(out[:, 0] == layout.genome[0]))
        # Arrow group sids 0-3 must still occupy only positions 0-3 (group overwrite disabled).
        for sid in range(4):
            for row in out:
                pos = int(np.where(row == sid)[0][0]) if sid in row else -1
                if pos >= 0:
                    self.assertIn(pos, [0, 1, 2, 3])

    def test_numba_and_python_fallback_similar_mutation_rates(self):
        """Numba path and pure-Python fallback should mutate a similar fraction."""
        import evolution
        layout = self._build_simple_mutation_layout()
        mutation = SwapMutation(
            prob=0.5,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.5,
            effort_swap_prob=0.5,
            smart_duplicate_prob=0.5,
        )
        n = 500
        np.random.seed(123)
        random.seed(123)
        pop = np.tile(layout.genome.astype(np.int32), (n, 1))
        pop[pop == 11] = -1

        # Numba path
        out_numba = mutation._do(None, pop.copy())
        changed_numba = np.sum(np.any(out_numba != pop, axis=1))

        # Force Python fallback by patching NUMBA_AVAILABLE
        orig = evolution.NUMBA_AVAILABLE
        try:
            evolution.NUMBA_AVAILABLE = False
            np.random.seed(123)
            random.seed(123)
            out_py = mutation._do(None, pop.copy())
        finally:
            evolution.NUMBA_AVAILABLE = orig
        changed_py = np.sum(np.any(out_py != pop, axis=1))

        # Both should mutate a non-trivial fraction of genomes.
        self.assertGreater(changed_numba, n * 0.1)
        self.assertGreater(changed_py, n * 0.1)
        # Rates should be within 20 percentage points (different RNGs, same probabilities).
        self.assertLess(abs(changed_numba - changed_py) / n, 0.20)

    def test_numba_kernel_is_deterministic_for_same_seeds(self):
        """Calling the Numba kernel directly with the same seeds must be deterministic."""
        from evolution import _mutate_batch_numba
        layout = self._build_simple_mutation_layout()
        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.5,
            effort_swap_prob=0.5,
            smart_duplicate_prob=0.5,
        )
        n = 50
        pop = np.tile(layout.genome.astype(np.int32), (n, 1))
        pop[pop == 11] = -1
        handled = np.zeros(n, dtype=np.bool_)
        seeds = np.random.randint(0, 2**63, size=n, dtype=np.uint64)
        probs = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        out1 = pop.copy()
        _mutate_batch_numba(
            out1, handled.copy(), probs, seeds,
            mutation._mutable_arr,
            mutation._pos_layer_arr,
            mutation._pos_hand_arr,
            mutation._pos_is_thumb_arr,
            mutation._pos_effort_arr,
            mutation._pos_x,
            mutation._pos_y,
            mutation._sid_importance_arr,
            mutation._access_target_lut,
            mutation._access_is_mo_lut,
            mutation._mo_access_target_lut,
            mutation._is_group_sid_lut,
            mutation._is_important_sid_lut,
            np.int32(mutation._return_toggle_sid if mutation._return_toggle_sid is not None else -1),
            mutation._dup_candidate_arr,
            mutation._dup_exp_w,
            mutation._frozen_sid_counts,
            mutation._assignable_arr,
            mutation._layer_mutable_flat,
            mutation._layer_mutable_start,
            mutation._mouse_button_sids,
            mutation._toggle_access_sids_arr,
            mutation._pos_physical_id,
            mutation._arrow_sid_by_type,
            mutation._arrow_quads,
            mutation._raw_completion_sid_by_order,
            mutation._raw_completion_quints,
            mutation._is_mouse_button_lut,
            mutation._pos_is_frozen_arr,
            mutation._hold_sid_for_target,
            mutation._is_l0_only_lut,
            mutation._dup_support_arr,
            np.int32(mutation.n_shortcuts),
        )
        out2 = pop.copy()
        _mutate_batch_numba(
            out2, handled.copy(), probs, seeds,
            mutation._mutable_arr,
            mutation._pos_layer_arr,
            mutation._pos_hand_arr,
            mutation._pos_is_thumb_arr,
            mutation._pos_effort_arr,
            mutation._pos_x,
            mutation._pos_y,
            mutation._sid_importance_arr,
            mutation._access_target_lut,
            mutation._access_is_mo_lut,
            mutation._mo_access_target_lut,
            mutation._is_group_sid_lut,
            mutation._is_important_sid_lut,
            np.int32(mutation._return_toggle_sid if mutation._return_toggle_sid is not None else -1),
            mutation._dup_candidate_arr,
            mutation._dup_exp_w,
            mutation._frozen_sid_counts,
            mutation._assignable_arr,
            mutation._layer_mutable_flat,
            mutation._layer_mutable_start,
            mutation._mouse_button_sids,
            mutation._toggle_access_sids_arr,
            mutation._pos_physical_id,
            mutation._arrow_sid_by_type,
            mutation._arrow_quads,
            mutation._raw_completion_sid_by_order,
            mutation._raw_completion_quints,
            mutation._is_mouse_button_lut,
            mutation._pos_is_frozen_arr,
            mutation._hold_sid_for_target,
            mutation._is_l0_only_lut,
            mutation._dup_support_arr,
            np.int32(mutation.n_shortcuts),
        )
        np.testing.assert_array_equal(out1, out2)

    def test_repair_momentary_key_reuse_relocates_minority_target(self):
        """The same physical key acting as a momentary-hold source for two
        different target layers (depending on which layer is active) must be
        detected and repaired by relocating the minority-target occurrence to
        an empty slot on its own source layer, matching
        fitness/kernel.py's momentary_key_reuse scoring. Pure fitness pressure
        alone was empirically confirmed to plateau on this exact pattern
        across thousands of real generations (see AGENTS.md), which is why
        this mutation proposes the relocation directly instead of relying on
        random mutation to stumble onto it.
        """
        # pos0/pos1 share physical coordinate (2.0, 1.0) across layers 0 and 3
        # but hold momentary access to different targets (1 vs 2) -- reuse.
        # pos2/pos3 are empty mutable slots on the same two source layers so
        # the repair has somewhere to relocate the minority occurrence to.
        positions = (
            Position(0, 0, 2.0, 1.0, "left", 1, 1.0),
            Position(1, 3, 2.0, 1.0, "left", 1, 1.0),
            Position(2, 0, 5.0, 1.0, "left", 1, 1.0),
            Position(3, 3, 5.0, 1.0, "left", 1, 1.0),
        )
        shortcuts = (
            Shortcut(0, "@access:L1:hold", "L1 hold", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=1, access_is_momentary=True),
            Shortcut(1, "@access:L2:hold", "L2 hold", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=2, access_is_momentary=True),
        )
        frozen = np.zeros(4, dtype=np.bool_)
        genome = np.array([0, 1, -1, -1], dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, frozen)

        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.0,
            effort_swap_prob=0.0,
            smart_duplicate_prob=0.0,
        )

        fixed = False
        for _ in range(50):
            g = genome.copy()
            if mutation._repair_momentary_key_reuse(g):
                fixed = True
                # One of sid0/sid1 relocated to an empty slot; the other stays put.
                remaining_sids = {int(v) for v in g if v >= 0}
                self.assertEqual(remaining_sids, {0, 1})
                # No same-physical-key reuse: at most one of pos0/pos1 still holds a sid.
                self.assertTrue(g[0] < 0 or g[1] < 0)
                break
        self.assertTrue(fixed, "momentary_key_reuse repair never relocated the minority-target occurrence")

    def test_repair_l0_hold_completion_adds_missing_hold_to_free_key(self):
        """L0 has a toggle-only path to L1 (no direct hold). A genuinely free
        physical key exists on L0 (not already doing hold duty for some other
        layer elsewhere), and a busy one also exists (shares its physical
        coordinate with an existing hold-to-L2 on another layer). The repair
        must add the missing @access:L1:hold onto the free key, never onto
        the busy one -- confirmed empirically 2026-07-15 that placing it on
        an already-busy key trips momentary_key_reuse and net-loses, which is
        exactly the mistake this operator exists to avoid.
        """
        positions = (
            Position(0, 0, 0.0, 0.0, "left", 1, 1.0),   # L0: toggle-to-L1 (already placed)
            Position(1, 0, 1.0, 0.0, "left", 1, 1.0),   # L0: empty, but physical key is BUSY (shared coord below)
            Position(2, 5, 1.0, 0.0, "left", 1, 1.0),   # L5: hold-to-L2, same coord as pos1 -> busies that phys key
            Position(3, 0, 2.0, 0.0, "left", 1, 1.0),   # L0: empty, genuinely free physical key
        )
        shortcuts = (
            Shortcut(0, "@access:L1:toggle", "L1 toggle", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=1, access_is_momentary=False),
            Shortcut(1, "@access:L2:hold", "L2 hold", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=2, access_is_momentary=True),
            Shortcut(2, "@access:L1:hold", "L1 hold", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=1, access_is_momentary=True),
        )
        frozen = np.zeros(4, dtype=np.bool_)
        genome = np.array([0, -1, 1, -1], dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, frozen)

        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.0,
            effort_swap_prob=0.0,
            smart_duplicate_prob=0.0,
        )
        self.assertEqual(int(mutation._hold_sid_for_target[1]), 2)

        fixed = False
        for _ in range(50):
            g = genome.copy()
            if mutation._repair_l0_hold_completion(g):
                fixed = True
                self.assertEqual(int(g[3]), 2, "hold should land on the free physical key (pos3)")
                self.assertEqual(int(g[1]), -1, "busy physical key (pos1) must stay untouched")
                self.assertEqual(int(g[0]), 0, "existing toggle must stay in place")
                break
        self.assertTrue(fixed, "l0_hold_completion repair never added the missing hold")

    def test_repair_l0_hold_completion_no_op_without_free_key(self):
        """If every L0 physical key is already busy with hold duty elsewhere,
        the repair must not force the new hold onto a busy key (that's the
        exact mistake it exists to avoid) -- it should no-op instead. Both L0
        positions here (including the one holding the toggle itself) share a
        physical coordinate with an existing hold job on another layer, so
        there is truly no eligible destination.
        """
        positions = (
            Position(0, 0, 0.0, 0.0, "left", 1, 1.0),   # L0: toggle-to-L1, BUSY (shared coord w/ pos2)
            Position(1, 0, 1.0, 0.0, "left", 1, 1.0),   # L0: empty, BUSY (shared coord w/ pos3)
            Position(2, 5, 0.0, 0.0, "left", 1, 1.0),   # L5: hold-to-L2, busies pos0's physical key
            Position(3, 6, 1.0, 0.0, "left", 1, 1.0),   # L6: hold-to-L3, busies pos1's physical key
        )
        shortcuts = (
            Shortcut(0, "@access:L1:toggle", "L1 toggle", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=1, access_is_momentary=False),
            Shortcut(1, "@access:L2:hold", "L2 hold", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=2, access_is_momentary=True),
            Shortcut(2, "@access:L1:hold", "L1 hold", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=1, access_is_momentary=True),
            Shortcut(3, "@access:L3:hold", "L3 hold", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=3, access_is_momentary=True),
        )
        frozen = np.zeros(4, dtype=np.bool_)
        genome = np.array([0, -1, 1, 3], dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, frozen)

        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.0,
            effort_swap_prob=0.0,
            smart_duplicate_prob=0.0,
        )

        for _ in range(50):
            g = genome.copy()
            result = mutation._repair_l0_hold_completion(g)
            self.assertFalse(result)
            np.testing.assert_array_equal(g, genome)

    def test_propose_arrow_cluster_builds_valid_same_line_shape(self):
        """Four scattered raw-arrow shortcuts must be moved into one of
        AGENTS.md's valid same-line shapes (Left Up Down Right, one row) when
        a geometrically-valid empty quad exists, instead of relying on
        undirected mutation to stumble into the exact arrangement -- the same
        plateau pattern already found and fixed for momentary_key_reuse.
        """
        positions = (
            # Scattered across four different rows so they never accidentally
            # form a second valid same-line quad -- the only valid quad is
            # (4, 5, 6, 7) below.
            Position(0, 0, 0.0, 0.0, "left", 1, 1.0),
            Position(1, 0, 1.0, 1.0, "left", 1, 1.0),
            Position(2, 0, 2.0, 2.0, "left", 1, 1.0),
            Position(3, 0, 3.0, 3.0, "left", 1, 1.0),
            Position(4, 0, 10.0, 5.0, "right", 1, 1.0),
            Position(5, 0, 11.0, 5.0, "right", 1, 1.0),
            Position(6, 0, 12.0, 5.0, "right", 1, 1.0),
            Position(7, 0, 13.0, 5.0, "right", 1, 1.0),
        )
        shortcuts = (
            Shortcut(0, "LeftArrow", "Left", "Nav", 4.0, base_key="LeftArrow"),
            Shortcut(1, "RightArrow", "Right", "Nav", 4.0, base_key="RightArrow"),
            Shortcut(2, "UpArrow", "Up", "Nav", 4.0, base_key="UpArrow"),
            Shortcut(3, "DownArrow", "Down", "Nav", 4.0, base_key="DownArrow"),
            Shortcut(4, "Ctrl+A", "Select All", "Editor", 3.0),
        )
        frozen = np.zeros(8, dtype=np.bool_)
        genome = np.array([0, 1, 2, 3, -1, -1, -1, -1], dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, frozen)

        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.0,
            effort_swap_prob=0.0,
            smart_duplicate_prob=0.0,
        )
        self.assertEqual(mutation._arrow_quads.shape[0], 1)
        np.testing.assert_array_equal(mutation._arrow_quads[0], [4, 5, 6, 7])

        fixed = False
        for _ in range(20):
            g = genome.copy()
            if mutation._propose_arrow_cluster(g):
                fixed = True
                self.assertEqual(g[4], 0)  # Left
                self.assertEqual(g[5], 2)  # Up
                self.assertEqual(g[6], 3)  # Down
                self.assertEqual(g[7], 1)  # Right
                self.assertTrue(all(g[i] < 0 for i in range(4)))
                break
        self.assertTrue(fixed, "arrow cluster proposal never constructed the valid same-line shape")

    def test_propose_raw_completion_cluster_gathers_family_onto_one_quint(self):
        """The Norwegian/raw-completion family's base keys, scattered across
        different rows, must be moved onto one precomputed increasing-x quint,
        matching fitness/kernel.py's raw_keyboard_completion_norwegian scoring
        -- the same coordinated-shape plateau risk as arrows, but for 5 keys.
        """
        positions = (
            # Scattered across five different rows so they never accidentally
            # form a second valid quint. Only (5, 6, 7, 8, 9) is valid.
            Position(0, 0, 0.0, 0.0, "left", 1, 1.0),
            Position(1, 0, 1.0, 1.0, "left", 1, 1.0),
            Position(2, 0, 2.0, 2.0, "left", 1, 1.0),
            Position(3, 0, 3.0, 3.0, "left", 1, 1.0),
            Position(4, 0, 4.0, 4.0, "left", 1, 1.0),
            Position(5, 0, 10.0, 5.0, "right", 1, 1.0),
            Position(6, 0, 11.0, 5.0, "right", 1, 1.0),
            Position(7, 0, 12.0, 5.0, "right", 1, 1.0),
            Position(8, 0, 13.0, 5.0, "right", 1, 1.0),
            Position(9, 0, 14.0, 5.0, "right", 1, 1.0),
        )
        shortcuts = (
            Shortcut(0, "-", "Dash", "Nav", 3.0, base_key="Dash and Underscore"),
            Shortcut(1, "=", "Equals", "Nav", 3.0, base_key="Equals and Plus"),
            Shortcut(2, "`", "Grave", "Nav", 3.0, base_key="Grave Accent and Tilde"),
            Shortcut(3, "]", "RBrace", "Nav", 3.0, base_key="Right Brace"),
            Shortcut(4, "\\", "Backslash", "Nav", 3.0, base_key="Backslash and Pipe"),
            Shortcut(5, "Ctrl+A", "Select All", "Editor", 3.0),
        )
        frozen = np.zeros(10, dtype=np.bool_)
        genome = np.array([0, 1, 2, 3, 4, -1, -1, -1, -1, -1], dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, frozen)

        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.0,
            effort_swap_prob=0.0,
            smart_duplicate_prob=0.0,
        )
        self.assertEqual(mutation._raw_completion_quints.shape[0], 1)
        np.testing.assert_array_equal(mutation._raw_completion_quints[0], [5, 6, 7, 8, 9])

        fixed = False
        for _ in range(20):
            g = genome.copy()
            if mutation._propose_raw_completion_cluster(g):
                fixed = True
                self.assertEqual(g[5], 0)
                self.assertEqual(g[6], 1)
                self.assertEqual(g[7], 2)
                self.assertEqual(g[8], 3)
                self.assertEqual(g[9], 4)
                self.assertTrue(all(g[i] < 0 for i in range(5)))
                break
        self.assertTrue(fixed, "raw completion cluster proposal never gathered the family onto the valid quint")

    def test_bias_access_to_thumb_relocates_existing_return_toggle(self):
        """An existing off-thumb return-to-L0 toggle must be relocatable onto
        a free thumb slot. Before this change, access_target_lut[sid] > 0
        excluded target == 0 (return toggles) entirely, so
        _numba_bias_access_to_thumb/_bias_access_to_thumb could never fix one
        that was already placed off-thumb -- only _numba_repair_return_toggles
        (create-if-missing) touched return toggles at all.
        """
        positions = (
            Position(0, 1, 0.0, 0.0, "left", 1, 1.0),                      # non-thumb, holds the return toggle
            Position(1, 1, 5.0, 3.0, "left", 0, 0.1, is_thumb=True),       # free thumb slot on the same layer
        )
        shortcuts = (
            Shortcut(0, "@access:L0:toggle", "L0 return", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=0, access_is_momentary=False),
        )
        frozen = np.zeros(2, dtype=np.bool_)
        genome = np.array([0, -1], dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, frozen)

        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.0,
            effort_swap_prob=0.0,
            smart_duplicate_prob=0.0,
        )

        fixed = False
        for _ in range(20):
            g = genome.copy()
            if mutation._bias_access_to_thumb(g):
                fixed = True
                self.assertEqual(g[1], 0)
                self.assertEqual(g[0], -1)
                break
        self.assertTrue(fixed, "bias_access_to_thumb never relocated the existing off-thumb return toggle")

    def test_repair_mouse_hold_conflict_relocates_colliding_access_key(self):
        """A momentary access key targeting the settled mouse layer must not
        sit at the exact physical (x, y) of one of that layer's mouse
        buttons, matching fitness/kernel.py's mouse_hold_position_conflict.
        """
        positions = (
            Position(0, 1, 5.0, 2.0, "right", 1, 1.0),   # MB1 on mouse layer 1
            Position(1, 1, 6.0, 2.0, "right", 1, 1.0),   # MB2
            Position(2, 1, 7.0, 2.0, "right", 1, 1.0),   # MB3
            Position(3, 0, 5.0, 2.0, "left", 1, 1.0),    # same (x,y) as MB1, but layer 0 -- conflict
            Position(4, 0, 9.0, 9.0, "left", 1, 1.0),    # empty mutable slot on layer 0
        )
        shortcuts = (
            Shortcut(0, "MB1", "Click", "Mouse", 10.0, base_key="MB1"),
            Shortcut(1, "MB2", "Click", "Mouse", 9.0, base_key="MB2"),
            Shortcut(2, "MB3", "Click", "Mouse", 7.0, base_key="MB3"),
            Shortcut(3, "@access:L1:hold", "L1 hold", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=1, access_is_momentary=True),
        )
        frozen = np.zeros(5, dtype=np.bool_)
        genome = np.array([0, 1, 2, 3, -1], dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, frozen)

        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.0,
            effort_swap_prob=0.0,
            smart_duplicate_prob=0.0,
        )

        fixed = False
        for _ in range(20):
            g = genome.copy()
            if mutation._repair_mouse_hold_conflict(g):
                fixed = True
                self.assertEqual(g[4], 3)
                self.assertEqual(g[3], -1)
                break
        self.assertTrue(fixed, "mouse_hold_conflict repair never relocated the colliding access key")

    def test_repair_thumb_occupancy_relocates_off_restricted_side(self):
        """A thumb position occupied on a side that AGENTS.md's dynamic
        thumb-clearance rule restricts (a momentary thumb key reaches the
        layer from that side only) must be relocated to a non-thumb slot on
        the same layer, matching fitness/kernel.py's thumb_occupancy scoring.
        """
        positions = (
            Position(0, 0, 0.0, 0.0, "right", 0, 0.1, is_thumb=True),   # momentary hold -> L1, right thumb
            Position(1, 1, 5.0, 2.0, "right", 0, 0.1, is_thumb=True),   # occupied right-thumb slot on L1 -- conflict
            Position(2, 1, 9.0, 9.0, "left", 1, 1.0),                  # empty non-thumb slot on L1
        )
        shortcuts = (
            Shortcut(0, "@access:L1:hold", "L1 hold", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=1, access_is_momentary=True),
            Shortcut(1, "Ctrl+A", "Select All", "Editor", 3.0),
        )
        frozen = np.zeros(3, dtype=np.bool_)
        genome = np.array([0, 1, -1], dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, frozen)

        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.0,
            effort_swap_prob=0.0,
            smart_duplicate_prob=0.0,
        )

        fixed = False
        for _ in range(20):
            g = genome.copy()
            if mutation._repair_thumb_occupancy(g):
                fixed = True
                self.assertEqual(g[2], 1)
                self.assertEqual(g[1], -1)
                break
        self.assertTrue(fixed, "thumb_occupancy repair never relocated the occupant off the restricted side")

    def test_repair_same_layer_duplicate_clears_one_copy(self):
        """same_layer_duplicate is a hard constraint (no shortcut may appear
        more than once on the same layer, L7 excluded, with a mouse-button
        left+right exception). A non-mouse-button shortcut duplicated twice
        on the same layer must have one mutable copy cleared, matching
        fitness/kernel.py's same_layer_duplicate scoring.
        """
        positions = (
            Position(0, 1, 0.0, 0.0, "left", 1, 1.0),
            Position(1, 1, 1.0, 0.0, "left", 1, 1.0),
            Position(2, 1, 2.0, 0.0, "left", 1, 1.0),
        )
        shortcuts = (
            Shortcut(0, "Ctrl+A", "Select All", "Editor", 3.0),
        )
        frozen = np.zeros(3, dtype=np.bool_)
        genome = np.array([0, 0, -1], dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, frozen)

        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.0,
            effort_swap_prob=0.0,
            smart_duplicate_prob=0.0,
        )

        fixed = False
        for _ in range(20):
            g = genome.copy()
            if mutation._repair_same_layer_duplicate(g):
                fixed = True
                cleared = sum(1 for v in g if v == 0)
                self.assertEqual(cleared, 1, "exactly one duplicate copy must be cleared")
                break
        self.assertTrue(fixed, "same_layer_duplicate repair never cleared a duplicate copy")

    def test_numba_random_reassign_pairs_return_toggle(self):
        """Numba random_reassign must place a return toggle when creating a toggle access."""
        from evolution import _numba_random_reassign_one
        # Build a layout with mutable positions on layer 2 so the return toggle has a home.
        positions = tuple(
            Position(i, 1 if i < 6 else 2, float(i % 6), float(i // 6), "left" if i < 6 else "right", 1, 1.0)
            for i in range(12)
        )
        shortcuts = tuple([
            Shortcut(0, "LeftArrow", "Left", "Nav", 1.0, base_key="LeftArrow"),
            Shortcut(1, "UpArrow", "Up", "Nav", 1.0, base_key="UpArrow"),
            Shortcut(2, "DownArrow", "Down", "Nav", 1.0, base_key="DownArrow"),
            Shortcut(3, "RightArrow", "Right", "Nav", 1.0, base_key="RightArrow"),
            Shortcut(4, "@access:L2:toggle", "L2 toggle", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=2, access_is_momentary=False),
            Shortcut(5, "@access:L0:toggle", "L0 return", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=0, access_is_momentary=False),
            *[Shortcut(i, f"K{i}", "", "App", 1.0) for i in range(6, 12)],
        ])
        frozen = np.array([False] * 12, dtype=np.bool_)
        genome = np.arange(12, dtype=np.int32)
        layout = Layout(genome, positions, shortcuts, frozen)
        mutation = SwapMutation(prob=0.0, frozen_mask=layout.frozen_mask, layout=layout)

        # Force assignable pool to only the L2 toggle so random_reassign must place it.
        mutation._assignable_arr = np.array([4], dtype=np.int32)

        g = genome.copy()
        g[5] = 11  # non-group position on layer 1
        state = np.array([42], dtype=np.uint64)
        ok = _numba_random_reassign_one(
            g, state,
            mutation._mutable_arr,
            mutation._pos_layer_arr,
            mutation._assignable_arr,
            mutation._is_group_sid_lut,
            mutation._is_important_sid_lut,
            mutation._access_target_lut,
            mutation._mo_access_target_lut,
            mutation.n_shortcuts,
            mutation._toggle_access_sids_arr,
            np.int32(mutation._return_toggle_sid),
            mutation._layer_mutable_flat,
            mutation._layer_mutable_start,
        )
        self.assertTrue(ok)
        self.assertIn(4, g, "L2 toggle was not placed")
        self.assertIn(5, g, "Return-to-L0 toggle missing on layer 2")


class TestUnsupportedDuplicateRepair(unittest.TestCase):
    """unsupported_duplicate_repair: blank extra copies of duplicated
    shortcuts that have some usage evidence but duplicate support < 0.25
    (the unsupported_duplicates_near_zero acceptance criterion)."""

    def _build_layout(self, frozen_idx=()):
        # 8 mutable positions across layers 1 and 2 with varying effort.
        efforts = [0.0, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 3.5]
        positions = tuple(
            Position(i, 1 if i < 4 else 2, float(i % 4), 0.0, "left", 1, efforts[i],
                     is_frozen=i in frozen_idx)
            for i in range(8)
        )
        shortcuts = tuple(Shortcut(i, f"K{i}", "", "App", 1.0) for i in range(8))
        frozen = np.zeros(8, dtype=np.bool_)
        for i in frozen_idx:
            frozen[i] = True
        genome = np.full(8, -1, dtype=np.int32)
        return Layout(genome, positions, shortcuts, frozen)

    def _mutation(self, layout):
        return SwapMutation(prob=0.0, frozen_mask=layout.frozen_mask, layout=layout)

    def test_blanks_extra_copies_keeps_lowest_effort(self):
        layout = self._build_layout()
        m = self._mutation(layout)
        m._dup_support_arr[5] = 0.1  # some evidence, support < 0.25
        genome = np.full(8, -1, dtype=np.int32)
        genome[1] = 5  # effort 1.0
        genome[3] = 5  # effort 3.0
        self.assertTrue(m._repair_unsupported_duplicate(genome))
        self.assertEqual(int((genome == 5).sum()), 1)
        self.assertEqual(int(genome[1]), 5, "lowest-effort copy should be kept")
        self.assertEqual(int(genome[3]), -1)

    def test_ignores_tolerated_and_supported_duplicates(self):
        layout = self._build_layout()
        m = self._mutation(layout)
        genome = np.full(8, -1, dtype=np.int32)
        genome[0] = 5
        genome[1] = 5
        before = genome.copy()
        # support == 0: zero-evidence duplicates are tolerated by acceptance.
        self.assertFalse(m._repair_unsupported_duplicate(genome))
        self.assertTrue((genome == before).all())
        # support >= 0.25: accepted/uncertain, also untouched.
        m._dup_support_arr[5] = 0.5
        self.assertFalse(m._repair_unsupported_duplicate(genome))
        self.assertTrue((genome == before).all())

    def test_prefers_frozen_copy_even_at_higher_effort(self):
        layout = self._build_layout(frozen_idx=(3,))
        m = self._mutation(layout)
        m._dup_support_arr[5] = 0.1
        genome = np.full(8, -1, dtype=np.int32)
        genome[0] = 5  # mutable, effort 0.0
        genome[3] = 5  # frozen, effort 3.0
        self.assertTrue(m._repair_unsupported_duplicate(genome))
        self.assertEqual(int(genome[3]), 5, "frozen copy is untouchable, must be kept")
        self.assertEqual(int(genome[0]), -1)

    def test_all_frozen_copies_returns_false(self):
        layout = self._build_layout(frozen_idx=(0, 1))
        m = self._mutation(layout)
        m._dup_support_arr[5] = 0.1
        genome = np.full(8, -1, dtype=np.int32)
        genome[0] = 5
        genome[1] = 5
        self.assertFalse(m._repair_unsupported_duplicate(genome))
        self.assertEqual(int((genome == 5).sum()), 2)

    def test_l0_only_shortcuts_excluded(self):
        # _base_*/is_l0_only shortcuts are outside the unsupported-duplicate
        # class (mirrors run_evolution.analyze_duplicates) -- the repair must
        # leave their duplicates alone even with low support.
        layout = self._build_layout()
        m = self._mutation(layout)
        m._is_l0_only_lut[5] = True
        m._dup_support_arr[5] = 0.1
        genome = np.full(8, -1, dtype=np.int32)
        genome[0] = 5
        genome[1] = 5
        self.assertFalse(m._repair_unsupported_duplicate(genome))
        self.assertEqual(int((genome == 5).sum()), 2)

    def test_numba_path_matches_fallback(self):
        from evolution import NUMBA_AVAILABLE
        if not NUMBA_AVAILABLE:
            self.skipTest("Numba unavailable")
        from evolution import _numba_repair_unsupported_duplicate
        layout = self._build_layout()
        m = self._mutation(layout)
        m._dup_support_arr[5] = 0.1
        genome = np.full(8, -1, dtype=np.int32)
        genome[1] = 5
        genome[3] = 5
        ok = _numba_repair_unsupported_duplicate(
            genome,
            m._pos_layer_arr,
            m._pos_is_frozen_arr,
            m._pos_effort_arr,
            m._is_mouse_button_lut,
            m._is_group_sid_lut,
            m._is_l0_only_lut,
            m._dup_support_arr,
            np.int32(m.n_shortcuts),
        )
        self.assertTrue(ok)
        self.assertEqual(int(genome[1]), 5)
        self.assertEqual(int(genome[3]), -1)


class TestThumbOccupancyRepairAlignment(unittest.TestCase):
    """_repair_thumb_occupancy must mirror acceptance's
    momentary_only_thumb_side_clear semantics: skip toggle-freed layers,
    ignore unreachable and self-referential access sources, and clear ALL
    occupants of a restricted side in one call."""

    def _hold(self, sid, target):
        return Shortcut(sid, f"@access:L{target}:hold", "h", "Layer Access", 5.0,
                        is_layer_access=True, access_target_layer=target,
                        access_is_momentary=True)

    def _toggle(self, sid, target):
        return Shortcut(sid, f"@access:L{target}:toggle", "t", "Layer Access", 5.0,
                        is_layer_access=True, access_target_layer=target,
                        access_is_momentary=False)

    def _mutation(self, positions, shortcuts, genome):
        frozen = np.zeros(len(positions), dtype=np.bool_)
        layout = Layout(np.array(genome, dtype=np.int32).copy(), positions, shortcuts, frozen)
        return SwapMutation(prob=0.0, frozen_mask=layout.frozen_mask, layout=layout)

    def test_clears_all_occupants_of_restricted_side_in_one_call(self):
        positions = (
            Position(0, 0, 0.0, 0.0, "right", 0, 0.1, is_thumb=True),  # hold -> L1
            Position(1, 1, 5.0, 2.0, "right", 0, 0.1, is_thumb=True),  # occupant 1
            Position(2, 1, 6.0, 2.0, "right", 0, 0.2, is_thumb=True),  # occupant 2
            Position(3, 1, 9.0, 9.0, "left", 1, 1.0),                  # empty non-thumb
            Position(4, 1, 10.0, 9.0, "right", 1, 1.0),                # empty non-thumb
        )
        shortcuts = (self._hold(0, 1), Shortcut(1, "Ctrl+A", "a", "App", 3.0),
                     Shortcut(2, "Ctrl+B", "a", "App", 3.0))
        m = self._mutation(positions, shortcuts, [0, 1, 2, -1, -1])
        genome = np.array([0, 1, 2, -1, -1], dtype=np.int32)
        self.assertTrue(m._repair_thumb_occupancy(genome))
        self.assertEqual(int(genome[1]), -1, "occupant 1 must be cleared off the restricted side")
        self.assertEqual(int(genome[2]), -1, "occupant 2 must be cleared in the same call")
        self.assertEqual(sorted(int(v) for v in genome[3:]), [1, 2],
                         "both occupants must survive on non-thumb slots")

    def test_skips_toggle_freed_layer(self):
        positions = (
            Position(0, 0, 0.0, 0.0, "right", 0, 0.1, is_thumb=True),  # hold -> L1
            Position(1, 0, 1.0, 0.0, "left", 1, 1.0),                  # toggle -> L1
            Position(2, 1, 5.0, 2.0, "right", 0, 0.1, is_thumb=True),  # occupant
            Position(3, 1, 9.0, 9.0, "left", 1, 1.0),
        )
        shortcuts = (self._hold(0, 1), self._toggle(1, 1), Shortcut(2, "Ctrl+A", "a", "App", 3.0))
        m = self._mutation(positions, shortcuts, [0, 1, 2, -1])
        genome = np.array([0, 1, 2, -1], dtype=np.int32)
        self.assertFalse(m._repair_thumb_occupancy(genome))
        self.assertEqual(int(genome[2]), 2, "toggle-freed layer's thumb occupant must be left alone")

    def test_self_referential_source_does_not_trigger(self):
        positions = (
            Position(0, 1, 0.0, 0.0, "right", 0, 0.1, is_thumb=True),  # self-ref hold ON L1
            Position(1, 1, 5.0, 2.0, "right", 0, 0.1, is_thumb=True),  # occupant
            Position(2, 1, 9.0, 9.0, "left", 1, 1.0),
        )
        shortcuts = (self._hold(0, 1), Shortcut(1, "Ctrl+A", "a", "App", 3.0))
        m = self._mutation(positions, shortcuts, [0, 1, -1])
        genome = np.array([0, 1, -1], dtype=np.int32)
        self.assertFalse(m._repair_thumb_occupancy(genome))
        self.assertEqual(int(genome[1]), 1)

    def test_unreachable_source_does_not_trigger(self):
        positions = (
            Position(0, 2, 0.0, 0.0, "right", 0, 0.1, is_thumb=True),  # hold -> L1 from unreachable L2
            Position(1, 1, 5.0, 2.0, "right", 0, 0.1, is_thumb=True),  # occupant
            Position(2, 1, 9.0, 9.0, "left", 1, 1.0),
        )
        shortcuts = (self._hold(0, 1), Shortcut(1, "Ctrl+A", "a", "App", 3.0))
        m = self._mutation(positions, shortcuts, [0, 1, -1])
        genome = np.array([0, 1, -1], dtype=np.int32)
        self.assertFalse(m._repair_thumb_occupancy(genome))
        self.assertEqual(int(genome[1]), 1)

    def test_numba_path_matches_fallback(self):
        from evolution import NUMBA_AVAILABLE
        if not NUMBA_AVAILABLE:
            self.skipTest("Numba unavailable")
        from evolution import _numba_repair_thumb_occupancy
        positions = (
            Position(0, 0, 0.0, 0.0, "right", 0, 0.1, is_thumb=True),
            Position(1, 1, 5.0, 2.0, "right", 0, 0.1, is_thumb=True),
            Position(2, 1, 6.0, 2.0, "right", 0, 0.2, is_thumb=True),
            Position(3, 1, 9.0, 9.0, "left", 1, 1.0),
            Position(4, 1, 10.0, 9.0, "right", 1, 1.0),
        )
        shortcuts = (self._hold(0, 1), Shortcut(1, "Ctrl+A", "a", "App", 3.0),
                     Shortcut(2, "Ctrl+B", "a", "App", 3.0))
        m = self._mutation(positions, shortcuts, [0, 1, 2, -1, -1])
        genome = np.array([0, 1, 2, -1, -1], dtype=np.int32)
        state = np.array([42], dtype=np.uint64)
        ok = _numba_repair_thumb_occupancy(
            genome, state,
            m._pos_layer_arr, m._pos_hand_arr, m._pos_is_thumb_arr,
            m._access_target_lut, m._access_is_mo_lut,
            m._layer_mutable_flat, m._layer_mutable_start,
            np.int32(m.n_shortcuts),
        )
        self.assertTrue(ok)
        self.assertEqual(int(genome[1]), -1)
        self.assertEqual(int(genome[2]), -1)
        self.assertEqual(sorted(int(v) for v in genome[3:]), [1, 2])


if __name__ == "__main__":
    unittest.main()
