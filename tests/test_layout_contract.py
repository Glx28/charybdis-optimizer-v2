"""Focused regression coverage for the versioned owner layout contract."""

import json
from pathlib import Path
import random
import unittest

import numpy as np

from evolution.layout_contract import (
    CONTRACT,
    audit_layout_contract,
    owner_protected_indices,
)
from tools._common import load_layout


SEED_PATH = Path(__file__).resolve().parents[1] / "seeds/gen25k_corrected_seed.json"


class TestOwnerLayoutContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = load_layout()
        genome = np.asarray(json.loads(SEED_PATH.read_text())['genome'], dtype=np.int32)
        cls.layout = base.clone_with(genome=genome)
        cls.by_coord = {
            (int(p.layer), int(p.x), int(p.y)): int(p.gene_idx)
            for p in cls.layout.positions
        }
        cls.sid = {str(s.keys): int(s.sid) for s in cls.layout.shortcuts}

    def changed(self, changes):
        genome = self.layout.genome.copy()
        for (layer, x, y), sid in changes.items():
            genome[self.by_coord[(layer, x, y)]] = self.sid[sid] if isinstance(sid, str) else sid
        return self.layout.clone_with(genome=genome)

    @classmethod
    def owner_model(cls):
        if not hasattr(cls, "_cpu_owner_model"):
            from tools._common import load_evaluator

            cls._cpu_owner_model = load_evaluator(use_cuda=False).model
        return cls._cpu_owner_model

    def owner_violation_code(self, genome):
        from fitness.model import _owner_layout_contract_v1_violation_batch

        model = self.owner_model()
        return int(_owner_layout_contract_v1_violation_batch(
            np.asarray(genome, dtype=np.int32)[None, :],
            model._position_layers, model._position_x, model._position_y,
            model._position_effort, model._position_hand, model._position_is_thumb,
            model._access_targets, model._access_is_momentary, model._is_scroll_access,
            model._owner_mouse_sids, model._mouse_button,
            model._owner_required_sids, model._owner_required_index_by_sid,
            model._owner_required_pair_sids, model._owner_required_pair_position_edges,
            model._owner_mb45_position_edges, model._owner_l1_hold_sid,
            model._owner_fixed_scroll_sid, model._owner_left_alt_sid,
            model._owner_right_alt_sid, model._owner_expected_l7,
        )[0, 0])

    def test_corrected_gen25k_seed_passes_every_owner_rule(self):
        report = audit_layout_contract(self.layout)
        self.assertTrue(report["acceptance_pass"], report["failed_checks"])

    def test_semantic_and_norwegian_groups_are_optimizer_hard_constraints(self):
        model = self.owner_model()
        semantic = next(c for c in self.layout.semantic_clusters
                        if c.get("name") == "pattern_Ctrl_switch_to_tab")
        semantic_sids = {int(m["sid"]) for m in semantic["members"]}
        broken_relative = self.layout.genome.copy()
        member_sid = int(semantic["members"][-1]["sid"])
        occurrences = [i for i, sid in enumerate(broken_relative) if int(sid) == member_sid]
        for member_idx in occurrences:
            layer = int(self.layout.positions[member_idx].layer)
            displaced_idx = next(
                i for i, sid in enumerate(broken_relative)
                if int(self.layout.positions[i].layer) == layer and int(sid) >= 0
                and int(sid) not in semantic_sids and not self.layout.positions[i].is_frozen
            )
            broken_relative[member_idx], broken_relative[displaced_idx] = (
                broken_relative[displaced_idx], broken_relative[member_idx]
            )
        relative_report = audit_layout_contract(self.layout.clone_with(genome=broken_relative))
        self.assertFalse(relative_report["checks"]["required_semantic_cluster_layouts"])

        norwegian_sids = {
            int(s.sid) for s in self.layout.shortcuts
            if str(s.base_key).upper() in {
                "DASH AND UNDERSCORE", "EQUALS AND PLUS", "GRAVE ACCENT AND TILDE",
                "RIGHT BRACE", "NON-US BACKSLASH AND PIPE",
            } and not s.modifiers
        }
        broken_norwegian = self.layout.genome.copy()
        raw_idx = next(i for i, sid in enumerate(broken_norwegian) if int(sid) in norwegian_sids)
        layer = int(self.layout.positions[raw_idx].layer)
        other_idx = next(
            i for i, sid in enumerate(broken_norwegian)
            if int(self.layout.positions[i].layer) == layer and int(sid) >= 0
            and int(sid) not in norwegian_sids and not self.layout.positions[i].is_frozen
        )
        broken_norwegian[raw_idx], broken_norwegian[other_idx] = (
            broken_norwegian[other_idx], broken_norwegian[raw_idx]
        )
        norwegian_report = audit_layout_contract(self.layout.clone_with(genome=broken_norwegian))
        self.assertFalse(norwegian_report["checks"]["norwegian_raw_key_cluster"])

        _, constraints = model.evaluate_batch(np.asarray(
            [broken_relative, broken_norwegian], dtype=np.int32
        ))
        owner_column = model.hard_constraints.index("owner_layout_contract_v1")
        self.assertTrue(np.all(constraints[:, owner_column] > 0))
        if model._use_cuda:
            _, cuda_constraints = model.evaluate_batch(np.asarray(
                [broken_relative, broken_norwegian], dtype=np.int32
            ))
        else:
            from fitness.cuda_kernel import cuda_available
            if cuda_available():
                model._use_cuda = True
                try:
                    _, cuda_constraints = model.evaluate_batch(np.asarray(
                        [broken_relative, broken_norwegian], dtype=np.int32
                    ))
                finally:
                    model._use_cuda = False
                np.testing.assert_array_equal(cuda_constraints, constraints)

    def test_manual_checkpoint_warning_uses_owner_contract(self):
        from tools._common import mouse_layer_quality_warnings

        report = audit_layout_contract(self.changed({(1, 8, 2): -1}))
        warnings = mouse_layer_quality_warnings(self.layout, arrays=(), mouse_report=report)
        self.assertTrue(any("fixed_l1_mouse_positions" in warning for warning in warnings))

    def test_missing_button_and_broken_fixed_coordinates_fail(self):
        genome = self.layout.genome.copy()
        for pos in self.layout.positions:
            if int(pos.layer) == 1 and int(genome[pos.gene_idx]) == self.sid["MB1"]:
                genome[pos.gene_idx] = -1
        mb1 = self.layout.clone_with(genome=genome)
        report = audit_layout_contract(mb1)
        self.assertFalse(report["checks"]["all_mouse_buttons_on_l1"])
        self.assertFalse(report["checks"]["fixed_l1_mouse_positions"])

        report = audit_layout_contract(self.changed({(1, 8, 2): -1}))
        self.assertFalse(report["checks"]["fixed_l1_mouse_positions"])
        report = audit_layout_contract(self.changed({(1, 9, 2): -1}))
        self.assertFalse(report["checks"]["fixed_l1_mouse_positions"])

    def test_separated_mb4_mb5_fail(self):
        genome = self.layout.genome.copy()
        for pos in self.layout.positions:
            if int(pos.layer) == 1 and int(genome[pos.gene_idx]) == self.sid["MB5"]:
                genome[pos.gene_idx] = -1
        genome[self.by_coord[(1, 10, 1)]] = self.sid["MB5"]
        report = audit_layout_contract(self.layout.clone_with(genome=genome))
        self.assertFalse(report["checks"]["mb4_mb5_adjacent_right_same_row_ordered"])

    def test_precomputed_pair_edges_preserve_mb45_and_required_pair_checks(self):
        valid = self.layout.genome.copy()
        self.assertEqual(self.owner_violation_code(valid), 0)

        mb45_broken = valid.copy()
        mb45_broken[self.by_coord[(1, 9, 1)]], mb45_broken[self.by_coord[(1, 11, 1)]] = (
            mb45_broken[self.by_coord[(1, 11, 1)]], mb45_broken[self.by_coord[(1, 9, 1)]]
        )
        report = audit_layout_contract(self.layout.clone_with(genome=mb45_broken))
        self.assertFalse(report["checks"]["mb4_mb5_adjacent_right_same_row_ordered"])
        self.assertTrue(self.owner_violation_code(mb45_broken) & 8)

        workflow_pair_rearranged = valid.copy()
        workflow_pair_rearranged[self.by_coord[(1, 2, 2)]], workflow_pair_rearranged[self.by_coord[(1, 4, 2)]] = (
            workflow_pair_rearranged[self.by_coord[(1, 4, 2)]], workflow_pair_rearranged[self.by_coord[(1, 2, 2)]]
        )
        report = audit_layout_contract(self.layout.clone_with(genome=workflow_pair_rearranged))
        self.assertFalse(
            report["checks"]["required_group:Win+Shift+Left|Win+Shift+Right"]
        )
        self.assertTrue(self.owner_violation_code(workflow_pair_rearranged) & 32)

    def test_missing_l1_lock_fails(self):
        genome = self.layout.genome.copy()
        for pos in self.layout.positions:
            if int(pos.layer) != 1:
                continue
            sid = int(genome[pos.gene_idx])
            if sid >= 0:
                shortcut = self.layout.shortcuts[sid]
                if (shortcut.is_layer_access and shortcut.access_target_layer == 1
                        and not shortcut.access_is_momentary):
                    genome[pos.gene_idx] = -1
        report = audit_layout_contract(self.layout.clone_with(genome=genome))
        self.assertFalse(report["checks"]["reachable_l1_self_lock"])

    def test_scroll_count_two_and_four_fail(self):
        two = self.changed({(2, 10, 2): -1})
        self.assertFalse(audit_layout_contract(two)["checks"]["exactly_three_momentary_scrolls"])

        four = self.changed({(3, 10, 2): "@scroll:L3:hold"})
        self.assertFalse(audit_layout_contract(four)["checks"]["exactly_three_momentary_scrolls"])

    def test_scroll_hold_cannot_replace_essential_ordinary_access(self):
        report = audit_layout_contract(self.changed({(0, 3, 4): "@scroll:L1:hold"}))
        self.assertFalse(report["checks"]["l0_x3_y4_ordinary_l1_hold"])

    def test_single_and_nested_both_side_thumb_holds_are_strict(self):
        single = self.changed({(1, 4, 4): "Ctrl+C"})
        self.assertFalse(audit_layout_contract(single)["checks"]["momentary_thumb_clearance_all_paths"])

        nested = self.changed({
            (3, 4, 5): "@access:L8:hold",
            (3, 8, 4): "@access:L8:hold",
            (8, 4, 4): "Ctrl+C",
            (8, 8, 4): "Ctrl+V",
            (8, 4, 3): "@access:L8:toggle",
        })
        report = audit_layout_contract(nested)
        self.assertFalse(report["checks"]["momentary_thumb_clearance_all_paths"])
        self.assertEqual(report["details"]["momentary_thumb_restrictions"]["8"], ["left", "right"])

    def test_reachable_same_layer_thumb_hold_still_restricts_its_target_thumb_side(self):
        genome = self.layout.genome.copy()
        genome[self.by_coord[(8, 4, 5)]] = self.sid["@access:L8:hold"]
        report = audit_layout_contract(self.layout.clone_with(genome=genome))
        self.assertFalse(report["checks"]["momentary_thumb_clearance_all_paths"])
        self.assertTrue(self.owner_violation_code(genome) & 128)
        model = self.owner_model()
        _, constraints = model.evaluate_batch(genome[None, :])
        owner_column = model.hard_constraints.index("owner_layout_contract_v1")
        self.assertGreater(constraints[0, owner_column], 0)

    def test_l7_is_frozen_and_exempt(self):
        report = audit_layout_contract(self.layout)
        self.assertTrue(report["checks"]["l7_frozen_and_exempt"])
        self.assertTrue(all(p.is_frozen for p in self.layout.positions if int(p.layer) == 7))

    def test_applied_l0_l7_hold_is_preserved_as_seed_specific_fixed_control(self):
        from evolution import SwapMutation

        genome = self.layout.genome.copy()
        idx = self.by_coord[(0, 7, 4)]
        sid = self.sid["@access:L7:hold"]
        genome[genome == sid] = -1
        genome[idx] = sid
        protected_indices = owner_protected_indices(self.layout, genome)
        self.assertIn(idx, protected_indices)
        protected = self.layout.frozen_mask.copy()
        protected[list(protected_indices)] = True
        mutation = SwapMutation(layout=self.layout, frozen_mask=protected, prob=1.0)
        self.assertNotIn(idx, mutation.safe_access_positions)
        proposal = genome.copy()
        self.assertFalse(mutation._propose_l7_access(proposal))
        self.assertEqual(int(proposal[idx]), sid)
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        try:
            np.random.seed(753)
            random.seed(753)
            for _ in range(20):
                candidate = genome.copy()
                mutation._do(None, candidate[None, :])
                self.assertEqual(int(candidate[idx]), sid)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)

    def test_required_l1_shortcuts_are_hard_contract_and_gpu_constraints(self):
        self.assertTrue(audit_layout_contract(self.layout)["acceptance_pass"])
        for key in CONTRACT["required_l1_shortcuts"]:
            genome = self.layout.genome.copy()
            sid = self.sid[key]
            indices = [p.gene_idx for p in self.layout.positions
                       if int(p.layer) == 1 and int(genome[p.gene_idx]) == sid]
            self.assertTrue(indices, key)
            for idx in indices:
                genome[idx] = -1
            report = audit_layout_contract(self.layout.clone_with(genome=genome))
            self.assertFalse(report["checks"][f"required_l1_shortcut:{key}"], key)
            self.assertFalse(report["acceptance_pass"], key)
            self.assertNotEqual(self.owner_violation_code(genome), 0, key)

    def test_required_l1_shortcut_cannot_move_to_top_row_or_break_its_group(self):
        genome = self.layout.genome.copy()
        source = self.by_coord[(1, 1, 1)]
        target = self.by_coord[(1, 1, 0)]
        genome[source], genome[target] = genome[target], genome[source]
        report = audit_layout_contract(self.layout.clone_with(genome=genome))
        self.assertFalse(report["checks"]["required_l1_shortcut:Win+Tab"])
        self.assertNotEqual(self.owner_violation_code(genome), 0)

        genome = self.layout.genome.copy()
        source = self.by_coord[(1, 4, 2)]  # Ctrl++
        target = self.by_coord[(1, 11, 3)]
        genome[source], genome[target] = genome[target], genome[source]
        report = audit_layout_contract(self.layout.clone_with(genome=genome))
        self.assertFalse(report["checks"]["required_group:Ctrl+-|Ctrl++"])
        self.assertNotEqual(self.owner_violation_code(genome), 0)

    def test_required_shortcut_groups_have_atomic_mutation_moves(self):
        from evolution import build_group_placements

        placements = build_group_placements(self.layout)
        sid_by_key = self.sid
        for group in CONTRACT["required_groups"]:
            required_sids = {sid_by_key[key] for key in group}
            self.assertTrue(
                any(required_sids.issubset(set(sids)) for sids, _ in placements),
                group,
            )

    def test_mb45_mutation_moves_pair_atomically_within_l1(self):
        from evolution import SwapMutation

        protected = self.layout.frozen_mask.copy()
        protected[list(owner_protected_indices(self.layout, self.layout.genome))] = True
        mutation = SwapMutation(layout=self.layout, frozen_mask=protected, mb45_move_prob=0.0)
        moved = None
        for _ in range(30):
            candidate = self.layout.genome.copy()
            if mutation._propose_mb45_pair_move(candidate):
                moved = candidate
                break
        self.assertIsNotNone(moved, "Expected a legal MB4/MB5 pair destination")
        report = audit_layout_contract(self.layout.clone_with(genome=moved))
        self.assertTrue(report["checks"]["mb4_mb5_adjacent_right_same_row_ordered"])
        fixed = owner_protected_indices(self.layout, self.layout.genome)
        np.testing.assert_array_equal(moved[list(fixed)], self.layout.genome[list(fixed)])

    def test_invalid_child_proposal_reverts_to_valid_parent(self):
        from evolution.custom_ga import _reject_invalid_children

        parent = self.layout.genome[None, :].copy()
        children = parent.copy()
        children[0, self.by_coord[(1, 8, 2)]] = -1
        scores = np.array([[99.0]], dtype=np.float32)
        cv = np.array([[1.0]], dtype=np.float32)
        rejected = _reject_invalid_children(
            children, scores, cv, parent, np.array([[5.0]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32), np.array([[1.0]], dtype=np.float32),
        )
        self.assertTrue(bool(rejected[0]))
        np.testing.assert_array_equal(children, parent)
        self.assertEqual(float(scores[0, 0]), 5.0)
        self.assertEqual(float(cv[0, 0]), 0.0)

    def test_randomized_crossover_mutation_rejects_owner_invalid_children(self):
        import random

        from evolution import StructuralGenomeSanitizer, SwapMutation
        from evolution.custom_ga import _crossover_batch, _reject_invalid_children

        random.seed(731)
        np.random.seed(731)
        protected = self.layout.frozen_mask.copy()
        protected[list(owner_protected_indices(self.layout, self.layout.genome))] = True
        mutation = SwapMutation(layout=self.layout, frozen_mask=protected, prob=0.8)
        sanitizer = StructuralGenomeSanitizer(
            n_shortcuts=self.layout.n_shortcuts,
            frozen_mask=protected,
            seed_genome=self.layout.genome,
            layout=self.layout,
        )
        parents = np.tile(self.layout.genome, (12, 1))
        protected_indices = np.flatnonzero(protected)

        for _ in range(24):
            parent_idx = np.random.permutation(len(parents))
            fallback = parents[parent_idx].copy()
            children = _crossover_batch(
                parents, parent_idx, crossover_prob=0.9,
                n_shortcuts=self.layout.n_shortcuts,
            )
            mutation._do(None, children)
            sanitizer._do(None, children)

            # Force one owner-contract violation each trial; all other rows
            # exercise randomized crossover and mutation proposals.
            children[0, self.by_coord[(1, 8, 2)]] = -1
            owner_valid = np.asarray([
                audit_layout_contract(self.layout.clone_with(genome=row))["acceptance_pass"]
                for row in children
            ], dtype=bool)
            self.assertFalse(owner_valid[0])
            constraints = (~owner_valid).astype(np.float32)[:, None]
            scores = np.full((len(children), 1), 99.0, dtype=np.float32)
            cv = constraints.copy()
            rejected = _reject_invalid_children(
                children, scores, cv, fallback,
                np.zeros_like(scores), np.zeros_like(cv), constraints,
            )
            np.testing.assert_array_equal(rejected, ~owner_valid)
            for row in children:
                self.assertTrue(
                    audit_layout_contract(self.layout.clone_with(genome=row))["acceptance_pass"]
                )
            np.testing.assert_array_equal(children[:, protected_indices], fallback[:, protected_indices])


if __name__ == "__main__":
    unittest.main()
