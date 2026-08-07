"""FitnessEvaluator tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import numpy as np

from config import DEFAULT_CONFIG
from core import Layout, Position, Shortcut, UsageData
from evolution.acceptance import (
    _dynamic_mouse_layer_report,
    _layer7_access_report,
    _momentary_only_thumb_clearance_report,
    build_acceptance_report,
)
from fitness.evaluator import FitnessEvaluator


class TestEvaluator(unittest.TestCase):
    def test_evaluate(self):
        positions = tuple(Position(i, 0, float(i), 0.0, "left", 1, 1.0) for i in range(5))
        shortcuts = tuple(Shortcut(i, f"Key{i}", f"action{i}", "app", 5.0) for i in range(5))
        genome = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        frozen = np.array([False]*5)
        layout = Layout(genome, positions, shortcuts, frozen)

        evaluator = FitnessEvaluator()
        result = evaluator.evaluate(layout)

        self.assertEqual(result.objectives.shape, (3,))
        self.assertIn("effort", result.factor_scores)
        self.assertIn("adjacency", result.factor_scores)
        self.assertIn("violations", result.factor_scores)
        self.assertIn("workflow_coherence", result.factor_scores)
        self.assertIn("finger_balance", result.factor_scores)
        self.assertIn("same_finger", result.factor_scores)

    def test_layer_specialization_penalizes_redundant_generated_layers(self):
        positions = tuple(
            Position(i, 1 if i < 4 else 2, float(i % 4), 0.0, "left", 1, 1.0)
            for i in range(8)
        )
        shortcuts = tuple(
            Shortcut(i, f"Shortcut{i}", f"Action {i}", "same-app", 10.0, base_key=f"Key{i}")
            for i in range(8)
        )
        frozen = np.array([False] * 8)
        redundant = Layout(
            np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        distinct = Layout(
            np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 0.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_specialization": 1.0,
        }
        evaluator = FitnessEvaluator(weights=weights)

        self.assertGreater(
            evaluator.evaluate(redundant).objectives[2],
            evaluator.evaluate(distinct).objectives[2],
        )

    def test_layer_specialization_discounts_exceptional_supported_repeats(self):
        positions = tuple(
            Position(i, 1 if i < 4 else 2, float(i % 4), 0.0, "left", 1, 1.0)
            for i in range(8)
        )
        shortcuts = tuple(
            Shortcut(i, f"Shortcut{i}", f"Action {i}", "same-app", 10.0, base_key=f"Key{i}")
            for i in range(4)
        )
        frozen = np.array([False] * 8)
        genome = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        unsupported = Layout(genome, positions, shortcuts, frozen)
        usage = UsageData(shortcuts={f"Shortcut{i}": {"count": 100} for i in range(4)})
        supported = Layout(genome, positions, shortcuts, frozen, usage_data=usage)
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 0.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_specialization": 1.0,
        }
        evaluator = FitnessEvaluator(weights=weights)

        self.assertGreater(
            evaluator.evaluate(unsupported).objectives[2],
            evaluator.evaluate(supported).objectives[2],
        )

    def test_layer_similarity_allows_one_everything_layer_to_overlap_more(self):
        positions = tuple(
            Position(i, 1 + (i // 4), float(i % 4), 0.0, "left", 1, 1.0)
            for i in range(12)
        )
        shortcuts = tuple(
            Shortcut(i, f"Shortcut{i}", f"Action {i}", "app", 8.0, base_key=f"Key{i}")
            for i in range(8)
        )
        usage = UsageData(shortcuts={
            "Shortcut0": {"count": 100},
            "Shortcut1": {"count": 90},
            "Shortcut2": {"count": 80},
            "Shortcut3": {"count": 70},
        })
        frozen = np.array([False] * 12)
        overlap_with_everything = Layout(
            np.array([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        ordinary_overlap = Layout(
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 0.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_similarity": 1.0,
            "everything_layer": 0.0,
        }
        evaluator = FitnessEvaluator(weights=weights)

        self.assertLess(
            evaluator.evaluate(overlap_with_everything).objectives[2],
            evaluator.evaluate(ordinary_overlap).objectives[2],
        )

    def test_familiarity_reward_is_gated_to_exceptional_repeats(self):
        positions = tuple(
            Position(i, 1 if i < 2 else 2, float(i % 2), 0.0, "left", 1, 1.0)
            for i in range(4)
        )
        shortcuts = (
            Shortcut(0, "Shortcut0", "Action 0", "same-app", 10.0, base_key="Key0"),
            Shortcut(1, "Shortcut1", "Action 1", "same-app", 10.0, base_key="Key1"),
        )
        frozen = np.array([False] * 4)
        genome = np.array([0, 1, 0, 1], dtype=np.int32)
        unsupported = Layout(genome, positions, shortcuts, frozen)
        usage = UsageData(shortcuts={"Shortcut0": {"count": 100}, "Shortcut1": {"count": 100}})
        supported = Layout(genome, positions, shortcuts, frozen, usage_data=usage)
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 0.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 1.0,
            "layer_specialization": 0.0,
        }
        evaluator = FitnessEvaluator(weights=weights)

        self.assertLess(
            evaluator.evaluate(supported).objectives[2],
            evaluator.evaluate(unsupported).objectives[2],
        )

    def test_familiarity_uses_pairwise_exponential_distance_decay(self):
        positions = (
            Position(0, 1, 5.0, 2.0, "left", 1, 1.0),
            Position(1, 2, 5.0, 2.0, "left", 1, 1.0),
            Position(2, 2, 6.0, 2.0, "left", 1, 1.0),
            Position(3, 2, 12.0, 5.0, "right", 4, 1.0),
        )
        shortcuts = (Shortcut(0, "Shortcut0", "Action 0", "same-app", 12.0, base_key="Key0"),)
        usage = UsageData(shortcuts={"Shortcut0": {"count": 100}})
        frozen = np.array([False] * 4)
        exact = Layout(np.array([0, 0, -1, -1], dtype=np.int32), positions, shortcuts, frozen, usage_data=usage)
        near = Layout(np.array([0, -1, 0, -1], dtype=np.int32), positions, shortcuts, frozen, usage_data=usage)
        far = Layout(np.array([0, -1, -1, 0], dtype=np.int32), positions, shortcuts, frozen, usage_data=usage)
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 0.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 1.0,
            "layer_similarity": 0.0,
            "everything_layer": 0.0,
        }
        evaluator = FitnessEvaluator(weights=weights)

        exact_score = evaluator.evaluate(exact).objectives[2]
        near_score = evaluator.evaluate(near).objectives[2]
        far_score = evaluator.evaluate(far).objectives[2]
        self.assertLess(exact_score, near_score)
        self.assertLess(near_score, far_score)

    def test_app_coherence_is_backup_after_layer_redundancy(self):
        positions = tuple(
            Position(i, 1 if i < 4 else 2, float(i % 4), 0.0, "left", 1, 1.0)
            for i in range(8)
        )
        shortcuts = tuple(
            Shortcut(i, f"Shortcut{i}", f"Action {i}", "same-app", 10.0, base_key=f"Key{i}")
            for i in range(4)
        )
        frozen = np.array([False] * 8)
        non_redundant = Layout(
            np.array([0, 1, 2, 3, -1, -1, -1, -1], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        redundant = Layout(
            np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 0.0,
            "workflow_coherence": 0.0,
            "app_coherence": 1.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_similarity": 0.0,
            "layer_specialization": 0.0,
            "everything_layer": 0.0,
        }
        evaluator = FitnessEvaluator(weights=weights)

        self.assertAlmostEqual(float(evaluator.evaluate(non_redundant).objectives[2]), 0.0, places=5)
        self.assertLess(float(evaluator.evaluate(redundant).objectives[2]), 0.0)

    def test_raw_completion_usage_prefers_more_accessible_anchor_layer(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 1, 8.0, 1.0, "right", 1, 1.0),
            Position(2, 1, 9.0, 1.0, "right", 1, 1.0),
            Position(3, 1, 10.0, 1.0, "right", 1, 1.0),
            Position(4, 1, 11.0, 1.0, "right", 1, 1.0),
            Position(5, 3, 8.0, 1.0, "right", 1, 1.0),
            Position(6, 3, 9.0, 1.0, "right", 1, 1.0),
            Position(7, 3, 10.0, 1.0, "right", 1, 1.0),
            Position(8, 3, 11.0, 1.0, "right", 1, 1.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L1:hold:Backup", "Backup", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=1,
                access_is_momentary=True,
            ),
            Shortcut(1, "-", "Dash", "Raw Keys", 3.0, "raw_completion", base_key="Dash and Underscore"),
            Shortcut(2, "=", "Equals", "Raw Keys", 3.0, "raw_completion", base_key="Equals and Plus"),
            Shortcut(3, "`", "Grave", "Raw Keys", 3.0, "raw_completion", base_key="Grave Accent and Tilde"),
            Shortcut(4, "]", "Right Brace", "Raw Keys", 3.0, "raw_completion", base_key="Right Brace"),
        )
        usage = UsageData(raw_completion_keys={
            "Dash and Underscore": {"count": 20},
            "Equals and Plus": {"count": 20},
            "Grave Accent and Tilde": {"count": 20},
            "Right Brace": {"count": 20},
        }, raw_completion_total=80)
        frozen = np.zeros(len(positions), dtype=np.bool_)
        accessible = Layout(
            np.array([0, 1, 2, 3, 4, -1, -1, -1, -1], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        inaccessible = Layout(
            np.array([0, -1, -1, -1, -1, 1, 2, 3, 4], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 1.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_specialization": 0.0,
        }
        vweights = {
            "duplicate": 0.0,
            "l0_displacement": 0.0,
            "missing_important": 0.0,
            "cross_layer_duplicate": 0.0,
            "group_split": 0.0,
            "thumb_occupancy": 0.0,
            "arrow_order": 0.0,
            "hand_bias": 0.0,
            "mouse_layer_access": 0.0,
            "arrow_scattered": 0.0,
            "mouse_scattered": 0.0,
            "layer7_access": 0.0,
            "duplicate_value_gap": 0.0,
            "access_layout": 0.0,
            "raw_keyboard_completion_norwegian": 1.0,
            "dynamic_mouse_layer": 0.0,
        }
        evaluator = FitnessEvaluator(
            weights=weights,
            reference_layout=accessible,
            violation_weights=vweights,
            hard_constraints=[],
            missing_important_threshold=99.0,
        )

        self.assertLess(
            evaluator.evaluate(accessible).objectives[2],
            evaluator.evaluate(inaccessible).objectives[2],
        )

    def test_workflow_coherence(self):
        """Test that splitting workflows across layers incurs penalty."""
        positions = (
            Position(0, 1, 0.0, 0.0, "left", 1, 1.0),   # L1
            Position(1, 1, 1.0, 0.0, "left", 1, 1.0),   # L1
            Position(2, 2, 0.0, 0.0, "left", 1, 1.0),   # L2
        )
        shortcuts = (
            Shortcut(0, "Ctrl+C", "Copy", "app", 10.0),
            Shortcut(1, "Ctrl+V", "Paste", "app", 10.0),
        )

        # Both on same layer - no penalty
        genome1 = np.array([0, 1, -1], dtype=np.int32)
        frozen = np.array([False, False, False])
        layout1 = Layout(genome1, positions, shortcuts, frozen)

        from fitness.factors.workflow_coherence import WorkflowCoherenceFactor
        factor = WorkflowCoherenceFactor()
        penalty1 = factor.compute(layout1)
        self.assertEqual(penalty1, 0.0)

        # Split across layers - penalty
        genome2 = np.array([0, -1, 1], dtype=np.int32)
        # Add usage data with a chain
        usage = UsageData(chains={"Ctrl+C -> Ctrl+V": {"count": 5}})
        layout2 = Layout(genome2, positions, shortcuts, frozen, usage_data=usage)
        penalty2 = factor.compute(layout2)
        self.assertGreater(penalty2, 0.0)
        # Penalty = 5 * 10.0 = 50
        self.assertEqual(penalty2, 50.0)

    def test_thumb_occupancy(self):
        """Test that shortcuts on occupied thumb positions get penalized."""
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),   # L0 left thumb access
            Position(1, 1, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),   # L1 left thumb (occupied)
            Position(2, 1, 7.0, 4.0, "right", 0, 0.8, is_thumb=True),  # L1 right thumb (free)
            Position(3, 1, 0.0, 0.0, "left", 1, 1.0),                  # L1 finger
        )
        shortcuts = (
            Shortcut(0, "Ctrl+A", "Select All", "app", 10.0),
            Shortcut(1, "Ctrl+C", "Copy", "app", 8.0),
            Shortcut(
                2, "@access:L0->L1:hold:Nav", "Nav", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=1,
                access_is_momentary=True,
            ),
        )
        genome = np.array([2, 0, -1, -1], dtype=np.int32)
        frozen = np.array([False, False, False, False])

        layout = Layout(genome, positions, shortcuts, frozen)

        from tests.legacy_factors import ViolationFactor
        factor = ViolationFactor()
        penalty = factor._thumb_occupancy(layout)

        # Shortcut on left thumb should be penalized
        self.assertGreater(penalty, 0.0)

        # Move shortcut to right thumb (free) - penalty should be 0
        genome2 = np.array([2, -1, 0, -1], dtype=np.int32)
        layout2 = Layout(genome2, positions, shortcuts, frozen)
        penalty2 = factor._thumb_occupancy(layout2)
        self.assertEqual(penalty2, 0.0)

        # Remove layer access - no penalty even on left thumb
        genome3 = np.array([-1, 0, -1, -1], dtype=np.int32)
        layout3 = Layout(genome3, positions, shortcuts, frozen)
        penalty3 = factor._thumb_occupancy(layout3)
        self.assertEqual(penalty3, 0.0)

    def test_compiled_occupied_thumb_penalizes_left_momentary_path(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 1, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(2, 1, 7.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(3, 7, 6.0, 4.0, "right", 0, 0.8, is_thumb=True),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L1:hold:Nav", "Nav", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=1,
                access_is_momentary=True,
            ),
            Shortcut(1, "Ctrl+A", "Select All", "app", 10.0),
            Shortcut(
                2, "@access:L0->L7:toggle:Game", "Game", "Layer Access", 8.0,
                "layer_access", is_layer_access=True, access_target_layer=7,
                access_is_momentary=False,
            ),
        )
        frozen = np.array([False, False, False, False])
        left_blocked = Layout(np.array([0, 1, -1, 2], dtype=np.int32), positions, shortcuts, frozen)
        right_free = Layout(np.array([0, -1, 1, 2], dtype=np.int32), positions, shortcuts, frozen)
        weights = {
            "effort": 0.0, "adjacency": 0.0, "finger_balance": 0.0,
            "same_finger": 0.0, "violations": 1.0, "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0, "familiarity": 0.0,
        }
        vweights = {"thumb_occupancy": 1000.0, "access_layout": 0.0}
        evaluator = FitnessEvaluator(
            weights=weights,
            reference_layout=left_blocked,
            violation_weights=vweights,
            hard_constraints=[],
            missing_important_threshold=99.0,
        )
        left_score = evaluator.evaluate(left_blocked).objectives[2]
        right_score = evaluator.evaluate(right_free).objectives[2]
        self.assertGreater(left_score, right_score)

    def test_dynamic_mouse_layer_requires_buttons_scroll_and_access(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(8, 4, 2.0, 2.0, "left", 1, 1.0),
            Position(2, 3, 8.0, 1.0, "right", 1, 1.0),
            Position(3, 3, 9.0, 1.0, "right", 1, 1.0),
            Position(4, 3, 10.0, 1.0, "right", 2, 1.0),
            Position(5, 3, 11.0, 1.0, "right", 3, 1.0),
            Position(6, 3, 12.0, 1.0, "right", 4, 1.2),
            Position(7, 3, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(8, 3, 9.0, 2.0, "right", 1, 1.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L3:hold:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L3:toggle:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(3, "MB2", "Click", "Mouse", 15.0, "mouse"),
            Shortcut(4, "MB3", "Click", "Mouse", 10.0, "mouse"),
            Shortcut(5, "MB4", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(6, "MB5", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(
                7, "@access:L3->L6:hold:Scroll", "Scroll", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=6,
                access_is_momentary=True,
            ),
            Shortcut(
                8, "@access:L4->L3:toggle:Mouse", "Mouse", "Layer Access", 10.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(
                9, "@access:L0->L4:hold:Source", "Source", "Layer Access", 10.0,
                "layer_access", is_layer_access=True, access_target_layer=4,
                access_is_momentary=True,
            ),
            Shortcut(
                10, "@access:L0->L3:hold:MouseRightThumb", "Mouse", "Layer Access", 10.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        layout = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        report = _dynamic_mouse_layer_report(layout)
        self.assertTrue(report["acceptance_pass"])
        self.assertEqual(report["mouse_layer"], 3)

        uncomfortable_scroll = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            tuple(
                Position(
                    p.gene_idx,
                    p.layer,
                    8.0 if i == 9 else p.x,
                    p.y,
                    p.hand,
                    p.finger,
                    p.effort,
                    is_thumb=p.is_thumb,
                    is_frozen=p.is_frozen,
                )
                for i, p in enumerate(positions)
            ),
            shortcuts,
            frozen,
        )
        uncomfortable_report = _dynamic_mouse_layer_report(uncomfortable_scroll)
        self.assertFalse(uncomfortable_report["acceptance_pass"])
        self.assertTrue(uncomfortable_report["best_candidate"]["uncomfortable_right_momentary_scroll_access"])

        missing_toggle = Layout(
            np.array([0, -1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertFalse(_dynamic_mouse_layer_report(missing_toggle)["acceptance_pass"])

        self_toggle_on_mouse_layer = Layout(
            np.array([0, -1, -1, 2, 3, 4, 5, 6, 1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        # A toggle shortcut placed ON the mouse layer itself (source layer ==
        # target layer) can only ever be pressed once the user is already on
        # that layer via some other entry path -- it is not a real external
        # toggle-in and must NOT satisfy the reachable-toggle-access
        # requirement, even though the shortcut's own label still says
        # "L0->L3". Placement determines the real source layer, not the label.
        self_toggle_report = _dynamic_mouse_layer_report(self_toggle_on_mouse_layer)
        self.assertFalse(self_toggle_report["acceptance_pass"])
        self.assertFalse(self_toggle_report["best_candidate"]["reachable_toggle_access"])

        no_momentary_access = Layout(
            np.array([-1, 1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertFalse(_dynamic_mouse_layer_report(no_momentary_access)["acceptance_pass"])
        self.assertFalse(
            _dynamic_mouse_layer_report(no_momentary_access)["best_candidate"]["direct_l0_momentary_access"]
        )

        # A momentary-hold shortcut placed ON the mouse layer itself (source
        # layer == target layer, e.g. sid=10's own "L0->L3" label but placed
        # at position 8, which is layer 3's right thumb) can only be pressed
        # once already on that layer via some other entry path. It must NOT
        # count as disqualifying right-thumb momentary access into the mouse
        # layer, even though it sits on the right thumb, or it would produce
        # exactly the false-negative found in the gen-29000 checkpoint audit
        # on 2026-07-13 (a genuinely valid mouse layer failing acceptance
        # solely due to a self-referential momentary hold on right thumb).
        self_momentary_on_mouse_layer = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, 10, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self_momentary_report = _dynamic_mouse_layer_report(self_momentary_on_mouse_layer)
        self.assertTrue(self_momentary_report["acceptance_pass"])
        self.assertFalse(self_momentary_report["best_candidate"]["right_thumb_momentary_access"])

        non_l0_toggle = Layout(
            np.array([0, 9, 8, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertTrue(_dynamic_mouse_layer_report(non_l0_toggle)["acceptance_pass"])

        extra_duplicate_elsewhere = Layout(
            np.array([0, 1, 2, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertTrue(_dynamic_mouse_layer_report(extra_duplicate_elsewhere)["acceptance_pass"])

        right_thumb_button = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, -1, 6, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        thumb_report = _dynamic_mouse_layer_report(right_thumb_button)
        self.assertFalse(thumb_report["acceptance_pass"])
        self.assertTrue(thumb_report["best_candidate"]["right_thumb_button_placements"])

        right_thumb_scroll = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, 7, -1], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        scroll_report = _dynamic_mouse_layer_report(right_thumb_scroll)
        self.assertFalse(scroll_report["acceptance_pass"])
        self.assertTrue(scroll_report["best_candidate"]["right_thumb_momentary_scroll_access"])

        right_thumb_momentary_access = Layout(
            np.array([-1, 10, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        access_report = _dynamic_mouse_layer_report(right_thumb_momentary_access)
        self.assertFalse(access_report["acceptance_pass"])
        self.assertTrue(access_report["best_candidate"]["right_thumb_momentary_access"])

        global_right_thumb_mouse = build_acceptance_report(
            right_thumb_button,
            duplicate_report={"unsupported_duplicates": []},
            completion_cluster_report={"acceptance_pass": True},
            arrow_report={"acceptance_pass": True},
        )
        self.assertFalse(
            global_right_thumb_mouse["optimizer_side_checks"]["no_mouse_buttons_on_right_thumb_area_global"]
        )
        self.assertIn(
            "global_right_thumb_mouse_button_count",
            global_right_thumb_mouse["numeric_distances"],
        )

    def test_dynamic_mouse_layer_report_same_side_duplicate_policy(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(2, 3, 8.0, 1.0, "right", 1, 1.0),
            Position(3, 3, 9.0, 1.0, "right", 1, 1.0),
            Position(4, 3, 10.0, 1.0, "right", 2, 1.0),
            Position(5, 3, 11.0, 1.0, "right", 3, 1.0),
            Position(6, 3, 12.0, 1.0, "right", 4, 1.2),
            Position(7, 3, 9.0, 2.0, "right", 1, 1.0),
            Position(8, 3, 2.0, 1.0, "left", 1, 1.0),
            Position(9, 3, 13.0, 1.0, "right", 4, 1.5),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L3:hold:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L3:toggle:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(3, "MB2", "Click", "Mouse", 15.0, "mouse"),
            Shortcut(4, "MB3", "Click", "Mouse", 10.0, "mouse"),
            Shortcut(5, "MB4", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(6, "MB5", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(
                7, "@access:L3->L6:hold:Scroll", "Scroll", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=6,
                access_is_momentary=True,
            ),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        single_right = Layout(
            np.array([0, 1, 2, 3, 4, 5, 6, 7, -1, -1], dtype=np.int32),
            positions, shortcuts, frozen,
        )
        self.assertTrue(_dynamic_mouse_layer_report(single_right)["acceptance_pass"])

        left_right_pair = Layout(
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 2, -1], dtype=np.int32),
            positions, shortcuts, frozen,
        )
        self.assertTrue(_dynamic_mouse_layer_report(left_right_pair)["acceptance_pass"])

        two_right_copies = Layout(
            np.array([0, 1, 2, 3, 4, 5, 6, 7, -1, 2], dtype=np.int32),
            positions, shortcuts, frozen,
        )
        two_right_report = _dynamic_mouse_layer_report(two_right_copies)
        self.assertFalse(two_right_report["acceptance_pass"])
        self.assertIn("MB1", two_right_report["best_candidate"]["duplicate_same_side_buttons"])

        unpaired_left_only = Layout(
            np.array([0, 1, -1, 3, 4, 5, 6, 7, 2, -1], dtype=np.int32),
            positions, shortcuts, frozen,
        )
        unpaired_report = _dynamic_mouse_layer_report(unpaired_left_only)
        self.assertFalse(unpaired_report["acceptance_pass"])
        self.assertIn("MB1", unpaired_report["best_candidate"]["unpaired_left_buttons"])

    def test_layer7_acceptance_checks_access_modes_only(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 4.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(2, 7, 5.0, 4.0, "left", 0, 0.8, is_thumb=True, is_frozen=True),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L7:hold:Fallback", "Fallback", "Layer Access", 8.0,
                "layer_access", is_layer_access=True, access_target_layer=7,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L7:toggle:Fallback", "Fallback", "Layer Access", 8.0,
                "layer_access", is_layer_access=True, access_target_layer=7,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Mouse 1", "Mouse", 8.0, "mouse"),
        )
        frozen = np.array([False, False, True])

        only_toggle = Layout(np.array([-1, 1, 2], dtype=np.int32), positions, shortcuts, frozen)
        both_modes = Layout(np.array([0, 1, 2], dtype=np.int32), positions, shortcuts, frozen)

        self.assertFalse(_layer7_access_report(only_toggle)["acceptance_pass"])
        report = _layer7_access_report(both_modes)
        self.assertTrue(report["acceptance_pass"])
        self.assertFalse(report["content_checked"])

    def test_dynamic_mouse_layer_penalty_rewards_natural_complete_layer(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(2, 3, 2.0, 1.0, "left", 1, 1.0),
            # Mouse-button/scroll positions use effort=0.0: this is the
            # "complete AND optimally-placed" natural mouse layer, so its
            # own candidate_penalty should be near zero. Before the
            # dynamic_mouse_layer fix, these carried effort=1.0 and the
            # test still passed by accident, because the min()-across-
            # layers bug capped every scenario at an unrelated empty
            # candidate layer's flat baseline instead of the natural
            # layer's real (inflated) penalty.
            Position(3, 3, 8.0, 2.0, "right", 1, 0.0),
            Position(4, 3, 9.0, 2.0, "right", 1, 0.0),
            # MB3/MB4/MB5 and Scroll ideal x-targets: MB3=11, MB4=12, MB5=13,
            # Scroll=10 -- Scroll's ideal sits one slot right of MB2 (not on
            # top of it), so MB1(x8)/MB2(x9) can land adjacent.
            Position(5, 3, 11.0, 2.0, "right", 2, 0.0),
            Position(6, 3, 12.0, 2.0, "right", 3, 0.0),
            Position(7, 3, 13.0, 2.0, "right", 4, 0.0),
            Position(8, 3, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(9, 3, 10.0, 2.0, "right", 1, 0.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L3:hold:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L3:toggle:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(3, "MB2", "Click", "Mouse", 15.0, "mouse"),
            Shortcut(4, "MB3", "Click", "Mouse", 10.0, "mouse"),
            Shortcut(5, "MB4", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(6, "MB5", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(
                7, "@access:L3->L6:hold:Scroll", "Scroll", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=6,
                access_is_momentary=True,
            ),
            Shortcut(
                8, "@access:L0->L3:hold:MouseRightThumb", "Mouse", "Layer Access", 10.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        valid = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        missing_toggle = Layout(
            np.array([0, -1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        left_button = Layout(
            np.array([0, 1, 2, -1, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 1.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_specialization": 0.0,
        }
        vweights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
        vweights["dynamic_mouse_layer"] = 1.0
        evaluator = FitnessEvaluator(
            weights=weights,
            reference_layout=valid,
            violation_weights=vweights,
            hard_constraints=[],
            missing_important_threshold=99.0,
        )
        valid_score = evaluator.evaluate(valid).objectives[2]
        self.assertGreater(evaluator.evaluate(missing_toggle).objectives[2], valid_score)
        self.assertGreater(evaluator.evaluate(left_button).objectives[2], valid_score)

        right_thumb_button = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, -1, 6, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertGreater(evaluator.evaluate(right_thumb_button).objectives[2], valid_score)

        right_thumb_scroll = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, 7, -1], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertGreater(evaluator.evaluate(right_thumb_scroll).objectives[2], valid_score)

        right_thumb_momentary_access = Layout(
            np.array([0, 8, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertGreater(evaluator.evaluate(right_thumb_momentary_access).objectives[2], valid_score)

        uncomfortable_scroll = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            tuple(
                Position(
                    p.gene_idx,
                    p.layer,
                    8.0 if i == 9 else p.x,
                    p.y,
                    p.hand,
                    p.finger,
                    p.effort,
                    is_thumb=p.is_thumb,
                    is_frozen=p.is_frozen,
                )
                for i, p in enumerate(positions)
            ),
            shortcuts,
            frozen,
        )
        self.assertGreater(evaluator.evaluate(uncomfortable_scroll).objectives[2], valid_score)

        # Regression test for the min()-across-all-layers decoupling bug: once
        # a natural_mouse_layer exists (all Pass-1 qualifying conditions
        # still hold -- unlike `uncomfortable_scroll` above, which moves
        # Scroll onto x=7/8 and so *disqualifies* the layer from being
        # natural_mouse_layer entirely, never exercising the override path),
        # dynamic_mouse_layer must track THAT layer's own candidate_penalty,
        # not the flat baseline of an unrelated near-empty candidate layer
        # (~338,000, i.e. missing_buttons(5)*50000 + no-scroll(25000) +
        # no-button-no-scroll(30000) + no-toggle(25000) + no-safe-
        # momentary(8000)). Before the fix, a natural mouse layer with a
        # badly-placed-but-still-qualifying Scroll (e.g. far from its ideal
        # y-row) still scored BELOW that baseline because dynamic_mouse_layer
        # capped out at the empty layer's minimum instead of reporting the
        # real, much larger penalty -- silently making further refinement of
        # the natural layer invisible to the scoring gradient.
        empty_layer_baseline = 338000.0
        bad_scroll_row_positions = tuple(
            Position(
                p.gene_idx,
                p.layer,
                p.x,
                4.0 if i == 9 else p.y,
                p.hand,
                p.finger,
                p.effort,
                is_thumb=p.is_thumb,
                is_frozen=p.is_frozen,
            )
            for i, p in enumerate(positions)
        )
        bad_scroll_row = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            bad_scroll_row_positions,
            shortcuts,
            frozen,
        )
        bad_scroll_row_score = evaluator.evaluate(bad_scroll_row).objectives[2]
        self.assertGreater(bad_scroll_row_score, empty_layer_baseline)
        self.assertGreater(bad_scroll_row_score, valid_score)

        even_worse_scroll_row_positions = tuple(
            Position(
                p.gene_idx,
                p.layer,
                p.x,
                6.0 if i == 9 else p.y,
                p.hand,
                p.finger,
                p.effort,
                is_thumb=p.is_thumb,
                is_frozen=p.is_frozen,
            )
            for i, p in enumerate(positions)
        )
        even_worse_scroll_row = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            even_worse_scroll_row_positions,
            shortcuts,
            frozen,
        )
        # Degrading the natural layer's own Scroll placement further must
        # keep making the score strictly worse -- proving the objective
        # keeps responding to changes on the natural layer itself, rather
        # than being frozen at whatever unrelated empty layer's baseline
        # happened to be lowest.
        self.assertGreater(
            evaluator.evaluate(even_worse_scroll_row).objectives[2],
            bad_scroll_row_score,
        )

    def test_dynamic_mouse_layer_mb2_and_scroll_ideal_x_do_not_collide(self):
        # Regression test: MB2's ideal x (9) and Scroll's ideal x used to be
        # identical (both 9), so Scroll's much larger weight always evicted
        # MB2 from that slot, forcing a gap between MB1(x8) and MB2 filled by
        # whatever else won the fight (e.g. MB3). Scroll's ideal is now one
        # slot further right (10) so MB1/MB2 can land adjacent without
        # Scroll's placement pressure fighting over the same coordinate.
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(2, 3, 2.0, 1.0, "left", 1, 1.0),
            Position(3, 3, 8.0, 2.0, "right", 1, 0.0),
            Position(4, 3, 9.0, 2.0, "right", 1, 0.0),
            Position(5, 3, 11.0, 2.0, "right", 2, 0.0),
            Position(6, 3, 12.0, 2.0, "right", 3, 0.0),
            Position(7, 3, 13.0, 2.0, "right", 4, 0.0),
            Position(8, 3, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(9, 3, 10.0, 2.0, "right", 1, 0.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L3:hold:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L3:toggle:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(3, "MB2", "Click", "Mouse", 15.0, "mouse"),
            Shortcut(4, "MB3", "Click", "Mouse", 10.0, "mouse"),
            Shortcut(5, "MB4", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(6, "MB5", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(
                7, "@access:L3->L6:hold:Scroll", "Scroll", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=6,
                access_is_momentary=True,
            ),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        # MB1@8, MB2@9 adjacent, Scroll@13 (uncontested, off to the side).
        mb2_adjacent = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32), positions, shortcuts, frozen,
        )
        # MB1@8, Scroll@9 (taking MB2's old slot), MB2 displaced to x=13.
        mb2_displaced_by_scroll = Layout(
            np.array([0, 1, -1, 2, 7, 4, 5, 6, -1, 3], dtype=np.int32), positions, shortcuts, frozen,
        )
        weights = {
            "effort": 0.0, "adjacency": 0.0, "finger_balance": 0.0, "same_finger": 0.0,
            "violations": 1.0, "workflow_coherence": 0.0, "app_coherence": 0.0,
            "trackball_proximity": 0.0, "familiarity": 0.0, "layer_specialization": 0.0,
        }
        vweights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
        vweights["dynamic_mouse_layer"] = 1.0
        evaluator = FitnessEvaluator(weights=weights, reference_layout=mb2_adjacent, violation_weights=vweights,
                                     hard_constraints=[], missing_important_threshold=99.0)
        self.assertLess(
            evaluator.evaluate(mb2_adjacent).objectives[2],
            evaluator.evaluate(mb2_displaced_by_scroll).objectives[2],
        )

    def test_dynamic_mouse_layer_penalty_uses_mouse_usage_for_placement(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(2, 3, 8.0, 1.0, "right", 1, 0.6),
            Position(3, 3, 9.0, 1.0, "right", 2, 2.8),
            Position(4, 3, 10.0, 1.0, "right", 2, 1.0),
            Position(5, 3, 11.0, 1.0, "right", 3, 1.0),
            Position(6, 3, 12.0, 1.0, "right", 4, 1.2),
            Position(7, 3, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(8, 3, 9.0, 2.0, "right", 1, 1.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L3:hold:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L3:toggle:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(3, "MB2", "Click", "Mouse", 15.0, "mouse"),
            Shortcut(4, "MB3", "Click", "Mouse", 10.0, "mouse"),
            Shortcut(5, "MB4", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(6, "MB5", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(
                7, "@access:L3->L6:hold:Scroll", "Scroll", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=6,
                access_is_momentary=True,
            ),
        )
        usage = UsageData(mouse_clicks={"MB1": {"count": 100}, "MB2": {"count": 1}})
        frozen = np.zeros(len(positions), dtype=np.bool_)
        better = Layout(
            np.array([0, 1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        worse = Layout(
            np.array([0, 1, 3, 2, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 1.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_specialization": 0.0,
        }
        vweights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
        vweights["dynamic_mouse_layer"] = 1.0
        evaluator = FitnessEvaluator(
            weights=weights,
            reference_layout=better,
            violation_weights=vweights,
            hard_constraints=[],
            missing_important_threshold=99.0,
        )
        self.assertLess(evaluator.evaluate(better).objectives[2], evaluator.evaluate(worse).objectives[2])

    def test_dynamic_mouse_layer_penalty_prefers_mb1_and_scroll_home_row(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 4.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(2, 3, 8.0, 2.0, "right", 1, 0.0),
            Position(3, 3, 8.0, 1.0, "right", 1, 1.0),
            Position(4, 3, 9.0, 2.0, "right", 2, 0.0),
            Position(5, 3, 10.0, 2.0, "right", 2, 0.0),
            Position(6, 3, 10.0, 3.0, "right", 2, 1.0),
            Position(7, 3, 11.0, 2.0, "right", 3, 0.2),
            Position(8, 3, 12.0, 2.0, "right", 4, 0.5),
        )
        shortcuts = (
            Shortcut(0, "@access:L0->L3:hold:Mouse", "Mouse", "Layer Access", 16.0,
                     "layer_access", is_layer_access=True, access_target_layer=3, access_is_momentary=True),
            Shortcut(1, "@access:L0->L3:toggle:Mouse", "Mouse", "Layer Access", 16.0,
                     "layer_access", is_layer_access=True, access_target_layer=3, access_is_momentary=False),
            Shortcut(2, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(3, "MB2", "Click", "Mouse", 18.0, "mouse"),
            Shortcut(4, "MB3", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(5, "MB4", "Click", "Mouse", 6.0, "mouse"),
            Shortcut(6, "MB5", "Click", "Mouse", 6.0, "mouse"),
            Shortcut(7, "@access:L3->L6:hold:Scroll", "Scroll", "Layer Access", 18.0,
                     "layer_access", is_layer_access=True, access_target_layer=6, access_is_momentary=True),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        better = Layout(np.array([0, 1, 2, -1, 3, 7, -1, 4, 5], dtype=np.int32), positions, shortcuts, frozen)
        worse = Layout(np.array([0, 1, -1, 2, 3, -1, 7, 4, 5], dtype=np.int32), positions, shortcuts, frozen)
        weights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["weights"]}
        weights["violations"] = 1.0
        vweights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
        vweights["dynamic_mouse_layer"] = 1.0
        evaluator = FitnessEvaluator(weights=weights, reference_layout=better, violation_weights=vweights,
                                     hard_constraints=[], missing_important_threshold=99.0)
        self.assertLess(evaluator.evaluate(better).objectives[2], evaluator.evaluate(worse).objectives[2])
        mb2_better = Layout(np.array([0, 1, 2, -1, 3, 7, -1, 4, 5], dtype=np.int32), positions, shortcuts, frozen)
        mb2_worse = Layout(np.array([0, 1, 2, -1, 4, 7, -1, 3, 5], dtype=np.int32), positions, shortcuts, frozen)
        self.assertLess(evaluator.evaluate(mb2_better).objectives[2], evaluator.evaluate(mb2_worse).objectives[2])
        scroll_better_than_mb3 = Layout(
            np.array([0, 1, 2, -1, 7, 3, 4, 5, 6], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        scroll_worse_than_mb3 = Layout(
            np.array([0, 1, 2, -1, 4, 3, 7, 5, 6], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertLess(
            evaluator.evaluate(scroll_better_than_mb3).objectives[2],
            evaluator.evaluate(scroll_worse_than_mb3).objectives[2],
        )

    def test_dynamic_mouse_layer_inner_group_order_enforced(self):
        # Reuses the same "complete AND optimally-placed" natural-mouse-layer
        # fixture as test_dynamic_mouse_layer_penalty_rewards_natural_complete_layer
        # so natural_mouse_layer actually settles on layer 3 -- otherwise the
        # min()-across-candidate-layers baseline masks the pair-order penalty.
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(2, 3, 2.0, 1.0, "left", 1, 1.0),
            Position(3, 3, 8.0, 2.0, "right", 1, 0.0),
            Position(4, 3, 9.0, 2.0, "right", 1, 0.0),
            Position(5, 3, 10.0, 2.0, "right", 2, 0.0),
            Position(6, 3, 11.0, 2.0, "right", 3, 0.0),
            Position(7, 3, 12.0, 2.0, "right", 4, 0.0),
            Position(8, 3, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(9, 3, 9.0, 2.0, "right", 1, 0.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L3:hold:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L3:toggle:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(3, "MB2", "Click", "Mouse", 15.0, "mouse"),
            Shortcut(4, "MB3", "Click", "Mouse", 10.0, "mouse"),
            Shortcut(5, "MB4", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(6, "MB5", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(
                7, "@access:L3->L6:hold:Scroll", "Scroll", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=6,
                access_is_momentary=True,
            ),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        # MB1 left of MB2 (correct inner-group order) vs MB2 left of MB1
        # (reversed) with every other placement identical.
        correct_pair12 = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32), positions, shortcuts, frozen,
        )
        reversed_pair12 = Layout(
            np.array([0, 1, -1, 3, 2, 4, 5, 6, -1, 7], dtype=np.int32), positions, shortcuts, frozen,
        )
        weights = {
            "effort": 0.0, "adjacency": 0.0, "finger_balance": 0.0, "same_finger": 0.0,
            "violations": 1.0, "workflow_coherence": 0.0, "app_coherence": 0.0,
            "trackball_proximity": 0.0, "familiarity": 0.0, "layer_specialization": 0.0,
        }
        vweights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
        vweights["dynamic_mouse_layer"] = 1.0
        evaluator = FitnessEvaluator(weights=weights, reference_layout=correct_pair12, violation_weights=vweights,
                                     hard_constraints=["mouse_button_order"], missing_important_threshold=99.0)
        correct_score = evaluator.evaluate(correct_pair12).objectives[2]
        reversed_score = evaluator.evaluate(reversed_pair12).objectives[2]
        self.assertGreater(reversed_score, correct_score + 100000.0)
        self.assertEqual(float(evaluator.evaluate(correct_pair12).constraints[0]), 0.0)
        self.assertGreater(float(evaluator.evaluate(reversed_pair12).constraints[0]), 0.0)

        # MB4 left of MB5 (correct) vs MB5 left of MB4 (reversed), holding
        # MB1/MB2 order fixed and identical across both variants.
        correct_pair45 = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32), positions, shortcuts, frozen,
        )
        reversed_pair45 = Layout(
            np.array([0, 1, -1, 2, 3, 4, 6, 5, -1, 7], dtype=np.int32), positions, shortcuts, frozen,
        )
        correct45_score = evaluator.evaluate(correct_pair45).objectives[2]
        reversed45_score = evaluator.evaluate(reversed_pair45).objectives[2]
        self.assertGreater(reversed45_score, correct45_score + 50000.0)

    def test_access_layout_penalizes_non_thumb_return_to_l0_toggle(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.3, is_thumb=True),
            Position(1, 1, 8.0, 2.0, "right", 1, 0.0),
            Position(2, 1, 5.0, 4.0, "left", 0, 0.4, is_thumb=True),
        )
        shortcuts = (
            Shortcut(0, "@access:L0->L1:hold:One", "One", "Layer Access", 12.0,
                     "layer_access", is_layer_access=True, access_target_layer=1, access_is_momentary=True),
            Shortcut(1, "@access:L1->L0:toggle:Back", "Back", "Layer Access", 12.0,
                     "layer_access", is_layer_access=True, access_target_layer=0, access_is_momentary=False),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        non_thumb_return = Layout(np.array([0, 1, -1], dtype=np.int32), positions, shortcuts, frozen)
        thumb_return = Layout(np.array([0, -1, 1], dtype=np.int32), positions, shortcuts, frozen)
        weights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["weights"]}
        weights["violations"] = 1.0
        vweights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
        vweights["access_layout"] = 1.0
        evaluator = FitnessEvaluator(weights=weights, reference_layout=thumb_return, violation_weights=vweights,
                                     hard_constraints=[], missing_important_threshold=99.0)
        self.assertGreater(
            evaluator.evaluate(non_thumb_return).objectives[2],
            evaluator.evaluate(thumb_return).objectives[2] + 400000.0,
        )

    def test_access_layout_penalizes_redundant_l0_access_and_nested_depth(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.3, is_thumb=True),
            Position(1, 0, 5.0, 4.0, "left", 0, 0.4, is_thumb=True),
            Position(2, 1, 8.0, 2.0, "right", 1, 0.0),
            Position(3, 2, 9.0, 2.0, "right", 1, 0.0),
            Position(4, 3, 10.0, 2.0, "right", 1, 0.0),
        )
        shortcuts = (
            Shortcut(0, "@access:L0->L1:hold:One", "One", "Layer Access", 12.0,
                     "layer_access", is_layer_access=True, access_target_layer=1, access_is_momentary=True),
            Shortcut(1, "@access:L0->L1:toggle:One", "One", "Layer Access", 12.0,
                     "layer_access", is_layer_access=True, access_target_layer=1, access_is_momentary=False),
            Shortcut(2, "@access:L1->L2:hold:Two", "Two", "Layer Access", 12.0,
                     "layer_access", is_layer_access=True, access_target_layer=2, access_is_momentary=True),
            Shortcut(3, "@access:L0->L2:hold:Two", "Two", "Layer Access", 12.0,
                     "layer_access", is_layer_access=True, access_target_layer=2, access_is_momentary=True),
            Shortcut(4, "@access:L2->L3:hold:Three", "Three", "Layer Access", 12.0,
                     "layer_access", is_layer_access=True, access_target_layer=3, access_is_momentary=True),
            Shortcut(5, "@access:L0->L3:hold:Three", "Three", "Layer Access", 12.0,
                     "layer_access", is_layer_access=True, access_target_layer=3, access_is_momentary=True),
            Shortcut(6, "Ctrl+A", "Action", "App", 10.0),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        redundant = Layout(np.array([0, 1, 6, -1, -1], dtype=np.int32), positions, shortcuts, frozen)
        single_direct = Layout(np.array([0, -1, 6, -1, -1], dtype=np.int32), positions, shortcuts, frozen)
        toggle_only = Layout(np.array([1, -1, 6, -1, -1], dtype=np.int32), positions, shortcuts, frozen)
        nested = Layout(np.array([0, -1, 2, 6, -1], dtype=np.int32), positions, shortcuts, frozen)
        direct = Layout(np.array([3, -1, -1, 6, -1], dtype=np.int32), positions, shortcuts, frozen)
        deep_nested = Layout(np.array([0, -1, 2, 4, 6], dtype=np.int32), positions, shortcuts, frozen)
        deep_direct = Layout(np.array([5, -1, -1, -1, 6], dtype=np.int32), positions, shortcuts, frozen)
        weights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["weights"]}
        weights["violations"] = 1.0
        vweights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
        vweights["access_layout"] = 1.0
        vweights["layer_depth_penalty"] = 1.0
        evaluator = FitnessEvaluator(weights=weights, reference_layout=redundant, violation_weights=vweights,
                                     hard_constraints=[], missing_important_threshold=99.0)
        self.assertGreater(evaluator.evaluate(redundant).objectives[2], evaluator.evaluate(single_direct).objectives[2])
        self.assertGreater(evaluator.evaluate(toggle_only).objectives[2], evaluator.evaluate(single_direct).objectives[2])
        self.assertGreater(evaluator.evaluate(nested).objectives[2], evaluator.evaluate(direct).objectives[2])
        self.assertGreater(
            evaluator.evaluate(deep_nested).objectives[2],
            evaluator.evaluate(deep_direct).objectives[2] + 10.0,
        )

    def test_momentary_key_reuse_penalizes_same_key_different_layer_jobs(self):
        # Position 0 (L0) and position 1 (L3) sit at the IDENTICAL physical
        # coordinate (5.0, 4.0) -- same physical key, different layer. If its
        # momentary-hold job targets a different layer depending on which
        # layer is currently active (reuse), that should score worse than
        # targeting the same layer from both places (no reuse), holding real
        # per-layer demand (positions 2/3 host real shortcuts on L1/L2)
        # equal between the two scenarios.
        positions = (
            Position(0, 0, 5.0, 4.0, "left", 0, 0.2, is_thumb=True),
            Position(1, 3, 5.0, 4.0, "left", 0, 0.2, is_thumb=True),
            Position(2, 1, 6.0, 2.0, "right", 1, 0.5),
            Position(3, 2, 7.0, 2.0, "right", 1, 0.5),
        )
        shortcuts = (
            Shortcut(0, "@access:L0->L1:hold:One", "One", "Layer Access", 12.0,
                     "layer_access", is_layer_access=True, access_target_layer=1, access_is_momentary=True),
            Shortcut(1, "@access:L3->L1:hold:One", "One", "Layer Access", 12.0,
                     "layer_access", is_layer_access=True, access_target_layer=1, access_is_momentary=True),
            Shortcut(2, "@access:L3->L2:hold:Two", "Two", "Layer Access", 12.0,
                     "layer_access", is_layer_access=True, access_target_layer=2, access_is_momentary=True),
            Shortcut(3, "Ctrl+A", "Action A", "App", 10.0),
            Shortcut(4, "Ctrl+B", "Action B", "App", 10.0),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        # Reuse: position 0/1's shared physical key holds to L1 from L0 but
        # to L2 from L3 -- two distinct jobs for the same physical key.
        reuse = Layout(np.array([0, 2, 3, 4], dtype=np.int32), positions, shortcuts, frozen)
        # Control: the same physical key holds to L1 from both L0 and L3 --
        # one consistent job, no reuse.
        no_reuse = Layout(np.array([0, 1, 3, 4], dtype=np.int32), positions, shortcuts, frozen)
        weights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["weights"]}
        weights["violations"] = 1.0
        vweights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
        vweights["momentary_key_reuse"] = 1.0
        evaluator = FitnessEvaluator(weights=weights, reference_layout=no_reuse, violation_weights=vweights,
                                     hard_constraints=[], missing_important_threshold=99.0)
        self.assertGreater(
            evaluator.evaluate(reuse).objectives[2],
            evaluator.evaluate(no_reuse).objectives[2],
        )

    def test_mouse_duplicates_clean_up_after_natural_mouse_layer_exists(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(2, 3, 8.0, 1.0, "right", 1, 1.0),
            Position(3, 3, 9.0, 1.0, "right", 2, 1.0),
            Position(4, 3, 10.0, 1.0, "right", 2, 1.0),
            Position(5, 3, 11.0, 1.0, "right", 3, 1.0),
            Position(6, 3, 12.0, 1.0, "right", 4, 1.2),
            Position(7, 3, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(8, 4, 8.0, 1.0, "right", 1, 1.0),
            Position(9, 3, 9.0, 2.0, "right", 1, 1.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L3:hold:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L3:toggle:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(3, "MB2", "Click", "Mouse", 15.0, "mouse"),
            Shortcut(4, "MB3", "Click", "Mouse", 10.0, "mouse"),
            Shortcut(5, "MB4", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(6, "MB5", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(
                7, "@access:L3->L6:hold:Scroll", "Scroll", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=6,
                access_is_momentary=True,
            ),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        natural = Layout(
            np.array([0, 1, 2, 3, 4, 5, 6, -1, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        extra_mouse = Layout(
            np.array([0, 1, 2, 3, 4, 5, 6, -1, 2, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 1.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_similarity": 0.0,
            "everything_layer": 0.0,
        }
        vweights = {
            "duplicate": 0.0,
            "l0_displacement": 0.0,
            "missing_important": 0.0,
            "cross_layer_duplicate": 0.0,
            "group_split": 0.0,
            "thumb_occupancy": 0.0,
            "arrow_order": 0.0,
            "hand_bias": 0.0,
            "mouse_layer_access": 0.0,
            "arrow_scattered": 0.0,
            "mouse_scattered": 1.0,
            "layer7_access": 0.0,
            "duplicate_value_gap": 0.0,
            "access_layout": 0.0,
            "raw_keyboard_completion_norwegian": 0.0,
            "dynamic_mouse_layer": 0.0,
        }
        evaluator = FitnessEvaluator(
            weights=weights,
            reference_layout=natural,
            violation_weights=vweights,
            hard_constraints=[],
            missing_important_threshold=99.0,
        )
        self.assertGreater(evaluator.evaluate(extra_mouse).objectives[2], evaluator.evaluate(natural).objectives[2])

    def test_dynamic_mouse_layer_same_side_duplicate_policy(self):
        # Positions 0-9 and the shortcuts below are the exact fixture from
        # test_dynamic_mouse_layer_penalty_rewards_natural_complete_layer,
        # which is already verified to make layer 3 the winning (lowest
        # penalty) dynamic-mouse-layer candidate. Positions 10-11 are extra
        # layer-3 slots (one right non-thumb, one left) used only to place
        # duplicate MB1 copies for this test.
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(2, 3, 2.0, 1.0, "left", 1, 1.0),
            # effort=0.0 on the mouse-button/scroll positions: a valid
            # natural mouse layer here must be optimally placed, or its own
            # candidate_penalty (post dynamic_mouse_layer fix, which reports
            # the natural layer's real penalty instead of an unrelated empty
            # candidate layer's flat baseline) would legitimately exceed the
            # baseline and break the "valid beats broken" comparisons below.
            Position(3, 3, 8.0, 2.0, "right", 1, 0.0),
            Position(4, 3, 9.0, 2.0, "right", 1, 0.0),
            # MB3/MB4/MB5 and Scroll ideal x-targets: MB3=11, MB4=12, MB5=13,
            # Scroll=10 -- Scroll's ideal sits one slot right of MB2 (not on
            # top of it), so MB1(x8)/MB2(x9) can land adjacent.
            Position(5, 3, 11.0, 2.0, "right", 2, 0.0),
            Position(6, 3, 12.0, 2.0, "right", 3, 0.0),
            Position(7, 3, 13.0, 2.0, "right", 4, 0.0),
            Position(8, 3, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(9, 3, 10.0, 2.0, "right", 1, 0.0),
            Position(10, 3, 14.0, 1.0, "right", 4, 0.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L3:hold:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L3:toggle:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(3, "MB2", "Click", "Mouse", 15.0, "mouse"),
            Shortcut(4, "MB3", "Click", "Mouse", 10.0, "mouse"),
            Shortcut(5, "MB4", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(6, "MB5", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(
                7, "@access:L3->L6:hold:Scroll", "Scroll", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=6,
                access_is_momentary=True,
            ),
            Shortcut(
                8, "@access:L0->L3:hold:MouseRightThumb", "Mouse", "Layer Access", 10.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        base = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, -1, 7, -1], dtype=np.int32),
            positions, shortcuts, frozen,
        )
        two_right_mb1 = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, -1, 7, 2], dtype=np.int32),
            positions, shortcuts, frozen,
        )
        left_right_pair_mb1 = Layout(
            np.array([0, 1, 2, 2, 3, 4, 5, 6, -1, 7, -1], dtype=np.int32),
            positions, shortcuts, frozen,
        )
        weights = {
            "effort": 0.0, "adjacency": 0.0, "finger_balance": 0.0, "same_finger": 0.0,
            "violations": 1.0, "workflow_coherence": 0.0, "app_coherence": 0.0,
            "trackball_proximity": 0.0, "familiarity": 0.0, "layer_specialization": 0.0,
        }
        vweights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
        vweights["dynamic_mouse_layer"] = 1.0
        evaluator = FitnessEvaluator(weights=weights, reference_layout=base, violation_weights=vweights,
                                     hard_constraints=[], missing_important_threshold=99.0)
        base_score = evaluator.evaluate(base).objectives[2]
        two_right_score = evaluator.evaluate(two_right_mb1).objectives[2]
        left_right_score = evaluator.evaluate(left_right_pair_mb1).objectives[2]
        # A single right-side MB1 copy (the valid dynamic mouse layer) beats
        # two right-side copies of the same button.
        self.assertGreater(two_right_score, base_score)
        # One right + one left copy of the same button is a much smaller
        # departure from the single-copy baseline than two right-side copies
        # (adding any extra mouse-button placement carries a small unrelated
        # access-cost baseline, so this isn't an exact equality check).
        self.assertLess(
            left_right_score - base_score,
            (two_right_score - base_score) * 0.5,
        )
        # Two right-side copies must score strictly worse than the
        # one-left-plus-one-right pairing.
        self.assertGreater(two_right_score, left_right_score)

    def test_dynamic_mouse_layer_incomplete_candidate_duplicate_penalty(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(2, 5, 8.0, 1.0, "right", 1, 1.0),
            Position(3, 5, 3.0, 1.0, "left", 1, 1.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L3:hold:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L3:toggle:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Click", "Mouse", 20.0, "mouse"),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        single_right = Layout(
            np.array([0, 1, 2, -1], dtype=np.int32), positions, shortcuts, frozen,
        )
        duplicate_pair = Layout(
            np.array([0, 1, 2, 2], dtype=np.int32), positions, shortcuts, frozen,
        )
        weights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["weights"]}
        weights["violations"] = 1.0
        vweights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
        vweights["dynamic_mouse_layer"] = 1.0
        evaluator = FitnessEvaluator(weights=weights, reference_layout=single_right, violation_weights=vweights,
                                     hard_constraints=[], missing_important_threshold=99.0)
        # Layer 5 is an incomplete mouse-layer candidate (only MB1 present).
        # Keeping only the right-side copy must score better than adding a
        # left-side duplicate, because incomplete candidates don't get the
        # paired-duplicate exception.
        self.assertGreater(
            evaluator.evaluate(duplicate_pair).objectives[2],
            evaluator.evaluate(single_right).objectives[2],
        )

    def test_l0_empty_position_is_no_longer_exempt(self):
        positions = (
            Position(0, 0, 5.0, 5.0, "left", 1, 1.5, is_thumb=True),
        )
        shortcuts = (
            Shortcut(0, "Ctrl+Z", "Undo", "App", 5.0),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        empty_l0 = Layout(np.array([-1], dtype=np.int32), positions, shortcuts, frozen)
        filled_l0 = Layout(np.array([0], dtype=np.int32), positions, shortcuts, frozen)

        # Violations objective (empty_pos_waste, raw_scores[16]): empty L0 must
        # now cost something (it was previously fully exempt).
        weights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["weights"]}
        weights["violations"] = 1.0
        vweights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
        vweights["empty_position"] = 1.0
        evaluator = FitnessEvaluator(weights=weights, reference_layout=filled_l0, violation_weights=vweights,
                                     hard_constraints=[], missing_important_threshold=99.0)
        self.assertGreater(evaluator.evaluate(empty_l0).objectives[2], 0.0)

        # Effort objective (the sigmoid empty-position penalty added directly
        # to effort): empty L0 must score worse than filled L0.
        weights2 = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["weights"]}
        weights2["effort"] = 1.0
        vweights2 = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
        evaluator2 = FitnessEvaluator(weights=weights2, reference_layout=filled_l0, violation_weights=vweights2,
                                      hard_constraints=[], missing_important_threshold=99.0)
        self.assertGreater(
            evaluator2.evaluate(empty_l0).objectives[0],
            evaluator2.evaluate(filled_l0).objectives[0],
        )

    def test_l0_empty_position_excludes_frozen(self):
        positions = (
            Position(0, 0, 5.0, 5.0, "left", 1, 1.5, is_thumb=True, is_frozen=True),
        )
        shortcuts = (
            Shortcut(0, "Ctrl+Z", "Undo", "App", 5.0),
        )
        frozen = np.array([True])
        empty_l0_frozen = Layout(np.array([-1], dtype=np.int32), positions, shortcuts, frozen)
        weights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["weights"]}
        weights["violations"] = 1.0
        weights["effort"] = 1.0
        vweights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
        vweights["empty_position"] = 1.0
        evaluator = FitnessEvaluator(weights=weights, reference_layout=empty_l0_frozen, violation_weights=vweights,
                                     hard_constraints=[], missing_important_threshold=99.0)
        r = evaluator.evaluate(empty_l0_frozen)
        self.assertEqual(r.objectives[2], 0.0)

    def test_l0_thumb_occupied_never_worse_than_empty(self):
        # A low-importance, zero-usage shortcut occupying an L0 thumb slot
        # must not score worse (in violations_raw terms) than leaving the
        # exact same slot empty.
        positions = (
            Position(0, 0, 5.0, 5.0, "left", 1, 1.5, is_thumb=True),
        )
        shortcuts = (
            Shortcut(0, "Ctrl+Z", "Undo", "App", 5.0),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        empty_l0 = Layout(np.array([-1], dtype=np.int32), positions, shortcuts, frozen)
        filled_l0 = Layout(np.array([0], dtype=np.int32), positions, shortcuts, frozen)
        weights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["weights"]}
        weights["violations"] = 1.0
        vweights = dict(DEFAULT_CONFIG["fitness"]["violation_sub_weights"])
        evaluator = FitnessEvaluator(weights=weights, reference_layout=filled_l0, violation_weights=vweights,
                                     hard_constraints=[], missing_important_threshold=99.0)
        self.assertLessEqual(
            evaluator.evaluate(filled_l0).objectives[2],
            evaluator.evaluate(empty_l0).objectives[2],
        )

    def test_archive_bootstrap_never_accepts_infeasible_first_entry(self):
        from evolution.custom_ga import CustomGARunner
        infeasible = {"constraints": [1.0, 0.0], "total_score": -500.0}
        feasible = {"constraints": [0.0, 0.0], "total_score": -10.0}
        # No incumbent yet: an infeasible candidate must NOT become the archive seed.
        self.assertFalse(CustomGARunner._is_better(None, infeasible, None))
        # A feasible candidate may become the archive seed.
        self.assertTrue(CustomGARunner._is_better(None, feasible, None))
        # Once a feasible incumbent exists, a worse-scoring feasible candidate loses,
        # and an infeasible candidate never beats a feasible incumbent.
        self.assertFalse(CustomGARunner._is_better(None, infeasible, feasible))

    def test_same_layer_duplicate_hard_constraint_basic(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 3, 8.0, 1.0, "right", 1, 1.0),
            Position(2, 3, 9.0, 1.0, "right", 1, 1.0),
        )
        shortcuts = (
            Shortcut(0, "Ctrl+A", "Action", "App", 10.0),
            Shortcut(1, "Ctrl+B", "Action", "App", 10.0),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        clean = Layout(np.array([-1, 0, 1], dtype=np.int32), positions, shortcuts, frozen)
        duplicated = Layout(np.array([-1, 0, 0], dtype=np.int32), positions, shortcuts, frozen)
        evaluator = FitnessEvaluator(
            weights=DEFAULT_CONFIG["fitness"]["weights"],
            reference_layout=clean,
            violation_weights=DEFAULT_CONFIG["fitness"]["violation_sub_weights"],
            hard_constraints=["same_layer_duplicate"],
            missing_important_threshold=99.0,
        )
        self.assertEqual(evaluator.evaluate(clean).constraints[0], 0.0)
        self.assertGreater(evaluator.evaluate(duplicated).constraints[0], 0.0)

    def test_same_layer_duplicate_excludes_l7(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 7, 8.0, 1.0, "right", 1, 1.0, is_frozen=True),
            Position(2, 7, 9.0, 1.0, "right", 1, 1.0, is_frozen=True),
        )
        shortcuts = (
            Shortcut(0, "Ctrl+A", "Action", "App", 10.0),
        )
        frozen = np.array([False, True, True])
        duplicated_on_l7 = Layout(np.array([-1, 0, 0], dtype=np.int32), positions, shortcuts, frozen)
        evaluator = FitnessEvaluator(
            weights=DEFAULT_CONFIG["fitness"]["weights"],
            reference_layout=duplicated_on_l7,
            violation_weights=DEFAULT_CONFIG["fitness"]["violation_sub_weights"],
            hard_constraints=["same_layer_duplicate"],
            missing_important_threshold=99.0,
        )
        self.assertEqual(evaluator.evaluate(duplicated_on_l7).constraints[0], 0.0)

    def test_same_layer_duplicate_mouse_pair_exception_and_orphan(self):
        # Reuses the verified fixture from
        # test_dynamic_mouse_layer_same_side_duplicate_policy: layer 3 is the
        # winning dynamic-mouse-layer candidate with positions 0-9 as-is.
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(2, 3, 2.0, 1.0, "left", 1, 1.0),
            # effort=0.0 on the mouse-button/scroll positions: a valid
            # natural mouse layer here must be optimally placed, or its own
            # candidate_penalty (post dynamic_mouse_layer fix, which reports
            # the natural layer's real penalty instead of an unrelated empty
            # candidate layer's flat baseline) would legitimately exceed the
            # baseline and break the "valid beats broken" comparisons below.
            Position(3, 3, 8.0, 2.0, "right", 1, 0.0),
            Position(4, 3, 9.0, 2.0, "right", 1, 0.0),
            Position(5, 3, 10.0, 2.0, "right", 2, 0.0),
            Position(6, 3, 11.0, 2.0, "right", 3, 0.0),
            Position(7, 3, 12.0, 2.0, "right", 4, 0.0),
            Position(8, 3, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(9, 3, 9.0, 2.0, "right", 1, 0.0),
            Position(10, 3, 13.0, 1.0, "right", 4, 0.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L3:hold:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L3:toggle:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(3, "MB2", "Click", "Mouse", 15.0, "mouse"),
            Shortcut(4, "MB3", "Click", "Mouse", 10.0, "mouse"),
            Shortcut(5, "MB4", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(6, "MB5", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(
                7, "@access:L3->L6:hold:Scroll", "Scroll", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=6,
                access_is_momentary=True,
            ),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        evaluator = FitnessEvaluator(
            weights=DEFAULT_CONFIG["fitness"]["weights"],
            reference_layout=Layout(
                np.array([0, 1, -1, 2, 3, 4, 5, 6, -1, 7, -1], dtype=np.int32), positions, shortcuts, frozen,
            ),
            violation_weights=DEFAULT_CONFIG["fitness"]["violation_sub_weights"],
            hard_constraints=["same_layer_duplicate"],
            missing_important_threshold=99.0,
        )
        # Valid left+right MB1 pair on the actual dynamic mouse layer: allowed.
        left_right_pair = Layout(
            np.array([0, 1, 2, 2, 3, 4, 5, 6, -1, 7, -1], dtype=np.int32), positions, shortcuts, frozen,
        )
        self.assertEqual(evaluator.evaluate(left_right_pair).constraints[0], 0.0)

        # Two right-side copies of MB1: never allowed, even on the mouse layer.
        two_right = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, -1, 7, 2], dtype=np.int32), positions, shortcuts, frozen,
        )
        self.assertGreater(evaluator.evaluate(two_right).constraints[0], 0.0)

        # Same left+right pair, but Scroll has moved to an uncomfortable x=7
        # position, so layer 3 no longer qualifies as natural_mouse_layer.
        # The left-side MB1 copy must immediately become an illegal duplicate.
        disqualified_positions = tuple(
            Position(
                p.gene_idx, p.layer,
                7.0 if p.gene_idx == 9 else p.x,
                p.y, p.hand, p.finger, p.effort,
                is_thumb=p.is_thumb, is_frozen=p.is_frozen,
            )
            for p in positions
        )
        orphaned_pair = Layout(
            np.array([0, 1, 2, 2, 3, 4, 5, 6, -1, 7, -1], dtype=np.int32),
            disqualified_positions, shortcuts, frozen,
        )
        self.assertGreater(evaluator.evaluate(orphaned_pair).constraints[0], 0.0)

    def test_no_same_layer_duplicates_report(self):
        from evolution.acceptance import _no_same_layer_duplicates_report
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 3, 8.0, 1.0, "right", 1, 1.0),
            Position(2, 3, 9.0, 1.0, "right", 1, 1.0),
        )
        shortcuts = (
            Shortcut(0, "Ctrl+A", "Action", "App", 10.0),
            Shortcut(1, "Ctrl+B", "Action", "App", 10.0),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        clean = Layout(np.array([-1, 0, 1], dtype=np.int32), positions, shortcuts, frozen)
        duplicated = Layout(np.array([-1, 0, 0], dtype=np.int32), positions, shortcuts, frozen)
        self.assertTrue(_no_same_layer_duplicates_report(clean)["acceptance_pass"])
        report = _no_same_layer_duplicates_report(duplicated)
        self.assertFalse(report["acceptance_pass"])
        self.assertEqual(len(report["offenders"]), 1)
        self.assertEqual(report["offenders"][0]["keys"], "Ctrl+A")

    def test_everything_layer_rewards_common_shortcuts_on_one_accessible_layer(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 1, 8.0, 1.0, "right", 1, 1.0),
            Position(2, 1, 9.0, 1.0, "right", 2, 1.0),
            Position(3, 1, 10.0, 1.0, "right", 3, 1.0),
            Position(4, 2, 8.0, 1.0, "right", 1, 1.0),
            Position(5, 3, 8.0, 1.0, "right", 1, 1.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L1:hold:General", "General", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=1,
                access_is_momentary=True,
            ),
            Shortcut(1, "Shortcut A", "A", "app", 10.0),
            Shortcut(2, "Shortcut B", "B", "app", 9.0),
            Shortcut(3, "Shortcut C", "C", "app", 8.0),
        )
        usage = UsageData(shortcuts={
            "Shortcut A": {"count": 100},
            "Shortcut B": {"count": 80},
            "Shortcut C": {"count": 60},
        })
        frozen = np.zeros(len(positions), dtype=np.bool_)
        concentrated = Layout(
            np.array([0, 1, 2, 3, -1, -1], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        scattered = Layout(
            np.array([0, 1, -1, -1, 2, 3], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 1.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_specialization": 0.0,
            "everything_layer": 10.0,
        }
        evaluator = FitnessEvaluator(
            weights=weights,
            reference_layout=concentrated,
            violation_weights={},
            hard_constraints=[],
            missing_important_threshold=99.0,
        )
        self.assertLess(
            evaluator.evaluate(concentrated).objectives[2],
            evaluator.evaluate(scattered).objectives[2],
        )

    def test_momentary_only_single_thumb_side_must_be_clear(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(2, 2, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(3, 2, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L2:hold:Layer", "Layer", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=2,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L2:toggle:Layer", "Layer", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=2,
                access_is_momentary=False,
            ),
            Shortcut(
                2, "@access:L0->L2:hold:LayerRight", "Layer", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=2,
                access_is_momentary=True,
            ),
            Shortcut(3, "Ctrl+A", "Select All", "app", 8.0),
            Shortcut(4, "Ctrl+C", "Copy", "app", 8.0),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)

        blocked = Layout(
            np.array([0, -1, 3, 4], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        report = _momentary_only_thumb_clearance_report(blocked)
        self.assertFalse(report["acceptance_pass"])
        self.assertEqual(report["violations"][0]["occupied_hand"], "left")

        toggle_access = Layout(
            np.array([0, 1, 3, 4], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        toggle_report = _momentary_only_thumb_clearance_report(toggle_access)
        self.assertTrue(toggle_report["acceptance_pass"])
        floor_rows = [
            item
            for layer in toggle_report["layers"]
            for item in layer.get("effort_floor_assignments", [])
        ]
        self.assertEqual(len(floor_rows), 1)
        self.assertEqual(floor_rows[0]["keys"], "Ctrl+A")
        self.assertEqual(floor_rows[0]["effective_effort_floor"], 4.0)
        toggle_other_side = Layout(
            np.array([0, 1, -1, 4], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        weights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["weights"]}
        weights["violations"] = 1.0
        vweights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
        vweights["thumb_occupancy"] = 1.0
        evaluator = FitnessEvaluator(
            weights=weights,
            reference_layout=toggle_access,
            violation_weights=vweights,
            hard_constraints=[],
            missing_important_threshold=99.0,
        )
        self.assertGreater(
            evaluator.evaluate(toggle_access).objectives[2],
            evaluator.evaluate(toggle_other_side).objectives[2],
        )

        both_momentary_sides = Layout(
            np.array([0, 2, 3, 4], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        both_report = _momentary_only_thumb_clearance_report(both_momentary_sides)
        self.assertTrue(both_report["acceptance_pass"])
        self.assertFalse(any(layer.get("effort_floor_assignments") for layer in both_report["layers"]))

        lost_second_side = Layout(
            np.array([0, -1, 3, -1], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        lost_report = _momentary_only_thumb_clearance_report(lost_second_side)
        self.assertFalse(lost_report["acceptance_pass"])
        self.assertEqual(lost_report["violations"][0]["restricted_hands"], ["left"])


if __name__ == "__main__":
    unittest.main()
