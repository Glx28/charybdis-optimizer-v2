"""Empty/transparent position penalty tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import numpy as np

from core import Layout, Position, Shortcut


class TestEmptyPositionPenalty(unittest.TestCase):
    """Verify the soft empty-position penalty (violation sub-weight 'empty_position').

    Policy:
    - Low-effort (prime) positions → high penalty when empty.
    - High-effort (far/corner) positions → near-zero penalty when empty.
    - L7 frozen content is never penalised.
    - L0 is NOT exempt (it is the only zero-cost, always-reachable layer) and
      gets an extra multiplier on top of the base penalty.
    - Penalty is soft: it adds to the violations objective but never causes a
      hard acceptance failure by itself.
    """

    def _make_evaluator(self, layout, empty_position_weight=50.0):
        """Build an evaluator with only empty_position active."""
        weights = {k: 0.0 for k in [
            "effort", "adjacency", "finger_balance", "same_finger",
            "violations", "workflow_coherence", "app_coherence",
            "trackball_proximity", "familiarity", "layer_similarity", "everything_layer",
        ]}
        weights["violations"] = 1.0
        vweights = {k: 0.0 for k in [
            "duplicate", "l0_displacement", "missing_important", "cross_layer_duplicate",
            "group_split", "thumb_occupancy", "arrow_order", "hand_bias",
            "mouse_layer_access", "arrow_scattered", "mouse_scattered", "layer7_access",
            "duplicate_value_gap", "access_layout", "raw_keyboard_completion_norwegian",
            "dynamic_mouse_layer", "natural_mouse_layer_exists",
            "layer_reachability", "layer_depth_penalty", "toggle_back_to_l0",
            "mouse_hold_position_conflict", "mouse_layer_depth_penalty",
        ]}
        vweights["empty_position"] = empty_position_weight
        from fitness.evaluator import FitnessEvaluator
        return FitnessEvaluator(
            weights=weights,
            reference_layout=layout,
            violation_weights=vweights,
            hard_constraints=[],
        )

    def _prime_effort_layout(self):
        """Layout where position 1 (layer 1) is a prime low-effort slot.

        Returns (layout_with_access, layout_empty_prime, layout_empty_far).
        """
        # position 0: L0 left thumb (access key to L1, effort=0.4)
        # position 1: L1 right home-row (low effort = 0.3) -- "prime"
        # position 2: L1 right far corner (high effort = 4.5)
        positions = (
            Position(0, 0, 1.0, 4.0, "left", 0, 0.4, is_thumb=True),
            Position(1, 1, 7.0, 2.0, "right", 1, 0.3),   # prime
            Position(2, 1, 0.0, 0.0, "right", 4, 4.5),   # far corner
        )
        shortcuts = (
            Shortcut(0, "@toggle:L0->L1:toggle", "Toggle", "Layer Access", 10.0,
                     "layer_access", is_layer_access=True, access_target_layer=1,
                     access_is_momentary=False),
            Shortcut(1, "Ctrl+C", "Copy", "app", 8.0),
        )
        frozen = np.zeros(3, dtype=np.bool_)

        # Base: access on pos 0, Ctrl+C on pos 1 (prime), pos 2 (far) empty
        filled_prime = Layout(np.array([0, 1, -1], dtype=np.int32), positions, shortcuts, frozen)
        # Variant: pos 1 (prime) empty, pos 2 (far) still empty
        empty_prime = Layout(np.array([0, -1, -1], dtype=np.int32), positions, shortcuts, frozen)
        # Variant: pos 2 (far) empty, pos 1 (prime) filled
        empty_far = Layout(np.array([0, 1, -1], dtype=np.int32), positions, shortcuts, frozen)

        return filled_prime, empty_prime, empty_far

    def test_prime_empty_scores_higher_than_far_empty(self):
        """An empty prime position gets a much larger penalty than an empty far position."""
        filled_prime, empty_prime, _ = self._prime_effort_layout()
        ev = self._make_evaluator(filled_prime, empty_position_weight=50.0)

        # Layout A: prime pos 1 empty, far pos 2 empty
        positions = filled_prime.positions
        shortcuts = filled_prime.shortcuts
        frozen = filled_prime.frozen_mask
        layout_prime_empty = Layout(np.array([0, -1, -1], dtype=np.int32), positions, shortcuts, frozen)
        # Layout B: prime pos 1 filled, far pos 2 empty
        layout_far_empty = Layout(np.array([0, 1, -1], dtype=np.int32), positions, shortcuts, frozen)

        # Need an evaluator built on a layout — rebuild for each
        ev_prime = self._make_evaluator(layout_prime_empty, empty_position_weight=50.0)
        ev_far = self._make_evaluator(layout_far_empty, empty_position_weight=50.0)

        score_prime = ev_prime.evaluate(layout_prime_empty).total_score
        score_far = ev_far.evaluate(layout_far_empty).total_score

        self.assertGreater(
            score_prime, score_far,
            f"Empty prime position (effort=0.3) should cost more than empty far (effort=4.5); "
            f"got prime={score_prime:.4f}, far={score_far:.4f}",
        )

    def test_l7_frozen_not_penalised(self):
        """L7 frozen positions are excluded from empty_position penalty."""
        # Two positions: L0 thumb (frozen access key) and L7 position (frozen, empty)
        positions = (
            Position(0, 0, 1.0, 4.0, "left", 0, 0.4, is_thumb=True, is_frozen=True),
            Position(1, 7, 7.0, 2.0, "right", 1, 0.3, is_frozen=True),   # L7, frozen
        )
        shortcuts = (
            Shortcut(0, "@toggle:L0->L7:toggle", "Toggle", "Layer Access", 10.0,
                     "layer_access", is_layer_access=True, access_target_layer=7,
                     access_is_momentary=False),
        )
        # L7 is frozen so both positions frozen; genome has access on pos 0, pos 1 empty
        frozen = np.array([True, True], dtype=np.bool_)
        layout = Layout(np.array([0, -1], dtype=np.int32), positions, shortcuts, frozen)
        ev = self._make_evaluator(layout, empty_position_weight=50.0)
        result = ev.evaluate(layout)
        # Empty position penalty contribution should be 0 (L7 excluded)
        self.assertEqual(result.total_score, 0.0,
                         f"L7 frozen empty position must not be penalised; got {result.total_score}")

    def test_l0_empty_is_penalised_by_empty_position(self):
        """L0 empty positions are NOT excluded from empty_position: L0 is the
        only zero-cost, always-reachable layer, so an empty L0 slot must cost
        at least as much as an equivalent-effort empty slot elsewhere (and, per
        fitness/kernel.py's L0 multiplier, more)."""
        positions = (
            Position(0, 0, 7.0, 2.0, "right", 1, 0.3),   # L0, prime effort, empty
            Position(1, 0, 0.0, 0.0, "right", 4, 4.5),   # L0, far, empty
        )
        shortcuts = (
            Shortcut(0, "Ctrl+C", "Copy", "app", 8.0),
        )
        frozen = np.zeros(2, dtype=np.bool_)
        layout = Layout(np.array([-1, -1], dtype=np.int32), positions, shortcuts, frozen)
        ev = self._make_evaluator(layout, empty_position_weight=50.0)
        result = ev.evaluate(layout)
        self.assertGreater(result.total_score, 0.0,
                           f"L0 empty positions must now be penalised; got {result.total_score}")

    def test_penalty_is_soft_not_hard_constraint(self):
        """empty_position never appears as a hard constraint."""
        from config import DEFAULT_CONFIG
        hard = DEFAULT_CONFIG["fitness"]["hard_constraints"]
        self.assertNotIn("empty_position", hard,
                         "empty_position must remain a soft penalty, not a hard constraint")

    def test_penalty_scales_with_position_effort(self):
        """Penalty decreases monotonically as position effort increases."""
        # Build a series of single-slot layouts on layer 1 with increasing effort.
        efforts = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]
        prev_score = None
        for eff in efforts:
            positions = (
                Position(0, 0, 1.0, 4.0, "left", 0, 0.4, is_thumb=True),  # access key (L0)
                Position(1, 1, 7.0, 2.0, "right", 1, eff),                 # empty slot
            )
            shortcuts = (
                Shortcut(0, "@toggle:L0->L1:toggle", "Toggle", "Layer Access", 10.0,
                         "layer_access", is_layer_access=True, access_target_layer=1,
                         access_is_momentary=False),
            )
            frozen = np.zeros(2, dtype=np.bool_)
            layout = Layout(np.array([0, -1], dtype=np.int32), positions, shortcuts, frozen)
            ev = self._make_evaluator(layout, empty_position_weight=50.0)
            score = ev.evaluate(layout).total_score
            if prev_score is not None:
                self.assertLessEqual(
                    score, prev_score + 1e-4,
                    f"Penalty should decrease (or stay flat) as effort increases; "
                    f"got score={score:.4f} > prev={prev_score:.4f} at effort={eff}",
                )
            prev_score = score


if __name__ == "__main__":
    unittest.main()
