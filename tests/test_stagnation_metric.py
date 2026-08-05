"""Stagnation metric tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import numpy as np


class TestStagnationMetric(unittest.TestCase):
    """Verify the stagnation metric uses overall quality, not just violations.

    Policy: effort improvement or adjacency gain alone must prevent false stagnation.
    Constraint violations add a small exploration penalty; exact archive ranking
    applies hard acceptance tiers separately.
    """

    def _make_pop(self, rows):
        """Build a numpy array shaped (n, 3) representing objective columns."""
        return np.array(rows, dtype=np.float64)

    def _quality(self, pop, constraints=None):
        from run_evolution import ExactEvalCallback
        return ExactEvalCallback._quality_scalar(pop, constraints)

    def test_effort_improvement_resets_stagnation(self):
        """Improving effort objective alone (violations unchanged) lowers pop sum."""
        # Gen t: best row sum = 100 + 50 + 200 = 350
        pop_t = self._make_pop([
            [100.0, 50.0, 200.0],
            [120.0, 60.0, 210.0],
        ])
        # Gen t+1: effort improved from 100 -> 80, sum = 80 + 50 + 200 = 330
        pop_t1 = self._make_pop([
            [80.0, 50.0, 200.0],
            [120.0, 60.0, 210.0],
        ])
        quality_t = float(np.min(self._quality(pop_t)))
        quality_t1 = float(np.min(self._quality(pop_t1)))
        self.assertLess(quality_t1, quality_t * 0.999,
                        "Effort improvement alone must lower best_quality and prevent stagnation")

    def test_violations_only_improvement_also_resets(self):
        """Improving violations alone also lowers the sum and prevents stagnation."""
        pop_t = self._make_pop([[100.0, 50.0, 300.0]])
        pop_t1 = self._make_pop([[100.0, 50.0, 50.0]])   # violations dropped sharply
        quality_t = float(np.min(self._quality(pop_t)))
        quality_t1 = float(np.min(self._quality(pop_t1)))
        self.assertLess(quality_t1, quality_t * 0.999)

    def test_no_improvement_triggers_stagnation(self):
        """Identical population quality triggers stagnation (sum unchanged)."""
        pop_t = self._make_pop([[100.0, 50.0, 200.0]])
        pop_t1 = self._make_pop([[100.0, 50.0, 200.0]])
        quality_t = float(np.min(self._quality(pop_t)))
        quality_t1 = float(np.min(self._quality(pop_t1)))
        # The condition in run_evolution.py: stagnation++ if NOT (quality_t1 < quality_t * 0.999)
        improved = quality_t1 < quality_t * 0.999
        self.assertFalse(improved, "Equal quality should count as stagnation, not improvement")

    def test_violations_alone_unchanged_but_effort_dropped(self):
        """Effort drops, violations unchanged — still counts as non-stagnant."""
        # Simulates a run that improved effort/workflow but not violations
        pop_t = self._make_pop([[150.0, 30.0, 500.0]])   # sum = 680
        pop_t1 = self._make_pop([[120.0, 30.0, 500.0]])  # sum = 650 (effort only)
        quality_t = float(np.min(self._quality(pop_t)))
        quality_t1 = float(np.min(self._quality(pop_t1)))
        self.assertLess(quality_t1, quality_t * 0.999,
                        "Effort-only improvement must prevent stagnation")

    def test_constraint_penalty_allows_temporary_exploration(self):
        """Constraint penalty is small enough that infeasible candidates can be explored."""
        pop = self._make_pop([
            [-60.0, 0.0, 0.0],
            [-50.0, 0.0, 0.0],
        ])
        constraints = np.array([
            [1.0],
            [0.0],
        ])
        quality = self._quality(pop, constraints)
        self.assertAlmostEqual(float(quality[0]), -58.0)
        self.assertAlmostEqual(float(quality[1]), -50.0)
        self.assertLess(
            float(quality[0]),
            float(quality[1]),
            "Population exploration may temporarily prefer a lower-score infeasible stepping stone",
        )

    def test_constraint_penalty_scales_with_objective_spread(self):
        """Soft constraint pressure is calibrated to population score spread."""
        from run_evolution import ExactEvalCallback

        narrow = self._make_pop([[-60.0, 0.0, 0.0], [-50.0, 0.0, 0.0]])
        wide = self._make_pop([[-100.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        narrow_penalty = ExactEvalCallback._constraint_penalty_from_objectives(narrow)
        wide_penalty = ExactEvalCallback._constraint_penalty_from_objectives(wide)
        self.assertGreaterEqual(narrow_penalty, 2.0)
        self.assertGreater(wide_penalty, narrow_penalty)
        self.assertLessEqual(wide_penalty, 25.0)

    def test_dynamic_mouse_failure_cannot_be_archive_best(self):
        """Exact archive ranking treats dynamic mouse failure as a hard worse tier."""
        from run_evolution import ExactEvalCallback

        cb = object.__new__(ExactEvalCallback)
        candidate = {
            "constraints": [0.0],
            "optimizer_side_pass": False,
            "acceptance_failed_checks": ["dynamic_mouse_layer_present"],
            "total_score": -60.0,
        }
        incumbent = {
            "constraints": [0.0],
            "optimizer_side_pass": False,
            "acceptance_failed_checks": ["mutable_raw_arrows_ok", "norwegian_completion_cluster"],
            "total_score": -50.0,
        }
        self.assertFalse(cb._is_better_exact(candidate, incumbent))
        self.assertTrue(cb._is_better_exact(incumbent, candidate))
        self.assertEqual(ExactEvalCallback._display_gap(candidate), 5.0)

    def test_custom_ga_uses_same_dynamic_mouse_archive_tier(self):
        """Custom GPU GA archive ranking follows the same hard mouse tier."""
        from evolution.custom_ga import CustomGARunner, _display_gap, _scalar

        candidate = {
            "constraints": [0.0],
            "optimizer_side_pass": False,
            "acceptance_failed_checks": ["dynamic_mouse_layer_present"],
            "total_score": -60.0,
        }
        incumbent = {
            "constraints": [0.0],
            "optimizer_side_pass": False,
            "acceptance_failed_checks": ["mutable_raw_arrows_ok"],
            "total_score": -50.0,
        }
        runner = object.__new__(CustomGARunner)
        self.assertFalse(runner._is_better(candidate, incumbent))
        self.assertTrue(runner._is_better(incumbent, candidate))
        self.assertEqual(_display_gap(candidate), 5.0)

        pop = self._make_pop([[-60.0, 0.0, 0.0], [-50.0, 0.0, 0.0]])
        constraints = np.array([[1.0], [0.0]])
        quality = _scalar(pop, constraints)
        self.assertAlmostEqual(float(quality[0]), -58.0)


if __name__ == "__main__":
    unittest.main()
