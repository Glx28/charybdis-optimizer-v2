"""CustomGARunner selection/instrumentation tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from unittest import mock

import numpy as np

from evolution.custom_ga import (
    _feasibility_log_line,
    _spearman_rank_corr,
    _tournament_select,
)


class TestFeasibilityFirstTournament(unittest.TestCase):
    """Deb (2000) feasibility rules in binary tournament selection."""

    def _run_pairs(self, scalar_F, cv, pairs):
        """Run _tournament_select with deterministic candidate pairs."""
        pairs = np.asarray(pairs, dtype=np.int64)
        with mock.patch("numpy.random.randint", return_value=pairs):
            return _tournament_select(
                np.asarray(scalar_F, dtype=np.float32),
                len(pairs),
                k=pairs.shape[1],
                cv=np.asarray(cv, dtype=np.float32),
            )

    def test_feasible_beats_higher_scoring_infeasible(self):
        # Index 0: feasible but terrible scalar score. Index 1: infeasible
        # with a much better scalar score. Feasible must always win.
        scalar_F = [100.0, 1.0]
        cv = [[0.0, 0.0], [3.0, 0.0]]
        pairs = [[0, 1], [1, 0], [0, 0], [1, 1]]
        winners = self._run_pairs(scalar_F, cv, pairs)
        self.assertEqual(winners.tolist(), [0, 0, 0, 1])

    def test_two_infeasible_lower_violation_wins(self):
        # Both infeasible: index 0 has the better scalar score but higher
        # total violation, so index 1 must win regardless of order.
        scalar_F = [1.0, 50.0]
        cv = [[5.0, 0.0], [2.0, 0.0]]
        pairs = [[0, 1], [1, 0]]
        winners = self._run_pairs(scalar_F, cv, pairs)
        self.assertEqual(winners.tolist(), [1, 1])

    def test_two_feasible_scalar_score_wins(self):
        scalar_F = [1.0, 50.0]
        cv = [[0.0, 0.0], [0.0, 0.0]]
        winners = self._run_pairs(scalar_F, cv, [[0, 1], [1, 0]])
        self.assertEqual(winners.tolist(), [0, 0])

    def test_cv_none_falls_back_to_scalar(self):
        scalar_F = np.array([1.0, 50.0], dtype=np.float32)
        pairs = np.array([[0, 1], [1, 0]], dtype=np.int64)
        with mock.patch("numpy.random.randint", return_value=pairs):
            winners = _tournament_select(scalar_F, 2)
        self.assertEqual(winners.tolist(), [0, 0])


class TestFeasibilityLogLine(unittest.TestCase):
    """Feasibility instrumentation must not crash on degenerate populations."""

    def test_all_feasible(self):
        cv = np.zeros((4, 2), dtype=np.float32)
        line = _feasibility_log_line(100, cv)
        self.assertIn("feasible=1.000", line)
        self.assertIn("mean_viol_infeasible=0.0000", line)

    def test_all_infeasible(self):
        cv = np.ones((4, 2), dtype=np.float32)
        line = _feasibility_log_line(100, cv)
        self.assertIn("feasible=0.000", line)
        self.assertIn("mean_viol_infeasible=2.0000", line)

    def test_no_constraints(self):
        line = _feasibility_log_line(100, None)
        self.assertIn("feasible=1.000", line)


class TestSpearmanRankCorr(unittest.TestCase):
    def test_perfect_ranking(self):
        rho = _spearman_rank_corr([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(rho, 1.0)

    def test_reversed_ranking(self):
        rho = _spearman_rank_corr([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0])
        self.assertAlmostEqual(rho, -1.0)


if __name__ == "__main__":
    unittest.main()
