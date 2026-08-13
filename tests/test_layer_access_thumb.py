"""Unit and parity tests for thumb-aware layer access scoring.

Covers both behavioral directionality of the new thumb-preference terms
and CUDA/Numba numerical parity on real and synthetic layouts.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from config import Config, DEFAULT_CONFIG
from core import Layout, Position, Shortcut
from core.loader import build_layout
from fitness.evaluator import FitnessEvaluator


class TestLayerAccessThumb(unittest.TestCase):
    """Behavioral and CUDA/Numba parity tests for thumb-aware layer access."""

    @staticmethod
    def _make_minimal_layout():
        positions = (
            # L0 left thumb
            Position(0, 0, 3.0, 4.0, "left", 0, 1.0, is_thumb=True),
            # L0 right thumb
            Position(1, 0, 8.0, 4.0, "right", 0, 1.0, is_thumb=True),
            # L0 left finger
            Position(2, 0, 2.0, 2.0, "left", 1, 1.0, is_thumb=False),
            # L1 left thumb
            Position(3, 1, 3.0, 4.0, "left", 0, 1.0, is_thumb=True),
            # L1 right thumb
            Position(4, 1, 8.0, 4.0, "right", 0, 1.0, is_thumb=True),
            # L1 left finger
            Position(5, 1, 2.0, 2.0, "left", 1, 1.0, is_thumb=False),
        )
        shortcuts = (
            Shortcut(0, "@access:L1:hold", "L1 hold", "Layer Access", 10.0,
                     category="layer_access", is_layer_access=True,
                     access_target_layer=1, access_is_momentary=True),
            Shortcut(1, "@access:L2:hold", "L2 hold", "Layer Access", 10.0,
                     category="layer_access", is_layer_access=True,
                     access_target_layer=2, access_is_momentary=True),
            Shortcut(2, "Ctrl+C", "Copy", "App", 5.0),
        )
        layer_to_indices = {
            0: np.array([0, 1, 2], dtype=np.int32),
            1: np.array([3, 4, 5], dtype=np.int32),
        }
        frozen_mask = np.zeros(len(positions), dtype=bool)
        return positions, shortcuts, layer_to_indices, frozen_mask

    def _build_layout(self):
        config = Config.load("config_v2.yaml")
        return config, build_layout("data", config.raw.get("fitness", {}))

    def _make_evaluator(self, weights=None, violation_weights=None):
        config, layout = self._build_layout()
        if weights is None:
            weights = config.get("fitness.weights", {})
        if violation_weights is None:
            violation_weights = config.get("fitness.violation_sub_weights", {})
        return FitnessEvaluator(
            weights=weights,
            reference_layout=layout,
            violation_weights=violation_weights,
            missing_important_threshold=config.get("fitness.missing_important_threshold", 6.0),
            hard_constraints=[],
            toggle_effort_multiplier=float(config.get("fitness.toggle_effort_multiplier", 2.5)),
        )

    def _sample_batch(self, layout, n_random=5, swaps=20, seed=54321):
        rng = np.random.default_rng(seed)
        samples = [layout.genome.copy()]
        for _ in range(n_random):
            g = layout.genome.copy()
            for _ in range(swaps):
                a, b = rng.choice(layout.mutable_indices, 2, replace=False)
                g[a], g[b] = g[b], g[a]
            samples.append(g)
        return np.asarray(samples, dtype=np.int32)

    def _eval_both(self, evaluator, batch):
        evaluator.model._use_cuda = True
        obj_cuda, _ = evaluator.model.evaluate_batch(batch)
        evaluator.model._use_cuda = False
        obj_numba, _ = evaluator.model.evaluate_batch(batch)
        return obj_cuda, obj_numba

    def test_cuda_numba_parity(self):
        try:
            from fitness.cuda_kernel import cuda_available
        except Exception:
            self.skipTest("CUDA kernel not importable")
        if not cuda_available():
            self.skipTest("CUDA not available")

        evaluator = self._make_evaluator()
        batch = self._sample_batch(evaluator.reference_layout)

        obj_cuda, obj_numba = self._eval_both(evaluator, batch)

        # Effort and adjacency are numerically stable; violations sum many
        # large terms in different orders and need a looser absolute tolerance.
        np.testing.assert_allclose(obj_cuda[:, 0], obj_numba[:, 0], rtol=1e-4, atol=2000.0)
        np.testing.assert_allclose(obj_cuda[:, 1], obj_numba[:, 1], rtol=1e-4, atol=1.0)
        np.testing.assert_allclose(obj_cuda[:, 2], obj_numba[:, 2], rtol=1e-4, atol=1e6)

    def _isolated_raw_score(self, target_name, weight=1.0):
        """Return raw_scores for target_name by differencing weighted/unweighted evals."""
        base_weights = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["weights"]}
        base_weights["violations"] = 1.0
        base_vw = {k: 0.0 for k in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
        base_vw[target_name] = weight
        off_vw = dict(base_vw)
        off_vw[target_name] = 0.0

        on_eval = self._make_evaluator(weights=base_weights, violation_weights=base_vw)
        off_eval = self._make_evaluator(weights=base_weights, violation_weights=off_vw)

        batch = self._sample_batch(on_eval.reference_layout)
        on_cuda, on_numba = self._eval_both(on_eval, batch)
        off_cuda, off_numba = self._eval_both(off_eval, batch)

        raw_cuda = (on_cuda[:, 2] - off_cuda[:, 2]) / weight
        raw_numba = (on_numba[:, 2] - off_numba[:, 2]) / weight
        return raw_cuda, raw_numba

    def test_layer_access_thumb_preference_raw_score_parity(self):
        try:
            from fitness.cuda_kernel import cuda_available
        except Exception:
            self.skipTest("CUDA kernel not importable")
        if not cuda_available():
            self.skipTest("CUDA not available")

        raw_cuda, raw_numba = self._isolated_raw_score("layer_access_thumb_preference", weight=1.0)
        np.testing.assert_allclose(raw_cuda, raw_numba, rtol=1e-4, atol=1e-3)

    def test_same_side_hold_flow_raw_score_parity(self):
        try:
            from fitness.cuda_kernel import cuda_available
        except Exception:
            self.skipTest("CUDA kernel not importable")
        if not cuda_available():
            self.skipTest("CUDA not available")

        raw_cuda, raw_numba = self._isolated_raw_score("same_side_hold_flow", weight=1.0)
        np.testing.assert_allclose(raw_cuda, raw_numba, rtol=1e-4, atol=1e-3)

    def test_layer_access_thumb_preference(self):
        positions, shortcuts, layer_to_indices, frozen_mask = self._make_minimal_layout()

        # L1 hold on left thumb -> good
        genome_good = np.array([0, -1, -1, -1, -1, -1], dtype=np.int32)
        layout_good = Layout(genome_good, positions, shortcuts, frozen_mask, layer_to_indices)

        # L1 hold on left finger -> bad; keep other slots unassigned so the
        # only difference is thumb vs. non-thumb placement.
        genome_bad = np.array([-1, -1, 0, -1, -1, -1], dtype=np.int32)
        layout_bad = Layout(genome_bad, positions, shortcuts, frozen_mask, layer_to_indices)

        ev = FitnessEvaluator()
        score_good = ev.evaluate(layout_good).total_score
        score_bad = ev.evaluate(layout_bad).total_score
        self.assertGreater(score_bad, score_good, "non-thumb layer access should score worse")

    def test_same_side_hold_flow(self):
        positions, shortcuts, layer_to_indices, frozen_mask = self._make_minimal_layout()

        # L0 left thumb -> L1, L1 left thumb -> L2 (same side, bad)
        genome_same = np.array([0, -1, -1, 1, -1, -1], dtype=np.int32)
        # L0 left thumb -> L1, L1 right thumb -> L2 (opposite, good)
        genome_opp = np.array([0, -1, -1, -1, 1, -1], dtype=np.int32)

        layout_same = Layout(genome_same, positions, shortcuts, frozen_mask, layer_to_indices)
        layout_opp = Layout(genome_opp, positions, shortcuts, frozen_mask, layer_to_indices)

        ev = FitnessEvaluator()
        score_same = ev.evaluate(layout_same).total_score
        score_opp = ev.evaluate(layout_opp).total_score
        self.assertGreater(score_same, score_opp, "same-side hold chain should score worse")


if __name__ == "__main__":
    unittest.main()
