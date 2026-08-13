"""Minimal parity test for thumb-aware layer access scoring."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from config import Config, DEFAULT_CONFIG
from core.loader import build_layout
from fitness.evaluator import FitnessEvaluator


class TestLayerAccessThumb(unittest.TestCase):
    """CUDA/Numba parity for layer_access_thumb_preference and same_side_hold_flow."""

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


if __name__ == "__main__":
    unittest.main()
