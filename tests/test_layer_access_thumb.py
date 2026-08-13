"""Minimal parity test for thumb-aware layer access scoring."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from config import Config
from core.loader import build_layout
from fitness.evaluator import FitnessEvaluator


class TestLayerAccessThumb(unittest.TestCase):
    """CUDA/Numba parity for layer_access_thumb_preference and same_side_hold_flow."""

    def _make_evaluator(self):
        config = Config.load("config_v2.yaml")
        layout = build_layout("data", config.raw.get("fitness", {}))
        return FitnessEvaluator(
            weights=config.get("fitness.weights", {}),
            reference_layout=layout,
            violation_weights=config.get("fitness.violation_sub_weights", {}),
            missing_important_threshold=config.get("fitness.missing_important_threshold", 6.0),
            hard_constraints=config.get("fitness.hard_constraints", []),
            toggle_effort_multiplier=float(config.get("fitness.toggle_effort_multiplier", 2.5)),
        )

    def test_cuda_numba_parity(self):
        try:
            from fitness.cuda_kernel import cuda_available
        except Exception:
            self.skipTest("CUDA kernel not importable")
        if not cuda_available():
            self.skipTest("CUDA not available")

        evaluator = self._make_evaluator()
        layout = evaluator.reference_layout
        rng = np.random.default_rng(54321)

        samples = [layout.genome.copy()]
        for _ in range(5):
            g = layout.genome.copy()
            for _ in range(20):
                a, b = rng.choice(layout.mutable_indices, 2, replace=False)
                g[a], g[b] = g[b], g[a]
            samples.append(g)

        batch = np.asarray(samples, dtype=np.int32)

        evaluator.model._use_cuda = True
        obj_cuda, constr_cuda = evaluator.model.evaluate_batch(batch)

        evaluator.model._use_cuda = False
        obj_numba, constr_numba = evaluator.model.evaluate_batch(batch)

        np.testing.assert_allclose(obj_cuda, obj_numba, rtol=1e-4, atol=2e7)
        np.testing.assert_allclose(constr_cuda, constr_numba, rtol=1e-4, atol=1.0)


if __name__ == "__main__":
    unittest.main()
