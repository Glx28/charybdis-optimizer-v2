"""CUDA vs Numba exact-evaluation parity tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import numpy as np

from fitness.evaluator import FitnessEvaluator


class TestCudaExactEvalParity(unittest.TestCase):
    """Parity between the CUDA exact-eval kernel and the Numba fallback."""

    def _make_evaluator(self):
        from config import Config
        from core.loader import build_layout
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

    def test_cuda_parity_seed_and_random(self):
        """CUDA and Numba objectives must agree within float32 tolerance."""
        try:
            import torch  # noqa: F401
            from fitness.cuda_kernel import cuda_available
        except Exception:
            self.skipTest("CUDA kernel not importable")
        if not cuda_available():
            self.skipTest("CUDA not available")

        evaluator = self._make_evaluator()
        layout = evaluator.reference_layout
        rng = np.random.default_rng(98765)

        # Seed genome
        samples = [layout.genome.copy()]
        # Mutated seed genomes
        for _ in range(7):
            g = layout.genome.copy()
            for _ in range(20):
                a, b = rng.choice(layout.mutable_indices, 2, replace=False)
                g[a], g[b] = g[b], g[a]
            samples.append(g)
        # Fully random genomes
        for _ in range(7):
            g = np.full(layout.n_positions, -1, dtype=np.int32)
            g[layout.frozen_indices] = layout.genome[layout.frozen_indices]
            n_assign = min(len(layout.mutable_indices), layout.n_shortcuts)
            assigned = rng.choice(layout.n_shortcuts, size=n_assign, replace=False)
            g[layout.mutable_indices[:n_assign]] = assigned
            samples.append(g)

        batch = np.asarray(samples, dtype=np.int32)

        evaluator.model._use_cuda = True
        obj_cuda, _ = evaluator.model.evaluate_batch(batch)

        evaluator.model._use_cuda = False
        obj_numba, _ = evaluator.model.evaluate_batch(batch)

        # Allow larger absolute tolerance on the violations objective because it
        # sums many large terms in different orders on GPU vs CPU.
        np.testing.assert_allclose(obj_cuda[:, 0], obj_numba[:, 0], rtol=1e-4, atol=2000.0)
        np.testing.assert_allclose(obj_cuda[:, 1], obj_numba[:, 1], rtol=1e-4, atol=1.0)
        np.testing.assert_allclose(obj_cuda[:, 2], obj_numba[:, 2], rtol=1e-4, atol=2e7)


if __name__ == "__main__":
    unittest.main()
