"""Single-source fitness model.

This is the public interface to the compiled fitness kernel. It precomputes
all static layout data once and exposes `evaluate()`, `evaluate_batch()`,
and constraint arrays.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from fitness.kernel import precompute, _evaluate_batch, _single_genome, NUMBA_AVAILABLE


@dataclass
class ParityResult:
    ok: bool
    max_abs_diff: float
    message: str


class FitnessModel:
    """Compiled fitness model: one source of truth for all evaluations."""

    def __init__(
        self,
        layout,
        weights: Dict[str, float],
        violation_weights: Dict[str, float],
        missing_important_threshold: float,
        scale_factors: Optional[np.ndarray] = None,
        reference_genome: Optional[np.ndarray] = None,
        hard_constraints: Optional[List[str]] = None,
        toggle_effort_multiplier: float = 2.5,
    ):
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba is required by the single-source fitness model")

        self.layout = layout
        self.weights = weights
        self.violation_weights = violation_weights
        self.missing_important_threshold = missing_important_threshold
        self.scale_factors = scale_factors if scale_factors is not None else np.ones(3, dtype=np.float32)
        self.reference_genome = reference_genome
        self.hard_constraints = hard_constraints or []

        self.toggle_effort_multiplier = toggle_effort_multiplier
        self.arrays = precompute(
            layout,
            weights=weights,
            violation_weights=violation_weights,
            missing_important_threshold=missing_important_threshold,
            scale_factors=self.scale_factors,
            reference_genome=reference_genome,
            hard_constraints=self.hard_constraints,
            toggle_effort_multiplier=toggle_effort_multiplier,
        )

        # Force one JIT compile/warm-up call on a single genome.
        _single_genome(layout.genome, *self.arrays)

    @property
    def n_constraints(self) -> int:
        return len(self.hard_constraints)

    def evaluate(self, genome: np.ndarray):
        """Return (objectives, constraints) for a single genome."""
        genome = np.asarray(genome, dtype=np.int32)
        return _single_genome(genome, *self.arrays)

    def evaluate_batch(self, genomes: np.ndarray):
        """Return ((batch, 3) objectives, (batch, n_constr) constraints)."""
        genomes = np.asarray(genomes, dtype=np.int32)
        if genomes.ndim == 1:
            genomes = genomes.reshape(1, -1)
        return _evaluate_batch(genomes, *self.arrays)

    def validate_parity(
        self,
        evaluator=None,
        n: int = 64,
        tolerance: float = 1e-4,
    ) -> ParityResult:
        """Check parity against a reference Python evaluator.

        If `evaluator` is None, parity is checked against the seed genome only
        and simply verifies the kernel runs without error.
        """
        rng = np.random.default_rng(12345)
        samples = [self.layout.genome.astype(np.int32).copy()]
        mutable = self.layout.mutable_indices
        for _ in range(max(0, n - 1)):
            genome = self.layout.genome.astype(np.int32).copy()
            if len(mutable) > 1:
                a, b = rng.choice(mutable, size=2, replace=False)
                genome[a], genome[b] = genome[b], genome[a]
            samples.append(genome)
        batch = np.asarray(samples, dtype=np.int32)
        compiled, _ = self.evaluate_batch(batch)

        if evaluator is None:
            return ParityResult(True, 0.0, "no reference evaluator provided; kernel executed successfully")

        oracle = np.asarray([
            evaluator.evaluate(self.layout.clone_with(genome=genome)).objectives
            for genome in batch
        ], dtype=np.float32)
        max_diff = float(np.max(np.abs(compiled - oracle)))
        ok = bool(np.allclose(compiled, oracle, atol=tolerance, rtol=1e-5))
        return ParityResult(ok, max_diff, "ok" if ok else f"max diff {max_diff:.6g} exceeds tolerance")
