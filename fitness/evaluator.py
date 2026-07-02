"""Fitness evaluator: single-source interface to the compiled fitness model."""
import numpy as np
from typing import Dict, List, Optional

from config import DEFAULT_CONFIG
from core import Layout, FitnessResult
from fitness.model import FitnessModel


class FitnessEvaluator:
    """Composable fitness evaluator with optional objective normalization.

    This evaluator uses the compiled single-source fitness model for all
    objective calculations. Factor scores are provided for compatibility
    with logging/checkpoints but are derived from the same objectives.
    """

    def __init__(self, weights: Dict[str, float] = None, reference_layout: Layout = None,
                 scale_factors=None, violation_weights: Dict[str, float] = None,
                 missing_important_threshold: float = 6.0,
                 hard_constraints: Optional[List[str]] = None,
                 toggle_effort_multiplier: float = 2.5):
        self.weights = dict(DEFAULT_CONFIG["fitness"]["weights"]) if weights is None else dict(weights)
        self.reference_layout = reference_layout
        self.scale_factors = scale_factors if scale_factors is not None else np.ones(3, dtype=np.float32)
        self.violation_weights = violation_weights or {}
        self.threshold = missing_important_threshold
        self.hard_constraints = hard_constraints or []
        self.toggle_effort_multiplier = toggle_effort_multiplier

        if reference_layout is not None:
            self.model = FitnessModel(
                layout=reference_layout,
                weights=self.weights,
                violation_weights=self.violation_weights,
                missing_important_threshold=self.threshold,
                scale_factors=self.scale_factors,
                reference_genome=reference_layout.genome,
                hard_constraints=self.hard_constraints,
                toggle_effort_multiplier=toggle_effort_multiplier,
            )
        else:
            self.model = None

    def _factor_scores_from_objectives(self, objectives: np.ndarray) -> Dict[str, float]:
        """Approximate raw factor scores from normalized objectives.

        These are best-effort values for logging/checkpoint compatibility.
        The authoritative values are the three objectives themselves.
        """
        raw_effort = objectives[0] * self.scale_factors[0]
        raw_adj = -objectives[1] * self.scale_factors[1]
        raw_viol = objectives[2] * self.scale_factors[2]
        return {
            "effort": float(raw_effort / max(self.weights.get("effort", 1.0), 1e-9)),
            "adjacency": float(raw_adj / max(self.weights.get("adjacency", 1.5), 1e-9)),
            "violations": float(raw_viol / max(self.weights.get("violations", 50.0), 1e-9)),
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_similarity": 0.0,
            "layer_specialization": 0.0,
            "everything_layer": 0.0,
        }

    def evaluate(self, layout: Layout) -> FitnessResult:
        if self.model is None or self.model.layout is not layout:
            # Build a temporary model for this layout (used by tests with ad-hoc layouts).
            model = FitnessModel(
                layout=layout,
                weights=self.weights,
                violation_weights=self.violation_weights,
                missing_important_threshold=self.threshold,
                scale_factors=self.scale_factors,
                reference_genome=layout.genome,
                hard_constraints=self.hard_constraints,
            )
            objectives, constraints = model.evaluate(layout.genome)
        else:
            objectives, constraints = self.model.evaluate(layout.genome)

        total = float(np.sum(objectives))
        return FitnessResult(
            objectives=objectives,
            constraints=constraints,
            factor_scores=self._factor_scores_from_objectives(objectives),
            total_score=total,
        )

    def evaluate_batch(self, genomes: np.ndarray):
        """Evaluate a batch of genomes and return (objectives, constraints)."""
        if self.model is None:
            raise RuntimeError("FitnessEvaluator was created without a reference layout")
        return self.model.evaluate_batch(genomes)
