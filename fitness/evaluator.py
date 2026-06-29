"""Fitness evaluator: wires all factors and produces a FitnessResult."""
import numpy as np
from typing import List, Dict
from core import Layout, FitnessResult
from fitness import FitnessFactor
from fitness.factors.effort import EffortFactor
from fitness.factors.adjacency import AdjacencyFactor
from fitness.factors.finger_balance import FingerBalanceFactor
from fitness.factors.same_finger import SameFingerFactor
from fitness.factors.violation import ViolationFactor
from fitness.factors.workflow_coherence import WorkflowCoherenceFactor
from fitness.factors.learning_curve import LearningCurveFactor
from fitness.factors.app_coherence import AppCoherenceFactor
from fitness.factors.trackball_proximity import TrackballProximityFactor

class FitnessEvaluator:
    """Composable fitness evaluator with optional objective normalization."""
    
    def __init__(self, weights: Dict[str, float] = None, reference_layout: Layout = None, scale_factors=None):
        self.weights = weights or {
            "effort": 1.0,
            "adjacency": 1.5,
            "finger_balance": 0.8,
            "same_finger": 2.0,
            "violations": 50.0,
            "workflow_coherence": 30.0,
            "learning_curve": 0.5,
            "app_coherence": 5.0,
            "trackball_proximity": 2.0,
        }
        self.reference_layout = reference_layout
        self.scale_factors = scale_factors if scale_factors is not None else np.ones(3, dtype=np.float32)
        self.factors = self._build_factors()
    
    def _build_factors(self) -> List[FitnessFactor]:
        return [
            EffortFactor(),
            AdjacencyFactor(),
            FingerBalanceFactor(),
            SameFingerFactor(),
            ViolationFactor(),
            WorkflowCoherenceFactor(),
            LearningCurveFactor(self.reference_layout),
            AppCoherenceFactor(),
            TrackballProximityFactor(),
        ]
    
    def evaluate(self, layout: Layout) -> FitnessResult:
        scores = {}
        for factor in self.factors:
            scores[factor.name] = factor.compute(layout)
        
        effort = scores["effort"] * self.weights.get("effort", 1.0)
        adjacency = -scores["adjacency"] * self.weights.get("adjacency", 1.5)
        violations = (
            scores["finger_balance"] * self.weights.get("finger_balance", 0.8) +
            scores["same_finger"] * self.weights.get("same_finger", 2.0) +
            scores["violations"] * self.weights.get("violations", 50.0) +
            scores["workflow_coherence"] * self.weights.get("workflow_coherence", 30.0) +
            scores["learning_curve"] * self.weights.get("learning_curve", 0.5) -
            scores["app_coherence"] * self.weights.get("app_coherence", 5.0) -
            scores["trackball_proximity"] * self.weights.get("trackball_proximity", 2.0)
        )
        
        objectives = np.array([effort, adjacency, violations], dtype=np.float32)
        objectives = objectives / self.scale_factors
        total = np.sum(objectives)
        
        return FitnessResult(
            objectives=objectives,
            factor_scores=scores,
            total_score=float(total)
        )
