"""Thumb utilization factor: rewards using thumb positions."""
import numpy as np
from core import Layout
from fitness import RewardFactor

class ThumbUtilizationFactor(RewardFactor):
    """Rewards placing shortcuts on thumb positions. Higher is better."""
    name = "thumb_utilization"
    
    def compute(self, layout: Layout) -> float:
        score = 0.0
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            pos = layout.positions[i]
            if pos.is_thumb:
                shortcut = layout.shortcuts[sid]
                score += shortcut.importance * 2.0
        return score
