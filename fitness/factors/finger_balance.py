"""Finger balance factor: penalizes uneven load across fingers."""
import numpy as np
from collections import defaultdict
from core import Layout
from fitness import FitnessFactor

class FingerBalanceFactor(FitnessFactor):
    """Penalizes uneven finger load. Lower is better."""
    name = "finger_balance"
    
    def compute(self, layout: Layout) -> float:
        finger_load = defaultdict(float)
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            shortcut = layout.shortcuts[sid]
            pos = layout.positions[i]
            finger_load[pos.finger] += shortcut.importance
        
        if not finger_load:
            return 0.0
        
        loads = np.array(list(finger_load.values()), dtype=np.float32)
        mean_load = np.mean(loads)
        if mean_load < 1e-6:
            return 0.0
        
        cv = np.std(loads) / mean_load
        return cv
