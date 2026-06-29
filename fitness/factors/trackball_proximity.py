"""Trackball proximity factor: rewards mouse-related shortcuts near trackball."""
import numpy as np
import re
from core import Layout
from fitness import RewardFactor

class TrackballProximityFactor(RewardFactor):
    """Rewards mouse-related shortcuts being close to the trackball."""
    name = "trackball_proximity"
    
    def __init__(self, trackball_x: float = 7.0, trackball_y: float = 3.5):
        self.trackball_pos = np.array([trackball_x, trackball_y], dtype=np.float32)
        self.mouse_terms = (
            "mouse", "click", "scroll", "wheel", "trackball", "cursor",
            "pointer", "drag", "mb1", "mb2", "mb3", "mb4", "mb5",
        )
    
    def compute(self, layout: Layout) -> float:
        coords = np.array([(p.x, p.y) for p in layout.positions], dtype=np.float32)
        score = 0.0
        
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            shortcut = layout.shortcuts[sid]
            if not self._is_trackball_related(shortcut):
                continue
            
            dist = np.linalg.norm(coords[i] - self.trackball_pos)
            proximity = max(0.0, 1.0 - dist * 0.3)
            score += shortcut.importance * proximity
        
        return score

    def _is_trackball_related(self, shortcut) -> bool:
        haystack = " ".join((
            shortcut.keys,
            shortcut.action,
            shortcut.category,
            shortcut.base_key,
        )).lower()
        if any(term in haystack for term in self.mouse_terms):
            return True
        return re.search(r"\bmb\s*[1-5]\b", haystack) is not None
