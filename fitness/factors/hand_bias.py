"""Hand bias factor: penalizes mouse shortcuts on wrong hand."""
from core import Layout
from fitness import FitnessFactor

class HandBiasFactor(FitnessFactor):
    """Penalizes placing shortcuts on the wrong hand.
    
    Mouse-category shortcuts strongly prefer the right hand (5x penalty).
    Shortcuts with explicit preferred_hand get a 2x penalty when placed wrong.
    """
    name = "hand_bias"
    
    def compute(self, layout: Layout) -> float:
        penalty = 0.0
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            shortcut = layout.shortcuts[sid]
            pos = layout.positions[i]
            
            # Mouse category: strongly prefer right hand
            if shortcut.category == "mouse":
                if pos.is_left:
                    penalty += shortcut.importance * 5.0
                continue
            
            # Explicit preferred hand
            if shortcut.preferred_hand == "right" and pos.is_left:
                penalty += shortcut.importance * 2.0
            elif shortcut.preferred_hand == "left" and pos.is_right:
                penalty += shortcut.importance * 2.0
        
        return penalty
