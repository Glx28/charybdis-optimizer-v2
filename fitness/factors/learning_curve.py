"""Learning curve factor: penalizes deviations from the current/preferred layout.

Lower is better. If a user has already learned a layout, changes cost learning effort.
"""
from core import Layout
from fitness import FitnessFactor

class LearningCurveFactor(FitnessFactor):
    """Penalizes layout changes from a reference genome. Lower is better."""
    name = "learning_curve"
    
    def __init__(self, reference_layout: Layout = None, swap_cost: float = 1.0):
        self.reference_genome = reference_layout.genome.copy() if reference_layout else None
        self.swap_cost = swap_cost
    
    def compute(self, layout: Layout) -> float:
        if self.reference_genome is None:
            return 0.0
        
        cost = 0.0
        for i in range(len(layout.genome)):
            if layout.genome[i] == self.reference_genome[i]:
                continue
            
            ref_sid = self.reference_genome[i]
            new_sid = layout.genome[i]
            
            # Both assigned: swap
            if ref_sid >= 0 and new_sid >= 0:
                imp = layout.shortcuts[ref_sid].importance if ref_sid < len(layout.shortcuts) else 5.0
                cost += self.swap_cost + imp * 0.5 + imp * imp * 0.01
            # Was empty, now assigned: addition
            elif ref_sid < 0 and new_sid >= 0:
                cost += 0.3
            # Was assigned, now empty: removal
            elif ref_sid >= 0 and new_sid < 0:
                imp = layout.shortcuts[ref_sid].importance if ref_sid < len(layout.shortcuts) else 5.0
                cost += 3.0 + imp * 1.0 + imp * imp * 0.02
        
        return cost
