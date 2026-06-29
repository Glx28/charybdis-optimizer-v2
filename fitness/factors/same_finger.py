"""Same-finger penalty: penalizes multiple shortcuts on the same finger+layer."""
from collections import defaultdict
from core import Layout
from fitness import FitnessFactor

class SameFingerFactor(FitnessFactor):
    """Penalizes shortcuts on the same finger within the same layer."""
    name = "same_finger"
    
    def compute(self, layout: Layout) -> float:
        layer_finger_sids = defaultdict(list)
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            pos = layout.positions[i]
            layer_finger_sids[(pos.layer, pos.finger)].append(sid)
        
        penalty = 0.0
        for (layer, finger), sids in layer_finger_sids.items():
            if len(sids) < 2:
                continue
            for i, sid_a in enumerate(sids):
                for sid_b in sids[i+1:]:
                    imp = layout.shortcuts[sid_a].importance * layout.shortcuts[sid_b].importance
                    penalty += imp * 0.5
        
        return penalty
