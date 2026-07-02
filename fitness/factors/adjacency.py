"""Adjacency factor: rewards workflow-related shortcuts being close together."""
import numpy as np
from collections import defaultdict
from core import Layout
from fitness import RewardFactor

class AdjacencyFactor(RewardFactor):
    """Rewards shortcuts being close together based on usage co-occurrence.
    
    Uses app/category proximity as a baseline, then gives stronger signal to
    shortcut sequences and chains observed in usage data. Same app does not
    imply same layer when the shortcut workflows are different.
    """
    name = "adjacency"
    
    def __init__(self, proximity_decay: float = 0.2, usage_weight: float = 5.0):
        self.proximity_decay = proximity_decay
        self.usage_weight = usage_weight
    
    def compute(self, layout: Layout) -> float:
        coords = np.array([(p.x, p.y) for p in layout.positions], dtype=np.float32)
        
        shortcut_pos = {}
        for i, sid in enumerate(layout.genome):
            if sid >= 0:
                shortcut_pos[sid] = i
        
        score = 0.0
        
        # 1. Same-app/category adjacency (baseline)
        for sid_a, pos_a in shortcut_pos.items():
            sc_a = layout.shortcuts[sid_a]
            for sid_b, pos_b in shortcut_pos.items():
                if sid_b <= sid_a:
                    continue
                sc_b = layout.shortcuts[sid_b]
                if sc_a.app != sc_b.app and sc_a.category != sc_b.category:
                    continue
                
                dist = np.linalg.norm(coords[pos_a] - coords[pos_b])
                proximity = max(0.0, 1.0 - dist * self.proximity_decay)
                weight = sc_a.importance * sc_b.importance
                score += weight * proximity
        
        # 2. Cross-app usage sequence adjacency (workflow-aware)
        if layout.usage_data.sequences:
            score += self._usage_adjacency(layout, shortcut_pos, coords)
        
        # 3. Usage chain adjacency (stronger for multi-step sequences)
        if layout.usage_data.chains:
            score += self._chain_adjacency(layout, shortcut_pos, coords)
        
        return score
    
    def _usage_adjacency(self, layout: Layout, shortcut_pos: dict, coords: np.ndarray) -> float:
        """Reward shortcuts that appear in usage sequences being close together."""
        score = 0.0
        
        # Build reverse lookup: keys -> sid
        keys_to_sid = {}
        for sid, pos in shortcut_pos.items():
            sc = layout.shortcuts[sid]
            keys_to_sid[sc.keys] = sid
        
        for seq_key, seq_data in layout.usage_data.sequences.items():
            parts = seq_key.split(" -> ")
            if len(parts) != 2:
                continue
            
            key_a, key_b = parts[0], parts[1]
            if key_a not in keys_to_sid or key_b not in keys_to_sid:
                continue
            
            sid_a = keys_to_sid[key_a]
            sid_b = keys_to_sid[key_b]
            pos_a = shortcut_pos[sid_a]
            pos_b = shortcut_pos[sid_b]
            
            count = seq_data.get("count", 0)
            avg_gap = seq_data.get("avg_gap_ms", 5000)
            
            # Faster transitions = stronger adjacency desire
            speed_weight = max(0.5, 2.0 - avg_gap / 2000.0)
            weight = count * speed_weight * self.usage_weight
            
            # Same layer bonus (0-1 based on same layer or not)
            layer_a = layout.positions[pos_a].layer
            layer_b = layout.positions[pos_b].layer
            if layer_a == layer_b:
                dist = np.linalg.norm(coords[pos_a] - coords[pos_b])
                proximity = max(0.0, 1.0 - dist * self.proximity_decay)
            else:
                # Cross-layer: much smaller bonus, but still non-zero if close
                # (adjacent layers with same physical position are somewhat accessible)
                dist = np.linalg.norm(coords[pos_a] - coords[pos_b])
                proximity = max(0.0, 0.3 - dist * self.proximity_decay)
            
            score += weight * proximity
        
        return score
    
    def _chain_adjacency(self, layout: Layout, shortcut_pos: dict, coords: np.ndarray) -> float:
        """Reward adjacent pairs in usage chains being close together."""
        score = 0.0
        
        keys_to_sid = {}
        for sid, pos in shortcut_pos.items():
            sc = layout.shortcuts[sid]
            keys_to_sid[sc.keys] = sid
        
        for chain_key, chain_data in layout.usage_data.chains.items():
            parts = chain_key.split(" -> ")
            count = chain_data.get("count", 0)
            if count < 2 or len(parts) < 2:
                continue
            
            # Adjacent pairs in chains get strong bonus
            for i in range(len(parts) - 1):
                key_a = parts[i]
                key_b = parts[i + 1]
                if key_a not in keys_to_sid or key_b not in keys_to_sid:
                    continue
                
                sid_a = keys_to_sid[key_a]
                sid_b = keys_to_sid[key_b]
                pos_a = shortcut_pos[sid_a]
                pos_b = shortcut_pos[sid_b]
                
                layer_a = layout.positions[pos_a].layer
                layer_b = layout.positions[pos_b].layer
                
                if layer_a == layer_b:
                    dist = np.linalg.norm(coords[pos_a] - coords[pos_b])
                    proximity = max(0.0, 1.0 - dist * self.proximity_decay)
                else:
                    dist = np.linalg.norm(coords[pos_a] - coords[pos_b])
                    proximity = max(0.0, 0.3 - dist * self.proximity_decay)
                
                weight = count * self.usage_weight * 2.0  # chains are stronger than sequences
                score += weight * proximity
        
        return score
