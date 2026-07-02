"""Workflow coherence factor: penalizes high-friction workflow splits.

Workflows are inferred from shortcuts and apps used together. This factor is
about reducing friction in real usage chains, not forcing app-pure layers.
"""
import numpy as np
from collections import defaultdict
from core import Layout
from fitness import FitnessFactor

MOD_ORDER = {"ctrl": 0, "control": 0, "shift": 1, "alt": 2, "win": 3, "gui": 3, "cmd": 3}

class WorkflowCoherenceFactor(FitnessFactor):
    """Penalizes splitting multi-step workflows across layers with high switching costs.
    
    If a user frequently performs a sequence of shortcut A -> shortcut B ->
    shortcut C, those actions should be on the same layer or layers with minimal
    switching cost.
    
    Lower is better (it's a penalty factor).
    """
    name = "workflow_coherence"
    
    def __init__(self, layer_switch_cost: float = 10.0):
        self.layer_switch_cost = layer_switch_cost
    
    def compute(self, layout: Layout) -> float:
        """Compute penalty for workflow incoherence."""
        if not layout.usage_data.chains and not layout.usage_data.workflows:
            return 0.0
        
        # Build reverse lookup: keys -> (sid, pos_idx, layer)
        keys_to_info = {}
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            sc = layout.shortcuts[sid]
            keys_to_info[sc.keys] = (sid, i, layout.positions[i].layer)
            keys_to_info[self._normalize_keys(sc.keys)] = (sid, i, layout.positions[i].layer)
        
        penalty = 0.0
        
        # 1. Penalize chains split across layers
        for chain_key, chain_data in layout.usage_data.chains.items():
            parts = chain_key.split(" -> ")
            count = chain_data.get("count", 0)
            if count < 2 or len(parts) < 2:
                continue
            
            # Get layer for each step in the chain
            layers = []
            for part in parts:
                info = keys_to_info.get(part) or keys_to_info.get(self._normalize_keys(part))
                if info is not None:
                    _, _, layer = info
                    layers.append(layer)
                else:
                    layers.append(None)  # not mapped
            
            # Penalty for each layer switch in the chain
            for i in range(len(layers) - 1):
                l_a = layers[i]
                l_b = layers[i + 1]
                if l_a is None or l_b is None:
                    continue  # one is not mapped, can't penalize
                if l_a != l_b:
                    # Layer switch penalty proportional to count and importance
                    penalty += count * self.layer_switch_cost
        
        # 2. Penalize workflows (longer sequences) split across layers
        for wf_key, wf_data in layout.usage_data.workflows.items():
            parts = wf_key.split(" -> ")
            count = wf_data.get("count", 0)
            if count < 3 or len(parts) < 3:
                continue
            
            layers = []
            for part in parts:
                info = keys_to_info.get(part) or keys_to_info.get(self._normalize_keys(part))
                if info is not None:
                    _, _, layer = info
                    layers.append(layer)
                else:
                    layers.append(None)
            
            # Penalty for each layer switch, stronger for workflows
            for i in range(len(layers) - 1):
                l_a = layers[i]
                l_b = layers[i + 1]
                if l_a is None or l_b is None:
                    continue
                if l_a != l_b:
                    # Workflows are more important than chains
                    penalty += count * self.layer_switch_cost * 2.0
        
        # 3. Penalize blind spots (frequently used but unmapped shortcuts)
        blind_spots = layout.usage_data.blind_spots
        if isinstance(blind_spots, dict):
            for keys, data in blind_spots.items():
                if keys not in keys_to_info and self._normalize_keys(keys) not in keys_to_info:
                    count = data.get("count", 0)
                    penalty += count * self.layer_switch_cost * 0.5
        elif isinstance(blind_spots, list):
            for item in blind_spots:
                keys = item.get("keys", "")
                if keys not in keys_to_info and self._normalize_keys(keys) not in keys_to_info:
                    score = item.get("blind_spot_score", 0)
                    penalty += score * self.layer_switch_cost * 0.5
        
        return penalty

    @staticmethod
    def _normalize_keys(keys: str) -> str:
        parts = [p.strip() for p in (keys or "").replace("++", "+Plus").split("+") if p.strip()]
        if not parts:
            return ""
        modifiers = []
        base = []
        for part in parts:
            norm = part.lower()
            if norm in MOD_ORDER:
                modifiers.append(norm)
            else:
                base.append(norm)
        modifiers.sort(key=lambda m: MOD_ORDER[m])
        return "+".join(modifiers + base)
