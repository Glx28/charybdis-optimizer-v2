"""Violation factor: aggregates all constraint violations."""
from collections import defaultdict
from core import Layout
from fitness import FitnessFactor

# Static key groups that should stay together on the same layer
KEY_GROUPS = [
    {"name": "arrows", "params": ["Left", "Right", "Up", "Down", "LeftArrow", "RightArrow", "UpArrow", "DownArrow"], "protected": True},
    {"name": "win_directions", "params": ["Left", "Right", "Up", "Down"], "mods_required": "win", "protected": True},
    {"name": "clipboard", "params": ["C", "V", "X", "Z", "Y"], "mods_required": "ctrl", "protected": True},
    {"name": "f_keys_low", "params": ["F1", "F2", "F3", "F4", "F5", "F6"], "protected": True, "base_only": True},
    {"name": "f_keys_high", "params": ["F7", "F8", "F9", "F10", "F11", "F12"], "protected": True, "base_only": True},
]


def shortcut_matches_group(shortcut, group):
    """Check if a shortcut belongs to a protected group."""
    if "params" not in group:
        return False
    params = {p.upper() for p in group.get("params", [])}
    if shortcut.base_key.upper() not in params:
        return False
    
    # base_only means the shortcut must have no modifiers (just the raw base key)
    if group.get("base_only") and shortcut.modifiers:
        return False
    
    mods_req = group.get("mods_required", "")
    if mods_req and not any(mods_req.lower() in m.lower() for m in shortcut.modifiers):
        return False
    return True


class ViolationFactor(FitnessFactor):
    """Aggregates constraint violations. Lower is better."""
    name = "violations"
    
    def __init__(self, weights: dict = None):
        self.sub_weights = weights or {
            "duplicate": 10.0,
            "l0_displacement": 50.0,
            "missing_important": 15.0,
            "cross_layer_duplicate": 8.0,
            "group_split": 200.0,
            "thumb_occupancy": 200.0,
            "arrow_order": 200.0,
            "hand_bias": 2000.0,
            "mouse_layer_access": 5000.0,
        }
    
    def compute(self, layout: Layout) -> float:
        total = 0.0
        total += self._duplicate_penalty(layout) * self.sub_weights["duplicate"]
        total += self._l0_displacement(layout) * self.sub_weights["l0_displacement"]
        total += self._missing_important(layout) * self.sub_weights["missing_important"]
        total += self._cross_layer_duplicate(layout) * self.sub_weights["cross_layer_duplicate"]
        total += self._group_split(layout) * self.sub_weights["group_split"]
        total += self._thumb_occupancy(layout) * self.sub_weights["thumb_occupancy"]
        total += self._arrow_order(layout) * self.sub_weights["arrow_order"]
        total += self._hand_bias(layout) * self.sub_weights["hand_bias"]
        total += self._mouse_layer_access(layout) * self.sub_weights["mouse_layer_access"]
        return total
    
    def _duplicate_penalty(self, layout: Layout) -> float:
        penalty = 0.0
        layer_base_keys = defaultdict(lambda: defaultdict(list))
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            sc = layout.shortcuts[sid]
            if not sc.base_key:
                continue
            layer = layout.positions[i].layer
            layer_base_keys[layer][sc.base_key.upper()].append(sid)
        
        for layer, base_map in layer_base_keys.items():
            for base_key, sids in base_map.items():
                if len(sids) > 1:
                    penalty += (len(sids) - 1) ** 2
        return penalty
    
    def _l0_displacement(self, layout: Layout) -> float:
        penalty = 0.0
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            sc = layout.shortcuts[sid]
            if sc.is_l0_only and layout.positions[i].layer != 0:
                penalty += 50.0 + sc.importance * 2.0
        return penalty
    
    def _missing_important(self, layout: Layout) -> float:
        penalty = 0.0
        assigned = set(layout.genome[layout.genome >= 0])
        for sc in layout.shortcuts:
            if sc.sid not in assigned and sc.importance >= 6.0:
                penalty += sc.importance
        return penalty
    
    def _cross_layer_duplicate(self, layout: Layout) -> float:
        sid_layers = defaultdict(set)
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            sid_layers[sid].add(layout.positions[i].layer)
        
        penalty = 0.0
        for sid, layers in sid_layers.items():
            if len(layers) >= 3:
                extra = len(layers) - 2
                penalty += extra * extra
        return penalty
    
    def _group_split(self, layout: Layout) -> float:
        """Penalize splitting protected groups across multiple layers.
        
        Each protected group should have all its members on the same layer.
        """
        penalty = 0.0
        
        all_groups = list(KEY_GROUPS) + list(layout.dynamic_groups)
        
        for group in all_groups:
            if not group.get("protected"):
                continue
            
            # Find all shortcuts in this group and their layers
            group_layers = defaultdict(list)
            group_sids = set(group.get("sids", []))
            
            for i, sid in enumerate(layout.genome):
                if sid < 0:
                    continue
                if sid in group_sids:
                    group_layers[layout.positions[i].layer].append(i)
                elif "sids" not in group:  # static group uses matching
                    sc = layout.shortcuts[sid]
                    if shortcut_matches_group(sc, group):
                        group_layers[layout.positions[i].layer].append(i)
            
            if len(group_layers) <= 1:
                continue  # All on one layer (or not found) — good
            
            # Penalize based on how split the group is
            # The more layers the group is split across, the higher the penalty
            n_layers = len(group_layers)
            total_members = sum(len(v) for v in group_layers.values())
            
            # Penalty: each extra layer beyond 1 adds a large penalty
            # Plus smaller penalty for uneven distribution
            penalty += (n_layers - 1) * 100.0
            
            # Bonus for having the group mostly on one layer
            max_on_layer = max(len(v) for v in group_layers.values())
            cohesion = max_on_layer / total_members if total_members > 0 else 0
            penalty += (1.0 - cohesion) * 50.0
        
        return penalty
    
    def _thumb_occupancy(self, layout: Layout) -> float:
        """Penalize shortcuts placed on thumb positions that are occupied by the layer access mechanism.
        
        If a layer is accessed via a momentary hold on a thumb, that thumb cannot be used
        to press other keys while the layer is active. Any shortcut placed on an occupied
        thumb position is effectively unreachable.
        """
        penalty = 0.0
        
        for access in layout.layer_access:
            if not access.is_momentary:
                continue  # Toggle access doesn't occupy the thumb
            
            target_layer = access.target_layer
            occupied_hand = access.hand
            
            # Find all thumb positions on the occupied hand for this layer
            for i, sid in enumerate(layout.genome):
                if sid < 0:
                    continue
                pos = layout.positions[i]
                if pos.layer != target_layer:
                    continue
                if not pos.is_thumb:
                    continue
                if pos.hand != occupied_hand:
                    continue
                
                # This shortcut is on an occupied thumb position
                sc = layout.shortcuts[sid]
                # Penalty proportional to importance - wasting a high-importance slot is worse
                penalty += 1.0 + sc.importance * 0.5
        
        return penalty

    def _arrow_order(self, layout: Layout) -> float:
        """Penalize arrow keys that are out of spatial order on the same layer.
        
        When multiple arrow keys are on the same layer:
        - LeftArrow should have lower x than RightArrow (left side)
        - UpArrow and DownArrow should be between LeftArrow and RightArrow
        """
        ARROW_KEYS = {"LEFTARROW", "RIGHTARROW", "UPARROW", "DOWNARROW"}
        
        # Group assigned arrow keys by layer
        layer_arrows = defaultdict(list)  # layer -> [(base_key, x)]
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            sc = layout.shortcuts[sid]
            if sc.base_key.upper() in ARROW_KEYS:
                layer_arrows[layout.positions[i].layer].append((sc.base_key.upper(), layout.positions[i].x))
        
        penalty = 0.0
        for layer, arrows in layer_arrows.items():
            if len(arrows) < 2:
                continue
            
            x_by_key = {k: x for k, x in arrows}
            
            # LeftArrow must be to the left of RightArrow
            if "LEFTARROW" in x_by_key and "RIGHTARROW" in x_by_key:
                left_x = x_by_key["LEFTARROW"]
                right_x = x_by_key["RIGHTARROW"]
                if left_x >= right_x:
                    penalty += (left_x - right_x + 1.0) * 100.0
            
            # UpArrow and DownArrow should be between LeftArrow and RightArrow
            if "LEFTARROW" in x_by_key and "RIGHTARROW" in x_by_key:
                left_x = x_by_key["LEFTARROW"]
                right_x = x_by_key["RIGHTARROW"]
                min_x = min(left_x, right_x)
                max_x = max(left_x, right_x)
                
                for key in ("UPARROW", "DOWNARROW"):
                    if key in x_by_key:
                        x = x_by_key[key]
                        if x < min_x:
                            penalty += (min_x - x + 1.0) * 60.0
                        elif x > max_x:
                            penalty += (x - max_x + 1.0) * 60.0
        
        return penalty

    def _hand_bias(self, layout: Layout) -> float:
        """Penalize mouse-category shortcuts and preferred-hand shortcuts on the wrong hand.
        
        Mouse category (MB1-MB5): 5x penalty for left-hand placement.
        Preferred_hand=right on left hand: 2x penalty.
        Preferred_hand=left on right hand: 2x penalty.
        """
        penalty = 0.0
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            shortcut = layout.shortcuts[sid]
            pos = layout.positions[i]
            
            if shortcut.category == "mouse":
                if pos.is_left:
                    penalty += shortcut.importance * 5.0
                continue
            
            if shortcut.preferred_hand == "right" and pos.is_left:
                penalty += shortcut.importance * 2.0
            elif shortcut.preferred_hand == "left" and pos.is_right:
                penalty += shortcut.importance * 2.0
        
        return penalty

    def _mouse_layer_access(self, layout: Layout) -> float:
        """Penalize mouse shortcuts placed on layers that require right-hand momentary access.
        
        The right hand operates the trackball. If the mouse layer requires holding a
        momentary button on the right side to access it directly, the right thumb is 
        occupied and cannot move the trackball. Toggle/lock access on the right side is fine.
        
        Uses get_occupied_thumbs which traces only momentary accesses TO the target layer.
        """
        penalty = 0.0
        
        # Precompute which layers have right-hand momentary access directly TO them
        right_required_layers = set()
        for layer in set(p.layer for p in layout.positions):
            occupied = layout.get_occupied_thumbs(layer)
            if "right" in occupied:
                right_required_layers.add(layer)
        
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            shortcut = layout.shortcuts[sid]
            if shortcut.category != "mouse":
                continue
            
            layer = layout.positions[i].layer
            if layer in right_required_layers:
                penalty += shortcut.importance * 100.0
        
        return penalty
