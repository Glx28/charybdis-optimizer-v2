"""Export layout to JSON and ZMK formats."""
import json
import os
from typing import Dict, Any
import numpy as np
from core import Layout


def export_layout(layout: Layout, path: str):
    """Export layout to JSON file."""
    assignments = {}
    for i, sid in enumerate(layout.genome):
        if sid < 0:
            continue
        pos = layout.positions[i]
        shortcut = layout.shortcuts[sid]
        key = f"L{pos.layer}_{pos.x}_{pos.y}"
        assignments[key] = {
            "sid": int(sid),
            "keys": shortcut.keys,
            "action": shortcut.action,
            "app": shortcut.app,
            "importance": shortcut.importance,
            "layer": pos.layer,
            "x": pos.x,
            "y": pos.y,
            "hand": pos.hand,
            "finger": pos.finger,
        }
    
    data = {
        "n_positions": layout.n_positions,
        "n_assigned": layout.n_assigned,
        "n_shortcuts": layout.n_shortcuts,
        "assignments": assignments,
    }
    
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"Layout exported to {path}")


def export_to_zmk(layout: Layout, path: str):
    """Export layout to ZMK keymap format."""
    # Group by layer
    layer_keys = {}
    for i, sid in enumerate(layout.genome):
        if sid < 0:
            continue
        pos = layout.positions[i]
        shortcut = layout.shortcuts[sid]
        layer = pos.layer
        if layer not in layer_keys:
            layer_keys[layer] = []
        layer_keys[layer].append({
            "x": pos.x,
            "y": pos.y,
            "behavior": shortcut.action or f"&kp {shortcut.keys}",
        })
    
    with open(path, "w", encoding="utf-8") as f:
        for layer, keys in sorted(layer_keys.items()):
            f.write(f"// Layer {layer}\n")
            for key in keys:
                f.write(f"  {{ x = {key['x']}, y = {key['y']}, behavior = {key['behavior']} }},\n")
            f.write("\n")
    
    print(f"ZMK keymap exported to {path}")
