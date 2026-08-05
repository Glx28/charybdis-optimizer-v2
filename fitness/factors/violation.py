"""Protected key groups used by production scoring.

The legacy ViolationFactor class (Python-only test reference path) now lives in
tests/legacy_factors.py. This module keeps only the group definitions imported
by fitness/kernel.py and evolution/__init__.py.

Scroll is intentionally absent from the groups: it is modeled as trackball
scroll-mode access, not ScrollUp/ScrollDown keys.
"""

# Static key groups that should be spatially close when multiple members appear
# on the same layer. They must not force members onto the same layer.
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
