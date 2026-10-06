"""Diagnostic inventory for mutable raw-arrow placements.

Raw arrows are available on frozen L7 and have no mutable-layer acceptance
rule. This report is informational only.
"""
from typing import Dict, List, Tuple

from core import Layout, Shortcut


ARROW_BASE = {
    "LEFTARROW": 1,
    "RIGHTARROW": 2,
    "UPARROW": 3,
    "DOWNARROW": 4,
}

def _arrow_type(shortcut: Shortcut) -> int:
    key = (shortcut.base_key or "").upper()
    return ARROW_BASE.get(key, 0)


def analyze_arrows(layout: Layout) -> Dict:
    """Inventory non-frozen raw arrow placements outside L7."""
    placements: Dict[int, List[Tuple[int, float, float]]] = {t: [] for t in range(1, 5)}
    layers: set = set()
    for i, sid in enumerate(layout.genome):
        if sid < 0 or sid >= layout.n_shortcuts:
            continue
        sc = layout.shortcuts[sid]
        atype = _arrow_type(sc)
        if atype == 0 or sc.modifiers:
            continue
        pos = layout.positions[i]
        if pos.is_frozen or pos.layer == 7:
            continue
        placements[atype].append((int(pos.layer), float(pos.x), float(pos.y)))
        layers.add(pos.layer)
    total = sum(len(v) for v in placements.values())
    return {
        "placements": placements,
        "layers": sorted(layers),
        "total": total,
        "review_only": True,
    }
