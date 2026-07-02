"""Analysis reporting for non-L7 raw arrow clusters.

The acceptance contract wants non-L7 raw arrows to be less desirable than L7
fallback arrows. If workflow evidence earns a mutable raw-arrow cluster, it
must be complete on one layer and use one of two explicit shapes:

- one-line: Left, Up, Down, Right
- two-line: Left, Down, Right on the bottom row, with Up directly above Down

This module reports whether the current genome satisfies that shape. It does
not repair or clear arrows. Whole-arrow-group movement is handled by mutation
proposals and scoring decides whether those proposals survive.
"""
from typing import Dict, List, Tuple

from core import Layout, Shortcut


ARROW_BASE = {
    "LEFTARROW": 1,
    "RIGHTARROW": 2,
    "UPARROW": 3,
    "DOWNARROW": 4,
}

_ARROW_NAME = {1: "LeftArrow", 2: "RightArrow", 3: "UpArrow", 4: "DownArrow"}


def _allowed_arrow_shape(type_positions: Dict[int, Tuple[float, float]]) -> bool:
    if set(type_positions) != {1, 2, 3, 4}:
        return False
    lx, ly = type_positions[1]
    rx, ry = type_positions[2]
    ux, uy = type_positions[3]
    dx, dy = type_positions[4]

    same_line = (
        abs(ly - uy) <= 0.25
        and abs(uy - dy) <= 0.25
        and abs(dy - ry) <= 0.25
        and lx < ux < dx < rx
        and (rx - lx) <= 4.5
    )
    split_cluster = (
        abs(ly - dy) <= 0.25
        and abs(dy - ry) <= 0.25
        and lx < dx < rx
        and uy < dy
        and abs(ux - dx) <= 0.25
        and (dy - uy) <= 2.0
        and (rx - lx) <= 3.5
    )
    return same_line or split_cluster


def _arrow_type(shortcut: Shortcut) -> int:
    key = (shortcut.base_key or "").upper()
    return ARROW_BASE.get(key, 0)


def _is_raw_arrow(shortcut: Shortcut) -> bool:
    return _arrow_type(shortcut) > 0 and len(shortcut.modifiers) == 0


def analyze_arrows(layout: Layout) -> Dict:
    """Report non-frozen raw arrow placements outside L7."""
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
    type_positions = {
        atype: (rows[0][1], rows[0][2])
        for atype, rows in placements.items()
        if len(rows) == 1
    }
    has_allowed_shape = total == 4 and len(layers) == 1 and _allowed_arrow_shape(type_positions)
    return {
        "placements": placements,
        "layers": sorted(layers),
        "total": total,
        "is_complete_cluster": total == 4 and len(layers) == 1 and len(type_positions) == 4,
        "allowed_cluster_shape": has_allowed_shape,
        "allowed_shapes": ["left-up-down-right same row", "left-down-right bottom row with up above down"],
        "acceptance_pass": total == 0 or has_allowed_shape,
    }
