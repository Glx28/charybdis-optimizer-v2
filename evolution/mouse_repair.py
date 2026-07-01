"""Targeted repair to keep high-frequency mouse buttons on the right hand."""
from typing import Dict, List, Optional

import numpy as np

from core import Layout


MOUSE_KEYS = {"MB1", "MB2", "MB3", "MB4", "MB5"}


def analyze_mouse(layout: Layout) -> Dict:
    """Report mouse button hand placement."""
    placements = []
    mutable_placements = []
    frozen_fallback_placements = []
    for i, sid in enumerate(layout.genome):
        if sid < 0 or sid >= layout.n_shortcuts:
            continue
        sc = layout.shortcuts[sid]
        if sc.keys in MOUSE_KEYS:
            pos = layout.positions[i]
            placement = (sc.keys, int(pos.layer), float(pos.x), float(pos.y), pos.hand)
            placements.append(placement)
            if pos.is_frozen:
                frozen_fallback_placements.append(placement)
            else:
                mutable_placements.append(placement)
    right = sum(1 for p in placements if p[4] == "right")
    mutable_right = sum(1 for p in mutable_placements if p[4] == "right")
    return {
        "placements": placements,
        "mutable_placements": mutable_placements,
        "frozen_fallback_placements": frozen_fallback_placements,
        "right_hand": f"{right}/{len(placements)}",
        "mutable_right_hand": f"{mutable_right}/{len(mutable_placements)}",
        "all_right": right == len(placements) and len(placements) > 0,
        "mutable_all_right": mutable_right == len(mutable_placements) and len(mutable_placements) > 0,
    }


def build_candidate_mouse_right_layout(layout: Layout) -> Optional[Layout]:
    """Return a candidate layout that moves any left-hand mouse buttons right.

    For each mouse button currently on the left hand, find the lowest-effort
    (least valuable) mutable right-hand position on the same layer and swap the
    assignments.  If no same-layer right-hand position is free/available, the
    button is moved to the globally lowest-effort mutable right-hand position.
    Non-L0/non-L7 layer numbers are not semantic; do not prefer a specific
    target layer such as L2.
    """
    mouse_sids = []
    for i, sid in enumerate(layout.genome):
        if sid < 0 or sid >= layout.n_shortcuts:
            continue
        sc = layout.shortcuts[sid]
        if sc.keys in MOUSE_KEYS:
            pos = layout.positions[i]
            if pos.is_frozen:
                continue
            mouse_sids.append((sid, i, pos.hand, pos.layer, pos.effort))

    if not mouse_sids:
        return None

    left_hand = [t for t in mouse_sids if t[2] == "left"]
    if not left_hand:
        return None

    genome = layout.genome.astype(np.int32).copy()

    def best_right_position(layer: int, excluded: set) -> Optional[int]:
        candidates = []
        for idx in layout.mutable_indices:
            if idx in excluded:
                continue
            pos = layout.positions[idx]
            if pos.hand != "right" or pos.is_frozen:
                continue
            assigned_sid = int(genome[idx])
            if assigned_sid >= 0 and assigned_sid < layout.n_shortcuts:
                if layout.shortcuts[assigned_sid].keys in MOUSE_KEYS:
                    continue
            if pos.layer == layer:
                candidates.append((0, pos.effort, idx))
            else:
                candidates.append((1, pos.effort, idx))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[:2])
        return candidates[0][2]

    excluded = set()
    for sid, src_idx, hand, layer, _ in left_hand:
        target_idx = best_right_position(layer, excluded)
        if target_idx is None:
            continue
        excluded.add(target_idx)
        displaced = int(genome[target_idx])
        genome[target_idx] = sid
        if displaced >= 0:
            genome[src_idx] = displaced
        else:
            genome[src_idx] = -1

    return layout.clone_with(genome=genome)
