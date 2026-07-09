"""Analysis reporting for the Norwegian extra-key cluster.

The Norwegian extra keys are the 5 physical keys that differ between Norwegian
and US International keyboard layouts.  They are referenced by their US
International HID names at their Norwegian physical positions.

This module provides cluster analysis and reporting only.  The optimizer
reaches good placement through normal scoring pressure and group-move
mutations that treat the 5 keys as an atomic unit.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from core import Layout, Shortcut
from evolution.group_shapes import NORWEGIAN_CLUSTER_OFFSETS


# Norwegian extra keys: the 5 physical keys that differ between Norwegian and
# US International keyboard layouts, referenced by US International HID names.
RAW_COMPLETION_FAMILY = {
    "DASH AND UNDERSCORE": 1,
    "EQUALS AND PLUS": 2,
    "GRAVE ACCENT AND TILDE": 3,
    "RIGHT BRACE": 4,
    "BACKSLASH AND PIPE": 5,
}

_ORDER_TO_DISPLAY_NAME = {
    1: "Dash and Underscore",
    2: "Equals and Plus",
    3: "Grave Accent and Tilde",
    4: "Right Brace",
    5: "Backslash and Pipe",
}


def _normalize_base_key(base_key: str) -> str:
    """Stable upper-case key for family membership tests."""
    return (base_key or "").upper().strip()


def completion_order(shortcut: Shortcut) -> int:
    """Return the completion family order for a shortcut, or 0 if not a member."""
    return RAW_COMPLETION_FAMILY.get(_normalize_base_key(shortcut.base_key), 0)


def is_raw_completion_base(shortcut: Shortcut) -> bool:
    """True if the shortcut is an unmodified raw member of the completion family."""
    return completion_order(shortcut) > 0 and len(shortcut.modifiers) == 0


def _cluster_positions(layout: Layout) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Return (base_sids_by_order, modified_sids_by_order) for assigned sids."""
    base_by_order: Dict[int, List[int]] = {o: [] for o in range(1, 9)}
    modified_by_order: Dict[int, List[int]] = {o: [] for o in range(1, 9)}
    for idx, sid in enumerate(layout.genome):
        if sid < 0 or sid >= layout.n_shortcuts:
            continue
        shortcut = layout.shortcuts[sid]
        order = completion_order(shortcut)
        if order <= 0:
            continue
        if is_raw_completion_base(shortcut):
            base_by_order[order].append(int(sid))
        else:
            modified_by_order[order].append(int(sid))
    return base_by_order, modified_by_order


def analyze_completion_cluster(layout: Layout) -> Dict:
    """Produce a diagnostic report for the completion cluster.

    Fields:
      - anchor_layer: layer with the most unique raw base members (excluding L7).
      - raw_base_keys_present: family members whose unmodified raw key is on anchor_layer.
      - raw_base_keys_missing: family members whose unmodified raw key is missing from anchor_layer.
      - modified_variants_demand: family members with a modified variant assigned anywhere.
      - layers_used_by_family: layers (excluding frozen L7) that hold any family member.
      - compactness_order_score: higher-is-better heuristic for anchor-layer compactness/order.
      - raw_base_layers_used: number of non-L7 layers holding unmodified raw members.
      - raw_total_count: total assigned family members (base + modified).
      - raw_base_total_count: total assigned unmodified raw members.
    """
    base_by_order, modified_by_order = _cluster_positions(layout)

    base_layers: Dict[int, set] = {o: set() for o in range(1, 9)}
    modified_present: set = set()
    base_positions: Dict[int, Tuple[int, float, float]] = {}
    raw_base_layers_all: set = set()
    modified_layers_all: set = set()
    all_layers: set = set()

    for order in range(1, 6):
        for sid in base_by_order[order]:
            idx = int(np.where(layout.genome == sid)[0][0])
            pos = layout.positions[idx]
            if not pos.is_frozen and pos.layer != 7:
                base_layers[order].add(pos.layer)
                raw_base_layers_all.add(pos.layer)
                all_layers.add(pos.layer)
            base_positions[order] = (idx, pos.x, pos.y)
        if modified_by_order[order]:
            modified_present.add(order)
            for sid in modified_by_order[order]:
                idx = int(np.where(layout.genome == sid)[0][0])
                pos = layout.positions[idx]
                if not pos.is_frozen and pos.layer != 7:
                    modified_layers_all.add(pos.layer)
                    all_layers.add(pos.layer)

    # Anchor layer: non-L7 layer with most unique base members, tie-break by count.
    anchor_layer: Optional[int] = None
    best_unique = -1
    best_count = -1
    layer_base_counts: Dict[int, int] = {}
    layer_base_unique: Dict[int, int] = {}
    for order in range(1, 6):
        for layer in base_layers[order]:
            layer_base_counts[layer] = layer_base_counts.get(layer, 0) + 1
            layer_base_unique[layer] = layer_base_unique.get(layer, 0) + 1
    for layer, unique in layer_base_unique.items():
        count = layer_base_counts[layer]
        if unique > best_unique or (unique == best_unique and count > best_count):
            best_unique = unique
            best_count = count
            anchor_layer = layer

    raw_base_keys_present = []
    raw_base_keys_missing = []
    for order in range(1, 6):
        name = _ORDER_TO_DISPLAY_NAME[order]
        on_anchor = anchor_layer is not None and anchor_layer in base_layers[order]
        if on_anchor:
            raw_base_keys_present.append(name)
        else:
            raw_base_keys_missing.append(name)

    modified_variants_demand = [_ORDER_TO_DISPLAY_NAME[o] for o in sorted(modified_present)]

    raw_base_layers_used = len({layer for layer in layer_base_counts if layer != 7})
    raw_base_total_count = sum(len(sids) for sids in base_by_order.values())
    raw_total_count = raw_base_total_count + sum(len(sids) for sids in modified_by_order.values())

    compactness_order_score = 0.0
    ordered_left_to_right = False
    exact_shape_preserved = False
    wrong_shape_members = []
    anchor_x_by_order: Dict[int, float] = {}
    if anchor_layer is not None:
        xs = []
        ys = []
        orders_on_anchor = []
        for order in range(1, 6):
            if anchor_layer in base_layers[order]:
                _, x, y = base_positions[order]
                xs.append(x)
                ys.append(y)
                orders_on_anchor.append(order)
                anchor_x_by_order[order] = x
        if xs:
            x_span = max(xs) - min(xs)
            y_span = max(ys) - min(ys)
            # Count inversions against left-to-right canonical order.
            inversions = 0
            for i in range(len(orders_on_anchor)):
                for j in range(i + 1, len(orders_on_anchor)):
                    if orders_on_anchor[i] < orders_on_anchor[j] and xs[i] > xs[j] + 0.5:
                        inversions += 1
                    elif orders_on_anchor[i] > orders_on_anchor[j] and xs[i] < xs[j] - 0.5:
                        inversions += 1
            # Reward presence, penalise spread and out-of-order placement.
            compactness_order_score = (
                len(xs) * 10.0
                - x_span * 1.5
                - max(0.0, y_span - 1.0) * 4.0
                - inversions * 5.0
            )
            if 2 in base_positions:
                _, ax, ay = base_positions[2]
                n_correct = 0
                for order, (dx, dy) in NORWEGIAN_CLUSTER_OFFSETS.items():
                    expected_x = ax + dx
                    expected_y = ay + dy
                    if (
                        order in base_positions
                        and abs(base_positions[order][1] - expected_x) <= 0.5
                        and abs(base_positions[order][2] - expected_y) <= 0.5
                    ):
                        n_correct += 1
                    else:
                        wrong_shape_members.append(_ORDER_TO_DISPLAY_NAME[order])
                exact_shape_preserved = n_correct == len(NORWEGIAN_CLUSTER_OFFSETS)
                ordered_left_to_right = exact_shape_preserved

    raw_base_layers = sorted(int(l) for l in raw_base_layers_all if l != 7)
    modified_variant_layers = sorted(int(l) for l in modified_layers_all if l != 7)
    all_family_layers = sorted(int(l) for l in all_layers if l != 7)
    anchor_contains_all_reachable = len(raw_base_keys_missing) == 0
    raw_base_concentrated = len(raw_base_layers) == 1
    compactness_positive = compactness_order_score > 0.0
    # Acceptance: the 5 unmodified Norwegian extra physical keys must appear as
    # one exact-shape movable cluster. The layer/anchor may change; the relative
    # offsets are fixed in evolution.group_shapes.NORWEGIAN_CLUSTER_OFFSETS.
    acceptance_pass = anchor_contains_all_reachable and raw_base_concentrated and exact_shape_preserved

    return {
        "anchor_layer": anchor_layer,
        "raw_base_keys_present": raw_base_keys_present,
        "raw_base_keys_missing": raw_base_keys_missing,
        "modified_variants_demand": modified_variants_demand,
        # Backward-compatible field: all non-L7 family layers including
        # modified variants.  Acceptance should use raw_base_layers below.
        "layers_used_by_family": all_family_layers,
        "raw_base_layers": raw_base_layers,
        "modified_variant_layers": modified_variant_layers,
        "all_family_layers": all_family_layers,
        "compactness_order_score": round(compactness_order_score, 3),
        "ordered_left_to_right": ordered_left_to_right,
        "exact_shape_preserved": exact_shape_preserved,
        "wrong_shape_members": wrong_shape_members,
        "anchor_contains_all_reachable_raw_base_keys": anchor_contains_all_reachable,
        "raw_base_concentrated_one_layer": raw_base_concentrated,
        "raw_base_concentrated_le_2_layers": len(raw_base_layers) <= 2,
        "compactness_order_score_positive": compactness_positive,
        "acceptance_pass": acceptance_pass,
        "raw_base_layers_used": raw_base_layers_used,
        "raw_total_count": raw_total_count,
        "raw_base_total_count": raw_base_total_count,
    }
