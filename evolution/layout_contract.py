"""Versioned owner layout contract and detailed CPU audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contracts" / "layout_contract_v1.json"
CONTRACT_BYTES = CONTRACT_PATH.read_bytes()
CONTRACT = json.loads(CONTRACT_BYTES)
CONTRACT_SHA256 = hashlib.sha256(CONTRACT_BYTES).hexdigest()
FROZEN_LAYOUT_PATH = CONTRACT_PATH.parents[1] / "data" / "layout.json"


def frozen_layer_content_sha256() -> str:
    canonical = json.loads(FROZEN_LAYOUT_PATH.read_text(encoding="utf-8"))
    content = canonical.get("l7_frozen", {})
    payload = json.dumps(content, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def owner_protected_indices(layout, genome=None):
    """Return fixed contract slots and any owner-fixed controls present in seed."""
    values = layout.genome if genome is None else genome
    fixed = [
        (0, 3, 4, "@access:L1:hold"),
        (1, 8, 2, "MB1"),
        (1, 9, 2, "MB2"),
        (1, 10, 2, "@scroll:L1:hold"),
    ]
    by_coord = {(int(p.layer), int(p.x), int(p.y)): p for p in layout.positions}
    protected = []
    for layer, x, y, required_key in fixed:
        pos = by_coord.get((layer, x, y))
        if pos is None:
            raise ValueError(f"Owner-fixed coordinate missing: L{layer} ({x},{y})")
        sid = int(values[pos.gene_idx])
        if sid < 0 or sid >= layout.n_shortcuts or layout.shortcuts[sid].keys != required_key:
            raise ValueError(
                f"Owner-fixed binding mismatch at L{layer} ({x},{y}); expected {required_key}"
            )
        protected.append(int(pos.gene_idx))
    # Seed-scoped controls are protected when present in the supplied seed;
    # unrelated layouts and synthetic fixtures need not contain them.
    for item in CONTRACT.get("seed_fixed_bindings", ()):
        coord = (int(item["layer"]), int(item["x"]), int(item["y"]))
        pos = by_coord.get(coord)
        if (pos is not None and 0 <= int(values[pos.gene_idx]) < layout.n_shortcuts
                and layout.shortcuts[int(values[pos.gene_idx])].keys == item["key"]):
            protected.append(int(pos.gene_idx))
    return tuple(protected)


def _assigned(layout):
    for idx, sid_value in enumerate(layout.genome):
        sid = int(sid_value)
        if 0 <= sid < layout.n_shortcuts:
            yield idx, layout.positions[idx], layout.shortcuts[sid]


def _is_return(shortcut) -> bool:
    return bool(shortcut.is_layer_access and shortcut.access_target_layer == 0)


def _ordinary_edges(layout):
    edges = []
    for idx, pos, shortcut in _assigned(layout):
        if (shortcut.is_layer_access and shortcut.access_is_momentary
                and not str(shortcut.keys).startswith("@scroll:")
                and 0 < int(shortcut.access_target_layer) < 32):
            edges.append({
                "source": int(pos.layer), "target": int(shortcut.access_target_layer),
                "side": str(pos.hand), "thumb": bool(pos.is_thumb),
                "idx": int(idx), "keys": shortcut.keys,
            })
    return edges


def _shortcut_positions(layout, keys: str):
    return [
        {"idx": int(idx), "layer": int(pos.layer), "x": float(pos.x),
         "y": float(pos.y), "effort": float(pos.effort),
         "hand": str(pos.hand), "thumb": bool(pos.is_thumb)}
        for idx, pos, shortcut in _assigned(layout) if shortcut.keys == keys
    ]


_NORWEGIAN_RAW_ORDER = {
    str(value).upper(): int(order)
    for order, value in CONTRACT["norwegian_raw_key_cluster"]["keys_by_order"].items()
}
_NORWEGIAN_RAW_OFFSETS = {
    int(order): (float(offset[0]), float(offset[1]))
    for order, offset in CONTRACT["norwegian_raw_key_cluster"]["offsets_from_order_2"].items()
}


def _required_semantic_layout_failures(layout):
    failures = []
    for cluster in getattr(layout, "semantic_clusters", ()):
        members = list(cluster.get("members", ()))
        if not any(abs(float(m.get("dx", 0))) > .01 or abs(float(m.get("dy", 0))) > .01
                   for m in members):
            continue
        positions = {int(m.get("sid", -1)): [] for m in members}
        for idx, raw_sid in enumerate(layout.genome):
            sid = int(raw_sid)
            if sid in positions:
                p = layout.positions[idx]
                positions[sid].append((int(p.layer), float(p.x), float(p.y)))
        per_layer = {}
        for sid, rows in positions.items():
            for layer, _, _ in rows:
                per_layer.setdefault(layer, set()).add(sid)
        full = [layer for layer, found in per_layer.items() if len(found) == len(positions)]
        dominant = max(full, key=lambda layer: len(per_layer[layer])) if full else (
            max(per_layer, key=lambda layer: len(per_layer[layer])) if per_layer else None
        )
        placed = []
        for sid in positions:
            placed.append(next((r for r in positions[sid] if r[0] == dominant),
                               positions[sid][0] if positions[sid] else None))
        offsets = [(float(m.get("dx", 0)), float(m.get("dy", 0))) for m in members]
        anchor = next((i for i, (off, row) in enumerate(zip(offsets, placed))
                       if row is not None and row[0] == dominant
                       and abs(off[0]) <= .01 and abs(off[1]) <= .01), None)
        ok = bool(full) and anchor is not None
        if ok:
            ax, ay = placed[anchor][1:]
            ok = all(row is None or row[0] != dominant or
                     ((row[1] - (ax + off[0])) ** 2 + (row[2] - (ay + off[1])) ** 2) <= .25
                     for off, row in zip(offsets, placed))
        if not ok:
            failures.append(str(cluster.get("name", "unnamed")))
    return failures


def _norwegian_raw_cluster_pass(layout):
    found = {}
    raw_layers = set()
    for idx, raw_sid in enumerate(layout.genome):
        sid = int(raw_sid)
        if sid < 0 or sid >= layout.n_shortcuts:
            continue
        shortcut = layout.shortcuts[sid]
        order = _NORWEGIAN_RAW_ORDER.get(str(shortcut.base_key).upper().strip(), 0)
        if not order or shortcut.modifiers:
            continue
        pos = layout.positions[idx]
        found.setdefault(order, (float(pos.x), float(pos.y)))
        if not pos.is_frozen and int(pos.layer) != 7:
            raw_layers.add(int(pos.layer))
    if set(found) != set(range(1, 6)) or len(raw_layers) != 1:
        return False
    ax, ay = found[2]
    return all(order in found and
               abs(found[order][0] - (ax + dx)) <= .5 and
               abs(found[order][1] - (ay + dy)) <= .5
               for order, (dx, dy) in _NORWEGIAN_RAW_OFFSETS.items())


def audit_layout_contract(layout) -> Dict:
    """Return named owner-contract checks and actionable evidence."""
    assigned = list(_assigned(layout))
    by_layer_key: Dict[int, Dict[str, List]] = {}
    scrolls = []
    for idx, pos, shortcut in assigned:
        layer = int(pos.layer)
        by_layer_key.setdefault(layer, {}).setdefault(str(shortcut.keys), []).append((idx, pos, shortcut))
        if str(shortcut.keys).startswith("@scroll:"):
            scrolls.append((idx, pos, shortcut))

    checks: Dict[str, bool] = {}
    details: Dict[str, object] = {}

    relative_failures = _required_semantic_layout_failures(layout)
    checks["required_semantic_cluster_layouts"] = not relative_failures
    details["required_semantic_cluster_failures"] = relative_failures
    norwegian_cluster_pass = _norwegian_raw_cluster_pass(layout)
    checks["norwegian_raw_key_cluster"] = norwegian_cluster_pass
    details["norwegian_raw_key_cluster_pass"] = norwegian_cluster_pass

    access = []
    for idx, pos, shortcut in assigned:
        if (int(pos.layer), float(pos.x), float(pos.y)) == (0, 3.0, 4.0):
            access.append({"idx": int(idx), "keys": shortcut.keys,
                           "target": int(shortcut.access_target_layer),
                           "momentary": bool(shortcut.access_is_momentary),
                           "scroll": str(shortcut.keys).startswith("@scroll:"),
                           "is_access": bool(shortcut.is_layer_access)})
    checks["l0_x3_y4_ordinary_l1_hold"] = any(
        row["is_access"] and row["target"] == 1 and row["momentary"] and not row["scroll"]
        for row in access
    )
    details["essential_access_positions"] = access

    fixed = {}
    for keys, expected in CONTRACT["fixed_mouse_bindings"].items():
        found = [
            {"idx": int(idx), "layer": int(pos.layer), "x": float(pos.x), "y": float(pos.y),
             "target": int(shortcut.access_target_layer), "momentary": bool(shortcut.access_is_momentary)}
            for idx, pos, shortcut in assigned if shortcut.keys == keys
        ]
        expected_tuple = (expected["layer"], expected["x"], expected["y"])
        fixed[keys] = {"expected": expected_tuple, "found": found,
                       "pass": any((r["layer"], r["x"], r["y"]) == expected_tuple for r in found)}
    checks["fixed_l1_mouse_positions"] = all(row["pass"] for row in fixed.values())
    details["fixed_mouse_positions"] = fixed

    button_rows = {}
    for key in CONTRACT["mouse_buttons"]:
        button_rows[key] = [
            {"idx": int(idx), "layer": int(pos.layer), "x": float(pos.x), "y": float(pos.y),
             "hand": str(pos.hand), "thumb": bool(pos.is_thumb)}
            for idx, pos, shortcut in assigned if shortcut.keys.upper() == key and int(pos.layer) == 1
        ]
    checks["all_mouse_buttons_on_l1"] = all(button_rows[key] for key in CONTRACT["mouse_buttons"])
    mb3_ok = any(row["hand"] == "right" and not row["thumb"] for row in button_rows["MB3"])
    checks["mb3_right_non_thumb"] = mb3_ok
    mb4 = [r for r in button_rows["MB4"] if r["hand"] == "right" and not r["thumb"]]
    mb5 = [r for r in button_rows["MB5"] if r["hand"] == "right" and not r["thumb"]]
    pairs = [(a, b) for a in mb4 for b in mb5
             if a["y"] == b["y"] and b["x"] - a["x"] == 1]
    checks["mb4_mb5_adjacent_right_same_row_ordered"] = bool(pairs)
    details["mouse_button_positions"] = button_rows
    details["mb4_mb5_pairs"] = pairs

    scroll_rows = [
        {"idx": int(idx), "layer": int(pos.layer), "x": float(pos.x), "y": float(pos.y),
         "thumb": bool(pos.is_thumb), "target": int(shortcut.access_target_layer),
         "momentary": bool(shortcut.access_is_momentary), "keys": shortcut.keys}
        for idx, pos, shortcut in scrolls
    ]
    fixed_scroll = [r for r in scroll_rows if r["keys"] == "@scroll:L1:hold"
                    and (r["layer"], r["x"], r["y"]) == (1, 10.0, 2.0)
                    and r["target"] == 1 and r["momentary"]]
    additional = [r for r in scroll_rows if r not in fixed_scroll]
    additional_layers = [r["layer"] for r in additional]
    checks["exactly_three_momentary_scrolls"] = (
        len(scroll_rows) == 3 and len(fixed_scroll) == 1
        and all(r["momentary"] for r in scroll_rows)
    )
    checks["two_current_layer_scrolls_on_distinct_eligible_layers"] = (
        len(additional) == 2
        and all(r["layer"] not in (0, 1, 7) and r["target"] == r["layer"] and not r["thumb"]
                for r in additional)
        and len(set(additional_layers)) == 2
    )
    details["scroll_bindings"] = scroll_rows

    reachable = {0}
    edges = _ordinary_edges(layout)
    for _ in range(32):
        newer = reachable | {edge["target"] for edge in edges if edge["source"] in reachable}
        if newer == reachable:
            break
        reachable = newer
    generated = set(range(1, 11)) - {7}
    checks["ordinary_hold_path_to_every_generated_layer"] = generated.issubset(reachable)
    details["ordinary_hold_reachable_layers"] = sorted(reachable)

    restricted_sides: Dict[int, set] = {}
    for edge in edges:
        if edge["source"] in reachable and edge["thumb"] and edge["target"] != 7:
            restricted_sides.setdefault(edge["target"], set()).add(edge["side"])
    thumb_violations = []
    for idx, pos, shortcut in assigned:
        layer, side = int(pos.layer), str(pos.hand)
        if layer == 7 or not pos.is_thumb or side not in restricted_sides.get(layer, set()):
            continue
        if not _is_return(shortcut):
            thumb_violations.append({"idx": int(idx), "layer": layer, "side": side,
                                     "keys": shortcut.keys, "action": shortcut.action})
    checks["momentary_thumb_clearance_all_paths"] = not thumb_violations
    details["momentary_thumb_restrictions"] = {str(layer): sorted(sides)
                                               for layer, sides in restricted_sides.items()}
    details["momentary_thumb_violations"] = thumb_violations

    l1_self_lock = [
        {"idx": int(idx), "keys": shortcut.keys, "x": float(pos.x), "y": float(pos.y)}
        for idx, pos, shortcut in assigned
        if int(pos.layer) == 1 and shortcut.is_layer_access
        and int(shortcut.access_target_layer) == 1 and not shortcut.access_is_momentary
    ]
    checks["reachable_l1_self_lock"] = bool(checks["l0_x3_y4_ordinary_l1_hold"] and l1_self_lock)
    details["l1_self_locks"] = l1_self_lock

    required = {}
    for keys in CONTRACT["required_l1_shortcuts"]:
        rows = _shortcut_positions(layout, keys)
        legal = [row for row in rows if row["layer"] == 1 and not row["thumb"]
                 and row["y"] > 0 and row["effort"] <= 1.0]
        required[keys] = {"all_positions": rows, "legal_l1_positions": legal, "pass": bool(legal)}
        checks[f"required_l1_shortcut:{keys}"] = bool(legal)
    details["workflow_placement_policy"] = {
        "mode": "usage_scored",
        "hard_required_l1_shortcuts": required,
        "note": "Layer use, sequence conjunction, effort, and incumbent familiarity determine workflow placement.",
    }

    group_rows = {}
    for group in CONTRACT["required_groups"]:
        a, b = group
        a_rows = required.get(a, {}).get("legal_l1_positions", [])
        b_rows = required.get(b, {}).get("legal_l1_positions", [])
        together = [(x, y) for x in a_rows for y in b_rows
                    if x["y"] == y["y"] and abs(x["x"] - y["x"]) == 1]
        group_rows[f"{a} | {b}"] = together
        checks[f"required_group:{a}|{b}"] = bool(together)
    details["workflow_group_positions"] = group_rows

    left_alt_l0 = [row for row in _shortcut_positions(layout, "LeftAlt") if row["layer"] == 0]
    right_alt = [row for row in _shortcut_positions(layout, "RightAlt") if row["layer"] != 0]
    checks["left_alt_l0"] = bool(left_alt_l0)
    checks["reachable_non_l0_altgr"] = bool(right_alt and any(r["layer"] in reachable for r in right_alt))
    details["left_alt_l0"] = left_alt_l0
    details["right_alt_non_l0"] = right_alt

    l7_positions = [p for p in layout.positions if int(p.layer) == 7]
    frozen_content_hash = frozen_layer_content_sha256()
    checks["l7_frozen_and_exempt"] = bool(
        l7_positions and all(p.is_frozen for p in l7_positions)
        and frozen_content_hash == CONTRACT["frozen_layer_content_sha256"]
    )
    details["l7_position_count"] = len(l7_positions)
    details["l7_content_source"] = str(FROZEN_LAYOUT_PATH)
    details["l7_content_sha256"] = frozen_content_hash

    return {
        "contract_version": CONTRACT["version"],
        "contract_sha256": CONTRACT_SHA256,
        "checks": checks,
        "acceptance_pass": all(checks.values()),
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "details": details,
    }
