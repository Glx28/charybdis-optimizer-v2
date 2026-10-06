"""Acceptance-contract reporting for evolved Charybdis layouts.

Three acceptance tiers:
  optimizer_side_pass  — all checks this module can evaluate without export.
  export_validation_pass — cleared externally once bad_literal_count == 0.
  overall_pass         — both above are true; layout is complete and valid.

This module intentionally does not claim export success.
"""
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from core import Layout


FAKE_SCROLL_KEYS = {"ScrollUp", "ScrollDown"}
MOUSE_BUTTON_KEYS = {"MB1", "MB2", "MB3", "MB4", "MB5"}
UNCOMFORTABLE_SCROLL_X = {7.0, 8.0}
REQUIRED_WINDOWS_SHORTCUTS = ("Win+H", "Win+Tab", "Alt+Tab", "Win+Alt+Space")


def _required_relative_layout_report(layout: Layout) -> Dict:
    """Check each semantic group with declared offsets against its layout."""
    rows = []
    failures = []
    for cluster in layout.semantic_clusters:
        members = list(cluster.get("members", []))
        offsets = [(float(m.get("dx", 0.0)), float(m.get("dy", 0.0))) for m in members]
        required = any(abs(dx) > 0.01 or abs(dy) > 0.01 for dx, dy in offsets)
        if not required:
            continue

        sids = [int(m.get("sid", -1)) for m in members]
        positions_by_sid = {sid: [] for sid in sids}
        for idx, sid in enumerate(layout.genome):
            sid = int(sid)
            if sid in positions_by_sid:
                pos = layout.positions[idx]
                positions_by_sid[sid].append((int(pos.layer), float(pos.x), float(pos.y)))

        layer_member_sets = defaultdict(set)
        for sid, places in positions_by_sid.items():
            for layer, _, _ in places:
                layer_member_sets[layer].add(sid)
        full_layers = [layer for layer, found in layer_member_sets.items() if len(found) == len(sids)]
        if full_layers:
            dominant_layer = max(full_layers, key=lambda layer: len(layer_member_sets[layer]))
        elif layer_member_sets:
            dominant_layer = max(layer_member_sets, key=lambda layer: len(layer_member_sets[layer]))
        else:
            dominant_layer = None

        placements = []
        for sid in sids:
            places = positions_by_sid[sid]
            dominant = next((p for p in places if p[0] == dominant_layer), None)
            placements.append(dominant if dominant is not None else (places[0] if places else None))

        anchor_idx = next((i for i, (offset, placement) in enumerate(zip(offsets, placements))
                           if placement is not None and placement[0] == dominant_layer
                           and abs(offset[0]) <= 0.01 and abs(offset[1]) <= 0.01), None)
        errors = []
        if anchor_idx is not None:
            anchor = placements[anchor_idx]
            for i, (offset, placement) in enumerate(zip(offsets, placements)):
                if i == anchor_idx or placement is None or placement[0] != dominant_layer:
                    continue
                distance = math.hypot(placement[1] - (anchor[1] + offset[0]),
                                      placement[2] - (anchor[2] + offset[1]))
                if distance > 0.5:
                    errors.append({"sid": sids[i], "distance": round(distance, 3)})
        passed = bool(full_layers) and anchor_idx is not None and not errors
        row = {"name": cluster.get("name"), "pass": passed,
               "fully_together": bool(full_layers), "order_errors": errors}
        rows.append(row)
        if not passed:
            failures.append(row)

    return {"acceptance_pass": not failures, "required_count": len(rows),
            "passed_count": len(rows) - len(failures), "failures": failures,
            "clusters": rows}


def _l0_toggle_hold_report(layout: Layout) -> Dict:
    """Require L0 layer access to use holds; place toggles in deeper layers."""
    holds = set()
    toggles = []
    for idx, sid in enumerate(layout.genome):
        pos = layout.positions[idx]
        if pos.layer != 0 or pos.is_frozen:
            continue
        sid = int(sid)
        if sid < 0 or sid >= layout.n_shortcuts:
            continue
        shortcut = layout.shortcuts[sid]
        if not shortcut.is_layer_access or shortcut.access_target_layer < 0:
            continue
        if shortcut.access_is_momentary and pos.is_thumb:
            holds.add(int(shortcut.access_target_layer))
        elif not shortcut.access_is_momentary:
            toggles.append({"idx": int(idx), "keys": shortcut.keys,
                            "target_layer": int(shortcut.access_target_layer)})
    without_hold = [row for row in toggles if row["target_layer"] not in holds]
    return {"preference_satisfied": not without_hold,
            "direct_l0_thumb_hold_targets": sorted(holds),
            "direct_l0_toggles": toggles,
            "toggles_without_direct_hold": without_hold}


def _nested_hold_side_report(layout: Layout) -> Dict:
    """Check that each reachable ordinary hold continuation can alternate thumbs."""
    incoming = [set() for _ in range(32)]
    outgoing = []
    for idx, pos, shortcut in _assigned_shortcuts(layout):
        if (not shortcut.is_layer_access or not shortcut.access_is_momentary
                or pos.is_frozen or not pos.is_thumb):
            continue
        text = f"{shortcut.keys} {shortcut.action} {shortcut.base_key}".lower()
        if "scroll" in text:
            continue
        source, target = int(pos.layer), int(shortcut.access_target_layer)
        if not (0 <= source < 32 and 0 <= target < 32) or source == target:
            continue
        side = 0 if pos.hand == "left" else 1
        incoming[target].add(side)
        outgoing.append({"idx": int(idx), "keys": shortcut.keys, "source_layer": source,
                         "target_layer": target, "side": side})

    failures = []
    for edge in outgoing:
        sides = incoming[edge["source_layer"]]
        if sides and (1 - edge["side"]) not in sides:
            failures.append({**edge, "incoming_thumb_sides": sorted(sides)})
    return {"preference_satisfied": not failures, "review_only": True, "failures": failures,
            "nested_hold_edges_checked": len(failures) + sum(
                1 for edge in outgoing if incoming[edge["source_layer"]]
                and (1 - edge["side"]) in incoming[edge["source_layer"]]
            )}


def _assigned_shortcuts(layout: Layout):
    for idx, sid in enumerate(layout.genome):
        sid = int(sid)
        if sid < 0 or sid >= layout.n_shortcuts:
            continue
        yield idx, layout.positions[idx], layout.shortcuts[sid]


def _find_assigned_keys(layout: Layout, keys: str) -> List[Dict]:
    rows = []
    for idx, pos, shortcut in _assigned_shortcuts(layout):
        if shortcut.keys != keys:
            continue
        rows.append({
            "idx": int(idx),
            "layer": int(pos.layer),
            "x": float(pos.x),
            "y": float(pos.y),
            "hand": pos.hand,
            "frozen": bool(pos.is_frozen),
        })
    return rows


def _fake_scroll_assignments(layout: Layout) -> List[Dict]:
    rows = []
    for idx, pos, shortcut in _assigned_shortcuts(layout):
        if shortcut.keys not in FAKE_SCROLL_KEYS:
            continue
        rows.append({
            "keys": shortcut.keys,
            "idx": int(idx),
            "layer": int(pos.layer),
            "x": float(pos.x),
            "y": float(pos.y),
        })
    return rows


def _scroll_mode_access(layout: Layout) -> List[Dict]:
    rows = []
    for idx, pos, shortcut in _assigned_shortcuts(layout):
        if not shortcut.is_layer_access:
            continue
        text = f"{shortcut.keys} {shortcut.action} {shortcut.base_key}".lower()
        if "scroll" not in text:
            continue
        rows.append({
            "keys": shortcut.keys,
            "target_layer": int(shortcut.access_target_layer),
            "momentary": bool(shortcut.access_is_momentary),
            "idx": int(idx),
            "layer": int(pos.layer),
            "x": float(pos.x),
            "y": float(pos.y),
            "hand": pos.hand,
            "frozen": bool(pos.is_frozen),
        })
    return rows


def _layer_access_assignments(layout: Layout) -> List[Dict]:
    rows = []
    for idx, pos, shortcut in _assigned_shortcuts(layout):
        if not shortcut.is_layer_access:
            continue
        rows.append({
            "keys": shortcut.keys,
            "source_layer": int(pos.layer),
            "target_layer": int(shortcut.access_target_layer),
            "momentary": bool(shortcut.access_is_momentary),
            "idx": int(idx),
            "x": float(pos.x),
            "y": float(pos.y),
            "hand": pos.hand,
            "thumb": bool(pos.is_thumb),
            "frozen": bool(pos.is_frozen),
        })
    return rows


def _reachable_layers_from_access_rows(access_rows: List[Dict]) -> Set[int]:
    reachable_layers: Set[int] = {0}
    for _ in range(32):
        changed = False
        for row in access_rows:
            source = int(row["source_layer"])
            target = int(row["target_layer"])
            if source in reachable_layers and target not in reachable_layers:
                reachable_layers.add(target)
                changed = True
        if not changed:
            break
    return reachable_layers


def _windows_shortcut_report(layout: Layout) -> Dict:
    """Report requested bindings and shortest live-genome access paths."""
    access_rows = _layer_access_assignments(layout)
    paths = {0: []}
    frontier = [0]
    for source in frontier:
        for row in access_rows:
            target = row["target_layer"]
            if row["source_layer"] == source and 0 <= target < 32 and target not in paths:
                paths[target] = paths[source] + [row]
                frontier.append(target)
    placements = {}
    for keys in REQUIRED_WINDOWS_SHORTCUTS:
        rows = _find_assigned_keys(layout, keys)
        for row in rows:
            row["reachable"] = row["layer"] in paths
            row["access_path"] = paths.get(row["layer"])
        placements[keys] = rows
    missing = [keys for keys, rows in placements.items() if not any(row["reachable"] for row in rows)]
    return {"acceptance_pass": not missing, "missing_or_unreachable": missing, "placements": placements}


def _always_available_alt_report(layout: Layout) -> Dict:
    """Require a dedicated, unmodified Alt key directly on L0."""
    placements = []
    for idx, sid in enumerate(layout.genome):
        if sid < 0 or sid >= layout.n_shortcuts:
            continue
        shortcut = layout.shortcuts[int(sid)]
        if shortcut.base_key == "LeftAlt" and not shortcut.modifiers:
            pos = layout.positions[idx]
            placements.append({"layer": int(pos.layer), "x": float(pos.x), "y": float(pos.y)})
    return {"acceptance_pass": any(row["layer"] == 0 for row in placements),
            "placements": placements}


def _norwegian_altgr_report(layout: Layout) -> Dict:
    """Require reachable RightAlt (AltGr), kept off L0 beside shortcut LeftAlt."""
    reachable = _reachable_layers_from_access_rows(_layer_access_assignments(layout))
    placements = []
    for idx, sid in enumerate(layout.genome):
        if sid < 0 or sid >= layout.n_shortcuts:
            continue
        shortcut = layout.shortcuts[int(sid)]
        if shortcut.base_key == "RightAlt" and not shortcut.modifiers:
            pos = layout.positions[idx]
            placements.append({"layer": int(pos.layer), "x": float(pos.x), "y": float(pos.y),
                               "reachable": int(pos.layer) in reachable})
    l0_placements = [row for row in placements if row["layer"] == 0]
    return {"acceptance_pass": any(row["reachable"] for row in placements) and not l0_placements,
            "reachable": any(row["reachable"] for row in placements),
            "right_alt_off_l0": not l0_placements,
            "l0_placements": l0_placements,
            "placements": placements}


def _dynamic_mouse_layer_report(layout: Layout) -> Dict:
    """Detect the generated mouse workflow layer by contents, not layer number.

    Required final shape:
    - one non-L0/non-L7 layer contains MB1-MB5 on right-hand positions
    - no mouse button on that layer occupies right-thumb positions
    - the same layer contains right-hand non-thumb momentary Scroll access
    - momentary access to the layer is not on the right thumb side
    - the layer has a reachable toggle access path
    """
    access_rows = _layer_access_assignments(layout)
    reachable_layers = _reachable_layers_from_access_rows(access_rows)
    candidates = []
    by_layer: Dict[int, Dict] = {}
    for idx, pos, shortcut in _assigned_shortcuts(layout):
        if shortcut.keys in MOUSE_BUTTON_KEYS:
            row = {
                "keys": shortcut.keys,
                "idx": int(idx),
                "layer": int(pos.layer),
                "x": float(pos.x),
                "y": float(pos.y),
                "hand": pos.hand,
                "thumb": bool(pos.is_thumb),
                "frozen": bool(pos.is_frozen),
            }
            if pos.layer == 7:
                continue
            if pos.layer == 0 or pos.is_frozen:
                continue
            layer = int(pos.layer)
            item = by_layer.setdefault(layer, {"layer": layer, "buttons": {}, "scroll_access": []})
            item["buttons"].setdefault(shortcut.keys, []).append(row)
        if shortcut.is_layer_access:
            text = f"{shortcut.keys} {shortcut.action} {shortcut.base_key}".lower()
            if "scroll" in text and pos.layer not in (0, 7) and not pos.is_frozen:
                layer = int(pos.layer)
                item = by_layer.setdefault(layer, {"layer": layer, "buttons": {}, "scroll_access": []})
                item["scroll_access"].append({
                    "keys": shortcut.keys,
                    "idx": int(idx),
                    "layer": layer,
                    "target_layer": int(shortcut.access_target_layer),
                    "momentary": bool(shortcut.access_is_momentary),
                    "x": float(pos.x),
                    "y": float(pos.y),
                    "hand": pos.hand,
                    "thumb": bool(pos.is_thumb),
                    "frozen": bool(pos.is_frozen),
                })

    for layer, item in sorted(by_layer.items()):
        buttons = item["buttons"]
        missing = sorted(MOUSE_BUTTON_KEYS - set(buttons))
        left_buttons = [
            row for rows in buttons.values() for row in rows
            if row["hand"] != "right"
        ]
        # A second copy of the same mouse button on the dynamic mouse layer is
        # only acceptable as exactly one left-side + one right-side pair.
        # Two right-side copies, two left-side copies, or a lone left-side
        # copy without a matching right-side copy, are all duplicate
        # violations on this layer.
        right_counts: Dict[str, int] = {}
        left_counts: Dict[str, int] = {}
        for key, rows in buttons.items():
            for row in rows:
                if row["hand"] == "right":
                    right_counts[key] = right_counts.get(key, 0) + 1
                else:
                    left_counts[key] = left_counts.get(key, 0) + 1
        duplicate_same_side = [
            key for key, count in right_counts.items() if count > 1
        ] + [
            key for key, count in left_counts.items() if count > 1
        ]
        unpaired_left_buttons = [
            key for key, count in left_counts.items()
            if not (count == 1 and right_counts.get(key, 0) == 1)
        ]
        right_thumb_buttons = [
            row for rows in buttons.values() for row in rows
            if row["hand"] == "right" and row["thumb"]
        ]
        right_scroll_momentary = [
            row for row in item["scroll_access"]
            if (
                row["momentary"]
                and row["hand"] == "right"
                and not row["thumb"]
                and row["x"] not in UNCOMFORTABLE_SCROLL_X
            )
        ]
        uncomfortable_right_scroll_momentary = [
            row for row in item["scroll_access"]
            if (
                row["momentary"]
                and row["hand"] == "right"
                and not row["thumb"]
                and row["x"] in UNCOMFORTABLE_SCROLL_X
            )
        ]
        right_thumb_scroll_momentary = [
            row for row in item["scroll_access"]
            if row["momentary"] and row["hand"] == "right" and row["thumb"]
        ]
        momentary_access = [
            row for row in access_rows
            if (
                row["target_layer"] == layer
                and row["momentary"]
                and not row["frozen"]
            )
        ]
        right_thumb_momentary_access = [
            row for row in momentary_access
            if row["hand"] == "right" and row["thumb"]
        ]
        reachable_toggle_access = [
            row for row in access_rows
            if (
                row["source_layer"] in reachable_layers
                and row["target_layer"] == layer
                and not row["momentary"]
                and not row["frozen"]
            )
        ]
        passed = (
            not missing
            and not duplicate_same_side
            and not unpaired_left_buttons
            and not right_thumb_buttons
            and not right_thumb_momentary_access
            and bool(right_scroll_momentary)
            and bool(reachable_toggle_access)
        )
        candidates.append({
            "layer": layer,
            "button_keys_present": sorted(buttons),
            "missing_buttons": missing,
            "non_right_button_placements": left_buttons,
            "duplicate_same_side_buttons": duplicate_same_side,
            "unpaired_left_buttons": unpaired_left_buttons,
            "right_thumb_button_placements": right_thumb_buttons,
            "right_momentary_scroll_access": right_scroll_momentary,
            "uncomfortable_right_momentary_scroll_access": uncomfortable_right_scroll_momentary,
            "right_thumb_momentary_scroll_access": right_thumb_scroll_momentary,
            "momentary_access": momentary_access,
            "right_thumb_momentary_access": right_thumb_momentary_access,
            "reachable_toggle_access": reachable_toggle_access,
            "acceptance_pass": passed,
        })

    passing = [row for row in candidates if row["acceptance_pass"]]
    best_candidate = None
    if candidates:
        best_candidate = max(
            candidates,
            key=lambda row: (
                len(row["button_keys_present"]),
                bool(row["right_momentary_scroll_access"]),
                -len(row["right_thumb_momentary_access"]),
                bool(row["reachable_toggle_access"]),
                -len(row["non_right_button_placements"]),
                -len(row["right_thumb_button_placements"]),
            ),
        )
    failure_guidance = []
    if not passing:
        if not candidates:
            failure_guidance.append("No non-L0/non-L7 layer contains mouse-button or scroll-mode candidates.")
        elif best_candidate is not None:
            if best_candidate["missing_buttons"]:
                failure_guidance.append(
                    "Concentrate missing mouse buttons on the best candidate layer: "
                    + ", ".join(best_candidate["missing_buttons"])
                )
            if best_candidate["duplicate_same_side_buttons"]:
                failure_guidance.append(
                    "Remove same-side duplicate mouse buttons on the candidate mouse layer (only one "
                    "left + one right copy of the same button is allowed): "
                    + ", ".join(best_candidate["duplicate_same_side_buttons"])
                )
            if best_candidate["unpaired_left_buttons"]:
                failure_guidance.append(
                    "Remove left-side mouse-button copies on the candidate mouse layer that have no "
                    "matching single right-side copy: " + ", ".join(best_candidate["unpaired_left_buttons"])
                )
            if best_candidate["right_thumb_button_placements"]:
                failure_guidance.append("Move mouse-button placements off the right-thumb area on the candidate mouse layer.")
            if not best_candidate["right_momentary_scroll_access"]:
                if best_candidate.get("uncomfortable_right_momentary_scroll_access"):
                    failure_guidance.append(
                        "Move momentary Scroll away from uncomfortable x7/x8 positions on the candidate mouse layer."
                    )
                else:
                    failure_guidance.append(
                        "Place a right-hand non-thumb momentary Scroll capability on the candidate mouse layer."
                    )
            if best_candidate["right_thumb_momentary_scroll_access"]:
                failure_guidance.append("Move momentary Scroll off the right-thumb area on the candidate mouse layer.")
            if best_candidate["right_thumb_momentary_access"]:
                failure_guidance.append("Move momentary mouse-layer access off the right-thumb side.")
            if not best_candidate["reachable_toggle_access"]:
                failure_guidance.append("Add or preserve a reachable toggle access path to the candidate mouse layer.")
    return {
        "acceptance_pass": bool(passing),
        "mouse_layer": passing[0]["layer"] if passing else None,
        "best_candidate": best_candidate,
        "candidates": candidates,
        "failure_guidance": failure_guidance,
        "requirements": {
            "non_l0_non_l7_layer": True,
            "mb1_to_mb5_right_hand_same_layer": True,
            "no_mouse_buttons_on_right_thumb_area": True,
            "right_hand_non_thumb_momentary_scroll_on_mouse_layer": True,
            "no_right_thumb_momentary_mouse_layer_access": True,
            "reachable_toggle_access": True,
        },
    }


def _no_same_layer_duplicates_report(layout: Layout) -> Dict:
    """No shortcut may appear more than once on the same layer.

    Exactly two exceptions: Layer 7 is frozen and fully excluded, and the
    dynamic mouse layer allows exactly one extra copy of a core mouse button
    (MB1-MB5), as one left-side + one right-side placement only — never two
    copies on the same side. The mouse exception is tied to the layer's live,
    fully-qualifying dynamic-mouse-layer status this generation (the same
    layer `_dynamic_mouse_layer_report` identifies as `mouse_layer`), not
    merely to whether the layer holds mouse buttons: if the layer stops fully
    qualifying as the dynamic mouse layer, its left-side copy immediately
    becomes an ordinary same-layer duplicate here too.
    """
    dynamic_mouse = _dynamic_mouse_layer_report(layout)
    natural_layer = dynamic_mouse.get("mouse_layer")

    counts: Dict[Tuple[int, int], List[Dict]] = {}
    scroll_access_counts: Dict[int, List[Dict]] = {}
    for idx, pos, shortcut in _assigned_shortcuts(layout):
        if pos.layer == 7:
            continue
        is_scroll_access = shortcut.is_layer_access and "scroll" in (
            f"{shortcut.keys} {shortcut.action} {shortcut.base_key}"
        ).lower()
        if is_scroll_access:
            scroll_access_counts.setdefault(int(pos.layer), []).append({
                "idx": int(idx), "keys": shortcut.keys, "hand": pos.hand,
            })
            continue
        key = (int(pos.layer), int(shortcut.sid))
        counts.setdefault(key, []).append({
            "idx": int(idx),
            "keys": shortcut.keys,
            "hand": pos.hand,
        })

    offenders = []
    for layer, rows in scroll_access_counts.items():
        if len(rows) > 1:
            offenders.append({
                "layer": layer, "sid": "scroll_mode_family", "keys": "Scroll mode access",
                "count": len(rows), "positions": rows,
            })
    for (layer, sid), rows in counts.items():
        if len(rows) <= 1:
            continue
        cap = 1
        if layer == natural_layer and rows[0]["keys"] in MOUSE_BUTTON_KEYS:
            right_ct = sum(1 for r in rows if r["hand"] == "right")
            left_ct = sum(1 for r in rows if r["hand"] != "right")
            if len(rows) == 2 and right_ct == 1 and left_ct == 1:
                cap = 2
        if len(rows) > cap:
            offenders.append({
                "layer": layer,
                "sid": sid,
                "keys": rows[0]["keys"],
                "count": len(rows),
                "positions": rows,
            })

    return {
        "acceptance_pass": len(offenders) == 0,
        "offenders": offenders,
    }


def _global_right_thumb_mouse_button_report(layout: Layout) -> Dict:
    placements = []
    for idx, pos, shortcut in _assigned_shortcuts(layout):
        if shortcut.keys not in MOUSE_BUTTON_KEYS:
            continue
        if pos.layer == 7 or pos.is_frozen:
            continue
        if pos.hand == "right" and pos.is_thumb:
            placements.append({
                "keys": shortcut.keys,
                "idx": int(idx),
                "layer": int(pos.layer),
                "x": float(pos.x),
                "y": float(pos.y),
                "hand": pos.hand,
                "thumb": bool(pos.is_thumb),
            })
    return {
        "acceptance_pass": len(placements) == 0,
        "placements": placements,
    }


def _momentary_only_thumb_clearance_report(layout: Layout) -> Dict:
    """Check thumb areas occupied by momentary access to a layer.

    If a non-L0/non-L7 layer has reachable toggle access, both thumb areas are
    available, but same-side thumb assignments are reported as effort-floor
    positions because the thumb still has to do access/mode work. If it has
    direct thumb momentary access from both sides, both thumb areas are freely
    available. Otherwise, every thumb side used for direct reachable momentary
    access to that layer is restricted and must be empty on the target layer.
    """
    access_rows = _layer_access_assignments(layout)
    reachable_layers = _reachable_layers_from_access_rows(access_rows)
    rows = []
    violations = []
    for layer in sorted({row["target_layer"] for row in access_rows if row["target_layer"] not in (0, 7)}):
        incoming = [
            row for row in access_rows
            if row["target_layer"] == layer and row["source_layer"] in reachable_layers
        ]
        toggles = [row for row in incoming if not row["momentary"]]
        thumb_momentary_hands: Set[str] = {
            row["hand"] for row in incoming
            if row["momentary"] and row["thumb"]
        }
        if toggles:
            effort_floor_assignments = []
            if thumb_momentary_hands != {"left", "right"}:
                for idx, pos, shortcut in _assigned_shortcuts(layout):
                    if pos.layer == layer and pos.is_thumb and pos.hand in thumb_momentary_hands:
                        effort_floor_assignments.append({
                            "keys": shortcut.keys,
                            "idx": int(idx),
                            "layer": int(pos.layer),
                            "x": float(pos.x),
                            "y": float(pos.y),
                            "hand": pos.hand,
                            "native_effort": float(pos.effort),
                            "effective_effort_floor": 4.0,
                        })
            rows.append({
                "layer": layer,
                "rule_applies": False,
                "reason": "reachable_toggle_access_present",
                "momentary_thumb_hands": sorted(thumb_momentary_hands),
                "toggle_access": toggles,
                "effort_floor_assignments": effort_floor_assignments,
                "acceptance_pass": True,
            })
            continue
        if len(thumb_momentary_hands) == 0:
            rows.append({
                "layer": layer,
                "rule_applies": False,
                "reason": "no_thumb_momentary_access",
                "momentary_thumb_hands": sorted(thumb_momentary_hands),
                "acceptance_pass": True,
            })
            continue
        if thumb_momentary_hands == {"left", "right"}:
            rows.append({
                "layer": layer,
                "rule_applies": False,
                "reason": "both_thumb_sides_momentary_access_present",
                "momentary_thumb_hands": sorted(thumb_momentary_hands),
                "acceptance_pass": True,
            })
            continue
        occupied_assignments = []
        restricted_hands = sorted(thumb_momentary_hands)
        for idx, pos, shortcut in _assigned_shortcuts(layout):
            if pos.layer == layer and pos.is_thumb and pos.hand in thumb_momentary_hands:
                occupied_assignments.append({
                    "keys": shortcut.keys,
                    "idx": int(idx),
                    "layer": int(pos.layer),
                    "x": float(pos.x),
                    "y": float(pos.y),
                    "hand": pos.hand,
                })
        row = {
            "layer": layer,
            "rule_applies": True,
            "restricted_hands": restricted_hands,
            "occupied_hand": restricted_hands[0] if len(restricted_hands) == 1 else None,
            "violating_assignments": occupied_assignments,
            "acceptance_pass": len(occupied_assignments) == 0,
        }
        rows.append(row)
        if occupied_assignments:
            violations.append(row)
    return {
        "acceptance_pass": len(violations) == 0,
        "layers": rows,
        "violations": violations,
    }


def _mutable_bluetooth_assignments(layout: Layout) -> List[Dict]:
    rows = []
    for idx, pos, shortcut in _assigned_shortcuts(layout):
        text = f"{shortcut.keys} {shortcut.action} {shortcut.base_key} {shortcut.category}".lower()
        if not ("bluetooth" in text or "bt_sel" in text or "output selection" in text or "out_sel" in text):
            continue
        if pos.is_frozen or pos.layer == 7:
            continue
        rows.append({
            "keys": shortcut.keys,
            "idx": int(idx),
            "layer": int(pos.layer),
            "x": float(pos.x),
            "y": float(pos.y),
        })
    return rows


def _layer7_access_report(layout: Layout) -> Dict:
    """L7 is frozen; only access mode is checked."""
    access_rows = _layer_access_assignments(layout)
    reachable_layers = _reachable_layers_from_access_rows(access_rows)
    momentary = [
        row for row in access_rows
        if row["target_layer"] == 7 and row["momentary"] and row["source_layer"] in reachable_layers
    ]
    toggle = [
        row for row in access_rows
        if row["target_layer"] == 7 and not row["momentary"] and row["source_layer"] in reachable_layers
    ]
    return {
        "acceptance_pass": bool(momentary) and bool(toggle),
        "reachable_momentary_access": momentary,
        "reachable_toggle_access": toggle,
        "content_checked": False,
    }


def _transparent_keys_report(layout: Layout) -> Dict:
    """Classify empty (transparent/fallthrough) positions per dynamic layer.

    A transparent position on a generated layer falls through to lower layers in
    firmware.  This may be intentional (the lower-layer key is still useful) or
    wasteful (a high-value position on a well-populated layer is simply unused).

    Classifications per position (informational only — not used for scoring):
      intentional_fallthrough — same (x, y) carries an assigned key on at least
          one other layer, so the empty spot exposes that lower-layer action.
      wasted_prime — non-thumb position on a layer that has at least half its
          non-thumb slots filled, with no fallthrough value at that (x, y).
      acceptable_empty — low-value or sparse-layer empty position.

    Actual scoring is handled by the compiled fitness kernel (fitness/kernel.py
    raw_scores[16] = empty_position).  The kernel applies a sigmoid-weighted
    penalty to ALL empty positions on reachable non-L0/non-L7 layers, based
    purely on position effort/value — not on these classifications.  The
    classifications here are for human inspection and reporting only.
    """
    n_pos = layout.n_positions
    genome = layout.genome

    # Build lookup: (x_rounded, y_rounded) -> list of (layer, sid) for assigned positions
    coord_to_assigned: Dict[tuple, List] = {}
    layer_stats: Dict[int, Dict] = {}

    for i in range(n_pos):
        pos = layout.positions[i]
        layer = int(pos.layer)
        if layer not in layer_stats:
            layer_stats[layer] = {"total": 0, "assigned": 0, "thumb_total": 0, "thumb_assigned": 0}
        layer_stats[layer]["total"] += 1
        if pos.is_thumb:
            layer_stats[layer]["thumb_total"] += 1

        sid = int(genome[i])
        if sid >= 0:
            layer_stats[layer]["assigned"] += 1
            if pos.is_thumb:
                layer_stats[layer]["thumb_assigned"] += 1
            key = (round(float(pos.x), 1), round(float(pos.y), 1))
            coord_to_assigned.setdefault(key, []).append((layer, sid))

    dynamic_layers = sorted(
        l for l in layer_stats if l != 0 and l != 7
    )

    per_layer = []
    total_intentional = 0
    total_wasted_prime = 0
    total_acceptable = 0

    for layer in dynamic_layers:
        stats = layer_stats[layer]
        non_thumb_total = stats["total"] - stats["thumb_total"]
        non_thumb_assigned = stats["assigned"] - stats["thumb_assigned"]
        # Layer is "well populated" if at least half its non-thumb slots are filled.
        well_populated = non_thumb_total > 0 and non_thumb_assigned >= non_thumb_total * 0.5

        intentional: List[Dict] = []
        wasted_prime: List[Dict] = []
        acceptable: List[Dict] = []

        for i in range(n_pos):
            pos = layout.positions[i]
            if int(pos.layer) != layer:
                continue
            if int(genome[i]) >= 0:
                continue  # assigned, not transparent

            key = (round(float(pos.x), 1), round(float(pos.y), 1))
            other_layers_at_coord = [
                (l, s) for (l, s) in coord_to_assigned.get(key, [])
                if l != layer
            ]
            has_fallthrough = bool(other_layers_at_coord)
            entry = {
                "idx": int(i),
                "layer": layer,
                "x": float(pos.x),
                "y": float(pos.y),
                "hand": pos.hand,
                "thumb": bool(pos.is_thumb),
                "fallthrough_layers": [l for l, _ in other_layers_at_coord],
            }

            if has_fallthrough:
                entry["classification"] = "intentional_fallthrough"
                intentional.append(entry)
                total_intentional += 1
            elif not pos.is_thumb and well_populated:
                entry["classification"] = "wasted_prime"
                wasted_prime.append(entry)
                total_wasted_prime += 1
            else:
                entry["classification"] = "acceptable_empty"
                acceptable.append(entry)
                total_acceptable += 1

        per_layer.append({
            "layer": layer,
            "non_thumb_total": non_thumb_total,
            "non_thumb_assigned": non_thumb_assigned,
            "well_populated": well_populated,
            "intentional_fallthrough_count": len(intentional),
            "wasted_prime_count": len(wasted_prime),
            "acceptable_empty_count": len(acceptable),
            "intentional_fallthrough": intentional,
            "wasted_prime": wasted_prime,
            "acceptable_empty": acceptable,
        })

    return {
        "per_layer": per_layer,
        "totals": {
            "intentional_fallthrough": total_intentional,
            "wasted_prime": total_wasted_prime,
            "acceptable_empty": total_acceptable,
        },
        "scoring_note": (
            "Classifications are informational only. The compiled fitness kernel "
            "applies a sigmoid-weighted effort-proportional penalty to ALL empty "
            "positions on reachable non-L0/non-L7 layers regardless of classification. "
            "There is no reward for intentional_fallthrough and no exemption for "
            "acceptable_empty — position effort determines penalty magnitude."
        ),
    }


def build_acceptance_report(
    layout: Layout,
    duplicate_report: Optional[Dict] = None,
    completion_cluster_report: Optional[Dict] = None,
    arrow_report: Optional[Dict] = None,
) -> Dict:
    """Return checkpoint-level acceptance facts.

    Three tiers (see module docstring):
      optimizer_side_pass  — all checks this module can evaluate.
      export_validation_pass — False until externally confirmed.
      overall_pass         — both tiers pass; layout is valid for deployment.

    During training, only optimizer_side_pass gates the global-best ranking.
    Run final validation at end-of-run; if it fails, consult failure_guidance
    and numeric_distances for which scoring pressures to adjust next run.
    """
    duplicate_report = duplicate_report or {}
    completion_cluster_report = completion_cluster_report or {}
    arrow_report = arrow_report or {}

    unsupported = duplicate_report.get("unsupported_duplicates", [])
    fake_scroll = _fake_scroll_assignments(layout)
    scroll_access = _scroll_mode_access(layout)
    dynamic_mouse = _dynamic_mouse_layer_report(layout)
    right_thumb_mouse = _global_right_thumb_mouse_button_report(layout)
    thumb_clearance = _momentary_only_thumb_clearance_report(layout)
    win_s = _find_assigned_keys(layout, "Win+S")
    mutable_bt = _mutable_bluetooth_assignments(layout)
    layer7_access = _layer7_access_report(layout)
    transparent = _transparent_keys_report(layout)
    no_same_layer_dup = _no_same_layer_duplicates_report(layout)
    windows_shortcuts = _windows_shortcut_report(layout)
    always_available_alt = _always_available_alt_report(layout)
    norwegian_altgr = _norwegian_altgr_report(layout)
    relative_layouts = _required_relative_layout_report(layout)
    l0_toggle_hold = _l0_toggle_hold_report(layout)

    optimizer_side_checks = {
        "requested_windows_shortcuts_reachable": windows_shortcuts["acceptance_pass"],
        "left_alt_directly_available_on_l0": always_available_alt["acceptance_pass"],
        "norwegian_altgr_reachable": norwegian_altgr["reachable"],
        "right_alt_off_l0": norwegian_altgr["right_alt_off_l0"],
        "norwegian_completion_cluster": bool(completion_cluster_report.get("acceptance_pass")),
        "unsupported_duplicates_near_zero": len(unsupported) == 0,
        "no_fake_scroll_keypresses": len(fake_scroll) == 0,
        "scroll_mode_access_present": len(scroll_access) > 0,
        "dynamic_mouse_layer_present": bool(dynamic_mouse.get("acceptance_pass")),
        "no_mouse_buttons_on_right_thumb_area_global": bool(right_thumb_mouse.get("acceptance_pass")),
        "momentary_only_thumb_side_clear": bool(thumb_clearance.get("acceptance_pass")),
        "layer7_momentary_and_toggle_access": bool(layer7_access.get("acceptance_pass")),
        "no_mutable_bluetooth_or_output_keys": len(mutable_bt) == 0,
        "win_s_present": len(win_s) > 0,
        "no_same_layer_duplicates": bool(no_same_layer_dup.get("acceptance_pass")),
        "required_relative_layouts": bool(relative_layouts["acceptance_pass"]),
        "l0_toggles_have_direct_hold_alternative": l0_toggle_hold["preference_satisfied"],
    }
    # Export check is always False during training; set externally after export.
    export_checks = {
        "norwegian_export_bad_literal_count_zero": False,
    }
    checks = {**optimizer_side_checks, **export_checks}

    optimizer_side_pass = all(optimizer_side_checks.values())
    export_validation_pass = all(export_checks.values())
    overall_pass = optimizer_side_pass and export_validation_pass

    # Numeric distances for failed checks — helps tune scoring pressure next run.
    numeric_distances: Dict[str, object] = {}
    if not optimizer_side_checks["unsupported_duplicates_near_zero"]:
        numeric_distances["unsupported_duplicates_count"] = len(unsupported)
    if not optimizer_side_checks["scroll_mode_access_present"]:
        numeric_distances["scroll_access_count"] = 0
    if not optimizer_side_checks["dynamic_mouse_layer_present"]:
        best = dynamic_mouse.get("best_candidate")
        if best:
            numeric_distances["mouse_missing_buttons"] = best.get("missing_buttons", [])
            numeric_distances["mouse_non_right_placements"] = len(best.get("non_right_button_placements", []))
            numeric_distances["mouse_right_thumb_placements"] = len(best.get("right_thumb_button_placements", []))
            numeric_distances["mouse_right_scroll_ok"] = bool(best.get("right_momentary_scroll_access"))
            numeric_distances["mouse_toggle_access_ok"] = bool(best.get("reachable_toggle_access"))
        else:
            numeric_distances["mouse_no_candidate_layer"] = True
    if not optimizer_side_checks["no_mouse_buttons_on_right_thumb_area_global"]:
        numeric_distances["global_right_thumb_mouse_button_count"] = len(
            right_thumb_mouse.get("placements", [])
        )
    if not optimizer_side_checks["momentary_only_thumb_side_clear"]:
        numeric_distances["thumb_clearance_violations"] = len(thumb_clearance.get("violations", []))
    if not optimizer_side_checks["layer7_momentary_and_toggle_access"]:
        numeric_distances["l7_momentary_access_count"] = len(layer7_access.get("reachable_momentary_access", []))
        numeric_distances["l7_toggle_access_count"] = len(layer7_access.get("reachable_toggle_access", []))
    if not optimizer_side_checks["no_same_layer_duplicates"]:
        numeric_distances["same_layer_duplicate_offenders"] = len(no_same_layer_dup.get("offenders", []))
    if not optimizer_side_checks["required_relative_layouts"]:
        numeric_distances["required_relative_layout_failures"] = len(relative_layouts["failures"])

    failure_guidance = list(dynamic_mouse.get("failure_guidance", []))
    if not optimizer_side_checks["no_mouse_buttons_on_right_thumb_area_global"]:
        failure_guidance.append("Move all mouse buttons off right-thumb positions on generated layers.")
    if not optimizer_side_checks["no_same_layer_duplicates"]:
        offenders = no_same_layer_dup.get("offenders", [])
        preview = ", ".join(f"{o['keys']} x{o['count']} on L{o['layer']}" for o in offenders[:5])
        failure_guidance.append(
            "Remove same-layer duplicate shortcuts (only the dynamic mouse layer's "
            f"MB1-MB5 may have one left+one right copy): {preview}"
        )
    if not optimizer_side_checks["required_relative_layouts"]:
        names = ", ".join(str(row["name"]) for row in relative_layouts["failures"])
        failure_guidance.append(f"Restore declared relative placement for semantic groups: {names}.")
    if not optimizer_side_checks["right_alt_off_l0"]:
        failure_guidance.append("Keep LeftAlt on L0 for shortcuts; place RightAlt/AltGr on a reachable deeper layer.")
    if not optimizer_side_checks["l0_toggles_have_direct_hold_alternative"]:
        failure_guidance.append("Give every direct L0 layer toggle a direct thumb hold alternative; place toggles in deeper layers.")
    return {
        "checks": checks,
        "optimizer_side_checks": optimizer_side_checks,
        "optimizer_side_pass": optimizer_side_pass,
        "export_validation_pass": export_validation_pass,
        "overall_pass": overall_pass,
        "external_export_validation_pending": not export_validation_pass,
        "export_validation_required": {
            "bad_literal_count": 0,
            "source": "/home/nos/charybdis/charybdis-tools/runtime/evolved_v2_export",
        },
        "numeric_distances": numeric_distances,
        "failure_guidance": failure_guidance,
        "details": {
            "requested_windows_shortcuts": windows_shortcuts,
            "always_available_left_alt": always_available_alt,
            "norwegian_altgr": norwegian_altgr,
            "win_s_positions": win_s,
            "fake_scroll_assignments": fake_scroll,
            "scroll_mode_access": scroll_access,
            "dynamic_mouse_layer": dynamic_mouse,
            "global_right_thumb_mouse_buttons": right_thumb_mouse,
            "momentary_only_thumb_clearance": thumb_clearance,
            "layer7_access": layer7_access,
            "mutable_bluetooth_or_output_assignments": mutable_bt,
            "transparent_keys": transparent,
            "no_same_layer_duplicates": no_same_layer_dup,
            "required_relative_layouts": relative_layouts,
            "l0_toggle_hold_alternative": l0_toggle_hold,
            "nested_hold_thumb_side_review": _nested_hold_side_report(layout),
        },
    }
