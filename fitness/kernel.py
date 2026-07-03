"""Single-source compiled fitness kernel.

This module is the one source of truth for layout fitness evaluation.
All precomputable data is assembled by `precompute()`; the JIT kernels
`_evaluate_single()` and `_evaluate_batch()` consume those arrays.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Iterable, Tuple

import numpy as np
from config import DEFAULT_CONFIG

try:
    from numba import njit, prange
except Exception:  # pragma: no cover
    njit = None
    prange = range

NUMBA_AVAILABLE = njit is not None

DEFAULT_FITNESS_WEIGHTS = DEFAULT_CONFIG["fitness"]["weights"]
DEFAULT_VIOLATION_WEIGHTS = DEFAULT_CONFIG["fitness"]["violation_sub_weights"]


def _id_map(values: Iterable[object]) -> Tuple[np.ndarray, Dict[object, int]]:
    mapping: Dict[object, int] = {}
    ids = []
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
        ids.append(mapping[value])
    return np.asarray(ids, dtype=np.int32), mapping


def _sid_lookup(layout) -> Dict[str, int]:
    """Map shortcut keys and normalized keys to sid."""
    from fitness.factors.workflow_coherence import WorkflowCoherenceFactor
    lookup: Dict[str, int] = {}
    normalizer = WorkflowCoherenceFactor._normalize_keys
    for shortcut in layout.shortcuts:
        lookup[shortcut.keys] = shortcut.sid
        lookup[normalizer(shortcut.keys)] = shortcut.sid
    return lookup


def _chain_rows(layout, source: dict, min_count: int, multiplier: float) -> np.ndarray:
    from fitness.factors.workflow_coherence import WorkflowCoherenceFactor
    rows = []
    lookup = _sid_lookup(layout)
    normalizer = WorkflowCoherenceFactor._normalize_keys
    for chain_key, data in source.items():
        parts = chain_key.split(" -> ")
        count = data.get("count", 0) if isinstance(data, dict) else 0
        if count < min_count or len(parts) < min_count:
            continue
        for a, b in zip(parts, parts[1:]):
            sid_a = lookup.get(a)
            if sid_a is None:
                sid_a = lookup.get(normalizer(a), -1)
            sid_b = lookup.get(b)
            if sid_b is None:
                sid_b = lookup.get(normalizer(b), -1)
            if sid_a is not None and sid_b is not None and sid_a >= 0 and sid_b >= 0:
                rows.append((int(sid_a), int(sid_b), float(count) * multiplier))
    return np.asarray(rows, dtype=np.float32).reshape((-1, 3)) if rows else np.empty((0, 3), dtype=np.float32)


def _sequence_rows(layout) -> np.ndarray:
    from fitness.factors.workflow_coherence import WorkflowCoherenceFactor

    rows = []
    lookup = _sid_lookup(layout)
    normalizer = WorkflowCoherenceFactor._normalize_keys
    for seq_key, data in layout.usage_data.sequences.items():
        parts = seq_key.split(" -> ")
        if len(parts) != 2 or not isinstance(data, dict):
            continue
        sid_a = lookup.get(parts[0])
        if sid_a is None:
            sid_a = lookup.get(normalizer(parts[0]), -1)
        sid_b = lookup.get(parts[1])
        if sid_b is None:
            sid_b = lookup.get(normalizer(parts[1]), -1)
        if sid_a < 0 or sid_b < 0:
            continue
        count = float(data.get("count", 0))
        avg_gap = float(data.get("avg_gap_ms", 5000))
        confidence = float(data.get("confidence", 1.0))
        speed_weight = max(0.5, 2.0 - avg_gap / 2000.0)
        rows.append((sid_a, sid_b, count * speed_weight * confidence * 5.0))
    return np.asarray(rows, dtype=np.float32).reshape((-1, 3)) if rows else np.empty((0, 3), dtype=np.float32)


@lru_cache(maxsize=4096)
def _normalize_app_name(name: str) -> str:
    lowered = (name or "").lower()
    for suffix in (".exe", " (chrome/edge)", "/powershell"):
        lowered = lowered.replace(suffix, "")
    return "".join(ch for ch in lowered if ch.isalnum() or ch.isspace()).strip()


APP_ALIASES = {
    "browser": ("msedge", "chrome", "firefox", "brave", "vivaldi", "browser chrome edge"),
    "chrome": ("msedge", "chrome", "browser"),
    "edge": ("msedge", "edge", "browser"),
    "visual studio code": ("code", "cursor", "vscode", "vs code"),
    "windows terminal": ("windowsterminal", "terminal", "powershell", "pwsh", "cmd"),
    "terminal": ("windowsterminal", "powershell", "pwsh", "cmd"),
    "file explorer": ("explorer",),
    "excel": ("excel",),
    "microsoft excel": ("excel",),
    "word": ("winword",),
    "microsoft word": ("winword",),
    "teams": ("teams", "ms teams"),
}


@lru_cache(maxsize=4096)
def _app_tokens(name: str) -> set:
    norm = _normalize_app_name(name)
    tokens = {norm}
    for key, values in APP_ALIASES.items():
        key_norm = _normalize_app_name(key)
        if key_norm in norm or norm in key_norm:
            tokens.update(_normalize_app_name(v) for v in values)
    return {t for t in tokens if t}


def _app_matches(a: str, b: str) -> bool:
    a_tokens = _app_tokens(a)
    b_tokens = _app_tokens(b)
    if not a_tokens or not b_tokens:
        return False
    for x in a_tokens:
        for y in b_tokens:
            if x == y or x in y or y in x:
                return True
    return False


def _app_workflow_rows(layout, app_map: Dict[object, int]) -> np.ndarray:
    rows = []
    app_names = list(app_map.keys())
    for cluster_key, data in layout.usage_data.app_workflows.items():
        if not isinstance(data, dict):
            continue
        count = float(data.get("count", 0))
        if count <= 0:
            continue
        shortcut_count = float(data.get("shortcut_count", 0))
        switch_count = float(data.get("switch_count", 0))
        span = float(data.get("avg_span_ms", 15000))
        cluster_apps = [p.strip() for p in str(cluster_key).split(" + ") if p.strip()]
        if "apps" in data and isinstance(data["apps"], list):
            cluster_apps.extend(str(x) for x in data["apps"])
        matched = []
        for app_name in app_names:
            if any(_app_matches(str(app_name), cluster_app) for cluster_app in cluster_apps):
                matched.append(app_map[app_name])
        matched = sorted(set(matched))
        if len(matched) < 2:
            continue
        span_weight = max(0.5, min(2.0, 15000.0 / max(span, 1.0)))
        weight = count * max(1.0, math.log1p(shortcut_count + switch_count)) * span_weight
        for i, app_a in enumerate(matched):
            for app_b in matched[i + 1:]:
                rows.append((app_a, app_b, weight))
    return np.asarray(rows, dtype=np.float32).reshape((-1, 3)) if rows else np.empty((0, 3), dtype=np.float32)


def _shortcut_duplicate_support(layout) -> np.ndarray:
    support = np.zeros(layout.n_shortcuts, dtype=np.float32)
    lookup = _sid_lookup(layout)

    counts = []
    for shortcut in layout.shortcuts:
        data = layout.usage_data.shortcuts.get(shortcut.keys, {})
        count = float(data.get("count", 0)) if isinstance(data, dict) else 0.0
        counts.append(count)
    max_count = max(counts) if counts else 0.0
    if max_count > 0.0:
        for shortcut, count in zip(layout.shortcuts, counts):
            if count > 0.0:
                support[shortcut.sid] += min(0.75, count / max_count)

    def add_parts(parts, amount):
        for part in parts:
            sid = lookup.get(part, -1)
            if sid >= 0:
                support[sid] += amount

    for seq_key, data in layout.usage_data.sequences.items():
        if not isinstance(data, dict):
            continue
        count = float(data.get("count", 0))
        confidence = float(data.get("confidence", 1.0))
        if count < 2 or confidence < 0.35:
            continue
        add_parts(seq_key.split(" -> "), min(0.45, 0.08 * count * confidence))

    for wf_key, data in list(layout.usage_data.chains.items()) + list(layout.usage_data.workflows.items()):
        if not isinstance(data, dict):
            continue
        count = float(data.get("count", 0))
        if count < 2:
            continue
        add_parts(wf_key.split(" -> "), min(0.75, 0.12 * count))

    for keys, data in layout.usage_data.mouse_session_shortcuts.items():
        sid = lookup.get(keys, -1)
        if sid >= 0 and isinstance(data, dict):
            support[sid] += min(0.5, 0.1 * float(data.get("count", 0)))

    for shortcut in layout.shortcuts:
        for cluster_key, data in layout.usage_data.app_workflows.items():
            if not isinstance(data, dict) or float(data.get("count", 0)) < 2:
                continue
            cluster_apps = [p.strip() for p in str(cluster_key).split(" + ") if p.strip()]
            if any(_app_matches(shortcut.app, app) for app in cluster_apps):
                support[shortcut.sid] += min(0.35, 0.05 * float(data.get("count", 0)))

    return np.minimum(support, 1.5).astype(np.float32)


def _blind_rows(layout) -> np.ndarray:
    from fitness.factors.workflow_coherence import WorkflowCoherenceFactor
    rows = []
    lookup = _sid_lookup(layout)
    normalizer = WorkflowCoherenceFactor._normalize_keys
    blind_spots = layout.usage_data.blind_spots
    if isinstance(blind_spots, dict):
        for keys, data in blind_spots.items():
            sid = lookup.get(keys)
            if sid is None:
                sid = lookup.get(normalizer(keys), -1)
            score = float(data.get("count", 0)) if isinstance(data, dict) else 0.0
            rows.append((sid if sid is not None else -1, score))
    elif isinstance(blind_spots, list):
        for item in blind_spots:
            keys = item.get("keys", "")
            sid = lookup.get(keys)
            if sid is None:
                sid = lookup.get(normalizer(keys), -1)
            rows.append((sid if sid is not None else -1, float(item.get("blind_spot_score", 0))))
    return np.asarray(rows, dtype=np.float32).reshape((-1, 2)) if rows else np.empty((0, 2), dtype=np.float32)


def _group_matrix(layout) -> np.ndarray:
    from fitness.factors.violation import KEY_GROUPS, shortcut_matches_group
    groups = []
    for group in list(KEY_GROUPS) + list(layout.dynamic_groups):
        if not group.get("protected"):
            continue
        members = np.zeros(layout.n_shortcuts, dtype=np.bool_)
        if "sids" in group:
            for sid in group.get("sids", []):
                if 0 <= int(sid) < layout.n_shortcuts:
                    members[int(sid)] = True
        else:
            for shortcut in layout.shortcuts:
                if shortcut_matches_group(shortcut, group):
                    members[shortcut.sid] = True
        if np.any(members):
            groups.append(members)
    return np.asarray(groups, dtype=np.bool_) if groups else np.zeros((0, layout.n_shortcuts), dtype=np.bool_)


def _layer_access_costs(layout) -> np.ndarray:
    """Legacy/static fallback only. Not authoritative for evolved dynamic access.

    WARNING: never call this from scoring, acceptance, or reporting paths.
    It reads ``layout.layer_access`` which is a legacy static fallback; an
    evolved candidate may have placed access shortcuts anywhere else.  The
    compiled kernel rebuilds layer-access costs from the live genome inside
    ``_single_genome()`` for every candidate — that is the authoritative path.
    This function is not called from any production path and exists only as a
    reference for a static access graph.
    """
    layer_access_cost = np.zeros(32, dtype=np.float32)
    if not layout.layer_access:
        return layer_access_cost

    access_graph = {}
    for access in layout.layer_access:
        if access.target_layer not in access_graph:
            access_graph[access.target_layer] = []
        access_effort = 2.0
        for pos in layout.positions:
            if pos.layer == access.source_layer and abs(pos.x - access.source_x) < 0.5 and abs(pos.y - access.source_y) < 0.5:
                access_effort = pos.effort
                break
        access_graph[access.target_layer].append((access.source_layer, access_effort))

    costs = {0: 0.0}
    for _ in range(10):
        for target, sources in access_graph.items():
            if target in costs:
                continue
            for source_layer, source_effort in sources:
                if source_layer in costs:
                    costs[target] = costs[source_layer] + source_effort
                    break
    for layer in range(32):
        layer_access_cost[layer] = costs.get(layer, 5.0)
    return layer_access_cost


def _layer_right_required(layout) -> np.ndarray:
    """Legacy/static fallback only. Not authoritative for evolved dynamic access.

    Reads ``layout.layer_access`` (legacy static fallback).  Not called from
    any production path.  The compiled kernel computes the equivalent per-genome
    inside ``_single_genome()``.
    """
    layer_right_required = np.zeros(32, dtype=np.bool_)
    if not layout.layer_access:
        return layer_right_required

    access_adj = {}
    for access in layout.layer_access:
        if access.target_layer not in access_adj:
            access_adj[access.target_layer] = []
        access_adj[access.target_layer].append((access.source_layer, access.is_momentary, access.hand))

    def _trace_right(layer, visited):
        if layer == 0:
            return False
        if layer in visited:
            return False
        visited.add(layer)
        if layer not in access_adj:
            return False
        for source_layer, is_momentary, hand in access_adj[layer]:
            if not is_momentary:
                continue
            if hand == "right":
                return True
            if source_layer != 0 and _trace_right(source_layer, visited):
                return True
        return False

    for layer in range(32):
        layer_right_required[layer] = _trace_right(layer, set())
    return layer_right_required


def _access_rows(layout) -> np.ndarray:
    """Legacy/static fallback only. Not authoritative for evolved dynamic access.

    Reads ``layout.layer_access`` (legacy static fallback).  Not called from
    any production path.
    """
    rows = []
    for access in layout.layer_access:
        if access.is_momentary:
            rows.append((access.target_layer, 0 if access.hand == "left" else 1))
    return np.asarray(rows, dtype=np.int32).reshape((-1, 2)) if rows else np.empty((0, 2), dtype=np.int32)


def precompute(layout, weights: dict, violation_weights: dict, missing_important_threshold: float,
               scale_factors: np.ndarray, reference_genome: np.ndarray = None,
               hard_constraints=None, toggle_effort_multiplier: float = 2.5):
    """Assemble all static arrays needed by the compiled kernels."""
    pos_effort = np.asarray([p.effort for p in layout.positions], dtype=np.float32)
    pos_layer = np.asarray([p.layer for p in layout.positions], dtype=np.int32)
    pos_finger = np.asarray([p.finger for p in layout.positions], dtype=np.int32)
    pos_hand = np.asarray([0 if p.hand == "left" else 1 for p in layout.positions], dtype=np.int32)
    pos_is_thumb = np.asarray([p.is_thumb for p in layout.positions], dtype=np.bool_)
    pos_is_frozen = np.asarray([p.is_frozen for p in layout.positions], dtype=np.bool_)
    coords = np.asarray([(p.x, p.y) for p in layout.positions], dtype=np.float32)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2).astype(np.float32)
    pos_x = np.asarray([p.x for p in layout.positions], dtype=np.float32)
    pos_y = np.asarray([p.y for p in layout.positions], dtype=np.float32)

    trackball_factor = None
    try:
        from fitness.factors.trackball_proximity import TrackballProximityFactor
        trackball_factor = TrackballProximityFactor()
        trackball_x, trackball_y = trackball_factor.trackball_x, trackball_factor.trackball_y
    except Exception:
        trackball_x, trackball_y = 7.0, 3.5
    trackball_dist = np.linalg.norm(coords - np.asarray([trackball_x, trackball_y], dtype=np.float32), axis=1).astype(np.float32)

    shortcut_importance = np.asarray([s.importance for s in layout.shortcuts], dtype=np.float32)
    shortcut_app, app_map = _id_map([s.app for s in layout.shortcuts])
    shortcut_category, _ = _id_map([s.category for s in layout.shortcuts])
    shortcut_base, _ = _id_map([s.base_key.upper() if s.base_key else "" for s in layout.shortcuts])
    shortcut_base = shortcut_base.astype(np.int32)
    for i, shortcut in enumerate(layout.shortcuts):
        if not shortcut.base_key:
            shortcut_base[i] = -1

    key_ids, key_mapping = _id_map([s.keys for s in layout.shortcuts])
    shortcut_key_group = key_ids.astype(np.int32)
    for i, shortcut in enumerate(layout.shortcuts):
        if not shortcut.keys:
            shortcut_key_group[i] = -1
    n_key_groups = len(key_mapping)

    shortcut_l0_only = np.asarray([s.is_l0_only for s in layout.shortcuts], dtype=np.bool_)
    shortcut_access_target = np.asarray(
        [s.access_target_layer if s.is_layer_access else -1 for s in layout.shortcuts],
        dtype=np.int32,
    )
    shortcut_access_momentary = np.asarray([s.access_is_momentary for s in layout.shortcuts], dtype=np.bool_)
    shortcut_scroll_mode_access = np.asarray([
        s.is_layer_access and (
            "scroll" in (s.keys or "").lower()
            or "scroll" in (s.base_key or "").lower()
            or "scroll" in (s.action or "").lower()
        )
        for s in layout.shortcuts
    ], dtype=np.bool_)

    if trackball_factor is not None:
        shortcut_trackball = np.asarray([trackball_factor._is_trackball_related(s) for s in layout.shortcuts], dtype=np.bool_)
    else:
        shortcut_trackball = np.zeros(layout.n_shortcuts, dtype=np.bool_)
    shortcut_is_mouse = np.asarray([s.category == "mouse" for s in layout.shortcuts], dtype=np.bool_)
    shortcut_mouse_button = np.zeros(layout.n_shortcuts, dtype=np.int32)
    for s in layout.shortcuts:
        key = (s.keys or "").upper().replace(" ", "")
        if key == "MB1":
            shortcut_mouse_button[s.sid] = 1
        elif key == "MB2":
            shortcut_mouse_button[s.sid] = 2
        elif key == "MB3":
            shortcut_mouse_button[s.sid] = 3
        elif key == "MB4":
            shortcut_mouse_button[s.sid] = 4
        elif key == "MB5":
            shortcut_mouse_button[s.sid] = 5
    shortcut_usage_count = np.zeros(layout.n_shortcuts, dtype=np.float32)
    for shortcut in layout.shortcuts:
        usage_entry = layout.usage_data.shortcuts.get(shortcut.keys, {})
        if isinstance(usage_entry, dict):
            shortcut_usage_count[shortcut.sid] += float(usage_entry.get("count", 0.0))
        elif isinstance(usage_entry, (int, float)):
            shortcut_usage_count[shortcut.sid] += float(usage_entry)
        if shortcut.keys in layout.usage_data.mouse_clicks:
            mouse_entry = layout.usage_data.mouse_clicks.get(shortcut.keys, {})
            if isinstance(mouse_entry, dict):
                shortcut_usage_count[shortcut.sid] += float(mouse_entry.get("count", 0.0))
        if shortcut_scroll_mode_access[shortcut.sid]:
            shortcut_usage_count[shortcut.sid] += float(layout.usage_data.scroll_total or 0)
        raw_entry = layout.usage_data.raw_completion_keys.get(shortcut.base_key, {})
        if isinstance(raw_entry, dict):
            shortcut_usage_count[shortcut.sid] += float(raw_entry.get("count", 0.0))
        elif isinstance(raw_entry, (int, float)):
            shortcut_usage_count[shortcut.sid] += float(raw_entry)
    shortcut_preferred_hand = np.asarray([
        1 if s.preferred_hand == "left" else (2 if s.preferred_hand == "right" else 0)
        for s in layout.shortcuts
    ], dtype=np.int32)

    shortcut_arrow_type = np.zeros(layout.n_shortcuts, dtype=np.int32)
    arrow_base = {"LEFTARROW": 1, "RIGHTARROW": 2, "UPARROW": 3, "DOWNARROW": 4}
    for s in layout.shortcuts:
        if s.base_key and s.base_key.upper() in arrow_base and not s.modifiers:
            shortcut_arrow_type[s.sid] = arrow_base[s.base_key.upper()]

    shortcut_raw_completion = np.zeros(layout.n_shortcuts, dtype=np.int32)
    shortcut_raw_completion_base = np.zeros(layout.n_shortcuts, dtype=np.int32)
    raw_completion_order = {
        "DASH AND UNDERSCORE": 1,
        "EQUALS AND PLUS": 2,
        "GRAVE ACCENT AND TILDE": 3,
        "RIGHT BRACE": 4,
        "BACKSLASH AND PIPE": 5,
    }
    for s in layout.shortcuts:
        key = (s.base_key or "").upper()
        if key in raw_completion_order and not s.is_l0_only:
            shortcut_raw_completion[s.sid] = raw_completion_order[key]
            if len(s.modifiers) == 0:
                shortcut_raw_completion_base[s.sid] = 1

    app_usage_weight = np.ones(len(app_map), dtype=np.float32)
    try:
        from fitness.factors.app_coherence import AppCoherenceFactor
        app_factor = AppCoherenceFactor()
        for app_name, app_id in app_map.items():
            app_usage_weight[app_id] = np.float32(app_factor._usage_weight(layout, str(app_name)))
    except Exception:
        pass

    objective_weights = np.asarray([
        weights.get("effort", DEFAULT_FITNESS_WEIGHTS["effort"]),
        weights.get("adjacency", DEFAULT_FITNESS_WEIGHTS["adjacency"]),
        weights.get("finger_balance", DEFAULT_FITNESS_WEIGHTS["finger_balance"]),
        weights.get("same_finger", DEFAULT_FITNESS_WEIGHTS["same_finger"]),
        weights.get("violations", DEFAULT_FITNESS_WEIGHTS["violations"]),
        weights.get("workflow_coherence", DEFAULT_FITNESS_WEIGHTS["workflow_coherence"]),
        weights.get("app_coherence", DEFAULT_FITNESS_WEIGHTS["app_coherence"]),
        weights.get("trackball_proximity", DEFAULT_FITNESS_WEIGHTS["trackball_proximity"]),
        weights.get("familiarity", DEFAULT_FITNESS_WEIGHTS["familiarity"]),
        weights.get("layer_similarity", weights.get("layer_specialization", DEFAULT_FITNESS_WEIGHTS["layer_similarity"])),
        weights.get("everything_layer", DEFAULT_FITNESS_WEIGHTS["everything_layer"]),
    ], dtype=np.float32)

    vw = violation_weights or {}
    violation_weight_arr = np.asarray([
        vw.get("duplicate", DEFAULT_VIOLATION_WEIGHTS["duplicate"]),
        vw.get("l0_displacement", DEFAULT_VIOLATION_WEIGHTS["l0_displacement"]),
        vw.get("missing_important", DEFAULT_VIOLATION_WEIGHTS["missing_important"]),
        vw.get("cross_layer_duplicate", DEFAULT_VIOLATION_WEIGHTS["cross_layer_duplicate"]),
        vw.get("group_split", DEFAULT_VIOLATION_WEIGHTS["group_split"]),
        vw.get("thumb_occupancy", DEFAULT_VIOLATION_WEIGHTS["thumb_occupancy"]),
        vw.get("arrow_order", DEFAULT_VIOLATION_WEIGHTS["arrow_order"]),
        vw.get("hand_bias", DEFAULT_VIOLATION_WEIGHTS["hand_bias"]),
        vw.get("mouse_layer_access", DEFAULT_VIOLATION_WEIGHTS["mouse_layer_access"]),
        vw.get("arrow_scattered", DEFAULT_VIOLATION_WEIGHTS["arrow_scattered"]),
        vw.get("mouse_scattered", DEFAULT_VIOLATION_WEIGHTS["mouse_scattered"]),
        vw.get("layer7_access", DEFAULT_VIOLATION_WEIGHTS["layer7_access"]),
        vw.get("duplicate_value_gap", DEFAULT_VIOLATION_WEIGHTS["duplicate_value_gap"]),
        vw.get("access_layout", DEFAULT_VIOLATION_WEIGHTS["access_layout"]),
        vw.get("raw_keyboard_completion_norwegian", DEFAULT_VIOLATION_WEIGHTS["raw_keyboard_completion_norwegian"]),
        vw.get("dynamic_mouse_layer", DEFAULT_VIOLATION_WEIGHTS["dynamic_mouse_layer"]),
        vw.get("empty_position", DEFAULT_VIOLATION_WEIGHTS["empty_position"]),
        vw.get("layer_reachability", DEFAULT_VIOLATION_WEIGHTS["layer_reachability"]),
        vw.get("layer_depth_penalty", DEFAULT_VIOLATION_WEIGHTS["layer_depth_penalty"]),
        vw.get("natural_mouse_layer_exists", DEFAULT_VIOLATION_WEIGHTS.get("natural_mouse_layer_exists", 50000.0)),
        vw.get("toggle_back_to_l0", DEFAULT_VIOLATION_WEIGHTS.get("toggle_back_to_l0", 50000.0)),
    ], dtype=np.float32)

    VIOLATION_NAMES = (
        "duplicate", "l0_displacement", "missing_important", "cross_layer_duplicate",
        "group_split", "thumb_occupancy", "arrow_order", "hand_bias",
        "mouse_layer_access", "arrow_scattered", "mouse_scattered", "layer7_access",
        "duplicate_value_gap", "access_layout", "raw_keyboard_completion_norwegian",
        "dynamic_mouse_layer", "empty_position",
        "layer_reachability", "layer_depth_penalty",
        "natural_mouse_layer_exists",
        "toggle_back_to_l0",
    )
    hard_constraints = hard_constraints or []
    hard_constraint_indices = np.asarray(
        [VIOLATION_NAMES.index(name) for name in hard_constraints if name in VIOLATION_NAMES],
        dtype=np.int32,
    )

    if reference_genome is None:
        reference_genome = np.full(layout.n_positions, -1, dtype=np.int32)
    else:
        reference_genome = np.asarray(reference_genome, dtype=np.int32)

    return (
        pos_effort, pos_layer, pos_finger, pos_hand, pos_is_thumb, pos_is_frozen, dist, trackball_dist, pos_x, pos_y,
        shortcut_importance, shortcut_app, shortcut_category, shortcut_base, shortcut_l0_only, shortcut_trackball,
        shortcut_is_mouse, shortcut_mouse_button, shortcut_preferred_hand, shortcut_arrow_type, shortcut_raw_completion,
        shortcut_raw_completion_base,
        shortcut_access_target, shortcut_access_momentary, shortcut_scroll_mode_access, shortcut_usage_count,
        app_usage_weight, _group_matrix(layout), _sequence_rows(layout), _app_workflow_rows(layout, app_map),
        _shortcut_duplicate_support(layout),
        _chain_rows(layout, layout.usage_data.chains, 2, 1.0),
        _chain_rows(layout, layout.usage_data.workflows, 3, 2.0),
        _blind_rows(layout), reference_genome, objective_weights, violation_weight_arr,
        np.asarray(scale_factors, dtype=np.float32),
        np.float32(missing_important_threshold),
        hard_constraint_indices, shortcut_key_group, np.int32(n_key_groups),
        np.float32(toggle_effort_multiplier),
    )


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _single_genome(
        genome, pos_effort, pos_layer, pos_finger, pos_hand, pos_is_thumb, pos_is_frozen, dist, trackball_dist, pos_x, pos_y,
        shortcut_importance, shortcut_app, shortcut_category, shortcut_base, shortcut_l0_only, shortcut_trackball,
        shortcut_is_mouse, shortcut_mouse_button, shortcut_preferred_hand, shortcut_arrow_type, shortcut_raw_completion, shortcut_raw_completion_base,
        shortcut_access_target, shortcut_access_momentary, shortcut_scroll_mode_access, shortcut_usage_count,
        app_usage_weight, group_matrix, sequence_rows, app_workflow_rows, duplicate_support,
        chain_rows, workflow_rows, blind_rows,
        reference_genome, objective_weights, violation_weights, scale_factors,
        threshold, hard_constraint_indices,
        shortcut_key_group, n_key_groups,
        toggle_effort_multiplier,
    ):
        n_pos = genome.shape[0]
        n_short = shortcut_importance.shape[0]
        n_apps = app_usage_weight.shape[0]

        sid_pos = np.full(n_short, -1, dtype=np.int32)
        sid_layer_seen = np.zeros((n_short, 32), dtype=np.bool_)
        assigned = np.zeros(n_short, dtype=np.bool_)
        sid_counts = np.zeros(n_short, dtype=np.int32)
        sid_mutable_counts = np.zeros(n_short, dtype=np.int32)

        effort = 0.0
        trackball = 0.0
        mouse_effective_access = 0.0
        mouse_workflow = 0.0
        hand_bias = 0.0
        mouse_layer_access = 0.0
        finger_load = np.zeros(8, dtype=np.float32)
        layer_finger_load = np.zeros((32, 8), dtype=np.float32)
        layer_finger_count = np.zeros((32, 8), dtype=np.int32)
        layer_base_counts = np.zeros((32, n_short), dtype=np.int32)
        layer_base_exception = np.zeros((32, n_short), dtype=np.float32)
        app_layer_counts = np.zeros((n_apps, 32), dtype=np.int32)
        app_total = np.zeros(n_apps, dtype=np.int32)
        app_layer_importance = np.zeros((n_apps, 32), dtype=np.float32)
        layer_demand = np.zeros(32, dtype=np.float32)
        layer_everything_value = np.zeros(32, dtype=np.float32)
        total_everything_value = 0.0
        layer_access_cost = np.full(32, 1000000.0, dtype=np.float32)
        layer_access_cost[0] = 0.0
        layer_left_required = np.zeros(32, dtype=np.bool_)
        layer_right_required = np.zeros(32, dtype=np.bool_)
        direct_left_thumb_momentary = np.zeros(32, dtype=np.bool_)
        direct_right_thumb_momentary = np.zeros(32, dtype=np.bool_)
        direct_toggle_access = np.zeros(32, dtype=np.bool_)
        layer_has_return_toggle = np.zeros(32, dtype=np.bool_)
        layer_has_mutable = np.zeros(32, dtype=np.bool_)
        direct_l0_thumb_access = np.zeros(32, dtype=np.bool_)
        safe_momentary_access = np.zeros(32, dtype=np.bool_)
        right_thumb_momentary_access = np.zeros(32, dtype=np.bool_)
        reachable_toggle_access = np.zeros(32, dtype=np.bool_)
        reachable_momentary_access = np.zeros(32, dtype=np.bool_)
        momentary_edge = np.zeros((32, 32), dtype=np.bool_)
        edge_cost = np.full((32, 32), 1000000.0, dtype=np.float32)
        edge_hand = np.full((32, 32), -1, dtype=np.int32)
        access_layout = 0.0
        mouse_button_right = np.zeros((32, 6), dtype=np.int32)
        mouse_button_right_thumb = np.zeros((32, 6), dtype=np.int32)
        mouse_button_x = np.full((32, 6), -1.0, dtype=np.float32)
        mouse_button_y = np.full((32, 6), -1.0, dtype=np.float32)
        mouse_button_effort = np.zeros((32, 6), dtype=np.float32)
        mouse_button_importance = np.zeros((32, 6), dtype=np.float32)
        mouse_button_usage = np.zeros((32, 6), dtype=np.float32)
        mouse_non_right_count = np.zeros(32, dtype=np.int32)
        mouse_l7_count = 0
        scroll_right_momentary = np.zeros(32, dtype=np.bool_)
        scroll_right_momentary_thumb = np.zeros(32, dtype=np.bool_)
        scroll_right_momentary_effort = np.zeros(32, dtype=np.float32)
        scroll_right_momentary_usage = np.zeros(32, dtype=np.float32)

        # Pre-scan: which layers have at least one mutable (non-frozen, non-L7) position.
        # Used to determine if the optimizer CAN place a return toggle on a given layer.
        for i in range(n_pos):
            if not pos_is_frozen[i] and 0 <= pos_layer[i] < 32 and pos_layer[i] != 7:
                layer_has_mutable[pos_layer[i]] = True

        # WARNING: access graph is rebuilt from assigned access shortcuts.
        # Do not replace this with layout.layer_access or any fixed canonical
        # access map; access buttons are first-class genome capabilities.
        for i in range(n_pos):
            sid = genome[i]
            if sid < 0 or sid >= n_short:
                continue
            target = shortcut_access_target[sid]
            if target < 0 or target >= 32:
                continue
            source = pos_layer[i]
            if source < 0 or source >= 32 or source == target:
                continue
            if shortcut_access_momentary[sid]:
                if pos_is_thumb[i] and pos_hand[i] == 1:
                    right_thumb_momentary_access[target] = True
                    direct_right_thumb_momentary[target] = True
                else:
                    safe_momentary_access[target] = True
                    if pos_is_thumb[i] and pos_hand[i] == 0:
                        direct_left_thumb_momentary[target] = True
            else:
                direct_toggle_access[target] = True
                # Track return-to-L0 toggles: records which source layers have one
                if target == 0:
                    layer_has_return_toggle[source] = True
            cost = pos_effort[i]
            if pos_is_thumb[i]:
                cost *= 0.45
            else:
                cost += 4.0
                access_layout += 2.0 + shortcut_importance[sid] * 0.2
            if source != 0:
                cost += 4.0
                access_layout += 3.0
                if shortcut_access_momentary[sid]:
                    cost += 8.0
                    access_layout += 8.0
            if shortcut_access_momentary[sid] and not pos_is_thumb[i]:
                access_layout += 4.0
            if cost < edge_cost[source, target]:
                edge_cost[source, target] = cost
                momentary_edge[source, target] = shortcut_access_momentary[sid]
                edge_hand[source, target] = pos_hand[i]

        for _ in range(32):
            changed = False
            for source in range(32):
                source_cost = layer_access_cost[source]
                if source_cost >= 999999.0:
                    continue
                for target in range(32):
                    ec = edge_cost[source, target]
                    if ec >= 999999.0:
                        continue
                    nested = 0.0
                    if source != 0:
                        nested += 8.0
                        if momentary_edge[source, target]:
                            nested += 12.0
                    cand = source_cost + ec + nested
                    if cand < layer_access_cost[target]:
                        layer_access_cost[target] = cand
                        layer_left_required[target] = layer_left_required[source] or (
                            momentary_edge[source, target] and edge_hand[source, target] == 0
                        )
                        layer_right_required[target] = layer_right_required[source] or (
                            momentary_edge[source, target] and edge_hand[source, target] == 1
                        )
                        changed = True
            if not changed:
                break

        # Integer hop-count BFS from L0 (separate from effort-weighted layer_access_cost).
        # Used for layer_depth_penalty: each extra hop beyond 1 incurs usage-weighted cost.
        layer_hop_depth = np.full(32, 999, dtype=np.int32)
        layer_hop_depth[0] = 0
        for _ in range(32):
            changed_hop = False
            for src in range(32):
                if layer_hop_depth[src] >= 999:
                    continue
                for tgt in range(32):
                    if edge_cost[src, tgt] < 999999.0:
                        cand = layer_hop_depth[src] + 1
                        if cand < layer_hop_depth[tgt]:
                            layer_hop_depth[tgt] = cand
                            changed_hop = True
            if not changed_hop:
                break

        for target in range(32):
            if edge_cost[0, target] < 999999.0:
                best_thumb = False
                for i in range(n_pos):
                    sid = genome[i]
                    if sid >= 0 and sid < n_short and shortcut_access_target[sid] == target:
                        if pos_layer[i] == 0 and pos_is_thumb[i]:
                            best_thumb = True
                direct_l0_thumb_access[target] = best_thumb

        for i in range(n_pos):
            sid = genome[i]
            if sid < 0 or sid >= n_short:
                continue
            target = shortcut_access_target[sid]
            if target < 0 or target >= 32:
                continue
            source = pos_layer[i]
            if source < 0 or source >= 32:
                continue
            if not shortcut_access_momentary[sid] and layer_access_cost[source] < 999999.0:
                reachable_toggle_access[target] = True
            if shortcut_access_momentary[sid] and layer_access_cost[source] < 999999.0:
                reachable_momentary_access[target] = True

        group_count = np.zeros(n_key_groups, dtype=np.float32)
        group_sum_x = np.zeros(n_key_groups, dtype=np.float32)
        group_sum_y = np.zeros(n_key_groups, dtype=np.float32)

        for i in range(n_pos):
            sid = genome[i]
            if sid < 0 or sid >= n_short:
                continue

            assigned[sid] = True
            sid_counts[sid] += 1
            if not pos_is_frozen[i] and pos_layer[i] != 7:
                sid_mutable_counts[sid] += 1
            sid_pos[sid] = i
            layer = pos_layer[i]
            finger = pos_finger[i]
            if 0 <= layer < 32 and not pos_is_frozen[i] and layer != 7:
                sid_layer_seen[sid, layer] = True

            imp = shortcut_importance[sid]
            access_cost = layer_access_cost[layer] if 0 <= layer < 32 else 0.0
            if access_cost >= 999999.0:
                access_cost = 40.0
                access_layout += imp
            # Toggle layer-access keys: exponential opportunity-cost for wasting
            # low-effort positions. effort=0 (home row) → very high penalty;
            # effort=2.75 (top corner) → near-zero. exp(-2*effort) gives ~250x ratio.
            pos_eff = pos_effort[i]
            if shortcut_access_target[sid] >= 0 and not shortcut_access_momentary[sid]:
                transparent_waste = 200.0 * math.exp(-2.0 * pos_effort[i])
                pos_eff = pos_effort[i] + transparent_waste
            effort += imp * (pos_eff + access_cost)
            if 0 <= layer < 32 and shortcut_access_target[sid] < 0:
                layer_demand[layer] += imp
                if layer != 0 and layer != 7 and not pos_is_frozen[i]:
                    usage_value = math.log1p(shortcut_usage_count[sid])
                    general_value = imp * (1.0 + usage_value * 0.75)
                    if shortcut_is_mouse[sid]:
                        general_value *= 0.65
                    layer_everything_value[layer] += general_value
                    total_everything_value += general_value
            if 0 <= finger < 8:
                finger_load[finger] += imp
            if 0 <= layer < 32 and 0 <= finger < 8:
                layer_finger_load[layer, finger] += imp
                layer_finger_count[layer, finger] += 1
            base = shortcut_base[sid]
            if 0 <= layer < 32 and base >= 0 and not shortcut_l0_only[sid]:
                layer_base_counts[layer, base] += 1
                # Cross-layer repeats should be rare unless the shortcut is
                # exceptional: high importance, strong workflow/logger support,
                # or both.  This cheap sigmoid-like score lets those few
                # shortcuts survive while ordinary repeats pay similarity
                # pressure and free slots for workflow-specific actions.
                exception_raw = shortcut_importance[sid] + duplicate_support[sid] * 10.0 - 16.0
                if shortcut_is_mouse[sid]:
                    # Mouse buttons can exist outside the generated mouse
                    # workflow, but copies are intentionally harder to justify
                    # than ordinary workflow shortcuts.
                    exception_raw -= 4.0
                exception_score = 1.0 / (1.0 + math.exp(-exception_raw * 0.45))
                if exception_score > layer_base_exception[layer, base]:
                    layer_base_exception[layer, base] = exception_score
            app = shortcut_app[sid]
            if 0 <= app < n_apps and 0 <= layer < 32:
                app_layer_counts[app, layer] += 1
                app_total[app] += 1
                app_layer_importance[app, layer] += imp
            if shortcut_trackball[sid]:
                proximity = 1.0 - trackball_dist[i] * 0.3
                if proximity > 0.0:
                    trackball += imp * proximity
            if shortcut_scroll_mode_access[sid]:
                if 0 <= layer < 32 and layer != 0 and layer != 7:
                    if shortcut_access_momentary[sid] and pos_hand[i] == 1:
                        if pos_is_thumb[i]:
                            scroll_right_momentary_thumb[layer] = True
                        else:
                            scroll_right_momentary[layer] = True
                            scroll_right_momentary_effort[layer] = pos_effort[i]
                            scroll_right_momentary_usage[layer] = shortcut_usage_count[sid]
                proximity = 1.0 - trackball_dist[i] * 0.25
                if proximity > 0.0:
                    trackball += imp * proximity * (1.0 + math.log1p(shortcut_usage_count[sid]) * 0.25)
                if pos_is_thumb[i]:
                    trackball += imp * 2.0
                usage_scale = 1.0 + math.log1p(shortcut_usage_count[sid]) * 0.25
                effective = pos_effort[i] + access_cost
                if shortcut_access_momentary[sid]:
                    effective += 0.7
                if 0 <= layer < 32 and layer_right_required[layer]:
                    effective += 2.0
                mouse_effective_access += imp * usage_scale * effective

            kg = shortcut_key_group[sid]
            if kg >= 0 and kg < n_key_groups:
                group_count[kg] += 1.0
                group_sum_x[kg] += pos_x[i]
                group_sum_y[kg] += pos_y[i]

            if shortcut_is_mouse[sid]:
                button = shortcut_mouse_button[sid]
                if button > 0:
                    if layer == 7:
                        mouse_l7_count += 1
                    elif layer != 0 and not pos_is_frozen[i]:
                        if pos_hand[i] == 1:
                            mouse_button_right[layer, button] = 1
                            if pos_is_thumb[i]:
                                mouse_button_right_thumb[layer, button] = 1
                            mouse_button_x[layer, button] = pos_x[i]
                            mouse_button_y[layer, button] = pos_y[i]
                            mouse_button_effort[layer, button] = pos_effort[i]
                            mouse_button_importance[layer, button] = imp
                            mouse_button_usage[layer, button] = shortcut_usage_count[sid]
                        else:
                            mouse_non_right_count[layer] += 1
                if pos_hand[i] == 0:
                    hand_bias += imp * 5.0
                if 0 <= layer < 32 and layer_right_required[layer]:
                    mouse_layer_access += imp * 100.0
                usage_scale = 1.0 + math.log1p(shortcut_usage_count[sid]) * 0.25
                effective = pos_effort[i] + access_cost
                if 0 <= layer < 32 and layer_right_required[layer]:
                    effective += 4.0
                if 0 <= layer < 32 and layer_left_required[layer]:
                    effective += 0.5
                mouse_effective_access += imp * usage_scale * effective
                proximity = 1.0 - trackball_dist[i] * 0.25
                if proximity > 0.0:
                    trackball += imp * usage_scale * proximity
            elif shortcut_preferred_hand[sid] == 2:
                if pos_hand[i] == 0:
                    hand_bias += imp * 2.0
            elif shortcut_preferred_hand[sid] == 1:
                if pos_hand[i] == 1:
                    hand_bias += imp * 2.0

        # Finger balance
        load_count = 0
        load_sum = 0.0
        for f in range(8):
            if finger_load[f] > 0.0:
                load_count += 1
                load_sum += finger_load[f]
        finger_balance = 0.0
        if load_count > 0:
            mean = load_sum / load_count
            if mean >= 1e-6:
                var = 0.0
                for f in range(8):
                    if finger_load[f] > 0.0:
                        d = finger_load[f] - mean
                        var += d * d
                finger_balance = math.sqrt(var / load_count) / mean

        # Same-finger penalty
        same_finger = 0.0
        for layer in range(32):
            for finger in range(8):
                if layer_finger_count[layer, finger] >= 2:
                    s = layer_finger_load[layer, finger]
                    sq = 0.0
                    for i in range(n_pos):
                        sid = genome[i]
                        if sid >= 0 and sid < n_short and pos_layer[i] == layer and pos_finger[i] == finger:
                            imp = shortcut_importance[sid]
                            sq += imp * imp
                    same_finger += ((s * s - sq) * 0.5) * 0.5

        # Familiarity: reward repeated shortcuts that keep the same physical
        # coordinate or nearby region across layers. This uses a pairwise
        # Euclidean exponential attraction: as placements get closer, the
        # reward rises sharply toward its maximum at the exact same coordinate.
        # Far-apart repeats receive little attraction and should not pull a
        # layer away from its workflow. The reward remains gated by
        # exceptionality, so ordinary repeats get little protection.
        familiarity = 0.0
        for i in range(n_pos):
            sid_i = genome[i]
            if sid_i < 0 or sid_i >= n_short:
                continue
            for j in range(i + 1, n_pos):
                sid_j = genome[j]
                if sid_j != sid_i:
                    continue
                if pos_layer[i] == pos_layer[j]:
                    continue
                dx = pos_x[i] - pos_x[j]
                dy = pos_y[i] - pos_y[j]
                dist_sq = dx * dx + dy * dy
                distance_reward = math.exp(-dist_sq * 0.9)
                exception_raw = shortcut_importance[sid_i] + duplicate_support[sid_i] * 10.0 - 16.0
                if shortcut_is_mouse[sid_i]:
                    exception_raw -= 4.0
                exception_score = 1.0 / (1.0 + math.exp(-exception_raw * 0.45))
                familiarity += shortcut_importance[sid_i] * exception_score * exception_score * distance_reward

        # Adjacency
        adjacency = 0.0
        for sid_a in range(n_short):
            pos_a = sid_pos[sid_a]
            if pos_a < 0:
                continue
            for sid_b in range(sid_a + 1, n_short):
                pos_b = sid_pos[sid_b]
                if pos_b < 0:
                    continue
                if shortcut_app[sid_a] != shortcut_app[sid_b] and shortcut_category[sid_a] != shortcut_category[sid_b]:
                    continue
                proximity = 1.0 - dist[pos_a, pos_b] * 0.2
                if proximity > 0.0:
                    adjacency += shortcut_importance[sid_a] * shortcut_importance[sid_b] * proximity

        for r in range(sequence_rows.shape[0]):
            sid_a = int(sequence_rows[r, 0])
            sid_b = int(sequence_rows[r, 1])
            pos_a = sid_pos[sid_a]
            pos_b = sid_pos[sid_b]
            if pos_a < 0 or pos_b < 0:
                continue
            if pos_layer[pos_a] == pos_layer[pos_b]:
                proximity = 1.0 - dist[pos_a, pos_b] * 0.2
            else:
                proximity = 0.3 - dist[pos_a, pos_b] * 0.2
            if proximity > 0.0:
                adjacency += sequence_rows[r, 2] * proximity

        for r in range(chain_rows.shape[0]):
            sid_a = int(chain_rows[r, 0])
            sid_b = int(chain_rows[r, 1])
            pos_a = sid_pos[sid_a]
            pos_b = sid_pos[sid_b]
            if pos_a < 0 or pos_b < 0:
                continue
            if pos_layer[pos_a] == pos_layer[pos_b]:
                proximity = 1.0 - dist[pos_a, pos_b] * 0.2
            else:
                proximity = 0.3 - dist[pos_a, pos_b] * 0.2
            if proximity > 0.0:
                adjacency += chain_rows[r, 2] * 5.0 * 2.0 * proximity

        for sid_a in range(n_short):
            pos_a = sid_pos[sid_a]
            if pos_a < 0:
                continue
            is_mouse_a = shortcut_is_mouse[sid_a] or shortcut_scroll_mode_access[sid_a]
            if not is_mouse_a:
                continue
            for sid_b in range(sid_a + 1, n_short):
                pos_b = sid_pos[sid_b]
                if pos_b < 0:
                    continue
                is_mouse_b = shortcut_is_mouse[sid_b] or shortcut_scroll_mode_access[sid_b]
                if not is_mouse_b:
                    continue
                transition = dist[pos_a, pos_b] * 0.35
                if pos_layer[pos_a] != pos_layer[pos_b]:
                    transition += abs(layer_access_cost[pos_layer[pos_a]] - layer_access_cost[pos_layer[pos_b]]) * 0.5
                    transition += 1.5
                if layer_right_required[pos_layer[pos_a]] or layer_right_required[pos_layer[pos_b]]:
                    transition += 2.0
                pair_weight = math.sqrt(shortcut_importance[sid_a] * shortcut_importance[sid_b])
                usage_pair = 1.0 + math.log1p(shortcut_usage_count[sid_a] + shortcut_usage_count[sid_b]) * 0.2
                mouse_workflow += pair_weight * usage_pair * transition

        for r in range(app_workflow_rows.shape[0]):
            app_a = int(app_workflow_rows[r, 0])
            app_b = int(app_workflow_rows[r, 1])
            if app_a < 0 or app_b < 0 or app_a >= n_apps or app_b >= n_apps:
                continue
            weight = app_workflow_rows[r, 2]
            for layer in range(32):
                if app_layer_counts[app_a, layer] > 0 and app_layer_counts[app_b, layer] > 0:
                    shared = app_layer_importance[app_a, layer]
                    if app_layer_importance[app_b, layer] < shared:
                        shared = app_layer_importance[app_b, layer]
                    adjacency += weight * math.log1p(shared) * 2.0

        # Violations
        duplicate = 0.0
        layer_base_support = np.zeros((32, n_short), dtype=np.float32)
        layer_base_position_value = np.zeros((32, n_short), dtype=np.float32)
        for i in range(n_pos):
            sid = genome[i]
            if sid < 0 or sid >= n_short:
                continue
            if pos_is_frozen[i] or pos_layer[i] == 7:
                continue
            layer = pos_layer[i]
            base = shortcut_base[sid]
            if 0 <= layer < 32 and base >= 0:
                layer_base_support[layer, base] += duplicate_support[sid]
                slot_value = 2.0 - pos_effort[i]
                if slot_value < 0.25:
                    slot_value = 0.25
                layer_base_position_value[layer, base] += slot_value * shortcut_importance[sid]
        for layer in range(32):
            for base in range(n_short):
                c = layer_base_counts[layer, base]
                if c > 1:
                    unsupported = (c - 1) - layer_base_support[layer, base]
                    if unsupported > 0.0:
                        avg_slot_value = layer_base_position_value[layer, base] / c
                        max_support = layer_base_support[layer, base]
                        uncertainty_factor = 1.0
                        if max_support <= 0.0:
                            uncertainty_factor = 0.25 + 0.10 * max(0.0, float(c - 3))
                            if uncertainty_factor > 0.75:
                                uncertainty_factor = 0.75
                        exception_raw = avg_slot_value + max_support * 10.0 - 16.0
                        exception_score = 1.0 / (1.0 + math.exp(-exception_raw * 0.45))
                        novelty_cost = 0.15 + (1.0 - exception_score) * (1.0 - exception_score)
                        duplicate += unsupported * unsupported * uncertainty_factor * novelty_cost * (1.0 + avg_slot_value * 0.1)

        l0_displacement = 0.0
        for i in range(n_pos):
            sid = genome[i]
            if sid >= 0 and sid < n_short and shortcut_l0_only[sid] and pos_layer[i] != 0:
                l0_displacement += 50.0 + shortcut_importance[sid] * 2.0

        missing = 0.0
        best_missing_importance = 0.0
        for sid in range(n_short):
            if not assigned[sid] and shortcut_importance[sid] >= threshold:
                missing += shortcut_importance[sid]
                if shortcut_importance[sid] > best_missing_importance:
                    best_missing_importance = shortcut_importance[sid]

        duplicate_value_gap = 0.0
        if best_missing_importance > 0.0:
            for sid in range(n_short):
                count = sid_mutable_counts[sid]
                if count <= 1:
                    continue
                if shortcut_l0_only[sid]:
                    continue
                # A duplicate should only displace a unique shortcut when it is
                # clearly more useful. The 1.5x margin keeps close calls unique,
                # and the sigmoid-like novelty gate makes ordinary duplicates
                # expensive while exceptional supported duplicates can survive.
                gap = best_missing_importance * 1.5 - shortcut_importance[sid]
                if gap > 0.0:
                    unsupported_extra = (count - 1) - duplicate_support[sid]
                    if unsupported_extra > 0.0:
                        uncertainty_factor = 1.0
                        if duplicate_support[sid] <= 0.0:
                            uncertainty_factor = 0.25 + 0.10 * max(0.0, float(count - 3))
                            if uncertainty_factor > 0.75:
                                uncertainty_factor = 0.75
                        exception_raw = shortcut_importance[sid] + duplicate_support[sid] * 10.0 - 16.0
                        if shortcut_is_mouse[sid]:
                            exception_raw -= 4.0
                        exception_score = 1.0 / (1.0 + math.exp(-exception_raw * 0.45))
                        novelty_cost = 0.15 + (1.0 - exception_score) * (1.0 - exception_score)
                        duplicate_value_gap += unsupported_extra * gap * uncertainty_factor * novelty_cost

        cross_dup = 0.0
        for sid in range(n_short):
            if shortcut_l0_only[sid]:
                continue
            layers = 0
            for layer in range(32):
                if sid_layer_seen[sid, layer]:
                    layers += 1
            if layers >= 2:
                extra = (layers - 1) - duplicate_support[sid]
                if extra > 0.0:
                    uncertainty_factor = 1.0
                    if duplicate_support[sid] <= 0.0:
                        uncertainty_factor = 0.35
                    exception_raw = shortcut_importance[sid] + duplicate_support[sid] * 10.0 - 16.0
                    if shortcut_is_mouse[sid]:
                        exception_raw -= 4.0
                    exception_score = 1.0 / (1.0 + math.exp(-exception_raw * 0.45))
                    novelty_cost = 0.15 + (1.0 - exception_score) * (1.0 - exception_score)
                    cross_dup += extra * extra * uncertainty_factor * novelty_cost

        group_split = 0.0
        for g in range(group_matrix.shape[0]):
            # Group scoring is same-layer compactness only. It must not pull a
            # group across layers: if A/B/C are on different workflow layers,
            # group logic is silent. If A/B/C already coexist on one layer,
            # they should be close enough to read as a local cluster.
            for layer in range(32):
                count = 0
                sum_x = 0.0
                sum_y = 0.0
                for i in range(n_pos):
                    sid = genome[i]
                    if sid < 0 or sid >= n_short or not group_matrix[g, sid]:
                        continue
                    if pos_layer[i] != layer:
                        continue
                    count += 1
                    sum_x += pos_x[i]
                    sum_y += pos_y[i]
                if count < 2:
                    continue
                mean_x = sum_x / float(count)
                mean_y = sum_y / float(count)
                for i in range(n_pos):
                    sid = genome[i]
                    if sid < 0 or sid >= n_short or not group_matrix[g, sid]:
                        continue
                    if pos_layer[i] != layer:
                        continue
                    dx = pos_x[i] - mean_x
                    dy = pos_y[i] - mean_y
                    spread = math.sqrt(dx * dx + dy * dy)
                    if spread > 1.5:
                        group_split += (spread - 1.5) * 20.0

        thumb_occ = 0.0
        for target_layer in range(32):
            if direct_toggle_access[target_layer]:
                continue
            if direct_left_thumb_momentary[target_layer] and direct_right_thumb_momentary[target_layer]:
                continue
            restrict_left = direct_left_thumb_momentary[target_layer]
            restrict_right = direct_right_thumb_momentary[target_layer]
            if not restrict_left and not restrict_right:
                continue
            for i in range(n_pos):
                sid = genome[i]
                if sid < 0 or sid >= n_short:
                    continue
                if pos_layer[i] != target_layer or not pos_is_thumb[i]:
                    continue
                if (pos_hand[i] == 0 and restrict_left) or (
                    pos_hand[i] == 1 and restrict_right
                ):
                    thumb_occ += 1.0 + shortcut_importance[sid] * 0.5

        for layer in range(1, 32):
            demand = layer_demand[layer]
            if demand <= 0.0:
                continue
            if layer_access_cost[layer] >= 999999.0:
                access_layout += demand * 5.0
            elif demand >= 30.0 and not direct_l0_thumb_access[layer]:
                access_layout += math.log1p(demand) * 12.0
            if layer_access_cost[layer] > 3.0:
                access_layout += math.log1p(demand) * (layer_access_cost[layer] - 3.0)

        layer7_access = 0.0
        if not reachable_momentary_access[7]:
            layer7_access += 25000.0
        if not reachable_toggle_access[7]:
            layer7_access += 25000.0

        arrow_order = 0.0
        for layer in range(32):
            if layer == 7:
                continue
            left_x = -1.0
            left_y = -1.0
            right_x = -1.0
            right_y = -1.0
            up_x = -1.0
            up_y = -1.0
            down_x = -1.0
            down_y = -1.0
            for i in range(n_pos):
                sid = genome[i]
                if sid < 0 or sid >= n_short:
                    continue
                atype = shortcut_arrow_type[sid]
                if atype == 0:
                    continue
                if pos_layer[i] != layer:
                    continue
                if atype == 1:
                    left_x = pos_x[i]
                    left_y = pos_y[i]
                elif atype == 2:
                    right_x = pos_x[i]
                    right_y = pos_y[i]
                elif atype == 3:
                    up_x = pos_x[i]
                    up_y = pos_y[i]
                elif atype == 4:
                    down_x = pos_x[i]
                    down_y = pos_y[i]
            if left_x >= 0.0 and right_x >= 0.0:
                if left_x >= right_x:
                    arrow_order += (left_x - right_x + 1.0) * 100.0
                min_x = min(left_x, right_x)
                max_x = max(left_x, right_x)
                if up_x >= 0.0:
                    if up_x < min_x:
                        arrow_order += (min_x - up_x + 1.0) * 60.0
                    elif up_x > max_x:
                        arrow_order += (up_x - max_x + 1.0) * 60.0
                if down_x >= 0.0:
                    if down_x < min_x:
                        arrow_order += (min_x - down_x + 1.0) * 60.0
                    elif down_x > max_x:
                        arrow_order += (down_x - max_x + 1.0) * 60.0
            if up_y >= 0.0 and down_y >= 0.0 and up_y >= down_y:
                arrow_order += (up_y - down_y + 1.0) * 100.0
            if left_x >= 0.0 and right_x >= 0.0 and up_x >= 0.0 and down_x >= 0.0:
                same_line = (
                    abs(left_y - up_y) <= 0.25
                    and abs(up_y - down_y) <= 0.25
                    and abs(down_y - right_y) <= 0.25
                    and left_x < up_x
                    and up_x < down_x
                    and down_x < right_x
                    and (right_x - left_x) <= 4.5
                )
                split_cluster = (
                    abs(left_y - down_y) <= 0.25
                    and abs(down_y - right_y) <= 0.25
                    and left_x < down_x
                    and down_x < right_x
                    and up_y < down_y
                    and abs(up_x - down_x) <= 0.25
                    and (down_y - up_y) <= 2.0
                    and (right_x - left_x) <= 3.5
                )
                if not same_line and not split_cluster:
                    arrow_order += 500.0

        arrow_scattered = 0.0
        arrow_layers = np.zeros(32, dtype=np.int32)
        arrow_layer_type_counts = np.zeros((32, 5), dtype=np.int32)
        arrow_layer_type_x = np.full((32, 5), -1.0, dtype=np.float32)
        arrow_layer_type_y = np.full((32, 5), -1.0, dtype=np.float32)
        non_l7_arrow_placements = 0
        for i in range(n_pos):
            sid = genome[i]
            if sid < 0 or sid >= n_short:
                continue
            atype = shortcut_arrow_type[sid]
            if atype != 0:
                layer = pos_layer[i]
                if 0 <= layer < 32:
                    if layer != 7:
                        non_l7_arrow_placements += 1
                        arrow_layers[layer] = 1
                        arrow_layer_type_counts[layer, atype] += 1
                        if arrow_layer_type_x[layer, atype] < 0.0:
                            arrow_layer_type_x[layer, atype] = pos_x[i]
                            arrow_layer_type_y[layer, atype] = pos_y[i]
        n_arrow_layers = 0
        best_arrow_layer = -1
        best_arrow_layer_count = 0
        best_arrow_layer_types = 0
        for layer in range(32):
            n_arrow_layers += arrow_layers[layer]
            placement_count = 0
            type_count = 0
            for atype in range(1, 5):
                if arrow_layer_type_counts[layer, atype] > 0:
                    type_count += 1
                    placement_count += arrow_layer_type_counts[layer, atype]
            if placement_count > best_arrow_layer_count:
                best_arrow_layer = layer
                best_arrow_layer_count = placement_count
                best_arrow_layer_types = type_count
        if n_arrow_layers > 1:
            arrow_scattered += float(n_arrow_layers - 1) * 10000.0
        if non_l7_arrow_placements > 0:
            # Mutable raw arrows are useful only as a complete ordered cluster.
            # L7 frozen arrows are the fallback; partial non-L7 fragments are noise.
            if not (n_arrow_layers == 1 and best_arrow_layer_count == 4 and best_arrow_layer_types == 4):
                arrow_scattered += 50000.0 + float(non_l7_arrow_placements) * 10000.0
                arrow_scattered += float(4 - best_arrow_layer_types) * 15000.0
                arrow_scattered += float(n_arrow_layers) * 15000.0
            else:
                left_x = arrow_layer_type_x[best_arrow_layer, 1]
                right_x = arrow_layer_type_x[best_arrow_layer, 2]
                up_x = arrow_layer_type_x[best_arrow_layer, 3]
                down_x = arrow_layer_type_x[best_arrow_layer, 4]
                left_y = arrow_layer_type_y[best_arrow_layer, 1]
                right_y = arrow_layer_type_y[best_arrow_layer, 2]
                up_y = arrow_layer_type_y[best_arrow_layer, 3]
                down_y = arrow_layer_type_y[best_arrow_layer, 4]
                same_line = (
                    abs(left_y - up_y) <= 0.25
                    and abs(up_y - down_y) <= 0.25
                    and abs(down_y - right_y) <= 0.25
                    and left_x < up_x
                    and up_x < down_x
                    and down_x < right_x
                    and (right_x - left_x) <= 4.5
                )
                split_cluster = (
                    abs(left_y - down_y) <= 0.25
                    and abs(down_y - right_y) <= 0.25
                    and left_x < down_x
                    and down_x < right_x
                    and up_y < down_y
                    and abs(up_x - down_x) <= 0.25
                    and (down_y - up_y) <= 2.0
                    and (right_x - left_x) <= 3.5
                )
                if not same_line and not split_cluster:
                    arrow_scattered += 50000.0
                # Valid mutable raw arrows are still lower value than the
                # frozen L7 fallback unless workflow pressure earns them.
                arrow_scattered += float(non_l7_arrow_placements) * 2000.0
        for layer in range(32):
            type_count = 0
            placement_count = 0
            duplicate_count = 0
            for atype in range(1, 5):
                c = arrow_layer_type_counts[layer, atype]
                if c > 0:
                    type_count += 1
                    placement_count += c
                    if c > 1:
                        duplicate_count += c - 1
            if placement_count == 0:
                continue
            if type_count < 4:
                arrow_scattered += float(4 - type_count) * 5000.0
                arrow_scattered += float(placement_count) * 5000.0
            if duplicate_count > 0:
                arrow_scattered += float(duplicate_count) * 5000.0

        raw_keyboard_completion_norwegian = 0.0
        raw_layer_counts = np.zeros(32, dtype=np.int32)
        raw_layer_order_counts = np.zeros((32, 6), dtype=np.int32)
        raw_base_layer_counts = np.zeros(32, dtype=np.int32)
        raw_base_layer_order_counts = np.zeros((32, 6), dtype=np.int32)
        raw_order_seen_anywhere = np.zeros(6, dtype=np.int32)
        raw_base_seen_anywhere = np.zeros(6, dtype=np.int32)
        raw_modified_seen_anywhere = np.zeros(6, dtype=np.int32)
        raw_base_assigned_count = np.zeros(n_short, dtype=np.int32)
        raw_usage_total = 0.0
        for i in range(n_pos):
            sid = genome[i]
            if sid < 0 or sid >= n_short:
                continue
            order = shortcut_raw_completion[sid]
            if order <= 0:
                continue
            raw_usage_total += shortcut_usage_count[sid]
            raw_order_seen_anywhere[order] = 1
            layer = pos_layer[i]
            if 0 <= layer < 32:
                raw_layer_counts[layer] += 1
                raw_layer_order_counts[layer, order] += 1
                if shortcut_raw_completion_base[sid] > 0:
                    raw_base_seen_anywhere[order] = 1
                    raw_base_layer_counts[layer] += 1
                    raw_base_layer_order_counts[layer, order] += 1
                    raw_base_assigned_count[sid] += 1
                else:
                    raw_modified_seen_anywhere[order] = 1

        best_raw_layer = -1
        best_raw_unique = 0
        best_raw_count = 0
        raw_total = 0
        raw_base_total = 0
        raw_unique_total = 0
        raw_layers_used = 0
        raw_base_layers_used = 0
        for layer in range(32):
            c = raw_layer_counts[layer]
            raw_total += c
            if c > 0 and layer != 7:
                raw_layers_used += 1
            base_c = raw_base_layer_counts[layer]
            raw_base_total += base_c
            if base_c > 0 and layer != 7:
                raw_base_layers_used += 1
            unique = 0
            for order in range(1, 6):
                if raw_base_layer_order_counts[layer, order] > 0:
                    unique += 1
            if layer != 7 and (unique > best_raw_unique or (unique == best_raw_unique and base_c > best_raw_count)):
                best_raw_unique = unique
                best_raw_count = base_c
                best_raw_layer = layer
        for order in range(1, 6):
            raw_unique_total += raw_order_seen_anywhere[order]
        if raw_total > 0:
            if raw_base_layers_used > 2:
                extra_layers = float(raw_base_layers_used - 2)
                raw_keyboard_completion_norwegian += extra_layers * extra_layers * 8000.0
            if raw_layers_used > 2:
                extra_layers_all = float(raw_layers_used - 2)
                raw_keyboard_completion_norwegian += extra_layers_all * extra_layers_all * 2500.0
            if raw_base_total == 0:
                raw_keyboard_completion_norwegian += float(raw_unique_total) * 6000.0
            raw_keyboard_completion_norwegian += float(raw_unique_total - best_raw_unique) * 4000.0
            raw_keyboard_completion_norwegian += float(raw_base_total - best_raw_count) * 1500.0
            raw_keyboard_completion_norwegian += float(raw_total - raw_base_total) * 250.0
            # Raw completion base keys are unique physical keys; duplicates waste
            # slots and fragment the cluster.
            raw_base_duplicates = 0
            for sid in range(n_short):
                c = raw_base_assigned_count[sid]
                if c > 1:
                    raw_base_duplicates += c - 1
            raw_keyboard_completion_norwegian += float(raw_base_duplicates) * 25000.0
            # Strong reward when all five unique family keys concentrate on one layer.
            if best_raw_unique >= 5:
                raw_keyboard_completion_norwegian -= 6000.0
            elif best_raw_unique >= 4:
                raw_keyboard_completion_norwegian -= 1500.0
            if best_raw_layer >= 0:
                min_x = 10000.0
                max_x = -10000.0
                min_y = 10000.0
                max_y = -10000.0
                last_x = -1000.0
                last_y = -1000.0
                found_unique = 0
                for order in range(1, 6):
                    found_x = -1000.0
                    found_y = -1000.0
                    found_count = raw_base_layer_order_counts[best_raw_layer, order]
                    if found_count <= 0:
                        if raw_order_seen_anywhere[order] > 0:
                            raw_keyboard_completion_norwegian += 2500.0
                        continue
                    found_unique += 1
                    if found_count > 1:
                        raw_keyboard_completion_norwegian += float(found_count - 1) * 600.0
                    for i in range(n_pos):
                        sid = genome[i]
                        if (
                            sid >= 0 and sid < n_short
                            and pos_layer[i] == best_raw_layer
                            and shortcut_raw_completion[sid] == order
                            and shortcut_raw_completion_base[sid] > 0
                        ):
                            if found_x < -999.0 or pos_x[i] < found_x:
                                found_x = pos_x[i]
                                found_y = pos_y[i]
                    if found_x < last_x:
                        raw_keyboard_completion_norwegian += (last_x - found_x + 1.0) * 600.0
                    if last_y >= 0.0 and abs(found_y - last_y) > 1.0:
                        raw_keyboard_completion_norwegian += (abs(found_y - last_y) - 1.0) * 900.0
                    last_x = found_x
                    last_y = found_y
                    if found_x < min_x:
                        min_x = found_x
                    if found_x > max_x:
                        max_x = found_x
                    if found_y < min_y:
                        min_y = found_y
                    if found_y > max_y:
                        max_y = found_y
                if found_unique > 0 and max_x > min_x:
                    if max_y > min_y and (max_y - min_y) > 1.0:
                        raw_keyboard_completion_norwegian += ((max_y - min_y) - 1.0) * 3000.0
                    if (max_x - min_x) > 3.0:
                        raw_keyboard_completion_norwegian += ((max_x - min_x) - 3.0) * 800.0
                    # These are backup physical keys missing from L0. Prefer a
                    # normal-keyboard-like block on the far right, but let real
                    # usage decide how accessible the mixed backup/workflow
                    # layer must be.
                    cluster_center_x = (min_x + max_x) * 0.5
                    if cluster_center_x < 8.0:
                        raw_keyboard_completion_norwegian += (8.0 - cluster_center_x) * 1800.0
                    raw_usage_scale = math.log1p(raw_usage_total)
                    anchor_access_cost = layer_access_cost[best_raw_layer]
                    if anchor_access_cost >= 999999.0:
                        anchor_access_cost = 40.0
                    raw_keyboard_completion_norwegian += raw_usage_scale * anchor_access_cost * 500.0
                    # Norwegian extra-key shape check: all 5 present but wrong positions.
                    # Offsets relative to EQUALS AND PLUS (order 2) as anchor.
                    # DASH(-1,0), EQUALS(0,0), GRAVE(-2,0), RBRACE(-2,1), BACKSLASH(-2,3)
                    if best_raw_unique >= 5:
                        shape_anchor_x = -1000.0
                        shape_anchor_y = -1000.0
                        for i in range(n_pos):
                            sid = genome[i]
                            if (sid >= 0 and sid < n_short
                                    and pos_layer[i] == best_raw_layer
                                    and shortcut_raw_completion[sid] == 2
                                    and shortcut_raw_completion_base[sid] > 0):
                                shape_anchor_x = pos_x[i]
                                shape_anchor_y = pos_y[i]
                                break
                        if shape_anchor_x > -999.0:
                            c_dx = np.empty(5, dtype=np.float32)
                            c_dy = np.empty(5, dtype=np.float32)
                            c_dx[0] = -1.0; c_dy[0] = 0.0
                            c_dx[1] =  0.0; c_dy[1] = 0.0
                            c_dx[2] = -2.0; c_dy[2] = 0.0
                            c_dx[3] = -2.0; c_dy[3] = 1.0
                            c_dx[4] = -2.0; c_dy[4] = 3.0
                            n_wrong_shape = 0
                            for shape_order in range(1, 6):
                                exp_x = shape_anchor_x + c_dx[shape_order - 1]
                                exp_y = shape_anchor_y + c_dy[shape_order - 1]
                                shape_found = False
                                for i in range(n_pos):
                                    sid = genome[i]
                                    if (sid >= 0 and sid < n_short
                                            and pos_layer[i] == best_raw_layer
                                            and shortcut_raw_completion[sid] == shape_order
                                            and shortcut_raw_completion_base[sid] > 0):
                                        if (abs(pos_x[i] - exp_x) <= 0.5
                                                and abs(pos_y[i] - exp_y) <= 0.5):
                                            shape_found = True
                                        break
                                if not shape_found:
                                    n_wrong_shape += 1
                            if n_wrong_shape > 0:
                                raw_keyboard_completion_norwegian += float(n_wrong_shape) * 5000.0
            for order in range(1, 6):
                if raw_order_seen_anywhere[order] > 0 and raw_base_seen_anywhere[order] == 0:
                    raw_keyboard_completion_norwegian += 2500.0
                elif (
                    raw_modified_seen_anywhere[order] > 0
                    and best_raw_layer >= 0
                    and raw_base_layer_order_counts[best_raw_layer, order] == 0
                ):
                    raw_keyboard_completion_norwegian += 1200.0

        mouse_scattered = 0.0
        mouse_layers = np.zeros(32, dtype=np.int32)
        for i in range(n_pos):
            sid = genome[i]
            if sid < 0 or sid >= n_short:
                continue
            if shortcut_is_mouse[sid]:
                layer = pos_layer[i]
                if 0 <= layer < 32:
                    mouse_layers[layer] = 1
        n_mouse_layers = 0
        for layer in range(32):
            n_mouse_layers += mouse_layers[layer]
        if n_mouse_layers > 1:
            mouse_scattered = float(n_mouse_layers - 1)

        # Dynamic mouse layer soft pressure. This is intentionally not a hard
        # constraint for intermediate generations; final acceptance still
        # invalidates a target checkpoint that lacks the complete generated
        # mouse layer. The penalty makes the natural search gradient point
        # toward the final shape instead of using post-hoc semantic patching.
        dynamic_mouse_layer = 100000.0
        natural_mouse_layer = -1
        for layer in range(32):
            if layer == 0 or layer == 7:
                continue
            button_count = 0
            right_thumb_count = 0
            for button in range(1, 6):
                if mouse_button_right[layer, button] > 0:
                    button_count += 1
                    if mouse_button_right_thumb[layer, button] > 0:
                        right_thumb_count += 1
            missing_buttons = 5 - button_count
            candidate_penalty = float(missing_buttons) * 15000.0
            candidate_penalty += float(mouse_non_right_count[layer]) * 20000.0
            for button in range(1, 6):
                if mouse_button_right[layer, button] <= 0:
                    continue
                usage_scale = 1.0 + math.log1p(mouse_button_usage[layer, button]) * 0.35
                imp_scale = mouse_button_importance[layer, button] if mouse_button_importance[layer, button] > 0.0 else 1.0
                candidate_penalty += mouse_button_effort[layer, button] * imp_scale * usage_scale * 30.0
                if mouse_button_right_thumb[layer, button] > 0:
                    candidate_penalty += 20000.0
            if mouse_button_right[layer, 1] > 0 and mouse_button_right[layer, 2] > 0:
                dx = mouse_button_x[layer, 2] - mouse_button_x[layer, 1]
                dy = abs(mouse_button_y[layer, 2] - mouse_button_y[layer, 1])
                dist12 = math.sqrt(dx * dx + dy * dy)
                if dx <= 0.0:
                    candidate_penalty += (1.0 - dx) * 1200.0
                candidate_penalty += dist12 * 250.0
                candidate_penalty += dy * 800.0
            if mouse_button_right[layer, 4] > 0 and mouse_button_right[layer, 5] > 0:
                dx = mouse_button_x[layer, 5] - mouse_button_x[layer, 4]
                dy = abs(mouse_button_y[layer, 5] - mouse_button_y[layer, 4])
                dist45 = math.sqrt(dx * dx + dy * dy)
                if dx <= 0.0:
                    candidate_penalty += (1.0 - dx) * 800.0
                candidate_penalty += dist45 * 180.0
                candidate_penalty += dy * 500.0
            if scroll_right_momentary[layer]:
                usage_scale = 1.0 + math.log1p(scroll_right_momentary_usage[layer]) * 0.35
                candidate_penalty += scroll_right_momentary_effort[layer] * usage_scale * 400.0
            else:
                candidate_penalty += 25000.0
            if scroll_right_momentary_thumb[layer]:
                candidate_penalty += 25000.0
            if right_thumb_momentary_access[layer]:
                candidate_penalty += 30000.0
            elif not safe_momentary_access[layer]:
                candidate_penalty += 8000.0
            if not reachable_toggle_access[layer]:
                candidate_penalty += 25000.0
            if button_count == 0 and not scroll_right_momentary[layer]:
                candidate_penalty += 30000.0
            if candidate_penalty < dynamic_mouse_layer:
                dynamic_mouse_layer = candidate_penalty
            if (
                natural_mouse_layer < 0
                and missing_buttons == 0
                and mouse_non_right_count[layer] == 0
                and right_thumb_count == 0
                and scroll_right_momentary[layer]
                and not right_thumb_momentary_access[layer]
                and reachable_toggle_access[layer]
            ):
                natural_mouse_layer = layer
        dynamic_mouse_layer += float(mouse_l7_count) * 500.0

        if natural_mouse_layer >= 0:
            # Once a natural generated mouse layer exists, it should dominate
            # mouse interaction. Mouse buttons elsewhere are still possible,
            # but they become exceptions that need enough usage/access value to
            # justify the duplicate instead of becoming random scatter.
            for i in range(n_pos):
                sid = genome[i]
                if sid < 0 or sid >= n_short:
                    continue
                if not shortcut_is_mouse[sid]:
                    continue
                layer = pos_layer[i]
                if layer == natural_mouse_layer or layer == 0 or layer == 7:
                    continue
                if pos_is_frozen[i]:
                    continue
                usage_relief = math.log1p(shortcut_usage_count[sid]) * 0.08
                if usage_relief > 0.75:
                    usage_relief = 0.75
                mouse_scattered += 0.35 + (1.0 - usage_relief)

        best_everything_layer = -1
        best_everything_value = 0.0
        for layer in range(32):
            if layer == 0 or layer == 7:
                continue
            value = layer_everything_value[layer]
            if value > best_everything_value:
                best_everything_value = value
                best_everything_layer = layer

        # Layer similarity: generated workflow layers should earn distinct
        # jobs and be worth switching to. Penalize layers that duplicate too
        # many non-exceptional shortcut/base assignments. The sigmoid novelty
        # gate lets truly exceptional shared shortcuts stay familiar. The one
        # emergent everything layer may overlap more broadly because being the
        # go-to layer is its job; ordinary workflow layers still need unique
        # purpose. Do not penalize layers just because they share apps.
        layer_similarity = 0.0
        layer_base_total = np.zeros(32, dtype=np.float32)
        for layer in range(32):
            if layer == 0 or layer == 7:
                continue
            for base in range(n_short):
                c = layer_base_counts[layer, base]
                if c > 0:
                    layer_base_total[layer] += float(c)
        for layer_a in range(32):
            if layer_a == 0 or layer_a == 7 or layer_base_total[layer_a] < 4.0:
                continue
            for layer_b in range(layer_a + 1, 32):
                if layer_b == 0 or layer_b == 7 or layer_base_total[layer_b] < 4.0:
                    continue
                weighted_overlap = 0.0
                for base in range(n_short):
                    ca = layer_base_counts[layer_a, base]
                    cb = layer_base_counts[layer_b, base]
                    if ca > 0 and cb > 0:
                        shared = 0.0
                        if ca < cb:
                            shared = float(ca)
                        else:
                            shared = float(cb)
                        exception_score = layer_base_exception[layer_a, base]
                        if layer_base_exception[layer_b, base] < exception_score:
                            exception_score = layer_base_exception[layer_b, base]
                        weighted_overlap += shared * (1.0 - exception_score)
                smaller = layer_base_total[layer_a]
                if layer_base_total[layer_b] < smaller:
                    smaller = layer_base_total[layer_b]
                if smaller <= 0.0:
                    continue
                overlap_ratio = weighted_overlap / smaller
                threshold_ratio = 0.45
                multiplier = 1.0
                if layer_a == best_everything_layer or layer_b == best_everything_layer:
                    threshold_ratio = 0.85
                    multiplier = 0.25
                if overlap_ratio > threshold_ratio:
                    demand = math.sqrt(max(layer_demand[layer_a], 1.0) * max(layer_demand[layer_b], 1.0))
                    excess = overlap_ratio - threshold_ratio
                    layer_similarity += excess * excess * demand * multiplier

        # Emergent "everything" layer: one generated layer should become the
        # user's go-to surface when they are unsure. This is not a fixed layer
        # number or hard role. It rewards the single best non-L0/non-L7 layer
        # for concentrating globally common/high-value shortcuts with cheap
        # access, while other generated layers remain free to specialize.
        everything_layer = 0.0
        if best_everything_layer >= 0 and total_everything_value > 0.0:
            coverage = best_everything_value / total_everything_value
            access_bonus = 1.0
            access_cost = layer_access_cost[best_everything_layer]
            if access_cost < 999999.0:
                access_bonus += 1.5 / (1.0 + access_cost)
            if direct_l0_thumb_access[best_everything_layer]:
                access_bonus += 0.4
            if reachable_toggle_access[best_everything_layer]:
                access_bonus += 0.25
            everything_layer = math.log1p(best_everything_value) * coverage * access_bonus

        # Empty-position penalty.
        # Each mutable, non-L0, non-L7, unassigned slot on a reachable layer
        # receives a sigmoid-weighted penalty that rises sharply for the best
        # (lowest-effort) positions and stays negligible for far/hard positions.
        #
        # Formula per empty slot i on layer L:
        #   pos_value  = 1 / (1 + effort[i])        -- [0,1], higher is better
        #   gate       = sigmoid(8 * (pos_value - 0.5))  -- sharp rise above 0.5
        #   layer_factor = min(3, 2 / (1 + access_cost[L] * 0.05))
        #   demand_scale = 1 + log1p(layer_demand[L]) * 0.1
        #   penalty   += gate * layer_factor * demand_scale
        #
        # Excluded: L0 (base typing), L7 (frozen), frozen positions, unreachable layers.
        # Soft pressure only; never a hard acceptance constraint.
        empty_position = 0.0
        for i in range(n_pos):
            if genome[i] >= 0:
                continue
            if pos_is_frozen[i]:
                continue
            layer = pos_layer[i]
            if layer == 0 or layer == 7:
                continue
            if layer_access_cost[layer] >= 999999.0:
                continue
            ev = pos_effort[i]
            pos_value = 1.0 / (1.0 + ev)
            gate = 1.0 / (1.0 + math.exp(-8.0 * (pos_value - 0.5)))
            lc = layer_access_cost[layer]
            layer_factor = 2.0 / (1.0 + lc * 0.05)
            if layer_factor > 3.0:
                layer_factor = 3.0
            demand_scale = 1.0 + math.log1p(layer_demand[layer]) * 0.1
            empty_position += gate * layer_factor * demand_scale

        # Layer graph: reachability and depth penalty.
        total_demand = 0.0
        for L in range(32):
            total_demand += layer_demand[L]
        layer_reachability = 0.0
        layer_depth_penalty = 0.0
        if total_demand > 0.0:
            for L in range(32):
                if layer_demand[L] <= 0.0:
                    continue
                if layer_access_cost[L] >= 999999.0:
                    layer_reachability += layer_demand[L]
                else:
                    extra = layer_hop_depth[L] - 1
                    if extra > 0:
                        layer_depth_penalty += float(extra) * (layer_demand[L] / total_demand)

        # Raw violation scores (0 = feasible).
        # toggle_back_to_l0: count mutable layers reachable via toggle that lack a return toggle.
        # Frozen-only layers (e.g. L7) are excluded: the optimizer can't place a return there.
        toggle_back_to_l0 = 0.0
        for lx in range(1, 32):
            if direct_toggle_access[lx] and not layer_has_return_toggle[lx] and layer_has_mutable[lx]:
                toggle_back_to_l0 += 1.0

        raw_scores = np.empty(21, dtype=np.float32)
        raw_scores[0] = duplicate
        raw_scores[1] = l0_displacement
        raw_scores[2] = missing
        raw_scores[3] = cross_dup
        raw_scores[4] = group_split
        raw_scores[5] = thumb_occ
        raw_scores[6] = arrow_order
        raw_scores[7] = hand_bias
        raw_scores[8] = mouse_layer_access
        raw_scores[9] = arrow_scattered
        raw_scores[10] = mouse_scattered
        raw_scores[11] = layer7_access
        raw_scores[12] = duplicate_value_gap
        raw_scores[13] = access_layout
        raw_scores[14] = raw_keyboard_completion_norwegian
        raw_scores[15] = dynamic_mouse_layer
        raw_scores[16] = empty_position
        raw_scores[17] = layer_reachability
        raw_scores[18] = layer_depth_penalty
        raw_scores[19] = 0.0 if natural_mouse_layer >= 0 else 1.0
        raw_scores[20] = toggle_back_to_l0

        # Hard constraints (g(x) <= 0 convention; raw_scores are >= 0).
        n_constr = hard_constraint_indices.shape[0]
        constraints = np.empty(n_constr, dtype=np.float32)
        for i in range(n_constr):
            constraints[i] = raw_scores[hard_constraint_indices[i]]

        # Soft penalties weighted and summed into the violations objective.
        violations_raw = 0.0
        for j in range(21):
            violations_raw += raw_scores[j] * violation_weights[j]

        workflow = 0.0
        for r in range(chain_rows.shape[0]):
            sid_a = int(chain_rows[r, 0])
            sid_b = int(chain_rows[r, 1])
            pos_a = sid_pos[sid_a]
            pos_b = sid_pos[sid_b]
            if pos_a >= 0 and pos_b >= 0 and pos_layer[pos_a] != pos_layer[pos_b]:
                workflow += chain_rows[r, 2] * 10.0
        for r in range(workflow_rows.shape[0]):
            sid_a = int(workflow_rows[r, 0])
            sid_b = int(workflow_rows[r, 1])
            pos_a = sid_pos[sid_a]
            pos_b = sid_pos[sid_b]
            if pos_a >= 0 and pos_b >= 0 and pos_layer[pos_a] != pos_layer[pos_b]:
                workflow += workflow_rows[r, 2] * 10.0
        for r in range(app_workflow_rows.shape[0]):
            app_a = int(app_workflow_rows[r, 0])
            app_b = int(app_workflow_rows[r, 1])
            if app_a < 0 or app_b < 0 or app_a >= n_apps or app_b >= n_apps:
                continue
            colocated = False
            for layer in range(32):
                if app_layer_counts[app_a, layer] > 0 and app_layer_counts[app_b, layer] > 0:
                    colocated = True
                    break
            if not colocated:
                workflow += app_workflow_rows[r, 2] * 8.0
        for r in range(blind_rows.shape[0]):
            sid = int(blind_rows[r, 0])
            if sid < 0 or sid >= n_short or sid_pos[sid] < 0:
                workflow += blind_rows[r, 1] * 10.0 * 0.5

        # App coherence is a fallback, not the primary target. Workflow and
        # sequence terms decide what belongs together first. Generic app
        # concentration only gets rewarded when layer_similarity has already
        # detected redundant/similar generated layers; then it can help one
        # layer become the specific workflow surface while another resolves
        # toward an app-specific workflow instead of becoming a duplicate.
        app_coherence = 0.0
        app_coherence_gate = 0.0
        if layer_similarity > 0.0:
            app_coherence_gate = layer_similarity / 25.0
            if app_coherence_gate > 1.0:
                app_coherence_gate = 1.0
        for app in range(n_apps):
            total = app_total[app]
            if total < 2:
                continue
            max_count = 0
            for layer in range(32):
                if app_layer_counts[app, layer] > max_count:
                    max_count = app_layer_counts[app, layer]
            coherence = max_count / total
            app_coherence += app_coherence_gate * coherence * 10.0 * math.log1p(total) * app_usage_weight[app]

        objective_effort = effort * objective_weights[0]
        objective_adj = -adjacency * objective_weights[1]
        objective_viol = (
            finger_balance * objective_weights[2] +
            same_finger * objective_weights[3] +
            violations_raw * objective_weights[4] +
            workflow * objective_weights[5] +
            -app_coherence * objective_weights[6] -
            trackball * objective_weights[7] -
            familiarity * objective_weights[8] +
            layer_similarity * objective_weights[9] +
            -everything_layer * objective_weights[10] +
            mouse_effective_access * 0.08 +
            mouse_workflow * 0.15
        )

        out = np.empty(3, dtype=np.float32)
        out[0] = objective_effort / scale_factors[0]
        out[1] = objective_adj / scale_factors[1]
        out[2] = objective_viol / scale_factors[2]
        return out, constraints

    @njit(parallel=True, cache=True)
    def _evaluate_batch(
        genomes, pos_effort, pos_layer, pos_finger, pos_hand, pos_is_thumb, pos_is_frozen, dist, trackball_dist, pos_x, pos_y,
        shortcut_importance, shortcut_app, shortcut_category, shortcut_base, shortcut_l0_only, shortcut_trackball,
        shortcut_is_mouse, shortcut_mouse_button, shortcut_preferred_hand, shortcut_arrow_type, shortcut_raw_completion, shortcut_raw_completion_base,
        shortcut_access_target, shortcut_access_momentary, shortcut_scroll_mode_access, shortcut_usage_count,
        app_usage_weight, group_matrix, sequence_rows, app_workflow_rows, duplicate_support,
        chain_rows, workflow_rows, blind_rows,
        reference_genome, objective_weights, violation_weights, scale_factors,
        threshold, hard_constraint_indices,
        shortcut_key_group, n_key_groups,
        toggle_effort_multiplier,
    ):
        batch = genomes.shape[0]
        n_constr = hard_constraint_indices.shape[0]
        out = np.empty((batch, 3), dtype=np.float32)
        constraints = np.empty((batch, n_constr), dtype=np.float32)
        for b in prange(batch):
            obj, constr = _single_genome(
                genomes[b],
                pos_effort, pos_layer, pos_finger, pos_hand, pos_is_thumb, pos_is_frozen, dist, trackball_dist, pos_x, pos_y,
                shortcut_importance, shortcut_app, shortcut_category, shortcut_base, shortcut_l0_only, shortcut_trackball,
                shortcut_is_mouse, shortcut_mouse_button, shortcut_preferred_hand, shortcut_arrow_type, shortcut_raw_completion, shortcut_raw_completion_base,
                shortcut_access_target, shortcut_access_momentary, shortcut_scroll_mode_access, shortcut_usage_count,
                app_usage_weight, group_matrix, sequence_rows, app_workflow_rows, duplicate_support,
                chain_rows, workflow_rows, blind_rows,
                reference_genome, objective_weights, violation_weights, scale_factors,
                threshold, hard_constraint_indices,
                shortcut_key_group, n_key_groups,
                toggle_effort_multiplier,
            )
            out[b] = obj
            constraints[b] = constr
        return out, constraints

else:
    def _evaluate_batch(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("numba is not available")

    def _single_genome(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("numba is not available")
