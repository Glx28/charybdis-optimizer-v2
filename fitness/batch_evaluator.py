"""Compiled batch exact evaluator for normalized fitness objectives."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from core import Layout
from fitness.evaluator import FitnessEvaluator
from fitness.factors.app_coherence import AppCoherenceFactor
from fitness.factors.trackball_proximity import TrackballProximityFactor
from fitness.factors.violation import KEY_GROUPS, shortcut_matches_group
from fitness.factors.workflow_coherence import WorkflowCoherenceFactor

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - exercised when numba is unavailable
    njit = None
    prange = range


NUMBA_AVAILABLE = njit is not None


@dataclass
class BatchParityResult:
    ok: bool
    max_abs_diff: float
    message: str


def _id_map(values: Iterable[object]) -> Tuple[np.ndarray, Dict[object, int]]:
    mapping: Dict[object, int] = {}
    ids = []
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
        ids.append(mapping[value])
    return np.asarray(ids, dtype=np.int32), mapping


def _sid_lookup(layout: Layout) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    normalizer = WorkflowCoherenceFactor._normalize_keys
    for shortcut in layout.shortcuts:
        lookup[shortcut.keys] = shortcut.sid
        lookup[normalizer(shortcut.keys)] = shortcut.sid
    return lookup


def _chain_rows(layout: Layout, source: dict, min_count: int, multiplier: float) -> np.ndarray:
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


def _sequence_rows(layout: Layout) -> np.ndarray:
    rows = []
    lookup = _sid_lookup(layout)
    for seq_key, data in layout.usage_data.sequences.items():
        parts = seq_key.split(" -> ")
        if len(parts) != 2 or not isinstance(data, dict):
            continue
        sid_a = lookup.get(parts[0], -1)
        sid_b = lookup.get(parts[1], -1)
        if sid_a < 0 or sid_b < 0:
            continue
        count = float(data.get("count", 0))
        avg_gap = float(data.get("avg_gap_ms", 5000))
        speed_weight = max(0.5, 2.0 - avg_gap / 2000.0)
        rows.append((sid_a, sid_b, count * speed_weight * 5.0))
    return np.asarray(rows, dtype=np.float32).reshape((-1, 3)) if rows else np.empty((0, 3), dtype=np.float32)


def _blind_sids(layout: Layout) -> np.ndarray:
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


def _group_matrix(layout: Layout) -> np.ndarray:
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


def _prepare_arrays(layout: Layout, evaluator: FitnessEvaluator):
    pos_effort = np.asarray([p.effort for p in layout.positions], dtype=np.float32)
    pos_layer = np.asarray([p.layer for p in layout.positions], dtype=np.int32)
    pos_finger = np.asarray([p.finger for p in layout.positions], dtype=np.int32)
    pos_hand = np.asarray([0 if p.hand == "left" else 1 for p in layout.positions], dtype=np.int32)
    pos_is_thumb = np.asarray([p.is_thumb for p in layout.positions], dtype=np.bool_)
    coords = np.asarray([(p.x, p.y) for p in layout.positions], dtype=np.float32)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2).astype(np.float32)
    trackball_dist = np.linalg.norm(coords - np.asarray([7.0, 3.5], dtype=np.float32), axis=1).astype(np.float32)

    shortcut_importance = np.asarray([s.importance for s in layout.shortcuts], dtype=np.float32)
    shortcut_app, app_map = _id_map([s.app for s in layout.shortcuts])
    shortcut_category, _ = _id_map([s.category for s in layout.shortcuts])
    shortcut_base, _ = _id_map([s.base_key.upper() if s.base_key else "" for s in layout.shortcuts])
    shortcut_base = shortcut_base.astype(np.int32)
    for i, shortcut in enumerate(layout.shortcuts):
        if not shortcut.base_key:
            shortcut_base[i] = -1
    shortcut_l0_only = np.asarray([s.is_l0_only for s in layout.shortcuts], dtype=np.bool_)
    trackball_factor = TrackballProximityFactor()
    shortcut_trackball = np.asarray([trackball_factor._is_trackball_related(s) for s in layout.shortcuts], dtype=np.bool_)
    shortcut_is_mouse = np.asarray([s.category == "mouse" for s in layout.shortcuts], dtype=np.bool_)
    shortcut_preferred_hand = np.asarray([
        1 if s.preferred_hand == "left" else (2 if s.preferred_hand == "right" else 0)
        for s in layout.shortcuts
    ], dtype=np.int32)

    pos_x = np.asarray([p.x for p in layout.positions], dtype=np.float32)

    shortcut_arrow_type = np.zeros(layout.n_shortcuts, dtype=np.int32)
    arrow_base = {"LEFTARROW": 1, "RIGHTARROW": 2, "UPARROW": 3, "DOWNARROW": 4}
    for s in layout.shortcuts:
        if s.base_key.upper() in arrow_base:
            shortcut_arrow_type[s.sid] = arrow_base[s.base_key.upper()]

    app_usage_weight = np.ones(len(app_map), dtype=np.float32)
    app_factor = AppCoherenceFactor()
    for app_name, app_id in app_map.items():
        app_usage_weight[app_id] = np.float32(app_factor._usage_weight(layout, str(app_name)))

    # Compute layer access costs from LayerAccess data
    layer_access_cost = np.zeros(32, dtype=np.float32)
    if layout.layer_access:
        access_graph = {}
        for access in layout.layer_access:
            if access.target_layer not in access_graph:
                access_graph[access.target_layer] = []
            # Find source position effort
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
            if layer not in costs:
                costs[layer] = 5.0
            layer_access_cost[layer] = costs[layer]

    # Compute which layers require right-hand momentary access (for mouse shortcut constraint)
    layer_right_required = np.zeros(32, dtype=np.bool_)
    if layout.layer_access:
        # Build adjacency: target -> list of (source, is_momentary, hand)
        access_adj = {}
        for access in layout.layer_access:
            if access.target_layer not in access_adj:
                access_adj[access.target_layer] = []
            access_adj[access.target_layer].append((access.source_layer, access.is_momentary, access.hand))
        
        # Trace back only through MOMENTARY accesses (matching get_occupied_thumbs logic)
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
                    continue  # Skip non-momentary (toggle, lock) — matching get_occupied_thumbs
                if hand == "right":
                    return True
                if source_layer != 0 and _trace_right(source_layer, visited):
                    return True
            return False
        
        for layer in range(32):
            if _trace_right(layer, set()):
                layer_right_required[layer] = True

    access_rows = []
    for access in layout.layer_access:
        if access.is_momentary:
            access_rows.append((access.target_layer, 0 if access.hand == "left" else 1))
    access = np.asarray(access_rows, dtype=np.int32).reshape((-1, 2)) if access_rows else np.empty((0, 2), dtype=np.int32)

    # Compute layer 7 reachability from L0 via BFS through all access types
    layer7_unreachable = 0.0
    if layout.layer_access:
        access_graph = {}
        for la in layout.layer_access:
            if la.source_layer not in access_graph:
                access_graph[la.source_layer] = []
            access_graph[la.source_layer].append(la.target_layer)
        visited = {0}
        queue = [0]
        reachable = False
        while queue:
            current = queue.pop(0)
            if current == 7:
                reachable = True
                break
            for neighbor in access_graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if not reachable:
            layer7_unreachable = 1.0

    weights = evaluator.weights
    objective_weights = np.asarray([
        weights.get("effort", 1.0),
        weights.get("adjacency", 1.5),
        weights.get("finger_balance", 0.8),
        weights.get("same_finger", 2.0),
        weights.get("violations", 50.0),
        weights.get("workflow_coherence", 30.0),
        weights.get("learning_curve", 0.5),
        weights.get("app_coherence", 5.0),
        weights.get("trackball_proximity", 2.0),
    ], dtype=np.float32)

    # Build violation weights from evaluator config or use defaults
    vw = getattr(evaluator, 'violation_weights', None)
    if vw is None:
        vw = {
            "duplicate": 10.0,
            "l0_displacement": 50.0,
            "missing_important": 5000000.0,
            "cross_layer_duplicate": 8.0,
            "group_split": 200.0,
            "thumb_occupancy": 200.0,
            "arrow_order": 500000.0,
            "hand_bias": 2000.0,
            "mouse_layer_access": 5000.0,
            "arrow_scattered": 5000000.0,
            "layer7_unreachable": 50000000.0,
        }
    
    violation_weights = np.asarray([
        vw.get("duplicate", 10.0),
        vw.get("l0_displacement", 50.0),
        vw.get("missing_important", 5000000.0),
        vw.get("cross_layer_duplicate", 8.0),
        vw.get("group_split", 200.0),
        vw.get("thumb_occupancy", 200.0),
        vw.get("arrow_order", 500000.0),
        vw.get("hand_bias", 2000.0),
        vw.get("mouse_layer_access", 5000.0),
        vw.get("arrow_scattered", 5000000.0),
        vw.get("layer7_unreachable", 50000000.0),
    ], dtype=np.float32)

    threshold = getattr(evaluator, 'threshold', 6.0)

    ref = evaluator.reference_layout.genome if evaluator.reference_layout is not None else np.full(layout.n_positions, -1, dtype=np.int32)

    return (
        pos_effort, pos_layer, pos_finger, pos_hand, pos_is_thumb, dist, trackball_dist, pos_x,
        shortcut_importance, shortcut_app, shortcut_category, shortcut_base, shortcut_l0_only, shortcut_trackball,
        shortcut_is_mouse, shortcut_preferred_hand, shortcut_arrow_type,
        app_usage_weight, access, _group_matrix(layout), _sequence_rows(layout),
        _chain_rows(layout, layout.usage_data.chains, 2, 1.0),
        _chain_rows(layout, layout.usage_data.workflows, 3, 2.0),
        _blind_sids(layout), ref.astype(np.int32), objective_weights, violation_weights,
        np.asarray(evaluator.scale_factors, dtype=np.float32),
        layer_access_cost,
        layer_right_required,
        np.float32(layer7_unreachable),
        np.float32(threshold),
    )


class BatchExactEvaluator:
    """Numba-compiled batch evaluator with CPU evaluator fallback/parity checks."""

    def __init__(self, layout: Layout, evaluator: FitnessEvaluator, validate: bool = True):
        if not NUMBA_AVAILABLE:
            raise RuntimeError("numba is not available")
        self.layout = layout
        self.evaluator = evaluator
        self.arrays = _prepare_arrays(layout, evaluator)
        self.enabled = True
        self.parity: Optional[BatchParityResult] = None
        _ = self.evaluate(layout.genome.reshape(1, -1))
        if validate:
            self.parity = self.validate_parity()
            self.enabled = self.parity.ok

    def evaluate(self, genomes: np.ndarray) -> np.ndarray:
        if not self.enabled:
            raise RuntimeError("batch evaluator disabled")
        return _evaluate_batch_numba(np.asarray(genomes, dtype=np.int32), *self.arrays)

    def validate_parity(self, n: int = 32, tolerance: float = 1e-4) -> BatchParityResult:
        tolerance = float(tolerance)
        rng = np.random.default_rng(12345)
        samples = [self.layout.genome.astype(np.int32).copy()]
        mutable = self.layout.mutable_indices
        for _ in range(max(0, n - 1)):
            genome = self.layout.genome.astype(np.int32).copy()
            if len(mutable) > 1:
                a, b = rng.choice(mutable, size=2, replace=False)
                genome[a], genome[b] = genome[b], genome[a]
            samples.append(genome)
        batch = np.asarray(samples, dtype=np.int32)
        compiled = self.evaluate(batch)
        oracle = np.asarray([
            self.evaluator.evaluate(self.layout.clone_with(genome=genome)).objectives
            for genome in batch
        ], dtype=np.float32)
        max_diff = float(np.max(np.abs(compiled - oracle)))
        ok = bool(np.allclose(compiled, oracle, atol=tolerance, rtol=1e-5))
        return BatchParityResult(ok, max_diff, "ok" if ok else f"max diff {max_diff:.6g} exceeds tolerance")


if NUMBA_AVAILABLE:
    @njit(parallel=False, cache=False)
    def _evaluate_batch_numba(
        genomes, pos_effort, pos_layer, pos_finger, pos_hand, pos_is_thumb, dist, trackball_dist, pos_x,
        shortcut_importance, shortcut_app, shortcut_category, shortcut_base, shortcut_l0_only, shortcut_trackball,
        shortcut_is_mouse, shortcut_preferred_hand, shortcut_arrow_type,
        app_usage_weight, access, group_matrix, sequence_rows, chain_rows, workflow_rows, blind_rows,
        reference_genome, objective_weights, violation_weights, scale_factors, layer_access_cost,
        layer_right_required, layer7_unreachable, threshold,
    ):
        batch = genomes.shape[0]
        n_pos = genomes.shape[1]
        n_short = shortcut_importance.shape[0]
        n_apps = app_usage_weight.shape[0]
        out = np.zeros((batch, 3), dtype=np.float32)

        for b in range(batch):
            genome = genomes[b]
            sid_pos = np.full(n_short, -1, dtype=np.int32)
            sid_layer_seen = np.zeros((n_short, 32), dtype=np.bool_)
            assigned = np.zeros(n_short, dtype=np.bool_)

            effort = 0.0
            trackball = 0.0
            hand_bias = 0.0
            mouse_layer_access = 0.0
            learning = 0.0
            finger_load = np.zeros(8, dtype=np.float32)
            layer_finger_load = np.zeros((32, 8), dtype=np.float32)
            layer_finger_count = np.zeros((32, 8), dtype=np.int32)
            layer_base_counts = np.zeros((32, n_short), dtype=np.int32)
            app_layer_counts = np.zeros((n_apps, 32), dtype=np.int32)
            app_total = np.zeros(n_apps, dtype=np.int32)

            for i in range(n_pos):
                sid = genome[i]
                ref_sid = reference_genome[i]
                if sid != ref_sid:
                    if ref_sid >= 0 and sid >= 0:
                        imp = shortcut_importance[ref_sid] if ref_sid < n_short else 5.0
                        learning += 1.0 + imp * 0.5 + imp * imp * 0.01
                    elif ref_sid < 0 and sid >= 0:
                        learning += 0.3
                    elif ref_sid >= 0 and sid < 0:
                        imp = shortcut_importance[ref_sid] if ref_sid < n_short else 5.0
                        learning += 3.0 + imp + imp * imp * 0.02

                if sid < 0 or sid >= n_short:
                    continue

                assigned[sid] = True
                sid_pos[sid] = i
                layer = pos_layer[i]
                finger = pos_finger[i]
                if 0 <= layer < 32:
                    sid_layer_seen[sid, layer] = True

                imp = shortcut_importance[sid]
                access_cost = layer_access_cost[layer] if 0 <= layer < 32 else 0.0
                effort += imp * (pos_effort[i] + access_cost)
                if 0 <= finger < 8:
                    finger_load[finger] += imp
                if 0 <= layer < 32 and 0 <= finger < 8:
                    layer_finger_load[layer, finger] += imp
                    layer_finger_count[layer, finger] += 1
                base = shortcut_base[sid]
                if 0 <= layer < 32 and base >= 0:
                    layer_base_counts[layer, base] += 1
                app = shortcut_app[sid]
                if 0 <= app < n_apps and 0 <= layer < 32:
                    app_layer_counts[app, layer] += 1
                    app_total[app] += 1
                if shortcut_trackball[sid]:
                    proximity = 1.0 - trackball_dist[i] * 0.3
                    if proximity > 0.0:
                        trackball += imp * proximity

                # Hand bias: mouse on left hand = strong penalty; wrong preferred_hand = moderate penalty
                if shortcut_is_mouse[sid]:
                    if pos_hand[i] == 0:  # left hand
                        hand_bias += imp * 5.0
                    # Mouse on a layer requiring right-hand momentary access = huge penalty
                    if 0 <= layer < 32 and layer_right_required[layer]:
                        mouse_layer_access += imp * 100.0
                elif shortcut_preferred_hand[sid] == 2:  # right preferred
                    if pos_hand[i] == 0:
                        hand_bias += imp * 2.0
                elif shortcut_preferred_hand[sid] == 1:  # left preferred
                    if pos_hand[i] == 1:  # right hand
                        hand_bias += imp * 2.0

            # Finger balance.
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

            # Same-finger penalty: sum pair products * .5 for each layer/finger.
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

            # Adjacency baseline.
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

            # Violation sub-factors.
            duplicate = 0.0
            for layer in range(32):
                for base in range(n_short):
                    c = layer_base_counts[layer, base]
                    if c > 1:
                        duplicate += (c - 1) * (c - 1)

            l0_displacement = 0.0
            for i in range(n_pos):
                sid = genome[i]
                if sid >= 0 and sid < n_short and shortcut_l0_only[sid] and pos_layer[i] != 0:
                    l0_displacement += 50.0 + shortcut_importance[sid] * 2.0

            missing = 0.0
            for sid in range(n_short):
                if not assigned[sid] and shortcut_importance[sid] >= threshold:
                    missing += shortcut_importance[sid]

            cross_dup = 0.0
            for sid in range(n_short):
                layers = 0
                for layer in range(32):
                    if sid_layer_seen[sid, layer]:
                        layers += 1
                if layers >= 3:
                    extra = layers - 2
                    cross_dup += extra * extra

            group_split = 0.0
            for g in range(group_matrix.shape[0]):
                layer_counts = np.zeros(32, dtype=np.int32)
                total_members = 0
                n_layers = 0
                max_on_layer = 0
                for i in range(n_pos):
                    sid = genome[i]
                    if sid < 0 or sid >= n_short or not group_matrix[g, sid]:
                        continue
                    layer = pos_layer[i]
                    if 0 <= layer < 32:
                        if layer_counts[layer] == 0:
                            n_layers += 1
                        layer_counts[layer] += 1
                        total_members += 1
                        if layer_counts[layer] > max_on_layer:
                            max_on_layer = layer_counts[layer]
                if n_layers > 1 and total_members > 0:
                    group_split += (n_layers - 1) * 100.0
                    group_split += (1.0 - (max_on_layer / total_members)) * 50.0

            thumb_occ = 0.0
            for a in range(access.shape[0]):
                target_layer = access[a, 0]
                occupied_hand = access[a, 1]
                for i in range(n_pos):
                    sid = genome[i]
                    if sid < 0 or sid >= n_short:
                        continue
                    if pos_layer[i] == target_layer and pos_is_thumb[i] and pos_hand[i] == occupied_hand:
                        thumb_occ += 1.0 + shortcut_importance[sid] * 0.5

            arrow_order = 0.0
            for layer in range(32):
                left_x = -1.0
                right_x = -1.0
                up_x = -1.0
                down_x = -1.0
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
                    elif atype == 2:
                        right_x = pos_x[i]
                    elif atype == 3:
                        up_x = pos_x[i]
                    elif atype == 4:
                        down_x = pos_x[i]
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

            arrow_scattered = 0.0
            arrow_layers = np.zeros(32, dtype=np.int32)
            for i in range(n_pos):
                sid = genome[i]
                if sid < 0 or sid >= n_short:
                    continue
                atype = shortcut_arrow_type[sid]
                if atype != 0:
                    layer = pos_layer[i]
                    if layer != 7 and 0 <= layer < 32:
                        arrow_layers[layer] = 1
            n_arrow_layers = 0
            for layer in range(32):
                n_arrow_layers += arrow_layers[layer]
            if n_arrow_layers > 1:
                arrow_scattered = float(n_arrow_layers - 1)

            violations_raw = (
                duplicate * violation_weights[0] +
                l0_displacement * violation_weights[1] +
                missing * violation_weights[2] +
                cross_dup * violation_weights[3] +
                group_split * violation_weights[4] +
                thumb_occ * violation_weights[5] +
                arrow_order * violation_weights[6] +
                hand_bias * violation_weights[7] +
                mouse_layer_access * violation_weights[8] +
                arrow_scattered * violation_weights[9] +
                layer7_unreachable * violation_weights[10]
            )

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
            for r in range(blind_rows.shape[0]):
                sid = int(blind_rows[r, 0])
                if sid < 0 or sid >= n_short or sid_pos[sid] < 0:
                    workflow += blind_rows[r, 1] * 10.0 * 0.5

            app_coherence = 0.0
            for app in range(n_apps):
                total = app_total[app]
                if total < 2:
                    continue
                max_count = 0
                for layer in range(32):
                    if app_layer_counts[app, layer] > max_count:
                        max_count = app_layer_counts[app, layer]
                coherence = max_count / total
                app_coherence += coherence * 10.0 * math.log1p(total) * app_usage_weight[app]

            objective_effort = effort * objective_weights[0]
            objective_adj = -adjacency * objective_weights[1]
            objective_viol = (
                finger_balance * objective_weights[2] +
                same_finger * objective_weights[3] +
                violations_raw * objective_weights[4] +
                workflow * objective_weights[5] +
                learning * objective_weights[6] -
                app_coherence * objective_weights[7] -
                trackball * objective_weights[8]
            )

            out[b, 0] = objective_effort / scale_factors[0]
            out[b, 1] = objective_adj / scale_factors[1]
            out[b, 2] = objective_viol / scale_factors[2]

        return out
else:
    def _evaluate_batch_numba(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("numba is not available")
