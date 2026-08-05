"""Legacy Python-only fitness factors, used exclusively by unit tests.

Production scoring goes through the compiled kernel (fitness/kernel.py); these
classes are the legacy reference implementations moved out of fitness/factors/
to keep the production package lean. Behavior is unchanged from the original
fitness/factors/{effort,adjacency,violation}.py sources.
"""
from collections import defaultdict
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from core import Layout
from config import DEFAULT_CONFIG
from fitness import FitnessFactor, RewardFactor
from fitness.factors.violation import KEY_GROUPS, shortcut_matches_group


class EffortFactor(FitnessFactor):
    """Sum of importance * (position effort + layer access cost) for all assigned positions. Lower is better."""
    name = "effort"

    def _compute_layer_access_costs_from_genome(self, layout: Layout) -> dict:
        """Compute layer access costs from the evolved genome bindings.

        Reads shortcuts assigned in the genome that have ``is_layer_access=True``
        to build the reachability graph.  This is authoritative for evolved
        candidates because access buttons are genome capabilities, not fixed
        canonical positions.

        If no access shortcuts exist in the genome at all (e.g. a minimal test
        layout), all layers are assumed directly accessible (cost 0.0) so that
        the factor degrades gracefully instead of adding large unreachability
        penalties to every layer.
        """
        all_layers = set(p.layer for p in layout.positions)
        costs: dict = {0: 0.0}
        access_graph: dict = {}
        for i, sid in enumerate(layout.genome):
            sid = int(sid)
            if sid < 0 or sid >= len(layout.shortcuts):
                continue
            shortcut = layout.shortcuts[sid]
            if not shortcut.is_layer_access or shortcut.access_target_layer < 0:
                continue
            pos = layout.positions[i]
            target = shortcut.access_target_layer
            source = pos.layer
            access_graph.setdefault(target, []).append((source, pos.effort))

        # If no access shortcuts are present at all, assume every layer is
        # directly accessible from L0 (test/fallback path only).
        if not access_graph:
            return {layer: 0.0 for layer in all_layers}

        changed = True
        iterations = 0
        while changed and iterations < 10:
            changed = False
            iterations += 1
            for target, sources in access_graph.items():
                if target in costs:
                    continue
                for source_layer, effort in sources:
                    if source_layer in costs:
                        costs[target] = costs[source_layer] + effort
                        changed = True
                        break

        for layer in all_layers:
            if layer not in costs:
                costs[layer] = 5.0
        return costs

    def _compute_layer_access_costs(self, layout: Layout) -> dict:
        """Legacy/static fallback only. Not authoritative for evolved dynamic access.

        Reads ``layout.layer_access`` which is a legacy static fallback.  An
        evolved candidate may have placed access shortcuts
        in completely different positions.  Call
        ``_compute_layer_access_costs_from_genome`` instead for evolved layouts.
        """
        if not layout.layer_access:
            all_layers = set(p.layer for p in layout.positions)
            return {layer: 0.0 for layer in all_layers}

        access_graph: dict = {}
        for access in layout.layer_access:
            access_effort = 2.0
            for pos in layout.positions:
                if (pos.layer == access.source_layer
                        and abs(pos.x - access.source_x) < 0.5
                        and abs(pos.y - access.source_y) < 0.5):
                    access_effort = pos.effort
                    break
            access_graph.setdefault(access.target_layer, []).append(
                (access.source_layer, access_effort)
            )

        costs = {0: 0.0}
        changed = True
        iterations = 0
        while changed and iterations < 10:
            changed = False
            iterations += 1
            for target, sources in access_graph.items():
                if target in costs:
                    continue
                for source_layer, effort in sources:
                    if source_layer in costs:
                        costs[target] = costs[source_layer] + effort
                        changed = True
                        break

        for layer in set(p.layer for p in layout.positions):
            if layer not in costs:
                costs[layer] = 5.0
        return costs

    def compute(self, layout: Layout, toggle_effort_multiplier: float = 2.5) -> float:
        """Python-only legacy path (used in unit tests).

        Uses dynamic genome-based access costs.  Production scoring uses the
        compiled kernel which performs an equivalent calculation inline.
        """
        access_costs = self._compute_layer_access_costs_from_genome(layout)
        total = 0.0
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            shortcut = layout.shortcuts[sid]
            pos = layout.positions[i]
            layer = pos.layer
            access_cost = access_costs.get(layer, 0.0)
            pos_eff = pos.effort
            if shortcut.is_layer_access and not shortcut.access_is_momentary:
                pos_eff *= toggle_effort_multiplier
            total += shortcut.importance * (pos_eff + access_cost)
        return total


class AdjacencyFactor(RewardFactor):
    """Rewards shortcuts being close together based on usage co-occurrence.

    Uses app/category proximity as a baseline, then gives stronger signal to
    shortcut sequences and chains observed in usage data. Same app does not
    imply same layer when the shortcut workflows are different.
    """
    name = "adjacency"

    def __init__(self, proximity_decay: float = 0.2, usage_weight: float = 5.0):
        self.proximity_decay = proximity_decay
        self.usage_weight = usage_weight

    def compute(self, layout: Layout) -> float:
        coords = np.array([(p.x, p.y) for p in layout.positions], dtype=np.float32)

        shortcut_pos = {}
        for i, sid in enumerate(layout.genome):
            if sid >= 0:
                shortcut_pos[sid] = i

        score = 0.0

        # 1. Same-app/category adjacency (baseline)
        for sid_a, pos_a in shortcut_pos.items():
            sc_a = layout.shortcuts[sid_a]
            for sid_b, pos_b in shortcut_pos.items():
                if sid_b <= sid_a:
                    continue
                sc_b = layout.shortcuts[sid_b]
                if sc_a.app != sc_b.app and sc_a.category != sc_b.category:
                    continue

                dist = np.linalg.norm(coords[pos_a] - coords[pos_b])
                proximity = max(0.0, 1.0 - dist * self.proximity_decay)
                weight = sc_a.importance * sc_b.importance
                score += weight * proximity

        # 2. Cross-app usage sequence adjacency (workflow-aware)
        if layout.usage_data.sequences:
            score += self._usage_adjacency(layout, shortcut_pos, coords)

        # 3. Usage chain adjacency (stronger for multi-step sequences)
        if layout.usage_data.chains:
            score += self._chain_adjacency(layout, shortcut_pos, coords)

        return score

    def _usage_adjacency(self, layout: Layout, shortcut_pos: dict, coords: np.ndarray) -> float:
        """Reward shortcuts that appear in usage sequences being close together."""
        score = 0.0

        # Build reverse lookup: keys -> sid
        keys_to_sid = {}
        for sid, pos in shortcut_pos.items():
            sc = layout.shortcuts[sid]
            keys_to_sid[sc.keys] = sid

        for seq_key, seq_data in layout.usage_data.sequences.items():
            parts = seq_key.split(" -> ")
            if len(parts) != 2:
                continue

            key_a, key_b = parts[0], parts[1]
            if key_a not in keys_to_sid or key_b not in keys_to_sid:
                continue

            sid_a = keys_to_sid[key_a]
            sid_b = keys_to_sid[key_b]
            pos_a = shortcut_pos[sid_a]
            pos_b = shortcut_pos[sid_b]

            count = seq_data.get("count", 0)
            avg_gap = seq_data.get("avg_gap_ms", 5000)

            # Faster transitions = stronger adjacency desire
            speed_weight = max(0.5, 2.0 - avg_gap / 2000.0)
            weight = count * speed_weight * self.usage_weight

            # Same layer bonus (0-1 based on same layer or not)
            layer_a = layout.positions[pos_a].layer
            layer_b = layout.positions[pos_b].layer
            if layer_a == layer_b:
                dist = np.linalg.norm(coords[pos_a] - coords[pos_b])
                proximity = max(0.0, 1.0 - dist * self.proximity_decay)
            else:
                # Cross-layer: much smaller bonus, but still non-zero if close
                # (adjacent layers with same physical position are somewhat accessible)
                dist = np.linalg.norm(coords[pos_a] - coords[pos_b])
                proximity = max(0.0, 0.3 - dist * self.proximity_decay)

            score += weight * proximity

        return score

    def _chain_adjacency(self, layout: Layout, shortcut_pos: dict, coords: np.ndarray) -> float:
        """Reward adjacent pairs in usage chains being close together."""
        score = 0.0

        keys_to_sid = {}
        for sid, pos in shortcut_pos.items():
            sc = layout.shortcuts[sid]
            keys_to_sid[sc.keys] = sid

        for chain_key, chain_data in layout.usage_data.chains.items():
            parts = chain_key.split(" -> ")
            count = chain_data.get("count", 0)
            if count < 2 or len(parts) < 2:
                continue

            # Adjacent pairs in chains get strong bonus
            for i in range(len(parts) - 1):
                key_a = parts[i]
                key_b = parts[i + 1]
                if key_a not in keys_to_sid or key_b not in keys_to_sid:
                    continue

                sid_a = keys_to_sid[key_a]
                sid_b = keys_to_sid[key_b]
                pos_a = shortcut_pos[sid_a]
                pos_b = shortcut_pos[sid_b]

                layer_a = layout.positions[pos_a].layer
                layer_b = layout.positions[pos_b].layer

                if layer_a == layer_b:
                    dist = np.linalg.norm(coords[pos_a] - coords[pos_b])
                    proximity = max(0.0, 1.0 - dist * self.proximity_decay)
                else:
                    dist = np.linalg.norm(coords[pos_a] - coords[pos_b])
                    proximity = max(0.0, 0.3 - dist * self.proximity_decay)

                weight = count * self.usage_weight * 2.0  # chains are stronger than sequences
                score += weight * proximity

        return score


def _exception_score(importance: float, support: float, is_mouse: bool = False) -> float:
    """Sigmoid-like gate for repeats that are worth preserving."""
    raw = float(importance) + float(support) * 10.0 - 16.0
    if is_mouse:
        raw -= 4.0
    return 1.0 / (1.0 + math.exp(-raw * 0.45))


def _novelty_cost(importance: float, support: float, is_mouse: bool = False) -> float:
    """High cost for ordinary repeats, low cost for exceptional repeats."""
    score = _exception_score(importance, support, is_mouse=is_mouse)
    return 0.15 + (1.0 - score) * (1.0 - score)


def _allowed_raw_arrow_shape(type_positions: dict) -> bool:
    if set(type_positions) != {"LEFTARROW", "RIGHTARROW", "UPARROW", "DOWNARROW"}:
        return False
    lx, ly = type_positions["LEFTARROW"]
    rx, ry = type_positions["RIGHTARROW"]
    ux, uy = type_positions["UPARROW"]
    dx, dy = type_positions["DOWNARROW"]
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


class ViolationFactor(FitnessFactor):
    """Aggregates constraint violations. Lower is better."""
    name = "violations"

    def __init__(self, weights: dict = None, threshold: float = 6.0):
        defaults = DEFAULT_CONFIG["fitness"]["violation_sub_weights"]
        self.sub_weights = dict(defaults)
        if weights:
            self.sub_weights.update(weights)
        self.threshold = threshold

    def compute(self, layout: Layout) -> float:
        total = 0.0
        total += self._duplicate_penalty(layout) * self.sub_weights["duplicate"]
        total += self._l0_displacement(layout) * self.sub_weights["l0_displacement"]
        total += self._missing_important(layout) * self.sub_weights["missing_important"]
        total += self._cross_layer_duplicate(layout) * self.sub_weights["cross_layer_duplicate"]
        total += self._group_split(layout) * self.sub_weights["group_split"]
        total += self._thumb_occupancy(layout) * self.sub_weights["thumb_occupancy"]
        total += self._arrow_order(layout) * self.sub_weights["arrow_order"]
        total += self._hand_bias(layout) * self.sub_weights["hand_bias"]
        total += self._mouse_layer_access(layout) * self.sub_weights["mouse_layer_access"]
        total += self._arrow_scattered(layout) * self.sub_weights["arrow_scattered"]
        total += self._layer7_access(layout) * self.sub_weights["layer7_access"]
        total += self._duplicate_value_gap(layout) * self.sub_weights["duplicate_value_gap"]
        total += self._access_layout(layout) * self.sub_weights["access_layout"]
        return total

    def _duplicate_penalty(self, layout: Layout) -> float:
        from fitness.kernel import _shortcut_duplicate_support

        support = _shortcut_duplicate_support(layout)
        penalty = 0.0
        layer_base_keys = defaultdict(lambda: defaultdict(list))
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            sc = layout.shortcuts[sid]
            if not sc.base_key or sc.is_l0_only:
                continue
            layer = layout.positions[i].layer
            layer_base_keys[layer][sc.base_key.upper()].append((sid, i))

        for layer, base_map in layer_base_keys.items():
            for base_key, placements in base_map.items():
                if len(placements) > 1:
                    support_sum = sum(float(support[int(sid)]) for sid, _ in placements)
                    unsupported = (len(placements) - 1) - support_sum
                    if unsupported > 0:
                        value = 0.0
                        for sid, pos_idx in placements:
                            pos = layout.positions[pos_idx]
                            sc = layout.shortcuts[int(sid)]
                            value += max(0.25, 2.0 - pos.effort) * sc.importance
                        avg_value = value / len(placements)
                        is_mouse = any(layout.shortcuts[int(sid)].category == "mouse" for sid, _ in placements)
                        penalty += (unsupported ** 2) * _novelty_cost(avg_value, support_sum, is_mouse=is_mouse) * (1.0 + avg_value * 0.1)
        return penalty

    def _l0_displacement(self, layout: Layout) -> float:
        penalty = 0.0
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            sc = layout.shortcuts[sid]
            if sc.is_l0_only and layout.positions[i].layer != 0:
                penalty += 50.0 + sc.importance * 2.0
        return penalty

    def _missing_important(self, layout: Layout) -> float:
        """Penalize missing high-importance shortcuts."""
        penalty = 0.0
        assigned = set(layout.genome[layout.genome >= 0])
        for sc in layout.shortcuts:
            if sc.sid in assigned or sc.importance < self.threshold:
                continue
            penalty += sc.importance
        return penalty

    def _cross_layer_duplicate(self, layout: Layout) -> float:
        from fitness.kernel import _shortcut_duplicate_support

        support = _shortcut_duplicate_support(layout)
        sid_layers = defaultdict(set)
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            if layout.shortcuts[int(sid)].is_l0_only:
                continue
            sid_layers[sid].add(layout.positions[i].layer)

        penalty = 0.0
        for sid, layers in sid_layers.items():
            if len(layers) >= 2:
                extra = (len(layers) - 1) - float(support[int(sid)])
                if extra > 0:
                    shortcut = layout.shortcuts[int(sid)]
                    penalty += extra * extra * _novelty_cost(
                        shortcut.importance,
                        float(support[int(sid)]),
                        is_mouse=shortcut.category == "mouse",
                    )
        return penalty

    def _duplicate_value_gap(self, layout: Layout) -> float:
        """Penalize duplicate slots unless they clearly beat missing shortcuts."""
        from fitness.kernel import _shortcut_duplicate_support

        support = _shortcut_duplicate_support(layout)
        assigned = {int(sid) for sid in layout.genome if sid >= 0}
        missing_importance = [
            sc.importance
            for sc in layout.shortcuts
            if sc.sid not in assigned and sc.importance >= self.threshold
        ]
        if not missing_importance:
            return 0.0

        best_missing = max(missing_importance)
        counts = defaultdict(int)
        for sid in layout.genome:
            if sid >= 0:
                counts[int(sid)] += 1

        penalty = 0.0
        for sid, count in counts.items():
            if count <= 1:
                continue
            shortcut = layout.shortcuts[sid]
            if shortcut.is_l0_only:
                continue
            gap = best_missing * 1.5 - shortcut.importance
            if gap > 0:
                unsupported = (count - 1) - float(support[int(sid)])
                if unsupported > 0:
                    penalty += unsupported * gap * _novelty_cost(
                        shortcut.importance,
                        float(support[int(sid)]),
                        is_mouse=shortcut.category == "mouse",
                    )
        return penalty

    def _group_split(self, layout: Layout) -> float:
        """Penalize same-layer group scatter, never cross-layer separation."""
        penalty = 0.0

        all_groups = list(KEY_GROUPS) + list(layout.dynamic_groups)

        for group in all_groups:
            if not group.get("protected"):
                continue

            group_layers = defaultdict(list)
            group_sids = set(group.get("sids", []))

            for i, sid in enumerate(layout.genome):
                if sid < 0:
                    continue
                if sid in group_sids:
                    group_layers[layout.positions[i].layer].append(i)
                elif "sids" not in group:  # static group uses matching
                    sc = layout.shortcuts[sid]
                    if shortcut_matches_group(sc, group):
                        group_layers[layout.positions[i].layer].append(i)

            for indices in group_layers.values():
                if len(indices) < 2:
                    continue
                mean_x = sum(layout.positions[i].x for i in indices) / len(indices)
                mean_y = sum(layout.positions[i].y for i in indices) / len(indices)
                for i in indices:
                    pos = layout.positions[i]
                    spread = ((pos.x - mean_x) ** 2 + (pos.y - mean_y) ** 2) ** 0.5
                    if spread > 1.5:
                        penalty += (spread - 1.5) * 20.0

        return penalty

    def _thumb_occupancy(self, layout: Layout) -> float:
        """Penalize thumb slots occupied by assigned momentary access paths."""
        penalty = 0.0
        occupied_by_layer = defaultdict(set)
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            access = layout.shortcuts[int(sid)]
            if not access.is_layer_access or not access.access_is_momentary:
                continue
            occupied_by_layer[access.access_target_layer].add(layout.positions[i].hand)

        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            pos = layout.positions[i]
            if pos.is_thumb and pos.hand in occupied_by_layer.get(pos.layer, set()):
                sc = layout.shortcuts[int(sid)]
                penalty += 1.0 + sc.importance * 0.5
        return penalty

    def _arrow_order(self, layout: Layout) -> float:
        """Penalize arrow keys that are out of spatial order on the same layer.

        When multiple arrow keys are on the same layer:
        - LeftArrow should have lower x than RightArrow (left side)
        - UpArrow and DownArrow should be between LeftArrow and RightArrow
        """
        ARROW_KEYS = {"LEFTARROW", "RIGHTARROW", "UPARROW", "DOWNARROW"}

        # Group assigned arrow keys by layer
        layer_arrows = defaultdict(list)  # layer -> [(base_key, x, y)]
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            sc = layout.shortcuts[sid]
            if sc.base_key.upper() in ARROW_KEYS and not sc.modifiers and layout.positions[i].layer != 7:
                pos = layout.positions[i]
                layer_arrows[pos.layer].append((sc.base_key.upper(), pos.x, pos.y))

        penalty = 0.0
        for layer, arrows in layer_arrows.items():
            if len(arrows) < 2:
                continue

            x_by_key = {k: x for k, x, _ in arrows}
            y_by_key = {k: y for k, _, y in arrows}

            # LeftArrow must be to the left of RightArrow
            if "LEFTARROW" in x_by_key and "RIGHTARROW" in x_by_key:
                left_x = x_by_key["LEFTARROW"]
                right_x = x_by_key["RIGHTARROW"]
                if left_x >= right_x:
                    penalty += (left_x - right_x + 1.0) * 100.0

            # UpArrow and DownArrow should be between LeftArrow and RightArrow
            if "LEFTARROW" in x_by_key and "RIGHTARROW" in x_by_key:
                left_x = x_by_key["LEFTARROW"]
                right_x = x_by_key["RIGHTARROW"]
                min_x = min(left_x, right_x)
                max_x = max(left_x, right_x)

                for key in ("UPARROW", "DOWNARROW"):
                    if key in x_by_key:
                        x = x_by_key[key]
                        if x < min_x:
                            penalty += (min_x - x + 1.0) * 60.0
                        elif x > max_x:
                            penalty += (x - max_x + 1.0) * 60.0

            if "UPARROW" in y_by_key and "DOWNARROW" in y_by_key:
                up_y = y_by_key["UPARROW"]
                down_y = y_by_key["DOWNARROW"]
                if up_y >= down_y:
                    penalty += (up_y - down_y + 1.0) * 100.0

            if all(k in x_by_key for k in ("LEFTARROW", "RIGHTARROW", "UPARROW", "DOWNARROW")):
                type_positions = {
                    key: (x_by_key[key], y_by_key[key])
                    for key in ("LEFTARROW", "RIGHTARROW", "UPARROW", "DOWNARROW")
                }
                if not _allowed_raw_arrow_shape(type_positions):
                    penalty += 500.0

        return penalty

    def _hand_bias(self, layout: Layout) -> float:
        """Penalize mouse-category shortcuts and preferred-hand shortcuts on the wrong hand.

        Mouse category (MB1-MB5): 5x penalty for left-hand placement.
        Preferred_hand=right on left hand: 2x penalty.
        Preferred_hand=left on right hand: 2x penalty.
        """
        penalty = 0.0
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            shortcut = layout.shortcuts[sid]
            pos = layout.positions[i]

            if shortcut.category == "mouse":
                if pos.is_left:
                    penalty += shortcut.importance * 5.0
                continue

            if shortcut.preferred_hand == "right" and pos.is_left:
                penalty += shortcut.importance * 2.0
            elif shortcut.preferred_hand == "left" and pos.is_right:
                penalty += shortcut.importance * 2.0

        return penalty

    def _mouse_layer_access(self, layout: Layout) -> float:
        """Penalize mouse shortcuts on layers reached through right-hand momentary access."""
        penalty = 0.0
        right_required_layers = set()
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            access = layout.shortcuts[int(sid)]
            if access.is_layer_access and access.access_is_momentary and layout.positions[i].hand == "right":
                right_required_layers.add(access.access_target_layer)

        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            shortcut = layout.shortcuts[sid]
            if shortcut.category != "mouse":
                continue

            layer = layout.positions[i].layer
            if layer in right_required_layers:
                penalty += shortcut.importance * 100.0

        return penalty

    def _arrow_scattered(self, layout: Layout) -> float:
        """Penalize arrows split across multiple non-L7 layers.

        L7 already owns frozen RPG/navigation arrows, so mutable raw arrows are
        less important. If they appear outside L7, they should either form a
        complete justified cluster or be absent.
        """
        ARROW_KEYS = {"LEFTARROW", "RIGHTARROW", "UPARROW", "DOWNARROW"}
        layer_arrow_types = defaultdict(lambda: defaultdict(int))
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            sc = layout.shortcuts[sid]
            if sc.base_key.upper() in ARROW_KEYS and not sc.modifiers:
                layer = layout.positions[i].layer
                if layer != 7:
                    layer_arrow_types[layer][sc.base_key.upper()] += 1

        penalty = 0.0
        n_layers = len(layer_arrow_types)
        if n_layers > 1:
            penalty += float(n_layers - 1) * 100.0
        for layer, type_counts in layer_arrow_types.items():
            type_count = len(type_counts)
            placement_count = sum(type_counts.values())
            duplicate_count = sum(max(0, c - 1) for c in type_counts.values())
            if type_count < 4:
                penalty += float(4 - type_count) * 25.0
                penalty += float(placement_count) * 10.0
            elif placement_count == 4:
                type_positions = {}
                for i, sid in enumerate(layout.genome):
                    if sid < 0:
                        continue
                    sc = layout.shortcuts[int(sid)]
                    pos = layout.positions[i]
                    key = sc.base_key.upper()
                    if pos.layer == layer and key in ARROW_KEYS and not sc.modifiers:
                        type_positions[key] = (pos.x, pos.y)
                if not _allowed_raw_arrow_shape(type_positions):
                    penalty += 500.0
                # Even valid mutable raw arrows are less desirable than relying
                # on frozen L7 unless workflow pressure clearly earns them.
                penalty += 20.0
            penalty += float(duplicate_count) * 20.0
        return penalty

    def _layer7_access(self, layout: Layout) -> float:
        """Penalize if frozen L7 lacks reachable momentary or toggle access."""
        access_rows = []
        access_graph = {}
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            shortcut = layout.shortcuts[int(sid)]
            if not shortcut.is_layer_access or shortcut.access_target_layer < 0:
                continue
            source_layer = layout.positions[i].layer
            target_layer = shortcut.access_target_layer
            access_rows.append((source_layer, target_layer, shortcut.access_is_momentary))
            access_graph.setdefault(source_layer, []).append(target_layer)

        visited = {0}
        queue = [0]
        while queue:
            current = queue.pop(0)
            for neighbor in access_graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        has_momentary = any(
            target == 7 and is_momentary and source in visited
            for source, target, is_momentary in access_rows
        )
        has_toggle = any(
            target == 7 and not is_momentary and source in visited
            for source, target, is_momentary in access_rows
        )
        return float((0 if has_momentary else 1) + (0 if has_toggle else 1))

    def _access_layout(self, layout: Layout) -> float:
        """Soft score for access-button quality.

        WARNING: this reads assigned access shortcuts from the genome. Do not
        replace it with layout.layer_access; canonical access metadata is only a
        seed/source, not fixed structure.
        """
        penalty = 0.0
        layer_demand = defaultdict(float)
        direct_thumb = set()
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            pos = layout.positions[i]
            shortcut = layout.shortcuts[int(sid)]
            if shortcut.is_layer_access:
                if shortcut.access_target_layer >= 0:
                    if not pos.is_thumb:
                        penalty += 2.0 + shortcut.importance * 0.2
                    if pos.layer != 0:
                        penalty += 3.0 + (8.0 if shortcut.access_is_momentary else 0.0)
                    if pos.layer == 0 and pos.is_thumb:
                        direct_thumb.add(shortcut.access_target_layer)
                continue
            layer_demand[pos.layer] += shortcut.importance

        for layer, demand in layer_demand.items():
            if layer == 0 or demand < 30.0:
                continue
            if layer not in direct_thumb:
                penalty += 12.0
        return penalty
