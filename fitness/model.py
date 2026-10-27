"""Single-source fitness model.

This is the public interface to the compiled fitness kernel. It precomputes
all static layout data once and exposes `evaluate()`, `evaluate_batch()`,
and constraint arrays.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numba import njit

from fitness.kernel import precompute, _evaluate_batch, _single_genome, NUMBA_AVAILABLE

try:
    from fitness.cuda_kernel import evaluate_batch_cuda, build_cuda_args, cuda_available
    _CUDA_AVAILABLE = cuda_available()
except Exception:
    _CUDA_AVAILABLE = False
    evaluate_batch_cuda = None
    build_cuda_args = None


@njit(cache=True)
def _semantic_geometry_raw_batch(genomes, pos_layer, pos_x, pos_y, edges, n_shortcuts):
    """Numba post-pass for ordered-cluster geometry, shared by CPU and CUDA results."""
    batch_size, n_positions = genomes.shape
    locations_layer = np.full((batch_size, n_shortcuts), -1, dtype=np.int32)
    locations_x = np.zeros((batch_size, n_shortcuts), dtype=np.float32)
    locations_y = np.zeros((batch_size, n_shortcuts), dtype=np.float32)
    for row in range(batch_size):
        for idx in range(n_positions):
            sid = genomes[row, idx]
            if sid < 0 or sid >= n_shortcuts or pos_layer[idx] == 7:
                continue
            if locations_layer[row, sid] < 0:
                locations_layer[row, sid] = pos_layer[idx]
                locations_x[row, sid] = pos_x[idx]
                locations_y[row, sid] = pos_y[idx]
    penalties = np.zeros(batch_size, dtype=np.float32)
    for row in range(batch_size):
        for edge in range(edges.shape[0]):
            sid_a = int(edges[edge, 0])
            sid_b = int(edges[edge, 1])
            layer_a = locations_layer[row, sid_a]
            if layer_a < 0 or layer_a != locations_layer[row, sid_b]:
                continue
            dx = (locations_x[row, sid_b] - locations_x[row, sid_a]) - edges[edge, 2]
            dy = (locations_y[row, sid_b] - locations_y[row, sid_a]) - edges[edge, 3]
            error = abs(dx) + abs(dy)
            if error >= 0.5:
                penalties[row] += edges[edge, 4] * error
    return penalties


@njit(cache=True)
def _semantic_and_sparse_contract_batch(
    genomes, pos_layer, cluster_starts, cluster_sids, cluster_weights,
    access_targets, semantic_penalty, sparse_base_penalty, sparse_gap_penalty,
):
    """Score explicit layout contracts outside the historically huge violation scale.

    Semantic co-location is measured as the fraction of member pairs that do not
    share any layer. Reachability is derived from each genome's placed access
    shortcuts, matching the dynamic layout rather than a static layer list.
    """
    batch_size, n_positions = genomes.shape
    n_shortcuts = access_targets.shape[0]
    n_clusters = cluster_weights.shape[0]
    penalties = np.zeros(batch_size, dtype=np.float32)
    for row in range(batch_size):
        layer_masks = np.zeros(n_shortcuts, dtype=np.uint32)
        for idx in range(n_positions):
            sid = int(genomes[row, idx])
            layer = int(pos_layer[idx])
            if sid < 0 or sid >= n_shortcuts or layer < 0 or layer >= 32:
                continue
            layer_masks[sid] |= np.uint32(1) << np.uint32(layer)

        reachable = np.zeros(32, dtype=np.bool_)
        reachable[0] = True
        for _ in range(32):
            changed = False
            for idx in range(n_positions):
                sid = int(genomes[row, idx])
                if sid < 0 or sid >= n_shortcuts:
                    continue
                target = int(access_targets[sid])
                source = int(pos_layer[idx])
                if (target >= 0 and target < 32 and source >= 0 and source < 32
                        and reachable[source] and not reachable[target]):
                    reachable[target] = True
                    changed = True
            if not changed:
                break

        for cluster in range(n_clusters):
            start = int(cluster_starts[cluster])
            end = int(cluster_starts[cluster + 1])
            total_pairs = 0
            split_pairs = 0
            for i in range(start, end):
                sid_a = int(cluster_sids[i])
                if sid_a < 0 or sid_a >= n_shortcuts:
                    continue
                for j in range(i + 1, end):
                    sid_b = int(cluster_sids[j])
                    if sid_b < 0 or sid_b >= n_shortcuts:
                        continue
                    total_pairs += 1
                    if (layer_masks[sid_a] & layer_masks[sid_b]) == np.uint32(0):
                        split_pairs += 1
            if total_pairs > 0 and split_pairs > 0:
                penalties[row] += cluster_weights[cluster] * semantic_penalty * (
                    float(split_pairs) / float(total_pairs)
                )

    return penalties


@njit(cache=True)
def _right_alt_l0_penalty_batch(genomes, pos_layer, right_alt_sid, penalty):
    """Penalize placing Norwegian AltGr beside the dedicated L0 shortcut Alt."""
    batch_size, n_positions = genomes.shape
    scores = np.zeros(batch_size, dtype=np.float32)
    if right_alt_sid < 0 or penalty <= 0.0:
        return scores
    for row in range(batch_size):
        for idx in range(n_positions):
            if int(genomes[row, idx]) == right_alt_sid and int(pos_layer[idx]) == 0:
                scores[row] = penalty
                break
    return scores


@njit(cache=True)
def _layer_usage_mismatch_batch(genomes, pos_layer, preference_index, preference_weights):
    """Return weighted loss for moving logged modifier shortcuts off their dominant layer."""
    batch_size, n_positions = genomes.shape
    scores = np.zeros(batch_size, dtype=np.float32)
    n_preferences = preference_weights.shape[0]
    if n_preferences == 0:
        return scores
    covered = np.zeros((batch_size, n_preferences), dtype=np.bool_)
    n_shortcuts = preference_index.shape[0]
    for row in range(batch_size):
        for idx in range(n_positions):
            sid = int(genomes[row, idx])
            layer = int(pos_layer[idx])
            if sid < 0 or sid >= n_shortcuts or layer < 0 or layer >= 32:
                continue
            preference = int(preference_index[sid, layer])
            if preference >= 0 and preference < n_preferences:
                covered[row, preference] = True
        for preference in range(n_preferences):
            if not covered[row, preference]:
                scores[row] += preference_weights[preference]
    return scores


@njit(cache=True)
def _l0_toggle_access_violation_batch(genomes, pos_layer, access_targets, access_is_momentary):
    """Return one hard-constraint violation when any toggle is assigned on L0."""
    batch_size, n_positions = genomes.shape
    scores = np.zeros((batch_size, 1), dtype=np.float32)
    n_shortcuts = access_targets.shape[0]
    for row in range(batch_size):
        for idx in range(n_positions):
            sid = int(genomes[row, idx])
            if sid < 0 or sid >= n_shortcuts or int(pos_layer[idx]) != 0:
                continue
            if int(access_targets[sid]) >= 0 and not access_is_momentary[sid]:
                scores[row, 0] = 1.0
                break
    return scores


@njit(cache=True)
def _right_alt_l0_violation_batch(genomes, pos_layer, right_alt_sid):
    batch_size, n_positions = genomes.shape
    scores = np.zeros((batch_size, 1), dtype=np.float32)
    if right_alt_sid < 0:
        return scores
    for row in range(batch_size):
        for idx in range(n_positions):
            if int(genomes[row, idx]) == right_alt_sid and int(pos_layer[idx]) == 0:
                scores[row, 0] = 1.0
                break
    return scores


@njit(cache=True)
def _mouse_layer_shortcut_penalty_batch(
    genomes, pos_layer, pos_x, pos_effort, pos_hand, pos_is_thumb,
    access_targets, access_is_momentary, is_mouse_button, mouse_button,
    is_scroll_access, preference_index, preference_weights, max_effort,
    pos_y=None, preferred_access_penalty=0.0,
):
    """Reward frequent mouse-session shortcuts on the contract-valid mouse layer."""
    batch_size, n_positions = genomes.shape
    scores = np.zeros(batch_size, dtype=np.float32)
    n_shortcuts = preference_index.shape[0]
    n_preferences = preference_weights.shape[0]
    if n_preferences == 0 and preferred_access_penalty <= 0.0:
        return scores
    for row in range(batch_size):
        right_buttons = np.zeros((32, 6), dtype=np.int32)
        left_buttons = np.zeros((32, 6), dtype=np.int32)
        right_thumb_buttons = np.zeros((32, 6), dtype=np.int32)
        scroll_right = np.zeros(32, dtype=np.bool_)
        scroll_bad_x = np.zeros(32, dtype=np.bool_)
        right_thumb_momentary = np.zeros(32, dtype=np.bool_)
        toggle_reachable_edge = np.zeros((32, 32), dtype=np.bool_)
        momentary_edge = np.zeros((32, 32), dtype=np.bool_)
        preferred_mouse_access = np.zeros(32, dtype=np.bool_)
        for idx in range(n_positions):
            sid = int(genomes[row, idx])
            source = int(pos_layer[idx])
            if sid < 0 or sid >= n_shortcuts or source < 0 or source >= 32:
                continue
            if is_mouse_button[sid]:
                button = int(mouse_button[sid])
                if 1 <= button <= 5 and source != 0 and source != 7:
                    if pos_hand[idx] == 1:
                        right_buttons[source, button] += 1
                        if pos_is_thumb[idx]:
                            right_thumb_buttons[source, button] += 1
                    else:
                        left_buttons[source, button] += 1
            target = int(access_targets[sid])
            if target < 0 or target >= 32:
                continue
            if (preferred_access_penalty > 0.0 and pos_y is not None
                    and source == 0 and pos_x[idx] == 3.0 and pos_y[idx] == 4.0
                    and not is_scroll_access[sid]):
                preferred_mouse_access[target] = True
            if is_scroll_access[sid] and source == target and source != 0 and source != 7 and pos_hand[idx] == 1 and not pos_is_thumb[idx]:
                scroll_right[target] = True
                if pos_x[idx] == 7.0 or pos_x[idx] == 8.0:
                    scroll_bad_x[target] = True
            if source != target and not is_scroll_access[sid]:
                if access_is_momentary[sid]:
                    momentary_edge[source, target] = True
                    if pos_hand[idx] == 1 and pos_is_thumb[idx]:
                        right_thumb_momentary[target] = True
                else:
                    toggle_reachable_edge[source, target] = True

        reachable = np.zeros(32, dtype=np.bool_)
        reachable[0] = True
        for _ in range(32):
            changed = False
            for source in range(32):
                if not reachable[source]:
                    continue
                for target in range(32):
                    if (toggle_reachable_edge[source, target] or momentary_edge[source, target]) and not reachable[target]:
                        reachable[target] = True
                        changed = True
            if not changed:
                break
        mouse_layer = -1
        for layer in range(1, 32):
            if layer == 7:
                continue
            valid = True
            for button in range(1, 6):
                if right_buttons[layer, button] == 0 or right_buttons[layer, button] > 1 or left_buttons[layer, button] > 1:
                    valid = False
                if right_thumb_buttons[layer, button] > 0:
                    valid = False
            toggle_accessible = False
            momentary_accessible = False
            for source in range(32):
                if reachable[source]:
                    if toggle_reachable_edge[source, layer]:
                        toggle_accessible = True
                    if momentary_edge[source, layer]:
                        momentary_accessible = True
            if (valid and scroll_right[layer] and not scroll_bad_x[layer]
                    and not right_thumb_momentary[layer] and reachable[layer]
                    and toggle_accessible and momentary_accessible):
                mouse_layer = layer
                break
        if mouse_layer < 0:
            scores[row] = 1.0
            continue
        if preferred_access_penalty > 0.0 and not preferred_mouse_access[mouse_layer]:
            scores[row] += preferred_access_penalty

        best_effort = np.full(n_shortcuts, max_effort, dtype=np.float32)
        present = np.zeros(n_shortcuts, dtype=np.bool_)
        for idx in range(n_positions):
            sid = int(genomes[row, idx])
            if sid < 0 or sid >= n_shortcuts or int(pos_layer[idx]) != mouse_layer:
                continue
            if preference_index[sid, mouse_layer] >= 0:
                present[sid] = True
                effort = float(pos_effort[idx])
                if effort < best_effort[sid]:
                    best_effort[sid] = effort
        for sid in range(n_shortcuts):
            preference = int(preference_index[sid, mouse_layer])
            if preference < 0 or preference >= n_preferences:
                continue
            weight = preference_weights[preference]
            if not present[sid]:
                scores[row] += weight
            elif max_effort > 0.0:
                scores[row] += weight * 0.25 * best_effort[sid] / max_effort
    return scores


def _build_layer_usage_preferences(layout, min_count=3, min_share=0.60):
    """Map confident logged modifier-shortcut use to observed layer IDs.

    Layer IDs remain data labels; this does not assign semantic roles to any
    dynamic layer. Unmodified keys are excluded because ordinary typing is not
    fully observed by the logger.
    """
    from fitness.factors.workflow_coherence import WorkflowCoherenceFactor

    normalize = WorkflowCoherenceFactor._normalize_keys
    sid_by_key = {}
    for shortcut in layout.shortcuts:
        sid_by_key[shortcut.keys] = int(shortcut.sid)
        sid_by_key[normalize(shortcut.keys)] = int(shortcut.sid)

    modifier_names = {"ctrl", "control", "shift", "alt", "leftalt", "rightalt", "win", "gui", "cmd", "command"}
    observed = {}
    for keys, by_layer in layout.usage_data.by_layer_shortcut.items():
        sid = sid_by_key.get(keys, sid_by_key.get(normalize(keys), -1))
        if sid < 0 or sid >= layout.n_shortcuts:
            continue
        shortcut = layout.shortcuts[sid]
        if shortcut.is_layer_access or shortcut.category == "mouse":
            continue
        parts = [part.strip().lower() for part in str(keys).split("+")]
        if not shortcut.modifiers and not any(part in modifier_names for part in parts[:-1]):
            continue
        for layer_text, raw_count in (by_layer or {}).items():
            try:
                layer, count = int(layer_text), int(raw_count)
            except (TypeError, ValueError):
                continue
            if 0 <= layer < 32 and layer != 7 and count > 0:
                observed.setdefault(sid, {})[layer] = observed.setdefault(sid, {}).get(layer, 0) + count

    preferences = []
    for sid, counts in observed.items():
        total = sum(counts.values())
        if total < min_count:
            continue
        target_layer, target_count = max(counts.items(), key=lambda item: (item[1], item[0] == 0, -item[0]))
        share = target_count / total
        if share < min_share:
            continue
        preferences.append((sid, target_layer, math.log1p(total) * (2.0 * share - 1.0)))

    preference_index = np.full((layout.n_shortcuts, 32), -1, dtype=np.int32)
    preference_weights = np.asarray([row[2] for row in preferences], dtype=np.float32)
    if preference_weights.size:
        total_weight = float(preference_weights.sum())
        if total_weight > 0.0:
            preference_weights /= np.float32(total_weight)
        for index, (sid, layer, _weight) in enumerate(preferences):
            preference_index[sid, layer] = index
    return preference_index, preference_weights


def _build_mouse_session_shortcut_preferences(layout, min_count=3):
    """Normalize logged shortcut demand observed during mouse-use sessions."""
    from fitness.factors.workflow_coherence import WorkflowCoherenceFactor

    normalize = WorkflowCoherenceFactor._normalize_keys
    sid_by_key = {}
    for shortcut in layout.shortcuts:
        sid_by_key[shortcut.keys] = int(shortcut.sid)
        sid_by_key[normalize(shortcut.keys)] = int(shortcut.sid)
    rows = []
    for keys, data in layout.usage_data.mouse_session_shortcuts.items():
        sid = sid_by_key.get(keys, sid_by_key.get(normalize(keys), -1))
        if sid < 0 or sid >= layout.n_shortcuts or not isinstance(data, dict):
            continue
        shortcut = layout.shortcuts[sid]
        if shortcut.is_layer_access or shortcut.category == "mouse":
            continue
        try:
            count = float(data.get("count", 0))
        except (TypeError, ValueError):
            continue
        if count < min_count:
            continue
        global_data = layout.usage_data.shortcuts.get(keys, {})
        global_count = float(global_data.get("count", 0)) if isinstance(global_data, dict) else 0.0
        weight = math.log1p(count) * (1.0 + 0.2 * math.log1p(global_count))
        if weight > 0.0:
            rows.append((sid, weight))
    total = sum(weight for _sid, weight in rows)
    weights = np.asarray([weight / total for _sid, weight in rows], dtype=np.float32) if total else np.asarray([], dtype=np.float32)
    preference_index = np.full((layout.n_shortcuts, 32), -1, dtype=np.int32)
    for index, (sid, _weight) in enumerate(rows):
        preference_index[sid, :] = index
    return preference_index, weights


def _build_owner_mouse_layer_shortcut_preferences(layout, keys):
    """Build explicit owner-requested shortcut placement on the dynamic mouse layer."""
    from fitness.factors.workflow_coherence import WorkflowCoherenceFactor

    normalize = WorkflowCoherenceFactor._normalize_keys
    sid_by_key = {}
    for shortcut in layout.shortcuts:
        sid_by_key[str(shortcut.keys)] = int(shortcut.sid)
        sid_by_key[normalize(shortcut.keys)] = int(shortcut.sid)

    preference_index = np.full((layout.n_shortcuts, 32), -1, dtype=np.int32)
    selected = []
    for key in keys or []:
        sid = sid_by_key.get(str(key), sid_by_key.get(normalize(key), -1))
        if sid < 0 or sid >= layout.n_shortcuts or sid in selected:
            continue
        shortcut = layout.shortcuts[sid]
        if shortcut.is_layer_access or shortcut.category == "mouse":
            continue
        selected.append(sid)

    weights = np.ones(len(selected), dtype=np.float32)
    for index, sid in enumerate(selected):
        preference_index[sid, :] = index
    return preference_index, weights


@njit(cache=True)
def _access_hold_target_position_conflict_batch(
    genomes, pos_layer, access_targets, access_is_momentary,
    is_scroll_access, target_position_by_source,
):
    """Count momentary layer holds masking a binding at the same target key."""
    batch_size, n_positions = genomes.shape
    scores = np.zeros((batch_size, 1), dtype=np.float32)
    n_shortcuts = access_targets.shape[0]
    for row in range(batch_size):
        for idx in range(n_positions):
            sid = int(genomes[row, idx])
            if sid < 0 or sid >= n_shortcuts:
                continue
            if not access_is_momentary[sid] or is_scroll_access[sid]:
                continue
            source = int(pos_layer[idx])
            target = int(access_targets[sid])
            if target < 0 or target >= 32 or target == source:
                continue
            target_idx = int(target_position_by_source[idx, target])
            if target_idx >= 0 and int(genomes[row, target_idx]) >= 0:
                scores[row, 0] += 1.0
    return scores


@njit(cache=True)
def _shortcut_pair_together_violation_batch(genomes, pos_layer, sid_a, sid_b):
    """Require a complementary shortcut pair to exist together on one layer."""
    batch_size, n_positions = genomes.shape
    scores = np.zeros((batch_size, 1), dtype=np.float32)
    if sid_a < 0 or sid_b < 0:
        scores[:, 0] = 1.0
        return scores
    for row in range(batch_size):
        in_layers = np.zeros(32, dtype=np.bool_)
        out_layers = np.zeros(32, dtype=np.bool_)
        for idx in range(n_positions):
            layer = int(pos_layer[idx])
            if layer < 0 or layer >= 32:
                continue
            sid = int(genomes[row, idx])
            if sid == sid_a:
                in_layers[layer] = True
            elif sid == sid_b:
                out_layers[layer] = True
        together = False
        for layer in range(32):
            if in_layers[layer] and out_layers[layer]:
                together = True
                break
        if not together:
            scores[row, 0] = 1.0
    return scores


@njit(cache=True)
def _l5_hold_self_toggle_path_violation_batch(genomes, pos_layer, hold_sid, toggle_sid):
    """Require the owner-requested L0 hold -> L5 self-toggle path."""
    batch_size, n_positions = genomes.shape
    scores = np.zeros((batch_size, 1), dtype=np.float32)
    for row in range(batch_size):
        has_l0_hold = False
        has_l5_self_toggle = False
        for idx in range(n_positions):
            sid = int(genomes[row, idx])
            layer = int(pos_layer[idx])
            if sid == hold_sid and layer == 0:
                has_l0_hold = True
            elif sid == toggle_sid and layer == 5:
                has_l5_self_toggle = True
        if not has_l0_hold or not has_l5_self_toggle:
            scores[row, 0] = 1.0
    return scores


@dataclass
class ParityResult:
    ok: bool
    max_abs_diff: float
    message: str


class FitnessModel:
    """Compiled fitness model: one source of truth for all evaluations.

    Args:
        layer_access_thumb_params: Optional dict with thumb-aware layer access
            coefficients (thumb_bonus, non_thumb_penalty, non_thumb_base,
            same_side_penalty, opposite_side_reward). When omitted, the kernel
            falls back to ``DEFAULT_CONFIG["fitness"]["layer_access_thumb"]``.
    """

    def __init__(
        self,
        layout,
        weights: Dict[str, float],
        violation_weights: Dict[str, float],
        missing_important_threshold: float,
        scale_factors: Optional[np.ndarray] = None,
        reference_genome: Optional[np.ndarray] = None,
        hard_constraints: Optional[List[str]] = None,
        toggle_effort_multiplier: float = 2.5,
        require_cuda: bool = False,
        use_cuda: Optional[bool] = None,
        layer_access_thumb_params: Optional[Dict] = None,
        semantic_cluster_multiplier: float = 200.0,
        semantic_cluster_pair_boost: float = 1.0,
        semantic_contract_penalty: float = 0.0,
        semantic_position_penalty: float = 0.0,
        sparse_layer_base_penalty: float = 0.0,
        sparse_layer_gap_penalty: float = 0.0,
        right_alt_l0_penalty: float = 5000.0,
        layer_usage_preference_penalty: float = 200.0,
        mouse_layer_shortcut_penalty: float = 10000.0,
        mouse_layer_access_position_penalty: float = 2.0,
        owner_mouse_layer_shortcut_preferences: Optional[List[str]] = None,
        owner_mouse_layer_shortcut_penalty: float = 50000.0,
    ):
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba is required by the single-source fitness model")

        self.layout = layout
        self.weights = weights
        self.violation_weights = violation_weights
        self.missing_important_threshold = missing_important_threshold
        self.scale_factors = scale_factors if scale_factors is not None else np.ones(3, dtype=np.float32)
        self.reference_genome = reference_genome
        self.hard_constraints = list(hard_constraints or [])
        custom_names = {
            "l0_toggle_access", "right_alt_off_l0",
            "cmdpal_on_mouse_layer", "access_hold_target_position_conflict",
            "zoom_controls_present_together", "monitor_move_pair_together",
            "l5_hold_to_self_toggle_path",
        }
        self._custom_hard_constraints = [name for name in self.hard_constraints if name in custom_names]
        self._kernel_hard_constraints = [name for name in self.hard_constraints if name not in custom_names]
        self.require_cuda = require_cuda
        self.layer_access_thumb_params = layer_access_thumb_params

        self.toggle_effort_multiplier = toggle_effort_multiplier
        self.semantic_cluster_multiplier = semantic_cluster_multiplier
        self.semantic_cluster_pair_boost = float(semantic_cluster_pair_boost)
        self.semantic_contract_penalty = float(semantic_contract_penalty)
        self.semantic_position_penalty = float(semantic_position_penalty)
        self.sparse_layer_base_penalty = float(sparse_layer_base_penalty)
        self.sparse_layer_gap_penalty = float(sparse_layer_gap_penalty)
        self.right_alt_l0_penalty = float(right_alt_l0_penalty)
        self.layer_usage_preference_penalty = float(layer_usage_preference_penalty)
        self.mouse_layer_shortcut_penalty = float(mouse_layer_shortcut_penalty)
        self.mouse_layer_access_position_penalty = float(mouse_layer_access_position_penalty)
        self.owner_mouse_layer_shortcut_penalty = float(owner_mouse_layer_shortcut_penalty)
        self._layer_usage_preference_index, self._layer_usage_preference_weights = (
            _build_layer_usage_preferences(layout)
        )
        self._mouse_layer_shortcut_preference_index, self._mouse_layer_shortcut_preference_weights = (
            _build_mouse_session_shortcut_preferences(layout)
        )
        self._owner_mouse_layer_shortcut_preference_index, self._owner_mouse_layer_shortcut_preference_weights = (
            _build_owner_mouse_layer_shortcut_preferences(
                layout, owner_mouse_layer_shortcut_preferences or []
            )
        )
        self._cmdpal_mouse_layer_preference_index, self._cmdpal_mouse_layer_preference_weights = (
            _build_owner_mouse_layer_shortcut_preferences(layout, ["Win+Alt+Space"])
        )
        self._right_alt_sid = next((int(s.sid) for s in layout.shortcuts
                                    if s.base_key == "RightAlt" and not s.modifiers), -1)
        self._zoom_in_sid = next((int(s.sid) for s in layout.shortcuts if s.keys == "Ctrl++"), -1)
        self._zoom_out_sid = next((int(s.sid) for s in layout.shortcuts if s.keys == "Ctrl+-"), -1)
        self._monitor_left_sid = next((int(s.sid) for s in layout.shortcuts if s.keys == "Win+Shift+Left"), -1)
        self._monitor_right_sid = next((int(s.sid) for s in layout.shortcuts if s.keys == "Win+Shift+Right"), -1)
        self._l5_hold_sid = next((int(s.sid) for s in layout.shortcuts if s.keys == "@access:L5:hold"), -1)
        self._l5_toggle_sid = next((int(s.sid) for s in layout.shortcuts if s.keys == "@access:L5:toggle"), -1)
        self._semantic_position_edges = self._build_semantic_position_edges(layout)
        self._semantic_position_edges_array = np.asarray(
            self._semantic_position_edges, dtype=np.float32
        ).reshape((-1, 5))
        self._position_layers = np.asarray([p.layer for p in layout.positions], dtype=np.int32)
        self._position_x = np.asarray([p.x for p in layout.positions], dtype=np.float32)
        self._position_y = np.asarray([p.y for p in layout.positions], dtype=np.float32)
        coordinate_positions = {
            (int(p.layer), float(p.x), float(p.y)): index
            for index, p in enumerate(layout.positions)
        }
        self._target_position_by_source = np.full(
            (len(layout.positions), 32), -1, dtype=np.int32
        )
        for source_idx, position in enumerate(layout.positions):
            for target_layer in range(32):
                self._target_position_by_source[source_idx, target_layer] = (
                    coordinate_positions.get((target_layer, float(position.x), float(position.y)), -1)
                )
        self._position_effort = np.asarray([p.effort for p in layout.positions], dtype=np.float32)
        self._position_hand = np.asarray([1 if p.hand == "right" else 0 for p in layout.positions], dtype=np.int32)
        self._position_is_thumb = np.asarray([p.is_thumb for p in layout.positions], dtype=np.bool_)
        self._max_position_effort = np.float32(max((p.effort for p in layout.positions), default=1.0))
        self._semantic_cluster_starts, self._semantic_cluster_sids, self._semantic_cluster_weights = (
            self._build_semantic_contract_groups(layout)
        )
        self._access_targets = np.full(layout.n_shortcuts, -1, dtype=np.int32)
        self._access_is_momentary = np.zeros(layout.n_shortcuts, dtype=np.bool_)
        self._is_mouse_button = np.zeros(layout.n_shortcuts, dtype=np.bool_)
        self._mouse_button = np.zeros(layout.n_shortcuts, dtype=np.int32)
        self._is_scroll_access = np.zeros(layout.n_shortcuts, dtype=np.bool_)
        for shortcut in layout.shortcuts:
            key = str(shortcut.keys).upper()
            if key in {"MB1", "MB2", "MB3", "MB4", "MB5"}:
                self._is_mouse_button[int(shortcut.sid)] = True
                self._mouse_button[int(shortcut.sid)] = int(key[-1])
            if not getattr(shortcut, "is_layer_access", False):
                continue
            target = int(getattr(shortcut, "access_target_layer", -1))
            if 0 <= target < 32:
                self._access_targets[int(shortcut.sid)] = target
                self._access_is_momentary[int(shortcut.sid)] = bool(shortcut.access_is_momentary)
                self._is_scroll_access[int(shortcut.sid)] = str(shortcut.keys).startswith("@scroll:")
        self.arrays = precompute(
            layout,
            weights=weights,
            violation_weights=violation_weights,
            missing_important_threshold=missing_important_threshold,
            scale_factors=self.scale_factors,
            reference_genome=reference_genome,
            hard_constraints=self._kernel_hard_constraints,
            toggle_effort_multiplier=toggle_effort_multiplier,
            layer_access_thumb_params=layer_access_thumb_params,
            semantic_cluster_multiplier=semantic_cluster_multiplier,
            semantic_cluster_pair_boost=self.semantic_cluster_pair_boost,
        )

        # CUDA is the only production batch-evaluation path. Numba remains
        # available for unit tests and explicit CPU diagnostics only.
        self._use_cuda = bool(_CUDA_AVAILABLE) if use_cuda is None else bool(use_cuda)
        if self.require_cuda and not self._use_cuda:
            raise RuntimeError(
                "CUDA exact evaluation is required but unavailable. "
                "Do not start a CPU-primary training run."
            )

        # Cache static CUDA tensors so each batch eval does not rebuild them.
        self._cuda_args = build_cuda_args(self.arrays) if self._use_cuda else None

        # Force one JIT compile/warm-up call on a single genome.
        _single_genome(layout.genome, *self.arrays)

    def set_semantic_cluster_multiplier(self, multiplier: float):
        """Rebuild precomputed arrays for a new semantic-cluster multiplier.

        Used by staged training schedules that ramp cluster pressure after the
        mouse layer, completion cluster, and other hard constraints are stable.
        """
        self.semantic_cluster_multiplier = float(multiplier)
        self.arrays = precompute(
            self.layout,
            weights=self.weights,
            violation_weights=self.violation_weights,
            missing_important_threshold=self.missing_important_threshold,
            scale_factors=self.scale_factors,
            reference_genome=self.reference_genome,
            hard_constraints=self._kernel_hard_constraints,
            toggle_effort_multiplier=self.toggle_effort_multiplier,
            layer_access_thumb_params=self.layer_access_thumb_params,
            semantic_cluster_multiplier=self.semantic_cluster_multiplier,
            semantic_cluster_pair_boost=self.semantic_cluster_pair_boost,
        )
        if self._use_cuda:
            self._cuda_args = build_cuda_args(self.arrays)

    @staticmethod
    def _build_semantic_position_edges(layout):
        """Build ordered member-to-anchor relations; compact-only groups add none."""
        edges = {}
        for cluster in getattr(layout, "semantic_clusters", ()):
            members = [m for m in cluster.get("members", ()) if int(m.get("sid", -1)) >= 0]
            if len(members) < 2:
                continue
            anchor = next((m for m in members if int(m.get("order", 0)) == 0), members[0])
            anchor_sid = int(anchor["sid"])
            anchor_dx, anchor_dy = float(anchor.get("dx", 0.0)), float(anchor.get("dy", 0.0))
            base_weight = float(cluster.get("weight", 1.0)) / math.sqrt(len(members))
            if cluster.get("is_critical", False):
                base_weight *= 8.0
            for member in members:
                sid = int(member["sid"])
                if sid == anchor_sid:
                    continue
                dx = float(member.get("dx", 0.0)) - anchor_dx
                dy = float(member.get("dy", 0.0)) - anchor_dy
                if abs(dx) <= 0.01 and abs(dy) <= 0.01:
                    continue
                key = (anchor_sid, sid, round(dx, 3), round(dy, 3))
                edges[key] = max(edges.get(key, 0.0), base_weight)
        return tuple((*edge, weight) for edge, weight in sorted(edges.items()))

    @staticmethod
    def _build_semantic_contract_groups(layout):
        starts = [0]
        sids = []
        weights = []
        for cluster in getattr(layout, "semantic_clusters", ()):
            members = []
            for member in cluster.get("members", ()):
                sid = int(member.get("sid", -1))
                if 0 <= sid < layout.n_shortcuts and sid not in members:
                    members.append(sid)
            if len(members) < 2:
                continue
            sids.extend(members)
            starts.append(len(sids))
            importance = min(1.0, max(0.5, float(cluster.get("weight", 1.0)) / 5.0))
            if cluster.get("is_critical", False):
                importance *= 1.5
            weights.append(importance)
        return (
            np.asarray(starts, dtype=np.int32),
            np.asarray(sids, dtype=np.int32),
            np.asarray(weights, dtype=np.float32),
        )

    def _add_semantic_position_score(self, objectives, genomes):
        """Score wrong relative placements identically after CPU or CUDA evaluation."""
        values = np.asarray(objectives, dtype=np.float32).copy()
        if values.ndim == 1:
            values = values.reshape(1, -1)
        batch = np.asarray(genomes, dtype=np.int32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        if self._semantic_position_edges:
            raw_penalty = _semantic_geometry_raw_batch(
                batch, self._position_layers, self._position_x, self._position_y,
                self._semantic_position_edges_array, self.layout.n_shortcuts,
            )
            values[:, 2] += raw_penalty * np.float32(self.semantic_position_penalty)
            multiplier = self.semantic_cluster_multiplier * self.semantic_cluster_pair_boost
            if multiplier > 0.0:
                workflow_weight = float(self.weights.get("workflow_coherence", 1.0))
                scale = max(float(self.scale_factors[2]), 1e-9)
                values[:, 2] += raw_penalty * np.float32(
                    multiplier * 10.0 * workflow_weight / scale
                )
        if self.semantic_contract_penalty > 0.0 or self.sparse_layer_base_penalty > 0.0:
            values[:, 2] += _semantic_and_sparse_contract_batch(
                batch, self._position_layers,
                self._semantic_cluster_starts, self._semantic_cluster_sids,
                self._semantic_cluster_weights, self._access_targets,
                np.float32(self.semantic_contract_penalty),
                np.float32(self.sparse_layer_base_penalty),
                np.float32(self.sparse_layer_gap_penalty),
            )
        right_alt_sid = getattr(self, "_right_alt_sid", -1)
        right_alt_penalty = float(getattr(self, "right_alt_l0_penalty", 0.0))
        if right_alt_sid >= 0 and right_alt_penalty > 0.0:
            values[:, 2] += _right_alt_l0_penalty_batch(
                batch, self._position_layers, right_alt_sid,
                np.float32(right_alt_penalty),
            )
        layer_usage_penalty = float(getattr(self, "layer_usage_preference_penalty", 0.0))
        preference_weights = getattr(self, "_layer_usage_preference_weights", np.asarray([]))
        if layer_usage_penalty > 0.0 and preference_weights.size:
            mismatch = _layer_usage_mismatch_batch(
                batch, self._position_layers,
                getattr(self, "_layer_usage_preference_index", np.empty((0, 0), dtype=np.int32)),
                preference_weights,
            )
            values[:, 2] += mismatch * np.float32(layer_usage_penalty)
        mouse_penalty = float(getattr(self, "mouse_layer_shortcut_penalty", 0.0))
        mouse_weights = getattr(self, "_mouse_layer_shortcut_preference_weights", np.asarray([]))
        if mouse_penalty > 0.0 and mouse_weights.size:
            mismatch = _mouse_layer_shortcut_penalty_batch(
                batch, self._position_layers, self._position_x,
                getattr(self, "_position_effort", np.ones(len(self._position_layers), dtype=np.float32)),
                getattr(self, "_position_hand", np.zeros(len(self._position_layers), dtype=np.int32)),
                getattr(self, "_position_is_thumb", np.zeros(len(self._position_layers), dtype=np.bool_)),
                self._access_targets, self._access_is_momentary,
                self._is_mouse_button, self._mouse_button, self._is_scroll_access,
                getattr(self, "_mouse_layer_shortcut_preference_index", np.empty((0, 0), dtype=np.int32)),
                mouse_weights, float(getattr(self, "_max_position_effort", 1.0)),
                self._position_y,
                float(getattr(self, "mouse_layer_access_position_penalty", 0.0)),
            )
            values[:, 2] += mismatch * np.float32(mouse_penalty)
        owner_penalty = float(getattr(self, "owner_mouse_layer_shortcut_penalty", 0.0))
        owner_weights = getattr(
            self, "_owner_mouse_layer_shortcut_preference_weights", np.asarray([])
        )
        if owner_penalty > 0.0 and owner_weights.size:
            owner_mismatch = _mouse_layer_shortcut_penalty_batch(
                batch, self._position_layers, self._position_x,
                self._position_effort, self._position_hand, self._position_is_thumb,
                self._access_targets, self._access_is_momentary,
                self._is_mouse_button, self._mouse_button, self._is_scroll_access,
                self._owner_mouse_layer_shortcut_preference_index, owner_weights,
                float(self._max_position_effort), self._position_y,
            )
            values[:, 2] += owner_mismatch * np.float32(owner_penalty)
        return values

    def _append_custom_constraints(self, genomes, constraints):
        if not self._custom_hard_constraints:
            return constraints
        batch = np.asarray(genomes, dtype=np.int32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        custom_columns = []
        for name in self._custom_hard_constraints:
            if name == "l0_toggle_access":
                custom_columns.append(_l0_toggle_access_violation_batch(
                    batch, self._position_layers, self._access_targets, self._access_is_momentary,
                ))
            elif name == "right_alt_off_l0":
                custom_columns.append(_right_alt_l0_violation_batch(
                    batch, self._position_layers, getattr(self, "_right_alt_sid", -1),
                ))
            elif name == "cmdpal_on_mouse_layer":
                cmdpal_weights = self._cmdpal_mouse_layer_preference_weights
                if cmdpal_weights.size:
                    missing = _mouse_layer_shortcut_penalty_batch(
                        batch, self._position_layers, self._position_x,
                        self._position_effort, self._position_hand, self._position_is_thumb,
                        self._access_targets, self._access_is_momentary,
                        self._is_mouse_button, self._mouse_button, self._is_scroll_access,
                        self._cmdpal_mouse_layer_preference_index, cmdpal_weights,
                        0.0, self._position_y,
                    )
                    custom_columns.append((missing > 0.0).astype(np.float32).reshape(-1, 1))
                else:
                    custom_columns.append(np.ones((len(batch), 1), dtype=np.float32))
            elif name == "access_hold_target_position_conflict":
                custom_columns.append(_access_hold_target_position_conflict_batch(
                    batch, self._position_layers, self._access_targets,
                    self._access_is_momentary, self._is_scroll_access,
                    self._target_position_by_source,
                ))
            elif name == "zoom_controls_present_together":
                custom_columns.append(_shortcut_pair_together_violation_batch(
                    batch, self._position_layers, self._zoom_in_sid, self._zoom_out_sid,
                ))
            elif name == "monitor_move_pair_together":
                custom_columns.append(_shortcut_pair_together_violation_batch(
                    batch, self._position_layers, self._monitor_left_sid, self._monitor_right_sid,
                ))
            elif name == "l5_hold_to_self_toggle_path":
                custom_columns.append(_l5_hold_self_toggle_path_violation_batch(
                    batch, self._position_layers, self._l5_hold_sid, self._l5_toggle_sid,
                ))
        custom = np.concatenate(custom_columns, axis=1) if custom_columns else np.zeros((len(batch), 0), dtype=np.float32)
        values = np.asarray(constraints, dtype=np.float32)
        if values.ndim == 1:
            return np.concatenate((values, custom[0]))
        return np.concatenate((values, custom), axis=1)

    @property
    def n_constraints(self) -> int:
        return len(self.hard_constraints)

    def evaluate(self, genome: np.ndarray):
        """Return (objectives, constraints) for a single genome."""
        genome = np.asarray(genome, dtype=np.int32)
        objectives, constraints = _single_genome(genome, *self.arrays)
        objectives = self._add_semantic_position_score(objectives, genome)[0]
        return objectives, self._append_custom_constraints(genome, constraints)

    def evaluate_batch(self, genomes: np.ndarray):
        """Return ((batch, 3) objectives, (batch, n_constr) constraints)."""
        genomes = np.asarray(genomes, dtype=np.int32)
        if genomes.ndim == 1:
            genomes = genomes.reshape(1, -1)
        if self.require_cuda and not self._use_cuda:
            raise RuntimeError(
                "CUDA exact evaluation is required but this model is CPU-only."
            )
        if self._use_cuda:
            objectives, constraints = evaluate_batch_cuda(genomes, self.arrays, cuda_args=self._cuda_args)
        else:
            objectives, constraints = _evaluate_batch(genomes, *self.arrays)
        objectives = self._add_semantic_position_score(objectives, genomes)
        return objectives, self._append_custom_constraints(genomes, constraints)

    def validate_parity(
        self,
        evaluator=None,
        n: int = 64,
        tolerance: float = 1e-4,
    ) -> ParityResult:
        """Check parity against a reference Python evaluator.

        If `evaluator` is None, parity is checked against the seed genome only
        and simply verifies the kernel runs without error.
        """
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
        compiled, _ = self.evaluate_batch(batch)

        if evaluator is None:
            return ParityResult(True, 0.0, "no reference evaluator provided; kernel executed successfully")

        oracle = np.asarray([
            evaluator.evaluate(self.layout.clone_with(genome=genome)).objectives
            for genome in batch
        ], dtype=np.float32)
        max_diff = float(np.max(np.abs(compiled - oracle)))
        ok = bool(np.allclose(compiled, oracle, atol=tolerance, rtol=1e-5))
        return ParityResult(ok, max_diff, "ok" if ok else f"max diff {max_diff:.6g} exceeds tolerance")
