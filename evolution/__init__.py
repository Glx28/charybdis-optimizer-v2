"""pymoo-based evolution engine with custom operators."""
import numpy as np
import random
from typing import Optional, Tuple
from collections import defaultdict
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair

from core import Layout, FitnessResult
from fitness.evaluator import FitnessEvaluator
from evolution.surrogate import LayoutSurrogate, SurrogateManager
from fitness.factors.violation import KEY_GROUPS, shortcut_matches_group


def build_group_placements(layout):
    """Return precomputed group data for both atomic groups.

    Returns a list of (sid_tuple, anchor_positions_list) — one entry per group.
    sid_tuple: ordered sids belonging to the group.
    anchor_positions_list: list of position-index lists, each a valid placement
        of the whole group in the correct shape on some mutable non-L7 layer.

    Used by SwapMutation to drive _overwrite_group_as_unit and by
    generate_random_layouts to seed initial genomes with groups in valid shape.
    """
    arrow_by_type = {}
    arrow_base = {"LEFTARROW": 1, "UPARROW": 2, "DOWNARROW": 3, "RIGHTARROW": 4}
    completion_by_order = {}
    completion_order_map = {
        "DASH AND UNDERSCORE": 1,
        "EQUALS AND PLUS": 2,
        "GRAVE ACCENT AND TILDE": 3,
        "RIGHT BRACE": 4,
        "BACKSLASH AND PIPE": 5,
    }
    for shortcut in layout.shortcuts:
        base = (shortcut.base_key or "").upper()
        if not shortcut.modifiers and base in arrow_base:
            arrow_by_type.setdefault(arrow_base[base], shortcut.sid)
        if not shortcut.modifiers and not shortcut.is_l0_only and base in completion_order_map:
            completion_by_order.setdefault(completion_order_map[base], shortcut.sid)

    pos_lookup = {
        (p.layer, round(p.x), round(p.y)): p.gene_idx
        for p in layout.positions
        if not p.is_frozen and p.layer != 7
    }

    def anchors_for_offsets(offsets):
        result = []
        for p in layout.positions:
            if p.is_frozen or p.layer == 7:
                continue
            ax, ay = round(p.x), round(p.y)
            target = {}
            valid = True
            for order, (dx, dy) in offsets.items():
                idx = pos_lookup.get((p.layer, ax + dx, ay + dy))
                if idx is None:
                    valid = False
                    break
                target[order] = idx
            if valid:
                result.append(target)
        return result

    groups = []

    if len(arrow_by_type) == 4:
        # Two valid arrow shapes; LEFT is type-1 anchor.
        arrow_shapes = [
            {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0)},   # same row
            {1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (2, 1)},   # T-cluster
        ]
        anchors = []
        for shape in arrow_shapes:
            anchors.extend(anchors_for_offsets(shape))
        if anchors:
            sid_tuple = tuple(arrow_by_type[i] for i in (1, 2, 3, 4))
            anchor_list = [[a[i] for i in (1, 2, 3, 4)] for a in anchors]
            groups.append((sid_tuple, anchor_list))

    if len(completion_by_order) == 5:
        # Norwegian extra-key group: fixed shape, EQUALS is anchor.
        completion_offsets = {1: (-1, 0), 2: (0, 0), 3: (-2, 0), 4: (-2, 1), 5: (-2, 3)}
        anchors = anchors_for_offsets(completion_offsets)
        if anchors:
            sid_tuple = tuple(completion_by_order[i] for i in range(1, 6))
            anchor_list = [[a[i] for i in range(1, 6)] for a in anchors]
            groups.append((sid_tuple, anchor_list))

    return groups


NUMBA_AVAILABLE = False

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    prange = None


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _cycle_crossover_pair_numba(p1, p2, n_shortcuts):
        n = len(p1)
        c1 = np.full(n, -1, dtype=np.int32)
        c2 = np.full(n, -1, dtype=np.int32)
        start_idx = 0
        while start_idx < n and (p1[start_idx] < 0 or p2[start_idx] < 0):
            start_idx += 1
        if start_idx >= n:
            return p1.astype(np.int32).copy(), p2.astype(np.int32).copy()
        p2_pos = np.full(n_shortcuts, -1, dtype=np.int32)
        for i in range(n):
            sid = p2[i]
            if sid >= 0:
                p2_pos[sid] = i
        cycle = np.zeros(n, dtype=np.bool_)
        idx = start_idx
        while True:
            cycle[idx] = True
            val = p1[idx]
            next_idx = -1
            if val >= 0 and val < n_shortcuts:
                next_idx = p2_pos[val]
            if next_idx < 0 or cycle[next_idx] or next_idx == start_idx:
                break
            idx = next_idx
        for i in range(n):
            if cycle[i]:
                c1[i] = p1[i]
                c2[i] = p2[i]
            else:
                c1[i] = p2[i]
                c2[i] = p1[i]
        return c1, c2

    @njit(parallel=True, cache=True)
    def _cycle_crossover_batch_numba(children, half, crossover_prob, n_shortcuts):
        """Parallel batch cycle crossover — each pair runs on its own Numba thread."""
        for i in prange(half):
            if np.random.random() < crossover_prob:
                p1 = children[i].copy()
                p2 = children[i + half].copy()
                c1, c2 = _cycle_crossover_pair_numba(p1, p2, n_shortcuts)
                children[i] = c1
                children[i + half] = c2
        return children

    @njit(parallel=True, cache=True)
    def _sanitize_batch_numba(X, frozen_idx, frozen_vals, mutable_idx, frozen_sid_lut, n_shortcuts):
        n = X.shape[0]
        n_frozen = frozen_idx.shape[0]
        n_mutable = mutable_idx.shape[0]
        for i in prange(n):
            # Clip invalid SID values
            for j in range(X.shape[1]):
                v = X[i, j]
                if v >= n_shortcuts or v < -1:
                    X[i, j] = -1
            # Restore frozen positions
            for k in range(n_frozen):
                X[i, frozen_idx[k]] = frozen_vals[k]
            # Zero-out frozen SIDs that ended up in mutable positions
            for k in range(n_mutable):
                pos = mutable_idx[k]
                sid = X[i, pos]
                if sid >= 0 and frozen_sid_lut[sid]:
                    X[i, pos] = -1

    # ------------------------------------------------------------------
    # Numba-accelerated simple mutation proposals for SwapMutation._do
    # ------------------------------------------------------------------

    @njit(cache=True)
    def _rng_step(state):
        """SplitMix64-style RNG step; mutates state array in place."""
        state[0] += np.uint64(0x9e3779b97f4a7c15)
        z = state[0]
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        z = z ^ (z >> np.uint64(31))
        state[0] = z
        return z

    @njit(cache=True)
    def _rand_float(state):
        return float(_rng_step(state) & np.uint64(0x7fffffffffffffff)) / float(np.uint64(0x7fffffffffffffff))

    @njit(cache=True)
    def _rand_int(state, n):
        return int(_rng_step(state) % np.uint64(n))

    @njit(cache=True)
    def _contains(arr, val):
        for i in range(len(arr)):
            if arr[i] == val:
                return True
        return False

    @njit(cache=True)
    def _numba_random_reassign_one(genome, state, mutable_arr, pos_layer_arr, assignable_arr,
                                   is_group_sid_lut, is_important_sid_lut, access_target_lut,
                                   mo_access_target_lut, n_shortcuts, toggle_access_sids_arr,
                                   return_toggle_sid, layer_mutable_flat, layer_mutable_start):
        n_mut = len(mutable_arr)
        if n_mut == 0 or len(assignable_arr) == 0:
            return False

        mutable_sids = genome[mutable_arr]
        counts = np.zeros(n_shortcuts, dtype=np.int32)
        for k in range(n_mut):
            sid = mutable_sids[k]
            if sid >= 0 and sid < n_shortcuts:
                counts[sid] += 1

        n_candidates = 0
        cand_positions = np.empty(n_mut, dtype=np.int32)
        for k in range(n_mut):
            sid = mutable_sids[k]
            pos = mutable_arr[k]
            if sid < 0 or sid >= n_shortcuts:
                continue
            if is_group_sid_lut[sid]:
                continue
            if is_important_sid_lut[sid] and counts[sid] <= 1:
                continue
            cand_positions[n_candidates] = pos
            n_candidates += 1

        if n_candidates == 0:
            return False

        chosen_pos = -1
        for _ in range(5):
            idx = _rand_int(state, n_candidates)
            pos = cand_positions[idx]
            sid = genome[pos]
            if sid < 0 or sid >= n_shortcuts:
                chosen_pos = pos
                break
            target = access_target_lut[sid]
            if target <= 0:
                chosen_pos = pos
                break
            broken = True
            for k in range(n_mut):
                if mutable_arr[k] == pos:
                    continue
                other_sid = genome[mutable_arr[k]]
                if other_sid >= 0 and other_sid < n_shortcuts and access_target_lut[other_sid] == target:
                    broken = False
                    break
            if not broken:
                chosen_pos = pos
                break

        if chosen_pos < 0:
            return False

        pos_layer = pos_layer_arr[chosen_pos]
        n_valid_assign = 0
        valid_assign = np.empty(len(assignable_arr), dtype=np.int32)
        for k in range(len(assignable_arr)):
            sid = assignable_arr[k]
            mo_tgt = mo_access_target_lut[sid]
            if mo_tgt >= 0 and mo_tgt == pos_layer:
                continue
            valid_assign[n_valid_assign] = sid
            n_valid_assign += 1
        if n_valid_assign == 0:
            valid_assign = assignable_arr
            n_valid_assign = len(assignable_arr)

        new_sid = valid_assign[_rand_int(state, n_valid_assign)]
        genome[chosen_pos] = new_sid

        if _contains(toggle_access_sids_arr, new_sid):
            target_layer = access_target_lut[new_sid]
            if target_layer > 0 and return_toggle_sid >= 0:
                n_layers = len(layer_mutable_start)
                start = layer_mutable_start[target_layer]
                end = layer_mutable_start[target_layer + 1] if target_layer + 1 < n_layers else len(layer_mutable_flat)
                already = False
                for p in range(start, end):
                    pos = layer_mutable_flat[p]
                    if genome[pos] == return_toggle_sid:
                        already = True
                        break
                if not already:
                    n_empty = 0
                    n_any = 0
                    empty_positions = np.empty(end - start, dtype=np.int32)
                    any_positions = np.empty(end - start, dtype=np.int32)
                    for p in range(start, end):
                        pos = layer_mutable_flat[p]
                        any_positions[n_any] = pos
                        n_any += 1
                        if genome[pos] < 0:
                            empty_positions[n_empty] = pos
                            n_empty += 1
                    if n_any > 0:
                        if n_empty > 0:
                            pool = empty_positions
                            n_pool = n_empty
                        else:
                            pool = any_positions
                            n_pool = n_any
                        genome[pool[_rand_int(state, n_pool)]] = return_toggle_sid
        return True

    @njit(cache=True)
    def _numba_detect_mouse_layer(genome, mutable_arr, pos_layer_arr, pos_hand_arr, pos_is_thumb_arr, mouse_button_sids):
        n_mut = len(mutable_arr)
        if mouse_button_sids[1] < 0:
            return -1
        layer_counts = np.zeros(32, dtype=np.int32)
        for k in range(n_mut):
            pos = mutable_arr[k]
            sid = genome[pos]
            if sid < 0:
                continue
            is_mouse = False
            for b in range(1, 6):
                if mouse_button_sids[b] == sid:
                    is_mouse = True
                    break
            if is_mouse:
                lyr = pos_layer_arr[pos]
                if lyr > 0 and lyr != 7 and lyr < 32 and pos_hand_arr[pos] == 1 and not pos_is_thumb_arr[pos]:
                    layer_counts[lyr] += 1
        best = 0
        best_layer = -1
        for lyr in range(32):
            if layer_counts[lyr] > best:
                best = layer_counts[lyr]
                best_layer = lyr
        if best < 2:
            return -1
        return best_layer

    @njit(cache=True)
    def _numba_mouse_ideal_penalty(button, x, y):
        # Mirrors fitness/kernel.py's MB1/MB2/MB3 ideal-position weights
        # (dynamic_mouse_layer). Used only to break effort-ties in
        # _numba_effort_swap -- MB1 and MB3 (and MB2) can share effort=0.0,
        # which leaves the effort-only swap criterion with zero gradient to
        # ever notice one is squatting on another's ideal slot.
        if button == 1:
            return abs(x - 8.0) * 28000.0 + abs(y - 2.0) * 30000.0
        if button == 2:
            return abs(x - 9.0) * 32000.0 + abs(y - 2.0) * 30000.0
        if button == 3:
            return abs(x - 11.0) * 6000.0 + abs(y - 2.0) * 5000.0
        return -1.0

    @njit(cache=True)
    def _numba_layer_sid_count(genome, layer_mutable_flat, layer_mutable_start, layer, sid, exclude_a, exclude_b):
        # Count how many mutable positions on `layer` already hold `sid`,
        # excluding the two positions involved in the swap under consideration
        # (their current contents are about to move elsewhere). Scoped to just
        # this layer's mutable positions via the precomputed per-layer index
        # (O(positions-on-layer), not O(n_pos)) — frozen positions are excluded
        # since they never change and each frozen sid occupies exactly one
        # fixed slot, so they carry no duplicate risk for this check.
        n_layers = len(layer_mutable_start)
        if layer < 0 or layer + 1 >= n_layers:
            return 0
        start = layer_mutable_start[layer]
        end = layer_mutable_start[layer + 1]
        count = 0
        for p in range(start, end):
            i = layer_mutable_flat[p]
            if i == exclude_a or i == exclude_b:
                continue
            if genome[i] == sid:
                count += 1
        return count

    @njit(cache=True)
    def _numba_swap_creates_illegal_duplicate(genome, pos_layer_arr, src_pos, target_pos, mouse_layer, mouse_button_sids,
                                              layer_mutable_flat, layer_mutable_start):
        # No shortcut may appear more than once on the same layer, except one
        # left+one right copy of a core mouse button on the dynamic mouse
        # layer (layer 7 is frozen and fully excluded). This is a mutation-
        # time heuristic guard using the swap operator's own simplified mouse-
        # layer detector as a proxy — the fitness kernel's same_layer_duplicate
        # hard constraint is the final, precise arbiter regardless.
        moving_sid = genome[src_pos]
        target_layer = pos_layer_arr[target_pos]
        if target_layer != 7:
            cap = 1
            if mouse_layer >= 0 and target_layer == mouse_layer:
                for b in range(1, 6):
                    if mouse_button_sids[b] == moving_sid:
                        cap = 2
                        break
            existing = _numba_layer_sid_count(genome, layer_mutable_flat, layer_mutable_start, target_layer, moving_sid, src_pos, target_pos)
            if existing + 1 > cap:
                return True
        displaced_sid = genome[target_pos]
        src_layer = pos_layer_arr[src_pos]
        if src_layer != 7:
            cap = 1
            if mouse_layer >= 0 and src_layer == mouse_layer:
                for b in range(1, 6):
                    if mouse_button_sids[b] == displaced_sid:
                        cap = 2
                        break
            existing = _numba_layer_sid_count(genome, layer_mutable_flat, layer_mutable_start, src_layer, displaced_sid, src_pos, target_pos)
            if existing + 1 > cap:
                return True
        return False

    @njit(cache=True)
    def _numba_effort_swap(genome, state, mutable_arr, pos_layer_arr, pos_hand_arr, pos_is_thumb_arr,
                           pos_effort_arr, pos_x_arr, pos_y_arr, sid_importance_arr, is_group_sid_lut, mouse_button_sids,
                           n_shortcuts, layer_mutable_flat, layer_mutable_start):
        n_mut = len(mutable_arr)
        if n_mut < 2 or len(sid_importance_arr) == 0:
            return False

        mouse_layer = _numba_detect_mouse_layer(genome, mutable_arr, pos_layer_arr, pos_hand_arr, pos_is_thumb_arr, mouse_button_sids)

        valid_positions = np.empty(n_mut, dtype=np.int32)
        valid_costs = np.empty(n_mut, dtype=np.float32)
        valid_mouse_button = np.full(n_mut, -1, dtype=np.int32)
        n_valid = 0
        for k in range(n_mut):
            pos = mutable_arr[k]
            sid = genome[pos]
            if sid < 0 or sid >= n_shortcuts:
                continue
            if is_group_sid_lut[sid]:
                continue
            imp = sid_importance_arr[sid]
            button = -1
            if mouse_layer >= 0:
                for b in range(1, 6):
                    if mouse_button_sids[b] == sid:
                        button = b
                        break
                if button > 0 and pos_layer_arr[pos] == mouse_layer:
                    imp *= 3.0
            valid_positions[n_valid] = pos
            cost = pos_effort_arr[pos] * imp
            # MB1/MB2/MB3 can sit at effort=0.0 while badly misplaced relative
            # to their ideal coordinate (e.g. squatting on another mouse
            # button's ideal slot) -- effort*importance alone is then always
            # 0, so this position could never even be considered as a swap
            # source. Add its ideal-position penalty so misplacement is
            # visible to the same top-k selection used for everything else.
            if button >= 1 and button <= 3 and pos_layer_arr[pos] == mouse_layer:
                cost += _numba_mouse_ideal_penalty(button, pos_x_arr[pos], pos_y_arr[pos])
                valid_mouse_button[n_valid] = button
            valid_costs[n_valid] = cost
            n_valid += 1

        if n_valid == 0:
            return False

        top_k = min(5, n_valid)
        order = np.argsort(valid_costs[:n_valid])
        top_idx = order[-top_k:]
        chosen_i = top_idx[_rand_int(state, top_k)]
        src_pos = valid_positions[chosen_i]
        src_effort = pos_effort_arr[src_pos]
        src_button = valid_mouse_button[chosen_i]
        if src_button < 1 and src_effort <= 0.0:
            return False

        n_lower = 0
        lower_positions = np.empty(n_valid, dtype=np.int32)
        for k in range(n_valid):
            if k == chosen_i:
                continue
            if pos_effort_arr[valid_positions[k]] < src_effort:
                lower_positions[n_lower] = valid_positions[k]
                n_lower += 1

        if n_lower == 0 and src_button >= 1:
            # Effort-tie fallback, mouse buttons only: no strictly-lower-effort
            # target exists (src is often already at effort=0.0, the
            # minimum), so look for an equal-effort position whose swap would
            # reduce this button's ideal-position penalty instead. Without
            # this, a misplaced mouse button stuck at effort=0.0 can never be
            # fixed by this operator no matter how many generations run.
            src_penalty = _numba_mouse_ideal_penalty(src_button, pos_x_arr[src_pos], pos_y_arr[src_pos])
            for k in range(n_valid):
                if k == chosen_i:
                    continue
                cand_pos = valid_positions[k]
                if pos_effort_arr[cand_pos] != src_effort:
                    continue
                cand_penalty = _numba_mouse_ideal_penalty(src_button, pos_x_arr[cand_pos], pos_y_arr[cand_pos])
                if cand_penalty < src_penalty:
                    lower_positions[n_lower] = cand_pos
                    n_lower += 1
        elif n_lower == 0:
            return False

        if n_lower == 0:
            return False

        src_layer = pos_layer_arr[src_pos]
        n_same = 0
        same_layer_positions = np.empty(n_lower, dtype=np.int32)
        for k in range(n_lower):
            if pos_layer_arr[lower_positions[k]] == src_layer:
                same_layer_positions[n_same] = lower_positions[k]
                n_same += 1

        # Try a bounded number of candidate targets (respecting the same-layer
        # preference each attempt) and skip any that would create an illegal
        # same-layer duplicate (see _numba_swap_creates_illegal_duplicate).
        target_pos = -1
        for _attempt in range(8):
            if n_same > 0 and _rand_float(state) < 0.75:
                candidate = same_layer_positions[_rand_int(state, n_same)]
            else:
                candidate = lower_positions[_rand_int(state, n_lower)]
            if not _numba_swap_creates_illegal_duplicate(genome, pos_layer_arr, src_pos, candidate, mouse_layer, mouse_button_sids,
                                                          layer_mutable_flat, layer_mutable_start):
                target_pos = candidate
                break
        if target_pos < 0:
            return False

        tmp = genome[src_pos]
        genome[src_pos] = genome[target_pos]
        genome[target_pos] = tmp
        return True

    @njit(cache=True)
    def _numba_smart_duplicate(genome, state, mutable_arr, pos_effort_arr, pos_layer_arr, dup_candidate_arr,
                               dup_exp_w, frozen_sid_counts, n_shortcuts):
        n_cand = len(dup_candidate_arr)
        if n_cand == 0:
            return False

        n_mut = len(mutable_arr)
        empty_positions = np.empty(n_mut, dtype=np.int32)
        n_empty = 0
        for k in range(n_mut):
            pos = mutable_arr[k]
            if genome[pos] < 0:
                empty_positions[n_empty] = pos
                n_empty += 1
        if n_empty == 0:
            return False

        efforts = np.empty(n_empty, dtype=np.float32)
        for k in range(n_empty):
            efforts[k] = pos_effort_arr[empty_positions[k]]
        top_n = max(1, n_empty // 3)
        order = np.argsort(efforts)
        target_idx = order[_rand_int(state, top_n)]
        target_pos = empty_positions[target_idx]
        target_layer = pos_layer_arr[target_pos]

        counts = np.zeros(n_shortcuts, dtype=np.int32)
        # No shortcut may appear more than once on the same layer (L7 is
        # frozen/excluded; mouse buttons never reach this function since
        # dup_candidate_arr excludes them entirely — mouse placement is
        # handled by the dedicated mouse-workflow proposal operator).
        layer_sid_count = np.zeros(n_shortcuts, dtype=np.int32)
        for k in range(n_mut):
            sid = genome[mutable_arr[k]]
            if sid >= 0 and sid < n_shortcuts:
                counts[sid] += 1
                if target_layer != 7 and pos_layer_arr[mutable_arr[k]] == target_layer:
                    layer_sid_count[sid] += 1

        n_frozen = len(frozen_sid_counts)
        weights = np.empty(n_cand, dtype=np.float32)
        total_w = 0.0
        for k in range(n_cand):
            sid = dup_candidate_arr[k]
            if target_layer != 7 and layer_sid_count[sid] >= 1:
                weights[k] = 0.0
                continue
            cnt = counts[sid]
            if n_frozen > 0 and sid < n_frozen:
                cnt += frozen_sid_counts[sid]
            count_discount = 1.0 + float(cnt)
            w = dup_exp_w[k] / (count_discount * count_discount)
            weights[k] = w
            total_w += w

        if total_w <= 0.0:
            return False

        r = _rand_float(state) * total_w
        cum = 0.0
        chosen = dup_candidate_arr[0]
        for k in range(n_cand):
            cum += weights[k]
            if cum >= r:
                chosen = dup_candidate_arr[k]
                break

        genome[target_pos] = chosen
        return True

    @njit(cache=True)
    def _numba_bias_toggle_to_own_layer(genome, state, pos_layer_arr, access_target_lut, access_is_mo_lut,
                                        layer_mutable_flat, layer_mutable_start, n_shortcuts):
        n_pos = len(genome)
        cand_positions = np.empty(n_pos, dtype=np.int32)
        cand_layers = np.empty(n_pos, dtype=np.int32)
        n_cand = 0
        for pos in range(n_pos):
            sid = genome[pos]
            if sid < 0 or sid >= n_shortcuts:
                continue
            tgt = access_target_lut[sid]
            if tgt <= 0 or tgt == 7:
                continue
            if access_is_mo_lut[sid]:
                continue
            if pos_layer_arr[pos] == tgt:
                continue
            cand_positions[n_cand] = pos
            cand_layers[n_cand] = tgt
            n_cand += 1
        if n_cand == 0:
            return False

        idx = _rand_int(state, n_cand)
        src_pos = cand_positions[idx]
        target_layer = cand_layers[idx]

        n_layers = len(layer_mutable_start)
        start = layer_mutable_start[target_layer]
        end = layer_mutable_start[target_layer + 1] if target_layer + 1 < n_layers else len(layer_mutable_flat)
        if start >= end:
            return False

        n_empty = 0
        n_any = 0
        empty_positions = np.empty(end - start, dtype=np.int32)
        any_positions = np.empty(end - start, dtype=np.int32)
        for p in range(start, end):
            pos = layer_mutable_flat[p]
            if pos == src_pos:
                continue
            any_positions[n_any] = pos
            n_any += 1
            if genome[pos] < 0:
                empty_positions[n_empty] = pos
                n_empty += 1

        if n_any == 0:
            return False
        if n_empty > 0:
            tgt_pos = empty_positions[_rand_int(state, n_empty)]
        else:
            tgt_pos = any_positions[_rand_int(state, n_any)]

        tmp = genome[src_pos]
        genome[src_pos] = genome[tgt_pos]
        genome[tgt_pos] = tmp
        return True

    @njit(cache=True)
    def _numba_bias_access_to_thumb(genome, state, mutable_arr, pos_layer_arr, pos_is_thumb_arr,
                                    access_target_lut, is_group_sid_lut, n_shortcuts):
        n_mut = len(mutable_arr)
        if n_mut == 0:
            return False

        cand_positions = np.empty(n_mut, dtype=np.int32)
        n_cand = 0
        for k in range(n_mut):
            pos = mutable_arr[k]
            sid = genome[pos]
            if sid < 0 or sid >= n_shortcuts:
                continue
            if access_target_lut[sid] <= 0:
                continue
            if pos_is_thumb_arr[pos]:
                continue
            cand_positions[n_cand] = pos
            n_cand += 1
        if n_cand == 0:
            return False

        src_pos = cand_positions[_rand_int(state, n_cand)]
        src_layer = pos_layer_arr[src_pos]

        thumb_positions = np.empty(n_mut, dtype=np.int32)
        n_thumb = 0
        for k in range(n_mut):
            pos = mutable_arr[k]
            if pos == src_pos:
                continue
            if not pos_is_thumb_arr[pos]:
                continue
            sid = genome[pos]
            if sid >= 0 and sid < n_shortcuts and is_group_sid_lut[sid]:
                continue
            thumb_positions[n_thumb] = pos
            n_thumb += 1
        if n_thumb == 0:
            return False

        n_same = 0
        same_layer = np.empty(n_thumb, dtype=np.int32)
        for k in range(n_thumb):
            if pos_layer_arr[thumb_positions[k]] == src_layer:
                same_layer[n_same] = thumb_positions[k]
                n_same += 1

        if n_same > 0:
            tgt_pos = same_layer[_rand_int(state, n_same)]
        else:
            tgt_pos = thumb_positions[_rand_int(state, n_thumb)]

        tmp = genome[src_pos]
        genome[src_pos] = genome[tgt_pos]
        genome[tgt_pos] = tmp
        return True

    @njit(cache=True)
    def _numba_repair_return_toggles(genome, state, pos_layer_arr, access_target_lut, access_is_mo_lut,
                                     layer_mutable_flat, layer_mutable_start, pos_is_thumb_arr,
                                     return_toggle_sid, n_shortcuts):
        if return_toggle_sid < 0:
            return False

        n_pos = len(genome)
        toggle_to = np.zeros(32, dtype=np.bool_)
        has_return = np.zeros(32, dtype=np.bool_)
        for pos in range(n_pos):
            sid = genome[pos]
            if sid < 0 or sid >= n_shortcuts:
                continue
            tgt = access_target_lut[sid]
            if tgt <= 0:
                continue
            if access_is_mo_lut[sid]:
                continue
            lyr = pos_layer_arr[pos]
            if tgt != 0:
                toggle_to[lyr] = True
            else:
                has_return[lyr] = True

        missing = np.empty(32, dtype=np.int32)
        n_missing = 0
        n_layers = len(layer_mutable_start)
        for lyr in range(32):
            if toggle_to[lyr] and not has_return[lyr]:
                start = layer_mutable_start[lyr]
                end = layer_mutable_start[lyr + 1] if lyr + 1 < n_layers else len(layer_mutable_flat)
                if start < end:
                    missing[n_missing] = lyr
                    n_missing += 1
        if n_missing == 0:
            return False

        lyr = missing[_rand_int(state, n_missing)]
        start = layer_mutable_start[lyr]
        end = layer_mutable_start[lyr + 1] if lyr + 1 < n_layers else len(layer_mutable_flat)

        n_te = 0
        n_ae = 0
        n_ap = 0
        thumb_empty = np.empty(end - start, dtype=np.int32)
        any_empty = np.empty(end - start, dtype=np.int32)
        any_positions = np.empty(end - start, dtype=np.int32)
        for p in range(start, end):
            pos = layer_mutable_flat[p]
            any_positions[n_ap] = pos
            n_ap += 1
            if genome[pos] < 0:
                any_empty[n_ae] = pos
                n_ae += 1
                if pos_is_thumb_arr[pos]:
                    thumb_empty[n_te] = pos
                    n_te += 1

        if n_te > 0:
            pool = thumb_empty
            n_pool = n_te
        elif n_ae > 0:
            pool = any_empty
            n_pool = n_ae
        else:
            pool = any_positions
            n_pool = n_ap

        genome[pool[_rand_int(state, n_pool)]] = return_toggle_sid
        return True

    @njit(parallel=True, cache=True)
    def _mutate_batch_numba(X, handled, probs, seeds,
                            mutable_arr, pos_layer_arr, pos_hand_arr, pos_is_thumb_arr, pos_effort_arr,
                            pos_x_arr, pos_y_arr,
                            sid_importance_arr, access_target_lut, access_is_mo_lut, mo_access_target_lut,
                            is_group_sid_lut, is_important_sid_lut,
                            return_toggle_sid,
                            dup_candidate_arr, dup_exp_w, frozen_sid_counts,
                            assignable_arr,
                            layer_mutable_flat, layer_mutable_start,
                            mouse_button_sids,
                            toggle_access_sids_arr,
                            n_shortcuts):
        n = X.shape[0]
        for i in prange(n):
            if handled[i]:
                continue
            state = np.empty(1, dtype=np.uint64)
            state[0] = seeds[i]

            if _rand_float(state) < probs[0]:
                if _numba_random_reassign_one(X[i], state, mutable_arr, pos_layer_arr, assignable_arr,
                                              is_group_sid_lut, is_important_sid_lut, access_target_lut,
                                              mo_access_target_lut, n_shortcuts, toggle_access_sids_arr,
                                              return_toggle_sid, layer_mutable_flat, layer_mutable_start):
                    handled[i] = True
                    continue

            if _rand_float(state) < probs[1]:
                if _numba_effort_swap(X[i], state, mutable_arr, pos_layer_arr, pos_hand_arr, pos_is_thumb_arr,
                                      pos_effort_arr, pos_x_arr, pos_y_arr, sid_importance_arr, is_group_sid_lut, mouse_button_sids,
                                      n_shortcuts, layer_mutable_flat, layer_mutable_start):
                    handled[i] = True
                    continue

            if _rand_float(state) < probs[2]:
                if _numba_smart_duplicate(X[i], state, mutable_arr, pos_effort_arr, pos_layer_arr, dup_candidate_arr,
                                          dup_exp_w, frozen_sid_counts, n_shortcuts):
                    handled[i] = True
                    continue

            if _rand_float(state) < probs[3]:
                if _numba_bias_toggle_to_own_layer(X[i], state, pos_layer_arr, access_target_lut, access_is_mo_lut,
                                                   layer_mutable_flat, layer_mutable_start, n_shortcuts):
                    handled[i] = True
                    continue

            if _rand_float(state) < probs[4]:
                if _numba_bias_access_to_thumb(X[i], state, mutable_arr, pos_layer_arr, pos_is_thumb_arr,
                                               access_target_lut, is_group_sid_lut, n_shortcuts):
                    handled[i] = True
                    continue

            if _rand_float(state) < probs[5]:
                if _numba_repair_return_toggles(X[i], state, pos_layer_arr, access_target_lut, access_is_mo_lut,
                                                layer_mutable_flat, layer_mutable_start, pos_is_thumb_arr,
                                                return_toggle_sid, n_shortcuts):
                    handled[i] = True
                    continue


    # ------------------------------------------------------------------
    # Numba helpers for semantic mutations (Pass 1/1b in SwapMutation._do)
    # ------------------------------------------------------------------

@njit(cache=True)
def _numba_place_sids(genome, sids, target_positions, pos_map):
    """Numba equivalent of SwapMutation._place_sids.

    Moves ``sids`` to ``target_positions`` by swapping displaced values back
    into the sids' old slots.  Returns True on success.
    """
    n = len(sids)
    if n != len(target_positions):
        return False

    current_positions = np.empty(n, dtype=np.int32)
    for k in range(n):
        sid = sids[k]
        if sid < 0 or sid >= len(pos_map):
            return False
        p = pos_map[sid]
        if p < 0:
            return False
        current_positions[k] = p

    displaced = np.empty(n, dtype=np.int32)
    for k in range(n):
        displaced[k] = genome[target_positions[k]]

    for k in range(n):
        genome[target_positions[k]] = sids[k]

    fill_positions = np.empty(n, dtype=np.int32)
    n_fill = 0
    for k in range(n):
        p = current_positions[k]
        found = False
        for t in range(n):
            if target_positions[t] == p:
                found = True
                break
        if not found:
            fill_positions[n_fill] = p
            n_fill += 1

    n_filled = 0
    for k in range(n):
        s = displaced[k]
        is_member = False
        for m in range(n):
            if s == sids[m]:
                is_member = True
                break
        if not is_member and n_filled < n_fill:
            genome[fill_positions[n_filled]] = s
            n_filled += 1

    while n_filled < n_fill:
        genome[fill_positions[n_filled]] = -1
        n_filled += 1

    return True


@njit(cache=True)
def _numba_contains_int(arr, n, val):
    for i in range(n):
        if arr[i] == val:
            return True
    return False


@njit(cache=True)
def _numba_insertion_sort(arr, n):
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


@njit(cache=True)
def _numba_sample_distinct(pool, n_take, state):
    n = len(pool)
    if n_take > n:
        n_take = n
    if n_take <= 0:
        return np.empty(0, dtype=np.int32)
    tmp = np.empty(n, dtype=np.int32)
    for i in range(n):
        tmp[i] = pool[i]
    for i in range(n_take):
        j = i + _rand_int(state, n - i)
        tmp_i = tmp[i]
        tmp[i] = tmp[j]
        tmp[j] = tmp_i
    out = np.empty(n_take, dtype=np.int32)
    for i in range(n_take):
        out[i] = tmp[i]
    return out


@njit(cache=True)
def _numba_thumb_exclude_mask(genome, mutable_arr, mutable_layer, mutable_hand,
                               mutable_is_thumb, access_target_lut, access_is_mo_lut,
                               n_shortcuts):
    n_mut = len(mutable_arr)
    thumb_exclude = np.zeros(n_mut, dtype=np.bool_)
    has_toggle = np.zeros(32, dtype=np.bool_)
    mo_count = np.zeros((32, 2), dtype=np.int32)
    for k in range(n_mut):
        sid = genome[mutable_arr[k]]
        if sid < 0 or sid >= n_shortcuts:
            continue
        tgt = access_target_lut[sid]
        if tgt <= 0 or tgt >= 32:
            continue
        if access_is_mo_lut[sid]:
            mo_count[tgt, mutable_hand[k]] += 1
        else:
            has_toggle[tgt] = True
    for k in range(n_mut):
        lyr = mutable_layer[k]
        if lyr <= 0 or lyr >= 32:
            continue
        if has_toggle[lyr]:
            continue
        total_mo = mo_count[lyr, 0] + mo_count[lyr, 1]
        if total_mo != 1:
            continue
        hand = 0 if mo_count[lyr, 0] == 1 else 1
        if mutable_is_thumb[k] and mutable_hand[k] == hand:
            thumb_exclude[k] = True
    return thumb_exclude


@njit(cache=True)
def _numba_effort_sort(positions, efforts, n):
    # Insertion sort `positions` ascending by (effort, position index), so the
    # lowest-effort (best) candidate slots come first, and ties resolve to a
    # fixed, deterministic order (not whatever order the random sample
    # happened to produce) — matching the old raw-index-sort's determinism
    # for equal-effort candidates.
    for i in range(1, n):
        key_pos = positions[i]
        key_eff = efforts[key_pos]
        j = i - 1
        while j >= 0 and (
            efforts[positions[j]] > key_eff
            or (efforts[positions[j]] == key_eff and positions[j] > key_pos)
        ):
            positions[j + 1] = positions[j]
            j -= 1
        positions[j + 1] = key_pos


@njit(cache=True)
def _numba_propose_mouse_workflow_layer(genome, pos_map, thumb_exclude, state,
                                        mouse_candidate_layers,
                                        right_non_thumb_flat, right_non_thumb_start,
                                        right_positions_flat, right_positions_start,
                                        right_thumb_positions_flat, right_thumb_positions_start,
                                        safe_access_positions, l0_safe_access_positions,
                                        layer_access_hold, layer_access_toggle,
                                        layer_scroll_access, return_toggle_sid,
                                        mouse_button_sids, n_shortcuts, pos_effort_arr, pos_x_arr):
    n_candidates = len(mouse_candidate_layers)
    if n_candidates == 0:
        return False
    layer = mouse_candidate_layers[_rand_int(state, n_candidates)]

    start = right_non_thumb_start[layer]
    end = right_non_thumb_start[layer + 1] if layer + 1 < 33 else len(right_non_thumb_flat)
    n_pos = min(12, end - start)
    if n_pos < 6:
        return False

    chosen = _numba_sample_distinct(right_non_thumb_flat[start:start + n_pos], 6, state)
    # Sort the 6 sampled positions by effort (ascending), not raw index, so the
    # lowest-effort slots go to the highest-priority mouse-group members
    # instead of an arbitrary index-order mapping. Priority (matching
    # fitness/kernel.py's button_weight and scroll effort-priority terms):
    # Scroll > MB2 > MB1 > MB3 ~= MB4 ~= MB5.
    _numba_effort_sort(chosen, pos_effort_arr, 6)
    # Momentary Scroll on x=7/x=8 is uncomfortable and fails final acceptance
    # outright (not just a soft scoring preference) — never propose it there.
    # If the best (chosen[0]) slot is x7/x8, swap it with the first later
    # slot that isn't, keeping the rest of the priority order intact.
    if pos_x_arr[chosen[0]] == 7.0 or pos_x_arr[chosen[0]] == 8.0:
        for k in range(1, 6):
            if pos_x_arr[chosen[k]] != 7.0 and pos_x_arr[chosen[k]] != 8.0:
                tmp = chosen[0]
                chosen[0] = chosen[k]
                chosen[k] = tmp
                break

    sids = np.empty(9, dtype=np.int32)
    targets = np.empty(9, dtype=np.int32)
    sids[0] = layer_scroll_access[layer]
    sids[1] = mouse_button_sids[2]
    sids[2] = mouse_button_sids[1]
    sids[3] = mouse_button_sids[3]
    sids[4] = mouse_button_sids[4]
    sids[5] = mouse_button_sids[5]
    for k in range(6):
        targets[k] = chosen[k]
    n_total = 6

    access_pool = l0_safe_access_positions if len(l0_safe_access_positions) >= 2 else safe_access_positions
    n_safe = len(access_pool)
    if n_safe < 2:
        return False

    # Hold access position
    hold_pos = -1
    for _ in range(50):
        idx = _rand_int(state, n_safe)
        pos = access_pool[idx]
        if not _numba_contains_int(targets, n_total, pos):
            hold_pos = pos
            break
    if hold_pos < 0:
        return False
    targets[n_total] = hold_pos
    sids[n_total] = layer_access_hold[layer]
    n_total += 1

    # Toggle access position
    toggle_pos = -1
    for _ in range(50):
        idx = _rand_int(state, n_safe)
        pos = access_pool[idx]
        if pos != hold_pos and not _numba_contains_int(targets, n_total, pos):
            toggle_pos = pos
            break
    if toggle_pos < 0:
        return False
    targets[n_total] = toggle_pos
    sids[n_total] = layer_access_toggle[layer]
    n_total += 1

    # Return-to-L0 toggle on the mouse layer
    if return_toggle_sid >= 0:
        r_start = right_positions_start[layer]
        r_end = right_positions_start[layer + 1] if layer + 1 < 33 else len(right_positions_flat)
        rt_start = right_thumb_positions_start[layer]
        rt_end = right_thumb_positions_start[layer + 1] if layer + 1 < 33 else len(right_thumb_positions_flat)

        pool = np.empty(r_end - r_start, dtype=np.int32)
        n_pool = 0
        # Prefer right thumb positions
        for k in range(rt_start, rt_end):
            pos = right_thumb_positions_flat[k]
            if not _numba_contains_int(targets, n_total, pos):
                pool[n_pool] = pos
                n_pool += 1
        if n_pool == 0:
            for k in range(r_start, r_end):
                pos = right_positions_flat[k]
                if not _numba_contains_int(targets, n_total, pos):
                    pool[n_pool] = pos
                    n_pool += 1
        if n_pool > 0:
            targets[n_total] = pool[_rand_int(state, n_pool)]
            sids[n_total] = return_toggle_sid
            n_total += 1

    return _numba_place_sids(
        genome, sids[:n_total], targets[:n_total], pos_map
    )


@njit(cache=True)
def _numba_propose_l7_access(genome, pos_map, state,
                             l0_safe_access_positions, safe_access_positions,
                             l7_hold_sid, l7_toggle_sid):
    pool = l0_safe_access_positions if len(l0_safe_access_positions) >= 2 else safe_access_positions
    if len(pool) < 2:
        return False
    n = len(pool)
    i0 = _rand_int(state, n)
    i1 = _rand_int(state, n)
    for _ in range(50):
        if i1 != i0:
            break
        i1 = _rand_int(state, n)
    if i1 == i0:
        return False
    sids = np.array([l7_hold_sid, l7_toggle_sid], dtype=np.int32)
    targets = np.array([pool[i0], pool[i1]], dtype=np.int32)
    return _numba_place_sids(genome, sids, targets, pos_map)


@njit(cache=True)
def _numba_overwrite_group_as_unit(genome, pos_map, state,
                                   group_sids_arr, group_sizes,
                                   group_anchors_flat, group_anchor_start,
                                   n_groups, n_shortcuts):
    if n_groups == 0:
        return False
    g = _rand_int(state, n_groups)
    gsize = group_sizes[g]
    a_start = group_anchor_start[g]
    a_end = group_anchor_start[g + 1]

    valid_anchors = np.empty(a_end - a_start, dtype=np.int32)
    n_valid = 0
    for ai in range(a_start, a_end):
        occupied = False
        for k in range(gsize):
            pos = group_anchors_flat[ai, k]
            sid = genome[pos]
            for m in range(gsize):
                if sid == group_sids_arr[g, m]:
                    occupied = True
                    break
            if occupied:
                break
        if not occupied:
            valid_anchors[n_valid] = ai
            n_valid += 1

    if n_valid == 0:
        return False

    chosen_ai = valid_anchors[_rand_int(state, n_valid)]
    sids = np.empty(gsize, dtype=np.int32)
    targets = np.empty(gsize, dtype=np.int32)
    for k in range(gsize):
        sids[k] = group_sids_arr[g, k]
        targets[k] = group_anchors_flat[chosen_ai, k]

    current_positions = np.empty(gsize, dtype=np.int32)
    present = np.zeros(gsize, dtype=np.bool_)
    for k in range(gsize):
        p = pos_map[sids[k]]
        current_positions[k] = p
        present[k] = p >= 0

    displaced = np.empty(gsize, dtype=np.int32)
    for k in range(gsize):
        displaced[k] = genome[targets[k]]

    for k in range(gsize):
        genome[targets[k]] = sids[k]

    vacated = np.empty(gsize, dtype=np.int32)
    n_vacated = 0
    for k in range(gsize):
        if present[k]:
            p = current_positions[k]
            if not _numba_contains_int(targets, gsize, p):
                vacated[n_vacated] = p
                n_vacated += 1

    n_filled = 0
    for k in range(gsize):
        s = displaced[k]
        is_member = False
        for m in range(gsize):
            if s == sids[m]:
                is_member = True
                break
        if not is_member and n_filled < n_vacated:
            genome[vacated[n_filled]] = s
            n_filled += 1

    while n_filled < n_vacated:
        genome[vacated[n_filled]] = -1
        n_filled += 1

    return True


@njit(cache=True)
def _numba_cluster_app_shortcut(genome, pos_map, thumb_exclude, state,
                                app_sids_flat, app_sids_start, n_apps,
                                n_app_sample, pos_x, pos_y, mutable_arr,
                                is_group_sid_lut, n_shortcuts):
    if n_apps == 0:
        return False
    n_sample = n_app_sample if n_app_sample < n_apps else n_apps

    app_order = np.arange(n_apps, dtype=np.int32)
    for i in range(n_sample):
        j = i + _rand_int(state, n_apps - i)
        tmp = app_order[i]
        app_order[i] = app_order[j]
        app_order[j] = tmp

    n_mut = len(mutable_arr)
    for idx in range(n_sample):
        app_i = app_order[idx]
        a_start = app_sids_start[app_i]
        a_end = app_sids_start[app_i + 1]
        n_asids = a_end - a_start
        if n_asids < 2:
            continue

        present_positions = np.empty(n_asids, dtype=np.int32)
        n_present = 0
        for k in range(n_asids):
            sid = app_sids_flat[a_start + k]
            p = pos_map[sid]
            if p >= 0:
                present_positions[n_present] = p
                n_present += 1

        if n_present < 2:
            continue

        cx = 0.0
        cy = 0.0
        for k in range(n_present):
            cx += pos_x[present_positions[k]]
            cy += pos_y[present_positions[k]]
        cx /= n_present
        cy /= n_present

        max_dist = 0.0
        oi = -1
        for k in range(n_present):
            dx = pos_x[present_positions[k]] - cx
            dy = pos_y[present_positions[k]] - cy
            d = np.sqrt(dx * dx + dy * dy)
            if d > max_dist:
                max_dist = d
                oi = k

        if max_dist <= 0.5:
            continue

        outlier_pos = present_positions[oi]

        candidates = np.empty(n_mut, dtype=np.int32)
        n_cand = 0
        for k in range(n_mut):
            pos = mutable_arr[k]
            if pos == outlier_pos:
                continue
            if thumb_exclude[k]:
                continue
            sid = genome[pos]
            if sid >= 0 and sid < n_shortcuts and is_group_sid_lut[sid]:
                continue
            candidates[n_cand] = pos
            n_cand += 1

        if n_cand == 0:
            continue

        best_pos = -1
        best_d = 1e9
        for k in range(n_cand):
            pos = candidates[k]
            dx = pos_x[pos] - cx
            dy = pos_y[pos] - cy
            d = np.sqrt(dx * dx + dy * dy)
            if d < best_d:
                best_d = d
                best_pos = pos

        if best_d >= max_dist:
            continue

        displaced = genome[best_pos]
        genome[best_pos] = genome[outlier_pos]
        genome[outlier_pos] = displaced
        return True

    return False


@njit(cache=True)
def _numba_drop_optional_raw_arrows(genome, state, mutable_arr,
                                    is_raw_arrow_lut, assignable_not_arrow,
                                    n_shortcuts):
    n_mut = len(mutable_arr)
    arrow_positions = np.empty(n_mut, dtype=np.int32)
    n_arrows = 0
    for k in range(n_mut):
        pos = mutable_arr[k]
        sid = genome[pos]
        if sid >= 0 and sid < n_shortcuts and is_raw_arrow_lut[sid]:
            arrow_positions[n_arrows] = pos
            n_arrows += 1
    if n_arrows == 0 or len(assignable_not_arrow) == 0:
        return False
    n_repl = len(assignable_not_arrow)
    for k in range(n_arrows):
        genome[arrow_positions[k]] = assignable_not_arrow[_rand_int(state, n_repl)]
    return True


@njit(cache=True)
def _numba_bulk_reassign(genome, state, mutable_arr, pos_layer_arr,
                         assignable_arr, mo_access_target_lut,
                         is_group_sid_lut, is_important_sid_lut, n_shortcuts):
    n_mut = len(mutable_arr)
    if n_mut == 0 or len(assignable_arr) == 0:
        return False

    counts = np.zeros(n_shortcuts, dtype=np.int32)
    for k in range(n_mut):
        sid = genome[mutable_arr[k]]
        if sid >= 0 and sid < n_shortcuts:
            counts[sid] += 1

    candidates = np.empty(n_mut, dtype=np.int32)
    n_cand = 0
    for k in range(n_mut):
        sid = genome[mutable_arr[k]]
        if sid < 0 or sid >= n_shortcuts:
            continue
        if is_group_sid_lut[sid]:
            continue
        if is_important_sid_lut[sid] and counts[sid] <= 1:
            continue
        candidates[n_cand] = mutable_arr[k]
        n_cand += 1

    if n_cand < 2:
        return False

    max_change = 8 if n_cand >= 8 else n_cand
    n_change = 2 + _rand_int(state, max_change - 1)
    if n_change > n_cand:
        n_change = n_cand

    tmp = np.empty(n_cand, dtype=np.int32)
    for i in range(n_cand):
        tmp[i] = candidates[i]
    for i in range(n_change):
        j = i + _rand_int(state, n_cand - i)
        tmp_i = tmp[i]
        tmp[i] = tmp[j]
        tmp[j] = tmp_i

    n_assign = len(assignable_arr)
    for i in range(n_change):
        pos = tmp[i]
        layer = pos_layer_arr[pos]
        valid = np.empty(n_assign, dtype=np.int32)
        n_valid = 0
        for ai in range(n_assign):
            sid = assignable_arr[ai]
            mo_tgt = mo_access_target_lut[sid]
            if mo_tgt >= 0 and mo_tgt == layer:
                continue
            valid[n_valid] = sid
            n_valid += 1
        if n_valid == 0:
            valid = assignable_arr
            n_valid = n_assign
        genome[pos] = valid[_rand_int(state, n_valid)]

    return True


@njit(parallel=True, cache=True)
def _semantic_mutations_batch_numba(
    X, handled, semantic_probs, seeds,
    pos_maps, thumb_excludes,
    mouse_candidate_layers,
    right_non_thumb_flat, right_non_thumb_start,
    right_positions_flat, right_positions_start,
    right_thumb_positions_flat, right_thumb_positions_start,
    safe_access_positions, l0_safe_access_positions,
    layer_access_hold, layer_access_toggle, layer_scroll_access,
    return_toggle_sid,
    mouse_button_sids,
    group_sids_arr, group_sizes, group_anchors_flat, group_anchor_start,
    n_groups,
    app_sids_flat, app_sids_start, n_apps, n_app_sample,
    pos_x, pos_y, mutable_arr,
    is_group_sid_lut, n_shortcuts,
    is_raw_arrow_lut, assignable_not_arrow,
    assignable_arr, pos_layer_arr, mo_access_target_lut, is_important_sid_lut,
    pos_effort_arr,
):
    """Parallel semantic-mutation pass (Pass 1/1b) for SwapMutation._do.

    Each unhandled genome attempts mutation types in priority order:
    mouse workflow, L7 access, protected-group overwrite, optional raw-arrow
    drop, bulk reassignment, and app-cluster compaction.  The first successful
    mutation marks the genome handled and stops further attempts for that row.
    """
    n = X.shape[0]
    for i in prange(n):
        if handled[i]:
            continue
        state = np.empty(1, dtype=np.uint64)
        state[0] = seeds[i]
        genome = X[i]
        pos_map = pos_maps[i]
        thumb_exclude = thumb_excludes[i]

        # 1. Mouse workflow layer
        if _rand_float(state) < semantic_probs[0]:
            if _numba_propose_mouse_workflow_layer(
                genome, pos_map, thumb_exclude, state,
                mouse_candidate_layers,
                right_non_thumb_flat, right_non_thumb_start,
                right_positions_flat, right_positions_start,
                right_thumb_positions_flat, right_thumb_positions_start,
                safe_access_positions, l0_safe_access_positions,
                layer_access_hold, layer_access_toggle, layer_scroll_access,
                return_toggle_sid,
                mouse_button_sids, n_shortcuts, pos_effort_arr, pos_x,
            ):
                handled[i] = True
                continue

        # 2. L7 access from L0
        if _rand_float(state) < semantic_probs[1]:
            if _numba_propose_l7_access(
                genome, pos_map, state,
                l0_safe_access_positions, safe_access_positions,
                layer_access_hold[7], layer_access_toggle[7],
            ):
                handled[i] = True
                continue

        # 3. Protected group overwrite-as-unit
        if _rand_float(state) < semantic_probs[2]:
            if _numba_overwrite_group_as_unit(
                genome, pos_map, state,
                group_sids_arr, group_sizes,
                group_anchors_flat, group_anchor_start,
                n_groups, n_shortcuts,
            ):
                handled[i] = True
                continue

        # 4. Drop optional raw arrows
        if _rand_float(state) < semantic_probs[3]:
            if _numba_drop_optional_raw_arrows(
                genome, state, mutable_arr,
                is_raw_arrow_lut, assignable_not_arrow, n_shortcuts,
            ):
                handled[i] = True
                continue

        # 5. Bulk reassignment
        if _rand_float(state) < semantic_probs[4]:
            if _numba_bulk_reassign(
                genome, state, mutable_arr, pos_layer_arr,
                assignable_arr, mo_access_target_lut,
                is_group_sid_lut, is_important_sid_lut, n_shortcuts,
            ):
                handled[i] = True
                continue

        # 6. App-cluster compaction
        if _rand_float(state) < semantic_probs[5]:
            if _numba_cluster_app_shortcut(
                genome, pos_map, thumb_exclude, state,
                app_sids_flat, app_sids_start, n_apps, n_app_sample,
                pos_x, pos_y, mutable_arr,
                is_group_sid_lut, n_shortcuts,
            ):
                handled[i] = True
                continue


class LayoutProblem(Problem):
    def __init__(self, n_positions, n_shortcuts, evaluator,
                 surrogate_manager=None, frozen_mask=None):
        self.evaluator = evaluator
        self.surrogate_manager = surrogate_manager
        self.frozen_mask = frozen_mask if frozen_mask is not None else np.zeros(n_positions, dtype=bool)
        super().__init__(
            n_var=n_positions, n_obj=3, n_constr=0,
            xl=-1, xu=n_shortcuts - 1, vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        n = x.shape[0]
        F = np.zeros((n, 3), dtype=np.float32)
        if self.surrogate_manager is not None and self.surrogate_manager.trainer.mean is not None:
            try:
                F = self.surrogate_manager.trainer.predict(x)
                out["F"] = F
                return
            except Exception:
                pass
        out["F"] = F


class PermutationSampling(Sampling):
    def __init__(self, n_shortcuts, frozen_mask=None, seed_genome=None, inject_seed=True, layout=None):
        super().__init__()
        self.n_shortcuts = n_shortcuts
        self.frozen_mask = frozen_mask
        self.seed_genome = seed_genome
        self.inject_seed = inject_seed
        self.layout = layout
        self.mutable = np.where(~frozen_mask)[0] if frozen_mask is not None else None
        self.frozen = np.where(frozen_mask)[0] if frozen_mask is not None else np.array([], dtype=int)
        if self.seed_genome is not None and len(self.frozen) > 0:
            self.frozen_assigned = {
                int(sid) for sid in np.asarray(self.seed_genome, dtype=int)[self.frozen]
                if int(sid) >= 0
            }
        else:
            self.frozen_assigned = set()
        self.available_sids = np.asarray(
            [sid for sid in range(n_shortcuts) if sid not in self.frozen_assigned],
            dtype=int,
        )
        # L0 thumb mutable positions and hold-access SIDs for the guaranteed L0 access key
        self._l0_thumb_mutable = []
        self._hold_access_sids = []
        if layout is not None:
            self._l0_thumb_mutable = [
                p.gene_idx for p in layout.positions
                if p.layer == 0 and p.is_thumb and not p.is_frozen
            ]
            self._hold_access_sids = [
                s.sid for s in layout.shortcuts
                if s.is_layer_access and s.access_is_momentary
                and s.access_target_layer not in (0, 7)
                and "scroll" not in s.keys.lower()
            ]

    def _random_genome(self, genome, mutable):
        """Fill mutable positions randomly, placing every available shortcut at least once."""
        n_mut = len(mutable)
        avail = list(self.available_sids)
        # Build assignment list: cycle through all shortcuts until we have enough
        assignments = []
        shuffled = avail[:]
        np.random.shuffle(shuffled)
        while len(assignments) < n_mut:
            assignments.extend(shuffled)
            np.random.shuffle(shuffled)
        assignments = assignments[:n_mut]
        # Randomize which shortcut goes to which position
        pos_order = list(mutable)
        np.random.shuffle(pos_order)
        for idx, sid in zip(pos_order, assignments):
            genome[idx] = sid
        # L0 thumb guarantee: one random momentary hold (not L7) on a random L0 thumb position
        if self._l0_thumb_mutable and self._hold_access_sids:
            chosen_pos = int(np.random.choice(self._l0_thumb_mutable))
            chosen_sid = int(np.random.choice(self._hold_access_sids))
            genome[chosen_pos] = chosen_sid
        return genome

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), -1, dtype=int)
        mutable = self.mutable if self.mutable is not None else np.arange(problem.n_var)

        # Always start from frozen genome base
        base = np.asarray(self.seed_genome, dtype=int).copy() if self.seed_genome is not None else np.full(problem.n_var, -1, dtype=int)

        for i in range(n_samples):
            genome = base.copy()
            # Clear mutable positions for fresh random assignment
            genome[mutable] = -1
            genome = self._random_genome(genome, mutable)
            X[i] = genome

        return X


class CycleCrossover(Crossover):
    def __init__(self, prob=0.9, n_shortcuts=None):
        super().__init__(2, 2, prob=prob)
        self.n_shortcuts = n_shortcuts
        self._p2_pos_buffer = None

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        assert n_parents == 2
        Y = np.full_like(X, -1)
        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]
            if random.random() > (float(self.prob.value) if hasattr(self.prob, "value") else float(self.prob)):
                Y[0, k] = p1.copy()
                Y[1, k] = p2.copy()
                continue
            c1, c2 = self._cycle_crossover_pair(p1, p2)
            # Preserve frozen positions from parents
            if problem.frozen_mask is not None:
                frozen = np.where(problem.frozen_mask)[0]
                c1[frozen] = p1[frozen]
                c2[frozen] = p2[frozen]
            Y[0, k] = c1
            Y[1, k] = c2
        return Y

    def _cycle_crossover_pair(self, p1, p2):
        # Fast path: Numba-compiled version (~165x speedup)
        if NUMBA_AVAILABLE and self.n_shortcuts is not None:
            return _cycle_crossover_pair_numba(p1, p2, self.n_shortcuts)
        # Pure-Python fallback with dict + boolarray (1.45x over original set+array)
        n = len(p1)
        c1 = np.full(n, -1, dtype=np.int32)
        c2 = np.full(n, -1, dtype=np.int32)
        start_idx = 0
        while start_idx < n and (p1[start_idx] < 0 or p2[start_idx] < 0):
            start_idx += 1
        if start_idx >= n:
            return p1.copy(), p2.copy()
        p2_pos = {}
        for i, sid in enumerate(p2):
            if sid >= 0:
                p2_pos[sid] = i
        cycle = np.zeros(n, dtype=bool)
        idx = start_idx
        while True:
            cycle[idx] = True
            val = p1[idx]
            next_idx = p2_pos.get(val, -1)
            if next_idx < 0 or cycle[next_idx] or next_idx == start_idx:
                break
            idx = next_idx
        for i in range(n):
            if cycle[i]:
                c1[i] = p1[i]
                c2[i] = p2[i]
            else:
                c1[i] = p2[i]
                c2[i] = p1[i]
        return c1, c2


class SwapMutation(Mutation):
    def __init__(
        self,
        prob=0.15,
        frozen_mask=None,
        layout=None,
        group_overwrite_prob=0.15,
        mouse_workflow_prob=0.06,
        l7_access_prob=0.03,
        random_assign_prob=0.08,
        bulk_assign_prob=0.04,
        optional_arrow_drop_prob=0.04,
        group_move_prob=None,
        cluster_app_prob=0.20,
        effort_swap_prob=0.06,
        smart_duplicate_prob=0.20,
    ):
        super().__init__()
        self.prob = prob
        # Backward-compatible alias for older tests/config.  The behavior is
        # overwrite-triggered group movement, not periodic group repair.
        self.group_overwrite_prob = group_overwrite_prob if group_move_prob is None else group_move_prob
        self.mouse_workflow_prob = mouse_workflow_prob
        self.l7_access_prob = l7_access_prob
        self.random_assign_prob = random_assign_prob
        self.bulk_assign_prob = bulk_assign_prob
        self.optional_arrow_drop_prob = optional_arrow_drop_prob
        self.cluster_app_prob = cluster_app_prob
        self.effort_swap_prob = effort_swap_prob
        self.smart_duplicate_prob = smart_duplicate_prob
        self.access_thumb_bias_prob = 0.15
        self.return_toggle_repair_prob = 0.10
        self.toggle_own_layer_bias_prob = 0.12
        self.mutable_indices = np.where(~frozen_mask)[0] if frozen_mask is not None else None
        self.mutable_list = self.mutable_indices.tolist() if self.mutable_indices is not None else None
        self.group_sid_sets = []
        self.group_anchor_positions = []
        self.group_anchor_sets: list = []  # precomputed frozensets for fast intersection test
        self.group_anchor_arrays: list = []  # precomputed numpy arrays for the hot path
        self.group_member_sids: set = set()
        self.assignable_sids = None
        self.important_sids = set()
        self.raw_arrow_sids = set()
        self.mouse_button_sids = {}
        self.scroll_access_by_target = {}
        self.access_hold_by_target = {}
        self.access_toggle_by_target = {}
        self.right_non_thumb_by_layer = defaultdict(list)
        self.safe_access_positions = []
        self.l0_safe_access_positions = []
        # Bug 3 & 5: per-layer position pools and access SID metadata
        self.right_positions_by_layer: dict = defaultdict(list)
        self.right_thumb_positions_by_layer: dict = defaultdict(list)
        self.left_thumb_positions_by_layer: dict = defaultdict(list)
        self._pos_layer_arr = np.zeros(len(layout.positions) if layout is not None else 0, dtype=np.int32)
        self._pos_hand_arr = np.zeros(len(layout.positions) if layout is not None else 0, dtype=np.int32)
        self._pos_is_thumb_arr = np.zeros(len(layout.positions) if layout is not None else 0, dtype=np.bool_)
        self._access_sid_targets: dict = {}
        self._access_sid_momentary: dict = {}
        if layout is not None:
            for idx, pos in enumerate(layout.positions):
                self._pos_layer_arr[idx] = int(pos.layer)
                self._pos_hand_arr[idx] = 1 if pos.hand == "right" else 0
                self._pos_is_thumb_arr[idx] = bool(pos.is_thumb)
                if pos.is_frozen or pos.layer == 7 or pos.layer == 0:
                    continue
                layer_i = int(pos.layer)
                if pos.hand == "right":
                    self.right_positions_by_layer[layer_i].append(idx)
                    if pos.is_thumb:
                        self.right_thumb_positions_by_layer[layer_i].append(idx)
                elif pos.is_thumb:
                    self.left_thumb_positions_by_layer[layer_i].append(idx)
            for s in layout.shortcuts:
                if s.is_layer_access:
                    self._access_sid_targets[s.sid] = int(s.access_target_layer)
                    self._access_sid_momentary[s.sid] = bool(s.access_is_momentary)
        if layout is not None and self.mutable_list:
            frozen_sids = {
                int(layout.genome[i])
                for i, pos in enumerate(layout.positions)
                if pos.is_frozen and int(layout.genome[i]) >= 0
            }
            self.assignable_sids = [
                s.sid for s in layout.shortcuts
                if s.sid not in frozen_sids
            ]
            self.important_sids = {
                s.sid for s in layout.shortcuts
                if s.sid not in frozen_sids and float(s.importance) >= 6.0
            }
            self.raw_arrow_sids = {
                s.sid for s in layout.shortcuts
                if s.sid not in frozen_sids
                and not s.modifiers
                and (s.base_key or "").upper() in {"LEFTARROW", "UPARROW", "DOWNARROW", "RIGHTARROW"}
            }
            self._build_protected_group_moves(layout)
            # Collect all group member sids so individual mutations can't scatter them.
            for sid_set in self.group_sid_sets:
                self.group_member_sids.update(sid_set)
            # Remove group members from assignable pool so random/bulk assign
            # never places them individually; only _overwrite_group_as_unit does.
            if self.assignable_sids and self.group_member_sids:
                self.assignable_sids = [
                    s for s in self.assignable_sids if s not in self.group_member_sids
                ]
            self._build_mouse_access_moves(layout)
            mouse_button_sid_set = set(self.mouse_button_sids.values())
            # Smart-duplication pool: all non-group, non-mouse shortcuts.
            # _base_* L0 keys (imp=0, is_l0_only=True) are included — they can
            # spawn as duplicates on mutable non-L0 layers so every layer has
            # base-key access. Group members are excluded: placed atomically only.
            # Mouse-button copies are handled by mouse-layer scoring/mutation,
            # not generic duplicate spawning; otherwise MB1-MB5 scatter across
            # unrelated layers and crowd out workflow shortcuts.
            # Selection uses softmax(imp/T) / (1 + total_count)^2 so high-importance
            # shortcuts can duplicate, but saturation quickly suppresses spam.
            _sid_to_imp = {s.sid: float(s.importance) for s in layout.shortcuts}
            _dup_sids = [
                s.sid for s in layout.shortcuts
                if s.sid not in self.group_member_sids
                and s.sid not in mouse_button_sid_set
            ]
            self._dup_candidate_arr = np.array(_dup_sids, dtype=np.int32)
            self._dup_imp_arr = np.array([_sid_to_imp[sid] for sid in _dup_sids], dtype=np.float32)
            # Precompute the softmax numerator (constant across genomes) so
            # _smart_duplicate avoids an np.exp call per invocation.
            T = 5.0
            dup_logits = self._dup_imp_arr / T
            dup_logits = dup_logits - float(dup_logits.max())
            self._dup_exp_w = np.exp(dup_logits)
            # Precompute frozen position counts: _base_* keys on L0 already have
            # count=1 from frozen placement, so they start with half the weight
            # of an unplaced shortcut when the discount is applied at runtime.
            self._frozen_sid_counts = np.zeros(len(layout.shortcuts), dtype=np.int32)
            for _idx, _pos in enumerate(layout.positions):
                if _pos.is_frozen:
                    _fsid = int(layout.genome[_idx])
                    if 0 <= _fsid < len(layout.shortcuts):
                        self._frozen_sid_counts[_fsid] += 1
        # Precomputed arrays for vectorized mutation
        self.n_shortcuts = len(layout.shortcuts) if layout is not None else 0
        n_sc = self.n_shortcuts
        # Fallback when layout is None (e.g. unit tests)
        if not hasattr(self, '_dup_candidate_arr'):
            self._dup_candidate_arr = np.array([], dtype=np.int32)
            self._dup_imp_arr = np.array([], dtype=np.float32)
            self._dup_exp_w = np.array([], dtype=np.float32)
            self._frozen_sid_counts = np.zeros(0, dtype=np.int32)
        self._mutable_arr = np.array(self.mutable_list or [], dtype=np.int32)
        self._assignable_arr = np.array(self.assignable_sids or [], dtype=np.int32)
        self._important_arr = np.array(sorted(self.important_sids), dtype=np.int32)
        self._group_arr = np.array(sorted(self.group_member_sids), dtype=np.int32)
        self._raw_arrow_arr = np.array(sorted(self.raw_arrow_sids), dtype=np.int32)
        # Precomputed mutable-position attribute slices to avoid repeated indexing.
        self._mutable_is_thumb = self._pos_is_thumb_arr[self._mutable_arr]
        self._mutable_layer = self._pos_layer_arr[self._mutable_arr]
        self._mutable_hand = self._pos_hand_arr[self._mutable_arr]
        # Boolean LUTs for O(1) per-element membership checks on numpy arrays
        self._is_group_sid_lut = np.zeros(n_sc, dtype=np.bool_) if n_sc > 0 else np.array([], dtype=np.bool_)
        self._is_important_sid_lut = np.zeros(n_sc, dtype=np.bool_) if n_sc > 0 else np.array([], dtype=np.bool_)
        if n_sc > 0:
            for sid in self.group_member_sids:
                if 0 <= sid < n_sc:
                    self._is_group_sid_lut[sid] = True
            for sid in self.important_sids:
                if 0 <= sid < n_sc:
                    self._is_important_sid_lut[sid] = True
            self._is_raw_arrow_lut = np.zeros(n_sc, dtype=np.bool_)
            for sid in self.raw_arrow_sids:
                if 0 <= sid < n_sc:
                    self._is_raw_arrow_lut[sid] = True
        else:
            self._is_raw_arrow_lut = np.array([], dtype=np.bool_)
        # Assignable SIDs that are not raw arrows (used by _drop_optional_raw_arrows)
        if len(self._raw_arrow_arr) > 0 and len(self._assignable_arr) > 0:
            self._assignable_not_arrow = self._assignable_arr[
                ~np.isin(self._assignable_arr, self._raw_arrow_arr)
            ]
        else:
            self._assignable_not_arrow = self._assignable_arr

        # LUTs for O(1) numpy-vectorized reachability and thumb-occupancy checks.
        # Replaces Python dict iteration in _would_break_reachability and
        # _get_layer_occupied_thumbs — one numpy gather replaces 510-iter loops.
        self._access_target_lut = np.full(n_sc, -1, dtype=np.int32)
        self._access_is_mo_lut = np.zeros(n_sc, dtype=np.bool_)
        # Momentary-only target layer: used to block self-referential hold placements.
        # Value = target layer for momentary access sids, -1 for everything else.
        # A hold key @LX:hold placed ON layer X is illegal: it never fires usefully.
        self._mo_access_target_lut = np.full(n_sc, -1, dtype=np.int32)
        for _sid, _tgt in self._access_sid_targets.items():
            if 0 <= _sid < n_sc:
                self._access_target_lut[_sid] = int(_tgt)
                if self._access_sid_momentary.get(_sid, False):
                    self._mo_access_target_lut[_sid] = int(_tgt)
        for _sid, _is_mo in self._access_sid_momentary.items():
            if 0 <= _sid < n_sc:
                self._access_is_mo_lut[_sid] = bool(_is_mo)

        # App-cluster mutation: precompute app → sids and position x,y arrays
        self._pos_x = np.array([p.x for p in layout.positions], dtype=np.float32) if layout is not None else np.array([], dtype=np.float32)
        self._pos_y = np.array([p.y for p in layout.positions], dtype=np.float32) if layout is not None else np.array([], dtype=np.float32)
        self._app_sids: dict = {}
        if layout is not None and self.mutable_list:
            frozen_sids_set = {
                int(layout.genome[i])
                for i, pos in enumerate(layout.positions)
                if pos.is_frozen and int(layout.genome[i]) >= 0
            }
            for s in layout.shortcuts:
                if s.sid in frozen_sids_set or s.sid in self.group_member_sids:
                    continue
                self._app_sids.setdefault(s.app, set()).add(s.sid)
            self._app_sids = {k: v for k, v in self._app_sids.items() if len(v) >= 2}
        self._app_ids = list(self._app_sids.keys())
        # Numpy arrays per app for vectorized sid presence checks
        self._app_sids_arrs: dict = {
            app_id: np.array(sorted(sids), dtype=np.int32)
            for app_id, sids in self._app_sids.items()
        }
        # Toggle pairing: return-to-L0 toggle SID and per-layer mutable positions.
        # Used by _ensure_return_toggle to place a return key whenever a toggle
        # access to a layer is newly created by mutation.
        self._return_toggle_sid = None
        self._toggle_access_sids: set = set()
        self._layer_mutable_positions: dict = {}
        if layout is not None:
            for s in layout.shortcuts:
                if s.is_layer_access and not s.access_is_momentary:
                    if s.access_target_layer == 0:
                        self._return_toggle_sid = s.sid
                    elif s.access_target_layer != 7:
                        self._toggle_access_sids.add(s.sid)
            for idx in (self.mutable_list or []):
                lyr = int(layout.positions[idx].layer)
                self._layer_mutable_positions.setdefault(lyr, []).append(idx)

        # Flattened layer mutable positions for O(1) Numba lookups.
        self._layer_mutable_flat = np.zeros(0, dtype=np.int32)
        self._layer_mutable_start = np.zeros(33, dtype=np.int32)
        if layout is not None and self._layer_mutable_positions:
            flat_parts = []
            start = 0
            for lyr in range(33):
                self._layer_mutable_start[lyr] = start
                positions = self._layer_mutable_positions.get(lyr, [])
                flat_parts.extend(positions)
                start += len(positions)
            self._layer_mutable_flat = np.array(flat_parts, dtype=np.int32)

        # ------------------------------------------------------------------
        # Flatten variable-length layout data for the Numba semantic-mutation
        # kernels.  All arrays are int32 with start/length tables so Numba can
        # index them without Python lists/dicts.
        # ------------------------------------------------------------------

        def _flatten_layer_positions(mapping, n_layers=33):
            """Return (flat_array, start_array) for a dict layer -> positions."""
            flat = []
            start = np.zeros(n_layers, dtype=np.int32)
            off = 0
            for lyr in range(n_layers):
                start[lyr] = off
                vals = mapping.get(lyr, [])
                flat.extend(vals)
                off += len(vals)
            return np.array(flat, dtype=np.int32), start

        self._right_non_thumb_flat, self._right_non_thumb_start = _flatten_layer_positions(
            self.right_non_thumb_by_layer
        )
        self._right_positions_flat, self._right_positions_start = _flatten_layer_positions(
            self.right_positions_by_layer
        )
        self._right_thumb_positions_flat, self._right_thumb_positions_start = _flatten_layer_positions(
            self.right_thumb_positions_by_layer
        )
        self._left_thumb_positions_flat, self._left_thumb_positions_start = _flatten_layer_positions(
            self.left_thumb_positions_by_layer
        )

        self._safe_access_positions_arr = np.array(self.safe_access_positions, dtype=np.int32)
        self._l0_safe_access_positions_arr = np.array(self.l0_safe_access_positions, dtype=np.int32)

        self._layer_access_hold = np.full(32, -1, dtype=np.int32)
        self._layer_access_toggle = np.full(32, -1, dtype=np.int32)
        self._layer_scroll_access = np.full(32, -1, dtype=np.int32)
        for lyr, sid in self.access_hold_by_target.items():
            if 0 <= lyr < 32:
                self._layer_access_hold[lyr] = int(sid)
        for lyr, sid in self.access_toggle_by_target.items():
            if 0 <= lyr < 32:
                self._layer_access_toggle[lyr] = int(sid)
        for lyr, sid in self.scroll_access_by_target.items():
            if 0 <= lyr < 32:
                self._layer_scroll_access[lyr] = int(sid)
        self._return_toggle_sid_arr = np.int32(
            self._return_toggle_sid if self._return_toggle_sid is not None else -1
        )

        # Mouse-layer candidates: non-L0/non-L7 layers with hold+toggle+scroll
        # access and at least 6 right-hand non-thumb positions.
        mouse_candidates = []
        for lyr in range(1, 32):
            if lyr == 7:
                continue
            start = int(self._right_non_thumb_start[lyr])
            end = int(self._right_non_thumb_start[lyr + 1]) if lyr + 1 < 33 else len(self._right_non_thumb_flat)
            if (
                self._layer_access_hold[lyr] >= 0
                and self._layer_access_toggle[lyr] >= 0
                and self._layer_scroll_access[lyr] >= 0
                and end - start >= 6
            ):
                mouse_candidates.append(lyr)
        self._mouse_candidate_layers = np.array(mouse_candidates, dtype=np.int32)

        # Flatten per-app sid lists.  _app_ids are string app names; Numba uses
        # integer app indices 0..n_apps-1 together with the flattened sid table.
        self._n_apps = len(self._app_ids)
        app_flat = []
        app_start = np.zeros(self._n_apps + 1, dtype=np.int32)
        for idx, app_id in enumerate(self._app_ids):
            app_start[idx] = len(app_flat)
            app_flat.extend(self._app_sids_arrs[app_id])
        app_start[self._n_apps] = len(app_flat)
        self._app_sids_flat = np.array(app_flat, dtype=np.int32)
        self._app_sids_start = app_start

        # Flatten protected-group anchors.  Group sizes are small (4 or 5) so
        # we pad every group/anchor to max_gs for Numba.
        max_gs = max((len(s) for s in self.group_sid_sets), default=0)
        n_groups = len(self.group_sid_sets)
        self._group_sizes = np.array([len(s) for s in self.group_sid_sets], dtype=np.int32)
        self._group_sids_arr = np.full((n_groups, max_gs), -1, dtype=np.int32)
        total_anchors = sum(len(a) for a in self.group_anchor_arrays)
        self._group_anchors_flat = np.full((total_anchors, max_gs), -1, dtype=np.int32)
        self._group_anchor_start = np.zeros(n_groups + 1, dtype=np.int32)
        anchor_off = 0
        for g in range(n_groups):
            self._group_sids_arr[g, : self._group_sizes[g]] = self.group_sid_sets[g]
            self._group_anchor_start[g] = anchor_off
            anchors = self.group_anchor_arrays[g]
            for k in range(len(anchors)):
                self._group_anchors_flat[anchor_off + k, : self._group_sizes[g]] = anchors[k]
            anchor_off += len(anchors)
        self._group_anchor_start[n_groups] = anchor_off

        # Mouse button sids as a fixed-size int32 array (index 1-5 used, 0 unused).
        self._mouse_button_sids = np.full(6, -1, dtype=np.int32)
        for btn_idx, sid in self.mouse_button_sids.items():
            if 1 <= btn_idx <= 5:
                self._mouse_button_sids[btn_idx] = int(sid)

        # Toggle access sids (not @L0, not @L7) as a sorted int32 array.
        self._toggle_access_sids_arr = np.array(
            sorted(self._toggle_access_sids), dtype=np.int32
        )

        # Effort-swap mutation: position efforts and shortcut importances as arrays
        self._pos_effort_arr = np.array(
            [float(p.effort) for p in layout.positions], dtype=np.float32
        ) if layout is not None else np.array([], dtype=np.float32)
        self._sid_importance_arr = np.zeros(n_sc, dtype=np.float32)
        if layout is not None:
            for s in layout.shortcuts:
                if 0 <= s.sid < n_sc:
                    self._sid_importance_arr[s.sid] = float(s.importance)

    def _build_protected_group_moves(self, layout):
        """Precompute whole-group mutation targets via build_group_placements."""
        for sid_tuple, anchor_list in build_group_placements(layout):
            self.group_sid_sets.append(sid_tuple)
            self.group_anchor_positions.append(anchor_list)
            # Precompute frozensets so _overwrite_group_as_unit uses isdisjoint()
            # instead of constructing set(anchor) on every call (saves 11ms/gen).
            self.group_anchor_sets.append([frozenset(a) for a in anchor_list])
            # Precompute numpy anchor array so the hot path avoids np.asarray().
            self.group_anchor_arrays.append(np.asarray(anchor_list, dtype=np.int32))

    def _build_mouse_access_moves(self, layout):
        """Precompute coordinated capability mutation ingredients.

        This does not assign a fixed mouse layer.  It only lets mutation create
        a candidate workflow surface on any non-L0/non-L7 layer; scoring and
        selection decide whether that candidate survives.
        """
        for shortcut in layout.shortcuts:
            key = (shortcut.keys or "").upper().replace(" ", "")
            if key in {"MB1", "MB2", "MB3", "MB4", "MB5"}:
                self.mouse_button_sids[int(key[2])] = shortcut.sid
            if shortcut.is_layer_access:
                target = int(shortcut.access_target_layer)
                text = f"{shortcut.keys} {shortcut.action} {shortcut.base_key}".lower()
                if "scroll" in text and shortcut.access_is_momentary:
                    self.scroll_access_by_target[target] = shortcut.sid
                elif shortcut.access_is_momentary:
                    self.access_hold_by_target[target] = shortcut.sid
                else:
                    self.access_toggle_by_target[target] = shortcut.sid

        for pos in layout.positions:
            if pos.is_frozen or pos.layer == 7:
                continue
            # Bug 1 fix: ALL non-frozen non-L7 positions (including both thumb clusters)
            # are valid for MO/TO coach buttons. Only MB/scroll are restricted to
            # right_non_thumb_by_layer below.
            self.safe_access_positions.append(pos.gene_idx)
            if pos.layer == 0:
                self.l0_safe_access_positions.append(pos.gene_idx)
            if pos.layer == 0:
                continue
            if pos.hand == "right" and not pos.is_thumb:
                self.right_non_thumb_by_layer[int(pos.layer)].append(pos.gene_idx)

        for layer in list(self.right_non_thumb_by_layer):
            self.right_non_thumb_by_layer[layer].sort(
                key=lambda idx: (layout.positions[idx].effort, layout.positions[idx].y, layout.positions[idx].x)
            )

    def _genome_pos_map(self, genome, row_pos_map=None):
        """Build sid→position array in one numpy pass (replaces repeated np.where calls)."""
        if row_pos_map is not None:
            return row_pos_map
        pos_map = np.full(self.n_shortcuts, -1, dtype=np.int32)
        valid = genome >= 0
        sids = genome[valid].astype(np.int32)
        sids = np.clip(sids, 0, self.n_shortcuts - 1)
        pos_map[sids] = np.where(valid)[0].astype(np.int32)
        return pos_map

    def _batched_pos_map(self, X):
        """Build sid→position map for every row in the batch at once.

        Returns an (n, n_shortcuts) array where pos_map[i, sid] is the position
        of ``sid`` in ``X[i]`` (or -1 if absent).  This replaces per-genome
        ``_genome_pos_map`` scans inside tight mutation loops.
        """
        n, n_pos = X.shape
        n_sc = max(1, self.n_shortcuts)
        pos_map = np.full((n, n_sc), -1, dtype=np.int32)
        rows = np.broadcast_to(np.arange(n, dtype=np.int32)[:, None], (n, n_pos))
        positions = np.broadcast_to(np.arange(n_pos, dtype=np.int32), (n, n_pos))
        valid = (X >= 0) & (X < n_sc)
        pos_map[rows[valid], X[valid]] = positions[valid]
        return pos_map

    def _batched_thumb_exclude(self, X):
        """Return a per-row thumb-exclusion mask over mutable positions.

        A hand's thumb area on layer L is restricted when L is reachable only via
        a single momentary hold from that hand.  The result has shape
        (n, n_mutable) and mirrors ``_thumb_exclude_mask`` computed for the
        whole batch in one vectorized pass.
        """
        n = X.shape[0]
        m = len(self._mutable_arr)
        if m == 0:
            return np.zeros((n, 0), dtype=np.bool_)

        mutable_sids = X[:, self._mutable_arr]
        valid = (mutable_sids >= 0) & (mutable_sids < self.n_shortcuts)
        safe_sids = np.where(valid, mutable_sids, 0)
        targets = self._access_target_lut[safe_sids]
        is_mo = self._access_is_mo_lut[safe_sids]
        is_access = valid & (targets > 0) & (targets < 32)

        # Per-row/layer booleans: does the layer have a toggle access?
        has_toggle = np.zeros((n, 32), dtype=np.bool_)
        toggle_mask = is_access & ~is_mo
        if toggle_mask.any():
            row_idx_t, layer_idx_t = np.where(toggle_mask)
            has_toggle[row_idx_t, targets[toggle_mask]] = True

        # Per-row/layer/hand: which hands have momentary access to the layer?
        has_mo_hand = np.zeros((n, 32, 2), dtype=np.bool_)
        mo_mask = is_access & is_mo
        if mo_mask.any():
            row_idx_m, pos_idx_m = np.where(mo_mask)
            hands = self._mutable_hand[pos_idx_m]
            np.add.at(has_mo_hand, (row_idx_m, targets[mo_mask], hands), True)

        n_mo_hands = has_mo_hand.sum(axis=2)
        occupied = (n_mo_hands == 1) & (~has_toggle)
        occupied_hand = np.where(occupied, np.argmax(has_mo_hand, axis=2), -1)

        occupied_m = occupied[np.arange(n)[:, None], self._mutable_layer]
        hand_m = occupied_hand[np.arange(n)[:, None], self._mutable_layer]
        thumb_exclude = (
            occupied_m
            & (hand_m == self._mutable_hand)
            & self._mutable_is_thumb
            & (self._mutable_layer > 0)
        )
        return thumb_exclude

    def _place_sids(self, genome, sids, target_positions, row_pos_map=None):
        """Move SIDs to positions by swapping displaced values into old slots."""
        if len(sids) != len(target_positions):
            return False
        # One numpy pass to find all sid positions (replaces len(sids) np.where calls)
        pos_map = self._genome_pos_map(genome, row_pos_map)
        current_positions = []
        for sid in sids:
            p = int(pos_map[sid]) if 0 <= sid < self.n_shortcuts else -1
            if p < 0:
                return False
            current_positions.append(p)

        target_set = set(target_positions)
        displaced = [int(genome[pos]) for pos in target_positions]
        for sid, pos in zip(sids, target_positions):
            genome[pos] = sid

        fill_positions = [pos for pos in current_positions if pos not in target_set]
        fill_values = [sid for sid in displaced if sid not in sids]
        for pos, sid in zip(fill_positions, fill_values):
            genome[pos] = sid
        for pos in fill_positions[len(fill_values):]:
            genome[pos] = -1
        return True

    def _propose_mouse_workflow_layer(self, genome, row_pos_map=None):
        if len(self.mouse_button_sids) < 5:
            return False
        candidate_layers = [
            layer for layer, positions in self.right_non_thumb_by_layer.items()
            if layer not in (0, 7)
            and len(positions) >= 6
            and layer in self.access_hold_by_target
            and layer in self.access_toggle_by_target
            and layer in self.scroll_access_by_target
        ]
        if not candidate_layers:
            return False
        layer = random.choice(candidate_layers)
        right_positions = self.right_non_thumb_by_layer[layer]
        target_positions = random.sample(right_positions[:min(len(right_positions), 12)], 6)
        # Sort by effort (ascending), not position index, so the lowest-effort
        # slots go to the highest-priority mouse-group members instead of an
        # arbitrary index-order mapping. Priority (matching fitness/kernel.py's
        # button_weight and scroll effort-priority terms):
        # Scroll > MB2 > MB1 > MB3 ~= MB4 ~= MB5.
        target_positions.sort(key=lambda pos: (self._pos_effort_arr[pos], pos))
        # Momentary Scroll on x=7/x=8 is uncomfortable and fails final
        # acceptance outright (not just a soft scoring preference) — never
        # propose it there. If the best slot is x7/x8, swap it with the
        # first later slot that isn't, keeping the rest of the priority order.
        if self._pos_x[target_positions[0]] in (7.0, 8.0):
            for k in range(1, 6):
                if self._pos_x[target_positions[k]] not in (7.0, 8.0):
                    target_positions[0], target_positions[k] = target_positions[k], target_positions[0]
                    break
        sids = [
            self.scroll_access_by_target[layer],
            self.mouse_button_sids[2],
            self.mouse_button_sids[1],
            self.mouse_button_sids[3],
            self.mouse_button_sids[4],
            self.mouse_button_sids[5],
        ]

        blocked = set(target_positions)
        # Proposal bias only: prefer L0 for initial mouse-layer access so the
        # candidate is reachable immediately. Scoring can still move access
        # elsewhere when the live genome provides another reachable path.
        access_pool = self.l0_safe_access_positions or [
            pos for pos in self.safe_access_positions
            if self._pos_layer_arr[pos] != layer
        ]
        safe_access = [pos for pos in access_pool if pos not in blocked]
        if len(safe_access) < 2:
            return False
        hold_pos = random.choice(safe_access)
        safe_access = [pos for pos in safe_access if pos != hold_pos]
        toggle_pos = random.choice(safe_access)
        sids.extend([self.access_hold_by_target[layer], self.access_toggle_by_target[layer]])
        target_positions.extend([hold_pos, toggle_pos])
        # Bug 3 fix: place return-to-L0 toggle ON the mouse layer (right side preferred).
        # Every toggle-accessible layer must have a return toggle back to L0.
        if 0 in self.access_toggle_by_target:
            return_sid = self.access_toggle_by_target[0]
            placed_set = set(target_positions)
            right_thumbs = [p for p in self.right_thumb_positions_by_layer.get(layer, []) if p not in placed_set]
            right_any = [p for p in self.right_positions_by_layer.get(layer, []) if p not in placed_set]
            return_pool = right_thumbs if right_thumbs else right_any
            if return_pool:
                sids.append(return_sid)
                target_positions.append(random.choice(return_pool))
        return self._place_sids(genome, sids, target_positions, row_pos_map)

    def _propose_l7_access(self, genome, row_pos_map=None):
        if 7 not in self.access_hold_by_target or 7 not in self.access_toggle_by_target:
            return False
        # L7 (arrows) is always accessed from L0: prefer l0_safe_access_positions,
        # fall back to all safe positions. Bug 2 fix only applies to mouse workflow layer.
        access_pool = self.l0_safe_access_positions or self.safe_access_positions
        if len(access_pool) < 2:
            return False
        positions = random.sample(access_pool, 2)
        sids = [self.access_hold_by_target[7], self.access_toggle_by_target[7]]
        return self._place_sids(genome, sids, positions, row_pos_map)

    def _reassign_candidates(self, genome):
        """Return array of mutable position indices safe to reassign.

        Uses precomputed boolean LUTs — one numpy pass + fast LUT indexing —
        which is ~15µs vs ~300µs for per-element Python numpy scalar access.
        """
        if len(self._mutable_arr) == 0 or len(self._assignable_arr) == 0:
            return None
        mutable_sids = genome[self._mutable_arr]          # one numpy gather: 510 elements
        safe_sids = np.maximum(mutable_sids, 0)           # -1 → 0 for LUT indexing
        is_valid = mutable_sids >= 0
        is_group = self._is_group_sid_lut[safe_sids] & is_valid
        is_important = self._is_important_sid_lut[safe_sids] & is_valid
        # Bincount for singleton check (fast: only 296 buckets)
        if is_valid.any():
            counts = np.bincount(
                mutable_sids[is_valid].astype(np.int64), minlength=self.n_shortcuts
            )
            sid_counts = counts[safe_sids]
        else:
            sid_counts = np.zeros(len(mutable_sids), dtype=np.int64)
        is_singleton_important = is_important & (sid_counts <= 1)
        ok = ~is_group & ~is_singleton_important
        return self._mutable_arr[ok]

    def _would_break_reachability(self, genome, pos_to_clear):
        """True if clearing genome[pos_to_clear] would make its target layer unreachable.

        One numpy gather over mutable positions replaces the 510-iter Python loop.
        """
        sid = int(genome[pos_to_clear])
        if sid < 0 or sid >= self.n_shortcuts:
            return False
        target = int(self._access_target_lut[sid])
        if target <= 0:
            return False
        mutable_sids = genome[self._mutable_arr]
        other_mask = (self._mutable_arr != pos_to_clear) & (mutable_sids >= 0) & (mutable_sids < self.n_shortcuts)
        if not other_mask.any():
            return True
        other_sids = mutable_sids[other_mask]
        return not bool(np.any(self._access_target_lut[other_sids] == target))

    def _get_layer_occupied_thumbs(self, genome):
        """Return dict layer -> set of hand ints (0=left,1=right) whose thumb is occupied.

        A hand's thumb is occupied on layer L if L is reachable ONLY via exactly one
        momentary hold from that hand (no toggle access, no other MO keys).

        One numpy gather + LUT lookup replaces the 510-iter Python loop.
        """
        mutable_sids = genome[self._mutable_arr]
        valid = (mutable_sids >= 0) & (mutable_sids < self.n_shortcuts)
        if not valid.any():
            return {}
        safe_sids = np.where(valid, mutable_sids, 0)
        targets = self._access_target_lut[safe_sids]
        is_mo = self._access_is_mo_lut[safe_sids]
        is_access = valid & (targets > 0) & (targets < 32)
        if not is_access.any():
            return {}
        access_targets = targets[is_access]
        access_mo = is_mo[is_access]
        access_hands = self._pos_hand_arr[self._mutable_arr[is_access]]
        occupied: dict = {}
        # Only iterate over layers that actually appear (usually a handful).
        for layer_id in np.unique(access_targets):
            lm = access_targets == layer_id
            layer_mo = access_mo[lm]
            layer_hands = access_hands[lm]
            has_toggle = not bool(np.all(layer_mo))
            mo_hands = layer_hands[layer_mo]
            if has_toggle or len(mo_hands) != 1:
                occupied[int(layer_id)] = set()
            else:
                occupied[int(layer_id)] = {int(mo_hands[0])}
        return occupied

    def _thumb_exclude_mask(self, genome, row_thumb_exclude=None):
        """Return a boolean mask over mutable positions excluded because of thumb occupancy.

        This is the direct mask form used by _cluster_app_shortcut, avoiding the
        intermediate dict construction in _get_layer_occupied_thumbs.  If a
        precomputed row mask is passed (from ``_batched_thumb_exclude``), return
        it directly.
        """
        if row_thumb_exclude is not None:
            return row_thumb_exclude
        thumb_exclude = np.zeros(len(self._mutable_arr), dtype=np.bool_)
        mutable_sids = genome[self._mutable_arr]
        valid = (mutable_sids >= 0) & (mutable_sids < self.n_shortcuts)
        if not valid.any():
            return thumb_exclude
        safe_sids = np.where(valid, mutable_sids, 0)
        targets = self._access_target_lut[safe_sids]
        is_mo = self._access_is_mo_lut[safe_sids]
        is_access = valid & (targets > 0) & (targets < 32)
        if not is_access.any():
            return thumb_exclude
        access_targets = targets[is_access]
        access_mo = is_mo[is_access]
        access_hands = self._mutable_hand[is_access]
        for layer_id in np.unique(access_targets):
            lm = access_targets == layer_id
            layer_mo = access_mo[lm]
            layer_hands = access_hands[lm]
            has_toggle = not bool(np.all(layer_mo))
            mo_hands = layer_hands[layer_mo]
            if has_toggle or len(mo_hands) != 1:
                continue
            hand = int(mo_hands[0])
            thumb_exclude |= (
                self._mutable_is_thumb
                & (self._mutable_layer == layer_id)
                & (self._mutable_hand == hand)
                & (self._mutable_layer > 0)
            )
        return thumb_exclude

    def _ensure_return_toggle(self, genome, layer):
        """Ensure layer has a return-to-L0 toggle after a toggle-access to it is placed.

        When mutation places a toggle to layer X on L0, the layout needs a
        corresponding @access:L0:toggle on layer X so the user can get back.
        This enforces structural pairing without hard constraints: mutation
        proposes it, fitness scoring and selection decide if it's worth keeping.
        """
        if self._return_toggle_sid is None:
            return
        layer_positions = self._layer_mutable_positions.get(layer, [])
        if not layer_positions:
            return
        # No-op if already placed on this layer
        for pos in layer_positions:
            if genome[pos] == self._return_toggle_sid:
                return
        # Prefer empty positions, fall back to any position on the layer
        empty = [pos for pos in layer_positions if genome[pos] < 0]
        pool = empty if empty else layer_positions
        genome[random.choice(pool)] = self._return_toggle_sid

    def _random_reassign_one(self, genome):
        """Generic multiplicity mutation: let shortcut counts evolve."""
        candidates = self._reassign_candidates(genome)
        if candidates is None or len(candidates) == 0:
            return False
        # Bug 4: avoid displacing the sole access path to any layer
        for _ in range(5):
            pos = int(candidates[np.random.randint(len(candidates))])
            if not self._would_break_reachability(genome, pos):
                break
        else:
            return False
        pos_layer = int(self._pos_layer_arr[pos])
        # Exclude momentary access sids that would self-ref on this layer
        valid_assignable = self._assignable_arr
        if self.n_shortcuts > 0:
            mo_tgts = self._mo_access_target_lut[valid_assignable]
            valid_assignable = valid_assignable[~((mo_tgts >= 0) & (mo_tgts == pos_layer))]
        if len(valid_assignable) == 0:
            valid_assignable = self._assignable_arr
        new_sid = int(valid_assignable[np.random.randint(len(valid_assignable))])
        genome[pos] = new_sid
        # Toggle pairing: if we just placed a toggle-access to layer X, ensure X has a return
        if new_sid in self._toggle_access_sids:
            target_layer = self._access_sid_targets.get(new_sid)
            if target_layer is not None:
                self._ensure_return_toggle(genome, target_layer)
        return True

    def _bulk_reassign(self, genome):
        """Generic larger count-changing mutation for escaping duplicate basins."""
        candidates = self._reassign_candidates(genome)
        if candidates is None or len(candidates) < 2:
            return False
        n_change = random.randint(2, min(8, len(candidates)))
        chosen = candidates[np.random.choice(len(candidates), n_change, replace=False)]
        new_sids = np.empty(n_change, dtype=np.int32)
        for _k, _pos in enumerate(chosen):
            _layer = int(self._pos_layer_arr[_pos])
            _pool = self._assignable_arr
            if self.n_shortcuts > 0:
                _mo = self._mo_access_target_lut[_pool]
                _filtered = _pool[~((_mo >= 0) & (_mo == _layer))]
                if len(_filtered) > 0:
                    _pool = _filtered
            new_sids[_k] = int(_pool[np.random.randint(len(_pool))])
        genome[chosen] = new_sids
        # Toggle pairing: ensure return toggles for any newly placed toggle-access sids
        for new_sid in new_sids:
            if int(new_sid) in self._toggle_access_sids:
                target_layer = self._access_sid_targets.get(int(new_sid))
                if target_layer is not None:
                    self._ensure_return_toggle(genome, target_layer)
        return True

    def _smart_duplicate(self, genome):
        """Fill an empty mutable position using softmax-weighted shortcut selection.

        Non-group, non-mouse shortcuts are eligible. Mouse-button copies are
        handled by mouse-workflow mutation/scoring so generic duplication does
        not scatter MB1-MB5 across unrelated layers. Selection weight is
        softmax(imp / T) discounted quadratically by total placement count
        (mutable + frozen).
        High-importance shortcuts dominate but never monopolise. _base_* keys
        start with count=1 from their frozen L0 placement, giving them naturally
        lower probability than unplaced shortcuts of equal importance.

        Targets the lowest-effort empty mutable positions first.
        """
        if len(self._dup_candidate_arr) == 0:
            return False

        mutable_sids = genome[self._mutable_arr]
        empty_mask = mutable_sids < 0
        if not empty_mask.any():
            return False

        empty_positions = self._mutable_arr[empty_mask]
        efforts = self._pos_effort_arr[empty_positions]
        top_n = max(1, len(empty_positions) // 3)
        order = np.argpartition(efforts, min(top_n - 1, len(efforts) - 1))[:top_n]
        target_pos = int(empty_positions[order[np.random.randint(len(order))]])
        target_layer = int(self._pos_layer_arr[target_pos])

        # Total count = mutable placements + frozen placements (so _base_* start at 1)
        valid_sids = mutable_sids[mutable_sids >= 0]
        mutable_counts = (
            np.bincount(valid_sids.astype(np.int64), minlength=self.n_shortcuts)
            if len(valid_sids) > 0 else np.zeros(self.n_shortcuts, dtype=np.int64)
        )
        n = len(self._frozen_sid_counts)
        if n > 0 and n <= self.n_shortcuts:
            total_counts = mutable_counts[:n] + self._frozen_sid_counts
        else:
            total_counts = mutable_counts

        # Softmax(imp / T) / (1 + count)^2 — exp part is precomputed in __init__
        cnts = total_counts[self._dup_candidate_arr].astype(np.float32)
        count_discount = 1.0 + cnts
        weights = self._dup_exp_w / (count_discount * count_discount)

        # No shortcut may appear more than once on the same layer (L7 is
        # frozen/excluded; mouse buttons never reach this pool — see docstring
        # above). Zero out any candidate that already occupies target_layer.
        if target_layer != 7:
            on_target_layer = mutable_sids[self._pos_layer_arr[self._mutable_arr] == target_layer]
            already_present = np.isin(self._dup_candidate_arr, on_target_layer)
            weights = np.where(already_present, 0.0, weights)

        total_w = float(weights.sum())
        if total_w <= 0.0:
            return False

        probs = weights / total_w
        chosen_sid = int(np.random.choice(self._dup_candidate_arr, p=probs))

        genome[target_pos] = chosen_sid
        return True

    def _drop_optional_raw_arrows(self, genome):
        """Mutation proposal: let L7 be the only raw-arrow source."""
        if len(self._raw_arrow_arr) == 0 or len(self._assignable_not_arrow) == 0:
            return False
        mutable_sids = genome[self._mutable_arr]
        is_raw = np.isin(mutable_sids, self._raw_arrow_arr)
        if not is_raw.any():
            return False
        arrow_positions = self._mutable_arr[is_raw]
        repl = self._assignable_not_arrow
        genome[arrow_positions] = repl[np.random.randint(0, len(repl), len(arrow_positions))]
        return True

    def _overwrite_group_as_unit(self, genome, row_pos_map=None):
        """Move or inject a randomly chosen group to a valid anchor position.

        Works whether the group is currently present or absent in the genome.
        When absent (sids were dropped), injects all members at a valid anchor.
        When present, moves all members to a different valid anchor.
        Displaced sids fill the vacated positions; extras become -1.
        """
        if not self.group_sid_sets:
            return False
        group_idx = random.randrange(len(self.group_sid_sets))
        group_sids = self.group_sid_sets[group_idx]
        anchors_arr = self.group_anchor_arrays[group_idx]
        if len(anchors_arr) == 0:
            return False
        group_sid_set = set(group_sids)

        # Find current genome positions for each group sid (None = absent).
        # Reuse a batched sid→position map when available.
        pos_map = self._genome_pos_map(genome, row_pos_map)
        current_positions = [
            (int(pos_map[sid]) if 0 <= sid < self.n_shortcuts and pos_map[sid] >= 0 else None)
            for sid in group_sids
        ]

        # Vectorized anchor validity: an anchor is valid if none of its positions
        # currently holds a member of this group.
        occupied = np.isin(genome[anchors_arr], group_sids).any(axis=1)
        alternatives = [anchors_arr[i].tolist() for i in range(len(anchors_arr)) if not occupied[i]]
        if not alternatives:
            return False
        target = list(random.choice(alternatives))

        # Sids currently at target positions (will be displaced by group placement).
        displaced = [int(genome[pos]) for pos in target]

        # Place the whole group at the target.
        for sid, pos in zip(group_sids, target):
            genome[pos] = sid

        # Vacated positions: where group sids were before (excluding target overlap).
        vacated = [p for p in current_positions if p is not None and p not in target]

        # Put displaced non-group sids into vacated positions.
        fill_sids = [s for s in displaced if s not in group_sid_set]
        for pos, sid in zip(vacated, fill_sids):
            genome[pos] = sid
        # Positions that can't be filled become -1 (unassigned — valid in genome).
        for pos in vacated[len(fill_sids):]:
            genome[pos] = -1

        return True

    def _cluster_app_shortcut(self, genome, row_pos_map=None, row_thumb_exclude=None):
        """Move the most-outlier shortcut of a random app toward the app's physical centroid.

        Fully vectorized: one _genome_pos_map call replaces per-app mutable_list iteration.
        Thumb exclusion mask and occupied thumbs computed once per genome, not per app.
        When called from the batched ``_do`` path, the per-row pos_map and thumb-exclude
        masks are passed in, avoiding repeated full-genome scans.
        """
        if not self._app_ids:
            return False

        # Build sid→pos map once (numpy) — replaces O(n_apps × n_mutable) Python loops.
        pos_map = self._genome_pos_map(genome, row_pos_map)

        # Thumb exclusion mask computed directly without intermediate dict.
        thumb_exclude = self._thumb_exclude_mask(genome, row_thumb_exclude)

        # Group-member exclusion mask over mutable positions.
        mutable_sids = genome[self._mutable_arr]
        safe_ms = np.where(mutable_sids >= 0, mutable_sids, 0)
        is_group_m = self._is_group_sid_lut[safe_ms] & (mutable_sids >= 0)
        base_exclude = is_group_m | thumb_exclude  # positions never valid as targets

        # Sample a random subset of apps per call.  The original code looped over
        # all apps; with a batched sid→position map the per-app work is cheaper,
        # but scanning every app for every candidate genome still adds up.  A
        # sample of 12 gives nearly the full search breadth while keeping the
        # per-generation cost well below the old path.
        app_ids = self._app_ids
        n_sample = min(12, len(app_ids))
        app_ids = random.sample(app_ids, n_sample)
        for app_id in app_ids:
            app_sids = self._app_sids_arrs[app_id]
            present_mask = pos_map[app_sids] >= 0
            if present_mask.sum() < 2:
                continue
            present_sids = app_sids[present_mask]
            present_positions = pos_map[present_sids]

            xs = self._pos_x[present_positions]
            ys = self._pos_y[present_positions]
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))

            dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
            max_dist = float(dists.max())
            if max_dist <= 0.5:
                continue
            oi = int(np.argmax(dists))
            outlier_sid = int(present_sids[oi])
            outlier_pos = int(present_positions[oi])

            # Candidate positions: mutable, not outlier, not group, not excluded thumb.
            not_outlier = self._mutable_arr != outlier_pos
            candidate_mask = not_outlier & ~base_exclude
            candidate_positions = self._mutable_arr[candidate_mask]
            if len(candidate_positions) == 0:
                continue

            cand_xs = self._pos_x[candidate_positions]
            cand_ys = self._pos_y[candidate_positions]
            cand_dists = np.sqrt((cand_xs - cx) ** 2 + (cand_ys - cy) ** 2)
            if float(cand_dists.min()) >= max_dist:
                continue

            best_target = int(candidate_positions[int(np.argmin(cand_dists))])
            displaced = int(genome[best_target])
            genome[best_target] = outlier_sid
            genome[outlier_pos] = displaced
            return True
        return False

    @staticmethod
    def _mouse_ideal_penalty(button, x, y):
        """Mirrors fitness/kernel.py's MB1/MB2/MB3 ideal-position weights.

        Used only to break effort-ties in _effort_swap -- MB1 and MB3 (and MB2)
        can share effort=0.0, which leaves the effort-only swap criterion with
        zero gradient to ever notice one is squatting on another's ideal slot.
        """
        if button == 1:
            return abs(x - 8.0) * 28000.0 + abs(y - 2.0) * 30000.0
        if button == 2:
            return abs(x - 9.0) * 32000.0 + abs(y - 2.0) * 30000.0
        if button == 3:
            return abs(x - 11.0) * 6000.0 + abs(y - 2.0) * 5000.0
        return -1.0

    def _detect_mouse_layer(self, genome):
        """Return the candidate mouse layer index (or -1) by replicating the kernel pre-scan.

        Called once per _effort_swap invocation so the mutation applies the same
        dynamic 3× importance boost the kernel uses, making MB1 visible as the
        highest-cost target when it sits at an elevated effort position.
        """
        if not self.mouse_button_sids:
            return -1
        mouse_sids = np.fromiter(self.mouse_button_sids.values(), dtype=np.int32)
        genome_arr = np.asarray(genome, dtype=np.int32)
        is_mouse = np.isin(genome_arr, mouse_sids)
        if not is_mouse.any():
            return -1
        valid = (
            is_mouse
            & (self._pos_layer_arr > 0)
            & (self._pos_layer_arr != 7)
            & (self._pos_layer_arr < 32)
            & (self._pos_hand_arr == 1)
            & (~self._pos_is_thumb_arr)
        )
        if not valid.any():
            return -1
        counts = np.bincount(self._pos_layer_arr[valid], minlength=32)
        best_mc = int(counts.max())
        if best_mc < 2:
            return -1
        return int(np.argmax(counts))

    def _swap_creates_illegal_duplicate(self, genome, src_pos, target_pos, mouse_layer):
        """True if swapping genome[src_pos] <-> genome[target_pos] would put the
        same shortcut twice on one layer. No shortcut may appear more than once
        on the same layer, except one left+one right copy of a core mouse
        button on the dynamic mouse layer; layer 7 is frozen and excluded.
        """
        moving_sid = int(genome[src_pos])
        displaced_sid = int(genome[target_pos])

        def layer_count(layer, sid):
            # Scoped to this layer's mutable positions only (precomputed
            # per-layer index), not a full-genome scan.
            if layer < 0 or layer + 1 >= len(self._layer_mutable_start):
                return 0
            start = self._layer_mutable_start[layer]
            end = self._layer_mutable_start[layer + 1]
            layer_positions = self._layer_mutable_flat[start:end]
            count = int(np.sum(genome[layer_positions] == sid))
            if int(genome[src_pos]) == sid and layer_positions.size and src_pos in layer_positions:
                count -= 1
            if int(genome[target_pos]) == sid and layer_positions.size and target_pos in layer_positions:
                count -= 1
            return count

        target_layer = int(self._pos_layer_arr[target_pos])
        if target_layer != 7:
            cap = 2 if (mouse_layer >= 0 and target_layer == mouse_layer
                        and moving_sid in self.mouse_button_sids.values()) else 1
            if layer_count(target_layer, moving_sid) + 1 > cap:
                return True

        src_layer = int(self._pos_layer_arr[src_pos])
        if src_layer != 7:
            cap = 2 if (mouse_layer >= 0 and src_layer == mouse_layer
                        and displaced_sid in self.mouse_button_sids.values()) else 1
            if layer_count(src_layer, displaced_sid) + 1 > cap:
                return True
        return False

    def _effort_swap(self, genome):
        """Propose swapping a high importance×effort shortcut with a lower-effort position.

        Uses the same dynamic 3× mouse layer boost as the kernel so MB1 is correctly
        prioritised when it sits at a worse position than lower-importance mouse buttons.
        """
        if len(self._mutable_arr) < 2 or len(self._sid_importance_arr) == 0:
            return False
        mutable_sids = genome[self._mutable_arr]
        valid = (mutable_sids >= 0) & (mutable_sids < self.n_shortcuts)
        if not valid.any():
            return False
        # Exclude group members from consideration
        safe_sids = np.where(valid, mutable_sids, 0)
        valid &= ~self._is_group_sid_lut[safe_sids]
        if not valid.any():
            return False

        valid_positions = self._mutable_arr[valid]
        valid_sids = mutable_sids[valid]
        pos_efforts = self._pos_effort_arr[valid_positions]

        # Apply dynamic mouse-layer 3× boost so MB1 on the mouse layer is
        # correctly identified as the highest-cost target. Also track which
        # button (1/2/3) each valid position holds, and add its ideal-position
        # penalty to cost: MB1/MB2/MB3 can sit at effort=0.0 while badly
        # misplaced (e.g. squatting on another button's ideal slot), and
        # effort*importance alone is then always 0 -- this position could
        # never even be considered as a swap source without it.
        sid_imps = self._sid_importance_arr[valid_sids].copy()
        button_of = np.full(len(valid_positions), -1, dtype=np.int32)
        mouse_layer = -1
        if self.mouse_button_sids:
            mouse_layer = self._detect_mouse_layer(genome)
            if mouse_layer >= 0:
                sid_to_button = {sid: b for b, sid in self.mouse_button_sids.items()}
                mouse_sids = np.fromiter(self.mouse_button_sids.values(), dtype=np.int32)
                valid_layers = self._pos_layer_arr[valid_positions]
                boost_mask = np.isin(valid_sids, mouse_sids) & (valid_layers == mouse_layer)
                sid_imps[boost_mask] *= 3.0
                for k in np.nonzero(boost_mask)[0]:
                    b = sid_to_button.get(int(valid_sids[k]), -1)
                    if 1 <= b <= 3:
                        button_of[k] = b

        costs = pos_efforts * sid_imps
        ideal_mask = button_of >= 1
        if ideal_mask.any():
            idx = np.nonzero(ideal_mask)[0]
            xs = self._pos_x[valid_positions[idx]]
            ys = self._pos_y[valid_positions[idx]]
            for j, k in enumerate(idx):
                costs[k] += self._mouse_ideal_penalty(int(button_of[k]), float(xs[j]), float(ys[j]))

        # Pick one of the top-5 highest-cost shortcuts at random (not deterministic max,
        # which would always target the same shortcut and prevent exploration).
        top_k = min(5, len(costs))
        top_idx = np.argpartition(costs, -top_k)[-top_k:]
        chosen_i = int(top_idx[random.randrange(top_k)])
        src_pos = int(valid_positions[chosen_i])
        src_effort = float(pos_efforts[chosen_i])
        src_button = int(button_of[chosen_i])

        if src_button < 1 and src_effort <= 0.0:
            return False

        # Target: prefer same-layer lower-effort positions to preserve adjacency clusters.
        # Fall back to cross-layer if no same-layer candidates exist.
        lower_mask = pos_efforts < src_effort
        lower_mask[chosen_i] = False
        candidates = valid_positions[lower_mask]

        if len(candidates) == 0 and src_button >= 1:
            # Effort-tie fallback, mouse buttons only: no strictly-lower-effort
            # target exists (src is often already at effort=0.0, the minimum),
            # so look for an equal-effort position whose swap would reduce
            # this button's ideal-position penalty instead.
            src_penalty = self._mouse_ideal_penalty(src_button, float(self._pos_x[src_pos]), float(self._pos_y[src_pos]))
            tie_mask = (pos_efforts == src_effort)
            tie_mask[chosen_i] = False
            tie_positions = valid_positions[tie_mask]
            if len(tie_positions) > 0:
                cand_penalties = np.array(
                    [self._mouse_ideal_penalty(src_button, float(self._pos_x[p]), float(self._pos_y[p])) for p in tie_positions],
                    dtype=np.float64,
                )
                candidates = tie_positions[cand_penalties < src_penalty]

        if len(candidates) == 0:
            return False

        src_layer = int(self._pos_layer_arr[src_pos])
        same_layer_mask = self._pos_layer_arr[candidates] == src_layer
        same_layer_candidates = candidates[same_layer_mask]
        mouse_layer = self._detect_mouse_layer(genome) if self.mouse_button_sids else -1

        # Try a bounded number of candidate targets (respecting the same-layer
        # preference each attempt) and skip any that would create an illegal
        # same-layer duplicate.
        target_pos = -1
        for _attempt in range(8):
            if len(same_layer_candidates) > 0 and random.random() < 0.75:
                candidate = int(same_layer_candidates[random.randrange(len(same_layer_candidates))])
            else:
                candidate = int(candidates[random.randrange(len(candidates))])
            if not self._swap_creates_illegal_duplicate(genome, src_pos, candidate, mouse_layer):
                target_pos = candidate
                break
        if target_pos < 0:
            return False

        displaced = int(genome[target_pos])
        genome[target_pos] = int(genome[src_pos])
        genome[src_pos] = displaced
        return True

    def sanitize_self_ref_momentary(self, genome):
        """Remove any momentary hold key placed on its own target layer (illegal placement).

        Called once on warmstart genomes. Replaces each violation with -1 (empty),
        letting subsequent mutations and smart_duplicate fill the slot properly.
        Returns the number of violations cleared.
        """
        cleared = 0
        for i, sid in enumerate(genome):
            if sid < 0 or sid >= self.n_shortcuts:
                continue
            tgt = int(self._mo_access_target_lut[sid])
            if tgt >= 0 and tgt == int(self._pos_layer_arr[i]):
                genome[i] = -1
                cleared += 1
        return cleared

    def _bias_toggle_to_own_layer(self, genome):
        """Search bias: move toggle-to-LX keys toward positions on layer X.

        @LX:toggle on layer X is desirable — it gives the layer a self-locking
        mechanism. This mutation finds toggle keys that are NOT on their target
        layer and proposes swapping them to a position on that layer.
        Prefers empty positions; falls back to any mutable position on the layer.
        The fitness decides whether the result is worth keeping.
        """
        # Find toggle access sids (non-momentary, not @L0, not @L7) NOT on own layer
        genome_arr = np.asarray(genome, dtype=np.int32)
        valid = (genome_arr >= 0) & (genome_arr < self.n_shortcuts)
        if not valid.any():
            return False
        sids = genome_arr
        tgts = self._access_target_lut[sids]
        cur_layers = self._pos_layer_arr
        is_toggle = valid & (tgts > 0) & (tgts != 7) & (~self._access_is_mo_lut[sids])
        not_on_own = is_toggle & (cur_layers != tgts)
        candidate_idx = np.where(not_on_own)[0]
        if len(candidate_idx) == 0:
            return False
        src_pos = int(random.choice(candidate_idx))
        target_layer = int(tgts[src_pos])
        positions_on_target = self._layer_mutable_positions.get(target_layer, [])
        if not positions_on_target:
            return False
        positions_on_target = np.asarray(positions_on_target, dtype=np.int32)
        empty = positions_on_target[(genome_arr[positions_on_target] < 0) & (positions_on_target != src_pos)]
        any_pos = positions_on_target[positions_on_target != src_pos]
        pool = empty if len(empty) > 0 else any_pos
        if len(pool) == 0:
            return False
        tgt_pos = int(random.choice(pool))
        genome[src_pos], genome[tgt_pos] = int(genome[tgt_pos]), int(genome[src_pos])
        return True

    def _bias_access_to_thumb(self, genome):
        """Search bias (not rule): move a non-thumb access key toward a thumb position.

        Proposes swapping a layer-access shortcut from a finger position to a thumb
        position on the same or any layer. The fitness function accepts or rejects —
        this only biases the search direction. Fires with access_thumb_bias_prob.
        Prefers same-layer targets to preserve layer coherence; falls back cross-layer.
        """
        if len(self._mutable_arr) == 0:
            return False
        mutable_sids = genome[self._mutable_arr]
        valid = (mutable_sids >= 0) & (mutable_sids < self.n_shortcuts)
        safe_sids = np.where(valid, mutable_sids, 0)
        # Non-thumb mutable positions holding an access key
        is_access = valid & (self._access_target_lut[safe_sids] > 0)
        is_nonthumb = ~self._pos_is_thumb_arr[self._mutable_arr]
        nonthumb_access_mask = is_access & is_nonthumb
        if not nonthumb_access_mask.any():
            return False
        candidates = self._mutable_arr[nonthumb_access_mask]
        src_pos = int(candidates[np.random.randint(len(candidates))])
        src_layer = int(self._pos_layer_arr[src_pos])
        # Find mutable thumb positions (prefer same layer)
        is_group_m = self._is_group_sid_lut[np.where(valid, safe_sids, 0)] & valid
        is_thumb_m = self._pos_is_thumb_arr[self._mutable_arr]
        not_src = self._mutable_arr != src_pos
        thumb_pool = self._mutable_arr[is_thumb_m & not_src & ~is_group_m]
        if len(thumb_pool) == 0:
            return False
        same_layer = thumb_pool[self._pos_layer_arr[thumb_pool] == src_layer]
        pool = same_layer if len(same_layer) > 0 else thumb_pool
        tgt_pos = int(pool[np.random.randint(len(pool))])
        genome[src_pos], genome[tgt_pos] = int(genome[tgt_pos]), int(genome[src_pos])
        return True

    def _repair_missing_return_toggles(self, genome):
        """Detect layers with toggle access but no return-to-L0 toggle and add one.

        Scans the genome for toggle-accessible layers missing @access:L0:return.
        Places sid=_return_toggle_sid on a mutable position on the affected layer.
        Prefers empty thumb positions, then any empty, then any mutable position.
        """
        if self._return_toggle_sid is None:
            return False
        ret_sid = self._return_toggle_sid
        # Find toggle-accessible layers
        genome_arr = np.asarray(genome, dtype=np.int32)
        valid = (genome_arr >= 0) & (genome_arr < self.n_shortcuts)
        if not valid.any():
            return False
        sids = genome_arr
        tgts = self._access_target_lut[sids]
        is_toggle = valid & (tgts > 0) & (~self._access_is_mo_lut[sids])
        if not is_toggle.any():
            return False
        cur_layers = self._pos_layer_arr
        toggle_to = set(cur_layers[is_toggle & (tgts != 0)].tolist())
        has_return = set(cur_layers[is_toggle & (tgts == 0)].tolist())
        missing = [lyr for lyr in toggle_to if lyr not in has_return
                   and lyr in self._layer_mutable_positions]
        if not missing:
            return False
        lyr = random.choice(missing)
        positions = np.asarray(self._layer_mutable_positions[lyr], dtype=np.int32)
        values = genome_arr[positions]
        thumb_empty = positions[(values < 0) & self._pos_is_thumb_arr[positions]]
        any_empty = positions[values < 0]
        pool = thumb_empty if len(thumb_empty) > 0 else (any_empty if len(any_empty) > 0 else positions)
        genome[int(random.choice(pool))] = ret_sid
        return True

    def _do(self, problem, X, **kwargs):
        n = X.shape[0]
        prob = float(self.prob.value if hasattr(self.prob, "value") else self.prob)
        handled = np.zeros(n, dtype=np.bool_)

        # Pass 1: complex semantic mutations.  When Numba is available we run a
        # single parallel dispatcher over the whole batch; otherwise fall back to
        # per-candidate Python attempt loops.
        if NUMBA_AVAILABLE and n > 0:
            semantic_probs = np.array([
                self.mouse_workflow_prob,
                self.l7_access_prob,
                self.group_overwrite_prob,
                self.optional_arrow_drop_prob,
                self.bulk_assign_prob,
                self.cluster_app_prob,
            ], dtype=np.float64)
            pos_maps = self._batched_pos_map(X)
            thumb_excludes = self._batched_thumb_exclude(X)
            seeds = np.random.randint(0, 2**63, size=n, dtype=np.uint64)
            _semantic_mutations_batch_numba(
                X,
                handled,
                semantic_probs,
                seeds,
                pos_maps,
                thumb_excludes,
                self._mouse_candidate_layers,
                self._right_non_thumb_flat,
                self._right_non_thumb_start,
                self._right_positions_flat,
                self._right_positions_start,
                self._right_thumb_positions_flat,
                self._right_thumb_positions_start,
                self._safe_access_positions_arr,
                self._l0_safe_access_positions_arr,
                self._layer_access_hold,
                self._layer_access_toggle,
                self._layer_scroll_access,
                self._return_toggle_sid_arr,
                self._mouse_button_sids,
                self._group_sids_arr,
                self._group_sizes,
                self._group_anchors_flat,
                self._group_anchor_start,
                np.int32(self._group_sizes.shape[0]),
                self._app_sids_flat,
                self._app_sids_start,
                np.int32(self._n_apps),
                np.int32(min(12, self._n_apps)),
                self._pos_x,
                self._pos_y,
                self._mutable_arr,
                self._is_group_sid_lut,
                np.int32(self.n_shortcuts),
                self._is_raw_arrow_lut,
                self._assignable_not_arrow,
                self._assignable_arr,
                self._pos_layer_arr,
                self._mo_access_target_lut,
                self._is_important_sid_lut,
                self._pos_effort_arr,
            )
        else:
            def _attempt(mutation_fn, prob, need_pos_map=False, need_thumb=False):
                if prob <= 0.0:
                    return
                candidates = np.where(~handled & (np.random.random(n) < prob))[0]
                if len(candidates) == 0:
                    return
                pm = self._batched_pos_map(X[candidates]) if need_pos_map else None
                te = self._batched_thumb_exclude(X[candidates]) if need_thumb else None
                for k, i in enumerate(candidates):
                    kwargs_call = {}
                    if need_pos_map:
                        kwargs_call["row_pos_map"] = pm[k]
                    if need_thumb:
                        kwargs_call["row_thumb_exclude"] = te[k]
                    if mutation_fn(X[i], **kwargs_call):
                        handled[i] = True

            _attempt(self._propose_mouse_workflow_layer, self.mouse_workflow_prob, need_pos_map=True)
            _attempt(self._propose_l7_access, self.l7_access_prob, need_pos_map=True)
            _attempt(self._overwrite_group_as_unit, self.group_overwrite_prob, need_pos_map=True)
            _attempt(self._drop_optional_raw_arrows, self.optional_arrow_drop_prob)
            _attempt(self._bulk_reassign, self.bulk_assign_prob)

            # Pass 1b: app-cluster mutation.  Same candidate-batch strategy; the
            # thumb-exclusion mask is built only for rows that attempt clustering.
            if self.cluster_app_prob > 0.0:
                _attempt(self._cluster_app_shortcut, self.cluster_app_prob, need_pos_map=True, need_thumb=True)

        # Pass 2: Numba-parallel simple mutations for remaining unhandled genomes
        if NUMBA_AVAILABLE and n > 0 and np.any(~handled):
            probs = np.array([
                self.random_assign_prob,
                self.effort_swap_prob,
                self.smart_duplicate_prob,
                self.toggle_own_layer_bias_prob,
                self.access_thumb_bias_prob,
                self.return_toggle_repair_prob,
            ], dtype=np.float64)
            seeds = np.random.randint(0, 2**63, size=n, dtype=np.uint64)
            _mutate_batch_numba(
                X,
                handled,
                probs,
                seeds,
                self._mutable_arr,
                self._pos_layer_arr,
                self._pos_hand_arr,
                self._pos_is_thumb_arr,
                self._pos_effort_arr,
                self._pos_x,
                self._pos_y,
                self._sid_importance_arr,
                self._access_target_lut,
                self._access_is_mo_lut,
                self._mo_access_target_lut,
                self._is_group_sid_lut,
                self._is_important_sid_lut,
                np.int32(self._return_toggle_sid if self._return_toggle_sid is not None else -1),
                self._dup_candidate_arr,
                self._dup_exp_w,
                self._frozen_sid_counts,
                self._assignable_arr,
                self._layer_mutable_flat,
                self._layer_mutable_start,
                self._mouse_button_sids,
                self._toggle_access_sids_arr,
                np.int32(self.n_shortcuts),
            )
        else:
            # Pure-Python fallback (also used when Numba is unavailable)
            for i in range(n):
                if handled[i]:
                    continue
                if random.random() < self.random_assign_prob and self._random_reassign_one(X[i]):
                    handled[i] = True
                    continue
                if random.random() < self.effort_swap_prob and self._effort_swap(X[i]):
                    handled[i] = True
                    continue
                if random.random() < self.smart_duplicate_prob and self._smart_duplicate(X[i]):
                    handled[i] = True
                    continue
                if random.random() < self.toggle_own_layer_bias_prob and self._bias_toggle_to_own_layer(X[i]):
                    handled[i] = True
                    continue
                if random.random() < self.access_thumb_bias_prob and self._bias_access_to_thumb(X[i]):
                    handled[i] = True
                    continue
                if random.random() < self.return_toggle_repair_prob and self._repair_missing_return_toggles(X[i]):
                    handled[i] = True
                    continue

        # Pass 3: vectorized swap for remaining unhandled genomes
        m = len(self._mutable_arr)
        if m < 2:
            return X
        rows = np.where(~handled)[0]
        if len(rows) == 0:
            return X
        swap_mask = np.random.random(len(rows)) < prob
        swap_rows = rows[swap_mask]
        if len(swap_rows) > 0:
            a_idx = np.random.randint(0, m, len(swap_rows))
            b_idx = np.random.randint(0, m, len(swap_rows))
            a_pos = self._mutable_arr[a_idx]
            b_pos = self._mutable_arr[b_idx]
            a_sids = X[swap_rows, a_pos]
            b_sids = X[swap_rows, b_pos]
            valid = np.ones(len(swap_rows), dtype=np.bool_)
            if len(self._group_arr) > 0:
                valid &= ~(np.isin(a_sids, self._group_arr) | np.isin(b_sids, self._group_arr))
            # Block swaps that would place a momentary hold key on its own target layer
            if self.n_shortcuts > 0:
                a_safe = np.where((a_sids >= 0) & (a_sids < self.n_shortcuts), a_sids, 0)
                b_safe = np.where((b_sids >= 0) & (b_sids < self.n_shortcuts), b_sids, 0)
                a_mo_tgt = self._mo_access_target_lut[a_safe]
                b_mo_tgt = self._mo_access_target_lut[b_safe]
                b_layer = self._pos_layer_arr[b_pos]
                a_layer = self._pos_layer_arr[a_pos]
                valid &= ~((a_mo_tgt >= 0) & (a_mo_tgt == b_layer))
                valid &= ~((b_mo_tgt >= 0) & (b_mo_tgt == a_layer))
            vr = swap_rows[valid]
            va = a_pos[valid]
            vb = b_pos[valid]
            if len(vr) > 0:
                tmp = X[vr, va].copy()
                X[vr, va] = X[vr, vb]
                X[vr, vb] = tmp
        return X


class StructuralGenomeSanitizer(Repair):
    """Minimal structural validity guard.

    This is not a layout repair mechanism.  It never moves mouse buttons,
    arrows, completion keys, workflow groups, or any semantic shortcut into a
    preferred place.  It only preserves immutable genome invariants that should
    never be part of search: valid SID bounds, frozen L0 base assignments, and
    no duplication of frozen base-key SIDs into mutable slots.
    """

    def __init__(self, n_shortcuts, frozen_mask=None, seed_genome=None, layout=None):
        super().__init__()
        self.n_shortcuts = n_shortcuts
        self.frozen_mask = frozen_mask
        self.seed_genome = seed_genome
        self.frozen = np.where(frozen_mask)[0] if frozen_mask is not None else np.array([], dtype=int)
        self.mutable = np.where(~frozen_mask)[0] if frozen_mask is not None else np.array([], dtype=int)
        if self.seed_genome is not None and len(self.frozen) > 0:
            self.frozen_assigned = {
                int(sid) for sid in np.asarray(self.seed_genome, dtype=int)[self.frozen]
                if int(sid) >= 0
            }
        else:
            self.frozen_assigned = set()
        # Precomputed arrays for the Numba prange sanitizer kernel.
        self._frozen_sid_lut = np.zeros(n_shortcuts, dtype=np.bool_)
        for sid in self.frozen_assigned:
            if 0 <= sid < n_shortcuts:
                self._frozen_sid_lut[sid] = True
        self._frozen_idx = self.frozen.astype(np.int32)
        self._frozen_vals = (
            np.asarray(self.seed_genome, dtype=np.int32)[self._frozen_idx]
            if self.seed_genome is not None and len(self._frozen_idx) > 0
            else np.array([], dtype=np.int32)
        )
        self._mutable_idx = self.mutable.astype(np.int32)

    def _do(self, problem, X, **kwargs):
        if NUMBA_AVAILABLE and X.flags["C_CONTIGUOUS"]:
            _sanitize_batch_numba(
                X,
                self._frozen_idx,
                self._frozen_vals,
                self._mutable_idx,
                self._frozen_sid_lut,
                np.int32(self.n_shortcuts),
            )
            return X
        # Numpy fallback
        X[X >= self.n_shortcuts] = -1
        X[X < -1] = -1
        if len(self._frozen_idx) > 0 and self.seed_genome is not None:
            X[:, self._frozen_idx] = self._frozen_vals
        if self.frozen_assigned:
            mb = X[:, self._mutable_idx].copy()
            valid = mb >= 0
            mb[self._frozen_sid_lut[np.maximum(mb, 0)] & valid] = -1
            X[:, self._mutable_idx] = mb
        return X


def create_algorithm(n_positions, n_shortcuts, frozen_mask=None, seed_genome=None, inject_seed=True,
                     pop_size=500, crossover_prob=0.7, mutation_prob=0.15,
                     eliminate_duplicates=False, layout=None):
    sampling = PermutationSampling(n_shortcuts=n_shortcuts, frozen_mask=frozen_mask, seed_genome=seed_genome, inject_seed=inject_seed, layout=layout)
    crossover = CycleCrossover(prob=crossover_prob, n_shortcuts=n_shortcuts)
    mutation = SwapMutation(prob=mutation_prob, frozen_mask=frozen_mask, layout=layout)
    repair = StructuralGenomeSanitizer(n_shortcuts=n_shortcuts, frozen_mask=frozen_mask, seed_genome=seed_genome, layout=layout)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        repair=repair,
        eliminate_duplicates=eliminate_duplicates,
    )
    return algorithm
