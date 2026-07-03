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
        self.mutable_indices = np.where(~frozen_mask)[0] if frozen_mask is not None else None
        self.mutable_list = self.mutable_indices.tolist() if self.mutable_indices is not None else None
        self.group_sid_sets = []
        self.group_anchor_positions = []
        self.group_anchor_sets: list = []  # precomputed frozensets for fast intersection test
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
        # Precomputed arrays for vectorized mutation
        self.n_shortcuts = len(layout.shortcuts) if layout is not None else 0
        n_sc = self.n_shortcuts
        self._mutable_arr = np.array(self.mutable_list or [], dtype=np.int32)
        self._assignable_arr = np.array(self.assignable_sids or [], dtype=np.int32)
        self._important_arr = np.array(sorted(self.important_sids), dtype=np.int32)
        self._group_arr = np.array(sorted(self.group_member_sids), dtype=np.int32)
        self._raw_arrow_arr = np.array(sorted(self.raw_arrow_sids), dtype=np.int32)
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
        for _sid, _tgt in self._access_sid_targets.items():
            if 0 <= _sid < n_sc:
                self._access_target_lut[_sid] = int(_tgt)
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

    def _genome_pos_map(self, genome):
        """Build sid→position array in one numpy pass (replaces repeated np.where calls)."""
        pos_map = np.full(self.n_shortcuts, -1, dtype=np.int32)
        valid = genome >= 0
        sids = genome[valid].astype(np.int32)
        sids = np.clip(sids, 0, self.n_shortcuts - 1)
        pos_map[sids] = np.where(valid)[0].astype(np.int32)
        return pos_map

    def _place_sids(self, genome, sids, target_positions):
        """Move SIDs to positions by swapping displaced values into old slots."""
        if len(sids) != len(target_positions):
            return False
        # One numpy pass to find all sid positions (replaces len(sids) np.where calls)
        pos_map = self._genome_pos_map(genome)
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

    def _propose_mouse_workflow_layer(self, genome):
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
        target_positions.sort()  # position index order only — fitness scoring drives quality placement
        sids = [
            self.mouse_button_sids[1],
            self.mouse_button_sids[2],
            self.mouse_button_sids[3],
            self.mouse_button_sids[4],
            self.mouse_button_sids[5],
            self.scroll_access_by_target[layer],
        ]

        blocked = set(target_positions)
        # Bug 2 fix: use safe_access_positions (any non-frozen non-L7 position),
        # not l0_safe_access_positions. Fitness scoring determines optimal placement.
        safe_access = [pos for pos in self.safe_access_positions if pos not in blocked]
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
        return self._place_sids(genome, sids, target_positions)

    def _propose_l7_access(self, genome):
        if 7 not in self.access_hold_by_target or 7 not in self.access_toggle_by_target:
            return False
        # L7 (arrows) is always accessed from L0: prefer l0_safe_access_positions,
        # fall back to all safe positions. Bug 2 fix only applies to mouse workflow layer.
        access_pool = self.l0_safe_access_positions or self.safe_access_positions
        if len(access_pool) < 2:
            return False
        positions = random.sample(access_pool, 2)
        sids = [self.access_hold_by_target[7], self.access_toggle_by_target[7]]
        return self._place_sids(genome, sids, positions)

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
        genome[pos] = int(self._assignable_arr[np.random.randint(len(self._assignable_arr))])
        return True

    def _bulk_reassign(self, genome):
        """Generic larger count-changing mutation for escaping duplicate basins."""
        candidates = self._reassign_candidates(genome)
        if candidates is None or len(candidates) < 2:
            return False
        n_change = random.randint(2, min(8, len(candidates)))
        chosen = candidates[np.random.choice(len(candidates), n_change, replace=False)]
        genome[chosen] = self._assignable_arr[np.random.randint(0, len(self._assignable_arr), n_change)]
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

    def _overwrite_group_as_unit(self, genome):
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
        anchors = self.group_anchor_positions[group_idx]

        # Find current genome positions for each group sid (None = absent).
        # One numpy pass replaces len(group_sids) np.where calls.
        pos_map = self._genome_pos_map(genome)
        current_positions = [
            (int(pos_map[sid]) if 0 <= sid < self.n_shortcuts and pos_map[sid] >= 0 else None)
            for sid in group_sids
        ]

        present_set = {p for p in current_positions if p is not None}

        # Use precomputed frozensets — isdisjoint avoids per-iteration set() construction.
        anchor_sets = self.group_anchor_sets[group_idx]
        alternatives = [
            anchors[i] for i, aset in enumerate(anchor_sets)
            if aset.isdisjoint(present_set)
        ]
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
        fill_sids = [s for s in displaced if s not in set(group_sids)]
        for pos, sid in zip(vacated, fill_sids):
            genome[pos] = sid
        # Positions that can't be filled become -1 (unassigned — valid in genome).
        for pos in vacated[len(fill_sids):]:
            genome[pos] = -1

        return True
    
    def _cluster_app_shortcut(self, genome):
        """Move the most-outlier shortcut of a random app toward the app's physical centroid.

        Fully vectorized: one _genome_pos_map call replaces per-app mutable_list iteration.
        Thumb exclusion mask and occupied thumbs computed once per genome, not per app.
        """
        if not self._app_ids:
            return False

        # Build sid→pos map once (numpy) — replaces O(n_apps × n_mutable) Python loops.
        pos_map = self._genome_pos_map(genome)

        # Occupied thumbs and the resulting exclusion mask — both computed once per genome.
        occupied_thumbs = self._get_layer_occupied_thumbs(genome)
        is_thumb_m = self._pos_is_thumb_arr[self._mutable_arr]
        layers_m = self._pos_layer_arr[self._mutable_arr]
        hands_m = self._pos_hand_arr[self._mutable_arr]
        thumb_exclude = np.zeros(len(self._mutable_arr), dtype=np.bool_)
        for _lid, _occ in occupied_thumbs.items():
            for _h in _occ:
                thumb_exclude |= is_thumb_m & (layers_m == _lid) & (hands_m == _h) & (layers_m > 0)

        # Group-member exclusion mask over mutable positions.
        mutable_sids = genome[self._mutable_arr]
        safe_ms = np.where(mutable_sids >= 0, mutable_sids, 0)
        is_group_m = self._is_group_sid_lut[safe_ms] & (mutable_sids >= 0)
        base_exclude = is_group_m | thumb_exclude  # positions never valid as targets

        app_ids = list(self._app_ids)
        random.shuffle(app_ids)
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

    def _effort_swap(self, genome):
        """Propose swapping a high importance×effort shortcut with a lower-effort position.

        Addresses remaining effort cost without cheating: we don't hardcode positions.
        We sample from the HIGH-COST end (many worse shortcuts = more candidates), then
        pick the TARGET uniformly from any lower-effort position — fitness scoring and
        selection decide whether the trade-off (effort vs adjacency) is worth it.
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
        sid_imps = self._sid_importance_arr[valid_sids]
        costs = pos_efforts * sid_imps

        # Pick one of the top-5 highest-cost shortcuts at random (not deterministic max,
        # which would always target the same shortcut and prevent exploration).
        top_k = min(5, len(costs))
        top_idx = np.argpartition(costs, -top_k)[-top_k:]
        chosen_i = int(top_idx[random.randrange(top_k)])
        src_pos = int(valid_positions[chosen_i])
        src_effort = float(pos_efforts[chosen_i])

        if src_effort <= 0.0:
            return False

        # Target: prefer same-layer lower-effort positions to preserve adjacency clusters.
        # Fall back to cross-layer if no same-layer candidates exist.
        lower_mask = pos_efforts < src_effort
        lower_mask[chosen_i] = False
        candidates = valid_positions[lower_mask]
        if len(candidates) == 0:
            return False

        src_layer = int(self._pos_layer_arr[src_pos])
        same_layer_mask = self._pos_layer_arr[candidates] == src_layer
        same_layer_candidates = candidates[same_layer_mask]
        if len(same_layer_candidates) > 0 and random.random() < 0.75:
            target_pos = int(same_layer_candidates[random.randrange(len(same_layer_candidates))])
        else:
            target_pos = int(candidates[random.randrange(len(candidates))])

        displaced = int(genome[target_pos])
        genome[target_pos] = int(genome[src_pos])
        genome[src_pos] = displaced
        return True

    def _do(self, problem, X, **kwargs):
        n = X.shape[0]
        prob = float(self.prob.value if hasattr(self.prob, "value") else self.prob)
        handled = np.zeros(n, dtype=np.bool_)

        # Pass 1: complex semantic mutations — sequential, ~40% of genomes
        for i in range(n):
            if random.random() < self.mouse_workflow_prob and self._propose_mouse_workflow_layer(X[i]):
                handled[i] = True
                continue
            if random.random() < self.l7_access_prob and self._propose_l7_access(X[i]):
                handled[i] = True
                continue
            if random.random() < self.group_overwrite_prob and self._overwrite_group_as_unit(X[i]):
                handled[i] = True
                continue
            if random.random() < self.optional_arrow_drop_prob and self._drop_optional_raw_arrows(X[i]):
                handled[i] = True
                continue
            if random.random() < self.bulk_assign_prob and self._bulk_reassign(X[i]):
                handled[i] = True
                continue
            if random.random() < self.random_assign_prob and self._random_reassign_one(X[i]):
                handled[i] = True
                continue
            if random.random() < self.cluster_app_prob and self._cluster_app_shortcut(X[i]):
                handled[i] = True
                continue
            if random.random() < self.effort_swap_prob and self._effort_swap(X[i]):
                handled[i] = True
                continue

        # Pass 2: vectorized swap for unhandled genomes (~60%)
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
