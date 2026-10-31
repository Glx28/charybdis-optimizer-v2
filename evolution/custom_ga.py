"""Custom single-objective GA replacing pymoo NSGA2.

Eliminates pymoo's Population objects, Pareto sort, crowding distance,
and Python tournament loop. Parent selection, crossover, mutation, repair,
and survival run on CPU; exact fitness evaluation runs on CUDA when enabled.

Expected speedup vs pymoo NSGA2: ~2x (from ~74ms/gen to ~38ms/gen).
Phase-2 vectorized swap brings this to ~3x (~23ms/gen).
"""

import glob
import json
import os
import random
import time
from collections import OrderedDict

import numpy as np

from evolution.acceptance import build_acceptance_report
from evolution.arrow_cluster import analyze_arrows
from evolution.completion_cluster import (
    analyze_completion_cluster, completion_order, is_raw_completion_base,
)
from evolution.group_shapes import NORWEGIAN_CLUSTER_OFFSETS
from evolution import NUMBA_AVAILABLE
from tools.semantic_cluster_report import _cluster_quality

if NUMBA_AVAILABLE:
    from evolution import _cycle_crossover_pair_numba, _cycle_crossover_batch_numba
    from numba import njit

    @njit(cache=True)
    def _completion_cluster_pass_fast(
        genome, shortcut_order, pos_layer, pos_x, pos_y, pos_frozen, offsets,
    ):
        """Match the completion acceptance check without allocating a report."""
        first_pos = np.full(len(shortcut_order), -1, dtype=np.int32)
        selected_sid = np.full(6, -1, dtype=np.int32)
        layer_seen = np.full(6, -1, dtype=np.int32)
        common_layer = -1
        for idx in range(len(genome)):
            sid = int(genome[idx])
            if sid < 0 or sid >= len(shortcut_order):
                continue
            order = int(shortcut_order[sid])
            if order < 1 or order > 5:
                continue
            if first_pos[sid] < 0:
                first_pos[sid] = idx
            selected_sid[order] = sid
            layer = int(pos_layer[idx])
            if pos_frozen[idx] or layer == 7:
                continue
            if layer_seen[order] < 0:
                layer_seen[order] = layer
            elif layer_seen[order] != layer:
                return False
            if common_layer < 0:
                common_layer = layer
            elif common_layer != layer:
                return False
        if common_layer < 0:
            return False
        for order in range(1, 6):
            sid = int(selected_sid[order])
            if sid < 0 or first_pos[sid] < 0 or layer_seen[order] < 0:
                return False
        anchor = first_pos[int(selected_sid[2])]
        anchor_x = float(pos_x[anchor])
        anchor_y = float(pos_y[anchor])
        for order in range(1, 6):
            idx = first_pos[int(selected_sid[order])]
            if (abs(float(pos_x[idx]) - (anchor_x + float(offsets[order, 0]))) > 0.5
                    or abs(float(pos_y[idx]) - (anchor_y + float(offsets[order, 1]))) > 0.5):
                return False
        return True

    @njit(cache=True)
    def _cluster_counts_fast(
        genome, pos_layer, pos_x, pos_y, n_shortcuts,
        cluster_starts, member_sids, member_dx, member_dy, requires_order,
    ):
        """Exact archive cluster metrics without rescanning the genome per cluster."""
        first_pos = np.full((n_shortcuts, 32), -1, dtype=np.int32)
        layer_masks = np.zeros(n_shortcuts, dtype=np.uint32)
        for idx in range(len(genome)):
            sid = int(genome[idx])
            layer = int(pos_layer[idx])
            if 0 <= sid < n_shortcuts and 0 <= layer < 32:
                layer_masks[sid] |= np.uint32(1) << np.uint32(layer)
                if first_pos[sid, layer] < 0:
                    first_pos[sid, layer] = idx

        together = 0
        order_ok = 0
        ordered_total = 0
        n_clusters = len(cluster_starts) - 1
        for cluster_idx in range(n_clusters):
            start = int(cluster_starts[cluster_idx])
            end = int(cluster_starts[cluster_idx + 1])
            if end <= start:
                continue
            shared_layers = np.uint32(0xFFFFFFFF)
            has_assignment = False
            for member_idx in range(start, end):
                sid = int(member_sids[member_idx])
                if sid < 0 or sid >= n_shortcuts:
                    shared_layers = np.uint32(0)
                    break
                if layer_masks[sid] != 0:
                    has_assignment = True
                shared_layers &= layer_masks[sid]
            if requires_order[cluster_idx] and has_assignment:
                ordered_total += 1
            if shared_layers == 0:
                continue
            together += 1
            if not requires_order[cluster_idx]:
                continue

            # _cluster_quality chooses the first full layer in member-order /
            # position-order traversal. Since every full layer has a copy of
            # the first member, its first position determines that same layer.
            first_sid = int(member_sids[start])
            dominant_layer = -1
            dominant_first = len(genome) + 1
            for layer in range(32):
                if (shared_layers & (np.uint32(1) << np.uint32(layer))) != 0:
                    idx = int(first_pos[first_sid, layer])
                    if 0 <= idx < dominant_first:
                        dominant_first = idx
                        dominant_layer = layer
            if dominant_layer < 0:
                continue

            anchor_member = -1
            for member_idx in range(start, end):
                if (abs(float(member_dx[member_idx])) <= 0.01
                        and abs(float(member_dy[member_idx])) <= 0.01):
                    anchor_member = member_idx
                    break
            # Preserve existing report semantics: absent anchor yields zero
            # order errors, so a complete relative cluster counts as ordered.
            if anchor_member < 0:
                order_ok += 1
                continue
            anchor_sid = int(member_sids[anchor_member])
            anchor_pos = int(first_pos[anchor_sid, dominant_layer])
            if anchor_pos < 0:
                order_ok += 1
                continue
            anchor_x = float(pos_x[anchor_pos])
            anchor_y = float(pos_y[anchor_pos])
            valid_order = True
            for member_idx in range(start, end):
                sid = int(member_sids[member_idx])
                idx = int(first_pos[sid, dominant_layer])
                if idx < 0:
                    continue
                dx = float(pos_x[idx]) - (anchor_x + float(member_dx[member_idx]))
                dy = float(pos_y[idx]) - (anchor_y + float(member_dy[member_idx]))
                if dx * dx + dy * dy > 0.25:
                    valid_order = False
                    break
            if valid_order:
                order_ok += 1
        return together, order_ok, n_clusters, ordered_total
else:
    _cluster_counts_fast = None
    _completion_cluster_pass_fast = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _constraint_penalty_from_objectives(F, min_penalty=2.0, max_penalty=25.0, spread_fraction=0.25):
    """Calibrate soft constraint pressure to the current objective scale.

    This is deliberately not a hard feasibility wall. A broken mouse layer can
    survive if it buys enough objective improvement to explore a new basin, but
    exact archive/final ranking still rejects dynamic-mouse failure.
    """
    values = np.asarray(F, dtype=np.float64)
    totals = values.sum(axis=1) if values.ndim == 2 else values
    if totals.size < 2:
        return float(min_penalty)
    q25, q75 = np.percentile(totals.astype(np.float64), [25, 75])
    spread = max(float(q75 - q25), float(np.std(totals)))
    if not np.isfinite(spread) or spread <= 0.0:
        return float(min_penalty)
    return float(np.clip(spread * spread_fraction, min_penalty, max_penalty))


def _has_unsupported_duplicate(genome, unsupported_sid_mask, frozen_mask):
    """Fast archive prefilter; detailed duplicate evidence remains in reports."""
    if unsupported_sid_mask is None or not np.any(unsupported_sid_mask):
        return False
    values = np.asarray(genome, dtype=np.int32)
    assigned = values >= 0
    total = np.bincount(values[assigned], minlength=len(unsupported_sid_mask))
    frozen_values = values[np.asarray(frozen_mask, dtype=np.bool_) & assigned]
    frozen = np.bincount(frozen_values, minlength=len(unsupported_sid_mask))
    return bool(np.any(
        unsupported_sid_mask & (total > 1) & (frozen < total)
    ))


def _scalar(F, cv=None, constraint_penalty=None):
    """Exploration scalar fitness. Lower is better.

    Constraint violations should guide selection without walling off useful
    intermediate states. Archive/final-best ranking applies hard acceptance
    tiers after exact evaluation.
    """
    values = np.asarray(F, dtype=np.float64)
    s = values.sum(axis=1) if values.ndim == 2 else values
    if cv is not None:
        if constraint_penalty is None:
            constraint_penalty = _constraint_penalty_from_objectives(np.asarray(F))
        penalty = np.maximum(cv, 0)
        if penalty.ndim > 1:
            penalty = penalty.sum(axis=1)
        if penalty.shape[0] == s.shape[0]:
            s = s + float(constraint_penalty) * penalty
    return s


def _tournament_select(scalar_F, n, k=2):
    """Return (n,) parent indices via binary tournament — vectorized CPU numpy."""
    idx = np.random.randint(0, len(scalar_F), (n, k))
    return idx[np.arange(n), scalar_F[idx].argmin(axis=1)]


def _prioritize_exact_eval_indices(sampled_indices, priority_indices, limit):
    """Reserve exact-evaluation slots for contract-targeted candidates."""
    limit = max(0, int(limit))
    if limit == 0:
        return np.empty(0, dtype=np.int32)
    selected = []
    seen = set()
    for values in (priority_indices, sampled_indices):
        for value in np.asarray(values, dtype=np.int32).reshape(-1):
            idx = int(value)
            if idx in seen:
                continue
            selected.append(idx)
            seen.add(idx)
            if len(selected) >= limit:
                return np.asarray(selected, dtype=np.int32)
    return np.asarray(selected, dtype=np.int32)


def _reject_invalid_children(children_X, children_F, children_cv, fallback_X,
                             fallback_F, fallback_cv, exact_constraints):
    """Restore each hard-invalid proposal to its known-valid selected parent."""
    invalid = np.any(np.asarray(exact_constraints) > 0, axis=1)
    if np.any(invalid):
        children_X[invalid] = fallback_X[invalid]
        children_F[invalid] = fallback_F[invalid]
        children_cv[invalid] = fallback_cv[invalid]
    return invalid


def _ordered_feasible_archive_indices(batch_F, batch_G, incumbent_total_score=None):
    """Return only feasible rows the archive loop can reach, lowest score first.

    The archive loop stops at the incumbent's total score. Filter rows at or
    above that cutoff before sorting, so converged batches do not sort 1,500
    candidates merely to discover that their best score cannot improve it.
    """
    feasible = np.flatnonzero(np.all(np.asarray(batch_G) <= 0, axis=1))
    if not len(feasible):
        return feasible
    scores = np.asarray(batch_F, dtype=np.float64).sum(axis=1)
    if incumbent_total_score is not None:
        feasible = feasible[scores[feasible] < float(incumbent_total_score)]
        if not len(feasible):
            return feasible
    return feasible[np.argsort(scores[feasible])]


def _crossover_batch(pop_X, parent_idx, crossover_prob, n_shortcuts):
    """Pair parents and apply parallel cycle crossover. Returns children array."""
    children = pop_X[parent_idx].copy().astype(np.int32)
    half = len(children) // 2
    if NUMBA_AVAILABLE:
        # All pairs run in parallel via Numba prange (uses all CPU threads)
        _cycle_crossover_batch_numba(children, half, crossover_prob, n_shortcuts)
    else:
        for i in range(half):
            if random.random() < crossover_prob:
                p1, p2 = children[i].copy(), children[i + half].copy()
                mask = np.random.random(len(p1)) < 0.5
                c1, c2 = p1.copy(), p2.copy()
                c1[mask], c2[mask] = p2[mask], p1[mask]
                children[i] = c1
                children[i + half] = c2
    return children


def _best_index(scalar_F):
    return int(np.argmin(scalar_F))


def _select_feasibility_first_scalar(scalar, cv, n):
    """Return top-n feasibility beacons, ordered by violations then quality."""
    if cv is None or cv.shape[1] == 0:
        return np.argpartition(scalar, n)[:n]
    feasible = cv.sum(axis=1) == 0
    n_feasible = int(feasible.sum())
    idx = np.arange(len(scalar))
    if n_feasible >= n:
        feasible_idx = idx[feasible]
        if n >= len(feasible_idx):
            return feasible_idx
        return feasible_idx[np.argpartition(scalar[feasible_idx], n)[:n]]
    feasible_idx = idx[feasible]
    infeasible_idx = idx[~feasible]
    n_needed = n - n_feasible
    if n_needed >= len(infeasible_idx):
        return np.concatenate([feasible_idx, infeasible_idx])
    # Preserve a narrow path toward feasibility without imposing a wall on
    # the rest of the population. The global soft-score selection keeps most
    # survivor slots and remains free to explore useful infeasible states.
    order = np.lexsort((scalar[infeasible_idx], cv[infeasible_idx].sum(axis=1)))
    ranked_infeasible = infeasible_idx[order[:n_needed]]
    return np.concatenate([feasible_idx, ranked_infeasible])


def _survivor_indices(scalar, cv, n_pop, forced_indices=(), relaxed=False):
    """Select a population while preserving explicit archive/anchor members."""
    forced = np.asarray(list(dict.fromkeys(int(i) for i in forced_indices)), dtype=np.int64)
    if len(forced) > n_pop:
        raise ValueError("forced survivor count exceeds population size")
    available = np.ones(len(scalar), dtype=bool)
    available[forced] = False
    candidates = np.flatnonzero(available)

    if relaxed:
        elite_n = min(max(2, n_pop // 10), len(candidates))
        elite_idx = candidates[np.argpartition(scalar[candidates], elite_n)[:elite_n]]
        remaining = np.setdiff1d(candidates, elite_idx, assume_unique=False)
        random_n = n_pop - elite_n - len(forced)
        random_n = min(random_n, len(remaining))
        random_idx = remaining[np.random.choice(len(remaining), random_n, replace=False)]
        return np.concatenate([elite_idx, random_idx, forced])

    beacon_n = min(max(1, n_pop // 10), len(candidates)) if cv is not None else 0
    if beacon_n:
        local = _select_feasibility_first_scalar(scalar[candidates], cv[candidates], beacon_n)
        beacons = candidates[local]
    else:
        beacons = np.empty(0, dtype=np.int64)
    remaining = candidates[~np.isin(candidates, beacons, assume_unique=False)]
    soft_n = min(n_pop - len(beacons) - len(forced), len(remaining))
    soft = remaining[np.argpartition(scalar[remaining], soft_n)[:soft_n]] if soft_n else np.empty(0, dtype=np.int64)
    return np.concatenate([beacons, soft, forced])


def _owner_contract_failed(entry):
    failed = set(entry.get("acceptance_failed_checks", []))
    return bool({"owner_layout_contract_v1", "l1_mouse_contract"} & failed)


def _warmstart_needs_access_sanitizing(entry):
    """Sanitize no-op holds only when the warmstart violates hard constraints.

    Acceptance has independent product checks (Norwegian keys,
    duplicates). Using any acceptance failure as a reason to clear holds can
    destroy a hard-feasible warmstart even when its only defect is elsewhere.
    """
    constraints = np.asarray(entry.get("constraints", ()), dtype=np.float32)
    return bool(np.any(constraints > 0.0))


def _display_gap(entry, target=None):
    """Return an optional calibrated gap; retain a sentinel for mouse failure."""
    if _owner_contract_failed(entry):
        return 5.0
    return None if target is None else float(entry["total_score"]) - float(target)


def _count_clusters_together(layout):
    """Return together, correctly ordered, total, and explicitly ordered counts."""
    clusters = list(getattr(layout, "semantic_clusters", ()))
    together = 0
    order_ok = 0
    ordered_total = 0
    for cluster in clusters:
        r = _cluster_quality(layout, cluster)
        requires_order = bool(r.get("relative_layout_required", False))
        if requires_order:
            ordered_total += 1
        if r.get("fully_together"):
            together += 1
            if requires_order and r.get("order_errors", 0) == 0:
                order_ok += 1
    return together, order_ok, len(clusters), ordered_total


def _cluster_count_arrays(layout):
    starts = [0]
    member_sids, member_dx, member_dy, requires_order = [], [], [], []
    for cluster in getattr(layout, "semantic_clusters", ()):
        members = list(cluster.get("members", ()))
        offsets = [(float(m.get("dx", 0.0)), float(m.get("dy", 0.0))) for m in members]
        member_sids.extend(int(m.get("sid", -1)) for m in members)
        member_dx.extend(dx for dx, _ in offsets)
        member_dy.extend(dy for _, dy in offsets)
        requires_order.append(any(abs(dx) > 0.01 or abs(dy) > 0.01 for dx, dy in offsets))
        starts.append(len(member_sids))
    return (
        np.asarray(starts, dtype=np.int32),
        np.asarray(member_sids, dtype=np.int32),
        np.asarray(member_dx, dtype=np.float32),
        np.asarray(member_dy, dtype=np.float32),
        np.asarray(requires_order, dtype=np.bool_),
    )


def _archive_cluster_precheck_reason(counts, completion_pass, incumbent):
    """Name an earlier archive tier that proves a candidate cannot win."""
    if incumbent is None or not incumbent.get("optimizer_side_pass", False):
        return None
    incumbent_completion = bool(incumbent.get("norwegian_completion_cluster_pass", True))
    if bool(completion_pass) != incumbent_completion:
        return "norwegian_completion_cluster" if not completion_pass else None
    together, ordered, total, ordered_total = counts
    for candidate_count, candidate_total, incumbent_count_key, incumbent_total_key in (
        (together, total, "clusters_together", "clusters_total"),
        (ordered, ordered_total, "clusters_order_ok", "clusters_ordered_total"),
    ):
        incumbent_count = incumbent.get(incumbent_count_key)
        incumbent_total = incumbent.get(incumbent_total_key)
        if (incumbent_count is None or incumbent_total is None
                or int(candidate_total) <= 0 or int(incumbent_total) <= 0):
            continue
        if (float(candidate_count) / int(candidate_total) + 1e-12
                < float(incumbent_count) / int(incumbent_total)):
            return "semantic_cluster_monotonicity"
    return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class CustomGARunner:
    """Self-contained GA loop; mirrors ExactEvalCallback without pymoo."""

    def __init__(
        self,
        layout,
        evaluator,
        surrogate_manager,
        mutation,
        sanitizer,
        analyze_duplicates_fn,
        pop_size,
        crossover_prob,
        n_shortcuts,
        checkpoint_every,
        build_dir,
        perf,
        hard_constraints,
        n_constraints: int = 0,
        semantic_multiplier_schedule=None,
        cluster_score_tolerance: float = 0.15,
        unsupported_duplicate_sid_mask=None,
    ):
        self.layout = layout
        self.evaluator = evaluator
        self.surrogate_manager = surrogate_manager
        self.mutation = mutation
        self.sanitizer = sanitizer
        self.analyze_duplicates = analyze_duplicates_fn
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.n_shortcuts = n_shortcuts
        self.checkpoint_every = checkpoint_every
        self.build_dir = build_dir
        self.perf = perf
        self.hard_constraints = hard_constraints
        self.n_factors = 3
        self.n_constraints = int(n_constraints)
        self._archive_pos_layer = np.asarray(
            [position.layer for position in layout.positions], dtype=np.int32
        )
        self._archive_pos_x = np.asarray(
            [position.x for position in layout.positions], dtype=np.float32
        )
        self._archive_pos_y = np.asarray(
            [position.y for position in layout.positions], dtype=np.float32
        )
        self._archive_pos_frozen = np.asarray(
            [position.is_frozen for position in layout.positions], dtype=np.bool_
        )
        self._shortcut_completion_order = np.asarray([
            completion_order(shortcut) if is_raw_completion_base(shortcut) else 0
            for shortcut in layout.shortcuts
        ], dtype=np.int32)
        self._completion_offsets = np.zeros((6, 2), dtype=np.float32)
        for order, (dx, dy) in NORWEGIAN_CLUSTER_OFFSETS.items():
            self._completion_offsets[int(order), 0] = float(dx)
            self._completion_offsets[int(order), 1] = float(dy)
        self.semantic_multiplier_schedule = list(semantic_multiplier_schedule or [])
        self.cluster_score_tolerance = float(cluster_score_tolerance)
        self.unsupported_duplicate_sid_mask = (
            None if unsupported_duplicate_sid_mask is None
            else np.asarray(unsupported_duplicate_sid_mask, dtype=np.bool_)
        )
        self._left_alt_sid = next(
            (shortcut.sid for shortcut in layout.shortcuts if shortcut.keys == "LeftAlt"),
            None,
        )
        self._l0_mutable_positions = np.asarray(
            [position.gene_idx for position in layout.positions
             if position.layer == 0 and not position.is_frozen],
            dtype=np.int32,
        )
        self._cluster_pos_layer = np.asarray(
            [position.layer for position in layout.positions], dtype=np.int32,
        )
        self._cluster_pos_x = np.asarray(
            [position.x for position in layout.positions], dtype=np.float32,
        )
        self._cluster_pos_y = np.asarray(
            [position.y for position in layout.positions], dtype=np.float32,
        )
        (self._cluster_starts, self._cluster_member_sids,
         self._cluster_member_dx, self._cluster_member_dy,
         self._cluster_requires_order) = _cluster_count_arrays(layout)
        self._layout_cluster_cache = OrderedDict()

        # State
        self.global_best_genome = None
        self.global_best_exact = None
        self.global_best_generation = None
        self.infeasible_anchor_genome = None
        self.infeasible_anchor_objectives = None
        self.infeasible_anchor_constraints = None
        self.infeasible_anchor_key = None
        self.stagnation_count = 0
        self.archive_stagnation = 0
        self.last_best_quality = float("inf")
        self.base_mutation_prob = float(
            mutation.prob.value if hasattr(mutation.prob, "value") else mutation.prob
        )
        self.last_diversity_reset = 0
        self.exact_history = []
        self._should_stop = False
        self.best_exact = None
        self.archive_acceptance_rejections = {}
        self.archive_duplicate_prefilter_rejections = 0
        self.archive_cluster_prefilter_rejections = 0
        self.archive_cluster_prefilter_reasons = {}
        self._relaxed_selection_until = 0  # gen until which to use diversity-preserving survival

    # ------------------------------------------------------------------
    # Helpers (ported from ExactEvalCallback)
    # ------------------------------------------------------------------

    def _apply_semantic_multiplier_schedule(self, gen):
        """Ramp semantic-cluster pressure according to the configured schedule."""
        if not self.semantic_multiplier_schedule:
            return
        target = None
        for sched_gen, sched_mult in self.semantic_multiplier_schedule:
            if gen >= sched_gen:
                target = float(sched_mult)
        if target is None:
            return
        current = getattr(self.evaluator.model, "semantic_cluster_multiplier", None)
        if current is None or abs(current - target) > 1e-9:
            # Finish and collect an old-stage background model before changing
            # labels. It must never overwrite the newly trained stage model.
            sm = self.surrogate_manager
            if sm is not None and sm._retrain_future is not None:
                sm._retrain_future.result()
                sm.maybe_collect_retrain()
            print(
                f"  Gen {gen}: ramping semantic_cluster_multiplier from {current} to {target}",
                flush=True,
            )
            self.evaluator.set_semantic_cluster_multiplier(target)
            if self.surrogate_manager is not None:
                self.surrogate_manager.clear_exact_cache()
            return True
        return False

    def _refresh_semantic_scores(self, pop_X):
        """Reprice parents, archive and teacher labels at a schedule milestone."""
        pop_F, pop_G = self.evaluator.evaluate_batch(pop_X.astype(np.int32))
        if getattr(self, "infeasible_anchor_genome", None) is not None:
            anchor_F, anchor_G = self.evaluator.evaluate_batch(self.infeasible_anchor_genome[None, :])
            self.infeasible_anchor_objectives = anchor_F[0].copy()
            self.infeasible_anchor_constraints = np.maximum(anchor_G[0], 0).astype(np.float32)
            self.infeasible_anchor_key = self._fallback_key(anchor_F[0], anchor_G[0])
        if self.global_best_genome is not None:
            layout = self.layout.clone_with(genome=self.global_best_genome.copy())
            result = self.evaluator.evaluate(layout)
            entry = self._exact_entry(result, self.global_best_generation)
            _, _, _, acceptance = self._layout_reports(layout)
            self._annotate(entry, acceptance)
            self._annotate_clusters(entry, layout)
            self.global_best_exact = entry
            self.best_exact = dict(entry)
        sm = self.surrogate_manager
        if sm is not None:
            sm.add_exact_evaluations(pop_X, pop_F, pop_G)
            sm.retrain()
        self.last_best_quality = float("inf")
        self.stagnation_count = 0
        return pop_F, np.maximum(pop_G, 0).astype(np.float32)

    @staticmethod
    def _fallback_key(objectives, constraints):
        cv = np.maximum(np.asarray(constraints, dtype=np.float64), 0.0)
        return (
            int(np.count_nonzero(cv)), float(cv.sum()),
            float(np.asarray(objectives, dtype=np.float64).sum()),
        )

    def _write_live_perf(self, generation, elapsed_seconds):
        """Persist per-stage timings at each checkpoint for live run review."""
        if self.perf is None or not self.build_dir:
            return
        report = {
            "generation": int(generation),
            "elapsed_seconds": float(elapsed_seconds),
            "seconds_per_generation": float(elapsed_seconds) / max(int(generation), 1),
            "events": self.perf.summary(),
        }
        path = os.path.join(self.build_dir, "v2_perf_live.json")
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        os.replace(temporary, path)

    def _record_generation_stage(self, name, started):
        """Record one non-overlapping stage in the current generation."""
        elapsed = time.perf_counter() - started
        if self.perf:
            self.perf.add(name, elapsed)
        self._generation_stage_seconds += elapsed
        return elapsed

    def _consider_infeasible_anchor(self, genome, objectives, constraints):
        """Retain the best exact low-violation genome for continued search."""
        key = self._fallback_key(objectives, constraints)
        incumbent_key = getattr(self, "infeasible_anchor_key", None)
        if incumbent_key is None or key < incumbent_key:
            self.infeasible_anchor_genome = np.asarray(genome, dtype=np.int32).copy()
            self.infeasible_anchor_objectives = np.asarray(objectives, dtype=np.float32).copy()
            self.infeasible_anchor_constraints = np.maximum(
                np.asarray(constraints, dtype=np.float32), 0.0
            )
            self.infeasible_anchor_key = key

    def _exact_entry(self, result, gen):
        return {
            "generation": int(gen),
            "objectives": [float(x) for x in result.objectives],
            "constraints": [float(x) for x in result.constraints],
            "factor_scores": {k: float(v) for k, v in result.factor_scores.items()},
            "total_score": float(result.total_score),
        }

    def _maybe_update_global_from_batch(self, batch_X, batch_F, batch_G, gen):
        """Check a batch of exact evaluations and update the archive if a better feasible genome exists."""
        if batch_X is None or len(batch_X) == 0:
            return
        if self.global_best_genome is None:
            anchor_t0 = time.perf_counter()
            for i in range(len(batch_X)):
                self._consider_infeasible_anchor(batch_X[i], batch_F[i], batch_G[i])
            if self.perf:
                self.perf.add("archive_anchor_scan", time.perf_counter() - anchor_t0)
        incumbent_score = (
            None if self.global_best_exact is None
            else float(self.global_best_exact["total_score"])
        )
        prefilter_t0 = time.perf_counter()
        feasible_indices = _ordered_feasible_archive_indices(
            batch_F, batch_G, incumbent_score
        )
        if self.perf:
            self.perf.add("archive_prefilter", time.perf_counter() - prefilter_t0)
        if not len(feasible_indices):
            return
        for i in feasible_indices:
            duplicate_t0 = time.perf_counter()
            duplicate_rejected = _has_unsupported_duplicate(
                batch_X[i], getattr(self, "unsupported_duplicate_sid_mask", None),
                self.layout.frozen_mask,
            )
            if self.perf:
                self.perf.add("archive_duplicate_prefilter", time.perf_counter() - duplicate_t0)
            if duplicate_rejected:
                self.archive_duplicate_prefilter_rejections += 1
                continue
            score = float(np.asarray(batch_F[i], dtype=np.float64).sum())
            candidate = batch_X[i]
            candidate_layout = self.layout.clone_with(genome=candidate.astype(np.int32))
            cluster_t0 = time.perf_counter()
            completion_report = None
            if all(hasattr(self, name) for name in (
                    "_shortcut_completion_order", "_archive_pos_layer",
                    "_archive_pos_x", "_archive_pos_y",
                    "_archive_pos_frozen", "_completion_offsets")):
                completion_pass = _completion_cluster_pass_fast(
                    candidate, self._shortcut_completion_order,
                    self._archive_pos_layer, self._archive_pos_x,
                    self._archive_pos_y, self._archive_pos_frozen,
                    self._completion_offsets,
                )
            else:
                completion_report = analyze_completion_cluster(candidate_layout)
                completion_pass = bool(completion_report.get("acceptance_pass", False))
            cluster_counts = self._cluster_counts(candidate)
            cluster_reject_reason = _archive_cluster_precheck_reason(
                cluster_counts, completion_pass, self.global_best_exact,
            )
            if self.perf:
                self.perf.add("archive_cluster_prefilter", time.perf_counter() - cluster_t0)
            if cluster_reject_reason:
                self.archive_cluster_prefilter_rejections += 1
                self.archive_cluster_prefilter_reasons[cluster_reject_reason] = (
                    self.archive_cluster_prefilter_reasons.get(cluster_reject_reason, 0) + 1
                )
                continue
            if completion_report is None:
                completion_report = analyze_completion_cluster(candidate_layout)
            entry = self._exact_entry(
                type("obj", (object,), {
                    "objectives": batch_F[i].copy(),
                    "constraints": batch_G[i].copy(),
                    "factor_scores": self.evaluator._factor_scores_from_objectives(batch_F[i]),
                    "total_score": score,
                })(), gen,
            )
            together, ordered, total, ordered_total = cluster_counts
            entry.update({
                "clusters_together": int(together),
                "clusters_order_ok": int(ordered),
                "clusters_total": int(total),
                "clusters_ordered_total": int(ordered_total),
            })
            report_t0 = time.perf_counter()
            _, _, _, acceptance = self._layout_reports(
                candidate_layout, completion_report=completion_report,
            )
            self._annotate(entry, acceptance)
            for check, passed in acceptance.get("optimizer_side_checks", {}).items():
                if not passed:
                    self.archive_acceptance_rejections[check] = (
                        self.archive_acceptance_rejections.get(check, 0) + 1
                    )
            if self.perf:
                self.perf.add("archive_candidate_reports", time.perf_counter() - report_t0)
            if not self._is_better(entry, self.global_best_exact):
                continue
            self._update_archive(candidate, entry)
            print(f"    Gen {gen}: global best improved to {entry['total_score']:.4f} (source=exact_eval)", flush=True)
            break

    def _is_better(self, candidate, incumbent):
        cand_cv = sum(max(0.0, float(x)) for x in candidate.get("constraints", []))
        if (cand_cv != 0.0 or _owner_contract_failed(candidate)
                or not candidate.get("optimizer_side_pass", False)):
            return False
        if incumbent is None:
            return True
        inc_cv = sum(max(0.0, float(x)) for x in incumbent.get("constraints", []))
        # Only prefer lower constraint violation when one solution is feasible (cv=0)
        # and the other isn't. When both violate constraints the violation penalty is
        # already baked into total_score (weight × count), so compare by score.
        cand_feasible = cand_cv == 0.0
        inc_feasible = inc_cv == 0.0
        if cand_feasible != inc_feasible:
            return cand_feasible
        cand_mouse_fail = _owner_contract_failed(candidate)
        inc_mouse_fail = _owner_contract_failed(incumbent)
        if cand_mouse_fail != inc_mouse_fail:
            return not cand_mouse_fail
        # The Norwegian raw-key cluster is a user acceptance requirement. Keep
        # it as an explicit archive tier: a small score advantage must not
        # preserve an invalid cluster over a valid one.
        cand_completion = bool(candidate.get(
            "norwegian_completion_cluster_pass",
            "norwegian_completion_cluster" not in candidate.get("acceptance_failed_checks", []),
        ))
        inc_completion = bool(incumbent.get(
            "norwegian_completion_cluster_pass",
            "norwegian_completion_cluster" not in incumbent.get("acceptance_failed_checks", []),
        ))
        if cand_completion != inc_completion:
            return cand_completion

        # A direct L0 LeftAlt pass may split the optional family cluster that
        # groups LeftAlt with RightAlt. Give that explicit owner contract one
        # narrow exception: exactly one co-location may be lost, with no order
        # regression, when the incumbent fails only the L0 Alt check.
        cand_pass = bool(candidate.get("optimizer_side_pass", False))
        inc_pass = bool(incumbent.get("optimizer_side_pass", False))
        inc_failures = set(incumbent.get("acceptance_failed_checks", []))
        candidate_clusters = candidate.get("clusters_together")
        incumbent_clusters = incumbent.get("clusters_together")
        candidate_total = candidate.get("clusters_total")
        incumbent_total = incumbent.get("clusters_total")
        candidate_order = candidate.get("clusters_order_ok")
        incumbent_order = incumbent.get("clusters_order_ok")
        candidate_order_total = candidate.get("clusters_ordered_total")
        incumbent_order_total = incumbent.get("clusters_ordered_total")
        if (
            cand_pass
            and not inc_pass
            and inc_failures == {"left_alt_directly_available_on_l0"}
            and None not in (
                candidate_clusters, incumbent_clusters, candidate_total, incumbent_total,
                candidate_order, incumbent_order, candidate_order_total, incumbent_order_total,
            )
            and int(candidate_total) == int(incumbent_total)
            and int(incumbent_clusters) - int(candidate_clusters) == 1
            and int(candidate_order_total) == int(incumbent_order_total)
            and int(candidate_order) >= int(incumbent_order)
        ):
            return True

        # A smaller acceptance-failure list must not erase a better semantic
        # archive. Keep both cluster co-location and relative placement
        # monotonic; rejected candidates remain available in the population.
        for count_key, total_key in (
            ("clusters_together", "clusters_total"),
            ("clusters_order_ok", "clusters_ordered_total"),
        ):
            candidate_count = candidate.get(count_key)
            incumbent_count = incumbent.get(count_key)
            candidate_total = candidate.get(total_key)
            incumbent_total = incumbent.get(total_key)
            if None in (candidate_count, incumbent_count, candidate_total, incumbent_total):
                continue
            candidate_total = int(candidate_total)
            incumbent_total = int(incumbent_total)
            if candidate_total <= 0 or incumbent_total <= 0:
                continue
            if float(candidate_count) / candidate_total + 1e-12 < float(incumbent_count) / incumbent_total:
                return False

        if cand_pass != inc_pass:
            return cand_pass
        if not cand_pass:
            cf = len(candidate.get("acceptance_failed_checks", []))
            inf = len(incumbent.get("acceptance_failed_checks", []))
            if cf != inf:
                return cf < inf

        cand_ordered = candidate.get("clusters_order_ok")
        inc_ordered = incumbent.get("clusters_order_ok")
        if cand_ordered is not None and inc_ordered is not None and cand_ordered != inc_ordered:
            cand_score = float(candidate["total_score"])
            inc_score = float(incumbent["total_score"])
            if cand_ordered > inc_ordered and cand_score <= inc_score * (1.0 + self.cluster_score_tolerance):
                return True
            if cand_ordered < inc_ordered:
                if cand_score < inc_score * (1.0 - self.cluster_score_tolerance):
                    return True
                return False

        # Cluster-aware archive selection: a feasible, accepted layout with more
        # semantic clusters together is preferred even if its raw score is slightly
        # worse. This drives the search toward the user's stated goal of keeping
        # related shortcuts on the same layer.
        cand_clusters = candidate.get("clusters_together")
        inc_clusters = incumbent.get("clusters_together")
        if cand_clusters is not None and inc_clusters is not None:
            if cand_clusters > inc_clusters:
                cand_score = float(candidate["total_score"])
                inc_score = float(incumbent["total_score"])
                if cand_score <= inc_score * (1.0 + self.cluster_score_tolerance):
                    return True
            elif cand_clusters < inc_clusters:
                # Prefer the incumbent unless the candidate is meaningfully better.
                cand_score = float(candidate["total_score"])
                inc_score = float(incumbent["total_score"])
                if cand_score < inc_score * (1.0 - self.cluster_score_tolerance):
                    return True
                return False

        return float(candidate["total_score"]) < float(incumbent["total_score"])

    def _update_archive(self, genome, entry):
        if self._is_better(entry, self.global_best_exact):
            archived = genome.astype(np.int32).copy()
            self.global_best_genome = archived
            self.global_best_exact = dict(entry)
            self.global_best_generation = int(entry["generation"])
            # Paranoia: ensure the genome we just archived matches the entry's
            # cluster count. If it doesn't, the exact entry was paired with the
            # wrong genome (likely a view-alias bug).
            entry_clusters = entry.get("clusters_together")
            if entry_clusters is not None:
                check_layout = self.layout.clone_with(genome=archived)
                real_together, _, _, _ = _count_clusters_together(check_layout)
                if real_together != entry_clusters:
                    print(
                        f"    WARNING: archived genome cluster mismatch "
                        f"(entry={entry_clusters}, genome={real_together}). "
                        f"Re-annotating archive entry.",
                        flush=True,
                    )
                    self._annotate_clusters(self.global_best_exact, check_layout)
            return True
        return False

    def _layout_reports(self, layout_obj, completion_report=None):
        # A converged population can propose the same genome many times. These
        # reports depend only on the genome and fixed layout metadata; retain a
        # bounded LRU so exact archive filtering does not re-audit repeats.
        cache = getattr(self, "_layout_reports_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._layout_reports_cache = cache
        genome = np.asarray(layout_obj.genome, dtype=np.int32)
        key = genome.tobytes()
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

        dup = self.analyze_duplicates(layout_obj)
        comp = (completion_report if completion_report is not None
                else analyze_completion_cluster(layout_obj))
        arr = analyze_arrows(layout_obj)
        acc = build_acceptance_report(
            layout_obj,
            duplicate_report=dup,
            completion_cluster_report=comp,
            arrow_report=arr,
        )
        reports = (dup, comp, arr, acc)
        cache[key] = reports
        if len(cache) > 512:
            cache.popitem(last=False)
        return reports

    def _annotate(self, entry, acceptance):
        failed = [
            k for k, ok in acceptance.get("checks", {}).items()
            if k != "norwegian_export_bad_literal_count_zero" and not ok
        ]
        entry["optimizer_side_pass"] = bool(acceptance.get("optimizer_side_pass", False))
        entry["acceptance_failed_checks"] = failed
        entry["norwegian_completion_cluster_pass"] = bool(
            acceptance.get("checks", {}).get("norwegian_completion_cluster", False)
        )
        return entry

    def _annotate_clusters(self, entry, layout):
        """Add semantic-cluster counts to an exact-eval entry."""
        counts = self._cluster_counts(layout.genome, layout_obj=layout)
        together, order_ok, total, ordered_total = counts
        entry["clusters_together"] = int(together)
        entry["clusters_order_ok"] = int(order_ok)
        entry["clusters_ordered_total"] = int(ordered_total)
        entry["clusters_total"] = int(total)
        return entry

    def _cluster_counts(self, genome, layout_obj=None):
        cache = getattr(self, "_layout_cluster_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._layout_cluster_cache = cache
        values = np.asarray(genome, dtype=np.int32)
        key = values.tobytes()
        counts = cache.get(key)
        if counts is None:
            if (_cluster_counts_fast is None
                    or not hasattr(self, "_cluster_pos_layer")):
                source_layout = layout_obj
                if source_layout is None:
                    source_layout = self.layout.clone_with(genome=values)
                counts = _count_clusters_together(source_layout)
            else:
                counts = _cluster_counts_fast(
                    values,
                    self._cluster_pos_layer,
                    self._cluster_pos_x,
                    self._cluster_pos_y,
                    self.layout.n_shortcuts,
                    self._cluster_starts,
                    self._cluster_member_sids,
                    self._cluster_member_dx,
                    self._cluster_member_dy,
                    self._cluster_requires_order,
                )
            cache[key] = counts
            if len(cache) > 512:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        return counts

    # ------------------------------------------------------------------
    # Adaptive mutation rate
    # ------------------------------------------------------------------

    def _adjust_mutation(self, best_quality, gen):
        if best_quality < self.last_best_quality * 0.999:
            if self.stagnation_count > 0:
                prob = self.base_mutation_prob
                if hasattr(self.mutation.prob, "value"):
                    self.mutation.prob.value = prob
                else:
                    self.mutation.prob = prob
                print(
                    f"    Gen {gen}: improvement detected (best_quality={best_quality:.3f})."
                    f" Restoring mutation rate to {prob:.3f}",
                    flush=True,
                )
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
            if self.stagnation_count >= 100 and self.stagnation_count % 100 == 0:
                cur = float(
                    self.mutation.prob.value
                    if hasattr(self.mutation.prob, "value")
                    else self.mutation.prob
                )
                new_prob = min(0.5, cur * 1.2)
                if hasattr(self.mutation.prob, "value"):
                    self.mutation.prob.value = new_prob
                else:
                    self.mutation.prob = new_prob
                print(
                    f"    Gen {gen}: stagnation for {self.stagnation_count} gens"
                    f" (best_quality={best_quality:.3f}). Increasing mutation rate to {new_prob:.3f}",
                    flush=True,
                )
        self.last_best_quality = best_quality

    # ------------------------------------------------------------------
    # Diversity injection
    # ------------------------------------------------------------------

    @staticmethod
    def _population_diversity(pop_X, n_sample=40):
        """Mean pairwise Hamming distance over a random sample of genome pairs.

        Returns a value in [0, 1] where 1 = all genomes completely different,
        0 = all genomes identical. Sampling keeps this O(n_sample² × genome_len).
        """
        n = len(pop_X)
        if n < 2:
            return 1.0
        idx = np.random.choice(n, size=min(n_sample, n), replace=False)
        sample = pop_X[idx]
        pairs = 0
        total_diff = 0
        genome_len = sample.shape[1]
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                total_diff += int(np.sum(sample[i] != sample[j]))
                pairs += 1
        return total_diff / max(pairs * genome_len, 1)

    def _inject_diversity(self, pop_X, pop_F, pop_cv, gen):
        # Trigger conditions (any one suffices):
        #   1. Surrogate stagnation (stagnation_count >= 1200)
        #   2. Archive stagnation (2+ checkpoint cycles with no improvement)
        #   3. Population genetic diversity collapsed below threshold
        archive_stagnant = self.archive_stagnation >= 2  # 1000 gens at checkpoint_every=500
        diversity = self._population_diversity(pop_X)
        diversity_collapsed = diversity < 0.08  # genomes are >92% identical on average
        if self.stagnation_count < 1200 and not archive_stagnant and not diversity_collapsed:
            return pop_X, pop_F, pop_cv
        if gen - self.last_diversity_reset < 500:
            return pop_X, pop_F, pop_cv
        trigger = (
            "surrogate_stagnation" if self.stagnation_count >= 1200
            else "archive_stagnation" if archive_stagnant
            else "diversity_collapsed"
        )
        print(f"    Diversity injection trigger: {trigger} (pop_diversity={diversity:.3f})", flush=True)

        n = len(pop_X)
        scalar = _scalar(pop_F, pop_cv)
        elite_count = max(2, n // 10)  # keep top 10% as elites
        elite_idx = set(np.argsort(scalar)[:elite_count].tolist())
        if self.global_best_genome is None and self.infeasible_anchor_genome is not None:
            elite_idx.add(0)
        replace_order = [i for i in np.argsort(scalar)[::-1] if i not in elite_idx]
        replace_idx = replace_order[: max(4, n // 4)]

        # Base perturbations on the feasible archive or the best exact
        # low-violation anchor. Never perturb the empty reference genome.
        base = (
            self.global_best_genome.astype(np.int32).copy()
            if self.global_best_genome is not None
            else self.infeasible_anchor_genome.astype(np.int32).copy()
            if self.infeasible_anchor_genome is not None
            else pop_X[_best_index(scalar)].astype(np.int32).copy()
        )

        mutable = self.layout.mutable_indices
        ml = mutable.tolist() if hasattr(mutable, "tolist") else list(mutable)

        # Exclude access/toggle SIDs from swap pool so structural constraints stay intact.
        access_sids = set()
        if hasattr(self.mutation, "_access_sid_targets"):
            access_sids = set(self.mutation._access_sid_targets.keys())
        safe_ml = [pos for pos in ml if int(base[pos]) not in access_sids] or ml

        # Effort arrays for targeted perturbation (if available from mutation operator)
        mut = self.mutation
        pos_effort = getattr(mut, "_pos_effort_arr", None)
        sid_imp = getattr(mut, "_sid_importance_arr", None)
        is_group_lut = getattr(mut, "_is_group_sid_lut", None)
        safe_ml_arr = np.array(safe_ml, dtype=np.int32)

        def _effort_targeted_swaps(g, n_swaps):
            """Swap high-importance×effort shortcuts toward lower-effort positions."""
            if pos_effort is None or sid_imp is None or is_group_lut is None:
                return  # fall back to random
            for _ in range(n_swaps):
                sids = g[safe_ml_arr]
                valid = (sids >= 0) & (sids < len(sid_imp))
                safe_sids = np.where(valid, sids, 0)
                valid &= ~is_group_lut[safe_sids]
                if not valid.any():
                    break
                vpos = safe_ml_arr[valid]
                vsids = sids[valid]
                costs = pos_effort[vpos] * sid_imp[vsids]
                top5 = min(5, len(costs))
                top_idx = np.argpartition(costs, -top5)[-top5:]
                ci = int(top_idx[random.randrange(top5)])
                src_pos = int(vpos[ci])
                src_effort = float(pos_effort[src_pos])
                if src_effort <= 0:
                    break
                lower = vpos[pos_effort[vpos] < src_effort]
                if len(lower) == 0:
                    break
                tgt_pos = int(lower[random.randrange(len(lower))])
                g[tgt_pos], g[src_pos] = int(g[src_pos]), int(g[tgt_pos])

        # Bring in fresh random genomes to break out of the local basin.
        # All-perturbation injection traps the algorithm after repeated firing.
        from run_evolution import generate_random_layouts
        n_replace = len(replace_idx)
        n_random = max(4, n_replace // 2)  # 50% fully random for stronger basin escape
        random_pool = generate_random_layouts(self.layout, n_random)

        new_genomes = []
        rand_assigned = 0
        for i, dst in enumerate(replace_idx):
            # Use random genomes for the last 25% of slots.
            if i >= n_replace - n_random and rand_assigned < n_random:
                g = random_pool[rand_assigned].copy()
                rand_assigned += 1
                pop_X[dst] = g
                new_genomes.append(g)
                continue

            g = base.copy()
            # Gradient: 25% light random → 30% medium random → 25% effort-targeted → 20% mixed
            t = i / max(n_replace - n_random - 1, 1)
            if t < 0.25:
                # Light random: stay close to global best
                n_swaps = random.randint(5, 25)
                pool = safe_ml
                for _ in range(n_swaps):
                    ia = random.randrange(len(pool))
                    ib = random.randrange(len(pool))
                    g[pool[ia]], g[pool[ib]] = g[pool[ib]], g[pool[ia]]
            elif t < 0.55:
                # Medium random: explore nearby basins
                n_swaps = random.randint(25, 80)
                pool = safe_ml
                for _ in range(n_swaps):
                    ia = random.randrange(len(pool))
                    ib = random.randrange(len(pool))
                    g[pool[ia]], g[pool[ib]] = g[pool[ib]], g[pool[ia]]
            elif t < 0.80:
                # Effort-targeted: move high-cost shortcuts to lower-effort positions.
                _effort_targeted_swaps(g, random.randint(30, 100))
            else:
                # Mixed: random + effort-targeted
                n_rand = random.randint(30, 80)
                pool = safe_ml
                for _ in range(n_rand):
                    ia = random.randrange(len(pool))
                    ib = random.randrange(len(pool))
                    g[pool[ia]], g[pool[ib]] = g[pool[ib]], g[pool[ia]]
                _effort_targeted_swaps(g, random.randint(20, 50))
            pop_X[dst] = g
            new_genomes.append(g)

        # Slot 0 always holds the global best so it stays in the live population
        # and can be selected for crossover/mutation every generation.
        if self.global_best_genome is not None:
            pop_X[0] = self.global_best_genome.astype(np.int32).copy()
        elif self.infeasible_anchor_genome is not None:
            pop_X[0] = self.infeasible_anchor_genome.astype(np.int32).copy()

        sm = self.surrogate_manager
        if new_genomes:
            # Include global best in the batch so its pop_F reflects exact fitness.
            eval_batch = np.array(new_genomes, dtype=np.int32)
            if self.global_best_genome is not None:
                eval_batch = np.vstack([
                    self.global_best_genome.reshape(1, -1).astype(np.int32),
                    eval_batch,
                ])
            elif self.infeasible_anchor_genome is not None:
                eval_batch = np.vstack([
                    self.infeasible_anchor_genome.reshape(1, -1).astype(np.int32),
                    eval_batch,
                ])
            t0 = time.perf_counter()
            new_F, new_G = self.evaluator.evaluate_batch(eval_batch)
            if self.perf:
                self.perf.add("exact_eval", time.perf_counter() - t0)
            offset = 0
            if self.global_best_genome is not None or self.infeasible_anchor_genome is not None:
                pop_F[0] = new_F[0]
                if pop_cv is not None and new_G.shape[1] > 0:
                    pop_cv[0] = np.maximum(new_G[0], 0)
                offset = 1
            for row, dst in enumerate(replace_idx):
                pop_F[dst] = new_F[row + offset]
                if pop_cv is not None and new_G.shape[1] > 0:
                    pop_cv[dst] = np.maximum(new_G[row + offset], 0)

            # Add injected genomes to surrogate training data so it can rank them
            # accurately. Without this, surrogate R² collapses after injection
            # (diverse genomes are out-of-distribution) → selection is nearly random
            # for 500 gens until the next scheduled retrain.
            if sm is not None:
                sm.add_exact_evaluations(eval_batch, new_F, new_G)
                sm.async_retrain()
                print(
                    f"    Surrogate async retrain submitted on {len(eval_batch)} injection evals.",
                    flush=True,
                )

        self.last_diversity_reset = gen
        # Reset archive stagnation so injection doesn't re-fire at the very next checkpoint.
        self.archive_stagnation = 0
        # Protect injected diverse genomes for 75 gens: use elitism+random survival
        # so random genomes can't immediately be killed off by the elite pool.
        self._relaxed_selection_until = gen + 75

        since_best = (gen - self.global_best_generation) if self.global_best_generation else "?"
        print(
            f"    Gen {gen}: no real improvement for ~{since_best} gens"
            f" (gen-{self.global_best_generation} best preserved at slot 0)."
            f" Injected {len(replace_idx)} global-best perturbations, kept {elite_count} elites.",
            flush=True,
        )
        return pop_X, pop_F, pop_cv

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _checkpoint(self, pop_X, pop_F, pop_cv, gen):
        scalar = _scalar(pop_F, pop_cv)
        best_i = _best_index(scalar)
        best_genome = pop_X[best_i].astype(np.int32)
        best_layout = self.layout.clone_with(genome=best_genome)

        t0 = time.perf_counter()
        exact = self.evaluator.evaluate(best_layout)
        if self.perf:
            self.perf.add("exact_eval", time.perf_counter() - t0)

        entry = self._exact_entry(exact, gen)
        dup, comp, arr, acc = self._layout_reports(best_layout)
        self._annotate(entry, acc)
        self._annotate_clusters(entry, best_layout)

        improved = self._update_archive(best_genome, entry)
        if improved:
            self.archive_stagnation = 0
            gap = _display_gap(entry)
            clusters = entry.get("clusters_together")
            details = []
            if clusters is not None:
                details.append(f"clusters={clusters}/{entry.get('clusters_total')}")
            if gap is not None:
                details.append(f"contract_failure_signal={gap:.1f}")
            details.append(f"optimizer_side_pass={entry['optimizer_side_pass']}")
            print(
                f"    Gen {gen}: global best improved to {entry['total_score']:.4f}"
                f" ({', '.join(details)})",
                flush=True,
            )
        else:
            self.archive_stagnation += 1
            stagnant_gens = self.archive_stagnation * self.checkpoint_every
            if stagnant_gens >= 100000 and self.global_best_exact is not None:
                print(
                    f"    Gen {gen}: early stop — archive stagnant for {stagnant_gens} gens"
                    f" (best score={self.global_best_exact['total_score']:.4f})",
                    flush=True,
                )
                self._should_stop = True

        if self.global_best_genome is None or self.global_best_exact is None:
            raise RuntimeError(f"No fully valid genome available for generation {gen} checkpoint")
        archive_genome = self.global_best_genome
        archive_layout = self.layout.clone_with(genome=archive_genome)
        archive_entry = dict(self.global_best_exact or entry)
        self.best_exact = archive_entry
        self.exact_history.append(entry)
        if len(self.exact_history) > 20:
            self.exact_history = self.exact_history[-20:]
        arc_dup, arc_comp, arc_arr, arc_acc = self._layout_reports(archive_layout)
        self._annotate_clusters(archive_entry, archive_layout)

        checkpoint = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generation": gen,
            "best_genome": [int(x) for x in archive_genome],
            "best_objectives": [float(x) for x in archive_entry["objectives"]],
            "best_constraints": [float(x) for x in archive_entry["constraints"]],
            "best_exact": archive_entry,
            "best_source": "global_exact_archive",
            "best_generation": self.global_best_generation,
            "population_best_genome": [int(x) for x in best_genome],
            "population_best_objectives": [float(x) for x in entry["objectives"]],
            "population_best_constraints": [float(x) for x in entry["constraints"]],
            "population_best_exact": entry,
            "population_acceptance_report": acc,
            "exact_eval_history": self.exact_history,
            "duplicate_report": arc_dup,
            "completion_cluster_report": arc_comp,
            "arrow_report": arc_arr,
            "acceptance_report": arc_acc,
            "population_size": self.pop_size,
            "stagnation_count": self.stagnation_count,
        }
        path = os.path.join(self.build_dir, f"v2_checkpoint_gen{gen}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self, keep=5):
        paths = glob.glob(os.path.join(self.build_dir, "v2_checkpoint_gen*.json"))
        if len(paths) <= keep:
            return
        for p in sorted(paths, key=os.path.getmtime, reverse=True)[keep:]:
            try:
                os.remove(p)
            except OSError:
                pass
        removed = len(paths) - keep
        if removed > 0:
            print(f"  Cleanup: removed {removed} old checkpoint(s), kept {keep}", flush=True)

    # ------------------------------------------------------------------
    # Surrogate teacher + retrain
    # ------------------------------------------------------------------

    def _maybe_teacher_update(self, pop_X, pop_F, pop_cv, gen):
        sm = self.surrogate_manager
        if sm is None:
            return pop_F, pop_cv
        sm.generation = gen
        exact_eval_every = sm.exact_eval_every
        exact_refresh = False
        if exact_eval_every > 0 and gen % exact_eval_every == 0:
            t0 = time.perf_counter()
            exact_F, exact_G = self.evaluator.evaluate_batch(pop_X.astype(np.int32))
            if self.perf:
                self.perf.add("surrogate_teacher_eval", time.perf_counter() - t0)
            sm.add_exact_evaluations(pop_X.astype(np.int32), exact_F, exact_G)
            # These labels are already paid for. Use them to rescore the live
            # population and to update exact feasibility/anchor tracking; never
            # leave parent fitness on stale surrogate estimates after refresh.
            pop_F = np.asarray(exact_F, dtype=np.float32)
            pop_cv = np.maximum(np.asarray(exact_G, dtype=np.float32), 0.0)
            self._maybe_update_global_from_batch(pop_X, exact_F, exact_G, gen)
            exact_refresh = True
        swapped = sm.maybe_collect_retrain()
        if swapped and not exact_refresh:
            # A model swap changes both prediction weights and normalization.
            # Reprice every parent before comparing it with children scored by
            # the new model on the next generation.
            pred = sm.trainer.predict(pop_X.astype(np.int32))
            pop_F = np.asarray(pred[:, :self.n_factors], dtype=np.float32)
            pop_cv = np.maximum(
                np.asarray(pred[:, self.n_factors:], dtype=np.float32), 0.0
            )
        if sm.should_retrain():
            t0 = time.perf_counter()
            sm.async_retrain()
            if self.perf:
                self.perf.add("surrogate_retrain_submit", time.perf_counter() - t0)
        return pop_F, pop_cv

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, n_gen, initial_pop_X=None):
        """Run the GA for n_gen generations. Returns results dict."""
        from run_evolution import generate_random_layouts

        t_start = time.perf_counter()

        # Initialize population
        if initial_pop_X is not None:
            pop_X = initial_pop_X.astype(np.int32).copy()
        else:
            pop_X = generate_random_layouts(self.layout, self.pop_size)
        pop_X = pop_X.astype(np.int32)

        # Constraint columns: surrogate predicts objectives + constraints, so cv arrays
        # must match the real constraint count.
        n_constraint_cols = self.n_constraints if self.n_constraints > 0 else len(self.hard_constraints)

        # Initial surrogate manager reference (used while seeding the archive).
        sm = self.surrogate_manager

        # Exact-evaluate the first individual so its label and constraint profile
        # are authoritative. On a fresh start this is only the empty mutable
        # reference canvas; a saved genome is present here only when explicitly
        # supplied as a warmstart.
        ws_F, ws_G = self.evaluator.evaluate_batch(pop_X[:1].astype(np.int32))
        if sm is not None:
            sm.add_exact_evaluations(pop_X[:1].astype(np.int32), ws_F, ws_G)
        # Seed the archive with the warmstart so the first checkpoint can only improve.
        ws_layout = self.layout.clone_with(genome=pop_X[0].astype(np.int32))
        ws_exact = self.evaluator.evaluate(ws_layout)
        ws_entry = self._exact_entry(ws_exact, 0)
        _, ws_comp, ws_arr, ws_acc = self._layout_reports(ws_layout)
        self._annotate(ws_entry, ws_acc)
        self._annotate_clusters(ws_entry, ws_layout)

        # Self-ref momentary holds are functionally no-ops, but they are not illegal
        # if the initial individual already passes acceptance. Sanitize only when
        # the initial individual is infeasible, then re-evaluate it.
        if _warmstart_needs_access_sanitizing(ws_entry) and hasattr(self.mutation, 'sanitize_self_ref_momentary'):
            n_cleared = self.mutation.sanitize_self_ref_momentary(pop_X[0])
            if n_cleared > 0:
                print(f"    Initial individual sanitized: {n_cleared} self-ref momentary hold keys cleared.", flush=True)
                ws_F, ws_G = self.evaluator.evaluate_batch(pop_X[:1].astype(np.int32))
                if sm is not None:
                    sm.add_exact_evaluations(pop_X[:1].astype(np.int32), ws_F, ws_G)
                ws_layout = self.layout.clone_with(genome=pop_X[0].astype(np.int32))
                ws_exact = self.evaluator.evaluate(ws_layout)
                ws_entry = self._exact_entry(ws_exact, 0)
                _, ws_comp, ws_arr, ws_acc = self._layout_reports(ws_layout)
                self._annotate(ws_entry, ws_acc)
                self._annotate_clusters(ws_entry, ws_layout)

        self._update_archive(pop_X[0].astype(np.int32), ws_entry)
        self._consider_infeasible_anchor(pop_X[0], ws_F[0], ws_G[0])
        ws_total = float(ws_F[0].sum())
        ws_gap = _display_gap(ws_entry)
        details = []
        if ws_gap is not None:
            details.append(f"contract_failure_signal={ws_gap:.1f}")
        details.extend((f"optimizer_side_pass={ws_entry['optimizer_side_pass']}",
                        "added to surrogate training data"))
        print(
            f"    Initial slot-0 exact score={ws_total:.4f} ({', '.join(details)})",
            flush=True,
        )

        # Initial surrogate evaluation
        if sm is not None and sm.trainer.mean is not None:
            pred = sm.trainer.predict(pop_X)
            pop_F = pred[:, : self.n_factors].astype(np.float32)
            pop_cv = np.maximum(pred[:, self.n_factors :], 0).astype(np.float32)
            if pop_cv.shape[1] == 0 and n_constraint_cols > 0:
                pop_cv = np.zeros((self.pop_size, n_constraint_cols), dtype=np.float32)
        else:
            t0 = time.perf_counter()
            pop_F, pop_G = self.evaluator.evaluate_batch(pop_X)
            if self.perf:
                self.perf.add("exact_eval", time.perf_counter() - t0)
            pop_cv = (
                np.maximum(pop_G, 0).astype(np.float32)
                if pop_G.shape[1] > 0
                else np.zeros((self.pop_size, n_constraint_cols), dtype=np.float32)
            )

        pop_F[0] = ws_F[0]
        if ws_G.shape[1] > 0:
            pop_cv[0] = np.maximum(ws_G[0], 0)

        gen_times = []

        for gen in range(1, n_gen + 1):
            t_gen = time.perf_counter()
            self._generation_stage_seconds = 0.0

            # --- Staged semantic-cluster multiplier ---
            stage_t0 = time.perf_counter()
            if self._apply_semantic_multiplier_schedule(gen):
                pop_F, pop_cv = self._refresh_semantic_scores(pop_X)
            self._record_generation_stage("semantic_schedule_and_refresh", stage_t0)

            # --- Tournament selection ---
            stage_t0 = time.perf_counter()
            scalar = _scalar(pop_F, pop_cv)
            parent_idx = _tournament_select(scalar, self.pop_size)
            fallback_X = pop_X[parent_idx].astype(np.int32).copy()
            self._record_generation_stage("parent_selection", stage_t0)

            # --- Crossover ---
            stage_t0 = time.perf_counter()
            children_X = _crossover_batch(
                pop_X, parent_idx, self.crossover_prob, self.n_shortcuts
            )
            self._record_generation_stage("crossover", stage_t0)

            # --- Mutation (call directly, problem=None is safe) ---
            stage_t0 = time.perf_counter()
            self.mutation._do(None, children_X)
            self._record_generation_stage("mutation", stage_t0)

            # --- Repair ---
            stage_t0 = time.perf_counter()
            self.sanitizer._do(None, children_X)
            self._record_generation_stage("sanitizer", stage_t0)

            # Exact-evaluate each child once. The GPU batch supplies both the
            # true objective for survivor selection and the hard-contract audit;
            # surrogate scores must never decide survival for audited children.
            t0 = time.perf_counter()
            children_F, children_G = self.evaluator.evaluate_batch(children_X)
            children_cv = np.maximum(children_G, 0).astype(np.float32)
            self._record_generation_stage("exact_eval", t0)

            stage_t0 = time.perf_counter()
            if sm is not None and sm.trainer.mean is not None:
                valid_rows = np.all(children_G <= 0, axis=1)
                if np.any(valid_rows):
                    sm.add_exact_evaluations(children_X[valid_rows], children_F[valid_rows],
                                             children_G[valid_rows])
                    self._maybe_update_global_from_batch(children_X[valid_rows],
                                                         children_F[valid_rows],
                                                         children_G[valid_rows], gen)
                _reject_invalid_children(
                    children_X, children_F, children_cv, fallback_X,
                    pop_F[parent_idx], pop_cv[parent_idx], children_G,
                )
            elif np.any(np.any(children_G > 0, axis=1)):
                _reject_invalid_children(
                    children_X, children_F, children_cv, fallback_X,
                    pop_F[parent_idx], pop_cv[parent_idx], children_G,
                )
            self._record_generation_stage("contract_filter_and_archive", stage_t0)

            # --- (µ+λ) survival: keep best pop_size from parents + children ---
            stage_t0 = time.perf_counter()
            all_X = np.concatenate([pop_X, children_X], axis=0)
            all_F = np.concatenate([pop_F, children_F], axis=0)
            all_cv = np.concatenate([pop_cv, children_cv], axis=0) if 'children_cv' in dir() else None
            anchor_index = None
            if self.global_best_genome is None and self.infeasible_anchor_genome is not None:
                anchor_index = len(all_X)
                all_X = np.concatenate([all_X, self.infeasible_anchor_genome[None, :]], axis=0)
                all_F = np.concatenate([all_F, self.infeasible_anchor_objectives[None, :]], axis=0)
                if all_cv is not None:
                    all_cv = np.concatenate([all_cv, self.infeasible_anchor_constraints[None, :]], axis=0)
            archive_index = None
            if self.global_best_genome is not None and self.global_best_exact is not None:
                archive_index = len(all_X)
                all_X = np.concatenate([all_X, self.global_best_genome[None, :].astype(np.int32)], axis=0)
                all_F = np.concatenate([
                    all_F, np.asarray(self.global_best_exact["objectives"], dtype=np.float32)[None, :],
                ], axis=0)
                if all_cv is not None:
                    all_cv = np.concatenate([
                        all_cv, np.maximum(
                            np.asarray(self.global_best_exact.get("constraints", ()), dtype=np.float32), 0.0
                        )[None, :],
                    ], axis=0)
            all_scalar = _scalar(all_F, all_cv)
            n_pop = self.pop_size
            forced_indices = [i for i in (anchor_index, archive_index) if i is not None]
            survivors = _survivor_indices(
                all_scalar, all_cv, n_pop, forced_indices,
                relaxed=gen < self._relaxed_selection_until,
            )
            pop_X = all_X[survivors].astype(np.int32)
            pop_F = all_F[survivors]
            pop_cv = all_cv[survivors] if all_cv is not None else pop_cv
            self._record_generation_stage("survivor_selection", stage_t0)

            # --- Adaptive mutation rate ---
            stage_t0 = time.perf_counter()
            best_quality = float(_scalar(pop_F, pop_cv).min())
            self._adjust_mutation(best_quality, gen)
            self._record_generation_stage("adaptive_mutation", stage_t0)

            # --- Periodic: teacher update, retrain, checkpoint ---
            stage_t0 = time.perf_counter()
            pop_F, pop_cv = self._maybe_teacher_update(pop_X, pop_F, pop_cv, gen)
            self._record_generation_stage("teacher_update_total", stage_t0)

            if gen % self.checkpoint_every == 0:
                t0 = time.perf_counter()
                self._checkpoint(pop_X, pop_F, pop_cv, gen)
                self._record_generation_stage("checkpoint", t0)
                # Diversity injection (uses stagnation_count updated above)
                stage_t0 = time.perf_counter()
                pop_X, pop_F, pop_cv = self._inject_diversity(pop_X, pop_F, pop_cv, gen)
                self._record_generation_stage("diversity_injection", stage_t0)

            gen_times.append(time.perf_counter() - t_gen)
            if self.perf:
                self.perf.add("generation_total", gen_times[-1])
                self.perf.add(
                    "generation_unattributed",
                    max(0.0, gen_times[-1] - self._generation_stage_seconds),
                )
            if gen % self.checkpoint_every == 0:
                elapsed = time.perf_counter() - t_start
                self._write_live_perf(gen, elapsed)
                rate = gen / max(elapsed, 1e-9)
                print(
                    f"  Throughput at gen {gen}: {rate:.3f} gen/s "
                    f"({1000.0 / max(rate, 1e-9):.1f} ms/gen)",
                    flush=True,
                )
            if self._should_stop:
                break

        total_time = time.perf_counter() - t_start
        completed_generations = len(gen_times)
        if completed_generations and completed_generations % self.checkpoint_every:
            checkpoint_t0 = time.perf_counter()
            self._checkpoint(pop_X, pop_F, pop_cv, completed_generations)
            self._record_generation_stage("checkpoint", checkpoint_t0)
            self._write_live_perf(completed_generations, time.perf_counter() - t_start)

        avg_gen_ms = 1000.0 * sum(gen_times) / max(len(gen_times), 1)
        gens_per_sec = len(gen_times) / max(total_time, 1e-9)
        print(
            f"\nCustomGA finished {len(gen_times)} gens in {total_time:.1f}s"
            f" ({gens_per_sec:.1f} gen/sec, {avg_gen_ms:.1f}ms/gen avg)",
            flush=True,
        )

        return {
            "pop_X": pop_X,
            "pop_F": pop_F,
            "global_best_genome": self.global_best_genome,
            "global_best_exact": self.global_best_exact,
            "global_best_generation": self.global_best_generation,
            "best_exact": self.best_exact,
            "total_time": total_time,
            "gens_run": len(gen_times),
            "archive_acceptance_rejections": dict(self.archive_acceptance_rejections),
            "archive_duplicate_prefilter_rejections": self.archive_duplicate_prefilter_rejections,
            "archive_cluster_prefilter_rejections": self.archive_cluster_prefilter_rejections,
            "archive_cluster_prefilter_reasons": dict(self.archive_cluster_prefilter_reasons),
        }
