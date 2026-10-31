"""Run the full pymoo evolution loop."""
import argparse
import sys
import os
import random
import numpy as np
import torch
import time
import json
import glob
import cProfile
import pstats
import hashlib

# Unbuffer stdout so manager can see progress in real-time
sys.stdout.reconfigure(line_buffering=True)

# Use all logical CPU threads for Numba prange (crossover + exact eval kernels)
import os as _os
_numba_threads = int(_os.environ.get("NUMBA_NUM_THREADS", _os.cpu_count() or 8))
_os.environ.setdefault("NUMBA_NUM_THREADS", str(_numba_threads))

sys.path.insert(0, os.path.dirname(__file__))

from core.loader import build_layout
from fitness.evaluator import FitnessEvaluator
from fitness.kernel import NUMBA_AVAILABLE
from fitness.kernel import _app_matches, _shortcut_duplicate_support
from evolution import create_algorithm, LayoutProblem
from evolution.surrogate import LayoutSurrogate, SurrogateTrainer, SurrogateManager
from evolution.completion_cluster import analyze_completion_cluster
from evolution.arrow_cluster import analyze_arrows
from evolution.acceptance import build_acceptance_report
from evolution.layout_contract import CONTRACT_SHA256
from config import Config


def _fitness_scale_fingerprint(config_path, data_dir):
    """Identify inputs that determine exact objective scales.

    Scale caches without this fingerprint can silently survive evaluator,
    config, or usage-data changes and badly distort the weighted objective.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.abspath(config_path), os.path.abspath(__file__)]
    data_root = os.path.abspath(data_dir)
    roots = (
        os.path.join(repo_root, "core"),
        os.path.join(repo_root, "fitness"),
        data_root,
        os.path.join(repo_root, "contracts"),
    )
    for root in roots:
        if not os.path.isdir(root):
            continue
        for current, _, names in os.walk(root):
            for name in names:
                path = os.path.join(current, name)
                if root == data_root:
                    candidates.append(path)
                elif name.endswith((".py", ".json", ".yaml", ".yml")):
                    candidates.append(path)
    digest = hashlib.sha256()
    for path in sorted(set(candidates)):
        if not os.path.isfile(path):
            continue
        digest.update(os.path.relpath(path).encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _scale_cache_matches(cache, fingerprint):
    return (
        isinstance(cache, dict)
        and cache.get("fingerprint") == fingerprint
        and isinstance(cache.get("scale_factors"), list)
        and len(cache["scale_factors"]) == 3
    )

from pymoo.optimize import minimize
from pymoo.core.callback import Callback


def generate_random_layouts(layout, n, warmstart_genome=None):
    from evolution import build_group_placements
    from evolution.layout_contract import owner_protected_indices

    seed = layout.genome.astype(np.int32).copy() if warmstart_genome is None else warmstart_genome.astype(np.int32).copy()
    protected = set(int(i) for i in layout.frozen_indices)
    if warmstart_genome is not None:
        protected.update(owner_protected_indices(layout, seed))
    frozen = np.asarray(sorted(protected), dtype=np.int32)
    mutable = np.asarray([i for i in range(layout.n_positions) if i not in protected], dtype=np.int32)
    n_shortcuts = layout.n_shortcuts
    layouts = np.full((n, layout.n_positions), -1, dtype=np.int32)
    if n <= 0:
        return layouts

    frozen_assigned = {int(sid) for sid in seed[frozen] if sid >= 0}

    # Exclude group sids from individual random assignment — they are placed
    # as complete groups at valid anchors so initial genomes are never scattered.
    groups = build_group_placements(layout)
    group_sids_all = {sid for sid_tuple, _ in groups for sid in sid_tuple}
    layer_access_sids = {
        shortcut.sid for shortcut in layout.shortcuts if shortcut.is_layer_access
    }

    available_sids = [
        sid for sid in range(n_shortcuts)
        if sid not in frozen_assigned and sid not in group_sids_all
    ]
    mutable_index = {int(pos): i for i, pos in enumerate(mutable)}
    group_records = []
    for sid_tuple, anchor_list in groups:
        anchor_matrix = np.asarray(anchor_list, dtype=np.int32)
        if anchor_matrix.ndim != 2 or anchor_matrix.shape[1] != len(sid_tuple):
            raise ValueError(f"Group anchor shape mismatch for shortcut group {sid_tuple}")
        anchor_mutable_indices = np.asarray(
            [[mutable_index.get(int(pos), -1) for pos in anchor] for anchor in anchor_matrix],
            dtype=np.int32,
        )
        group_records.append((sid_tuple, anchor_matrix, anchor_mutable_indices))
    layer_access_lut = np.zeros(n_shortcuts, dtype=np.bool_)
    if layer_access_sids:
        layer_access_lut[np.fromiter(layer_access_sids, dtype=np.int32)] = True

    if warmstart_genome is not None:
        layouts[0] = warmstart_genome.astype(np.int32).copy()
        layouts[0, frozen] = seed[frozen]
        n_perturbed = min(n - 1, max(1, n // 5))
        mutable_list = mutable.tolist()
        for i in range(1, n_perturbed + 1):
            g = warmstart_genome.astype(np.int32).copy()
            g[frozen] = seed[frozen]
            n_swaps = random.randint(20, 80)
            for _ in range(n_swaps):
                ia = random.randrange(len(mutable_list))
                ib = random.randrange(len(mutable_list))
                g[mutable_list[ia]], g[mutable_list[ib]] = g[mutable_list[ib]], g[mutable_list[ia]]
            layouts[i] = g
        rand_start = n_perturbed + 1
    else:
        layouts[0] = seed
        rand_start = 1

    for i in range(rand_start, n):
        layouts[i, frozen] = seed[frozen]
        n_assign = min(len(mutable), len(available_sids))
        assigned = random.sample(available_sids, n_assign)
        layouts[i, mutable[:n_assign]] = assigned
        free_mask = layouts[i, mutable] < 0
        occupied_group_positions = np.zeros(layout.n_positions, dtype=np.bool_)
        row_values = layouts[i]
        occupied_access_positions = np.zeros(layout.n_positions, dtype=np.bool_)
        assigned_positions = np.flatnonzero(row_values >= 0)
        if len(assigned_positions):
            assigned_sids = row_values[assigned_positions]
            occupied_access_positions[assigned_positions] = layer_access_lut[assigned_sids]

        # Place groups without clobbering members placed for earlier groups.
        # Group definitions can overlap in shortcut IDs and geometry. The
        # complete Norwegian shape and explicit relative groups appear first;
        # later overlapping groups must not scatter those assignments.
        placed_group_sids = set()
        placed_group_positions = set()
        for sid_tuple, anchor_matrix, anchor_mutable_indices in group_records:
            group_sid_set = set(sid_tuple)
            if group_sid_set & placed_group_sids:
                continue
            conflicts = (
                occupied_group_positions[anchor_matrix].any(axis=1)
                | occupied_access_positions[anchor_matrix].any(axis=1)
            )
            compatible_indices = np.flatnonzero(~conflicts)
            if not len(compatible_indices):
                continue
            chosen = int(compatible_indices[random.randrange(len(compatible_indices))])
            anchor_array = anchor_matrix[chosen]
            anchor = tuple(int(pos) for pos in anchor_array)
            current_group_mutable_indices = anchor_mutable_indices[chosen]
            # Read what's at the anchor positions now (to displace them).
            displaced = [int(sid) for sid in layouts[i, anchor_array]]
            # Place group sids at anchor.
            for sid, pos in zip(sid_tuple, anchor):
                layouts[i, pos] = sid
            # Displaced non-group sids go to the first free mutable slots.
            fill_sids = [s for s in displaced if s >= 0 and s not in group_sids_all]
            anchor_free_mask = free_mask.copy()
            valid_anchor_indices = current_group_mutable_indices[current_group_mutable_indices >= 0]
            anchor_free_mask[valid_anchor_indices] = False
            free_slot_indices = np.flatnonzero(anchor_free_mask)
            free_slots = mutable[free_slot_indices]
            for pos, sid in zip(free_slots, fill_sids):
                layouts[i, pos] = sid
            free_mask[valid_anchor_indices] = False
            free_mask[free_slot_indices[:len(fill_sids)]] = False
            occupied_group_positions[anchor_array] = True
            occupied_access_positions[anchor_array] = np.fromiter(
                (int(sid) in layer_access_sids for sid in sid_tuple),
                dtype=np.bool_, count=len(sid_tuple),
            )
            placed_group_sids.update(group_sid_set)
            placed_group_positions.update(anchor)

        # Groups skipped because of shared members or unavailable nonoverlap
        # anchors still need assignable instances. Fill unclaimed positions
        # after all protected group placements are complete.
        mutable_values = layouts[i, mutable]
        assigned_group_sids = set(
            int(sid) for sid in mutable_values[np.isin(mutable_values, tuple(group_sids_all))]
        )
        missing_group_sids = sorted(
            group_sids_all - frozen_assigned - assigned_group_sids
        )
        free_slots = mutable[np.flatnonzero(layouts[i, mutable] < 0)]
        for pos, sid in zip(free_slots, missing_group_sids):
            layouts[i, pos] = sid

    return layouts


def evaluate_exact_batch(layouts, layout, evaluator, perf=None, label="exact_eval"):
    """Evaluate exact fitness for a batch of layouts using the compiled model."""
    t0 = time.perf_counter()
    try:
        scores, constraints = evaluator.evaluate_batch(layouts)
        if perf is not None:
            perf.add(label, time.perf_counter() - t0)
        return scores, constraints
    except Exception as exc:
        print(f"  Compiled exact evaluator failed, falling back to sequential: {exc}", flush=True)

    n = layouts.shape[0]
    scores = np.zeros((n, 3), dtype=np.float32)
    constraints = np.zeros((n, len(evaluator.hard_constraints)), dtype=np.float32)
    for i in range(n):
        result = evaluator.evaluate(layout.clone_with(genome=layouts[i]))
        scores[i] = result.objectives
        constraints[i] = result.constraints

    if perf is not None:
        perf.add(label, time.perf_counter() - t0)
    return scores, constraints


def validate_exact_evaluator(layout, evaluator, tolerance=1e-4, n=16):
    """Validate the compiled evaluator against sequential Python evaluation."""
    tolerance = float(tolerance)
    try:
        parity = evaluator.model.validate_parity(evaluator, n=n, tolerance=tolerance)
        if parity.ok:
            print(f"  Compiled exact evaluator enabled (parity max diff={parity.max_abs_diff:.3g}).", flush=True)
        else:
            print(f"  Compiled exact evaluator parity failed: {parity.message}", flush=True)
        return parity.ok
    except Exception as exc:
        print(f"  Compiled exact evaluator unavailable: {exc}", flush=True)
        return False


def maybe_validate_exact_evaluator(config, layout, evaluator):
    """Run expensive compiled/parity validation only when explicitly enabled."""
    if not bool(config.get("exact_eval.validate_parity", False)):
        print("  Exact evaluator parity validation skipped (exact_eval.validate_parity=false).", flush=True)
        return True
    return validate_exact_evaluator(
        layout,
        evaluator,
        tolerance=config.get("exact_eval.parity_tolerance", 1e-4),
        n=config.get("exact_eval.parity_samples", 16),
    )


def build_evaluator(config, layout, scale_factors=None):
    multiplier = float(config.get("fitness.semantic_cluster_multiplier", 200.0))
    for generation, value in config.get("fitness.semantic_cluster_multiplier_schedule", []) or []:
        if generation <= 0:
            multiplier = float(value)
    return FitnessEvaluator(
        weights=config.get("fitness.weights", {}),
        reference_layout=layout,
        scale_factors=scale_factors,
        violation_weights=config.get("fitness.violation_sub_weights", {}),
        missing_important_threshold=config.get("fitness.missing_important_threshold", 6.0),
        hard_constraints=config.get("fitness.hard_constraints", []),
        toggle_effort_multiplier=float(config.get("fitness.toggle_effort_multiplier", 2.5)),
        layer_access_thumb_params=config.get("fitness.layer_access_thumb", {}),
        require_cuda=bool(config.get("training.require_cuda", True)),
        semantic_cluster_multiplier=multiplier,
        semantic_cluster_pair_boost=float(config.get("fitness.semantic_cluster_pair_boost", 1.0)),
        semantic_contract_penalty=float(config.get("fitness.semantic_contract_penalty", 0.0)),
        semantic_position_penalty=float(config.get("fitness.semantic_position_penalty", 0.0)),
        sparse_layer_base_penalty=float(config.get("fitness.sparse_layer_base_penalty", 0.0)),
        sparse_layer_gap_penalty=float(config.get("fitness.sparse_layer_gap_penalty", 0.0)),
        right_alt_l0_penalty=float(config.get("fitness.right_alt_l0_penalty", 5000.0)),
        layer_usage_preference_penalty=float(
            config.get("fitness.layer_usage_preference_penalty", 10000.0)
        ),
        mouse_layer_shortcut_penalty=float(
            config.get("fitness.mouse_layer_shortcut_penalty", 10000.0)
        ),
        mouse_layer_access_position_penalty=float(
            config.get("fitness.mouse_layer_access_position_penalty", 2.0)
        ),
        owner_mouse_layer_shortcut_preferences=config.get(
            "fitness.owner_mouse_layer_shortcut_preferences", []
        ),
        owner_mouse_layer_shortcut_penalty=float(
            config.get("fitness.owner_mouse_layer_shortcut_penalty", 50000.0)
        ),
    )


def enforce_training_device_policy(config):
    """Fail fast when the configured training path would run CPU-primary.

    This project is GPU-focused.  CPU-only exact evaluation may be useful for
    unit tests or one-off diagnostics, but it must not silently become the
    training path for an evolution run.
    """
    from fitness.model import _CUDA_AVAILABLE

    require_cuda = bool(config.get("training.require_cuda", True))
    require_gpu_primary = bool(config.get("training.require_gpu_primary", True))
    allow_cpu_exact_validation = bool(config.get("training.allow_cpu_exact_validation", False))
    cuda_available = bool(torch.cuda.is_available())
    surrogate_enabled = bool(config.get("surrogate.enabled", False))

    if require_cuda and not cuda_available:
        raise SystemExit(
            "GPU training policy violation: CUDA is required, but torch.cuda.is_available() is false. "
            "Do not start a CPU training run."
        )
    if require_gpu_primary and not surrogate_enabled:
        raise SystemExit(
            "GPU training policy violation: GPU-primary training is required, but surrogate.enabled=false. "
            "The surrogate model is what keeps selection GPU-primary. "
            "Enable surrogate.enabled before starting a training run."
        )
    if require_gpu_primary and not allow_cpu_exact_validation and not _CUDA_AVAILABLE:
        raise SystemExit(
            "GPU training policy violation: allow_cpu_exact_validation=false, but the CUDA exact "
            "evaluator is not available (import/compilation failed). Fix the CUDA build or explicitly "
            "allow CPU validation while keeping training GPU-primary."
        )


_DUPLICATE_USAGE_INDEX_CACHE = {}


def _duplicate_usage_index(layout):
    """Pre-index static usage evidence once for this catalog and usage log."""
    signature = tuple((s.sid, s.keys, s.app) for s in layout.shortcuts)
    cache_key = (id(layout.usage_data), signature)
    cached = _DUPLICATE_USAGE_INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] is layout.usage_data:
        return cached[1]

    by_key = {}
    by_app = {}
    for shortcut in layout.shortcuts:
        by_key.setdefault(shortcut.keys, []).append(shortcut.sid)
        by_app.setdefault(shortcut.app, []).append(shortcut.sid)
    indexed = {name: {} for name in ("sequences", "workflows", "apps")}

    def add_key(bucket, row_key):
        for part in set(row_key.split(" -> ")):
            for sid in by_key.get(part, ()):
                bucket.setdefault(sid, []).append(row_key)

    for row_key, data in layout.usage_data.sequences.items():
        if isinstance(data, dict) and data.get("count", 0) >= 2:
            add_key(indexed["sequences"], row_key)
    for source in (layout.usage_data.chains, layout.usage_data.workflows):
        for row_key, data in source.items():
            if isinstance(data, dict) and data.get("count", 0) >= 2:
                add_key(indexed["workflows"], row_key)
    for row_key, data in layout.usage_data.app_workflows.items():
        if not isinstance(data, dict) or data.get("count", 0) < 2:
            continue
        apps = [part.strip() for part in row_key.split(" + ") if part.strip()]
        for app_name, sids in by_app.items():
            if any(_app_matches(app_name, app) for app in apps):
                for sid in sids:
                    indexed["apps"].setdefault(sid, []).append(row_key)

    if len(_DUPLICATE_USAGE_INDEX_CACHE) >= 16:
        _DUPLICATE_USAGE_INDEX_CACHE.pop(next(iter(_DUPLICATE_USAGE_INDEX_CACHE)))
    _DUPLICATE_USAGE_INDEX_CACHE[cache_key] = (layout.usage_data, indexed)
    return indexed


def unsupported_duplicate_sid_mask(layout):
    """Precompute the SIDs whose repeated placements fail the duplicate rule."""
    support = _shortcut_duplicate_support(layout)
    usage_index = _duplicate_usage_index(layout)
    mask = np.zeros(layout.n_shortcuts, dtype=np.bool_)
    for shortcut in layout.shortcuts:
        if shortcut.is_l0_only or shortcut.keys.startswith("_base_"):
            continue
        usage_row = layout.usage_data.shortcuts.get(shortcut.keys, {})
        has_usage = isinstance(usage_row, dict) and int(usage_row.get("count", 0)) > 0
        mouse_row = layout.usage_data.mouse_session_shortcuts.get(shortcut.keys, {})
        has_mouse_usage = isinstance(mouse_row, dict) and int(mouse_row.get("count", 0)) > 0
        has_reason = (
            has_usage
            or has_mouse_usage
            or bool(usage_index["sequences"].get(shortcut.sid))
            or bool(usage_index["workflows"].get(shortcut.sid))
            or bool(usage_index["apps"].get(shortcut.sid))
        )
        mask[int(shortcut.sid)] = float(support[int(shortcut.sid)]) < 0.25 and has_reason
    return mask


def analyze_duplicates(layout):
    """Explain which duplicate placements have usage support."""
    support = _shortcut_duplicate_support(layout)
    usage_index = _duplicate_usage_index(layout)
    counts = {}
    layers = {}
    positions = {}
    frozen_counts = {}
    for i, sid in enumerate(layout.genome):
        sid = int(sid)
        if sid < 0:
            continue
        counts[sid] = counts.get(sid, 0) + 1
        pos = layout.positions[i]
        layers.setdefault(sid, set()).add(pos.layer)
        if layout.frozen_mask[i]:
            frozen_counts[sid] = frozen_counts.get(sid, 0) + 1
        positions.setdefault(sid, []).append({
            "idx": int(i),
            "layer": int(pos.layer),
            "x": float(pos.x),
            "y": float(pos.y),
            "hand": pos.hand,
        })

    supported = []
    uncertain = []
    unsupported = []
    multi_workflow = []
    for sid, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count <= 1:
            continue
        shortcut = layout.shortcuts[sid]
        if shortcut.is_l0_only or shortcut.keys.startswith("_base_"):
            continue
        if frozen_counts.get(sid, 0) == count:
            continue
        seq_hits = usage_index["sequences"].get(sid, [])
        workflow_hits = usage_index["workflows"].get(sid, [])
        app_hits = usage_index["apps"].get(sid, [])
        mouse_hit = layout.usage_data.mouse_session_shortcuts.get(shortcut.keys, {})
        usage_count = 0
        usage_row = layout.usage_data.shortcuts.get(shortcut.keys, {})
        if isinstance(usage_row, dict):
            usage_count = int(usage_row.get("count", 0))
        reasons = []
        if usage_count > 0:
            reasons.append(f"shortcut_count={usage_count}")
        if seq_hits:
            reasons.append(f"sequence_hits={len(seq_hits)}")
        if workflow_hits:
            reasons.append(f"workflow_hits={len(workflow_hits)}")
        if app_hits:
            reasons.append(f"app_workflow_clusters={len(app_hits)}")
        if isinstance(mouse_hit, dict) and mouse_hit.get("count", 0) > 0:
            reasons.append(f"mouse_sessions={mouse_hit.get('count', 0)}")
        row = {
            "sid": int(sid),
            "keys": shortcut.keys,
            "app": shortcut.app,
            "count": int(count),
            "layers": sorted(int(x) for x in layers.get(sid, set())),
            "support": float(support[sid]),
            "reasons": reasons,
            "positions": positions.get(sid, []),
        }
        if support[sid] >= 0.75:
            row["classification"] = "workflow_supported"
            supported.append(row)
            if len(workflow_hits) + len(app_hits) >= 2:
                multi_workflow.append(row)
        elif not reasons:
            row["classification"] = "needs_more_logger_data"
            uncertain.append(row)
        elif support[sid] >= 0.25:
            row["classification"] = "partial_support_uncertain"
            uncertain.append(row)
        else:
            row["classification"] = "unsupported_or_wasteful"
            unsupported.append(row)
    return {
        "workflow_supported_duplicates": supported,
        "uncertain_duplicates_needing_more_data": uncertain,
        "unsupported_duplicates": unsupported,
        "multi_workflow_duplicates": multi_workflow,
    }


class PerfStats:
    def __init__(self):
        self.events = {}

    def add(self, name, seconds):
        item = self.events.setdefault(name, {"count": 0, "total_seconds": 0.0, "max_seconds": 0.0})
        item["count"] += 1
        item["total_seconds"] += float(seconds)
        item["max_seconds"] = max(item["max_seconds"], float(seconds))

    def summary(self):
        summary = {}
        for name, item in self.events.items():
            count = item["count"]
            summary[name] = {
                **item,
                "avg_seconds": item["total_seconds"] / count if count else 0.0,
            }
        return summary


class ExactEvalCallback(Callback):
    """Callback for exact-evaluation mode (no surrogate). Handles adaptive mutation,
    diversity injection, surrogate teacher updates, and checkpointing."""

    def __init__(
        self,
        layout,
        evaluator,
        build_dir,
        checkpoint_every=100,
        perf=None,
        surrogate_manager=None,
        semantic_multiplier_schedule=None,
    ):
        super().__init__()
        self.layout = layout
        self.evaluator = evaluator
        self.build_dir = build_dir
        self.checkpoint_every = checkpoint_every
        self.perf = perf
        self.surrogate_manager = surrogate_manager
        self.semantic_multiplier_schedule = list(semantic_multiplier_schedule or [])
        self.exact_history = []
        self.evolved_accuracy_history = []
        self.best_exact = None
        self.global_best_genome = None
        self.global_best_exact = None
        self.global_best_generation = None
        self.stagnation_count = 0
        self.last_best_quality = float('inf')
        self.base_mutation_prob = None
        self.last_diversity_reset = 0
        self.archive_stagnation = 0

    def _get_mutation(self, algorithm):
        mating = getattr(algorithm, 'mating', None)
        if mating is not None:
            return getattr(mating, 'mutation', None)
        return None

    def _best_index(self, pop_F, pop_G=None):
        """Select best individual: feasible first, then minimum sum of objectives."""
        if pop_F.ndim == 1 or pop_F.shape[1] == 1:
            return int(np.argmin(pop_F.ravel()))
        scalar = self._quality_scalar(pop_F, pop_G)
        return int(np.argmin(scalar))

    @staticmethod
    @staticmethod
    def _constraint_penalty_from_objectives(pop_F, min_penalty=2.0, max_penalty=25.0, spread_fraction=0.25):
        pop_F = np.asarray(pop_F, dtype=np.float64)
        totals = pop_F.sum(axis=1) if pop_F.ndim == 2 else pop_F
        if totals.size < 2:
            return float(min_penalty)
        q25, q75 = np.percentile(totals.astype(np.float64), [25, 75])
        spread = max(float(q75 - q25), float(np.std(totals)))
        if not np.isfinite(spread) or spread <= 0.0:
            return float(min_penalty)
        return float(np.clip(spread * spread_fraction, min_penalty, max_penalty))

    @classmethod
    def _quality_scalar(cls, pop_F, pop_G=None, constraint_penalty=None):
        """Exploration scalar quality for ranking and stagnation.

        Constraint violations should be costly enough to guide the population,
        but not so large that intermediate infeasible states cannot be explored.
        Final/archive selection uses _is_better_exact for hard acceptance tiers.
        """
        pop_F = np.asarray(pop_F, dtype=np.float64)
        if pop_F.ndim == 1:
            base = np.asarray(pop_F, dtype=np.float64).reshape(1, -1).sum(axis=1)
        else:
            base = np.asarray(pop_F, dtype=np.float64).sum(axis=1)
        if pop_G is not None and getattr(pop_G, "shape", (0, 0))[1] > 0:
            if constraint_penalty is None:
                constraint_penalty = cls._constraint_penalty_from_objectives(pop_F)
            cv = np.maximum(np.asarray(pop_G, dtype=np.float64), 0.0).sum(axis=1)
        else:
            cv = np.zeros(base.shape[0], dtype=np.float64)
            constraint_penalty = 0.0
        return base + float(constraint_penalty) * cv

    @staticmethod
    def _owner_contract_failed(exact_entry):
        failed = set(exact_entry.get("acceptance_failed_checks", []))
        return bool({"owner_layout_contract_v1", "l1_mouse_contract"} & failed)

    @classmethod
    def _display_gap(cls, exact_entry, target=None):
        if cls._owner_contract_failed(exact_entry):
            return 5.0
        return None if target is None else float(exact_entry["total_score"]) - float(target)

    def _exact_entry(self, exact_result, gen):
        return {
            "generation": int(gen),
            "objectives": [float(x) for x in exact_result.objectives],
            "constraints": [float(x) for x in exact_result.constraints],
            "factor_scores": {k: float(v) for k, v in exact_result.factor_scores.items()},
            "total_score": float(exact_result.total_score),
        }

    def _is_better_exact(self, candidate_entry, incumbent_entry):
        cand_cv = sum(max(0.0, float(x)) for x in candidate_entry.get("constraints", []))
        if (cand_cv != 0.0 or not candidate_entry.get("optimizer_side_pass", False)
                or self._owner_contract_failed(candidate_entry)):
            return False
        if incumbent_entry is None:
            return True
        inc_cv = sum(max(0.0, float(x)) for x in incumbent_entry.get("constraints", []))
        if cand_cv != inc_cv:
            return cand_cv < inc_cv
        cand_mouse_fail = self._owner_contract_failed(candidate_entry)
        inc_mouse_fail = self._owner_contract_failed(incumbent_entry)
        if cand_mouse_fail != inc_mouse_fail:
            return not cand_mouse_fail
        cand_completion = "norwegian_completion_cluster" not in candidate_entry.get(
            "acceptance_failed_checks", []
        )
        inc_completion = "norwegian_completion_cluster" not in incumbent_entry.get(
            "acceptance_failed_checks", []
        )
        if cand_completion != inc_completion:
            return cand_completion
        cand_accept = bool(candidate_entry.get("optimizer_side_pass", False))
        inc_accept = bool(incumbent_entry.get("optimizer_side_pass", False))
        if cand_accept != inc_accept:
            return cand_accept
        if not cand_accept:
            cand_failed = len(candidate_entry.get("acceptance_failed_checks", []))
            inc_failed = len(incumbent_entry.get("acceptance_failed_checks", []))
            if cand_failed != inc_failed:
                return cand_failed < inc_failed
        return float(candidate_entry["total_score"]) < float(incumbent_entry["total_score"])

    def _update_global_best(self, genome, exact_entry):
        if self._is_better_exact(exact_entry, self.global_best_exact):
            self.global_best_genome = genome.astype(np.int32).copy()
            self.global_best_exact = dict(exact_entry)
            self.global_best_generation = int(exact_entry["generation"])
            return True
        return False

    def _layout_reports(self, candidate_layout):
        duplicate = analyze_duplicates(candidate_layout)
        completion = analyze_completion_cluster(candidate_layout)
        arrow = analyze_arrows(candidate_layout)
        acceptance = build_acceptance_report(
            candidate_layout,
            duplicate_report=duplicate,
            completion_cluster_report=completion,
            arrow_report=arrow,
        )
        return duplicate, completion, arrow, acceptance

    def _annotate_acceptance(self, entry, acceptance):
        failed = [
            key for key, ok in acceptance.get("checks", {}).items()
            if key != "norwegian_export_bad_literal_count_zero" and not ok
        ]
        entry["optimizer_side_pass"] = bool(acceptance.get("optimizer_side_pass", False))
        entry["acceptance_failed_checks"] = failed
        return entry

    def _apply_semantic_multiplier_schedule(self, gen):
        """Ramp semantic-cluster pressure according to the configured schedule."""
        if not self.semantic_multiplier_schedule:
            return
        # Schedule is a list of [generation, multiplier]; apply the last entry
        # whose generation is <= current gen.
        target = None
        for sched_gen, sched_mult in self.semantic_multiplier_schedule:
            if gen >= sched_gen:
                target = float(sched_mult)
        if target is None:
            return
        current = getattr(self.evaluator.model, "semantic_cluster_multiplier", None)
        if current is None or abs(current - target) > 1e-9:
            print(
                f"  Gen {gen}: ramping semantic_cluster_multiplier from {current} to {target}",
                flush=True,
            )
            self.evaluator.set_semantic_cluster_multiplier(target)
            if self.surrogate_manager is not None:
                # Old exact-eval cache is on the previous fitness landscape; clear it
                # so the surrogate retrains on the new objective.
                self.surrogate_manager.clear_exact_cache()

    def _adjust_mutation_rate(self, algorithm, gen):
        mutation = self._get_mutation(algorithm)
        if mutation is None:
            return

        if self.base_mutation_prob is None:
            prob = getattr(mutation, 'prob', None)
            if prob is not None:
                self.base_mutation_prob = float(prob.value) if hasattr(prob, 'value') else float(prob)

        pop = algorithm.pop.get("F")
        if pop is None or pop.shape[0] == 0:
            return

        pop_G = algorithm.pop.get("G")
        # Overall layout quality improvement, feasible first. Objective gains
        # count only inside the same constraint-violation tier.
        best_quality = float(np.min(self._quality_scalar(pop, pop_G)))

        if best_quality < self.last_best_quality * 0.999:
            if self.stagnation_count > 0:
                current_prob = getattr(mutation, 'prob', None)
                if current_prob is not None and self.base_mutation_prob is not None:
                    if hasattr(current_prob, 'value'):
                        current_prob.value = self.base_mutation_prob
                    else:
                        mutation.prob = self.base_mutation_prob
                    print(f"    Gen {gen}: improvement detected (best_quality={best_quality:.3f}). Restoring mutation rate to {self.base_mutation_prob:.3f}", flush=True)
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
            if self.stagnation_count >= 100 and self.stagnation_count % 100 == 0:
                current_prob = getattr(mutation, 'prob', None)
                if current_prob is not None and self.base_mutation_prob is not None:
                    current_val = float(current_prob.value) if hasattr(current_prob, 'value') else float(current_prob)
                    new_prob = min(0.5, current_val * 1.2)
                    if hasattr(current_prob, 'value'):
                        current_prob.value = new_prob
                    else:
                        mutation.prob = new_prob
                    print(f"    Gen {gen}: stagnation for {self.stagnation_count} gens (best_quality={best_quality:.3f}). Increasing mutation rate to {new_prob:.3f}", flush=True)

        self.last_best_quality = best_quality

    def _inject_diversity_on_stagnation(self, algorithm, gen):
        if self.stagnation_count < 1200:
            return
        if gen - self.last_diversity_reset < 600:
            return

        pop_X = algorithm.pop.get("X")
        pop_F = algorithm.pop.get("F")
        pop_G = algorithm.pop.get("G")
        if pop_X is None or pop_F is None or pop_X.shape[0] < 8:
            return

        n_pop = pop_X.shape[0]
        n_replace = max(4, n_pop // 4)
        mutable = self.layout.mutable_indices
        if len(mutable) < 2:
            return

        scalar = self._quality_scalar(pop_F, pop_G)
        elite_count = max(2, n_pop // 20)
        elite_idx = set(np.argsort(scalar)[:elite_count].tolist())
        replace_order = [idx for idx in np.argsort(scalar)[::-1] if int(idx) not in elite_idx]
        replace_idx = replace_order[:n_replace]
        if not replace_idx:
            return

        seed = self.layout.genome.astype(np.int32)
        assigned_frozen = {int(seed[i]) for i in self.layout.frozen_indices if int(seed[i]) >= 0}
        available = np.asarray([sid for sid in range(self.layout.n_shortcuts) if sid not in assigned_frozen], dtype=np.int32)
        if len(available) == 0:
            return

        exact_genomes = []
        exact_indices = []
        for dst in replace_idx:
            genome = seed.copy()
            genome[mutable] = -1
            n_assign = min(len(mutable), len(available))
            shuffled_pos = np.random.permutation(mutable)
            shuffled_sids = np.random.permutation(available)[:n_assign]
            genome[shuffled_pos[:n_assign]] = shuffled_sids
            pop_X[dst] = genome
            exact_genomes.append(genome)
            exact_indices.append(dst)

        try:
            exact_F, exact_G = self.evaluator.evaluate_batch(np.asarray(exact_genomes, dtype=np.int32))
            for row, dst in enumerate(exact_indices):
                pop_F[dst] = exact_F[row]
                if pop_G is not None and exact_G.shape[1] > 0:
                    pop_G[dst] = exact_G[row]
        except Exception:
            pass

        algorithm.pop.set("X", pop_X)
        algorithm.pop.set("F", pop_F)
        if pop_G is not None:
            algorithm.pop.set("G", pop_G)

        self.last_diversity_reset = gen
        print(
            f"    Gen {gen}: stagnation for {self.stagnation_count} gens. Injected {len(replace_idx)} diverse individuals, kept {elite_count} elites.",
            flush=True,
        )

    def _save_checkpoint(self, algorithm, gen):
        pop_X = algorithm.pop.get("X")
        pop_F = algorithm.pop.get("F")
        pop_G = algorithm.pop.get("G")
        best_idx = self._best_index(pop_F, pop_G)
        population_best_genome = pop_X[best_idx].astype(np.int32)
        population_best_layout = self.layout.clone_with(genome=population_best_genome)
        population_exact_result = self.evaluator.evaluate(population_best_layout)
        population_exact_entry = self._exact_entry(population_exact_result, gen)

        (
            population_duplicate_report,
            population_completion_report,
            population_arrow_report,
            population_acceptance_report,
        ) = self._layout_reports(population_best_layout)
        self._annotate_acceptance(population_exact_entry, population_acceptance_report)

        improved_archive = self._update_global_best(population_best_genome, population_exact_entry)
        if improved_archive:
            self.archive_stagnation = 0
            print(
                f"    Gen {gen}: global best improved to {population_exact_entry['total_score']:.4f} "
                f"(optimizer_side_pass={population_exact_entry['optimizer_side_pass']})",
                flush=True,
            )
        else:
            self.archive_stagnation += 1
            # Early stop: archive hasn't improved for 5000 gens (measured in checkpoint steps)
            # and current best already passes all optimizer-side checks.
            stagnant_gens = self.archive_stagnation * self.checkpoint_every
            if (
                stagnant_gens >= 5000
                and self.global_best_exact is not None
                and self.global_best_exact.get("optimizer_side_pass", False)
            ):
                print(
                    f"    Gen {gen}: early stop — archive stagnant for {stagnant_gens} gens "
                    f"with optimizer_side_pass=True (best score={self.global_best_exact['total_score']:.4f})",
                    flush=True,
                )
                algorithm.termination.force_termination = True

        if self.global_best_genome is None or self.global_best_exact is None:
            raise RuntimeError(f"No fully valid genome available for generation {gen} checkpoint")
        best_genome = self.global_best_genome
        best_layout = self.layout.clone_with(genome=best_genome)
        exact_entry = dict(self.global_best_exact or population_exact_entry)
        self.best_exact = exact_entry
        self.exact_history.append(population_exact_entry)
        if len(self.exact_history) > 20:
            self.exact_history = self.exact_history[-20:]
        duplicate_report, completion_report, arrow_report, acceptance_report = self._layout_reports(best_layout)
        checkpoint = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generation": gen,
            "contract_sha256": CONTRACT_SHA256,
            "best_genome": [int(x) for x in best_genome],
            "best_objectives": [float(x) for x in exact_entry["objectives"]],
            "best_constraints": [float(x) for x in exact_entry["constraints"]],
            "best_exact": exact_entry,
            "best_source": "global_exact_archive",
            "best_generation": self.global_best_generation,
            "population_best_genome": [int(x) for x in population_best_genome],
            "population_best_objectives": [float(x) for x in population_exact_entry["objectives"]],
            "population_best_constraints": [float(x) for x in population_exact_entry["constraints"]],
            "population_best_exact": population_exact_entry,
            "population_acceptance_report": population_acceptance_report,
            "exact_eval_history": self.exact_history,
            "duplicate_report": duplicate_report,
            "completion_cluster_report": completion_report,
            "arrow_report": arrow_report,
            "acceptance_report": acceptance_report,
            "population_size": algorithm.pop_size,
            "stagnation_count": self.stagnation_count,
        }
        ckpt_path = os.path.join(self.build_dir, f"v2_checkpoint_gen{gen}.json")
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self, keep=5):
        paths = glob.glob(os.path.join(self.build_dir, "v2_checkpoint_gen*.json"))
        if len(paths) <= keep:
            return
        paths_sorted = sorted(paths, key=os.path.getmtime, reverse=True)
        for path in paths_sorted[keep:]:
            try:
                os.remove(path)
            except OSError as exc:
                print(f"  Warning: could not remove old checkpoint {path}: {exc}", flush=True)
        removed = len(paths_sorted) - keep
        if removed > 0:
            print(f"  Cleanup: removed {removed} old checkpoint(s), kept {keep}", flush=True)

    def notify(self, algorithm):
        gen = algorithm.n_iter

        self._apply_semantic_multiplier_schedule(gen)
        self._adjust_mutation_rate(algorithm, gen)
        self._inject_diversity_on_stagnation(algorithm, gen)
        if self.surrogate_manager is not None:
            self.surrogate_manager.generation = gen
            if (
                gen > 0
                and self.surrogate_manager.exact_eval_every > 0
                and gen % self.surrogate_manager.exact_eval_every == 0
            ):
                pop_X = algorithm.pop.get("X")
                if pop_X is not None:
                    t0 = time.perf_counter()
                    exact_F, _ = self.evaluator.evaluate_batch(pop_X.astype(np.int32))
                    self.surrogate_manager.add_exact_evaluations(pop_X.astype(np.int32), exact_F)
                    if self.perf is not None:
                        self.perf.add("surrogate_teacher_eval", time.perf_counter() - t0)
            if self.surrogate_manager.should_retrain():
                t0 = time.perf_counter()
                self.surrogate_manager.retrain()
                if self.perf is not None:
                    self.perf.add("surrogate_retrain", time.perf_counter() - t0)

        if gen > 0 and gen % self.checkpoint_every == 0:
            t0 = time.perf_counter()
            self._save_checkpoint(algorithm, gen)
            if self.perf is not None:
                self.perf.add("checkpoint", time.perf_counter() - t0)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the Charybdis v2 layout evolution.",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory containing layout.json, app_shortcut_scores.json, usage_stats.json (default: data)",
    )
    parser.add_argument(
        "--config", default="config_v2.yaml",
        help="Path to config YAML (default: config_v2.yaml)",
    )
    parser.add_argument(
        "--output-dir", default="build",
        help="Directory for checkpoints, results, and profiling output (default: build)",
    )
    parser.add_argument(
        "--generations", "-g", type=int, default=None,
        help="Override evolution.n_generations from config",
    )
    parser.add_argument(
        "--pop-size", "-p", type=int, default=None,
        help="Override evolution.pop_size from config",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override evolution.seed from config",
    )
    parser.add_argument(
        "--no-inject-seed", action="store_true",
        help="Disable injection of the pre-seeded genome",
    )
    parser.add_argument(
        "--profile-fast", action="store_true",
        help="Run cProfile and write v2_profile_fast.pstats/txt to output-dir",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    config = Config.load(args.config)
    perf = PerfStats()
    enforce_training_device_policy(config)

    n_gen = args.generations if args.generations is not None else config.get("evolution.n_generations", 10)
    pop_size = args.pop_size if args.pop_size is not None else config.get("evolution.pop_size", 50)
    seed = args.seed if args.seed is not None else config.get("evolution.seed", 42)
    inject_seed = not args.no_inject_seed and config.get("evolution.inject_seed", True)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    print("=" * 60, flush=True)
    print("CHARYBDIS V2 EVOLUTION", flush=True)
    print("=" * 60, flush=True)
    print(f"  torch={torch.__version__}, cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  Numba available: {NUMBA_AVAILABLE}", flush=True)

    print(f"Loading data from {args.data_dir}...", flush=True)
    fitness_config = config.raw.get("fitness", {})
    layout = build_layout(args.data_dir, fitness_config)
    print(f"  Positions: {layout.n_positions}, Shortcuts: {layout.n_shortcuts}", flush=True)
    print(f"  Mutable: {len(layout.mutable_indices)}", flush=True)
    frozen_filled = sum(1 for p in layout.positions if p.is_frozen and layout.genome[p.gene_idx] >= 0)
    print(f"  Frozen genome positions filled: {frozen_filled} (L0 only — L7 is frozen outside training)", flush=True)

    hard_constraints = config.get("fitness.hard_constraints", [])

    # Try to warmstart from local search result.
    # Prefer an explicit copy in the run directory, but fall back to the project-wide
    # warmstart so we do not accidentally launch from a random population.
    warmstart_genome = None
    if not inject_seed:
        print("  Warmstart disabled by --no-inject-seed; starting from random population.", flush=True)
    warmstart_candidates = [
        os.path.join(args.output_dir, 'v2_local_search_result.json'),
        os.path.join(os.path.dirname(__file__), 'build', 'v2_local_search_result.json'),
        os.path.join(args.data_dir, 'default_layout_genome.json'),
    ]
    for warmstart_path in warmstart_candidates:
        if not inject_seed:
            break
        if not os.path.exists(warmstart_path):
            continue
        try:
            with open(warmstart_path) as f:
                ws = json.load(f)
            # Use top-level 'genome' which is updated by each run's best find;
            # 'best_exact' was set during local search with different scale factors.
            warmstart_genome = np.array(ws['genome'], dtype=np.int32)
            ws_score = ws.get('best_score', None)
            if ws_score is not None:
                ws_exact = dict(ws.get("best_exact", {}))
                ws_exact.setdefault("total_score", float(ws_score))
                ws_signal = ExactEvalCallback._display_gap(ws_exact)
                score_str = f"raw_score={ws_score:.4f}"
                if ws_signal is not None:
                    score_str += f", contract_failure_signal={ws_signal:+.2f}"
            else:
                score_str = "score=N/A"
            print(f"  Warmstart genome loaded from {warmstart_path} ({score_str})", flush=True)
            break
        except Exception as e:
            print(f"  Warmstart load failed for {warmstart_path}: {e}", flush=True)

    if warmstart_genome is not None:
        from evolution.layout_contract import audit_layout_contract
        warmstart_layout = layout.clone_with(genome=warmstart_genome)
        seed_acceptance = build_acceptance_report(
            warmstart_layout,
            analyze_duplicates(warmstart_layout),
            analyze_completion_cluster(warmstart_layout),
            analyze_arrows(warmstart_layout),
        )
        contract_audit = audit_layout_contract(warmstart_layout)
        failed_acceptance = [
            name for name, passed in seed_acceptance["optimizer_side_checks"].items()
            if not passed
        ]
        if not contract_audit["acceptance_pass"]:
            raise RuntimeError(
                "Warmstart fails the versioned owner layout contract: "
                + ", ".join(contract_audit["failed_checks"])
            )
        if failed_acceptance:
            raise RuntimeError(
                "Warmstart fails optimizer acceptance; refusing population fallback: "
                + ", ".join(failed_acceptance)
            )

    surrogate_enabled = bool(config.get("surrogate.enabled", False))
    evaluator = build_evaluator(config, layout)
    maybe_validate_exact_evaluator(config, layout, evaluator)

    # Score the actual warmstart when present; the reference layout is only
    # the frozen canvas and otherwise produces a misleading empty-seed score.
    seed_score_layout = (
        layout.clone_with(genome=warmstart_genome)
        if warmstart_genome is not None else layout
    )
    seed_result = evaluator.evaluate(seed_score_layout)
    print(f"  Seed objectives (unscaled): {seed_result.objectives.tolist()}", flush=True)

    # Compute or load stable IQR-based scale factors.
    # Scale factors must be stable across runs so that scores are comparable and the
    # optimizer doesn't over-weight objectives when the warmstart is already good.
    # Problem: generate_random_layouts perturbs the warmstart, so as the warmstart
    # improves the IQR shrinks, biasing the scale factors.
    # Fix: compute once from a fresh (unbiased) random sample and cache to disk.
    _scale_cache = os.path.join(args.output_dir, "v2_scale_factors.json")
    _scale_fingerprint = _fitness_scale_fingerprint(args.config, args.data_dir)
    scale_factors = None
    if os.path.exists(_scale_cache):
        try:
            with open(_scale_cache) as _f:
                _cached = json.load(_f)
            if _scale_cache_matches(_cached, _scale_fingerprint):
                scale_factors = np.array(_cached["scale_factors"], dtype=np.float64)
                print(f"  IQR scale factors (cached, fingerprint matched): effort={scale_factors[0]:.2f}, adj={scale_factors[1]:.2f}, viol={scale_factors[2]:.2f}", flush=True)
            else:
                print("  Scale cache is stale or unversioned; recomputing.", flush=True)
        except Exception as _e:
            print(f"  Scale cache load failed: {_e}; recomputing.", flush=True)
            scale_factors = None

    if scale_factors is None:
        # Use a fresh layout (no warmstart) so IQR reflects the true random distribution.
        # Problem: generate_random_layouts perturbs the warmstart genome, so as the warmstart
        # improves, the IQR shrinks — biasing scale factors each run. Fix: compute from fresh.
        # Calibrate against the active generation-zero landscape. Keep these scales fixed
        # during training so later scheduled cluster increases retain their extra pressure.
        fresh_layout = build_layout(args.data_dir, config.get("fitness.weights", {}))
        n_sample = 200  # more samples for stable IQR
        sample_layouts = generate_random_layouts(fresh_layout, n_sample)
        scale_evaluator = build_evaluator(config, fresh_layout)
        sample_scores, _ = evaluate_exact_batch(
            sample_layouts, fresh_layout, scale_evaluator, perf=perf, label="scale_sample_eval",
        )
        q25 = np.percentile(sample_scores, 25, axis=0)
        q75 = np.percentile(sample_scores, 75, axis=0)
        iqr = q75 - q25
        seed_result_scale = scale_evaluator.evaluate(fresh_layout)
        seed_scores = np.abs(seed_result_scale.objectives)
        scale_factors = np.maximum(iqr, seed_scores * 0.1)
        scale_factors = np.maximum(scale_factors, 1.0)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(_scale_cache, "w") as _f:
            json.dump({"scale_factors": scale_factors.tolist(), "fingerprint": _scale_fingerprint}, _f)
        print(f"  IQR scale factors (computed, cached): effort={scale_factors[0]:.2f}, adj={scale_factors[1]:.2f}, viol={scale_factors[2]:.2f}", flush=True)
    evaluator = build_evaluator(config, layout, scale_factors=scale_factors)
    maybe_validate_exact_evaluator(config, layout, evaluator)
    seed_result = evaluator.evaluate(seed_score_layout)
    print(f"  Seed objectives (scaled): {seed_result.objectives.tolist()}", flush=True)
    n_constraints = len(seed_result.constraints)

    surrogate_manager = None

    if surrogate_enabled:
        print("  GPU-primary surrogate training enabled.", flush=True)
        surrogate = LayoutSurrogate(
            layout.n_positions,
            layout.n_shortcuts,
            n_factors=3,
            n_constraints=n_constraints,
            hidden_dim=config.get("surrogate.hidden_dim", 256),
            embedding_dim=config.get("surrogate.embedding_dim", 32),
        )
        trainer = SurrogateTrainer(
            surrogate,
            device="cuda",
            mixed_precision=True,
            compile_model=config.get("surrogate.compile_model", False),
        )
        surrogate_manager = SurrogateManager(
            surrogate,
            trainer,
            retrain_every=config.get("surrogate.retrain_every", 500),
            exact_eval_every=config.get("surrogate.exact_eval_every", 100),
            retrain_epochs=config.get("surrogate.retrain_epochs", 10),
            retrain_batch_size=config.get("surrogate.batch_size", 1024),
            max_retrain_samples=config.get("surrogate.max_retrain_samples", 20000),
        )
        n_initial = int(config.get("surrogate.initial_exact_samples", 1000))
        t0_train = time.perf_counter()
        print(f"  Preparing {n_initial} initial surrogate teacher layouts...", flush=True)
        initial_layouts = generate_random_layouts(layout, n_initial, warmstart_genome=warmstart_genome)
        print(
            f"  Initial teacher layouts generated in {time.perf_counter() - t0_train:.1f}s; "
            "evaluating exact teacher labels...",
            flush=True,
        )
        initial_scores, initial_constraints = evaluate_exact_batch(
            initial_layouts, layout, evaluator, perf=perf, label="surrogate_initial_teacher_eval",
        )
        print(
            f"  Initial teacher labels evaluated in {time.perf_counter() - t0_train:.1f}s; "
            "training surrogate...",
            flush=True,
        )
        initial_combined = np.concatenate(
            [initial_scores, initial_constraints], axis=1
        ).astype(np.float32)
        surrogate_manager.add_exact_evaluations(initial_layouts, initial_combined)
        trainer.train(
            initial_layouts,
            initial_combined,
            epochs=config.get("surrogate.train_epochs", 30),
            batch_size=config.get("surrogate.batch_size", 256),
        )
        perf.add("surrogate_initial_train", time.perf_counter() - t0_train)
        print(
            f"  Surrogate trained on CUDA device={trainer.device} with {n_initial} exact teacher samples.",
            flush=True,
        )

    # Create pymoo problem.
    from evolution import SwapMutation, StructuralGenomeSanitizer
    from evolution.custom_ga import CustomGARunner, _has_unsupported_duplicate
    from evolution.layout_contract import owner_protected_indices

    protected_seed = (layout.genome.copy() if warmstart_genome is None
                      else warmstart_genome.astype(np.int32).copy())
    mutation_frozen_mask = layout.frozen_mask.copy()
    if warmstart_genome is not None:
        mutation_frozen_mask[list(owner_protected_indices(layout, protected_seed))] = True
    mutation_layout = layout.clone_with(genome=protected_seed)

    mutation = SwapMutation(
        prob=config.get("evolution.mutation_prob", 0.15),
        frozen_mask=mutation_frozen_mask,
        layout=mutation_layout,
        group_overwrite_prob=config.get("evolution.group_overwrite_prob", 0.15),
        mouse_workflow_prob=config.get("evolution.mouse_workflow_prob", 0.06),
        l7_access_prob=config.get("evolution.l7_access_prob", 0.03),
        random_assign_prob=config.get("evolution.random_assign_prob", 0.08),
        bulk_assign_prob=config.get("evolution.bulk_assign_prob", 0.04),
        optional_arrow_drop_prob=config.get("evolution.optional_arrow_drop_prob", 0.04),
        cluster_app_prob=config.get("evolution.cluster_app_prob", 0.20),
        effort_swap_prob=config.get("evolution.effort_swap_prob", 0.06),
        smart_duplicate_prob=config.get("evolution.smart_duplicate_prob", 0.20),
        mb45_move_prob=config.get("evolution.mb45_move_prob", 0.04),
    )
    sanitizer = StructuralGenomeSanitizer(
        n_shortcuts=layout.n_shortcuts,
        frozen_mask=mutation_frozen_mask,
        seed_genome=protected_seed,
        layout=layout,
    )
    crossover_prob = config.get("evolution.crossover_prob", 0.7)
    unsupported_duplicate_mask = unsupported_duplicate_sid_mask(layout)

    if warmstart_genome is not None:
        print(
            f"  Initial population: 1 warmstart + {pop_size - 1} mutated seed variants; "
            "hard-invalid or nonaccepted variants revert to the warmstart.",
            flush=True,
        )
    else:
        print("  All individuals start from fully random shortcut assignment (L0 thumb gets one random momentary hold).", flush=True)
    print(f"\nRunning evolution (CustomGA): pop={pop_size}, gens={n_gen}", flush=True)
    t0 = time.time()

    if warmstart_genome is not None:
        initial_pop_X = np.tile(warmstart_genome.astype(np.int32), (pop_size, 1))
        if pop_size > 1:
            mutation._do(None, initial_pop_X[1:])
            sanitizer._do(None, initial_pop_X[1:])
        _, initial_G = evaluator.evaluate_batch(initial_pop_X)
        valid_variations = 1
        for row in range(1, pop_size):
            if np.any(initial_G[row] > 0):
                initial_pop_X[row] = warmstart_genome
                continue
            if _has_unsupported_duplicate(
                initial_pop_X[row], unsupported_duplicate_mask,
                layout.frozen_mask,
            ):
                initial_pop_X[row] = warmstart_genome
                continue
            candidate_layout = layout.clone_with(genome=initial_pop_X[row])
            report = build_acceptance_report(
                candidate_layout, analyze_duplicates(candidate_layout),
                analyze_completion_cluster(candidate_layout), analyze_arrows(candidate_layout),
            )
            if report["optimizer_side_pass"]:
                valid_variations += 1
            else:
                initial_pop_X[row] = warmstart_genome
        print(f"  Valid seeded population: {valid_variations}/{pop_size} candidates under full optimizer acceptance.", flush=True)
    else:
        initial_pop_X = generate_random_layouts(layout, pop_size)

    # Parse semantic-cluster multiplier schedule if present.
    # Format: list of [generation, multiplier] entries, e.g.
    #   [[0, 200], [2000, 10000], [5000, 1000000]]
    semantic_schedule = config.get("fitness.semantic_cluster_multiplier_schedule", None)
    if semantic_schedule is None:
        # No explicit schedule: use the fixed multiplier from the start.
        semantic_schedule = []

    runner = CustomGARunner(
        layout=layout,
        evaluator=evaluator,
        surrogate_manager=surrogate_manager if surrogate_enabled else None,
        mutation=mutation,
        sanitizer=sanitizer,
        analyze_duplicates_fn=analyze_duplicates,
        pop_size=pop_size,
        crossover_prob=crossover_prob,
        n_shortcuts=layout.n_shortcuts,
        checkpoint_every=config.get("output.checkpoint_interval", 500),
        build_dir=args.output_dir,
        perf=perf,
        hard_constraints=hard_constraints,
        n_constraints=n_constraints,
        semantic_multiplier_schedule=semantic_schedule,
        cluster_score_tolerance=float(config.get("evolution.cluster_score_tolerance", 0.15)),
        unsupported_duplicate_sid_mask=unsupported_duplicate_mask,
    )
    ga_result = runner.run(n_gen, initial_pop_X=initial_pop_X)

    t1 = time.time()
    print(f"\nEvolution completed in {t1-t0:.1f}s", flush=True)

    # Show best results
    if ga_result["global_best_genome"] is not None and ga_result["global_best_exact"] is not None:
        best_genome = ga_result["global_best_genome"].copy()
        best_f = np.asarray(ga_result["global_best_exact"]["objectives"], dtype=np.float32)
        print(
            f"Best layout from global exact archive: gen={ga_result['global_best_generation']}, "
            f"objectives={best_f.tolist()}",
            flush=True,
        )
    else:
        raise RuntimeError("No fully valid seed or candidate entered the exact archive")
    print(f"Best layout (exact): objectives={best_f.tolist()}", flush=True)

    # Exact evaluation of best
    best_layout = layout.clone_with(genome=best_genome.astype(np.int32))
    exact_result = evaluator.evaluate(best_layout)
    print(f"Exact eval (normalized): effort={exact_result.effort:.2f}, adj={exact_result.adjacency:.2f}, viol={exact_result.violations:.2f}", flush=True)
    # Also show raw (unscaled) values for comparison
    raw_effort = exact_result.effort * scale_factors[0]
    raw_adj = exact_result.adjacency * scale_factors[1]
    raw_viol = exact_result.violations * scale_factors[2]
    print(f"Exact eval (raw):        effort={raw_effort:.0f}, adj={raw_adj:.0f}, viol={raw_viol:.0f}", flush=True)

    # Save results
    duplicate_report = analyze_duplicates(best_layout)
    completion_report = analyze_completion_cluster(best_layout)
    arrow_report = analyze_arrows(best_layout)
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config.raw,
        "scale_factors": [float(s) for s in scale_factors],
        "seed_fitness": {
            "effort": float(seed_result.objectives[0]),
            "adjacency": float(seed_result.objectives[1]),
            "violations": float(seed_result.objectives[2]),
        },
        "best_objective": {
            "source": "exact",
            "objectives": [float(x) for x in best_f],
            "total": float(best_f.sum()),
        },
        "best_exact": {
            "effort": float(exact_result.effort),
            "adjacency": float(exact_result.adjacency),
            "violations": float(exact_result.violations),
            "raw_effort": float(raw_effort),
            "raw_adjacency": float(raw_adj),
            "raw_violations": float(raw_viol),
            "factor_scores": {k: float(v) for k, v in exact_result.factor_scores.items()},
            "genome": [int(x) for x in best_genome],
            "duplicate_report": duplicate_report,
            "completion_cluster_report": completion_report,
            "arrow_report": arrow_report,
            "acceptance_report": build_acceptance_report(
                best_layout,
                duplicate_report=duplicate_report,
                completion_cluster_report=completion_report,
                arrow_report=arrow_report,
            ),
        },
        "exact_eval_history": runner.exact_history,
        "archive_acceptance_rejections": ga_result.get("archive_acceptance_rejections", {}),
        "archive_cluster_prefilter_rejections": ga_result.get("archive_cluster_prefilter_rejections", 0),
        "archive_cluster_prefilter_reasons": ga_result.get("archive_cluster_prefilter_reasons", {}),
        "pareto_front": [],
        "generation": ga_result["gens_run"],
        "population_size": pop_size,
        "elapsed_seconds": t1 - t0,
        "perf": perf.summary(),
        "compiled_exact_enabled": True,
        "compiled_exact_parity": getattr(evaluator.model, "parity", None),
        "training_path": "cuda_surrogate_primary" if surrogate_enabled else "cpu_exact",
    }

    results_path = os.path.join(args.output_dir, "v2_evolution_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}", flush=True)
    perf_path = os.path.join(args.output_dir, "v2_perf_report.json")
    with open(perf_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numba_available": NUMBA_AVAILABLE,
            "training_path": "cuda_surrogate_primary" if surrogate_enabled else "cpu_exact",
            "surrogate_enabled": surrogate_enabled,
            "surrogate_device": getattr(getattr(surrogate_manager, "trainer", None), "device", None) if surrogate_manager else None,
            "profile_fast": args.profile_fast or config.get("profiling.enabled", False),
            "archive_acceptance_rejections": ga_result.get("archive_acceptance_rejections", {}),
            "archive_cluster_prefilter_rejections": ga_result.get("archive_cluster_prefilter_rejections", 0),
            "archive_cluster_prefilter_reasons": ga_result.get("archive_cluster_prefilter_reasons", {}),
            "events": perf.summary(),
        }, f, indent=2, default=str)
    print(f"Performance report saved to {perf_path}", flush=True)

    print("\nv2 evolution complete!", flush=True)


if __name__ == "__main__":
    args = _parse_args()
    if args.profile_fast:
        profile = cProfile.Profile()
        try:
            profile.enable()
            main()
        finally:
            profile.disable()
            os.makedirs(args.output_dir, exist_ok=True)
            stats_path = os.path.join(args.output_dir, "v2_profile_fast.pstats")
            txt_path = os.path.join(args.output_dir, "v2_profile_fast.txt")
            profile.dump_stats(stats_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                stats = pstats.Stats(profile, stream=f).sort_stats("cumtime")
                stats.print_stats(80)
            print(f"Profile saved to {stats_path} and {txt_path}", flush=True)
    else:
        main()
