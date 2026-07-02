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
from config import Config

from pymoo.optimize import minimize
from pymoo.core.callback import Callback


def generate_random_layouts(layout, n):
    from evolution import build_group_placements

    mutable = layout.mutable_indices
    mutable_set = set(mutable.tolist()) if hasattr(mutable, 'tolist') else set(mutable)
    n_shortcuts = layout.n_shortcuts
    layouts = np.full((n, layout.n_positions), -1, dtype=np.int32)
    if n <= 0:
        return layouts

    layouts[0] = layout.genome.astype(np.int32).copy()
    frozen = layout.frozen_indices
    frozen_assigned = {int(sid) for sid in layout.genome[frozen] if sid >= 0}

    # Exclude group sids from individual random assignment — they are placed
    # as complete groups at valid anchors so initial genomes are never scattered.
    groups = build_group_placements(layout)
    group_sids_all = {sid for sid_tuple, _ in groups for sid in sid_tuple}

    available_sids = [
        sid for sid in range(n_shortcuts)
        if sid not in frozen_assigned and sid not in group_sids_all
    ]

    for i in range(1, n):
        layouts[i, frozen] = layout.genome[frozen]
        n_assign = min(len(mutable), len(available_sids))
        assigned = random.sample(available_sids, n_assign)
        layouts[i, mutable[:n_assign]] = assigned

        # Place each group at a random valid anchor.
        for sid_tuple, anchor_list in groups:
            anchor = list(random.choice(anchor_list))
            anchor_set = set(anchor)
            # Read what's at the anchor positions now (to displace them).
            displaced = [int(layouts[i, pos]) for pos in anchor]
            # Place group sids at anchor.
            for sid, pos in zip(sid_tuple, anchor):
                layouts[i, pos] = sid
            # Displaced non-group sids go to the first free mutable slots.
            fill_sids = [s for s in displaced if s >= 0 and s not in group_sids_all]
            free_slots = [
                pos for pos in mutable
                if pos not in anchor_set and int(layouts[i, pos]) == -1
            ]
            for pos, sid in zip(free_slots, fill_sids):
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
    return FitnessEvaluator(
        weights=config.get("fitness.weights", {}),
        reference_layout=layout,
        scale_factors=scale_factors,
        violation_weights=config.get("fitness.violation_sub_weights", {}),
        missing_important_threshold=config.get("fitness.missing_important_threshold", 6.0),
        hard_constraints=config.get("fitness.hard_constraints", []),
        toggle_effort_multiplier=float(config.get("fitness.toggle_effort_multiplier", 2.5)),
    )


def enforce_training_device_policy(config):
    """Fail fast when the configured training path would run CPU-primary.

    This project is GPU-focused.  CPU-only exact evaluation may be useful for
    unit tests or one-off diagnostics, but it must not silently become the
    training path for an evolution run.
    """
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
            "The current run_evolution.py exact-evaluation path is CPU/Numba-primary. "
            "Enable or implement a CUDA-backed training path before starting a training run."
        )
    if require_gpu_primary and not allow_cpu_exact_validation:
        raise SystemExit(
            "GPU training policy violation: allow_cpu_exact_validation=false, but this runner still uses "
            "CPU exact evaluation as teacher labels/checkpoint validation. Implement a CUDA exact kernel or "
            "explicitly allow CPU validation while keeping training GPU-primary."
        )


def analyze_duplicates(layout):
    """Explain which duplicate placements have usage support."""
    support = _shortcut_duplicate_support(layout)
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
        seq_hits = [
            key for key, data in layout.usage_data.sequences.items()
            if shortcut.keys in key.split(" -> ") and isinstance(data, dict) and data.get("count", 0) >= 2
        ]
        workflow_hits = [
            key for key, data in list(layout.usage_data.chains.items()) + list(layout.usage_data.workflows.items())
            if shortcut.keys in key.split(" -> ") and isinstance(data, dict) and data.get("count", 0) >= 2
        ]
        app_hits = [
            key for key, data in layout.usage_data.app_workflows.items()
            if (
                isinstance(data, dict)
                and data.get("count", 0) >= 2
                and any(_app_matches(shortcut.app, part.strip()) for part in key.split(" + "))
            )
        ]
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
    ):
        super().__init__()
        self.layout = layout
        self.evaluator = evaluator
        self.build_dir = build_dir
        self.checkpoint_every = checkpoint_every
        self.perf = perf
        self.surrogate_manager = surrogate_manager
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
        n = pop_F.shape[0]
        if pop_G is not None and pop_G.shape[1] > 0:
            cv = np.maximum(pop_G, 0).sum(axis=1)
        else:
            cv = np.zeros(n, dtype=np.float32)
        scalar = pop_F.sum(axis=1) + 1e9 * cv
        return int(np.argmin(scalar))

    def _exact_entry(self, exact_result, gen):
        return {
            "generation": int(gen),
            "objectives": [float(x) for x in exact_result.objectives],
            "constraints": [float(x) for x in exact_result.constraints],
            "factor_scores": {k: float(v) for k, v in exact_result.factor_scores.items()},
            "total_score": float(exact_result.total_score),
        }

    def _is_better_exact(self, candidate_entry, incumbent_entry):
        if incumbent_entry is None:
            return True
        cand_cv = sum(max(0.0, float(x)) for x in candidate_entry.get("constraints", []))
        inc_cv = sum(max(0.0, float(x)) for x in incumbent_entry.get("constraints", []))
        if cand_cv != inc_cv:
            return cand_cv < inc_cv
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

        # Overall layout quality improvement: any objective improving counts.
        # Sum all objectives so effort/adjacency/workflow gains prevent false stagnation.
        best_quality = float(np.min(pop.sum(axis=1)))

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

        cv = np.maximum(pop_G, 0).sum(axis=1) if pop_G is not None and pop_G.shape[1] > 0 else np.zeros(n_pop)
        scalar = pop_F.sum(axis=1) + 1e9 * cv
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

        best_genome = self.global_best_genome if self.global_best_genome is not None else population_best_genome
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

    surrogate_enabled = bool(config.get("surrogate.enabled", False))
    evaluator = build_evaluator(config, layout)
    maybe_validate_exact_evaluator(config, layout, evaluator)

    # Evaluate pre-seeded genome
    seed_result = evaluator.evaluate(layout)
    print(f"  Seed fitness (raw): effort={seed_result.objectives[0]*1:.0f}, adj={seed_result.objectives[1]*1:.0f}, viol={seed_result.objectives[2]*1:.0f}", flush=True)

    # Compute robust scale factors from a small random sample (IQR-based, not seed-relative)
    n_sample = 50
    sample_layouts = generate_random_layouts(layout, n_sample)
    sample_scores, _ = evaluate_exact_batch(
        sample_layouts, layout, evaluator, perf=perf, label="scale_sample_eval",
    )
    q25 = np.percentile(sample_scores, 25, axis=0)
    q75 = np.percentile(sample_scores, 75, axis=0)
    iqr = q75 - q25
    seed_scores = np.abs(seed_result.objectives)
    scale_factors = np.maximum(iqr, seed_scores * 0.1)
    scale_factors = np.maximum(scale_factors, 1.0)
    print(f"  IQR scale factors: effort={scale_factors[0]:.2f}, adj={scale_factors[1]:.2f}, viol={scale_factors[2]:.2f}", flush=True)
    evaluator = build_evaluator(config, layout, scale_factors=scale_factors)
    maybe_validate_exact_evaluator(config, layout, evaluator)
    seed_result = evaluator.evaluate(layout)
    print(f"  Seed fitness (normalized): effort={seed_result.objectives[0]:.2f}, adj={seed_result.objectives[1]:.2f}, viol={seed_result.objectives[2]:.2f}", flush=True)

    surrogate_manager = None

    if surrogate_enabled:
        print("  GPU-primary surrogate training enabled.", flush=True)
        surrogate = LayoutSurrogate(
            layout.n_positions,
            layout.n_shortcuts,
            n_factors=3,
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
        initial_layouts = generate_random_layouts(layout, n_initial)
        initial_scores, _ = evaluate_exact_batch(
            initial_layouts, layout, evaluator, perf=perf, label="surrogate_initial_teacher_eval",
        )
        surrogate_manager.add_exact_evaluations(initial_layouts, initial_scores)
        trainer.train(
            initial_layouts,
            initial_scores,
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
    from evolution.custom_ga import CustomGARunner

    mutation = SwapMutation(
        prob=config.get("evolution.mutation_prob", 0.15),
        frozen_mask=layout.frozen_mask,
        layout=layout,
    )
    sanitizer = StructuralGenomeSanitizer(
        n_shortcuts=layout.n_shortcuts,
        frozen_mask=layout.frozen_mask,
        seed_genome=layout.genome,
        layout=layout,
    )
    crossover_prob = config.get("evolution.crossover_prob", 0.7)

    print("  All individuals start from fully random shortcut assignment (L0 thumb gets one random momentary hold).", flush=True)
    print(f"\nRunning evolution (CustomGA): pop={pop_size}, gens={n_gen}", flush=True)
    t0 = time.time()

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
    )
    ga_result = runner.run(n_gen)

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
        print("Warning: no global best found; falling back to seed genome.", flush=True)
        best_genome = layout.genome.copy()
        best_f = seed_result.objectives.copy()
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
