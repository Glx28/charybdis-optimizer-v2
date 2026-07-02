"""Custom single-objective GA replacing pymoo NSGA2.

Eliminates pymoo's Population objects, Pareto sort, crowding distance,
and Python tournament loop. Uses GPU tournament selection (basic PyTorch
tensor ops, CUDA 6.1 compatible) and keeps all existing operators
(SwapMutation, StructuralGenomeSanitizer) unchanged.

Expected speedup vs pymoo NSGA2: ~2x (from ~74ms/gen to ~38ms/gen).
Phase-2 vectorized swap brings this to ~3x (~23ms/gen).
"""

import concurrent.futures
import glob
import json
import os
import random
import time

import numpy as np
import torch

from evolution.acceptance import build_acceptance_report
from evolution.arrow_cluster import analyze_arrows
from evolution.completion_cluster import analyze_completion_cluster
from evolution import NUMBA_AVAILABLE

if NUMBA_AVAILABLE:
    from evolution import _cycle_crossover_pair_numba, _cycle_crossover_batch_numba


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _scalar(F, cv=None):
    """Feasibility-first scalar fitness. Lower is better."""
    s = F.sum(axis=1) if F.ndim == 2 else np.asarray(F, dtype=np.float32)
    if cv is not None:
        penalty = np.maximum(cv, 0)
        if penalty.ndim > 1:
            penalty = penalty.sum(axis=1)
        if penalty.shape[0] == s.shape[0]:
            s = s + 1e9 * penalty
    return s


def _tournament_select(scalar_F, n, k=2):
    """Return (n,) parent indices via binary tournament — vectorized CPU numpy."""
    idx = np.random.randint(0, len(scalar_F), (n, k))
    return idx[np.arange(n), scalar_F[idx].argmin(axis=1)]


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

        # Background thread pool for concurrent mini exact eval during GPU predict
        self._eval_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # State
        self.global_best_genome = None
        self.global_best_exact = None
        self.global_best_generation = None
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

    # ------------------------------------------------------------------
    # Helpers (ported from ExactEvalCallback)
    # ------------------------------------------------------------------

    def _exact_entry(self, result, gen):
        return {
            "generation": int(gen),
            "objectives": [float(x) for x in result.objectives],
            "constraints": [float(x) for x in result.constraints],
            "factor_scores": {k: float(v) for k, v in result.factor_scores.items()},
            "total_score": float(result.total_score),
        }

    def _is_better(self, candidate, incumbent):
        if incumbent is None:
            return True
        cand_cv = sum(max(0.0, float(x)) for x in candidate.get("constraints", []))
        inc_cv = sum(max(0.0, float(x)) for x in incumbent.get("constraints", []))
        if cand_cv != inc_cv:
            return cand_cv < inc_cv
        cand_pass = bool(candidate.get("optimizer_side_pass", False))
        inc_pass = bool(incumbent.get("optimizer_side_pass", False))
        if cand_pass != inc_pass:
            return cand_pass
        if not cand_pass:
            cf = len(candidate.get("acceptance_failed_checks", []))
            inf = len(incumbent.get("acceptance_failed_checks", []))
            if cf != inf:
                return cf < inf
        return float(candidate["total_score"]) < float(incumbent["total_score"])

    def _update_archive(self, genome, entry):
        if self._is_better(entry, self.global_best_exact):
            self.global_best_genome = genome.astype(np.int32).copy()
            self.global_best_exact = dict(entry)
            self.global_best_generation = int(entry["generation"])
            return True
        return False

    def _layout_reports(self, layout_obj):
        dup = self.analyze_duplicates(layout_obj)
        comp = analyze_completion_cluster(layout_obj)
        arr = analyze_arrows(layout_obj)
        acc = build_acceptance_report(
            layout_obj,
            duplicate_report=dup,
            completion_cluster_report=comp,
            arrow_report=arr,
        )
        return dup, comp, arr, acc

    def _annotate(self, entry, acceptance):
        failed = [
            k for k, ok in acceptance.get("checks", {}).items()
            if k != "norwegian_export_bad_literal_count_zero" and not ok
        ]
        entry["optimizer_side_pass"] = bool(acceptance.get("optimizer_side_pass", False))
        entry["acceptance_failed_checks"] = failed
        return entry

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

    def _inject_diversity(self, pop_X, pop_F, pop_cv, gen):
        # Trigger on surrogate stagnation OR archive stagnation (so a slightly
        # optimistic surrogate that keeps stagnation_count low doesn't prevent injection)
        archive_stagnant = self.archive_stagnation >= 6  # 3000 gens at checkpoint_every=500
        if self.stagnation_count < 1200 and not archive_stagnant:
            return pop_X, pop_F, pop_cv
        if gen - self.last_diversity_reset < 600:
            return pop_X, pop_F, pop_cv

        n = len(pop_X)
        scalar = _scalar(pop_F, pop_cv)
        elite_count = max(2, n // 20)
        elite_idx = set(np.argsort(scalar)[:elite_count].tolist())
        replace_order = [i for i in np.argsort(scalar)[::-1] if i not in elite_idx]
        replace_idx = replace_order[: max(4, n // 4)]

        seed = self.layout.genome.astype(np.int32)
        frozen_assigned = {
            int(seed[i]) for i in self.layout.frozen_indices if int(seed[i]) >= 0
        }
        available = np.array(
            [s for s in range(self.layout.n_shortcuts) if s not in frozen_assigned],
            dtype=np.int32,
        )
        mutable = self.layout.mutable_indices

        new_genomes = []
        for dst in replace_idx:
            g = seed.copy()
            g[mutable] = -1
            n_assign = min(len(mutable), len(available))
            g[np.random.permutation(mutable)[:n_assign]] = np.random.permutation(available)[:n_assign]
            pop_X[dst] = g
            new_genomes.append(g)

        if new_genomes:
            batch = np.array(new_genomes, dtype=np.int32)
            t0 = time.perf_counter()
            new_F, new_G = self.evaluator.evaluate_batch(batch)
            if self.perf:
                self.perf.add("exact_eval", time.perf_counter() - t0)
            for row, dst in enumerate(replace_idx):
                pop_F[dst] = new_F[row]
                if pop_cv is not None and new_G.shape[1] > 0:
                    pop_cv[dst] = np.maximum(new_G[row], 0)

        self.last_diversity_reset = gen
        print(
            f"    Gen {gen}: stagnation for {self.stagnation_count} gens."
            f" Injected {len(replace_idx)} diverse individuals, kept {elite_count} elites.",
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

        improved = self._update_archive(best_genome, entry)
        if improved:
            self.archive_stagnation = 0
            print(
                f"    Gen {gen}: global best improved to {entry['total_score']:.4f}"
                f" (optimizer_side_pass={entry['optimizer_side_pass']})",
                flush=True,
            )
        else:
            self.archive_stagnation += 1
            stagnant_gens = self.archive_stagnation * self.checkpoint_every
            if stagnant_gens >= 20000 and self.global_best_exact is not None:
                print(
                    f"    Gen {gen}: early stop — archive stagnant for {stagnant_gens} gens"
                    f" (best score={self.global_best_exact['total_score']:.4f})",
                    flush=True,
                )
                self._should_stop = True

        archive_genome = (
            self.global_best_genome if self.global_best_genome is not None else best_genome
        )
        archive_layout = self.layout.clone_with(genome=archive_genome)
        archive_entry = dict(self.global_best_exact or entry)
        self.best_exact = archive_entry
        self.exact_history.append(entry)
        if len(self.exact_history) > 20:
            self.exact_history = self.exact_history[-20:]
        arc_dup, arc_comp, arc_arr, arc_acc = self._layout_reports(archive_layout)

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

    def _maybe_teacher_update(self, pop_X, pop_F, gen):
        sm = self.surrogate_manager
        if sm is None:
            return
        sm.generation = gen
        exact_eval_every = sm.exact_eval_every
        if exact_eval_every > 0 and gen % exact_eval_every == 0:
            t0 = time.perf_counter()
            exact_F, _ = self.evaluator.evaluate_batch(pop_X.astype(np.int32))
            if self.perf:
                self.perf.add("surrogate_teacher_eval", time.perf_counter() - t0)
            sm.add_exact_evaluations(pop_X.astype(np.int32), exact_F)
        if sm.should_retrain():
            t0 = time.perf_counter()
            sm.retrain()
            if self.perf:
                self.perf.add("surrogate_retrain", time.perf_counter() - t0)

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

        # Initial surrogate evaluation
        sm = self.surrogate_manager
        if sm is not None and sm.trainer.mean is not None:
            pop_F = sm.trainer.predict(pop_X)
        else:
            t0 = time.perf_counter()
            pop_F, _ = self.evaluator.evaluate_batch(pop_X)
            if self.perf:
                self.perf.add("exact_eval", time.perf_counter() - t0)
        pop_cv = None  # constraint violations only tracked at checkpoints

        gen_times = []

        for gen in range(1, n_gen + 1):
            t_gen = time.perf_counter()

            # --- Tournament selection (GPU) ---
            scalar = _scalar(pop_F)
            parent_idx = _tournament_select(scalar, self.pop_size)

            # --- Crossover ---
            children_X = _crossover_batch(
                pop_X, parent_idx, self.crossover_prob, self.n_shortcuts
            )

            # --- Mutation (call directly, problem=None is safe) ---
            self.mutation._do(None, children_X)

            # --- Repair ---
            self.sanitizer._do(None, children_X)

            # --- Hybrid exact/surrogate evaluation ---
            if sm is not None and sm.trainer.mean is not None:
                # Submit 150-genome exact eval to background before GPU predict.
                # Numba JIT releases the GIL, so it runs concurrently with CUDA forward.
                # Exact scores for 10% of children act as selection "beacons" that guide
                # the search toward hard-constraint-satisfying regions.
                mini_idx = np.random.choice(len(children_X), 150, replace=False)
                mini_batch = children_X[mini_idx].copy()
                mini_future = self._eval_executor.submit(
                    self.evaluator.evaluate_batch, mini_batch
                )
                children_F = sm.trainer.predict(children_X)
                # Collect exact results and splice back — overrides surrogate predictions
                # for these 150 children with ground-truth fitness.
                mini_F, _ = mini_future.result()
                children_F[mini_idx] = mini_F
                sm.add_exact_evaluations(mini_batch, mini_F)
            else:
                t0 = time.perf_counter()
                children_F, _ = self.evaluator.evaluate_batch(children_X)
                if self.perf:
                    self.perf.add("exact_eval", time.perf_counter() - t0)

            # --- (µ+λ) survival: keep best pop_size from parents + children ---
            all_X = np.concatenate([pop_X, children_X], axis=0)
            all_F = np.concatenate([pop_F, children_F], axis=0)
            all_scalar = _scalar(all_F)
            survivors = np.argpartition(all_scalar, self.pop_size)[: self.pop_size]
            pop_X = all_X[survivors].astype(np.int32)
            pop_F = all_F[survivors]

            # --- Adaptive mutation rate ---
            best_quality = float(pop_F.sum(axis=1).min())
            self._adjust_mutation(best_quality, gen)

            # --- Periodic: teacher update, retrain, checkpoint ---
            self._maybe_teacher_update(pop_X, pop_F, gen)

            if gen % self.checkpoint_every == 0:
                t0 = time.perf_counter()
                self._checkpoint(pop_X, pop_F, pop_cv, gen)
                if self.perf:
                    self.perf.add("checkpoint", time.perf_counter() - t0)
                # Diversity injection (uses stagnation_count updated above)
                pop_X, pop_F, pop_cv = self._inject_diversity(pop_X, pop_F, pop_cv, gen)
                if self._should_stop:
                    break

            gen_times.append(time.perf_counter() - t_gen)

        total_time = time.perf_counter() - t_start
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
        }
