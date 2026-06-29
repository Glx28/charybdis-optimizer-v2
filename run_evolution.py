"""Run the full pymoo evolution loop with surrogate."""
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

sys.path.insert(0, os.path.dirname(__file__))

from core.loader import build_layout
from fitness.evaluator import FitnessEvaluator
from fitness.batch_evaluator import BatchExactEvaluator, NUMBA_AVAILABLE
from evolution.surrogate import LayoutSurrogate, SurrogateTrainer, SurrogateManager
from evolution import create_algorithm
from config import Config

from pymoo.optimize import minimize
from pymoo.core.callback import Callback


def generate_random_layouts(layout, n):
    mutable = layout.mutable_indices
    n_shortcuts = layout.n_shortcuts
    layouts = np.full((n, layout.n_positions), -1, dtype=np.int32)
    if n <= 0:
        return layouts

    layouts[0] = layout.genome.astype(np.int32).copy()
    frozen = layout.frozen_indices
    frozen_assigned = {int(sid) for sid in layout.genome[frozen] if sid >= 0}
    available_sids = [sid for sid in range(n_shortcuts) if sid not in frozen_assigned]

    for i in range(1, n):
        layouts[i, frozen] = layout.genome[frozen]
        n_assign = min(len(mutable), n_shortcuts, len(available_sids))
        assigned = random.sample(available_sids, n_assign)
        layouts[i, mutable[:n_assign]] = assigned
    return layouts


def _setup_mp_env():
    """Set PYTHONPATH so child processes can find the venv and project."""
    import os, sys
    paths = []
    project_root = os.path.dirname(os.path.abspath(__file__))
    paths.append(project_root)
    exe_dir = os.path.dirname(sys.executable)
    for candidate in [
        os.path.join(exe_dir, 'Lib', 'site-packages'),
        os.path.join(os.path.dirname(exe_dir), 'Lib', 'site-packages'),
        os.path.join(project_root, '.venv', 'Lib', 'site-packages'),
    ]:
        if os.path.exists(candidate) and candidate not in paths:
            paths.append(candidate)
    current = os.environ.get('PYTHONPATH', '')
    new_paths = os.pathsep.join(paths)
    if current:
        os.environ['PYTHONPATH'] = new_paths + os.pathsep + current
    else:
        os.environ['PYTHONPATH'] = new_paths


def _eval_one(args):
    """Module-level worker for parallel exact eval."""
    (idx, genome, positions, shortcuts, frozen_mask, layer_to_indices, usage_data,
     layer_access, dynamic_groups, evaluator_weights, evaluator_scale, reference_genome) = args
    from core import Layout, UsageData
    from fitness.evaluator import FitnessEvaluator
    import numpy as np
    
    l = Layout(
        genome=genome,
        positions=positions,
        shortcuts=shortcuts,
        frozen_mask=frozen_mask,
        layer_to_indices=layer_to_indices,
        usage_data=usage_data if usage_data is not None else UsageData(),
        layer_access=layer_access,
        dynamic_groups=dynamic_groups if dynamic_groups is not None else tuple(),
    )
    ref = Layout(
        genome=reference_genome,
        positions=positions,
        shortcuts=shortcuts,
        frozen_mask=frozen_mask,
        layer_to_indices=layer_to_indices,
        usage_data=usage_data if usage_data is not None else UsageData(),
        layer_access=layer_access,
        dynamic_groups=dynamic_groups if dynamic_groups is not None else tuple(),
    ) if reference_genome is not None else None
    ev = FitnessEvaluator(weights=evaluator_weights, reference_layout=ref, scale_factors=evaluator_scale)
    result = ev.evaluate(l)
    return idx, result.objectives


def evaluate_exact_batch(layouts, layout, evaluator, n_workers=None, batch_evaluator=None, perf=None, label="exact_eval"):
    """Evaluate exact fitness for a batch of layouts, using all CPU cores."""
    t0 = time.perf_counter()
    if batch_evaluator is not None and getattr(batch_evaluator, "enabled", False):
        try:
            scores = batch_evaluator.evaluate(layouts)
            if perf is not None:
                perf.add(label, time.perf_counter() - t0)
            return scores
        except Exception as exc:
            print(f"  Compiled exact evaluator failed, falling back to CPU: {exc}", flush=True)
            batch_evaluator.enabled = False

    import multiprocessing as mp
    n = layouts.shape[0]
    scores = np.zeros((n, 3), dtype=np.float32)
    
    # For small batches, just use sequential evaluation
    if n <= 20 or n_workers == 1:
        for i in range(n):
            result = evaluator.evaluate(layout.clone_with(genome=layouts[i]))
            scores[i] = result.objectives
        if perf is not None:
            perf.add(label, time.perf_counter() - t0)
        return scores
    
    # Parallel evaluation for larger batches
    n_workers = n_workers or max(1, mp.cpu_count() - 1)
    
    # Fix environment for child processes (Windows spawn doesn't inherit venv)
    _setup_mp_env()
    
    # Prepare args - pickleable objects
    args_list = []
    for i in range(n):
        args_list.append((
            i,
            layouts[i].copy(),
            layout.positions,
            layout.shortcuts,
            layout.frozen_mask.copy(),
            layout.layer_to_indices,
            layout.usage_data,
            layout.layer_access,
            layout.dynamic_groups,
            evaluator.weights.copy() if hasattr(evaluator.weights, 'copy') else evaluator.weights,
            evaluator.scale_factors.copy() if evaluator.scale_factors is not None else None,
            evaluator.reference_layout.genome.copy() if evaluator.reference_layout is not None else None,
        ))
    
    try:
        with mp.Pool(n_workers) as pool:
            for idx, obj in pool.imap_unordered(_eval_one, args_list):
                scores[idx] = obj
    except Exception as e:
        print(f"  Warning: parallel eval failed ({e}), falling back to sequential", flush=True)
        for i in range(n):
            result = evaluator.evaluate(layout.clone_with(genome=layouts[i]))
            scores[i] = result.objectives
    
    if perf is not None:
        perf.add(label, time.perf_counter() - t0)
    return scores


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


def make_batch_evaluator(layout, evaluator, enabled=True, tolerance=1e-4):
    if not enabled:
        return None
    if not NUMBA_AVAILABLE:
        print("  Numba not available; exact eval uses CPU fallback.", flush=True)
        return None
    try:
        batch = BatchExactEvaluator(layout, evaluator, validate=False)
        parity = batch.validate_parity(n=32, tolerance=tolerance)
        batch.parity = parity
        batch.enabled = parity.ok
        if parity.ok:
            print(f"  Numba exact evaluator enabled (parity max diff={parity.max_abs_diff:.3g}).", flush=True)
            return batch
        print(f"  Numba exact evaluator disabled: {parity.message}", flush=True)
    except Exception as exc:
        print(f"  Numba exact evaluator unavailable: {exc}", flush=True)
    return None


class SurrogateCallback(Callback):
    """Callback that manages surrogate retraining, exact evaluation, and checkpointing."""
    
    def __init__(
        self,
        layout,
        evaluator,
        manager,
        build_dir,
        exact_eval_batch=20,
        checkpoint_every=100,
        batch_evaluator=None,
        perf=None,
    ):
        super().__init__()
        self.layout = layout
        self.evaluator = evaluator
        self.manager = manager
        self.build_dir = build_dir
        self.exact_eval_batch = exact_eval_batch
        self.checkpoint_every = checkpoint_every
        self.batch_evaluator = batch_evaluator
        self.perf = perf
        self.exact_history = []
        self.evolved_accuracy_history = []
        self.best_exact = None
        self.stagnation_count = 0
        self.last_best_quality = float('inf')
        self.base_mutation_prob = None
    
    def _get_mutation(self, algorithm):
        """Safely access the mutation operator from the algorithm."""
        mating = getattr(algorithm, 'mating', None)
        if mating is not None:
            return getattr(mating, 'mutation', None)
        return None
    
    def _adjust_mutation_rate(self, algorithm, gen):
        """Adaptive mutation: increase when stagnated, restore when improving."""
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
        
        # Use violations (last column) as quality metric
        best_quality = float(np.min(pop[:, -1]))
        
        if best_quality < self.last_best_quality * 0.999:
            # Improvement detected; reset stagnation and restore base rate if needed
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
    
    def _repair_best_groups(self, algorithm, gen):
        """Post-process the best individual to fix group splits. Very fast: only 1 individual per gen."""
        if gen % 10 != 0:  # Run every 10 generations
            return
        
        from fitness.factors.violation import KEY_GROUPS, shortcut_matches_group
        
        pop_X = algorithm.pop.get("X")
        pop_F = algorithm.pop.get("F")
        if pop_X is None or pop_F is None or len(pop_X) == 0:
            return
        
        # Find best individual (min total objective)
        best_idx = int(np.argmin(pop_F.ravel()))
        genome = pop_X[best_idx].copy()
        n_var = len(genome)
        
        # Build sid -> position mapping
        sid_to_pos = {}
        for pos, sid in enumerate(genome):
            if sid >= 0:
                sid_to_pos[sid] = pos
        
        # Build layer -> mutable positions mapping
        pos_layer = np.array([p.layer for p in self.layout.positions], dtype=np.int32)
        mutable = self.layout.mutable_indices
        layer_to_empty = {}
        for layer in set(pos_layer):
            layer_pos = np.where((pos_layer == layer) & (~self.layout.frozen_mask))[0]
            empty = [p for p in layer_pos if genome[p] < 0]
            layer_to_empty[layer] = empty
        
        all_groups = list(KEY_GROUPS) + list(self.layout.dynamic_groups)
        repaired = False
        
        for group in all_groups:
            if not group.get("protected"):
                continue
            
            # Find group members
            sids = set()
            if "sids" in group:
                for sid in group.get("sids", []):
                    if 0 <= int(sid) < self.layout.n_shortcuts:
                        sids.add(int(sid))
            else:
                for shortcut in self.layout.shortcuts:
                    if shortcut_matches_group(shortcut, group):
                        sids.add(shortcut.sid)
            
            if len(sids) <= 1:
                continue
            
            # Find members and their layers
            member_info = []  # (pos, sid, layer)
            for sid in sids:
                if sid in sid_to_pos:
                    pos = sid_to_pos[sid]
                    member_info.append((pos, sid, pos_layer[pos]))
            
            if len(member_info) <= 1:
                continue
            
            # Count per layer
            layer_counts = {}
            for pos, sid, layer in member_info:
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            
            if len(layer_counts) <= 1:
                continue
            
            dominant_layer = max(layer_counts, key=layer_counts.get)
            
            # Move non-dominant members to empty positions on dominant layer
            for pos, sid, layer in member_info:
                if layer == dominant_layer:
                    continue
                if not layer_to_empty.get(dominant_layer):
                    continue
                target_pos = layer_to_empty[dominant_layer].pop(0)
                genome[target_pos] = sid
                genome[pos] = -1
                # Update sid_to_pos
                sid_to_pos[sid] = target_pos
                # Add old position back to empty list
                layer_to_empty[layer].append(pos)
                repaired = True
        
        if repaired:
            # Re-evaluate and update if improved
            best_layout = self.layout.clone_with(genome=genome.astype(np.int32))
            try:
                exact_result = self.evaluator.evaluate(best_layout)
                new_viol = exact_result.violations
                old_viol = pop_F[best_idx, -1]
                if new_viol < old_viol:
                    pop_X[best_idx] = genome
                    pop_F[best_idx] = exact_result.objectives
                    print(f"    Gen {gen}: group repair improved best individual viol={old_viol:.2f} -> {new_viol:.2f}", flush=True)
            except Exception:
                pass
    
    def notify(self, algorithm):
        self.manager.step()
        gen = self.manager.generation
        
        # Adaptive mutation rate
        self._adjust_mutation_rate(algorithm, gen)
        
        # Post-process best individual for group repair
        self._repair_best_groups(algorithm, gen)
        
        # Periodic exact evaluation
        if self.manager.should_exact_eval():
            pop = algorithm.pop.get("X")
            n_eval = min(self.exact_eval_batch, len(pop))
            indices = np.random.choice(len(pop), n_eval, replace=False)
            exact_layouts = pop[indices]
            surrogate_total = self.manager.trainer.predict(exact_layouts)
            exact_scores = evaluate_exact_batch(
                exact_layouts,
                self.layout,
                self.evaluator,
                batch_evaluator=self.batch_evaluator,
                perf=self.perf,
                label="callback_exact_eval",
            )
            exact_total = exact_scores.sum(axis=1, keepdims=True)
            drift = self._surrogate_drift_metrics(surrogate_total, exact_total)
            self.evolved_accuracy_history.append((gen, drift))
            self.manager.add_exact_evaluations(exact_layouts, exact_total)
            self.exact_history.append((gen, exact_scores.mean(axis=0)))
            if np.isfinite(drift["corr_mean"]) and drift["corr_mean"] < 0.70:
                self.manager.exact_eval_every = max(25, self.manager.exact_eval_every // 2)
                print(
                    f"    Warning: evolved surrogate corr={drift['corr_mean']:.3f}; "
                    f"increasing exact eval cadence to every {self.manager.exact_eval_every} generations.",
                    flush=True,
                )
            print(
                f"    Gen {gen}: exact eval {n_eval} individuals, avg obj={exact_scores.mean(axis=0)}, "
                f"evolved R^2={drift['r2_mean']:.3f}, corr={drift['corr_mean']:.3f}",
                flush=True,
            )
        
        # Periodic retraining
        if self.manager.should_retrain():
            self.manager.retrain()
        
        # Periodic checkpointing
        if gen > 0 and gen % self.checkpoint_every == 0:
            t0 = time.perf_counter()
            self._save_checkpoint(algorithm, gen)
            if self.perf is not None:
                self.perf.add("checkpoint", time.perf_counter() - t0)

    @staticmethod
    def _surrogate_drift_metrics(predicted, exact):
        residual = predicted - exact
        denom = np.sum((exact - exact.mean(axis=0)) ** 2, axis=0)
        r2 = np.full(exact.shape[1], np.nan, dtype=np.float32)
        valid = denom > 1e-6
        r2[valid] = 1 - np.sum(residual[:, valid] ** 2, axis=0) / denom[valid]

        corr = np.full(exact.shape[1], np.nan, dtype=np.float32)
        for j in range(exact.shape[1]):
            if np.std(predicted[:, j]) > 1e-6 and np.std(exact[:, j]) > 1e-6:
                corr[j] = np.corrcoef(predicted[:, j], exact[:, j])[0, 1]

        return {
            "r2": [None if np.isnan(x) else float(x) for x in r2],
            "r2_mean": float(np.nanmean(r2)) if np.any(~np.isnan(r2)) else float("nan"),
            "corr": [None if np.isnan(x) else float(x) for x in corr],
            "corr_mean": float(np.nanmean(corr)) if np.any(~np.isnan(corr)) else float("nan"),
            "mae": [float(x) for x in np.mean(np.abs(residual), axis=0)],
        }
    
    def _save_checkpoint(self, algorithm, gen):
        """Save intermediate results so progress is not lost if process is killed."""
        pop = algorithm.pop.get("X")
        F = algorithm.pop.get("F")
        best_idx = np.argmin(F.ravel()) if F is not None else 0
        best_genome = pop[best_idx] if pop is not None else None
        
        checkpoint = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generation": gen,
            "population_size": len(pop) if pop is not None else 0,
            "surrogate_r2_history": self.manager.accuracy_history,
            "exact_eval_history": [(int(g), [float(x) for x in s]) for g, s in self.exact_history],
            "evolved_surrogate_accuracy_history": [
                {"generation": int(g), **metrics} for g, metrics in self.evolved_accuracy_history
            ],
        }
        
        if best_genome is not None:
            best_layout = self.layout.clone_with(genome=best_genome.astype(np.int32))
            exact_result = self.evaluator.evaluate(best_layout)
            checkpoint["best_exact"] = {
                "effort": float(exact_result.effort),
                "adjacency": float(exact_result.adjacency),
                "violations": float(exact_result.violations),
                "factor_scores": {k: float(v) for k, v in exact_result.factor_scores.items()},
                "genome": [int(x) for x in best_genome],
            }
            self.best_exact = checkpoint["best_exact"]
            print(f"  Checkpoint gen {gen}: best_exact viol={exact_result.violations:.0f}", flush=True)
        
        ckpt_path = os.path.join(self.build_dir, f"v2_checkpoint_gen{gen}.json")
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self, keep=5):
        paths = glob.glob(os.path.join(self.build_dir, "v2_checkpoint_gen*.json"))
        if len(paths) <= keep:
            return
        # Sort by modification time (most recent first) so the newest checkpoints are kept
        paths_sorted = sorted(paths, key=os.path.getmtime, reverse=True)
        for path in paths_sorted[keep:]:
            try:
                os.remove(path)
            except OSError as exc:
                print(f"  Warning: could not remove old checkpoint {path}: {exc}", flush=True)
        # Also print how many were kept/removed for clarity
        removed = len(paths_sorted) - keep
        if removed > 0:
            print(f"  Cleanup: removed {removed} old checkpoint(s), kept {keep}", flush=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_evolution.py <build_dir> [--profile-fast]")
        sys.exit(1)

    profile_fast = "--profile-fast" in sys.argv
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    build_dir = args[0]
    config = Config.load(os.path.join(build_dir, "config_v2.yaml"))
    perf = PerfStats()

    seed = config.get("evolution.seed", 42)
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

    print(f"Loading data from {build_dir}...", flush=True)
    layout = build_layout(build_dir)
    print(f"  Positions: {layout.n_positions}, Shortcuts: {layout.n_shortcuts}", flush=True)
    print(f"  Mutable: {len(layout.mutable_indices)}", flush=True)
    print(f"  Pre-seeded assignments: {layout.n_assigned}", flush=True)

    # We'll create evaluator after computing scale factors from training data
    evaluator = FitnessEvaluator(weights=config.get("fitness.weights", {}), reference_layout=layout)
    raw_batch_evaluator = make_batch_evaluator(
        layout,
        evaluator,
        enabled=config.get("exact_eval.use_numba", True),
        tolerance=config.get("exact_eval.parity_tolerance", 1e-4),
    )

    # Evaluate pre-seeded genome
    seed_result = evaluator.evaluate(layout)
    print(f"  Seed fitness (raw): effort={seed_result.objectives[0]*1:.0f}, adj={seed_result.objectives[1]*1:.0f}, viol={seed_result.objectives[2]*1:.0f}", flush=True)

    # Initial exact evaluations for surrogate training
    n_initial = config.get("surrogate.n_initial_samples", 500)
    print(f"\nGenerating {n_initial} random layouts for surrogate training...", flush=True)
    train_layouts = generate_random_layouts(layout, n_initial)

    print("Evaluating exact fitness...", flush=True)
    t0 = time.perf_counter()
    train_scores = evaluate_exact_batch(
        train_layouts,
        layout,
        evaluator,
        batch_evaluator=raw_batch_evaluator,
        perf=perf,
        label="initial_raw_exact_eval",
    )
    t1 = time.perf_counter()
    print(f"  Done in {t1-t0:.1f}s ({(t1-t0)/n_initial*1000:.1f}ms per layout)", flush=True)
    print(f"  Score range: effort=[{train_scores[:,0].min():.2f}, {train_scores[:,0].max():.2f}], "
          f"adj=[{train_scores[:,1].min():.2f}, {train_scores[:,1].max():.2f}], "
          f"viol=[{train_scores[:,2].min():.2f}, {train_scores[:,2].max():.2f}]", flush=True)

    # Compute scale factors from training data std dev
    scale_factors = np.std(train_scores, axis=0)
    scale_factors = np.maximum(scale_factors, 1.0)  # prevent division by zero
    print(f"  Scale factors: effort={scale_factors[0]:.2f}, adj={scale_factors[1]:.2f}, viol={scale_factors[2]:.2f}", flush=True)

    # Re-create evaluator with scale factors
    evaluator = FitnessEvaluator(
        weights=config.get("fitness.weights", {}),
        reference_layout=layout,
        scale_factors=scale_factors,
    )
    normalized_batch_evaluator = make_batch_evaluator(
        layout,
        evaluator,
        enabled=config.get("exact_eval.use_numba", True),
        tolerance=config.get("exact_eval.parity_tolerance", 1e-4),
    )
    # CRITICAL: Re-evaluate training layouts with normalized evaluator so surrogate learns normalized scores
    train_scores = evaluate_exact_batch(
        train_layouts,
        layout,
        evaluator,
        batch_evaluator=normalized_batch_evaluator,
        perf=perf,
        label="initial_normalized_exact_eval",
    )
    # Compute total score for single-objective surrogate training
    total_scores = train_scores.sum(axis=1, keepdims=True)
    # Re-evaluate seed with normalized objectives
    seed_result = evaluator.evaluate(layout)
    print(f"  Seed fitness (normalized): effort={seed_result.objectives[0]:.2f}, adj={seed_result.objectives[1]:.2f}, viol={seed_result.objectives[2]:.2f}", flush=True)

    # Train surrogate on total scores (single-objective)
    surrogate = LayoutSurrogate(
        n_positions=layout.n_positions,
        n_shortcuts=layout.n_shortcuts,
        n_factors=1,
        hidden_dim=config.get("surrogate.hidden_dim", 128),
        embedding_dim=config.get("surrogate.embedding_dim", 32),
    )
    trainer = SurrogateTrainer(
        surrogate,
        device=None,
        mixed_precision=config.get("surrogate.mixed_precision", True),
        compile_model=config.get("surrogate.compile", True),
    )  # auto-detect GPU
    print(f"\nTraining surrogate ({surrogate.count_parameters()} params) on {trainer.device}...", flush=True)
    t_train = time.perf_counter()
    trainer.train(
        train_layouts,
        total_scores,
        epochs=config.get("surrogate.surrogate_epochs", 30),
        batch_size=config.get("surrogate.batch_size", 256),
    )
    perf.add("surrogate_initial_train", time.perf_counter() - t_train)

    t_pred = time.perf_counter()
    acc = trainer.evaluate(train_layouts[:500], total_scores[:500])
    perf.add("surrogate_initial_eval", time.perf_counter() - t_pred)
    r2_mean = float(np.mean(acc["r2"]))
    print(f"  Surrogate R^2 = {r2_mean:.4f}", flush=True)
    if r2_mean < config.get("surrogate.min_r2", 0.90):
        print("  Surrogate R^2 below gate; retraining with safer 1000-sample/60-epoch settings.", flush=True)
        target_n = max(n_initial, 1000)
        if len(train_layouts) < target_n:
            extra_layouts = generate_random_layouts(layout, target_n - len(train_layouts))
            extra_scores = evaluate_exact_batch(
                extra_layouts,
                layout,
                evaluator,
                batch_evaluator=normalized_batch_evaluator,
                perf=perf,
                label="r2_fallback_exact_eval",
            )
            extra_total = extra_scores.sum(axis=1, keepdims=True)
            train_layouts = np.vstack([train_layouts, extra_layouts])
            total_scores = np.vstack([total_scores, extra_total])
        t_train = time.perf_counter()
        trainer.train(train_layouts, total_scores, epochs=60, batch_size=config.get("surrogate.batch_size", 256))
        perf.add("surrogate_fallback_train", time.perf_counter() - t_train)
        acc = trainer.evaluate(train_layouts[:min(500, len(train_layouts))], total_scores[:min(500, len(total_scores))])
        r2_mean = float(np.mean(acc["r2"]))
        print(f"  Surrogate R^2 after fallback = {r2_mean:.4f}", flush=True)

    manager = SurrogateManager(surrogate, trainer,
                               retrain_every=config.get("surrogate.retrain_every", 200),
                               exact_eval_every=config.get("surrogate.exact_eval_every", 50))
    manager.add_exact_evaluations(train_layouts, total_scores)

    # Create pymoo problem that uses surrogate for fast evaluation
    from pymoo.core.problem import Problem

    class FastLayoutProblem(Problem):
        def __init__(self, n_positions, n_shortcuts, frozen_mask, manager):
            self.manager = manager
            self.frozen_mask = frozen_mask
            super().__init__(n_var=n_positions, n_obj=1, n_constr=0,
                           xl=-1, xu=n_shortcuts-1, vtype=int)

        def _evaluate(self, x, out, *args, **kwargs):
            t_eval = time.perf_counter()
            try:
                F = self.manager.trainer.predict(x)
            except Exception:
                F = np.zeros((x.shape[0], 1), dtype=np.float32)
            out["F"] = F
            perf.add("surrogate_predict", time.perf_counter() - t_eval)

    problem = FastLayoutProblem(
        n_positions=layout.n_positions,
        n_shortcuts=layout.n_shortcuts,
        frozen_mask=layout.frozen_mask,
        manager=manager,
    )

    algorithm = create_algorithm(
        n_positions=layout.n_positions,
        n_shortcuts=layout.n_shortcuts,
        frozen_mask=layout.frozen_mask,
        seed_genome=layout.genome,
        inject_seed=config.get("evolution.inject_seed", True),
        pop_size=config.get("evolution.pop_size", 100),
        crossover_prob=config.get("evolution.crossover_prob", 0.7),
        mutation_prob=config.get("evolution.mutation_prob", 0.15),
        eliminate_duplicates=config.get("evolution.eliminate_duplicates", False),
    )

    # The custom sampler injects the pre-seeded genome as the first initial individual.
    inject_seed = config.get("evolution.inject_seed", True)
    if inject_seed and layout.n_assigned > 0:
        print("  Seed genome will be injected as initial population individual 0.", flush=True)
    elif not inject_seed:
        print("  Seed injection DISABLED — starting from fresh random population.", flush=True)

    n_gen = config.get("evolution.n_generations", 999999)
    print(f"\nRunning evolution: pop={algorithm.pop_size}, gens={n_gen}", flush=True)
    t0 = time.time()

    callback = SurrogateCallback(
        layout,
        evaluator,
        manager,
        build_dir,
        exact_eval_batch=config.get("exact_eval.batch_size", 5),
        checkpoint_every=config.get("output.checkpoint_interval", 500),
        batch_evaluator=normalized_batch_evaluator,
        perf=perf,
    )
    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False, callback=callback)

    t1 = time.time()
    print(f"\nEvolution completed in {t1-t0:.1f}s", flush=True)

    # Show best results
    if res.X.ndim == 1:
        # Single solution returned (1D array)
        best_genome = res.X.copy()
        best_f = res.F
    else:
        best_idx = np.argmin(res.F.ravel())
        best_genome = res.X[best_idx]
        best_f = res.F[best_idx]
    print(f"Best layout (surrogate): total={float(best_f.ravel()[0]):.2f}", flush=True)

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
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config.raw,
        "scale_factors": [float(s) for s in scale_factors],
        "seed_fitness": {
            "effort": float(seed_result.objectives[0]),
            "adjacency": float(seed_result.objectives[1]),
            "violations": float(seed_result.objectives[2]),
        },
        "best_surrogate": {
            "total": float(best_f.ravel()[0]),
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
        },
        "surrogate_r2_initial": r2_mean,
        "surrogate_r2_history": manager.accuracy_history,
        "exact_eval_history": [(int(g), [float(x) for x in s]) for g, s in callback.exact_history],
        "evolved_surrogate_accuracy_history": [
            {"generation": int(g), **metrics} for g, metrics in callback.evolved_accuracy_history
        ],
        "pareto_front": (
            [{
                "genome": [int(x) for x in res.X],
                "objectives": [float(res.F.ravel()[0])],
            }] if res.X.ndim == 1 else [
                {
                    "genome": [int(x) for x in g],
                    "objectives": [float(obj)] if np.ndim(obj) == 0 else [float(o) for o in obj],
                }
                for g, obj in zip(res.X, res.F)
            ]
        ),
        "generation": n_gen,
        "population_size": algorithm.pop_size,
        "elapsed_seconds": t1 - t0,
        "perf": perf.summary(),
        "numba_exact_enabled": bool(normalized_batch_evaluator and normalized_batch_evaluator.enabled),
        "numba_parity": (
            normalized_batch_evaluator.parity.__dict__
            if normalized_batch_evaluator is not None and normalized_batch_evaluator.parity is not None
            else None
        ),
    }
    
    results_path = os.path.join(build_dir, "v2_evolution_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}", flush=True)
    perf_path = os.path.join(build_dir, "v2_perf_report.json")
    with open(perf_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numba_available": NUMBA_AVAILABLE,
            "profile_fast": profile_fast or config.get("profiling.enabled", False),
            "events": perf.summary(),
        }, f, indent=2, default=str)
    print(f"Performance report saved to {perf_path}", flush=True)

    print("\nv2 evolution complete!", flush=True)


if __name__ == "__main__":
    if "--profile-fast" in sys.argv:
        profile = cProfile.Profile()
        try:
            profile.enable()
            main()
        finally:
            profile.disable()
            args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
            build_dir = args[0] if args else "."
            stats_path = os.path.join(build_dir, "v2_profile_fast.pstats")
            txt_path = os.path.join(build_dir, "v2_profile_fast.txt")
            profile.dump_stats(stats_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                stats = pstats.Stats(profile, stream=f).sort_stats("cumtime")
                stats.print_stats(80)
            print(f"Profile saved to {stats_path} and {txt_path}", flush=True)
    else:
        main()
