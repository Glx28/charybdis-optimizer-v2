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


NUMBA_AVAILABLE = False

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    njit = None


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
        if self.surrogate_manager is not None and len(self.surrogate_manager.exact_cache) > 100:
            try:
                F = self.surrogate_manager.trainer.predict(x)
                out["F"] = F
                return
            except Exception:
                pass
        out["F"] = F


class PermutationSampling(Sampling):
    def __init__(self, n_shortcuts, frozen_mask=None, seed_genome=None, inject_seed=True):
        super().__init__()
        self.n_shortcuts = n_shortcuts
        self.frozen_mask = frozen_mask
        self.seed_genome = seed_genome
        self.inject_seed = inject_seed
        self.mutable = np.where(~frozen_mask)[0] if frozen_mask is not None else None
        self.frozen = np.where(frozen_mask)[0] if frozen_mask is not None else np.array([], dtype=int)
    
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), -1, dtype=int)
        mutable = self.mutable if self.mutable is not None else np.arange(problem.n_var)
        start = 0
        if self.inject_seed and self.seed_genome is not None and n_samples > 0:
            X[0] = np.asarray(self.seed_genome, dtype=int).copy()
            start = 1

        for i in range(n_samples):
            if i < start:
                continue
            n_assign = min(len(mutable), self.n_shortcuts)
            X[i, mutable[:n_assign]] = np.random.permutation(self.n_shortcuts)[:n_assign]
            # Preserve frozen positions from seed
            if self.frozen_mask is not None and self.seed_genome is not None:
                X[i, self.frozen] = self.seed_genome[self.frozen]
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
    def __init__(self, prob=0.15, frozen_mask=None):
        super().__init__()
        self.prob = prob
        self.mutable_indices = np.where(~frozen_mask)[0] if frozen_mask is not None else None
        self.mutable_list = self.mutable_indices.tolist() if self.mutable_indices is not None else None
    
    def _do(self, problem, X, **kwargs):
        n_samples, n_var = X.shape
        for i in range(n_samples):
            if random.random() > (float(self.prob.value) if hasattr(self.prob, "value") else float(self.prob)):
                continue
            if self.mutable_list is not None and len(self.mutable_list) >= 2:
                a, b = random.sample(self.mutable_list, 2)
                X[i, a], X[i, b] = X[i, b], X[i, a]
            else:
                a, b = random.sample(range(n_var), 2)
                X[i, a], X[i, b] = X[i, b], X[i, a]
        return X


class LayoutRepair(Repair):
    def __init__(self, n_shortcuts, frozen_mask=None, seed_genome=None):
        super().__init__()
        self.n_shortcuts = n_shortcuts
        self.frozen_mask = frozen_mask
        self.seed_genome = seed_genome
        self.frozen = np.where(frozen_mask)[0] if frozen_mask is not None else np.array([], dtype=int)

    def _do(self, problem, X, **kwargs):
        X[X >= self.n_shortcuts] = -1
        X[X < -1] = -1
        if self.frozen_mask is not None and self.seed_genome is not None and len(self.frozen) > 0:
            X[:, self.frozen] = self.seed_genome[self.frozen]
        return X


def create_algorithm(n_positions, n_shortcuts, frozen_mask=None, seed_genome=None, inject_seed=True,
                     pop_size=500, crossover_prob=0.7, mutation_prob=0.15,
                     eliminate_duplicates=False, layout=None):
    sampling = PermutationSampling(n_shortcuts=n_shortcuts, frozen_mask=frozen_mask, seed_genome=seed_genome, inject_seed=inject_seed)
    crossover = CycleCrossover(prob=crossover_prob, n_shortcuts=n_shortcuts)
    mutation = SwapMutation(prob=mutation_prob, frozen_mask=frozen_mask)
    repair = LayoutRepair(n_shortcuts=n_shortcuts, frozen_mask=frozen_mask, seed_genome=seed_genome)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        repair=repair,
        eliminate_duplicates=eliminate_duplicates,
    )
    return algorithm
