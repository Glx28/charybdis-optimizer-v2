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
    ):
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba is required by the single-source fitness model")

        self.layout = layout
        self.weights = weights
        self.violation_weights = violation_weights
        self.missing_important_threshold = missing_important_threshold
        self.scale_factors = scale_factors if scale_factors is not None else np.ones(3, dtype=np.float32)
        self.reference_genome = reference_genome
        self.hard_constraints = hard_constraints or []
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
        self._right_alt_sid = next((int(s.sid) for s in layout.shortcuts
                                    if s.base_key == "RightAlt" and not s.modifiers), -1)
        self._semantic_position_edges = self._build_semantic_position_edges(layout)
        self._semantic_position_edges_array = np.asarray(
            self._semantic_position_edges, dtype=np.float32
        ).reshape((-1, 5))
        self._position_layers = np.asarray([p.layer for p in layout.positions], dtype=np.int32)
        self._position_x = np.asarray([p.x for p in layout.positions], dtype=np.float32)
        self._position_y = np.asarray([p.y for p in layout.positions], dtype=np.float32)
        self._semantic_cluster_starts, self._semantic_cluster_sids, self._semantic_cluster_weights = (
            self._build_semantic_contract_groups(layout)
        )
        self._access_targets = np.full(layout.n_shortcuts, -1, dtype=np.int32)
        for shortcut in layout.shortcuts:
            if not getattr(shortcut, "is_layer_access", False):
                continue
            target = int(getattr(shortcut, "access_target_layer", -1))
            if 0 <= target < 32:
                self._access_targets[int(shortcut.sid)] = target
        self.arrays = precompute(
            layout,
            weights=weights,
            violation_weights=violation_weights,
            missing_important_threshold=missing_important_threshold,
            scale_factors=self.scale_factors,
            reference_genome=reference_genome,
            hard_constraints=self.hard_constraints,
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
            hard_constraints=self.hard_constraints,
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
        if self._right_alt_sid >= 0 and self.right_alt_l0_penalty > 0.0:
            values[:, 2] += _right_alt_l0_penalty_batch(
                batch, self._position_layers, self._right_alt_sid,
                np.float32(self.right_alt_l0_penalty),
            )
        return values

    @property
    def n_constraints(self) -> int:
        return len(self.hard_constraints)

    def evaluate(self, genome: np.ndarray):
        """Return (objectives, constraints) for a single genome."""
        genome = np.asarray(genome, dtype=np.int32)
        objectives, constraints = _single_genome(genome, *self.arrays)
        objectives = self._add_semantic_position_score(objectives, genome)[0]
        return objectives, constraints

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
        return self._add_semantic_position_score(objectives, genomes), constraints

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
