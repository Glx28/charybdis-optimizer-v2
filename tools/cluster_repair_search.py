#!/usr/bin/env python3
"""Greedy cluster-repair local search.

Takes a checkpoint (default: the current warmstart) and tries to consolidate
split semantic clusters onto single layers without breaking acceptance.

Usage:
    .venv/bin/python tools/cluster_repair_search.py \
        --checkpoint build/runs/.../v2_checkpoint_gen*.json \
        --output build/v2_local_search_result.json
"""
import argparse
import copy
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evolution.acceptance import build_acceptance_report
from evolution.arrow_cluster import analyze_arrows
from evolution.completion_cluster import analyze_completion_cluster
from tools._common import load_checkpoint, load_evaluator, load_layout
from tools.semantic_cluster_report import _cluster_quality


def _layout_reports(layout):
    from run_evolution import analyze_duplicates

    dup = analyze_duplicates(layout)
    comp = analyze_completion_cluster(layout)
    arr = analyze_arrows(layout)
    acc = build_acceptance_report(layout, duplicate_report=dup, completion_cluster_report=comp, arrow_report=arr)
    return acc


def _acceptance_passes(layout) -> bool:
    acc = _layout_reports(layout)
    return bool(acc.get("optimizer_side_pass", False))


def _count_clusters_together(layout) -> Tuple[int, int]:
    together = 0
    order_ok = 0
    for cluster in layout.semantic_clusters:
        r = _cluster_quality(layout, cluster)
        if r.get("fully_together"):
            together += 1
            if r.get("relative_layout_required") and r.get("order_errors", 0) == 0:
                order_ok += 1
    return together, order_ok


def _sid_positions(layout) -> Dict[int, int]:
    """Map sid -> gene index for currently assigned sids."""
    pos = {}
    for idx, sid in enumerate(layout.genome):
        if sid >= 0:
            pos[int(sid)] = int(idx)
    return pos


def _positions_on_layer(layout, layer: int, exclude_sids: set,
                        genome: Optional[np.ndarray] = None) -> List[int]:
    """Return mutable gene indices on layer excluding frozen/L0/L7 and given sids."""
    genome_arr = layout.genome if genome is None else genome
    out = []
    for p in layout.positions:
        if p.is_frozen or p.layer in (0, 7):
            continue
        if p.layer != layer:
            continue
        sid = int(genome_arr[p.gene_idx])
        if sid in exclude_sids:
            continue
        out.append(int(p.gene_idx))
    return out


def _move_cluster_to_layer(layout, cluster, target_layer: int, rng,
                           base_genome: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Move a complete cluster without dropping bindings or breaking access.

    Ordered groups move only when target-layer positions match their declared
    offsets. Return None for unsafe, infeasible, or geometry-breaking moves.
    """
    members = list(cluster.get("members", []))
    sids = [int(m.get("sid", -1)) for m in members]
    sids = [sid for sid in sids if 0 <= sid < layout.n_shortcuts]
    if len(sids) < 2:
        return None

    genome = np.asarray(base_genome if base_genome is not None else layout.genome, dtype=np.int32).copy()
    # Select one primary placement while preserving any intentional extra copies.
    sid_pos = {}
    for idx, sid in enumerate(genome):
        if sid >= 0:
            sid_pos.setdefault(int(sid), int(idx))

    # Classify members and validate they are movable.
    to_move = []
    unassigned = []
    for sid in sids:
        sc = layout.shortcuts[sid]
        if sc.is_layer_access or sc.category == "mouse" or getattr(sc, "is_l0_only", False):
            return None  # don't touch structural shortcuts
        idx = sid_pos.get(sid)
        if idx is None:
            unassigned.append(sid)
            continue
        pos = layout.positions[idx]
        if pos.is_frozen or pos.layer == 7:
            return None  # cannot move frozen/L7 members
        # Move every primary member together; leaving one in place prevents an
        # ordered group from being repaired atomically.
        to_move.append((sid, idx))

    # Ordered groups move only when their complete relative shape fits. A random
    offsets = []
    for m in members:
        sid = int(m.get("sid", -1))
        if sid in sids:
            offsets.append((
                sid,
                int(round(float(m.get("dx", 0.0)))),
                int(round(float(m.get("dy", 0.0)))),
            ))

    move_sids = [sid for sid, _ in to_move] + unassigned
    relative_required = any(dx != 0 or dy != 0 for _, dx, dy in offsets)
    if relative_required:
        if len(move_sids) != len(offsets) or len(set(sids)) != len(sids):
            return None

        positions_by_coord = {}
        for pos in layout.positions:
            if pos.is_frozen or pos.layer != target_layer:
                continue
            key = (round(pos.x), round(pos.y))
            positions_by_coord[key] = int(pos.gene_idx)
        anchor_keys = [key for key in positions_by_coord
                       if any(dx == 0 and dy == 0 for _, dx, dy in offsets)]
        rng.shuffle(anchor_keys)
        source_positions = [sid_pos[sid] for sid in sids if sid in sid_pos]
        member_set = set(sids)
        candidates = []
        for ax_key in anchor_keys:
            ax, ay = ax_key
            shape_by_sid = {}
            valid = True
            for sid, dx, dy in offsets:
                tidx = positions_by_coord.get((ax + dx, ay + dy))
                if tidx is None:
                    valid = False
                    break
                shape_by_sid[sid] = tidx
            if not valid:
                continue

            target_by_sid = shape_by_sid
            targets = [target_by_sid[sid] for sid in sids]
            target_set = set(targets)
            displaced = [int(genome[idx]) for idx in targets
                         if genome[idx] >= 0 and int(genome[idx]) not in member_set]
            candidate = genome.copy()
            for idx in source_positions:
                if idx not in target_set:
                    candidate[idx] = -1
            for sid, idx in zip(sids, targets):
                candidate[idx] = sid

            free_slots = [int(pos.gene_idx) for pos in layout.positions
                          if not pos.is_frozen and pos.layer not in (0, 7)
                          and int(pos.gene_idx) not in target_set
                          and candidate[pos.gene_idx] < 0]
            if len(free_slots) < len(displaced):
                continue
            for sid, idx in zip(displaced, free_slots):
                candidate[idx] = sid
            displaced_weight = sum(
                float(layout.shortcuts[sid].importance) for sid in displaced
            )
            movement_count = sum(genome[idx] != candidate[idx]
                                 for idx in range(len(genome)))
            candidates.append((displaced_weight, movement_count, candidate))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    n_needed = len(move_sids)
    target_positions = _positions_on_layer(layout, target_layer, set(sids), genome)
    if len(target_positions) < n_needed:
        return None
    # An unassigned member has no source slot where a displaced occupant can
    # be preserved. Give those members genuinely empty targets first; random
    # target selection used to reject nearly every otherwise feasible repair
    # when it happened to place an unassigned SID over an occupied key.
    unassigned_set = set(unassigned)
    unassigned_targets = [None] * len(unassigned)
    empty_targets = [idx for idx in target_positions if int(genome[idx]) < 0]
    if len(empty_targets) < len(unassigned):
        return None
    if unassigned:
        selected_empty = rng.choice(empty_targets, size=len(unassigned), replace=False).tolist()
        unassigned_targets = selected_empty
    reserved = set(unassigned_targets)
    remaining_targets = [idx for idx in target_positions if idx not in reserved]
    assigned_targets = rng.choice(
        remaining_targets, size=len(to_move), replace=False,
    ).tolist() if to_move else []
    target_by_sid = dict(zip(unassigned, unassigned_targets))
    target_by_sid.update({sid: idx for (sid, _), idx in zip(to_move, assigned_targets)})
    placed_targets = [target_by_sid[sid] for sid in move_sids]

    # Source positions for moved members; unassigned members have no source.
    source_positions = [idx for _, idx in to_move] + [None] * len(unassigned)
    occupant_sids = [int(genome[t]) for t in placed_targets]

    # A cluster move cannot discard a displaced shortcut. Access-key relocation
    # is allowed only because _evaluate immediately checks dynamic reachability,
    # thumb rules, and optimizer acceptance on the resulting full genome.
    for occupant, source_idx in zip(occupant_sids, source_positions):
        if occupant < 0:
            continue
        sc = layout.shortcuts[occupant]
        if sc.category == "mouse" or getattr(sc, "is_l0_only", False):
            return None
        if source_idx is None:
            return None

    # Sanity: no overlap between source and target for assigned moves.
    assigned_sources = [idx for idx in source_positions if idx is not None]
    if set(assigned_sources) & set(placed_targets):
        return None

    # Place members on target layer.
    for sid, tidx in zip(move_sids, placed_targets):
        genome[tidx] = sid

    # Send every displaced occupant back to its corresponding source position.
    for occ_sid, src_idx in zip(occupant_sids, source_positions):
        if src_idx is not None:
            genome[src_idx] = occ_sid

    return genome


def _evaluate(layout, evaluator, genome):
    candidate = layout.clone_with(genome=genome.astype(np.int32))
    result = evaluator.evaluate(candidate)
    if np.any(np.asarray(result.constraints) > 0.0):
        return None, result, {}, 0, 0
    acc = _layout_reports(candidate)
    if not acc.get("optimizer_side_pass", False):
        return None, result, acc, 0, 0
    together, order_ok = _count_clusters_together(candidate)
    return candidate, result, acc, together, order_ok


def main():
    parser = argparse.ArgumentParser(description="Greedy semantic-cluster repair local search")
    parser.add_argument("--checkpoint", default="build/v2_local_search_result.json",
                        help="Checkpoint JSON or warmstart JSON with 'genome' field")
    parser.add_argument("--output", default="build/v2_local_search_result.json",
                        help="Where to write the repaired genome")
    parser.add_argument("--max-moves", type=int, default=1000,
                        help="Maximum successful consolidation moves")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-score-regression", type=float, default=0.10,
                        help="Allow up to this fraction of score regression if clusters improve")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Simulated-annealing temperature (0 = greedy only)")
    parser.add_argument("--pair-moves", action="store_true",
                        help="Try simultaneous two-cluster moves when single moves stall")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    evaluator = load_evaluator(require_cuda=False)
    # Disable semantic-cluster multiplier for local search so acceptance and
    # base score are stable; we optimize cluster count directly.
    evaluator.set_semantic_cluster_multiplier(0.0)
    # Use the evaluator's reference layout so semantic clusters and shortcut
    # importance overrides match the evolution config exactly.
    layout = evaluator.reference_layout

    # Load starting genome.
    if os.path.basename(args.checkpoint) == "v2_local_search_result.json" and not os.path.exists(args.checkpoint):
        # Fallback to best known checkpoint if warmstart file missing.
        args.checkpoint = "build/runs/v2_cluster_final_wfc100_g10000_20260901_054232/v2_checkpoint_gen10000.json"
    ckpt = load_checkpoint(args.checkpoint)
    genome = np.asarray(ckpt.get("best_genome") or ckpt.get("genome"), dtype=np.int32)
    layout = layout.clone_with(genome=genome)

    base_result = evaluator.evaluate(layout)
    base_acc = _layout_reports(layout)
    base_together, base_order_ok = _count_clusters_together(layout)
    base_score = float(base_result.total_score)
    print(f"Start: {base_together}/{len(layout.semantic_clusters)} clusters together, "
          f"score={base_score:.4f}, acceptance_pass={base_acc.get('optimizer_side_pass', False)}")

    if not base_acc.get("optimizer_side_pass", False):
        print("WARNING: starting layout does not pass acceptance; repair may fail.", flush=True)

    best_layout = layout
    best_together = base_together
    best_order_ok = base_order_ok
    best_score = base_score
    best_genome = genome.copy()

    clusters = list(layout.semantic_clusters)
    # Sort by weight descending; also prioritize critical clusters.
    clusters_sorted = sorted(clusters, key=lambda c: (-float(c.get("is_critical", False)) * 1000.0,
                                                     -float(c.get("weight", 1.0))))

    # Candidate layers: all mutable non-L0/L7 layers.
    all_candidate_layers = sorted({
        int(p.layer) for p in layout.positions
        if not p.is_frozen and p.layer not in (0, 7)
    })

    move_count = 0
    no_progress_generations = 0
    start_time = time.time()
    temperature = args.temperature

    def _candidate_layers_for_cluster(cluster, current_layout):
        members = list(cluster.get("members", []))
        sids = [int(m.get("sid", -1)) for m in members]
        layer_counts = defaultdict(int)
        sid_pos = _sid_positions(current_layout)
        for sid in sids:
            if sid in sid_pos:
                layer = current_layout.positions[sid_pos[sid]].layer
                layer_counts[int(layer)] += 1
        dominant_layers = sorted(layer_counts.keys(), key=lambda l: (-layer_counts[l], l))
        other_layers = [l for l in all_candidate_layers if l not in layer_counts]
        return dominant_layers + other_layers

    def _try_cluster_move(cluster, current_layout, current_score, current_together, current_order_ok):
        """Try moving cluster to all candidate layers; return best valid move."""
        best_move = None
        for target_layer in _candidate_layers_for_cluster(cluster, current_layout):
            if target_layer in (0, 7):
                continue
            new_genome = _move_cluster_to_layer(current_layout, cluster, target_layer, rng)
            if new_genome is None:
                continue
            cand_layout, cand_result, cand_acc, cand_together, cand_order_ok = _evaluate(
                current_layout, evaluator, new_genome
            )
            if cand_layout is None:
                continue
            if cand_order_ok < current_order_ok:
                continue
            cand_score = float(cand_result.total_score)
            # Preserve both co-location and required relative geometry.
            is_better = False
            if cand_together > current_together or cand_order_ok > current_order_ok:
                # Allow some score regression; tighter regression for bigger count jumps.
                regression_limit = 1 + args.allow_score_regression
                if cand_together > best_together or cand_order_ok > best_order_ok:
                    regression_limit = 1 + args.allow_score_regression * 1.5
                if cand_score <= current_score * regression_limit:
                    is_better = True
            elif (cand_together == current_together and cand_order_ok == current_order_ok
                  and cand_score < current_score * 0.9995):
                is_better = True

            if (not is_better and temperature > 0 and cand_together >= current_together
                    and cand_order_ok >= current_order_ok):
                delta = cand_score - current_score
                prob = np.exp(-delta / (temperature * abs(current_score) + 1e-9))
                if rng.random() < prob:
                    is_better = True

            if is_better:
                if best_move is None:
                    best_move = (cand_layout, new_genome, cand_score, cand_together, cand_order_ok, target_layer)
                else:
                    _, _, bs, bt, _, _ = best_move
                    if (cand_order_ok > best_move[4]
                            or (cand_order_ok == best_move[4] and cand_together > bt)
                            or (cand_order_ok == best_move[4] and cand_together == bt and cand_score < bs)):
                        best_move = (cand_layout, new_genome, cand_score, cand_together, cand_order_ok, target_layer)
        return best_move

    def _try_pair_move(cluster_a, cluster_b, current_layout, current_score, current_together,
                       current_order_ok):
        """Try moving two split clusters simultaneously; return best valid move."""
        sids_a = {int(m.get("sid", -1)) for m in cluster_a.get("members", [])}
        sids_b = {int(m.get("sid", -1)) for m in cluster_b.get("members", [])}
        if sids_a & sids_b:
            return None
        layers_a = _candidate_layers_for_cluster(cluster_a, current_layout)[:4]
        layers_b = _candidate_layers_for_cluster(cluster_b, current_layout)[:4]
        best_move = None
        for la in layers_a:
            for lb in layers_b:
                if la in (0, 7) or lb in (0, 7):
                    continue
                g1 = _move_cluster_to_layer(current_layout, cluster_a, la, rng)
                if g1 is None:
                    continue
                g2 = _move_cluster_to_layer(current_layout, cluster_b, lb, rng, base_genome=g1)
                if g2 is None:
                    continue
                cand_layout, cand_result, cand_acc, cand_together, cand_order_ok = _evaluate(
                    current_layout, evaluator, g2
                )
                if cand_layout is None:
                    continue
                cand_score = float(cand_result.total_score)
                # Pair moves must strictly increase cluster count.
                if cand_together < current_together or cand_order_ok < current_order_ok:
                    continue
                if cand_together == current_together and cand_order_ok == current_order_ok:
                    continue
                regression_limit = 1 + args.allow_score_regression * 2.0
                if cand_score > current_score * regression_limit:
                    continue
                if best_move is None:
                    best_move = (cand_layout, g2, cand_score, cand_together, cand_order_ok, la, lb)
                else:
                    _, _, bs, bt, _, _, _ = best_move
                    if (cand_order_ok > best_move[4]
                            or (cand_order_ok == best_move[4] and cand_together > bt)
                            or (cand_order_ok == best_move[4] and cand_together == bt and cand_score < bs)):
                        best_move = (cand_layout, g2, cand_score, cand_together, cand_order_ok, la, lb)
        return best_move

    while move_count < args.max_moves and no_progress_generations < 3:
        improved_this_round = False
        # --- Single-cluster phase ---
        for cluster in clusters_sorted:
            r = _cluster_quality(best_layout, cluster)
            if (r.get("fully_together") and
                    (not r.get("relative_layout_required") or r.get("order_errors", 0) == 0)):
                continue

            best_move = _try_cluster_move(
                cluster, best_layout, best_score, best_together, best_order_ok,
            )
            if best_move is not None:
                cand_layout, new_genome, cand_score, cand_together, cand_order_ok, target_layer = best_move
                cluster_name = cluster.get("name", "unknown")
                print(f"  Move {move_count+1}: consolidated {cluster_name} to L{target_layer} "
                      f"({cand_together}/{len(clusters)} clusters, score={cand_score:.4f})", flush=True)
                best_layout = cand_layout
                best_genome = new_genome.copy()
                best_score = cand_score
                best_together = cand_together
                best_order_ok = cand_order_ok
                move_count += 1
                improved_this_round = True
                if move_count >= args.max_moves:
                    break

        # --- Pair-move phase (only if singles stalled this round) ---
        if not improved_this_round and args.pair_moves:
            split_clusters = [c for c in clusters_sorted if not _cluster_quality(best_layout, c).get("fully_together")]
            # Try highest-weight pairs first.
            for i, cluster_a in enumerate(split_clusters[:12]):
                for cluster_b in split_clusters[i+1:13]:
                    pair_move = _try_pair_move(
                        cluster_a, cluster_b, best_layout, best_score, best_together, best_order_ok,
                    )
                    if pair_move is not None:
                        cand_layout, new_genome, cand_score, cand_together, cand_order_ok, la, lb = pair_move
                        name_a = cluster_a.get("name", "unknown")
                        name_b = cluster_b.get("name", "unknown")
                        print(f"  Move {move_count+1}: pair {name_a}→L{la}, {name_b}→L{lb} "
                              f"({cand_together}/{len(clusters)} clusters, score={cand_score:.4f})", flush=True)
                        best_layout = cand_layout
                        best_genome = new_genome.copy()
                        best_score = cand_score
                        best_together = cand_together
                        best_order_ok = cand_order_ok
                        move_count += 1
                        improved_this_round = True
                        if move_count >= args.max_moves:
                            break
                if move_count >= args.max_moves:
                    break

        if not improved_this_round:
            no_progress_generations += 1
        else:
            no_progress_generations = 0
            # Cool down annealing temperature.
            temperature *= 0.9

    elapsed = time.time() - start_time
    final_together, final_order_ok = _count_clusters_together(best_layout)
    final_acc = _layout_reports(best_layout)
    final_result = evaluator.evaluate(best_layout)
    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Clusters together: {base_together} -> {final_together}/{len(clusters)} "
          f"(order-correct: {base_order_ok} -> {final_order_ok})")
    print(f"Score: {base_score:.4f} -> {float(final_result.total_score):.4f}")
    print(f"Acceptance pass: {final_acc.get('optimizer_side_pass', False)}")

    # Write output compatible with warmstart.
    output = {
        "genome": [int(x) for x in best_genome],
        "best_score": float(final_result.total_score),
        "best_exact": {
            "objectives": [float(x) for x in final_result.objectives],
            "constraints": [float(x) for x in final_result.constraints],
            "total_score": float(final_result.total_score),
            "optimizer_side_pass": bool(final_acc.get("optimizer_side_pass", False)),
            "acceptance_failed_checks": [k for k, ok in final_acc.get("checks", {}).items() if not ok],
        },
        "clusters_together": {
            "before": base_together,
            "after": final_together,
            "total": len(clusters),
        },
        "moves": move_count,
        "elapsed_seconds": elapsed,
        "source_checkpoint": args.checkpoint,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Wrote repaired genome to {args.output}")


if __name__ == "__main__":
    main()
