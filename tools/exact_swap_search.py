#!/usr/bin/env python3
"""Exact, contract-preserving one-swap hill climb for a checkpoint genome."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools._common import load_checkpoint, load_evaluator
from tools.run_contract_audit import audit_genome


def swap_pairs(genome, layout):
    """Enumerate distinct swaps among mutable positions only."""
    mutable = [i for i, pos in enumerate(layout.positions) if not pos.is_frozen]
    return [(a, b) for offset, a in enumerate(mutable)
            for b in mutable[offset + 1:] if genome[a] != genome[b]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Checkpoint or warmstart JSON")
    parser.add_argument("--config", default="config_v2.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--output", default="build/diagnostics/exact-swap-search.json")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--audit-candidates", type=int, default=250)
    parser.add_argument("--minimum-improvement", type=float, default=0.01)
    parser.add_argument("--cpu", action="store_true", help="Use Numba for diagnostic score estimation")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    generation = int(checkpoint.get("generation", checkpoint.get("source_generation", 0)) or 0)
    genome = np.asarray(checkpoint.get("best_genome", checkpoint.get("genome")), dtype=np.int32)
    evaluator = load_evaluator(args.config, args.data_dir, args.build_dir,
                               require_cuda=not args.cpu, use_cuda=False if args.cpu else None,
                               generation=generation)
    layout = evaluator.reference_layout
    if len(genome) != len(layout.positions):
        raise ValueError(f"genome has {len(genome)} genes; expected {len(layout.positions)}")

    base_audit = audit_genome(genome, generation, evaluator)
    if not base_audit["run_contract_pass"]:
        raise RuntimeError("starting checkpoint fails run contracts; repair/audit it before score search")
    objectives, constraints = evaluator.evaluate_batch(genome[None, :])
    score = float(objectives[0].sum())
    output = {
        "method": "exact_one_swap_hill_climb_cpu" if args.cpu else "exact_cuda_contract_preserving_one_swap_hill_climb",
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "source_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "generation": generation,
        "genome": genome.tolist(),
        "best_score": score,
        "objectives": objectives[0].tolist(),
        "constraints": constraints[0].tolist(),
        "audit": base_audit,
        "history": [],
    }
    start_time = time.time()
    for iteration in range(args.iterations):
        pairs = swap_pairs(genome, layout)
        improvers = []
        for start in range(0, len(pairs), args.batch_size):
            batch_pairs = pairs[start:start + args.batch_size]
            candidates = np.repeat(genome[None, :], len(batch_pairs), axis=0)
            for row, (a, b) in enumerate(batch_pairs):
                candidates[row, a], candidates[row, b] = candidates[row, b], candidates[row, a]
            batch_objectives, batch_constraints = evaluator.evaluate_batch(candidates)
            batch_scores = batch_objectives.sum(axis=1)
            feasible = np.all(batch_constraints <= 0, axis=1)
            indexes = np.flatnonzero(feasible & (batch_scores < score - args.minimum_improvement))
            improvers.extend((float(batch_scores[i]), batch_pairs[int(i)], candidates[int(i)].copy(),
                              batch_objectives[int(i)].copy(), batch_constraints[int(i)].copy())
                             for i in indexes)

        improvers.sort(key=lambda row: row[0])
        selected = None
        checked = 0
        for candidate_score, pair, candidate, candidate_objectives, candidate_constraints in improvers:
            if checked >= args.audit_candidates:
                break
            checked += 1
            audit = audit_genome(candidate, generation, evaluator)
            if audit["run_contract_pass"]:
                selected = (candidate_score, pair, candidate, candidate_objectives,
                            candidate_constraints, audit)
                break

        if selected is None:
            print(f"iteration={iteration + 1} local optimum; swaps={len(pairs)} "
                  f"feasible_score_improvers={len(improvers)} audited={checked}", flush=True)
            break

        score, pair, genome, objectives, constraints, audit = selected
        entry = {
            "iteration": iteration + 1, "swap": list(pair), "score": score,
            "objectives": objectives.tolist(), "constraints": constraints.tolist(),
            "audited_candidates": checked, "contract_checks": audit["checks"],
        }
        output["history"].append(entry)
        output.update(genome=genome.tolist(), best_score=score,
                      objectives=objectives.tolist(), constraints=constraints.tolist(), audit=audit)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
        print(f"iteration={iteration + 1} score={score:.6f} swap={pair} "
              f"swaps={len(pairs)} feasible_score_improvers={len(improvers)} "
              f"audited={checked} elapsed={time.time() - start_time:.1f}s", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(f"final_score={output['best_score']:.6f} steps={len(output['history'])} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
