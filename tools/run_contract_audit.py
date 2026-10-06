#!/usr/bin/env python3
"""Audit the exact-best genome against the validated-run contracts."""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution.acceptance import (
    _layer_access_assignments,
    _reachable_layers_from_access_rows,
    build_acceptance_report,
)
from run_evolution import analyze_duplicates
from evolution.arrow_cluster import analyze_arrows
from evolution.completion_cluster import analyze_completion_cluster
from tools._common import load_checkpoint, load_evaluator
from tools.semantic_cluster_report import _cluster_quality


def _contract_checks(optimizer_acceptance, constraints, frozen_positions, ordered_failures):
    return {
        "optimizer_acceptance": bool(optimizer_acceptance),
        "hard_constraints": bool(np.all(constraints <= 0)),
        "frozen_positions_unchanged": bool(frozen_positions),
        "required_relative_layouts": not ordered_failures,
    }


def audit_genome(genome, generation, evaluator):
    """Return the validated-run contract audit for one genome."""
    base = evaluator.reference_layout
    layout = base.clone_with(genome=np.asarray(genome, dtype=np.int32))
    objectives, constraints = evaluator.evaluate_batch(layout.genome[None, :])
    acceptance = build_acceptance_report(
        layout, duplicate_report=analyze_duplicates(layout),
        completion_cluster_report=analyze_completion_cluster(layout), arrow_report=analyze_arrows(layout),
    )
    reachable = _reachable_layers_from_access_rows(_layer_access_assignments(layout))
    occupancy = Counter(pos.layer for pos, sid in zip(layout.positions, layout.genome) if sid >= 0)
    functional_occupancy = Counter(
        pos.layer for pos, sid in zip(layout.positions, layout.genome)
        if sid >= 0 and not layout.shortcuts[int(sid)].is_layer_access
    )
    # Occupancy is evidence for review, not an acceptance threshold. A
    # reachable layer can be useful with few bindings when its workflow merits it.
    active_generated_occupancy = {
        layer: count for layer, count in occupancy.items()
        if layer not in (0, 7) and layer in reachable
    }
    frozen = bool(np.array_equal(layout.genome[base.frozen_mask], base.genome[base.frozen_mask]))
    clusters = [_cluster_quality(layout, cluster) for cluster in layout.semantic_clusters]
    critical_split = [row.get("name") for cluster, row in zip(layout.semantic_clusters, clusters)
                      if cluster.get("is_critical") and not row.get("fully_together", False)]
    relative_report = acceptance["details"]["required_relative_layouts"]
    ordered_failures = [row["name"] for row in relative_report["failures"]]
    checks = _contract_checks(acceptance["optimizer_side_pass"], constraints,
                              frozen, ordered_failures)
    return {
        "generation": generation, "best_generation": generation,
        "checks": checks, "run_contract_pass": all(checks.values()),
        "objectives": objectives[0].tolist(), "constraints": constraints[0].tolist(),
        "reachable_generated_layer_occupancy": active_generated_occupancy,
        "reachable_generated_functional_occupancy": {
            layer: int(functional_occupancy[layer]) for layer in sorted(reachable)
            if layer not in (0, 7)
        },
        "reachable_access_only_layers": [
            layer for layer in sorted(reachable)
            if layer not in (0, 7) and functional_occupancy[layer] == 0
        ],
        "layer_occupancy": dict(occupancy),
        "critical_split_clusters": critical_split, "semantic_clusters": clusters,
        "ordered_relation_failures": ordered_failures,
        "acceptance": acceptance, "export_validation_pending": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Run audit evaluator on CPU")
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    evaluator = load_evaluator(require_cuda=not args.cpu, use_cuda=False if args.cpu else None,
                               generation=checkpoint["generation"])
    audit = audit_genome(checkpoint.get("best_genome", checkpoint.get("genome")),
                         checkpoint["generation"], evaluator)
    print(json.dumps(audit, indent=2))
    return 0 if audit["run_contract_pass"] else 5


if __name__ == "__main__":
    sys.exit(main())
