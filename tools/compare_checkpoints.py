#!/usr/bin/env python3
"""Compare checkpoints with fresh same-stage exact and contract audits."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools._common import checkpoint_to_layout, load_checkpoint, load_evaluator, resolve_checkpoint_path
from tools.run_contract_audit import audit_genome

OBJECTIVE_NAMES = ("effort", "adjacency", "violations")


def _comparison_verdict(score_delta, a_pass, b_pass):
    if b_pass and not a_pass:
        return "B passes where A fails"
    if a_pass and not b_pass:
        return "B fails where A passes"
    if not a_pass and not b_pass:
        return "Both fail run contract"
    if score_delta < -0.01:
        return "B has lower (better) exact score; both pass"
    if score_delta > 0.01:
        return "B has higher (worse) exact score; both pass"
    return "Similar exact score; both pass"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _side_report(path, checkpoint, audit):
    acceptance = audit["acceptance"]
    owner = acceptance["details"]["owner_layout_contract_v1"]
    owner_details = owner["details"]
    required = owner_details.get("required_l1_shortcuts", {})
    return {
        "path": str(Path(path).resolve()),
        "checkpoint_sha256": _sha256(path),
        "generation": int(checkpoint.get("generation", 0)),
        "best_generation": int(checkpoint.get("best_generation", checkpoint.get("generation", 0))),
        "objectives": [float(x) for x in audit["objectives"]],
        "exact_score": float(sum(audit["objectives"])),
        "constraints": [float(x) for x in audit["constraints"]],
        "run_contract_pass": bool(audit["run_contract_pass"]),
        "run_contract_failures": [k for k, v in audit["checks"].items() if not v],
        "optimizer_side_pass": bool(acceptance["optimizer_side_pass"]),
        "optimizer_side_failures": [
            k for k, v in acceptance["optimizer_side_checks"].items() if not v
        ],
        "owner_contract_pass": bool(owner["acceptance_pass"]),
        "owner_contract_failures": list(owner.get("failed_checks", [])),
        "required_l1_shortcuts": {
            key: {
                "pass": bool(value.get("pass")),
                "positions": value.get("legal_l1_positions", []),
            }
            for key, value in sorted(required.items())
        },
        "fixed_mouse_positions": owner_details.get("fixed_mouse_positions", {}),
        "mouse_button_positions": owner_details.get("mouse_button_positions", {}),
        "scroll_bindings": owner_details.get("scroll_bindings", []),
        "mb4_mb5_pairs": owner_details.get("mb4_mb5_pairs", []),
        "l1_lock": owner_details.get("l1_self_locks", []),
        "l7_content_sha256": owner_details.get("l7_content_sha256"),
        "semantic_clusters_together": sum(
            bool(row.get("fully_together")) for row in audit.get("semantic_clusters", [])
        ),
        "semantic_clusters_total": len(audit.get("semantic_clusters", [])),
        "relative_layout_failures": list(audit.get("ordered_relation_failures", [])),
    }


def compare(path_a, path_b, *, cpu=False):
    ckpt_a, ckpt_b = load_checkpoint(path_a), load_checkpoint(path_b)
    gen_a, gen_b = int(ckpt_a.get("generation", 0)), int(ckpt_b.get("generation", 0))
    # Both layouts are rescored with one evaluator at the later checkpoint's
    # scheduled stage, avoiding incomparable stored scores and stage weights.
    reference_generation = max(gen_a, gen_b)
    evaluator = load_evaluator(
        require_cuda=not cpu,
        use_cuda=False if cpu else None,
        generation=reference_generation,
    )
    base = evaluator.reference_layout
    layout_a = checkpoint_to_layout(ckpt_a, base)
    layout_b = checkpoint_to_layout(ckpt_b, base)
    audit_a = audit_genome(layout_a.genome, gen_a, evaluator)
    audit_b = audit_genome(layout_b.genome, gen_b, evaluator)
    side_a = _side_report(path_a, ckpt_a, audit_a)
    side_b = _side_report(path_b, ckpt_b, audit_b)

    changed, by_layer = [], {}
    for idx, (sid_a, sid_b) in enumerate(zip(layout_a.genome, layout_b.genome)):
        sid_a, sid_b = int(sid_a), int(sid_b)
        if sid_a == sid_b:
            continue
        pos = base.positions[idx]
        layer = int(pos.layer)
        by_layer[layer] = by_layer.get(layer, 0) + 1
        changed.append({
            "layer": layer, "x": float(pos.x), "y": float(pos.y),
            "from": base.shortcuts[sid_a].keys if sid_a >= 0 else "Transparent",
            "to": base.shortcuts[sid_b].keys if sid_b >= 0 else "Transparent",
        })

    score_delta = side_b["exact_score"] - side_a["exact_score"]
    objective_deltas = {
        name: float(b - a)
        for name, a, b in zip(OBJECTIVE_NAMES, side_a["objectives"], side_b["objectives"])
    }
    verdict = _comparison_verdict(
        score_delta, side_a["run_contract_pass"], side_b["run_contract_pass"]
    )

    return {
        "comparison_version": 1,
        "fitness_reference_generation": reference_generation,
        "same_stage_exact_evaluation": True,
        "objective_names": list(OBJECTIVE_NAMES),
        "contract_sha256": audit_b["acceptance"]["details"]["owner_layout_contract_v1"]["contract_sha256"],
        "a": side_a,
        "b": side_b,
        "delta_b_minus_a": score_delta,
        "objective_deltas_b_minus_a": objective_deltas,
        "positions_changed": len(changed),
        "positions_changed_by_layer": {str(k): v for k, v in sorted(by_layer.items())},
        "changed_bindings": changed,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Use CPU evaluator; avoids CUDA contention")
    parser.add_argument("--output", type=Path, help="Write detailed JSON comparison artifact")
    parser.add_argument("ckpt_a", help="First checkpoint path or 'latest'")
    parser.add_argument("ckpt_b", help="Second checkpoint path or 'latest'")
    args = parser.parse_args()

    path_a, path_b = resolve_checkpoint_path(args.ckpt_a), resolve_checkpoint_path(args.ckpt_b)
    report = compare(path_a, path_b, cpu=args.cpu)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("=== Compare Checkpoints ===")
    print(f"A: {Path(path_a).name} (gen {report['a']['generation']})")
    print(f"B: {Path(path_b).name} (gen {report['b']['generation']})")
    print(f"Common exact-fitness stage: gen {report['fitness_reference_generation']}")
    print(f"A: {report['a']['exact_score']:.4f}; run contract pass={report['a']['run_contract_pass']}")
    print(f"B: {report['b']['exact_score']:.4f}; run contract pass={report['b']['run_contract_pass']}")
    print(f"Delta B−A: {report['delta_b_minus_a']:+.4f}; changed positions={report['positions_changed']}")
    print(f"Owner contract: A={report['a']['owner_contract_pass']} B={report['b']['owner_contract_pass']}")
    print(f"Verdict: {report['verdict']}")
    if args.output:
        print(f"Detailed comparison: {args.output}")


if __name__ == "__main__":
    main()
