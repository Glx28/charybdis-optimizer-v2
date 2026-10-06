#!/usr/bin/env python3
"""Validate a candidate genome, install it as the project warmstart, and export it."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export import export_layout
from tools._common import load_evaluator
from tools.run_contract_audit import audit_genome


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def apply(candidate_path, cpu=False):
    candidate_path = Path(candidate_path).resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    genome = candidate.get("best_genome", candidate.get("genome"))
    if not isinstance(genome, list):
        raise ValueError("Candidate must contain a genome or best_genome list")

    evaluator = load_evaluator(require_cuda=not cpu, use_cuda=False if cpu else None,
                               generation=int(candidate.get("generation", 0)))
    layout = evaluator.reference_layout.clone_with(genome=np.asarray(genome, dtype=np.int32))
    audit = audit_genome(layout.genome, int(candidate.get("generation", 0)), evaluator)
    if not audit["run_contract_pass"]:
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise RuntimeError("Candidate failed layout contracts: " + ", ".join(failed))

    objectives, constraints = evaluator.evaluate_batch(layout.genome[None, :])
    raw_score = float(objectives[0].sum())
    if np.any(constraints[0] > 0):
        raise RuntimeError("Candidate has positive hard constraints")

    try:
        source_name = str(candidate_path.relative_to(ROOT))
    except ValueError:
        source_name = candidate_path.name
    source_name = candidate.get("source_candidate", source_name)
    warmstart = {
        "genome": layout.genome.tolist(),
        "best_score": raw_score,
        "accepted": True,
        "generation": int(candidate.get("generation", 0)),
        "best_generation": int(candidate.get("best_generation", candidate.get("generation", 0))),
        "source_candidate": source_name,
        "source_score": candidate.get(
            "source_score", candidate.get("total_score", candidate.get("best_score")),
        ),
        "source_audit": audit["checks"],
    }
    atomic_json(ROOT / "data/default_layout_genome.json", warmstart)
    build = ROOT / "build"
    warmstart_path = build / "v2_local_search_result.json"
    backup_path = build / "v2_local_search_result.pre_candidate_apply.json"
    if warmstart_path.exists() and not backup_path.exists():
        shutil.copy2(warmstart_path, backup_path)
    atomic_json(warmstart_path, warmstart)
    export_layout(layout, str(build / "current_layout.json"))
    atomic_json(build / "current_layout_audit.json", audit)
    print(f"Applied candidate: {candidate_path}")
    print(f"Warmstart: {warmstart_path}")
    print(f"Exported layout: {build / 'current_layout.json'}")
    print(f"CUDA/CPU audited score: {raw_score:.6f}; all run contracts pass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate", nargs="?",
        default="data/default_layout_genome.json",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU evaluator instead of requiring CUDA")
    args = parser.parse_args()
    apply(args.candidate, cpu=args.cpu)


if __name__ == "__main__":
    main()
