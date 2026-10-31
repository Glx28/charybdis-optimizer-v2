#!/usr/bin/env python3
"""Validate a candidate genome, install it as the project warmstart, and export it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export import export_layout
from evolution.layout_contract import CONTRACT_SHA256
from tools._common import load_evaluator
from tools.run_contract_audit import audit_genome


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_promotion_evidence(candidate, checkpoint_path, audit_path,
                              export_validation_path, csv_path, apply_js_path):
    checkpoint_path = Path(checkpoint_path).resolve()
    audit_path = Path(audit_path).resolve()
    export_validation_path = Path(export_validation_path).resolve()
    csv_path = Path(csv_path).resolve()
    apply_js_path = Path(apply_js_path).resolve()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    export_validation = json.loads(export_validation_path.read_text(encoding="utf-8"))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    genome = candidate.get("best_genome", candidate.get("genome"))
    if not isinstance(genome, list):
        raise ValueError("Candidate must contain a genome or best_genome list")
    checkpoint_hash = sha(checkpoint_path)
    audit_hash = sha(audit_path)
    checkpoint_genome = checkpoint.get("best_genome", checkpoint.get("genome"))
    owner_audit = audit.get("acceptance", {}).get("details", {}).get("owner_layout_contract_v1", {})
    checkpoint_contract_hash = checkpoint.get("contract_sha256") or checkpoint.get(
        "acceptance_report", {}).get("details", {}).get(
            "owner_layout_contract_v1", {}).get("contract_sha256")
    checks = {
        "candidate_matches_checkpoint": checkpoint_genome == np.asarray(genome, dtype=np.int32).tolist(),
        "candidate_checkpoint_hash_matches": candidate.get("checkpoint_sha256") == checkpoint_hash,
        "candidate_audit_hash_matches": candidate.get("audit_sha256") == audit_hash,
        "candidate_contract_hash_matches": candidate.get("contract_sha256") == CONTRACT_SHA256,
        "checkpoint_contract_matches": checkpoint_contract_hash == CONTRACT_SHA256,
        "audit_passes": bool(audit.get("run_contract_pass")),
        "audit_checkpoint_hash_matches": audit.get("checkpoint_sha256") == checkpoint_hash,
        "audit_generation_matches": audit.get("generation") == checkpoint.get("generation"),
        "audit_contract_hash_matches": (
            audit.get("contract_sha256") == CONTRACT_SHA256
            and owner_audit.get("contract_sha256") == CONTRACT_SHA256
        ),
        "export_validation_passes": bool(export_validation.get("acceptance_pass")),
        "export_validation_checks_pass": bool(export_validation.get("checks")) and all(
            export_validation["checks"].values()
        ),
        "export_checkpoint_hash_matches": export_validation.get("checkpoint_sha256") == checkpoint_hash,
        "export_contract_hash_matches": export_validation.get("contract_sha256") == CONTRACT_SHA256,
        "export_audit_hash_matches": export_validation.get("audit_sha256") == audit_hash,
        "candidate_exported_binding_hash_matches": (
            candidate.get("exported_binding_sha256") == export_validation.get("exported_binding_sha256")
            and len(candidate.get("exported_binding_sha256", "")) == 64
        ),
        "export_csv_hash_matches": export_validation.get("csv_sha256") == sha(csv_path),
        "export_studio_payload_hash_matches": export_validation.get("studio_apply_sha256") == sha(apply_js_path),
        "exported_binding_hash_present": len(export_validation.get("exported_binding_sha256", "")) == 64,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("Promotion evidence failed: " + ", ".join(failed))
    return checks


def apply(candidate_path, *, checkpoint_path, audit_path, export_validation_path,
          csv_path, apply_js_path, cpu=False):
    candidate_path = Path(candidate_path).resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    genome = candidate.get("best_genome", candidate.get("genome"))
    if not isinstance(genome, list):
        raise ValueError("Candidate must contain a genome or best_genome list")
    verify_promotion_evidence(
        candidate, checkpoint_path, audit_path, export_validation_path, csv_path, apply_js_path,
    )

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
    print(f"CUDA/CPU audited score: {raw_score:.6f}; all run contracts and promotion hashes pass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate", nargs="?",
        default="data/default_layout_genome.json",
    )
    parser.add_argument("--checkpoint", required=True, help="Exact checkpoint backing candidate")
    parser.add_argument("--audit", required=True, help="Contract audit for that checkpoint")
    parser.add_argument("--export-validation", required=True, help="Passing CSV/Studio binding audit")
    parser.add_argument("--csv", required=True, help="CSV whose hash is recorded in export validation")
    parser.add_argument("--apply-js", required=True, help="Studio payload whose hash is recorded in export validation")
    parser.add_argument("--cpu", action="store_true", help="Use CPU evaluator instead of requiring CUDA")
    args = parser.parse_args()
    apply(args.candidate, checkpoint_path=args.checkpoint, audit_path=args.audit,
          export_validation_path=args.export_validation, csv_path=args.csv,
          apply_js_path=args.apply_js, cpu=args.cpu)


if __name__ == "__main__":
    main()
