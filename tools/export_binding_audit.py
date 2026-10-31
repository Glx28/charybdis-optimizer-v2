#!/usr/bin/env python3
"""Compare corrected checkpoint assignments with exported CSV and Studio payload."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evolution.layout_contract import CONTRACT_SHA256
from tools._common import load_layout


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("apply_js", type=Path)
    parser.add_argument("audit", type=Path, help="run_contract_audit JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    genome = np.asarray(checkpoint["best_genome"], dtype=np.int32)
    layout = load_layout().clone_with(genome=genome)
    csv_rows = list(csv.DictReader(args.csv.open(encoding="utf-8", newline="")))
    csv_by_coord = {(int(r["layer"]), int(r["x"]), int(r["y"])): r for r in csv_rows}

    marker = "window.CHARYBDIS_FINAL_LAYOUT = "
    source = args.apply_js.read_text(encoding="utf-8")
    start = source.find(marker)
    end = source.find("\n};", start)
    if start < 0 or end < 0:
        raise ValueError("Studio apply script has no parseable CHARYBDIS_FINAL_LAYOUT payload")
    payload = json.loads(source[start + len(marker):end + 2])
    payload_by_coord = {
        (int(r["layer"]), int(r["x"]), int(r["y"])): r
        for r in payload.get("keys", [])
    }

    assignment_mismatches = []
    for idx, sid_value in enumerate(genome):
        pos = layout.positions[idx]
        sid = int(sid_value)
        if sid < 0 or int(pos.layer) in (0, 7):
            continue
        coord = (int(pos.layer), int(pos.x), int(pos.y))
        row = csv_by_coord.get(coord)
        shortcut = layout.shortcuts[sid]
        expected = str(shortcut.keys)
        label_matches = bool(row and (
            expected in row["purpose"] if shortcut.is_layer_access
            else row["visual_label"] == expected
            or (row["visual_label"].startswith("AltGr+")
                and "Ctrl" in shortcut.modifiers and "Alt" in shortcut.modifiers
                and row["modifiers"] == "R Alt")
        ))
        if row is None or not label_matches or row["source"] != "evolved":
            assignment_mismatches.append({
                "coordinate": coord,
                "expected_shortcut": expected,
                "csv_label": row.get("visual_label") if row else None,
                "csv_source": row.get("source") if row else None,
            })

    payload_mismatches = []
    fields = (("visual_label", "label"), ("behavior", "behavior"),
              ("parameter", "parameter"), ("modifiers", "modifiers"))
    for coord, row in csv_by_coord.items():
        item = payload_by_coord.get(coord)
        for csv_field, payload_field in fields:
            expected = row[csv_field]
            actual = item.get(payload_field) if item else None
            if csv_field == "modifiers":
                actual = "+".join(actual or []) if isinstance(actual, list) else str(actual or "")
            if csv_field == "parameter":
                expected = str(expected or "").removeprefix("Keyboard ")
                actual = str(actual or "").removeprefix("Keyboard ")
                expected = {
                    "Shift": "LeftShift",
                    "9 and PageUp": "Keypad 9 and PageUp",
                    "3 and PageDn": "Keypad 3 and PageDn",
                }.get(expected, expected)
            if str(actual or "") != str(expected or ""):
                payload_mismatches.append({
                    "coordinate": coord, "field": csv_field,
                    "csv": expected, "studio_payload": actual,
                })
                break

    contract_hash = checkpoint.get("contract_sha256") or checkpoint.get(
        "acceptance_report", {}).get("details", {}).get(
            "owner_layout_contract_v1", {}).get("contract_sha256")
    audit_data = json.loads(args.audit.read_text(encoding="utf-8"))
    owner_audit = audit_data.get("acceptance", {}).get("details", {}).get(
        "owner_layout_contract_v1", {}
    )
    normalized_payload = [
        {"coordinate": coord,
         "label": payload_by_coord[coord].get("label", ""),
         "behavior": payload_by_coord[coord].get("behavior", ""),
         "parameter": payload_by_coord[coord].get("parameter", ""),
         "modifiers": payload_by_coord[coord].get("modifiers", [])}
        for coord in sorted(payload_by_coord)
    ]
    binding_bytes = json.dumps(normalized_payload, sort_keys=True,
                               separators=(",", ":")).encode("utf-8")
    checks = {
        "contract_hash_matches": contract_hash == CONTRACT_SHA256,
        "audit_contract_hash_matches": owner_audit.get("contract_sha256") == CONTRACT_SHA256,
        "audit_top_level_contract_hash_matches": audit_data.get("contract_sha256") == CONTRACT_SHA256,
        "audit_matches_checkpoint_hash": audit_data.get("checkpoint_sha256") == sha256(args.checkpoint),
        "audit_matches_checkpoint_generation": audit_data.get("generation") == checkpoint.get("generation"),
        "audit_passes": bool(audit_data.get("run_contract_pass")),
        "csv_has_616_unique_positions": len(csv_rows) == 616 and len(csv_by_coord) == 616,
        "studio_payload_has_616_unique_positions": len(payload_by_coord) == 616,
        "checkpoint_assignments_match_csv": not assignment_mismatches,
        "csv_matches_studio_payload": not payload_mismatches,
    }
    result = {
        "checks": checks,
        "acceptance_pass": all(checks.values()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "genome_sha256": hashlib.sha256(json.dumps(genome.astype(int).tolist(),
                                                     separators=(",", ":")).encode()).hexdigest(),
        "contract_sha256": CONTRACT_SHA256,
        "audit_sha256": sha256(args.audit),
        "exported_binding_sha256": hashlib.sha256(binding_bytes).hexdigest(),
        "csv_sha256": sha256(args.csv),
        "studio_apply_sha256": sha256(args.apply_js),
        "assignment_mismatches": assignment_mismatches,
        "payload_mismatches": payload_mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"acceptance_pass": result["acceptance_pass"], "checks": checks,
                      "assignment_mismatch_count": len(assignment_mismatches),
                      "payload_mismatch_count": len(payload_mismatches)}, indent=2))
    return 0 if result["acceptance_pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
