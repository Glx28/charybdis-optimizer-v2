import hashlib
import json

import pytest

from evolution.layout_contract import CONTRACT_SHA256
from tools.apply_layout import verify_promotion_evidence


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("top_level_contract", [True, False])
def test_promotion_requires_matching_checkpoint_contract_audit_and_export_hashes(tmp_path, top_level_contract):
    checkpoint_path = tmp_path / "checkpoint.json"
    audit_path = tmp_path / "audit.json"
    export_path = tmp_path / "export-validation.json"
    csv_path = tmp_path / "layout.csv"
    js_path = tmp_path / "apply.js"
    checkpoint = {
        "generation": 5000,
        "best_genome": [1, 2, 3],
    }
    if top_level_contract:
        checkpoint["contract_sha256"] = CONTRACT_SHA256
    else:
        checkpoint["acceptance_report"] = {"details": {"owner_layout_contract_v1": {
            "contract_sha256": CONTRACT_SHA256,
        }}}
    _write_json(checkpoint_path, checkpoint)
    _write_json(audit_path, {
        "generation": 5000,
        "checkpoint_sha256": _sha(checkpoint_path),
        "contract_sha256": CONTRACT_SHA256,
        "run_contract_pass": True,
        "acceptance": {"details": {"owner_layout_contract_v1": {
            "contract_sha256": CONTRACT_SHA256,
        }}},
    })
    csv_path.write_text("layer,x,y\n1,8,2\n", encoding="utf-8")
    js_path.write_text("window.CHARYBDIS_FINAL_LAYOUT = {};\n", encoding="utf-8")
    _write_json(export_path, {
        "acceptance_pass": True,
        "checks": {"all": True},
        "checkpoint_sha256": _sha(checkpoint_path),
        "contract_sha256": CONTRACT_SHA256,
        "audit_sha256": _sha(audit_path),
        "csv_sha256": _sha(csv_path),
        "studio_apply_sha256": _sha(js_path),
        "exported_binding_sha256": "a" * 64,
    })
    candidate = {
        "genome": [1, 2, 3],
        "checkpoint_sha256": _sha(checkpoint_path),
        "audit_sha256": _sha(audit_path),
        "contract_sha256": CONTRACT_SHA256,
        "exported_binding_sha256": "a" * 64,
    }

    checks = verify_promotion_evidence(
        candidate, checkpoint_path, audit_path, export_path, csv_path, js_path,
    )
    assert all(checks.values())

    _write_json(audit_path, {"run_contract_pass": False})
    with pytest.raises(RuntimeError, match="hash_matches"):
        verify_promotion_evidence(
            candidate, checkpoint_path, audit_path, export_path, csv_path, js_path,
        )
