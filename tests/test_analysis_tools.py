"""Smoke tests for the shared analysis infrastructure and tools."""
import json
import os
import shutil
import subprocess
import sys

# Ensure repo root and tools/ are importable.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pytest

from tools._common import (
    checkpoint_to_layout,
    compute_reachability,
    find_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    load_evaluator,
    load_layout,
)
from tools.compare_checkpoints import _comparison_verdict


def test_aggregate_usage_preserves_layer_symbols_and_shortcut_sequences(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for usage aggregation")

    usage_log = tmp_path / "shortcut_usage.jsonl"
    output_path = tmp_path / "usage_stats.json"
    events = [
        {"ts": "2026-10-29T10:00:00Z", "type": "shortcut", "keys": "^c",
         "app": "Code.exe", "layer": "0", "count": 1},
        {"ts": "2026-10-29T10:00:00.120Z", "type": "shortcut", "keys": "^v",
         "prev": "^c", "prev_layer": "0", "gap_ms": 120, "app": "Code.exe", "layer": "1", "count": 1},
        {"ts": "2026-10-29T10:00:00.200Z", "type": "norwegian_symbol_key",
         "physical_key": "2", "produced_output": "@", "modifier_state": "altgr",
         "app": "Code.exe", "layer": "1"},
        {"ts": "2026-10-29T10:00:00.300Z", "type": "norwegian_symbol_key",
         "physical_key": "SC056", "produced_output": "<", "modifier_state": "shift",
         "app": "Code.exe", "layer": "10"},
        {"ts": "2026-10-29T10:00:00.400Z", "type": "norwegian_symbol_key",
         "physical_key": "SC01A", "produced_output": "å", "modifier_state": "normal",
         "app": "Code.exe", "layer": "9"},
        {"ts": "2026-10-29T10:00:00.500Z", "type": "norwegian_symbol_key",
         "physical_key": "SC027", "produced_output": "ø", "modifier_state": "normal",
         "app": "Code.exe", "layer": "1"},
        {"ts": "2026-10-29T10:00:00.600Z", "type": "norwegian_symbol_key",
         "physical_key": "SC028", "produced_output": "æ", "modifier_state": "normal",
         "app": "Code.exe", "layer": "2"},
        {"ts": "2026-10-29T10:00:00.700Z", "type": "norwegian_symbol_key",
         "physical_key": "SC00C", "produced_output": "+", "modifier_state": "normal",
         "app": "Code.exe", "layer": "5"},
    ]
    usage_log.write_text(
        "\n".join(json.dumps(event) for event in events)
        + "\n\x00\x00\x00" + json.dumps({
            "ts": "2026-10-29T10:00:00.800Z", "type": "shortcut", "keys": "^s",
            "app": "Code.exe", "layer": "1", "count": 1,
        })
        + "\n\x00truncated\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["CHARYBDIS_USAGE_LOG"] = str(usage_log)
    env["CHARYBDIS_USAGE_STATS_OUT"] = str(output_path)
    result = subprocess.run(
        [node, "pipeline/aggregate_usage.js"], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, check=True,
    )
    assert "Usage aggregation:" in result.stdout
    assert "9 events" in result.stdout
    assert "1 malformed JSON event line(s) skipped" in result.stderr
    assert "1 NUL-padded JSON event line(s) recovered" in result.stderr
    stats = json.loads(output_path.read_text(encoding="utf-8"))
    assert stats["total_events"] == 9
    assert stats["malformed_event_lines"] == 1
    assert stats["repaired_padded_event_lines"] == 1

    assert stats["by_layer_shortcut"]["Ctrl+C"] == {"0": 1}
    assert stats["by_layer_shortcut"]["Ctrl+V"] == {"1": 1}
    assert stats["by_layer_shortcut"]["Ctrl+S"] == {"1": 1}
    assert stats["shortcut_sequences"]["Ctrl+C -> Ctrl+V"]["count"] == 1
    assert stats["shortcut_sequences"]["Ctrl+C -> Ctrl+V"]["by_layer_pair"] == {"0->1": 1}
    assert stats["chains"]["Ctrl+C -> Ctrl+V"]["by_layer_path"] == {"0->1": 1}
    assert stats["norwegian_symbol_usage"]["by_output"]["@"]["by_layer"] == {"1": 1}
    assert stats["norwegian_symbol_usage"]["by_output"]["@"]["physical_keys"] == {"2": 1}
    assert stats["norwegian_symbol_usage"]["by_output"]["<"]["by_layer"] == {"10": 1}
    assert stats["norwegian_symbol_usage"]["by_output"]["å"]["by_layer"] == {"9": 1}
    assert stats["norwegian_symbol_usage"]["by_output"]["ø"]["by_layer"] == {"1": 1}
    assert stats["norwegian_symbol_usage"]["by_output"]["æ"]["by_layer"] == {"2": 1}
    assert stats["norwegian_symbol_usage"]["by_output"]["+"]["physical_keys"] == {"SC00C": 1}


def test_compare_checkpoint_verdict_treats_lower_penalty_as_better():
    assert _comparison_verdict(-10.0, True, True) == (
        "B has lower (better) exact score; both pass"
    )
    assert _comparison_verdict(10.0, True, True) == (
        "B has higher (worse) exact score; both pass"
    )


def test_load_layout_returns_layout():
    layout = load_layout()
    assert hasattr(layout, "genome")
    assert hasattr(layout, "n_positions")
    assert layout.n_positions > 0


def test_load_evaluator_returns_evaluator():
    ev = load_evaluator()
    assert ev is not None
    assert ev.model is not None
    assert ev.model.arrays is not None


def test_find_checkpoints_returns_sorted_list():
    paths = find_checkpoints()
    assert isinstance(paths, list)
    if paths:
        assert all(p.endswith(".json") for p in paths)
        gens = [int(os.path.basename(p).split("gen")[-1].split(".")[0]) for p in paths]
        assert gens == sorted(gens)


def test_find_latest_checkpoint_returns_path_or_none():
    latest = find_latest_checkpoint()
    assert latest is None or latest.endswith(".json")


def test_load_checkpoint_returns_dict():
    latest = find_latest_checkpoint()
    if latest is None:
        pytest.skip("no checkpoints available")
    ckpt = load_checkpoint(latest)
    assert isinstance(ckpt, dict)
    assert "best_genome" in ckpt


def test_checkpoint_to_layout_matches_genome():
    layout = load_layout()
    latest = find_latest_checkpoint()
    if latest is None:
        pytest.skip("no checkpoints available")
    ckpt = load_checkpoint(latest)
    laid_out = checkpoint_to_layout(ckpt, layout)
    assert list(laid_out.genome) == ckpt["best_genome"]


def test_compute_reachability_returns_expected_keys():
    layout = load_layout()
    ev = load_evaluator()
    latest = find_latest_checkpoint()
    if latest is None:
        pytest.skip("no checkpoints available")
    ckpt = load_checkpoint(latest)
    laid_out = checkpoint_to_layout(ckpt, layout)
    r = compute_reachability(laid_out.genome, laid_out, arrays=ev.model.arrays)
    assert set(r.keys()) >= {"all_hops", "hold_hops", "toggle_hops", "reachable_layers"}
    assert 0 in r["reachable_layers"]


def _run_tool(args):
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def test_tool_best_runs():
    result = _run_tool(["tools/best.py"])
    assert result.returncode == 0


def test_tool_check_runs():
    result = _run_tool(["tools/check.py"])
    assert result.returncode == 0


def test_tool_human_audit_runs():
    result = _run_tool(["tools/human_audit.py"])
    assert result.returncode == 0


def test_tool_workflow_check_runs():
    result = _run_tool(["tools/workflow_check.py"])
    assert result.returncode == 0


def test_tool_run_report_runs():
    result = _run_tool(["tools/run_report.py", "build"])
    assert result.returncode == 0
    assert "Run Report" in result.stdout or "checkpoints" in result.stdout.lower()


def test_tool_checkpoint_audit_runs():
    result = _run_tool(["tools/checkpoint_audit.py", "latest"])
    assert result.returncode == 0
    assert "Checkpoint Audit" in result.stdout


def test_tool_layer_profile_runs():
    result = _run_tool(["tools/layer_profile.py", "latest"])
    assert result.returncode == 0
    assert "Layer Profile" in result.stdout


def test_tool_shortcut_audit_runs():
    result = _run_tool(["tools/shortcut_audit.py", "latest", "--top", "5"])
    assert result.returncode == 0
    assert "Shortcut Audit" in result.stdout


def test_tool_mouse_layer_report_runs():
    result = _run_tool(["tools/mouse_layer_report.py", "latest"])
    assert result.returncode == 0
    assert "Mouse Layer Report" in result.stdout


def test_tool_arrow_cluster_report_runs():
    result = _run_tool(["tools/arrow_cluster_report.py", "latest"])
    assert result.returncode == 0
    assert "Arrow Cluster Report" in result.stdout


def test_tool_completion_cluster_report_runs():
    result = _run_tool(["tools/completion_cluster_report.py", "latest"])
    assert result.returncode == 0
    assert "Completion Cluster Report" in result.stdout


def test_tool_constraint_trace_runs():
    result = _run_tool(["tools/constraint_trace.py", "build"])
    assert result.returncode == 0
    assert "Constraint Trace" in result.stdout


def test_tool_compare_checkpoints_runs():
    files = find_checkpoints()
    if len(files) < 2:
        pytest.skip("need at least two checkpoints")
    result = _run_tool(["tools/compare_checkpoints.py", files[0], files[-1]])
    assert result.returncode == 0
    assert "Compare Checkpoints" in result.stdout


def test_tool_generate_run_report_runs():
    result = _run_tool(["tools/generate_run_report.py", "build"])
    assert result.returncode == 0
    assert "Run report bundle written to" in result.stdout
