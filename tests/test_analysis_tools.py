"""Smoke tests for the shared analysis infrastructure and tools."""
import os
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
