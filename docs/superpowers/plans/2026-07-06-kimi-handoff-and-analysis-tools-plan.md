# Kimi Handoff + Token-Saving Analysis Tools — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Kimi-optimized handoff and a suite of one-command analysis tools that eliminate repetitive loader boilerplate and give instant answers about runs, layers, shortcuts, mouse/arrow/Norwegian clusters, and constraints.

**Architecture:** A shared `tools/_common.py` module centralizes layout/evaluator/checkpoint loading and reachability. Existing tools are refactored to use it. New focused CLI reports read checkpoints via `_common.py` and output markdown/text summaries. `just` recipes wrap the most common invocations. Handoff docs (`KIMI.md`, `.kimi-code/config.toml`, hook) tell future Kimi sessions exactly how to use everything.

**Tech Stack:** Python 3.12, existing repo venv, NumPy, existing `core`, `fitness`, `evolution` modules. No new dependencies.

## Global Constraints

- No new Python dependencies.
- GPU-primary training policy must remain intact; do not add processor-primary training escape hatches.
- Existing tests must still pass after refactor.
- All new files live inside `/home/nos/charybdis/charybdis-optimizer-v2`.
- All file paths are exact and relative to repo root.
- Minimal diffs; do not rewrite unrelated logic.
- Follow existing naming and output conventions in `tools/`.

---

## File Structure

### New files
- `.kimi-code/config.toml` — Kimi project-level config (model, hooks, instructions reference).
- `.kimi-code/hooks/block-dangerous-bash.py` — Python destructive-command guard (more portable than Node).
- `tools/_common.py` — Shared loader/evaluator/checkpoint/reachability helpers.
- `tools/run_report.py` — Run-level summary and stagnation diagnosis.
- `tools/checkpoint_audit.py` — One-page checkpoint audit.
- `tools/layer_profile.py` — Per-layer role profile.
- `tools/shortcut_audit.py` — Misplaced/high-value shortcut audit.
- `tools/mouse_layer_report.py` — Mouse-layer focused report.
- `tools/arrow_cluster_report.py` — Arrow-cluster focused report.
- `tools/completion_cluster_report.py` — Norwegian completion-cluster report.
- `tools/constraint_trace.py` — Constraint trends across checkpoints.
- `tools/compare_checkpoints.py` — Diff two checkpoints.
- `tools/generate_run_report.py` — Orchestrator that writes `build/run_report/`.
- `.agent-tools/quick-context.sh` — Generates minimal context pack.

### Modified files
- `KIMI.md` — Rewritten as full Kimi handoff.
- `AGENTS.md` — Add cross-reference to `KIMI.md`.
- `justfile` — Add new recipes.
- `tools/best.py` — Use `tools/_common.py`.
- `tools/check.py` — Use `tools/_common.py`.
- `tools/human_audit.py` — Use `tools/_common.py`.
- `tools/workflow_check.py` — Use `tools/_common.py`.
- `.ai/context/agent-tooling-research.md` — Document new tooling.
- `tests/test_analysis_tools.py` — Smoke tests for new tools.

---

## Task 1: Shared Analysis Infrastructure — `tools/_common.py`

**Files:**
- Create: `tools/_common.py`
- Test: `tests/test_analysis_tools.py` (initial tests)

**Interfaces:**
- Produces:
  - `load_layout(data_dir="data") -> Layout`
  - `load_evaluator(config_path="config_v2.yaml", data_dir="data", build_dir="build") -> FitnessEvaluator`
  - `load_checkpoint(path: str) -> dict`
  - `checkpoint_to_layout(checkpoint: dict, layout: Layout) -> Layout`
  - `find_latest_checkpoint(build_dir="build") -> str`
  - `find_checkpoints(build_dir="build") -> list[str]`
  - `compute_reachability(genome, layout) -> dict`
  - `label_shortcut(sid: int, layout, arrays) -> dict`
  - `layer_inventory(layout) -> list[dict]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_analysis_tools.py`:

```python
import os
import pytest
from tools._common import (
    load_layout,
    load_checkpoint,
    checkpoint_to_layout,
    find_latest_checkpoint,
    find_checkpoints,
    compute_reachability,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_load_layout_returns_layout():
    layout = load_layout()
    assert hasattr(layout, "genome")
    assert hasattr(layout, "n_positions")


def test_find_checkpoints_returns_sorted_list():
    paths = find_checkpoints()
    assert isinstance(paths, list)
    if paths:
        assert all(p.endswith(".json") for p in paths)
        assert paths == sorted(paths, key=lambda p: int(os.path.basename(p).split("gen")[-1].split(".")[0]))


def test_find_latest_checkpoint_returns_one_path_or_none():
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
    latest = find_latest_checkpoint()
    if latest is None:
        pytest.skip("no checkpoints available")
    ckpt = load_checkpoint(latest)
    laid_out = checkpoint_to_layout(ckpt, layout)
    r = compute_reachability(laid_out.genome, layout)
    assert set(r.keys()) >= {"all_hops", "hold_hops", "toggle_hops", "reachable_layers"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/pytest tests/test_analysis_tools.py -v
```

Expected: failures because `tools/_common.py` does not exist.

- [ ] **Step 3: Implement `tools/_common.py`**

Create `tools/_common.py`:

```python
"""Shared helpers for Charybdis analysis tools."""
import glob
import json
import os
import sys

import numpy as np

# Ensure repo root is importable when scripts are run directly.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.loader import load_data, load_usage_stats
from fitness.evaluator import FitnessEvaluator


def load_layout(data_dir="data"):
    """Load the base Layout object from data_dir."""
    layout, _arrays, _usage = load_data(data_dir=data_dir)
    return layout


def load_evaluator(config_path="config_v2.yaml", data_dir="data", build_dir="build"):
    """Load a FitnessEvaluator using the production config."""
    # Import config loader lazily to avoid circular imports.
    from config import get_config

    config = get_config(config_path)
    layout = load_layout(data_dir=data_dir)
    usage = load_usage_stats(data_dir=data_dir)
    return FitnessEvaluator(
        layout=layout,
        usage_stats=usage,
        config=config,
        build_dir=build_dir,
        hard_constraints=config.get("fitness.hard_constraints", []),
    )


def load_checkpoint(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_to_layout(checkpoint: dict, layout) -> object:
    """Return a Layout clone with the checkpoint's best genome."""
    return layout.clone_with(genome=np.asarray(checkpoint["best_genome"], dtype=np.int32))


def _checkpoint_generation(path: str) -> int:
    base = os.path.basename(path)
    # v2_checkpoint_gen<N>.json
    try:
        return int(base.split("gen")[-1].split(".")[0])
    except ValueError:
        return -1


def find_checkpoints(build_dir="build") -> list:
    pattern = os.path.join(build_dir, "v2_checkpoint_gen*.json")
    paths = glob.glob(pattern)
    return sorted(paths, key=_checkpoint_generation)


def find_latest_checkpoint(build_dir="build"):
    paths = find_checkpoints(build_dir=build_dir)
    return paths[-1] if paths else None


def compute_reachability(genome, layout):
    """Build a reachability summary for a genome."""
    from fitness.kernel import build_layer_reachability

    all_hops, hold_hops, toggle_hops = build_layer_reachability(genome, layout)
    reachable_layers = set(int(layer) for layer in all_hops.keys())
    return {
        "all_hops": all_hops,
        "hold_hops": hold_hops,
        "toggle_hops": toggle_hops,
        "reachable_layers": reachable_layers,
    }


def label_shortcut(sid: int, layout, arrays) -> dict:
    """Return a human-readable label dict for shortcut sid."""
    if sid < 0 or sid >= len(arrays.shortcuts):
        return {"name": "(empty)", "app": None, "category": None, "importance": 0.0, "usage": 0.0}
    sc = arrays.shortcuts[sid]
    return {
        "name": sc.name,
        "app": getattr(sc, "app", None),
        "category": getattr(sc, "category", None),
        "importance": float(getattr(sc, "importance", 0.0)),
        "usage": float(getattr(sc, "usage", 0.0)),
    }


def layer_inventory(layout) -> list:
    """Return a list of dicts, one per layer, with basic metadata."""
    inv = []
    for layer_idx in range(layout.n_layers):
        layer = layout.layers[layer_idx]
        inv.append({
            "index": layer_idx,
            "name": getattr(layer, "name", f"L{layer_idx}"),
            "n_positions": len(layer.positions),
        })
    return inv
```

Note: verify `build_layer_reachability` signature in `fitness/kernel.py` before finalizing; adjust imports if it lives elsewhere.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/pytest tests/test_analysis_tools.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/_common.py tests/test_analysis_tools.py
git commit -m "feat(tools): shared analysis infrastructure"
```

---

## Task 2: Refactor Existing Tools to Use `_common.py`

**Files:**
- Modify: `tools/best.py`, `tools/check.py`, `tools/human_audit.py`, `tools/workflow_check.py`

**Interfaces:**
- Consumes: `tools/_common.load_layout`, `load_evaluator`, `load_checkpoint`, `checkpoint_to_layout`, `find_latest_checkpoint`, `find_checkpoints`.
- Produces: same CLI behavior; only duplicated setup removed.

- [ ] **Step 1: Refactor `tools/best.py`**

Replace the top-level loader/evaluator setup with imports from `tools._common`. Keep command-line interface and output identical.

Example change near top of file:

```python
from tools._common import load_evaluator, find_checkpoints, load_checkpoint, checkpoint_to_layout
```

Then remove duplicated `load_data`, `FitnessEvaluator`, and `glob` setup.

- [ ] **Step 2: Refactor `tools/check.py`**

Same pattern: import `load_layout`, `load_evaluator`, `find_latest_checkpoint`, `load_checkpoint`, `checkpoint_to_layout` from `tools._common`. Remove duplicated setup. Keep output.

- [ ] **Step 3: Refactor `tools/human_audit.py`**

Same pattern.

- [ ] **Step 4: Refactor `tools/workflow_check.py`**

Same pattern.

- [ ] **Step 5: Run smoke tests on each tool**

Run:

```bash
.venv/bin/python tools/best.py
.venv/bin/python tools/check.py
.venv/bin/python tools/human_audit.py
.venv/bin/python tools/workflow_check.py
```

Expected: all exit 0 and produce sensible output.

- [ ] **Step 6: Run full test suite**

Run:

```bash
just ai-smoke
```

Expected: passes.

- [ ] **Step 7: Commit**

```bash
git add tools/best.py tools/check.py tools/human_audit.py tools/workflow_check.py
git commit -m "refactor(tools): use shared _common helpers"
```

---

## Task 3: New Report Tools (Batch 1) — Run + Checkpoint Audit

**Files:**
- Create: `tools/run_report.py`, `tools/checkpoint_audit.py`
- Test: `tests/test_analysis_tools.py` (add tests)

**Interfaces:**
- Consumes: `tools._common`.
- Produces: stdout markdown/text reports; `tools/run_report.py --write` creates `build/run_report.md`.

- [ ] **Step 1: Implement `tools/run_report.py`**

CLI: `python3 tools/run_report.py [build_dir] [--write]`

Behavior:
1. Find all checkpoints in `build_dir` (default `build`).
2. Load each and extract `total_score`, `optimizer_side_pass`, `generation`, `best_constraints`, `population_best_constraints`.
3. Compute gap = total_score + 49.30.
4. Find best generation (lowest total_score where constraints all zero and `optimizer_side_pass` is True, or lowest total_score overall as fallback).
5. Count diversity-injection events by scanning `build_dir/runs/*/run.log` for `Diversity injection trigger`.
6. If `--write`, write markdown to `build/run_report.md`.

Output sections:
- Run summary (build dir, checkpoints found, best gen, best gap).
- Score trajectory table (gen, gap, total_score, pass/fail).
- Stagnation diagnosis (gens since last improvement, likely cause).
- Constraint trace summary (which constraints are violated in latest checkpoint).
- Surrogate health if info available.

- [ ] **Step 2: Implement `tools/checkpoint_audit.py`**

CLI: `python3 tools/checkpoint_audit.py <checkpoint_path>`

Behavior:
1. Load checkpoint via `_common.load_checkpoint`.
2. Build layout via `_common.checkpoint_to_layout`.
3. Run acceptance report using `evolution.acceptance.build_acceptance_report`.
4. Print one-page audit with:
   - Generation, gap, total score, objectives, constraints.
   - Acceptance status and failed checks.
   - Mouse layer present?.
   - Arrow cluster OK?.
   - Norwegian completion cluster OK?.
   - L7 access OK?.
   - Top 5 empty prime positions (lowest effort mutable positions with sid == -1).

- [ ] **Step 3: Add smoke tests**

Append to `tests/test_analysis_tools.py`:

```python
def test_run_report_runs():
    import subprocess
    result = subprocess.run([sys.executable, "tools/run_report.py", "build"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Run summary" in result.stdout or "checkpoints" in result.stdout.lower()


def test_checkpoint_audit_runs():
    import subprocess, sys
    latest = find_latest_checkpoint()
    if latest is None:
        pytest.skip("no checkpoints available")
    result = subprocess.run([sys.executable, "tools/checkpoint_audit.py", latest], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Generation" in result.stdout
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_analysis_tools.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tools/run_report.py tools/checkpoint_audit.py tests/test_analysis_tools.py
git commit -m "feat(tools): run summary and checkpoint audit reports"
```

---

## Task 4: New Report Tools (Batch 2) — Layer + Shortcut + Mouse

**Files:**
- Create: `tools/layer_profile.py`, `tools/shortcut_audit.py`, `tools/mouse_layer_report.py`
- Test: `tests/test_analysis_tools.py` (add tests)

**Interfaces:**
- Consumes: `tools._common`, `evolution.acceptance`, `evolution.arrow_cluster`, `evolution.completion_cluster`.
- Produces: stdout markdown/text reports.

- [ ] **Step 1: Implement `tools/layer_profile.py`**

CLI: `python3 tools/layer_profile.py <checkpoint_path>`

Output one markdown table row per layer with:
- Layer index/name.
- Access cost (best hold/toggle hop count to reach layer).
- Reachable (yes/no).
- Shortcut count (non-empty mutable positions).
- Top 3 apps by shortcut count.
- Top 3 categories.
- Mouse buttons present.
- Arrow keys present.
- Norwegian keys present.
- Empty slot count.
- Dominant workflow (app with highest total usage on layer).

- [ ] **Step 2: Implement `tools/shortcut_audit.py`**

CLI: `python3 tools/shortcut_audit.py <checkpoint_path> [--top N]`

Behavior:
1. Load layout and arrays.
2. Iterate all non-empty mutable positions.
3. For each shortcut, collect: effort, layer, access hops, importance, usage, duplicate status.
4. Flag:
   - `unreachable` if layer not reachable.
   - `unsupported_duplicate` if duplicate is not workflow-supported.
   - `misplaced` if importance * usage / effort is in bottom quartile for that app.
5. Sort by severity: infeasible first, then by (importance * usage) descending.
6. Print top N (default 30).

- [ ] **Step 3: Implement `tools/mouse_layer_report.py`**

CLI: `python3 tools/mouse_layer_report.py <checkpoint_path>`

Behavior:
1. Build layout and acceptance report.
2. Find layer(s) containing MB1–MB5.
3. For candidate layer(s), report:
   - Layer index.
   - MB1–MB5 positions (row/col).
   - Right-hand non-thumb scroll capability.
   - Hold-hop depth from L0.
   - Any right-thumb position occupied by a mouse button.
   - Toggle access reachable?.
4. Print acceptance `dynamic_mouse_layer_present` details.

- [ ] **Step 4: Add smoke tests**

Append subprocess tests for each tool, skipping if no checkpoints.

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_analysis_tools.py -v
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add tools/layer_profile.py tools/shortcut_audit.py tools/mouse_layer_report.py tests/test_analysis_tools.py
git commit -m "feat(tools): layer, shortcut, and mouse-layer reports"
```

---

## Task 5: New Report Tools (Batch 3) — Arrow + Norwegian + Constraints + Diff

**Files:**
- Create: `tools/arrow_cluster_report.py`, `tools/completion_cluster_report.py`, `tools/constraint_trace.py`, `tools/compare_checkpoints.py`
- Test: `tests/test_analysis_tools.py` (add tests)

**Interfaces:**
- Consumes: `tools._common`, `evolution.arrow_cluster`, `evolution.completion_cluster`, `evolution.acceptance`.
- Produces: stdout markdown/text reports; `tools/constraint_trace.py --write` creates `build/constraint_trace.md`.

- [ ] **Step 1: Implement `tools/arrow_cluster_report.py`**

CLI: `python3 tools/arrow_cluster_report.py <checkpoint_path>`

Behavior:
1. Run `evolution.arrow_cluster.analyze_arrows` on the layout.
2. Report non-L7 arrow placements, shape compliance, and scattered penalty contributors.
3. Print acceptance `mutable_raw_arrows_ok` status.

- [ ] **Step 2: Implement `tools/completion_cluster_report.py`**

CLI: `python3 tools/completion_cluster_report.py <checkpoint_path>`

Behavior:
1. Run `evolution.completion_cluster.analyze_completion_cluster`.
2. Report anchor layer, present/missing keys, compactness, ordered-left-to-right flag.
3. Print acceptance `norwegian_completion_cluster` status.

- [ ] **Step 3: Implement `tools/constraint_trace.py`**

CLI: `python3 tools/constraint_trace.py [build_dir] [--write]`

Behavior:
1. Load all checkpoints.
2. Build markdown table: generation, archive constraints (each), population best constraints (each).
3. If `--write`, save to `build/constraint_trace.md`.

- [ ] **Step 4: Implement `tools/compare_checkpoints.py`**

CLI: `python3 tools/compare_checkpoints.py <ckpt_a> <ckpt_b>`

Behavior:
1. Load both checkpoints.
2. Compare total score, gap, objectives, constraints, acceptance.
3. Count positions changed between genomes.
4. Print whether `b` is progression/regression/neutral with reasons.

- [ ] **Step 5: Add smoke tests and run**

Append subprocess tests; run `pytest tests/test_analysis_tools.py -v`.

- [ ] **Step 6: Commit**

```bash
git add tools/arrow_cluster_report.py tools/completion_cluster_report.py tools/constraint_trace.py tools/compare_checkpoints.py tests/test_analysis_tools.py
git commit -m "feat(tools): arrow, completion, constraint trace, and checkpoint diff reports"
```

---

## Task 6: Report Orchestrator — `tools/generate_run_report.py`

**Files:**
- Create: `tools/generate_run_report.py`

**Interfaces:**
- Consumes: all tools created above.
- Produces: `build/run_report/` directory with timestamped reports.

- [ ] **Step 1: Implement `tools/generate_run_report.py`**

CLI: `python3 tools/generate_run_report.py [build_dir]`

Behavior:
1. Determine output directory: `build/run_report/YYYY-MM-DD_HH-MM-SS/`.
2. Run and capture output from:
   - `tools/run_report.py [build_dir] --write` (writes `build/run_report.md`, also copy into output dir).
   - `tools/checkpoint_audit.py <latest>` → `checkpoint_audit.md`.
   - `tools/layer_profile.py <latest>` → `layer_profile.md`.
   - `tools/shortcut_audit.py <latest>` → `shortcut_audit.md`.
   - `tools/mouse_layer_report.py <latest>` → `mouse_layer_report.md`.
   - `tools/arrow_cluster_report.py <latest>` → `arrow_cluster_report.md`.
   - `tools/completion_cluster_report.py <latest>` → `completion_cluster_report.md`.
   - `tools/constraint_trace.py [build_dir] --write` → `constraint_trace.md`.
3. Also write a `README.md` in the output dir listing the files.
4. Print the output directory path at the end.

- [ ] **Step 2: Add smoke test**

Append subprocess test that runs `tools/generate_run_report.py build` and checks exit code 0 and that `build/run_report/` contains files.

- [ ] **Step 3: Run tests**

```bash
.venv/bin/pytest tests/test_analysis_tools.py -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add tools/generate_run_report.py tests/test_analysis_tools.py
git commit -m "feat(tools): orchestrated run report generator"
```

---

## Task 7: Kimi Handoff Docs and Config

**Files:**
- Create: `.kimi-code/config.toml`, `.kimi-code/hooks/block-dangerous-bash.py`
- Modify: `KIMI.md`, `AGENTS.md`

**Interfaces:**
- Produces: project-level Kimi instructions and destructive-command guard.

- [ ] **Step 1: Rewrite `KIMI.md`**

Replace contents with:

```markdown
# Kimi Handoff — Charybdis Optimizer V2

GPU-primary evolutionary keyboard-layout optimizer. Read `AGENTS.md` first; this file adds Kimi-specific commands.

## Startup Checklist

1. `just ai-status` — repo state, tools, skills, MCP.
2. `just ai-context` — regenerate repo map and context.
3. Read `AGENTS.md` for GPU policy, dynamic layer rules, mouse/scroll requirements, agent tooling rules.
4. Read `SESSION_HANDOFF.md` for current work state.

## Skills

Activate when relevant:
- `repo-first-fix` — bug fixes.
- `cuda-preservation` — any CUDA/GPU/kernel work.
- `context-triage` — large/noisy context.
- `tool-before-code` — before writing custom scripts.
- `testing-verification` — before declaring task complete.
- `agent-tooling-research` — setup/MCP/skills questions.

## MCP Servers

Configured in `.kimi-code/mcp.json`:
- `filesystem` — repo file access.
- `memory` — persistent key-value memory.
- `playwright` — browser automation.

## Just Recipes

| Recipe | Purpose |
|--------|---------|
| `just ai-status` | Repo state and tool check. |
| `just ai-context` | Regenerate `.ai/context/repo-map.md` and `repomix.xml`. |
| `just ai-guard` | Block >80 changed files and GPU bypass in CUDA diffs. |
| `just ai-smoke` | Guard + ruff + pytest + biome/cargo checks. |
| `just ai-research` | Show verified tooling research. |
| `just run-report` | Generate full `build/run_report/`. |
| `just checkpoint-audit [CKPT]` | One-page audit of latest or given checkpoint. |
| `just layer-profile [CKPT]` | Per-layer role profile. |
| `just shortcut-audit [CKPT]` | High-value/misplaced shortcut audit. |
| `just constraint-trace` | Hard-constraint trends across checkpoints. |
| `just quick-context` | Minimal context pack for Kimi sessions. |

## Run Analysis Commands

```bash
# Full run report
just run-report

# Audit latest checkpoint
just checkpoint-audit

# Mouse/arrow/Norwegian focused reports
python3 tools/mouse_layer_report.py build/v2_checkpoint_gen<N>.json
python3 tools/arrow_cluster_report.py build/v2_checkpoint_gen<N>.json
python3 tools/completion_cluster_report.py build/v2_checkpoint_gen<N>.json

# Compare two checkpoints
python3 tools/compare_checkpoints.py build/v2_checkpoint_genA.json build/v2_checkpoint_genB.json
```

## GPU Policy Reminder

Training must never silently fall back to CPU. If CUDA is unavailable or the active training path is CPU/Numba-primary, the run must abort. CPU-only commands are allowed for tests, static checks, and diagnostics only.
```

- [ ] **Step 2: Add `.kimi-code/config.toml`**

```toml
[core]
model = "kimi-code/kimi-for-coding"

[instructions]
project = "KIMI.md"

[[hooks]]
name = "block-dangerous-bash"
type = "PreToolUse"
script = ".kimi-code/hooks/block-dangerous-bash.py"
```

Note: if Kimi does not read repo-local `config.toml`, move this content into `KIMI.md` and delete the file.

- [ ] **Step 3: Add `.kimi-code/hooks/block-dangerous-bash.py`**

```python
#!/usr/bin/env python3
"""PreToolUse hook: block destructive Bash commands."""
import json
import re
import sys

FORBIDDEN_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\b.*of=/dev/"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
]


def main():
    payload = json.load(sys.stdin)
    tool_name = payload.get("tool", "")
    arguments = payload.get("arguments", {})
    if tool_name != "Bash":
        json.dump({"allow": True}, sys.stdout)
        return
    command = arguments.get("command", "")
    if any(p.search(command) for p in FORBIDDEN_PATTERNS):
        json.dump({"allow": False, "reason": "Destructive command blocked by project hook."}, sys.stdout)
        return
    json.dump({"allow": True}, sys.stdout)


if __name__ == "__main__":
    main()
```

Make executable:

```bash
chmod +x .kimi-code/hooks/block-dangerous-bash.py
```

- [ ] **Step 4: Update `AGENTS.md`**

Add near the top, after the title:

```markdown
## Agent-Specific Entry Points

- **Kimi Code CLI:** read `KIMI.md` after this file for Kimi-specific skills, MCP servers, recipes, and run-analysis commands.
- **Claude Code:** `CLAUDE.md` redirects here.
- **Codex CLI:** see `CODEX_PROMPT.md`.
```

- [ ] **Step 5: Commit**

```bash
git add KIMI.md .kimi-code/config.toml .kimi-code/hooks/block-dangerous-bash.py AGENTS.md
git commit -m "docs(kimi): handoff config, hook, and updated entry files"
```

---

## Task 8: Just Recipes and Quick-Context Helper

**Files:**
- Modify: `justfile`
- Create: `.agent-tools/quick-context.sh`

**Interfaces:**
- Produces: new `just` recipes and a minimal context generator.

- [ ] **Step 1: Add recipes to `justfile`**

Append:

```just
# Generate full run report under build/run_report/
run-report:
    python3 tools/generate_run_report.py

# Audit latest or specified checkpoint
checkpoint-audit CKPT="":
    python3 tools/checkpoint_audit.py {{ if CKPT == "" { shell("python3 -c 'from tools._common import find_latest_checkpoint; print(find_latest_checkpoint() or \"\")'") } else { CKPT } }}

# Layer profile for latest or specified checkpoint
layer-profile CKPT="":
    python3 tools/layer_profile.py {{ if CKPT == "" { shell("python3 -c 'from tools._common import find_latest_checkpoint; print(find_latest_checkpoint() or \"\")'") } else { CKPT } }}

# Shortcut audit for latest or specified checkpoint
shortcut-audit CKPT="":
    python3 tools/shortcut_audit.py {{ if CKPT == "" { shell("python3 -c 'from tools._common import find_latest_checkpoint; print(find_latest_checkpoint() or \"\")'") } else { CKPT } }}

# Hard-constraint trace across checkpoints
constraint-trace:
    python3 tools/constraint_trace.py

# Generate minimal context pack for Kimi sessions
quick-context:
    bash .agent-tools/quick-context.sh
```

If just's conditional shell interpolation is awkward, simplify each recipe to call a Python script that resolves the default itself, e.g.:

```just
checkpoint-audit CKPT="latest":
    python3 tools/checkpoint_audit.py {{CKPT}}
```

And handle `"latest"` inside `checkpoint_audit.py`.

- [ ] **Step 2: Create `.agent-tools/quick-context.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.ai/context/quick-context.md"
mkdir -p "$(dirname "$OUT")"

cat > "$OUT" <<'HEAD'
# Quick Context — Charybdis Optimizer V2

This is a minimal context pack for Kimi sessions. For full repo context, run `just ai-context`.

HEAD

for f in AGENTS.md KIMI.md SESSION_HANDOFF.md justfile; do
    echo "## $f" >> "$OUT"
    echo '```' >> "$OUT"
    head -n 200 "$ROOT/$f" >> "$OUT" || true
    echo '```' >> "$OUT"
    echo "" >> "$OUT"
done

for f in run_evolution.py fitness/kernel.py evolution/custom_ga.py; do
    echo "## $f (first 200 lines)" >> "$OUT"
    echo '```python' >> "$OUT"
    head -n 200 "$ROOT/$f" >> "$OUT" || true
    echo '```' >> "$OUT"
    echo "" >> "$OUT"
done

echo "Generated: $OUT"
```

Make executable:

```bash
chmod +x .agent-tools/quick-context.sh
```

- [ ] **Step 3: Test recipes**

Run:

```bash
just run-report
just checkpoint-audit
just layer-profile
just shortcut-audit
just constraint-trace
just quick-context
```

Expected: all succeed.

- [ ] **Step 4: Commit**

```bash
git add justfile .agent-tools/quick-context.sh
git commit -m "build(just): analysis recipes and quick-context helper"
```

---

## Task 9: Documentation Update and Final Verification

**Files:**
- Modify: `.ai/context/agent-tooling-research.md`
- Test: full suite via `just ai-smoke`

- [ ] **Step 1: Update `.ai/context/agent-tooling-research.md`**

Append a new section:

```markdown
## Analysis Tools Added (2026-07-06)

Run-level:
- `tools/run_report.py` — gap trend, stagnation, surrogate health.
- `tools/generate_run_report.py` — writes full `build/run_report/` bundle.

Checkpoint-level:
- `tools/checkpoint_audit.py` — one-page acceptance/objectives/constraints audit.
- `tools/layer_profile.py` — per-layer app/category inventory.
- `tools/shortcut_audit.py` — misplaced/high-value shortcut ranking.
- `tools/mouse_layer_report.py` — mouse-layer focused report.
- `tools/arrow_cluster_report.py` — arrow cluster shape/scattered report.
- `tools/completion_cluster_report.py` — Norwegian completion cluster report.

Cross-checkpoint:
- `tools/constraint_trace.py` — hard-constraint trends.
- `tools/compare_checkpoints.py` — diff two checkpoints.

Shared infrastructure:
- `tools/_common.py` — loader, evaluator, checkpoint, reachability helpers.

Just recipes:
- `just run-report`, `just checkpoint-audit`, `just layer-profile`, `just shortcut-audit`, `just constraint-trace`, `just quick-context`.
```

- [ ] **Step 2: Final verification**

Run:

```bash
just ai-status
just ai-smoke
```

Expected: `ai-status` shows new tools/recipes; `ai-smoke` passes.

- [ ] **Step 3: Commit**

```bash
git add .ai/context/agent-tooling-research.md
git commit -m "docs: update tooling research with new analysis tools"
```

---

## Spec Coverage Check

| Spec Section | Implementing Task |
|--------------|-------------------|
| Rewrite `KIMI.md` | Task 7 |
| `.kimi-code/config.toml` | Task 7 |
| Destructive-command hook | Task 7 |
| `AGENTS.md` cross-reference | Task 7 |
| `tools/_common.py` | Task 1 |
| Refactor existing tools | Task 2 |
| `tools/run_report.py` | Task 3 |
| `tools/checkpoint_audit.py` | Task 3 |
| `tools/layer_profile.py` | Task 4 |
| `tools/shortcut_audit.py` | Task 4 |
| `tools/mouse_layer_report.py` | Task 4 |
| `tools/arrow_cluster_report.py` | Task 5 |
| `tools/completion_cluster_report.py` | Task 5 |
| `tools/constraint_trace.py` | Task 5 |
| `tools/compare_checkpoints.py` | Task 5 |
| `tools/generate_run_report.py` | Task 6 |
| Just recipes | Task 8 |
| Quick-context helper | Task 8 |
| Tests | Tasks 1–6 |
| No new dependencies | Global constraint |
| GPU policy untouched | Global constraint |

## Open Questions from Spec

1. **Repo-local `.kimi-code/config.toml`:** implement and test; if Kimi ignores it, document in `KIMI.md` and remove.
2. **Existing tool output normalization:** keep output semantically identical; only remove duplicated setup code.
3. **Hook language:** use Python for portability.
