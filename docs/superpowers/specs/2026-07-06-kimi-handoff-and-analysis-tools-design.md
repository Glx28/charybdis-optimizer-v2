# Kimi Handoff + Token-Saving Analysis Tools — Design Spec

**Date:** 2026-07-06
**Approach:** B (full analysis suite + handoff docs, no custom MCP server)
**Goal:** Make the Charybdis Optimizer V2 repo maximally efficient for future Kimi Code CLI sessions, with one-command answers to repetitive run/layout analysis questions.

## Background

The repo already has strong GPU-primary evolution, agent CLIs, fast local tools, MCP servers, and skills. The bottleneck for future sessions is not missing infrastructure — it is that repetitive analysis ("how is the run doing?", "why did it stagnate?", "what is each layer for?", "which shortcuts are misplaced?") still requires ad-hoc Python snippets that re-do the same loader/arrays/reachability setup every time. This wastes tokens and produces inconsistent answers.

## Scope

In scope:
- Rewrite `KIMI.md` into a proper Kimi handoff.
- Add shared analysis infrastructure in `tools/_common.py`.
- Refactor existing `tools/*.py` to use the shared infrastructure.
- Add new CLI analysis tools for run, checkpoint, layer, shortcut, mouse/arrow/Norwegian, constraints, and diffs.
- Add `just` recipes so agents do not have to remember Python invocations.
- Add a quick-context helper to avoid dumping the full 1.17 MB `repomix.xml`.

Out of scope:
- Custom MCP server (approach C). The CLI tools are the first layer; MCP wrapping can be added later if needed.
- New evolution algorithms or fitness changes.
- Major refactoring of `run_evolution.py`, `evolution/custom_ga.py`, or CUDA kernels.

## Components

### 1. Kimi Handoff Docs

#### `KIMI.md` (rewrite)
Must contain:
- One-sentence project description.
- Startup checklist: `just ai-status`, `just ai-context`, read `AGENTS.md`, activate relevant skills.
- Skills list with when to use each.
- MCP servers configured (`filesystem`, `memory`, `playwright`) and their purpose.
- Just recipes cheat sheet.
- GPU policy summary (fail-fast on CPU-primary training).
- Run-analysis command cheat sheet.
- Pointer to `SESSION_HANDOFF.md` for current work state.

#### `.kimi-code/config.toml` (new)
If Kimi supports repo-local config, add:
- Model preference: `kimi-code/kimi-for-coding`.
- One `[[hooks]]` entry pointing to `.kimi-code/hooks/block-dangerous-bash.mjs` (or equivalent Python script) to block destructive commands.
- Reference to `KIMI.md` as project instructions.

If repo-local config is not supported, document the equivalent global config in `KIMI.md`.

#### `.kimi-code/hooks/block-dangerous-bash.mjs` (new)
Mirror the existing global hook and Codex/Claude guards. Blocks: `rm -rf /`, `mkfs`, `dd`, `shutdown`, `reboot`, and any command that would write outside `/home/nos/charybdis/charybdis-optimizer-v2`.

#### `AGENTS.md` update
Add a short cross-reference so Kimi users know `KIMI.md` exists and contains Kimi-specific commands.

### 2. Shared Analysis Infrastructure — `tools/_common.py`

Functions to expose:
- `load_layout(data_dir="data") -> Layout`
- `load_evaluator(config_path="config_v2.yaml", data_dir="data", build_dir="build") -> FitnessEvaluator`
- `load_checkpoint(path: str) -> dict`
- `checkpoint_to_layout(checkpoint: dict, layout: Layout) -> Layout`
- `compute_reachability(genome, layout) -> dict` with `all_hops`, `hold_hops`, `toggle_hops`, `reachable_layers`.
- `label_shortcut(sid: int, layout, arrays) -> dict` with `name`, `app`, `category`, `importance`, `usage`.
- `layer_inventory(layout) -> list[dict]` — per-layer summary.
- `genome_to_layer_assignments(layout) -> dict` — position -> shortcut mapping per layer.
- `find_latest_checkpoint(build_dir="build") -> str`
- `find_checkpoints(build_dir="build") -> list[str]` sorted by generation.

All functions must work inside the repo venv without extra installs.

### 3. Refactor Existing Tools

Update these to use `tools/_common.py` and produce cleaner output:
- `tools/best.py`
- `tools/check.py`
- `tools/human_audit.py`
- `tools/workflow_check.py`

No behavior changes; only deduplicate setup and normalize output format.

### 4. New CLI Analysis Tools

#### `tools/run_report.py [build_dir]`
Outputs to stdout (and optionally `build/run_report.md` with `--write`):
- Run config summary (pop size, gens planned, device, surrogate enabled).
- Best result: gen, gap, total score, acceptance pass/fail, failed checks.
- Score trajectory: best score per checkpoint, time to best, stagnation count.
- Diversity / injection events: count, triggers, lowest diversity observed.
- Surrogate health: R² history, cache size, retrain events.
- Gen/sec and total runtime.
- One-paragraph "likely status" (improving / stagnated / infeasible / etc.).

#### `tools/checkpoint_audit.py <checkpoint>`
Concise one-page audit:
- Gap, objectives, constraints, acceptance status.
- Failed checks with guidance.
- Layer role summary.
- Mouse layer status.
- Arrow cluster status.
- Norwegian completion cluster status.
- Top 5 misplaced shortcuts.
- Top 5 empty prime positions.

#### `tools/layer_profile.py <checkpoint>`
Markdown table, one row per layer:
- Layer index.
- Access cost (best hold/toggle).
- Reachable?.
- Assigned shortcut count.
- Top 3 apps by shortcut count.
- Top 3 categories.
- Mouse buttons present.
- Arrow keys present.
- Norwegian completion keys present.
- Empty slot count.
- Dominant inferred workflow.

#### `tools/shortcut_audit.py <checkpoint>`
Ranked list of shortcuts with flags:
- Importance / usage.
- Effort of assigned position.
- Layer and access hops.
- Duplicates (supported / unsupported / multi-workflow).
- Flags: `misplaced`, `unreachable`, `unsupported_duplicate`, `missing_important`.
- Sort by `(importance * usage) / effort` descending, with infeasible placements first.

#### `tools/mouse_layer_report.py <checkpoint>`
Focused report:
- Candidate mouse layer(s).
- MB1–MB5 positions.
- Right-hand non-thumb scroll capability.
- Hold-hop depth to layer.
- Right-thumb conflicts.
- Toggle reachability.
- Acceptance `dynamic_mouse_layer_present` pass/fail with details.

#### `tools/arrow_cluster_report.py <checkpoint>`
- Non-L7 arrow placements.
- Allowed shape check (`Left Up Down Right` single row, or `Left Down Right` + `Up` above `Down`).
- Scattered penalty contributors.
- Acceptance `mutable_raw_arrows_ok` status.

#### `tools/completion_cluster_report.py <checkpoint>`
- Anchor layer for Norwegian/raw completion keys.
- Present/missing keys.
- Compactness score.
- Ordered-left-to-right flag.
- Acceptance `norwegian_completion_cluster` status.

#### `tools/constraint_trace.py [build_dir]`
Read all checkpoints and output a markdown/CSV table:
- Generation.
- Each hard-constraint value.
- Population best constraints.
- Archive best constraints.
- Top raw-score violation contributor.

#### `tools/compare_checkpoints.py <ckpt_a> <ckpt_b>`
- Objective/constraints delta.
- Acceptance delta.
- Number of positions changed.
- Layer-role changes.
- Whether b is a progression or regression.

#### `tools/generate_run_report.py [build_dir]`
Orchestrator that runs:
1. `run_report.py --write`
2. `checkpoint_audit.py` on latest checkpoint
3. `layer_profile.py` on latest checkpoint
4. `shortcut_audit.py` on latest checkpoint
5. `mouse_layer_report.py` on latest checkpoint
6. `constraint_trace.py --write`

Writes all outputs to `build/run_report/` with timestamps.

### 5. Just Recipes

Add to `justfile`:

```just
# Full post-run report suite
run-report:
    python3 tools/generate_run_report.py

# Audit a specific checkpoint (defaults to latest)
checkpoint-audit CKPT="":
    python3 tools/checkpoint_audit.py {{ if CKPT == "" { shell("python3 -c 'from tools._common import find_latest_checkpoint; print(find_latest_checkpoint())'") } else { CKPT } }}

# Layer profile for latest or given checkpoint
layer-profile CKPT="":
    python3 tools/layer_profile.py {{ if CKPT == "" { shell("python3 -c 'from tools._common import find_latest_checkpoint; print(find_latest_checkpoint())'") } else { CKPT } }}

# Constraint trace across all checkpoints
constraint-trace:
    python3 tools/constraint_trace.py

# Shortcut audit for latest or given checkpoint
shortcut-audit CKPT="":
    python3 tools/shortcut_audit.py {{ if CKPT == "" { shell("python3 -c 'from tools._common import find_latest_checkpoint; print(find_latest_checkpoint())'") } else { CKPT } }}

# Regenerate minimal context pack (repo map + key files, not full repomix)
quick-context:
    bash .agent-tools/quick-context.sh
```

If just's conditional syntax is too awkward for default args, implement the default-latest logic inside each Python script and keep the recipe simple:

```just
checkpoint-audit CKPT="build/v2_checkpoint_gen_latest.json":
    python3 tools/checkpoint_audit.py {{CKPT}}
```

With the Python script resolving the symlink/path if the literal file does not exist.

### 6. Quick-Context Helper — `.agent-tools/quick-context.sh`

Generate a context pack for a Kimi session without the 1.17 MB `repomix.xml`:
- `.ai/context/repo-map.md`
- `AGENTS.md`
- `KIMI.md`
- `SESSION_HANDOFF.md`
- `justfile`
- `run_evolution.py` (first 200 lines only)
- `fitness/kernel.py` (first 200 lines only)
- `evolution/custom_ga.py` (first 200 lines only)

Output to `.ai/context/quick-context.md`.

## Dependencies

No new Python dependencies. All new tools use existing repo code and standard library.
If we later add CSV output, use the standard `csv` module.
If we later add plotting, use `matplotlib` only if already present in venv.

## Testing

- Existing `just ai-smoke` must still pass.
- Add a new smoke test in `tests/test_v2.py` or a new `tests/test_analysis_tools.py` that:
  - Imports `tools._common` and calls `load_layout`, `find_latest_checkpoint`.
  - Runs `tools/check.py` on the latest checkpoint and checks exit code 0.
  - Runs `tools/run_report.py` on `build/` and checks exit code 0.
- The refactor of existing tools must not change their CLI output format in a way that breaks any callers.

## File List

New files:
- `.kimi-code/config.toml`
- `.kimi-code/hooks/block-dangerous-bash.mjs`
- `tools/_common.py`
- `tools/run_report.py`
- `tools/checkpoint_audit.py`
- `tools/layer_profile.py`
- `tools/shortcut_audit.py`
- `tools/mouse_layer_report.py`
- `tools/arrow_cluster_report.py`
- `tools/completion_cluster_report.py`
- `tools/constraint_trace.py`
- `tools/compare_checkpoints.py`
- `tools/generate_run_report.py`
- `.agent-tools/quick-context.sh`
- `docs/superpowers/specs/2026-07-06-kimi-handoff-and-analysis-tools-design.md`

Modified files:
- `KIMI.md` (rewritten)
- `AGENTS.md` (cross-reference added)
- `justfile` (new recipes)
- `tools/best.py` (refactored to use `_common`)
- `tools/check.py` (refactored to use `_common`)
- `tools/human_audit.py` (refactored to use `_common`)
- `tools/workflow_check.py` (refactored to use `_common`)
- `.ai/context/agent-tooling-research.md` (document new tools)
- `tests/test_v2.py` or new `tests/test_analysis_tools.py`

## Success Criteria

1. A future Kimi session can run `just ai-status` and see the new tools/recipes.
2. `just run-report` produces a complete `build/run_report/` directory from existing checkpoints.
3. `just checkpoint-audit` audits the latest checkpoint without needing the path.
4. `just ai-smoke` passes.
5. No new Python dependencies required.
6. `KIMI.md` is the single source of truth for Kimi-specific commands.

## Open Questions

1. Does Kimi Code CLI read repo-local `.kimi-code/config.toml`? If not, the config section moves to `KIMI.md` documentation only.
2. Should `tools/generate_run_report.py` overwrite `build/run_report/` or append timestamps? (Spec says overwrite with timestamps inside.)
3. Should we preserve exact current output of `tools/check.py`/`human_audit.py`, or is minor format normalization acceptable?

## Risks

- Refactoring existing tools could break subtle behavior. Mitigation: keep behavior changes minimal; test with `just ai-smoke`.
- New tools may duplicate logic already in `evolution/acceptance.py`. Mitigation: call existing functions rather than reimplement.
- Hook script may not be executable cross-platform. Mitigation: use Python for the hook if Node is not guaranteed.
