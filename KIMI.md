# Kimi Handoff — Charybdis Optimizer V2

GPU-primary evolutionary keyboard-layout optimizer for the Charybdis split keyboard.

**Start here:** read `AGENTS.md` first for project rules (GPU policy, dynamic layer assignment, mouse/scroll requirements, agent tooling rules). Then use this file for Kimi-specific commands and shortcuts.

## Startup Checklist

1. `just ai-status` — repo state, available tools, skills, MCP.
2. `just ai-context` — regenerate `.ai/context/repo-map.md` and `repomix.xml`.
3. `just ai-research` — show verified tooling research.
4. Read `AGENTS.md` and `SESSION_HANDOFF.md`.

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
| `just run-report` | Generate full `build/run_report/` bundle. |
| `just checkpoint-audit [CKPT]` | One-page audit of latest or given checkpoint. |
| `just layer-profile [CKPT]` | Per-layer role profile. |
| `just shortcut-audit [CKPT]` | High-value/misplaced shortcut audit. |
| `just constraint-trace` | Hard-constraint trends across checkpoints. |
| `just quick-context` | Minimal context pack for Kimi sessions. |

## Run Analysis Commands

```bash
# Full run report (writes build/run_report/YYYY-MM-DD_HH-MM-SS/)
just run-report

# Audit latest checkpoint
just checkpoint-audit

# Layer roles, shortcut audit, mouse layer
just layer-profile
just shortcut-audit
python3 tools/mouse_layer_report.py latest
python3 tools/arrow_cluster_report.py latest
python3 tools/completion_cluster_report.py latest

# Stagnation diagnosis
python3 tools/run_report.py build
python3 tools/constraint_trace.py build

# Compare two checkpoints
python3 tools/compare_checkpoints.py build/v2_checkpoint_gen9000.json build/v2_checkpoint_gen11000.json

# Find best checkpoint by gap
python3 tools/best.py
```

## GPU Policy Reminder

Training must never silently fall back to CPU. If CUDA is unavailable or the active training path is CPU/Numba-primary, the run must abort before warmup. CPU-only commands are allowed for tests, static checks, and diagnostics only.

## Token-Saving Infrastructure

- `tools/_common.py` — shared loader/evaluator/checkpoint helpers; all new tools use it.
- `tools/generate_run_report.py` — one-command full run analysis bundle.
- `.agent-tools/quick-context.sh` — minimal context pack (avoids dumping the full 1.17 MB `repomix.xml`).

## Cross-Agent Files

- `AGENTS.md` — canonical project rules.
- `CLAUDE.md` — redirects to `AGENTS.md`.
- `SESSION_HANDOFF.md` — current work state.

## Self-Improving Agent Setup

For a session whose goal is to finish incomplete work, improve repo automation, and optimize Kimi's own workflow, paste the contents of `.kimi-code/prompts/self-improving.md` as the system prompt. It will bootstrap from `AGENTS.md`/`KIMI.md`, scan agent logs for recurring tasks, and add skills/recipes/scripts that save tokens on future sessions.
