# Session Handoff — Charybdis Optimizer V2

Updated: 2026-08-05. Rules live in `AGENTS.md`; Kimi commands in `KIMI.md`;
cross-repo promises in `ECOSYSTEM_CONTRACT.md`; decision/incident history in
`DECISION_LOG.md`. This file is only *current state* — rewrite it at the end
of any session that changes it.

## Current state

- **No active run.** Last run: 30,000-gen GPU run completed 2026-08-05
  (~2.9h, 2.9 gen/sec). Final archive best: gap **−19.72**, all hard
  constraints zero, all acceptance checks green (`build/v2_evolution_results.json`,
  checkpoints `build/v2_checkpoint_gen*.json`, log `build/run_logs/`).
- Last commit: `2e085e0` (2026-07-14). Working tree holds reviewed, tested
  but **uncommitted** changes: 2026-07-15 scale-cache/rebase work,
  2026-08-05 warmstart validation, token-reduction pass, config-driven
  mutation probs, test split, GA constraint-handling fixes (see
  `DECISION_LOG.md`). Full suite: 156 passed + 1 known failure (below);
  `just ai-guard` perf gate green on idle GPU.
- Warmstart: `build/v2_local_search_result.json` (loaded with sid-range
  validation since 2026-08-05).

## Known issues / TODOs

1. **CUDA/Numba parity drift:** `tests/test_cuda_exact_eval_parity.py`
   fails — one viol element differs ~0.7% between the `.cu` and Numba
   kernels. Pre-existing at HEAD; root-cause not yet done.
2. GA fixes from 2026-08-05 (feasibility-first selection, feasibility
   logging, surrogate Spearman metric, `unsupported_duplicate_repair`) are
   unit-tested but have **not yet been exercised by a real training run** —
   the next `just run` is their validation. Watch: feasible-fraction log
   line, surrogate rho, and whether the archive unfreezes earlier than it
   did in the 2026-08-05 run (gen ~29852).
3. Early stop in `evolution/custom_ga.py` requires 100k stagnant gens —
   never fires within a 30k run; the unused `run_evolution.py` callback path
   uses 5k. Reconcile if early stopping matters.
4. `evolution/__init__.py` (4.8k lines) is the next modularization target;
   do it only with the perf gate green and no active run.

## Next logical step

Commit the reviewed working-tree changes (user approval required), then
launch a fresh `just run` to validate the GA fixes. Root-cause the
CUDA/Numba parity drift when convenient.

## Sister project sync

No layout promotion performed from this repo. The 2026-08-05 run's final
layout (gap −19.72, all checks green) is a promotion candidate via
charybdis-tools (`ECOSYSTEM_CONTRACT.md`). Sister repos untouched.
