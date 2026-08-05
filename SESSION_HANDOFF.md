# Session Handoff — Charybdis Optimizer V2

Updated: 2026-08-05. Rules live in `AGENTS.md`; Kimi commands in `KIMI.md`;
cross-repo promises in `ECOSYSTEM_CONTRACT.md`; decision/incident history in
`DECISION_LOG.md`. This file is only *current state* — rewrite it at the end
of any session that changes it.

## Current state

- **Validation run FINISHED (2026-08-05 15:32–18:27, log
  `build/run_logs/run_20260805_153230_g30000.log`): the 2026-08-05 GA fixes
  FAILED to unfreeze the archive.** Archive best stayed at gen 386
  (gap +10.77) for all 30k gens — it never self-resolved (unlike the morning
  baseline's gen ~29852). Population best improved to ~−66 while failing
  acceptance on `unsupported_duplicates_near_zero` (1-2 dups: Win+E 0.164,
  Ctrl+Shift+Up 0.10) and `momentary_only_thumb_side_clear` (4 layers).
  Root cause: both checks were soft viols ~1000x below the selection noise
  floor and not in `hard_constraints`, so Deb selection was blind to them and
  discarded repaired offspring (DECISION_LOG.md 2026-08-05, first entry).
  feasible=1.000 every gen (the trap: feasibility-first was satisfied while
  acceptance failed). Surrogate Spearman rho stayed NEGATIVE all run
  (−0.5..−0.62, relaxing to −0.11 only as diversity collapsed at the end) —
  metric verified correct; population drifts to exploit the surrogate between
  teacher updates. Needs its own fix (TODO #2), not yet started.
- **Hard-constraint fixes implemented, uncommitted:** new kernel terms
  `unsupported_duplicate` + `thumb_occupancy_restricted` (kernel.py + .cu
  mirror, both in `hard_constraints`), thumb-occupancy repair aligned to
  acceptance semantics (clears whole restricted side per call).
  181 passed, 0 failed (1 known deselect). Perf gate: ran after the run on
  idle GPU (see git status / next session for result).
- Morning run (2026-08-05): final archive best gap **−19.72**, all checks
  green — its `build/v2_evolution_results.json` was OVERWRITTEN by the
  afternoon validation run; the genome survives only if a copy exists
  elsewhere (run_report bundles don't store genomes). Best genome on disk is
  `build/runs/v2_scalefix_20260715/v2_checkpoint_gen56000.json` (gap −20.95,
  constraints zero, optimizer-side acceptance false at the time).
- Last commit: `6d5edb6` (2026-08-05, GA fixes + docs/config/test consolidation).
- Warmstart: `build/v2_local_search_result.json` (loaded with sid-range
  validation since 2026-08-05).

## Known issues / TODOs

1. **CUDA/Numba parity drift:** `tests/test_cuda_exact_eval_parity.py`
   fails — one viol element differs ~0.7% between the `.cu` and Numba
   kernels. Pre-existing at HEAD; root-cause not yet done. NOTE: the
   kernel now has 2 new viol terms; the drift may shift — re-check when
   root-causing.
2. **Surrogate anti-correlation:** Spearman rho surrogate-vs-exact is
   persistently negative (−0.5..−0.62) on the live population while retrain
   R² reads ~0.89 on the training cache — the GA drifts to exploit the
   surrogate between teacher updates, so surrogate pre-screening actively
   misranks. Candidate fixes: rank-based (Spearman/pairwise) surrogate loss,
   shorter exact-eval cadence, or trust-region on surrogate-selected
   candidates. Not yet started.
3. Early stop in `evolution/custom_ga.py` requires 100k stagnant gens —
   never fires within a 30k run; the unused `run_evolution.py` callback path
   uses 5k. Reconcile if early stopping matters.
4. `evolution/__init__.py` (~5k lines) is the next modularization target;
   do it only with the perf gate green and no active run.

## Next logical step

When the live run finishes: run `just ai-guard` (perf gate, idle GPU) on the
uncommitted hard-constraint fixes, commit them (user approval), then launch a
fresh `just run` — it validates both the 2026-08-05 GA fixes AND the new
hard constraints. Watch: does the archive now track the population best
instead of freezing; surrogate rho (expected still negative — separate fix);
does `unsupported_duplicate_repair` finally stick. Root-cause the CUDA/Numba
parity drift when convenient.

## Sister project sync

No layout promotion performed from this repo. The 2026-08-05 run's final
layout (gap −19.72, all checks green) is a promotion candidate via
charybdis-tools (`ECOSYSTEM_CONTRACT.md`). Sister repos untouched.
