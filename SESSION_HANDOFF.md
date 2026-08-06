# Session Handoff — Charybdis Optimizer V2

Updated: 2026-08-05. Rules live in `AGENTS.md`; Kimi commands in `KIMI.md`;
cross-repo promises in `ECOSYSTEM_CONTRACT.md`; decision/incident history in
`DECISION_LOG.md`. This file is only *current state* — rewrite it at the end
of any session that changes it.

## Current state

- **Third validation run FINISHED (2026-08-06 00:12–03:18, log
  `build/run_logs/run_20260806_001228_g30000.log`): FULL SUCCESS — the
  archive-freeze failure mode is fixed.** Final archive best: gen 29950,
  gap **−19.70, ALL acceptance checks green** (external export validation
  pending as always). The archive improved right up to the last 50 gens —
  it NEVER froze (run 1: froze at gen 386, +10.77; run 2: froze at gen
  18500, −18.02). It sailed through gen 22500 (where run 2's cluster check
  failed) with zero acceptance failures. Nominally 0.02 short of the
  morning baseline (−19.72) — a tie for practical purposes, and the morning
  number was scored under the pre-fcc6c07 weights so cross-config
  comparison is approximate. Genome in `build/v2_evolution_results.json`
  + `build/v2_checkpoint_gen30000.json`.
- All three selection-invisibility fixes are now validated by real runs:
  `unsupported_duplicate` + `thumb_occupancy_restricted` (fcc6c07) and
  `norwegian_completion_cluster` (ef85678). feasible=1.000 and zero
  hard-constraint violations across the whole third run.
- Surrogate Spearman rho stayed negative (−0.45..−0.6) all third run —
  confirmed chronic; the surrogate contributes pre-screening misranking in
  every run. Top unfixed GA issue (TODO #2).
- Best genome on disk remains `build/runs/v2_scalefix_20260715/
  v2_checkpoint_gen56000.json` (gap −20.95, but carries 10
  thumb_occupancy_restricted violations under the current config — it was
  pre-fix). The third run's −19.70 all-green layout is the best
  CURRENT-CONFIG, acceptance-green candidate for promotion.
- Last commits: `ef85678` (cluster hard constraint), `fcc6c07` (first two
  hard constraints), `6d5edb6` (GA fixes + consolidation).
- Warmstart: `build/v2_local_search_result.json` (loaded with sid-range
  validation since 2026-08-05).

## Known issues / TODOs

1. **CUDA/Numba parity drift:** `tests/test_cuda_exact_eval_parity.py`
   fails — one viol element differs ~0.7% between the `.cu` and Numba
   kernels. Pre-existing at HEAD; root-cause not yet done. NOTE: the
   kernel gained 3 new viol terms on 2026-08-05/06 (slots 25-27); the
   drifted element may have shifted — re-check when root-causing.
2. **Surrogate anti-correlation (top unfixed GA issue):** Spearman rho
   surrogate-vs-exact is persistently negative (−0.45..−0.6) on the live
   population in every run while retrain R² reads ~0.9 on the training
   cache — the GA drifts to exploit the surrogate between teacher updates,
   so surrogate pre-screening actively misranks. Confirmed in three runs,
   incl. the injection→rho→re-convergence cycle. Candidate fixes:
   rank-based (Spearman/pairwise) surrogate loss, shorter exact-eval
   cadence, or trust-region on surrogate-selected candidates.
3. **Remaining selection-invisible acceptance checks:** three have now
   become archive blockers and been hard-constrained (unsupported dups,
   thumb occupancy, norwegian cluster — all validated). Audit the REST of
   the acceptance list (`evolution/acceptance.py`) for soft-only checks
   that could become the next blocker: candidates visible in checkpoint
   reports include `win_s_present`, `scroll_mode_access_present`,
   `mutable_raw_arrows_ok`, `no_fake_scroll_keypresses`. Do the audit
   BEFORE the next long run instead of discovering the fourth mole at
   gen 25k.
4. Early stop in `evolution/custom_ga.py` requires 100k stagnant gens —
   never fires within a 30k run; the unused `run_evolution.py` callback path
   uses 5k. Reconcile if early stopping matters.
5. `evolution/__init__.py` (~5k lines) is the next modularization target;
   do it only with the perf gate green and no active run.
6. `just checkpoint-audit` with no args audits the best-by-gap checkpoint
   under `build/runs/` (stale July scalefix gen56000), not the latest run's
   checkpoint — pass the path explicitly or fix the default.

## Next logical step

The 2026-08-06 00:12 run's layout (gap −19.70, all acceptance checks
green, `build/v2_evolution_results.json`) is the best current-config
promotion candidate — promotion path is charybdis-tools per
ECOSYSTEM_CONTRACT.md (sister repo, READ-ONLY from here; external export
validation still pending). GA-side: audit remaining acceptance checks for
selection visibility (TODO #3) before the next long run, then fix the
surrogate anti-correlation (TODO #2). Root-cause the CUDA/Numba parity
drift when convenient.

## Sister project sync

No layout promotion performed from this repo. The 2026-08-05 run's final
layout (gap −19.72, all checks green) is a promotion candidate via
charybdis-tools (`ECOSYSTEM_CONTRACT.md`). Sister repos untouched.
