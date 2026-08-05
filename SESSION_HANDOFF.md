# Session Handoff — Charybdis Optimizer V2

Updated: 2026-08-05. Rules live in `AGENTS.md`; Kimi commands in `KIMI.md`;
cross-repo promises in `ECOSYSTEM_CONTRACT.md`; decision/incident history in
`DECISION_LOG.md`. This file is only *current state* — rewrite it at the end
of any session that changes it.

## Current state

- **Hard-constraint validation run FINISHED (2026-08-05 20:10–23:05, log
  `build/run_logs/run_20260805_201043_g30000.log`): the fcc6c07 fix WORKED.**
  Final archive best: gen 18500, gap **−18.02, ALL acceptance checks green**
  (only external export validation pending, as always). Archive tracked the
  population best for 18.5k gens straight (previous run: frozen at gen 386
  forever, final +10.77). `unsupported_duplicates` and
  `thumb_occupancy_restricted` stayed at zero violations all run;
  feasible=1.000 every gen. vs morning baseline (−19.72): 1.7 points short —
  see next bullet for why.
- **New blocker surfaced (same systemic class):** from ~gen 22500 the
  population best (−67.85, gap −19.0+) exceeded the archive while failing
  `norwegian_completion_cluster` — another acceptance check that is only a
  soft pressure, invisible to feasibility-first selection. Pattern is now
  proven twice: ANY acceptance check that isn't selection-visible can become
  the archive blocker. Candidate fix: same hard-constraint treatment, or a
  stronger completion-cluster operator. NOT yet decided/implemented.
- Surrogate Spearman rho stayed negative all run (−0.3..−0.7); twice relaxed
  toward 0 right after diversity injections, then re-anti-correlated as the
  population re-converged — strong confirmation of the exploitation-drift
  model (TODO #2).
- First validation run (15:32–18:27) FAILED as designed-to-detect: archive
  frozen at gen 386 (+10.77) for 30k gens; root-caused to selection-blind
  soft violations; fixed in fcc6c07.
- Morning run (2026-08-05): final −19.72 all green; its results json was
  overwritten. Best genome on disk:
  `build/runs/v2_scalefix_20260715/v2_checkpoint_gen56000.json` (gap −20.95,
  constraints zero, July acceptance false). NOTE: `just checkpoint-audit`
  with no args audits THAT stale best checkpoint, not the latest run's —
  pass the checkpoint path explicitly.
- Last commit: `fcc6c07` (2026-08-05, hard-constraint fixes).
- Warmstart: `build/v2_local_search_result.json` (loaded with sid-range
  validation since 2026-08-05).

## Known issues / TODOs

1. **CUDA/Numba parity drift:** `tests/test_cuda_exact_eval_parity.py`
   fails — one viol element differs ~0.7% between the `.cu` and Numba
   kernels. Pre-existing at HEAD; root-cause not yet done. NOTE: the
   kernel now has 2 new viol terms; the drift may shift — re-check when
   root-causing.
2. **Surrogate anti-correlation:** Spearman rho surrogate-vs-exact is
   persistently negative (−0.3..−0.7) on the live population while retrain
   R² reads ~0.9 on the training cache — the GA drifts to exploit the
   surrogate between teacher updates. Confirmed twice, incl. the
   injection→rho→re-convergence cycle. Candidate fixes: rank-based
   (Spearman/pairwise) surrogate loss, shorter exact-eval cadence, or
   trust-region on surrogate-selected candidates. Not yet started.
3. **Selection-invisible acceptance checks (systemic):** twice now an
   acceptance check that is only soft pressure became the archive blocker
   (unsupported dups + thumb occupancy, fixed in fcc6c07; then
   `norwegian_completion_cluster` in the last third of the validation run).
   Decide: hard-constraint treatment for `norwegian_completion_cluster`
   (needs a kernel term mirroring the cluster-shape rule — see
   `evolution/completion_cluster.py` and the acceptance check), or a
   stronger completion-cluster operator. Longer term: audit EVERY acceptance
   check for selection visibility instead of whack-a-mole.
4. Early stop in `evolution/custom_ga.py` requires 100k stagnant gens —
   never fires within a 30k run; the unused `run_evolution.py` callback path
   uses 5k. Reconcile if early stopping matters.
5. `evolution/__init__.py` (~5k lines) is the next modularization target;
   do it only with the perf gate green and no active run.
6. `just checkpoint-audit` with no args audits the best-by-gap checkpoint
   under `build/runs/` (stale July scalefix gen56000), not the latest run's
   checkpoint — pass the path explicitly or fix the default.

## Next logical step

Decide the `norwegian_completion_cluster` treatment (TODO #3, user decision),
implement it with kernel + .cu mirror + tests, then a fresh `just run` —
the archive best from this run (gap −18.02, all green, in
`build/v2_evolution_results.json`) is also the best warmstart-adjacent
candidate now. Root-cause the CUDA/Numba parity drift when convenient.

## Sister project sync

No layout promotion performed from this repo. The 2026-08-05 run's final
layout (gap −19.72, all checks green) is a promotion candidate via
charybdis-tools (`ECOSYSTEM_CONTRACT.md`). Sister repos untouched.
