# Decision Log

Format: `[date] | chose [X] | because [Y] | confirmed by user`. Incident
narratives moved out of source comments live in the History section below;
source comments carry the design rationale, this file carries the archaeology.

## Decisions

- 2026-08-05 | made unsupported duplicates and restricted thumb occupancy hard
  constraints: new kernel terms `unsupported_duplicate` (weight 200000) and
  `thumb_occupancy_restricted` (weight 50000, split out of soft
  `thumb_occupancy` so toggle-freed effort-floor pressure stays soft and
  acceptance-PASS layers can never hard-fail), both added to
  `hard_constraints`; `_numba_repair_thumb_occupancy` aligned to acceptance
  semantics (reachable-source gating, self-referential exclusion, toggle-freed
  skip, clears the whole restricted side per call); repair gained the
  `is_l0_only` exclusion | the 30k validation run's archive froze at gen ~386
  because Deb feasibility-first selection is blind to both checks (soft
  violations ~1000x below the noise floor, cv=0 everywhere, `feasible=1.000`),
  so repaired offspring always lost tournaments to violation carriers; the
  repair operators produced correct genomes that selection discarded —
  root-caused against checkpoints gen9000-14000 (pop best stuck at
  unsup_dup 1-2 = Win+E support 0.164 / Ctrl+Shift+Up support 0.10, thumb_viol
  4 layers, oscillating) | hard-constraint option pre-approved 2026-08-05;
  direction confirmed by user ("keep running + fix in parallel"). Perf gate
  pending idle GPU; new terms add same-complexity-class per-eval work —
  if perf_benchmark flags it, update baseline with an explicit reason.
- 2026-08-05 | validate warmstart genomes at load (`load_warmstart_genome` in
  `run_evolution.py`), masking out-of-range shortcut ids to -1 | a stale
  `build/v2_local_search_result.json` (2026-07-05, `source:
  recovered_gap-5.032_gen4379`) carried sids 293-295 against a 293-shortcut
  data snapshot and crashed surrogate training with a CUDA embedding index
  assertion | confirmed by user
- 2026-08-05 | consolidated policy docs into `AGENTS.md` as single source of
  truth; `LAYER_ACCESS_POLICY.md`/`GOAL_PROMPT.md`/`CODEX_PROMPT.md` are now
  redirects; deleted `SESSION_HANDOFF_WSL.md`, `SPEED_HANDOFF.md`,
  `docs/superpowers/*` | four drifting copies contradicted the canonical rules
  (fixed L10 mouse layer, weak MB-pair language, no 30k target) |
  confirmed by user (token reduction plan, item 1)
- 2026-08-05 | deleted `export.py` (dead, nothing imported it; the real export
  path is charybdis-tools' `export_and_analyze_linux.py`), moved
  `local_search.py` to `tools/` | dead-code removal | confirmed by user (item 2)
- 2026-08-05 | synced `config/__init__.py` `DEFAULT_CONFIG` to
  `config_v2.yaml` | 13 defaults had diverged, some 3-4 orders of magnitude
  stale (e.g. `toggle_back_to_l0` 1.5e11 vs 2e5) | confirmed by user (item 3)
- 2026-08-05 | moved the 10 hardcoded `SwapMutation` probabilities to
  `config_v2.yaml` (`evolution.*_prob`) | tunables belong in config |
  confirmed by user (item 7)
- 2026-08-05 | excluded `data/` from the context pack
  (`.agent-tools/repo-map.sh`); trimmed `config_v2.yaml` narratives to
  one-line comments; rewrote `SESSION_HANDOFF.md` | machine-generated JSON
  (1.3MB) and stale prose were 55%+ of the AI context pack | confirmed by
  user (token pass 2)
- 2026-08-05 | migrated production-dead factor classes (`EffortFactor`,
  `AdjacencyFactor`, `ViolationFactor`, etc.) from `fitness/factors/` to
  `tests/legacy_factors.py` | they were only imported by tests; production
  scoring is the compiled kernel | confirmed by user
- 2026-08-05 | GA constraint-handling fixes after literature review +
  diagnosis of the archive-freeze pattern (population best improving but
  acceptance-failing for ~29k gens of the 2026-08-05 run before self-
  resolving at gen ~29852): (1) Deb (2000) feasibility-first tournament
  selection, config-gated by `evolution.feasibility_first_selection`;
  (2) per-generation feasible-fraction logging; (3) surrogate health metric
  (Spearman rho surrogate-vs-exact on exact batches); (4) new
  `unsupported_duplicate_repair` operator (numba + Python fallback, prob
  `evolution.unsupported_duplicate_repair_prob`) blanking extra copies of
  evidence-backed support<0.25 duplicates -- the penalty for those sat
  ~1000x below their adjacency benefit (below the selection noise floor)
  and no operator removed them | confirmed by user ("do the 1-6 fixes").
  Deliberately not done: epsilon-constraint (fallback if Deb's rules stall)
  and making unsupported duplicates a hard constraint (option kept for the
  next fresh run).

## History (incident narratives moved from source comments)

- 2026-07-13, thumb-clearance vs access-to-thumb bias (`evolution/__init__.py`):
  a gen-30000 checkpoint had 4 layers stuck violating the dynamic
  thumb-clearance rule identically across the last 2000 generations — the
  access-to-thumb bias operator refilled slots that
  `_numba_repair_thumb_occupancy` cleared. The bias operator now skips
  restricted sides.
- 2026-07-11/12, `momentary_key_reuse` plateau (`evolution/__init__.py`): pure
  fitness pressure plateaued on multi-job momentary keys across thousands of
  generations even at a strongly recalibrated weight; escaping needs a
  coordinated single-generation move, hence the dedicated repair mutation.
- 2026-07-13, MB same-row/adjacency (`fitness/kernel.py` + `.cu`): MB1/MB2 and
  MB4/MB5 same-row requirements confirmed as user-required properties; a
  checkpoint had MB4/MB5 on the same row but 2 slots apart, uncorrected by
  proximity-only pressure — hence the exact x-adjacency term (dx==1).
- 2026-07-13, empty_position layer_factor floor (`fitness/kernel.py` + `.cu`):
  6 genuinely prime (effort==0) empty slots on L5/L9/L10 sat empty and flat
  across 1000+ generations late in a 30k run; access-cost scaling crushed
  their pressure to near-zero. Floor at 0.5 added.
- 2026-07-13/14, display rebasing (`evolution/custom_ga.py`): SCORE_REBASE_OFFSET
  anchored to the then-current live best (total_score=-350.35 at gen 5626 of
  v2_recalibrated3_20260714) so the live best reads as ~0; raw total_score is
  no longer printed.
- 2026-07-14, `same_layer_duplicate` repair (`evolution/__init__.py`): no
  dedicated repair existed despite it being one of the 6 hard constraints;
  found missing during a repo sweep after a run's archive got stuck unable to
  promote a fully-filled-but-duplicated genome (cv=4) past a sparser but
  feasible one.
- 2026-07-15, stale scale-factor cache (`run_evolution.py`): a cache computed
  2026-07-06 (viol divisor ~1.8e13) survived the 2026-07-13/14 weight
  recalibration; the stale divisor crushed the viol objective to near-zero
  (a shortcut duplicated across 10/11 layers: ~22.7M raw → ~0.0000012 after
  division). Soft-violation pressure was effectively dead for every run since
  while hard constraints looked "all green". The cache is now keyed on a hash
  of the weight config. Related: random-genome IQR (~1e14-1e15) is a different
  order of problem than real-search variance (extra^2/extra^3 saturation
  terms), so viol is anchored to the seed genome's own magnitude instead.
- 2026-07-15, L0 hold completion (`evolution/__init__.py`): a naive same-key
  toggle→hold swap collides with `momentary_key_reuse` when that physical key
  is already busy elsewhere, so pure fitness pressure can leave a layer
  toggle-only indefinitely; the repair proposes the hold onto a genuinely
  free key.
- 2026-07-12/13, violation-weight "nuclear pass" (`config_v2.yaml`): every
  violation_sub_weights entry was measured against a real gen-7500 checkpoint
  using a contamination-cancelling zero-others method (violations-only
  weights, target term at 1.0 vs all-zero baseline, subtract). Findings:
  `layer_depth_penalty` (~6.7% of viol budget) and `momentary_key_reuse`
  (~4.4%) dominate as intended; every other term was either exactly zero
  (genuinely satisfied hard constraint) or a small-but-real residual.
  `momentary_key_reuse` history: 40000 too weak (~0.003% of viol budget) →
  5e7 still too weak (~2.2%) → 5e8 "worked" by count but contributed 4.1e10,
  175x every other active term combined — the escalate-until-count-drops
  methodology never checks weighted contribution, and a swamping term always
  looks like it's working. Lesson now baked into AGENTS.md: calibrate by
  measured weighted contribution, never by count. The 1.5e11 placeholder
  values (`toggle_back_to_l0`, `mouse_hold_position_conflict`,
  `mouse_layer_depth_penalty`) were copy-paste artifacts, never calibrated.
  Small-residual terms rely on dedicated constructive/repair mutation
  operators rather than strong gradient pressure.
