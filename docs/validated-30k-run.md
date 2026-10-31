# Validated 30,000-generation run

## Completed continuation: local generations 0–5,000

Run snapshot: `build/runs/validated_5000_20261028T234239Z`. Exactly 5,000
local generations ran from corrected gen25k seed SHA-256
`8cb0e3b38f0d23d46765d5dbdf9ca5e145207581493718fb454528cff81d57bc`, with
fresh optimizer/surrogate state. Reporting range: local 0–5,000, nominal
cumulative 25,000–30,000; not an exact-state resume. Manifest: population
1,500, seed 42. The worker finished at local gen5000; best archive genome came
from gen4806, raw exact score `184618.46875`. Final scheduled checkpoint audit
passes with six zero hard constraints, frozen positions, required relations,
and owner/run contract. Gate artifact: `reviews/generation-5000-gate.json`.

CUDA was the production path on NVIDIA GeForce GTX 1070
(`cuda_surrogate_primary`). CustomGA reports 26,212.3 seconds for 5,000
generations (5.242 s/gen, about 1 h 27 min per 1,000); the exact evaluator's
5,020 batches averaged 0.319 s. This confirms the run was CUDA-enabled but the
full loop remained mostly CPU-bound. See `v2_perf_report.json` and
[`next-run-performance-plan.md`](next-run-performance-plan.md).

Final raw gen5000 export detects one binding mismatch: `Win+Pause` at L2
(1,1) exports transparent. Raw Norwegian/owner acceptance passes with zero
bad literals; the run-contract checkpoint is valid. The review-only final
candidate clears that one unsupported binding, then passes fresh CPU and CUDA
contract audits, Norwegian/owner checks, generated-binding comparison, and all
18 promotion-evidence checks. CPU/CUDA objective delta is `0.00018883`, with
identical six zero constraints. Same-stage CPU comparison against corrected
gen25k seed: repaired candidate `185075.801873`, seed `239910.612684`, delta
`-54834.810811`, 292 positions changed; both pass. Review package is marked
`ready_unapplied`; no apply or propagation occurred.

Artifacts: raw export under `reviews/export-gen5000/`; repaired candidate,
audits, comparison, and hash-chain verification under
`reviews/repair-gen5000-exportability/`; checkpoint evidence and hashes in
`reviews/checkpoint-review-index.json`. Run state remains
`validated_export_pending` for the raw checkpoint mismatch; the separate
review-only repaired candidate is ready for user choice.

Final verification: all ten checkpoints (500 through 5000) have matching saved
checkpoint and passing audit hashes. The Coach owner-contract test passes, and
the two preview endpoints return HTTP 200 on ports 8765 and 8766. Tracked
default layout and warmstart were not changed by final review preparation.

Run snapshot: `build/runs/validated_5000_20261028T234239Z`. It started from the
corrected gen25k seed with fresh optimizer/surrogate
state; it is not an exact-state resume. Intended reporting range: local 0–5,000
and nominal cumulative 25,000–30,000. At 2026-10-29 06:20 CET, worker PID
48650 and supervisor 46954 were live at 99.9% of one CPU core. Latest logged
event then was gen4187; latest saved/audited checkpoint was gen4000.
Checkpoints gen500–4000, every 500 generations, pass current-source CUDA
contract audits with six zero constraints. Gen4000's best genome is from
gen2533; current-source total score is `184620.140268`, violation objective
`184647.484375`. Its audit has 58/59 clusters together and 29/29 ordered
relations passing. The external Norwegian/owner audit passes (15 owner checks;
zero bad literals). Current-source same-stage CPU comparison against the
corrected seed: candidate total `184620.140079`, seed `239910.612684`,
seed-minus-candidate `+55290.472605`, 292 changed positions; both pass the run
contract. Artifact: `reviews/current-l1-seed-vs-gen4000.json`.

Binding equality still finds one assigned `Win+Pause` rendered transparent at
L2 x1,y1, so export binding acceptance remains false. All gen4000 evidence and
hashes are under `reviews/export-gen4000/` and indexed in
`reviews/checkpoint-review-index.json`. Comparison wording now states explicitly
that lower exact penalty score is better; focused regression test passes.

A review-only gen4000 repair removes only that unsupported `Win+Pause` at L2
x1,y1. The corrected candidate passes run-contract, Norwegian/owner (zero bad
literals), and binding audits. Recomputed total `185077.470227`; same-stage seed
`239910.612684`, candidate-minus-seed `-54833.142457`, 292 changed positions.
Candidate SHA `c3c832818e85e2184e5389d737073b54f1602d14584f67fd16304c7d5c38ea37`;
exported binding SHA `0c91ef477bda8af7c138d77ca6733017e13cc28a87248f611995f6420483fba0`.
This diagnostic derivative remains unapplied. Its separate review wrapper passes all 18 hash-chain checks in
`promotion-evidence-verification.json`; verifier only, with no promotion or apply.

At 2026-10-29 06:23 CET, a point sample showed 20% GPU utilization, 4,810 MiB
VRAM used, 53°C, and 13.74 W while worker PID 48650 remained CUDA-resident. This
is a single sample; it does not establish sustained utilization.

At 06:25 CET, worker PID 48650 had 24 threads with the main thread using one
full CPU core; host exposes 6 CPUs. GPU peaked briefly at 79%, then measured 4–5%
across six 2-second samples, with 4,802 MiB VRAM and 53–54°C. CUDA remains active,
but sustained utilization is low. Run preserved.

Checkpoint pacing: gen2500→3000 took 38m15s, gen3000→3500 40m08s, and
gen3500→4000 52m32s. Extrapolation places gen4500 near 06:59–07:08 and
gen5000 near 07:42–08:00 CET; estimate only.

At 06:39 CET, latest log event is gen4187 (checkpoint still gen4000). Gen4087→4187
took 11m53s; estimated gen4500 near 07:12–07:16 and gen5000 near 08:04–08:16 CET.
These remain rough extrapolations.

At 06:51 CET, worker PID 48650 and supervisor 46954 remain live. Latest log
event advanced to gen4287; latest saved/audited checkpoint remains gen4000.
Gen4187→4287 took 9m57s. One GPU sample showed 4% utilization and 4,855 MiB
VRAM; worker remained at 99.9% of one CPU core. This sample does not establish
sustained utilization. The gen4500 checkpoint has not yet been written.
Supervisor inspection confirms it watches each new `v2_checkpoint_gen*.json`,
copies it into `reviews/`, recomputes the run-contract audit, and terminates
production if any checkpoint fails. The separate final exporter audit remains
required after local gen5000.

At 07:03 CET, the worker logged gen4387; gen4000 remains the latest saved and
audited checkpoint. Gen4287→4387 took 11m45s. Worker remains live at about one
CPU core; gen4500 has not yet been written.

At 07:19 CET, local gen4500 checkpoint and automatic run-contract audit pass:
six zero hard constraints, frozen-position and required-relation checks pass.
Independent Norwegian/owner audit also passes with zero bad literals. Raw
binding export has one mismatch, `Win+Pause` at L2 (1,1) exported transparent.
A separate review-only candidate clears that one binding, then passes fresh
run-contract, Norwegian/owner, binding, and all 18 promotion-evidence checks.
Same-stage CPU comparison against corrected gen25k seed: repaired score
`185077.470227` vs seed `239910.612684`, delta `-54833.142457`, 292 positions
changed. No apply or propagation. Artifacts and hashes are indexed under
`reviews/repair-gen4500-exportability/`; `checkpoint-review-index.json` now
records generation 4500. Worker PID 48650 remains live; final 500 generations
and final package are pending.

At 07:25 CET, worker PID 48650 logged gen4587; gen4500 remains the latest
saved/audited checkpoint. The next scheduled checkpoint is gen5000.
Observed intervals: gen4287→4387 11m45s, 4387→4487 10m45s, and
4487→4587 11m19s. At that recent average, local gen5000 is roughly 46–47
minutes away (about 08:12–08:13 CET); estimate only.

At 07:36 CET, worker logged gen4687; gen4500 remains the latest saved/audited
checkpoint. Gen4587→4687 took 10m49s. At this pace, local gen5000 is about
34 minutes away (roughly 08:10 CET); estimate only.

At 07:47 CET, worker logged gen4787; gen4500 remains the latest saved/audited
checkpoint. Gen4687→4787 took 11m06s. The latest 200 generations averaged
10m58s per 100; gen5000 is roughly 23–24 minutes away (about 08:10 CET),
estimate only.

At 07:49 CET, the exact archive improved at local gen4806 to score
`184618.4688`; gen4500 remains the latest saved checkpoint. The scheduled
gen5000 checkpoint will capture this archive and receive the supervisor audit.

At 07:57 CET, worker logged gen4887; gen4500 remains the latest saved/audited
checkpoint. Gen4787→4887 took 9m40s. The final 113 generations are estimated
at about 11 minutes (around 08:08 CET); estimate only.

Review-only post-run repair diagnostics clear only the skipped binding and
recompute the candidate. Gen1500 repaired total score `185077.5217`, same-stage
seed delta `-54833.091`; gen2000 repaired total score `185077.489956`, same-stage
seed delta `-54833.123`. Both have six zero constraints and pass run-contract,
Norwegian/owner, and generated-binding audits. These diagnostics are
unapplied and are not final candidates. Artifacts:
`reviews/repair-gen1500-exportability/` and
`reviews/repair-gen2000-exportability/`.
Do not infer exact live generation from milestone logs.

Manifest confirms cap 5,000, population 1,500, seed 42, warmstart source
generation 25,000, and source checkpoint SHA-256
`df9ac8d9cd73e33e93baf50814110f0c8c8b9cec5267eadd6f01a9c4bccc7d39`.

The corrected seed's current-contract export bundle is independently verified:
all 10 binding/Studio checks and all 18 promotion hash-chain checks pass in
`build/runs/gen25k_repair_work/export/current_contract/`; exported-binding
SHA-256 is `e69240a46e1a64336269593dd5c2816d5dcfd4a33c0557ac6808e6e6f1d87b7b`.
Current-source CPU and CUDA audits for
`build/runs/gen25k_repair_work/v2_checkpoint_gen0_recomputed.json` both pass
with six zero hard constraints and current owner/run-contract acceptance. Both
report total scores `239910.6126835` (CPU) and `239910.6128500` (CUDA), within
`1e-3`; both have violation objective `239939.96875`, matching constraints,
and passing acceptance. Objective max absolute delta is `0.00016785`. Audit
hashes are in `reviews/checkpoint-review-index.json`. The prior recorded total
`239910.61268353462` agrees within this tolerance but does not establish current
validity. `v2_checkpoint_gen0.json` remains historical; its embedded score and
acceptance report are stale. The corrected seed is not applied.

## Earlier run attempts and performance diagnostics

Next-run optimization decision and go/no-go gates are recorded in
[`next-run-performance-plan.md`](next-run-performance-plan.md). Current diagnosis
is CPU archive-audit work, not inadequate CUDA kernel throughput; the live run
remains untouched while it completes.

The prior snapshot `validated_5000_20261028T223133Z` started CUDA production but
stopped at local gen500 when its frozen checkpoint auditor raised `NameError:
name 'Path' is not defined`. The saved gen500 candidate passes the fixed
`run_contract_audit.py`: owner contract and optimizer acceptance pass, all six
hard constraints are zero, and no ordered-relation failures remain. External
export validation remains pending. The failure traceback and candidate are
preserved under that snapshot's `reviews/` directory.

The active snapshot passed 92 focused tests, 261 full tests, CUDA parity,
`just ai-guard`, performance benchmark, and 10-generation CUDA smoke before
production. It started from the corrected gen25k source checkpoint with fresh
optimizer and surrogate state, a 5,000-generation cap, population 1,500, and
seed 42. Audit each 500-generation checkpoint.

Current-run gen500 checkpoint: best generation 497, score `230476.890625`,
58/59 semantic clusters together. Optimizer, hard constraints, frozen-position
comparison, required relative layouts, Norwegian owner audit (15/15), and bad
literal checks pass. Checkpoint SHA-256:
`27b028c6be17c24d9a3364b896b4aabfc2c8a752c5a94fefee74231bb28ff66a`.
External binding comparison has one mismatch: `Win+Pause` at L3 x4,y3 is
rendered transparent by the existing `NON_EXPORTABLE_BASE_KEYS` exporter rule.
Full artifacts: `reviews/contract-audit-gen500.json` and
`reviews/export-gen500/`.

Performance check at local gen535 (2026-10-29): worker remained live at 24m51s
elapsed / 22m07s CPU, about 89% of one core. A 10-second `nvidia-smi dmon`
sample showed 0–1% SM utilization. Source audit of the frozen run confirms one
CUDA exact evaluation of all 1,500 children per generation; CUDA also trains and
serves the surrogate. Parent selection (`_tournament_select`), crossover,
mutation, repair, and survival are CPU-side. The current run is not GPU-bound
and its observed pace is below the intended turnaround. The code's adjacent
`GPU` selection comments are inaccurate: that helper uses NumPy. Do not describe
this as high-GPU-utilization training. No generation-1000 checkpoint had been
written at this observation; gen500 remains the latest fully audited
checkpoint. A five-second `/proc/<pid>/task/*/stat` sample attributed 5.01 CPU
seconds to the main thread and no measurable CPU time to worker threads, despite
the host exposing six CPUs and Numba reporting six threads in a separate
process. This does not identify which Python call owns the time, but confirms
that the live search was effectively serial during the sample. Keep the live
process and Gradle workload untouched while tracking its next checkpoint.
Follow-up observation: local gen545 at 28m28s worker elapsed; supervisor and
worker remain live, state remains `running`, and no gen1000 checkpoint exists.

Short profile diagnostic (separate from the required 5,000-generation run):
same corrected warmstart, population 1,500, 10 generations, CUDA exact
evaluation; reduced bootstrap to 128 teacher samples / 8 epochs to keep this a
short diagnostic. `CustomGARunner` measured 5.46s for its 10-generation loop
under cProfile, with exact-eval batches averaging 0.321s. Startup acceptance
audits were the dominant profile cost: `build_acceptance_report` took 57.7s
over 835 seeded variants under cProfile. Artifacts are in
`build/diag/gen10_profile/`; this diagnostic is not counted toward local
generations 0–5,000. Its small bootstrap and cProfile overhead mean its loop
timing is diagnostic only, not a production throughput guarantee.

Live CPU profile (100 Hz × 5s) located the production bottleneck: archive
candidate handling (`_maybe_update_global_from_batch`) occupied 376/504 sampled
stacks, with `_layout_reports` present in 330/504. The repeated reports audit
exact children that improve the current score but fail archive tiers. Added a
bounded 512-genome report LRU in source and to the live worker; report contents
and archive decisions are unchanged. Focused cache and archive tests pass. A
second live sample after patch showed `_maybe_update_global_from_batch` in
252/500 stacks, `_layout_reports` in 227/500, and `build_acceptance_report` in
64/500 versus 91/504 before. This is a sample-share reduction, not an end-to-end
throughput claim. Runtime patch and hashes are recorded in
`reviews/live-runtime-patches.json`; patch source is alongside it. The live
archive has logged improvement through gen593; gen500 remains the latest
scheduled audited checkpoint. A later live query recorded 11,266 cache hits and
3,280 misses (77.5% hit rate), and the archive subsequently logged improvement
through gen603. Detailed samples and query time are in
`reviews/live-report-cache-observations.json`. The worker remains active and
gen1000 has not yet been audited.

The `validated_5000_20261028T224540Z` worker later terminated with exit `-11`
while I queried in-process cache statistics through GDB. This was my mistake;
the query crashed the worker. Its gen500 checkpoint and audit remain intact;
archive improvements logged through gen623 were not checkpointed and are not
used as final evidence. A subsequent startup attempt,
`validated_5000_20261028T233741Z`, correctly rejected the repository's invalid
generation-4000 global warmstart at the fixed L0 access position before
production. It ran zero production generations.

Replacement snapshot `validated_5000_20261028T234239Z` explicitly seeds from
`seeds/gen25k_corrected_seed.json` (source generation 25,000; SHA-256
`8cb0e3b38f0d23d46765d5dbdf9ca5e145207581493718fb454528cff81d57bc`) with the
source-level acceptance and cluster caches. The immediately prior preflight
on the same optimizer source revision passed 94 focused tests, 263 full tests,
the perf benchmark and AI guard; the replacement snapshot repeated them. Its
10-generation CUDA-primary smoke passed and production started as PID 48650.
Current authoritative run artifacts are under that snapshot's `production.log`,
`state.json`, and `reviews/`; the 5,000 local-generation count starts at 0.

CUDA scratch caps now match the measured layout more closely (320 shortcuts,
16 layers, 32 apps, 96 groups; current use 307/11/19/79), with pre-launch
capacity checks. Exact-eval throughput rose to 10,200 genomes/s from the 6,359/s
baseline; mutation reached 50,192 individuals/s from 33,488/s. The previous
startup-only snapshot `validated_5000_20261028T214559Z` and its contention
failure logs remain preserved; it was not used as the production source.

The previous snapshot, `build/runs/validated_5000_20261028T085946Z`, was
interrupted by host restart. Its state is `interrupted_by_host_restart`; last
saved checkpoint is local 3000 and last logged event local 3174. It is
preserved for diagnostics and will not be exact-state-resumed or counted as the
completed 5,000-generation continuation. Its generation-3000 audit passed
optimizer acceptance, hard constraints, frozen-position checks, and required
relative layouts; archive best was generation 2998, score 184939.453125,
58/59 semantic clusters together, 29/29 order relations.

Before the restart, three llama.cpp servers were stopped at the user's request.
GPU memory fell from 6.6 GiB to 640 MiB. The corrected seed has a separate
current-contract export/promotion evidence bundle at
`build/runs/gen25k_repair_work/export/current_contract/`; all 18 hash-chain
checks pass, CPU/CUDA parity passes, and the seed remains unapplied. See the
bundle README for artifact hashes and the superseded audit records.

Review previews are live in separate Edge windows: corrected gen25k at
`http://127.0.0.1:8766/charybdis-coach/` and rejected gen30k at
`http://127.0.0.1:8765/charybdis-coach/`. Their isolated static data copies are
under `build/previews/`. The rejected export shows its non-exportable Win+Pause
assignment as transparent at L10 (1,1).

Audit every 500 generations. If a saved candidate violates the contract,
preserve diagnostics and stop to fix the implementation. Final deliverable
still requires local generation 5000, full checkpoint audit, exported-binding
hash comparison, required shortcut/effort and Norwegian audits, and a final
comparison against the corrected seed. Do not apply or propagate the candidate.

## Promote or apply a layout

Apply the currently audited candidate with:

```bash
rtk proxy .venv/bin/python tools/apply_layout.py <candidate.json> \
  --checkpoint <checkpoint.json> \
  --audit <contract-audit.json> \
  --export-validation <binding-audit.json> \
  --csv <layout.csv> \
  --apply-js <studio-apply.js>
```

Promotion requires matching checkpoint, owner-contract, audit, CSV, Studio
payload, and exported-binding evidence before writing anything. The script then
re-runs the contract audit. It installs the
genome in tracked `data/default_layout_genome.json` and the local optimizer
warmstart `build/v2_local_search_result.json`, then writes
`build/current_layout.json` and its audit. The first prior local warmstart is
preserved as `build/v2_local_search_result.pre_candidate_apply.json`. New
optimizer runs and validated-run snapshots use this genome by default; an
explicit warmstart or run-directory warmstart still takes precedence. Use
`--cpu` only on hosts without CUDA. Re-run the script with another candidate
path to promote a later result. Candidate JSON must carry `checkpoint_sha256`,
`audit_sha256`, `contract_sha256`, and `exported_binding_sha256` from the same
passing evidence set.

Run from the local WSL repository with its existing `.venv`:

```bash
rtk proxy .venv/bin/python tools/validated_run.py --prepare-only --fresh-start
rtk proxy .venv/bin/python tools/validated_run.py --run-dir build/runs/validated_30k_TIMESTAMP
```

The snapshot includes current tracked/untracked source, all data inputs, the
selected checkpoint warmstart, original warmstart, scale factors, dependency versions, source hashes, HEAD,
and the dirty-worktree patch. The sparse-layer cutoff is removed; reachable
occupancy is diagnostic only. No reset, rebase or cherry-pick is needed.
Historical checkpoints copied into the source build directory are analysis-test
fixtures, not resume inputs. `--fresh-start` disables every saved-genome
injection and starts the entire population from random mutable assignments over
the frozen L0/L7 base. `--warmstart` accepts `best_genome` or `genome` only when
intentionally continuing from an audited layout; it records source and
generation separately.
The frozen config retains its 500,000 ceiling;
the production command explicitly overrides it with 30,000 generations,
population 1,500 and seed 42.

Startup requires at least 4,096 MiB free GPU memory and utilization at most 10%.
Exit 3 means the snapshot is prepared but the GPU gate remains closed. Retry the
same snapshot once the other workload finishes. CUDA remains mandatory; CPU
training is never substituted. Direct focused and full pytest exit codes,
`just ai-guard`, and a direct performance benchmark (contention skips fail) gate
a 10-generation GPU smoke at population 1,500. The redundant `just ai-smoke`
wrapper is skipped because it repeats tests and the performance guard while
suppressing test failures. Only after the direct checks and CUDA smoke pass may
production start. Raw subprocess logs preserve exact failure evidence.

The supervisor copies every observed 500-generation checkpoint into `reviews/`
before the runner prunes old checkpoints. `reviews/progress.json` tracks exact
archive generation, constraint values, acceptance failures and semantic clusters.
Production logs record stages 0, 3,000, 5,000, 7,000, 9,000 and 10,000. Stage
changes reprice the archive and parents and retrain the surrogate with current
labels; raw scores from different stages must not be compared as improvements.
Every scheduled full-population exact refresh replaces stale parent scores and
checks the exact feasible archive. When asynchronous surrogate weights or
normalization change, the live population is repriced before the next
parent/child comparison.
Exact archive ranking cannot trade lower semantic co-location or relative
placement for a shorter acceptance-failure list. PowerToys shortcuts are grouped
by launcher, visual, mouse, and workspace functions instead of one app-wide
cluster. Browser Find excludes Find My Mouse, browser panels exclude clipboard
and version history, window management excludes saved workspaces, and the
numbered Ctrl+1…9 tab grid is separate from tab navigation. Same-stem aliases
such as Ctrl+Y and Ctrl+Shift+Z remain adjacent in the Undo/Redo sequence.
Contextual shortcut groups retain explicit left/right, up/down, before/after,
and sequence geometry. These relations apply to shortcut actions, not raw arrow
keys, which have no mutable-layer placement requirement.

Audit every 500-generation checkpoint for acceptance results, hard constraints,
reachable-layer occupancy diagnostics, semantic co-location, and relative
placement. Scheduled 5,000-generation reviews compare the exact-best archive.
Do not stop for score plateaus, sparse occupancy, or soft cluster trends. Stop
and preserve a run if the archive fails a named hard contract at two consecutive
scheduled reviews without corrective progress; fix the cause before restarting.
Audit-tool errors are recorded and do not count as failed acceptance reviews.
Passing archives continue to the configured generation cap; early acceptance
remains possible after the required final review.
The run-contract gate fails when any required relative layout differs from its
declared offsets. The audit records failed cluster names so a numeric score
improvement cannot hide broken shortcut order.

Semantic split and relative-position scores are added after legacy objective
normalization. Sparse occupancy is diagnostic only, with no count cutoff or
standalone penalty. Production values are recorded in `config_v2.yaml`.

The 10-generation CUDA smoke checks completion on the CUDA-primary path, finite
exact-evaluator scores, and a returned genome. It records acceptance diagnostics
but does not demand a fully accepted layout after only 10 generations. Scheduled
production reviews enforce the hard layout contract and stagnation guard.
Snapshot preparation does not inject or arrange raw arrow keys.

Final reports use the exact-best archive and stage-correct evaluator. The contract
audit reports reachable generated-layer occupancy without a cutoff, ordinary
momentary thumb access with Scroll exemption, semantic clusters, completion keys, dynamic
mouse and Scroll access, L7 access, frozen positions, duplicates, and requested
Windows shortcuts with their live access paths. Occupancy and split critical
clusters remain review evidence without standalone cutoffs. Export validation remains a separate requirement;
a run passing optimizer checks is not automatically approved for firmware use.

### Final review package procedure

After local gen5000 exists and its scheduled audit passes, export into the run's
review directory. Do not use `tools/apply_layout.py` to prepare or apply the
candidate.

```bash
RUN_DIR=/home/nos/charybdis/charybdis-optimizer-v2/build/runs/validated_5000_20261028T234239Z
EXPORT_DIR="$RUN_DIR/reviews/export-gen5000"
/home/nos/charybdis/charybdis-optimizer-v2/.venv/bin/python \
  /home/nos/charybdis/charybdis-tools/runtime/evolved_v2_export/export_and_analyze_linux.py \
  --build-dir "$RUN_DIR" \
  --data-dir "$RUN_DIR/source/data" \
  --out-dir "$EXPORT_DIR" \
  --checkpoint "$RUN_DIR/v2_checkpoint_gen5000.json" \
  --prefix validated_30k_gen5000
/home/nos/charybdis/charybdis-optimizer-v2/.venv/bin/python \
  /home/nos/charybdis/charybdis-tools/runtime/evolved_v2_export/acceptance_check.py \
  "$RUN_DIR/v2_checkpoint_gen5000.json" \
  --data-dir "$RUN_DIR/source/data" \
  --config "$RUN_DIR/source/config_v2.yaml" > "$EXPORT_DIR/norwegian-owner-audit.json"
```

Run `tools/run_contract_audit.py` from the frozen `source/` snapshot and
`tools/export_binding_audit.py` against its gen5000 checkpoint, generated CSV,
Studio apply payload, and matching run-contract audit. Require all exporter and
Norwegian literal checks to pass, then use `tools/compare_checkpoints.py` at
`--cpu` with the recomputed gen25k seed checkpoint and gen5000 checkpoint. The
comparison reports both genomes under one common fitness stage. Package the
candidate with the matching checkpoint, audit, contract, and exported-binding
hashes; verify it with `verify_promotion_evidence`. Keep every artifact under
`reviews/`; do not invoke `apply()` or propagate the Studio script.

### Replacement run checkpoint: local generation 500

Run `validated_5000_20261028T234239Z`, seeded from corrected gen25k SHA-256
`8cb0e3b38f0d23d46765d5dbdf9ca5e145207581493718fb454528cff81d57bc`.
The local generation-500 checkpoint selected best generation 497 (score
`230476.890625`; 58/59 semantic clusters together). The full run-contract audit
passed optimizer acceptance, hard constraints, frozen-position comparison,
and required relative layouts. Checkpoint SHA-256:
`27b028c6be17c24d9a3364b896b4aabfc2c8a752c5a94fefee74231bb28ff66a`.
Owner contract SHA-256:
`619338390f55a5bc8f4464fc4ff8f0283b321e8ee63670cfa010832a84a8e0f7`.
The checkpoint's Norwegian export check remains pending until the separate
exporter audit; no export-validation hash exists yet. Production continued
past local generation 500 after the passing audit.

### Replacement run checkpoint: local generation 1,000

The local generation-1,000 checkpoint passed optimizer acceptance, hard
constraints, frozen-position comparison, required relative layouts, and all
owner-contract checks. Best archive entry remained generation 891 with exact
score `184771.171875`, zero constraints, 58/59 semantic clusters together,
and all 29/29 ordered relations passing. Checkpoint SHA-256:
`8a630443ba1b66fef9d00a9c0663f5f2cb659f1c7678489f5088d4b7066b13ca`.
Owner contract SHA-256 remains
`619338390f55a5bc8f4464fc4ff8f0283b321e8ee63670cfa010832a84a8e0f7`.
The Norwegian owner audit passes (15/15 owner checks; zero bad literals),
while the external binding comparison has one mismatch: `Win+Pause` at L3 x4,y3
is rendered transparent by the existing exporter rule. Full artifacts are in
`reviews/export-gen1000/`.
At this checkpoint diversity injection triggered at population diversity
`0.013`; the async surrogate reported R² `-48663156.0`. This diagnostic is
recorded for review; exact child evaluation and hard-contract audits remain
active. At the observed gen-1187 stagnation milestone, elapsed worker time was
about 69 minutes and the async surrogate R² had recovered to `0.6637`. GPU
utilization samples ranged from 4% to 21%, with the worker's main Python thread
near 100% CPU. No new scheduled checkpoint had been written at that observation.

### Runtime recheck: 2026-10-29 02:15 CET

The production worker (PID 48650) remained live and consumed one CPU core. Its
log was unchanged since local generation 1,287 at 02:08:27; latest scheduled
checkpoint remained generation 1,000. Six `nvidia-smi dmon` samples over ten
seconds were 6%, 10%, 99%, 4%, 5%, and 6% SM utilization. This confirms brief
GPU bursts separated by CPU-bound work, not sustained high GPU use. The run was
left running; no process in the separate Gradle project was touched. Runtime
optimization cannot affect this already-running Python process without
restarting it, so preserve this run and apply any loop-level speedup only to a
future launch unless a safe in-process control is established. `perf` is not
installed in this environment; `py-spy` attach was previously denied by the
host's ptrace restrictions, so there is no permitted live stack profiler to
identify the CPU hot function without changing the run. The gen-1,000
checkpoint contains the best genome and summary fields, but no live population,
RNG, optimizer, or surrogate state; it cannot recover this run's exact state if
the worker exits. A fresh process check confirmed the worker is allowed on CPUs
0–5 and has 24 threads, while only its main thread consumes CPU (100%); the
latest GPU sample was 6%. The loop is effectively serial, so changing source
code or CPU affinity cannot accelerate this already-loaded run safely.

At 02:20:04 CET, the same worker logged local generation 1,387 after the
previous gen-1,287 milestone at 02:08:27. No gen-1,500 checkpoint exists yet.
That 100-generation interval took 11m37s; at the same pace, roughly seven
additional hours would be needed for the remaining 3,613 generations. This is
a rough throughput projection from two logged milestones, not an exact ETA.

At 03:23 CET, worker 48650 and supervisor 46954 remained live; worker used
99.1% of one CPU core. Gen2000 triggered diversity injection at population
diversity `0.001`, injecting 375 perturbations and retaining 150 elites. The
asynchronous surrogate retrain reported R² `-48442320.0`; exact archive best
remained gen1512. The R² is averaged across concatenated objective and
constraint outputs; near-constant constraint heads can make this aggregate
very negative, so it does not isolate objective prediction quality. The
measured diversity collapse is independent evidence. Neither metric indicates
a hard-contract failure.
Gradle/Kotlin processes for the separate project remain active and untouched.

The extended `tools/compare_checkpoints.py` now freshly audits both genomes
and reevaluates them at one common fitness stage; it records required L1
shortcut placements/effort, owner and run-contract checks, fixed mouse and
Scroll positions, grouping, L7 hash, objective deltas, and changed bindings.
An interim CPU comparison of the corrected seed against local gen1000 is saved
at `reviews/interim-seed-vs-gen1000.json`: both pass the run and owner
contracts, all required L1 shortcuts pass, clusters together are 57/59 vs
58/59, and gen1000's same-stage exact score is 55,139.441 lower across 261
changed positions. This is diagnostic only; final comparison will use local
gen5000 after its complete audit.

### Coach runtime check

The Coach project is served on ports 8766 and 8765. Runtime inspection at
`/charybdis-coach/` showed the corrected gen25k preview on 8766 reporting
`L1 mouse contract verified from current bindings.` The rejected gen30k preview
on 8765 reported the L1 mouse contract broken, listing the missing/misplaced
mouse buttons, Scroll bindings, L0 hold, and lock. Both previews use the
Windows Norwegian (`nb-NO`) host configuration and remain labeled by candidate.
Coach and optimizer contract files have identical SHA-256
`619338390f55a5bc8f4464fc4ff8f0283b321e8ee63670cfa010832a84a8e0f7`; the Coach
owner-contract Node test passed (1/1).

```bash
rtk proxy .venv/bin/python tools/run_contract_audit.py build/runs/validated_30k_TIMESTAMP/v2_checkpoint_gen30000.json
```

For an existing snapshot, execute the audit from its `source/` directory using
the original repository's `.venv/bin/python`, so frozen data/config are used.

### Current corrected-data review

The subsequent fresh-logger 5k continuation and its review-only export package
are recorded in `docs/next-run-performance-plan.md` and
`build/runs/validated_5000_20261030T104142Z/reviews/repair-gen5000-exportability/`.
The raw checkpoint remains `validated_export_pending` for one unsupported
`Win+Pause` binding; the separate candidate clears that binding and passes its
hash-chain evidence. Port 8765 now displays that candidate, labeled
`UNAPPLIED`. The Portable Coach page contains no mouse-contract pass/fail line
or scoring/task-completion metric. Nothing was applied to the keyboard.
