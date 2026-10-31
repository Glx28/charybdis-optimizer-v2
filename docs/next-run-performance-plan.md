# Next-run performance plan

## Decision gate: no long run until throughput is explained

The completed run averaged 5.24 seconds/generation (about 87 minutes per
1,000). Its saved performance report attributes 1,599.8 seconds of 26,230.0
seconds to exact evaluation (6.1%). This means roughly 24,630 seconds are not
accounted for by that report. The live CPU samples implicated archive/report
work, but did not time it. The archive prefilter microbenchmark saves only
about 0.1 ms/generation; it is immaterial to a 5.24-second generation.
Calling that a fix would be unjustified.

Do not start another 5,000-generation job until a short instrumented run
accounts for at least 95% of wall time and demonstrates a material gain. Avoid
serial multi-hour trial runs:

1. Run one 20-generation diagnostic on the same host, same seed, population,
   CUDA path, and checkpoint cadence. Capture stage timers from generation 1,
   process CPU by core, GPU utilization/memory, CUDA synchronization and copy
   time, archive/report call counts, cache hit rates, and contract-valid share.
   This is a diagnostic, not an optimization claim.
2. Select the two largest measured costs. Change only those paths. Keep exact
   CPU/CUDA outputs and candidate validity identical. Prefer batch/vector
   processing for Python scans; use bounded CPU workers only if a 20-generation
   A/B shows net wall-time savings. For CUDA gaps, move the repeated population
   operators/checks onto device and retain population and reusable buffers
   there; avoid per-generation host copies and synchronization. Do not enlarge
   the batch until profiling shows device saturation and memory headroom.
3. A/B the optimized build against the recorded diagnostic for 50 generations
   per variant, alternating order to reduce thermal/background bias. Capture
   identical metrics and candidate-quality/diversity measures. If these trials
   do not establish at least 2x end-to-end speedup, continue optimization or
   stop: no 5,000-generation run.
4. Run one 100-generation confirmation only after the 50-generation A/B passes.
   Verify contract and CPU/CUDA parity at every checkpoint. Its measured rate
   becomes the production estimate; no extrapolation from kernel-only numbers.
5. Production, if authorized by these gates, uses checkpoint audits every 100
   generations initially (not every 500), with automatic stop on any contract
   failure. This catches regressions before a long interval of wasted compute.

The current result report does not provide attribution for the unexplained
24,630 seconds. Existing post-run stage instrumentation is useful for a new
process but cannot reconstruct the completed run. First diagnostic must expose
timer names and counts in the live report, and assert their summed duration is
close to elapsed time; otherwise the instrumentation itself fails the gate.

## Evidence and diagnosis

Current 5,000-generation run remains authoritative and must finish unchanged.
Its population is 1,500; CUDA exact evaluation is active. Preflight measured
10,091.8 genomes/s for exact CUDA evaluation and 40,503.3 individuals/s for
CPU mutation. These isolated numbers do not represent complete-generation
throughput.

The production worker has run near one full CPU core on a six-core host, while
GPU samples were usually 0–6% after brief spikes. A 100 Hz, five-second live
CPU sample attributed 376/504 stacks to archive candidate handling and
330/504 to `_layout_reports`; before the bounded 512-genome report LRU, the
same profile showed `_layout_reports` in 330/504 samples. After the live cache
patch, a second sample showed 227/500. A later cache read recorded a 77.5% hit
rate (11,266 hits / 3,280 misses). This reduces repeated work, but it does not
prove an end-to-end speedup. The current source also replaces nested owner-rule
pair scans with precomputed legal edges; a narrow 512-genome microbenchmark
measured 1.10x for that one check before the later inverse-lookup change.

Conclusion: low GPU use is downstream of serial CPU generation and archive
work. Raising GPU batch size alone cannot fix this run's bottleneck. No
interactive profiler was attached to the live process because host ptrace
restrictions rejected it; Nsight Systems and Nsight Compute are installed for
the next process launch.

## Work before another production run

1. Preserve the completed run and its artifacts. No profiling or benchmark
   process may contend with production CUDA work.
2. Capture the instrumented 20-generation diagnostic described above. Record wall-clock time,
   generation throughput, per-stage cumulative/average time, CPU utilization,
   GPU utilization, memory use, archive candidates examined, report-cache hit
   rate, diversity, and contract-valid population fraction.
3. Run Nsight Systems for 30–60 seconds around steady-state generations, then
   Nsight Compute on `evaluate_batch_kernel` only if the trace shows a kernel
   bottleneck. Keep profiling runs short and separate from production.
4. Optimize only measured hot paths. Add a score/feasibility
   prefilter before archive reports; batch and deduplicate candidate genome
   audits; retain the bounded report cache; cache immutable acceptance metadata
   where sound. Preserve exact archive ordering and all hard-rule checks.
5. Measure CPU parallelism only after batching. Python threads are unlikely to
   help Python-heavy report construction under the GIL. Compare vectorized or
   Numba batch audits first; then benchmark a bounded process pool only if
   serialization and startup costs are lower than the measured audit cost.
   Six host CPUs are available, so cap workers at five and leave capacity for
   CUDA/runtime and the desktop.
6. Then reduce GPU idle gaps: persist reusable device input/output buffers to
   avoid per-generation allocation and synchronization; batch fitness and
   hard-rule checks; prototype device-side tournament selection and genome
   operators only if the trace shows host/device gaps remain dominant. Keep
   CUDA as required exact evaluator. Consider CUDA graphs only after buffer
   reuse and correctness are stable.
7. Compare population sizes 1,500, 3,000, and 6,000 in bounded trials. Select
   by valid improvement and diversity per wall-clock hour, not utilization or
   generations alone. Keep the current 1,500 setting unless a larger batch
   improves that measure without unacceptable latency or memory pressure.

## Go/no-go checks

- Use `tools/perf_benchmark.py` and the documented `just ai-guard` path when
  touching its covered hot paths; run only after the active CUDA job ends.
- Compare CPU and CUDA objectives, constraints, archive choice, and owner
  contract on identical deterministic batches. Reject any performance change
  with parity drift or invalid candidates.
- Run focused contract, initialization, mutation, crossover, export, and
  checkpoint audits before throughput trials.
- Require a 50-generation A/B and 100-generation confirmation, both showing at
  least 2x end-to-end throughput over the completed run on the same
  host/settings, before starting another 5,000-generation run. Report both
  throughput and quality/diversity per hour.
- Save the benchmark configuration, source revision/hash, device/runtime
  versions, profiler trace, raw timings, and audit results with the trial.

## Instrumentation added for the next process

`CustomGARunner` now records per-stage cumulative, average, and maximum times
for selection, crossover, mutation, sanitizer, exact evaluation, contract
filter/archive, survivor selection, adaptive mutation, teacher update,
diversity injection, checkpointing, and total generation. It atomically writes
`v2_perf_live.json` at each checkpoint and logs elapsed milliseconds per
generation. This source change does not alter the already-running worker.

## Archive prefilter change

The archive batch path now filters feasible rows below the incumbent score
before sorting, preserving the previous loop's incumbent cutoff and ascending
score order among considered rows. Focused tests cover the incumbent cutoff
and the no-incumbent case. On a pinned CPU core with a
representative 1,500-row synthetic batch, seven medians of 1,000 calls measured
188.6 μs/call for the previous full sort and 85.7 μs/call for the prefilter
(2.20x for this helper). The sample had no score improvements, matching the
stagnant-run case. It saves about 0.1 ms per generation in that case; it is a
small, bounded optimization and does not explain the hour-per-thousand pace.
Production remains untouched; this measurement does not claim a whole-run
speedup.

## Measured follow-up — 2026-10-29

Optimized `generate_random_layouts` after measuring its warmstart teacher-set
path. There are 89 shortcut groups and 28,502 possible group anchors. The old
code filtered those anchors one at a time in Python for every layout, also
rebuilding anchor sets and scanning mutable positions for free slots. The new
code precomputes anchor matrices and mutable-position indexes, then checks
occupied group/access positions with NumPy masks. It preserves anchor order,
the random-choice distribution, displacement order, and corrected seed in
slot zero.

| Measurement | Before | After |
|---|---:|---:|
| 100 warmstart teacher layouts | 26.561 s | 0.617 s |
| 5,000 warmstart teacher layouts | not run | 21.641 s (231 layouts/s) |

Focused tests pass for fresh Norwegian group placement and deterministic
warmstart generation with owner-fixed positions held constant.

Two CUDA loop trials used the corrected gen25k seed, population 1,500, the
configured 5,000 teacher samples, and GTX 1070. The 20-generation run measured
9.369 s in the generation loop (468 ms/gen). The separate 50-generation
confirmation measured 20.594 s (412 ms/gen). Its accounted stages were 20.587
seconds, leaving 7 ms unattributed. Average stage costs: CUDA exact evaluation
318 ms/gen, mutation 50 ms/gen, contract/archive 30 ms/gen, crossover 4 ms/gen,
parent selection 3 ms/gen, survivor selection 6 ms/gen. This source is about
12.7x faster per generation than the completed run's 5.24 s/gen, but the
comparison is not a controlled old-code/new-code A/B: archive caching,
prefiltering, valid warmstart population creation, and other changes also
differ. Do not attribute the whole gain to anchor vectorization.

An overlapping 500 ms `nvidia-smi` sample across the 50-generation process
and its startup measured 34.2% mean GPU utilization, 100% peak, and 3,314 MiB
peak memory. It includes surrogate bootstrap/training and idle periods, so it
does not isolate generation utilization.

The first standard benchmark result (10,068 GPU genomes/s, 41,545 CPU
mutations/s) used the unrelated repository default genome. It overstated the
corrected-seed workload. Updated `tools/perf_benchmark.py` to use the corrected
gen25k seed and the production owner-protected mutation mask. The corrected
workload measured 4,555 GPU genomes/s and 24,194 CPU mutations/s. Rebased
`tools/perf_baseline.json` to that workload, recording its seed hash and why;
the old 2026-10-05 baseline was not an apples-to-apples comparison. The
subsequent standard check measured 4,552 GPU genomes/s and 32,032 CPU
mutations/s, passing against the corrected-seed baseline. The
50-generation loop trial is about 12.7x faster per generation than the old
production run, but it is not a controlled code A/B.

Nsight Systems completed the 10-generation run but failed while converting
the trace (`No GPU associated to the given UUID`). Nsight Compute could not
collect hardware counters (`ERR_NVGPUCTRPERM`: access denied). No usable
Nsight trace or kernel-counter diagnosis exists. Direct timers show the
corrected workload's full CUDA evaluator at roughly 0.33 seconds per
1,500-genome batch. Further device-side work needs profiler access or a
focused component benchmark; aggregate utilization does not prove saturation.

At the measured 50-generation rate, 5,000 generations project to about 34
minutes before checkpoint, refresh, and retraining overhead. This is a
projection, not a production result. No additional 5,000-generation run was
started. Keep the 100-generation confirmation and full acceptance gates before
any future production continuation.

## Layer-aware conjunction scoring — 2026-10-29

The portable `layer-binding-v2` logger records `prev_layer` and the current
`layer` on shortcut events. Aggregation previously reduced shortcut pairs to a
global count, and exact scoring selected one global position per shortcut ID;
when a shortcut had copies on several layers, sequence pressure could attach to
the wrong copy. The local change now retains `by_layer_pair` counts, emits
layer-specific sequence rows, and scores those rows against the matching
source/destination layer bindings in both Numba and CUDA. Legacy aggregate
files without layer pairs keep their global-sequence fallback. The older Python
adjacency factor uses the same observed-layer mapping.

Validation against the live Windows log, written to `/tmp` without replacing
checked-in usage data:

- 207,475 events across 30 days; last logged event `2026-10-29T10:25:50Z`.
- 461 distinct shortcut pairs; all 55,515 sequence observations have source
  and destination layers. 120 pair types originate on L1, totaling 2,598
  observations.
- 188 pairs map to shortcut IDs in the current production layout; all 188
  retain layer-pair data. An 8-genome deterministic batch passed exact
  Numba/CUDA objective and constraint parity with this real usage payload.
- Synthetic duplicate-binding tests prove proximity follows the observed
  layers even when copies on another layer are far apart. Aggregation,
  layer-familiarity, contract, promotion, CUDA scratch, and production parity
  tests passed.
- The combined 225-test batch had one incomplete `CustomGARunner.__new__`
  fixture missing `perf`; after adding `runner.perf = None`, all 57
  `test_run_contract.py` tests passed. The other 168 tests passed in the
  combined batch.
- `just ai-guard` passed. Corrected-gen25k CUDA throughput was 4,277.7
  genomes/s against 4,555.2 baseline (-6.1%, within the 20% tolerance); CPU
  mutation throughput was 28,382.1/s against 24,193.7/s (+17.3%). Baseline was
  not rewritten. The first run rebuilt the CUDA extension; compilation took
  4m22s and is one-time build cost, separate from evaluator throughput.

Portable runtime was checked on Windows at `C:\Users\nos\portable-charybdis`.
`Start-Charybdis.ps1 -Action Restart -NoBrowser` replaced the prior Coach and
logger processes without opening a browser. The Coach is healthy on port 8765;
the active `layer-binding-v2` logger registered 594 hooks with no errors. Its
source update is pushed as Portable Charybdis commit `145db5f`. Process
inspection found exactly one Coach server and one AHK helper. No Charybdis Run
registry entries or scheduled tasks exist; the sole Startup shortcut points
to this checkout's current `Start-Charybdis.ps1`.

The preserved runtime log is 32,383,281 bytes through `2026-10-29T21:33:35Z`.
The corrected aggregator reports 208,158 valid events, 118 unique shortcuts,
30 days, 55,553 observed shortcut transitions, and 94 blind spots. Layer event
totals: L0 68,210; L1 8,856; L2 848; L3 1; L5 880; L7 118,004; L9 2,906; L10
17. Three historic lines contain embedded NUL bytes; aggregation now skips and
reports them instead of counting them as valid events. The eight observed
Norwegian symbol events predate the new physical-scan hooks: seven `-` and one
apostrophe, all on L0. A Windows `nb-NO` `ToUnicodeEx` probe verified the added
physical mappings (`SC00C` → `+`, `SC01A` → `å`, `SC027` → `ø`, `SC028` → `æ`),
and all 32 new hotkey registrations succeeded. Wait for real post-restart use
before treating those symbols as observed behavior.

The changed CUDA kernel rebuilt after removal of a zero-byte stale PyTorch
extension lock left by an earlier interrupted build. Updated CPU/CUDA parity
passed (`1 passed`, 221.73 seconds including the rebuild). No evolution run was
started and no layout was applied; the current applied layout remains untouched.

Scoring regressions pass five focused tests: increasing per-layer familiarity,
low-use layer-copy demotion, exemption of fixed access/mouse/L0-Alt bindings,
observed-layer sequence proximity when shortcuts have duplicates, and keeping
distance pressure for frequent action pairs. The aggregation regression passes
with `@`, `<`, `+`, and Norwegian `å/ø/æ` output attributed to their logged
layers, and verifies malformed-line reporting.

After that rebuild, the exact real-log aggregation was loaded alongside the
corrected gen25k seed (`source_generation=25000`) for an 8-genome parity batch:
616 positions, 467 mapped sequence rows, 118 layer-specific shortcut keys,
and 8 recorded Norwegian-symbol events. CUDA evaluation took 0.687 seconds;
maximum absolute objective error was `0.0002251` and constraints matched
exactly. This was an in-memory evaluator check only; no checkpoint or layout
was written or applied.

## Ctrl+6 L0 placement postmortem — 2026-10-29

The review candidate's `Ctrl+6` at L0 x7,y4 came from seed repair, before the
5,000-generation run. `tools/repair_gen25k_seed.py` moved the L2 current-layer
scroll into L2 x10,y2 with a two-slot swap. That slot held `Ctrl+6`, so the
swap left `Ctrl+6` in the scroll's former L0 x7,y4 position. The later Ctrl+1
through Ctrl+9 group restoration placed another copy at L8 x2,y1. The repair
report records both moves and their coordinates.

The run did not discover or promote the L0 placement: gene 51 (`Ctrl+6` at L0
x7,y4) and gene 462 (the L8 copy) match the warmstart seed and remain identical
in the best genomes at checkpoints 3,000, 3,500, 4,000, 4,500, and 5,000.
The Oct 28 run used a 67,824-event snapshot ending Oct 23; it contained no
`Ctrl+6` usage count. The fresh Oct 29 aggregate also contains no `Ctrl+6`
events, so there is no observed layer usage supporting this placement. The
run's result therefore reflects a seed-repair placement that survived scoring,
not evidence that `Ctrl+6` deserved an L0 position. The independent scoring
weakness remains: the run did not move an unused, low-importance binding out
of that position.

Fixed the seed repair operation. When the L2 scroll target contains the
required `Ctrl+6` group member, repair now moves that member directly to its
L8 group slot and cycles the old slot contents into the vacated scroll
position. Regeneration passes the complete owner contract and optimizer
acceptance; the corrected report leaves L0 x7,y4 empty and keeps the group
member at L8 x2,y1. The versioned seed and correction report were refreshed.
The applied keyboard was not changed. The earlier prepared snapshot using the
prior seed is stale and must not be launched.

Prepared replacement snapshot
`build/runs/validated_5000_20261029T214731Z` from the corrected seed. Manifest
records local cap 5,000, warmstart source generation 25,000, seed SHA-256
`43726c6f67979547a8cc54c9df3cba98de1d9e444dce8ffbc08efc143c7ebb5c`, and live
logger SHA-256 `2404505fb61cbff1f9c2a624fc9806cde0d1edf14582a16e0013ce48c22fd0dd`
(208,158 events, through `2026-10-29T21:33:35Z`). Snapshot state is `prepared`
with `production_started=false`. Current GTX 1070 free memory is 2,119 MiB,
below the 4,096 MiB startup gate. No run was launched; rerun readiness and
short-run confirmation before starting the 5,000-generation continuation.

After the objective and ranking fixes, `just ai-guard` passed using the
corrected gen25k seed: CUDA evaluator 4,563.7 genomes/s against 4,555.2/s
baseline (-0.2% slower); CPU mutation 29,822.7 individuals/s against
24,193.7/s (+23.3% faster). This is the documented throughput check, not a
generation run. The latest prepared snapshot remains
`build/runs/validated_5000_20261029T220016Z`; its source manifest matches the
tested CPU/CUDA code. GPU readiness is still 2,158 MiB free, so production has
not started.

## L0 opportunity cost and archive precision — 2026-10-29

The L0 thumb occupancy cost was calculated but added to `access_layout`, which
feeds the large violation objective. On that historical scale snapshot, adding
a zero-use `Ctrl+6` copy at L0 x7,y4 changed the effort channel by only
7.15e-7. The archive loop also converted `batch_F[i].sum()` from float32,
losing some small differences. Neither fact alone proves why the binding
survived; the checkpoint was selected by the global exact archive, not merely
because it was newest. The later measured cause is that low static importance
and a weak, globally relieved thumb opportunity cost gave the unused L0 copy
almost no score consequence.

The smoothly usage-relieved occupancy cost remains in the effort objective.
The current scorer also adds an L0-thumb opportunity cost in score units,
relieved only by L0-specific observed use. A synthetic test verifies that
L1-only use does not count as L0 familiarity. Archive candidate totals now sum
float32 channels in float64.

Validation: eight focused scoring tests passed for low-use placement, mature
binding familiarity, layer-specific sequence proximity, and
precision-preserving score/archive ranking. The focused CUDA check rebuilt the
extension from the changed kernel and verified low-use versus mature placement
preference plus CPU/CUDA objective and constraint parity on synthetic genomes.
No production run started. Snapshot
`build/runs/validated_5000_20261029T214731Z` predates these fitness changes and
is stale. Prepared current snapshot
`build/runs/validated_5000_20261029T220016Z` from the corrected gen25k seed and
same 208,158-event logger export. Its manifest hashes match current
`fitness/kernel.py`, `fitness/cuda/fitness_kernel.cu`, `fitness/evaluator.py`,
`evolution/custom_ga.py`, `run_evolution.py`, and `tools/repair_gen25k_seed.py`.
State is `prepared`, `production_started=false`, cap 5,000; warmstart source
generation 25,000. GTX 1070 currently has 2,158 MiB free, below the 4,096 MiB
training gate. No run was launched.

## Scoring correction, archive speed, refreshed run inputs — 2026-10-30

Rechecked the historical gen5k `Ctrl+6` placement against the old checkpoint
and frozen 30-day logger export. The checkpoint says `best_source=global_exact_archive`,
`best_generation=4806`; there is no evidence it was selected as the latest
wild mutation. `Ctrl+6` had no observed events or layer counts and importance
0.19. Under the prior scorer, removing its L0 x7,y4 copy improved total score
by only 7.15e-7, while moving it to an empty high-effort position was worse.
The L0 thumb term used global usage relief and was normalized beside a very
large violation objective.

The corrected L0 thumb penalty is a soft, layer-local occupancy cost. L0 use
reduces it; use on L1 or another layer does not. On the historical layout,
removing or moving the unused Ctrl+6 copy from the L0 thumb now improves score
by 5.14. The earlier effort cost remains. Added coverage for L0-only relief.

Found two additional consistency issues. The archived scale cache was dated
2026-08-31, unversioned, and reused from the shared `build/` directory despite
changes to evaluator code and data. Scale caches now live under each run
output and carry a fingerprint of config, evaluator source, and data. Sampling
uses the active generation-zero objective landscape; the training schedule
then increases cluster pressure without changing those fixed scales. For the
frozen prior inputs, the new cached violation scale is
523,860,986,298,368, versus the stale 323,001,352,192. Single-genome evaluation
now uses the same batch scoring path and reference model as population
evaluation. Archive candidate totals now sum in float64.

The repeated Norwegian completion report was built for every archive candidate.
Added a compiled fast acceptance check and retained full reports only for
candidates that survive the prefilter. `tests/test_archive_fast_prefilter.py`
compares it with the full report on the corrected seed and randomized genomes.
Focused checks passed: 5 archive-prefilter tests; 7 scoring, familiarity,
sequence, CUDA parity, and evaluator/cache tests.

100-generation diagnostic, CUDA surrogate-primary, population 1,500, seed 42,
from the earlier frozen 208,159-event input:

| Measurement | Before fast prefilter | After fast prefilter |
|---|---:|---:|
| Generation time | 62.94 s / 100 | 55.95 s / 100 |
| Exact evaluation | 29.40 s | 36.86 s |
| Contract/archive stage | 27.71 s | 6.48 s |
| Norwegian cluster prefilter | 22.95 s | 0.02 s |

These are diagnostic measurements, not a controlled whole-run A/B: archive
scoring and candidate ranking also changed between runs. The optimized
prefilter check matches the full report in focused tests. The gen100
checkpoint passes the run contract with six zero constraints; external export
validation remains pending. Neither diagnostic changed the applied keyboard.

The current live log was refreshed from
`C:\Users\nos\portable-charybdis\runtime\shortcut_usage.jsonl`, SHA-256
`421ff72d602c540c1be80f6e8ce9d65703570d0ff2b86309b1e896f326d3c48b`:
208,352 events through `2026-10-30T07:22:48Z`, 119 unique shortcuts. Aggregation
reports 93 blind spots and three malformed JSON lines skipped. Norwegian
physical-output events total 21 across five outputs; the new scan-code samples
include eight `SC035` dashes, one `SC033` comma, and one `SC034` period. No
`@`, `+`, `å`, `ø`, or `æ` event was observed in this log window.

Prepared `build/runs/validated_5000_20261030T073002Z` from the corrected gen25k
seed (SHA-256 `7e63415110a4b9ab6c1c349539ce6e4c9d49aa778daa97b18524e5283a2a29b0`)
and that refreshed log. CPU seed audit passes the owner/run contract and all
six hard constraints. Export binding validation is still pending. State is
`waiting_for_gpu`, `production_started=false`; the last readiness sample had
2,267 MiB free versus the 4,096 MiB gate. `validated_run.py --run-dir` returned
before tests or startup smoke because of this gate. Gradle was left running.
No 5k production run started and no layout was applied.

## Corrected-data continuation — 2026-10-30

The first production attempt,
`validated_5000_20261030T095408Z`, completed 5,000 local generations but froze
the older 67,824-event aggregate ending Oct 23, despite the newer Windows log
being available. Keep it as diagnostic evidence only.

The authoritative continuation is
`build/runs/validated_5000_20261030T104142Z`. It ran local generations 0–5,000
from the corrected gen25k seed with fresh optimizer/surrogate state, nominal
cumulative 25,000–30,000. Its frozen logger input contains 208,672 events from
2026-09-29 13:37:07Z through 2026-10-30 09:47:21Z (raw SHA-256
`9c082ba5e474efb925fec9c11eb3dfcf963b13c9f9e9323dcebccabae2e1e7e2`), 31 days,
and 469 shortcut transitions. Aggregated usage SHA-256 is
`6b7b9e81f1fda6da31d469eae4d3ee043b86938caf92f8a1ddbdd0e95cd3193a5ed8`.
The snapshot retains per-layer shortcut counts; e.g. Ctrl+C was observed 430
times on L0 and 19 times on L1, while Ctrl+V was 706 on L0 and 62 on L1. The
raw log contains 22 Norwegian-symbol events; it has no events for `@`, `+`,
`å`, `ø`, or `æ` in this period.

All ten saved checkpoints (500–5,000) pass run and optimizer contracts with
six zero hard constraints. Archive best is generation 4,998 with raw score
`69523.8919`; gen5,000 has 58/59 sequence clusters together and 29/29 required
ordering relations. The GTX 1070 run took 1,830 seconds at 2.732 generations/sec
(366 ms/gen); sampled training utilization ranged 29–100% with about 4.5 GiB
allocated. Pre-run validation passed 293 tests, CUDA benchmark, and 10-gen CUDA
startup smoke.

Raw export reports one unsupported `Win+Pause` at L10 x0,y2 as transparent.
The raw checkpoint remains unchanged. A separate review-only derivative clears
that binding and passes CPU/CUDA contract audits, the Norwegian/owner audit
(zero bad literals), all 616 generated-binding checks, and all 18 promotion
hash checks. CPU/CUDA objective delta is at most `0.0005112`; both report six
zero constraints. Same-stage CPU comparison: corrected seed `178931.4259`,
candidate `71185.0776`, delta `-107746.3482` (60.2% lower), across 404 changed
positions. All 13 hard-required L1 shortcuts pass at effort `<= 1.0`; L7 stays
frozen.

Review package:
`build/runs/validated_5000_20261030T104142Z/reviews/repair-gen5000-exportability/`.
The existing Windows Coach endpoint on port 8765 now serves the 616-row
review-candidate CSV and labels it `UNAPPLIED`. The current Portable Coach
source is unchanged; candidate data is local preview state. The previous
preview is backed up in the package. No keyboard layout was applied. Run state
remains `validated_export_pending` because the raw archive export contains the
unsupported binding; the separate review-only derivative is ready for a layout
choice.

## Usage-log padding audit — 2026-10-30

The live Windows logger remains active (`LoggerVersion=layer-binding-v2`, 594
hooks, one AutoHotkey helper process). Re-aggregation found three records with
leading NUL padding. Each decoded to a valid `startup` event (`logger_init`)
from Oct 12, 17, and 22; no shortcut or symbol usage was lost. The aggregator
now recovers only leading-NUL-prefixed lines whose remainder parses as complete
JSON and reports recovered and malformed counts separately. Focused synthetic
coverage confirms padded valid records are recovered and truncated invalid
records remain excluded.

The actual Windows `Start-Charybdis.ps1 -Action Start -NoBrowser` path was
exercised against the already-running stack. It replaced the prior Coach and
AutoHotkey processes, restarted Coach on port 8765, and registered 594 logger
hooks. A follow-up CIM query found exactly one Portable Coach server and one
logger process; `-Action Status` reported both running. Deployed startup/logger
source hashes matched this repository before the check. A fresh `startup` event
appeared in the shared usage log and aggregated successfully. The refreshed
aggregate contains 208,676 records: 3 recovered padded startup events plus the
new startup event account for its four-event increase over the run snapshot.
A full stats diff found all optimizer-relevant keyboard usage, layer counts,
sequences, and symbol counts unchanged; only the synthetic `Logger_init`
shortcut/app counts, overall L0 event totals, timestamps, and malformed/recovery
counters differ. No browser opened and no keyboard layout changed.
