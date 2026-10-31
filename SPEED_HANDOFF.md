# Optimizer performance handoff — current

Current implementation and active-run status. Detailed historical diagnosis is
preserved in [the July 5 handoff](docs/archive/SPEED_HANDOFF_2026-07-05.md).

## Current execution path

- Production requires CUDA and uses the CUDA exact evaluator plus GPU-primary
  surrogate training. CPU-primary fallback is forbidden.
- Child batches are exact-evaluated once; those objective and constraint values
  drive survivor selection and owner-contract filtering. There is no CPU
  mini-evaluation subset in the active optimizer.
- Surrogate exact-evaluation cache is bounded by `max_retrain_samples`.
- Surrogate inference and asynchronous training use separate CUDA streams.
- CUDA exact evaluation uses one thread per candidate. Per-candidate scratch
  capacities now match this layout more closely: 320 shortcuts, 16 layers,
  32 apps, and 96 key groups, guarded before launch. This cuts oversized
  scratch initialization and memory traffic while retaining headroom over the
  measured layout (307 shortcuts, 11 layers, 19 apps, 79 groups).
- Candidate mutation and much genetic-operator work remain CPU/Numba. Simple
  GA selection also remains CPU; moving it to CUDA was measured at 1.08 ms per
  call versus 0.074 ms in NumPy on this GTX 1070, so that change was rejected.

## Active 5,000-generation continuation

Current snapshot: `build/runs/validated_5000_20261028T224540Z`, local
generations 0–5,000 (nominal cumulative 25,000–30,000), population 1,500, seed
42. Fresh optimizer/surrogate state from corrected gen25k checkpoint
`df9ac8d9cd73e33e93baf50814110f0c8c8b9cec5267eadd6f01a9c4bccc7d39`.
Startup passed focused/full tests, CUDA parity, ai-guard, performance benchmark,
and 10-generation CUDA smoke before launch. The prior production attempt stopped
at gen500 because its frozen checkpoint auditor missed a `Path` import. The
saved checkpoint passes the fixed owner audit; the active snapshot includes the
auditor fix and runs from gen0. See [the validated run record](docs/validated-30k-run.md).

Measured after CUDA scratch-capacity reduction: exact evaluation 10,200
genomes/s versus the 6,359/s baseline; mutation 50,192 individuals/s versus
33,488/s. Direct evaluation of 1,500 candidates fell from about 251 ms to
about 147 ms. The CUDA kernel remains the largest measured component; CPU
post-processing adds about 22 ms per batch.

## Performance measurement

The performance comparison above was measured before production started. Do
not rerun `tools/perf_benchmark.py` against the shared GPU while production is
active: it would contend with the run. The optimized kernel is included in the
active frozen snapshot and passed the startup smoke.

## Before a future production launch

For later runs, preserve the 5,000-generation cap, 500-generation audits,
corrected seed, and fresh optimizer/surrogate state. Re-run tests, performance
checks, and CUDA smoke before launch.
