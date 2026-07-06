# Charybdis Optimizer V2 Handoff

Date: 2026-07-06
Repo: `/home/nos/charybdis/charybdis-optimizer-v2`
Commit: `a0770cb` on `master`
Pushed: `https://github.com/Glx28/charybdis-optimizer-v2`

This handoff covers the full CUDA exact-fitness implementation, performance analysis, and remaining work.

---

## GPU Training Policy (unchanged, strict)

Production training must be GPU-primary. Do not start or continue `run_evolution.py` if CUDA is unavailable or if the active training path would be CPU/Numba-primary. CPU is allowed only for unit tests, static checks, diagnostics, or an explicitly requested CPU-only smoke test. See `GPU_TRAINING_POLICY.md`.

The current code enforces this:
- `run_evolution.py` aborts if `training.require_cuda=true` and `torch.cuda.is_available()` is false.
- `FitnessModel` raises if `require_cuda=true` but the CUDA kernel is unavailable.
- The production `evaluate_batch` path no longer silently falls back to Numba.

---

## What Was Done This Session

### 1. Full CUDA exact-fitness kernel

- Added `fitness/cuda/fitness_kernel.cu` — a complete CUDA C++ port of the Numba `_single_genome` / `_evaluate_batch` logic.
- Added `fitness/cuda_kernel.py` to load the kernel via `torch.utils.cpp_extension.load`.
- Wired CUDA as the primary batch evaluator in `fitness/model.py`.
- Kept Numba as a fallback only for unit tests and explicit diagnostics.

### 2. Hang fix

The original stall was Numba `parallel=True` batch compilation in `fitness/kernel.py::_evaluate_batch`. Changed to `parallel=False`. This is what allowed the run to proceed while the CUDA kernel was being built.

### 3. Fail-fast GPU policy

- Added `require_cuda` flag to `FitnessModel` and `FitnessEvaluator`.
- Production runner (`run_evolution.py`) passes `require_cuda=true`.
- `enforce_training_device_policy` updated to check actual CUDA exact-eval availability.
- Config defaults changed: `training.allow_cpu_exact_validation=false`; removed dead `exact_eval.use_numba` flag.

### 4. CUDA optimizations

- `FitnessModel` now caches precomputed CUDA tensors so each batch evaluation does not rebuild them.
- `CustomGARunner` mini-eval count is configurable via `surrogate.mini_eval_fraction` (default `0.1` = 150 genomes for pop 1500).
- Reuses batch exact-eval scores for the best mini-eval genome instead of re-running single-genome Numba eval.

### 5. Configurable mutation probabilities

All `SwapMutation` probabilities are now configurable from `config_v2.yaml`:
`group_overwrite_prob`, `mouse_workflow_prob`, `l7_access_prob`, `random_assign_prob`, `bulk_assign_prob`, `optional_arrow_drop_prob`, `cluster_app_prob`, `effort_swap_prob`, `smart_duplicate_prob`.

### 6. Tests

- Added `TestCudaExactEvalParity::test_cuda_parity_seed_and_random` in `tests/test_v2.py`.
- Full suite: `pytest tests/ -v` → **76/76 passed**.

### 7. Committed and pushed

Commit `a0770cb` includes all CUDA work plus pre-existing session changes (surrogate CUDA streams, loader LeftAlt/arrow fix, mouse-layer x-position preferences, `SPEED_HANDOFF.md`, `tools/`).

---

## Current State

### No active run

No long-running `run_evolution.py` process is currently active. The last runs were short verification runs under `/tmp/`:

- `/tmp/v2_cuda_500gen/` — 500 gen, pop 1500, completed in 114.2 s, best score `-24.51`.
- `/tmp/v2_cuda_profile2/` — 50-gen profile run.
- `/tmp/v2_cuda_150mini/` — 50-gen profile with 150 mini-eval.
- `/tmp/v2_final_smoke/` — final 10-gen smoke test before commit.

### Verified performance

| Config | Speed | 50-gen best | Notes |
|---|---|---|---|
| Default `config_v2.yaml` | ~4.5–4.8 gen/sec | ~-14 to -16 | Quality-focused default |
| Reduced `cluster_app_prob=0.08`, `group_overwrite_prob=0.08` | ~6.3 gen/sec | -13.68 | Faster, worse convergence |
| Semantic mutations disabled | ~3.4–4.1 gen/sec | -15.47 | Slower and worse |

Full default 500-gen run: **4.7 gen/sec**, total 114.2 s, best `-24.51`.

Hardware: NVIDIA GeForce GTX 1070.

---

## Critical Performance Finding

**The CUDA exact-eval path is not the bottleneck.**

From `cProfile` (50-gen default run):

| Cost center | Per generation |
|---|---|
| `SwapMutation._do` (all mutation passes) | **~127–162 ms** |
| Surrogate `predict` (1500 children) | ~86–98 ms |
| CUDA exact mini-eval (150 genomes) | ~28–30 ms |
| `_layout_reports` / acceptance | ~18–22 ms per event |

The dominant cost is the Python-level semantic mutation operators, especially:
- `_cluster_app_shortcut` (~84 ms/gen)
- `_overwrite_group_as_unit` (~30–35 ms/gen)
- `_thumb_exclude_mask` (~28 ms/gen)
- `_propose_mouse_workflow_layer`, `_bulk_reassign`, etc.

The pre-existing `SPEED_HANDOFF.md` already identified the previous bottleneck as CUDA stream contention between surrogate training and inference. That stream-separation fix is in `evolution/surrogate.py` and is working. With that fixed and exact eval moved to GPU, the next bottleneck is the Python mutation layer.

### Why the old 20–40 gen/sec is not reachable now

The current pipeline does substantially more work per generation than the old fast runs:
- Semantic mutations (`cluster_app`, group overwrite, mouse workflow, L7 access, bulk reassign, arrow drop).
- 10% mini exact-eval beacons every generation.
- Full acceptance report on promising mini-eval genomes.

That design buys per-generation search quality but costs ~200 ms/gen.

To reach 20–40 gen/sec without quality loss, the semantic mutation operators would need a major rewrite in Numba or CUDA. Reducing mutation probabilities trades quality for speed and is now configurable.

---

## Remaining / Next Work

1. **Mutation performance (biggest leverage)**
   - Rewrite `_cluster_app_shortcut`, `_overwrite_group_as_unit`, and `_thumb_exclude_mask` to avoid Python-level per-genome loops.
   - Options: Numba-compiled variants, vectorized numpy over batches, or CUDA kernels.
   - This is the only path to 20–40 gen/sec without sacrificing quality.

2. **Continue long training run**
   - The current best from the 500-gen verification run is `-24.51`.
   - If the user wants a real long run, start one with bounded generations (not `999999` unless explicitly requested).
   - Use the restart snippet below.

3. **Monitor completion cluster and dynamic mouse layer**
   - Existing handoff policy still applies: Norwegian raw completion family should concentrate on one anchor layer.
   - Final acceptance requires a generated non-L0/non-L7 dynamic mouse layer with MB1-MB5 on the right side, right-hand non-thumb momentary Scroll, no mouse button on right-thumb, no right-thumb momentary access to the layer, and reachable toggle access.

4. **CUDA kernel parity / robustness**
   - Seed genome parity is exact.
   - Random-genome differences are float32 accumulation-order noise (effort ~528, adjacency ~0.32, violations ~1.68e7 against raw violations ~1e26).
   - If stricter parity is needed, implement deterministic accumulation (double-precision partials or Kahan) inside `fitness/cuda/fitness_kernel.cu`. This is not required for optimization quality.

---

## Useful Commands

Run tests:

```bash
cd /home/nos/charybdis/charybdis-optimizer-v2
.venv/bin/python -m pytest tests/ -v
```

Quick CUDA smoke:

```bash
cd /home/nos/charybdis/charybdis-optimizer-v2
timeout 600 .venv/bin/python run_evolution.py --generations 10 --pop-size 1500 --output-dir /tmp/v2_smoke --config config_v2.yaml
```

Start a bounded real run:

```bash
cd /home/nos/charybdis/charybdis-optimizer-v2
.venv/bin/python - <<'PY'
import os, subprocess, sys, datetime
root = os.getcwd()
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(root, 'build', 'runs', f'v2_cuda_{stamp}')
os.makedirs(run_dir, exist_ok=True)
log_path = os.path.join(run_dir, 'run.log')
log = open(log_path, 'ab', buffering=0)
proc = subprocess.Popen([
    sys.executable, 'run_evolution.py',
    '--generations', '5000',
    '--pop-size', '1500',
    '--output-dir', run_dir,
    '--config', 'config_v2.yaml',
], cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
with open(os.path.join(run_dir, 'run.pid'), 'w') as f:
    f.write(str(proc.pid))
print('pid', proc.pid)
print('run_dir', run_dir)
print('log', log_path)
PY
```

Monitor active run:

```bash
ps -eo pid,etime,pcpu,pmem,args | rg 'run_evolution.py|PID'
tail -n 80 build/runs/<run>/run.log
```

Checkpoint summary:

```bash
.venv/bin/python - <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
be = d.get('best_exact') or {}
objs = be.get('objectives') or d.get('best_objectives') or []
print('checkpoint', p.resolve())
print('generation', d.get('generation'))
print('objectives', objs)
print('total', be.get('total_score', sum(objs) if objs else None))
print('constraints', be.get('constraints') or d.get('best_constraints'))
PY build/runs/<run>/v2_checkpoint_gen500.json
```

---

## Key Files

- `fitness/cuda/fitness_kernel.cu` — CUDA exact-fitness kernel.
- `fitness/cuda_kernel.py` — loader / Python wrapper.
- `fitness/model.py` — single-source fitness model, CUDA primary.
- `fitness/kernel.py` — Numba kernel (`parallel=False` fix).
- `evolution/custom_ga.py` — custom GA loop, mini-eval logic.
- `evolution/surrogate.py` — surrogate trainer with CUDA stream separation.
- `evolution/__init__.py` — `SwapMutation` and semantic mutation operators (performance bottleneck).
- `run_evolution.py` — training runner.
- `config_v2.yaml` — production config.
- `SPEED_HANDOFF.md` — previous session's detailed speed analysis.

---

## Pitfalls

- Do not revive CPU-primary production training. The code now fails fast; keep it that way.
- Do not reduce `mini_eval_fraction` below `0.1` to chase speed — it removes exact training signal from the surrogate cache.
- Do not judge 50-gen quality as final; this optimizer typically needs thousands of generations.
- Do not trust old analyzer output for raw arrows unless the empty-slot fallback bug is fixed.
- Do not hardcode a mouse layer, scroll layer, or layer numbers other than L0/L7.
- If exported CSV and standalone analyzer disagree, fix the decoder source first.
