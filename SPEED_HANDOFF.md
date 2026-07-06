# Optimizer Speed Handoff — 2026-07-05

## Problem: ~1 gen/sec instead of ~29 gen/sec

The optimizer is running at ~1 gen/sec. The previous session measured 29.5 gen/sec.
This handoff documents exactly what is slow and what the correct fix is.

---

## What was already measured (do not re-measure)

**Exact eval (Numba, CPU):** 150 genomes = 47.8ms = 0.32ms each  
**Surrogate predict (GPU):** estimated ~5-8ms for 1500 genomes  
**Gen rate:** ~1 gen/sec = 1000ms/gen (checkpoints: gen500→gen1000 = 10 minutes)  
**GPU utilization:** only 20% (should be much higher)

---

## Root cause: GPU CUDA stream contention

The async surrogate retrain and the main-loop `predict()` share the **same default CUDA stream**.

### The blocking chain (every ~300 gens):

```
Background thread: trainer.train_on_copy(...)
  → 40 epochs × batch_size=8192 → ~2 seconds of CUDA kernels
  → All queued into the DEFAULT CUDA stream

Main GA thread: sm.trainer.predict(children_X)
  → surrogate forward pass on same default CUDA stream
  → pred = self.surrogate(X).float().cpu().numpy()
                                          ↑
                               .cpu() is a HARD SYNC:
                               blocks until ALL pending CUDA work clears
                               including the 2 seconds of training kernels
```

**Result:** every `predict()` call blocks for up to 2 seconds waiting for training to clear.

### Why it compounds:
- retrain_every=300, mini-eval runs every gen (150 genomes)
- During 2-second training hold: ~33 gens try to predict, each blocks
- Net: ~60ms true gen work + ~940ms GPU sync wait = ~1000ms/gen

---

## The correct fix (do NOT reduce anything to CPU)

### Fix A — Separate CUDA streams (proper fix)

Give training and inference their own streams so they don't block each other:

```python
# In SurrogateTrainer.__init__:
self._inference_stream = torch.cuda.Stream()
self._training_stream = torch.cuda.Stream()

# In predict():
with torch.cuda.stream(self._inference_stream):
    pred = self.surrogate(X).float()
    # Use stream-aware sync instead of blocking .cpu():
    pred_cpu = pred.cpu()  # only syncs inference_stream, not training_stream
return pred_cpu.numpy() * self.std + self.mean

# In train_on_copy():
with torch.cuda.stream(self._training_stream):
    # all training kernels go to training_stream
    ...
```

This lets training and inference run concurrently on separate GPU hardware queues.

### Fix B — Config reduction (partial workaround, not proper)

Reducing `retrain_epochs: 40→10` and `batch_size: 8192→1024` reduces training hold-time from
~2000ms to ~150ms. Less blocking but still wrong architecture. The streams fix is correct.

**Do NOT reduce mini-eval from 150 to 50 — that removes exact data from surrogate training
and moves computation away from GPU.**

---

## Secondary bottleneck: CPU mini-eval (NOT the main issue)

Each gen, 150 genomes are evaluated on **CPU (Numba)** taking 47.8ms.
This runs concurrently with GPU predict via a background thread:

```python
mini_future = executor.submit(evaluator.evaluate_batch, mini_batch)  # CPU, background
children_F = sm.trainer.predict(children_X)                          # GPU, main thread
mini_F, mini_G = mini_future.result()                                # wait for slower one
```

If GPU predict takes 5ms and CPU eval takes 47.8ms: net bottleneck = 47.8ms → cap ~21 gen/sec.

**This is NOT the current problem** (current is 1000ms/gen, not 47ms/gen).
After fixing the CUDA stream issue, this becomes the next bottleneck.

The fix for this: move exact eval to GPU (requires rewriting Numba kernel in PyTorch/CUDA),
or accept ~21 gen/sec and focus on layout quality. Do not reduce mini-eval size.

---

## What was changed this session (kernel fix — keep this)

**`fitness/kernel.py`** — MB1/MB2/scroll x-position preferences on mouse layer:

```python
# New array (line ~652):
scroll_right_momentary_x = np.full(32, -1.0, dtype=np.float32)

# Tracking (line ~917):
scroll_right_momentary_x[layer] = pos_x[i]

# Candidate mouse layer penalty (after line ~1716):
if mouse_button_right[layer, 1] > 0 and mouse_button_right_thumb[layer, 1] == 0:
    candidate_penalty += abs(mouse_button_x[layer, 1] - 8.0) * 12000.0   # MB1 → x=8
if mouse_button_right[layer, 2] > 0 and mouse_button_right_thumb[layer, 2] == 0:
    candidate_penalty += abs(mouse_button_x[layer, 2] - 9.0) * 12000.0   # MB2 → x=9
if scroll_right_momentary[layer] and scroll_right_momentary_x[layer] >= 0.0:
    candidate_penalty += abs(scroll_right_momentary_x[layer] - 10.0) * 12000.0  # scroll → x=10
```

**Goal:** Force mouse layer to have MB1 at index finger (x=8), MB2 at middle (x=9),
scroll at ring finger (x=10). Currently MB1 is at pinky (x=11). 71/71 tests pass.

**Warmstart:** `build/v2_local_search_result.json` contains the gen19500 best genome.
**Scale factors:** `build/v2_scale_factors.json` has been deleted (will recompute on restart).

---

## Current optimizer state

- PID: `ps aux | grep run_evolution | grep -v grep`
- Log: `/tmp/run_v20.log`
- The optimizer may still be running (PID 124710/124712) with old config — kill before restart
- Best genome: gen 4379, gap=-5.07 (warmstart written to build/v2_local_search_result.json)
- G=[0,0,0,0,0] — all hard constraints satisfied
- BAD: Ctrl+S at eff=1.25, MB1 at pinky (x=11) instead of index (x=8)

### Restart sequence:
```bash
kill 124710 124712
rm -f build/v2_scale_factors.json
nohup .venv/bin/python3 run_evolution.py > /tmp/run_v20.log 2>&1 &
```

---

## Files to fix for the CUDA stream issue

| File | What to change |
|---|---|
| `evolution/surrogate.py` | Add `_inference_stream` and `_training_stream` in `__init__`; wrap `predict()` in inference stream; wrap `train_on_copy()` in training stream |
| `evolution/surrogate.py` line 226 | `.cpu().numpy()` → use stream-synchronized copy |

**Files NOT to change:** `fitness/kernel.py` (keep MB1/MB2/scroll fix), `config_v2.yaml` (keep current), `evolution/custom_ga.py` (keep 150 mini-eval)

---

## Expected result after CUDA stream fix

- Training and inference run in parallel on GPU
- predict() no longer blocks on training kernels
- Gen/sec: ~21 gen/sec (then limited by CPU mini-eval at 47.8ms)
- If CPU mini-eval is also moved to GPU: potentially 100+ gen/sec

