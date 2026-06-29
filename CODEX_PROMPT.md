# Codex Task Assignment: Charybdis V2 Surrogate Removal & Exact Eval Mode

## Context

The Charybdis V2 keyboard layout optimizer is stuck. The surrogate model has collapsed (R² = -0.1 to 0.5, corr = nan). The population has been frozen at the same genome for 3,800+ generations. The root cause is clear: a neural network trained on ~1 sample per dimension cannot learn a meaningful fitness landscape.

**The fix:** Disable the surrogate entirely. Use exact evaluation via Numba batch evaluator for every generation. With Numba at ~0.5ms per layout and pop=500, exact eval of the full population takes ~250ms per generation — fast enough for real search.

## Your Tasks (Codex)

### 1. Modify `run_evolution.py` to Support `surrogate.enabled: false` Mode

The current `main()` function unconditionally builds and trains the surrogate. You need to add a `surrogate_enabled` branch that:

**When `surrogate.enabled: false` in config:**
- **Skip** all surrogate creation, training, and `SurrogateManager` setup
- **Skip** the `n_initial_samples` random layout generation for surrogate training
- **Still compute** `scale_factors` from a small sample (e.g., 100 random layouts) for normalization — this is needed for the objectives to be comparable
- Create `FastLayoutProblem` with `use_surrogate=False`, passing `layout`, `evaluator`, and `batch_evaluator` so `_evaluate` calls `evaluate_exact_batch()` instead of `trainer.predict()`
- Use `ExactEvalCallback` instead of `SurrogateCallback`
- The `ExactEvalCallback` should handle:
  - Adaptive mutation rate (increase on stagnation, restore on improvement)
  - Group repair every 10 generations
  - Checkpointing every N generations
  - **No** surrogate drift metrics, no retraining, no exact eval batch (everything is already exact)

**When `surrogate.enabled: true` (default):**
- Keep existing behavior unchanged

### 2. Update `config_v2.yaml`

Add `surrogate.enabled: false` and clean up the config:
```yaml
evolution:
  pop_size: 500
  n_generations: 999999
  crossover_prob: 0.7
  mutation_prob: 0.15
  seed: 42
  inject_seed: false
surrogate:
  enabled: false  # <-- ADD THIS
  hidden_dim: 256
  # ... rest stays
exact_eval:
  use_numba: true
  parity_tolerance: 1e-4
fitness:
  weights:
    effort: 1.0
    adjacency: 1.5
    finger_balance: 0.8
    same_finger: 2.0
    violations: 50.0
    workflow_coherence: 30.0
    trackball_proximity: 5.0
    app_coherence: 5.0
    learning_curve: 0.5
output:
  build_dir: "build"
  checkpoint_interval: 500
  verbose: true
```

### 3. Verify the `ExactEvalCallback` Class

I already added a basic `ExactEvalCallback` class to `run_evolution.py`. Verify it:
- Has `_adjust_mutation_rate`, `_repair_best_groups`, `_save_checkpoint`, `_cleanup_old_checkpoints`, and `notify` methods
- `notify` correctly gets `gen = algorithm.n_iter` and calls the sub-methods
- Works with pymoo's `Callback` interface

### 4. Verify `FastLayoutProblem._evaluate` for Exact Mode

I already modified `FastLayoutProblem` to accept `use_surrogate` flag. Verify:
- When `use_surrogate=False`, it calls `evaluate_exact_batch(x, self.layout, self.evaluator, batch_evaluator=self.batch_evaluator, ...)`
- It sums the 3 objectives into a single total: `total = scores.sum(axis=1, keepdims=True)`
- The output shape is `(batch, 1)` for single-objective NSGA2

### 5. Test the Changes

Run these commands and verify they pass:
```bash
cd C:/Users/nos/charybdis-optimizer/charybdis-optimizer-v2
../charybdis-optimizer/charybdis-optimizer-v2/.venv/Scripts/python.exe -m pytest tests/test_v2.py -v
```

And verify Numba parity:
```bash
cd C:/Users/nos/charybdis-optimizer/charybdis-optimizer-v2
../charybdis-optimizer/charybdis-optimizer-v2/.venv/Scripts/python.exe -c "
from core.loader import build_layout
from fitness.evaluator import FitnessEvaluator
from fitness.batch_evaluator import BatchExactEvaluator
layout = build_layout('../charybdis-optimizer/build')
evaluator = FitnessEvaluator()
batch = BatchExactEvaluator(layout, evaluator, validate=True)
print(f'Parity: {batch.parity.ok} (max_diff={batch.parity.max_abs_diff:.6g})')
print(f'Enabled: {batch.enabled}')
"
```

### 6. Copy Updated Files to Run Directory

After all changes are verified, copy them to the actual run directory:
```bash
cp C:/Users/nos/charybdis-optimizer-v2/run_evolution.py C:/Users/nos/charybdis-optimizer/charybdis-optimizer-v2/run_evolution.py
cp C:/Users/nos/charybdis-optimizer-v2/fitness/batch_evaluator.py C:/Users/nos/charybdis-optimizer/charybdis-optimizer-v2/fitness/batch_evaluator.py
cp C:/Users/nos/charybdis-optimizer-v2/fitness/factors/effort.py C:/Users/nos/charybdis-optimizer/charybdis-optimizer-v2/fitness/factors/effort.py
cp C:/Users/nos/charybdis-optimizer-v2/fitness/factors/violation.py C:/Users/nos/charybdis-optimizer/charybdis-optimizer-v2/fitness/factors/violation.py
cp C:/Users/nos/charybdis-optimizer-v2/core/__init__.py C:/Users/nos/charybdis-optimizer/charybdis-optimizer-v2/core/__init__.py
cp C:/Users/nos/charybdis-optimizer-v2/evolution/__init__.py C:/Users/nos/charybdis-optimizer/charybdis-optimizer-v2/evolution/__init__.py
```

And copy the config:
```bash
cp C:/Users/nos/charybdis-optimizer/build/config_v2.yaml C:/Users/nos/charybdis-optimizer/build/config_v2.yaml.bak
cp C:/Users/nos/charybdis-optimizer-v2/build/config_v2.yaml C:/Users/nos/charybdis-optimizer/build/config_v2.yaml
```

## Important Notes

### The Constraint "Wiring" Issue is Already Fixed

The analysis file claims `hand_bias.py`, `arrow_order`, and `mouse_layer_access` are not wired. **This is incorrect for the current code.** These were all moved INTO `ViolationFactor` as sub-methods:
- `ViolationFactor._hand_bias()` — weight 2000
- `ViolationFactor._arrow_order()` — weight 200  
- `ViolationFactor._mouse_layer_access()` — weight 5000

All are called from `ViolationFactor.compute()` which IS called by `FitnessEvaluator.evaluate()`. The analysis was from the OLD running process (gen 1000), not from the updated code.

**Do NOT create a separate `HandBiasFactor` class.** The constraints are already inside `ViolationFactor` where they belong (they are violation-type penalties, not separate objectives).

### Numba is Already Fixed

I already removed `fastmath=True` from the `@njit` decorator. The Numba batch evaluator is working (parity: True, max_diff=64). Do not change the Numba code.

### Layer Access Cost is Already Implemented

The `EffortFactor` now computes `layer_access_cost` for each position based on the access path from L0. This is already in the code. Do not change it.

### What NOT to Do
- Do NOT create new factor files (`hand_bias.py`, `arrow_order.py`, etc.)
- Do NOT modify `ViolationFactor` — the sub-methods are already there
- Do NOT modify `fitness/batch_evaluator.py` — Numba is already working
- Do NOT modify `fitness/factors/effort.py` — layer access cost is already implemented
- Do NOT modify `core/__init__.py` — `get_occupied_thumbs` is already fixed
- Do NOT kill the running process — Kimi will handle that

## Acceptance Criteria

1. `run_evolution.py` runs without errors when `surrogate.enabled: false`
2. `pytest tests/test_v2.py` passes all 10 tests
3. Numba parity check shows `Parity: True`
4. Config file has `surrogate.enabled: false`
5. All updated files are copied to the nested run directory

Report back when done. Kimi will then verify and start the actual run.
