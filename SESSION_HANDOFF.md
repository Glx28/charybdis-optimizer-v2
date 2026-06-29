# Charybdis V2 Evolution — Session Handoff

**Date:** 2026-06-29
**Status:** Run killed, new run starting with increased constraint weights
**Next check:** Verify startup completes, then monitor for gen 1000

---

## What Works (Good Context)

### 1. Surrogate is DISABLED (Working)
- Config: `surrogate.enabled: false` in `build/config_v2.yaml`
- `run_evolution.py` has `surrogate_enabled` branch — skips all surrogate training
- Uses `ExactEvalCallback` instead of `SurrogateCallback`
- Exact Numba eval every generation

### 2. Numba Exact Evaluator is WORKING (Fast)
- Removed `fastmath=True` from `@njit(parallel=False, cache=False)`
- `tolerance = float(tolerance)` fix in `make_batch_evaluator()` and `validate_parity()`
- Parity verified: `ok=True, max_diff=96`
- Speed: ~0.2s per generation (pop=500, 500 layouts exact eval in ~15ms)
- Startup: ~42s Numba compilation (one-time), then fast

### 3. Constraints ARE Wired Inside ViolationFactor
- `_hand_bias()` — weight 2000. Mouse on left hand = `importance * 5.0` penalty
- `_arrow_order()` — weight 500000. LeftArrow must be left of RightArrow
- `_mouse_layer_access()` — weight 5000. Mouse on layers requiring right-hand momentary access
- All are called from `ViolationFactor.compute()` → `FitnessEvaluator.evaluate()`
- The analysis claim that "constraints are not wired" is FALSE — it was from the OLD pre-fix run

### 4. IQR Scale Factors ARE Working
- Computed from 50 random layouts using `np.percentile(q25, q75)`
- Violations IQR = ~358M (large but correct for the range)
- Seed-relative: effort=10, adj=-10, viol=1.89

### 5. L0 Raw Duplicate Filtering WORKS
- Prevents raw no-modifier shortcuts from duplicating base L0 keys
- `_base_j`, `_base_spacebar`, `_base_returnenter` internal records
- Frozen L0 thumb keys: Spacebar at (4,4), Return at (7,5)

### 6. Evolution IS Working (Not Frozen)
- Previous run reached gen 9000+ with best objective -19.55
- Stagnation count = 0 (still exploring)
- Mouse improved from 20% → 80% right-hand over generations
- NOT the old stuck run (gen 3900 with R²=-0.1)

---

## What Was Fixed This Session

| File | Change | Status |
|------|--------|--------|
| `fitness/factors/violation.py` | `missing_important`: 500 → **5,000,000** | Applied |
| `fitness/factors/violation.py` | `arrow_order`: 500 → **500,000** | Applied |
| `fitness/batch_evaluator.py` | Numba `violation_weights` array synced to match | Applied |
| `fitness/batch_evaluator.py` | `parallel=True` → `parallel=False` (stability) | Applied |
| `run_evolution.py` | IQR scale factors (not seed-relative) | Applied |
| `run_evolution.py` | `setdefault` fix in `_repair_best_groups` | Applied |
| `run_evolution.py` | `ExactEvalCallback` with mutation, repair, checkpoint | Applied |

---

## Current Files in Run Directory

```
C:\Users\nos\charybdis-optimizer\charybdis-optimizer-v2\run_evolution.py
C:\Users\nos\charybdis-optimizer\charybdis-optimizer-v2\fitness\factors\violation.py
C:\Users\nos\charybdis-optimizer\charybdis-optimizer-v2\fitness\batch_evaluator.py
C:\Users\nos\charybdis-optimizer\charybdis-optimizer-v2\launch_bg.py
```

Also synced to working tree:
```
C:\Users\nos\charybdis-optimizer-v2\run_evolution.py
C:\Users\nos\charybdis-optimizer-v2\fitness\factors\violation.py
C:\Users\nos\charybdis-optimizer-v2\fitness\batch_evaluator.py
```

---

## Known Issues (Bad Context to Remove)

### ❌ "Constraints are not wired into evaluator" — FALSE
The analysis claims `hand_bias.py`, `arrow_order`, and `mouse_layer_access` are not wired. This is from the OLD running process (pre-fix code). The current code has all three inside `ViolationFactor.compute()` and they ARE called by `FitnessEvaluator`.

### ❌ "Run is frozen at gen 3900" — FALSE
That was the OLD run. The new exact-eval run reached gen 9000+ with best objective -19.55 and stagnation_count=0. The process is NOT frozen.

### ❌ "Numba evaluator is broken (isfinite error)" — FIXED
The `isfinite` error was caused by `tolerance` being a string. Fixed with `tolerance = float(tolerance)` in both `make_batch_evaluator` and `validate_parity`. Numba parity is now `ok=True`.

### ❌ "Process is dead/hung" — FALSE (was Numba compile time)
The 42-second startup silence is Numba JIT compilation. The process is NOT dead. After compile, it runs at ~0.2s/gen.

### ❌ "Missing_important doesn't penalize arrows" — FIXED
The `_missing_important` function already penalizes ALL shortcuts with `importance >= 6.0` regardless of category. The issue was the scale factor making the penalty invisible, not the function itself.

---

## Remaining Problems to Solve

### 1. Arrow Keys Still Missing (LeftArrow, DownArrow unassigned)
- **Root cause:** Scale factor (358M) drowns the missing penalty (4,000) in normalized space
- **Fix applied:** Increased `missing_important` weight from 500 → 5,000,000
- **Expected:** Missing penalty now = 8 * 5,000,000 = 40,000,000. Normalized = 40M/358M = 0.11. Should be visible.
- **Verification needed:** Check gen 1000+ checkpoint for arrow placement

### 2. Arrow Keys Not Grouped (split across layers)
- **Root cause:** `group_split` penalty (200) is too small relative to total violations
- **Fix applied:** Increased `arrow_order` weight from 500 → 500,000
- **Expected:** Arrow ordering penalty now visible. Grouping still depends on `group_split` weight.

### 3. MB4 Still on Left Hand (at L2, (0,2))
- **Root cause:** `hand_bias` weight (2000) works but not strong enough for all mouse buttons
- **Status:** 4/5 mouse buttons on right hand at gen 9000 (80%)
- **Fix:** The increased `missing_important` and `arrow_order` might indirectly help by freeing up good right-hand positions

### 4. Scale Factor Still Dominates
- **Root cause:** Violations IQR = 358M because random layouts have ~500M-1B violations
- **Fix applied:** Not changed — we fixed via weights instead
- **Alternative:** If weights don't work, consider computing scale factors from a GOOD sample (not random) or using multi-objective optimization

---

## Next Steps for New Session

### Immediate (Next 30 min)
1. **Verify startup:** Check `../build/run_logs/v2_restart_bg.log` for successful startup with new weights
2. **Verify speed:** Should see 0.2-0.4s per generation after 42s compile
3. **Check first checkpoint:** At gen 500, check `v2_checkpoint_gen500.json` exists

### Short-term (Next 2 hours)
4. **Monitor gen 1000:** Check if arrows are placed (LeftArrow, RightArrow, UpArrow, DownArrow)
5. **Check mouse placement:** Verify MB4 moved to right hand
6. **If arrows still missing:** Need further weight increases or scale factor changes

### Medium-term
7. **Gen 5000 checkpoint:** Analyze for grouping, ordering, mouse hand
8. **If layout is good:** Export to `v2_evolution_results.json` and give to keyboard analysis AI
9. **If layout is bad:** Consider multi-objective optimization (change `n_obj=1` to `n_obj=3`)

---

## Key Commands

```bash
# Check process
ps -W | grep charybdis-optimizer-v2 | grep python

# Check log
tail -30 ../build/run_logs/v2_restart_bg.log

# Check latest checkpoint
ls -t ../build/v2_checkpoint_gen*.json | head -1

# Read checkpoint
python -c "import json; d=json.load(open('../build/v2_checkpoint_genN.json')); print(f'Gen {d[\"generation\"]}: best={d[\"best_objectives\"]}')"

# Quick analysis of checkpoint (find arrows, mouse, scroll)
python -c "
import json, sys
sys.path.insert(0, '.')
from core.loader import build_layout
cp = json.load(open('../build/v2_checkpoint_genN.json'))
layout = build_layout('../build')
layout.genome[:] = cp['best_genome']
for sid in layout.genome:
    if sid < 0: continue
    sc = layout.shortcuts[sid]
    if sc.keys in ['LeftArrow', 'RightArrow', 'UpArrow', 'DownArrow', 'MB1', 'MB2', 'MB3', 'MB4', 'MB5', 'ScrollUp', 'ScrollDown']:
        pos = layout.positions[list(layout.genome).index(sid)]
        print(f'{sc.keys}: L{pos.layer} ({pos.x},{pos.y}) hand={pos.hand}')
"

# Kill process if needed
C:/Users/nos/charybdis-optimizer/charybdis-optimizer-v2/.venv/Scripts/python.exe -c "import os; os.system('taskkill /F /PID <PID>')"

# Restart
C:/Users/nos/charybdis-optimizer/charybdis-optimizer-v2/.venv/Scripts/python.exe launch_bg.py
```

---

## Checkpoint Path for Analysis AI

Latest checkpoint (when ready):
```
"C:\Users\nos\charybdis-optimizer\build\v2_checkpoint_gen{generation}.json"
```

To find the latest:
```bash
ls -t C:/Users/nos/charybdis-optimizer/build/v2_checkpoint_gen*.json | head -1
```

---

## Learned Lessons (Keep)

1. **Numba compile time is ~42s** — don't kill the process during startup silence
2. **Log file doesn't show gen progress** — only checkpoints and mutation rate changes. Use checkpoint timestamps to measure speed
3. **IQR scale factors make small penalties invisible** — weights must be large enough to be visible after normalization
4. **The analysis AI's "not wired" claim is from OLD code** — always verify current code before believing analysis
5. **Two directories exist** — `charybdis-optimizer-v2` (working tree) and `charybdis-optimizer/charybdis-optimizer-v2` (run directory). Must sync both.
6. **Pymoo `Callback` requires `super().__init__()`** — otherwise `is_initialized` attribute error
7. **Windows Git Bash path normalization** — use `pwd -W` or convert `/c/` → `C:/`

## Pitfalls to Avoid (Remove)

1. Don't re-run the `evaluator.py` wiring analysis — constraints ARE already wired
2. Don't trust "process is dead" based on log silence — check checkpoint timestamps
3. Don't use `parallel=True` in Numba on Windows — causes hangs
4. Don't modify files in only one directory — sync to both working tree and run tree
5. Don't create separate factor files for constraints — add them inside `ViolationFactor`

