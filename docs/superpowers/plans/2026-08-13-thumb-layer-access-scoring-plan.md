# Thumb-Aware Layer Access Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. For CUDA/kernel work, also invoke `superpowers:cuda-preservation`.

**Goal:** Add two new fitness terms to `charybdis-optimizer-v2` so the optimizer learns to place momentary/scroll layer-access keys on thumb clusters and to alternate thumb sides along hold chains.

**Architecture:** Extend the existing compiled Numba/CUDA kernels with `edge_is_thumb` tracking, `incoming_thumb_hand` propagation during the shortest-path relaxation, and two new raw scores (`layer_access_thumb_preference` and `same_side_hold_flow`). Add config weights, evaluator factor-score placeholders, and unit/parity tests.

**Tech Stack:** Python 3.11, Numba, PyTorch CUDA extension, pytest, `just`.

## Global Constraints

- GPU-primary training must remain the only production path; any kernel change must be mirrored in `fitness/cuda/fitness_kernel.cu`.
- No frozen positions and no post-hoc manual layout patches.
- CPU-only commands are allowed only for tests, static checks, and short diagnostics.
- Any change to `fitness/kernel.py`, `fitness/cuda/fitness_kernel.cu`, or mutation hot paths must pass `python3 tools/perf_benchmark.py` / `just ai-guard` without regression beyond tolerance.
- Dynamic layer assignment remains unchanged; no fixed semantic layer roles are introduced.

---

## File map

| File | Responsibility |
|------|----------------|
| `config/__init__.py` | Default violation weights and tunable coefficients for the two new raw scores. |
| `fitness/kernel.py` | Numba kernel: access-graph build, shortest-path relaxation, new raw-score computation. |
| `fitness/cuda_kernel.py` | Tensor wrapping; passes the new coefficients array into the CUDA kernel. |
| `fitness/cuda/fitness_kernel.cu` | CUDA kernel mirror of the Numba changes. |
| `fitness/evaluator.py` | Passes coefficients to `FitnessModel` and adds factor-score keys for logging. |
| `tests/test_layer_access_thumb.py` | New unit tests for thumb preference, same-side flow, and CUDA/Numba parity. |

---

## Task 1: Config weights and tunables

**Files:**
- Modify: `config/__init__.py:76-101`
- Test: `tests/test_layer_access_thumb.py` (created in Task 5)

**Interfaces:**
- Produces: `DEFAULT_CONFIG["fitness"]["violation_sub_weights"]` gains `layer_access_thumb_preference` and `same_side_hold_flow`.
- Produces: `DEFAULT_CONFIG["fitness"]["layer_access_thumb"]` dict with five coefficients.

- [ ] **Step 1: Add violation weights**

Inside `violation_sub_weights` add:

```python
"layer_access_thumb_preference": 2500.0,
"same_side_hold_flow": 1500.0,
```

- [ ] **Step 2: Add coefficient dict**

Add a sibling dict inside `fitness`:

```python
"layer_access_thumb": {
    "thumb_bonus": 0.8,
    "non_thumb_penalty": 1.0,
    "non_thumb_base": 2.0,
    "same_side_penalty": 1.0,
    "opposite_side_reward": 0.7,
},
```

- [ ] **Step 3: Run existing weight-sync test**

Run: `python -m pytest tests/test_v2.py::TestDataStructures::test_fallback_weights_are_synchronized_with_default_config -v`
Expected: PASS once `DEFAULT_VIOLATION_WEIGHTS` is updated in Task 2.

- [ ] **Step 4: Commit**

```bash
git add config/__init__.py
git commit -m "config: add thumb-aware layer access weights and tunables"
```

---

## Task 2: Extend Numba kernel data structures and scoring

**Files:**
- Modify: `fitness/kernel.py:504-544`, `fitness/kernel.py:680-760`, `fitness/kernel.py:2247-2293`

**Interfaces:**
- Consumes: `layer_access_thumb` coefficients from config (packed into a 5-float `lat_params` array).
- Produces: `raw_scores` grows from 24 to 26 entries; indices 24 and 25 are the new terms.

- [ ] **Step 1: Extend `precompute` return tuple**

Add a new `lat_params` array at the end of the returned tuple:

```python
lat = layer_access_thumb_params or {}
lat_params = np.asarray([
    float(lat.get("thumb_bonus", 0.8)),
    float(lat.get("non_thumb_penalty", 1.0)),
    float(lat.get("non_thumb_base", 2.0)),
    float(lat.get("same_side_penalty", 1.0)),
    float(lat.get("opposite_side_reward", 0.7)),
], dtype=np.float32)
```

Append `lat_params` to the returned tuple. Update `_single_genome`, `_evaluate_batch`, and `cuda_kernel.py::_build_args` to accept the extra positional argument (see Tasks 3 and 4).

- [ ] **Step 2: Extend violation names/weights arrays**

Add to `VIOLATION_NAMES`:

```python
"layer_access_thumb_preference",
"same_side_hold_flow",
```

Add matching defaults to `violation_weight_arr`.

- [ ] **Step 3: Track thumb edges in access graph**

Declare alongside `edge_cost`, `momentary_edge`, `edge_hand`:

```python
edge_is_thumb = np.zeros((32, 32), dtype=np.bool_)
```

In the access-graph pre-scan, when an edge improves, also record:

```python
edge_is_thumb[source, target] = pos_is_thumb[i]
```

- [ ] **Step 4: Propagate `incoming_thumb_hand` during relaxation**

Declare:

```python
incoming_thumb_hand = np.full(32, -1, dtype=np.int32)
```

In the relaxation loop, when `cand < layer_access_cost[target]` and the improving edge is momentary and on a thumb:

```python
if momentary_edge[source, target] and edge_is_thumb[source, target]:
    incoming_thumb_hand[target] = edge_hand[source, target]
```

- [ ] **Step 5: Compute the two new raw scores**

Before the `raw_scores = np.empty(24, ...)` line, add a position scan:

```python
layer_access_thumb_preference = 0.0
same_side_hold_flow = 0.0
for i in range(n_pos):
    sid = genome[i]
    if sid < 0 or sid >= n_short:
        continue
    if not shortcut_access_momentary[sid]:
        continue
    target = shortcut_access_target[sid]
    if target <= 0 or target >= 32:
        continue
    layer = pos_layer[i]
    if layer < 0 or layer >= 32:
        continue
    imp = shortcut_importance[sid]
    demand = layer_demand[target]
    if pos_is_thumb[i]:
        layer_access_thumb_preference -= imp * lat_params[0] * (1.0 + math.log1p(demand))
    else:
        layer_access_thumb_preference += imp * lat_params[1] + lat_params[2]
    if pos_is_thumb[i]:
        inc = incoming_thumb_hand[layer]
        if inc >= 0:
            if pos_hand[i] == inc:
                same_side_hold_flow += imp * lat_params[3]
            else:
                same_side_hold_flow -= imp * lat_params[4]
```

Note: scroll-mode holds are included because they are created with `access_is_momentary=True`.

- [ ] **Step 6: Wire new scores into raw_scores and violations loop**

Change:

```python
raw_scores = np.empty(26, dtype=np.float32)
```

Assign:

```python
raw_scores[24] = layer_access_thumb_preference
raw_scores[25] = same_side_hold_flow
```

Update the violations loop:

```python
for j in range(26):
    violations_raw += raw_scores[j] * violation_weights[j]
```

- [ ] **Step 7: Run a quick kernel import test**

Run: `python -c "from fitness.kernel import precompute; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 8: Commit**

```bash
git add fitness/kernel.py
git commit -m "feat(kernel): thumb-aware layer access raw scores (Numba)"
```

---

## Task 3: Update Numba kernel call signatures

**Files:**
- Modify: `fitness/kernel.py:2372-2414`

**Interfaces:**
- Consumes: `lat_params` produced by `precompute`.
- Produces: `_single_genome` and `_evaluate_batch` accept and forward `lat_params`.

- [ ] **Step 1: Update `_single_genome` signature**

Append `lat_params` to the end of the parameter list.

- [ ] **Step 2: Update `_evaluate_batch` signature and call**

Append `lat_params` and pass it through to `_single_genome`.

- [ ] **Step 3: Verify import still works**

Run: `python -c "from fitness.kernel import _evaluate_batch, _single_genome; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add fitness/kernel.py
git commit -m "refactor(kernel): pass lat_params through Numba kernel signatures"
```

---

## Task 4: Update CUDA tensor wrapper

**Files:**
- Modify: `fitness/cuda_kernel.py:70-166`

**Interfaces:**
- Consumes: `lat_params` from the arrays tuple.
- Produces: `_build_args` returns a CUDA tensor for `lat_params`.

- [ ] **Step 1: Unpack `lat_params` in `_build_args`**

Add `lat_params` to the destructuring assignment and to the returned tuple as `_to_tensor(lat_params)`.

- [ ] **Step 2: Update `build_cuda_args`**

No change needed if `_build_args` already returns the new tensor in the same position; `build_cuda_args` only replaces `n_key_groups`.

- [ ] **Step 3: Verify CUDA wrapper import**

Run: `python -c "from fitness.cuda_kernel import build_cuda_args; print('ok')"`
Expected: prints `ok` (CUDA not required for import).

- [ ] **Step 4: Commit**

```bash
git add fitness/cuda_kernel.py
git commit -m "feat(cuda_wrapper): pass lat_params tensor to CUDA kernel"
```

---

## Task 5: Mirror changes in CUDA kernel

**Files:**
- Modify: `fitness/cuda/fitness_kernel.cu` (access-graph build ~lines 320-410, raw score assembly ~lines 2018-2066)

**Interfaces:**
- Consumes: `lat_params` tensor (5 floats).
- Produces: CUDA kernel computes identical `raw_scores[24]` and `raw_scores[25]`.

- [ ] **Step 1: Add `lat_params` to device kernel signature**

Append `const float* lat_params` to the device kernel argument list.

- [ ] **Step 2: Track thumb edges**

Add `bool edge_is_thumb[MAX_LAYERS][MAX_LAYERS]` in shared/per-thread state, initialized to false. Set it when an edge improves:

```cpp
edge_is_thumb[source][target] = pos_is_thumb[i];
```

- [ ] **Step 3: Propagate incoming thumb hand**

Add `int incoming_thumb_hand[MAX_LAYERS]` initialized to `-1`. In the relaxation loop:

```cpp
if (momentary_edge[source][target] && edge_is_thumb[source][target]) {
    incoming_thumb_hand[target] = edge_hand[source][target];
}
```

- [ ] **Step 4: Compute new raw scores**

After `layer_demand` is populated and before `raw_scores` assembly, add a position loop matching the Numba version:

```cpp
float layer_access_thumb_preference = 0.0f;
float same_side_hold_flow = 0.0f;
for (int i = 0; i < n_pos; i++) {
    int sid = genome[i];
    if (sid < 0 || sid >= n_short) continue;
    if (!shortcut_access_momentary[sid]) continue;
    int target = shortcut_access_target[sid];
    if (target <= 0 || target >= MAX_LAYERS) continue;
    int layer = pos_layer[i];
    if (layer < 0 || layer >= MAX_LAYERS) continue;
    float imp = shortcut_importance[sid];
    float demand = layer_demand[target];
    if (pos_is_thumb[i]) {
        layer_access_thumb_preference -= imp * lat_params[0] * (1.0f + log1pf(demand));
    } else {
        layer_access_thumb_preference += imp * lat_params[1] + lat_params[2];
    }
    if (pos_is_thumb[i]) {
        int inc = incoming_thumb_hand[layer];
        if (inc >= 0) {
            if (pos_hand[i] == inc) {
                same_side_hold_flow += imp * lat_params[3];
            } else {
                same_side_hold_flow -= imp * lat_params[4];
            }
        }
    }
}
```

- [ ] **Step 5: Extend raw_scores and violation loop**

Change:

```cpp
float raw_scores[26];
```

Assign:

```cpp
raw_scores[24] = layer_access_thumb_preference;
raw_scores[25] = same_side_hold_flow;
```

Update the loop:

```cpp
for (int j = 0; j < 26; j++) {
    violations_raw += raw_scores[j] * violation_weights[j];
}
```

- [ ] **Step 6: Update host wrapper launch signature**

Add the `torch::Tensor lat_params` argument and pass `lat_params.data_ptr<float>()` to the device kernel.

- [ ] **Step 7: Compile and run CUDA parity test**

Run: `python -m pytest tests/test_layer_access_thumb.py::TestLayerAccessThumb::test_cuda_numba_parity -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add fitness/cuda/fitness_kernel.cu
python3 -c "from fitness.cuda_kernel import _load_cuda_extension; _load_cuda_extension()"  # force compile
git commit -m "feat(cuda): thumb-aware layer access raw scores"
```

---

## Task 6: Update evaluator factor-score mapping

**Files:**
- Modify: `fitness/evaluator.py:48-70`
- Modify: `fitness/evaluator.py:18-44` (pass coefficients)

**Interfaces:**
- Consumes: `layer_access_thumb` from config.
- Produces: `FitnessResult.factor_scores` includes the two new keys.

- [ ] **Step 1: Pass coefficients to FitnessModel**

In `FitnessEvaluator.__init__`, extract `layer_access_thumb` from config and pass as `layer_access_thumb_params` to `FitnessModel`. Do the same for the temporary model built in `evaluate()`.

- [ ] **Step 2: Add factor-score placeholders**

Add to `_factor_scores_from_objectives`:

```python
"layer_access_thumb_preference": 0.0,
"same_side_hold_flow": 0.0,
```

(The exact raw values are not returned by the compiled model; placeholders keep checkpoint logs consistent.)

- [ ] **Step 3: Run evaluator import test**

Run: `python -c "from fitness.evaluator import FitnessEvaluator; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add fitness/evaluator.py
git commit -m "feat(evaluator): pass thumb coefficients and log factor placeholders"
```

---

## Task 7: Unit and parity tests

**Files:**
- Create: `tests/test_layer_access_thumb.py`

**Interfaces:**
- Consumes: `FitnessEvaluator`, `Layout`, `Position`, `Shortcut`.

- [ ] **Step 1: Write synthetic layout helper**

```python
def _make_minimal_layout():
    positions = (
        # L0 left thumb
        Position(0, 0, 3.0, 4.0, "left", 0, 1.0, is_thumb=True),
        # L0 right thumb
        Position(1, 0, 8.0, 4.0, "right", 0, 1.0, is_thumb=True),
        # L0 left finger
        Position(2, 0, 2.0, 2.0, "left", 1, 1.0, is_thumb=False),
        # L1 left thumb
        Position(3, 1, 3.0, 4.0, "left", 0, 1.0, is_thumb=True),
        # L1 right thumb
        Position(4, 1, 8.0, 4.0, "right", 0, 1.0, is_thumb=True),
        # L1 left finger
        Position(5, 1, 2.0, 2.0, "left", 1, 1.0, is_thumb=False),
    )
    shortcuts = (
        Shortcut(0, "@access:L1:hold", "L1 hold", "Layer Access", 10.0,
                 category="layer_access", is_layer_access=True,
                 access_target_layer=1, access_is_momentary=True),
        Shortcut(1, "@access:L2:hold", "L2 hold", "Layer Access", 10.0,
                 category="layer_access", is_layer_access=True,
                 access_target_layer=2, access_is_momentary=True),
        Shortcut(2, "Ctrl+C", "Copy", "App", 5.0),
    )
    layer_to_indices = {
        0: np.array([0, 1, 2], dtype=np.int32),
        1: np.array([3, 4, 5], dtype=np.int32),
    }
    frozen_mask = np.zeros(len(positions), dtype=bool)
    return positions, shortcuts, layer_to_indices, frozen_mask
```

- [ ] **Step 2: Write thumb-preference test**

```python
def test_layer_access_thumb_preference():
    positions, shortcuts, layer_to_indices, frozen_mask = _make_minimal_layout()

    # L1 hold on left thumb -> good
    genome_good = np.array([0, -1, -1, -1, -1, -1], dtype=np.int32)
    layout_good = Layout(genome_good, positions, shortcuts, frozen_mask, layer_to_indices)

    # L1 hold on left finger -> bad
    genome_bad = np.array([2, -1, 0, -1, -1, -1], dtype=np.int32)
    layout_bad = Layout(genome_bad, positions, shortcuts, frozen_mask, layer_to_indices)

    ev = FitnessEvaluator()
    score_good = ev.evaluate(layout_good).total_score
    score_bad = ev.evaluate(layout_bad).total_score
    assert score_bad > score_good, "non-thumb layer access should score worse"
```

- [ ] **Step 3: Write same-side hold-flow test**

```python
def test_same_side_hold_flow():
    positions, shortcuts, layer_to_indices, frozen_mask = _make_minimal_layout()

    # L0 left thumb -> L1, L1 left thumb -> L2 (same side, bad)
    genome_same = np.array([0, -1, -1, 1, -1, -1], dtype=np.int32)
    # L0 left thumb -> L1, L1 right thumb -> L2 (opposite, good)
    genome_opp = np.array([0, -1, -1, -1, 1, -1], dtype=np.int32)

    layout_same = Layout(genome_same, positions, shortcuts, frozen_mask, layer_to_indices)
    layout_opp = Layout(genome_opp, positions, shortcuts, frozen_mask, layer_to_indices)

    ev = FitnessEvaluator()
    score_same = ev.evaluate(layout_same).total_score
    score_opp = ev.evaluate(layout_opp).total_score
    assert score_same > score_opp, "same-side hold chain should score worse"
```

- [ ] **Step 4: Write CUDA/Numba parity test**

```python
@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
def test_cuda_numba_parity():
    from fitness.model import FitnessModel
    from core.loader import build_layout
    layout = build_layout("data")
    ev = FitnessEvaluator()
    model = ev.model
    objectives_numba, _ = model.evaluate(layout.genome)
    objectives_cuda, _ = model.evaluate_batch(layout.genome.reshape(1, -1))
    assert np.allclose(objectives_numba, objectives_cuda[0], atol=1e-4)
```

- [ ] **Step 5: Run the new tests**

Run: `python -m pytest tests/test_layer_access_thumb.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_layer_access_thumb.py
git commit -m "test: thumb-aware layer access scoring and parity"
```

---

## Task 8: Performance and smoke verification

**Files:**
- All of the above.

- [ ] **Step 1: Run unit test suite**

Run: `python -m pytest tests/test_v2.py tests/test_layer_access_thumb.py -q`
Expected: all pass.

- [ ] **Step 2: Run performance guard**

Run: `python3 tools/perf_benchmark.py`
Expected: no regression beyond tolerance; if it regresses, investigate before updating the baseline.

- [ ] **Step 3: Run project smoke checks**

Run: `just ai-smoke`
Expected: passes (ruff, pytest, any biome/cargo checks).

- [ ] **Step 4: Short CPU-only evolution smoke**

Run a diagnostic 100-generation CPU-only smoke to ensure the new terms do not crash training:

```bash
.venv/bin/python run_evolution.py --no-inject-seed --require-cuda False --n-generations 100 --output-dir build/smoke_thumb_access
```

Expected: completes without error; checkpoint written.

- [ ] **Step 5: Run `just ai-guard`**

Run: `just ai-guard`
Expected: zero exit.

- [ ] **Step 6: Commit any final fixes**

```bash
git commit -m "verify: perf, smoke, and ai-guard pass for thumb-aware scoring"
```

---

## Self-review checklist

1. **Spec coverage:**
   - Thumb-cluster preference for momentary/scroll layer-access keys → Task 2/5.
   - Same-side hold-flow penalty and opposite-side reward → Task 2/5.
   - Normal shortcuts unaffected → implicit (no code touches non-layer-access keys).
   - Configurable weights and tunables → Task 1.
   - CUDA/Numba parity → Task 5/7.
   - No manual patches → enforced by only changing scoring.

2. **Placeholder scan:** No TBD/TODO/fill-in-details; every step has concrete code or exact command.

3. **Type consistency:** `lat_params` is added as the final positional argument in the arrays tuple and forwarded through Numba, CUDA wrapper, and CUDA kernel signatures consistently.

4. **Performance safety:** Task 8 explicitly runs `tools/perf_benchmark.py` and `just ai-guard`; no O(n) full-genome rescans are introduced inside mutation hot paths.
