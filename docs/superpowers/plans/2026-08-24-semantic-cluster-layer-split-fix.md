# Semantic Cluster Layer-Split Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Make the v2 optimizer strongly penalize semantic shortcut clusters (Copy/Paste, Undo/Redo, browser tabs, PowerToys, Excel navigation, etc.) being split across multiple layers, so evolution groups them on a single layer.

**Architecture:** Move semantic-cluster same-layer pressure out of `chain_rows` (which is proximity-based and parity-inconsistent between Numba single-genome and CUDA batch) and into `workflow_rows` (which is a pure same-layer penalty and is handled identically by Numba batch and CUDA). Add the missing `workflow_rows` handling to Numba `_single_genome` so exact single-genome evaluation matches CUDA. Increase the semantic cluster row weight so the penalty is comparable to other violation pressures. Keep semantic clusters as dynamic groups so `group_split` still enforces within-layer compactness.

**Tech Stack:** Python 3, Numba, CUDA (torch cpp_extension), PyMoo.

## Global Constraints

- GPU training must not silently fall back to CPU; CPU-only diagnostics are allowed for tests/perf, but production `run_evolution.py` requires CUDA.
- Any change to `fitness/kernel.py`, `fitness/cuda/fitness_kernel.cu`, or `evolution/__init__.py` hot paths must pass `python3 tools/perf_benchmark.py` per `AGENTS.md`.
- Keep Numba/CUDA fitness parity; single-genome and batch evaluation must agree.
- Minimal diffs; do not rewrite broad systems.
- Run `just ai-guard` and `just ai-smoke` before finishing (or equivalent if `just` unavailable).

---

## Task 1: Add workflow_rows handling to Numba `_single_genome`

**Files:**
- Modify: `fitness/kernel.py` (`_single_genome` around the existing `mouse_workflow` / `app_workflow_rows` loops)

**Interfaces:**
- Consumes: `workflow_rows` array already passed into `_single_genome` (currently unused)
- Produces: `workflow` local variable incremented for each split semantic/workflow pair

- [ ] **Step 1: Locate the workflow computation block in `_single_genome`**

Find the block after `mouse_workflow` and `app_workflow_rows` loops (around line 1231-1242). The `workflow` variable should already exist and be initialized to `0.0`.

- [ ] **Step 2: Add `workflow_rows` loop matching `_evaluate_batch` / CUDA**

```python
for r in range(workflow_rows.shape[0]):
    sid_a = int(workflow_rows[r, 0])
    sid_b = int(workflow_rows[r, 1])
    pos_a = sid_pos[sid_a]
    pos_b = sid_pos[sid_b]
    if pos_a >= 0 and pos_b >= 0 and pos_layer[pos_a] != pos_layer[pos_b]:
        workflow += workflow_rows[r, 2] * 10.0
```

- [ ] **Step 3: Verify `workflow` flows into the violations/objective formula**

Confirm the existing final objective formula includes `workflow * objective_weights[5]` (it already does in `_evaluate_batch`; mirror if needed).

- [ ] **Step 4: Run semantic-cluster unit tests**

Run: `.venv/bin/python -m pytest tests/test_semantic_clusters.py -v`
Expected: PASS (no regression).

- [ ] **Step 5: Commit**

```bash
git add fitness/kernel.py
git commit -m "fix(kernel): process workflow_rows in Numba single-genome eval for parity"
```

---

## Task 2: Route semantic clusters to workflow_rows with strong weight

**Files:**
- Modify: `fitness/kernel.py` (`precompute` and `_semantic_chain_rows`)
- Modify: `config_v2.yaml` (optional semantic-cluster weight multiplier; prefer code constant for now)

**Interfaces:**
- Consumes: `layout.semantic_clusters`
- Produces: `workflow_rows` containing semantic cluster pairs; `chain_rows` no longer carrying semantic cluster pressure (or carrying a much smaller proximity-only weight)

- [ ] **Step 1: Add `_semantic_workflow_rows(layout)` helper**

Create a new helper next to `_semantic_chain_rows` that returns pairwise rows for semantic clusters with a high multiplier chosen to make split clusters pay a penalty comparable to `group_split` (2M weight). Use multiplier `200.0` initially (row weight = cluster_weight * 200.0; per split pair raw penalty = row_weight * 10.0; objective penalty = * workflow_weight 20.0).

```python
def _semantic_workflow_rows(layout) -> np.ndarray:
    """Generate high-weight workflow rows that penalize semantic clusters split across layers."""
    rows = []
    for cluster in getattr(layout, "semantic_clusters", ()):
        members = list(cluster.get("members", []))
        weight = float(cluster.get("weight", 1.0))
        if len(members) < 2:
            continue
        sids = [int(m.get("sid", -1)) for m in members]
        sids = [sid for sid in sids if 0 <= sid < layout.n_shortcuts]
        if len(sids) < 2:
            continue
        pair_weight = weight * 200.0
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                rows.append((sids[i], sids[j], pair_weight))
    return np.asarray(rows, dtype=np.float32).reshape((-1, 3)) if rows else np.empty((0, 3), dtype=np.float32)
```

- [ ] **Step 2: Append semantic workflow rows to `workflow_rows` in `precompute()`**

Change:
```python
_chain_rows(layout, layout.usage_data.workflows, 3, 2.0),
```
to:
```python
np.concatenate([
    _chain_rows(layout, layout.usage_data.workflows, 3, 2.0),
    _semantic_workflow_rows(layout),
], axis=0) if len(_semantic_workflow_rows(layout)) > 0 else _chain_rows(layout, layout.usage_data.workflows, 3, 2.0),
```

Avoid calling `_semantic_workflow_rows(layout)` twice by assigning to a local variable.

- [ ] **Step 3: Reduce or remove semantic pressure from `chain_rows`**

Either set `_semantic_chain_rows` to return empty (proximity is already handled by `group_split` dynamic groups), or keep it with a very small weight (e.g., `weight * 0.5`) for fine-grained within-layer attraction. Recommended: remove from chain_rows to avoid double-counting and parity issues.

Change:
```python
np.concatenate([
    _chain_rows(layout, layout.usage_data.chains, 2, 1.0),
    _semantic_chain_rows(layout),
], axis=0) if len(_semantic_chain_rows(layout)) > 0 else _chain_rows(layout, layout.usage_data.chains, 2, 1.0),
```
to simply:
```python
_chain_rows(layout, layout.usage_data.chains, 2, 1.0),
```

- [ ] **Step 4: Add unit test verifying strong cross-layer penalty**

Create/update `tests/test_semantic_clusters.py`:

```python
def test_semantic_cluster_split_penalty_is_strong(tmp_path):
    from core.loader import build_layout
    from fitness.model import FitnessModel
    from config import DEFAULT_CONFIG
    import numpy as np

    layout = build_layout("data", DEFAULT_CONFIG)
    # Find a semantic cluster with at least 2 members
    cluster = next(c for c in layout.semantic_clusters if len(c["sids"]) >= 2)
    sids = cluster["sids"]

    # Place members on two different mutable non-L7 layers
    genome = layout.genome.copy()
    mutable_by_layer = {}
    for idx in layout.mutable_indices:
        layer = layout.positions[idx].layer
        if layer != 7:
            mutable_by_layer.setdefault(layer, []).append(idx)
    layers = list(mutable_by_layer.keys())[:2]
    for i, sid in enumerate(sids):
        layer = layers[i % 2]
        genome[mutable_by_layer[layer][i]] = sid

    split_layout = layout.clone_with(genome=genome)
    model = FitnessModel(layout=split_layout, weights=DEFAULT_CONFIG["fitness"]["weights"],
                         violation_weights=DEFAULT_CONFIG["fitness"]["violation_sub_weights"],
                         missing_important_threshold=DEFAULT_CONFIG["fitness"]["missing_important_threshold"],
                         hard_constraints=DEFAULT_CONFIG["fitness"]["hard_constraints"],
                         toggle_effort_multiplier=DEFAULT_CONFIG["fitness"]["toggle_effort_multiplier"],
                         require_cuda=False)
    objectives_split, _ = model.evaluate(split_layout.genome)

    # Place members on the same layer close together
    genome2 = layout.genome.copy()
    layer = layers[0]
    for i, sid in enumerate(sids):
        genome2[mutable_by_layer[layer][i]] = sid
    together_layout = layout.clone_with(genome=genome2)
    model2 = FitnessModel(layout=together_layout, weights=DEFAULT_CONFIG["fitness"]["weights"],
                          violation_weights=DEFAULT_CONFIG["fitness"]["violation_sub_weights"],
                          missing_important_threshold=DEFAULT_CONFIG["fitness"]["missing_important_threshold"],
                          hard_constraints=DEFAULT_CONFIG["fitness"]["hard_constraints"],
                          toggle_effort_multiplier=DEFAULT_CONFIG["fitness"]["toggle_effort_multiplier"],
                          require_cuda=False)
    objectives_together, _ = model2.evaluate(together_layout.genome)

    # The split layout should have a noticeably higher violations objective
    assert objectives_split[2] > objectives_together[2] + 1000.0, \
        f"Expected large violations increase when cluster {cluster['name']} is split"
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_semantic_clusters.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add fitness/kernel.py tests/test_semantic_clusters.py
git commit -m "feat(kernel): route semantic clusters to high-weight workflow rows"
```

---

## Task 3: Fix compactness-only atomic group placement (optional but recommended)

**Files:**
- Modify: `evolution/__init__.py` (`build_group_placements`)

**Interfaces:**
- Consumes: `layout.semantic_clusters` with compactness-only members (all order=0, dx=0, dy=0)
- Produces: valid `(sid_tuple, anchor_positions_list)` entries for keyword-family clusters

- [ ] **Step 1: Detect compactness clusters in `build_group_placements`**

After the existing semantic-cluster loop, add a second pass for clusters where all members have order=0 and offset (0,0):

```python
# Compactness-only clusters (keyword families): no fixed shape, just need same-layer proximity.
for cluster in getattr(layout, "semantic_clusters", ()):
    members = list(cluster.get("members", []))
    if len(members) < 2:
        continue
    sids = [int(m.get("sid", -1)) for m in members]
    sids = [sid for sid in sids if sid >= 0]
    if len(sids) < 2:
        continue
    # Skip clusters that already have explicit offsets
    has_explicit_offset = any(
        abs(float(m.get("dx", 0.0))) > 0.01 or abs(float(m.get("dy", 0.0))) > 0.01
        for m in members
    )
    if has_explicit_offset:
        continue
    # Generate anchors: pick any mutable non-L7 position and a set of nearby positions on the same layer
    sid_tuple = tuple(sids)
    anchor_list = []
    for p in layout.positions:
        if p.is_frozen or p.layer == 7:
            continue
        # Collect nearest mutable positions on the same layer
        layer_positions = [q for q in layout.positions if q.layer == p.layer and not q.is_frozen]
        layer_positions.sort(key=lambda q: abs(q.x - p.x) + abs(q.y - p.y))
        if len(layer_positions) >= len(sids):
            anchor_list.append([q.gene_idx for q in layer_positions[:len(sids)]])
    if anchor_list:
        groups.append((sid_tuple, anchor_list))
```

(Implementation can be optimized with a precomputed layer-to-mutable-positions index.)

- [ ] **Step 2: Add unit test for compactness cluster placement**

In `tests/test_semantic_clusters.py`, assert that `build_group_placements(layout)` returns at least one placement for a known keyword-family cluster.

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_semantic_clusters.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add evolution/__init__.py tests/test_semantic_clusters.py
git commit -m "fix(evolution): allow compactness-only semantic clusters in atomic group moves"
```

---

## Task 4: Performance guardrail

**Files:**
- Run: `tools/perf_benchmark.py`

- [ ] **Step 1: Run perf benchmark**

```bash
cd /home/nos/charybdis/charybdis-optimizer-v2
.venv/bin/python tools/perf_benchmark.py
```

- [ ] **Step 2: Check result**

Expected: no regression beyond tolerance vs `tools/perf_baseline.json`. If regression is unavoidable, update baseline with `--update-baseline --reason "added workflow_rows loop to single-genome kernel"`.

- [ ] **Step 3: Commit baseline update if needed**

```bash
git add tools/perf_baseline.json
git commit -m "chore: update perf baseline for workflow_rows single-genome handling"
```

---

## Task 5: Start new evolution run

**Files:**
- Run: `run_evolution.py`

- [ ] **Step 1: Choose run name and config**

Use a fresh run directory, e.g.:
```bash
cd /home/nos/charybdis/charybdis-optimizer-v2
mkdir -p build/runs/v2_cluster_fix_$(date +%Y%m%d_%H%M%S)
```

- [ ] **Step 2: Launch training**

```bash
.venv/bin/python run_evolution.py --config config_v2.yaml --run-dir build/runs/v2_cluster_fix_$(date +%Y%m%d_%H%M%S)
```

(Adjust actual CLI flags to match `run_evolution.py` argument parser.)

- [ ] **Step 3: Verify GPU utilization and first checkpoints**

After a few minutes, confirm:
- `nvidia-smi` shows Python using GPU.
- Checkpoints are written every 500 generations.
- `stderr` is clean (no CUDA/Numba parity errors).

- [ ] **Step 4: Early cluster check**

After ~1000 generations, run the cluster inspection script and verify fewer split clusters than the gen62000 baseline.

---

## Spec Coverage

- Strong cross-layer cluster penalty: Task 2.
- Numba/CUDA parity: Task 1.
- Compactness cluster atomic moves: Task 3.
- Performance guardrail: Task 4.
- New evolution run: Task 5.

## Placeholder Scan

No TBD/TODO/fill-in-details remain; each step has concrete code or commands.
