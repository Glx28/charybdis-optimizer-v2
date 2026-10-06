# Thumb-Aware Layer Access Scoring

> Superseded: opposite-side hold flow is not an accepted rule and its score is
> disabled. See [LAYOUT_ACCEPTANCE.md](../../LAYOUT_ACCEPTANCE.md).

## Summary

Add two new fitness terms to `charybdis-optimizer-v2` so the optimizer learns to:

1. Place coach layer-access keys (momentary holds and scroll-mode holds) on thumb clusters.
2. Prefer thumb placement for ordinary momentary layer access.

The change is implemented inside the compiled fitness kernel (`fitness/kernel.py` and `fitness/cuda/fitness_kernel.cu`) so it affects GPU batch evaluation, the surrogate model, and acceptance checks. No positions are frozen and no post-hoc layout patch is applied.

## Goals

- Coach layer keys should live on thumb clusters, not in the finger area.
- Normal shortcuts should compete for thumb clusters only on effort/usage merit; they receive no thumb-specific bonus.
- Scroll-mode access remains exempt from ordinary thumb access preference.
- Return-to-L0 and toggle access are not part of the hold-flow rule.

## Definitions

- **Thumb cluster**: any position where `pos_is_thumb` is true. Left/right is given by `pos_hand` (`0 = left`, `1 = right`). No new geometry is introduced.
- **Momentary layer access**: shortcuts with `is_layer_access = true`, `access_is_momentary = true`, and `access_target_layer != 0`. This includes both plain `@access:L{N}:hold` and `@scroll:L{N}:hold`.
- **Incoming thumb hand**: the hand of the last thumb-cluster momentary edge on the shortest access path from L0 to a layer. `-1` means the layer is not reached through any thumb momentary edge.

## New raw scores

The kernel's `raw_scores` array grows from 24 to 26 entries.

### Index 24 — `layer_access_thumb_preference`

For every assigned momentary/scroll layer-access key:

- If it is on a thumb cluster: reward `importance * thumb_bonus * (1 + log1p(layer_demand))`.
- If it is not on a thumb cluster: penalty `importance * non_thumb_penalty + base_penalty`.

Return-to-L0 keys and toggles are ignored. The demand scaling ensures high-traffic layers pay a larger price for non-thumb access.

Suggested tunable parameters:

- `thumb_bonus = 0.8`
- `non_thumb_penalty = 1.0`
- `non_thumb_base = 2.0`

### Index 25 — `same_side_hold_flow`

During the existing shortest-path relaxation over the access graph:

- Track `incoming_thumb_hand[layer]` = hand of the thumb momentary edge that improved `layer_access_cost[layer]`. Only update when the improving edge is momentary and originates from a thumb cluster.
- After convergence, for each momentary/scroll layer-access key on a thumb cluster on layer `L`:
  - Same hand as `incoming_thumb_hand[L]`: penalty `importance * same_side_penalty`.
  - Opposite hand: reward `importance * opposite_side_reward`.
  - `incoming_thumb_hand[L] == -1`: no score (neutral).

Suggested tunable parameters:

- `same_side_penalty = 1.0`
- `opposite_side_reward = 0.7`

## Algorithm details

### Access graph build

The existing pre-scan over positions already records `edge_cost[source, target]`, `momentary_edge[source, target]`, and `edge_hand[source, target]`. Add a new array:

- `edge_is_thumb[source, target] = pos_is_thumb[i]` for the position that provided the best edge cost.

### Shortest-path relaxation

The existing double loop over `(source, target)` updates `layer_access_cost[target]`. When an improvement occurs:

```
if momentary_edge[source, target] and edge_is_thumb[source, target]:
    incoming_thumb_hand[target] = edge_hand[source, target]
```

This records the hand of the final thumb hold on the shortest path, which is the thumb the user must keep pressed to be on that layer.

### Score accumulation

After the relaxation, iterate over positions once:

- For layer-access keys on thumb clusters, add thumb-preference reward.
- For layer-access keys off thumb clusters, add thumb-preference penalty.
- For layer-access keys on thumb clusters, compare `pos_hand` to `incoming_thumb_hand[pos_layer]` and add same-side/opposite-side flow score.

Both scores are weighted through the normal `violation_weights` array and summed into the violations objective.

## Files changed

- `config/__init__.py`
  - Add `layer_access_thumb_preference` and `same_side_hold_flow` to `violation_sub_weights`.
  - Add optional tunable sub-parameters under a new `layer_access_thumb` dict.
- `fitness/kernel.py`
  - Extend `violation_weight_arr` and `VIOLATION_NAMES` by two entries.
  - Add `edge_is_thumb` during access-graph build.
  - Propagate `incoming_thumb_hand` during relaxation.
  - Compute and assign `raw_scores[24]` and `raw_scores[25]`.
  - Update the `violations_raw` loop to cover 26 entries.
- `fitness/cuda/fitness_kernel.cu`
  - Mirror the Numba changes exactly: extend arrays, add edge/thumb tracking, compute new raw scores, extend loops.
- `fitness/evaluator.py`
  - Add placeholder factor-score keys for the two new terms so checkpoint logs show them.
- `tests/test_v2.py` (or new `tests/test_layer_access_thumb.py`)
  - Unit tests for thumb-preference penalty/reward.
  - Unit tests for same-side hold-flow penalty/reward.
  - CUDA/Numba parity test for raw scores including the new indices.

## Config defaults

```yaml
fitness:
  violation_sub_weights:
    layer_access_thumb_preference: 2500.0
    same_side_hold_flow: 1500.0
  layer_access_thumb:
    thumb_bonus: 0.8
    non_thumb_penalty: 1.0
    non_thumb_base: 2.0
    same_side_penalty: 1.0
    opposite_side_reward: 0.7
```

The weights put the new terms in the same range as `access_layout` (5000) and `thumb_occupancy` (30000), so they can compete with existing pressures without swamping everything else.

## Testing

1. **Unit tests** on synthetic layouts:
   - A layer reached by a left-thumb hold with a left-thumb hold key scores worse than one with a right-thumb hold key.
   - A momentary layer-access key off the thumb scores worse than the same key on the thumb.
   - A non-layer shortcut on a thumb is unaffected by the new terms.
2. **Parity**: CUDA and Numba produce identical objectives for a sampled batch after the change.
3. **Smoke run**: a short evolution run completes without crash.

## Acceptance criteria

- New raw scores are produced by both Numba and CUDA kernels.
- Unit tests pass.
- Ordinary momentary layer access uses thumb positions where practical.

## Out of scope

- No changes to the coach website, firmware, or AHK scripts.
- No manual assignment or freezing of layer-access positions.
- Toggle keys are not forced to thumbs.
