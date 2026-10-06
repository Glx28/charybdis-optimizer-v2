# Layout acceptance contract

This file resolves conflicting older prompts and handoffs for candidate
reviews. The already-applied keyboard remains outside this optimizer change.
The candidate is review-only.

## Authority

Hard requirements come from the current canonical `AGENTS.md` and
`LAYER_ACCESS_POLICY.md` statements below. Older run plans, generated checklists,
and historical handoffs are not additional acceptance requirements. Raw arrow
keys are available on frozen L7 and have no mutable-layer shape, placement,
scoring, mutation, or acceptance requirement. Complex modifier shortcuts still
use semantic direction and sequence relations where their group context makes
those relations clear.

## Hard checks

- L7 stays frozen; L7 must be reachable through both momentary and toggle
  access. L7 content is excluded from generated-layer acceptance.
- A generated non-L0/non-L7 mouse workflow layer contains MB1–MB5 on the right,
  with no mouse button on a right-thumb position. It has right-hand,
  non-thumb momentary Scroll access, no right-thumb momentary access to that
  layer, and reachable toggle access.
- Mouse buttons are forbidden on right-thumb positions on every generated
  layer. No shortcut is duplicated on one layer, subject to the documented
  live mouse-layer exception in `AGENTS.md`.
- Ordinary momentary layer access uses a thumb. Scroll-mode access is exempt.
- Nested momentary holds receive soft scoring pressure to alternate thumb sides;
  opposite-side access receives a reward. This is not a hard acceptance gate.
- `Win+H`, `Win+Tab` (Task View), `Alt+Tab`, and PowerToys Command Palette
  (`Win+Alt+Space`) must be assigned and reachable; audit their live access paths.
- Norwegian completion keys and export literals are audited as named product
  requirements; export validation remains separate from optimizer acceptance.

## Soft preferences

- Prefer momentary access for frequently used layers; the configured
  `toggle_effort_multiplier` must affect production CPU and CUDA access-path
  costs. Toggle remains available where useful.
- Keep contextually ordered shortcut groups (direction, sequence, undo/redo,
  before/after, and similar pairs) in the relative arrangement their actions
  imply. This does not apply to unmodified arrow keys.
- Measure sparse reachable-layer occupancy and workflow utility for review.
  Occupancy has no minimum-count acceptance cutoff or standalone score penalty.
- Semantic co-location, shortcut familiarity, duplicate selectivity, and layer
  distinction inform candidate review; no review signal replaces the named
  hard checks.

## Score reporting

Report raw objective score and score relative to the current local floor
`80710.5625`, the best contract-passing single swap found by exhaustive CPU
search from the gen-4000 candidate. That neighbor scores `0`; the original
candidate scores about `55.82` above it. This is a practical local reference,
not a global lower bound. Scores from historical configs or different schedule
stages are not comparable. Layout criteria above remain authoritative.
