# Layout acceptance contract

`contracts/layout_contract_v1.json` is the versioned source of truth for seed
validation, generation, mutation, exact evaluation, archive selection, export,
and promotion. Historical handoffs and score metadata do not establish
validity. Frozen L7 keeps its existing contents and is exempt from mutable
layer acceptance.

## Hard checks

- L0 `(3,4)` accesses the fixed mouse layer L1 through an ordinary momentary
  hold. L1 has all five mouse buttons and an accessible self-lock.
- Preserve the current applied seed's L0 `(7,4)` L7 hold across variation and
  crossover when that seed control is present.
- L1 MB1 `(8,2)`, MB2 `(9,2)`, and momentary Scroll `(10,2)` are fixed. MB3 is
  right-hand non-thumb. MB4 and MB5 are adjacent on one row, MB4 left of MB5.
- Exactly three momentary scroll bindings exist: the fixed L1 binding plus two
  current-layer bindings on distinct layers outside L0 and frozen L7.
- Every generated layer except frozen L7 has an ordinary momentary hold path
  from L0. Thumb clearance applies across every hold path, including nested
  paths; a toggle or lock never waives the restriction.
- Preserve Norwegian keys/grouping, L0 LeftAlt, reachable non-L0 AltGr,
  existing duplicate restrictions, and all other rules listed in the contract.

Workflow shortcut locations and groups are scored, not hard-pinned. Layer-local
use controls familiarity and demotion; observed sequences reward proximity.
Mouse-session and explicit-favorite bonuses require observed use on L1, so
global use on L0 cannot force a low-use duplicate onto the mouse layer. Only
fixed owner controls and Norwegian physical grouping remain position-constrained.

Every candidate checkpoint must pass the contract before selection or export.
Scoring and generation count are review data, not acceptance evidence.
