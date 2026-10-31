# Layer Access Policy

GPU training policy is repo-wide: production optimizer training must be
GPU-primary. Do not start or continue training if CUDA is unavailable or if the
active path is CPU/Numba-primary. See `GPU_TRAINING_POLICY.md`.

Layer access buttons are mutable except the required L0 x3,y4 ordinary hold to L1.

They must be represented as first-class assignable `Shortcut` capabilities with
`is_layer_access=True`. The optimizer may move, duplicate, remove, or replace
them, subject to structural safety constraints and fitness penalties.

Do not freeze layer access positions in `core/loader.py`.

Do not score reachability or access effort from any static
`layout.layer_access` list in the compiled evaluator. The kernel must rebuild
the access graph from the candidate genome on every evaluation.

Allowed fixed structure:

- base typing keys that are required for normal L0 typing
- required recovery/safety constraints expressed as fitness constraints
- frozen L7 canonical RPG/arrows/navigation/keyboard-system positions when intentionally
  excluded from the mutable optimization pool

Required scoring behavior:

- workflow evidence comes first. Layers are evaluated by the apps, shortcuts,
  mouse actions, scroll actions, and transitions that usage data shows are used
  together, not by fixed semantic layer numbers
- app coherence is only a backup pressure. It should activate when generated
  workflow layers are too similar or redundant, helping one layer remain the
  workflow-specific surface while another becomes a useful app-specific
  workflow. It must not override direct workflow evidence
- layer similarity is a first-class penalty. Two generated layers that are too
  similar should be pushed apart until each is unique enough to be worth
  switching to. The single emergent everything layer is the only layer allowed
  to overlap broadly with many others, because broad coverage is its role
- generated layers should be distinct enough to be useful for different
  workflows. They may share app focus while containing different shortcut sets;
  that is allowed when usage data supports those workflows
- an "everything" layer with the most common cross-workflow actions should
  naturally emerge when usage data supports it. This is the user's go-to layer
  when unsure: it should concentrate the globally most common/high-value
  actions with cheap access. It is not a fixed layer number, and additional
  layers should not become redundant copies of that layer
- if the same shortcut appears on several layers, familiarity scoring should
  prefer the same physical key or nearby region across layers unless a
  higher-value workflow action earns that position. Familiarity is pairwise
  Euclidean exponential attraction between placements: the closer placements
  are, the more sharply the reward rises toward its maximum at exact coordinate
  matches. Far-apart repeats have little useful attraction. This familiarity reward
  should itself be exception-weighted so ordinary repeats can receive worse
  placement or be overwritten by shortcuts that are more useful for that
  generated layer's workflow
- cross-layer repetition should pass an exception threshold. Use a
  sigmoid-weighted novelty style of scoring where high-importance,
  high-frequency, workflow-supported shared shortcuts can remain across layers,
  but ordinary repeats are penalized so layer similarity pressure can populate those
  slots with workflow-specific shortcuts. This does not require an expensive
  exact sigmoid implementation if a cheaper monotonic approximation gives the
  same behavior
- duplicate scoring, familiarity scoring, and layer-similarity scoring must
  be balanced by the same novelty idea. Non-exceptional duplicates should be
  removed, penalized, or made less likely. Exceptional duplicates can remain,
  should usually keep familiar physical placement, and must still compete
  against stronger workflow-specific shortcuts for prime positions
- protected group scoring only applies inside a layer. If group members
  already coexist on a layer, reward them toward a compact local cluster. If
  group members are on different layers, group scoring must not penalize or
  move them across layers. Group-aware mutation may move a whole protected
  group as one unit when a large overwrite would replace half or more of that
  group; scoring then decides whether that moved group survives. Seed repair is governed by the owner contract below.
- Norwegian/raw completion keys are a backup group for physical keys missing
  from L0. They should be grouped together in normal-keyboard-like order,
  preferably on the far-right side, but not as a hardcoded layer role. Since the
  group is too small to fill a whole layer, it should be mixed with compatible
  workflow shortcuts. Logged usage determines access priority relative to
  workflows: lower usage means lower-priority access; higher usage can outrank
  less-used workflows
## Canonical mouse and access contract

`contracts/layout_contract_v1.json` is the versioned source of truth for generation, mutation, exact evaluation, archive selection, export, and promotion. All checks must use its CPU audit semantics.

- Mouse role is fixed to L1. L0 x3,y4 is an ordinary momentary hold to L1, preserving cursor movement.
- L1 MB1, MB2, and momentary Scroll are fixed at x8,y2, x9,y2, and x10,y2. MB3 remains movable within L1 right-hand non-thumb positions. MB4 and MB5 stay adjacent, ordered left-to-right, on that region.
- Exactly three momentary current-layer Scroll bindings exist: fixed L1 Scroll plus two on distinct non-L0, non-L1, non-L7 layers. They never serve essential layer access.
- L1 has a reachable self-lock usable while holding L0's L1 access key; release preserves lock.
- Every generated layer L1-L10 except frozen L7 has an ordinary momentary hold path from L0. Every occupied target thumb side on those paths contains only empty or return-to-L0 bindings. Toggles and locks never waive this restriction. L7 is byte-for-byte frozen and exempt.
- Preserve Norwegian key grouping, L0 LeftAlt, reachable non-L0 RightAlt, required shortcut groups, and duplicate restrictions. Required L1 shortcuts and effort limits are listed in the contract.

Other layers can evolve useful workflows; only L1 has the mouse role. Never select or publish a candidate that fails the versioned contract.
