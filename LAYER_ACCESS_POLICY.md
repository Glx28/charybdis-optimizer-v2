# Layer Access Policy

GPU training policy is repo-wide: production optimizer training must be
GPU-primary. Do not start or continue training if CUDA is unavailable or if the
active path is CPU/Numba-primary. See `GPU_TRAINING_POLICY.md`.

Layer access buttons are not fixed structure.

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
  group; scoring then decides whether that moved group survives. Post-hoc
  semantic repairs for mouse, arrows, completion, app, or workflow groups are
  not allowed
- Norwegian/raw completion keys are a backup group for physical keys missing
  from L0. They should be grouped together in normal-keyboard-like order,
  preferably on the far-right side, but not as a hardcoded layer role. Since the
  group is too small to fill a whole layer, it should be mixed with compatible
  workflow shortcuts. Logged usage determines access priority relative to
  workflows: lower usage means lower-priority access; higher usage can outrank
  less-used workflows
- thumb access is preferred over finger access
- direct L0 thumb access is preferred for high-traffic layers
- nested access is expensive
- momentary-into-momentary access is very expensive
- mouse shortcuts and mouse-layer access can compete for L0/thumb slots when
  usage data supports them
- scroll is a trackball pointer-to-wheel mode switch with a layer side effect;
  do not model `ScrollUp`/`ScrollDown` as ordinary keypress shortcuts and do
  not treat any layer as a hardcoded scroll-only destination
- `ScrollUp` and `ScrollDown` must not appear in protected/static/dynamic
  groups. Scroll grouping is represented by the scroll-mode access capability,
  not by fake wheel-direction keypresses
- target-generation run validity requires a generated dynamic mouse workflow
  layer. If the run reaches its target generation and no non-L0/non-L7 layer
  contains every core mouse button, MB1-MB2-MB3-MB4-MB5, accessible on the right
  side of that same layer, the entire run is invalid. Mouse buttons on that
  generated mouse layer must not occupy the right-thumb area. That same layer
  must also include right-hand non-thumb momentary Scroll, no momentary
  mouse-layer access on the right thumb side, and a reachable toggle access path. This is a final acceptance
  constraint, not a requirement that every intermediate generation already
  satisfy it
- momentary Scroll is part of the core mouse group on the generated mouse
  layer. Because scrolling is a high-frequency trackball/mouse action, a toggle
  Scroll alone does not satisfy dynamic mouse-layer success; the required
  Scroll capability is right-hand, non-thumb, and momentary
- mouse-button placement on the generated mouse layer is usage-weighted. More
  used mouse buttons should receive better positions, but they still compete
  with stronger workflow shortcuts. Mouse duplicates outside that layer are
  allowed but have lower duplicate value and must be justified by usage/access
  evidence. MB1/MB2 and MB4/MB5 receive small soft bonuses for left-to-right,
  close, same-row pair placement
- after a natural generated mouse layer exists, it should dominate mouse
  interactions. Mouse buttons on other layers receive lower value and extra
  cleanup pressure, but exceptions are allowed when usage/access evidence shows
  a workflow uses mouse buttons heavily enough that switching to the mouse layer
  each time would be more expensive
- layer-aware logger data must record which shortcuts and mouse/scroll actions
  occur while each layer is active. The optimizer may use this as evidence, but
  must not convert observed layer numbers into permanent semantic layer roles
- L7 is not a generated mouse layer. It is frozen RPG/arrows/navigation/
  keyboard-system fallback and must not satisfy mutable/generated mouse
  workflow requirements. Frozen L7 content is not an optimizer acceptance
  surface.
- L7 access is the only L7 acceptance check: L7 must be reachable by both a
  momentary layer access capability and a toggle layer access capability
- L7 affects generated layers only by owning Bluetooth/output/keyboard-system
  keys and by making mutable raw arrows less important because frozen L7 arrows
  already exist
- mutable raw arrows are lower value than before. If a workflow/layer uses raw
  arrows enough to earn them outside L7, all four arrows must be on one layer
  in one of two shapes only: a single row ordered `Left Up Down Right`, or a
  two-row cluster with `Left Down Right` on the bottom row and `Up` directly
  above `Down`
- thumb clearance is strict and dynamic. For any non-L0/non-L7 layer, thumb
  positions on a side are restricted if that layer is accessed by a momentary
  thumb key from that side. The restricted side must be empty on the target
  layer. Both thumb areas become available only when the layer has reachable
  toggle access or has momentary thumb access from both left and right sides.
  If either freeing condition is later lost, any keys in the newly restricted
  thumb area make the layout invalid until moved
