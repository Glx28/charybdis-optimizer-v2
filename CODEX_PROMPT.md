# Charybdis Optimizer V2 Goal Prompt

Work only in `/home/nos/charybdis/charybdis-optimizer-v2`.

Non-negotiable GPU rule:

- Production training must be GPU-primary.
- Do not start or continue `run_evolution.py` if CUDA is unavailable or if the
  active training path is CPU/Numba-primary.
- CPU is allowed only for tests, static checks, diagnostics, or an explicitly
  requested CPU-only smoke test.
- If GPU-primary training is not implemented, stop and implement/report that
  blocker instead of running CPU training.
- Preserve `GPU_TRAINING_POLICY.md` in every future agent/handoff update.

The optimizer must discover a coherent keyboard layout from real usage data:
apps, shortcut sequences, shortcut workflow windows, app transitions, mouse
actions, scroll-mode actions, and access-path effort. The layout should feel
like a natural power-user office workflow surface after a reasonable learning
period with the coach app.

Core framing:

- The optimizer is workflow-focused first. A workflow is what the usage data
  shows the user does together, not a fixed app or layer name.
- Only L0 and L7 have stable roles. L0 is base typing/thumb access. L7 is
  frozen RPG/arrows/navigation plus keyboard-system controls such as
  Bluetooth/output selection.
- All other layers are dynamically assigned. Do not hardcode a mouse layer,
  scroll layer, travel layer, app layer, or system layer by number.
- Generated layers should be distinct enough to improve real workflows. A
  high-frequency "everything" layer should naturally emerge when usage data
  supports it: this is the go-to layer when the user is unsure, concentrating
  the globally most common/high-value actions. Additional layers must not
  become redundant copies of it.
- Layers may share app focus while serving different shortcut workflows.
  Do not enforce app-pure layers.
- Workflow coherence is primary. App coherence is backup only: when two
  generated workflow layers become too similar, app coherence may help one
  resolve toward an app-specific workflow while the other remains the stronger
  workflow-specific layer. App coherence must not override logged workflow
  sequences, workflow windows, or app workflow clusters.
- Layer similarity is directly penalized. If two generated layers are too
  similar, the optimizer should push them apart until each is unique enough to
  be worth switching to. The one emergent everything layer may overlap broadly
  because broad coverage is its job; specialized workflow layers still need
  distinct purpose.
- When the same shortcut appears across multiple layers, prefer the same
  physical key or nearby region for familiarity. Familiarity is pairwise
  Euclidean exponential attraction: the closer two placements are, the more
  sharply the reward rises toward its maximum at the exact same coordinate.
  Far-apart repeats get little or no useful attraction. This is a strong soft preference, not a hard
  rule; a more important workflow action may earn that position. Gate this
  familiarity reward by exceptionality so ordinary repeated shortcuts can be
  displaced or receive worse placement when a layer-specific workflow shortcut
  is more useful.
- Cross-layer shortcut repetition should be exceptional, not default. Use a
  sigmoid-weighted novelty style of pressure: keep the few shared shortcuts
  that are genuinely high-value across workflows, then let layer similarity pressure
  replace ordinary repeats with workflow-specific shortcuts so generated layers
  become diverse and useful.
- Apply that same novelty principle to duplicate scoring, familiarity, and
  layer similarity as one balanced system. Non-exceptional duplicates should
  be removed or become unlikely. Exceptional duplicates may stay and should
  receive familiarity support, but they still lose prime positions when a
  stronger workflow-specific shortcut earns that layer.
- Group scoring means compactness on an existing layer, not forcing a group
  onto one layer. If grouped shortcuts already coexist on a layer, keep them
  close there. If they are on different layers, group scoring should not
  penalize that or move them across layers. Group-aware mutation may move a
  whole protected group as a unit when a large overwrite would replace half or
  more of that group, then fill the old positions with the overwriting
  shortcuts. Do not add post-hoc semantic repair for mouse, arrows,
  completion, app, or workflow groups.
- Norwegian/raw completion keys are backup physical keys missing from L0. They
  should form a normal-keyboard-like, preferably far-right cluster, but not a
  dedicated hardcoded layer. The cluster should share a mixed workflow/backup
  layer, and its access priority should follow logged usage relative to real
  workflows.

Hard final-run rule:

At target generation, the whole run is invalid unless a generated non-L0/non-L7
dynamic mouse workflow layer exists. That layer must contain MB1-MB2-MB3-MB4-MB5
on the right side of the same layer, with no mouse button on the right-thumb
area. It must also contain right-hand non-thumb momentary Scroll, no momentary
mouse-layer access on the right thumb side, and a reachable toggle access path.
L7 is excluded from generated mouse-layer candidates and cannot count as mouse
success. Frozen L7 content is not acceptance-checked by the optimizer.

L7 access is checked separately from L7 content. Frozen L7 must be reachable by
both a momentary layer access capability and a toggle layer access capability.
The optimizer must not inspect or fail L7 because of its frozen key contents.

Mutable raw arrows are no longer strongly desirable because frozen L7 already
provides fallback arrows. If workflow evidence earns raw arrows on a generated
layer, they must be all four arrows on one layer in exactly one of two shapes:
one row ordered `Left Up Down Right`, or two rows with `Left Down Right` on the
bottom row and `Up` directly above `Down`. Partial or differently shaped raw
arrow fragments should be penalized or cleared.

Momentary Scroll is part of the core mouse group. It is highly important
because scrolling is a major trackball/mouse action. Toggle Scroll may exist,
but it does not satisfy the generated dynamic mouse-layer condition; the
required capability is momentary Scroll on the right side and off the
right-thumb area. Momentary Scroll on x7 or x8 is uncomfortable and does not
satisfy this condition.

Mouse placement inside the generated mouse layer is usage-weighted rather than
coordinate-fixed. More-used mouse buttons should get better positions unless a
more important workflow shortcut earns that position. MB2 is more valuable than
MB3/MB4/MB5 and should win better placement when usage does not prove otherwise.
Mouse duplicates outside the generated mouse layer are allowed only when
usage/access support justifies them, and they are harder to justify than
ordinary shortcut duplicates. Mouse buttons are forbidden on right-thumb
positions on every generated layer. Keep small soft relative biases: MB1
left/close/same-row with MB2, and MB4 left/close/same-row with MB5.

After a natural generated mouse layer exists, it should dominate mouse
interactions and create natural cleanup pressure against mouse buttons scattered
on other layers. Do not ban exceptions: if a workflow uses mouse buttons so
heavily that switching to the mouse layer every time is more costly, usage data
may justify copies on that workflow layer.

This is a final acceptance constraint. Intermediate generations may explore
without a complete mouse layer, but the target checkpoint must either pass this
constraint or report precisely what is missing.

Run discipline:

- Use bounded generation targets appropriate to the change. Do not default to
  extreme generation counts unless there is a documented reason.
- Inspect runs sparingly: start, one meaningful middle checkpoint, and end are
  usually enough.
- If the target checkpoint fails the hard final-run rule, mark the whole run
  invalid and explain which condition was missing.
- Do not patch generated apply/verify JavaScript by hand. Fix generator or
  optimizer logic.
