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
- L0 is the base typing/thumb-access layer, L1 has the fixed mouse role in
  `contracts/layout_contract_v1.json`, and L7 remains frozen with its existing
  contents. Infer other generated-layer roles from their actual bindings.
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

Hard run rule:

The versioned owner contract is a gate for seed preparation, initialization,
mutation, exact evaluation, archive selection, export, and promotion. Every
saved checkpoint must pass it. L1 is the only mouse layer; L7 is frozen and
exempt from mutable-layer checks. See the contract for fixed coordinates,
scroll count, access, lock, thumb, shortcut, grouping, effort, and Norwegian
layout requirements.

Run discipline:

- Use bounded generation targets appropriate to the change. Do not default to
  extreme generation counts unless there is a documented reason.
- Inspect runs sparingly: start, one meaningful middle checkpoint, and end are
  usually enough.
- If the target checkpoint fails the hard final-run rule, mark the whole run
  invalid and explain which condition was missing.
- Do not patch generated apply/verify JavaScript by hand. Fix generator or
  optimizer logic.
