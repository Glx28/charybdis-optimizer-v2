# Charybdis Optimizer V2

Evolutionary layout optimizer for the Charybdis keyboard.

## Agent-Specific Entry Points

- **Kimi Code CLI:** read `KIMI.md` after this file for Kimi-specific skills, MCP servers, recipes, and run-analysis commands.
- **Claude Code:** `CLAUDE.md` redirects here.
- **Codex CLI:** see `CODEX_PROMPT.md`.

## GPU Training Policy

Training must never silently fall back to CPU-primary evolution. This is a
GPU-focused optimizer. If CUDA is unavailable, or if the active training path is
CPU/Numba-primary, the run must abort before evaluator warmup and before any
generation starts. CPU-only commands are allowed for unit tests, static checks,
small diagnostics, and explicit one-off analysis, but not for production
training runs.

Any agent changing training execution must preserve this fail-fast rule. Do not
start `run_evolution.py` unless the configured path satisfies the GPU policy or
the user explicitly asks for a diagnostic CPU-only smoke test.

See `GPU_TRAINING_POLICY.md`. This rule applies to all agents, prompts,
handoffs, configs, and runner code in this repo. Do not weaken it by moving
training back to CPU, disabling the fail-fast check, or treating unused GPU code
as sufficient.

## Dynamic Layer Assignment

Only L0 and L7 have stable semantic roles. L0 is base typing/thumb access. L7 is frozen RPG/arrows/navigation plus keyboard-system controls such as Bluetooth/output selection.

Every other layer is dynamically assigned during each generation. Do not encode assumptions such as a fixed mouse layer, scroll layer, travel layer, app layer, code layer, Excel layer, DMS layer, or system layer by layer number.

This optimizer is workflow-focused first. A workflow is the set of apps,
shortcuts, mouse actions, scroll-mode actions, and app transitions that usage
data shows are often used together. Layers should become distinct workflow
surfaces, not redundant copies of each other and not app-pure buckets by
default. It is valid to have an "everything" layer for the most common actions,
but other generated layers must objectively improve high-use workflows from
the logged data.

The "everything" layer is an emergent go-to layer, not a fixed layer number or
hardcoded role. It should naturally concentrate the most globally common and
highest-value shortcuts/buttons from usage data so that when the user is
unsure, the most likely needed actions are on one easy-to-access layer. This
layer can overlap with specialized workflow layers, but it must not make the
rest of the layout redundant; other generated layers should still become better
surfaces for specific high-value workflows.

Workflow coherence has priority over app coherence. App coherence is only a
fallback pressure when generated workflow layers become too similar. In that
case, one layer should continue toward the specific workflow while another may
resolve toward a useful app-specific workflow. App coherence must not override
logged shortcut sequences, workflow windows, or app workflow clusters.

Layer similarity is a direct penalty. If two generated non-L0/non-L7 layers are
too similar, the optimizer should punish that overlap until at least one layer
develops a distinct workflow purpose worth switching to. The only layer allowed
to overlap broadly with many others is the single emergent everything layer,
because broad coverage is its purpose. Even then, specialized layers must stay
unique enough to serve their own workflows.

Mouse, scroll-mode access, app workflows, completion keys, and duplicates must be scored from actual bindings, usage data, access paths, physical effort, and workflow support. Behavior names such as `coach_l2_hold`, `coach_l6_toggle`, `coach_mouse_lock`, and `coach_travel_toggle` are access/beacon names only; they do not prove the target layer's role.

When the same shortcut appears on multiple layers, familiarity matters: the
optimizer should strongly prefer placing that shared shortcut in the same
physical position or nearby region across layers. Familiarity is pairwise
Euclidean exponential attraction between placements: the closer the placements
are, the more sharply the reward rises toward its maximum at the exact same
coordinate. Far-apart repeats have little attraction. This is a soft rule, not a
hard constraint; a more important workflow action may displace the shared
shortcut when usage and effort justify it. Familiarity itself must be gated by
exceptionality, so normal repeated shortcuts do not receive enough protection
to block stronger workflow-specific shortcuts.

Cross-layer repetition must be selective. The optimizer should behave like it
has a sigmoid-weighted novelty gate: only the most useful exceptional shared
shortcuts, justified by importance, frequency, workflow support, or duplicate
support, should survive across many layers. Ordinary repeated shortcuts should
lose to layer similarity pressure so empty/redundant spots become workflow-specific
actions that make generated layers diverse and useful.

Use that same novelty gate to balance duplicate creation, familiarity, and
layer similarity. A duplicate should be likely only when the shortcut is
exceptional enough to justify multiple placements. Familiarity should keep
those exceptional duplicates near the same physical area. Layer similarity pressure
should reclaim ordinary duplicates and repeated shortcuts for workflow-specific
actions.

Group scoring is same-layer compactness only. If a protected group has members
already placed on one layer, those members should be close together on that
layer. If the same group appears on different layers, group scoring must not
penalize that separation; workflow scoring decides whether those shortcuts
belong on the same layer. Group movement is allowed only as an evolutionary
mutation proposal: when a large overwrite would replace half or more of a
protected group, the whole group may move elsewhere as a unit and the
overwriting shortcuts fill the vacated slots. Do not add post-hoc repair that
forces mouse, arrow, completion, app, or workflow groups into a target layer.

Norwegian/raw completion keys are a backup physical-key group for keys missing
from L0. They should be grouped in normal-keyboard-like order, preferably as a
far-right cluster, but they are not a full semantic layer. Because the family
does not fill a whole layer, it should live on a mixed workflow/backup layer.
Its layer accessibility is usage-ranked: if logged raw-completion usage is
lower than a workflow, it belongs on a lower-priority layer than that workflow;
if usage is higher, it should beat less-used workflows to a more accessible
layer.

At target generation, the whole run is invalid unless a generated non-L0/non-L7
dynamic mouse workflow layer exists. That layer must contain every core mouse
button, MB1-MB2-MB3-MB4-MB5, accessible on the right side of that same dynamic
mouse layer, with no mouse button occupying the right-thumb area. Mouse-button
placement on that layer is usage-weighted: more-used mouse buttons should earn
better physical positions, but they still compete with highly used workflow
shortcuts. It must also contain right-hand non-thumb momentary Scroll capability, have
no momentary mouse-layer access on the right thumb side, and have a reachable toggle access path. L7
is excluded from generated mouse-layer candidates and must not count as mouse
workflow success. Frozen L7 content is not an optimizer acceptance surface.
This is a final acceptance constraint, not a requirement that every intermediate
generation already has the complete mouse layer.

L7 access is checked separately from L7 content. L7 must be reachable by both a
momentary layer access capability and a toggle layer access capability.

Momentary Scroll is part of the core mouse group for the generated mouse layer.
Because scrolling is a major part of mouse/trackball use, the dynamic mouse
layer is invalid without a right-hand non-thumb momentary Scroll capability on
that same layer. Momentary Scroll on x7 or x8 is considered uncomfortable and
does not satisfy the generated mouse-layer condition. Prefer momentary Scroll
over toggle Scroll for this condition.
Do not create static or dynamic groups for `ScrollUp`/`ScrollDown`; those are
fake wheel-direction keypresses, not real shortcuts.

Mouse buttons may appear on other generated layers only when usage/access data
justifies them, and mouse duplicates are harder to justify than ordinary
shortcut duplicates. Mouse buttons are forbidden on right-thumb positions on
every generated layer, not only on the dynamic mouse layer. The dynamic mouse
layer receives the primary mouse-workflow bonus. Within that layer, MB2 is
more valuable than MB3/MB4/MB5 and should win better placement when usage does
not prove otherwise. MB1 is slightly preferred left of and close to MB2; MB4 is
slightly preferred left of and close to MB5; each pair gets a small same-row
bonus. These are soft relative biases, not fixed coordinates.

Once a natural generated mouse layer exists, it should dominate mouse
interactions. Mouse buttons on other layers receive lower value and extra
cleanup pressure so random scatter disappears naturally. Exceptions are still
allowed when usage/access data shows a workflow uses mouse buttons so heavily
that switching to the mouse layer every time would be too costly.

No shortcut may appear more than once on the same layer. This is a global,
absolute rule with exactly two exceptions. First, layer 7 is frozen and
entirely excluded from this rule. Second, the dynamic mouse layer allows
exactly one extra copy of a core mouse button (MB1-MB5), and only as one
left-side plus one right-side placement — never two copies on the same side.
This mouse exception is tied to the layer's live, fully-qualifying
dynamic-mouse-layer status for the current generation, not merely to whether
the layer holds mouse buttons: if the layer stops fully qualifying as the
dynamic mouse layer for any reason (for example Scroll or toggle access
moving away from it), its left-side mouse-button copy immediately becomes an
ordinary same-layer duplicate violation. There is no post-hoc repair or
deletion of the orphaned copy; feasibility-first selection removes it over
generations, the same mechanism already used for every other hard constraint
in this document.

The logger and aggregation pipeline must preserve active-layer shortcut counts
(`by_layer_shortcut` and `layer_shortcuts`) so future runs can learn what was
actually used while a layer was active. These counts are evidence, not stable
semantic layer roles.

`layout.layer_access` is a legacy static access field. It is NOT the source of
truth for evolved layouts. New layout data should leave it empty. Layer-access shortcuts are
first-class genome capabilities: any shortcut with `is_layer_access=True` can
be placed anywhere in the mutable genome by the optimizer. All scoring,
acceptance, and reporting must infer access from the live genome (shortcut
fields `is_layer_access`, `access_target_layer`, `access_is_momentary`) and
never from `layout.layer_access`. The compiled fitness kernel enforces this
by rebuilding the reachability graph from the genome on every evaluation.
Helper functions that read `layout.layer_access` are explicitly marked as
"Legacy/static fallback only" and must not be called from scoring or acceptance
paths.

Transparent/empty keys are generally penalized according to position value.
An empty slot on a prime, low-effort position is especially costly; an empty
slot on a far, high-effort position incurs only a small penalty. The penalty
uses a sigmoid-weighted position-value function: `pos_value = 1/(1+effort)`,
gate = sigmoid(8×(pos_value − 0.5)), scaled by layer access cost and layer
demand. Cheap-access, high-demand layers produce stronger empty-position
pressure. L7 (frozen), frozen positions, and positions on unreachable layers
are excluded from this penalty. This is a soft scoring pressure (violation
sub-weight `empty_position`, default 3.0), not a hard acceptance constraint.
Intentional transparent fall-through is not banned, but it competes against
placing useful actions at each position.

L0 is deliberately NOT excluded from empty-position pressure, and gets an
extra multiplier on top: it is the only zero-cost, always-reachable layer, so
every L0 key is high-value and should be fiercely contested. An empty L0 slot
must never be cheaper than filling it with any valid shortcut — the L0-thumb
occupied-position penalty (which discourages low-usage shortcuts from
squatting on L0 thumb slots) is capped so it can never exceed what leaving
the same slot empty would cost. This is still soft pressure, not a hard
constraint, matching how mouse-layer effort quality is handled — but it
should be strong enough that a mutable L0 position sitting empty for
thousands of generations should not happen.

Thumb clearance is strict and dynamic. If a layer is accessed by a momentary
thumb key from one side, that same side's thumb area on the target layer is
restricted and must be empty. Both thumb areas become available only when the
layer has reachable toggle access or momentary thumb access from both left and
right sides. If either freeing condition is lost later, keys in the newly
restricted thumb area make the layout invalid until moved.

L7 affects generated layers only in two ways: Bluetooth/output/keyboard-system
keys are removed from the mutable genome because L7 owns them, and frozen L7
arrows make mutable raw arrows lower value unless a workflow genuinely earns
them. Mutable raw arrows, when present outside L7, must be complete on one
layer and use exactly one of two shapes: one row ordered `Left Up Down Right`,
or two rows with `Left Down Right` on the bottom row and `Up` directly above
`Down`.

## Agent Tooling Rules

Before editing: run `just ai-status` and `just ai-context`. Prefer existing repo tools (rg, fd, ast-grep, just recipes, MCP, tests, linters) over custom scripts. Make minimal diffs. Do not rewrite broad systems. Do not replace CUDA/GPU/Numba/Triton/NVIDIA logic with CPU-only logic. Do not add processor-side escape hatches to hide CUDA bugs. Do not delete tests. Keep final answers short unless asked for detail. For CUDA work: reproduce the GPU failure, inspect the smallest failing path, fix the GPU path, run relevant tests, then `just ai-guard`. Before finishing: `just ai-guard` and `just ai-smoke`.
