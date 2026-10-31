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

L0 is base typing/thumb access, L1 is the fixed mouse surface, and L7 is the frozen RPG/arrows/navigation and keyboard-system layer. The owner contract in `contracts/layout_contract_v1.json` is authoritative for L1 and must be enforced by generation, evaluation, archive selection, export, and promotion.

Every layer other than L1 and frozen L7 remains dynamically assigned. Do not hardcode other workflow roles by layer number.

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

Modifier placement is an owner contract: keep bare LeftAlt directly on L0 for
frequent application shortcuts; keep bare RightAlt (Norwegian AltGr) on a
reachable non-L0 layer. Do not place both Alt keys on L0. Score RightAlt on L0
as a major violation while preserving the AltGr key in the layout.

The owner contract fixes L0 x3,y4 as an ordinary momentary hold to L1; fixes L1 MB1, MB2, and momentary Scroll at x8,y2, x9,y2, and x10,y2; requires all five mouse buttons, an L1 self-lock, exactly two additional current-layer scroll holds, ordinary hold access to every generated layer, strict path-based thumb clearance, and named low-effort L1 shortcuts. L7 remains byte-for-byte frozen and is exempt from mutable-layer contract checks. Do not infer or assign another mouse layer. See `contracts/layout_contract_v1.json` for the complete versioned rules.

Do not create static or dynamic groups for `ScrollUp`/`ScrollDown`; they are fake wheel-direction keypresses, not real shortcuts.

No shortcut may appear more than once on the same layer. L7 is frozen and excluded from this rule. Preserve the existing duplicate restrictions on mutable layers; the L1 contract does not authorize arbitrary mouse-button duplication.

The logger and aggregation pipeline must preserve active-layer shortcut counts
(`by_layer_shortcut` and `layer_shortcuts`) so future runs can learn what was
actually used while a layer was active. These counts are evidence, not stable
semantic layer roles.

Layer familiarity is per shortcut and per layer. Repeated observed use builds a
bounded memory weight; each mutable copy on a layer with lower observed use pays
more, especially at low-effort positions. Mature bindings also resist moving
away from their incumbent coordinate on that same layer until use supports
another layer. Low-use duplicates should move to a high-effort position or
another layer, even when the shortcut is popular elsewhere. Frozen bindings,
L7, layer-access controls, and mouse controls are exempt. Norwegian symbol
events feed the same model through produced-output counts by layer. Do not
collapse counts across layers before scoring.

Shortcut sequences and workflow chains provide a continuous distance signal:
far-apart pairs still receive a smaller closeness reward so mutation can improve
their placement incrementally. Apply the same formula in CPU and CUDA kernels.

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

Transparent/empty positions on reachable generated layers are scored by
position value. An empty slot on a prime, low-effort position has a larger
penalty; a far, high-effort slot has a smaller one. L0 is excluded: empty base
positions are valid, and filling them with arbitrary low-use shortcuts must not
earn a reward. L7 is frozen, frozen positions, and unreachable layers are also
excluded. L0 thumb bindings instead pay a soft opportunity cost that decreases
with observed shortcut use. This uses the logger's measured counts; it does not
make any binding or coordinate a hard rule.

A layer's return-to-L0 toggle existing at all is a hard constraint
(`toggle_back_to_l0`), but existing is not enough: if that toggle is placed
away from the thumb area, the user has to hunt across the whole layer to find
the way back. Every layer with a toggle back to L0 gets extra soft scoring
pressure toward putting at least one such toggle on a thumb key, on top of the
general access-layout thumb preference. This is soft pressure, not a hard
constraint — a layer can still legally keep its return toggle elsewhere, it
just competes against this pressure like any other placement choice.

Thumb clearance is strict and dynamic. If a layer is accessed by a momentary
thumb key from one side, that same side's thumb area on the target layer is
restricted and must be empty. Both thumb areas become available only when the
layer has reachable toggle access or momentary thumb access from both left and
right sides. If either freeing condition is lost later, keys in the newly
restricted thumb area make the layout invalid until moved.

L7 affects generated layers by owning Bluetooth/output/keyboard-system keys
and the frozen raw-arrow fallback. Raw arrows have no mutable-layer placement
requirement or special score.

## Run-Length Expectations

Do not treat `evolution.n_generations: 500000` as a normal target. It is an absolute ceiling for an overnight run. A generation cap limits work; it is never a minimum acceptance generation.

- **Short diagnostic / sanity run:** a few hundred generations.
- **Normal production run:** cap at 30 000 generations. Review the best exact archive at any generation; accept it when the named hard checks pass and required layout review is complete.
- Do not stop/restart a run for a score plateau, sparse occupancy, or soft cluster trend. Record those as diagnostics while the run proceeds.
- At the scheduled 5,000-generation reviews, stop and preserve a run only when its exact-best archive still fails a named hard contract at two consecutive reviews with no corrective progress. Fix the cause before restarting. A passing archive continues to the configured cap unless the user accepts it earlier.
- **Overnight ceiling:** 500 000 generations must only be used for an unattended overnight run, and the user must be told it is an overnight ceiling, not a standard target.

If the best archive fails a named hard check, use the scheduled reviews to determine whether acceptance, hard constraints, or required relative placement are correcting. Do not turn an unapproved clustering or occupancy target into a new failure gate.

## Agent Tooling Rules

Before editing: run `just ai-status` and `just ai-context`. Prefer existing repo tools (rg, fd, ast-grep, just recipes, MCP, tests, linters) over custom scripts. Make minimal diffs. Do not rewrite broad systems. Do not replace CUDA/GPU/Numba/Triton/NVIDIA logic with CPU-only logic. Do not add processor-side escape hatches to hide CUDA bugs. Do not delete tests. Keep final answers short unless asked for detail. For CUDA work: reproduce the GPU failure, inspect the smallest failing path, fix the GPU path, run relevant tests, then `just ai-guard`. Before finishing: `just ai-guard` and `just ai-smoke`.

**MANDATORY performance check.** On 2026-07-09, a correctness fix added O(n_pos)
full-genome rescans into the per-individual mutation hot path in
`evolution/__init__.py`, silently halving real training throughput (~4-5
gen/sec to ~2 gen/sec) for hours before anyone noticed. This must never
happen again. Any change to `fitness/kernel.py`, `fitness/cuda/fitness_kernel.cu`,
or `evolution/__init__.py`'s mutation/genome-generation operators is not
considered done until `python3 tools/perf_benchmark.py` has been run and shows
no regression beyond its tolerance against `tools/perf_baseline.json`. This is
already wired into `just ai-guard`, which runs it automatically and blocks
(non-zero exit) on regression whenever those files changed — do not bypass or
weaken that check. When adding correctness logic to a hot path (duplicate
checks, safety guards, validation), always reach for an existing precomputed
index (e.g. a per-layer position index) instead of a full O(n) rescan; if a
regression is genuinely unavoidable and accepted, update the baseline
explicitly with `--update-baseline --reason "<why>"`, never by deleting or
ignoring `tools/perf_baseline.json`.
