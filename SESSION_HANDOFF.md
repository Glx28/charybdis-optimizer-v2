# Charybdis Optimizer V2 Handoff

Date: 2026-07-01 after dynamic mouse-layer policy correction
Repo: `/home/nos/charybdis/charybdis-optimizer-v2`

This handoff is for continuing optimizer-agent code work. Keep work in this v2 repo. Do not port optimizer logic from the old sibling repo unless you first recreate or adapt it here.

## GPU Training Policy

Production training must be GPU-primary. Do not start or continue
`run_evolution.py` if CUDA is unavailable or if the active training path is
CPU/Numba-primary. CPU is allowed only for unit tests, static checks,
diagnostics, or an explicitly requested CPU-only smoke test. If GPU-primary
training is not implemented, stop and report/implement that blocker before any
real run. See `GPU_TRAINING_POLICY.md`.

## Current Run

Active run:

- PID: `10947`
- Status: stopped because user correctly rejected extreme `999999` generation runs unless necessary.

- No active run should be trusted until the corrected dynamic mouse-layer policy is fully validated.
- The previous bounded run `v2_acceptance_archive_30k_20260701_023134` used an obsolete mouse acceptance definition: right-hand mouse placement was enough. It was stopped/invalidated after the user clarified the final product is invalid without a generated dynamic mouse layer.
- The prior bounded run `v2_acceptance_archive_30k_20260701_022713` was stopped at gen500 because the first checkpoint archived `optimizer_side_pass: false` under an obsolete mouse-right-hand-only check.
- Mouse repair was removed. Do not use direct genome patching to make a mouse layer pass. The reward system must make a generated mouse workflow layer naturally attractive, and final acceptance must invalidate target checkpoints that still lack it.
- Validation after patch: `py_compile` passed for the optimizer entrypoints and `pytest tests/test_v2.py tests/test_completion_cluster.py -q` passed.

Previous long run:

- Folder: `/home/nos/charybdis/charybdis-optimizer-v2/build/runs/v2_global_best_20260630_181630`
- Reached retained checkpoint `v2_checkpoint_gen86500.json`; log continued to about gen86820, then process exited.
- Best archived layout was gen84500 total `-20.5515`, constraints `[0.0, 0.0]`, but `optimizer_side_pass: false` because mutable mouse placement was `0/3` right hand.
- That exposed a bug: global archive compared hard constraints and total score only, so an acceptance-invalid high-scoring layout could become the saved best.
- Gen5500 from that run remained externally clean (`bad_literal_count 0`, no fake scroll/chord shortcuts, no mutable Bluetooth/output rows, no raw non-L7 arrows), but it did not prove the corrected dynamic mouse-layer requirement.

Current run monitoring policy:

- Be efficient. Do not repeatedly inspect every small percentage interval.
- Do not start a new run until the corrected dynamic mouse-layer checks pass tests.
- The next run should use a bounded target, not `999999`, unless a specific reason is documented.
- Checkpoint `factor_scores` for `workflow_coherence`, `app_coherence`, `trackball_proximity`, and `familiarity` are compatibility placeholders and may show `0.0`. Do not read that as missing scoring pressure; the compiled kernel folds workflow, app coherence, trackball, familiarity, mouse effective access, and mouse workflow terms into objective 3.

There is currently no trusted active run. Do not start one until the user approves the corrected policy or asks for the next bounded run.

Useful checks:

```bash
ps -eo pid,etime,pcpu,pmem,args | rg 'run_evolution.py|PID'
tail -n 80 build/runs/v2_global_best_20260630_181630/run.log
ls -t build/runs/v2_global_best_20260630_181630/v2_checkpoint_gen*.json | head
```

## Non-Negotiable Policy

- The optimizer is workflow-focused first. Workflows are inferred from apps,
  shortcuts, mouse actions, scroll-mode actions, shortcut sequences, app
  transitions, and repeated windows that usage data shows are used together.
- Generated layers should be distinct workflow surfaces. A high-frequency
  "everything" layer should naturally emerge from global usage data as the
  user's go-to layer when unsure, but other generated layers must not drift
  into redundant copies of it.
- Layers may share the same app focus while serving different shortcut
  workflows. Do not enforce app-pure layers.
- Workflow coherence comes before app coherence. App coherence is a fallback
  for resolving overly similar generated workflow layers, not a primary
  app-pure-layer objective.
- Layer similarity is a direct penalty. If two generated workflow layers are
  too similar, the optimizer should push them apart until each has a distinct
  purpose worth switching to. The single emergent everything layer may overlap
  broadly because broad coverage is its purpose; specialized layers still need
  unique workflows.
- Repeated shortcuts across layers should stay in the same physical key or
  nearby region when possible. Familiarity is pairwise Euclidean exponential
  attraction between placements: the closer placements are, the more sharply
  the reward rises toward its maximum at exact coordinate matches. Far-apart
  repeats provide little useful attraction. This rule is soft; a stronger workflow action may take
  the position when usage and effort justify it. Familiarity is also
  exception-weighted, so ordinary repeated shortcuts do not keep prime
  positions just because they repeat.
- Cross-layer repetition should be exceptional. The intended scoring shape is
  sigmoid-weighted novelty: keep only the most useful shared shortcuts across
  layers when importance/frequency/workflow evidence justifies them, and let
  similarity pressure replace ordinary repeats with workflow-specific shortcuts.
- Duplicate scoring, familiarity, and layer similarity should use one
  balanced novelty concept. Non-exceptional duplicates should be penalized or
  become unlikely; exceptional duplicates can stay familiar, but only when they
  beat the workflow-specific alternatives for that layer.
- Group scoring is same-layer compactness only. It may reward compact group
  members that already coexist on a layer, but it must not penalize or move
  group members merely because they are on different layers. Whole-group
  movement belongs in mutation proposals, not post-hoc layout patching.
- Norwegian/raw completion keys are a backup physical-key group for keys missing
  from L0. They should be normal-keyboard-like, preferably far right, and mixed
  with workflow shortcuts rather than treated as a dedicated layer. Their layer
  priority should be based on logged usage versus workflow usage.
- Layer access buttons are not fixed. They are first-class evolvable capabilities.
- Never preserve/freeze layer access positions as a shortcut around scoring.
- Scroll is a trackball pointer-to-wheel mode switch with layer side effect. Do not model fake `ScrollUp` / `ScrollDown` keypress shortcuts.
- Do not hardcode a scroll layer, mouse layer, or `MB1` location.
- Dynamic layers are desired. App-pure layers are not required when workflow data supports mixed layers.
- Layer 7 is frozen and intentionally important as RPG/arrows/navigation/keyboard-system fallback. L7 is not a generated mouse layer and must not satisfy generated mouse workflow success. Frozen L7 content is not an optimizer acceptance surface.
- L7 access is checked separately from content: L7 must be reachable through both momentary and toggle layer access capabilities.
- L7 affects generated layers only by owning Bluetooth/output/keyboard-system keys and by lowering the value of mutable raw arrows because frozen L7 arrows already exist.
- Target-generation run validity requires a generated non-L0/non-L7 dynamic mouse workflow layer. If a run reaches its target generation without a layer containing MB1-MB2-MB3-MB4-MB5 all accessible on the right side of that same layer, no mouse buttons on that layer's right-thumb area, right-hand non-thumb momentary Scroll on that layer, no momentary mouse-layer access on the right thumb side, and a reachable toggle access path to it, the whole run is invalid and must be reported as such. This is final acceptance only; intermediate generations do not have to satisfy the complete mouse-layer shape.
- Momentary Scroll is part of the core mouse group. Toggle Scroll may exist, but it does not satisfy dynamic mouse-layer success; the required Scroll capability is right-hand, non-thumb, and momentary.
- Mouse buttons on the generated mouse layer are placed by usage-weighted effective effort. Higher-frequency mouse buttons should earn better positions, but still compete with stronger workflow shortcuts. Mouse duplicates outside the generated mouse layer are allowed only with usage/access support and are harder to justify than ordinary shortcut duplicates. Soft pair biases prefer MB1 left/close/same-row with MB2 and MB4 left/close/same-row with MB5.
- Once a natural generated mouse layer exists, it should dominate mouse
  interactions. Mouse buttons on other layers get lower value and extra cleanup
  pressure, but high-usage/access-supported workflow exceptions can remain when
  switching to the mouse layer every time would be too costly.
- The logger/aggregator must preserve active-layer shortcut evidence through `by_layer_shortcut` and `layer_shortcuts`. This is usage evidence for future re-evolution, not permission to assign fixed semantic roles to layer numbers.
- Thumb clearance is strict and dynamic. For any non-L0/non-L7 layer, a thumb
  side is restricted when that layer is accessed by a momentary thumb key from
  that side, and the same side's thumb area on the target layer must be empty.
  Both thumb areas become available only with reachable toggle access or
  momentary thumb access from both left and right sides. If either freeing
  condition is later lost, keys in the newly restricted thumb area make the
  layout invalid until moved.
- Lack of logger evidence means “uncertain / needs more data”, not automatically “bad”.
- Use Norwegian Windows HID semantics. Studio/ZMK parameters are HID names; display labels are Norwegian OS-layout results.

## Current Architecture

Main files:

- `core/loader.py`: builds positions, shortcuts, usage stats, dynamic layer-access shortcuts, seed genome.
- `core/__init__.py`: dataclasses. `Shortcut` includes layer-access metadata; `UsageData` includes workflow, app, mouse, and scroll telemetry.
- `core/norwegian_keys.py`: Norwegian HID metadata, literal canonicalization, raw completion family.
- `fitness/kernel.py`: authoritative compiled scoring model. Most active optimizer behavior lives here.
- `fitness/evaluator.py`: wrapper around `FitnessModel`.
- `fitness/factors/violation.py`: Python-side compatibility/fallback factor; keep it aligned enough for tests, but the kernel is the real scoring path.
- `run_evolution.py`: exact-eval evolution loop, checkpointing, duplicate reporting, stagnation escape.
- `config_v2.yaml` and `config/__init__.py`: keep defaults aligned.
- `LAYER_ACCESS_POLICY.md`: warning/policy doc to prevent fixed-access regressions.

Tools-side scripts that matter for validation:

- `/home/nos/charybdis/charybdis-tools/runtime/evolved_v2_export/export_and_analyze_linux.py`
- `/home/nos/charybdis/charybdis-tools/runtime/evolved_v2_export/analyze_checkpoint_standalone.py`

The analyzer was fixed tools-side so empty mutable slots no longer fall back to stale canonical labels. If an analyzer report says arrows are present, verify it is using the fixed script.

## Already Implemented

- Workflow telemetry ingestion structures:
  - `shortcut_sequences`
  - `shortcut_workflows`
  - `app_sequences`
  - `app_workflows`
  - mouse click/session and scroll mode usage fields
- Workflow-aware duplicate classification:
  - workflow-supported duplicates
  - uncertain duplicates needing more data
  - unsupported duplicates
  - multi-workflow duplicates
- Supported duplicates are allowed; unsupported duplicates are penalized softly. Frozen L7 RPG/arrows/keyboard-system duplicates are ignored for duplicate failure reporting.
- Bluetooth/output-selection keys are removed from the mutable genome because L7 owns them.
- Fake/non-direct shortcuts are filtered from normal shortcut sampling:
  - `ScrollUp`
  - `ScrollDown`
  - `gg`
  - `gi`
  - `yy`
  - `Ctrl+K S`
- Dynamic layer access:
  - layer access entries are represented as assignable shortcut capabilities like `@access:Lx->Ly:hold/toggle:Label`
  - access graph is rebuilt from the candidate genome inside the kernel
  - nested access, momentary-into-momentary access, non-thumb access, and access depth are scored
- left/right occupied thumb path logic is modeled so a held thumb access key makes that side’s thumb keys unusable or expensive on the target layer unless there is toggle access or another viable thumb-side path
- Mouse/scroll scoring is generalized:
  - physical action effort
  - layer access path effort
  - hold/toggle cost
  - trackball-hand conflict
  - usage frequency
  - workflow transition cost
- Norwegian HID export canonicalization is working. Literal symbols like `+`, `-`, `=`, `` ` ``, `]`, `Page Up`, and `Page Down` export as Studio/HID names such as `Equals and Plus`, `Dash and Underscore`, `Grave Accent and Tilde`, `Right Brace`, `PageUp`, `PageDown`.
- Mutable raw non-L7 arrows are not currently present in corrected analyzer reports. Earlier reports of raw arrow fragments were analyzer artifacts.
- Stagnation escape exists in `run_evolution.py`: after long stagnation it keeps elites and injects diverse structurally valid candidates rather than only increasing mutation.

## Latest Code Change

The latest patches changed Norwegian completion reporting, mouse reporting/scoring, arrow reporting, shortcut importance overrides, checkpoint-level acceptance reporting, checkpoint retention, and group-aware mutation.

Problems fixed:

- Modified shortcuts like `Ctrl+-`, `Alt+=`, `Ctrl+Page Up`, etc. were allowed to satisfy the raw Norwegian completion family.
- This let the optimizer scatter physical leftover keys across many layers while still getting partial credit.
- Old completion repair functions were removed. Completion-key movement now happens through group-aware mutation proposals plus scoring pressure, not post-hoc layout patching.
- The completion report did not clearly distinguish raw base layers from modified variant layers.
- Mouse reporting previously treated L7 mouse as fallback context. This is now wrong: L7 must not be a generated mouse-layer candidate and cannot count as mouse success. L7 content itself is not acceptance-checked.
- The old mouse repair approach was removed. Dynamic mouse workflow must emerge from scoring pressure and final acceptance checks, not from manually moving mouse buttons after evolution.
- Old arrow repair functions were removed. Mutable arrows now survive only through scoring/selection and whole-arrow-group mutation proposals.
- Config importance overrides only matched `Shortcut.keys`, so aliases like Teams `Up` with `base_key='UpArrow'` stayed hard-important and conflicted with the no-mutable-raw-arrow policy.

New behavior:

- `shortcut_raw_completion` marks demand for the physical key family.
- `shortcut_raw_completion_base` marks only unmodified raw HID keys.
- Modified variants create demand, but only unmodified raw keys can satisfy the completion cluster.
- The scoring now strongly prefers one non-L7 anchor layer with compact, ordered, adjacent raw HID completion keys.
- If modified variants exist but the raw base key is missing from the anchor, the layout pays a large penalty.
- `completion_cluster_report` is analysis-only and no longer builds repaired candidate layouts.
- `completion_cluster_report` includes explicit raw-base fields and booleans: `raw_base_layers`, `ordered_left_to_right`, `anchor_contains_all_reachable_raw_base_keys`, `raw_base_concentrated_le_2_layers`, and `acceptance_pass`.
- `arrow_report` includes `acceptance_pass`; it reports partial mutable raw arrows but does not clear, move, or complete them.
- `acceptance_report.dynamic_mouse_layer` detects mouse workflow candidates by contents and access path while excluding L7 as a generated mouse-layer candidate. `acceptance_report.layer7_access` checks only momentary plus toggle access to frozen L7.
- There is no mouse repair helper. Final success requires the generated dynamic mouse-layer acceptance check; intermediate generations receive soft scoring pressure only.
- Loader importance overrides match both `sc.keys` and `sc.base_key`, so `Up`/`Down` aliases inherit `UpArrow`/`DownArrow` override weights.
- Checkpoints now include `acceptance_report` with optimizer-side checks and an explicit external export-validation pending flag.
- `ExactEvalCallback` now maintains a global exact best archive. Checkpoints write `best_*` from the best feasible exact layout seen so far and also include `population_best_*` diagnostics. This prevents mature checkpoints from regressing when the population drifts.

Validation before restart:

```bash
.venv/bin/python -m pytest tests/test_v2.py -q
.venv/bin/python -m pytest tests/test_v2.py tests/test_completion_cluster.py -q
```

Compiled smoke also passed with feasible hard constraints.

## Remaining Main Task

The unsolved optimizer problem is the Norwegian raw physical-key completion cluster.

Desired behavior:

- Treat this as a family/set, not scattered individual shortcuts.
- The family is:
  - `Dash and Underscore`
  - `Equals and Plus`
  - `Grave Accent and Tilde`
  - `Right Brace`
  - `PageUp`
  - `PageDown`
  - `Home`
  - `End`
- If modified variants of these keys exist, the unmodified raw key should be available somewhere memorable.
- Prefer all or most raw base keys on one chosen anchor layer.
- Allow at most two layers when there is a strong reason, but penalize 3+ layers heavily.
- Within the anchor layer, reward compact region placement, row adjacency, and left-to-right physical-key order.
- Do not solve this by hardcoding a layer number. The optimizer should choose the anchor layer.

Important current diagnostic pattern:

- Earlier mature runs had good mouse/scroll/duplicates but completion keys stayed scattered over about 8 layers.
- Stronger scalar weights alone did not solve it.
- The scoring needs structural pressure and group-aware mutation support aimed at moving the family together. Do not add post-hoc semantic repair.

## Suggested Next Optimizer Work

1. Inspect the latest run after gen5000 or later.
   - Check whether the raw base completion keys now concentrate on one anchor layer.
   - Do not overreact to gen500/gen1500 quality; this optimizer often needs several thousand generations.

2. If completion is still scattered, improve group-aware mutation and scoring.
   - Allow large overwrite mutations to displace half or more of a protected group.
   - Move the full group elsewhere as a unit when that happens.
   - Fill the vacated slots with the overwriting shortcuts.
   - Keep feasibility and frozen positions intact.
   - Do not use repaired candidate layouts or hardcoded final placement.

3. Add explicit report fields for completion cluster quality.
   - anchor layer
   - raw base keys present on anchor
   - raw base keys missing
   - modified variants that create demand
   - layers used by the family
   - compactness/order score

4. Keep improving effective access scoring only if analysis shows a real issue.
   - Do not hardcode a mouse layer number.
   - Enforce that a generated dynamic mouse workflow exists by contents and access path.
   - Evaluate low effective access cost for high-frequency mouse/scroll capabilities.

5. Keep duplicate policy nuanced.
   - Supported, exceptional duplicates are good.
   - Ordinary duplicates should be removed, softly penalized, or made unlikely.
   - Uncertain duplicates should be monitored or softly penalized.
   - Unsupported duplicates can be penalized.
   - Frozen L7 duplicates should generally be ignored.

6. Keep analyzer/export in sync with optimizer decoding.
   - If exported CSV and standalone analyzer disagree, fix the decoder source first.
   - Empty mutable slots must not inherit stale labels from canonical files.

## Acceptance Checks

- `Win+S` remains present and important.
- No fake `ScrollUp` / `ScrollDown` normal keypress assignments.
- Scroll access is modeled as trackball scroll-mode access with layer side effect.
- No hardcoded `L6` scroll layer assumption.
- Direct/non-L7 raw arrows either do not appear, or all four form a workflow-justified cluster on one layer in one of the two accepted shapes: `Left Up Down Right` on one row, or `Left Down Right` on the bottom row with `Up` directly above `Down`.
- L7 content is frozen and not optimizer acceptance-checked. Acceptance only requires both momentary and toggle access to L7.
- L7 still affects mutable layers: Bluetooth/output keys stay out of the mutable genome, and frozen L7 arrows lower the need for mutable raw arrows.
- Norwegian export has `bad_literal_count 0`.
- Norwegian completion family becomes a coherent memorably placed physical-key cluster.
- Duplicate report has unsupported duplicates at or near zero; uncertain duplicates are not labeled as bad just because logger data is new.

## Useful Commands

Run tests:

```bash
.venv/bin/python -m pytest tests/test_v2.py -q
```

Compile/evaluator smoke:

```bash
.venv/bin/python - <<'PY'
from core.loader import build_layout
from fitness.evaluator import FitnessEvaluator
from config import Config
cfg = Config.load('config_v2.yaml')
fitness = cfg.raw.get('fitness', {})
layout = build_layout('data', fitness)
e = FitnessEvaluator(
    weights=cfg.get('fitness.weights', {}),
    reference_layout=layout,
    violation_weights=cfg.get('fitness.violation_sub_weights', {}),
    missing_important_threshold=cfg.get('fitness.missing_important_threshold', 6.0),
    hard_constraints=cfg.get('fitness.hard_constraints', []),
)
obj, con = e.evaluate_batch(layout.genome.reshape(1, -1))
print('objectives', obj[0].tolist())
print('constraints', con[0].tolist())
PY
```

Start a fresh run after code changes:

```bash
.venv/bin/python - <<'PY'
import os, subprocess, sys, datetime
root = os.getcwd()
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(root, 'build', 'runs', f'v2_NEXT_NAME_{stamp}')
os.makedirs(run_dir, exist_ok=True)
log_path = os.path.join(run_dir, 'run.log')
log = open(log_path, 'ab', buffering=0)
proc = subprocess.Popen([
    sys.executable, 'run_evolution.py',
    '--generations', '999999',
    '--output-dir', run_dir,
], cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
with open(os.path.join(run_dir, 'run.pid'), 'w') as f:
    f.write(str(proc.pid))
print('pid', proc.pid)
print('run_dir', run_dir)
print('log', log_path)
PY
```

Checkpoint summary:

```bash
.venv/bin/python - <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
be = d.get('best_exact') or {}
objs = be.get('objectives') or d.get('best_objectives') or []
print('checkpoint', p.resolve())
print('generation', d.get('generation'))
print('stagnation', d.get('stagnation_count'))
print('objectives', objs)
print('total', be.get('total_score', sum(objs) if objs else None))
print('constraints', be.get('constraints') or d.get('best_constraints'))
PY build/runs/<run>/v2_checkpoint_gen500.json
```

## Pitfalls

- Do not trust old analyzer output for raw arrows unless the empty-slot fallback bug is fixed.
- Do not restore the removed placement seed file. Structural keyboard data lives in data/layout.json; mutable placements must start random.
- Do not revive `fitness/batch_evaluator.py` as the primary scoring path. The current authoritative path is `fitness/kernel.py`.
- Do not make layer access fixed again.
- Do not judge missing logger support as negative proof while the logger is new.
- Do not use US symbol names as if they were Norwegian output characters.
- Do not judge current run quality from gen500 alone.
