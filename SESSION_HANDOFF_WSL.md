# Charybdis Optimizer V2 WSL Notes

Primary repo:

```text
/home/nos/charybdis/charybdis-optimizer-v2
```

Use this v2 repo as source of truth. Do not copy optimizer files into or out of
the old sibling optimizer unless the user explicitly asks.

GPU training rule:

- Production optimizer training must be GPU-primary.
- Do not start or continue `run_evolution.py` if CUDA is unavailable or if the
  active training path is CPU/Numba-primary.
- CPU is allowed only for tests, static checks, diagnostics, or an explicitly
  requested CPU-only smoke test.
- If the runner cannot satisfy this, stop and fix/report GPU-primary training
  before any real run.
- See `GPU_TRAINING_POLICY.md`.

Environment:

- Python: `.venv/bin/python`
- Tests: `.venv/bin/python -m pytest tests/test_v2.py tests/test_completion_cluster.py -q`
- Process check: `ps -eo pid,etime,pcpu,pmem,args | rg 'run_evolution.py|PID'`

Current policy:

- Workflow-focused first: layers should be useful for apps, shortcuts, mouse
  actions, scroll-mode actions, and transitions that usage data shows are used
  together.
- Layers should not be redundant copies. A common "everything" layer is valid
  only when high-frequency usage earns it.
- Same app does not mean same layer. A single app may have several distinct
  workflows with different shortcuts.
- Repeated shortcuts across layers should stay on the same physical key or
  nearby region when possible for familiarity.
- Only L0 and L7 have stable roles. L7 is frozen RPG/arrows/navigation and
  keyboard-system fallback, not a generated mouse layer. Frozen L7 content is
  not an optimizer acceptance surface.
- L7 access is checked separately from content: L7 must be reachable by both a
  momentary layer access capability and a toggle layer access capability.
- Mutable raw arrows are lower value because L7 already provides fallback
  arrows. If raw arrows appear outside L7, they must be all four arrows on one
  layer in either one-row `Left Up Down Right` order or a two-row cluster with
  `Left Down Right` on the bottom row and `Up` directly above `Down`.
- At target generation, a run is invalid unless a generated non-L0/non-L7
  dynamic mouse layer has MB1-MB2-MB3-MB4-MB5 on the right side, right-hand
  non-thumb momentary Scroll, no mouse button on that layer's right-thumb area,
  no momentary mouse-layer access on the right thumb side, and a reachable
  toggle access path. The toggle access path does not have to originate on L0.
- Mouse-button placement on that dynamic mouse layer is usage-weighted. Mouse
  duplicates elsewhere are allowed but harder to justify than ordinary shortcut
  duplicates.
- This mouse-layer rule is final acceptance only. Intermediate generations do
  not have to satisfy it yet.

Run discipline:

- No trusted run is active unless explicitly reported in `SESSION_HANDOFF.md`.
- Prefer bounded runs sized to the current scoring change.
- Do not use old Windows paths or old nested run-directory sync instructions.
