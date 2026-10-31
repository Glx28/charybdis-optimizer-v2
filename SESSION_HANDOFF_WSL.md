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

- The owner contract is versioned in `contracts/layout_contract_v1.json` and
  enforced during generation, evaluation, archive selection, export, and
  promotion. L1 is always the mouse layer; L7 remains frozen and exempt.
- Exact CUDA evaluation now drives child filtering and survivor selection. The
  active optimizer no longer uses a surrogate mini-evaluation subset.

- Workflow-focused first: layers should be useful for apps, shortcuts, mouse
  actions, scroll-mode actions, and transitions that usage data shows are used
  together.
- Layers should not be redundant copies. A common "everything" layer is valid
  only when high-frequency usage earns it.
- Same app does not mean same layer. A single app may have several distinct
  workflows with different shortcuts.
- Repeated shortcuts across layers should stay on the same physical key or
  nearby region when possible for familiarity.
- L1 has the fixed mouse role. Other mutable layers can evolve useful
  workflows. L7 is frozen RPG/arrows/navigation and keyboard-system fallback;
  its existing content is preserved and exempt from mutable-layer checks.
- L7 access is checked separately from content: L7 must be reachable by both a
  momentary layer access capability and a toggle layer access capability.
- Raw arrows have no mutable-layer placement requirement; frozen L7 provides
  raw-arrow access.
- Contract violations invalidate candidates at every generation. Essential
  L0→L1 access must be ordinary momentary hold at x3,y4; fixed L1 mouse
  bindings and the exact three-scroll count follow the contract file.

Run discipline:

- Current trusted run status is recorded in `docs/validated-30k-run.md`;
  `SESSION_HANDOFF.md` is a historical snapshot.
- Production runs use the configured generation cap; checkpoint trends are review evidence, not automatic stop/restart triggers.
- Do not use old Windows paths or old nested run-directory sync instructions.
# Historical handoff — superseded for current work by
# `contracts/layout_contract_v1.json` and `docs/LAYOUT_ACCEPTANCE.md`.
