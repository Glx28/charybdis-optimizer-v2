# Validated 30,000-generation run

## Promote or apply a layout

Apply the currently audited candidate with:

```bash
rtk proxy .venv/bin/python tools/apply_layout.py
```

The script runs the contract audit before writing anything. It installs the
genome in tracked `data/default_layout_genome.json` and the local optimizer
warmstart `build/v2_local_search_result.json`, then writes
`build/current_layout.json` and its audit. The first prior local warmstart is
preserved as `build/v2_local_search_result.pre_candidate_apply.json`. New
optimizer runs and validated-run snapshots use this genome by default; an
explicit warmstart or run-directory warmstart still takes precedence. Use
`--cpu` only on hosts without CUDA. Re-run the script with another candidate
path to promote a later result.

Run from the local WSL repository with its existing `.venv`:

```bash
rtk proxy .venv/bin/python tools/validated_run.py --prepare-only --fresh-start
rtk proxy .venv/bin/python tools/validated_run.py --run-dir build/runs/validated_30k_TIMESTAMP
```

The snapshot includes current tracked/untracked source, all data inputs, the
selected checkpoint warmstart, original warmstart, scale factors, dependency versions, source hashes, HEAD,
and the dirty-worktree patch. The sparse-layer cutoff is removed; reachable
occupancy is diagnostic only. No reset, rebase or cherry-pick is needed.
Historical checkpoints copied into the source build directory are analysis-test
fixtures, not resume inputs. `--fresh-start` disables every saved-genome
injection and starts the entire population from random mutable assignments over
the frozen L0/L7 base. `--warmstart` accepts `best_genome` or `genome` only when
intentionally continuing from an audited layout; it records source and
generation separately.
The frozen config retains its 500,000 ceiling;
the production command explicitly overrides it with 30,000 generations,
population 1,500 and seed 42.

Startup requires at least 4,096 MiB free GPU memory and utilization at most 10%.
Exit 3 means the snapshot is prepared but the GPU gate remains closed. Retry the
same snapshot once the other workload finishes. CUDA remains mandatory; CPU
training is never substituted. Direct focused and full pytest exit codes,
`just ai-guard`, and a direct performance benchmark (contention skips fail) gate
a 10-generation GPU smoke at population 1,500. The redundant `just ai-smoke`
wrapper is skipped because it repeats tests and the performance guard while
suppressing test failures. Only after the direct checks and CUDA smoke pass may
production start. Raw subprocess logs preserve exact failure evidence.

The supervisor copies every observed 500-generation checkpoint into `reviews/`
before the runner prunes old checkpoints. `reviews/progress.json` tracks exact
archive generation, constraint values, acceptance failures and semantic clusters.
Production logs record stages 0, 3,000, 5,000, 7,000, 9,000 and 10,000. Stage
changes reprice the archive and parents and retrain the surrogate with current
labels; raw scores from different stages must not be compared as improvements.
Every scheduled full-population exact refresh replaces stale parent scores and
checks the exact feasible archive. When asynchronous surrogate weights or
normalization change, the live population is repriced before the next
parent/child comparison.
Exact archive ranking cannot trade lower semantic co-location or relative
placement for a shorter acceptance-failure list. PowerToys shortcuts are grouped
by launcher, visual, mouse, and workspace functions instead of one app-wide
cluster. Browser Find excludes Find My Mouse, browser panels exclude clipboard
and version history, window management excludes saved workspaces, and the
numbered Ctrl+1…9 tab grid is separate from tab navigation. Same-stem aliases
such as Ctrl+Y and Ctrl+Shift+Z remain adjacent in the Undo/Redo sequence.
Contextual shortcut groups retain explicit left/right, up/down, before/after,
and sequence geometry. These relations apply to shortcut actions, not raw arrow
keys, which have no mutable-layer placement requirement.

Audit every 500-generation checkpoint for acceptance results, hard constraints,
reachable-layer occupancy diagnostics, semantic co-location, and relative
placement. Scheduled 5,000-generation reviews compare the exact-best archive.
Do not stop for score plateaus, sparse occupancy, or soft cluster trends. Stop
and preserve a run if the archive fails a named hard contract at two consecutive
scheduled reviews without corrective progress; fix the cause before restarting.
Audit-tool errors are recorded and do not count as failed acceptance reviews.
Passing archives continue to the configured generation cap; early acceptance
remains possible after the required final review.
The run-contract gate fails when any required relative layout differs from its
declared offsets. The audit records failed cluster names so a numeric score
improvement cannot hide broken shortcut order.

Semantic split and relative-position scores are added after legacy objective
normalization. Sparse occupancy is diagnostic only, with no count cutoff or
standalone penalty. Production values are recorded in `config_v2.yaml`.

The 10-generation CUDA smoke checks completion on the CUDA-primary path, finite
exact-evaluator scores, and a returned genome. It records acceptance diagnostics
but does not demand a fully accepted layout after only 10 generations. Scheduled
production reviews enforce the hard layout contract and stagnation guard.
Snapshot preparation does not inject or arrange raw arrow keys.

Final reports use the exact-best archive and stage-correct evaluator. The contract
audit reports reachable generated-layer occupancy without a cutoff, ordinary
momentary thumb access with Scroll exemption, semantic clusters, completion keys, dynamic
mouse and Scroll access, L7 access, frozen positions, duplicates, and requested
Windows shortcuts with their live access paths. Occupancy and split critical
clusters remain review evidence without standalone cutoffs. Export validation remains a separate requirement;
a run passing optimizer checks is not automatically approved for firmware use.

```bash
rtk proxy .venv/bin/python tools/run_contract_audit.py build/runs/validated_30k_TIMESTAMP/v2_checkpoint_gen30000.json
```

For an existing snapshot, execute the audit from its `source/` directory using
the original repository's `.venv/bin/python`, so frozen data/config are used.
