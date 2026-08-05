# Ecosystem Contract

What `charybdis-optimizer-v2` promises to output, what each sister project
expects, and where the exchange points live. Changing anything in the
"Produced" sections is a **breaking change** — flag it explicitly and update
this file.

## Produced by optimizer-v2

| Artifact | Path | Consumer |
|---|---|---|
| Run checkpoints | `build/runs/<run>/…gen<N>.json` (checkpoint genome JSON) | charybdis-tools promotion/analysis |
| Latest-run pointer | `build/latest_run_dir` | charybdis-tools agent rules |
| Best-run results | `build/v2_evolution_results.json` (incl. top-level `genome`) | warmstart chain, diagnostics |
| Scale-factor cache | `build/v2_scale_factors.json` (keyed on weight-config hash) | internal; invalidate on weight change |
| Run logs | `build/run_logs/*.log` | humans/agents |
| Acceptance report | embedded in run outputs (`evolution/acceptance.py`) | mirrored by charybdis-tools `acceptance_check.py` |

Optimizer-v2 writes **nothing** directly into sister repos. Promotion is
performed by charybdis-tools.

## Consumed by optimizer-v2

| Input | Path | Producer |
|---|---|---|
| Layout/position spec | `data/layout.json` | derived from zmk-config `config/charybdis.json` + `layout/layout_spec.json` |
| Shortcut scores | `data/app_shortcut_scores.json` | pipeline aggregation of charybdis-tools telemetry |
| Usage stats | `data/usage_stats.json` (must preserve `by_layer_shortcut` / `layer_shortcuts`) | `pipeline/aggregate_usage.js` over charybdis-tools `runtime/shortcut_usage.jsonl` |
| Warmstart (optional) | `build/v2_local_search_result.json` (`genome` validated at load) | `tools/local_search.py` / manual recovery |

## The real export path (lives in charybdis-tools)

`charybdis-tools/runtime/evolved_v2_export/` is the exchange directory:

- `export_and_analyze_linux.py` (Linux) / `export_and_analyze.py` (Windows)
  turn a checkpoint into `evolved_apply.js`, `evolved_verify.js`,
  `evolved_keybindings_explained.csv`, `evolved_diff.txt`.
- `promote.py` validates, then copies into charybdis-zmk-config
  (`layout/final_user_layout_v2.json`, `layout/keybindings_explained.csv`,
  ZMK Studio baseline scripts) and charybdis-coach (`data/` copies).
- `acceptance_check.py` re-implements optimizer-v2's acceptance contract
  standalone (drift risk — the canonical version is `evolution/acceptance.py`
  here; the 2026-07-13 self-referential-toggle fix had to land in both).
- `release_manifest.json` hash-locks the last promotion.

`evolved_apply.js` / `evolved_verify.js` are pasted into ZMK Studio. Never
hand-edit generated JS; fix the exporter and regenerate.

## Format versioning

No explicit version field exists today. Compatibility is by file-name
convention and genome shape (`n_positions = 616`, sids in `[-1, n_shortcuts)`).
A genome written against an older data snapshot may carry stale sids —
`run_evolution.py` masks those at load rather than crashing.
