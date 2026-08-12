# Canonical Shortcut Corpus

These JSON files are the single source of truth for the keyboard shortcuts that
the optimizer can place and the coach can explain.

## File format

Each file is an app:

```json
{
  "id": "powertoys",
  "name": "PowerToys",
  "user_weight": 1,
  "exeNames": ["powertoys.exe"],
  "expected_count": 15,
  "shortcuts": [
    {
      "keys": "Alt+Space",
      "action": "PowerToys Run launcher",
      "category": "System Shortcuts",
      "frequency": "constant",
      "importance": 10,
      "preferred_hand": "either"
    }
  ]
}
```

- `expected_count` is the estimated number of shortcuts this app should have. If
  the actual count is much lower, `tools/validate_shortcut_coverage.py` treats it
  as a critical error.
- `frequency` is one of `constant`, `high`, `medium`, `low`, `rare`.
- `importance` is 1-10. Shortcuts with `importance >= 6` are considered
  "mapped" in the generated optimizer summary.

## Workflow

### Validate coverage

```bash
python3 tools/validate_shortcut_coverage.py
```

### Regenerate outputs

```bash
python3 tools/generate_shortcut_corpus.py
```

This produces:

- `data/app_shortcut_scores.json` (optimizer input)
- `../charybdis-coach/data/app_shortcut_reference.json` (coach reference)

### Capture shortcuts from the user's Windows PC

On the Windows host, run:

```powershell
.\powershell\export_app_shortcuts.ps1
```

Then merge the exported JSON into this corpus:

```bash
python3 ../charybdis-tools/python/merge_shortcut_export.py runtime/discovered_shortcuts.json
```

## Adding a new app

1. Create `<app_id>.json` in this directory.
2. Set a realistic `expected_count`.
3. Run `validate_shortcut_coverage.py` and `generate_shortcut_corpus.py`.
