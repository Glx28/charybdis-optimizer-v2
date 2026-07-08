#!/usr/bin/env bash
set -euo pipefail
for t in codex claude kimi-code kimi rg fd ast-grep just tokei repomix uv ruff pytest biome pre-commit smithery node npm python3 files-to-prompt gitingest; do
  if command -v "$t" >/dev/null 2>&1; then
    printf "OK  %-18s %s\n" "$t" "$(command -v "$t")"
  else
    printf "MISS %-18s\n" "$t"
  fi
done
