#!/usr/bin/env bash
set -euo pipefail
echo "== repo =="
pwd
git status --short 2>/dev/null || true
echo
echo "== tools =="
for t in codex claude kimi-code kimi rg fd ast-grep just tokei repomix uv ruff pytest biome pre-commit smithery node npm python3 files-to-prompt gitingest; do
  command -v "$t" >/dev/null 2>&1 && printf "%-18s %s\n" "$t" "$(command -v "$t")" || true
done
echo
echo "== project markers =="
ls -1 package.json pyproject.toml Cargo.toml go.mod CMakeLists.txt requirements.txt 2>/dev/null || true
echo
echo "== skills =="
find .agents/skills .kimi-code/skills -name "SKILL.md" 2>/dev/null | sed 's#/SKILL.md##; s#^./##' | sort || true
echo
echo "== mcp =="
ls -1 .mcp.json .kimi-code/mcp.json 2>/dev/null || true
