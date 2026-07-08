#!/usr/bin/env bash
set -euo pipefail
mkdir -p .ai/context
{
  echo "# Repo map"
  echo
  echo "Generated: $(date -Iseconds)"
  echo
  echo "## Git status"
  git status --short 2>/dev/null || true
  echo
  echo "## Files"
  if command -v rg >/dev/null 2>&1; then
    rg --files -g '!node_modules' -g '!dist' -g '!build' -g '!target' -g '!.git' -g '!.venv' -g '!venv' -g '!__pycache__' -g '!.ai' | sed 's#^#- #'
  else
    find . -type f | sed 's#^\./#- #'
  fi
  echo
  echo "## Language summary"
  command -v tokei >/dev/null 2>&1 && tokei . || true
} > .ai/context/repo-map.md
if command -v repomix >/dev/null 2>&1; then
  repomix --output .ai/context/repomix.xml --style xml --remove-comments --remove-empty-lines --ignore "node_modules,dist,build,target,.git,.venv,venv,__pycache__,.ai" >/dev/null 2>&1 || true
fi
echo "Wrote .ai/context/repo-map.md"
[ -f .ai/context/repomix.xml ] && echo "Wrote .ai/context/repomix.xml"
