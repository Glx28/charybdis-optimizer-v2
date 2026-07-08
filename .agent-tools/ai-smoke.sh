#!/usr/bin/env bash
set -euo pipefail
bash .agent-tools/ai-guard.sh
if find . -maxdepth 4 -name "*.py" -not -path "./.venv/*" -not -path "./venv/*" | grep -q .; then
  command -v ruff >/dev/null 2>&1 && ruff check . --output-format=concise || true
  if [ -d tests ] || find . -maxdepth 4 \( -name "test_*.py" -o -name "*_test.py" \) | grep -q .; then
    # Try venv first, then system pytest
    if [ -f .venv/bin/pytest ] && [ -d tests ]; then
      .venv/bin/pytest -q tests/ || true
    elif [ -f .venv/bin/python ] && [ -d tests ]; then
      .venv/bin/python -m pytest -q tests/ || true
    else
      command -v pytest >/dev/null 2>&1 && pytest -q || true
    fi
  fi
fi
if [ -f package.json ]; then
  npm run lint --if-present || true
  npm test --if-present || true
  command -v biome >/dev/null 2>&1 && biome check . || true
fi
if [ -f Cargo.toml ]; then
  cargo check || true
  cargo test || true
fi
echo "Smoke completed."
