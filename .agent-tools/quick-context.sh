#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.ai/context/quick-context.md"
mkdir -p "$(dirname "$OUT")"

cat > "$OUT" <<'HEAD'
# Quick Context — Charybdis Optimizer V2

This is a minimal context pack for Kimi sessions. For full repo context, run `just ai-context`.

HEAD

for f in AGENTS.md KIMI.md SESSION_HANDOFF.md justfile; do
    echo "## $f" >> "$OUT"
    echo '```' >> "$OUT"
    head -n 200 "$ROOT/$f" >> "$OUT" || true
    echo '```' >> "$OUT"
    echo "" >> "$OUT"
done

for f in run_evolution.py fitness/kernel.py evolution/custom_ga.py; do
    echo "## $f (first 200 lines)" >> "$OUT"
    echo '```python' >> "$OUT"
    head -n 200 "$ROOT/$f" >> "$OUT" || true
    echo '```' >> "$OUT"
    echo "" >> "$OUT"
done

echo "Generated: $OUT"
