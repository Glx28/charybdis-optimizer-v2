#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GENS="${GENS:-500000}"
POP_SIZE="${POP_SIZE:-1500}"
RUN="${RUN:-v2_full_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-$(date +%Y%m%d%H%M%S)}"
SESSION="${SESSION:-charybdis_v2_$RUN}"
LOG="build/run_logs/${RUN}.log"
OUT_DIR="build/runs/${RUN}"

mkdir -p "$OUT_DIR" build/run_logs

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "bash -lc 'cd \"$ROOT\" && .venv/bin/python -X faulthandler run_evolution.py --config config_v2.yaml --generations \"$GENS\" --pop-size \"$POP_SIZE\" --seed \"$SEED\" --no-inject-seed --output-dir \"$OUT_DIR\" 2>&1 | tee \"$LOG\"'"

cat <<EOF
Started Charybdis optimizer v2 run.

Session: $SESSION
Run:     $RUN
Seed:    $SEED
Gens:    $GENS
Pop:     $POP_SIZE
Output:  $ROOT/$OUT_DIR
Log:     $ROOT/$LOG

Attach:
  tmux attach -t $SESSION

Watch log:
  tail -f "$ROOT/$LOG"
EOF
