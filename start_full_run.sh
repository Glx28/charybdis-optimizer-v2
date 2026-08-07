#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GENS="${GENS:-500000}"
POP_SIZE="${POP_SIZE:-1500}"
RUN="${RUN:-v2_full_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-$(date +%s)}"
SESSION="${SESSION:-charybdis_v2_$RUN}"
LOG="build/run_logs/${RUN}.log"
OUT_DIR="build/runs/${RUN}"

mkdir -p "$OUT_DIR" build/run_logs

# This launcher owns sessions with the charybdis_v2_ prefix. Rerunning it is
# intentionally a restart: stop the previous optimizer before starting the
# corrected code, instead of leaving two GPU runs competing for the device.
# tmux exits 1 when no server exists; that is the normal first-run state.
OLD_SESSIONS="$(tmux list-sessions -F '#S' 2>/dev/null || true)"
OLD_SESSIONS="$(printf '%s\n' "$OLD_SESSIONS" | awk '/^charybdis_v2_/ {print}')"
if [[ -n "$OLD_SESSIONS" ]]; then
  while IFS= read -r old_session; do
    [[ -z "$old_session" ]] && continue
    tmux kill-session -t "$old_session" 2>/dev/null || true
    echo "Stopped previous optimizer session: $old_session"
  done <<< "$OLD_SESSIONS"
fi

OLD_PIDS="$(pgrep -f -- "$ROOT/run_evolution.py" || true)"
if [[ -n "$OLD_PIDS" ]]; then
  while IFS= read -r old_pid; do
    [[ -z "$old_pid" ]] && continue
    kill "$old_pid" 2>/dev/null || true
    echo "Stopped previous optimizer process: $old_pid"
  done <<< "$OLD_PIDS"
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
