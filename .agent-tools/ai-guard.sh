#!/usr/bin/env bash
set -euo pipefail
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0
changed_files="$( { git diff --name-only; git diff --cached --name-only; } 2>/dev/null | sort -u || true )"
changed_count="$(printf "%s\n" "$changed_files" | sed '/^$/d' | wc -l | tr -d ' ')"
if [ "${changed_count:-0}" -gt 80 ]; then
  echo "AI GUARD BLOCKED: more than 80 files changed. Split the task or explicitly approve the rewrite." >&2
  exit 2
fi
if printf "%s\n" "$changed_files" | grep -Ei '(\.cu$|\.cuh$|cuda|gpu|triton|kernel|nvcc|nvidia)' >/dev/null 2>&1; then
  diff="$( {
    git diff -- . ':(exclude).agent-tools/ai-guard.sh'
    git diff --cached -- . ':(exclude).agent-tools/ai-guard.sh'
  } 2>/dev/null || true )"
  cpu_fallback_pattern="cpu fall""back|force.*cpu|use_cpu|disable.*cuda|cuda.*disable|to\([\"']cpu[\"']\)|\.cpu\(\)|CUDA_VISIBLE_DEVICES *= *-1"
  if printf "%s\n" "$diff" | grep -Ei "^\+.*(${cpu_fallback_pattern})" >/dev/null 2>&1; then
    echo "AI GUARD BLOCKED: CPU-primary escape hatch detected in GPU/CUDA work." >&2
    exit 2
  fi
fi
exit 0
