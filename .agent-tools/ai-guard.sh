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

# Perf-regression guard: a same-layer-duplicate correctness fix on 2026-07-09
# silently halved real training throughput (~4-5 gen/sec -> ~2 gen/sec) for
# hours by adding O(n_pos) full-genome rescans into the mutation hot path.
# Any change touching the fitness kernels or the mutation/genome operators
# MUST be benchmarked; a regression beyond tolerance blocks the change.
if printf "%s\n" "$changed_files" | grep -Ei '^(fitness/kernel\.py|fitness/cuda/fitness_kernel\.cu|evolution/__init__\.py)$' >/dev/null 2>&1; then
  if [ -f tools/perf_benchmark.py ] && [ -x .venv/bin/python ]; then
    echo "AI GUARD: fitness kernel or mutation operators changed — running perf_benchmark.py..." >&2
    if .venv/bin/python tools/perf_benchmark.py; then
      perf_rc=0
    else
      perf_rc=$?
    fi
    if [ "$perf_rc" -eq 3 ]; then
      echo "AI GUARD WARNING: perf_benchmark.py skipped (likely GPU contention with another" >&2
      echo "active process) -- not blocking, but re-run it manually once the GPU is free." >&2
    elif [ "$perf_rc" -ne 0 ]; then
      echo "AI GUARD BLOCKED: performance regression detected. See tools/perf_benchmark.py output above." >&2
      echo "Fix the regression, or explicitly accept it with:" >&2
      echo "  .venv/bin/python tools/perf_benchmark.py --update-baseline --reason \"<why>\"" >&2
      exit 2
    fi
  fi
fi
exit 0
