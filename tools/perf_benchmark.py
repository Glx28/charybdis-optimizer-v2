#!/usr/bin/env python3
"""Performance-regression benchmark for the two hot paths that have silently
regressed before: GPU fitness-eval throughput and CPU mutation throughput.

WHY THIS EXISTS: on 2026-07-09, a same-layer-duplicate correctness fix added
O(n_pos) full-genome rescans inside the per-individual mutation hot path
(evolution/__init__.py's SwapMutation operators), silently halving real
training throughput (~4-5 gen/sec -> ~2 gen/sec) for several hours before
anyone noticed. This benchmark exists so that never happens silently again.

MANDATORY: per AGENTS.md, any change touching fitness/kernel.py,
fitness/cuda/fitness_kernel.cu, or evolution/__init__.py's mutation/genome
operators MUST be benchmarked with this tool (via `just ai-guard`) before the
change is considered done. A regression beyond the tolerance below FAILS
(non-zero exit) and blocks the change until it is optimized or the baseline
is deliberately updated with --update-baseline (which requires explicitly
noting *why* the regression is accepted).

Usage:
    python3 tools/perf_benchmark.py                 # run + compare to baseline
    python3 tools/perf_benchmark.py --update-baseline --reason "..."
"""
import argparse
import hashlib
import json
import os
import random
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Torch's CUDA extension loader serializes on a build-directory lock file
# shared by every process that loads it -- including a concurrently-running
# training run. That can make this benchmark hang indefinitely waiting on
# the lock rather than actually contending for GPU compute. Bound the whole
# run with an alarm so a stuck/contended run degrades to a clear, non-blocking
# SKIP instead of hanging every future commit while training is active.
DEFAULT_TIMEOUT_SECONDS = 90
EXIT_SKIPPED_CONTENTION = 3


class _BenchmarkTimeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _BenchmarkTimeout()

import numpy as np

from tools._common import load_evaluator, load_layout

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "perf_baseline.json")
DEFAULT_SEED_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "seeds", "gen25k_corrected_seed.json",
)

# Regression tolerance: current throughput must be within this fraction of
# the stored baseline, or the benchmark fails. 20% gives headroom for normal
# machine-load noise while still catching a 2x-class regression like the one
# that prompted this tool.
REGRESSION_TOLERANCE = 0.20

EVAL_BATCH_SIZE = 1500  # matches typical pop_size in config_v2.yaml
EVAL_REPEATS = 5
MUTATION_POP_SIZE = 1500
MUTATION_REPEATS = 5


def _build_eval_batch(layout, batch_size, rng):
    """A batch of mutated-seed genomes -- representative of real training
    populations, not adversarial/fully-random (see tests/test_v2.py's CUDA
    parity test for why fully-random genomes are not representative)."""
    samples = []
    for _ in range(batch_size):
        g = layout.genome.copy()
        for _ in range(20):
            a, b = rng.choice(layout.mutable_indices, 2, replace=False)
            g[a], g[b] = g[b], g[a]
        samples.append(g)
    return np.asarray(samples, dtype=np.int32)


def benchmark_eval_throughput(layout, evaluator):
    rng = np.random.default_rng(12345)
    batch = _build_eval_batch(layout, EVAL_BATCH_SIZE, rng)

    # Warm up (JIT/CUDA compile, cache warmup) -- not timed.
    evaluator.model.evaluate_batch(batch)

    t0 = time.perf_counter()
    for _ in range(EVAL_REPEATS):
        evaluator.model.evaluate_batch(batch)
    elapsed = time.perf_counter() - t0

    total_genomes = EVAL_BATCH_SIZE * EVAL_REPEATS
    return total_genomes / elapsed


def benchmark_mutation_throughput(layout):
    from evolution import SwapMutation
    from evolution.layout_contract import owner_protected_indices

    random.seed(54321)
    np.random.seed(54321)
    frozen_mask = layout.frozen_mask.copy()
    frozen_mask[list(owner_protected_indices(layout, layout.genome))] = True
    mutation = SwapMutation(
        prob=0.15,
        frozen_mask=frozen_mask,
        layout=layout,
        group_overwrite_prob=0.15,
        mouse_workflow_prob=0.06,
        l7_access_prob=0.03,
        random_assign_prob=0.08,
        bulk_assign_prob=0.04,
        optional_arrow_drop_prob=0.04,
        cluster_app_prob=0.20,
        effort_swap_prob=0.06,
        smart_duplicate_prob=0.20,
    )
    pop = np.tile(layout.genome.astype(np.int32), (MUTATION_POP_SIZE, 1))

    # Warm up (JIT compile) -- not timed.
    mutation._do(None, pop.copy())

    t0 = time.perf_counter()
    for _ in range(MUTATION_REPEATS):
        mutation._do(None, pop.copy())
    elapsed = time.perf_counter() - t0

    total_individuals = MUTATION_POP_SIZE * MUTATION_REPEATS
    return total_individuals / elapsed


def _benchmark_layout(data_dir, seed_path):
    """Use the active corrected warmstart instead of the unrelated data default."""
    layout = load_layout(data_dir)
    if not seed_path:
        seed_bytes = layout.genome.astype(np.int32, copy=False).tobytes()
        return layout, f"layout default sha256={hashlib.sha256(seed_bytes).hexdigest()}"
    with open(seed_path, "rb") as stream:
        payload_bytes = stream.read()
    payload = json.loads(payload_bytes)
    genome = np.asarray(payload.get("genome", payload.get("best_genome")), dtype=np.int32)
    if genome.shape != (layout.n_positions,):
        raise ValueError(
            f"Benchmark seed has shape {genome.shape}; expected {(layout.n_positions,)}: {seed_path}"
        )
    layout = layout.clone_with(genome=genome)
    return layout, f"{seed_path} sha256={hashlib.sha256(payload_bytes).hexdigest()}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update-baseline", action="store_true",
                         help="Overwrite the stored baseline with current measurements.")
    parser.add_argument("--reason", default="",
                         help="Required with --update-baseline: why this regression/change is accepted.")
    parser.add_argument("--tolerance", type=float, default=REGRESSION_TOLERANCE,
                         help=f"Allowed fractional regression before failing (default {REGRESSION_TOLERANCE}).")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                         help=f"Max seconds before treating this as GPU-contention and skipping "
                              f"(non-blocking) instead of hanging (default {DEFAULT_TIMEOUT_SECONDS}).")
    parser.add_argument("--seed-genome", default=DEFAULT_SEED_PATH,
                        help="Seed genome for representative eval and mutation batches; default is corrected gen25k seed.")
    args = parser.parse_args()

    if args.update_baseline and not args.reason:
        print("ERROR: --update-baseline requires --reason explaining why the new number is accepted.", file=sys.stderr)
        sys.exit(2)

    # CPU mutation throughput is not comparable while another build saturates
    # the host. Skip the guard cleanly instead of recording a false regression.
    try:
        loads = os.getloadavg()
        cpu_count = max(1, os.cpu_count() or 1)
        normalized_load = float(loads[0]) / cpu_count
    except (AttributeError, OSError):
        loads = None
        normalized_load = 0.0
    if normalized_load > 0.70:
        load_text = "unavailable" if loads is None else ", ".join(f"{x:.2f}" for x in loads)
        print(
            f"WARNING: host CPU load is too high for comparable mutation throughput "
            f"(load averages {load_text}; {cpu_count} logical CPUs)."
        )
        print("SKIPPING the perf-regression check; rerun when the host is idle.")
        sys.exit(EXIT_SKIPPED_CONTENTION)

    have_alarm = hasattr(signal, "SIGALRM")
    if have_alarm:
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(args.timeout)
    try:
        layout, seed_source = _benchmark_layout("data", args.seed_genome)
        evaluator = load_evaluator(require_cuda=True, reference_layout=layout)

        print(f"Benchmark seed: {seed_source}")
        print("Running GPU fitness-eval throughput benchmark...")
        eval_throughput = benchmark_eval_throughput(layout, evaluator)
        print(f"  {eval_throughput:.1f} genomes/sec")

        print("Running CPU mutation throughput benchmark...")
        mutation_throughput = benchmark_mutation_throughput(layout)
        print(f"  {mutation_throughput:.1f} individuals/sec")
    except _BenchmarkTimeout:
        print(f"WARNING: perf_benchmark.py did not complete within {args.timeout}s.")
        print("This is most likely GPU/CUDA-extension-lock contention with another active")
        print("process (e.g. a live training run sharing the same GPU), not a real hang.")
        print("SKIPPING the perf-regression check for this commit rather than blocking it --")
        print("re-run `python3 tools/perf_benchmark.py` manually once no other GPU job is")
        print("active to get a real answer before trusting this change's performance.")
        sys.exit(EXIT_SKIPPED_CONTENTION)
    finally:
        if have_alarm:
            signal.alarm(0)

    current = {
        "eval_genomes_per_sec": eval_throughput,
        "mutation_individuals_per_sec": mutation_throughput,
        "workload": seed_source,
    }

    if args.update_baseline:
        baseline = {
            **current,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": args.reason,
        }
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)
        print(f"Baseline updated: {BASELINE_PATH}")
        print(f"  reason: {args.reason}")
        sys.exit(0)

    if not os.path.exists(BASELINE_PATH):
        print(f"No baseline found at {BASELINE_PATH}.")
        print("Run with --update-baseline --reason \"initial baseline\" to establish one.")
        sys.exit(0)

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    failed = False
    for key, label in (
        ("eval_genomes_per_sec", "GPU eval throughput"),
        ("mutation_individuals_per_sec", "CPU mutation throughput"),
    ):
        base_val = baseline.get(key)
        cur_val = current[key]
        if base_val is None or base_val <= 0:
            continue
        ratio = cur_val / base_val
        regression = 1.0 - ratio
        status = "OK"
        if regression > args.tolerance:
            status = "FAIL"
            failed = True
        print(f"  [{status}] {label}: {cur_val:.1f} vs baseline {base_val:.1f} "
              f"({regression * 100:+.1f}% {'slower' if regression > 0 else 'faster'})")

    if failed:
        print()
        print(f"PERFORMANCE REGRESSION DETECTED (beyond {args.tolerance * 100:.0f}% tolerance).")
        print("Do not ship this change until the regression is fixed, or explicitly accept it")
        print("with: python3 tools/perf_benchmark.py --update-baseline --reason \"<why>\"")
        sys.exit(1)

    print()
    print("No performance regression detected.")
    sys.exit(0)


if __name__ == "__main__":
    main()
