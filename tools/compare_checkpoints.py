#!/usr/bin/env python3
"""Diff two checkpoints.

Usage:
    python3 tools/compare_checkpoints.py <ckpt_a> <ckpt_b>
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tools._common import checkpoint_to_layout, load_checkpoint, load_evaluator, resolve_checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Diff two checkpoints")
    parser.add_argument("ckpt_a", help="First checkpoint path or 'latest'")
    parser.add_argument("ckpt_b", help="Second checkpoint path or 'latest'")
    args = parser.parse_args()

    path_a = resolve_checkpoint_path(args.ckpt_a)
    path_b = resolve_checkpoint_path(args.ckpt_b)
    ckpt_a = load_checkpoint(path_a)
    ckpt_b = load_checkpoint(path_b)

    ev = load_evaluator()
    layout_a = checkpoint_to_layout(ckpt_a, ev.model.layout)
    layout_b = checkpoint_to_layout(ckpt_b, ev.model.layout)

    F_a, G_a = ev.model.evaluate_batch(layout_a.genome.reshape(1, -1))
    F_b, G_b = ev.model.evaluate_batch(layout_b.genome.reshape(1, -1))

    total_a = float(F_a[0].sum())
    total_b = float(F_b[0].sum())
    gap_a = total_a + 49.30
    gap_b = total_b + 49.30

    diff_count = int(np.sum(layout_a.genome != layout_b.genome))

    pass_a = bool(ckpt_a.get("acceptance_report", {}).get("optimizer_side_pass", False))
    pass_b = bool(ckpt_b.get("acceptance_report", {}).get("optimizer_side_pass", False))

    print("=== Compare Checkpoints ===")
    print(f"A: {os.path.basename(path_a)} (gen {ckpt_a.get('generation', 0)})")
    print(f"B: {os.path.basename(path_b)} (gen {ckpt_b.get('generation', 0)})")
    print()
    print(f"A total: {total_a:.4f} (gap {gap_a:+.2f}) pass={pass_a}")
    print(f"B total: {total_b:.4f} (gap {gap_b:+.2f}) pass={pass_b}")
    print(f"Δ total: {total_b - total_a:+.4f}  Δ gap: {gap_b - gap_a:+.2f}")
    print(f"Positions changed: {diff_count}")
    print()

    if pass_b and not pass_a:
        print("Verdict: B is a progression (feasible where A was not).")
    elif not pass_b and pass_a:
        print("Verdict: B is a regression (lost feasibility).")
    elif pass_a and pass_b:
        if total_b < total_a - 0.01:
            print("Verdict: B is a progression (lower score, both feasible).")
        elif total_b > total_a + 0.01:
            print("Verdict: B is a regression (higher score, both feasible).")
        else:
            print("Verdict: Neutral (similar score, both feasible).")
    else:
        if total_b < total_a - 0.01:
            print("Verdict: B improves raw score but both are infeasible.")
        else:
            print("Verdict: Neutral/infeasible.")


if __name__ == "__main__":
    main()
