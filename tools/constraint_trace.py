#!/usr/bin/env python3
"""Trace hard-constraint values across checkpoints.

Usage:
    python3 tools/constraint_trace.py [build_dir] [--write]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools._common import find_checkpoints, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Hard-constraint trends across checkpoints")
    parser.add_argument("build_dir", nargs="?", default="build", help="Build directory")
    parser.add_argument("--write", action="store_true", help="Write markdown report")
    args = parser.parse_args()

    files = find_checkpoints(args.build_dir)
    if not files:
        print(f"No checkpoints found in {args.build_dir}")
        sys.exit(1)

    hard_constraint_names = None
    rows = []
    for path in files:
        ckpt = load_checkpoint(path)
        gen = ckpt.get("generation", 0)
        best_exact = ckpt.get("best_exact", {})
        pop_exact = ckpt.get("population_best_exact", {})
        best_constraints = best_exact.get("constraints", [])
        pop_constraints = pop_exact.get("constraints", [])
        if hard_constraint_names is None:
            # Try to infer names from config if length matches known hard constraints
            hard_constraint_names = [f"C{i}" for i in range(len(best_constraints))]
        rows.append({
            "gen": gen,
            "best": [float(c) for c in best_constraints],
            "pop": [float(c) for c in pop_constraints],
        })

    lines = ["# Constraint Trace", ""]
    hc_header = " | ".join(hard_constraint_names)
    hc_sep = "|".join("-" * (len(n) + 2) for n in hard_constraint_names)
    pop_hc_sep = "|".join("-" * (len(n) + 4) for n in hard_constraint_names)
    header = f"| Gen | {hc_header} | Pop {hc_header} |"
    sep = f"|-----|{hc_sep}|{pop_hc_sep}|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        best_str = " | ".join(f"{v:.0f}" for v in r["best"])
        pop_str = " | ".join(f"{v:.0f}" for v in r["pop"])
        lines.append(f"| {r['gen']:>5} | {best_str} | {pop_str} |")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    if args.write:
        out_path = os.path.join(args.build_dir, "constraint_trace.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
