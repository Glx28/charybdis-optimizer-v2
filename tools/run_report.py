#!/usr/bin/env python3
"""Run-level summary and stagnation diagnosis.

Usage:
    python3 tools/run_report.py [build_dir] [--write]
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools._common import find_checkpoints, load_checkpoint


def _count_diversity_injections(build_dir: str) -> dict:
    """Scan run logs for diversity-injection events."""
    triggers = {}
    for log_path in glob.glob(os.path.join(build_dir, "runs", "*", "run.log")):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Diversity injection trigger:" in line:
                    trigger = line.split("Diversity injection trigger:")[1].split("(")[0].strip()
                    triggers[trigger] = triggers.get(trigger, 0) + 1
    return triggers


def _format_constraints(constraints: list) -> str:
    return ",".join(str(int(c)) for c in constraints)


def main():
    parser = argparse.ArgumentParser(description="Run-level summary and stagnation diagnosis")
    parser.add_argument("build_dir", nargs="?", default="build", help="Build directory")
    parser.add_argument("--write", action="store_true", help="Write markdown report to build/run_report.md")
    args = parser.parse_args()

    files = find_checkpoints(args.build_dir)
    if not files:
        print(f"No checkpoints found in {args.build_dir}")
        sys.exit(1)

    rows = []
    for path in files:
        ckpt = load_checkpoint(path)
        gen = ckpt.get("generation", 0)
        best_exact = ckpt.get("best_exact", {})
        pop_exact = ckpt.get("population_best_exact", {})
        best_total = float(best_exact.get("total_score", 0.0))
        pop_total = float(pop_exact.get("total_score", 0.0))
        best_gap = best_total + 49.30
        pop_gap = pop_total + 49.30
        best_pass = bool(best_exact.get("optimizer_side_pass", False))
        pop_pass = bool(pop_exact.get("optimizer_side_pass", False))
        best_constraints = best_exact.get("constraints", [])
        pop_constraints = pop_exact.get("constraints", [])
        rows.append({
            "gen": gen,
            "path": path,
            "best_gap": best_gap,
            "pop_gap": pop_gap,
            "best_total": best_total,
            "pop_total": pop_total,
            "best_pass": best_pass,
            "pop_pass": pop_pass,
            "best_cv": sum(max(0.0, float(c)) for c in best_constraints),
            "pop_cv": sum(max(0.0, float(c)) for c in pop_constraints),
            "best_constraints": best_constraints,
            "pop_constraints": pop_constraints,
        })

    # Best feasible (pass=True, cv=0) or fallback lowest total score
    feasible = [r for r in rows if r["best_pass"] and r["best_cv"] == 0.0]
    best_row = min(feasible, key=lambda r: r["best_total"]) if feasible else min(rows, key=lambda r: r["best_total"])

    latest = rows[-1]
    gens_since_best = latest["gen"] - best_row["gen"]

    # Diversity injections
    injections = _count_diversity_injections(args.build_dir)
    total_injections = sum(injections.values())

    lines = []
    lines.append("# Run Report")
    lines.append("")
    lines.append(f"**Build directory:** `{args.build_dir}`")
    lines.append(f"**Checkpoints analyzed:** {len(files)}")
    lines.append(f"**Latest generation:** {latest['gen']}")
    lines.append("")
    lines.append("## Best Result")
    lines.append("")
    lines.append(f"- Generation: {best_row['gen']}")
    lines.append(f"- Total score: {best_row['best_total']:.4f}")
    lines.append(f"- Gap: {best_row['best_gap']:+.2f}")
    lines.append(f"- Optimizer-side pass: {best_row['best_pass']}")
    lines.append(f"- Constraints: [{_format_constraints(best_row['best_constraints'])}]")
    lines.append("")
    lines.append("## Latest Population")
    lines.append("")
    lines.append(f"- Generation: {latest['gen']}")
    lines.append(f"- Population best total: {latest['pop_total']:.4f} (gap {latest['pop_gap']:+.2f})")
    lines.append(f"- Archive best total: {latest['best_total']:.4f} (gap {latest['best_gap']:+.2f})")
    lines.append(f"- Population pass: {latest['pop_pass']}")
    lines.append(f"- Constraints: [{_format_constraints(latest['pop_constraints'])}]")
    lines.append("")
    lines.append("## Stagnation Diagnosis")
    lines.append("")
    if gens_since_best == 0:
        lines.append("✓ The best result was found at the latest checkpoint.")
    else:
        lines.append(f"✗ No archive improvement for {gens_since_best} generations (since gen {best_row['gen']}).")
        if total_injections > 0:
            injection_str = ", ".join(f"{k}: {v}" for k, v in injections.items())
            lines.append(f"- Diversity injections fired: {total_injections} ({injection_str})")
        else:
            lines.append("- No diversity-injection events logged.")
        if latest["pop_cv"] > 0 and latest["best_cv"] == 0:
            lines.append("- The population best is infeasible while the archive best is feasible: "
                         "selection may be favoring high-scoring infeasible genomes.")
        elif latest["pop_cv"] == 0 and latest["best_cv"] == 0 and latest["pop_total"] > latest["best_total"] + 0.01:
            lines.append("- The population best has a worse total score than the archive best: "
                         "search may be stuck in a different basin.")
        else:
            lines.append("- Search appears stagnant; consider increasing mutation rate, "
                         "constraint-aware surrogate training, or a warm restart from the archive best.")
    lines.append("")
    lines.append("## Score Trajectory")
    lines.append("")
    lines.append("| Gen | Archive Gap | Pop Gap | Archive Pass | Pop Pass | Archive CV | Pop CV |")
    lines.append("|-----|-------------|---------|--------------|----------|------------|--------|")
    for r in rows:
        lines.append(
            f"| {r['gen']:>5} | {r['best_gap']:>+11.2f} | {r['pop_gap']:>+7.2f} | "
            f"{str(r['best_pass']):>12} | {str(r['pop_pass']):>8} | "
            f"{r['best_cv']:>10.0f} | {r['pop_cv']:>6.0f} |"
        )
    lines.append("")
    if injections:
        lines.append("## Diversity Injection Events")
        lines.append("")
        for trigger, count in injections.items():
            lines.append(f"- {trigger}: {count}")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    if args.write:
        out_path = os.path.join(args.build_dir, "run_report.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
