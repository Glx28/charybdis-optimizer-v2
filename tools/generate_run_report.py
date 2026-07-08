#!/usr/bin/env python3
"""Generate a complete run-report bundle under build/run_report/.

Usage:
    python3 tools/generate_run_report.py [build_dir]
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools._common import find_latest_checkpoint


def _run(args, cwd=None):
    result = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate full run report bundle")
    parser.add_argument("build_dir", nargs="?", default="build", help="Build directory")
    args = parser.parse_args()

    build_dir = args.build_dir
    latest = find_latest_checkpoint(build_dir)
    if latest is None:
        print(f"No checkpoints found in {build_dir}")
        sys.exit(1)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(build_dir, "run_report", stamp)
    os.makedirs(out_dir, exist_ok=True)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python = sys.executable

    reports = [
        ("run_report.md", [python, "tools/run_report.py", build_dir, "--write"]),
        ("checkpoint_audit.md", [python, "tools/checkpoint_audit.py", latest]),
        ("layer_profile.md", [python, "tools/layer_profile.py", latest]),
        ("shortcut_audit.md", [python, "tools/shortcut_audit.py", latest]),
        ("mouse_layer_report.md", [python, "tools/mouse_layer_report.py", latest]),
        ("arrow_cluster_report.md", [python, "tools/arrow_cluster_report.py", latest]),
        ("completion_cluster_report.md", [python, "tools/completion_cluster_report.py", latest]),
        ("constraint_trace.md", [python, "tools/constraint_trace.py", build_dir, "--write"]),
    ]

    for filename, cmd in reports:
        result = _run(cmd, cwd=repo_root)
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        if result.returncode != 0:
            print(f"WARNING: {cmd} exited {result.returncode}", file=sys.stderr)
            print(result.stderr, file=sys.stderr)

    # Copy the top-level run_report.md into the bundle too.
    top_report = os.path.join(build_dir, "run_report.md")
    if os.path.exists(top_report):
        with open(top_report, "r", encoding="utf-8") as src:
            with open(os.path.join(out_dir, "run_report_top.md"), "w", encoding="utf-8") as dst:
                dst.write(src.read())

    readme = os.path.join(out_dir, "README.md")
    with open(readme, "w", encoding="utf-8") as f:
        f.write("# Run Report Bundle\n\n")
        f.write(f"Generated: {stamp}\n\n")
        f.write("| File | Description |\n")
        f.write("|------|-------------|\n")
        for filename, _ in reports:
            f.write(f"| {filename} | see file |\n")

    print(f"Run report bundle written to: {out_dir}")


if __name__ == "__main__":
    main()
