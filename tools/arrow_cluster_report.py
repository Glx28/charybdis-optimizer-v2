#!/usr/bin/env python3
"""Arrow cluster report.

Usage:
    python3 tools/arrow_cluster_report.py [checkpoint_path|latest]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools._common import load_checkpoint, resolve_checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Arrow cluster report")
    parser.add_argument("checkpoint", nargs="?", default="latest", help="Checkpoint path or 'latest'")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)

    arrow = ckpt.get("arrow_report", {})

    print(f"=== Arrow Cluster Report: {os.path.basename(ckpt_path)} ===")
    print()
    print("acceptance: not checked (diagnostic only)")
    print(f"Total raw arrow placements: {arrow.get('total', 0)}")
    print(f"Layers used: {sorted(arrow.get('layers', []))}")
    print()

    placements = arrow.get("placements", [])
    if placements:
        print("--- Non-L7 Arrow Placements ---")
        for p in placements:
            print(f"  {p}")
        print()



if __name__ == "__main__":
    main()
