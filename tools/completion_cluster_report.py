#!/usr/bin/env python3
"""Norwegian / raw completion cluster report.

Usage:
    python3 tools/completion_cluster_report.py [checkpoint_path|latest]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools._common import load_checkpoint, resolve_checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Completion cluster report")
    parser.add_argument("checkpoint", nargs="?", default="latest", help="Checkpoint path or 'latest'")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)

    comp = ckpt.get("completion_cluster_report", {})
    checks = ckpt.get("acceptance_report", {}).get("checks", {})

    print(f"=== Completion Cluster Report: {os.path.basename(ckpt_path)} ===")
    print()
    print(f"norwegian_completion_cluster: {checks.get('norwegian_completion_cluster', False)}")
    print(f"Anchor layer: L{comp.get('anchor_layer', -1)}")
    print(f"Raw base keys present: {comp.get('raw_base_keys_present', [])}")
    print(f"Raw base keys missing: {comp.get('raw_base_keys_missing', [])}")
    print(f"Layers used by family: {sorted(comp.get('layers_used_by_family', []))}")
    print(f"Compactness order score: {comp.get('compactness_order_score', 0.0):.3f}")
    print(f"Ordered left-to-right: {comp.get('ordered_left_to_right', False)}")
    print(f"Raw total count: {comp.get('raw_total_count', 0)}")


if __name__ == "__main__":
    main()
