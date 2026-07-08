#!/usr/bin/env python3
"""Report dynamic thumb-clearance/access-side rules for a checkpoint.

Usage:
    python3 tools/thumb_clearance_report.py [checkpoint_path|latest]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution.acceptance import _momentary_only_thumb_clearance_report
from tools._common import checkpoint_to_layout, load_checkpoint, load_evaluator, resolve_checkpoint_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Report momentary-access thumb clearance")
    parser.add_argument("checkpoint", nargs="?", default="latest", help="Checkpoint path or 'latest'")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)
    ev = load_evaluator()
    layout = checkpoint_to_layout(ckpt, ev.model.layout)
    report = _momentary_only_thumb_clearance_report(layout)

    print(f"=== Thumb Clearance Report: {os.path.basename(ckpt_path)} ===")
    print(f"acceptance_pass: {bool(report.get('acceptance_pass'))}")
    print()

    for row in report.get("layers", []):
        layer = row["layer"]
        reason = row.get("reason", "restricted")
        print(f"L{layer}: {reason}")
        if row.get("momentary_thumb_hands") is not None:
            print(f"  momentary thumb sides: {row.get('momentary_thumb_hands')}")
        if row.get("restricted_hands"):
            print(f"  restricted sides: {row.get('restricted_hands')}")
        if row.get("toggle_access"):
            toggles = [
                f"{t['keys']}@L{t['source_layer']} pos{t['idx']} {t['hand']}"
                for t in row.get("toggle_access", [])
            ]
            print(f"  reachable toggles: {', '.join(toggles)}")
        if row.get("effort_floor_assignments"):
            print("  allowed by toggle, but effective effort floor applies:")
            for item in row["effort_floor_assignments"]:
                print(
                    f"    pos{item['idx']} {item['hand']} thumb "
                    f"native_effort={item['native_effort']:.2f} "
                    f"effective_floor={item['effective_effort_floor']:.1f} "
                    f"{item['keys']}"
                )
        if row.get("violating_assignments"):
            print("  violations:")
            for item in row["violating_assignments"]:
                print(f"    pos{item['idx']} {item['hand']} thumb {item['keys']}")
        print()

    if report.get("violations"):
        print("Violating layers:", ", ".join(f"L{v['layer']}" for v in report["violations"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
