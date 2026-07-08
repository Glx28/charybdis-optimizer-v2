#!/usr/bin/env python3
"""Per-layer role profile for a checkpoint.

Usage:
    python3 tools/layer_profile.py [checkpoint_path|latest]
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools._common import (
    checkpoint_to_layout,
    compute_reachability,
    load_checkpoint,
    load_evaluator,
    resolve_checkpoint_path,
)


def main():
    parser = argparse.ArgumentParser(description="Per-layer role profile")
    parser.add_argument("checkpoint", nargs="?", default="latest", help="Checkpoint path or 'latest'")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)

    ev = load_evaluator()
    layout = checkpoint_to_layout(ckpt, ev.model.layout)
    arrays = ev.model.arrays
    pos_layer = arrays[1]
    pos_frozen = arrays[5]
    sc_is_mouse = arrays[16]
    sc_mouse_btn = arrays[17]

    reach = compute_reachability(layout.genome, layout, arrays=arrays)
    all_hops = reach["all_hops"]
    hold_hops = reach["hold_hops"]
    toggle_hops = reach["toggle_hops"]

    # Per-layer aggregates
    layer_apps = defaultdict(lambda: defaultdict(float))
    layer_categories = defaultdict(lambda: defaultdict(float))
    layer_usage = defaultdict(float)
    layer_shortcuts = defaultdict(int)
    layer_empty = defaultdict(int)
    layer_mouse = defaultdict(set)
    layer_arrows = defaultdict(set)
    layer_norwegian = defaultdict(set)

    for i, sid in enumerate(layout.genome):
        lyr = int(pos_layer[i])
        if pos_frozen[i] and lyr == 7:
            continue  # skip frozen L7 positions from mutable analysis
        if sid < 0:
            layer_empty[lyr] += 1
            continue
        s = layout.shortcuts[sid]
        layer_shortcuts[lyr] += 1
        usage = float(getattr(s, "usage", 0.0))
        importance = float(getattr(s, "importance", 0.0))
        layer_usage[lyr] += usage * importance
        app = getattr(s, "app", None) or "Unknown"
        cat = getattr(s, "category", None) or "Unknown"
        layer_apps[lyr][app] += usage * importance
        layer_categories[lyr][cat] += usage * importance

        if sc_is_mouse[sid] and sc_mouse_btn[sid] > 0:
            layer_mouse[lyr].add(int(sc_mouse_btn[sid]))
        if s.keys in {"Left", "Right", "Up", "Down"}:
            layer_arrows[lyr].add(s.keys)
        if s.keys in {"Å", "Ø", "Æ", "å", "ø", "æ"}:
            layer_norwegian[lyr].add(s.keys)

    print(f"=== Layer Profile: {os.path.basename(ckpt_path)} ===")
    print()
    print("| Lyr | Reach | Hold | Tog | Shortcuts | Empty | Top App | Top Category | Mouse | Arrows | NO |")
    print("|-----|-------|------|-----|-----------|-------|---------|--------------|-------|--------|----|")
    for lyr in sorted(layer_shortcuts.keys() | layer_empty.keys()):
        if lyr == 7:
            continue
        ah = all_hops.get(lyr, -1)
        hh = hold_hops.get(lyr, -1)
        th = toggle_hops.get(lyr, -1)
        reach_s = f"{ah}" if ah >= 0 else "-"
        hold_s = f"{hh}" if hh >= 0 else "-"
        tog_s = f"{th}" if th >= 0 else "-"
        top_app = max(layer_apps[lyr], key=layer_apps[lyr].get) if layer_apps[lyr] else "-"
        top_cat = max(layer_categories[lyr], key=layer_categories[lyr].get) if layer_categories[lyr] else "-"
        mouse_s = ",".join(str(b) for b in sorted(layer_mouse[lyr])) or "-"
        arrow_s = ",".join(sorted(layer_arrows[lyr])) or "-"
        no_s = ",".join(sorted(layer_norwegian[lyr])) or "-"
        print(
            f"| {lyr:>3} | {reach_s:>5} | {hold_s:>4} | {tog_s:>3} | "
            f"{layer_shortcuts[lyr]:>9} | {layer_empty[lyr]:>5} | {top_app[:7]:>7} | "
            f"{top_cat[:12]:>12} | {mouse_s:>5} | {arrow_s:>6} | {no_s:>2} |"
        )


if __name__ == "__main__":
    main()
