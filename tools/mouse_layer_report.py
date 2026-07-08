#!/usr/bin/env python3
"""Focused mouse-layer report.

Usage:
    python3 tools/mouse_layer_report.py [checkpoint_path|latest]
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
from evolution.acceptance import _dynamic_mouse_layer_report


def main():
    parser = argparse.ArgumentParser(description="Mouse-layer report")
    parser.add_argument("checkpoint", nargs="?", default="latest", help="Checkpoint path or 'latest'")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)

    ev = load_evaluator()
    layout = checkpoint_to_layout(ckpt, ev.model.layout)
    arrays = ev.model.arrays
    pos_layer = arrays[1]
    pos_x = arrays[8]
    pos_y = arrays[9]
    pos_hand = arrays[3]
    pos_is_thumb = arrays[4]
    sc_is_mouse = arrays[16]
    sc_mouse_btn = arrays[17]

    reach = compute_reachability(layout.genome, layout, arrays=arrays)
    hold_hops = reach["hold_hops"]
    toggle_hops = reach["toggle_hops"]

    # Candidate mouse layers by distinct right-hand non-thumb buttons
    mr = defaultdict(set)
    btn_positions = defaultdict(list)
    for i, sid in enumerate(layout.genome):
        if sid < 0 or sid >= len(layout.shortcuts):
            continue
        if sc_is_mouse[sid] and sc_mouse_btn[sid] > 0 and pos_hand[i] == 1 and not pos_is_thumb[i]:
            lyr = int(pos_layer[i])
            if 0 < lyr < 32 and lyr != 7:
                mr[lyr].add(int(sc_mouse_btn[sid]))
                btn_positions[int(sc_mouse_btn[sid])].append((lyr, i, float(pos_x[i]), float(pos_y[i])))

    candidate = -1
    if mr:
        best_lyr = max(mr, key=lambda lyr: len(mr[lyr]))
        if len(mr[best_lyr]) >= 2:
            candidate = best_lyr

    details = _dynamic_mouse_layer_report(layout)

    print(f"=== Mouse Layer Report: {os.path.basename(ckpt_path)} ===")
    print()
    print(f"Candidate mouse layer: L{candidate}")
    best_detail = details.get("best_candidate") or {}
    print(f"dynamic_mouse_layer_present: {details.get('acceptance_pass', False)}")
    print(f"scroll_mode_access_present: {bool(best_detail.get('right_momentary_scroll_access'))}")
    print()

    if candidate >= 0:
        print(f"--- Buttons on L{candidate} ---")
        for btn in sorted(mr[candidate]):
            positions = [(i, x, y) for lyr, i, x, y in btn_positions[btn] if lyr == candidate]
            pos_strs = [f"pos{i}({x:.1f},{y:.1f})" for i, x, y in positions]
            print(f"  MB{btn}: {', '.join(pos_strs)}")
        print()
        print(f"Hold hops from L0: {hold_hops.get(candidate, -1)}")
        print(f"Toggle hops from L0: {toggle_hops.get(candidate, -1)}")

        # Right-thumb conflict on candidate layer
        rt_conflicts = []
        for i, sid in enumerate(layout.genome):
            if sid < 0 or int(pos_layer[i]) != candidate:
                continue
            if pos_hand[i] == 1 and pos_is_thumb[i] and sc_is_mouse[sid] and sc_mouse_btn[sid] > 0:
                rt_conflicts.append(i)
        print(f"Right-thumb mouse-button conflicts: {rt_conflicts or 'none'}")
        print()

    print("--- Scroll Capability on Candidate Layer ---")
    scroll_found = False
    for i, sid in enumerate(layout.genome):
        if sid < 0 or int(pos_layer[i]) != candidate:
            continue
        s = layout.shortcuts[sid]
        if getattr(s, "is_scroll_mode_access", False) or "scroll" in s.keys.lower():
            if pos_hand[i] == 1 and not pos_is_thumb[i]:
                print(f"  {s.keys} at pos{i} (R-finger, non-thumb)")
                scroll_found = True
    if not scroll_found:
        print("  No right-hand non-thumb scroll access detected.")
    print()

    if details:
        print("--- Acceptance Details ---")
        for k, v in details.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
