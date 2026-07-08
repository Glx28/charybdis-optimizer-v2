#!/usr/bin/env python3
"""High-value / misplaced shortcut audit.

Usage:
    python3 tools/shortcut_audit.py [checkpoint_path|latest] [--top N]
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
    parser = argparse.ArgumentParser(description="Shortcut placement audit")
    parser.add_argument("checkpoint", nargs="?", default="latest", help="Checkpoint path or 'latest'")
    parser.add_argument("--top", type=int, default=30, help="Number of shortcuts to show")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)

    ev = load_evaluator()
    layout = checkpoint_to_layout(ckpt, ev.model.layout)
    arrays = ev.model.arrays
    pos_layer = arrays[1]
    pos_effort = arrays[0]
    pos_hand = arrays[3]
    pos_is_thumb = arrays[4]
    pos_frozen = arrays[5]

    reach = compute_reachability(layout.genome, layout, arrays=arrays)
    reachable = reach["reachable_layers"]
    hold_hops = reach["hold_hops"]

    # Collect positions per shortcut
    placements = defaultdict(list)
    for i, sid in enumerate(layout.genome):
        if sid < 0 or pos_frozen[i] or int(pos_layer[i]) == 7:
            continue
        lyr = int(pos_layer[i])
        s = layout.shortcuts[sid]
        placements[s.keys].append({
            "pos": i,
            "layer": lyr,
            "effort": float(pos_effort[i]),
            "hand": "R" if pos_hand[i] == 1 else "L",
            "thumb": bool(pos_is_thumb[i]),
            "reachable": lyr in reachable,
            "hops": hold_hops.get(lyr, -1),
            "sid": sid,
        })

    scored = []
    for keys, positions in placements.items():
        best = min(positions, key=lambda p: p["effort"])
        s = layout.shortcuts[best["sid"]]
        importance = float(getattr(s, "importance", 0.0))
        usage = float(getattr(s, "usage", 0.0))
        score = importance * usage
        flags = []
        if not best["reachable"]:
            flags.append("unreachable")
        if best["hops"] > 2:
            flags.append("deep")
        if best["effort"] > 2.0 and score > 5.0:
            flags.append("misplaced")
        if len(positions) > 1:
            flags.append(f"{len(positions)}x")
        scored.append({
            "keys": keys,
            "score": score,
            "effort": best["effort"],
            "layer": best["layer"],
            "hops": best["hops"],
            "flags": flags,
            "app": getattr(s, "app", "") or "",
            "category": getattr(s, "category", "") or "",
        })

    # Sort: infeasible/deep first, then by score desc
    scored.sort(key=lambda x: (0 if x["flags"] else 1, -x["score"]))

    print(f"=== Shortcut Audit: {os.path.basename(ckpt_path)} ===")
    print()
    print("| Keys | App | Category | Score | Effort | L | Hops | Flags |")
    print("|------|-----|----------|-------|--------|---|------|-------|")
    for item in scored[:args.top]:
        flags_s = ",".join(item["flags"]) or "-"
        print(
            f"| {item['keys'][:20]:<20} | {item['app'][:10]:<10} | {item['category'][:12]:<12} | "
            f"{item['score']:>5.1f} | {item['effort']:>6.2f} | {item['layer']:>1} | "
            f"{item['hops']:>4} | {flags_s} |"
        )


if __name__ == "__main__":
    main()
