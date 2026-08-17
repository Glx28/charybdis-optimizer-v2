#!/usr/bin/env python3
"""Report semantic cluster quality for a checkpoint.

Usage:
    python3 tools/semantic_cluster_report.py [checkpoint_path|latest]
"""
import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools._common import load_checkpoint, load_layout, resolve_checkpoint_path


def _cluster_quality(layout, cluster):
    members = list(cluster.get("members", []))
    sids = [int(m.get("sid", -1)) for m in members]
    orders = [int(m.get("order", 0)) for m in members]
    offsets = [(float(m.get("dx", 0.0)), float(m.get("dy", 0.0))) for m in members]

    # Find positions.
    placements = []
    for sid in sids:
        idx = layout.get_position_of(sid)
        if idx is None:
            placements.append(None)
        else:
            pos = layout.positions[idx]
            placements.append({
                "idx": idx,
                "layer": int(pos.layer),
                "x": float(pos.x),
                "y": float(pos.y),
                "hand": pos.hand,
            })

    # Layer distribution.
    layer_counts = defaultdict(int)
    for p in placements:
        if p is not None:
            layer_counts[p["layer"]] += 1

    if not layer_counts:
        return {"name": cluster.get("name"), "status": "unassigned"}

    dominant_layer = max(layer_counts, key=lambda k: layer_counts[k])
    total_assigned = sum(layer_counts.values())
    split = total_assigned - layer_counts[dominant_layer]
    fully_together = split == 0 and len(members) == total_assigned

    # Relative position check on dominant layer.
    anchor_pos = None
    for sid, order, placement in zip(sids, orders, placements):
        if placement is not None and order == 0 and placement["layer"] == dominant_layer:
            anchor_pos = placement
            break

    order_errors = 0
    total_order_error = 0.0
    if anchor_pos is not None:
        ax, ay = anchor_pos["x"], anchor_pos["y"]
        for sid, order, off, placement in zip(sids, orders, offsets, placements):
            if placement is None or order <= 0:
                continue
            if placement["layer"] != dominant_layer:
                continue
            expected_x = ax + off[0]
            expected_y = ay + off[1]
            dx = placement["x"] - expected_x
            dy = placement["y"] - expected_y
            err = math.sqrt(dx * dx + dy * dy)
            if err > 0.5:
                order_errors += 1
                total_order_error += err

    shortcut_labels = []
    for sid, placement in zip(sids, placements):
        sc = layout.shortcuts[sid]
        label = f"{sc.keys} ({sc.action})"
        if placement is not None:
            label += f" → L{placement['layer']} x{placement['x']:.0f}y{placement['y']:.0f}"
        else:
            label += " → unassigned"
        shortcut_labels.append(label)

    return {
        "name": cluster.get("name"),
        "weight": float(cluster.get("weight", 1.0)),
        "dominant_layer": dominant_layer,
        "layer_counts": dict(layer_counts),
        "split": split,
        "fully_together": fully_together,
        "order_errors": order_errors,
        "total_order_error": total_order_error,
        "shortcuts": shortcut_labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Semantic cluster quality report")
    parser.add_argument("checkpoint", nargs="?", default="latest", help="Checkpoint path or 'latest'")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)
    layout = load_layout()

    # Load checkpoint genome into layout.
    best_genome = ckpt.get("best_genome") or ckpt.get("genome")
    if best_genome is not None:
        layout = layout.clone_with(genome=np.asarray(best_genome, dtype=np.int32))

    clusters = list(layout.semantic_clusters)
    if not clusters:
        print("No semantic clusters defined in this layout.")
        return

    print(f"=== Semantic Cluster Report: {os.path.basename(ckpt_path)} ===")
    print(f"Clusters: {len(clusters)}")
    print()

    results = [_cluster_quality(layout, c) for c in clusters]
    together = sum(1 for r in results if r.get("fully_together"))
    order_ok = sum(1 for r in results if r.get("order_errors", 0) == 0 and r.get("fully_together"))

    print(f"Fully together on one layer: {together}/{len(clusters)}")
    print(f"Together AND correct relative position: {order_ok}/{len(clusters)}")
    print()

    for r in sorted(results, key=lambda x: -x.get("weight", 1.0)):
        status = "OK" if r.get("fully_together") and r.get("order_errors", 0) == 0 else "SPLIT"
        print(f"[{status}] {r['name']}  weight={r.get('weight', 1.0):.2f}")
        print(f"  layers: {r.get('layer_counts', {})}")
        if not r.get("fully_together"):
            print(f"  split members: {r.get('split', 0)}")
        if r.get("order_errors", 0) > 0:
            print(f"  order errors: {r['order_errors']}  total error: {r['total_order_error']:.2f}")
        for label in r.get("shortcuts", []):
            print(f"    {label}")
        print()


if __name__ == "__main__":
    main()
