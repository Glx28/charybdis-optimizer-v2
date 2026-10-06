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
    offsets = [(float(m.get("dx", 0.0)), float(m.get("dy", 0.0))) for m in members]
    relative_required = any(abs(dx) > 0.01 or abs(dy) > 0.01 for dx, dy in offsets)

    # Collect all assigned positions for each member (shortcuts may be duplicated
    # across layers for accessibility).  For cluster-quality purposes a group is
    # "together" if a single layer contains at least one copy of every member.
    positions_by_sid = {sid: [] for sid in sids}
    for idx, sid in enumerate(layout.genome):
        if sid in positions_by_sid:
            pos = layout.positions[idx]
            positions_by_sid[sid].append({
                "idx": int(idx),
                "layer": int(pos.layer),
                "x": float(pos.x),
                "y": float(pos.y),
                "hand": pos.hand,
            })

    # Layer distribution using unique members per layer (permissive w.r.t. duplicates).
    layer_member_sets = defaultdict(set)
    for sid, places in positions_by_sid.items():
        for p in places:
            layer_member_sets[p["layer"]].add(sid)

    if not layer_member_sets:
        return {"name": cluster.get("name"), "status": "unassigned"}

    # Dominant layer: one that contains all members if possible, otherwise most members.
    full_layers = [layer for layer, sid_set in layer_member_sets.items()
                   if len(sid_set) == len(sids)]
    if full_layers:
        dominant_layer = full_layers[0]
        # Prefer the layer with the most physical placements (most copies) for reporting.
        dominant_layer = max(full_layers, key=lambda l: len(layer_member_sets[l]))
    else:
        dominant_layer = max(layer_member_sets, key=lambda l: len(layer_member_sets[l]))

    total_assigned = sum(len(s) for s in layer_member_sets.values())
    split = total_assigned - len(layer_member_sets[dominant_layer])
    fully_together = len(full_layers) > 0

    # Primary placements for reporting: prefer the dominant layer when a shortcut
    # appears there, otherwise fall back to its first assigned position.
    placements = []
    for sid in sids:
        places = positions_by_sid[sid]
        dom_place = next((p for p in places if p["layer"] == dominant_layer), None)
        if dom_place is not None:
            placements.append(dom_place)
        elif places:
            placements.append(places[0])
        else:
            placements.append(None)

    # Relative position check on dominant layer.
    anchor_index = None
    if relative_required:
        anchor_index = next((i for i, (off, placement) in enumerate(zip(offsets, placements))
                             if placement is not None and placement["layer"] == dominant_layer
                             and abs(off[0]) <= 0.01 and abs(off[1]) <= 0.01), None)

    order_errors = 0
    total_order_error = 0.0
    if anchor_index is not None:
        anchor_pos = placements[anchor_index]
        ax, ay = anchor_pos["x"], anchor_pos["y"]
        for i, (off, placement) in enumerate(zip(offsets, placements)):
            if placement is None or i == anchor_index:
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

    # Report layer counts as member counts per layer.
    layer_counts_report = {layer: len(sid_set) for layer, sid_set in layer_member_sets.items()}
    return {
        "name": cluster.get("name"),
        "category": cluster.get("category"),
        "weight": float(cluster.get("weight", 1.0)),
        "dominant_layer": dominant_layer,
        "layer_counts": layer_counts_report,
        "split": split,
        "fully_together": fully_together,
        "order_errors": order_errors,
        "total_order_error": total_order_error,
        "relative_layout_required": relative_required,
        "relative_layout_pass": bool(
            fully_together and order_errors == 0
            and any(abs(dx) > 0.01 or abs(dy) > 0.01 for dx, dy in offsets)
        ),
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
