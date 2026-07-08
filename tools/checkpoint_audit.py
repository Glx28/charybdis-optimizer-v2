#!/usr/bin/env python3
"""One-page checkpoint audit.

Usage:
    python3 tools/checkpoint_audit.py [checkpoint_path|latest]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from tools._common import (
    checkpoint_to_layout,
    load_checkpoint,
    load_evaluator,
    mouse_layer_quality_warnings,
    resolve_checkpoint_path,
)
from evolution.acceptance import _dynamic_mouse_layer_report


def _shortcut_usage_count(layout, shortcut):
    """Best-effort usage evidence for a shortcut/action."""
    usage = 0
    data = layout.usage_data

    shortcut_stats = data.shortcuts.get(shortcut.keys, {})
    usage += int(shortcut_stats.get("count", 0) or 0)

    mouse_stats = data.mouse_clicks.get(shortcut.keys, {})
    usage += int(mouse_stats.get("count", 0) or 0)

    raw_stats = data.raw_completion_keys.get(shortcut.keys, {})
    usage += int(raw_stats.get("count", 0) or 0)

    key_lower = shortcut.keys.lower()
    action_lower = shortcut.action.lower()
    if "scroll" in key_lower or "scroll" in action_lower:
        usage += int(data.scroll_total or 0)

    return usage


def _layer_demands(layout):
    """Approximate layer demand from assigned, non-access shortcut importance/usage."""
    demand = {}
    for idx, sid in enumerate(layout.genome):
        sid = int(sid)
        if sid < 0 or sid >= len(layout.shortcuts):
            continue
        shortcut = layout.shortcuts[sid]
        if shortcut.is_layer_access:
            continue
        pos = layout.positions[idx]
        usage = _shortcut_usage_count(layout, shortcut)
        demand[pos.layer] = demand.get(pos.layer, 0.0) + float(shortcut.importance) * (1.0 + usage ** 0.5)
    return demand


def _l0_mutable_assignments(layout, top_n=24):
    """Return mutable L0 positions with usage evidence for audit/reporting."""
    demand = _layer_demands(layout)
    rows = []
    for idx, sid in enumerate(layout.genome):
        pos = layout.positions[idx]
        if pos.layer != 0 or pos.is_frozen:
            continue
        sid = int(sid)
        if sid < 0 or sid >= len(layout.shortcuts):
            rows.append({
                "idx": idx,
                "x": pos.x,
                "y": pos.y,
                "effort": pos.effort,
                "hand": pos.hand,
                "kind": "empty",
                "keys": "Transparent",
                "usage": 0,
                "importance": 0.0,
                "target": "",
                "target_demand": 0.0,
            })
            continue
        shortcut = layout.shortcuts[sid]
        target = ""
        target_demand = 0.0
        kind = "shortcut"
        if shortcut.is_layer_access:
            kind = "hold" if shortcut.access_is_momentary else "toggle"
            target = f"L{shortcut.access_target_layer}"
            target_demand = demand.get(shortcut.access_target_layer, 0.0)
        rows.append({
            "idx": idx,
            "x": pos.x,
            "y": pos.y,
            "effort": pos.effort,
            "hand": pos.hand,
            "kind": kind,
            "keys": shortcut.keys,
            "usage": _shortcut_usage_count(layout, shortcut),
            "importance": float(shortcut.importance),
            "target": target,
            "target_demand": target_demand,
        })

    rows.sort(key=lambda r: (r["effort"], r["idx"]))
    return rows[:top_n]


def _prime_empty_positions(genome, layout, arrays, top_n=5):
    """Return the lowest-effort mutable positions that are empty."""
    pos_effort = arrays[0]
    pos_layer = arrays[1]
    pos_frozen = arrays[5]
    empty = []
    for i, sid in enumerate(genome):
        if sid >= 0:
            continue
        if pos_frozen[i] or int(pos_layer[i]) == 7:
            continue
        empty.append((float(pos_effort[i]), i, int(pos_layer[i])))
    empty.sort()
    return empty[:top_n]


def main():
    parser = argparse.ArgumentParser(description="One-page checkpoint audit")
    parser.add_argument("checkpoint", nargs="?", default="latest", help="Checkpoint path or 'latest'")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)
    gen = ckpt.get("generation", 0)

    ev = load_evaluator()
    layout = checkpoint_to_layout(ckpt, ev.model.layout)
    arrays = ev.model.arrays

    F, G = ev.model.evaluate_batch(layout.genome.reshape(1, -1))
    total = float(F[0].sum())
    gap = total + 49.30

    best_exact = ckpt.get("best_exact", {})
    pop_exact = ckpt.get("population_best_exact", {})
    acc = ckpt.get("acceptance_report", {})
    current_mouse = _dynamic_mouse_layer_report(layout)
    checks = acc.get("checks", {})
    checks = dict(checks)
    checks["dynamic_mouse_layer_present"] = bool(current_mouse.get("acceptance_pass", False))
    pending_checks = set()
    if acc.get("external_export_validation_pending", False):
        pending_checks.add("norwegian_export_bad_literal_count_zero")
    failed = [k for k, ok in checks.items() if not ok and k not in pending_checks]

    print(f"=== Checkpoint Audit: {os.path.basename(ckpt_path)} ===")
    print(f"Generation: {gen}")
    archive_total = best_exact.get("total_score", total)
    pop_total = pop_exact.get("total_score", 0.0)
    print(f"Archive best total: {archive_total:.4f} (gap {archive_total + 49.30:+.2f})")
    print(f"Population best total: {pop_total:.4f} (gap {pop_total + 49.30:+.2f})")
    print(f"Objectives (recomputed): effort={F[0,0]:.4f} adj={F[0,1]:.4f} viol={F[0,2]:.4f}")
    print(f"Constraints: {[int(c) for c in G[0]]}")
    optimizer_side_pass = bool(acc.get("optimizer_side_pass", False)) and all(int(c) == 0 for c in G[0])
    print(f"Optimizer-side pass: {optimizer_side_pass}")
    print()

    print("--- Acceptance Checks ---")
    for check, ok in checks.items():
        if check in pending_checks:
            status = "?"
        else:
            status = "✓" if ok else "✗"
        print(f"  [{status}] {check}")
    print()

    if pending_checks:
        print("--- Pending External Validation ---")
        for check in sorted(pending_checks):
            print(f"  {check}: run the external export/apply validator and require bad_literal_count == 0.")
        print()

    if failed:
        print("--- Failed Check Guidance ---")
        guidance = acc.get("failure_guidance", [])
        guidance_map = {}
        if isinstance(guidance, dict):
            guidance_map = guidance
        elif isinstance(guidance, list):
            # Flat list of strings; print generically.
            guidance_map = {check: guidance for check in failed}
        for check in failed:
            print(f"  {check}:")
            lines = guidance_map.get(check, ["No specific guidance."])
            if not lines:
                lines = ["No specific guidance."]
            for line in lines:
                print(f"    - {line}")
        print()

    # Mouse/arrow/Norwegian quick status
    print("--- Key Subsystem Status ---")
    print(f"  Mouse layer present: {checks.get('dynamic_mouse_layer_present', False)}")
    print(f"  Scroll mode access: {checks.get('scroll_mode_access_present', False)}")
    print(f"  L7 access: {checks.get('layer7_momentary_and_toggle_access', False)}")
    print(f"  Raw arrows OK: {checks.get('mutable_raw_arrows_ok', False)}")
    print(f"  Norwegian cluster: {checks.get('norwegian_completion_cluster', False)}")
    print(f"  Thumb side clear: {checks.get('momentary_only_thumb_side_clear', False)}")
    print()

    mouse_warnings = mouse_layer_quality_warnings(layout, arrays, current_mouse)
    if mouse_warnings:
        print("--- Mouse Layer Quality Warnings ---")
        for warning in mouse_warnings:
            print(f"  ! {warning}")
        print()

    l0_rows = _l0_mutable_assignments(layout)
    if l0_rows:
        print("--- L0 Mutable Assignment Audit ---")
        print("  Prime L0 slots must be defended by shortcut usage or target-layer demand.")
        for row in l0_rows:
            target = f" target={row['target']} demand={row['target_demand']:.1f}" if row["target"] else ""
            print(
                f"  pos{row['idx']:3d} x={row['x']:.0f} y={row['y']:.0f} "
                f"effort={row['effort']:.2f} {row['hand']:<5} {row['kind']:<8} "
                f"usage={row['usage']:<5d} importance={row['importance']:.1f}{target} "
                f"{row['keys']}"
            )
        print()

    empty = _prime_empty_positions(layout.genome, layout, arrays)
    if empty:
        print("--- Top Empty Prime Positions ---")
        for eff, pos, lyr in empty:
            print(f"  L{lyr} pos{pos:3d} effort={eff:.2f}")
        print()

    print(f"Re-evaluated total score: {total:.4f} (gap {gap:+.2f})")


if __name__ == "__main__":
    main()
