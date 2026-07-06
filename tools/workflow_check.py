#!/usr/bin/env python3
"""Workflow clustering check: VSCode shortcuts, Norwegian chars.
Replaces the old Windows-path-hardcoded workflow_analysis.py for v2.
Usage: python3 tools/workflow_check.py [checkpoint.json]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, yaml, numpy as np, glob
from collections import defaultdict

class DC:
    def __init__(self, d): self._d = d
    def get(self, key, default=None):
        parts = key.split('.'); v = self._d
        for p in parts:
            if isinstance(v, dict) and p in v: v = v[p]
            else: return default
        return v

VSCODE_KEYS = ['Ctrl+Shift+P', 'Ctrl+P', 'F5', 'F12', 'Ctrl+K Ctrl+S', 'Ctrl+`',
               'Ctrl+Shift+`', 'Shift+F5', 'Ctrl+Shift+F6', 'Ctrl+,', 'Ctrl+B', 'Ctrl+Shift+E']
NORWEGIAN_KEYS = ['Å', 'Ø', 'Æ', 'å', 'ø', 'æ']

def load_ev():
    from core.loader import build_layout
    from fitness.evaluator import FitnessEvaluator
    cfg = DC(yaml.safe_load(open('config_v2.yaml')))
    layout = build_layout('data', cfg.get('fitness', {}))
    sf = np.array(json.load(open('build/v2_scale_factors.json'))['scale_factors'], dtype=np.float32)
    ev = FitnessEvaluator(
        weights=cfg.get('fitness.weights', {}), reference_layout=layout, scale_factors=sf,
        violation_weights=cfg.get('fitness.violation_sub_weights', {}),
        missing_important_threshold=cfg.get('fitness.missing_important_threshold', 6.0),
        hard_constraints=cfg.get('fitness.hard_constraints', []),
        toggle_effort_multiplier=float(cfg.get('fitness.toggle_effort_multiplier', 2.5)),
    )
    return ev, layout

def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    if not ckpt:
        files = sorted(glob.glob('build/v2_checkpoint_gen*.json'), key=os.path.getmtime)
        ckpt = files[-1] if files else None
    if not ckpt:
        print('No checkpoint'); sys.exit(1)

    ev, layout = load_ev()
    arrays = ev.model.arrays
    pos_layer = arrays[1]; pos_effort = arrays[0]

    g = np.array(json.load(open(ckpt))['best_genome'], dtype=np.int32)

    key_positions = defaultdict(list)
    for i, sid in enumerate(g):
        if sid < 0 or sid >= len(layout.shortcuts): continue
        s = layout.shortcuts[sid]
        if not s.is_layer_access:
            key_positions[s.keys].append((i, int(pos_layer[i]), float(pos_effort[i])))

    print(f'=== WORKFLOW CHECK: {os.path.basename(ckpt)} ===')
    print('\n--- VSCODE CLUSTERING ---')
    layers_seen = defaultdict(int)
    for k in VSCODE_KEYS:
        positions = key_positions.get(k, [])
        if not positions:
            print(f'  [MISS] {k}: not in layout')
            continue
        best = min(positions, key=lambda x: x[2])
        pos, lyr, eff = best
        layers_seen[lyr] += 1
        print(f'  {k:16s}: L{lyr} pos{pos:3d} eff={eff:.2f}')
    if layers_seen:
        dominant = max(layers_seen.values())
        total = sum(layers_seen.values())
        spread = len(layers_seen)
        tag = 'GOOD' if spread <= 2 else ('WARN' if spread <= 3 else 'BAD')
        print(f'  [{tag}] Spread across {spread} layer(s): {dict(layers_seen)} ({dominant}/{total} on top layer)')

    print('\n--- NORWEGIAN CHARS ---')
    for k in NORWEGIAN_KEYS:
        positions = key_positions.get(k, [])
        if not positions:
            print(f'  [MISS] {k}: not in layout')
            continue
        best = min(positions, key=lambda x: x[2])
        pos, lyr, eff = best
        tag = 'GOOD' if eff <= 1.0 else ('WARN' if eff <= 2.0 else 'BAD')
        print(f'  [{tag}] {k}: L{lyr} pos{pos:3d} eff={eff:.2f}')

if __name__ == '__main__':
    main()
