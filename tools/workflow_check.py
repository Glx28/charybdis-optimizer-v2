#!/usr/bin/env python3
"""Workflow clustering check: VSCode shortcuts, Norwegian chars.
Replaces the old Windows-path-hardcoded workflow_analysis.py for v2.
Usage: python3 tools/workflow_check.py [checkpoint.json]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict

import numpy as np

from tools._common import (
    find_latest_checkpoint,
    load_checkpoint,
    load_evaluator,
)

VSCODE_KEYS = ['Ctrl+Shift+P', 'Ctrl+P', 'F5', 'F12', 'Ctrl+K Ctrl+S', 'Ctrl+`',
               'Ctrl+Shift+`', 'Shift+F5', 'Ctrl+Shift+F6', 'Ctrl+,', 'Ctrl+B', 'Ctrl+Shift+E']
NORWEGIAN_KEYS = ['å', 'ø', 'æ']  # frozen L0 raw keys; uppercase is shift+key, not a distinct binding

def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint()
    if not ckpt_path:
        print('No checkpoint'); sys.exit(1)

    ev = load_evaluator()
    layout = ev.model.layout
    arrays = ev.model.arrays
    pos_layer = arrays[1]; pos_effort = arrays[0]

    ckpt = load_checkpoint(ckpt_path)
    g = np.array(ckpt['best_genome'], dtype=np.int32)

    key_positions = defaultdict(list)
    base_key_positions = defaultdict(list)
    for i, sid in enumerate(g):
        if sid < 0 or sid >= len(layout.shortcuts): continue
        s = layout.shortcuts[sid]
        if not s.is_layer_access:
            key_positions[s.keys].append((i, int(pos_layer[i]), float(pos_effort[i])))
            if s.base_key:
                base_key_positions[s.base_key].append((i, int(pos_layer[i]), float(pos_effort[i])))

    print(f'=== WORKFLOW CHECK: {os.path.basename(ckpt_path)} ===')
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
    print('  (frozen L0 raw-completion keys; matched by base_key, not by shortcut label)')
    for k in NORWEGIAN_KEYS:
        positions = base_key_positions.get(k, [])
        if not positions:
            print(f'  [MISS] {k}: not in layout')
            continue
        best = min(positions, key=lambda x: x[2])
        pos, lyr, eff = best
        tag = 'GOOD' if eff <= 1.0 else ('WARN' if eff <= 2.0 else 'BAD')
        print(f'  [{tag}] {k}: L{lyr} pos{pos:3d} eff={eff:.2f}')

if __name__ == '__main__':
    main()
