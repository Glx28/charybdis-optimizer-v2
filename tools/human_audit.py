#!/usr/bin/env python3
"""Human usability audit. Usage: python3 tools/human_audit.py [checkpoint.json]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict, deque

import numpy as np

from tools._common import (
    find_latest_checkpoint,
    load_checkpoint,
    load_evaluator,
)

ESSENTIAL = {
    'Ctrl+C': 'Copy',      'Ctrl+V': 'Paste',    'Ctrl+Z': 'Undo',
    'Ctrl+X': 'Cut',       'Ctrl+S': 'Save',      'Ctrl+F': 'Find',
    'Ctrl+A': 'Select all','Ctrl+Y': 'Redo',      'Alt+Tab': 'Switch app',
    'Ctrl+T': 'New tab',   'Ctrl+W': 'Close tab', 'Ctrl+N': 'New',
}
NAV_CLUSTER = ['Ctrl+Left','Ctrl+Right','Page Up','Page Down','Ctrl+Home','Ctrl+End']

def bfs_all(edges):
    dist = {0: 0}; q = deque([0])
    while q:
        node = q.popleft()
        for to in edges[node]:
            if to not in dist: dist[to] = dist[node] + 1; q.append(to)
    return dist

def tag(condition, good='GOOD', warn='WARN', bad='BAD'):
    return f'[{good}]' if condition is True else (f'[{bad}]' if condition is False else f'[{warn}]')

def tag3(good_cond, warn_cond, good='GOOD', warn='WARN', bad='BAD'):
    return f'[{good}]' if good_cond else (f'[{warn}]' if warn_cond else f'[{bad}]')

def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint()
    if not ckpt_path:
        print('No checkpoint'); sys.exit(1)

    ev = load_evaluator()
    layout = ev.model.layout
    arrays = ev.model.arrays
    pos_layer = arrays[1]; pos_effort = arrays[0]; pos_hand = arrays[3]
    pos_is_thumb = arrays[4]; pos_frozen = arrays[5]
    sc_is_mouse = arrays[16]; sc_mouse_btn = arrays[17]

    ckpt = load_checkpoint(ckpt_path)
    g = np.array(ckpt['best_genome'], dtype=np.int32)
    F, G = ev.model.evaluate_batch(g.reshape(1, -1))
    gap = float(F[0].sum()) + 49.30

    # Build lookup: keys -> list of (pos, layer, effort, hand, is_thumb)
    key_positions = defaultdict(list)
    for i, sid in enumerate(g):
        if sid < 0 or sid >= len(layout.shortcuts): continue
        s = layout.shortcuts[sid]
        if not s.is_layer_access:
            key_positions[s.keys].append((i, int(pos_layer[i]), float(pos_effort[i]),
                                          int(pos_hand[i]), bool(pos_is_thumb[i])))

    # Layer edges
    edges_all = defaultdict(set); edges_hold = defaultdict(set)
    for i, sid in enumerate(g):
        if sid < 0 or sid >= len(layout.shortcuts): continue
        s = layout.shortcuts[sid]
        if not s.is_layer_access: continue
        src = int(pos_layer[i]); tgt = s.access_target_layer
        edges_all[src].add(tgt)
        if s.access_is_momentary: edges_hold[src].add(tgt)
    da = bfs_all(edges_all); dh = bfs_all(edges_hold)

    # Mouse layer: pick the layer with the most *distinct* R-hand, non-thumb
    # mouse buttons (not raw count) so duplicate-button layers can't win a tie
    # over the true, complete 5-button layer.
    mr = defaultdict(set)
    for i, sid in enumerate(g):
        if sid < 0 or sid >= len(layout.shortcuts): continue
        if sc_is_mouse[sid] and sc_mouse_btn[sid] > 0 and pos_hand[i] == 1 and not pos_is_thumb[i]:
            lyr = int(pos_layer[i])
            if 0 < lyr < 32 and lyr != 7: mr[lyr].add(int(sc_mouse_btn[sid]))
    ml = max(mr, key=lambda lyr: len(mr[lyr])) if mr and max(len(v) for v in mr.values()) >= 2 else -1

    issues = []
    print(f'=== HUMAN AUDIT: {os.path.basename(ckpt_path)} ===')
    print(f'gap={gap:+.3f}  G={[int(G[0,i]) for i in range(G.shape[1])]}')
    print()

    # 1. Mouse layer access
    print('--- 1. MOUSE LAYER ---')
    ml_hold_hops = dh.get(ml, 999)
    rt_holds = [(i, int(pos_layer[i])) for i, sid in enumerate(g)
                if sid >= 0 and sid < len(layout.shortcuts)
                and layout.shortcuts[sid].is_layer_access
                and layout.shortcuts[sid].access_is_momentary
                and layout.shortcuts[sid].access_target_layer == ml
                and pos_hand[i] == 1 and pos_is_thumb[i]]
    mbs = [(int(sc_mouse_btn[sid]), 'R' if pos_hand[i]==1 else 'L', bool(pos_is_thumb[i]))
           for i, sid in enumerate(g)
           if sid >= 0 and sid < len(layout.shortcuts)
           and sc_is_mouse[sid] and sc_mouse_btn[sid] > 0 and int(pos_layer[i]) == ml]
    mb_ok = all(h == 'R' and not t for _, h, t in mbs)

    print(f'  {tag(ml_hold_hops == 1)} Hold-hops to L{ml}: {ml_hold_hops} (want 1)')
    print(f'  {tag(not rt_holds)} Right-thumb holds to mouse: {rt_holds or "none"}')
    print(f'  {tag(mb_ok)} Mouse buttons all R-finger: {[(b,h) for b,h,t in sorted(mbs)]}')
    if ml_hold_hops != 1: issues.append('Mouse layer needs hold-hop=1')
    if rt_holds: issues.append('Right-thumb hold to mouse layer exists')
    if not mb_ok: issues.append('Mouse buttons on wrong hand/thumb')

    # L0 direct mouse hold
    l0_mouse_holds = [(i, float(pos_effort[i]), 'R' if pos_hand[i]==1 else 'L', bool(pos_is_thumb[i]))
                      for i, sid in enumerate(g)
                      if sid >= 0 and sid < len(layout.shortcuts)
                      and layout.shortcuts[sid].is_layer_access
                      and layout.shortcuts[sid].access_is_momentary
                      and layout.shortcuts[sid].access_target_layer == ml
                      and int(pos_layer[i]) == 0]
    holds_str = [(p, e, h, "T" if t else "F") for p, e, h, t in l0_mouse_holds]
    print(f'  {tag(bool(l0_mouse_holds))} L0 direct @L{ml}:hold: {holds_str}')
    print()

    # 2. Essential shortcuts placement
    print('--- 2. ESSENTIAL SHORTCUTS ---')
    for keys, name in ESSENTIAL.items():
        positions = key_positions.get(keys, [])
        if not positions:
            print(f'  [MISS] {keys} ({name}): NOT IN LAYOUT')
            issues.append(f'{keys} missing from layout')
            continue
        best = min(positions, key=lambda x: x[2])
        pos, lyr, eff, hand, thumb = best
        eff_tag = tag3(eff <= 0.5, eff <= 1.0)
        dups = f' ({len(positions)}x)' if len(positions) > 1 else ''
        print(f'  {eff_tag} {keys:20s} ({name:12s}): L{lyr} pos{pos:3d} eff={eff:.2f}{dups}')
        if eff > 1.0:
            issues.append(f'{keys} at high effort {eff:.2f} — should be closer to home row')
    print()

    # 3. Navigation cluster coherence
    print('--- 3. NAVIGATION CLUSTER ---')
    nav_layers = defaultdict(list)
    for k in NAV_CLUSTER:
        for pos, lyr, eff, hand, thumb in key_positions.get(k, []):
            nav_layers[lyr].append((k, eff, pos))
    if nav_layers:
        dominant = max(nav_layers, key=lambda lyr: len(nav_layers[lyr]))
        coverage = len(nav_layers[dominant])
        covered = [k for k, e, p in nav_layers[dominant]]
        missing_nav = [k for k in NAV_CLUSTER if k not in covered]
        status = tag3(coverage >= 4, coverage >= 3)
        print(f'  {status} Nav cluster: L{dominant} has {coverage}/{len(NAV_CLUSTER)} nav keys')
        if missing_nav: print(f'    Missing from L{dominant}: {missing_nav}')
        if coverage < 3: issues.append(f'Nav cluster only {coverage}/{len(NAV_CLUSTER)} on L{dominant}')
    else:
        print('  [MISS] No nav keys found')
    print()

    # 4. Duplicate shortcut waste (same keys at similar effort on different layers)
    print('--- 4. DUPLICATE WASTE ---')
    wasted = []
    for keys, positions in key_positions.items():
        if len(positions) < 2: continue
        effs = sorted(set(round(e, 1) for _, _, e, _, _ in positions))
        lyrs = sorted(set(lyr for _, lyr, _, _, _ in positions))
        if len(lyrs) > 1 and min(effs) <= 1.0:
            wasted.append((keys, lyrs, effs))
    if wasted:
        wasted.sort(key=lambda x: len(x[1]), reverse=True)
        print(f'  [WARN] {len(wasted)} shortcuts appear on multiple layers:')
        for keys, lyrs, effs in wasted[:8]:
            print(f'    {keys:25s} on L{lyrs} effs={effs}')
    else:
        print('  [GOOD] No significant duplicate waste')
    print()

    # 5. Layer depth — no shortcut layer deeper than 2 hops
    print('--- 5. LAYER DEPTH ---')
    deep = [(lyr, da[lyr], dh.get(lyr, 999)) for lyr in da if lyr not in (0, 7) and da[lyr] > 2]
    print(f'  {tag(not deep)} Layers deeper than 2 hops: {deep or "none"}')
    if deep: issues.append(f'Layers at depth >2: {[lyr for lyr, _, _ in deep]}')
    print()

    # 6. Thumb cluster efficiency — are thumbs used for layer switches or wasted?
    print('--- 6. THUMB CLUSTER QUALITY ---')
    thumb_positions = [(i, int(pos_layer[i]), float(pos_effort[i]), int(g[i]))
                       for i in range(len(g))
                       if pos_is_thumb[i] and not pos_frozen[i] and int(pos_layer[i]) != 7]
    thumb_layer_switches = sum(1 for _, _, _, sid in thumb_positions
                               if 0 <= sid < len(layout.shortcuts) and layout.shortcuts[sid].is_layer_access)
    thumb_total = len(thumb_positions)
    thumb_empty = sum(1 for _, _, _, sid in thumb_positions if sid < 0)
    # Not a hard requirement: thumbs cost 0.45x normal effort, so the optimizer
    # legitimately trades some layer-switch duty for high-value shortcuts there.
    # No fitness weight targets this ratio directly, so treat low values as WARN.
    thumb_switch_ratio = thumb_layer_switches/max(thumb_total,1)
    thumb_status = tag3(thumb_switch_ratio > 0.5, thumb_switch_ratio > 0.3)
    print(f'  {thumb_status} Thumb cluster: {thumb_layer_switches}/{thumb_total} are layer switches, '
          f'{thumb_empty} empty')

    # High-effort thumb switches (bad)
    bad_thumb = [(i, lyr, eff) for i, lyr, eff, sid in thumb_positions
                 if eff >= 1.5 and 0 <= sid < len(layout.shortcuts) and layout.shortcuts[sid].is_layer_access]
    if bad_thumb:
        print(f'  [WARN] High-effort (≥1.5) thumb layer switches: {len(bad_thumb)} '
              f'(acceptable for rare layers)')
    print()

    # 7. Return toggles
    print('--- 7. RETURN TOGGLES ---')
    toggle_layers = set()
    for i, sid in enumerate(g):
        if sid < 0 or sid >= len(layout.shortcuts): continue
        s = layout.shortcuts[sid]
        if (s.is_layer_access and not s.access_is_momentary and s.access_target_layer != 0
                and s.access_target_layer != int(pos_layer[i])):
            toggle_layers.add(s.access_target_layer)
    missing_returns = []
    for lyr in sorted(toggle_layers):
        if lyr == 7: continue  # frozen, exempt
        has_ret = any(int(pos_layer[i]) == lyr
                      and layout.shortcuts[int(g[i])].is_layer_access
                      and layout.shortcuts[int(g[i])].access_target_layer == 0
                      and not layout.shortcuts[int(g[i])].access_is_momentary
                      for i in range(len(g))
                      if int(g[i]) >= 0 and int(g[i]) < len(layout.shortcuts) and int(pos_layer[i]) == lyr)
        if not has_ret:
            missing_returns.append(lyr)
    print(f'  {tag(not missing_returns)} Toggle layers missing @L0:return: {missing_returns or "none"}')
    if missing_returns: issues.append(f'Toggle layers missing return: {missing_returns}')
    print()

    # Summary
    print('=== SUMMARY ===')
    print(f'gap={gap:+.3f}  issues={len(issues)}')
    for iss in issues:
        print(f'  ✗ {iss}')
    if not issues:
        print('  ✓ All checks passed')

if __name__ == '__main__':
    main()
