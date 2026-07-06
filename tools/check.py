#!/usr/bin/env python3
"""Fast layout snapshot. Usage: python3 tools/check.py [checkpoint.json]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, yaml, numpy as np, glob
from collections import defaultdict, deque

class DC:
    def __init__(self, d): self._d = d
    def get(self, key, default=None):
        parts = key.split('.'); v = self._d
        for p in parts:
            if isinstance(v, dict) and p in v: v = v[p]
            else: return default
        return v

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

def bfs(edges):
    dist = {0: 0}; q = deque([0])
    while q:
        node = q.popleft()
        for to, *_ in edges[node]:
            if to not in dist:
                dist[to] = dist[node] + 1; q.append(to)
    return dist

def sc_label(s, sid, sc_is_mouse, sc_mouse_btn):
    if sid < 0: return '(empty)'
    if s.is_layer_access:
        return '@L{}:{}'.format(s.access_target_layer, 'hold' if s.access_is_momentary else 'tog')
    if sc_is_mouse[sid] and sc_mouse_btn[sid] > 0:
        return 'MB{}'.format(int(sc_mouse_btn[sid]))
    return s.keys

def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    if not ckpt:
        files = sorted(glob.glob('build/v2_checkpoint_gen*.json'), key=os.path.getmtime)
        ckpt = files[-1] if files else None
    if not ckpt:
        print('No checkpoint found'); sys.exit(1)

    ev, layout = load_ev()
    arrays = ev.model.arrays
    pos_layer = arrays[1]; pos_effort = arrays[0]; pos_hand = arrays[3]
    pos_is_thumb = arrays[4]; pos_frozen = arrays[5]
    pos_x = arrays[8]; pos_y = arrays[9]
    sc_is_mouse = arrays[16]; sc_mouse_btn = arrays[17]

    g = np.array(json.load(open(ckpt))['best_genome'], dtype=np.int32)
    F, G = ev.model.evaluate_batch(g.reshape(1, -1))
    gap = float(F[0].sum()) + 49.30
    adj = float(F[0, 1])

    # Layer BFS
    edges_all = defaultdict(list); edges_hold = defaultdict(list)
    for i, sid in enumerate(g):
        if sid < 0 or sid >= len(layout.shortcuts): continue
        s = layout.shortcuts[sid]
        if not s.is_layer_access: continue
        src = int(pos_layer[i]); tgt = s.access_target_layer
        edges_all[src].append((tgt,))
        if s.access_is_momentary: edges_hold[src].append((tgt,))
    da = bfs(edges_all); dh = bfs(edges_hold)

    # Mouse layer: count distinct button numbers per layer (not raw count, avoids dup inflation)
    mr = defaultdict(set)
    for i, sid in enumerate(g):
        if sid < 0 or sid >= len(layout.shortcuts): continue
        if sc_is_mouse[sid] and sc_mouse_btn[sid] > 0 and pos_hand[i] == 1 and not pos_is_thumb[i]:
            lyr = int(pos_layer[i])
            if 0 < lyr < 32 and lyr != 7: mr[lyr].add(int(sc_mouse_btn[sid]))
    ml = max(mr, key=lambda l: len(mr[l])) if mr and max(len(v) for v in mr.values()) >= 2 else -1

    gen = int(ckpt.split('gen')[1].split('.')[0]) if 'gen' in ckpt else 0
    print(f'=== {os.path.basename(ckpt)} ===')
    print(f'gap={gap:+.3f}  adj={adj:.3f}  G={[int(G[0,i]) for i in range(G.shape[1])]}')
    print(f'Mouse L{ml}: all-hops={da.get(ml,"?")} hold-hops={dh.get(ml,"NONE")}')
    print()

    # L0 thumb cluster
    print('L0 thumb cluster:')
    for i, sid in enumerate(g):
        if int(pos_layer[i]) != 0 or pos_frozen[i]: continue
        hand = 'R' if pos_hand[i] == 1 else 'L'; th = '+T' if pos_is_thumb[i] else ''
        s = layout.shortcuts[sid] if 0 <= sid < len(layout.shortcuts) else None
        lbl = sc_label(s, sid, sc_is_mouse, sc_mouse_btn) if s else '(empty)'
        mark = ' ★MOUSE' if (s and s.is_layer_access and s.access_target_layer == ml and s.access_is_momentary) else ''
        print(f'  pos{i:3d} {hand}{th} eff={pos_effort[i]:.2f}  {lbl}{mark}')
    print()

    # Top 10 easiest positions and what's on them
    print('Top 10 easiest mutable positions:')
    mutable = [(float(pos_effort[i]), i, int(g[i])) for i in range(len(g))
               if not pos_frozen[i] and int(pos_layer[i]) != 7]
    for eff, i, sid in sorted(mutable)[:10]:
        lyr = int(pos_layer[i])
        hand = 'R' if pos_hand[i] == 1 else 'L'; th = 'T' if pos_is_thumb[i] else 'F'
        s = layout.shortcuts[sid] if 0 <= sid < len(layout.shortcuts) else None
        lbl = sc_label(s, sid, sc_is_mouse, sc_mouse_btn) if s else '(empty)'
        print(f'  eff={eff:.2f} L{lyr} pos{i:3d} {hand}{th}  {lbl}')
    print()

    # Layer reachability summary
    print('Layer hops (all / hold):')
    for lyr in sorted(set(int(pos_layer[i]) for i in range(len(g)))):
        if lyr == 7: continue
        ah = da.get(lyr, 999); hh = dh.get(lyr, 999)
        mark = ' ★' if lyr == ml else ''
        print(f'  L{lyr}: {ah}/{hh if hh<999 else "none"}{mark}')

if __name__ == '__main__':
    main()
