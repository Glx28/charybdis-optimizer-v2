#!/usr/bin/env python3
"""Find the best checkpoint by gap score. Usage: python3 tools/best.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, yaml, numpy as np, glob

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
        weights=cfg.get('fitness.weights', {}),
        reference_layout=layout,
        scale_factors=sf,
        violation_weights=cfg.get('fitness.violation_sub_weights', {}),
        missing_important_threshold=cfg.get('fitness.missing_important_threshold', 6.0),
        hard_constraints=cfg.get('fitness.hard_constraints', []),
        toggle_effort_multiplier=float(cfg.get('fitness.toggle_effort_multiplier', 2.5)),
    )
    return ev, layout

def main():
    ev, layout = load_ev()
    files = sorted(glob.glob('build/v2_checkpoint_gen*.json'),
                   key=lambda f: int(f.split('gen')[1].split('.')[0]))
    results = []
    for f in files:
        g = np.array(json.load(open(f))['best_genome'], dtype=np.int32)
        F, G = ev.model.evaluate_batch(g.reshape(1, -1))
        gap = float(F[0].sum()) + 49.30
        gen = int(f.split('gen')[1].split('.')[0])
        results.append((gap, gen, f))
        print(f'  gen{gen:6d}: gap={gap:+.3f}  G={[int(G[0,i]) for i in range(G.shape[1])]}')

    results.sort(key=lambda x: x[0])
    best_gap, best_gen, best_f = results[0]
    print(f'\nBEST: {os.path.basename(best_f)}  gap={best_gap:+.3f}')
    print(best_f)  # last line = path, easy to capture

if __name__ == '__main__':
    main()
