#!/usr/bin/env python3
"""Find the best checkpoint by gap score. Usage: python3 tools/best.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tools._common import find_checkpoints, load_checkpoint, load_evaluator


def main():
    ev = load_evaluator()
    files = find_checkpoints()
    if not files:
        print("No checkpoints found", file=sys.stderr)
        sys.exit(1)

    results = []
    for f in files:
        ckpt = load_checkpoint(f)
        g = np.array(ckpt["best_genome"], dtype=np.int32)
        F, G = ev.model.evaluate_batch(g.reshape(1, -1))
        gap = float(F[0].sum()) + 49.30
        gen = int(os.path.basename(f).split("gen")[1].split(".")[0])
        results.append((gap, gen, f))
        print(f'  gen{gen:6d}: gap={gap:+.3f}  G={[int(G[0, i]) for i in range(G.shape[1])]}')

    results.sort(key=lambda x: x[0])
    best_gap, best_gen, best_f = results[0]
    print(f'\nBEST: {os.path.basename(best_f)}  gap={best_gap:+.3f}')
    print(best_f)  # last line = path, easy to capture


if __name__ == '__main__':
    main()
