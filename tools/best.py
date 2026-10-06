#!/usr/bin/env python3
"""Find the best checkpoint by raw score. Usage: python3 tools/best.py"""
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
        total = float(F[0].sum())
        gen = int(os.path.basename(f).split("gen")[1].split(".")[0])
        results.append((total, gen, f))
        print(f'  gen{gen:6d}: raw_score={total:.3f}  G={[int(G[0, i]) for i in range(G.shape[1])]}')

    results.sort(key=lambda x: x[0])
    best_score, best_gen, best_f = results[0]
    print(f'\nBEST: {os.path.basename(best_f)}  raw_score={best_score:.3f}')
    print(best_f)  # last line = path, easy to capture


if __name__ == '__main__':
    main()
