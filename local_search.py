"""
Greedy local search: cluster same-app shortcuts to improve adjacency.

Strategy: for each app (sorted by importance), try pulling same-app shortcuts
to adjacent physical positions. Accept any improvement.

Run: .venv/bin/python3 local_search.py
"""
import sys, numpy as np, yaml, json, time
sys.path.insert(0, '.')

from core.loader import build_layout
from fitness.kernel import precompute
from fitness.evaluator import FitnessEvaluator

# ── Load config + data ──────────────────────────────────────────────────────
with open('config_v2.yaml') as f:
    cfg = yaml.safe_load(f)
with open('build/v2_evolution_results.json') as f:
    d = json.load(f)

fw  = cfg['fitness']['weights']
vw  = cfg['fitness']['violation_sub_weights']
hc  = cfg['fitness'].get('hard_constraints', [])
thr = cfg['fitness'].get('missing_important_threshold', 6.0)
tem = cfg['fitness'].get('toggle_effort_multiplier', 2.5)
sf  = np.array(d['scale_factors'], dtype=np.float32)

layout   = build_layout('data', fw)
arrays   = precompute(layout, fw, vw, thr, sf.astype(np.float64),
                      reference_genome=layout.genome, hard_constraints=hc,
                      toggle_effort_multiplier=tem)
evaluator = FitnessEvaluator(
    weights=fw, reference_layout=layout, scale_factors=sf,
    violation_weights=vw, missing_important_threshold=thr,
    hard_constraints=hc, toggle_effort_multiplier=tem,
)

dist                = arrays[6]
shortcut_importance = arrays[10]
shortcut_app        = arrays[11]
shortcut_category   = arrays[12]
pos_x               = arrays[8]
pos_y               = arrays[9]
pos_layer           = arrays[1]
pos_is_frozen       = arrays[5]
n_sc  = len(shortcut_importance)
n_pos = len(pos_layer)

FLOOR = -49.30  # theoretical floor for gap score

def total_score(objs):
    return float(objs[0] + objs[1] + objs[2])

def gap(objs):
    return total_score(objs) - FLOOR

def eval_genome(g):
    objs, _ = evaluator.evaluate_batch(g[np.newaxis])
    return objs[0]

# ── Load best genome ─────────────────────────────────────────────────────────
genome = np.array(d['best_exact']['genome'], dtype=np.int32)
base_objs = eval_genome(genome)
base_total = total_score(base_objs)
print(f"Starting genome: total={base_total:.4f}  gap=+{gap(base_objs):.2f}")
print(f"  e={base_objs[0]:.4f}  a={base_objs[1]:.4f}  v={base_objs[2]:.4f}")
print()

# ── Build helper maps ────────────────────────────────────────────────────────
def sid_to_pos(g):
    m = np.full(n_sc, -1, dtype=np.int32)
    for pos, sid in enumerate(g):
        if 0 <= sid < n_sc:
            m[sid] = pos
    return m

def mutable_positions():
    return [i for i in range(n_pos) if not pos_is_frozen[i]]

MUTABLE = mutable_positions()

# All same-app pairs sorted by importance product (highest first)
pairs_by_app = {}
for a in range(n_sc):
    if shortcut_importance[a] <= 0:
        continue
    app = int(shortcut_app[a])
    for b in range(a + 1, n_sc):
        if shortcut_importance[b] <= 0:
            continue
        if shortcut_app[b] != app and shortcut_category[a] != shortcut_category[b]:
            continue
        prod = float(shortcut_importance[a]) * float(shortcut_importance[b])
        pairs_by_app.setdefault(app, []).append((prod, a, b))

for app in pairs_by_app:
    pairs_by_app[app].sort(reverse=True)

# Physical-key-to-positions map: (x,y) -> list of mutable position indices
xy_to_pos = {}
for i in MUTABLE:
    key = (float(pos_x[i]), float(pos_y[i]))
    xy_to_pos.setdefault(key, []).append(i)

# ── Move: swap shortcut b to a position near shortcut a ──────────────────────
def try_cluster_pair(genome, sid_a, sid_b, current_total):
    """Try moving sid_b to be physically closer to sid_a."""
    pos_a = sid_to_pos(genome)[sid_a]
    pos_b = sid_to_pos(genome)[sid_b]
    if pos_a < 0 or pos_b < 0:
        return genome, current_total, False

    xa, ya = float(pos_x[pos_a]), float(pos_y[pos_a])

    # Find the closest mutable positions to (xa, ya) that aren't pos_a
    # and aren't frozen
    candidates = []
    for (xi, yi), positions in xy_to_pos.items():
        d_to_a = np.sqrt((xi - xa)**2 + (yi - ya)**2)
        for p in positions:
            if p != pos_a:
                candidates.append((d_to_a, p))

    candidates.sort()

    best_genome = genome
    best_total  = current_total
    improved    = False

    # Try up to top-10 nearest positions
    for _, target_pos in candidates[:10]:
        if pos_is_frozen[target_pos]:
            continue
        # Swap: put sid_b at target_pos, put whatever is at target_pos to pos_b
        g2 = genome.copy()
        sid_at_target = int(g2[target_pos])
        g2[target_pos] = sid_b
        g2[pos_b] = sid_at_target
        objs2 = eval_genome(g2)
        t2 = total_score(objs2)
        if t2 < best_total:
            best_genome = g2
            best_total  = t2
            improved    = True
            break  # greedy: accept first improvement

    return best_genome, best_total, improved

# ── Main search loop ─────────────────────────────────────────────────────────
current_genome = genome.copy()
current_objs   = base_objs
current_total  = base_total

n_moves = 0
t0 = time.perf_counter()

for iteration in range(20):
    improved_this_iter = False

    # Process apps sorted by importance (most important first)
    app_importance = {
        app: sum(prod for prod, _, _ in pairs)
        for app, pairs in pairs_by_app.items()
    }
    for app in sorted(pairs_by_app, key=lambda a: -app_importance[a]):
        for prod, sid_a, sid_b in pairs_by_app[app]:
            if prod < 1.0:
                break  # skip low-importance pairs
            current_genome, new_total, improved = try_cluster_pair(
                current_genome, sid_a, sid_b, current_total
            )
            if improved:
                current_total = new_total
                n_moves += 1
                improved_this_iter = True

    current_objs = eval_genome(current_genome)
    elapsed = time.perf_counter() - t0
    print(f"Iteration {iteration+1:2d}: total={current_total:.4f}  gap=+{gap(current_objs):.2f}"
          f"  moves={n_moves}  elapsed={elapsed:.1f}s")

    if not improved_this_iter:
        print("  → No more improvements, converged.")
        break

print()
print("=== FINAL RESULT ===")
final_objs  = eval_genome(current_genome)
final_total = total_score(final_objs)
print(f"  Start: total={base_total:.4f}  gap=+{gap(base_objs):.2f}")
print(f"  End:   total={final_total:.4f}  gap=+{gap(final_objs):.2f}")
print(f"  Improvement: {final_total - base_total:.4f} points")
print(f"  e={final_objs[0]:.4f}  a={final_objs[1]:.4f}  v={final_objs[2]:.4f}")
print(f"  Total moves: {n_moves}")

# Save result
out = dict(d)
out['best_exact']['genome'] = current_genome.tolist()
out['best_exact']['effort']    = float(final_objs[0])
out['best_exact']['adjacency'] = float(final_objs[1])
out['best_exact']['violations']= float(final_objs[2])
out['local_search_gap_before'] = gap(base_objs)
out['local_search_gap_after']  = gap(final_objs)
with open('build/v2_local_search_result.json', 'w') as f:
    json.dump(out, f)
print(f"\nSaved to build/v2_local_search_result.json")
