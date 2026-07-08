"""Shared helpers for Charybdis analysis tools.

Eliminates the duplicated loader/evaluator/checkpoint setup that every
script in tools/ used to repeat. All functions work inside the repo venv
and require no new dependencies.
"""
import glob
import json
import os
import sys
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure repo root is importable when scripts are run directly.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.loader import build_layout
from fitness.evaluator import FitnessEvaluator


def load_layout(data_dir: str = "data") -> object:
    """Load the base Layout object from data_dir."""
    return build_layout(data_dir, config=None)


def load_evaluator(
    config_path: str = "config_v2.yaml",
    data_dir: str = "data",
    build_dir: str = "build",
    require_cuda: bool = False,
) -> FitnessEvaluator:
    """Load a FitnessEvaluator using production config and cached scale factors."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    fitness_cfg = cfg.get("fitness", {})

    layout = build_layout(data_dir, config=fitness_cfg)

    sf_path = os.path.join(build_dir, "v2_scale_factors.json")
    if os.path.exists(sf_path):
        with open(sf_path, "r", encoding="utf-8") as f:
            sf = np.array(json.load(f)["scale_factors"], dtype=np.float32)
    else:
        sf = np.ones(3, dtype=np.float32)

    return FitnessEvaluator(
        weights=fitness_cfg.get("weights", {}),
        reference_layout=layout,
        scale_factors=sf,
        violation_weights=fitness_cfg.get("violation_sub_weights", {}),
        missing_important_threshold=fitness_cfg.get("missing_important_threshold", 6.0),
        hard_constraints=fitness_cfg.get("hard_constraints", []),
        toggle_effort_multiplier=float(fitness_cfg.get("toggle_effort_multiplier", 2.5)),
        require_cuda=require_cuda,
    )


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_to_layout(checkpoint: dict, layout: object) -> object:
    """Return a Layout clone with the checkpoint's best genome."""
    return layout.clone_with(genome=np.asarray(checkpoint["best_genome"], dtype=np.int32))


def _checkpoint_generation(path: str) -> int:
    base = os.path.basename(path)
    try:
        return int(base.split("gen")[-1].split(".")[0])
    except ValueError:
        return -1


def _checkpoint_sort_key(path: str) -> Tuple[int, str]:
    return (_checkpoint_generation(path), path)


def _latest_run_dir(build_dir: str) -> Optional[str]:
    run_root = os.path.join(build_dir, "runs")
    if not os.path.isdir(run_root):
        return None
    candidates = []
    for path in glob.glob(os.path.join(run_root, "*")):
        if not os.path.isdir(path):
            continue
        if glob.glob(os.path.join(path, "v2_checkpoint_gen*.json")):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: (os.path.getmtime(p), p))


def find_checkpoints(build_dir: str = "build") -> List[str]:
    """Return v2 checkpoint paths sorted by generation.

    If ``build_dir`` is the top-level ``build`` directory, prefer the newest
    nested production run under ``build/runs/<run_id>/``. This keeps reports on
    the active run instead of mixing unrelated historical generations.
    """
    run_dir = _latest_run_dir(build_dir)
    search_dir = run_dir if run_dir is not None else build_dir
    paths = [os.path.normpath(p) for p in glob.glob(os.path.join(search_dir, "v2_checkpoint_gen*.json"))]
    return sorted(paths, key=_checkpoint_sort_key)


def find_latest_checkpoint(build_dir: str = "build") -> Optional[str]:
    """Return the most recent checkpoint path, or None if none exist."""
    paths = find_checkpoints(build_dir=build_dir)
    return paths[-1] if paths else None


def _bfs(edges: Dict[int, List[int]], start: int = 0) -> Dict[int, int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        node = q.popleft()
        for to in edges.get(node, []):
            if to not in dist:
                dist[to] = dist[node] + 1
                q.append(to)
    return dist


def compute_reachability(
    genome: np.ndarray,
    layout: object,
    arrays: Optional[Tuple] = None,
) -> Dict[str, Dict[int, int]]:
    """Compute layer reachability from L0 via all, momentary-hold, and toggle edges.

    Args:
        genome: int32 genome array.
        layout: Layout object (provides shortcuts and layer metadata).
        arrays: Optional precomputed arrays tuple (e.g., evaluator.model.arrays).
                If omitted, ``layout.position_layer`` is tried as a fallback.

    Returns a dict with keys:
      - all_hops: shortest hops using any access edge.
      - hold_hops: shortest hops using only momentary (hold) access edges.
      - toggle_hops: shortest hops using only toggle access edges.
      - reachable_layers: set of layers reachable by any edge.
    """
    edges_all: Dict[int, List[int]] = defaultdict(list)
    edges_hold: Dict[int, List[int]] = defaultdict(list)
    edges_toggle: Dict[int, List[int]] = defaultdict(list)

    if arrays is not None:
        pos_layer = arrays[1]
    elif hasattr(layout, "position_layer"):
        pos_layer = layout.position_layer
    else:
        raise ValueError(
            "compute_reachability needs layer metadata. Pass arrays=evaluator.model.arrays "
            "or a Layout with position_layer."
        )

    for i, sid in enumerate(genome):
        if sid < 0 or sid >= len(layout.shortcuts):
            continue
        s = layout.shortcuts[sid]
        if not s.is_layer_access:
            continue
        src = int(pos_layer[i])
        tgt = s.access_target_layer
        edges_all[src].append(tgt)
        if s.access_is_momentary:
            edges_hold[src].append(tgt)
        else:
            edges_toggle[src].append(tgt)

    all_hops = _bfs(edges_all)
    hold_hops = _bfs(edges_hold)
    toggle_hops = _bfs(edges_toggle)
    reachable_layers = set(all_hops.keys())

    return {
        "all_hops": all_hops,
        "hold_hops": hold_hops,
        "toggle_hops": toggle_hops,
        "reachable_layers": reachable_layers,
    }


def label_shortcut(sid: int, layout: object, arrays: Optional[Tuple] = None) -> Dict[str, object]:
    """Return a human-readable label dict for shortcut sid."""
    if sid < 0 or sid >= len(layout.shortcuts):
        return {"name": "(empty)", "app": None, "category": None, "importance": 0.0, "usage": 0.0}
    sc = layout.shortcuts[sid]
    return {
        "name": sc.name,
        "app": getattr(sc, "app", None),
        "category": getattr(sc, "category", None),
        "importance": float(getattr(sc, "importance", 0.0)),
        "usage": float(getattr(sc, "usage", 0.0)),
    }


def layer_inventory(layout: object) -> List[Dict[str, object]]:
    """Return a list of dicts, one per layer, with basic metadata."""
    inv = []
    for layer_idx in range(layout.n_layers):
        layer = layout.layers[layer_idx]
        inv.append({
            "index": layer_idx,
            "name": getattr(layer, "name", f"L{layer_idx}"),
            "n_positions": len(layer.positions),
        })
    return inv


def resolve_checkpoint_path(path_or_keyword: str, build_dir: str = "build") -> str:
    """Resolve 'latest' or a concrete checkpoint path."""
    if path_or_keyword.lower() in ("latest", "last"):
        latest = find_latest_checkpoint(build_dir)
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found in {build_dir}")
        return latest
    if os.path.exists(path_or_keyword):
        return path_or_keyword
    raise FileNotFoundError(f"Checkpoint not found: {path_or_keyword}")
