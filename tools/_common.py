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


def mouse_layer_quality_warnings(layout: object, arrays: Tuple, mouse_report: Optional[Dict] = None) -> List[str]:
    """Return human-facing warnings for accepted-but-low-quality mouse layers.

    Acceptance intentionally checks only final validity. This helper catches
    quality regressions inside a valid dynamic mouse layer, especially Scroll
    being placed worse than lower-priority mouse buttons.
    """
    if mouse_report is None:
        from evolution.acceptance import _dynamic_mouse_layer_report

        mouse_report = _dynamic_mouse_layer_report(layout)

    layer = mouse_report.get("mouse_layer")
    if layer is None:
        best = mouse_report.get("best_candidate") or {}
        layer = best.get("layer")
    if layer is None or int(layer) < 0:
        return ["No dynamic mouse layer candidate found for quality audit."]
    layer = int(layer)

    pos_effort = arrays[0]
    pos_layer = arrays[1]
    pos_hand = arrays[3]
    pos_is_thumb = arrays[4]
    pos_x = arrays[8]
    pos_y = arrays[9]
    sc_is_mouse = arrays[16]
    sc_mouse_btn = arrays[17]

    button_best: Dict[int, Tuple[float, int, float, float]] = {}
    scroll_best: Optional[Tuple[float, int, float, float]] = None
    for idx, sid in enumerate(layout.genome):
        sid = int(sid)
        if sid < 0 or sid >= len(layout.shortcuts) or int(pos_layer[idx]) != layer:
            continue
        if sc_is_mouse[sid] and sc_mouse_btn[sid] > 0 and pos_hand[idx] == 1 and not pos_is_thumb[idx]:
            button = int(sc_mouse_btn[sid])
            row = (float(pos_effort[idx]), idx, float(pos_x[idx]), float(pos_y[idx]))
            if button not in button_best or row[0] < button_best[button][0]:
                button_best[button] = row
        shortcut = layout.shortcuts[sid]
        is_scroll = bool(getattr(shortcut, "is_scroll_mode_access", False)) or "scroll" in shortcut.keys.lower()
        if (
            is_scroll
            and getattr(shortcut, "access_is_momentary", False)
            and pos_hand[idx] == 1
            and not pos_is_thumb[idx]
        ):
            row = (float(pos_effort[idx]), idx, float(pos_x[idx]), float(pos_y[idx]))
            if scroll_best is None or row[0] < scroll_best[0]:
                scroll_best = row

    warnings = []
    if scroll_best is None:
        warnings.append(f"L{layer}: no right-hand non-thumb momentary Scroll found for quality audit.")
        return warnings

    scroll_effort, scroll_idx, scroll_x, scroll_y = scroll_best
    for button in (3, 4):
        if button not in button_best:
            continue
        btn_effort, btn_idx, btn_x, btn_y = button_best[button]
        if scroll_effort > btn_effort + 1e-6:
            warnings.append(
                f"L{layer}: momentary Scroll pos{scroll_idx} "
                f"(x={scroll_x:.0f}, y={scroll_y:.0f}, effort={scroll_effort:.2f}) "
                f"is worse than MB{button} pos{btn_idx} "
                f"(x={btn_x:.0f}, y={btn_y:.0f}, effort={btn_effort:.2f})."
            )
    if abs(scroll_y - 2.0) > 1e-6:
        warnings.append(
            f"L{layer}: momentary Scroll pos{scroll_idx} is on y={scroll_y:.0f}; "
            "expected the preferred mouse row y=2 unless usage strongly proves otherwise."
        )
    if scroll_x in (7.0, 8.0):
        warnings.append(f"L{layer}: momentary Scroll pos{scroll_idx} is on uncomfortable x={scroll_x:.0f}.")
    return warnings


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
