"""Tests for core/semantic_clusters.py"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.semantic_clusters import detect_semantic_clusters
from core import Shortcut


def _sc(sid, keys, action, app="test", importance=5.0, category="general", modifiers=(), base_key=""):
    return Shortcut(
        sid=sid,
        keys=keys,
        action=action,
        app=app,
        importance=importance,
        category=category,
        modifiers=modifiers,
        base_key=base_key,
    )


def test_detects_copy_paste():
    shortcuts = [
        _sc(0, "Ctrl+C", "Copy", modifiers=("ctrl",), base_key="C"),
        _sc(1, "Ctrl+V", "Paste", modifiers=("ctrl",), base_key="V"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    assert len(clusters) == 1
    c = clusters[0]
    assert c.name == "sequence_clipboard"
    assert c.is_critical
    assert [m.sid for m in c.members] == [0, 1]
    assert [m.order for m in c.members] == [0, 1]


def test_detects_undo_redo():
    shortcuts = [
        _sc(0, "Ctrl+Z", "Undo", modifiers=("ctrl",), base_key="Z"),
        _sc(1, "Ctrl+Shift+Z", "Redo", modifiers=("ctrl", "shift"), base_key="Z"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    assert len(clusters) == 1
    c = clusters[0]
    assert c.name == "sequence_history"
    assert [m.sid for m in c.members] == [0, 1]


def test_detects_directional_action_stem():
    shortcuts = [
        _sc(0, "Win+Left", "Snap window left", modifiers=("win",), base_key="Left"),
        _sc(1, "Win+Right", "Snap window right", modifiers=("win",), base_key="Right"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    assert len(clusters) >= 1
    # The window-state keyword family now claims snap actions before the generic
    # directional stem detector.
    names = {c.name for c in clusters}
    assert "family_window_state" in names
    sids = {m.sid for c in clusters for m in c.members}
    assert sids == {0, 1}


def test_ignores_mouse_buttons():
    shortcuts = [
        _sc(0, "MB1", "Left click", category="mouse"),
        _sc(1, "MB2", "Right click", category="mouse"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    assert len(clusters) == 0


def test_clusters_raw_arrows():
    shortcuts = [
        _sc(0, "LeftArrow", "Left arrow", base_key="LeftArrow"),
        _sc(1, "RightArrow", "Right arrow", base_key="RightArrow"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    assert len(clusters) == 1
    # Directional-stem detection wins over raw-key detection for labelled arrows.
    assert clusters[0].name == "direction_arrow"


def test_keeps_best_importance_per_order_for_direction_stems():
    # These actions do not match any keyword family, so they fall through to the
    # directional stem detector which keeps the highest-importance shortcut per
    # distinct order.
    shortcuts = [
        _sc(0, "Ctrl+Shift+G", "My previous item", importance=3.0, modifiers=("ctrl", "shift"), base_key="G"),
        _sc(1, "Shift+F3", "My previous item", importance=5.0, modifiers=("shift",), base_key="F3"),
        _sc(2, "Ctrl+G", "My next item", importance=4.0, modifiers=("ctrl",), base_key="G"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    assert len(clusters) == 1
    c = clusters[0]
    # Should keep Shift+F3 (highest importance) as the previous representative.
    assert [m.sid for m in c.members] == [1, 2]


def test_clipboard_includes_cut():
    shortcuts = [
        _sc(0, "Ctrl+X", "Cut", modifiers=("ctrl",), base_key="X"),
        _sc(1, "Ctrl+C", "Copy", modifiers=("ctrl",), base_key="C"),
        _sc(2, "Ctrl+V", "Paste", modifiers=("ctrl",), base_key="V"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    assert len(clusters) == 1
    c = clusters[0]
    assert [m.sid for m in c.members] == [0, 1, 2]
    assert [m.order for m in c.members] == [-1, 0, 1]


def test_keyword_family_prefers_whole_words():
    shortcuts = [
        _sc(0, "Ctrl+C", "Copy", modifiers=("ctrl",), base_key="C"),
        _sc(1, "Ctrl+Shift+C", "Copy line up", modifiers=("ctrl", "shift"), base_key="C"),
        _sc(2, "Ctrl+V", "Paste", modifiers=("ctrl",), base_key="V"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    names = {c.name for c in clusters}
    # "Copy line up" must not be treated as a clipboard member.
    clipboard = next((c for c in clusters if "clipboard" in c.name), None)
    assert clipboard is not None
    assert sorted(m.sid for m in clipboard.members) == [0, 2]


def test_keyword_family_detects_browser_tab_group():
    shortcuts = [
        _sc(0, "Ctrl+T", "New tab", app="Browser (Chrome/Edge)", modifiers=("ctrl",), base_key="T"),
        _sc(1, "Ctrl+W", "Close tab", app="Browser (Chrome/Edge)", modifiers=("ctrl",), base_key="W"),
        _sc(2, "Ctrl+Shift+T", "Reopen closed tab", app="Browser (Chrome/Edge)", modifiers=("ctrl", "shift"), base_key="T"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    names = {c.name for c in clusters}
    assert "family_browser_tab" in names
    tab = next(c for c in clusters if c.name == "family_browser_tab")
    assert sorted(m.sid for m in tab.members) == [0, 1, 2]


def test_semantic_workflow_rows_are_high_weight():
    """Semantic clusters must produce strong same-layer workflow rows."""
    from fitness.kernel import _semantic_workflow_rows

    shortcuts = [
        _sc(0, "Ctrl+C", "Copy", modifiers=("ctrl",), base_key="C"),
        _sc(1, "Ctrl+V", "Paste", modifiers=("ctrl",), base_key="V"),
    ]
    layout = type("Layout", (), {
        "n_shortcuts": 2,
        "semantic_clusters": ({
            "name": "sequence_clipboard",
            "sids": [0, 1],
            "weight": 5.0,
            "members": [{"sid": 0, "order": 0, "dx": 0.0, "dy": 0.0},
                        {"sid": 1, "order": 1, "dx": 1.0, "dy": 0.0}],
        },),
    })()
    rows = _semantic_workflow_rows(layout)
    assert rows.shape == (1, 3)
    assert rows[0, 0] == 0.0
    assert rows[0, 1] == 1.0
    # Multiplier is 5e7, so pair weight = 5.0 * 5e7 = 2.5e8
    assert rows[0, 2] == 250_000_000.0


def test_semantic_cluster_split_penalty_is_strong():
    """Splitting a semantic cluster across layers must increase the violations objective."""
    import numpy as np
    from core.loader import build_layout
    from fitness.model import FitnessModel
    from config import DEFAULT_CONFIG

    layout = build_layout("data", DEFAULT_CONFIG)
    cluster = next(c for c in layout.semantic_clusters if len(c["sids"]) >= 2)
    sids = list(cluster["sids"])

    # Mutable positions grouped by layer, excluding L7
    mutable_by_layer = {}
    for idx in layout.mutable_indices:
        layer = layout.positions[idx].layer
        if layer != 7:
            mutable_by_layer.setdefault(layer, []).append(idx)
    layers = sorted(mutable_by_layer.keys())[:2]
    assert len(layers) == 2, "need two mutable layers for the test"

    def make_genome(layer_for_sid):
        genome = layout.genome.copy()
        # clear old placements of cluster sids
        for i, sid in enumerate(genome):
            if sid in layer_for_sid:
                genome[i] = -1
        for sid, layer in layer_for_sid.items():
            for idx in mutable_by_layer[layer]:
                if genome[idx] < 0:
                    genome[idx] = sid
                    break
            else:
                raise RuntimeError(f"no empty slot on layer {layer}")
        return genome

    split_map = {sids[i]: layers[i % 2] for i in range(len(sids))}
    together_map = {sid: layers[0] for sid in sids}

    model = FitnessModel(
        layout=layout,
        weights=DEFAULT_CONFIG["fitness"]["weights"],
        violation_weights=DEFAULT_CONFIG["fitness"]["violation_sub_weights"],
        missing_important_threshold=DEFAULT_CONFIG["fitness"]["missing_important_threshold"],
        hard_constraints=DEFAULT_CONFIG["fitness"]["hard_constraints"],
        toggle_effort_multiplier=DEFAULT_CONFIG["fitness"]["toggle_effort_multiplier"],
        require_cuda=False,
    )

    split_layout = layout.clone_with(genome=make_genome(split_map))
    together_layout = layout.clone_with(genome=make_genome(together_map))

    obj_split, _ = model.evaluate(split_layout.genome)
    obj_together, _ = model.evaluate(together_layout.genome)

    assert obj_split[2] > obj_together[2] + 1000.0, \
        f"Expected large violations increase when cluster {cluster['name']} is split; " \
        f"split={obj_split[2]:.1f}, together={obj_together[2]:.1f}"


def test_compactness_cluster_group_placements():
    """Compactness-only semantic clusters must produce atomic group placements."""
    from evolution import build_group_placements
    from core.loader import build_layout
    from config import DEFAULT_CONFIG

    layout = build_layout("data", DEFAULT_CONFIG)
    groups = build_group_placements(layout)
    # Find a keyword-family cluster with all offsets (0, 0)
    for cluster in layout.semantic_clusters:
        members = cluster.get("members", [])
        if len(members) < 2:
            continue
        if all(abs(m.get("dx", 0.0)) < 0.01 and abs(m.get("dy", 0.0)) < 0.01 for m in members):
            sids = set(cluster["sids"])
            placements = [g for g in groups if set(g[0]) == sids]
            assert len(placements) >= 1, \
                f"Compactness cluster {cluster['name']} should have atomic placements"
            # Each placement should put all members on the same layer
            for sid_tuple, anchor_list in placements:
                for placement in anchor_list[:3]:
                    layers = {layout.positions[idx].layer for idx in placement}
                    assert len(layers) == 1, \
                        f"Atomic placement for {cluster['name']} spans layers {layers}"
            break
    else:
        # No compactness cluster found in this corpus; skip assertively
        pass


if __name__ == "__main__":
    test_detects_copy_paste()
    test_detects_undo_redo()
    test_detects_directional_action_stem()
    test_ignores_mouse_buttons()
    test_clusters_raw_arrows()
    test_keeps_best_importance_per_order_for_direction_stems()
    test_clipboard_includes_cut()
    test_keyword_family_prefers_whole_words()
    test_keyword_family_detects_browser_tab_group()
    test_semantic_workflow_rows_are_high_weight()
    test_semantic_cluster_split_penalty_is_strong()
    test_compactness_cluster_group_placements()
    print("All semantic cluster tests passed.")
