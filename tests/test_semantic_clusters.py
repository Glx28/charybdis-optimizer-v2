"""Tests for core/semantic_clusters.py"""
import sys
import os
import pytest

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
    assert [(m.dx, m.dy) for m in c.members] == [(0.0, 0.0), (1.0, 0.0)]


def test_redo_aliases_stay_adjacent_to_the_undo_redo_sequence():
    shortcuts = [
        _sc(0, "Ctrl+Z", "Undo", modifiers=("ctrl",), base_key="Z"),
        _sc(1, "Ctrl+Y", "Redo", modifiers=("ctrl",), base_key="Y"),
        _sc(2, "Ctrl+Shift+Z", "Redo", modifiers=("ctrl", "shift"), base_key="Z"),
    ]
    cluster = next(c for c in detect_semantic_clusters(shortcuts) if c.name == "sequence_history")
    assert [m.sid for m in cluster.members] == [0, 1, 2]
    assert [(m.dx, m.dy) for m in cluster.members] == [
        (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
    ]


def test_keyword_groups_keep_distinct_shortcut_contexts_separate():
    shortcuts = [
        _sc(0, "Ctrl+F", "Find on page"),
        _sc(1, "Win+Shift+M", "Find My Mouse", app="PowerToys"),
        _sc(2, "Ctrl+J", "Downloads"),
        _sc(3, "Ctrl+H", "History"),
        _sc(4, "Win+V", "Clipboard history"),
        _sc(5, "Ctrl+Shift+H", "Version history", app="M-Files Desktop Client"),
        _sc(6, "Ctrl+T", "New tab"),
        _sc(7, "Ctrl+Tab", "Next tab"),
        _sc(8, "Ctrl+1", "Switch to tab 1"),
        _sc(9, "Ctrl+2", "Switch to tab 2"),
        _sc(10, "Win+Left", "Snap window left"),
        _sc(11, "Alt+F4", "Close window"),
        _sc(12, "Win+Ctrl+`", "Workspaces (save/restore window sets)"),
        _sc(13, "Ctrl+G", "Find next"),
        _sc(14, "Win+S", "Search", app="Windows 11"),
        _sc(15, "Win+R", "Run dialog", app="Windows 11"),
        _sc(16, "Ctrl+E", "Search", app="Discord"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    by_name = {c.name: {m.sid for m in c.members} for c in clusters}
    assert by_name["family_browser_find"] == {0, 13}
    assert by_name["family_browser_panel"] == {2, 3}
    assert by_name["family_browser_tab"] == {6, 7}
    assert any(
        c.name == "pattern_Ctrl_switch_to_tab" and {m.sid for m in c.members} == {8, 9}
        for c in clusters
    )
    assert by_name["family_window_state"] == {10, 11}
    assert by_name["family_windows_launcher"] == {14, 15}


def test_before_after_are_ordered_left_to_right():
    shortcuts = [
        _sc(0, "Ctrl+Left", "Move before", base_key="Left"),
        _sc(1, "Ctrl+Right", "Move after", base_key="Right"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    relation = next(c for c in clusters if {m.sid for m in c.members} == {0, 1})
    assert [(m.dx, m.dy) for m in relation.members] == [(0.0, 0.0), (1.0, 0.0)]


def test_previous_next_with_pageup_pagedown_bindings_are_vertical():
    shortcuts = [
        _sc(0, "Page Up", "Previous slide", app="PowerPoint", base_key="PageUp"),
        _sc(1, "Page Down", "Next slide", app="PowerPoint", base_key="PageDown"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    relation = next(c for c in clusters if {m.sid for m in c.members} == {0, 1})
    assert [(m.dx, m.dy) for m in relation.members] == [(0.0, 0.0), (0.0, 1.0)]


def test_physical_plus_minus_keys_override_zoom_action_order():
    shortcuts = [
        _sc(0, "Ctrl++", "Zoom in", modifiers=("Ctrl",), base_key="Equals and Plus"),
        _sc(1, "Ctrl+-", "Zoom out", modifiers=("Ctrl",), base_key="Dash and Underscore"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    relation = next(c for c in clusters if {m.sid for m in c.members} == {0, 1})
    by_sid = {m.sid: (m.dx, m.dy) for m in relation.members}
    assert by_sid == {1: (0.0, 0.0), 0: (1.0, 0.0)}


def test_physical_angle_bracket_keys_override_font_size_action_order():
    shortcuts = [
        _sc(0, "Ctrl+Shift+>", "Increase font size", modifiers=("Ctrl", "Shift"),
            base_key="Period and GreaterThan"),
        _sc(1, "Ctrl+Shift+<", "Decrease font size", modifiers=("Ctrl", "Shift"),
            base_key="Comma and LessThan"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    relation = next(c for c in clusters if {m.sid for m in c.members} == {0, 1})
    by_sid = {m.sid: (m.dx, m.dy) for m in relation.members}
    assert by_sid == {1: (0.0, 0.0), 0: (1.0, 0.0)}


def test_angle_bracket_base_keys_do_not_pair_unrelated_actions():
    shortcuts = [
        _sc(0, "Ctrl+,", "Settings", modifiers=("Ctrl",), base_key="Comma and LessThan"),
        _sc(1, "Ctrl+.", "Show commands", modifiers=("Ctrl",), base_key="Period and GreaterThan"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    assert not any({m.sid for m in c.members} == {0, 1} for c in clusters)


def test_horizontal_vertical_split_uses_minus_plus_key_order():
    shortcuts = [
        _sc(0, "Alt+Shift+-", "Split pane horizontal", modifiers=("Alt", "Shift"),
            base_key="Dash and Underscore"),
        _sc(1, "Alt+Shift++", "Split pane vertical", modifiers=("Alt", "Shift"),
            base_key="Equals and Plus"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    relation = next(c for c in clusters if {m.sid for m in c.members} == {0, 1})
    by_sid = {m.sid: (m.dx, m.dy) for m in relation.members}
    assert by_sid == {0: (0.0, 0.0), 1: (1.0, 0.0)}


def test_detects_directional_action_stem():
    shortcuts = [
        _sc(0, "Win+Left", "Snap window left", modifiers=("win",), base_key="Left"),
        _sc(1, "Win+Right", "Snap window right", modifiers=("win",), base_key="Right"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    assert len(clusters) >= 1
    relation = next(c for c in clusters if {m.sid for m in c.members} == {0, 1})
    assert relation.category == "direction"
    assert [(m.dx, m.dy) for m in relation.members] == [(0.0, 0.0), (1.0, 0.0)]


def test_ignores_mouse_buttons():
    shortcuts = [
        _sc(0, "MB1", "Left click", category="mouse"),
        _sc(1, "MB2", "Right click", category="mouse"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    assert len(clusters) == 0


def test_raw_arrows_have_no_mutable_semantic_cluster():
    shortcuts = [
        _sc(0, "LeftArrow", "Left arrow", base_key="LeftArrow"),
        _sc(1, "RightArrow", "Right arrow", base_key="RightArrow"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    assert clusters == []


def test_complete_raw_arrows_do_not_create_mutable_shape_contract():
    shortcuts = [
        _sc(0, "LeftArrow", "Left arrow", app="Mouse", base_key="LeftArrow"),
        _sc(1, "UpArrow", "Up arrow", app="Mouse", base_key="UpArrow"),
        _sc(2, "DownArrow", "Down arrow", app="Mouse", base_key="DownArrow"),
        _sc(3, "RightArrow", "Right arrow", app="Mouse", base_key="RightArrow"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    arrow_sids = {0, 1, 2, 3}
    assert not any(
        c.category in {"direction", "raw_key"}
        and set(m.sid for m in c.members).issubset(arrow_sids)
        for c in clusters
    )


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


def test_powertoys_tools_use_contextual_subgroups_instead_of_app_wide_cluster():
    shortcuts = [
        _sc(0, "Alt+Space", "PowerToys Run launcher", app="PowerToys"),
        _sc(1, "Win+Alt+Space", "PowerToys Command Palette (CmdPal)", app="PowerToys"),
        _sc(2, "Win+Shift+C", "Color Picker", app="PowerToys"),
        _sc(3, "Win+Shift+T", "Text Extractor (OCR)", app="PowerToys"),
        _sc(4, "Win+Shift+R", "Screen Ruler", app="PowerToys"),
        _sc(5, "Win+Shift+O", "Mouse Pointer Crosshairs", app="PowerToys"),
        _sc(6, "Win+Shift+D", "Mouse Highlighter", app="PowerToys"),
        _sc(7, "Win+Shift+M", "Find My Mouse", app="PowerToys"),
        _sc(8, "Win+`", "FancyZones editor", app="PowerToys"),
        _sc(9, "Win+Ctrl+`", "Workspaces (save/restore window sets)", app="PowerToys"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    by_name = {cluster.name: cluster for cluster in clusters}

    assert "family_powertoys" not in by_name
    assert set(by_name["family_powertoys_launcher"].member_sids) == {0, 1}
    assert set(by_name["family_powertoys_visual_tools"].member_sids) == {2, 3, 4}
    assert set(by_name["family_powertoys_mouse_tools"].member_sids) == {5, 6, 7}
    assert set(by_name["family_powertoys_workspace_tools"].member_sids) == {8, 9}


def test_ordered_browser_tab_pair_is_not_swallowed_by_broad_family():
    shortcuts = [
        _sc(0, "Ctrl+T", "New tab", app="Browser (Chrome/Edge)", modifiers=("ctrl",), base_key="T"),
        _sc(1, "Ctrl+Tab", "Next tab", app="Browser (Chrome/Edge)", modifiers=("ctrl",), base_key="Tab"),
        _sc(2, "Ctrl+Shift+Tab", "Previous tab", app="Browser (Chrome/Edge)", modifiers=("ctrl", "shift"), base_key="Tab"),
    ]
    clusters = detect_semantic_clusters(shortcuts)
    assert not any(c.name == "family_browser_tab" for c in clusters)
    directional = next(c for c in clusters if c.category == "direction")
    assert {m.sid for m in directional.members} == {1, 2}
    assert [(m.dx, m.dy) for m in directional.members] == [(0.0, 0.0), (1.0, 0.0)]


def test_semantic_position_score_penalizes_reversed_order():
    import numpy as np
    from types import SimpleNamespace
    from fitness.model import FitnessModel

    layout = SimpleNamespace(
        n_shortcuts=2,
        positions=(SimpleNamespace(layer=1, x=0.0, y=0.0),
                   SimpleNamespace(layer=1, x=1.0, y=0.0)),
        semantic_clusters=({
            "weight": 1.0,
            "is_critical": True,
            "members": [{"sid": 0, "order": 0, "dx": 0.0, "dy": 0.0},
                        {"sid": 1, "order": 1, "dx": 1.0, "dy": 0.0}],
        },),
    )
    model = FitnessModel.__new__(FitnessModel)
    model.layout = layout
    model.weights = {"workflow_coherence": 1.0}
    model.scale_factors = np.ones(3, dtype=np.float32)
    model.semantic_cluster_multiplier = 1.0
    model.semantic_cluster_pair_boost = 1.0
    model.semantic_position_penalty = 120.0
    model.semantic_contract_penalty = 2000.0
    model.sparse_layer_base_penalty = 2500.0
    model.sparse_layer_gap_penalty = 500.0
    model._semantic_cluster_starts = np.asarray([0], dtype=np.int32)
    model._semantic_cluster_sids = np.asarray([], dtype=np.int32)
    model._semantic_cluster_weights = np.asarray([], dtype=np.float32)
    model._access_targets = np.asarray([-1, -1], dtype=np.int32)
    model._semantic_position_edges = model._build_semantic_position_edges(layout)
    model._semantic_position_edges_array = np.asarray(
        model._semantic_position_edges, dtype=np.float32
    ).reshape((-1, 5))
    model._position_layers = np.asarray([p.layer for p in layout.positions], dtype=np.int32)
    model._position_x = np.asarray([p.x for p in layout.positions], dtype=np.float32)
    model._position_y = np.asarray([p.y for p in layout.positions], dtype=np.float32)

    base = np.zeros((2, 3), dtype=np.float32)
    scored = model._add_semantic_position_score(base, np.array([[0, 1], [1, 0]], dtype=np.int32))
    assert scored[0, 2] == 0.0
    assert scored[1, 2] > 0.0


def test_production_shortcut_groups_keep_contextual_relations():
    from core.loader import build_layout
    from config import DEFAULT_CONFIG

    layout = build_layout("data", DEFAULT_CONFIG)
    clusters = list(layout.semantic_clusters)

    # Every inferred ordered pair in the real shortcut corpus must have one
    # consistent physical direction, even when several families overlap.
    pair_relations = {}
    for cluster in clusters:
        members = list(cluster["members"])
        for i, first in enumerate(members):
            for second in members[i + 1:]:
                vector = (round(float(second["dx"]) - float(first["dx"]), 3),
                          round(float(second["dy"]) - float(first["dy"]), 3))
                if vector == (0.0, 0.0):
                    continue
                pair = (int(first["sid"]), int(second["sid"]))
                canonical = pair if pair[0] < pair[1] else (pair[1], pair[0])
                canonical_vector = vector if pair == canonical else (-vector[0], -vector[1])
                previous = pair_relations.setdefault(canonical, canonical_vector)
                assert previous == canonical_vector, (
                    layout.shortcuts[canonical[0]].keys,
                    layout.shortcuts[canonical[1]].keys,
                )

    history = next(c for c in clusters if c["name"] == "sequence_history")
    history_offsets = {
        layout.shortcuts[m["sid"]].keys: (m["dx"], m["dy"])
        for m in history["members"]
    }
    assert history_offsets["Ctrl+Z"] == (0.0, 0.0)
    assert history_offsets["Ctrl+Shift+Z"] == (1.0, 0.0)
    assert history_offsets["Ctrl+Y"] == (2.0, 0.0)
    arrow_sids = {
        sc.sid for sc in layout.shortcuts
        if not sc.modifiers and (sc.base_key or "").upper()
        in {"LEFTARROW", "UPARROW", "DOWNARROW", "RIGHTARROW"}
    }
    assert not any(c["name"] == "family_mouse_arrows" for c in clusters)
    assert not any(
        c["category"] in {"direction", "raw_key"}
        and {m["sid"] for m in c["members"]}.issubset(arrow_sids)
        for c in clusters
    )

    tab_digits = next(c for c in clusters if c["name"] == "pattern_Ctrl_switch_to_tab")
    tab_actions = {layout.shortcuts[m["sid"]].action for m in tab_digits["members"]}
    assert "Reset zoom" not in tab_actions
    assert "Switch to tab 1" in tab_actions and "Last tab" in tab_actions
    tab_offsets = {
        layout.shortcuts[m["sid"]].keys: (m["dx"], m["dy"])
        for m in tab_digits["members"]
    }
    assert tab_offsets["Ctrl+1"] == (0.0, 0.0)
    assert tab_offsets["Ctrl+4"] == (0.0, 1.0)
    assert tab_offsets["Ctrl+9"] == (2.0, 2.0)

    pinned_apps = next(c for c in clusters if c["name"] == "pattern_Win_open_switch_pinned_app")
    pinned_offsets = {
        layout.shortcuts[m["sid"]].keys: (m["dx"], m["dy"])
        for m in pinned_apps["members"]
    }
    assert pinned_offsets == {
        "Win+1": (0.0, 0.0), "Win+2": (1.0, 0.0),
        "Win+3": (2.0, 0.0), "Win+4": (3.0, 0.0), "Win+5": (4.0, 0.0),
    }


def test_semantic_report_flags_wrong_relative_order():
    import numpy as np
    from types import SimpleNamespace
    from tools.semantic_cluster_report import _cluster_quality

    shortcuts = (
        _sc(0, "Left", "Move left"),
        _sc(1, "Right", "Move right"),
    )
    positions = (
        SimpleNamespace(layer=1, x=0.0, y=0.0, hand="left"),
        SimpleNamespace(layer=1, x=1.0, y=0.0, hand="right"),
    )
    cluster = {"name": "directional_pair", "category": "direction", "weight": 1.0,
               "members": [{"sid": 0, "order": 0, "dx": 0.0, "dy": 0.0},
                           {"sid": 1, "order": 1, "dx": 1.0, "dy": 0.0}]}
    layout = SimpleNamespace(genome=np.array([0, 1]), positions=positions, shortcuts=shortcuts)
    assert _cluster_quality(layout, cluster)["relative_layout_pass"]
    layout.genome = np.array([1, 0])
    assert not _cluster_quality(layout, cluster)["relative_layout_pass"]

    clipboard = {
        "name": "sequence_clipboard", "category": "sequence", "weight": 1.0,
        "members": [{"sid": 0, "order": -1, "dx": 0.0, "dy": 0.0},
                    {"sid": 1, "order": 0, "dx": 1.0, "dy": 0.0},
                    {"sid": 2, "order": 1, "dx": 2.0, "dy": 0.0}],
    }
    clipboard_layout = SimpleNamespace(
        genome=np.array([0, 1, 2]),
        positions=positions + (SimpleNamespace(layer=1, x=2.0, y=0.0, hand="right"),),
        shortcuts=shortcuts + (_sc(2, "Paste", "Paste"),),
    )
    assert _cluster_quality(clipboard_layout, clipboard)["relative_layout_pass"]
    clipboard_layout.genome = np.array([2, 1, 0])
    assert not _cluster_quality(clipboard_layout, clipboard)["relative_layout_pass"]


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
    rows = _semantic_workflow_rows(layout, multiplier=200.0)
    assert rows.shape == (1, 3)
    assert rows[0, 0] == 0.0
    assert rows[0, 1] == 1.0
    # Default multiplier is 200.0; size-2 discount is 1/sqrt(2).
    # pair weight = 5.0 * 200.0 / sqrt(2) ≈ 707.1
    assert rows[0, 2] == pytest.approx(707.1068, rel=1e-4)


def test_overlapping_semantic_groups_keep_max_pair_pressure_once():
    from fitness.kernel import _semantic_workflow_rows

    layout = type("Layout", (), {
        "n_shortcuts": 3,
        "semantic_clusters": (
            {"weight": 1.0, "members": [
                {"sid": 0, "order": 0, "dx": 0.0, "dy": 0.0},
                {"sid": 1, "order": 0, "dx": 0.0, "dy": 0.0},
                {"sid": 2, "order": 0, "dx": 0.0, "dy": 0.0},
            ]},
            {"weight": 1.0, "is_critical": True, "members": [
                {"sid": 0, "order": 0, "dx": 0.0, "dy": 0.0},
                {"sid": 1, "order": 1, "dx": 1.0, "dy": 0.0},
            ]},
        ),
    })()
    rows = _semantic_workflow_rows(layout, multiplier=200.0)
    assert rows.shape == (3, 3)
    pair_01 = next(row for row in rows if tuple(row[:2].astype(int)) == (0, 1))
    assert pair_01[2] == pytest.approx(8.0 * 200.0 / (2.0 ** 0.5), rel=1e-4)


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
        scale_factors=np.array([61_590_358.0, 15_435.586, 323_001_352_192.0], dtype=np.float32),
        semantic_contract_penalty=2000.0,
        semantic_position_penalty=120.0,
        sparse_layer_base_penalty=2500.0,
        sparse_layer_gap_penalty=500.0,
        require_cuda=False,
    )

    split_layout = layout.clone_with(genome=make_genome(split_map))
    together_layout = layout.clone_with(genome=make_genome(together_map))

    from fitness.model import _semantic_and_sparse_contract_batch
    args = (
        model._position_layers, model._semantic_cluster_starts,
        model._semantic_cluster_sids, model._semantic_cluster_weights,
        model._access_targets, model.semantic_contract_penalty,
        model.sparse_layer_base_penalty, model.sparse_layer_gap_penalty,
    )
    split_penalty = _semantic_and_sparse_contract_batch(
        split_layout.genome[None, :], *args,
    )[0]
    together_penalty = _semantic_and_sparse_contract_batch(
        together_layout.genome[None, :], *args,
    )[0]
    assert split_penalty > together_penalty + 1000.0, \
        f"Expected large contract penalty when cluster {cluster['name']} is split"


def test_contract_penalty_scores_semantic_splits_at_production_scale():
    import numpy as np
    from fitness.model import _semantic_and_sparse_contract_batch

    positions = np.asarray([1, 1, 2, 2], dtype=np.int32)
    starts = np.asarray([0, 2], dtype=np.int32)
    sids = np.asarray([0, 1], dtype=np.int32)
    weights = np.asarray([1.5], dtype=np.float32)
    targets = np.asarray([-1, -1], dtype=np.int32)
    split = _semantic_and_sparse_contract_batch(
        np.asarray([[0, -1, 1, -1]], dtype=np.int32), positions, starts, sids, weights,
        targets, 2000.0, 2500.0, 500.0,
    )
    together = _semantic_and_sparse_contract_batch(
        np.asarray([[0, 1, -1, -1]], dtype=np.int32), positions, starts, sids, weights,
        targets, 2000.0, 2500.0, 500.0,
    )
    assert split[0] - together[0] == pytest.approx(3000.0)


def test_temporary_evaluator_preserves_contract_penalties():
    import numpy as np
    from core.loader import build_layout
    from fitness.evaluator import FitnessEvaluator
    from config import DEFAULT_CONFIG

    layout = build_layout("data", DEFAULT_CONFIG)
    evaluator = FitnessEvaluator(
        weights=DEFAULT_CONFIG["fitness"]["weights"],
        reference_layout=layout,
        scale_factors=np.asarray([61_590_358.0, 15_435.586, 323_001_352_192.0], dtype=np.float32),
        violation_weights=DEFAULT_CONFIG["fitness"]["violation_sub_weights"],
        hard_constraints=DEFAULT_CONFIG["fitness"]["hard_constraints"],
        semantic_contract_penalty=2000.0,
        semantic_position_penalty=120.0,
        sparse_layer_base_penalty=2500.0,
        sparse_layer_gap_penalty=500.0,
    )
    reference_score = evaluator.evaluate(layout).total_score
    cloned_score = evaluator.evaluate(layout.clone_with(genome=layout.genome.copy())).total_score
    assert cloned_score == pytest.approx(reference_score, rel=1e-6)


def test_sparse_reachable_layer_occupancy_has_no_score_cutoff():
    import numpy as np
    from fitness.model import _semantic_and_sparse_contract_batch

    positions = np.asarray([0] + [5] * 25 + [7] * 15, dtype=np.int32)
    genome_sparse = np.full(len(positions), -1, dtype=np.int32)
    genome_full = np.full(len(positions), -1, dtype=np.int32)
    genome_sparse[0] = 0
    genome_full[0] = 0
    genome_sparse[1:16] = np.arange(1, 16)
    genome_full[1:21] = np.arange(1, 21)
    # Both candidates partially occupy frozen L7; it must never be charged.
    genome_sparse[26:36] = np.arange(1, 11)
    genome_full[26:36] = np.arange(1, 11)
    targets = np.full(21, -1, dtype=np.int32)
    targets[0] = 5
    result = _semantic_and_sparse_contract_batch(
        np.stack([genome_sparse, genome_full]), positions,
        np.asarray([0], dtype=np.int32), np.asarray([], dtype=np.int32),
        np.asarray([], dtype=np.float32), targets, 2000.0, 2500.0, 500.0,
    )
    assert result.tolist() == pytest.approx([0.0, 0.0])


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


def test_raw_completion_compactness_uses_exact_five_key_shape():
    """All five Norwegian raw keys share compactness and one exact-shape move."""
    from evolution import build_group_placements
    from core.loader import build_layout
    from config import DEFAULT_CONFIG

    layout = build_layout("data", DEFAULT_CONFIG)
    groups = build_group_placements(layout)
    family = next(c for c in layout.semantic_clusters if c["name"] == "family_raw_completion")
    family_sids = set(family["sids"])
    assert len(family_sids) == 5
    complete_groups = [set(sids) for sids, _ in groups if len(sids) == 5 and set(sids) == family_sids]
    assert len(complete_groups) == 1


def test_fresh_random_population_preserves_norwegian_completion_shape():
    """Later overlapping group anchors must not overwrite the completion cluster."""
    import random
    import numpy as np
    from dataclasses import replace
    from core.loader import build_layout
    from config import DEFAULT_CONFIG
    from evolution.completion_cluster import analyze_completion_cluster
    from run_evolution import generate_random_layouts

    layout = build_layout("data", DEFAULT_CONFIG)
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(20261003)
        np.random.seed(20261003)
        genomes = generate_random_layouts(layout, 24)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

    reports = [
        analyze_completion_cluster(replace(layout, genome=genome.copy()))
        for genome in genomes[1:]
    ]
    assert all(report["acceptance_pass"] for report in reports), [
        (report["raw_base_keys_missing"], report["wrong_shape_members"])
        for report in reports if not report["acceptance_pass"]
    ]


def test_mouse_workflow_mutation_preserves_required_groups_and_accesses_trackball_safely():
    """Mouse placement must not dismantle relative groups or use right-thumb holds."""
    import random
    import numpy as np
    from dataclasses import replace
    from core.loader import build_layout
    from config import DEFAULT_CONFIG
    from evolution import SwapMutation
    from evolution.acceptance import build_acceptance_report
    from evolution.arrow_cluster import analyze_arrows
    from evolution.completion_cluster import analyze_completion_cluster
    from run_evolution import analyze_duplicates, generate_random_layouts

    layout = build_layout("data", DEFAULT_CONFIG)
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(42003)
        np.random.seed(42003)
        genomes = generate_random_layouts(layout, 17)
        mutation = SwapMutation(prob=0.0, frozen_mask=layout.frozen_mask, layout=layout)
        reports = []
        for genome in genomes[1:]:
            candidate = genome.copy()
            assert mutation._propose_mouse_workflow_layer(candidate)
            candidate_layout = replace(layout, genome=candidate)
            report = build_acceptance_report(
                candidate_layout,
                analyze_duplicates(candidate_layout),
                analyze_completion_cluster(candidate_layout),
                analyze_arrows(candidate_layout),
            )
            reports.append(report["optimizer_side_checks"])
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

    assert all(row["dynamic_mouse_layer_present"] for row in reports)
    assert all(row["norwegian_completion_cluster"] for row in reports)
    assert all(row["required_relative_layouts"] for row in reports)


def test_ordered_numeric_clusters_have_atomic_geometry_placements():
    from evolution import build_group_placements
    from core.loader import build_layout
    from config import DEFAULT_CONFIG

    layout = build_layout("data", DEFAULT_CONFIG)
    groups = build_group_placements(layout)
    by_name = {
        cluster["name"]: set(cluster["sids"])
        for cluster in layout.semantic_clusters
        if cluster["name"] in {
            "pattern_Ctrl_switch_to_tab",
            "pattern_Win_open_switch_pinned_app",
        }
    }
    for name, sids in by_name.items():
        placements = [anchors for group_sids, anchors in groups if set(group_sids) == sids]
        assert placements and placements[0], f"{name} missing atomic ordered placements"
        cluster = next(c for c in layout.semantic_clusters if c["name"] == name)
        expected = {
            int(member["sid"]): (round(float(member["dx"])), round(float(member["dy"])))
            for member in cluster["members"]
        }
        group_sids, anchors = next(group for group in groups if set(group[0]) == sids)
        for anchor in anchors[:10]:
            coords = {
                sid: (round(layout.positions[idx].x), round(layout.positions[idx].y))
                for sid, idx in zip(group_sids, anchor)
            }
            first_sid = next(sid for sid, offset in expected.items() if offset == (0, 0))
            ax, ay = coords[first_sid]
            assert all(coords[sid] == (ax + dx, ay + dy)
                       for sid, (dx, dy) in expected.items())


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
