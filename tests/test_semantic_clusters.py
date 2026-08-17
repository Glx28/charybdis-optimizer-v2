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
    print("All semantic cluster tests passed.")
