"""Data structure tests (Position, Shortcut, Layout, UsageData)."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import numpy as np

from config import DEFAULT_CONFIG
from core import Layout, Position, Shortcut, UsageData
from core.loader import (
    _discover_dynamic_groups,
    _parse_layer_from_behavior,
    build_frozen_genome,
    load_layout,
    load_shortcuts,
    load_usage_stats,
)
from core.norwegian_keys import parse_shortcut_keys_norwegian
from fitness.evaluator import FitnessEvaluator
from tests.legacy_factors import ViolationFactor
from fitness.kernel import DEFAULT_FITNESS_WEIGHTS, DEFAULT_VIOLATION_WEIGHTS


class TestDataStructures(unittest.TestCase):
    def test_fallback_weights_are_synchronized_with_default_config(self):
        self.assertEqual(DEFAULT_FITNESS_WEIGHTS, DEFAULT_CONFIG["fitness"]["weights"])
        self.assertEqual(DEFAULT_VIOLATION_WEIGHTS, DEFAULT_CONFIG["fitness"]["violation_sub_weights"])
        self.assertEqual(FitnessEvaluator().weights, DEFAULT_CONFIG["fitness"]["weights"])
        self.assertEqual(ViolationFactor().sub_weights, DEFAULT_CONFIG["fitness"]["violation_sub_weights"])

    def test_position_creation(self):
        p = Position(0, 0, 0.0, 0.0, "left", 1, 1.0)
        self.assertEqual(p.gene_idx, 0)
        self.assertEqual(p.hand, "left")
        self.assertTrue(p.is_left)

    def test_layout_validity(self):
        p = Position(0, 0, 0.0, 0.0, "left", 1, 1.0)
        s = Shortcut(0, "Ctrl+C", "copy", "windows", 8.0)
        g = np.array([0, -1], dtype=np.int32)
        m = np.array([False, False])
        layout = Layout(g, (p, p), (s,), m)
        self.assertTrue(layout.is_valid())
        self.assertEqual(layout.n_assigned, 1)

    def test_l0_raw_duplicates_are_filtered(self):
        layout_data = {
            "n_layers": 1,
            "l0_frozen": {
                "8:2": {"x": 8, "y": 2, "label": "J", "behavior": "Key Press", "parameter": "J", "modifiers": []},
                "4:4": {"x": 4, "y": 4, "label": "Space", "behavior": "Key Press", "parameter": "Spacebar", "modifiers": []},
                "7:5": {"x": 7, "y": 5, "label": "Ret", "behavior": "Key Press", "parameter": "Return Enter", "modifiers": []},
            }
        }
        app_scores = {
            "apps": [{
                "name": "Browser",
                "shortcuts": [
                    {"keys": "J", "action": "Previous tab", "importance": 5.7},
                    {"keys": "Ctrl+J", "action": "Downloads", "importance": 1.0},
                    {"keys": "Spacebar", "action": "Page down", "importance": 5.0},
                    {"keys": "Enter", "action": "Send", "importance": 5.0},
                    {"keys": "LeftAlt", "action": "Menu", "importance": 5.0},
                ],
            }]
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump(app_scores, f)
            path = f.name
        try:
            shortcuts = load_shortcuts(path, layout_data)
        finally:
            os.unlink(path)

        keys = {s.keys for s in shortcuts}
        self.assertIn("_base_j", keys)
        self.assertIn("_base_spacebar", keys)
        self.assertIn("_base_returnenter", keys)
        self.assertIn("Ctrl+J", keys)
        self.assertIn("LeftAlt", keys)
        self.assertNotIn("J", keys)
        self.assertNotIn("Spacebar", keys)
        self.assertNotIn("Enter", keys)

    def test_non_exportable_sequences_filtered_but_mouse_click_shortcuts_remain(self):
        layout_data = {"n_layers": 1, "l0_frozen": {}}
        app_scores = {
            "apps": [{
                "name": "Mixed",
                "shortcuts": [
                    {"keys": "ScrollUp", "action": "Scroll up", "importance": 8.0},
                    {"keys": "ScrollDown", "action": "Scroll down", "importance": 8.0},
                    {"keys": "yy", "action": "Yank line", "category": "vimium", "importance": 6.0},
                    {"keys": "gg", "action": "Top", "category": "vimium", "importance": 6.0},
                    {"keys": "gi", "action": "Focus input", "category": "vimium", "importance": 6.0},
                    {"keys": "Ctrl+K Ctrl+F", "action": "Format selection", "importance": 8.0},
                    {"keys": "Ctrl+K Ctrl+S", "action": "Keyboard shortcuts", "importance": 8.0},
                    {"keys": "Ctrl+Click", "action": "Open in new tab", "importance": 9.0},
                    {"keys": "Shift+Click", "action": "Range select", "importance": 8.0},
                    {"keys": "Alt+Click", "action": "Alternate click", "importance": 7.0},
                    {"keys": "Right Click", "action": "Context menu", "importance": 7.0},
                ],
            }]
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump(app_scores, f)
            path = f.name
        try:
            shortcuts = load_shortcuts(path, layout_data)
        finally:
            os.unlink(path)

        by_key = {s.keys: s for s in shortcuts}
        for invalid in ("ScrollUp", "ScrollDown", "yy", "gg", "gi", "Ctrl+K Ctrl+F", "Ctrl+K Ctrl+S"):
            self.assertNotIn(invalid, by_key)
        for click_key in ("Ctrl+Click", "Shift+Click", "Alt+Click"):
            self.assertIn(click_key, by_key)
            self.assertEqual(by_key[click_key].base_key, "Click")
        self.assertIn("Right Click", by_key)
        self.assertEqual(by_key["Right Click"].base_key, "Right Click")

    def test_layer_access_positions_are_mutable_capabilities(self):
        layout_data = {
            "n_layers": 2,
            "physical_grid": {
                "positions": [
                    {"x": 3, "y": 4, "hand": "left", "finger": "thumb", "zone": "thumb", "row_type": "thumb"},
                    {"x": 4, "y": 4, "hand": "left", "finger": "thumb", "zone": "thumb", "row_type": "thumb"},
                ]
            },
            "l0_frozen": {
                "4:4": {"x": 4, "y": 4, "label": "Space", "behavior": "Key Press", "parameter": "Spacebar", "modifiers": []},
            },
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump(layout_data, f)
            path = f.name
        app_scores = {"apps": []}
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump(app_scores, f)
            shortcuts_path = f.name
        try:
            positions, frozen, _ = load_layout(path)
            shortcuts = load_shortcuts(shortcuts_path, layout_data)
            genome = build_frozen_genome(layout_data, positions, shortcuts)
        finally:
            os.unlink(path)
            os.unlink(shortcuts_path)

        access_shortcuts = [s for s in shortcuts if s.is_layer_access]
        self.assertTrue(any(s.access_target_layer == 1 and s.access_is_momentary for s in access_shortcuts))
        self.assertFalse(bool(frozen[0]))
        self.assertTrue(bool(frozen[1]))
        self.assertEqual(int(genome[0]), -1)
        self.assertGreaterEqual(int(genome[1]), 0)

    def test_l7_has_momentary_and_toggle_access_capabilities(self):
        layout_data = {"n_layers": 8, "physical_grid": {"positions": []}, "l0_frozen": {}}
        app_scores = {"apps": []}
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump(app_scores, f)
            shortcuts_path = f.name
        try:
            shortcuts = load_shortcuts(shortcuts_path, layout_data)
        finally:
            os.unlink(shortcuts_path)

        l7_access = [s for s in shortcuts if s.is_layer_access and s.access_target_layer == 7]
        self.assertTrue(any(s.access_is_momentary for s in l7_access))
        self.assertTrue(any(not s.access_is_momentary for s in l7_access))

    def test_raw_arrow_shortcuts_are_single_physical_capabilities(self):
        layout_data = {"n_layers": 8, "physical_grid": {"positions": []}, "l0_frozen": {}}
        app_scores = {
            "apps": [
                {
                    "name": "App A",
                    "shortcuts": [
                        {"keys": "Up", "base_key": "UpArrow", "importance": 2.0},
                        {"keys": "Down", "base_key": "DownArrow", "importance": 2.0},
                    ],
                },
                {
                    "name": "Raw",
                    "shortcuts": [
                        {"keys": "UpArrow", "base_key": "UpArrow", "importance": 8.0},
                        {"keys": "DownArrow", "base_key": "DownArrow", "importance": 8.0},
                        {"keys": "LeftArrow", "base_key": "LeftArrow", "importance": 8.0},
                        {"keys": "RightArrow", "base_key": "RightArrow", "importance": 8.0},
                    ],
                },
            ]
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump(app_scores, f)
            shortcuts_path = f.name
        try:
            shortcuts = load_shortcuts(shortcuts_path, layout_data)
        finally:
            os.unlink(shortcuts_path)

        raw_arrows = [
            s for s in shortcuts
            if not s.modifiers and s.base_key in {"LeftArrow", "RightArrow", "UpArrow", "DownArrow"}
        ]
        self.assertEqual(len(raw_arrows), 4)
        self.assertEqual({s.base_key for s in raw_arrows}, {"LeftArrow", "RightArrow", "UpArrow", "DownArrow"})
        self.assertTrue(all(s.importance == 8.0 for s in raw_arrows))

    def test_layer_target_parsing_ignores_semantic_labels(self):
        self.assertEqual(_parse_layer_from_behavior("", "", "Layer::5"), 5)
        self.assertEqual(_parse_layer_from_behavior("", "coach_l3_hold", ""), 3)
        self.assertEqual(_parse_layer_from_behavior("", "coach_game_lock", ""), 7)
        self.assertEqual(_parse_layer_from_behavior("", "coach_base", ""), 0)

        self.assertIsNone(_parse_layer_from_behavior("Excel", "", ""))
        # Numeric coach behaviors are correctly parsed
        self.assertEqual(_parse_layer_from_behavior("Toggle L8", "coach_l8_toggle", ""), 8)
        self.assertEqual(_parse_layer_from_behavior("Toggle L2", "coach_l2_toggle", ""), 2)
        # Function-named behaviors with no numeric layer return None
        self.assertIsNone(_parse_layer_from_behavior("Hold L2", "coach_mouse_lock", ""))

    def test_load_new_workflow_usage_stats(self):
        usage_stats = {
            "shortcut_sequences": {
                "Ctrl+C -> Alt+Tab": {
                    "count": 4,
                    "avg_gap_ms": 250,
                    "p50_gap_ms": 220,
                    "same_app_count": 0,
                    "cross_app_count": 4,
                    "apps": {"code.exe": 4, "chrome.exe": 4},
                    "confidence": 0.9,
                }
            },
            "shortcut_workflows": {
                "Ctrl+C -> Alt+Tab -> Ctrl+V": {
                    "count": 3,
                    "avg_span_ms": 1400,
                    "apps": {"code.exe": 3, "chrome.exe": 3},
                    "app_count": 2,
                    "layer_count": 1,
                }
            },
            "app_sequences": {"code.exe -> chrome.exe": {"count": 5, "avg_prev_duration_ms": 10000}},
            "app_workflows": {
                "chrome.exe + code.exe + windowsterminal.exe": {
                    "count": 3,
                    "switch_count": 6,
                    "shortcut_count": 12,
                    "avg_span_ms": 12000,
                }
            },
            "raw_completion_keys": {
                "Dash and Underscore": {"count": 5},
                "PageUp": {"count": 2},
            },
            "raw_completion_total": 7,
            "by_layer_shortcut": {"Ctrl+C": {"2": 4}},
            "layer_shortcuts": {"2": {"total": 4, "shortcuts": {"Ctrl+C": 4}}},
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump(usage_stats, f)
            path = f.name
        try:
            usage = load_usage_stats(path)
        finally:
            os.unlink(path)

        self.assertIn("Ctrl+C -> Alt+Tab", usage.sequences)
        self.assertIn("Ctrl+C -> Alt+Tab -> Ctrl+V", usage.chains)
        self.assertIn("code.exe -> chrome.exe", usage.app_sequences)
        self.assertIn("chrome.exe + code.exe + windowsterminal.exe", usage.app_workflows)
        self.assertEqual(usage.raw_completion_total, 7)
        self.assertEqual(usage.raw_completion_keys["Dash and Underscore"]["count"], 5)
        self.assertEqual(usage.by_layer_shortcut["Ctrl+C"]["2"], 4)
        self.assertEqual(usage.layer_shortcuts["2"]["shortcuts"]["Ctrl+C"], 4)

    def test_non_keypress_shortcuts_are_not_loaded_as_plain_keys(self):
        canonical = {
            "physical_grid": {"positions": []},
            "layers": {},
            "_usage_stats": {"scroll_total": 4000},
        }
        app_scores = {
            "apps": [{
                "name": "Browser",
                "shortcuts": [
                    {"keys": "ScrollUp", "action": "Scroll up", "category": "navigation", "importance": 7.0},
                    {"keys": "ScrollDown", "action": "Scroll down", "category": "navigation", "importance": 7.0},
                    {"keys": "gg", "action": "Scroll to top", "category": "Vimium Extension", "importance": 3.0},
                    {"keys": "gi", "action": "Focus input", "category": "Vimium Extension", "importance": 3.0},
                    {"keys": "yy", "action": "Copy URL", "category": "Vimium Extension", "importance": 3.0},
                    {"keys": "Ctrl+K S", "action": "Save all", "category": "File Operations", "importance": 3.0},
                    {"keys": "Ctrl+S", "action": "Save", "category": "File Operations", "importance": 9.0},
                ],
            }]
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump(app_scores, f)
            path = f.name
        try:
            shortcuts = load_shortcuts(path, canonical)
        finally:
            os.unlink(path)
        keys = {s.keys for s in shortcuts}
        self.assertIn("Ctrl+S", keys)
        for key in ("ScrollUp", "ScrollDown", "gg", "gi", "yy", "Ctrl+K S"):
            self.assertNotIn(key, keys)

    def test_scroll_up_down_do_not_create_dynamic_groups(self):
        shortcuts = [
            Shortcut(0, "ScrollUp", "Scroll up", "Browser", 7.0, "navigation"),
            Shortcut(1, "ScrollDown", "Scroll down", "Browser", 7.0, "navigation"),
            Shortcut(2, "Ctrl+C", "Copy", "Browser", 9.0, "editing"),
            Shortcut(3, "Ctrl+V", "Paste", "Browser", 9.0, "editing"),
        ]
        usage = UsageData(
            sequences={
                "ScrollUp -> ScrollDown": {"count": 50},
                "Ctrl+C -> Ctrl+V": {"count": 50},
            },
            chains={
                "ScrollUp -> ScrollDown -> Ctrl+C": {"count": 50},
                "Ctrl+C -> Ctrl+V -> Ctrl+C": {"count": 50},
            },
        )
        groups = _discover_dynamic_groups(usage, shortcuts)
        grouped_keys = {
            shortcuts[sid].keys
            for group in groups
            for sid in group.get("sids", [])
        }

        self.assertNotIn("ScrollUp", grouped_keys)
        self.assertNotIn("ScrollDown", grouped_keys)
        self.assertIn("Ctrl+C", grouped_keys)

    def test_bluetooth_keys_do_not_enter_evolvable_genome(self):
        layout_data = {
            "n_layers": 8,
            "physical_grid": {
                "positions": [
                    {"x": 1, "y": 1, "hand": "left", "finger": "index", "zone": "finger", "row_type": "middle"},
                    {"x": 7, "y": 4, "hand": "right", "finger": "thumb", "zone": "thumb", "row_type": "thumb"},
                ]
            },
            "l0_frozen": {},
            "l7_frozen": {
                "1:1": {"x": 1, "y": 1, "label": "BT1", "behavior": "Bluetooth", "parameter": "BT_SEL 1", "modifiers": []},
                "7:4": {"x": 7, "y": 4, "label": "Exit Base", "behavior": "To Layer", "parameter": "Layer::0", "modifiers": []},
            },
        }
        app_scores = {
            "apps": [{
                "name": "System",
                "shortcuts": [
                    {"keys": "BT_SEL 1", "action": "Bluetooth", "category": "system", "importance": 10.0},
                    {"keys": "Ctrl+S", "action": "Save", "category": "File", "importance": 9.0},
                ],
            }]
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump(layout_data, f)
            layout_path = f.name
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump(app_scores, f)
            shortcuts_path = f.name
        try:
            positions, _, _ = load_layout(layout_path)
            shortcuts = load_shortcuts(shortcuts_path, layout_data)
            genome = build_frozen_genome(layout_data, positions, shortcuts)
        finally:
            os.unlink(layout_path)
            os.unlink(shortcuts_path)

        keys = {s.keys for s in shortcuts}
        self.assertNotIn("BT_SEL 1", keys)
        self.assertIn("Ctrl+S", keys)
        self.assertTrue(all("BT_SEL" not in s.keys for s in shortcuts))
        self.assertTrue(all(int(genome[p.gene_idx]) == -1 for p in positions if p.layer == 7))

    def test_norwegian_hid_shortcut_parsing(self):
        cases = {
            "Ctrl+Page Up": (["Ctrl"], "PageUp"),
            "Ctrl+Page Down": (["Ctrl"], "PageDown"),
            "Ctrl+-": (["Ctrl"], "Dash and Underscore"),
            "Alt+=": (["Alt"], "Equals and Plus"),
            "Ctrl+Shift+`": (["Ctrl", "Shift"], "Grave Accent and Tilde"),
            "Ctrl+]": (["Ctrl"], "Right Brace"),
            "?": ([], "ForwardSlash and QuestionMark"),
        }
        for keys, expected in cases.items():
            self.assertEqual(parse_shortcut_keys_norwegian(keys), expected)

    def test_pointer_mode_capabilities_are_appended_last(self):
        layout_data = {"n_layers": 3, "l0_frozen": {}}
        app_scores = {
            "apps": [{
                "name": "Browser",
                "shortcuts": [
                    {"keys": "Ctrl+J", "action": "Downloads", "importance": 5.0},
                ],
            }]
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump(app_scores, f)
            path = f.name
        try:
            shortcuts = load_shortcuts(path, layout_data)
        finally:
            os.unlink(path)

        expected = {
            "@ptr:snipe_hold": "SnipeHold",
            "@ptr:fast_hold": "FastHold",
            "@ptr:snipe_mode": "SnipeMode",
            "@ptr:normal_mode": "NormalMode",
            "@ptr:fast_mode": "FastMode",
        }
        by_key = {s.keys: s for s in shortcuts}
        for keys, base_key in expected.items():
            self.assertIn(keys, by_key)
            sc = by_key[keys]
            self.assertEqual(sc.base_key, base_key)
            self.assertEqual(sc.category, "pointer_mode")
            self.assertEqual(sc.preferred_hand, "right")
            self.assertEqual(sc.importance, 3.0)
            self.assertTrue(sc.is_capability)
            self.assertFalse(sc.is_layer_access)
            self.assertNotIn("scroll", (sc.keys + sc.action + sc.base_key).lower())
        # Appended at the very end, in declaration order, so every pre-existing
        # sid keeps its index (warmstart safety).
        self.assertEqual([s.keys for s in shortcuts[-5:]], list(expected.keys()))
        self.assertEqual([s.sid for s in shortcuts], list(range(len(shortcuts))))


if __name__ == "__main__":
    unittest.main()
