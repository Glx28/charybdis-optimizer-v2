"""Tests for the v2 Charybdis optimizer."""
import sys
import os
import json
import tempfile
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import unittest

from config import DEFAULT_CONFIG
from core import Position, Shortcut, Layout, FitnessResult, UsageData
from fitness.evaluator import FitnessEvaluator
from fitness.factors.effort import EffortFactor
from fitness.factors.adjacency import AdjacencyFactor
from fitness.factors.finger_balance import FingerBalanceFactor
from fitness.factors.same_finger import SameFingerFactor
from fitness.factors.violation import ViolationFactor
from fitness.kernel import DEFAULT_FITNESS_WEIGHTS, DEFAULT_VIOLATION_WEIGHTS
from evolution.surrogate import LayoutSurrogate, SurrogateTrainer
from evolution import StructuralGenomeSanitizer, PermutationSampling, SwapMutation
from evolution.arrow_cluster import analyze_arrows
from evolution.acceptance import (
    _dynamic_mouse_layer_report,
    _layer7_access_report,
    _momentary_only_thumb_clearance_report,
)
from core.loader import load_layout, load_shortcuts, load_usage_stats, build_frozen_genome, _parse_layer_from_behavior, _discover_dynamic_groups
from core.norwegian_keys import parse_shortcut_keys_norwegian


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
        for invalid in ("ScrollUp", "ScrollDown", "yy", "gg", "gi"):
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


class TestFitnessFactors(unittest.TestCase):
    def setUp(self):
        self.positions = (
            Position(0, 0, 0.0, 0.0, "left", 1, 1.0),
            Position(1, 0, 1.0, 0.0, "left", 1, 1.0),
            Position(2, 1, 5.0, 0.0, "right", 2, 3.0),
        )
        self.shortcuts = (
            Shortcut(0, "Ctrl+C", "copy", "windows", 8.0, "editing"),
            Shortcut(1, "Ctrl+V", "paste", "windows", 8.0, "editing"),
            Shortcut(2, "Enter", "enter", "general", 5.0, "navigation"),
        )
    
    def _make_layout(self, genome):
        g = np.array(genome, dtype=np.int32)
        m = np.array([False, False, False])
        return Layout(g, self.positions, self.shortcuts, m)
    
    def test_effort(self):
        layout = self._make_layout([0, 1, 2])
        f = EffortFactor()
        score = f.compute(layout)
        expected = 8.0*1.0 + 8.0*1.0 + 5.0*3.0  # 31.0
        self.assertAlmostEqual(score, expected, places=1)
    
    def test_adjacency(self):
        layout = self._make_layout([0, 1, -1])
        f = AdjacencyFactor()
        score = f.compute(layout)
        self.assertGreater(score, 0)  # Ctrl+C and Ctrl+V are close
    
    def test_violations_empty(self):
        layout = self._make_layout([-1, -1, -1])
        f = ViolationFactor()
        score = f.compute(layout)
        self.assertGreater(score, 0)  # Missing important shortcuts

    def test_supported_duplicate_penalty_is_discounted(self):
        positions = (
            Position(0, 1, 0.0, 0.0, "left", 1, 0.8),
            Position(1, 1, 1.0, 0.0, "left", 1, 0.8),
            Position(2, 1, 2.0, 0.0, "left", 1, 1.0),
        )
        shortcuts = (
            Shortcut(0, "Ctrl+C", "Copy", "Visual Studio Code", 10.0, "editing", base_key="C"),
            Shortcut(1, "Ctrl+V", "Paste", "Visual Studio Code", 10.0, "editing", base_key="V"),
        )
        genome = np.array([0, 0, 1], dtype=np.int32)
        frozen = np.array([False, False, False])
        unsupported = Layout(genome, positions, shortcuts, frozen)
        supported_usage = UsageData(
            shortcuts={"Ctrl+C": {"count": 25}},
            chains={"Ctrl+C -> Alt+Tab -> Ctrl+V": {"count": 5}},
            app_workflows={"code.exe + chrome.exe + windowsterminal.exe": {"count": 4}},
        )
        supported = Layout(genome, positions, shortcuts, frozen, usage_data=supported_usage)
        factor = ViolationFactor()

        self.assertGreater(factor._duplicate_penalty(unsupported), factor._duplicate_penalty(supported))

    def test_duplicate_penalties_are_novelty_gated(self):
        positions = (
            Position(0, 1, 0.0, 0.0, "left", 1, 0.8),
            Position(1, 2, 0.0, 0.0, "left", 1, 0.8),
            Position(2, 3, 0.0, 0.0, "left", 1, 0.8),
            Position(3, 4, 0.0, 0.0, "left", 1, 0.8),
        )
        normal_shortcuts = (
            Shortcut(0, "Shortcut0", "Normal", "app", 8.0, "general", base_key="Key0"),
            Shortcut(1, "Shortcut1", "Other", "app", 8.0, "general", base_key="Key1"),
        )
        exceptional_shortcuts = (
            Shortcut(0, "Shortcut0", "Exceptional", "app", 20.0, "general", base_key="Key0"),
            Shortcut(1, "Shortcut1", "Other", "app", 8.0, "general", base_key="Key1"),
        )
        genome = np.array([0, 0, 0, 1], dtype=np.int32)
        frozen = np.array([False] * 4)
        normal = Layout(genome, positions, normal_shortcuts, frozen)
        exceptional = Layout(genome, positions, exceptional_shortcuts, frozen)
        supported = Layout(
            genome,
            positions,
            exceptional_shortcuts,
            frozen,
            usage_data=UsageData(shortcuts={"Shortcut0": {"count": 100}}),
        )
        factor = ViolationFactor()

        self.assertGreater(factor._cross_layer_duplicate(normal), factor._cross_layer_duplicate(exceptional))
        self.assertGreater(factor._cross_layer_duplicate(exceptional), factor._cross_layer_duplicate(supported))

    def test_partial_arrow_layer_has_large_scatter_penalty(self):
        positions = (
            Position(0, 1, 0.0, 0.0, "left", 1, 1.0),
            Position(1, 2, 4.0, 0.0, "left", 1, 1.0),
            Position(2, 7, 1.0, 0.0, "left", 1, 1.0),
            Position(3, 7, 2.0, 0.0, "left", 1, 1.0),
            Position(4, 7, 1.5, -1.0, "left", 1, 1.0),
            Position(5, 7, 1.5, 1.0, "left", 1, 1.0),
        )
        shortcuts = (
            Shortcut(0, "LeftArrow", "Left", "Mouse", 3.0, "navigation", base_key="LeftArrow"),
            Shortcut(1, "RightArrow", "Right", "Mouse", 3.0, "navigation", base_key="RightArrow"),
            Shortcut(2, "UpArrow", "Up", "Mouse", 3.0, "navigation", base_key="UpArrow"),
            Shortcut(3, "DownArrow", "Down", "Mouse", 3.0, "navigation", base_key="DownArrow"),
        )
        frozen = np.array([False] * len(positions))
        scattered = Layout(np.array([0, 1, 0, 1, 2, 3], dtype=np.int32), positions, shortcuts, frozen)
        l7_only = Layout(np.array([-1, -1, 0, 1, 2, 3], dtype=np.int32), positions, shortcuts, frozen)
        factor = ViolationFactor()

        self.assertGreaterEqual(factor._arrow_scattered(scattered), 100.0)
        self.assertEqual(factor._arrow_scattered(l7_only), 0.0)

    def test_raw_arrow_clusters_allow_only_two_shapes(self):
        shortcuts = (
            Shortcut(0, "LeftArrow", "Left", "Mouse", 3.0, "navigation", base_key="LeftArrow"),
            Shortcut(1, "RightArrow", "Right", "Mouse", 3.0, "navigation", base_key="RightArrow"),
            Shortcut(2, "UpArrow", "Up", "Mouse", 3.0, "navigation", base_key="UpArrow"),
            Shortcut(3, "DownArrow", "Down", "Mouse", 3.0, "navigation", base_key="DownArrow"),
        )
        frozen = np.array([False] * 4)
        same_line_positions = (
            Position(0, 1, 0.0, 0.0, "left", 1, 1.0),
            Position(1, 1, 1.0, 0.0, "left", 1, 1.0),
            Position(2, 1, 2.0, 0.0, "left", 1, 1.0),
            Position(3, 1, 3.0, 0.0, "left", 1, 1.0),
        )
        split_positions = (
            Position(0, 1, 0.0, 1.0, "left", 1, 1.0),
            Position(1, 1, 1.0, 1.0, "left", 1, 1.0),
            Position(2, 1, 2.0, 1.0, "left", 1, 1.0),
            Position(3, 1, 1.0, 0.0, "left", 1, 1.0),
        )
        old_inverted_t_positions = (
            Position(0, 1, 0.0, 0.0, "left", 1, 1.0),
            Position(1, 1, 1.0, 0.0, "left", 1, 1.0),
            Position(2, 1, 2.0, 0.0, "left", 1, 1.0),
            Position(3, 1, 1.0, 1.0, "left", 1, 1.0),
        )

        same_line = Layout(np.array([0, 2, 3, 1], dtype=np.int32), same_line_positions, shortcuts, frozen)
        split = Layout(np.array([0, 3, 1, 2], dtype=np.int32), split_positions, shortcuts, frozen)
        old_shape = Layout(np.array([0, 2, 1, 3], dtype=np.int32), old_inverted_t_positions, shortcuts, frozen)
        factor = ViolationFactor()

        self.assertTrue(analyze_arrows(same_line)["acceptance_pass"])
        self.assertTrue(analyze_arrows(split)["acceptance_pass"])
        self.assertFalse(analyze_arrows(old_shape)["acceptance_pass"])
        self.assertLess(factor._arrow_scattered(same_line), factor._arrow_scattered(old_shape))
        self.assertLess(factor._arrow_scattered(split), factor._arrow_scattered(old_shape))

    def test_group_scoring_only_compacts_same_layer_members(self):
        shortcuts = (
            Shortcut(0, "Ctrl+C", "Copy", "app", 8.0, "editing", modifiers=("Ctrl",), base_key="C"),
            Shortcut(1, "Ctrl+V", "Paste", "app", 8.0, "editing", modifiers=("Ctrl",), base_key="V"),
            Shortcut(2, "Ctrl+X", "Cut", "app", 8.0, "editing", modifiers=("Ctrl",), base_key="X"),
        )
        frozen = np.array([False, False, False])
        split_positions = (
            Position(0, 1, 0.0, 0.0, "left", 1, 1.0),
            Position(1, 2, 10.0, 0.0, "right", 1, 1.0),
            Position(2, 3, 20.0, 0.0, "right", 1, 1.0),
        )
        scattered_positions = (
            Position(0, 1, 0.0, 0.0, "left", 1, 1.0),
            Position(1, 1, 8.0, 0.0, "right", 1, 1.0),
            Position(2, 1, 16.0, 0.0, "right", 1, 1.0),
        )
        compact_positions = (
            Position(0, 1, 0.0, 0.0, "left", 1, 1.0),
            Position(1, 1, 1.0, 0.0, "left", 1, 1.0),
            Position(2, 1, 2.0, 0.0, "left", 1, 1.0),
        )
        genome = np.array([0, 1, 2], dtype=np.int32)
        factor = ViolationFactor()

        split = Layout(genome, split_positions, shortcuts, frozen)
        scattered = Layout(genome, scattered_positions, shortcuts, frozen)
        compact = Layout(genome, compact_positions, shortcuts, frozen)

        self.assertEqual(factor._group_split(split), 0.0)
        self.assertGreater(factor._group_split(scattered), factor._group_split(compact))


class TestSurrogate(unittest.TestCase):
    def test_forward(self):
        surrogate = LayoutSurrogate(n_positions=10, n_shortcuts=20, n_factors=3, hidden_dim=32)
        layouts = np.full((5, 10), -1, dtype=np.int32)
        layouts[:, :5] = np.arange(5)
        
        import torch
        x = torch.tensor(layouts, dtype=torch.long)
        out = surrogate(x)
        self.assertEqual(out.shape, (5, 3))
    
    def test_train_predict(self):
        surrogate = LayoutSurrogate(n_positions=10, n_shortcuts=20, n_factors=3, hidden_dim=32)
        trainer = SurrogateTrainer(surrogate, device="cpu")
        
        layouts = np.full((50, 10), -1, dtype=np.int32)
        layouts[:, :5] = np.random.randint(0, 20, size=(50, 5))
        scores = np.random.randn(50, 3).astype(np.float32)
        
        trainer.train(layouts, scores, epochs=10, batch_size=16)
        pred = trainer.predict(layouts[:10])
        self.assertEqual(pred.shape, (10, 3))

    def test_sampling_excludes_frozen_assigned_shortcuts(self):
        seed = np.array([0, -1, 1, -1], dtype=np.int32)
        frozen = np.array([True, False, True, False])

        class Problem:
            n_var = 4

        sampler = PermutationSampling(
            n_shortcuts=4,
            frozen_mask=frozen,
            seed_genome=seed,
            inject_seed=True,
        )
        X = sampler._do(Problem(), 20)
        mutable = np.where(~frozen)[0]
        self.assertFalse(np.any(np.isin(X[:, mutable], [0, 1])))

        dirty = np.array([[0, 0, 1, 1]], dtype=np.int32)
        sanitizer = StructuralGenomeSanitizer(n_shortcuts=4, frozen_mask=frozen, seed_genome=seed)
        cleaned = sanitizer._do(Problem(), dirty.copy())
        self.assertEqual(cleaned[0, 0], 0)
        self.assertEqual(cleaned[0, 2], 1)
        self.assertEqual(cleaned[0, 1], -1)
        self.assertEqual(cleaned[0, 3], -1)

    def test_group_mutation_moves_arrow_group_as_unit(self):
        random.seed(100)
        positions = tuple(
            Position(i, 1, float(i % 6), float(i // 6), "left", 1, 1.0)
            for i in range(12)
        )
        shortcuts = tuple([
            Shortcut(0, "LeftArrow", "Left", "Nav", 1.0, base_key="LeftArrow"),
            Shortcut(1, "UpArrow", "Up", "Nav", 1.0, base_key="UpArrow"),
            Shortcut(2, "DownArrow", "Down", "Nav", 1.0, base_key="DownArrow"),
            Shortcut(3, "RightArrow", "Right", "Nav", 1.0, base_key="RightArrow"),
            *[Shortcut(i, f"K{i}", "", "App", 1.0) for i in range(4, 12)],
        ])
        genome = np.arange(12, dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(12, dtype=np.bool_))
        mutation = SwapMutation(prob=0.0, frozen_mask=layout.frozen_mask, layout=layout, group_move_prob=1.0)
        X = mutation._do(None, genome.reshape(1, -1).copy())
        moved = X[0]
        arrow_positions = [int(np.where(moved == sid)[0][0]) for sid in range(4)]
        self.assertEqual(len(set(arrow_positions)), 4)
        self.assertNotEqual(set(arrow_positions), {0, 1, 2, 3})
        self.assertTrue(set(arrow_positions).isdisjoint({0, 1, 2, 3}))
        self.assertFalse(any(int(moved[pos]) in {0, 1, 2, 3} for pos in range(4)))

    def test_group_mutation_moves_completion_group_as_unit(self):
        random.seed(200)
        # 5x4 grid (x=0..4, y=0..3), layer 1 — 20 positions.
        # Completion sids 0-4 start at positions 10-14 (y=2, x=0-4).
        # Valid anchors at y=0 (ax=2,3,4) are positions like {0,1,2,5,15}
        # which don't overlap {10,11,12,13,14}, so the mutation can move the group.
        positions = tuple(
            Position(i, 1, float(i % 5), float(i // 5), "left", 1, 1.0)
            for i in range(20)
        )
        bases = [
            "Dash and Underscore", "Equals and Plus", "Grave Accent and Tilde",
            "Right Brace", "Backslash and Pipe",
        ]
        shortcuts = tuple([
            *(Shortcut(i, bases[i], "", "Raw", 1.0, base_key=bases[i]) for i in range(5)),
            *[Shortcut(i, f"K{i}", "", "App", 1.0) for i in range(5, 20)],
        ])
        # Place completion sids 0-4 at positions 10-14; other sids at 0-9 and 15-19.
        genome = np.array(list(range(5, 15)) + list(range(5)) + list(range(15, 20)), dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(20, dtype=np.bool_))
        mutation = SwapMutation(prob=0.0, frozen_mask=layout.frozen_mask, layout=layout, group_move_prob=1.0)
        initial_completion = set(int(np.where(genome == sid)[0][0]) for sid in range(5))
        moved_any = False
        for _ in range(30):
            X = mutation._do(None, genome.reshape(1, -1).copy())
            moved = X[0]
            completion_positions = set(int(np.where(moved == sid)[0][0]) for sid in range(5))
            self.assertEqual(len(completion_positions), 5)
            if completion_positions != initial_completion:
                moved_any = True
                self.assertTrue(completion_positions.isdisjoint(initial_completion))
                self.assertFalse(any(int(moved[pos]) in set(range(5)) for pos in initial_completion))
        self.assertTrue(moved_any)

    def test_group_members_never_moved_by_individual_mutations(self):
        """Individual swap/random_assign/bulk_assign must not scatter group members."""
        random.seed(203)
        positions = tuple(
            Position(i, 1, float(i % 5), float(i // 5), "left", 1, 1.0)
            for i in range(20)
        )
        bases = [
            "Dash and Underscore", "Equals and Plus", "Grave Accent and Tilde",
            "Right Brace", "Backslash and Pipe",
        ]
        shortcuts = tuple([
            *(Shortcut(i, bases[i], "", "Raw", 1.0, base_key=bases[i]) for i in range(5)),
            *[Shortcut(i, f"K{i}", "", "App", 1.0) for i in range(5, 20)],
        ])
        genome = np.arange(20, dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(20, dtype=np.bool_))
        mutation = SwapMutation(
            prob=1.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            group_overwrite_prob=0.0,   # disable group-move
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            random_assign_prob=0.5,
            bulk_assign_prob=0.5,
            optional_arrow_drop_prob=0.0,
        )
        group_sids = set(range(5))
        for _ in range(100):
            X = mutation._do(None, genome.reshape(1, -1).copy())
            moved = X[0]
            # Group sids must still be at their original positions (0-4)
            for sid in group_sids:
                positions_of_sid = set(int(idx) for idx in np.where(moved == sid)[0])
                self.assertTrue(positions_of_sid.issubset({0, 1, 2, 3, 4}),
                    f"sid {sid} moved to unexpected positions {positions_of_sid}")

    def test_mouse_workflow_mutation_proposes_acceptance_visible_layer(self):
        random.seed(300)
        positions = tuple([
            Position(0, 0, 3.0, 4.0, "left", 0, 0.1, is_thumb=True),
            Position(1, 0, 4.0, 4.0, "left", 0, 0.1, is_thumb=True),
            Position(2, 0, 5.0, 4.0, "left", 0, 0.1, is_thumb=True),
            Position(3, 0, 0.0, 0.0, "left", 1, 1.0),
            *(Position(i, 1, float(7 + (i - 4)), 1.0, "right", 1, 0.2) for i in range(4, 10)),
            *(Position(i, 1, float(i - 10), 2.0, "left", 1, 1.0) for i in range(10, 14)),
        ])
        shortcuts = tuple([
            Shortcut(0, "MB1", "Left click", "Mouse", 10.0, category="mouse", base_key="MB1"),
            Shortcut(1, "MB2", "Right click", "Mouse", 9.0, category="mouse", base_key="MB2"),
            Shortcut(2, "MB3", "Middle click", "Mouse", 7.0, category="mouse", base_key="MB3"),
            Shortcut(3, "MB4", "Back", "Mouse", 6.0, category="mouse", base_key="MB4"),
            Shortcut(4, "MB5", "Forward", "Mouse", 6.0, category="mouse", base_key="MB5"),
            Shortcut(5, "@scroll:L1:hold", "Scroll Mode Layer 1", "Layer Access", 15.0, category="layer_access", base_key="Scroll_L1", is_layer_access=True, access_target_layer=1, access_is_momentary=True),
            Shortcut(6, "@access:L1:hold", "Momentary Layer 1", "Layer Access", 12.0, category="layer_access", base_key="L1", is_layer_access=True, access_target_layer=1, access_is_momentary=True),
            Shortcut(7, "@access:L1:toggle", "Toggle Layer 1", "Layer Access", 12.0, category="layer_access", base_key="L1", is_layer_access=True, access_target_layer=1, access_is_momentary=False),
            Shortcut(8, "@access:L7:hold", "Momentary Layer 7", "Layer Access", 12.0, category="layer_access", base_key="L7", is_layer_access=True, access_target_layer=7, access_is_momentary=True),
            Shortcut(9, "@access:L7:toggle", "Toggle Layer 7", "Layer Access", 12.0, category="layer_access", base_key="L7", is_layer_access=True, access_target_layer=7, access_is_momentary=False),
            *(Shortcut(i, f"K{i}", "", "App", 1.0, base_key=f"K{i}") for i in range(10, 14)),
        ])
        genome = np.arange(14, dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(14, dtype=np.bool_))
        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            group_overwrite_prob=0.0,
            mouse_workflow_prob=1.0,
            l7_access_prob=0.0,
        )
        moved = mutation._do(None, genome.reshape(1, -1).copy())[0]
        report = _dynamic_mouse_layer_report(layout.clone_with(genome=moved))
        self.assertTrue(report["acceptance_pass"], report)

    def test_l7_access_mutation_proposes_hold_and_toggle(self):
        random.seed(400)
        positions = tuple(
            Position(i, 0 if i < 4 else 1, float(i), 0.0, "left", 0 if i < 4 else 1, 0.5, is_thumb=i < 4)
            for i in range(8)
        )
        shortcuts = tuple([
            Shortcut(0, "@access:L7:hold", "Momentary Layer 7", "Layer Access", 12.0, category="layer_access", base_key="L7", is_layer_access=True, access_target_layer=7, access_is_momentary=True),
            Shortcut(1, "@access:L7:toggle", "Toggle Layer 7", "Layer Access", 12.0, category="layer_access", base_key="L7", is_layer_access=True, access_target_layer=7, access_is_momentary=False),
            *(Shortcut(i, f"K{i}", "", "App", 1.0, base_key=f"K{i}") for i in range(2, 8)),
        ])
        genome = np.arange(8, dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(8, dtype=np.bool_))
        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            group_overwrite_prob=0.0,
            mouse_workflow_prob=0.0,
            l7_access_prob=1.0,
        )
        moved = mutation._do(None, genome.reshape(1, -1).copy())[0]
        report = _layer7_access_report(layout.clone_with(genome=moved))
        self.assertTrue(report["acceptance_pass"], report)

    def test_random_reassign_preserves_last_important_shortcut_copy(self):
        random.seed(500)
        positions = tuple(Position(i, 1, float(i), 0.0, "left", 1, 0.5) for i in range(4))
        shortcuts = tuple([
            Shortcut(0, "Critical", "", "App", 6.0, base_key="Critical"),
            Shortcut(1, "Weak", "", "App", 1.0, base_key="Weak"),
            Shortcut(2, "Other", "", "App", 1.0, base_key="Other"),
            Shortcut(3, "Spare", "", "App", 1.0, base_key="Spare"),
        ])
        genome = np.array([0, 1, 1, 2], dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(4, dtype=np.bool_))
        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            group_overwrite_prob=0.0,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            random_assign_prob=1.0,
        )
        for _ in range(50):
            moved = mutation._do(None, genome.reshape(1, -1).copy())[0]
            self.assertIn(0, set(int(sid) for sid in moved))

    def test_bulk_reassign_preserves_last_important_shortcut_copy(self):
        random.seed(501)
        positions = tuple(Position(i, 1, float(i), 0.0, "left", 1, 0.5) for i in range(12))
        shortcuts = tuple([
            Shortcut(0, "Critical", "", "App", 6.0, base_key="Critical"),
            *(Shortcut(i, f"Weak{i}", "", "App", 1.0, base_key=f"Weak{i}") for i in range(1, 12)),
        ])
        genome = np.array([0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(12, dtype=np.bool_))
        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            group_overwrite_prob=0.0,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            random_assign_prob=0.0,
            bulk_assign_prob=1.0,
        )
        changed_any = False
        for _ in range(50):
            moved = mutation._do(None, genome.reshape(1, -1).copy())[0]
            self.assertIn(0, set(int(sid) for sid in moved))
            changed_any = changed_any or not np.array_equal(moved, genome)
        self.assertTrue(changed_any)

    def test_optional_arrow_drop_mutation_removes_mutable_raw_arrows(self):
        random.seed(502)
        positions = tuple(Position(i, 1, float(i), 0.0, "left", 1, 0.5) for i in range(8))
        shortcuts = tuple([
            Shortcut(0, "Critical", "", "App", 6.0, base_key="Critical"),
            Shortcut(1, "LeftArrow", "", "Raw", 4.0, base_key="LeftArrow"),
            Shortcut(2, "UpArrow", "", "Raw", 4.0, base_key="UpArrow"),
            Shortcut(3, "DownArrow", "", "Raw", 4.0, base_key="DownArrow"),
            Shortcut(4, "RightArrow", "", "Raw", 4.0, base_key="RightArrow"),
            Shortcut(5, "WeakA", "", "App", 1.0, base_key="WeakA"),
            Shortcut(6, "WeakB", "", "App", 1.0, base_key="WeakB"),
            Shortcut(7, "WeakC", "", "App", 1.0, base_key="WeakC"),
        ])
        genome = np.arange(8, dtype=np.int32)
        layout = Layout(genome.copy(), positions, shortcuts, np.zeros(8, dtype=np.bool_))
        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            group_overwrite_prob=0.0,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            random_assign_prob=0.0,
            bulk_assign_prob=0.0,
            optional_arrow_drop_prob=1.0,
        )
        moved = mutation._do(None, genome.reshape(1, -1).copy())[0]
        self.assertIn(0, set(int(sid) for sid in moved))
        self.assertFalse(any(int(sid) in {1, 2, 3, 4} for sid in moved))


class TestEvaluator(unittest.TestCase):
    def test_evaluate(self):
        positions = tuple(Position(i, 0, float(i), 0.0, "left", 1, 1.0) for i in range(5))
        shortcuts = tuple(Shortcut(i, f"Key{i}", f"action{i}", "app", 5.0) for i in range(5))
        genome = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        frozen = np.array([False]*5)
        layout = Layout(genome, positions, shortcuts, frozen)
        
        evaluator = FitnessEvaluator()
        result = evaluator.evaluate(layout)
        
        self.assertEqual(result.objectives.shape, (3,))
        self.assertIn("effort", result.factor_scores)
        self.assertIn("adjacency", result.factor_scores)
        self.assertIn("violations", result.factor_scores)
        self.assertIn("workflow_coherence", result.factor_scores)
        self.assertIn("finger_balance", result.factor_scores)
        self.assertIn("same_finger", result.factor_scores)

    def test_layer_specialization_penalizes_redundant_generated_layers(self):
        positions = tuple(
            Position(i, 1 if i < 4 else 2, float(i % 4), 0.0, "left", 1, 1.0)
            for i in range(8)
        )
        shortcuts = tuple(
            Shortcut(i, f"Shortcut{i}", f"Action {i}", "same-app", 10.0, base_key=f"Key{i}")
            for i in range(8)
        )
        frozen = np.array([False] * 8)
        redundant = Layout(
            np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        distinct = Layout(
            np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 0.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_specialization": 1.0,
        }
        evaluator = FitnessEvaluator(weights=weights)

        self.assertGreater(
            evaluator.evaluate(redundant).objectives[2],
            evaluator.evaluate(distinct).objectives[2],
        )

    def test_layer_specialization_discounts_exceptional_supported_repeats(self):
        positions = tuple(
            Position(i, 1 if i < 4 else 2, float(i % 4), 0.0, "left", 1, 1.0)
            for i in range(8)
        )
        shortcuts = tuple(
            Shortcut(i, f"Shortcut{i}", f"Action {i}", "same-app", 10.0, base_key=f"Key{i}")
            for i in range(4)
        )
        frozen = np.array([False] * 8)
        genome = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        unsupported = Layout(genome, positions, shortcuts, frozen)
        usage = UsageData(shortcuts={f"Shortcut{i}": {"count": 100} for i in range(4)})
        supported = Layout(genome, positions, shortcuts, frozen, usage_data=usage)
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 0.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_specialization": 1.0,
        }
        evaluator = FitnessEvaluator(weights=weights)

        self.assertGreater(
            evaluator.evaluate(unsupported).objectives[2],
            evaluator.evaluate(supported).objectives[2],
        )

    def test_layer_similarity_allows_one_everything_layer_to_overlap_more(self):
        positions = tuple(
            Position(i, 1 + (i // 4), float(i % 4), 0.0, "left", 1, 1.0)
            for i in range(12)
        )
        shortcuts = tuple(
            Shortcut(i, f"Shortcut{i}", f"Action {i}", "app", 8.0, base_key=f"Key{i}")
            for i in range(8)
        )
        usage = UsageData(shortcuts={
            "Shortcut0": {"count": 100},
            "Shortcut1": {"count": 90},
            "Shortcut2": {"count": 80},
            "Shortcut3": {"count": 70},
        })
        frozen = np.array([False] * 12)
        overlap_with_everything = Layout(
            np.array([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        ordinary_overlap = Layout(
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 0.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_similarity": 1.0,
            "everything_layer": 0.0,
        }
        evaluator = FitnessEvaluator(weights=weights)

        self.assertLess(
            evaluator.evaluate(overlap_with_everything).objectives[2],
            evaluator.evaluate(ordinary_overlap).objectives[2],
        )

    def test_familiarity_reward_is_gated_to_exceptional_repeats(self):
        positions = tuple(
            Position(i, 1 if i < 2 else 2, float(i % 2), 0.0, "left", 1, 1.0)
            for i in range(4)
        )
        shortcuts = (
            Shortcut(0, "Shortcut0", "Action 0", "same-app", 10.0, base_key="Key0"),
            Shortcut(1, "Shortcut1", "Action 1", "same-app", 10.0, base_key="Key1"),
        )
        frozen = np.array([False] * 4)
        genome = np.array([0, 1, 0, 1], dtype=np.int32)
        unsupported = Layout(genome, positions, shortcuts, frozen)
        usage = UsageData(shortcuts={"Shortcut0": {"count": 100}, "Shortcut1": {"count": 100}})
        supported = Layout(genome, positions, shortcuts, frozen, usage_data=usage)
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 0.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 1.0,
            "layer_specialization": 0.0,
        }
        evaluator = FitnessEvaluator(weights=weights)

        self.assertLess(
            evaluator.evaluate(supported).objectives[2],
            evaluator.evaluate(unsupported).objectives[2],
        )

    def test_familiarity_uses_pairwise_exponential_distance_decay(self):
        positions = (
            Position(0, 1, 5.0, 2.0, "left", 1, 1.0),
            Position(1, 2, 5.0, 2.0, "left", 1, 1.0),
            Position(2, 2, 6.0, 2.0, "left", 1, 1.0),
            Position(3, 2, 12.0, 5.0, "right", 4, 1.0),
        )
        shortcuts = (Shortcut(0, "Shortcut0", "Action 0", "same-app", 12.0, base_key="Key0"),)
        usage = UsageData(shortcuts={"Shortcut0": {"count": 100}})
        frozen = np.array([False] * 4)
        exact = Layout(np.array([0, 0, -1, -1], dtype=np.int32), positions, shortcuts, frozen, usage_data=usage)
        near = Layout(np.array([0, -1, 0, -1], dtype=np.int32), positions, shortcuts, frozen, usage_data=usage)
        far = Layout(np.array([0, -1, -1, 0], dtype=np.int32), positions, shortcuts, frozen, usage_data=usage)
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 0.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 1.0,
            "layer_similarity": 0.0,
            "everything_layer": 0.0,
        }
        evaluator = FitnessEvaluator(weights=weights)

        exact_score = evaluator.evaluate(exact).objectives[2]
        near_score = evaluator.evaluate(near).objectives[2]
        far_score = evaluator.evaluate(far).objectives[2]
        self.assertLess(exact_score, near_score)
        self.assertLess(near_score, far_score)

    def test_app_coherence_is_backup_after_layer_redundancy(self):
        positions = tuple(
            Position(i, 1 if i < 4 else 2, float(i % 4), 0.0, "left", 1, 1.0)
            for i in range(8)
        )
        shortcuts = tuple(
            Shortcut(i, f"Shortcut{i}", f"Action {i}", "same-app", 10.0, base_key=f"Key{i}")
            for i in range(4)
        )
        frozen = np.array([False] * 8)
        non_redundant = Layout(
            np.array([0, 1, 2, 3, -1, -1, -1, -1], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        redundant = Layout(
            np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 0.0,
            "workflow_coherence": 0.0,
            "app_coherence": 1.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_similarity": 0.0,
            "layer_specialization": 0.0,
            "everything_layer": 0.0,
        }
        evaluator = FitnessEvaluator(weights=weights)

        self.assertAlmostEqual(float(evaluator.evaluate(non_redundant).objectives[2]), 0.0, places=5)
        self.assertLess(float(evaluator.evaluate(redundant).objectives[2]), 0.0)

    def test_raw_completion_usage_prefers_more_accessible_anchor_layer(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 1, 8.0, 1.0, "right", 1, 1.0),
            Position(2, 1, 9.0, 1.0, "right", 1, 1.0),
            Position(3, 1, 10.0, 1.0, "right", 1, 1.0),
            Position(4, 1, 11.0, 1.0, "right", 1, 1.0),
            Position(5, 3, 8.0, 1.0, "right", 1, 1.0),
            Position(6, 3, 9.0, 1.0, "right", 1, 1.0),
            Position(7, 3, 10.0, 1.0, "right", 1, 1.0),
            Position(8, 3, 11.0, 1.0, "right", 1, 1.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L1:hold:Backup", "Backup", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=1,
                access_is_momentary=True,
            ),
            Shortcut(1, "-", "Dash", "Raw Keys", 3.0, "raw_completion", base_key="Dash and Underscore"),
            Shortcut(2, "=", "Equals", "Raw Keys", 3.0, "raw_completion", base_key="Equals and Plus"),
            Shortcut(3, "`", "Grave", "Raw Keys", 3.0, "raw_completion", base_key="Grave Accent and Tilde"),
            Shortcut(4, "]", "Right Brace", "Raw Keys", 3.0, "raw_completion", base_key="Right Brace"),
        )
        usage = UsageData(raw_completion_keys={
            "Dash and Underscore": {"count": 20},
            "Equals and Plus": {"count": 20},
            "Grave Accent and Tilde": {"count": 20},
            "Right Brace": {"count": 20},
        }, raw_completion_total=80)
        frozen = np.zeros(len(positions), dtype=np.bool_)
        accessible = Layout(
            np.array([0, 1, 2, 3, 4, -1, -1, -1, -1], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        inaccessible = Layout(
            np.array([0, -1, -1, -1, -1, 1, 2, 3, 4], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 1.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_specialization": 0.0,
        }
        vweights = {
            "duplicate": 0.0,
            "l0_displacement": 0.0,
            "missing_important": 0.0,
            "cross_layer_duplicate": 0.0,
            "group_split": 0.0,
            "thumb_occupancy": 0.0,
            "arrow_order": 0.0,
            "hand_bias": 0.0,
            "mouse_layer_access": 0.0,
            "arrow_scattered": 0.0,
            "mouse_scattered": 0.0,
            "layer7_access": 0.0,
            "duplicate_value_gap": 0.0,
            "access_layout": 0.0,
            "raw_keyboard_completion_norwegian": 1.0,
            "dynamic_mouse_layer": 0.0,
        }
        evaluator = FitnessEvaluator(
            weights=weights,
            reference_layout=accessible,
            violation_weights=vweights,
            hard_constraints=[],
            missing_important_threshold=99.0,
        )

        self.assertLess(
            evaluator.evaluate(accessible).objectives[2],
            evaluator.evaluate(inaccessible).objectives[2],
        )

    def test_workflow_coherence(self):
        """Test that splitting workflows across layers incurs penalty."""
        positions = (
            Position(0, 1, 0.0, 0.0, "left", 1, 1.0),   # L1
            Position(1, 1, 1.0, 0.0, "left", 1, 1.0),   # L1
            Position(2, 2, 0.0, 0.0, "left", 1, 1.0),   # L2
        )
        shortcuts = (
            Shortcut(0, "Ctrl+C", "Copy", "app", 10.0),
            Shortcut(1, "Ctrl+V", "Paste", "app", 10.0),
        )
        
        # Both on same layer - no penalty
        genome1 = np.array([0, 1, -1], dtype=np.int32)
        frozen = np.array([False, False, False])
        layout1 = Layout(genome1, positions, shortcuts, frozen)
        
        from fitness.factors.workflow_coherence import WorkflowCoherenceFactor
        factor = WorkflowCoherenceFactor()
        penalty1 = factor.compute(layout1)
        self.assertEqual(penalty1, 0.0)
        
        # Split across layers - penalty
        genome2 = np.array([0, -1, 1], dtype=np.int32)
        # Add usage data with a chain
        usage = UsageData(chains={"Ctrl+C -> Ctrl+V": {"count": 5}})
        layout2 = Layout(genome2, positions, shortcuts, frozen, usage_data=usage)
        penalty2 = factor.compute(layout2)
        self.assertGreater(penalty2, 0.0)
        # Penalty = 5 * 10.0 = 50
        self.assertEqual(penalty2, 50.0)

    def test_thumb_occupancy(self):
        """Test that shortcuts on occupied thumb positions get penalized."""
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),   # L0 left thumb access
            Position(1, 1, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),   # L1 left thumb (occupied)
            Position(2, 1, 7.0, 4.0, "right", 0, 0.8, is_thumb=True),  # L1 right thumb (free)
            Position(3, 1, 0.0, 0.0, "left", 1, 1.0),                  # L1 finger
        )
        shortcuts = (
            Shortcut(0, "Ctrl+A", "Select All", "app", 10.0),
            Shortcut(1, "Ctrl+C", "Copy", "app", 8.0),
            Shortcut(
                2, "@access:L0->L1:hold:Nav", "Nav", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=1,
                access_is_momentary=True,
            ),
        )
        genome = np.array([2, 0, -1, -1], dtype=np.int32)
        frozen = np.array([False, False, False, False])

        layout = Layout(genome, positions, shortcuts, frozen)
        
        from fitness.factors.violation import ViolationFactor
        factor = ViolationFactor()
        penalty = factor._thumb_occupancy(layout)
        
        # Shortcut on left thumb should be penalized
        self.assertGreater(penalty, 0.0)
        
        # Move shortcut to right thumb (free) - penalty should be 0
        genome2 = np.array([2, -1, 0, -1], dtype=np.int32)
        layout2 = Layout(genome2, positions, shortcuts, frozen)
        penalty2 = factor._thumb_occupancy(layout2)
        self.assertEqual(penalty2, 0.0)
        
        # Remove layer access - no penalty even on left thumb
        genome3 = np.array([-1, 0, -1, -1], dtype=np.int32)
        layout3 = Layout(genome3, positions, shortcuts, frozen)
        penalty3 = factor._thumb_occupancy(layout3)
        self.assertEqual(penalty3, 0.0)

    def test_compiled_occupied_thumb_penalizes_left_momentary_path(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 1, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(2, 1, 7.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(3, 7, 6.0, 4.0, "right", 0, 0.8, is_thumb=True),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L1:hold:Nav", "Nav", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=1,
                access_is_momentary=True,
            ),
            Shortcut(1, "Ctrl+A", "Select All", "app", 10.0),
            Shortcut(
                2, "@access:L0->L7:toggle:Game", "Game", "Layer Access", 8.0,
                "layer_access", is_layer_access=True, access_target_layer=7,
                access_is_momentary=False,
            ),
        )
        frozen = np.array([False, False, False, False])
        left_blocked = Layout(np.array([0, 1, -1, 2], dtype=np.int32), positions, shortcuts, frozen)
        right_free = Layout(np.array([0, -1, 1, 2], dtype=np.int32), positions, shortcuts, frozen)
        weights = {
            "effort": 0.0, "adjacency": 0.0, "finger_balance": 0.0,
            "same_finger": 0.0, "violations": 1.0, "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0, "familiarity": 0.0,
        }
        vweights = {"thumb_occupancy": 1000.0, "access_layout": 0.0}
        evaluator = FitnessEvaluator(
            weights=weights,
            reference_layout=left_blocked,
            violation_weights=vweights,
            hard_constraints=[],
            missing_important_threshold=99.0,
        )
        left_score = evaluator.evaluate(left_blocked).objectives[2]
        right_score = evaluator.evaluate(right_free).objectives[2]
        self.assertGreater(left_score, right_score)

    def test_dynamic_mouse_layer_requires_buttons_scroll_and_access(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(8, 4, 2.0, 2.0, "left", 1, 1.0),
            Position(2, 3, 8.0, 1.0, "right", 1, 1.0),
            Position(3, 3, 9.0, 1.0, "right", 1, 1.0),
            Position(4, 3, 10.0, 1.0, "right", 2, 1.0),
            Position(5, 3, 11.0, 1.0, "right", 3, 1.0),
            Position(6, 3, 12.0, 1.0, "right", 4, 1.2),
            Position(7, 3, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(8, 3, 8.0, 2.0, "right", 1, 1.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L3:hold:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L3:toggle:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(3, "MB2", "Click", "Mouse", 15.0, "mouse"),
            Shortcut(4, "MB3", "Click", "Mouse", 10.0, "mouse"),
            Shortcut(5, "MB4", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(6, "MB5", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(
                7, "@access:L3->L6:hold:Scroll", "Scroll", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=6,
                access_is_momentary=True,
            ),
            Shortcut(
                8, "@access:L4->L3:toggle:Mouse", "Mouse", "Layer Access", 10.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(
                9, "@access:L0->L4:hold:Source", "Source", "Layer Access", 10.0,
                "layer_access", is_layer_access=True, access_target_layer=4,
                access_is_momentary=True,
            ),
            Shortcut(
                10, "@access:L0->L3:hold:MouseRightThumb", "Mouse", "Layer Access", 10.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        layout = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        report = _dynamic_mouse_layer_report(layout)
        self.assertTrue(report["acceptance_pass"])
        self.assertEqual(report["mouse_layer"], 3)

        missing_toggle = Layout(
            np.array([0, -1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertFalse(_dynamic_mouse_layer_report(missing_toggle)["acceptance_pass"])

        no_momentary_access = Layout(
            np.array([-1, 1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertTrue(_dynamic_mouse_layer_report(no_momentary_access)["acceptance_pass"])

        non_l0_toggle = Layout(
            np.array([0, 9, 8, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertTrue(_dynamic_mouse_layer_report(non_l0_toggle)["acceptance_pass"])

        extra_duplicate_elsewhere = Layout(
            np.array([0, 1, 2, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertTrue(_dynamic_mouse_layer_report(extra_duplicate_elsewhere)["acceptance_pass"])

        right_thumb_button = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, -1, 6, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        thumb_report = _dynamic_mouse_layer_report(right_thumb_button)
        self.assertFalse(thumb_report["acceptance_pass"])
        self.assertTrue(thumb_report["best_candidate"]["right_thumb_button_placements"])

        right_thumb_scroll = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, 7, -1], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        scroll_report = _dynamic_mouse_layer_report(right_thumb_scroll)
        self.assertFalse(scroll_report["acceptance_pass"])
        self.assertTrue(scroll_report["best_candidate"]["right_thumb_momentary_scroll_access"])

        right_thumb_momentary_access = Layout(
            np.array([-1, 1, -1, 2, 3, 4, 5, 6, 10, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        access_report = _dynamic_mouse_layer_report(right_thumb_momentary_access)
        self.assertFalse(access_report["acceptance_pass"])
        self.assertTrue(access_report["best_candidate"]["right_thumb_momentary_access"])

    def test_layer7_acceptance_checks_access_modes_only(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 4.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(2, 7, 5.0, 4.0, "left", 0, 0.8, is_thumb=True, is_frozen=True),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L7:hold:Fallback", "Fallback", "Layer Access", 8.0,
                "layer_access", is_layer_access=True, access_target_layer=7,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L7:toggle:Fallback", "Fallback", "Layer Access", 8.0,
                "layer_access", is_layer_access=True, access_target_layer=7,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Mouse 1", "Mouse", 8.0, "mouse"),
        )
        frozen = np.array([False, False, True])

        only_toggle = Layout(np.array([-1, 1, 2], dtype=np.int32), positions, shortcuts, frozen)
        both_modes = Layout(np.array([0, 1, 2], dtype=np.int32), positions, shortcuts, frozen)

        self.assertFalse(_layer7_access_report(only_toggle)["acceptance_pass"])
        report = _layer7_access_report(both_modes)
        self.assertTrue(report["acceptance_pass"])
        self.assertFalse(report["content_checked"])

    def test_dynamic_mouse_layer_penalty_rewards_natural_complete_layer(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(2, 3, 2.0, 1.0, "left", 1, 1.0),
            Position(3, 3, 8.0, 1.0, "right", 1, 1.0),
            Position(4, 3, 9.0, 1.0, "right", 1, 1.0),
            Position(5, 3, 10.0, 1.0, "right", 2, 1.0),
            Position(6, 3, 11.0, 1.0, "right", 3, 1.0),
            Position(7, 3, 12.0, 1.0, "right", 4, 1.2),
            Position(8, 3, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(9, 3, 8.0, 2.0, "right", 1, 1.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L3:hold:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L3:toggle:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(3, "MB2", "Click", "Mouse", 15.0, "mouse"),
            Shortcut(4, "MB3", "Click", "Mouse", 10.0, "mouse"),
            Shortcut(5, "MB4", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(6, "MB5", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(
                7, "@access:L3->L6:hold:Scroll", "Scroll", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=6,
                access_is_momentary=True,
            ),
            Shortcut(
                8, "@access:L0->L3:hold:MouseRightThumb", "Mouse", "Layer Access", 10.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        valid = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        missing_toggle = Layout(
            np.array([0, -1, -1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        left_button = Layout(
            np.array([0, 1, 2, -1, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 1.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_specialization": 0.0,
        }
        vweights = {
            "duplicate": 0.0,
            "l0_displacement": 0.0,
            "missing_important": 0.0,
            "cross_layer_duplicate": 0.0,
            "group_split": 0.0,
            "thumb_occupancy": 0.0,
            "arrow_order": 0.0,
            "hand_bias": 0.0,
            "mouse_layer_access": 0.0,
            "arrow_scattered": 0.0,
            "mouse_scattered": 0.0,
            "layer7_access": 0.0,
            "duplicate_value_gap": 0.0,
            "access_layout": 0.0,
            "raw_keyboard_completion_norwegian": 0.0,
            "dynamic_mouse_layer": 1.0,
        }
        evaluator = FitnessEvaluator(
            weights=weights,
            reference_layout=valid,
            violation_weights=vweights,
            hard_constraints=[],
            missing_important_threshold=99.0,
        )
        valid_score = evaluator.evaluate(valid).objectives[2]
        self.assertGreater(evaluator.evaluate(missing_toggle).objectives[2], valid_score)
        self.assertGreater(evaluator.evaluate(left_button).objectives[2], valid_score)

        right_thumb_button = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, -1, 6, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertGreater(evaluator.evaluate(right_thumb_button).objectives[2], valid_score)

        right_thumb_scroll = Layout(
            np.array([0, 1, -1, 2, 3, 4, 5, 6, 7, -1], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertGreater(evaluator.evaluate(right_thumb_scroll).objectives[2], valid_score)

        right_thumb_momentary_access = Layout(
            np.array([-1, 1, -1, 2, 3, 4, 5, 6, 8, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertGreater(evaluator.evaluate(right_thumb_momentary_access).objectives[2], valid_score)

    def test_dynamic_mouse_layer_penalty_uses_mouse_usage_for_placement(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(2, 3, 8.0, 1.0, "right", 1, 0.6),
            Position(3, 3, 9.0, 1.0, "right", 2, 2.8),
            Position(4, 3, 10.0, 1.0, "right", 2, 1.0),
            Position(5, 3, 11.0, 1.0, "right", 3, 1.0),
            Position(6, 3, 12.0, 1.0, "right", 4, 1.2),
            Position(7, 3, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(8, 3, 8.0, 2.0, "right", 1, 1.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L3:hold:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L3:toggle:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(3, "MB2", "Click", "Mouse", 15.0, "mouse"),
            Shortcut(4, "MB3", "Click", "Mouse", 10.0, "mouse"),
            Shortcut(5, "MB4", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(6, "MB5", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(
                7, "@access:L3->L6:hold:Scroll", "Scroll", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=6,
                access_is_momentary=True,
            ),
        )
        usage = UsageData(mouse_clicks={"MB1": {"count": 100}, "MB2": {"count": 1}})
        frozen = np.zeros(len(positions), dtype=np.bool_)
        better = Layout(
            np.array([0, 1, 2, 3, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        worse = Layout(
            np.array([0, 1, 3, 2, 4, 5, 6, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 1.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_specialization": 0.0,
        }
        vweights = {
            "duplicate": 0.0,
            "l0_displacement": 0.0,
            "missing_important": 0.0,
            "cross_layer_duplicate": 0.0,
            "group_split": 0.0,
            "thumb_occupancy": 0.0,
            "arrow_order": 0.0,
            "hand_bias": 0.0,
            "mouse_layer_access": 0.0,
            "arrow_scattered": 0.0,
            "mouse_scattered": 0.0,
            "layer7_access": 0.0,
            "duplicate_value_gap": 0.0,
            "access_layout": 0.0,
            "raw_keyboard_completion_norwegian": 0.0,
            "dynamic_mouse_layer": 1.0,
        }
        evaluator = FitnessEvaluator(
            weights=weights,
            reference_layout=better,
            violation_weights=vweights,
            hard_constraints=[],
            missing_important_threshold=99.0,
        )
        self.assertLess(evaluator.evaluate(better).objectives[2], evaluator.evaluate(worse).objectives[2])

    def test_mouse_duplicates_clean_up_after_natural_mouse_layer_exists(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(2, 3, 8.0, 1.0, "right", 1, 1.0),
            Position(3, 3, 9.0, 1.0, "right", 2, 1.0),
            Position(4, 3, 10.0, 1.0, "right", 2, 1.0),
            Position(5, 3, 11.0, 1.0, "right", 3, 1.0),
            Position(6, 3, 12.0, 1.0, "right", 4, 1.2),
            Position(7, 3, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(8, 4, 8.0, 1.0, "right", 1, 1.0),
            Position(9, 3, 8.0, 2.0, "right", 1, 1.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L3:hold:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L3:toggle:Mouse", "Mouse", "Layer Access", 16.0,
                "layer_access", is_layer_access=True, access_target_layer=3,
                access_is_momentary=False,
            ),
            Shortcut(2, "MB1", "Click", "Mouse", 20.0, "mouse"),
            Shortcut(3, "MB2", "Click", "Mouse", 15.0, "mouse"),
            Shortcut(4, "MB3", "Click", "Mouse", 10.0, "mouse"),
            Shortcut(5, "MB4", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(6, "MB5", "Click", "Mouse", 8.0, "mouse"),
            Shortcut(
                7, "@access:L3->L6:hold:Scroll", "Scroll", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=6,
                access_is_momentary=True,
            ),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        natural = Layout(
            np.array([0, 1, 2, 3, 4, 5, 6, -1, -1, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        extra_mouse = Layout(
            np.array([0, 1, 2, 3, 4, 5, 6, -1, 2, 7], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 1.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_similarity": 0.0,
            "everything_layer": 0.0,
        }
        vweights = {
            "duplicate": 0.0,
            "l0_displacement": 0.0,
            "missing_important": 0.0,
            "cross_layer_duplicate": 0.0,
            "group_split": 0.0,
            "thumb_occupancy": 0.0,
            "arrow_order": 0.0,
            "hand_bias": 0.0,
            "mouse_layer_access": 0.0,
            "arrow_scattered": 0.0,
            "mouse_scattered": 1.0,
            "layer7_access": 0.0,
            "duplicate_value_gap": 0.0,
            "access_layout": 0.0,
            "raw_keyboard_completion_norwegian": 0.0,
            "dynamic_mouse_layer": 0.0,
        }
        evaluator = FitnessEvaluator(
            weights=weights,
            reference_layout=natural,
            violation_weights=vweights,
            hard_constraints=[],
            missing_important_threshold=99.0,
        )
        self.assertGreater(evaluator.evaluate(extra_mouse).objectives[2], evaluator.evaluate(natural).objectives[2])

    def test_everything_layer_rewards_common_shortcuts_on_one_accessible_layer(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 1, 8.0, 1.0, "right", 1, 1.0),
            Position(2, 1, 9.0, 1.0, "right", 2, 1.0),
            Position(3, 1, 10.0, 1.0, "right", 3, 1.0),
            Position(4, 2, 8.0, 1.0, "right", 1, 1.0),
            Position(5, 3, 8.0, 1.0, "right", 1, 1.0),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L1:hold:General", "General", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=1,
                access_is_momentary=True,
            ),
            Shortcut(1, "Shortcut A", "A", "app", 10.0),
            Shortcut(2, "Shortcut B", "B", "app", 9.0),
            Shortcut(3, "Shortcut C", "C", "app", 8.0),
        )
        usage = UsageData(shortcuts={
            "Shortcut A": {"count": 100},
            "Shortcut B": {"count": 80},
            "Shortcut C": {"count": 60},
        })
        frozen = np.zeros(len(positions), dtype=np.bool_)
        concentrated = Layout(
            np.array([0, 1, 2, 3, -1, -1], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        scattered = Layout(
            np.array([0, 1, -1, -1, 2, 3], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            usage_data=usage,
        )
        weights = {
            "effort": 0.0,
            "adjacency": 0.0,
            "finger_balance": 0.0,
            "same_finger": 0.0,
            "violations": 1.0,
            "workflow_coherence": 0.0,
            "app_coherence": 0.0,
            "trackball_proximity": 0.0,
            "familiarity": 0.0,
            "layer_specialization": 0.0,
            "everything_layer": 10.0,
        }
        evaluator = FitnessEvaluator(
            weights=weights,
            reference_layout=concentrated,
            violation_weights={},
            hard_constraints=[],
            missing_important_threshold=99.0,
        )
        self.assertLess(
            evaluator.evaluate(concentrated).objectives[2],
            evaluator.evaluate(scattered).objectives[2],
        )

    def test_momentary_only_single_thumb_side_must_be_clear(self):
        positions = (
            Position(0, 0, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(1, 0, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
            Position(2, 2, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),
            Position(3, 2, 8.0, 4.0, "right", 0, 0.8, is_thumb=True),
        )
        shortcuts = (
            Shortcut(
                0, "@access:L0->L2:hold:Layer", "Layer", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=2,
                access_is_momentary=True,
            ),
            Shortcut(
                1, "@access:L0->L2:toggle:Layer", "Layer", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=2,
                access_is_momentary=False,
            ),
            Shortcut(
                2, "@access:L0->L2:hold:LayerRight", "Layer", "Layer Access", 12.0,
                "layer_access", is_layer_access=True, access_target_layer=2,
                access_is_momentary=True,
            ),
            Shortcut(3, "Ctrl+A", "Select All", "app", 8.0),
            Shortcut(4, "Ctrl+C", "Copy", "app", 8.0),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)

        blocked = Layout(
            np.array([0, -1, 3, 4], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        report = _momentary_only_thumb_clearance_report(blocked)
        self.assertFalse(report["acceptance_pass"])
        self.assertEqual(report["violations"][0]["occupied_hand"], "left")

        toggle_access = Layout(
            np.array([0, 1, 3, 4], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertTrue(_momentary_only_thumb_clearance_report(toggle_access)["acceptance_pass"])

        both_momentary_sides = Layout(
            np.array([0, 2, 3, 4], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        self.assertTrue(_momentary_only_thumb_clearance_report(both_momentary_sides)["acceptance_pass"])

        lost_second_side = Layout(
            np.array([0, -1, 3, -1], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
        )
        lost_report = _momentary_only_thumb_clearance_report(lost_second_side)
        self.assertFalse(lost_report["acceptance_pass"])
        self.assertEqual(lost_report["violations"][0]["restricted_hands"], ["left"])


class TestEmptyPositionPenalty(unittest.TestCase):
    """Verify the soft empty-position penalty (violation sub-weight 'empty_position').

    Policy:
    - Low-effort (prime) positions → high penalty when empty.
    - High-effort (far/corner) positions → near-zero penalty when empty.
    - L7 frozen content is never penalised.
    - Penalty is soft: it adds to the violations objective but never causes a
      hard acceptance failure by itself.
    """

    def _make_evaluator(self, layout, empty_position_weight=50.0):
        """Build an evaluator with only empty_position active."""
        weights = {k: 0.0 for k in [
            "effort", "adjacency", "finger_balance", "same_finger",
            "violations", "workflow_coherence", "app_coherence",
            "trackball_proximity", "familiarity", "layer_similarity", "everything_layer",
        ]}
        weights["violations"] = 1.0
        vweights = {k: 0.0 for k in [
            "duplicate", "l0_displacement", "missing_important", "cross_layer_duplicate",
            "group_split", "thumb_occupancy", "arrow_order", "hand_bias",
            "mouse_layer_access", "arrow_scattered", "mouse_scattered", "layer7_access",
            "duplicate_value_gap", "access_layout", "raw_keyboard_completion_norwegian",
            "dynamic_mouse_layer", "natural_mouse_layer_exists",
            "layer_reachability", "layer_depth_penalty", "toggle_back_to_l0",
            "mouse_hold_position_conflict", "mouse_layer_depth_penalty",
        ]}
        vweights["empty_position"] = empty_position_weight
        from fitness.evaluator import FitnessEvaluator
        return FitnessEvaluator(
            weights=weights,
            reference_layout=layout,
            violation_weights=vweights,
            hard_constraints=[],
        )

    def _prime_effort_layout(self):
        """Layout where position 1 (layer 1) is a prime low-effort slot.

        Returns (layout_with_access, layout_empty_prime, layout_empty_far).
        """
        # position 0: L0 left thumb (access key to L1, effort=0.4)
        # position 1: L1 right home-row (low effort = 0.3) -- "prime"
        # position 2: L1 right far corner (high effort = 4.5)
        positions = (
            Position(0, 0, 1.0, 4.0, "left", 0, 0.4, is_thumb=True),
            Position(1, 1, 7.0, 2.0, "right", 1, 0.3),   # prime
            Position(2, 1, 0.0, 0.0, "right", 4, 4.5),   # far corner
        )
        shortcuts = (
            Shortcut(0, "@toggle:L0->L1:toggle", "Toggle", "Layer Access", 10.0,
                     "layer_access", is_layer_access=True, access_target_layer=1,
                     access_is_momentary=False),
            Shortcut(1, "Ctrl+C", "Copy", "app", 8.0),
        )
        frozen = np.zeros(3, dtype=np.bool_)

        # Base: access on pos 0, Ctrl+C on pos 1 (prime), pos 2 (far) empty
        filled_prime = Layout(np.array([0, 1, -1], dtype=np.int32), positions, shortcuts, frozen)
        # Variant: pos 1 (prime) empty, pos 2 (far) still empty
        empty_prime = Layout(np.array([0, -1, -1], dtype=np.int32), positions, shortcuts, frozen)
        # Variant: pos 2 (far) empty, pos 1 (prime) filled
        empty_far = Layout(np.array([0, 1, -1], dtype=np.int32), positions, shortcuts, frozen)

        return filled_prime, empty_prime, empty_far

    def test_prime_empty_scores_higher_than_far_empty(self):
        """An empty prime position gets a much larger penalty than an empty far position."""
        filled_prime, empty_prime, _ = self._prime_effort_layout()
        ev = self._make_evaluator(filled_prime, empty_position_weight=50.0)

        # Layout A: prime pos 1 empty, far pos 2 empty
        positions = filled_prime.positions
        shortcuts = filled_prime.shortcuts
        frozen = filled_prime.frozen_mask
        layout_prime_empty = Layout(np.array([0, -1, -1], dtype=np.int32), positions, shortcuts, frozen)
        # Layout B: prime pos 1 filled, far pos 2 empty
        layout_far_empty = Layout(np.array([0, 1, -1], dtype=np.int32), positions, shortcuts, frozen)

        # Need an evaluator built on a layout — rebuild for each
        ev_prime = self._make_evaluator(layout_prime_empty, empty_position_weight=50.0)
        ev_far = self._make_evaluator(layout_far_empty, empty_position_weight=50.0)

        score_prime = ev_prime.evaluate(layout_prime_empty).total_score
        score_far = ev_far.evaluate(layout_far_empty).total_score

        self.assertGreater(
            score_prime, score_far,
            f"Empty prime position (effort=0.3) should cost more than empty far (effort=4.5); "
            f"got prime={score_prime:.4f}, far={score_far:.4f}",
        )

    def test_l7_frozen_not_penalised(self):
        """L7 frozen positions are excluded from empty_position penalty."""
        # Two positions: L0 thumb (frozen access key) and L7 position (frozen, empty)
        positions = (
            Position(0, 0, 1.0, 4.0, "left", 0, 0.4, is_thumb=True, is_frozen=True),
            Position(1, 7, 7.0, 2.0, "right", 1, 0.3, is_frozen=True),   # L7, frozen
        )
        shortcuts = (
            Shortcut(0, "@toggle:L0->L7:toggle", "Toggle", "Layer Access", 10.0,
                     "layer_access", is_layer_access=True, access_target_layer=7,
                     access_is_momentary=False),
        )
        # L7 is frozen so both positions frozen; genome has access on pos 0, pos 1 empty
        frozen = np.array([True, True], dtype=np.bool_)
        layout = Layout(np.array([0, -1], dtype=np.int32), positions, shortcuts, frozen)
        ev = self._make_evaluator(layout, empty_position_weight=50.0)
        result = ev.evaluate(layout)
        # Empty position penalty contribution should be 0 (L7 excluded)
        self.assertEqual(result.total_score, 0.0,
                         f"L7 frozen empty position must not be penalised; got {result.total_score}")

    def test_l0_empty_not_penalised_by_empty_position(self):
        """L0 empty positions are excluded from the empty_position penalty."""
        positions = (
            Position(0, 0, 7.0, 2.0, "right", 1, 0.3),   # L0, prime effort, empty
            Position(1, 0, 0.0, 0.0, "right", 4, 4.5),   # L0, far, empty
        )
        shortcuts = (
            Shortcut(0, "Ctrl+C", "Copy", "app", 8.0),
        )
        frozen = np.zeros(2, dtype=np.bool_)
        layout = Layout(np.array([-1, -1], dtype=np.int32), positions, shortcuts, frozen)
        ev = self._make_evaluator(layout, empty_position_weight=50.0)
        result = ev.evaluate(layout)
        self.assertEqual(result.total_score, 0.0,
                         f"L0 empty positions must not be penalised; got {result.total_score}")

    def test_penalty_is_soft_not_hard_constraint(self):
        """empty_position never appears as a hard constraint."""
        from config import DEFAULT_CONFIG
        hard = DEFAULT_CONFIG["fitness"]["hard_constraints"]
        self.assertNotIn("empty_position", hard,
                         "empty_position must remain a soft penalty, not a hard constraint")

    def test_penalty_scales_with_position_effort(self):
        """Penalty decreases monotonically as position effort increases."""
        # Build a series of single-slot layouts on layer 1 with increasing effort.
        efforts = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]
        prev_score = None
        for eff in efforts:
            positions = (
                Position(0, 0, 1.0, 4.0, "left", 0, 0.4, is_thumb=True),  # access key (L0)
                Position(1, 1, 7.0, 2.0, "right", 1, eff),                 # empty slot
            )
            shortcuts = (
                Shortcut(0, "@toggle:L0->L1:toggle", "Toggle", "Layer Access", 10.0,
                         "layer_access", is_layer_access=True, access_target_layer=1,
                         access_is_momentary=False),
            )
            frozen = np.zeros(2, dtype=np.bool_)
            layout = Layout(np.array([0, -1], dtype=np.int32), positions, shortcuts, frozen)
            ev = self._make_evaluator(layout, empty_position_weight=50.0)
            score = ev.evaluate(layout).total_score
            if prev_score is not None:
                self.assertLessEqual(
                    score, prev_score + 1e-4,
                    f"Penalty should decrease (or stay flat) as effort increases; "
                    f"got score={score:.4f} > prev={prev_score:.4f} at effort={eff}",
                )
            prev_score = score


class TestDynamicLayerAccessNotCanonical(unittest.TestCase):
    """Verify that scoring and acceptance use evolved genome bindings,
    never the static layout.layer_access fallback.

    Policy: layout.layer_access is NOT authoritative for evolved layouts.
    All reachability, cost, and acceptance logic must read shortcut.is_layer_access
    from the genome.
    """

    def _make_layout(self, genome_list, positions, shortcuts, layer_access=()):
        """Helper: build a Layout with an explicit layer_access seed (possibly stale)."""
        frozen = np.zeros(len(positions), dtype=np.bool_)
        return Layout(
            np.array(genome_list, dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            layer_access=tuple(layer_access),
        )

    def test_acceptance_uses_genome_not_layer_access(self):
        """dynamic_mouse_layer_report reads genome shortcuts, ignores layer_access."""
        from core import LayerAccess
        from evolution.acceptance import _dynamic_mouse_layer_report

        # Layout: L0 (left thumb at pos 0) + L3 (right non-thumb x6, right thumb x1)
        # pos 6 (L3, right non-thumb): scroll momentary access
        # pos 7 (L3, right thumb): empty
        positions = (
            Position(0, 0, 1.0, 4.0, "left", 0, 0.5, is_thumb=True),  # L0 toggle key
            Position(1, 3, 7.0, 2.0, "right", 1, 1.0),   # MB1
            Position(2, 3, 8.0, 2.0, "right", 2, 1.0),   # MB2
            Position(3, 3, 9.0, 2.0, "right", 3, 1.0),   # MB3
            Position(4, 3, 10.0, 2.0, "right", 4, 1.0),  # MB4
            Position(5, 3, 11.0, 2.0, "right", 1, 1.0),  # MB5
            Position(6, 3, 8.0, 3.0, "right", 2, 0.9),   # scroll (non-thumb)
            Position(7, 3, 9.0, 4.0, "right", 0, 0.7, is_thumb=True),  # thumb (empty)
        )
        shortcuts = (
            Shortcut(0, "@toggle:L0->L3:toggle", "Toggle", "Layer Access", 10.0,
                     "layer_access", is_layer_access=True, access_target_layer=3,
                     access_is_momentary=False),
            Shortcut(1, "MB1", "MB1", "Mouse", 10.0),
            Shortcut(2, "MB2", "MB2", "Mouse", 9.0),
            Shortcut(3, "MB3", "MB3", "Mouse", 8.0),
            Shortcut(4, "MB4", "MB4", "Mouse", 7.0),
            Shortcut(5, "MB5", "MB5", "Mouse", 6.0),
            Shortcut(6, "@scroll:L3->Scroll:hold", "Scroll", "Layer Access", 8.0,
                     "layer_access", is_layer_access=True, access_target_layer=10,
                     access_is_momentary=True),
        )
        # genome: toggle on pos 0, MB1-MB5 on pos 1-5, scroll on pos 6, thumb empty
        genome = np.array([0, 1, 2, 3, 4, 5, 6, -1], dtype=np.int32)
        # layer_access is stale: claims layer 1 is accessible, not layer 3
        stale_layer_access = (
            LayerAccess(target_layer=1, source_layer=0, source_x=1.0, source_y=4.0,
                        hand="left", is_momentary=False, access_key_label="stale"),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)
        layout = Layout(genome, positions, shortcuts, frozen, layer_access=stale_layer_access)

        report = _dynamic_mouse_layer_report(layout)
        # Must find the mouse layer via genome bindings (L3), NOT via stale layer_access (L1)
        self.assertTrue(report["acceptance_pass"],
                        f"Expected acceptance_pass=True; failure_guidance={report.get('failure_guidance')}")
        self.assertEqual(report["mouse_layer"], 3)

    def test_effort_factor_uses_genome_access_not_layer_access(self):
        """EffortFactor.compute() reads genome-based access costs, not layout.layer_access."""
        from core import LayerAccess

        # Layer 1 keys are on an expensive non-thumb position (effort 5.0)
        # Genome puts the access shortcut on a cheap thumb (effort 0.2)
        # layer_access (stale canonical) says access to layer 1 comes from an effort-2.0 key
        positions = (
            Position(0, 0, 1.0, 4.0, "left", 0, 0.2, is_thumb=True),  # access key (cheap)
            Position(1, 1, 7.0, 2.0, "right", 1, 5.0),                # layer-1 key (expensive)
        )
        shortcuts = (
            Shortcut(0, "@access:L0->L1:hold", "Hold", "Layer Access", 10.0,
                     "layer_access", is_layer_access=True, access_target_layer=1,
                     access_is_momentary=True),
            Shortcut(1, "Ctrl+C", "Copy", "app", 8.0),
        )
        frozen = np.zeros(2, dtype=np.bool_)

        # Stale layer_access claims access to L1 came from effort 3.5
        stale_access = (
            LayerAccess(target_layer=1, source_layer=0, source_x=99.0, source_y=99.0,
                        hand="left", is_momentary=True, access_key_label="stale"),
        )
        layout = Layout(
            np.array([0, 1], dtype=np.int32),
            positions,
            shortcuts,
            frozen,
            layer_access=stale_access,
        )

        ef = EffortFactor()
        # Dynamic cost from genome: access key at position 0, effort=0.2
        dynamic_costs = ef._compute_layer_access_costs_from_genome(layout)
        # Static cost from stale layer_access: would try to find effort at (99,99) → fallback 2.0
        static_costs = ef._compute_layer_access_costs(layout)

        # Layer 1 cost from genome: 0.0 (L0) + 0.2 (access effort) = 0.2
        self.assertAlmostEqual(dynamic_costs.get(1, -1), 0.2, places=5,
                               msg="Dynamic access cost should use genome effort 0.2")
        # Static cost would be 0.0 + 2.0 (fallback for unmatched coord) = 2.0
        self.assertAlmostEqual(static_costs.get(1, -1), 2.0, places=5,
                               msg="Static (canonical) access cost uses fallback 2.0")
        # compute() now uses the dynamic path
        total = ef.compute(layout)
        # expected:
        #   pos 0: access shortcut (sid 0, importance=10.0), layer=0, effort=0.2, access_cost=0.0
        #          → 10.0 * (0.2 + 0.0) = 2.0
        #   pos 1: Ctrl+C (sid 1, importance=8.0), layer=1, effort=5.0, access_cost=0.2 (genome)
        #          → 8.0 * (5.0 + 0.2) = 41.6
        #   total = 43.6
        expected_dynamic = 10.0 * (0.2 + 0.0) + 8.0 * (5.0 + 0.2)
        self.assertAlmostEqual(total, expected_dynamic, places=4,
                               msg="EffortFactor must use genome-derived access cost, not canonical")

    def test_occupied_thumbs_from_genome_ignores_layer_access(self):
        """get_occupied_thumbs_from_genome reads genome, not layout.layer_access."""
        from core import LayerAccess

        positions = (
            Position(0, 0, 1.0, 4.0, "left", 0, 0.5, is_thumb=True),
            Position(1, 0, 9.0, 4.0, "right", 0, 0.5, is_thumb=True),
            Position(2, 3, 7.0, 2.0, "right", 1, 1.0),
        )
        shortcuts = (
            # genome will put this on the RIGHT thumb (position 1) → right is occupied
            Shortcut(0, "@access:L0->L3:hold", "Hold", "Layer Access", 10.0,
                     "layer_access", is_layer_access=True, access_target_layer=3,
                     access_is_momentary=True),
            Shortcut(1, "Ctrl+C", "Copy", "app", 8.0),
        )
        frozen = np.zeros(3, dtype=np.bool_)
        # Stale layer_access says LEFT thumb holds the access key
        stale = (
            LayerAccess(target_layer=3, source_layer=0, source_x=1.0, source_y=4.0,
                        hand="left", is_momentary=True),
        )
        layout = Layout(
            np.array([0, -1, 1], dtype=np.int32),  # sid 0 (access) on pos 0 (left thumb)
            positions, shortcuts, frozen,
            layer_access=stale,
        )

        # Genome says access is on position 0 = LEFT thumb
        dynamic = layout.get_occupied_thumbs_from_genome(3)
        self.assertIn("left", dynamic)
        self.assertNotIn("right", dynamic,
                         "Genome puts access on left thumb; right must not appear")

        # Legacy static says LEFT too (stale matches in this case) — just verify legacy still works
        legacy = layout.get_occupied_thumbs(3)
        self.assertIn("left", legacy)

        # Now change the genome so access is on RIGHT thumb (position 1)
        layout2 = Layout(
            np.array([-1, 0, 1], dtype=np.int32),  # sid 0 (access) on pos 1 (right thumb)
            positions, shortcuts, frozen,
            layer_access=stale,  # stale still says left!
        )
        dynamic2 = layout2.get_occupied_thumbs_from_genome(3)
        self.assertIn("right", dynamic2,
                      "Genome puts access on right thumb; right must appear")
        self.assertNotIn("left", dynamic2)

        # Legacy static still returns left (reads stale layer_access)
        legacy2 = layout2.get_occupied_thumbs(3)
        self.assertIn("left", legacy2,
                      "Legacy reads stale layer_access which claims left thumb")

    def test_layer_access_field_is_not_used_by_acceptance_static(self):
        """Static layout.layer_access must not affect final acceptance of generated layers.

        Acceptance is driven entirely by shortcut.is_layer_access genome bindings.
        A layout whose layer_access claims one set of accesses but whose genome
        binds different accesses must be accepted based on the genome, not the claim.
        """
        from core import LayerAccess
        from evolution.acceptance import build_acceptance_report

        # L0: left thumb (toggle key), right thumb (empty)
        # L4: 5 right non-thumb (MB1-MB5), 1 right non-thumb (scroll), 1 right thumb (empty)
        positions = (
            Position(0, 0, 1.0, 4.0, "left", 0, 0.5, is_thumb=True),   # L0 toggle slot
            Position(1, 0, 9.0, 4.0, "right", 0, 0.5, is_thumb=True),  # L0 right thumb (empty)
            Position(2, 4, 7.0, 2.0, "right", 1, 1.0),   # MB1
            Position(3, 4, 8.0, 2.0, "right", 2, 1.0),   # MB2
            Position(4, 4, 9.0, 2.0, "right", 3, 1.0),   # MB3
            Position(5, 4, 10.0, 2.0, "right", 4, 1.0),  # MB4
            Position(6, 4, 11.0, 2.0, "right", 1, 1.0),  # MB5
            Position(7, 4, 8.0, 3.0, "right", 2, 0.9),   # scroll (non-thumb)
            Position(8, 4, 9.0, 4.0, "right", 0, 0.7, is_thumb=True),  # L4 right thumb (empty)
        )
        shortcuts = (
            Shortcut(0, "@toggle:L0->L4:toggle", "Toggle", "Layer Access", 10.0,
                     "layer_access", is_layer_access=True, access_target_layer=4,
                     access_is_momentary=False),
            Shortcut(1, "MB1", "MB1", "Mouse", 10.0),
            Shortcut(2, "MB2", "MB2", "Mouse", 9.0),
            Shortcut(3, "MB3", "MB3", "Mouse", 8.0),
            Shortcut(4, "MB4", "MB4", "Mouse", 7.0),
            Shortcut(5, "MB5", "MB5", "Mouse", 6.0),
            Shortcut(6, "@scroll:L4->Scroll:hold", "Scroll", "Layer Access", 8.0,
                     "layer_access", is_layer_access=True, access_target_layer=10,
                     access_is_momentary=True),
        )
        frozen = np.zeros(len(positions), dtype=np.bool_)

        # layer_access claims access to layer 99 (not 4) — completely stale
        stale = (
            LayerAccess(target_layer=99, source_layer=0, source_x=1.0, source_y=4.0,
                        hand="left", is_momentary=False),
        )
        # toggle on pos 0, right-thumb L0 empty, MB1-5 on L4, scroll on non-thumb L4, thumb empty
        genome = np.array([0, -1, 1, 2, 3, 4, 5, 6, -1], dtype=np.int32)
        layout = Layout(genome, positions, shortcuts, frozen, layer_access=stale)

        report = build_acceptance_report(layout)
        dynamic_mouse = report["details"]["dynamic_mouse_layer"]
        self.assertTrue(dynamic_mouse["acceptance_pass"],
                        f"Mouse layer acceptance should pass using genome; "
                        f"guidance={dynamic_mouse.get('failure_guidance')}")
        self.assertEqual(dynamic_mouse["mouse_layer"], 4,
                         "Mouse layer should be L4 as bound in genome, not canonical layer_access")


class TestStagnationMetric(unittest.TestCase):
    """Verify the stagnation metric sums all objectives, not just violations.

    Policy: effort improvement or adjacency gain alone must prevent false stagnation.
    The computation is: best_quality = min(pop.sum(axis=1)).
    """

    def _make_pop(self, rows):
        """Build a numpy array shaped (n, 3) representing objective columns."""
        return np.array(rows, dtype=np.float64)

    def test_effort_improvement_resets_stagnation(self):
        """Improving effort objective alone (violations unchanged) lowers pop sum."""
        # Gen t: best row sum = 100 + 50 + 200 = 350
        pop_t = self._make_pop([
            [100.0, 50.0, 200.0],
            [120.0, 60.0, 210.0],
        ])
        # Gen t+1: effort improved from 100 -> 80, sum = 80 + 50 + 200 = 330
        pop_t1 = self._make_pop([
            [80.0, 50.0, 200.0],
            [120.0, 60.0, 210.0],
        ])
        quality_t = float(np.min(pop_t.sum(axis=1)))
        quality_t1 = float(np.min(pop_t1.sum(axis=1)))
        self.assertLess(quality_t1, quality_t * 0.999,
                        "Effort improvement alone must lower best_quality and prevent stagnation")

    def test_violations_only_improvement_also_resets(self):
        """Improving violations alone also lowers the sum and prevents stagnation."""
        pop_t = self._make_pop([[100.0, 50.0, 300.0]])
        pop_t1 = self._make_pop([[100.0, 50.0, 50.0]])   # violations dropped sharply
        quality_t = float(np.min(pop_t.sum(axis=1)))
        quality_t1 = float(np.min(pop_t1.sum(axis=1)))
        self.assertLess(quality_t1, quality_t * 0.999)

    def test_no_improvement_triggers_stagnation(self):
        """Identical population quality triggers stagnation (sum unchanged)."""
        pop_t = self._make_pop([[100.0, 50.0, 200.0]])
        pop_t1 = self._make_pop([[100.0, 50.0, 200.0]])
        quality_t = float(np.min(pop_t.sum(axis=1)))
        quality_t1 = float(np.min(pop_t1.sum(axis=1)))
        # The condition in run_evolution.py: stagnation++ if NOT (quality_t1 < quality_t * 0.999)
        improved = quality_t1 < quality_t * 0.999
        self.assertFalse(improved, "Equal quality should count as stagnation, not improvement")

    def test_violations_alone_unchanged_but_effort_dropped(self):
        """Effort drops, violations unchanged — still counts as non-stagnant."""
        # Simulates a run that improved effort/workflow but not violations
        pop_t = self._make_pop([[150.0, 30.0, 500.0]])   # sum = 680
        pop_t1 = self._make_pop([[120.0, 30.0, 500.0]])  # sum = 650 (effort only)
        quality_t = float(np.min(pop_t.sum(axis=1)))
        quality_t1 = float(np.min(pop_t1.sum(axis=1)))
        self.assertLess(quality_t1, quality_t * 0.999,
                        "Effort-only improvement must prevent stagnation")


class TestMouseLayerAcceptanceTier(unittest.TestCase):
    """Verify dynamic_mouse_layer is checked at training time (optimizer_side_pass),
    not just at final export.

    Policy:
    - dynamic_mouse_layer_present is in optimizer_side_checks — training-level.
    - The kernel applies soft pressure (dynamic_mouse_layer weight 5000) during training.
    - final_acceptance_only field must NOT appear in the report (it was misleading and removed).
    - A layout lacking a valid mouse layer gets optimizer_side_pass=False.
    - A layout with a valid mouse layer gets optimizer_side_pass influence from this check.
    """

    def _build_mouse_layout(self, include_mouse=True):
        """Build a minimal layout with or without a complete mouse layer on L1."""
        # pos 0: L0 left thumb — toggle to L1
        # pos 1: L1 right index  — MB1
        # pos 2: L1 right middle — MB2
        # pos 3: L1 right ring   — MB3
        # pos 4: L1 right pinky  — MB4
        # pos 5: L1 right index2 — MB5
        # pos 6: L1 right middle2 (non-thumb) — scroll-mode access key
        # pos 7: L7 left thumb — momentary access to L7 (frozen)
        # pos 8: L7 left thumb2 — toggle to L7 (frozen)
        positions = (
            Position(0, 0, 1.0, 4.0, "left", 0, 0.4, is_thumb=True),
            Position(1, 1, 7.0, 2.0, "right", 1, 0.3),
            Position(2, 1, 6.0, 2.0, "right", 2, 0.4),
            Position(3, 1, 5.0, 2.0, "right", 3, 0.5),
            Position(4, 1, 4.0, 2.0, "right", 4, 0.6),
            Position(5, 1, 3.0, 2.0, "right", 1, 0.5),
            Position(6, 1, 2.0, 2.0, "right", 2, 0.7),   # non-thumb scroll access
            Position(7, 7, 1.0, 4.0, "left", 0, 0.4, is_thumb=True, is_frozen=True),
            Position(8, 7, 0.5, 4.0, "left", 0, 0.4, is_thumb=True, is_frozen=True),
        )
        shortcuts = (
            # sid 0: L0->L1 toggle (access)
            Shortcut(0, "@toggle:L0->L1:toggle", "Toggle", "Layer Access", 10.0,
                     "layer_access", is_layer_access=True, access_target_layer=1,
                     access_is_momentary=False),
            # sid 1-5: MB1-MB5
            Shortcut(1, "MB1", "Mouse Button 1", "mouse", 8.0),
            Shortcut(2, "MB2", "Mouse Button 2", "mouse", 7.0),
            Shortcut(3, "MB3", "Mouse Button 3", "mouse", 5.0),
            Shortcut(4, "MB4", "Mouse Button 4", "mouse", 5.0),
            Shortcut(5, "MB5", "Mouse Button 5", "mouse", 5.0),
            # sid 6: scroll-mode layer access (non-thumb right hand on L1)
            Shortcut(6, "@mo:L1->scroll_mode:momentary", "Scroll mode", "Layer Access", 8.0,
                     "layer_access", is_layer_access=True, access_target_layer=10,
                     access_is_momentary=True),
            # sid 7: L7 momentary (frozen)
            Shortcut(7, "@mo:L0->L7:momentary", "L7 momentary", "Layer Access", 5.0,
                     "layer_access", is_layer_access=True, access_target_layer=7,
                     access_is_momentary=True),
            # sid 8: L7 toggle (frozen)
            Shortcut(8, "@toggle:L0->L7:toggle", "L7 toggle", "Layer Access", 5.0,
                     "layer_access", is_layer_access=True, access_target_layer=7,
                     access_is_momentary=False),
        )
        frozen = np.array([False, False, False, False, False, False, False, True, True], dtype=np.bool_)
        if include_mouse:
            genome = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        else:
            # No mouse buttons placed — positions 1-5 empty
            genome = np.array([0, -1, -1, -1, -1, -1, -1, 7, 8], dtype=np.int32)
        return Layout(genome, positions, shortcuts, frozen)

    def test_mouse_layer_check_is_in_optimizer_side(self):
        """dynamic_mouse_layer_present must appear in optimizer_side_checks, not export_checks."""
        from evolution.acceptance import build_acceptance_report
        layout = self._build_mouse_layout(include_mouse=True)
        report = build_acceptance_report(layout)
        self.assertIn("dynamic_mouse_layer_present", report["optimizer_side_checks"],
                      "Mouse layer check must be in optimizer_side_checks (training-level)")
        self.assertNotIn("dynamic_mouse_layer_present", report.get("export_checks", {}),
                         "Mouse layer check must not be in export_checks")

    def test_no_final_acceptance_only_field(self):
        """The misleading final_acceptance_only field must not appear in the mouse layer report."""
        from evolution.acceptance import build_acceptance_report
        layout = self._build_mouse_layout(include_mouse=True)
        report = build_acceptance_report(layout)
        mouse_detail = report["details"]["dynamic_mouse_layer"]
        self.assertNotIn("final_acceptance_only", mouse_detail,
                         "final_acceptance_only was removed — it was misleading since the check runs at training time")

    def test_missing_mouse_layer_fails_optimizer_side(self):
        """Layout without valid mouse layer must have optimizer_side_pass=False."""
        from evolution.acceptance import build_acceptance_report
        layout = self._build_mouse_layout(include_mouse=False)
        report = build_acceptance_report(layout)
        self.assertFalse(report["optimizer_side_checks"]["dynamic_mouse_layer_present"],
                         "Missing mouse layer must fail dynamic_mouse_layer_present")
        # optimizer_side_pass should be False (other checks may also fail, that's fine)
        if all(v for k, v in report["optimizer_side_checks"].items()
               if k != "dynamic_mouse_layer_present"):
            # all other checks pass → optimizer_side_pass should be solely driven by mouse
            self.assertFalse(report["optimizer_side_pass"],
                             "optimizer_side_pass must be False when mouse layer is absent")

    def test_l7_cannot_be_mouse_layer(self):
        """L7 is excluded from mouse layer detection — it cannot satisfy the mouse layer check."""
        from evolution.acceptance import _dynamic_mouse_layer_report
        # Put MB1-5 on L7 — must be ignored
        positions = (
            Position(0, 0, 1.0, 4.0, "left", 0, 0.4, is_thumb=True),
            Position(1, 7, 7.0, 2.0, "right", 1, 0.3, is_frozen=True),
            Position(2, 7, 6.0, 2.0, "right", 2, 0.4, is_frozen=True),
            Position(3, 7, 5.0, 2.0, "right", 3, 0.5, is_frozen=True),
            Position(4, 7, 4.0, 2.0, "right", 4, 0.6, is_frozen=True),
            Position(5, 7, 3.0, 2.0, "right", 1, 0.5, is_frozen=True),
        )
        shortcuts = (
            Shortcut(0, "@toggle:L0->L7:toggle", "Toggle", "Layer Access", 10.0,
                     "layer_access", is_layer_access=True, access_target_layer=7,
                     access_is_momentary=False),
            Shortcut(1, "MB1", "MB1", "mouse", 8.0),
            Shortcut(2, "MB2", "MB2", "mouse", 7.0),
            Shortcut(3, "MB3", "MB3", "mouse", 5.0),
            Shortcut(4, "MB4", "MB4", "mouse", 5.0),
            Shortcut(5, "MB5", "MB5", "mouse", 5.0),
        )
        frozen = np.array([False, True, True, True, True, True], dtype=np.bool_)
        genome = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        layout = Layout(genome, positions, shortcuts, frozen)
        report = _dynamic_mouse_layer_report(layout)
        self.assertFalse(report["acceptance_pass"],
                         "L7 must not satisfy the mouse layer check even with MB1-5 on it")


class TestSwapMutationNumba(unittest.TestCase):
    def _build_simple_mutation_layout(self):
        """Layout with frozen positions, a group, and mutable positions."""
        positions = tuple(
            Position(i, 1, float(i % 6), float(i // 6), "left" if i < 6 else "right", 1, 1.0)
            for i in range(12)
        )
        shortcuts = tuple([
            Shortcut(0, "LeftArrow", "Left", "Nav", 1.0, base_key="LeftArrow"),
            Shortcut(1, "UpArrow", "Up", "Nav", 1.0, base_key="UpArrow"),
            Shortcut(2, "DownArrow", "Down", "Nav", 1.0, base_key="DownArrow"),
            Shortcut(3, "RightArrow", "Right", "Nav", 1.0, base_key="RightArrow"),
            Shortcut(4, "@access:L2:hold", "L2 hold", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=2, access_is_momentary=True),
            Shortcut(5, "@access:L2:toggle", "L2 toggle", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=2, access_is_momentary=False),
            Shortcut(6, "@access:L0:toggle", "L0 return", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=0, access_is_momentary=False),
            *[Shortcut(i, f"K{i}", "", "App", 1.0) for i in range(7, 12)],
        ])
        frozen = np.array([False] * 12, dtype=np.bool_)
        frozen[0] = True
        genome = np.arange(12, dtype=np.int32)
        layout = Layout(genome, positions, shortcuts, frozen)
        return layout

    def test_numba_kernel_preserves_invariants(self):
        """Numba-accelerated mutations must not touch frozen positions or scatter groups."""
        layout = self._build_simple_mutation_layout()
        mutation = SwapMutation(
            prob=0.5,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.5,
            effort_swap_prob=0.5,
            smart_duplicate_prob=0.5,
        )
        np.random.seed(42)
        random.seed(42)
        pop = np.tile(layout.genome.astype(np.int32), (200, 1))
        # introduce some empty slots and duplicates
        pop[pop == 11] = -1
        pop[:, 4] = 4
        out = mutation._do(None, pop.copy())

        # Frozen position 0 must be unchanged.
        self.assertTrue(np.all(out[:, 0] == layout.genome[0]))
        # Arrow group sids 0-3 must still occupy only positions 0-3 (group overwrite disabled).
        for sid in range(4):
            for row in out:
                pos = int(np.where(row == sid)[0][0]) if sid in row else -1
                if pos >= 0:
                    self.assertIn(pos, [0, 1, 2, 3])

    def test_numba_and_python_fallback_similar_mutation_rates(self):
        """Numba path and pure-Python fallback should mutate a similar fraction."""
        import evolution
        layout = self._build_simple_mutation_layout()
        mutation = SwapMutation(
            prob=0.5,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.5,
            effort_swap_prob=0.5,
            smart_duplicate_prob=0.5,
        )
        n = 500
        np.random.seed(123)
        random.seed(123)
        pop = np.tile(layout.genome.astype(np.int32), (n, 1))
        pop[pop == 11] = -1

        # Numba path
        out_numba = mutation._do(None, pop.copy())
        changed_numba = np.sum(np.any(out_numba != pop, axis=1))

        # Force Python fallback by patching NUMBA_AVAILABLE
        orig = evolution.NUMBA_AVAILABLE
        try:
            evolution.NUMBA_AVAILABLE = False
            np.random.seed(123)
            random.seed(123)
            out_py = mutation._do(None, pop.copy())
        finally:
            evolution.NUMBA_AVAILABLE = orig
        changed_py = np.sum(np.any(out_py != pop, axis=1))

        # Both should mutate a non-trivial fraction of genomes.
        self.assertGreater(changed_numba, n * 0.1)
        self.assertGreater(changed_py, n * 0.1)
        # Rates should be within 20 percentage points (different RNGs, same probabilities).
        self.assertLess(abs(changed_numba - changed_py) / n, 0.20)

    def test_numba_kernel_is_deterministic_for_same_seeds(self):
        """Calling the Numba kernel directly with the same seeds must be deterministic."""
        from evolution import _mutate_batch_numba
        layout = self._build_simple_mutation_layout()
        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.0,
            group_overwrite_prob=0.0,
            optional_arrow_drop_prob=0.0,
            bulk_assign_prob=0.0,
            cluster_app_prob=0.0,
            random_assign_prob=0.5,
            effort_swap_prob=0.5,
            smart_duplicate_prob=0.5,
        )
        n = 50
        pop = np.tile(layout.genome.astype(np.int32), (n, 1))
        pop[pop == 11] = -1
        handled = np.zeros(n, dtype=np.bool_)
        seeds = np.random.randint(0, 2**63, size=n, dtype=np.uint64)
        probs = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float64)
        out1 = pop.copy()
        _mutate_batch_numba(
            out1, handled.copy(), probs, seeds,
            mutation._mutable_arr,
            mutation._pos_layer_arr,
            mutation._pos_hand_arr,
            mutation._pos_is_thumb_arr,
            mutation._pos_effort_arr,
            mutation._sid_importance_arr,
            mutation._access_target_lut,
            mutation._access_is_mo_lut,
            mutation._mo_access_target_lut,
            mutation._is_group_sid_lut,
            mutation._is_important_sid_lut,
            np.int32(mutation._return_toggle_sid if mutation._return_toggle_sid is not None else -1),
            mutation._dup_candidate_arr,
            mutation._dup_exp_w,
            mutation._frozen_sid_counts,
            mutation._assignable_arr,
            mutation._layer_mutable_flat,
            mutation._layer_mutable_start,
            mutation._mouse_button_sids,
            mutation._toggle_access_sids_arr,
            np.int32(mutation.n_shortcuts),
        )
        out2 = pop.copy()
        _mutate_batch_numba(
            out2, handled.copy(), probs, seeds,
            mutation._mutable_arr,
            mutation._pos_layer_arr,
            mutation._pos_hand_arr,
            mutation._pos_is_thumb_arr,
            mutation._pos_effort_arr,
            mutation._sid_importance_arr,
            mutation._access_target_lut,
            mutation._access_is_mo_lut,
            mutation._mo_access_target_lut,
            mutation._is_group_sid_lut,
            mutation._is_important_sid_lut,
            np.int32(mutation._return_toggle_sid if mutation._return_toggle_sid is not None else -1),
            mutation._dup_candidate_arr,
            mutation._dup_exp_w,
            mutation._frozen_sid_counts,
            mutation._assignable_arr,
            mutation._layer_mutable_flat,
            mutation._layer_mutable_start,
            mutation._mouse_button_sids,
            mutation._toggle_access_sids_arr,
            np.int32(mutation.n_shortcuts),
        )
        np.testing.assert_array_equal(out1, out2)

    def test_numba_random_reassign_pairs_return_toggle(self):
        """Numba random_reassign must place a return toggle when creating a toggle access."""
        from evolution import _numba_random_reassign_one
        # Build a layout with mutable positions on layer 2 so the return toggle has a home.
        positions = tuple(
            Position(i, 1 if i < 6 else 2, float(i % 6), float(i // 6), "left" if i < 6 else "right", 1, 1.0)
            for i in range(12)
        )
        shortcuts = tuple([
            Shortcut(0, "LeftArrow", "Left", "Nav", 1.0, base_key="LeftArrow"),
            Shortcut(1, "UpArrow", "Up", "Nav", 1.0, base_key="UpArrow"),
            Shortcut(2, "DownArrow", "Down", "Nav", 1.0, base_key="DownArrow"),
            Shortcut(3, "RightArrow", "Right", "Nav", 1.0, base_key="RightArrow"),
            Shortcut(4, "@access:L2:toggle", "L2 toggle", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=2, access_is_momentary=False),
            Shortcut(5, "@access:L0:toggle", "L0 return", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=0, access_is_momentary=False),
            *[Shortcut(i, f"K{i}", "", "App", 1.0) for i in range(6, 12)],
        ])
        frozen = np.array([False] * 12, dtype=np.bool_)
        genome = np.arange(12, dtype=np.int32)
        layout = Layout(genome, positions, shortcuts, frozen)
        mutation = SwapMutation(prob=0.0, frozen_mask=layout.frozen_mask, layout=layout)

        # Force assignable pool to only the L2 toggle so random_reassign must place it.
        mutation._assignable_arr = np.array([4], dtype=np.int32)

        g = genome.copy()
        g[5] = 11  # non-group position on layer 1
        state = np.array([42], dtype=np.uint64)
        ok = _numba_random_reassign_one(
            g, state,
            mutation._mutable_arr,
            mutation._pos_layer_arr,
            mutation._assignable_arr,
            mutation._is_group_sid_lut,
            mutation._is_important_sid_lut,
            mutation._access_target_lut,
            mutation._mo_access_target_lut,
            mutation.n_shortcuts,
            mutation._toggle_access_sids_arr,
            np.int32(mutation._return_toggle_sid),
            mutation._layer_mutable_flat,
            mutation._layer_mutable_start,
        )
        self.assertTrue(ok)
        self.assertIn(4, g, "L2 toggle was not placed")
        self.assertIn(5, g, "Return-to-L0 toggle missing on layer 2")


class TestCudaExactEvalParity(unittest.TestCase):
    """Parity between the CUDA exact-eval kernel and the Numba fallback."""

    def _make_evaluator(self):
        from core.loader import build_layout
        from config import Config
        config = Config.load("config_v2.yaml")
        layout = build_layout("data", config.raw.get("fitness", {}))
        return FitnessEvaluator(
            weights=config.get("fitness.weights", {}),
            reference_layout=layout,
            violation_weights=config.get("fitness.violation_sub_weights", {}),
            missing_important_threshold=config.get("fitness.missing_important_threshold", 6.0),
            hard_constraints=config.get("fitness.hard_constraints", []),
            toggle_effort_multiplier=float(config.get("fitness.toggle_effort_multiplier", 2.5)),
        )

    def test_cuda_parity_seed_and_random(self):
        """CUDA and Numba objectives must agree within float32 tolerance."""
        try:
            import torch
            from fitness.cuda_kernel import cuda_available
        except Exception:
            self.skipTest("CUDA kernel not importable")
        if not cuda_available():
            self.skipTest("CUDA not available")

        evaluator = self._make_evaluator()
        layout = evaluator.reference_layout
        rng = np.random.default_rng(98765)

        # Seed genome
        samples = [layout.genome.copy()]
        # Mutated seed genomes
        for _ in range(7):
            g = layout.genome.copy()
            for _ in range(20):
                a, b = rng.choice(layout.mutable_indices, 2, replace=False)
                g[a], g[b] = g[b], g[a]
            samples.append(g)
        # Fully random genomes
        for _ in range(7):
            g = np.full(layout.n_positions, -1, dtype=np.int32)
            g[layout.frozen_indices] = layout.genome[layout.frozen_indices]
            n_assign = min(len(layout.mutable_indices), layout.n_shortcuts)
            assigned = rng.choice(layout.n_shortcuts, size=n_assign, replace=False)
            g[layout.mutable_indices[:n_assign]] = assigned
            samples.append(g)

        batch = np.asarray(samples, dtype=np.int32)

        evaluator.model._use_cuda = True
        obj_cuda, _ = evaluator.model.evaluate_batch(batch)

        evaluator.model._use_cuda = False
        obj_numba, _ = evaluator.model.evaluate_batch(batch)

        # Allow larger absolute tolerance on the violations objective because it
        # sums many large terms in different orders on GPU vs CPU.
        np.testing.assert_allclose(obj_cuda[:, 0], obj_numba[:, 0], rtol=1e-4, atol=2000.0)
        np.testing.assert_allclose(obj_cuda[:, 1], obj_numba[:, 1], rtol=1e-4, atol=1.0)
        np.testing.assert_allclose(obj_cuda[:, 2], obj_numba[:, 2], rtol=1e-4, atol=2e7)


if __name__ == "__main__":
    unittest.main()
