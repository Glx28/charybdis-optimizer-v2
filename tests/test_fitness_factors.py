"""Fitness factor tests (effort, adjacency, violation weights)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import numpy as np

from core import Layout, Position, Shortcut, UsageData
from evolution.arrow_cluster import analyze_arrows
from tests.legacy_factors import AdjacencyFactor, EffortFactor, ViolationFactor


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


if __name__ == "__main__":
    unittest.main()
