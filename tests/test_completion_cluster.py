"""Tests for Norwegian raw completion-cluster analysis."""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from core import Position, Shortcut, Layout
from evolution.completion_cluster import (
    RAW_COMPLETION_FAMILY,
    completion_order,
    is_raw_completion_base,
    analyze_completion_cluster,
)


class TestCompletionCluster(unittest.TestCase):
    def _make_shortcut(self, sid, keys, base_key, modifiers=()):
        return Shortcut(
            sid=sid,
            keys=keys,
            action="test",
            app="test",
            importance=5.0,
            category="general",
            modifiers=tuple(modifiers),
            base_key=base_key,
        )

    def test_family_membership(self):
        dash = self._make_shortcut(0, "-", "Dash and Underscore")
        ctrl_dash = self._make_shortcut(1, "Ctrl+-", "Dash and Underscore", ["Ctrl"])
        other = self._make_shortcut(2, "Ctrl+C", "C", ["Ctrl"])

        self.assertEqual(completion_order(dash), RAW_COMPLETION_FAMILY["DASH AND UNDERSCORE"])
        self.assertTrue(is_raw_completion_base(dash))
        self.assertEqual(completion_order(ctrl_dash), RAW_COMPLETION_FAMILY["DASH AND UNDERSCORE"])
        self.assertFalse(is_raw_completion_base(ctrl_dash))
        self.assertEqual(completion_order(other), 0)
        self.assertFalse(is_raw_completion_base(other))

    def test_analyze_cluster_chooses_anchor_and_reports_missing(self):
        # L1 positions: 0,1,2 on one row, L2 position 3, L3 position 4.
        positions = (
            Position(0, 1, 0.0, 0.0, "left", 1, 1.0),
            Position(1, 1, 1.0, 0.0, "left", 1, 1.0),
            Position(2, 1, 2.0, 0.0, "left", 1, 1.0),
            Position(3, 2, 0.0, 0.0, "left", 1, 1.0),
            Position(4, 3, 0.0, 0.0, "left", 1, 1.0),
        )
        shortcuts = (
            self._make_shortcut(0, "-", "Dash and Underscore"),
            self._make_shortcut(1, "=", "Equals and Plus"),
            self._make_shortcut(2, "`", "Grave Accent and Tilde"),
            self._make_shortcut(3, "]", "Right Brace"),
            self._make_shortcut(4, "Ctrl+\\", "Backslash and Pipe", ["Ctrl"]),
        )
        genome = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        frozen = np.array([False] * 5)
        layout = Layout(genome, positions, shortcuts, frozen)

        report = analyze_completion_cluster(layout)
        self.assertEqual(report["anchor_layer"], 1)
        self.assertEqual(set(report["raw_base_keys_present"]), {"Dash and Underscore", "Equals and Plus", "Grave Accent and Tilde"})
        self.assertIn("Right Brace", report["raw_base_keys_missing"])
        self.assertIn("Backslash and Pipe", report["modified_variants_demand"])
        self.assertEqual(report["raw_base_layers_used"], 2)

    def test_compactness_score_penalises_inversions(self):
        positions = (
            Position(0, 1, 2.0, 0.0, "left", 1, 1.0),
            Position(1, 1, 1.0, 0.0, "left", 1, 1.0),
            Position(2, 1, 0.0, 0.0, "left", 1, 1.0),
        )
        shortcuts = (
            self._make_shortcut(0, "-", "Dash and Underscore"),
            self._make_shortcut(1, "=", "Equals and Plus"),
            self._make_shortcut(2, "`", "Grave Accent and Tilde"),
        )
        genome = np.array([0, 1, 2], dtype=np.int32)
        frozen = np.array([False] * 3)
        inverted = Layout(genome, positions, shortcuts, frozen)

        ordered_positions = (
            Position(0, 1, 0.0, 0.0, "left", 1, 1.0),
            Position(1, 1, 1.0, 0.0, "left", 1, 1.0),
            Position(2, 1, 2.0, 0.0, "left", 1, 1.0),
        )
        ordered = Layout(genome, ordered_positions, shortcuts, frozen)

        self.assertGreater(
            analyze_completion_cluster(ordered)["compactness_order_score"],
            analyze_completion_cluster(inverted)["compactness_order_score"],
        )

    def test_acceptance_requires_exact_fixed_shape(self):
        shortcuts = (
            self._make_shortcut(0, "-", "Dash and Underscore"),
            self._make_shortcut(1, "=", "Equals and Plus"),
            self._make_shortcut(2, "`", "Grave Accent and Tilde"),
            self._make_shortcut(3, "]", "Right Brace"),
            self._make_shortcut(4, "\\", "Backslash and Pipe"),
        )
        frozen = np.array([False] * 5)
        # Equals is the anchor at x=10,y=1. Required offsets are:
        # Dash(-1,0), Equals(0,0), Grave(-2,0), RightBrace(-2,1), Backslash(-2,3).
        valid_positions = (
            Position(0, 4, 9.0, 1.0, "right", 1, 1.0),
            Position(1, 4, 10.0, 1.0, "right", 1, 1.0),
            Position(2, 4, 8.0, 1.0, "right", 1, 1.0),
            Position(3, 4, 8.0, 2.0, "right", 1, 1.0),
            Position(4, 4, 8.0, 4.0, "right", 1, 1.0),
        )
        wrong_positions = (
            Position(0, 4, 9.0, 1.0, "right", 1, 1.0),
            Position(1, 4, 10.0, 1.0, "right", 1, 1.0),
            Position(2, 4, 8.0, 1.0, "right", 1, 1.0),
            Position(3, 4, 8.0, 2.0, "right", 1, 1.0),
            Position(4, 4, 11.0, 4.0, "right", 1, 1.0),
        )
        genome = np.array([0, 1, 2, 3, 4], dtype=np.int32)

        valid = analyze_completion_cluster(Layout(genome, valid_positions, shortcuts, frozen))
        wrong = analyze_completion_cluster(Layout(genome, wrong_positions, shortcuts, frozen))

        self.assertTrue(valid["exact_shape_preserved"])
        self.assertTrue(valid["acceptance_pass"])
        self.assertFalse(wrong["exact_shape_preserved"])
        self.assertFalse(wrong["acceptance_pass"])


if __name__ == "__main__":
    unittest.main()
