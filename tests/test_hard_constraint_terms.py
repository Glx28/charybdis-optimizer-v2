"""Tests for the 2026-08-05 hard-constraint additions:
unsupported_duplicate and thumb_occupancy_restricted.

Both terms exist so acceptance checks that were previously invisible to Deb
feasibility-first selection (soft, ~1000x below the objective noise floor)
become hard constraints. Classification must match acceptance
(run_evolution.analyze_duplicates / evolution/acceptance.py
momentary_only_thumb_side_clear).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import numpy as np

from config import DEFAULT_CONFIG, Config
from core import Layout, Position, Shortcut, UsageData
from fitness.kernel import precompute, _single_genome, _evaluate_batch


def _precompute(layout, hard_constraints):
    return precompute(
        layout,
        weights={"effort": 1.0},
        violation_weights={},
        missing_important_threshold=99.0,
        scale_factors=np.ones(3, dtype=np.float32),
        hard_constraints=hard_constraints,
    )


class TestUnsupportedDuplicateConstraint(unittest.TestCase):
    """unsupported_duplicate: cross-layer duplicate (same sid on 2+ distinct
    generated layers, L7 excluded) with usage evidence (support > 0) but
    duplicate support < 0.25. Magnitude = extra copies beyond the first."""

    HC = ["unsupported_duplicate"]

    def _build(self, genome, shortcuts, usage, frozen=None, dynamic_groups=()):
        n = len(genome)
        positions = tuple(
            Position(i, 1 if i < 4 else (2 if i < 8 else 7), float(i % 4), 0.0,
                     "left", 1, 1.0, is_frozen=(i >= 8))
            for i in range(n)
        )
        if frozen is None:
            frozen = np.zeros(n, dtype=np.bool_)
            for i in range(8, n):
                frozen[i] = True
        return Layout(np.array(genome, dtype=np.int32), positions, shortcuts,
                      frozen, usage_data=usage, dynamic_groups=dynamic_groups)

    def _shortcuts(self):
        return (
            Shortcut(0, "Win+E", "a", "App", 5.0),
            Shortcut(1, "Ctrl+V", "a", "App", 5.0),
        )

    def _usage_low_support(self):
        # support[Win+E] = min(0.75, 1/100) = 0.01 -> evidence present, < 0.25
        return UsageData(shortcuts={"Win+E": {"count": 1}, "Ctrl+V": {"count": 100}})

    def test_low_support_cross_layer_duplicate_counts(self):
        layout = self._build(
            [0, -1, -1, -1, 0, 1, -1, -1], self._shortcuts(), self._usage_low_support())
        arrays = _precompute(layout, self.HC)
        self.assertGreater(_single_genome(layout.genome, *arrays)[1][0], 0.0)

    def test_extra_copies_beyond_first_are_counted(self):
        # 3 copies across 2 layers -> magnitude 2.
        layout = self._build(
            [0, 0, -1, -1, 0, 1, -1, -1], self._shortcuts(), self._usage_low_support())
        arrays = _precompute(layout, self.HC)
        self.assertEqual(float(_single_genome(layout.genome, *arrays)[1][0]), 2.0)

    def test_supported_duplicate_does_not_count(self):
        usage = UsageData(shortcuts={"Win+E": {"count": 50}, "Ctrl+V": {"count": 100}})
        layout = self._build(
            [0, -1, -1, -1, 0, 1, -1, -1], self._shortcuts(), usage)
        arrays = _precompute(layout, self.HC)
        self.assertEqual(float(_single_genome(layout.genome, *arrays)[1][0]), 0.0)

    def test_zero_evidence_duplicate_tolerated(self):
        layout = self._build(
            [0, -1, -1, -1, 0, 1, -1, -1], self._shortcuts(), UsageData())
        arrays = _precompute(layout, self.HC)
        self.assertEqual(float(_single_genome(layout.genome, *arrays)[1][0]), 0.0)

    def test_single_layer_duplicate_not_in_scope(self):
        # Two copies on ONE layer only: that is same_layer_duplicate's job,
        # not this cross-layer term's.
        layout = self._build(
            [0, 0, -1, -1, -1, 1, -1, -1], self._shortcuts(), self._usage_low_support())
        arrays = _precompute(layout, self.HC)
        self.assertEqual(float(_single_genome(layout.genome, *arrays)[1][0]), 0.0)

    def test_mouse_button_excluded(self):
        shortcuts = (
            Shortcut(0, "MB1", "Click", "Mouse", 10.0, "mouse"),
            Shortcut(1, "Ctrl+V", "a", "App", 5.0),
        )
        usage = UsageData(shortcuts={"MB1": {"count": 1}, "Ctrl+V": {"count": 100}})
        layout = self._build([0, -1, -1, -1, 0, 1, -1, -1], shortcuts, usage)
        arrays = _precompute(layout, self.HC)
        self.assertEqual(float(_single_genome(layout.genome, *arrays)[1][0]), 0.0)

    def test_protected_group_member_excluded(self):
        layout = self._build(
            [0, -1, -1, -1, 0, 1, -1, -1], self._shortcuts(), self._usage_low_support(),
            dynamic_groups=({"protected": True, "sids": [0]},))
        arrays = _precompute(layout, self.HC)
        self.assertEqual(float(_single_genome(layout.genome, *arrays)[1][0]), 0.0)

    def test_l0_only_excluded(self):
        shortcuts = (
            Shortcut(0, "_base_A", "a", "Base", 5.0, is_l0_only=True),
            Shortcut(1, "Ctrl+V", "a", "App", 5.0),
        )
        usage = UsageData(shortcuts={"_base_A": {"count": 1}, "Ctrl+V": {"count": 100}})
        layout = self._build([0, -1, -1, -1, 0, 1, -1, -1], shortcuts, usage)
        arrays = _precompute(layout, self.HC)
        self.assertEqual(float(_single_genome(layout.genome, *arrays)[1][0]), 0.0)

    def test_layer7_copies_excluded(self):
        # Copies on L7 + one generated layer: L7 is outside the class, so only
        # one generated layer carries the sid -> no violation.
        layout = self._build(
            [0, -1, -1, -1, -1, 1, -1, -1, 0, -1], self._shortcuts(),
            self._usage_low_support())
        arrays = _precompute(layout, self.HC)
        self.assertEqual(float(_single_genome(layout.genome, *arrays)[1][0]), 0.0)

    def test_numba_single_and_batch_paths_agree(self):
        dup = self._build(
            [0, -1, -1, -1, 0, 1, -1, -1], self._shortcuts(), self._usage_low_support())
        clean = self._build(
            [-1, -1, -1, -1, 0, 1, -1, -1], self._shortcuts(), self._usage_low_support())
        arrays = _precompute(dup, self.HC)
        _, c_single_dup = _single_genome(dup.genome, *arrays)
        _, c_single_clean = _single_genome(clean.genome, *arrays)
        _, c_batch = _evaluate_batch(np.stack([dup.genome, clean.genome]), *arrays)
        np.testing.assert_array_equal(c_batch[0], c_single_dup)
        np.testing.assert_array_equal(c_batch[1], c_single_clean)
        self.assertGreater(c_batch[0][0], 0.0)
        self.assertEqual(float(c_batch[1][0]), 0.0)


class TestThumbOccupancyRestrictedConstraint(unittest.TestCase):
    """thumb_occupancy_restricted: only the genuinely-restricted component of
    thumb_occupancy -- reachable-gated, non-self-referential momentary thumb
    access, no reachable toggle access, single-side restriction. Matches
    acceptance's momentary_only_thumb_side_clear semantics."""

    HC = ["thumb_occupancy_restricted"]

    def _eval(self, positions, shortcuts, genome):
        layout = Layout(np.array(genome, dtype=np.int32), positions, shortcuts,
                        np.zeros(len(positions), dtype=np.bool_))
        arrays = _precompute(layout, self.HC)
        return float(_single_genome(layout.genome, *arrays)[1][0]), layout

    def _hold(self, sid, target):
        return Shortcut(sid, f"@access:L{target}:hold", "h", "Layer Access", 5.0,
                        is_layer_access=True, access_target_layer=target,
                        access_is_momentary=True)

    def _toggle(self, sid, target):
        return Shortcut(sid, f"@access:L{target}:toggle", "t", "Layer Access", 5.0,
                        is_layer_access=True, access_target_layer=target,
                        access_is_momentary=False)

    def test_restricted_side_occupancy_counts(self):
        positions = (
            Position(0, 0, 0.0, 0.0, "right", 0, 0.1, is_thumb=True),  # hold -> L1
            Position(1, 1, 5.0, 2.0, "right", 0, 0.1, is_thumb=True),  # occupant 1
            Position(2, 1, 6.0, 2.0, "right", 0, 0.2, is_thumb=True),  # occupant 2
            Position(3, 1, 9.0, 9.0, "left", 1, 1.0),
        )
        shortcuts = (self._hold(0, 1), Shortcut(1, "Ctrl+A", "a", "App", 3.0),
                     Shortcut(2, "Ctrl+B", "a", "App", 3.0))
        value, _ = self._eval(positions, shortcuts, [0, 1, 2, -1])
        self.assertEqual(value, 2.0, "each occupied restricted thumb slot counts")

    def test_toggle_freed_layer_passes(self):
        positions = (
            Position(0, 0, 0.0, 0.0, "right", 0, 0.1, is_thumb=True),  # hold -> L1
            Position(1, 0, 1.0, 0.0, "left", 1, 1.0),                  # toggle -> L1
            Position(2, 1, 5.0, 2.0, "right", 0, 0.1, is_thumb=True),  # occupant
            Position(3, 1, 9.0, 9.0, "left", 1, 1.0),
        )
        shortcuts = (self._hold(0, 1), self._toggle(1, 1), Shortcut(2, "Ctrl+A", "a", "App", 3.0))
        value, layout = self._eval(positions, shortcuts, [0, 1, 2, -1])
        self.assertEqual(value, 0.0)
        # Cross-check against acceptance: the same layout must PASS.
        from evolution.acceptance import _momentary_only_thumb_clearance_report
        self.assertTrue(_momentary_only_thumb_clearance_report(layout)["acceptance_pass"])

    def test_self_referential_source_does_not_restrict(self):
        positions = (
            Position(0, 1, 0.0, 0.0, "right", 0, 0.1, is_thumb=True),  # self-ref hold ON L1
            Position(1, 1, 5.0, 2.0, "right", 0, 0.1, is_thumb=True),  # occupant
            Position(2, 1, 9.0, 9.0, "left", 1, 1.0),
        )
        shortcuts = (self._hold(0, 1), Shortcut(1, "Ctrl+A", "a", "App", 3.0))
        value, layout = self._eval(positions, shortcuts, [0, 1, -1])
        self.assertEqual(value, 0.0)
        from evolution.acceptance import _momentary_only_thumb_clearance_report
        self.assertTrue(_momentary_only_thumb_clearance_report(layout)["acceptance_pass"])

    def test_unreachable_source_does_not_restrict(self):
        # Momentary thumb hold sits on L2, which nothing reaches: acceptance
        # gates incoming access on source reachability, so L1 is not restricted.
        positions = (
            Position(0, 2, 0.0, 0.0, "right", 0, 0.1, is_thumb=True),  # hold -> L1 from unreachable L2
            Position(1, 1, 5.0, 2.0, "right", 0, 0.1, is_thumb=True),  # occupant
            Position(2, 1, 9.0, 9.0, "left", 1, 1.0),
        )
        shortcuts = (self._hold(0, 1), Shortcut(1, "Ctrl+A", "a", "App", 3.0))
        value, layout = self._eval(positions, shortcuts, [0, 1, -1])
        self.assertEqual(value, 0.0)
        from evolution.acceptance import _momentary_only_thumb_clearance_report
        self.assertTrue(_momentary_only_thumb_clearance_report(layout)["acceptance_pass"])

    def test_both_side_momentary_frees_both_sides(self):
        positions = (
            Position(0, 0, 0.0, 0.0, "right", 0, 0.1, is_thumb=True),  # right hold -> L1
            Position(1, 0, 1.0, 0.0, "left", 0, 0.1, is_thumb=True),   # left hold -> L1
            Position(2, 1, 5.0, 2.0, "right", 0, 0.1, is_thumb=True),  # occupant
            Position(3, 1, 9.0, 9.0, "left", 1, 1.0),
        )
        shortcuts = (self._hold(0, 1), self._hold(1, 1), Shortcut(2, "Ctrl+A", "a", "App", 3.0))
        value, layout = self._eval(positions, shortcuts, [0, 1, 2, -1])
        self.assertEqual(value, 0.0)
        from evolution.acceptance import _momentary_only_thumb_clearance_report
        self.assertTrue(_momentary_only_thumb_clearance_report(layout)["acceptance_pass"])

    def test_violation_matches_acceptance_failure(self):
        positions = (
            Position(0, 0, 0.0, 0.0, "right", 0, 0.1, is_thumb=True),
            Position(1, 1, 5.0, 2.0, "right", 0, 0.1, is_thumb=True),
            Position(2, 1, 9.0, 9.0, "left", 1, 1.0),
        )
        shortcuts = (self._hold(0, 1), Shortcut(1, "Ctrl+A", "a", "App", 3.0))
        value, layout = self._eval(positions, shortcuts, [0, 1, -1])
        from evolution.acceptance import _momentary_only_thumb_clearance_report
        self.assertGreater(value, 0.0)
        self.assertFalse(_momentary_only_thumb_clearance_report(layout)["acceptance_pass"])


class TestHardConstraintConfigShape(unittest.TestCase):
    def test_default_config_contains_new_hard_constraints(self):
        hard = DEFAULT_CONFIG["fitness"]["hard_constraints"]
        self.assertIn("unsupported_duplicate", hard)
        self.assertIn("thumb_occupancy_restricted", hard)

    def test_config_v2_yaml_contains_new_hard_constraints(self):
        cfg = Config.load(os.path.join(os.path.dirname(__file__), "..", "config_v2.yaml"))
        hard = cfg.get("fitness.hard_constraints")
        self.assertIn("unsupported_duplicate", hard)
        self.assertIn("thumb_occupancy_restricted", hard)

    def test_violation_sub_weights_contain_new_terms(self):
        vw = DEFAULT_CONFIG["fitness"]["violation_sub_weights"]
        self.assertIn("unsupported_duplicate", vw)
        self.assertIn("thumb_occupancy_restricted", vw)
        cfg = Config.load(os.path.join(os.path.dirname(__file__), "..", "config_v2.yaml"))
        yaml_vw = cfg.get("fitness.violation_sub_weights")
        self.assertEqual(vw["unsupported_duplicate"], yaml_vw["unsupported_duplicate"])
        self.assertEqual(vw["thumb_occupancy_restricted"], yaml_vw["thumb_occupancy_restricted"])


if __name__ == "__main__":
    unittest.main()
