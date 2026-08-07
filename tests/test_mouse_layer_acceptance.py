"""Mouse-layer acceptance tier tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import numpy as np

from core import Layout, Position, Shortcut


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
        # pos 0: L0 left thumb — direct momentary hold to L1
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
            Position(9, 0, 2.0, 4.0, "left", 0, 0.5, is_thumb=True),
        )
        shortcuts = (
            # sid 0: L0->L1 direct momentary hold (required mouse entry path)
            Shortcut(0, "@toggle:L0->L1:toggle", "Toggle", "Layer Access", 10.0,
                     "layer_access", is_layer_access=True, access_target_layer=1,
                     access_is_momentary=True),
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
            # sid 9: separate L0 toggle keeps the external toggle requirement
            Shortcut(9, "@toggle:L0->L1:toggle", "Toggle", "Layer Access", 10.0,
                     "layer_access", is_layer_access=True, access_target_layer=1,
                     access_is_momentary=False),
        )
        frozen = np.array([False, False, False, False, False, False, False, True, True, False], dtype=np.bool_)
        if include_mouse:
            genome = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
        else:
            # No mouse buttons placed — positions 1-5 empty
            genome = np.array([0, -1, -1, -1, -1, -1, -1, 7, 8, 9], dtype=np.int32)
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

    def test_primary_mouse_button_order_is_checked(self):
        from evolution.acceptance import _mouse_button_order_report
        report = _mouse_button_order_report(self._build_mouse_layout(include_mouse=True))
        self.assertFalse(report["acceptance_pass"], "The fixture intentionally places MB1 right of MB2")

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


if __name__ == "__main__":
    unittest.main()
