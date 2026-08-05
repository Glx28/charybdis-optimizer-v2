"""Dynamic layer-access genome canonicalization tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import numpy as np

from core import Layout, Position, Shortcut
from tests.legacy_factors import EffortFactor


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
            Position(6, 3, 12.0, 3.0, "right", 2, 0.9),  # scroll (non-thumb)
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
            Position(7, 4, 12.0, 3.0, "right", 2, 0.9),  # scroll (non-thumb)
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


if __name__ == "__main__":
    unittest.main()
