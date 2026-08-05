"""Semantic mutation (Numba) tests."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import numpy as np

from core import Layout, Position, Shortcut
from evolution import SwapMutation


class TestSemanticMutationNumba(unittest.TestCase):
    def test_numba_place_sids_matches_python(self):
        from evolution import _numba_place_sids
        n = 20
        genome = np.arange(n, dtype=np.int32)
        pos_map = np.full(n, -1, dtype=np.int32)
        for p, sid in enumerate(genome):
            pos_map[sid] = p
        sids = np.array([0, 1, 2], dtype=np.int32)
        targets = np.array([10, 11, 12], dtype=np.int32)

        py_genome = genome.copy()
        current_positions = [int(pos_map[sid]) for sid in sids]
        target_set = set(targets.tolist())
        displaced = [int(py_genome[pos]) for pos in targets]
        for sid, pos in zip(sids, targets):
            py_genome[pos] = sid
        fill_positions = [p for p in current_positions if p not in target_set]
        fill_values = [s for s in displaced if s not in sids.tolist()]
        for pos, sid in zip(fill_positions, fill_values):
            py_genome[pos] = sid
        for pos in fill_positions[len(fill_values):]:
            py_genome[pos] = -1

        nb_genome = genome.copy()
        ok = _numba_place_sids(nb_genome, sids, targets, pos_map)
        self.assertTrue(ok)
        np.testing.assert_array_equal(nb_genome, py_genome)

    def _build_semantic_mutation_layout(self):
        """Layout exercising semantic mutations: group, L7 access, apps."""
        positions = []
        for i in range(12):
            positions.append(Position(
                i, 1, float(i % 6), float(i // 6),
                "left" if i < 6 else "right", 1, 1.0,
                is_frozen=(i == 0),
            ))
        positions = tuple(positions)
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
            Shortcut(7, "@access:L7:hold", "L7 hold", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=7, access_is_momentary=True),
            Shortcut(8, "@access:L7:toggle", "L7 toggle", "Layer Access", 5.0,
                     is_layer_access=True, access_target_layer=7, access_is_momentary=False),
            Shortcut(9, "AppA1", "Action A1", "AppA", 5.0),
            Shortcut(10, "AppA2", "Action A2", "AppA", 5.0),
            Shortcut(11, "AppB1", "Action B1", "AppB", 5.0),
        ])
        frozen = np.array([False] * 12, dtype=np.bool_)
        frozen[0] = True
        # Place a non-group, non-access sid at the frozen position so group
        # overwrite does not displace values back into it.
        genome = np.arange(12, dtype=np.int32)
        genome[0] = 11
        genome[11] = 0
        layout = Layout(genome, positions, shortcuts, frozen)
        return layout

    def test_numba_semantic_dispatcher_preserves_invariants(self):
        """The Numba semantic dispatcher must not touch frozen positions or scatter groups."""
        layout = self._build_semantic_mutation_layout()
        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.3,
            group_overwrite_prob=0.3,
            optional_arrow_drop_prob=0.1,
            bulk_assign_prob=0.1,
            cluster_app_prob=0.3,
            random_assign_prob=0.0,
            effort_swap_prob=0.0,
            smart_duplicate_prob=0.0,
        )
        np.random.seed(42)
        random.seed(42)
        pop = np.tile(layout.genome.astype(np.int32), (200, 1))
        out = mutation._do(None, pop.copy())

        # Frozen position 0 must be unchanged.
        self.assertTrue(np.all(out[:, 0] == layout.genome[0]))

    def test_numba_vs_python_semantic_mutation_rates(self):
        """Numba semantic dispatcher and Python fallback mutate a similar fraction."""
        import evolution
        layout = self._build_semantic_mutation_layout()
        mutation = SwapMutation(
            prob=0.0,
            frozen_mask=layout.frozen_mask,
            layout=layout,
            mouse_workflow_prob=0.0,
            l7_access_prob=0.4,
            group_overwrite_prob=0.4,
            optional_arrow_drop_prob=0.2,
            bulk_assign_prob=0.2,
            cluster_app_prob=0.4,
            random_assign_prob=0.0,
            effort_swap_prob=0.0,
            smart_duplicate_prob=0.0,
        )
        n = 500
        np.random.seed(123)
        random.seed(123)
        pop = np.tile(layout.genome.astype(np.int32), (n, 1))
        pop[pop == 11] = -1

        out_numba = mutation._do(None, pop.copy())
        changed_numba = np.sum(np.any(out_numba != pop, axis=1))

        orig = evolution.NUMBA_AVAILABLE
        try:
            evolution.NUMBA_AVAILABLE = False
            np.random.seed(123)
            random.seed(123)
            out_py = mutation._do(None, pop.copy())
        finally:
            evolution.NUMBA_AVAILABLE = orig
        changed_py = np.sum(np.any(out_py != pop, axis=1))

        self.assertGreater(changed_numba, n * 0.1)
        self.assertGreater(changed_py, n * 0.1)
        self.assertLess(abs(changed_numba - changed_py) / n, 0.25)


if __name__ == "__main__":
    unittest.main()
