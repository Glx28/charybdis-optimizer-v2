"""Tests for the v2 Charybdis optimizer."""
import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import unittest

from core import Position, Shortcut, Layout, FitnessResult, UsageData
from fitness.evaluator import FitnessEvaluator
from fitness.factors.effort import EffortFactor
from fitness.factors.adjacency import AdjacencyFactor
from fitness.factors.finger_balance import FingerBalanceFactor
from fitness.factors.same_finger import SameFingerFactor
from fitness.factors.violation import ViolationFactor
from evolution.surrogate import LayoutSurrogate, SurrogateTrainer
from core.loader import load_shortcuts


class TestDataStructures(unittest.TestCase):
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
        canonical = {
            "layers": {
                "0": {
                    "keys": {
                        "8:2": {
                            "x": 8,
                            "y": 2,
                            "label": "J",
                            "behavior": "Key Press",
                            "parameter": "J",
                            "modifiers": [],
                        },
                        "4:4": {
                            "x": 4,
                            "y": 4,
                            "label": "Space",
                            "behavior": "Key Press",
                            "parameter": "Spacebar",
                            "modifiers": [],
                        },
                        "5:4": {
                            "x": 5,
                            "y": 4,
                            "label": "Alt",
                            "behavior": "Key Press",
                            "parameter": "LeftAlt",
                            "modifiers": [],
                        },
                        "7:5": {
                            "x": 7,
                            "y": 5,
                            "label": "Ret",
                            "behavior": "Key Press",
                            "parameter": "Return Enter",
                            "modifiers": [],
                        },
                    }
                }
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
            shortcuts = load_shortcuts(path, canonical)
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
        from core import LayerAccess
        positions = (
            Position(0, 1, 3.0, 4.0, "left", 0, 0.8, is_thumb=True),   # L1 left thumb (occupied)
            Position(1, 1, 7.0, 4.0, "right", 0, 0.8, is_thumb=True),  # L1 right thumb (free)
            Position(2, 1, 0.0, 0.0, "left", 1, 1.0),                  # L1 finger
        )
        shortcuts = (
            Shortcut(0, "Ctrl+A", "Select All", "app", 10.0),
            Shortcut(1, "Ctrl+C", "Copy", "app", 8.0),
        )
        genome = np.array([0, -1, -1], dtype=np.int32)
        frozen = np.array([False, False, False])
        
        # L1 accessed via left thumb momentary from L0
        layer_access = (LayerAccess(1, 0, 3.0, 4.0, "left", True, "Nav"),)
        layout = Layout(genome, positions, shortcuts, frozen, layer_access=layer_access)
        
        from fitness.factors.violation import ViolationFactor
        factor = ViolationFactor()
        penalty = factor._thumb_occupancy(layout)
        
        # Shortcut on left thumb should be penalized
        self.assertGreater(penalty, 0.0)
        
        # Move shortcut to right thumb (free) - penalty should be 0
        genome2 = np.array([-1, 0, -1], dtype=np.int32)
        layout2 = Layout(genome2, positions, shortcuts, frozen, layer_access=layer_access)
        penalty2 = factor._thumb_occupancy(layout2)
        self.assertEqual(penalty2, 0.0)
        
        # Remove layer access - no penalty even on left thumb
        layout3 = Layout(genome, positions, shortcuts, frozen)
        penalty3 = factor._thumb_occupancy(layout3)
        self.assertEqual(penalty3, 0.0)


if __name__ == "__main__":
    unittest.main()
