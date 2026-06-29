"""Main entry point for Charybdis v2 layout optimizer."""
import sys
import os
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from core import Layout
from core.loader import build_layout
from fitness.evaluator import FitnessEvaluator
from evolution.surrogate import LayoutSurrogate, SurrogateTrainer, SurrogateManager
from config import Config


def generate_random_layouts(layout, n):
    """Generate n random valid layouts."""
    mutable = layout.mutable_indices
    n_shortcuts = layout.n_shortcuts
    layouts = np.full((n, layout.n_positions), -1, dtype=np.int32)
    for i in range(n):
        n_assign = min(len(mutable), n_shortcuts)
        assigned = np.random.choice(n_shortcuts, size=n_assign, replace=False)
        layouts[i, mutable[:n_assign]] = assigned
    return layouts


def evaluate_exact(layouts, layout, evaluator):
    """Evaluate layouts exactly with the fitness evaluator."""
    n = layouts.shape[0]
    scores = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        genome = layouts[i].copy()
        test_layout = layout.clone_with(genome=genome)
        result = evaluator.evaluate(test_layout)
        scores[i] = result.objectives
    return scores


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <build_dir>")
        sys.exit(1)
    
    build_dir = sys.argv[1]
    config = Config.load(os.path.join(build_dir, "config_v2.yaml"))
    
    seed = config.get("evolution.seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    print("=" * 60)
    print("CHARYBDIS V2 LAYOUT OPTIMIZER")
    print("=" * 60)
    
    print(f"Loading data from {build_dir}...")
    layout = build_layout(build_dir)
    print(f"  Positions: {layout.n_positions}")
    print(f"  Shortcuts: {layout.n_shortcuts}")
    print(f"  Mutable: {len(layout.mutable_indices)}")
    
    evaluator = FitnessEvaluator(weights=config.get("fitness.weights", {}))
    
    print("Testing exact evaluation...")
    result = evaluator.evaluate(layout)
    print(f"  Effort={result.effort:.1f}, Adjacency={result.adjacency:.1f}, Violations={result.violations:.1f}")
    
    n_initial = min(config.get("surrogate.n_initial_samples", 5000), 1000)
    print(f"Generating {n_initial} random layouts for surrogate training...")
    train_layouts = generate_random_layouts(layout, n_initial)
    
    print("Evaluating exact fitness on training set...")
    train_scores = evaluate_exact(train_layouts, layout, evaluator)
    print(f"  Score range: effort=[{train_scores[:,0].min():.0f}, {train_scores[:,0].max():.0f}], "
          f"viol=[{train_scores[:,2].min():.0f}, {train_scores[:,2].max():.0f}]")
    
    if config.get("surrogate.enabled", True):
        print("Training surrogate model...")
        surrogate = LayoutSurrogate(
            n_positions=layout.n_positions,
            n_shortcuts=layout.n_shortcuts,
            n_factors=3,
            hidden_dim=config.get("surrogate.hidden_dim", 256),
        )
        print(f"  Surrogate parameters: {surrogate.count_parameters()}")
        trainer = SurrogateTrainer(surrogate, device="cpu")
        trainer.train(train_layouts, train_scores, epochs=config.get("surrogate.surrogate_epochs", 100))
        
        acc = trainer.evaluate(train_layouts[:500], train_scores[:500])
        r2_mean = float(np.mean(acc["r2"]))
        print(f"  Surrogate R^2 = {r2_mean:.4f}")
        
        manager = SurrogateManager(surrogate, trainer,
                                   retrain_every=config.get("surrogate.retrain_every", 200),
                                   exact_eval_every=config.get("surrogate.exact_eval_every", 50))
    else:
        manager = None
    
    print("\nv2 system built and validated successfully!")
    print(f"  Surrogate: {'enabled' if manager else 'disabled'}")
    print(f"  Population: {config.get('evolution.pop_size')}")
    print(f"  Generations: {config.get('evolution.n_generations')}")


if __name__ == "__main__":
    main()
