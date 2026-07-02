import os
import yaml
from copy import deepcopy
from typing import Dict, Any

DEFAULT_CONFIG = {
    "evolution": {
        "pop_size": 500,
        "n_generations": 30000,
        "crossover_prob": 0.7,
        "mutation_prob": 0.2,
        "eliminate_duplicates": False,
        "seed": 42,
        "inject_seed": True,
    },
    "surrogate": {
        "enabled": True,
        "initial_exact_samples": 1000,
        "exact_eval_every": 100,
        "retrain_every": 500,
        "train_epochs": 30,
        "retrain_epochs": 10,
        "batch_size": 1024,
        "max_retrain_samples": 20000,
        "hidden_dim": 256,
        "embedding_dim": 32,
    },
    "training": {
        "require_cuda": True,
        "require_gpu_primary": True,
        "allow_cpu_exact_validation": True,
    },
    "exact_eval": {
        "batch_size": 50,
        "use_numba": True,
        "validate_parity": False,
        "parity_samples": 16,
        "parity_tolerance": 1e-4,
    },
    "fitness": {
        "weights": {
            "effort": 1.0,
            "adjacency": 1.5,
            "finger_balance": 0.8,
            "same_finger": 2.0,
            "violations": 120.0,
            "workflow_coherence": 30.0,
            "app_coherence": 5.0,
            "trackball_proximity": 5.0,
            "familiarity": 12.0,
            "layer_similarity": 8.0,
            "everything_layer": 8.0,
        },
        "shortcut_importance_overrides": {
            "Win+S": 12.0,
            "LeftArrow": 4.0,
            "RightArrow": 4.0,
            "UpArrow": 4.0,
            "DownArrow": 4.0,
        },
        "missing_important_threshold": 6.0,
        "violation_sub_weights": {
            "duplicate": 10.0,
            "l0_displacement": 50.0,
            "missing_important": 500.0,
            "cross_layer_duplicate": 25.0,
            "group_split": 200.0,
            "thumb_occupancy": 30000.0,
            "arrow_order": 10000.0,
            "hand_bias": 25000.0,
            "mouse_layer_access": 25000.0,
            "arrow_scattered": 50000.0,
            "mouse_scattered": 500.0,
            "layer7_access": 50000.0,
            "duplicate_value_gap": 500.0,
            "access_layout": 3000.0,
            "raw_keyboard_completion_norwegian": 80000.0,
            "dynamic_mouse_layer": 200000.0,
            "empty_position": 5.0,
            "layer_reachability": 50000.0,
            "layer_depth_penalty": 2000.0,
            "natural_mouse_layer_exists": 200000.0,
        },
        "hard_constraints": [
            "missing_important",
            "layer7_access",
            "natural_mouse_layer_exists",
            "layer_reachability",
        ],
    },
    "output": {
        "build_dir": "build",
        "checkpoint_interval": 500,
        "verbose": True,
    },
    "profiling": {
        "enabled": False,
    },
}

class Config:
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    @property
    def raw(self):
        return self._data.copy()
    
    def get(self, key: str, default=None):
        parts = key.split(".")
        data = self._data
        for part in parts:
            if isinstance(data, dict) and part in data:
                data = data[part]
            else:
                return default
        return data
    
    def to_dict(self):
        return self._data.copy()
    
    @classmethod
    def load(cls, path: str) -> "Config":
        if not os.path.exists(path):
            return cls(deepcopy(DEFAULT_CONFIG))
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        merged = _merge_dicts(deepcopy(DEFAULT_CONFIG), data)
        return cls(merged)
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False)

def _merge_dicts(base: Dict, override: Dict) -> Dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
