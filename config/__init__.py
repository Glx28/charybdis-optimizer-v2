import os
import yaml
from copy import deepcopy
from typing import Dict, Any

DEFAULT_CONFIG = {
    "evolution": {
        "pop_size": 1500,
        "n_generations": 500000,
        "crossover_prob": 0.7,
        "mutation_prob": 0.2,
        "group_overwrite_prob": 0.15,
        "mouse_workflow_prob": 0.20,
        "l7_access_prob": 0.03,
        "random_assign_prob": 0.08,
        "bulk_assign_prob": 0.04,
        "optional_arrow_drop_prob": 0.04,
        "cluster_app_prob": 0.20,
        "effort_swap_prob": 0.06,
        "smart_duplicate_prob": 0.20,
        "access_thumb_bias_prob": 0.30,
        "return_toggle_repair_prob": 0.10,
        "toggle_own_layer_bias_prob": 0.12,
        "momentary_reuse_repair_prob": 0.10,
        "arrow_cluster_prob": 0.06,
        "raw_completion_cluster_prob": 0.06,
        "mouse_hold_conflict_repair_prob": 0.05,
        "thumb_occupancy_repair_prob": 0.05,
        "same_layer_duplicate_repair_prob": 0.05,
        "l0_hold_completion_prob": 0.08,
        "unsupported_duplicate_repair_prob": 0.05,
        "eliminate_duplicates": False,
        "feasibility_first_selection": True,
        "seed": 42,
        "inject_seed": True,
    },
    "surrogate": {
        "enabled": True,
        "initial_exact_samples": 5000,
        "exact_eval_every": 300,
        "retrain_every": 300,
        "train_epochs": 100,
        "retrain_epochs": 40,
        "batch_size": 8192,
        "max_retrain_samples": 9000,
        "hidden_dim": 256,
        "embedding_dim": 32,
        "mini_eval_fraction": 0.1,
    },
    "training": {
        "require_cuda": True,
        "require_gpu_primary": True,
        "allow_cpu_exact_validation": False,
    },
    "exact_eval": {
        "batch_size": 100,
        "validate_parity": False,
        "parity_samples": 16,
        "parity_tolerance": 1e-4,
    },
    "fitness": {
        "weights": {
            "effort": 8.0,
            "adjacency": 1.5,
            "finger_balance": 0.8,
            "same_finger": 2.0,
            "violations": 120.0,
            "workflow_coherence": 20.0,
            "app_coherence": 5.0,
            "trackball_proximity": 5.0,
            "familiarity": 6.0,
            "layer_similarity": 4.0,
            "everything_layer": 4.0,
        },
        "shortcut_importance_overrides": {
            "Win+S": 12.0,
            "LeftArrow": 4.0,
            "RightArrow": 4.0,
            "UpArrow": 4.0,
            "DownArrow": 4.0,
            "Ctrl+S": 14.0,
            "Ctrl+A": 6.0,
            "Ctrl+Y": 12.0,
            "Shift+Enter": 9.5,
        },
        "missing_important_threshold": 6.0,
        "toggle_effort_multiplier": 4.0,
        "violation_sub_weights": {
            # Must mirror config_v2.yaml's violation_sub_weights. These
            # defaults only apply when a key is missing from the yaml; keeping
            # them in sync prevents silent fallback to stale magnitudes.
            "duplicate": 35.0,
            "l0_displacement": 50.0,
            "missing_important": 200000.0,
            "cross_layer_duplicate": 40000.0,
            "group_split": 200.0,
            "thumb_occupancy": 50000.0,
            "arrow_order": 10000.0,
            "hand_bias": 25000.0,
            "mouse_layer_access": 50000.0,
            "arrow_scattered": 400.0,
            "mouse_scattered": 500.0,
            "layer7_access": 50000.0,
            "duplicate_value_gap": 45000.0,
            "access_layout": 5000.0,
            "raw_keyboard_completion_norwegian": 20.0,
            "dynamic_mouse_layer": 50.0,
            "empty_position": 200.0,
            "layer_reachability": 50000.0,
            "layer_depth_penalty": 50000.0,
            "natural_mouse_layer_exists": 200000.0,
            "toggle_back_to_l0": 200000.0,
            "mouse_hold_position_conflict": 50000.0,
            "mouse_layer_depth_penalty": 50000.0,
            "same_layer_duplicate": 200000.0,
            "momentary_key_reuse": 50000.0,
            "unsupported_duplicate": 200000.0,
            "thumb_occupancy_restricted": 50000.0,
            "norwegian_completion_cluster": 200000.0,
            "mouse_right_thumb_global": 200000.0,
            "mutable_raw_arrows": 200000.0,
            "mouse_button_order": 200000.0,
            "mouse_layer_l0_hold": 300000.0,
        },
        "hard_constraints": [
            "missing_important",
            "layer7_access",
            "natural_mouse_layer_exists",
            "layer_reachability",
            "toggle_back_to_l0",
            "same_layer_duplicate",
            "unsupported_duplicate",
            "thumb_occupancy_restricted",
            "norwegian_completion_cluster",
            "mouse_right_thumb_global",
            "mutable_raw_arrows",
            "mouse_button_order",
            "mouse_layer_l0_hold",
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
