from types import SimpleNamespace

import numpy as np

from tools.repair_gen25k_seed import SeedRepair


def _repairer(layers, efforts, layer_use):
    repairer = SeedRepair.__new__(SeedRepair)
    repairer.genome = np.ones(len(layers), dtype=np.int32)
    repairer.positions = [
        SimpleNamespace(layer=layer, effort=effort, x=0, y=0)
        for layer, effort in zip(layers, efforts)
    ]
    repairer.layout = SimpleNamespace(
        usage_data=SimpleNamespace(by_layer_shortcut={"Ctrl+K": layer_use})
    )
    repairer.shortcut_by_sid = {1: SimpleNamespace(keys="Ctrl+K")}
    repairer.moves = []
    repairer.record = lambda idx, before, after, reason: repairer.moves.append(
        (idx, before, after, reason)
    )
    return repairer


def test_unsupported_duplicate_keeps_binding_specific_used_layer():
    repairer = _repairer([2, 8], [1.0, 0.0], {"2": 1, "8": 7})
    repairer.remove_unsupported_cross_layer_duplicates({
        "unsupported_duplicates": [{
            "sid": 1,
            "positions": [{"idx": 0, "layer": 2}, {"idx": 1, "layer": 8}],
        }],
    })

    assert repairer.genome.tolist() == [-1, 1]


def test_unsupported_duplicate_without_layer_data_keeps_shallower_layer():
    repairer = _repairer([4, 1], [0.0, 1.0], {})
    repairer.remove_unsupported_cross_layer_duplicates({
        "unsupported_duplicates": [{
            "sid": 1,
            "positions": [{"idx": 0, "layer": 4}, {"idx": 1, "layer": 1}],
        }],
    })

    assert repairer.genome.tolist() == [-1, 1]
