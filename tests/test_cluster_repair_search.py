import numpy as np
from types import SimpleNamespace

from tools.cluster_repair_search import _evaluate, _move_cluster_to_layer


class TinyLayout:
    def __init__(self, genome, positions, shortcuts):
        self.genome = np.asarray(genome, dtype=np.int32)
        self.positions = positions
        self.shortcuts = shortcuts
        self.n_shortcuts = len(shortcuts)

    def clone_with(self, genome):
        return TinyLayout(genome, self.positions, self.shortcuts)


def _position(idx, layer, x=0.0, y=0.0):
    return SimpleNamespace(gene_idx=idx, layer=layer, x=x, y=y, is_frozen=False)


def test_cluster_repair_rechecks_access_displacement_hard_constraints():
    layout = TinyLayout(
        [0, 1, 2, -1], [
            _position(0, 1), _position(1, 1),
            _position(2, 2), _position(3, 2),
        ],
        [
            SimpleNamespace(is_layer_access=False, category="general", is_l0_only=False),
            SimpleNamespace(is_layer_access=False, category="general", is_l0_only=False),
            SimpleNamespace(is_layer_access=True, category="layer_access", is_l0_only=False),
            SimpleNamespace(is_layer_access=False, category="general", is_l0_only=False),
        ],
    )
    cluster = {"members": [{"sid": 0}, {"sid": 1}]}
    moved = _move_cluster_to_layer(layout, cluster, 2, np.random.default_rng(1))
    assert moved is not None
    evaluator = SimpleNamespace(evaluate=lambda _: SimpleNamespace(
        constraints=np.asarray([1.0]),
    ))
    candidate, _, _, _, _ = _evaluate(layout, evaluator, moved)
    assert candidate is None


def test_cluster_repair_rejects_hard_constraint_violations():
    layout = TinyLayout([0], [_position(0, 1)], [SimpleNamespace()])
    evaluator = SimpleNamespace(evaluate=lambda _: SimpleNamespace(
        constraints=np.asarray([1.0]),
    ))
    candidate, _, report, together, ordered = _evaluate(layout, evaluator, layout.genome)
    assert candidate is None
    assert report == {}
    assert together == ordered == 0


def test_cluster_repair_places_unassigned_members_into_empty_targets_first():
    layout = TinyLayout(
        [0, -1, 2, -1],
        [_position(0, 1), _position(1, 1), _position(2, 2), _position(3, 2)],
        [
            SimpleNamespace(is_layer_access=False, category="general", is_l0_only=False),
            SimpleNamespace(is_layer_access=False, category="general", is_l0_only=False),
            SimpleNamespace(is_layer_access=False, category="general", is_l0_only=False),
        ],
    )
    cluster = {"members": [{"sid": 0}, {"sid": 1}]}
    moved = _move_cluster_to_layer(layout, cluster, 2, np.random.default_rng(0))
    assert moved is not None
    assert set(moved.tolist()) == {0, 1, 2, -1}
    assert moved[2] == 0 or moved[3] == 0
    assert moved[2] == 1 or moved[3] == 1
    assert moved[0] == 2


def test_ordered_cluster_repair_requires_the_declared_relative_shape():
    shortcuts = [
        SimpleNamespace(is_layer_access=False, category="general", is_l0_only=False),
        SimpleNamespace(is_layer_access=False, category="general", is_l0_only=False),
    ]
    cluster = {"members": [
        {"sid": 0, "order": 0, "dx": 0, "dy": 0},
        {"sid": 1, "order": 1, "dx": 0, "dy": 1},
    ]}
    source = [_position(0, 1), _position(1, 1, x=1)]
    horizontal_target = [_position(2, 2), _position(3, 2, x=1)]
    no_shape = TinyLayout([0, 1, -1, -1], source + horizontal_target, shortcuts)
    assert _move_cluster_to_layer(no_shape, cluster, 2, np.random.default_rng(1)) is None

    vertical_target = [_position(2, 2), _position(3, 2, y=1)]
    valid_shape = TinyLayout([0, 1, -1, -1], source + vertical_target, shortcuts)
    moved = _move_cluster_to_layer(valid_shape, cluster, 2, np.random.default_rng(1))
    assert moved is not None
    assert moved.tolist() == [-1, -1, 0, 1]
