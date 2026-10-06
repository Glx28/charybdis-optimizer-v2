from types import SimpleNamespace

from tools.exact_swap_search import swap_pairs


def test_swap_pairs_omit_frozen_positions_and_identical_assignments():
    layout = SimpleNamespace(positions=[
        SimpleNamespace(is_frozen=False),
        SimpleNamespace(is_frozen=True),
        SimpleNamespace(is_frozen=False),
        SimpleNamespace(is_frozen=False),
        SimpleNamespace(is_frozen=False),
    ])

    pairs = swap_pairs([1, 9, 2, 2, 3], layout)

    assert pairs == [(0, 2), (0, 3), (0, 4), (2, 4), (3, 4)]
    assert all(1 not in pair for pair in pairs)
