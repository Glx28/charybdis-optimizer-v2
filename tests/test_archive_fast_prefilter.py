import json
from pathlib import Path

import numpy as np

from evolution.custom_ga import (
    _archive_cluster_precheck_reason,
    _cluster_count_arrays,
    _cluster_counts_fast,
    _count_clusters_together,
    _completion_cluster_pass_fast,
    _has_unsupported_duplicate,
)
from tools._common import load_layout
from evolution.completion_cluster import completion_order, is_raw_completion_base, analyze_completion_cluster
from evolution.group_shapes import NORWEGIAN_CLUSTER_OFFSETS


def test_unsupported_duplicate_prefilter_matches_mutable_and_frozen_cases():
    disallowed = np.asarray([False, True, False], dtype=np.bool_)
    frozen = np.asarray([False, False, True, True], dtype=np.bool_)

    assert not _has_unsupported_duplicate(
        np.asarray([1, -1, -1, -1]), disallowed, frozen,
    )
    assert _has_unsupported_duplicate(
        np.asarray([1, 1, -1, -1]), disallowed, frozen,
    )
    # A duplicate confined to frozen positions is ignored by the report rule.
    assert not _has_unsupported_duplicate(
        np.asarray([-1, -1, 1, 1]), disallowed, frozen,
    )
    # One frozen copy plus a mutable copy remains an unsupported duplicate.
    assert _has_unsupported_duplicate(
        np.asarray([1, -1, 1, -1]), disallowed, frozen,
    )


def test_supported_or_non_duplicate_sids_are_not_prefiltered():
    disallowed = np.asarray([False, False, True], dtype=np.bool_)
    frozen = np.zeros(4, dtype=np.bool_)

    assert not _has_unsupported_duplicate(
        np.asarray([1, 1, -1, -1]), disallowed, frozen,
    )
    assert not _has_unsupported_duplicate(
        np.asarray([2, -1, -1, -1]), disallowed, frozen,
    )


def test_fast_cluster_metrics_match_full_report_for_random_genomes():
    layout = load_layout()
    starts, sids, dx, dy, relative = _cluster_count_arrays(layout)
    pos_layer = np.asarray([p.layer for p in layout.positions], dtype=np.int32)
    pos_x = np.asarray([p.x for p in layout.positions], dtype=np.float32)
    pos_y = np.asarray([p.y for p in layout.positions], dtype=np.float32)
    rng = np.random.default_rng(917)
    genomes = [np.asarray(layout.genome, dtype=np.int32)]
    genomes.extend(
        rng.integers(-1, layout.n_shortcuts, size=len(layout.genome), dtype=np.int32)
        for _ in range(12)
    )

    for genome in genomes:
        expected = _count_clusters_together(layout.clone_with(genome=genome))
        actual = _cluster_counts_fast(
            genome, pos_layer, pos_x, pos_y, layout.n_shortcuts,
            starts, sids, dx, dy, relative,
        )
        assert actual == expected


def test_completion_cluster_fast_prefilter_matches_full_report():
    layout = load_layout()
    shortcut_order = np.asarray([
        completion_order(shortcut) if is_raw_completion_base(shortcut) else 0
        for shortcut in layout.shortcuts
    ], dtype=np.int32)
    pos_layer = np.asarray([p.layer for p in layout.positions], dtype=np.int32)
    pos_x = np.asarray([p.x for p in layout.positions], dtype=np.float32)
    pos_y = np.asarray([p.y for p in layout.positions], dtype=np.float32)
    pos_frozen = np.asarray([p.is_frozen for p in layout.positions], dtype=np.bool_)
    offsets = np.zeros((6, 2), dtype=np.float32)
    for order, (dx, dy) in NORWEGIAN_CLUSTER_OFFSETS.items():
        offsets[order] = (dx, dy)

    seed_path = Path("seeds/gen25k_corrected_seed.json")
    genomes = [layout.genome.astype(np.int32)]
    if seed_path.exists():
        genomes.append(np.asarray(json.loads(seed_path.read_text())["genome"], dtype=np.int32))
    rng = np.random.default_rng(1031)
    genomes.extend(
        rng.integers(-1, layout.n_shortcuts, size=layout.n_positions, dtype=np.int32)
        for _ in range(20)
    )

    for genome in genomes:
        expected = bool(
            analyze_completion_cluster(layout.clone_with(genome=genome))
            .get("acceptance_pass", False)
        )
        actual = _completion_cluster_pass_fast(
            genome, shortcut_order, pos_layer, pos_x, pos_y, pos_frozen, offsets,
        )
        assert bool(actual) == expected


def test_archive_cluster_prefilter_only_rejects_proven_monotonic_losses():
    incumbent = {
        "optimizer_side_pass": True,
        "norwegian_completion_cluster_pass": True,
        "clusters_together": 58,
        "clusters_total": 59,
        "clusters_order_ok": 29,
        "clusters_ordered_total": 29,
    }
    assert _archive_cluster_precheck_reason((57, 29, 59, 29), True, incumbent) == (
        "semantic_cluster_monotonicity"
    )
    assert _archive_cluster_precheck_reason((58, 28, 59, 29), True, incumbent) == (
        "semantic_cluster_monotonicity"
    )
    assert _archive_cluster_precheck_reason((58, 29, 59, 29), True, incumbent) is None
    assert _archive_cluster_precheck_reason((57, 29, 59, 29), False, incumbent) == (
        "norwegian_completion_cluster"
    )
    incumbent["norwegian_completion_cluster_pass"] = False
    assert _archive_cluster_precheck_reason((57, 29, 59, 29), True, incumbent) is None
