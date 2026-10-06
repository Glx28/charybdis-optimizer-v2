"""Focused production contracts, checked against both exact kernels."""
import json
from pathlib import Path

import numpy as np
import pytest

from config import Config, DEFAULT_CONFIG
from core import Layout, Position, Shortcut
from core.loader import build_layout
from fitness.kernel import _single_genome, precompute


CONSTRAINTS = ["layer_access_thumb_preference", "same_layer_duplicate"]
WINDOWS_KEYS = ("Win+H", "Win+Tab", "Alt+Tab", "Win+Alt+Space")


def _arrays(layout):
    weights = {key: 0.0 for key in DEFAULT_CONFIG["fitness"]["weights"]}
    weights["effort"] = 1.0
    violations = {key: 0.0 for key in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
    return precompute(layout, weights, violations, 6.0, np.ones(3, dtype=np.float32),
                      hard_constraints=CONSTRAINTS)


def _check(layout, genomes, expected_constraints, backend):
    arrays = _arrays(layout)
    batch = np.asarray(genomes, dtype=np.int32)
    if backend == "cuda":
        from fitness.cuda_kernel import cuda_available, evaluate_batch_cuda
        if not cuda_available():
            pytest.fail("CUDA required for production parity checks")
        objectives, constraints = evaluate_batch_cuda(batch, arrays)
        oracle = [_single_genome(genome, *arrays) for genome in batch]
        np.testing.assert_allclose(objectives, [row[0] for row in oracle], rtol=1e-4, atol=0.01)
        np.testing.assert_allclose(constraints, [row[1] for row in oracle], rtol=1e-4, atol=0.001)
    else:
        results = [_single_genome(genome, *arrays) for genome in batch]
        objectives = np.asarray([row[0] for row in results])
        constraints = np.asarray([row[1] for row in results])
    np.testing.assert_allclose(constraints, expected_constraints, rtol=1e-4, atol=0.001)
    return objectives


def _access_layout():
    positions = (
        Position(0, 0, 3, 4, "left", 0, 0, is_thumb=True),
        Position(1, 0, 8, 4, "right", 0, 0, is_thumb=True),
        Position(2, 0, 2, 2, "left", 1, 0),
        Position(3, 1, 3, 4, "left", 0, 0, is_thumb=True),
        Position(4, 1, 8, 4, "right", 0, 0, is_thumb=True),
        Position(5, 1, 9, 2, "right", 1, 0),
    )
    shortcuts = (
        Shortcut(0, "@access:L1:hold", "L1 hold", "Layer Access", 10,
                 is_layer_access=True, access_target_layer=1, access_is_momentary=True),
        Shortcut(1, "@access:L2:hold", "L2 hold", "Layer Access", 10,
                 is_layer_access=True, access_target_layer=2, access_is_momentary=True),
        Shortcut(2, "@scroll:hold", "Scroll mode", "Layer Access", 10,
                 is_layer_access=True, access_target_layer=2, access_is_momentary=True),
        Shortcut(3, "@scroll:other", "Scroll mode", "Layer Access", 10,
                 is_layer_access=True, access_target_layer=2, access_is_momentary=True),
    )
    return Layout(np.full(6, -1, dtype=np.int32), positions, shortcuts, np.zeros(6, dtype=bool))


@pytest.mark.parametrize("backend", ["numba", "cuda"])
def test_momentary_access_contract(backend):
    layout = _access_layout()
    genomes = [
        [0, -1, -1, -1, -1, -1],  # thumb
        [-1, -1, 0, -1, -1, -1],  # finger
        [0, -1, -1, 1, -1, -1],   # nested same side
        [0, -1, -1, -1, 1, -1],   # nested opposite side
        [0, -1, -1, -1, -1, 2],   # Scroll finger is exempt
        [0, -1, -1, 2, -1, -1],   # Scroll same side is exempt
    ]
    _check(layout, genomes, [[0, 0], [12, 0], [0, 0],
                             [0, 0], [0, 0], [0, 0]], backend)


def test_toggle_effort_multiplier_changes_production_access_path_score():
    from dataclasses import replace

    base = _access_layout()
    shortcuts = base.shortcuts + (Shortcut(4, "Ctrl+C", "Copy", "Editor", 50),)
    positions = list(base.positions)
    positions[0] = replace(positions[0], effort=2.0)
    layout = Layout(base.genome, tuple(positions), shortcuts, base.frozen_mask)
    genome = np.array([0, -1, -1, -1, -1, 4], dtype=np.int32)
    weights = {key: 0.0 for key in DEFAULT_CONFIG["fitness"]["weights"]}
    weights["violations"] = 1.0
    violation_weights = {key: 0.0 for key in DEFAULT_CONFIG["fitness"]["violation_sub_weights"]}
    violation_weights["access_layout"] = 1.0

    def score(access, multiplier):
        candidate = Layout(genome, layout.positions, (access,) + shortcuts[1:], layout.frozen_mask)
        arrays = precompute(
            candidate, weights, violation_weights, 6.0, np.ones(3, dtype=np.float32),
            hard_constraints=[], toggle_effort_multiplier=multiplier,
        )
        return float(_single_genome(genome, *arrays)[0][2])

    toggle = replace(shortcuts[0], access_is_momentary=False)
    hold = shortcuts[0]
    assert score(toggle, 4.0) > score(toggle, 1.0)
    assert score(hold, 4.0) == pytest.approx(score(hold, 1.0))


@pytest.mark.parametrize("backend", ["numba", "cuda"])
@pytest.mark.parametrize("momentary", [True, False])
def test_scroll_capability_variants_are_same_layer_duplicates(backend, momentary):
    from dataclasses import replace
    from evolution.acceptance import _no_same_layer_duplicates_report

    base = _access_layout()
    shortcuts = tuple(replace(shortcut, access_is_momentary=momentary) if shortcut.sid == 3 else shortcut
                      for shortcut in base.shortcuts)
    genome = np.array([0, -1, -1, 2, -1, 3], dtype=np.int32)
    layout = Layout(genome, base.positions, shortcuts, base.frozen_mask)
    _check(layout, [genome], [[0, 1]], backend)
    assert not _no_same_layer_duplicates_report(layout)["acceptance_pass"]


@pytest.mark.parametrize("backend", ["numba", "cuda"])
@pytest.mark.parametrize("layer,reachable", [(1, True), (1, False), (7, True), (0, True)])
def test_sparse_layer_occupancy_is_diagnostic_only(backend, layer, reachable):
    # Frozen zero-importance slots isolate occupancy from all other pressures.
    positions = (Position(0, 0, 3, 4, "left", 0, 0, is_thumb=True, is_frozen=True),) + tuple(
        Position(i + 1, layer, i, 2, "right", 1, 0, is_frozen=True) for i in range(20)
    )
    shortcuts = (
        Shortcut(0, "@access:toggle", "Toggle", "Access", 0,
                 is_layer_access=True, access_target_layer=layer),
    ) + tuple(Shortcut(i + 1, f"neutral{i}", "Neutral", "Test", 0) for i in range(20))
    layout = Layout(np.full(21, -1, dtype=np.int32), positions, shortcuts, np.ones(21, dtype=bool))
    counts = [0, 1, 19, 20]
    genomes = [[0 if reachable else -1] + list(range(1, count + 1)) + [-1] * (20 - count)
               for count in counts]
    objectives = _check(layout, genomes, np.zeros((4, 2)), backend)
    np.testing.assert_allclose(objectives[:, 0], [0, 0, 0, 0], rtol=1e-4, atol=0.01)


def test_production_shortcuts_hard_constraints_and_frozen_l7():
    config = Config.load("config_v2.yaml")
    layout = build_layout("data", config.raw["fitness"])
    shortcuts = {shortcut.keys: shortcut for shortcut in layout.shortcuts}
    for keys in WINDOWS_KEYS:
        assert keys in shortcuts
        assert shortcuts[keys].importance >= config.get("fitness.missing_important_threshold")
    assert set(CONSTRAINTS[:2]).issubset(config.get("fitness.hard_constraints"))
    assert "same_side_hold_flow" not in config.get("fitness.hard_constraints")
    assert config.get("fitness.violation_sub_weights.same_side_hold_flow") >= 1_000_000_000.0
    assert config.get("fitness.layer_access_thumb.same_side_penalty") > 0
    assert config.get("fitness.layer_access_thumb.opposite_side_reward") > 0
    assert config.get("fitness.sparse_layer_base_penalty") == 0
    assert config.get("fitness.sparse_layer_gap_penalty") == 0
    data = json.loads(Path("data/layout.json").read_text())
    assert data["l7_frozen"]
    # L7 contents live outside the training genome; mutations cannot change them.
    assert all(layout.positions[i].layer != 7 for i in layout.mutable_indices)
    assert all(pos.is_frozen and layout.frozen_mask[pos.gene_idx]
               for pos in layout.positions if pos.layer == 7)


@pytest.mark.parametrize("generation", [0, 3000, 10000])
def test_full_production_layout_cpu_cuda_parity(generation):
    from tools._common import load_evaluator

    evaluator = load_evaluator(require_cuda=True, generation=generation)
    layout = evaluator.reference_layout
    warmstart = json.loads(Path("build/v2_local_search_result.json").read_text())
    genome = np.asarray(warmstart["genome"], dtype=np.int32)
    # Exercise both complete and incomplete mouse layouts, including the
    # importance boost and missing-button coefficients omitted by tiny fixtures.
    incomplete = genome.copy()
    for index, sid in enumerate(incomplete):
        if sid >= 0 and layout.shortcuts[int(sid)].keys == "MB2":
            incomplete[index] = -1
    batch = np.stack([genome, incomplete])
    gpu_objectives, gpu_constraints = evaluator.evaluate_batch(batch)
    cpu = [evaluator.model.evaluate(candidate) for candidate in batch]
    np.testing.assert_allclose(gpu_objectives, [row[0] for row in cpu], rtol=1e-4, atol=0.01)
    np.testing.assert_allclose(gpu_constraints, [row[1] for row in cpu], rtol=1e-4, atol=0.001)


def test_windows_shortcut_report_uses_live_access_paths():
    from evolution.acceptance import _windows_shortcut_report

    base = _access_layout()
    shortcuts = base.shortcuts + tuple(
        Shortcut(i + 4, keys, keys, "Windows", 6) for i, keys in enumerate(WINDOWS_KEYS)
    )
    layout = Layout(np.array([0, 4, -1, 5, 6, 7], dtype=np.int32), base.positions,
                    shortcuts, base.frozen_mask)
    report = _windows_shortcut_report(layout)
    assert report["acceptance_pass"]
    assert report["placements"]["Win+H"][0]["access_path"] == []
    assert report["placements"]["Win+Tab"][0]["access_path"][0]["keys"] == "@access:L1:hold"
    genome = layout.genome.copy()
    genome[0] = -1
    missing = _windows_shortcut_report(layout.clone_with(genome=genome))
    assert not missing["acceptance_pass"]
    assert set(missing["missing_or_unreachable"]) == set(WINDOWS_KEYS[1:])


@pytest.mark.parametrize("frozen", [False, True])
def test_warmstart_sanitizer_preserves_same_layer_scroll_mode(frozen):
    from dataclasses import replace
    from evolution import SwapMutation

    base = _access_layout()
    shortcuts = tuple(replace(shortcut, access_target_layer=1) if shortcut.sid == 2 else shortcut
                      for shortcut in base.shortcuts)
    genome = np.array([0, -1, -1, 0, -1, 2], dtype=np.int32)
    mask = base.frozen_mask.copy()
    mask[3] = frozen
    layout = Layout(genome.copy(), base.positions, shortcuts, mask)
    mutation = SwapMutation(layout=layout, frozen_mask=layout.frozen_mask)
    assert mutation.sanitize_self_ref_momentary(genome) == (0 if frozen else 1)
    assert genome[3] == (0 if frozen else -1)  # ordinary hold cleared only when mutable
    assert genome[5] == 2   # pointer-mode hold is useful on its own layer


def test_acceptance_failure_alone_does_not_sanitize_feasible_warmstart():
    from evolution.custom_ga import _warmstart_needs_access_sanitizing

    # E.g. raw-arrow acceptance may fail while access constraints remain clean.
    assert not _warmstart_needs_access_sanitizing({
        "optimizer_side_pass": False,
        "acceptance_failed_checks": ["dynamic_mouse_layer_present"],
        "constraints": [0, 0, 0],
    })
    assert _warmstart_needs_access_sanitizing({
        "optimizer_side_pass": False,
        "constraints": [0, 2, 0],
    })


def test_atomic_injection_preserves_bindings_displaced_by_missing_group():
    from evolution import _numba_overwrite_group_as_unit

    genome = np.array([3, 4, 5, 0, -1, -1, -1, -1], dtype=np.int32)
    pos_map = np.array([-1, -1, -1, 0, 1, 2], dtype=np.int32)
    state = np.array([42], dtype=np.uint64)
    group_sids = np.array([[1, 2]], dtype=np.int32)
    group_sizes = np.array([2], dtype=np.int32)
    anchors = np.array([[0, 1]], dtype=np.int32)
    anchor_start = np.array([0, 1], dtype=np.int32)
    mutable = np.arange(8, dtype=np.int32)

    assert _numba_overwrite_group_as_unit(
        genome, pos_map, state, group_sids, group_sizes, anchors,
        anchor_start, 1, 6, mutable, 0,
    )
    assert genome[0:2].tolist() == [1, 2]
    assert np.count_nonzero(genome == 3) == 1
    assert np.count_nonzero(genome == 4) == 1


def test_milestone_gate_keeps_search_alive_when_contract_metrics_improve():
    from tools.validated_run import contract_gate_should_stop, generation_contract_review

    first = {"best_generation": 5, "failed_checks": ["dynamic_mouse_layer_present"],
             "clusters_together": 51, "clusters_total": 67,
             "clusters_order_ok": 24, "clusters_ordered_total": 28,
             "constraints": [0, 0]}
    current = {**first, "best_generation": 100, "clusters_together": 52}
    audit = {"generation": 5000, "run_contract_pass": False,
             "checks": {"optimizer_acceptance": False},
             "critical_split_clusters": ["undo_redo"], "ordered_relation_failures": ["before_after"]}
    review = generation_contract_review(first, current, audit, audit)
    assert any("cluster co-location improved" in item for item in review["improvements"])

    stop, stalls, _ = contract_gate_should_stop(review, audit, 0)
    assert not stop and stalls == 0


def test_schedule_refresh_reprices_archive_and_teacher_before_selection():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from core import FitnessResult
    from evolution.custom_ga import CustomGARunner

    runner = CustomGARunner.__new__(CustomGARunner)
    layout = _access_layout()
    runner.layout = layout
    runner.global_best_genome = layout.genome.copy()
    runner.global_best_generation = 6
    runner.global_best_exact = {"total_score": 1.0}
    runner.evaluator = Mock()
    scores = np.array([[2, 3, 4]], dtype=np.float32)
    constraints = np.array([[0, 1, 0]], dtype=np.float32)
    runner.evaluator.evaluate_batch.return_value = (scores, constraints)
    runner.evaluator.evaluate.return_value = FitnessResult(
        objectives=scores[0], constraints=constraints[0], factor_scores={}, total_score=9.0,
    )
    runner.evaluator.model = SimpleNamespace(semantic_cluster_multiplier=200.0)
    runner.semantic_multiplier_schedule = [[0, 0], [3000, 200]]
    runner.surrogate_manager = Mock(_retrain_future=None)
    runner._layout_reports = Mock(return_value=({}, {}, {}, {"optimizer_side_pass": False, "checks": {}}))
    runner._annotate_clusters = Mock()

    assert runner._apply_semantic_multiplier_schedule(0)
    runner.evaluator.set_semantic_cluster_multiplier.assert_called_once_with(0.0)
    runner.surrogate_manager.clear_exact_cache.assert_called_once()
    new_scores, new_constraints = runner._refresh_semantic_scores(layout.genome[None, :])
    assert runner.global_best_exact["total_score"] == 9.0
    assert runner.global_best_generation == 6
    np.testing.assert_array_equal(new_scores, scores)
    np.testing.assert_array_equal(new_constraints, constraints)
    runner.surrogate_manager.add_exact_evaluations.assert_called_once()
    runner.surrogate_manager.retrain.assert_called_once()


def test_surrogate_teacher_exact_refresh_reprices_population_and_checks_archive():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from evolution.custom_ga import CustomGARunner

    runner = CustomGARunner.__new__(CustomGARunner)
    runner.perf = None
    runner.n_factors = 3
    genomes = np.arange(12, dtype=np.int32).reshape(2, 6)
    exact_F = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    exact_G = np.asarray([[0, 1], [2, 0]], dtype=np.float32)
    runner.evaluator = Mock()
    runner.evaluator.evaluate_batch.return_value = (exact_F, exact_G)
    runner._maybe_update_global_from_batch = Mock()
    manager = Mock(exact_eval_every=10)
    manager.maybe_collect_retrain.return_value = False
    manager.should_retrain.return_value = False
    runner.surrogate_manager = manager

    scores, cv = runner._maybe_teacher_update(
        genomes, np.full((2, 3), 99, dtype=np.float32),
        np.full((2, 2), 99, dtype=np.float32), 10,
    )

    np.testing.assert_array_equal(scores, exact_F)
    np.testing.assert_array_equal(cv, exact_G)
    assert manager.add_exact_evaluations.call_count == 1
    added = manager.add_exact_evaluations.call_args.args
    np.testing.assert_array_equal(added[0], genomes)
    np.testing.assert_array_equal(added[1], exact_F)
    np.testing.assert_array_equal(added[2], exact_G)
    assert runner._maybe_update_global_from_batch.call_count == 1
    updated = runner._maybe_update_global_from_batch.call_args.args
    np.testing.assert_array_equal(updated[0], genomes)
    np.testing.assert_array_equal(updated[1], exact_F)
    np.testing.assert_array_equal(updated[2], exact_G)
    assert updated[3] == 10


def test_surrogate_model_swap_reprices_parent_population():
    from unittest.mock import Mock

    from evolution.custom_ga import CustomGARunner

    runner = CustomGARunner.__new__(CustomGARunner)
    runner.perf = None
    runner.n_factors = 3
    genomes = np.arange(12, dtype=np.int32).reshape(2, 6)
    manager = Mock(exact_eval_every=10)
    manager.maybe_collect_retrain.return_value = True
    manager.should_retrain.return_value = False
    predictions = np.asarray([[10, 11, 12, -1, 2], [20, 21, 22, 3, -2]], dtype=np.float32)
    manager.trainer.predict.return_value = predictions
    runner.surrogate_manager = manager
    runner.evaluator = Mock()

    scores, cv = runner._maybe_teacher_update(
        genomes, np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32), 11,
    )

    np.testing.assert_array_equal(scores, predictions[:, :3])
    np.testing.assert_array_equal(cv, [[0, 2], [3, 0]])
    assert manager.trainer.predict.call_count == 1
    np.testing.assert_array_equal(manager.trainer.predict.call_args.args[0], genomes)
    runner.evaluator.evaluate_batch.assert_not_called()


def test_exact_archive_considers_feasible_batch_member_above_soft_best():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from evolution.custom_ga import CustomGARunner

    runner = CustomGARunner.__new__(CustomGARunner)
    runner.layout = _access_layout()
    runner.global_best_genome = None
    runner.global_best_exact = None
    runner.evaluator = SimpleNamespace(_factor_scores_from_objectives=lambda _: {})
    runner._layout_reports = Mock(return_value=({}, {}, {}, {"optimizer_side_pass": True, "checks": {}}))
    invalid = runner.layout.genome.copy()
    feasible = invalid.copy()
    feasible[0] = 0
    # Soft selection prefers the invalid score 0 + 25*1 over feasible score 40.
    # Exact archive must nevertheless preserve the feasible candidate.
    runner._maybe_update_global_from_batch(np.asarray([invalid, feasible]),
                                          np.asarray([[0, 0, 0], [40, 0, 0]], dtype=np.float32),
                                          np.asarray([[1], [0]], dtype=np.float32), 10)
    np.testing.assert_array_equal(runner.global_best_genome, feasible)
    assert runner.global_best_exact["constraints"] == [0.0]


def test_infeasible_anchor_keeps_best_exact_violation_profile():
    from evolution.custom_ga import CustomGARunner

    runner = CustomGARunner.__new__(CustomGARunner)
    runner.infeasible_anchor_genome = None
    runner.infeasible_anchor_objectives = None
    runner.infeasible_anchor_constraints = None
    runner.infeasible_anchor_key = None
    runner._consider_infeasible_anchor([3, 4], [0.0, 0.0, 50.0], [1.0, 1.0])
    runner._consider_infeasible_anchor([5, 6], [0.0, 0.0, 500.0], [0.0, 5.0])
    # Fewer violated hard-constraint families outrank scalar quality; within
    # the same family count, lower exact violation sum wins.
    runner._consider_infeasible_anchor([7, 8], [0.0, 0.0, 800.0], [0.0, 4.0])
    np.testing.assert_array_equal(runner.infeasible_anchor_genome, [7, 8])
    np.testing.assert_array_equal(runner.infeasible_anchor_constraints, [0.0, 4.0])
    runner._consider_infeasible_anchor([9, 10], [0.0, 0.0, 0.0], [0.0, 0.0])
    np.testing.assert_array_equal(runner.infeasible_anchor_genome, [9, 10])


@pytest.mark.parametrize("samples,expected", [
    (["7200, 50\n", "7200, 0\n"], True),
    (["2600, 0\n"] * 3, False),
    (["7200, 50\n"] * 3, False),
])
def test_gpu_startup_gate_requires_free_memory_and_idle_compute(tmp_path, monkeypatch, samples, expected):
    from types import SimpleNamespace
    from tools import validated_run

    outputs = iter(samples)
    monkeypatch.setattr(validated_run.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=next(outputs), stderr=""))
    monkeypatch.setattr(validated_run.time, "sleep", lambda _: None)
    assert validated_run.gpu_ready(tmp_path, 4096) is expected


def test_gpu_startup_gate_allows_recorded_desktop_utilization_ceiling(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools import validated_run

    outputs = iter(["7200, 22\n"] * 3)
    monkeypatch.setattr(validated_run.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=next(outputs), stderr=""))
    monkeypatch.setattr(validated_run.time, "sleep", lambda _: None)
    assert validated_run.gpu_ready(tmp_path, 4096, max_utilization=25)


def test_redundant_ai_smoke_status_is_recorded_without_masking_direct_gates(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools import validated_run

    monkeypatch.setattr(validated_run.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=2))
    assert validated_run.diagnostic(tmp_path, "ai-smoke", ["just", "ai-smoke"], tmp_path) == 2
    import json
    status = json.loads((tmp_path / "ai-smoke.status.json").read_text())
    assert status["exit_code"] == 2
    assert "unit-tests" in status["authoritative_checks"]


def test_generation_gate_rejects_acceptance_gain_when_semantics_regress():
    from tools.validated_run import generation_contract_review

    first = {"best_generation": None, "failed_checks": ["mouse", "duplicates"],
             "clusters_together": 38, "clusters_total": 65,
             "clusters_order_ok": 14, "clusters_ordered_total": 30,
             "constraints": [0, 0, 1, 0, 0, 42, 272, 28]}
    current = {"best_generation": None, "failed_checks": ["duplicates"],
               "clusters_together": 32, "clusters_total": 65,
               "clusters_order_ok": 12, "clusters_ordered_total": 28,
               "constraints": [0, 0, 0, 0, 0, 2, 0, 0]}
    audit = {"checks": {"optimizer_acceptance": True},
             "critical_split_clusters": ["a", "b", "c"],
             "ordered_relation_failures": ["a"] * 16}
    current_audit = {"checks": {"optimizer_acceptance": True},
                     "critical_split_clusters": ["a"] * 7,
                     "ordered_relation_failures": ["a"] * 18}
    review = generation_contract_review(first, current, audit, current_audit)
    assert not review["improved"]
    assert any("cluster co-location regressed" in item for item in review["regressions"])
    assert any("relative placement regressed" in item for item in review["regressions"])
    assert any("critical cluster splits increased" in item for item in review["regressions"])


def test_run_contract_rejects_broken_required_semantic_order():
    from types import SimpleNamespace
    from evolution.acceptance import _required_relative_layout_report
    from tools.run_contract_audit import _contract_checks

    layout = SimpleNamespace(
        genome=np.asarray([0, 1], dtype=np.int32),
        positions=(
            SimpleNamespace(layer=1, x=1.0, y=0.0, hand="left"),
            SimpleNamespace(layer=1, x=0.0, y=0.0, hand="right"),
        ),
        shortcuts=(
            SimpleNamespace(keys="Win+1", action="Open pinned app 1"),
            SimpleNamespace(keys="Win+2", action="Open pinned app 2"),
        ),
        semantic_clusters=({
            "name": "ordered_apps",
            "category": "sequence",
            "members": [
                {"sid": 0, "order": 0, "dx": 0.0, "dy": 0.0},
                {"sid": 1, "order": 1, "dx": 1.0, "dy": 0.0},
            ],
        },),
    )
    relative_report = _required_relative_layout_report(layout)
    failures = [row["name"] for row in relative_report["failures"]]
    assert failures == ["ordered_apps"]
    assert not relative_report["acceptance_pass"]
    checks = _contract_checks(True, np.asarray([0.0]), True, failures)
    assert not checks["required_relative_layouts"]
    assert not all(checks.values())

    layout.positions = (
        SimpleNamespace(layer=1, x=0.0, y=0.0, hand="left"),
        SimpleNamespace(layer=1, x=1.0, y=0.0, hand="right"),
    )
    relative_report = _required_relative_layout_report(layout)
    failures = [row["name"] for row in relative_report["failures"]]
    assert failures == []
    assert relative_report["acceptance_pass"]
    assert all(_contract_checks(True, np.asarray([0.0]), True, failures).values())


def test_l0_toggle_hold_alternative_is_reported_as_a_preference():
    from types import SimpleNamespace
    from evolution.acceptance import _l0_toggle_hold_report
    from tools.run_contract_audit import _contract_checks

    layout = SimpleNamespace(
        genome=np.asarray([0], dtype=np.int32),
        positions=(SimpleNamespace(layer=0, is_frozen=False, is_thumb=True),),
        shortcuts=(SimpleNamespace(is_layer_access=True, access_target_layer=4,
                                   access_is_momentary=False, keys="@access:L4:toggle"),),
        n_shortcuts=1,
    )
    report = _l0_toggle_hold_report(layout)
    assert not report["preference_satisfied"]
    assert report["toggles_without_direct_hold"][0]["target_layer"] == 4
    checks = _contract_checks(True, np.asarray([0.0]), True, [])
    assert all(checks.values())
    assert "l0_toggles_have_direct_hold_alternative" not in checks

    layout.genome = np.asarray([0, 1], dtype=np.int32)
    layout.positions = (
        SimpleNamespace(layer=0, is_frozen=False, is_thumb=True),
        SimpleNamespace(layer=0, is_frozen=False, is_thumb=True),
    )
    layout.shortcuts = (
        layout.shortcuts[0],
        SimpleNamespace(is_layer_access=True, access_target_layer=4,
                        access_is_momentary=True, keys="@access:L4:hold"),
    )
    layout.n_shortcuts = 2
    report = _l0_toggle_hold_report(layout)
    assert report["preference_satisfied"]
    assert report["direct_l0_thumb_hold_targets"] == [4]


def test_nested_hold_audit_requires_opposite_thumbs_and_exempts_scroll():
    from types import SimpleNamespace
    from evolution.acceptance import _nested_hold_side_report

    layout = SimpleNamespace(
        genome=np.asarray([0, 1], dtype=np.int32), n_shortcuts=2,
        positions=(
            SimpleNamespace(layer=0, is_frozen=False, is_thumb=True, hand="left"),
            SimpleNamespace(layer=1, is_frozen=False, is_thumb=True, hand="left"),
        ),
        shortcuts=(
            SimpleNamespace(is_layer_access=True, access_is_momentary=True,
                            access_target_layer=1, keys="@access:L1:hold", action="Hold", base_key=""),
            SimpleNamespace(is_layer_access=True, access_is_momentary=True,
                            access_target_layer=2, keys="@access:L2:hold", action="Hold", base_key=""),
        ),
    )
    report = _nested_hold_side_report(layout)
    assert not report["preference_satisfied"]
    assert len(report["failures"]) == 1

    layout.positions = (layout.positions[0],
                        SimpleNamespace(layer=1, is_frozen=False, is_thumb=True, hand="right"))
    assert _nested_hold_side_report(layout)["preference_satisfied"]

    layout.positions = (layout.positions[0],
                        SimpleNamespace(layer=1, is_frozen=False, is_thumb=True, hand="left"))
    layout.shortcuts = (layout.shortcuts[0], SimpleNamespace(
        is_layer_access=True, access_is_momentary=True, access_target_layer=2,
        keys="@scroll:L2:hold", action="Scroll mode", base_key="",
    ))
    assert _nested_hold_side_report(layout)["preference_satisfied"]


def test_generation_gate_accepts_monotonic_contract_progress():
    from tools.validated_run import generation_contract_review

    first = {"best_generation": None, "failed_checks": ["mouse", "duplicates"],
             "clusters_together": 20, "clusters_total": 65,
             "clusters_order_ok": 8, "clusters_ordered_total": 30,
             "constraints": [0, 0, 5]}
    current = {"best_generation": 3000, "failed_checks": ["duplicates"],
               "clusters_together": 23, "clusters_total": 65,
               "clusters_order_ok": 9, "clusters_ordered_total": 30,
               "constraints": [0, 0, 1]}
    audit = {"checks": {"optimizer_acceptance": True},
             "critical_split_clusters": ["a"], "ordered_relation_failures": ["a", "b"]}
    current_audit = {"checks": {"optimizer_acceptance": True},
                     "critical_split_clusters": ["a"], "ordered_relation_failures": ["a"]}
    assert generation_contract_review(first, current, audit, current_audit)["improved"]


def test_sparse_layer_occupancy_is_diagnostic_not_a_generation_gate():
    from tools.validated_run import generation_contract_review

    first = {"best_generation": 0, "failed_checks": ["mouse"],
             "clusters_together": 50, "clusters_total": 65,
             "clusters_order_ok": 20, "clusters_ordered_total": 30,
             "constraints": [0, 0, 0]}
    current = {**first, "generation": 5000}
    sparse_audit = {"checks": {}, "reachable_generated_layer_occupancy": {2: 3}}
    fuller_audit = {"checks": {}, "reachable_generated_layer_occupancy": {2: 20}}
    review = generation_contract_review(first, current, sparse_audit, fuller_audit)
    assert not review["improved"]
    assert not review["regressions"]


def test_exact_archive_score_noise_does_not_mask_unresolved_layout_contracts():
    from tools.validated_run import generation_contract_review

    first = {"best_generation": 0, "score": 80745.09375, "failed_checks": [],
             "clusters_together": 51, "clusters_total": 67,
             "clusters_order_ok": 24, "clusters_ordered_total": 28,
             "constraints": [0, 0, 0]}
    current = {**first, "best_generation": 3678, "score": 80745.0}
    audit = {"checks": {"optimizer_acceptance": False},
             "critical_split_clusters": ["undo_redo"], "ordered_relation_failures": ["before_after"]}
    review = generation_contract_review(first, current, audit, audit)
    assert review["archive_updated"]
    assert not review["improved"]
    assert not review["improvements"]
    valid_audit = {"run_contract_pass": True,
                   "checks": {"optimizer_acceptance": True},
                   "critical_split_clusters": [], "ordered_relation_failures": []}
    valid = generation_contract_review(first, current, valid_audit, valid_audit)
    assert valid["improved"]
    assert valid["score_improved"]
    assert any("exact-best score improved" in item for item in valid["improvements"])


def test_contract_gate_ignores_score_roundoff_and_resets_on_score_stage_change():
    from tools.validated_run import (
        _semantic_multiplier_at, contract_gate_should_stop, generation_contract_review,
    )

    staged = {"fitness": {"semantic_cluster_multiplier": 200.0,
                           "semantic_cluster_multiplier_schedule": [[0, 5000], [10000, 20000]]}}
    assert _semantic_multiplier_at(staged, 5000) == 5000.0
    assert _semantic_multiplier_at(staged, 10000) == 20000.0

    audit = {"generation": 5000, "run_contract_pass": True, "checks": {}}
    first = {"best_generation": 1, "score": 80766.7265625, "semantic_cluster_multiplier": 5000.0}
    roundoff = {**first, "best_generation": 2, "score": 80766.71875}
    review = generation_contract_review(first, roundoff, audit, audit)
    assert review["archive_updated"]
    assert not review["score_improved"]
    assert not review["improvements"]
    stop, stalls, reason = contract_gate_should_stop(review, audit, 0)
    assert not stop and stalls == 0
    assert "passes the run contract" in reason

    two_ulp_gain = {**first, "best_generation": 1458, "score": 80766.7109375}
    review = generation_contract_review(first, two_ulp_gain, audit, audit)
    assert review["score_improved"]
    assert review["improved"]
    stop, stalls, reason = contract_gate_should_stop(review, audit, 1)
    assert not stop and stalls == 0
    assert "passes the run contract" in reason

    repriced = {**roundoff, "best_generation": 3, "score": 8.0,
                "semantic_cluster_multiplier": 20000.0}
    review = generation_contract_review(first, repriced, audit, audit)
    assert review["score_stage_changed"]
    assert not review["score_improved"]
    stop, stalls, reason = contract_gate_should_stop(review, audit, 1)
    assert not stop and stalls == 0
    assert "passes the run contract" in reason


def test_startup_smoke_records_but_does_not_require_early_contract_acceptance():
    from tools.validated_run import startup_smoke_passes

    smoke = {
        "generation": 10,
        "training_path": "cuda_surrogate_primary",
        "best_exact": {
            "effort": 0.2,
            "adjacency": 1.0,
            "violations": 5.0,
            "genome": [1, 2, 3],
            "acceptance_report": {
                "optimizer_side_pass": False,
                "optimizer_side_checks": {"dynamic_mouse_layer_present": False},
            },
            "arrow_report": {"acceptance_pass": False},
        },
    }
    assert startup_smoke_passes(smoke) == (True, "")


def test_startup_smoke_rejects_missing_or_nonfinite_exact_metrics():
    from tools.validated_run import startup_smoke_passes

    smoke = {
        "generation": 10,
        "training_path": "cuda_surrogate_primary",
        "best_exact": {
            "effort": float("nan"),
            "adjacency": 0.0,
            "violations": 0.0,
            "genome": [1, 2, 3],
            "acceptance_report": {"optimizer_side_pass": True, "optimizer_side_checks": {}},
            "arrow_report": {"acceptance_pass": True},
        },
    }
    passed, reason = startup_smoke_passes(smoke)
    assert not passed
    assert "non-finite" in reason


def test_fresh_run_snapshot_does_not_select_or_inject_saved_genome(tmp_path, monkeypatch):
    from tools import validated_run

    (tmp_path / "config_v2.yaml").write_text("evolution: {}\n")
    data = tmp_path / "data"
    data.mkdir()
    (data / "layout.json").write_text("{}\n")
    (data / "default_layout_genome.json").write_text('{"genome": [1]}\n')
    build = tmp_path / "build"
    build.mkdir()
    (build / "v2_local_search_result.json").write_text('{"genome": [2]}\n')
    listed = b"config_v2.yaml\0data/layout.json\0data/default_layout_genome.json\0"

    def fake_capture(args):
        if args[:2] == ["git", "ls-files"]:
            return listed
        if args[:2] == ["git", "rev-parse"]:
            return b"test-head\n"
        if args[:2] == ["uname", "-a"]:
            return b"test-host\n"
        return b""

    monkeypatch.setattr(validated_run, "ROOT", tmp_path)
    monkeypatch.setattr(validated_run, "capture", fake_capture)
    run = validated_run.prepare(fresh_start=True)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["initialization_mode"] == "fresh_random_population"
    assert manifest["warmstart"] == {
        "enabled": False, "source": None, "source_generation": None, "sha256": None,
    }
    assert not (run / "v2_local_search_result.json").exists()
    # Historical genomes may remain as analysis-test fixtures in frozen source,
    # but the production argv must explicitly disable their injection.
    assert (run / "source/build/v2_local_search_result.json").exists()
    assert "--no-inject-seed" in validated_run.evolution_runner(run, manifest)


def test_fresh_run_rejects_a_warmstart_argument():
    from tools.validated_run import prepare

    with pytest.raises(ValueError, match="cannot be combined"):
        prepare(warmstart_path=Path("old-genome.json"), fresh_start=True)

def test_contract_review_schedule_covers_requested_milestones():
    from tools.validated_run import CONTRACT_REVIEW_GENERATIONS

    assert CONTRACT_REVIEW_GENERATIONS == (5000, 10000, 15000, 20000, 25000)


def test_checkpoint_reports_raw_score_without_stale_score_floor():
    from tools.validated_run import SCORE_REBASE_FLOOR, summarize

    summary = summarize({
        "generation": 4000, "best_generation": 4,
        "best_exact": {"total_score": 80766.40625, "constraints": []},
    })
    assert summary["score"] == summary["raw_score"] == 80766.40625
    assert SCORE_REBASE_FLOOR is None
    assert summary["score_floor"] is None
    assert summary["rebased_score"] is None
    assert summary["rebased_target"] == 0.0


def test_contract_gate_stops_after_two_stagnant_invalid_milestones():
    from tools.validated_run import contract_gate_should_stop

    review = {"regressions": [], "improvements": []}
    invalid_audit = {"generation": 1000, "run_contract_pass": False}
    stop, stalls, reason = contract_gate_should_stop(review, invalid_audit, 0)
    assert not stop and stalls == 0
    assert "not a scheduled" in reason

    invalid_audit["generation"] = 5000
    stop, stalls, reason = contract_gate_should_stop(review, invalid_audit, 0)
    assert not stop and stalls == 1
    assert "first stagnant" in reason

    invalid_audit["generation"] = 10000
    stop, stalls, reason = contract_gate_should_stop(review, invalid_audit, stalls)
    assert stop and stalls == 2
    assert "two consecutive" in reason


def test_contract_gate_retains_improving_but_still_invalid_search():
    from tools.validated_run import contract_gate_should_stop

    review = {"regressions": [], "improvements": ["relative placement improved"]}
    stop, stalls, reason = contract_gate_should_stop(
        review, {"generation": 5000, "run_contract_pass": False}, 1,
    )
    assert not stop and stalls == 0
    assert "evidence is improving" in reason


def test_contract_review_does_not_stop_for_regressions_or_plateaus():
    from tools.validated_run import contract_gate_should_stop

    stop, stalls, _ = contract_gate_should_stop(
        {"regressions": ["ordered placement regressed"], "improvements": []},
        {"generation": 5000, "run_contract_pass": False}, 0,
    )
    assert not stop and stalls == 1

    stop, stalls, reason = contract_gate_should_stop(
        {"regressions": [], "improvements": []},
        {"generation": 10000, "run_contract_pass": True}, 1,
    )
    assert not stop and stalls == 0
    assert "passes the run contract" in reason

    stop, stalls, reason = contract_gate_should_stop(
        {"regressions": [], "improvements": []},
        {"generation": 10000, "run_contract_pass": True}, 0,
    )
    assert not stop and stalls == 0
    assert "passes the run contract" in reason
