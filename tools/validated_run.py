#!/usr/bin/env python3
"""Freeze inputs, validate startup, supervise the bounded production run.

COMPOSE: existing runner, direct pytest, perf benchmark and report tools.
No dependencies or alternate evaluator/training paths are introduced.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONTRACT_REVIEW_GENERATIONS = (5000, 10000, 15000, 20000, 25000)
# Calibrate a score floor from the selected exact candidate after the run.
# Live training and milestone gates compare raw exact scores.
SCORE_REBASE_FLOOR = None
REBASED_SCORE_TARGET = 0.0
# At the current ~80k score scale, float32 ULP is 0.0078125. Require two ULPs
# to count a clean exact-archive update as meaningful progress.
MIN_EXACT_SCORE_IMPROVEMENT = 0.01


def capture(args):
    return subprocess.check_output(args, cwd=ROOT)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(warmstart_path=None, fresh_start=False):
    if fresh_start and warmstart_path is not None:
        raise ValueError("--fresh-start cannot be combined with --warmstart")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = ROOT / "build/runs" / f"validated_30k_{stamp}"
    source = run / "source"
    source.mkdir(parents=True)
    # Raw NUL-delimited Git output is necessary to preserve exact paths.
    names = capture(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"])
    hashes = {}
    run_artifact_hashes = {}
    for name in sorted(set(os.fsdecode(item) for item in names.split(b"\0") if item)):
        src = ROOT / name
        if not src.is_file():
            continue
        dst = source / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    # Include ignored inputs too; the loader's source of truth is the data directory.
    for src in sorted((ROOT / "data").rglob("*")):
        if not src.is_file():
            continue
        name = str(src.relative_to(ROOT))
        dst = source / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    # Existing analysis-tool tests require representative checkpoints. These
    # are test fixtures only; the runner never resumes from them.
    for name in ("v2_scale_factors.json", "v2_local_search_result.json",
                 "v2_checkpoint_gen29500.json", "v2_checkpoint_gen30000.json"):
        src = ROOT / "build" / name
        if src.exists():
            dst = source / "build" / name
            dst.parent.mkdir(exist_ok=True)
            shutil.copy2(src, dst)
            hashes[f"build/{name}"] = hashlib.sha256(dst.read_bytes()).hexdigest()
    # Keep the baseline seed and selected seed identifiable in the frozen run.
    base_warmstart = source / "build/v2_local_search_result.json"
    default_layout_seed = source / "data/default_layout_genome.json"
    if not fresh_start and not base_warmstart.exists() and default_layout_seed.exists():
        base_warmstart.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(default_layout_seed, base_warmstart)
        hashes["build/v2_local_search_result.json"] = hashlib.sha256(
            base_warmstart.read_bytes()
        ).hexdigest()
    if not fresh_start and base_warmstart.exists():
        shutil.copy2(base_warmstart, run / "base_v2_local_search_result.json")
        run_artifact_hashes["base_v2_local_search_result.json"] = hashlib.sha256(
            (run / "base_v2_local_search_result.json").read_bytes()
        ).hexdigest()
    selected_warmstart = None if fresh_start else base_warmstart
    warmstart_source = None if fresh_start else "repository default"
    warmstart_generation = None
    warmstart_sha256 = None
    if warmstart_path is not None:
        input_path = Path(warmstart_path).expanduser().resolve()
        input_bytes = input_path.read_bytes()
        payload = json.loads(input_bytes)
        genome = payload.get("best_genome", payload.get("genome"))
        if not isinstance(genome, list):
            raise ValueError(f"Warmstart must contain best_genome or genome list: {input_path}")
        warmstart_generation = payload.get("best_generation", payload.get("generation"))
        warmstart_sha256 = hashlib.sha256(input_bytes).hexdigest()
        archived_checkpoint = run / "warmstart_source_checkpoint.json"
        archived_checkpoint.write_bytes(input_bytes)
        run_artifact_hashes[archived_checkpoint.name] = hashlib.sha256(
            archived_checkpoint.read_bytes()
        ).hexdigest()
        selected = {
            "genome": genome,
            "source_checkpoint": str(input_path),
            "source_generation": warmstart_generation,
        }
        selected_warmstart = source / "build/v2_local_search_result.json"
        selected_warmstart.parent.mkdir(parents=True, exist_ok=True)
        write_json(selected_warmstart, selected)
        warmstart_source = str(input_path)
        hashes["build/v2_local_search_result.json"] = hashlib.sha256(
            selected_warmstart.read_bytes()
        ).hexdigest()
    if selected_warmstart is not None and selected_warmstart.exists():
        shutil.copy2(selected_warmstart, run / "v2_local_search_result.json")
        run_artifact_hashes["v2_local_search_result.json"] = hashlib.sha256(
            (run / "v2_local_search_result.json").read_bytes()
        ).hexdigest()
    shutil.copy2(source / "config_v2.yaml", run / "config_v2.yaml")
    (run / "worktree.patch").write_bytes(capture(["git", "diff", "HEAD", "--binary"]))
    (run / "git-status.txt").write_bytes(capture(["git", "status", "--short"]))
    (run / "dependencies.txt").write_bytes(capture(["uv", "pip", "freeze", "--python", str(PYTHON)]))
    write_json(run / "manifest.json", {
        "created_utc": stamp, "head": capture(["git", "rev-parse", "HEAD"]).decode().strip(),
        "sparse_upstream_commit": "ce81e536323c1739b841678caf2acc797275bd5f",
        "sparse_reconciliation": "Threshold score and cutoff removed; occupancy is diagnostic only",
        "host": capture(["uname", "-a"]).decode().strip(),
        "python": str(PYTHON), "generations": 30000, "population": 1500, "seed": 42,
        "initialization_mode": "fresh_random_population" if fresh_start else "warmstart",
        "warmstart": {"enabled": not fresh_start, "source": warmstart_source,
                      "source_generation": warmstart_generation,
                      "sha256": warmstart_sha256},
        "run_artifact_sha256": run_artifact_hashes,
        "sha256": hashes,
    })
    write_json(run / "state.json", {
        "status": "prepared", "production_started": False,
        "initialization_mode": "fresh_random_population" if fresh_start else "warmstart",
    })
    return run


def evolution_runner(run, manifest):
    """Build runner argv, making fresh-start initialization explicit."""
    runner = [str(PYTHON), "-u", "run_evolution.py", "--config", "config_v2.yaml",
              "--data-dir", "data", "--pop-size", "1500", "--seed", "42"]
    if manifest.get("initialization_mode") == "fresh_random_population":
        runner.append("--no-inject-seed")
    return runner


def gpu_ready(run, min_free_mib, max_utilization=10):
    samples = []
    for attempt in range(3):
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits",
        ], capture_output=True, text=True)
        samples.append(result.stdout + result.stderr)
        (run / "gpu-readiness.txt").write_text("".join(samples))
        if result.returncode != 0:
            return False
        free, utilization = map(int, result.stdout.splitlines()[0].split(","))
        # Allow our own preceding check's utilization sample to decay. Never
        # start training while memory headroom or low utilization is absent.
        if free >= min_free_mib and utilization <= max_utilization:
            return True
        if attempt < 2:
            time.sleep(2)
    return False


def checked(run, name, args, cwd):
    with (run / f"{name}.log").open("w") as log:
        result = subprocess.run(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        write_json(run / "state.json", {"status": "startup_failed", "check": name,
                                       "exit_code": result.returncode, "production_started": False})
        raise RuntimeError(f"{name} failed with exit code {result.returncode}; see {run / (name + '.log')}")


def diagnostic(run, name, args, cwd):
    """Record a redundant wrapper's status; direct checks remain authoritative."""
    with (run / f"{name}.log").open("w") as log:
        result = subprocess.run(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    write_json(run / f"{name}.status.json", {"exit_code": result.returncode,
                                               "authoritative_checks": ["focused-tests", "unit-tests",
                                                                        "ai-guard", "perf-benchmark"]})
    return result.returncode


def startup_smoke_passes(smoke):
    """Check CUDA startup/evaluator health; production reviews enforce acceptance."""
    if smoke.get("generation") != 10 or smoke.get("training_path") != "cuda_surrogate_primary":
        return False, "Startup smoke did not complete 10 GPU-primary generations"
    best = smoke.get("best_exact") or {}
    acceptance = best.get("acceptance_report") or {}
    if not isinstance(acceptance.get("optimizer_side_pass"), bool):
        return False, "Startup smoke did not return exact acceptance diagnostics"
    scores = [best.get(name) for name in ("effort", "adjacency", "violations")]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in scores):
        return False, "Startup smoke exact evaluator returned missing or non-finite scores"
    if not isinstance(best.get("genome"), list) or not best["genome"]:
        return False, "Startup smoke exact evaluator returned no genome"
    return True, ""


def _semantic_multiplier_at(config, generation):
    fitness = (config or {}).get("fitness", {})
    multiplier = float(fitness.get("semantic_cluster_multiplier", 200.0))
    for stage, value in fitness.get("semantic_cluster_multiplier_schedule", []) or []:
        if int(generation) >= int(stage):
            multiplier = float(value)
    return multiplier


def summarize(checkpoint, config=None):
    best = checkpoint["best_exact"]
    score = float(best["total_score"])
    return {
        "generation": checkpoint["generation"], "best_generation": checkpoint["best_generation"],
        "semantic_cluster_multiplier": _semantic_multiplier_at(config, checkpoint["generation"]),
        "score": score, "raw_score": score, "score_floor": None,
        "rebased_score": None,
        "rebased_target": REBASED_SCORE_TARGET,
        "constraints": best["constraints"],
        "clusters_together": best.get("clusters_together"), "clusters_total": best.get("clusters_total"),
        "clusters_order_ok": best.get("clusters_order_ok"),
        "clusters_ordered_total": best.get("clusters_ordered_total"),
        "failed_checks": best.get("acceptance_failed_checks", []),
    }


def _ratio(summary, numerator, denominator):
    total = summary.get(denominator) or 0
    return (summary.get(numerator) or 0) / total if total else 0.0


def generation_contract_review(first, current, first_audit, current_audit):
    """Compare milestone quality; callers allow one stalled review interval."""
    regressions = []
    improvements = []
    metrics = (
        ("cluster co-location", "clusters_together", "clusters_total"),
        ("relative placement", "clusters_order_ok", "clusters_ordered_total"),
    )
    for label, numerator, denominator in metrics:
        before = _ratio(first, numerator, denominator)
        after = _ratio(current, numerator, denominator)
        if after + 1e-9 < before:
            regressions.append(f"{label} regressed: {before:.3f} -> {after:.3f}")
        elif after > before + 1e-9:
            improvements.append(f"{label} improved: {before:.3f} -> {after:.3f}")

    before_failures = len(first.get("failed_checks", []))
    after_failures = len(current.get("failed_checks", []))
    baseline_failure_set = set(first.get("failed_checks", []))
    current_failure_set = set(current.get("failed_checks", []))
    recovered_failures = sorted(baseline_failure_set - current_failure_set)
    unresolved_failures = sorted(baseline_failure_set & current_failure_set)
    if recovered_failures:
        improvements.append("acceptance checks recovered: " + ", ".join(recovered_failures))
    if after_failures > before_failures:
        regressions.append(f"failed acceptance checks increased: {before_failures} -> {after_failures}")
    elif after_failures < before_failures:
        improvements.append(f"failed acceptance checks decreased: {before_failures} -> {after_failures}")

    before_cv = sum(max(0.0, float(value)) for value in first.get("constraints", []))
    after_cv = sum(max(0.0, float(value)) for value in current.get("constraints", []))
    if after_cv > before_cv + 1e-9:
        regressions.append(f"constraint violation increased: {before_cv:g} -> {after_cv:g}")
    elif after_cv + 1e-9 < before_cv:
        improvements.append(f"constraint violation decreased: {before_cv:g} -> {after_cv:g}")

    for label, key in (("critical cluster splits", "critical_split_clusters"),
                       ("ordered relation failures", "ordered_relation_failures")):
        before = len(first_audit.get(key, []))
        after = len(current_audit.get(key, []))
        if after > before:
            regressions.append(f"{label} increased: {before} -> {after}")
        elif after < before:
            improvements.append(f"{label} decreased: {before} -> {after}")

    baseline_contract_failures = {
        name for name, passed in first_audit.get("checks", {}).items() if not passed
    }
    current_contract_failures = {
        name for name, passed in current_audit.get("checks", {}).items() if not passed
    }
    recovered_contract_checks = sorted(baseline_contract_failures - current_contract_failures)
    unresolved_contract_checks = sorted(baseline_contract_failures & current_contract_failures)
    if recovered_contract_checks:
        improvements.append("contract checks recovered: " + ", ".join(recovered_contract_checks))
    if current_contract_failures - baseline_contract_failures:
        regressions.append("new contract checks failed: " + ", ".join(
            sorted(current_contract_failures - baseline_contract_failures)
        ))

    before_best = first.get("best_generation")
    after_best = current.get("best_generation")
    archive_updated = after_best is not None and (before_best is None or after_best > before_best)
    before_score = first.get("score")
    after_score = current.get("score")
    before_stage = first.get("semantic_cluster_multiplier")
    current_stage = current.get("semantic_cluster_multiplier")
    score_stage_changed = before_stage is not None and current_stage is not None and before_stage != current_stage
    # Ignore one-ULP float32 changes; two ULPs count as exact progress.
    score_improved = (not score_stage_changed and before_score is not None and after_score is not None
                      and float(after_score) < float(before_score) - MIN_EXACT_SCORE_IMPROVEMENT)
    # Small objective-score changes do not count as recovery while hard layout
    # contracts remain unresolved; otherwise harmless score noise can reset the
    # no-progress window indefinitely.
    if score_improved and current_audit.get("run_contract_pass", False):
        improvements.append(f"exact-best score improved: {before_score:.4f} -> {after_score:.4f}")
    elif (archive_updated and current_audit.get("run_contract_pass", False)
          and (before_score is None or after_score is None)):
        improvements.append(f"exact archive updated at generation {after_best}")

    return {
        "improved": not regressions and bool(improvements),
        "archive_updated": archive_updated,
        "score_improved": score_improved,
        "score_stage_changed": score_stage_changed,
        "recovered_acceptance_failures": recovered_failures,
        "unresolved_acceptance_failures": unresolved_failures,
        "recovered_contract_checks": recovered_contract_checks,
        "unresolved_contract_checks": unresolved_contract_checks,
        "regressions": regressions,
        "improvements": improvements,
    }


def contract_gate_should_stop(review, current_audit, consecutive_stalls):
    """Stop after repeated milestone-level hard-contract failure without progress."""
    generation = int(current_audit.get("generation", 0))
    if generation not in CONTRACT_REVIEW_GENERATIONS:
        return False, consecutive_stalls, "not a scheduled contract milestone"
    if current_audit.get("run_contract_pass", False):
        return False, 0, "exact-best archive passes the run contract"
    if review.get("improvements"):
        return False, 0, "exact-best archive still fails, but contract evidence is improving"
    stalls = consecutive_stalls + 1
    if stalls >= 2:
        return True, stalls, (
            "exact-best archive failed a named hard contract at two consecutive "
            "5,000-generation reviews with no corrective progress"
        )
    return False, stalls, "first stagnant invalid milestone; continue to next scheduled review"


def contract_audit_checkpoint(run, checkpoint, source):
    checkpoint = Path(checkpoint)
    generation = json.loads(checkpoint.read_text())["generation"]
    audit_path = Path(run) / "reviews" / f"contract-audit-gen{generation}.json"
    result = subprocess.run(
        [str(PYTHON), "tools/run_contract_audit.py", "--cpu", str(checkpoint)],
        cwd=source, capture_output=True, text=True,
    )
    audit_path.write_text(result.stdout)
    (audit_path.with_suffix(".stderr")).write_text(result.stderr)
    if result.returncode not in (0, 5):
        raise RuntimeError(f"contract audit failed to run for generation {generation}: {result.stderr}")
    audit = json.loads(result.stdout)
    audit["audit_exit_code"] = result.returncode
    return audit


def watch_existing(run, pid):
    """Monitor a live worker, applying contract gates after supervisor restart."""
    run = Path(run).resolve()
    reviews = run / "reviews"
    reviews.mkdir(exist_ok=True)
    progress_path = reviews / "progress.json"
    summaries = json.loads(progress_path.read_text()) if progress_path.exists() else []
    seen = {path.name for path in reviews.glob("v2_checkpoint_gen*.json")}
    source = run / "source"
    import yaml
    with (source / "config_v2.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    summaries = [
        {**summary, "semantic_cluster_multiplier": summary.get(
            "semantic_cluster_multiplier",
            _semantic_multiplier_at(config, summary["generation"]),
        ), "raw_score": summary.get("raw_score", summary["score"]),
            "score_floor": None,
            "rebased_score": None,
            "rebased_target": REBASED_SCORE_TARGET}
        for summary in summaries
    ]
    write_json(progress_path, summaries)
    passed_gates = []
    for path in reviews.glob("generation-*-gate.json"):
        try:
            data = json.loads(path.read_text())
            generation = int(path.stem.split("-")[1])
            passed_gates.append((generation, data))
        except (ValueError, json.JSONDecodeError):
            continue
    if passed_gates:
        _, prior_gate = max(passed_gates, key=lambda item: item[0])
        consecutive_stalls = int(prior_gate.get("consecutive_stalls", 0))
        gate_baseline = dict(prior_gate["current"])
        gate_baseline.setdefault(
            "semantic_cluster_multiplier",
            _semantic_multiplier_at(config, gate_baseline["generation"]),
        )
        gate_baseline.setdefault("raw_score", gate_baseline["score"])
        gate_baseline.update({"score_floor": None,
                              "rebased_score": None,
                              "rebased_target": REBASED_SCORE_TARGET})
        baseline_audit_path = reviews / f"contract-audit-gen{gate_baseline['generation']}.json"
        gate_baseline_audit = json.loads(baseline_audit_path.read_text())
    else:
        consecutive_stalls = 0
        gate_baseline = summaries[0] if summaries else None
        baseline_audit_path = reviews / "contract-audit-gen500.json"
        gate_baseline_audit = json.loads(baseline_audit_path.read_text()) if baseline_audit_path.exists() else None
    monitor_log = reviews / "continuation-monitor.log"
    stop_reason = None
    stop_generation = None
    with monitor_log.open("a", buffering=1) as log:
        log.write(f"Monitoring live process pid={pid} at {datetime.now(timezone.utc).isoformat()}\n")
        while True:
            for path in sorted(run.glob("v2_checkpoint_gen*.json"), key=lambda p: int(p.stem.split("gen")[-1])):
                if path.name in seen:
                    continue
                try:
                    checkpoint = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue
                seen.add(path.name)
                shutil.copy2(path, reviews / path.name)
                summary = summarize(checkpoint, config)
                summaries.append(summary)
                write_json(progress_path, summaries)
                log.write(json.dumps(summary) + "\n")
                if gate_baseline is None:
                    gate_baseline = summary
                    try:
                        gate_baseline_audit = contract_audit_checkpoint(
                            run, reviews / path.name, source,
                        )
                    except Exception as exc:
                        write_json(reviews / f"generation-{checkpoint['generation']}-audit-error.json", {"reason": str(exc)})
                        log.write(f"Checkpoint audit failed; continuing run: {exc}\n")
                        gate_baseline_audit = {"checks": {}, "audit_error": str(exc)}
                    continue
                try:
                    current_audit = contract_audit_checkpoint(run, reviews / path.name, source)
                    if current_audit.get("audit_error"):
                        continue
                    gate_generation = checkpoint["generation"]
                    if gate_generation not in CONTRACT_REVIEW_GENERATIONS:
                        continue
                    review = generation_contract_review(
                        gate_baseline, summary, gate_baseline_audit, current_audit,
                    )
                except Exception as exc:
                    write_json(reviews / f"generation-{checkpoint['generation']}-audit-error.json", {"reason": str(exc)})
                    log.write(f"Contract review failed; continuing run: {exc}\n")
                    gate_baseline = summary
                    gate_baseline_audit = {"checks": {}, "audit_error": str(exc)}
                    continue
                stop, consecutive_stalls, decision = contract_gate_should_stop(
                    review, current_audit, consecutive_stalls,
                )
                review.update({
                    "consecutive_stalls": consecutive_stalls,
                    "decision": decision,
                    "scheduled_milestone": gate_generation in CONTRACT_REVIEW_GENERATIONS,
                })
                write_json(reviews / f"generation-{gate_generation}-gate.json", {
                    **review, "baseline_generation": gate_baseline["generation"],
                    "baseline": gate_baseline, "current": summary,
                    "baseline_contract_checks": gate_baseline_audit["checks"],
                    "current_contract_checks": current_audit["checks"],
                })
                gate_baseline = summary
                gate_baseline_audit = current_audit
                if stop:
                    stop_reason = decision
                    stop_generation = gate_generation
                    log.write(f"STOPPING at generation {gate_generation}: {decision}\n")
                    try:
                        os.kill(pid, 15)
                    except ProcessLookupError:
                        pass
                    break
            try:
                stat = Path(f"/proc/{pid}/stat").read_text().split()
                alive = stat[2] != "Z"
            except FileNotFoundError:
                alive = False
            if not alive:
                break
            time.sleep(10)

    if stop_reason:
        write_json(run / "state.json", {
            "status": "stopped_invalid_archive_stall", "production_started": True,
            "generation": stop_generation, "reason": stop_reason,
            "checkpoint": str(reviews / f"v2_checkpoint_gen{stop_generation}.json"),
        })
        return 5

    result_path = run / "v2_evolution_results.json"
    if not result_path.exists():
        write_json(run / "state.json", {"status": "production_failed", "production_started": True,
                                         "pid": pid, "last_checkpoint": summaries[-1] if summaries else None})
        return 5
    source = run / "source"
    for name, args in (
        ("final-report", [str(PYTHON), "tools/generate_run_report.py", str(run)]),
        ("semantic-report", [str(PYTHON), "tools/semantic_cluster_report.py",
                             str(run / "v2_checkpoint_gen30000.json")]),
    ):
        with (run / f"{name}.log").open("w") as log:
            result = subprocess.run(args, cwd=source, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(run / "state.json", {"status": "report_failed", "check": name,
                                             "exit_code": result.returncode, "generation": 30000})
            return result.returncode
    with (run / "final-contract-audit.json").open("w") as log:
        audit = subprocess.run([str(PYTHON), "tools/run_contract_audit.py",
                                str(run / "v2_checkpoint_gen30000.json")], cwd=source, stdout=log,
                               stderr=subprocess.PIPE, text=True)
    (run / "final-contract-audit.stderr").write_text(audit.stderr)
    final = json.loads(result_path.read_text())
    acceptance = final["best_exact"]["acceptance_report"]
    valid = acceptance["optimizer_side_pass"] and audit.returncode == 0
    status = "validated_export_pending" if valid else "invalid"
    write_json(run / "state.json", {"status": status, "production_started": True,
                                    "generation": final["generation"], "acceptance": acceptance})
    return 0 if valid else 5


def execute(run, min_free_mib, max_utilization):
    manifest = json.loads((run / "manifest.json").read_text())
    source = run / "source"
    import yaml
    with (source / "config_v2.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source/input changed: {name}")
    if not gpu_ready(run, min_free_mib, max_utilization):
        write_json(run / "state.json", {"status": "waiting_for_gpu", "production_started": False,
                                       "min_free_mib": min_free_mib,
                                       "max_utilization": max_utilization})
        print(f"GPU unavailable; snapshot preserved at {run}", flush=True)
        return 3
    write_json(run / "state.json", {"status": "validating_startup", "production_started": False,
                                   "max_utilization": max_utilization})
    # Check the frozen code. Guard additionally runs in the dirty Git worktree,
    # where changed-file detection triggers its mandatory performance check.
    checked(run, "focused-tests", [str(PYTHON), "-m", "pytest", "-q", "tests/test_run_contract.py",
                                   "tests/test_layer_access_thumb.py", "tests/test_semantic_clusters.py"], source)
    checked(run, "unit-tests", [str(PYTHON), "-m", "pytest", "-q", "tests/"], source)
    checked(run, "ai-guard", ["just", "ai-guard"], ROOT)
    # Explicit benchmark exit code rejects contention skips (guard permits them).
    checked(run, "perf-benchmark", [str(PYTHON), "tools/perf_benchmark.py"], source)
    if not gpu_ready(run, min_free_mib, max_utilization):
        raise RuntimeError("GPU readiness lost before startup smoke")
    runner = evolution_runner(run, manifest)
    checked(run, "startup-smoke", runner + ["--generations", "10", "--output-dir", str(run / "smoke")], source)
    smoke = json.loads((run / "smoke/v2_evolution_results.json").read_text())
    smoke_pass, smoke_reason = startup_smoke_passes(smoke)
    if not smoke_pass:
        write_json(run / "state.json", {
            "status": "startup_failed", "check": "startup-smoke-acceptance",
            "reason": smoke_reason, "production_started": False,
        })
        raise RuntimeError(smoke_reason)
    if not gpu_ready(run, min_free_mib, max_utilization):
        raise RuntimeError("GPU readiness lost before production")
    reviews = run / "reviews"
    reviews.mkdir(exist_ok=True)
    seen = set()
    summaries = []
    gate_baseline = None
    gate_baseline_audit = None
    consecutive_stalls = 0
    stop_reason = None
    gate_generation = None
    with (run / "production.log").open("w") as log:
        process = subprocess.Popen(runner + ["--generations", "30000", "--output-dir", str(run)],
                                   cwd=source, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(run / "state.json", {"status": "running", "production_started": True,
                                       "pid": process.pid, "max_utilization": max_utilization})
        while True:
            for path in sorted(run.glob("v2_checkpoint_gen*.json"), key=lambda p: int(p.stem.split("gen")[-1])):
                if path.name in seen:
                    continue
                try:
                    checkpoint = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue  # Checkpoint writer is still finishing.
                seen.add(path.name)
                shutil.copy2(path, reviews / path.name)
                summary = summarize(checkpoint, config)
                summaries.append(summary)
                write_json(reviews / "progress.json", summaries)
                print(json.dumps(summary), flush=True)
                try:
                    if gate_baseline is None:
                        gate_baseline = summary
                        gate_baseline_audit = contract_audit_checkpoint(run, path, source)
                        continue
                    current_audit = contract_audit_checkpoint(run, path, source)
                    if gate_baseline_audit.get("audit_error"):
                        gate_baseline = summary
                        gate_baseline_audit = current_audit
                        continue
                    gate_generation = checkpoint["generation"]
                    if gate_generation not in CONTRACT_REVIEW_GENERATIONS:
                        continue
                    review = generation_contract_review(
                        gate_baseline, summary, gate_baseline_audit, current_audit,
                    )
                    stop, consecutive_stalls, decision = contract_gate_should_stop(
                        review, current_audit, consecutive_stalls,
                    )
                    review.update({
                        "consecutive_stalls": consecutive_stalls,
                        "decision": decision,
                        "scheduled_milestone": gate_generation in CONTRACT_REVIEW_GENERATIONS,
                    })
                    write_json(reviews / f"generation-{gate_generation}-gate.json", {
                        **review, "baseline_generation": gate_baseline["generation"],
                        "baseline": gate_baseline, "current": summary,
                        "baseline_contract_checks": gate_baseline_audit["checks"],
                        "current_contract_checks": current_audit["checks"],
                    })
                    gate_baseline = summary
                    gate_baseline_audit = current_audit
                    if stop:
                        stop_reason = decision
                        gate_generation = checkpoint["generation"]
                        log.write(f"STOPPING at generation {gate_generation}: {decision}\n")
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        break
                except Exception as exc:
                    generation = checkpoint["generation"]
                    write_json(reviews / f"generation-{generation}-audit-error.json", {"reason": str(exc)})
                    print(f"Generation {generation} contract audit failed; continuing run: {exc}", flush=True)
                    gate_baseline = summary
                    gate_baseline_audit = {"checks": {}, "audit_error": str(exc)}
            if process.poll() is not None:
                break
            time.sleep(10)
    if stop_reason:
        write_json(run / "state.json", {
            "status": "stopped_invalid_archive_stall", "production_started": True,
            "generation": gate_generation, "reason": stop_reason,
            "checkpoint": str(run / "reviews" / f"v2_checkpoint_gen{gate_generation}.json"),
        })
        return 5
    if process.returncode:
        write_json(run / "state.json", {"status": "production_failed", "exit_code": process.returncode,
                                       "production_started": True})
        return process.returncode
    checked(run, "final-report", [str(PYTHON), "tools/generate_run_report.py", str(run)], source)
    checked(run, "semantic-report", [str(PYTHON), "tools/semantic_cluster_report.py",
                                    str(run / "v2_checkpoint_gen30000.json")], source)
    final = json.loads((run / "v2_evolution_results.json").read_text())
    acceptance = final["best_exact"]["acceptance_report"]
    with (run / "final-contract-audit.json").open("w") as log:
        audit = subprocess.run([str(PYTHON), "tools/run_contract_audit.py",
                                str(run / "v2_checkpoint_gen30000.json")], cwd=source, stdout=log,
                               stderr=subprocess.PIPE, text=True)
    (run / "final-contract-audit.stderr").write_text(audit.stderr)
    valid = acceptance["optimizer_side_pass"] and audit.returncode == 0 and final["generation"] == 30000
    status = "validated_export_pending" if valid else "invalid"
    write_json(run / "state.json", {"status": status, "production_started": True,
                                   "generation": final["generation"], "acceptance": acceptance})
    return 0 if valid else 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-dir", type=Path, help="Execute a previously prepared snapshot")
    parser.add_argument("--watch-pid", type=int, help="Resume monitoring an existing production PID")
    parser.add_argument("--warmstart", type=Path,
                        help="Seed from a checkpoint's best_genome or a warmstart genome")
    parser.add_argument("--fresh-start", action="store_true",
                        help="Use a random initial population; do not inject a saved genome")
    parser.add_argument("--min-free-mib", type=int, default=4096)
    parser.add_argument("--max-gpu-utilization", type=int, default=10,
                        help="Startup GPU utilization ceiling; record justified desktop-use overrides")
    args = parser.parse_args()
    if args.run_dir and (args.fresh_start or args.warmstart):
        parser.error("--run-dir cannot be combined with --fresh-start or --warmstart")
    run = args.run_dir.resolve() if args.run_dir else prepare(args.warmstart, args.fresh_start)
    print(f"Run snapshot: {run}", flush=True)
    if args.prepare_only:
        return 0
    if args.watch_pid:
        return watch_existing(run, args.watch_pid)
    return execute(run, args.min_free_mib, args.max_gpu_utilization)


if __name__ == "__main__":
    sys.exit(main())
