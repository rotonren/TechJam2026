from __future__ import annotations

import hashlib
import json
import os
import statistics
from copy import deepcopy
from pathlib import Path

import pytest


def _report(*, suite: str = "representative", runtime: str = "a" * 64, config: str = "b" * 64, all_hits: bool = False, boundary_hits: int = 1) -> dict[str, object]:
    scenarios = ("buying", "browsing", "intent_override", "boundary")
    folds = []
    fold_ids = (1, 2, 3, 4) if suite == "representative" else (None,)
    for fold in fold_ids:
        sessions = [
            {"sample_id": f"{suite}-{fold}-{name}", "scenario_type": name, "hit": all_hits or (name == "boundary" and boundary_hits > 0), "first_hit_turn": 1 if all_hits or (name == "boundary" and boundary_hits > 0) else None, "best_rank": 1 if all_hits or (name == "boundary" and boundary_hits > 0) else None, "reciprocal_rank": 1.0 if all_hits or (name == "boundary" and boundary_hits > 0) else 0.0}
            for name in scenarios
        ]
        hit_rate = sum(item["hit"] for item in sessions) / len(sessions)
        mrr = sum(item["reciprocal_rank"] for item in sessions) / len(sessions)
        mttc = sum(item["first_hit_turn"] if item["first_hit_turn"] is not None else 11 for item in sessions) / len(sessions)
        efficiency = round(max(0, min(1, (11 - mttc) / 10)), 6)
        score = round(.5 * hit_rate + .3 * mrr + .2 * efficiency, 6)
        scenario_metrics = {name: {"sample_count": 1, "hit_rate_at_10": float(sessions[index]["hit"]), "mrr": sessions[index]["reciprocal_rank"], "mttc": float(sessions[index]["first_hit_turn"] or 11)} for index, name in enumerate(scenarios)}
        folds.append({"fold": fold, "sample_count": 4, "aggregate": {"sample_count": 4, "hit_rate_at_10": round(hit_rate, 6), "mrr": round(mrr, 6), "mttc": round(mttc, 6), "efficiency": efficiency, "recommended_technical_score": score, "reported_token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "scenario_metrics": scenario_metrics, "invalid_response_count": 0}, "latency_ms": {"count": 1, "p50": 1.0, "p95": 1.0, "max": 1.0}, "fallback_count": 0, "route_distribution": {}, "sessions": sessions})
    score = folds[0]["aggregate"]["recommended_technical_score"]
    return {"created_at": "2026-08-27T00:00:00+00:00", "commit": "commit", "config_hash": config, "runtime_hash": runtime, "manifest_hash": "c" * 64, "dataset_hash": "d" * 64, "suite": suite, "fallback_count": 0, "invalid_response_count": 0, "folds": folds, "mean_technical_score": score, "std_technical_score": 0.0, "selection_score": score, "api_cost_usd": 0.0}


def _resource_report(runtime: str = "a" * 64, config: str = "b" * 64) -> dict[str, object]:
    from tools import benchmark_release as bench

    baseline_path = Path("var/balanced-hardening/benchmark-r0-wall-v2.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = deepcopy(bench.validate_baseline_report(baseline))
    for trial in candidate["trials"]:
        trial["runtime_hash"] = runtime
        trial["config_hash"] = config
    candidate.update(bench.aggregate_trials(candidate["trials"]))
    candidate["comparison"] = {
        **bench.compare_reports(candidate, baseline, comparison_mode="resource"),
        "baseline_reference": str(baseline_path.resolve()),
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
    }
    return candidate


def _rebuild(report: dict[str, object]) -> None:
    from tools.run_cv import selection_score
    from tools.run_proxy import metric_summary

    scores = []
    invalid = 0
    for fold in report["folds"]:
        sessions = fold["sessions"]
        overall = metric_summary(sessions)
        efficiency = round(max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0)), 6)
        score = round(.5 * overall["hit_rate_at_10"] + .3 * overall["mrr"] + .2 * efficiency, 6)
        scenarios = {name: metric_summary([item for item in sessions if item["scenario_type"] == name]) for name in ("buying", "browsing", "intent_override", "boundary")}
        fold["aggregate"].update({**overall, "efficiency": efficiency, "recommended_technical_score": score, "scenario_metrics": scenarios})
        scores.append(score)
        invalid += fold["aggregate"]["invalid_response_count"]
    report["invalid_response_count"] = invalid
    report["fallback_count"] = sum(fold["fallback_count"] for fold in report["folds"])
    report["mean_technical_score"] = round(statistics.fmean(scores), 6)
    report["std_technical_score"] = round(statistics.pstdev(scores), 6)
    report["selection_score"] = selection_score(scores, 0.0, invalid / sum(fold["sample_count"] for fold in report["folds"]))


def _make_miss(report: dict[str, object], scenario: str, *, fold_index: int = 0) -> None:
    session = next(item for item in report["folds"][fold_index]["sessions"] if item["scenario_type"] == scenario)
    session.update({"hit": False, "first_hit_turn": None, "best_rank": None, "reciprocal_rank": 0.0})
    _rebuild(report)


def test_development_requires_selection_threshold_and_writes_aggregate_only_receipt(tmp_path: Path) -> None:
    from tools.compare_proxy_stages import (
        compare_development,
        write_development_receipt,
    )

    parent = _report()
    candidate = _report(all_hits=True)
    result = compare_development(parent, candidate, stage="S3")

    assert result["accepted"] is True
    assert all("sample" not in json.dumps(value) for value in result.values())
    receipt = tmp_path / "receipt.json"
    write_development_receipt(receipt, candidate, stage="S3", result=result)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["accepted"] is True
    assert "sample_id" not in json.dumps(payload)


def test_development_receipt_publishes_canonical_bytes_exclusively_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import compare_proxy_stages

    candidate = _report(all_hits=True)
    result = compare_proxy_stages.compare_development(_report(), candidate, stage="S3")
    destination = tmp_path / "receipt.json"
    compare_proxy_stages.write_development_receipt(destination, candidate, stage="S3", result=result)

    expected = json.dumps(
        compare_proxy_stages._receipt_payload(candidate, stage="S3", result=result),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8") + b"\n"
    assert destination.read_bytes() == expected
    with pytest.raises(FileExistsError):
        compare_proxy_stages.write_development_receipt(destination, candidate, stage="S3", result=result)

    failing = tmp_path / "failing.json"
    monkeypatch.setattr(compare_proxy_stages.os, "link", lambda *_args: (_ for _ in ()).throw(OSError("no link")))
    with pytest.raises(RuntimeError, match="hardlink"):
        compare_proxy_stages.write_development_receipt(failing, candidate, stage="S3", result=result)
    assert not failing.exists()
    assert not list(tmp_path.glob(".failing.json.*.tmp"))


def test_compare_cli_preflights_existing_output_before_reading_reports(tmp_path: Path) -> None:
    from tools import compare_proxy_stages

    output = tmp_path / "result.json"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        compare_proxy_stages.main([
            "--phase", "development", "--parent-dev", str(tmp_path / "missing-parent.json"),
            "--candidate-dev", str(tmp_path / "missing-candidate.json"), "--stage", "S3",
            "--output", str(output),
        ])


def test_development_rejects_provenance_and_scoring_stage_cannot_use_correctness_tolerance() -> None:
    from tools.compare_proxy_stages import compare_development

    parent = _report()
    candidate = _report()
    candidate["dataset_hash"] = "e" * 64

    assert compare_development(parent, candidate, stage="S3")["accepted"] is False
    with pytest.raises(ValueError, match="correctness"):
        compare_development(parent, _report(), stage="S3", correctness_fix=True)


def test_complete_requires_matching_dev_stress_and_resource_fingerprints(tmp_path: Path) -> None:
    from tools.compare_proxy_stages import (
        compare_complete,
        compare_development,
        write_development_receipt,
    )

    parent_dev = _report()
    candidate_dev = _report(all_hits=True)
    receipt = tmp_path / "receipt.json"
    development = compare_development(parent_dev, candidate_dev, stage="S3")
    write_development_receipt(receipt, candidate_dev, stage="S3", result=development)
    parent_stress = _report(suite="stress")
    candidate_stress = _report(suite="stress")
    resource = _resource_report()

    accepted = compare_complete(parent_dev, candidate_dev, parent_stress, candidate_stress, resource, receipt, stage="S3", current_runtime_hash="a" * 64, current_config_hash="b" * 64)
    assert accepted["accepted"] is True
    rejected = compare_complete(parent_dev, candidate_dev, parent_stress, _report(suite="stress", runtime="f" * 64), resource, receipt, stage="S3", current_runtime_hash="a" * 64, current_config_hash="b" * 64)
    assert rejected["accepted"] is False


def test_metrics_accepts_real_p0_six_decimal_scenario_aggregates() -> None:
    from tools.compare_proxy_stages import _metrics

    report = json.loads(Path("var/balanced-hardening/proxy-v1/dev-p0.json").read_text(encoding="utf-8"))

    assert _metrics(report)["suite"] == "representative"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report.update({"unexpected": True}),
        lambda report: report["folds"][0].update({"unexpected": True}),
        lambda report: report["folds"][0]["aggregate"].update({"unexpected": True}),
        lambda report: report["folds"][0]["aggregate"]["scenario_metrics"]["buying"].update({"unexpected": True}),
        lambda report: report["folds"][0]["sessions"][0].update({"target": "secret"}),
        lambda report: report["folds"][0]["latency_ms"].update({"unexpected": True}),
        lambda report: report["folds"][0]["aggregate"]["reported_token_usage"].update({"unexpected": True}),
        lambda report: report["folds"][0].update({"route_distribution": {"buying": "one"}}),
    ],
)
def test_metrics_rejects_unexpected_or_malformed_normal_report_schema(mutate) -> None:
    from tools.compare_proxy_stages import _metrics

    report = _report()
    mutate(report)

    with pytest.raises((TypeError, ValueError), match="schema|sessions|latency|usage|route"):
        _metrics(report)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report.pop("api_cost_usd"),
        lambda report: report["folds"][0].pop("sessions"),
        lambda report: report["folds"][0]["aggregate"].pop("reported_token_usage"),
        lambda report: report["folds"][0]["aggregate"]["scenario_metrics"]["buying"].pop("mrr"),
        lambda report: report["folds"][0]["sessions"][0].pop("best_rank"),
        lambda report: report["folds"][0]["latency_ms"].pop("p95"),
        lambda report: report["folds"][0]["aggregate"]["reported_token_usage"].pop("total_tokens"),
    ],
)
def test_metrics_rejects_missing_normal_report_schema_fields(mutate) -> None:
    from tools.compare_proxy_stages import _metrics

    report = _report()
    mutate(report)

    with pytest.raises((TypeError, ValueError), match="schema|sessions|latency|usage|metrics"):
        _metrics(report)


def test_technical_score_uses_unrounded_efficiency_before_six_decimal_output() -> None:
    from tools.compare_proxy_stages import _technical_score

    assert _technical_score(0.696517, 0.310539, 5.893035) == 0.54356


def test_metrics_recomputes_scores_and_rejects_forged_top_level_aggregate() -> None:
    from tools.compare_proxy_stages import _metrics

    report = _report()
    report["mean_technical_score"] = 1.0

    with pytest.raises(ValueError, match="mean"):
        _metrics(report)


def test_development_allows_candidate_runtime_and_config_to_evolve() -> None:
    from tools.compare_proxy_stages import compare_development

    parent = _report()
    candidate = _report(runtime="e" * 64, config="f" * 64, all_hits=True)

    assert compare_development(parent, candidate, stage="S3")["accepted"] is True


@pytest.mark.parametrize(("value", "expected"), [(0.003, 3_000), (0.002999999999, 3_000), (0.0029989, 2_999)])
def test_six_decimal_threshold_units_are_exact(value: float, expected: int) -> None:
    from tools.compare_proxy_stages import _units

    assert _units(value) == expected


def test_complete_rejects_resource_fingerprint_stub(tmp_path: Path) -> None:
    from tools.compare_proxy_stages import (
        compare_complete,
        compare_development,
        write_development_receipt,
    )

    parent_dev, candidate_dev = _report(), _report(all_hits=True)
    result = compare_development(parent_dev, candidate_dev, stage="S3")
    receipt = tmp_path / "receipt.json"
    write_development_receipt(receipt, candidate_dev, stage="S3", result=result)
    stub = {"runtime_hash": "a" * 64, "config_hash": "b" * 64, "comparison": {"accepted": True, "comparison_mode": "resource"}}

    outcome = compare_complete(parent_dev, candidate_dev, _report(suite="stress"), _report(suite="stress"), stub, receipt, stage="S3", current_runtime_hash="a" * 64, current_config_hash="b" * 64)

    assert outcome["accepted"] is False
    assert "resource_report_rejected" in outcome["failure_codes"]


def test_resource_baseline_is_anchored_to_repo_root_outside_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import compare_proxy_stages

    report = _resource_report()
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        assert compare_proxy_stages._resource_is_accepted(report) is True
    finally:
        os.chdir(original_cwd)



@pytest.mark.parametrize("field", ["policy", "deltas", "failure_codes"])
def test_complete_rejects_tampered_development_receipt(tmp_path: Path, field: str) -> None:
    from tools.compare_proxy_stages import (
        compare_complete,
        compare_development,
        write_development_receipt,
    )

    parent_dev, candidate_dev = _report(), _report(all_hits=True)
    receipt = tmp_path / "receipt.json"
    result = compare_development(parent_dev, candidate_dev, stage="S3")
    write_development_receipt(receipt, candidate_dev, stage="S3", result=result)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload[field] = {"tampered": True} if field != "failure_codes" else ["tampered"]
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="receipt"):
        compare_complete(parent_dev, candidate_dev, _report(suite="stress"), _report(suite="stress"), _resource_report(), receipt, stage="S3", current_runtime_hash="a" * 64, current_config_hash="b" * 64)


def test_development_rejects_candidate_missing_runtime_hash() -> None:
    from tools.compare_proxy_stages import compare_development

    candidate = _report(all_hits=True)
    candidate.pop("runtime_hash")

    assert compare_development(_report(), candidate, stage="S3")["accepted"] is False


def test_development_rejects_samples_moved_between_folds() -> None:
    from tools.compare_proxy_stages import compare_development

    parent, candidate = _report(all_hits=True), _report(all_hits=True)
    candidate["folds"][0]["sessions"][0]["sample_id"], candidate["folds"][1]["sessions"][0]["sample_id"] = (
        candidate["folds"][1]["sessions"][0]["sample_id"],
        candidate["folds"][0]["sessions"][0]["sample_id"],
    )

    outcome = compare_development(parent, candidate, stage="S3")

    assert outcome["accepted"] is False
    assert "development_fold_sample_set_mismatch" in outcome["failure_codes"]


def test_development_rejects_sample_scenario_relabeling() -> None:
    from tools.compare_proxy_stages import compare_development

    parent, candidate = _report(all_hits=True), _report(all_hits=True)
    candidate["folds"][0]["sessions"][0]["scenario_type"], candidate["folds"][0]["sessions"][1]["scenario_type"] = (
        candidate["folds"][0]["sessions"][1]["scenario_type"],
        candidate["folds"][0]["sessions"][0]["scenario_type"],
    )

    outcome = compare_development(parent, candidate, stage="S3")

    assert outcome["accepted"] is False
    assert "development_sample_scenario_mismatch" in outcome["failure_codes"]


def test_metrics_rejects_hit_at_rank_eleven() -> None:
    from tools.compare_proxy_stages import _metrics

    report = _report(all_hits=True)
    report["folds"][0]["sessions"][0].update({"best_rank": 11, "reciprocal_rank": 1 / 11})

    with pytest.raises(ValueError, match="sessions"):
        _metrics(report)


def test_six_decimal_aggregate_validation_rejects_one_unit_difference() -> None:
    from tools.compare_proxy_stages import _same_six

    with pytest.raises(ValueError, match="aggregate"):
        _same_six(0.500001, 0.5, "aggregate")


def test_compare_stress_and_compare_stage_helpers_apply_the_same_gates() -> None:
    from tools.compare_proxy_stages import compare_stage, compare_stress

    parent, candidate = _report(suite="stress"), _report(suite="stress")

    assert compare_stress(parent, candidate)["accepted"] is True
    result = compare_stage(_report(), _report(all_hits=True), parent, candidate, stage="S3")
    assert result["accepted"] is True


@pytest.mark.parametrize(
    ("mutate", "failure_code"),
    [
        (lambda parent_dev, candidate_dev, parent_stress, candidate_stress: parent_stress.update({"config_hash": "e" * 64}), "parent_source_identity_mismatch"),
        (lambda parent_dev, candidate_dev, parent_stress, candidate_stress: parent_stress.update({"runtime_hash": "e" * 64}), "parent_source_identity_mismatch"),
        (lambda parent_dev, candidate_dev, parent_stress, candidate_stress: candidate_stress.update({"config_hash": "e" * 64}), "candidate_config_identity_mismatch"),
        (lambda parent_dev, candidate_dev, parent_stress, candidate_stress: candidate_stress.update({"runtime_hash": "e" * 64}), "candidate_runtime_identity_mismatch"),
    ],
)
def test_stage_rejects_cross_report_lineage_mismatches(mutate, failure_code: str) -> None:
    from tools.compare_proxy_stages import compare_stage

    parent_dev, candidate_dev = _report(), _report(all_hits=True)
    parent_stress, candidate_stress = _report(suite="stress"), _report(suite="stress")
    mutate(parent_dev, candidate_dev, parent_stress, candidate_stress)

    outcome = compare_stage(parent_dev, candidate_dev, parent_stress, candidate_stress, stage="S3")

    assert failure_code in outcome["failure_codes"]


def test_stage_accepts_legacy_parent_with_same_valid_commit() -> None:
    from tools.compare_proxy_stages import compare_stage

    parent_dev, candidate_dev = _report(), _report(all_hits=True)
    parent_stress, candidate_stress = _report(suite="stress"), _report(suite="stress")
    for report in (parent_dev, parent_stress):
        report.pop("runtime_hash")
        report["commit"] = "ac36def"

    outcome = compare_stage(parent_dev, candidate_dev, parent_stress, candidate_stress, stage="S3")

    assert outcome["accepted"] is True


def _gate_metrics() -> dict[str, object]:
    return {
        "selection": 1.0,
        "mean": 1.0,
        "scores": {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        "scenario_rates": {"buying": 1.0, "browsing": 1.0, "intent_override": 1.0},
    }


def _set_gate_delta(metrics: dict[str, object], metric: str, delta: float) -> None:
    if metric in {"selection", "mean"}:
        metrics[metric] = 1.0 + delta
    elif metric.startswith("fold_"):
        metrics["scores"][int(metric.removeprefix("fold_"))] = 1.0 + delta
    else:
        metrics["scenario_rates"][metric] = 1.0 + delta


@pytest.mark.parametrize(
    ("stage", "metric", "threshold", "failure_code"),
    [
        ("S3", "selection", 0.003, "development_selection_regression"),
        ("S1", "selection", -0.001, "development_selection_regression"),
        ("S3", "mean", 0.0, "development_mean_regression"),
        ("S3", "fold_1", -0.015, "development_fold_regression"),
        ("S3", "fold_2", -0.015, "development_fold_regression"),
        ("S3", "fold_3", -0.015, "development_fold_regression"),
        ("S3", "fold_4", -0.015, "development_fold_regression"),
        ("S3", "buying", -0.02, "development_buying_regression"),
        ("S3", "browsing", -0.02, "development_browsing_regression"),
        ("S3", "intent_override", -0.02, "development_intent_override_regression"),
    ],
)
def test_development_gate_accepts_exact_threshold_and_rejects_one_micro_unit_lower(stage: str, metric: str, threshold: float, failure_code: str) -> None:
    from tools.compare_proxy_stages import _development_gate, _policy

    parent = _gate_metrics()
    candidate = deepcopy(parent)
    candidate["selection"] = 1.0 + _policy(stage).minimum_selection_delta
    _set_gate_delta(candidate, metric, threshold)

    assert _development_gate(parent, candidate, _policy(stage))["failure_codes"] == []

    _set_gate_delta(candidate, metric, threshold - 0.000001)

    assert failure_code in _development_gate(parent, candidate, _policy(stage))["failure_codes"]


@pytest.mark.parametrize(
    ("metric", "threshold", "failure_code"),
    [
        ("mean", -0.01, "stress_mean_regression"),
        ("buying", -0.025, "stress_buying_regression"),
        ("browsing", -0.025, "stress_browsing_regression"),
        ("intent_override", -0.025, "stress_intent_override_regression"),
    ],
)
def test_stress_gate_accepts_exact_threshold_and_rejects_one_micro_unit_lower(metric: str, threshold: float, failure_code: str) -> None:
    from tools.compare_proxy_stages import _policy, _stress_gate

    parent = _gate_metrics()
    candidate = deepcopy(parent)
    _set_gate_delta(candidate, metric, threshold)

    assert _stress_gate(parent, candidate, _policy("S3"))["failure_codes"] == []

    _set_gate_delta(candidate, metric, threshold - 0.000001)

    assert failure_code in _stress_gate(parent, candidate, _policy("S3"))["failure_codes"]


@pytest.mark.parametrize("commit", [None, "not-a-sha", "a" * 41])
def test_complete_rejects_legacy_parent_without_matching_recorded_git_commit(
    tmp_path: Path, commit: object
) -> None:
    from tools.compare_proxy_stages import (
        compare_complete,
        compare_development,
        write_development_receipt,
    )

    parent_dev, candidate_dev = _report(), _report(all_hits=True)
    parent_stress = _report(suite="stress")
    parent_dev.pop("runtime_hash")
    parent_stress.pop("runtime_hash")
    parent_dev["commit"] = commit
    parent_stress["commit"] = commit
    receipt = tmp_path / "receipt.json"
    development = compare_development(parent_dev, candidate_dev, stage="S3")
    write_development_receipt(receipt, candidate_dev, stage="S3", result=development)

    outcome = compare_complete(parent_dev, candidate_dev, parent_stress, _report(suite="stress"), _resource_report(), receipt, stage="S3", current_runtime_hash="a" * 64, current_config_hash="b" * 64)

    assert outcome["accepted"] is False
    assert "parent_source_identity_mismatch" in outcome["failure_codes"]


@pytest.mark.parametrize(
    ("mutate", "failure_code"),
    [
        (lambda report: report.update({"manifest_hash": "e" * 64}), "development_manifest_hash_mismatch"),
        (lambda report: report["folds"][0]["sessions"][0].update({"sample_id": "new-id"}), "development_sample_set_mismatch"),
        (lambda report: report["folds"][0]["sessions"].__setitem__(0, {**report["folds"][0]["sessions"][0], "scenario_type": "browsing"}), "development_sample_scenario_mismatch"),
    ],
)
def test_development_provenance_matrix(mutate, failure_code: str) -> None:
    from tools.compare_proxy_stages import compare_development

    candidate = _report(all_hits=True)
    mutate(candidate)
    if candidate["suite"] == "representative":
        _rebuild(candidate)

    outcome = compare_development(_report(all_hits=True), candidate, stage="S3")

    assert failure_code in outcome["failure_codes"]


def test_development_rejects_valid_stress_suite_as_mismatched_provenance() -> None:
    from tools.compare_proxy_stages import compare_development

    outcome = compare_development(_report(all_hits=True), _report(suite="stress", all_hits=True), stage="S3")

    assert "development_suite_mismatch" in outcome["failure_codes"]


@pytest.mark.parametrize("scenario", ["buying", "browsing", "intent_override", "boundary"])
def test_development_scenario_and_boundary_gate_matrix(scenario: str) -> None:
    from tools.compare_proxy_stages import compare_development

    parent, candidate = _report(all_hits=True), _report(all_hits=True)
    _make_miss(candidate, scenario)

    outcome = compare_development(parent, candidate, stage="S3")

    expected = "development_boundary_regression" if scenario == "boundary" else f"development_{scenario}_regression"
    assert expected in outcome["failure_codes"]


def test_development_fold_fallback_and_invalid_gate_matrix() -> None:
    from tools.compare_proxy_stages import compare_development

    parent, fold_regression = _report(all_hits=True), _report(all_hits=True)
    _make_miss(fold_regression, "buying", fold_index=0)
    assert "development_fold_regression" in compare_development(parent, fold_regression, stage="S3")["failure_codes"]
    runtime_invalid = _report(all_hits=True)
    runtime_invalid["folds"][0]["fallback_count"] = 1
    runtime_invalid["folds"][0]["aggregate"]["invalid_response_count"] = 1
    _rebuild(runtime_invalid)
    assert "candidate_runtime_invalid" in compare_development(parent, runtime_invalid, stage="S3")["failure_codes"]


@pytest.mark.parametrize("scenario", ["buying", "browsing", "intent_override", "boundary"])
def test_stress_scenario_and_boundary_gate_matrix(scenario: str) -> None:
    from tools.compare_proxy_stages import compare_stress

    parent, candidate = _report(suite="stress", all_hits=True), _report(suite="stress", all_hits=True)
    _make_miss(candidate, scenario)

    outcome = compare_stress(parent, candidate)

    expected = "stress_boundary_regression" if scenario == "boundary" else f"stress_{scenario}_regression"
    assert expected in outcome["failure_codes"]


def test_stress_overall_fallback_and_invalid_gate_matrix() -> None:
    from tools.compare_proxy_stages import compare_stress

    parent, candidate = _report(suite="stress", all_hits=True), _report(suite="stress", all_hits=True)
    _make_miss(candidate, "buying")
    assert "stress_mean_regression" in compare_stress(parent, candidate)["failure_codes"]
    candidate = _report(suite="stress", all_hits=True)
    candidate["folds"][0]["fallback_count"] = 1
    candidate["folds"][0]["aggregate"]["invalid_response_count"] = 1
    _rebuild(candidate)
    assert "candidate_runtime_invalid" in compare_stress(parent, candidate)["failure_codes"]


def test_complete_accepts_legacy_parent_with_same_valid_commit(tmp_path: Path) -> None:
    from tools.compare_proxy_stages import (
        compare_complete,
        compare_development,
        write_development_receipt,
    )

    parent_dev, candidate_dev = _report(), _report(all_hits=True)
    parent_stress = _report(suite="stress")
    for report in (parent_dev, parent_stress):
        report.pop("runtime_hash")
        report["commit"] = "ac36def"
    receipt = tmp_path / "receipt.json"
    development = compare_development(parent_dev, candidate_dev, stage="S3")
    write_development_receipt(receipt, candidate_dev, stage="S3", result=development)

    outcome = compare_complete(parent_dev, candidate_dev, parent_stress, _report(suite="stress"), _resource_report(), receipt, stage="S3", current_runtime_hash="a" * 64, current_config_hash="b" * 64)

    assert outcome["accepted"] is True
