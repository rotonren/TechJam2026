from __future__ import annotations

import hashlib
import json
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
        folds.append({"fold": fold, "sample_count": 4, "aggregate": {"sample_count": 4, "hit_rate_at_10": round(hit_rate, 6), "mrr": round(mrr, 6), "mttc": round(mttc, 6), "efficiency": efficiency, "recommended_technical_score": score, "scenario_metrics": scenario_metrics, "invalid_response_count": 0}, "latency_ms": {"count": 1, "p50": 1.0, "p95": 1.0, "max": 1.0}, "fallback_count": 0, "route_distribution": {}, "sessions": sessions})
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
