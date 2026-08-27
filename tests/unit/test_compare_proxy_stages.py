from __future__ import annotations

import json
from pathlib import Path

import pytest


def _report(*, suite: str = "representative", runtime: str = "a" * 64, config: str = "b" * 64, selection: float = 0.5, score: float = 0.5, scenario_score: float = 0.5, boundary_hits: int = 1) -> dict[str, object]:
    scenarios = {name: {"sample_count": 1, "hit_rate_at_10": float(name == "boundary" and boundary_hits > 0), "mrr": scenario_score, "mttc": 1.0} for name in ("buying", "browsing", "intent_override", "boundary")}
    folds = []
    fold_ids = (1, 2, 3, 4) if suite == "representative" else (None,)
    for fold in fold_ids:
        sessions = [
            {"sample_id": f"{suite}-{fold}-{name}", "scenario_type": name, "hit": name == "boundary" and boundary_hits > 0}
            for name in scenarios
        ]
        folds.append({"fold": fold, "sample_count": 4, "aggregate": {"recommended_technical_score": score, "scenario_metrics": scenarios, "invalid_response_count": 0}, "latency_ms": {"count": 1, "p50": 1.0, "p95": 1.0, "max": 1.0}, "fallback_count": 0, "route_distribution": {}, "sessions": sessions})
    return {"created_at": "2026-08-27T00:00:00+00:00", "commit": "commit", "config_hash": config, "runtime_hash": runtime, "manifest_hash": "c" * 64, "dataset_hash": "d" * 64, "suite": suite, "fallback_count": 0, "invalid_response_count": 0, "folds": folds, "mean_technical_score": score, "std_technical_score": 0.0, "selection_score": selection, "api_cost_usd": 0.0}


def test_development_requires_selection_threshold_and_writes_aggregate_only_receipt(tmp_path: Path) -> None:
    from tools.compare_proxy_stages import (
        compare_development,
        write_development_receipt,
    )

    parent = _report()
    candidate = _report(selection=0.503)
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
    candidate = _report(selection=0.499)
    candidate["dataset_hash"] = "e" * 64

    assert compare_development(parent, candidate, stage="S3")["accepted"] is False
    with pytest.raises(ValueError, match="correctness"):
        compare_development(parent, _report(selection=0.499), stage="S3", correctness_fix=True)


def test_complete_requires_matching_dev_stress_and_resource_fingerprints(tmp_path: Path) -> None:
    from tools.compare_proxy_stages import (
        compare_complete,
        compare_development,
        write_development_receipt,
    )

    parent_dev = _report()
    candidate_dev = _report(selection=0.503)
    receipt = tmp_path / "receipt.json"
    development = compare_development(parent_dev, candidate_dev, stage="S3")
    write_development_receipt(receipt, candidate_dev, stage="S3", result=development)
    parent_stress = _report(suite="stress")
    candidate_stress = _report(suite="stress")
    resource = {"runtime_hash": "a" * 64, "config_hash": "b" * 64, "comparison": {"accepted": True, "comparison_mode": "resource"}}

    accepted = compare_complete(parent_dev, candidate_dev, parent_stress, candidate_stress, resource, receipt, stage="S3", current_runtime_hash="a" * 64, current_config_hash="b" * 64)
    assert accepted["accepted"] is True
    rejected = compare_complete(parent_dev, candidate_dev, parent_stress, _report(suite="stress", runtime="f" * 64), resource, receipt, stage="S3", current_runtime_hash="a" * 64, current_config_hash="b" * 64)
    assert rejected["accepted"] is False
