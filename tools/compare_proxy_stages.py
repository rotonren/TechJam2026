"""Fail-closed aggregate gates for CompassCart proxy stage experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from compasscart.config import RuntimeConfig
from tools.runtime_fingerprint import config_hash, runtime_hash

_HASH_RE: Final = __import__("re").compile(r"^[0-9a-f]{64}$")
_SCENARIOS: Final = frozenset({"buying", "browsing", "intent_override", "boundary"})
_PRIMARY_SCENARIOS: Final = ("buying", "browsing", "intent_override")


@dataclass(frozen=True)
class StageGatePolicy:
    minimum_selection_delta: float
    minimum_mean_delta: float = 0.0
    maximum_fold_decline: float = 0.015
    maximum_dev_scenario_decline: float = 0.02
    maximum_stress_decline: float = 0.01
    maximum_stress_scenario_decline: float = 0.025


def _policy(stage: str, correctness_fix: bool = False) -> StageGatePolicy:
    family = stage.split("-", 1)[0].upper()
    if family not in {"S1", "S2", "S3", "S4", "S5"}:
        raise ValueError("stage is invalid")
    if correctness_fix and family not in {"S1", "S2"}:
        raise ValueError("correctness tolerance is only available to S1/S2")
    return StageGatePolicy(-0.001 if family in {"S1", "S2"} else 0.003)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("report contains a non-finite number")
    return float(value)


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")
    return value


def _sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _read(path: str | Path) -> dict[str, object]:
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("report is invalid") from error
    if not isinstance(report, dict):
        raise TypeError("report is invalid")
    return report


def _metrics(report: dict[str, object]) -> dict[str, object]:
    suite = report.get("suite")
    if suite not in {"representative", "stress"}:
        raise ValueError("suite is invalid")
    for name in ("config_hash", "manifest_hash", "dataset_hash"):
        _hash(report.get(name), name)
    if "runtime_hash" in report:
        _hash(report["runtime_hash"], "runtime_hash")
    for name in ("fallback_count", "invalid_response_count"):
        value = report.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("count is invalid")
    folds = report.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("folds are invalid")
    expected_folds = {1, 2, 3, 4} if suite == "representative" else {None}
    by_fold: dict[object, dict[str, object]] = {}
    sample_ids: set[str] = set()
    scenario_set: set[str] = set()
    scenario_rates: dict[str, list[float]] = {name: [] for name in _SCENARIOS}
    boundary_hits = 0
    scores: dict[object, float] = {}
    fallback = invalid = 0
    for fold in folds:
        if not isinstance(fold, dict) or {"fold", "sample_count", "aggregate", "latency_ms", "fallback_count", "route_distribution", "sessions"} - set(fold):
            raise ValueError("fold schema is invalid")
        fold_id = fold["fold"]
        if fold_id not in expected_folds or fold_id in by_fold:
            raise ValueError("folds are invalid")
        if isinstance(fold["sample_count"], bool) or not isinstance(fold["sample_count"], int) or fold["sample_count"] < 1:
            raise ValueError("sample count is invalid")
        aggregate = fold["aggregate"]
        if not isinstance(aggregate, dict) or not isinstance(aggregate.get("scenario_metrics"), dict):
            raise TypeError("aggregate is invalid")
        scores[fold_id] = _number(aggregate.get("recommended_technical_score"))
        invalid_value = aggregate.get("invalid_response_count", 0)
        if isinstance(invalid_value, bool) or not isinstance(invalid_value, int) or invalid_value < 0:
            raise ValueError("invalid response count is invalid")
        invalid += invalid_value
        fallback_value = fold["fallback_count"]
        if isinstance(fallback_value, bool) or not isinstance(fallback_value, int) or fallback_value < 0:
            raise ValueError("fallback count is invalid")
        fallback += fallback_value
        sessions = fold["sessions"]
        if not isinstance(sessions, list) or len(sessions) != fold["sample_count"]:
            raise ValueError("sessions are invalid")
        session_scenarios: dict[str, list[bool]] = {name: [] for name in _SCENARIOS}
        for session in sessions:
            if not isinstance(session, dict) or not isinstance(session.get("sample_id"), str) or not session["sample_id"]:
                raise ValueError("sessions are invalid")
            scenario = session.get("scenario_type")
            if scenario not in _SCENARIOS or not isinstance(session.get("hit"), bool):
                raise ValueError("sessions are invalid")
            if session["sample_id"] in sample_ids:
                raise ValueError("duplicate sample IDs")
            sample_ids.add(session["sample_id"])
            scenario_set.add(scenario)
            session_scenarios[scenario].append(session["hit"])
            boundary_hits += int(scenario == "boundary" and session["hit"])
        metrics = aggregate["scenario_metrics"]
        if set(metrics) != _SCENARIOS:
            raise ValueError("scenario metrics are invalid")
        for scenario, metric in metrics.items():
            if not isinstance(metric, dict):
                raise TypeError("scenario metrics are invalid")
            sample_count = metric.get("sample_count")
            rate = _number(metric.get("hit_rate_at_10"))
            if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count != len(session_scenarios[scenario]):
                raise ValueError("scenario metrics are inconsistent")
            expected_rate = sum(session_scenarios[scenario]) / sample_count
            if not math.isclose(rate, expected_rate, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("scenario metrics are inconsistent")
            scenario_rates[scenario].append(rate)
        by_fold[fold_id] = fold
    if set(by_fold) != expected_folds or scenario_set != _SCENARIOS:
        raise ValueError("fold or scenario set is invalid")
    mean = _number(report.get("mean_technical_score"))
    if not math.isclose(mean, sum(scores.values()) / len(scores), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("mean score is inconsistent")
    selection = _number(report.get("selection_score"))
    if fallback != report["fallback_count"] or invalid != report["invalid_response_count"]:
        raise ValueError("aggregate counts are inconsistent")
    return {"suite": suite, "runtime_hash": report.get("runtime_hash"), "config_hash": report["config_hash"], "manifest_hash": report["manifest_hash"], "dataset_hash": report["dataset_hash"], "commit": report.get("commit"), "selection": selection, "mean": mean, "scores": scores, "scenario_rates": {name: sum(values) / len(values) for name, values in scenario_rates.items()}, "boundary_hits": boundary_hits, "fallback": fallback, "invalid": invalid, "sample_ids": frozenset(sample_ids)}


def _result(deltas: dict[str, float], failures: list[str]) -> dict[str, object]:
    return {"accepted": not failures, "deltas": {key: round(value, 6) for key, value in sorted(deltas.items())}, "failure_codes": sorted(set(failures))}


def _same_provenance(parent: dict[str, object], candidate: dict[str, object], *, prefix: str, require_runtime: bool = True) -> list[str]:
    failures = []
    for name in ("suite", "manifest_hash", "dataset_hash", "config_hash"):
        if parent[name] != candidate[name]:
            failures.append(f"{prefix}_{name}_mismatch")
    if require_runtime and (not parent["runtime_hash"] or not candidate["runtime_hash"] or parent["runtime_hash"] != candidate["runtime_hash"]):
        failures.append(f"{prefix}_runtime_hash_mismatch")
    if parent["sample_ids"] != candidate["sample_ids"]:
        failures.append(f"{prefix}_sample_set_mismatch")
    return failures


def compare_development(parent_dev: dict[str, object], candidate_dev: dict[str, object], *, stage: str, correctness_fix: bool = False) -> dict[str, object]:
    policy = _policy(stage, correctness_fix)
    parent, candidate = _metrics(parent_dev), _metrics(candidate_dev)
    failures = _same_provenance(parent, candidate, prefix="development")
    deltas = {"selection": candidate["selection"] - parent["selection"], "mean": candidate["mean"] - parent["mean"]}
    if deltas["selection"] < policy.minimum_selection_delta:
        failures.append("development_selection_regression")
    if deltas["mean"] < policy.minimum_mean_delta:
        failures.append("development_mean_regression")
    for fold_id, parent_score in parent["scores"].items():
        delta = candidate["scores"][fold_id] - parent_score
        deltas[f"fold_{fold_id}"] = delta
        if delta < -policy.maximum_fold_decline:
            failures.append("development_fold_regression")
    for scenario in _PRIMARY_SCENARIOS:
        delta = candidate["scenario_rates"][scenario] - parent["scenario_rates"][scenario]
        deltas[f"{scenario}_hit_rate"] = delta
        if delta < -policy.maximum_dev_scenario_decline:
            failures.append(f"development_{scenario}_regression")
    deltas["boundary_hits"] = candidate["boundary_hits"] - parent["boundary_hits"]
    if deltas["boundary_hits"] < 0:
        failures.append("development_boundary_regression")
    if candidate["fallback"] or candidate["invalid"]:
        failures.append("candidate_runtime_invalid")
    return _result(deltas, failures)


def write_development_receipt(destination: str | Path, candidate_dev: dict[str, object], *, stage: str, result: dict[str, object]) -> None:
    candidate = _metrics(candidate_dev)
    receipt = {"stage": stage, "policy": asdict(_policy(stage)), "candidate_report_sha256": _sha256_json(candidate_dev), "runtime_hash": candidate["runtime_hash"], "config_hash": candidate["config_hash"], "accepted": result["accepted"], "deltas": result["deltas"], "failure_codes": result["failure_codes"]}
    Path(destination).write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def _load_receipt(path: str | Path, candidate: dict[str, object], stage: str) -> dict[str, object]:
    receipt = _read(path)
    expected = {"stage", "policy", "candidate_report_sha256", "runtime_hash", "config_hash", "accepted", "deltas", "failure_codes"}
    if set(receipt) != expected or receipt["stage"] != stage or receipt["candidate_report_sha256"] != _sha256_json(candidate) or receipt["accepted"] is not True:
        raise ValueError("development receipt is invalid")
    return receipt


def compare_complete(parent_dev: dict[str, object], candidate_dev: dict[str, object], parent_stress: dict[str, object], candidate_stress: dict[str, object], resource_report: dict[str, object], development_receipt: str | Path, *, stage: str, current_runtime_hash: str | None = None, current_config_hash: str | None = None) -> dict[str, object]:
    receipt = _load_receipt(development_receipt, candidate_dev, stage)
    development = compare_development(parent_dev, candidate_dev, stage=stage)
    parent_development, candidate_development = _metrics(parent_dev), _metrics(candidate_dev)
    parent, candidate = _metrics(parent_stress), _metrics(candidate_stress)
    failures = list(development["failure_codes"])
    failures.extend(_same_provenance(parent, candidate, prefix="stress"))
    legacy_parent = not parent["runtime_hash"] and not parent_development["runtime_hash"]
    parent_identity_matches = parent["config_hash"] == parent_development["config_hash"] and (
        (legacy_parent and parent["commit"] == parent_development["commit"])
        or (not legacy_parent and parent["runtime_hash"] == parent_development["runtime_hash"])
    )
    if not parent_identity_matches:
        failures.append("parent_source_identity_mismatch")
    fingerprints = (candidate_development["runtime_hash"], candidate["runtime_hash"], resource_report.get("runtime_hash"), receipt["runtime_hash"], current_runtime_hash or runtime_hash())
    configs = (candidate_development["config_hash"], candidate["config_hash"], resource_report.get("config_hash"), receipt["config_hash"], current_config_hash or config_hash(RuntimeConfig()))
    if not all(isinstance(item, str) and item == fingerprints[0] for item in fingerprints):
        failures.append("candidate_runtime_identity_mismatch")
    if not all(isinstance(item, str) and item == configs[0] for item in configs):
        failures.append("candidate_config_identity_mismatch")
    comparison = resource_report.get("comparison")
    if not isinstance(comparison, dict) or comparison.get("comparison_mode") != "resource" or comparison.get("accepted") is not True:
        failures.append("resource_report_rejected")
    deltas = dict(development["deltas"])
    overall = candidate["mean"] - parent["mean"]
    deltas["stress_mean"] = overall
    if overall < -_policy(stage).maximum_stress_decline:
        failures.append("stress_mean_regression")
    for scenario in _PRIMARY_SCENARIOS:
        delta = candidate["scenario_rates"][scenario] - parent["scenario_rates"][scenario]
        deltas[f"stress_{scenario}_hit_rate"] = delta
        if delta < -_policy(stage).maximum_stress_scenario_decline:
            failures.append(f"stress_{scenario}_regression")
    deltas["stress_boundary_hits"] = candidate["boundary_hits"] - parent["boundary_hits"]
    if deltas["stress_boundary_hits"] < 0:
        failures.append("stress_boundary_regression")
    if candidate["fallback"] or candidate["invalid"]:
        failures.append("candidate_runtime_invalid")
    return _result(deltas, failures)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "complete"), required=True)
    parser.add_argument("--parent-dev", required=True)
    parser.add_argument("--candidate-dev", required=True)
    parser.add_argument("--parent-stress")
    parser.add_argument("--candidate-stress")
    parser.add_argument("--resource-report")
    parser.add_argument("--development-receipt")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    parent_dev, candidate_dev = _read(args.parent_dev), _read(args.candidate_dev)
    if args.phase == "development":
        result = compare_development(parent_dev, candidate_dev, stage=args.stage)
        write_development_receipt(args.output, candidate_dev, stage=args.stage, result=result)
    else:
        required = (args.parent_stress, args.candidate_stress, args.resource_report, args.development_receipt)
        if not all(required):
            parser.error("complete phase requires stress, resource, and development receipt inputs")
        result = compare_complete(parent_dev, candidate_dev, _read(args.parent_stress), _read(args.candidate_stress), _read(args.resource_report), args.development_receipt, stage=args.stage)
        Path(args.output).write_text(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
