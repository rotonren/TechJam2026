from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl


def assign_folds(samples: list[dict], *, seed: int = 2026) -> dict[str, int]:
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sample in samples:
        key = (
            str(sample.get("scenario_type", "unknown")),
            str(sample.get("difficulty_bucket", "unknown")),
        )
        strata[key].append(str(sample["sample_id"]))

    assignments: dict[str, int] = {}
    for key in sorted(strata):
        identifiers = sorted(strata[key])
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(identifiers)
        for position, identifier in enumerate(identifiers):
            assignments[identifier] = position % 5 + 1
    return assignments


def select_samples(
    samples: list[dict],
    assignments: dict[str, int],
    folds: list[int],
    *,
    audit: bool,
) -> list[dict]:
    allowed = set(folds)
    if not audit:
        allowed.discard(5)
    return [
        sample for sample in samples if assignments[str(sample["sample_id"])] in allowed
    ]


def selection_score(
    technical_scores: list[float],
    timeout_rate: float,
    invalid_response_rate: float,
) -> float:
    if not technical_scores:
        return 0.0
    deviation = statistics.pstdev(technical_scores)
    return round(
        statistics.fmean(technical_scores)
        - 0.5 * deviation
        - timeout_rate
        - invalid_response_rate,
        6,
    )


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "p50": round(statistics.median(ordered), 3),
        "p95": round(ordered[p95_index], 3),
        "max": round(ordered[-1], 3),
    }


def _trace_summary(records: list[dict[str, object]]) -> dict[str, object]:
    """Summarize route and controlled-relaxation evidence for an evaluation fold."""
    route_distribution = Counter(str(record.get("route", "unknown")) for record in records)
    relaxed_turn_count = sum(
        bool(record.get("relaxed_count", 0)) for record in records
    )
    hard_filter_violation_count = sum(
        int(record.get("relaxed_count", 0) or 0) for record in records
    )
    return {
        "route_distribution": dict(sorted(route_distribution.items())),
        "relaxed_turn_count": relaxed_turn_count,
        "hard_filter_violation_count": hard_filter_violation_count,
    }


def _load_agent(specification: str):
    module_name, class_name = specification.split(":", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _canonical_config_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_config_value(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {
            str(key): _canonical_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_config_value(item) for item in value]
    return value


def _config_hash(agent: object) -> str:
    canonical = _canonical_config_value(getattr(agent, "config", None))
    payload = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--agent", default="agent:Agent")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments"))
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    assignments = assign_folds(samples, seed=args.seed)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent_class = _load_agent(args.agent)
    fold_results: list[dict[str, object]] = []
    config_hash = ""
    for fold in sorted(set(args.folds)):
        fold_samples = select_samples(samples, assignments, [fold], audit=args.audit)
        if not fold_samples:
            continue
        agent = agent_class(args.catalog)
        config_hash = config_hash or _config_hash(agent)
        result = evaluate(agent, fold_samples, catalog_ids, categories, products)
        trace_sink = getattr(agent, "traces", None)
        records = trace_sink.records if hasattr(trace_sink, "records") else []
        fold_report = {
            "fold": fold,
            "sample_count": len(fold_samples),
            "aggregate": {
                key: value for key, value in result.items() if key != "sessions"
            },
            "latency_ms": _latency_summary(
                [float(record.get("elapsed_ms", 0.0)) for record in records]
            ),
            "fallback_count": sum(bool(record.get("fallbacks")) for record in records),
            **_trace_summary(records),
        }
        if not args.audit:
            fold_report["sessions"] = result["sessions"]
        fold_results.append(fold_report)

    technical_scores = [
        float(item["aggregate"]["recommended_technical_score"]) for item in fold_results
    ]
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "commit": _git_commit(),
        "config_hash": config_hash,
        "seed": args.seed,
        "audit": args.audit,
        "folds": fold_results,
        "mean_technical_score": round(statistics.fmean(technical_scores), 6)
        if technical_scores
        else 0.0,
        "std_technical_score": round(statistics.pstdev(technical_scores), 6)
        if technical_scores
        else 0.0,
        "selection_score": selection_score(technical_scores, 0.0, 0.0),
        "api_cost_usd": 0.0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"cv-{stamp}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
