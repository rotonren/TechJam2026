from __future__ import annotations

from types import SimpleNamespace

from src.compasscart.config import RuntimeConfig
from tools.analyze_failures import summarize_failures
from tools.compare_results import compare_results
from tools.run_cv import (
    _config_hash,
    _latency_summary,
    assign_folds,
    select_samples,
    selection_score,
)


def test_config_hash_is_canonical_and_stable():
    agent = SimpleNamespace(config=RuntimeConfig())

    assert _config_hash(agent) == (
        "4400c69e62a123979d7cadef7b9384f975b9fda28dd18aa93a8a3758544b65e5"
    )


def _samples() -> list[dict]:
    result = []
    for scenario in ("buying", "browsing", "intent_override", "boundary"):
        for difficulty in ("easy", "medium", "hard"):
            for index in range(5):
                result.append(
                    {
                        "sample_id": f"{scenario}-{difficulty}-{index}",
                        "scenario_type": scenario,
                        "difficulty_bucket": difficulty,
                    }
                )
    return result


def test_five_fold_assignment_is_deterministic_and_stratified():
    samples = _samples()

    first = assign_folds(samples, seed=2026)
    second = assign_folds(list(reversed(samples)), seed=2026)

    assert first == second
    assert set(first) == {sample["sample_id"] for sample in samples}
    for scenario in ("buying", "browsing", "intent_override", "boundary"):
        for difficulty in ("easy", "medium", "hard"):
            identifiers = [
                sample["sample_id"]
                for sample in samples
                if sample["scenario_type"] == scenario
                and sample["difficulty_bucket"] == difficulty
            ]
            assert {first[identifier] for identifier in identifiers} == {1, 2, 3, 4, 5}


def test_sealed_fold_is_excluded_without_audit_flag():
    samples = _samples()
    assignments = assign_folds(samples, seed=2026)

    development = select_samples(samples, assignments, [1, 5], audit=False)
    audit = select_samples(samples, assignments, [5], audit=True)

    assert all(assignments[item["sample_id"]] == 1 for item in development)
    assert audit and all(assignments[item["sample_id"]] == 5 for item in audit)


def test_selection_score_penalizes_variance_and_invalid_outputs():
    assert selection_score([0.4, 0.5, 0.4, 0.5], 0.01, 0.02) < 0.45
    assert selection_score([0.45] * 4, 0.0, 0.0) == 0.45


def test_latency_summary_is_compact_and_uses_nearest_rank_p95():
    summary = _latency_summary([1.0, 2.0, 3.0, 4.0, 100.0])

    assert summary == {"count": 5, "p50": 3.0, "p95": 100.0, "max": 100.0}


def test_failure_summary_groups_misses_without_exposing_hits():
    sessions = [
        {
            "sample_id": "a",
            "scenario_type": "intent_override",
            "route": "buying",
            "final_turn": 10,
            "override_state": "applied",
            "fallback": "category",
            "hit": False,
            "first_hit_turn": None,
            "best_rank": None,
        },
        {
            "sample_id": "b",
            "scenario_type": "buying",
            "route": "buying",
            "final_turn": 10,
            "override_state": "none",
            "fallback": "none",
            "hit": False,
            "first_hit_turn": None,
            "best_rank": None,
        },
        {
            "sample_id": "c",
            "scenario_type": "buying",
            "hit": True,
            "first_hit_turn": 1,
            "best_rank": 1,
        },
    ]

    summary = summarize_failures(sessions)

    assert summary["total_misses"] == 2
    assert summary["by_scenario"] == {"buying": 1, "intent_override": 1}
    assert summary["by_route"] == {"buying": 2}
    assert summary["by_override_state"] == {"applied": 1, "none": 1}
    assert summary["by_fallback"] == {"category": 1, "none": 1}
    assert set(summary["sample_ids"]) == {"a", "b"}


def test_compare_results_reports_recoveries_regressions_and_metric_deltas():
    baseline = {
        "recommended_technical_score": 0.5,
        "hit_rate_at_10": 0.5,
        "mrr": 0.2,
        "mttc": 6.0,
        "sessions": [
            {
                "sample_id": "a",
                "scenario_type": "buying",
                "hit": False,
                "first_hit_turn": None,
                "best_rank": None,
            },
            {
                "sample_id": "b",
                "scenario_type": "browsing",
                "hit": True,
                "first_hit_turn": 1,
                "best_rank": 1,
            },
            {
                "sample_id": "c",
                "scenario_type": "buying",
                "hit": True,
                "first_hit_turn": 3,
                "best_rank": 5,
            },
        ],
    }
    candidate = {
        "recommended_technical_score": 0.6,
        "hit_rate_at_10": 2 / 3,
        "mrr": 0.3,
        "mttc": 5.0,
        "sessions": [
            {
                "sample_id": "a",
                "scenario_type": "buying",
                "hit": True,
                "first_hit_turn": 2,
                "best_rank": 3,
            },
            {
                "sample_id": "b",
                "scenario_type": "browsing",
                "hit": False,
                "first_hit_turn": None,
                "best_rank": None,
            },
            {
                "sample_id": "c",
                "scenario_type": "buying",
                "hit": True,
                "first_hit_turn": 2,
                "best_rank": 2,
            },
        ],
    }

    report = compare_results(baseline, candidate)

    assert report["metric_delta"]["recommended_technical_score"] == 0.1
    assert report["recovered"] == ["a"]
    assert report["regressed"] == ["b"]
    assert report["rank_or_turn_improved"] == ["c"]
    assert report["by_scenario"]["buying"]["recovered"] == 1
    assert report["by_scenario"]["browsing"]["regressed"] == 1
