from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

_METRICS = (
    "recommended_technical_score",
    "hit_rate_at_10",
    "mrr",
    "mttc",
)


def compare_results(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    baseline_sessions = {
        str(item["sample_id"]): item for item in baseline.get("sessions", [])
    }
    candidate_sessions = {
        str(item["sample_id"]): item for item in candidate.get("sessions", [])
    }
    if set(baseline_sessions) != set(candidate_sessions):
        raise ValueError("baseline and candidate sample IDs differ")

    recovered: list[str] = []
    regressed: list[str] = []
    improved: list[str] = []
    worsened: list[str] = []
    scenario_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "recovered": 0,
            "regressed": 0,
            "rank_or_turn_improved": 0,
            "rank_or_turn_worsened": 0,
        }
    )
    for sample_id in sorted(baseline_sessions):
        before = baseline_sessions[sample_id]
        after = candidate_sessions[sample_id]
        scenario = str(after.get("scenario_type", "unknown"))
        before_hit = bool(before.get("hit"))
        after_hit = bool(after.get("hit"))
        if not before_hit and after_hit:
            recovered.append(sample_id)
            scenario_counts[scenario]["recovered"] += 1
        elif before_hit and not after_hit:
            regressed.append(sample_id)
            scenario_counts[scenario]["regressed"] += 1
        elif before_hit and after_hit:
            before_key = (
                int(before.get("first_hit_turn") or 11),
                int(before.get("best_rank") or 11),
            )
            after_key = (
                int(after.get("first_hit_turn") or 11),
                int(after.get("best_rank") or 11),
            )
            if after_key < before_key:
                improved.append(sample_id)
                scenario_counts[scenario]["rank_or_turn_improved"] += 1
            elif after_key > before_key:
                worsened.append(sample_id)
                scenario_counts[scenario]["rank_or_turn_worsened"] += 1

    metric_delta = {
        metric: round(float(candidate[metric]) - float(baseline[metric]), 6)
        for metric in _METRICS
    }
    return {
        "metric_delta": metric_delta,
        "recovered": recovered,
        "regressed": regressed,
        "rank_or_turn_improved": improved,
        "rank_or_turn_worsened": worsened,
        "by_scenario": dict(sorted(scenario_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = compare_results(baseline, candidate)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
