from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize_failures(sessions: list[dict]) -> dict[str, object]:
    misses = [session for session in sessions if not session.get("hit")]

    def grouped(field: str, default: str = "unknown") -> dict[str, int]:
        return dict(
            sorted(Counter(str(item.get(field, default)) for item in misses).items())
        )

    return {
        "total_misses": len(misses),
        "by_scenario": grouped("scenario_type"),
        "by_route": grouped("route"),
        "by_final_turn": grouped("final_turn"),
        "by_override_state": grouped("override_state"),
        "by_fallback": grouped("fallback"),
        "sample_ids": [str(item.get("sample_id")) for item in misses[:50]],
    }


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# CompassCart Failure Analysis",
        "",
        f"Total misses: {summary['total_misses']}",
        "",
    ]
    lines.append("## By Scenario")
    lines.append("")
    for scenario, count in summary["by_scenario"].items():
        lines.append(f"- {scenario}: {count}")
    for field, title in (
        ("by_route", "By Route"),
        ("by_override_state", "By Override State"),
        ("by_fallback", "By Fallback"),
    ):
        lines.extend(["", f"## {title}", ""])
        for value, count in summary[field].items():
            lines.append(f"- {value}: {count}")
    lines.extend(["", "## Representative Samples", ""])
    lines.extend(f"- {identifier}" for identifier in summary["sample_ids"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    summary = summarize_failures(payload.get("sessions", []))
    markdown = render_markdown(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
