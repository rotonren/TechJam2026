"""Run the sealed proxy dialogue suites without exposing target-level audit data."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import (
    ALLOWED_ATTRIBUTES,
    MAX_TURNS,
    TOP_K,
    catalog_index,
    classify_constraint,
    coarse_category,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)
from tools.proxy_dataset import sha256_file
from tools.run_cv import _latency_summary, selection_score

INITIAL_BUYING = (
    "I'm looking for {category}. A key requirement is: {constraint}.",
    "Please find {category}; {constraint} is essential.",
    "I need {category}, and it must have: {constraint}.",
    "Help me choose {category}. Prioritize {constraint}.",
)
INITIAL_BROWSING = (
    "I'm looking for {category}, but I'm still exploring.",
    "I'm browsing {category} and have not decided on the details.",
    "Show me some {category}; I'm open to options.",
    "I want to explore {category} before narrowing it down.",
)
INITIAL_OVERRIDE = (
    "I'm looking for {category}. I had been considering {old_value}.",
    "Please find {category}; I was initially leaning toward {old_value}.",
    "I need {category}. My earlier preference was {old_value}.",
    "Help me explore {category}; I started out wanting {old_value}.",
)
NO_PREFERENCE = (
    "I don't have a preference for {attribute}; please use your judgment.",
    "No preference on {attribute}; choose what fits best.",
    "I'm flexible about {attribute}.",
    "Any {attribute} is fine with me.",
)
CONTROL_REPLY = (
    "Those options are not quite right yet. Ask me about one specific attribute.",
    "Please narrow this down by asking one concrete question.",
    "I need another direction; ask about a specific preference.",
    "Keep searching and ask me for one useful detail.",
)
DISCLOSURE_REPLY = (
    "For that, what matters is: {values}.",
    "Please prioritize {values}.",
    "My preference there is {values}.",
    "Use this requirement: {values}.",
)
NO_ADDITIONAL = (
    "I don't have an additional preference for {attribute}.",
    "Nothing more to add about {attribute}.",
    "I'm flexible on {attribute} beyond that.",
    "No other requirement for {attribute} right now.",
)


class ProxyDialogue:
    """A wording-only dialogue adapter preserving evaluator state transitions."""

    def __init__(self, sample: dict[str, Any], dialogue_variant: object = None) -> None:
        self.sample = sample
        self.variant = int(dialogue_variant or 0) % 4

    def initial_message(self, category: str, disclosed: set[str]) -> str:
        scenario = self.sample["scenario_type"]
        if scenario == "buying" and self.sample["intent_card"].get("hard_constraints"):
            constraint = str(self.sample["intent_card"]["hard_constraints"][0])
            disclosed.add(constraint)
            return INITIAL_BUYING[self.variant].format(category=category, constraint=constraint)
        if scenario == "intent_override":
            old_value = str(self.sample["behavior"]["override"]["old_value"])
            return INITIAL_OVERRIDE[self.variant].format(category=category, old_value=old_value)
        return INITIAL_BROWSING[self.variant].format(category=category)

    def customer_reply(
        self, ask_attribute: object, disclosed: set[str], boundary_used: bool
    ) -> tuple[str, bool]:
        attribute = ask_attribute if isinstance(ask_attribute, str) else None
        if self.sample["scenario_type"] == "boundary" and not boundary_used and attribute:
            return NO_PREFERENCE[self.variant].format(attribute=attribute), True
        if not attribute:
            return CONTROL_REPLY[self.variant], boundary_used
        if attribute not in ALLOWED_ATTRIBUTES:
            attribute = "other"
        constraints = [
            *[str(value) for value in self.sample["intent_card"].get("hard_constraints", [])],
            *[str(value) for value in self.sample["intent_card"].get("soft_preferences", [])],
        ]
        matches = [
            value
            for value in constraints
            if value not in disclosed
            and (attribute == "other" or classify_constraint(value) == attribute)
        ][:2]
        if not matches:
            return NO_ADDITIONAL[self.variant].format(attribute=attribute), boundary_used
        disclosed.update(matches)
        return DISCLOSURE_REPLY[self.variant].format(values="; ".join(matches)), boundary_used


def evaluate_proxy(
    agent: object,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> dict[str, object]:
    """Evaluate an agent with the frozen evaluator's control flow and metrics."""
    sessions: list[dict[str, object]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    invalid_response_count = 0
    for sample in samples:
        sample_id = str(sample["sample_id"])
        digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:20]
        session_id = f"proxy_eval_{digest}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
        effective_sample = {
            **sample,
            "intent_card": effective_intent_card,
            "behavior": effective_behavior,
        }
        dialogue = ProxyDialogue(effective_sample, sample.get("dialogue_variant"))
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = dialogue.initial_message(
            coarse_category(categories.get(target, [])), disclosed
        )
        hit_turn: int | None = None
        best_rank: int | None = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:  # noqa: BLE001 - evaluator always turns errors into legal blanks.
                response = {"message": "", "ask_attribute": None, "recommendations": []}
                invalid_response_count += 1
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
                invalid_response_count += 1
            usage = response.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                    total_prompt_tokens += prompt_tokens
                if isinstance(completion_tokens, int) and completion_tokens >= 0:
                    total_completion_tokens += completion_tokens
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(
                    override.get("message", "Actually, please ignore my earlier preference.")
                )
            else:
                user_message, boundary_used = dialogue.customer_reply(
                    response.get("ask_attribute"), disclosed, boundary_used
                )
        sessions.append(
            {
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "hit": hit_turn is not None,
                "first_hit_turn": hit_turn,
                "best_rank": best_rank,
                "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            }
        )

    overall = metric_summary(sessions)
    hit_rate = overall["hit_rate_at_10"]
    mrr = overall["mrr"]
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical_score = (0.50*hit_rate + 0.30*mrr + 0.20*efficiency)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "reported_token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        "scenario_metrics": {
            name: metric_summary(grouped[name]) for name in sorted(grouped)
        },
        "sessions": sessions,
        "invalid_response_count": invalid_response_count,
    }


def load_proxy_suite(proxy_root: str | Path, suite: str) -> tuple[dict, list[dict]]:
    if suite not in {"representative", "stress"}:
        raise ValueError("suite must be representative or stress")
    root = Path(proxy_root)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid proxy manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise TypeError("proxy manifest must be an object")
    filename = f"{suite}.jsonl"
    output_hashes = manifest.get("output_hashes")
    counts = manifest.get("counts")
    expected_hash = output_hashes.get(filename) if isinstance(output_hashes, dict) else None
    expected_count = counts.get(suite) if isinstance(counts, dict) else None
    if not isinstance(expected_hash, str) or not isinstance(expected_count, int):
        raise TypeError("proxy manifest does not define suite hash and count")
    dataset_path = root / filename
    try:
        actual_hash = sha256_file(dataset_path)
        rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid proxy suite: {error}") from error
    if actual_hash != expected_hash:
        raise ValueError("proxy suite hash does not match manifest")
    if len(rows) != expected_count:
        raise ValueError("proxy suite count does not match manifest")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("proxy suite rows must be objects")
    return manifest, rows


def select_proxy_rows(
    rows: list[dict], suite: str, folds: list[int] | None, audit_label: str | None
) -> list[tuple[int | None, list[dict]]]:
    if suite == "stress":
        if audit_label is not None:
            raise ValueError("stress suite cannot be used for audit")
        if not rows:
            raise ValueError("proxy selection is empty")
        return [(None, rows)]
    if suite != "representative":
        raise ValueError("suite must be representative or stress")
    if audit_label is not None:
        if audit_label not in {"baseline", "final"}:
            raise ValueError("audit label must be baseline or final")
        if folds is not None and folds != [5]:
            raise ValueError("representative audit is restricted to fold 5")
        audit_rows = [row for row in rows if row.get("proxy_fold") == 5]
        if not audit_rows:
            raise ValueError("proxy selection is empty")
        return [(5, audit_rows)]
    selected_folds = sorted(set(folds if folds is not None else [1, 2, 3, 4]))
    if not selected_folds or any(fold not in {1, 2, 3, 4} for fold in selected_folds):
        raise ValueError("representative development folds must be in 1..4")
    selected: list[tuple[int, list[dict]]] = []
    for fold in selected_folds:
        fold_rows = [row for row in rows if row.get("proxy_fold") == fold]
        if not fold_rows:
            raise ValueError(f"proxy selection is empty for fold {fold}")
        selected.append((fold, fold_rows))
    return selected


def _contains_sessions(value: object) -> bool:
    if isinstance(value, dict):
        return "sessions" in value or any(_contains_sessions(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_sessions(item) for item in value)
    return False


def write_audit_report(destination: Path, result: dict[str, object], metadata: dict[str, object]) -> None:
    """Write a one-shot audit report that can never include session-level evidence."""
    aggregate = {key: value for key, value in result.items() if key != "sessions"}
    if _contains_sessions(metadata) or _contains_sessions(aggregate):
        raise ValueError("audit report must not contain sessions")
    payload = {**metadata, "aggregate": aggregate}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")


def _load_agent(specification: str) -> type:
    module_name, class_name = specification.split(":", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _config_hash(agent: object) -> str:
    return hashlib.sha256(repr(getattr(agent, "config", None)).encode("utf-8")).hexdigest()


def _trace_details(agent: object) -> tuple[dict[str, object], int, dict[str, int]]:
    sink = getattr(agent, "traces", None)
    records = getattr(sink, "records", [])
    records = records if isinstance(records, list) else []
    latency_values = [
        float(record["elapsed_ms"])
        for record in records
        if isinstance(record, dict) and isinstance(record.get("elapsed_ms"), (int, float))
    ]
    fallback_count = sum(
        bool(record.get("fallbacks")) for record in records if isinstance(record, dict)
    )
    routes = Counter(
        str(record.get("route", "unknown")) for record in records if isinstance(record, dict)
    )
    return _latency_summary(latency_values), fallback_count, dict(sorted(routes.items()))


def _write_normal_report(destination: Path, report: dict[str, object]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def run_proxy(args: argparse.Namespace) -> None:
    _manifest, rows = load_proxy_suite(args.proxy_root, args.suite)
    selections = select_proxy_rows(rows, args.suite, args.folds, args.audit_label)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent_class = _load_agent(args.agent)
    manifest_hash = sha256_file(args.proxy_root / "manifest.json")
    dataset_hash = sha256_file(args.proxy_root / f"{args.suite}.jsonl")
    fold_reports: list[dict[str, object]] = []
    config_hash = ""
    total_fallback_count = 0
    total_invalid_count = 0
    audit_result: dict[str, object] | None = None
    for fold, selected_rows in selections:
        agent = agent_class(args.catalog)
        config_hash = config_hash or _config_hash(agent)
        result = evaluate_proxy(agent, selected_rows, catalog_ids, categories, products)
        latency, fallback_count, routes = _trace_details(agent)
        invalid_count = int(result["invalid_response_count"])
        total_fallback_count += fallback_count
        total_invalid_count += invalid_count
        aggregate = {key: value for key, value in result.items() if key != "sessions"}
        fold_report: dict[str, object] = {
            "fold": fold,
            "sample_count": len(selected_rows),
            "aggregate": aggregate,
            "latency_ms": latency,
            "fallback_count": fallback_count,
            "route_distribution": routes,
        }
        if args.audit_label is None:
            fold_report["sessions"] = result["sessions"]
        else:
            audit_result = result
        fold_reports.append(fold_report)
        # A real agent owns native dense-runtime buffers.  Each fold must use a
        # fresh instance, so release the completed fold before constructing the next.
        del agent
        gc.collect()

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "commit": _git_commit(),
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "dataset_hash": dataset_hash,
        "suite": args.suite,
        "fallback_count": total_fallback_count,
        "invalid_response_count": total_invalid_count,
    }
    if args.audit_label is not None:
        if audit_result is None:
            raise RuntimeError("audit selection did not produce a result")
        if total_fallback_count or total_invalid_count:
            raise SystemExit(1)
        write_audit_report(args.output, audit_result, {**metadata, "audit_label": args.audit_label})
        print(json.dumps({**metadata, "audit_label": args.audit_label, "aggregate": {key: value for key, value in audit_result.items() if key != "sessions"}}, sort_keys=True))
        return

    scores = [float(item["aggregate"]["recommended_technical_score"]) for item in fold_reports]
    selected_sample_count = sum(len(selected_rows) for _, selected_rows in selections)
    invalid_rate = total_invalid_count / max(selected_sample_count, 1)
    report = {
        **metadata,
        "folds": fold_reports,
        "mean_technical_score": round(statistics.fmean(scores), 6) if scores else 0.0,
        "std_technical_score": round(statistics.pstdev(scores), 6) if scores else 0.0,
        "selection_score": selection_score(scores, 0.0, invalid_rate),
        "api_cost_usd": 0.0,
    }
    _write_normal_report(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "folds"}, sort_keys=True))
    if total_fallback_count or total_invalid_count:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sealed CompassCart proxy suites")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--suite", choices=("representative", "stress"), required=True)
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--audit-label", choices=("baseline", "final"))
    parser.add_argument("--agent", default="agent:Agent")
    parser.add_argument("--output", type=Path, required=True)
    run_proxy(parser.parse_args())


if __name__ == "__main__":
    main()
