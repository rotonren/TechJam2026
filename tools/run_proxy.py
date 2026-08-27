"""Run the sealed proxy dialogue suites without exposing target-level audit data."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
import secrets
import stat
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
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
from tools.run_cv import _latency_summary, selection_score
from tools.runtime_fingerprint import config_hash, runtime_hash

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

AUDIT_METADATA_KEYS = frozenset({
    "created_at", "commit", "config_hash", "manifest_hash", "dataset_hash", "suite",
    "fallback_count", "invalid_response_count", "audit_label",
})
AUDIT_AGGREGATE_KEYS = frozenset({
    "sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency",
    "recommended_technical_score", "reported_token_usage", "scenario_metrics",
    "invalid_response_count",
})
SENSITIVE_AUDIT_KEYS = frozenset({
    "sessions", "sample_id", "session_id", "target", "targets", "parent_asin",
    "ground_truth", "misses", "miss", "recommendations", "intent_card", "behavior",
})
AUDIT_SCENARIOS = frozenset({"boundary", "browsing", "buying", "intent_override"})
METRIC_SUMMARY_KEYS = frozenset({"sample_count", "hit_rate_at_10", "mrr", "mttc"})
TOKEN_USAGE_KEYS = frozenset({"prompt_tokens", "completion_tokens", "total_tokens"})


@dataclass(frozen=True)
class _VerifiedProxySuite:
    manifest: dict[str, object]
    rows: list[dict]
    manifest_hash: str
    dataset_hash: str


@dataclass(frozen=True)
class _CallEvidence:
    session_id: str
    turn: int
    trace_required: bool


@dataclass(frozen=True)
class _AuditLockOwnership:
    path: Path
    token: str
    identity: tuple[int, int, int, int, int]


def opaque_session_id(sample_id: str) -> str:
    """Build the deterministic opaque ID used for proxy agent sessions."""
    return "proxy_eval_" + hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:20]


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
    *,
    call_evidence: list[_CallEvidence] | None = None,
) -> dict[str, object]:
    """Evaluate an agent with the frozen evaluator's control flow and metrics."""
    sessions: list[dict[str, object]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    invalid_response_count = 0
    for sample in samples:
        sample_id = str(sample["sample_id"])
        session_id = opaque_session_id(sample_id)
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
                if call_evidence is not None:
                    call_evidence.append(_CallEvidence(session_id, turn, False))
                response = {"message": "", "ask_attribute": None, "recommendations": []}
                invalid_response_count += 1
            else:
                if call_evidence is not None:
                    call_evidence.append(_CallEvidence(session_id, turn, True))
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


def _load_verified_proxy_suite(proxy_root: str | Path, suite: str) -> _VerifiedProxySuite:
    """Read, hash, and parse the sealed manifest and suite from the same bytes."""
    if suite not in {"representative", "stress"}:
        raise ValueError("suite must be representative or stress")
    root = Path(proxy_root)
    manifest_path = root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid proxy manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise TypeError("proxy manifest must be an object")
    filename = f"{suite}.jsonl"
    output_hashes = manifest.get("output_hashes")
    counts = manifest.get("counts")
    expected_hash = output_hashes.get(filename) if isinstance(output_hashes, dict) else None
    expected_count = counts.get(suite) if isinstance(counts, dict) else None
    if not isinstance(expected_hash, str):
        raise TypeError("proxy manifest does not define suite hash")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 0:
        raise ValueError("proxy manifest count must be a nonnegative integer")
    dataset_path = root / filename
    try:
        dataset_bytes = dataset_path.read_bytes()
        rows = [
            json.loads(line)
            for line in dataset_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid proxy suite: {error}") from error
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    actual_hash = hashlib.sha256(dataset_bytes).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("proxy suite hash does not match manifest")
    if len(rows) != expected_count:
        raise ValueError("proxy suite count does not match manifest")
    identifiers: set[str] = set()
    session_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("proxy suite rows must be objects")  # noqa: TRY004
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("proxy suite rows require a nonempty sample_id")
        if sample_id in identifiers:
            raise ValueError("proxy suite has duplicate sample_id")
        identifiers.add(sample_id)
        session_id = opaque_session_id(sample_id)
        if session_id in session_ids:
            raise ValueError("proxy suite opaque session ID collision")
        session_ids.add(session_id)
        if row.get("scenario_type") not in AUDIT_SCENARIOS:
            raise ValueError("proxy suite scenario_type is invalid")
        if not isinstance(row.get("user_profile"), dict):
            raise ValueError("proxy suite user_profile is invalid")  # noqa: TRY004
        ground_truth = row.get("ground_truth")
        if not isinstance(ground_truth, dict):
            raise ValueError("proxy suite ground_truth is invalid")  # noqa: TRY004
        parent_asin = ground_truth.get("parent_asin")
        if not isinstance(parent_asin, str) or not parent_asin.strip():
            raise ValueError("proxy suite ground_truth parent_asin is invalid")
        if not isinstance(row.get("intent_card"), dict):
            raise ValueError("proxy suite intent_card is invalid")  # noqa: TRY004
        if not isinstance(row.get("behavior"), dict):
            raise ValueError("proxy suite behavior is invalid")  # noqa: TRY004
        variant = row.get("dialogue_variant")
        if isinstance(variant, bool) or not isinstance(variant, int):
            raise ValueError("proxy suite dialogue_variant is invalid")  # noqa: TRY004
        if row.get("proxy_suite") != suite:
            raise ValueError("proxy suite proxy_suite is invalid")
        for key in ("category_bucket", "difficulty_bucket"):
            value = row.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"proxy suite {key} is invalid")
        if suite == "representative":
            fold = row.get("proxy_fold")
            if isinstance(fold, bool) or not isinstance(fold, int) or fold not in {1, 2, 3, 4, 5}:
                raise ValueError("proxy suite proxy_fold is invalid")
        elif "proxy_fold" in row:
            raise ValueError("stress proxy suite must not include proxy_fold")
    return _VerifiedProxySuite(manifest, rows, manifest_hash, actual_hash)


def load_proxy_suite(proxy_root: str | Path, suite: str) -> tuple[dict, list[dict]]:
    verified = _load_verified_proxy_suite(proxy_root, suite)
    return verified.manifest, verified.rows


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


def _contains_audit_key(value: object, keys: frozenset[str]) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower() in keys
            or _contains_audit_key(item, keys)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_audit_key(item, keys) for item in value)
    return False


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_finite_number(value: object, *, minimum: float | None = None, maximum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)


def _validate_metric_summary(value: object) -> int:
    if not isinstance(value, dict) or set(value) != METRIC_SUMMARY_KEYS:
        raise ValueError("audit scenario metric schema is invalid")
    count = value["sample_count"]
    mttc = value["mttc"]
    if not _is_nonnegative_int(count):
        raise ValueError("audit scenario sample_count is invalid")
    if not _is_finite_number(value["hit_rate_at_10"], minimum=0.0, maximum=1.0):
        raise ValueError("audit scenario hit rate is invalid")
    if not _is_finite_number(value["mrr"], minimum=0.0, maximum=1.0):
        raise ValueError("audit scenario mrr is invalid")
    if mttc is None:
        if count != 0:
            raise ValueError("audit scenario mttc is invalid")
    elif not _is_finite_number(mttc, minimum=1.0, maximum=float(MAX_TURNS + 1)):
        raise ValueError("audit scenario mttc is invalid")
    if count == 0 and (value["hit_rate_at_10"] != 0.0 or value["mrr"] != 0.0 or mttc is not None):
        raise ValueError("audit empty scenario metrics are invalid")
    return count


def _validate_audit_metadata(metadata: dict[str, object]) -> None:
    if set(metadata) != AUDIT_METADATA_KEYS:
        raise ValueError("unknown or missing audit metadata field")
    for key in ("created_at", "commit", "config_hash", "manifest_hash", "dataset_hash"):
        if not isinstance(metadata[key], str) or not metadata[key]:
            raise ValueError("audit metadata scalar is invalid")
    if metadata["suite"] != "representative":
        raise ValueError("audit metadata suite is invalid")
    if metadata["audit_label"] not in {"baseline", "final"}:
        raise ValueError("audit metadata label is invalid")
    if (
        not _is_nonnegative_int(metadata["fallback_count"])
        or not _is_nonnegative_int(metadata["invalid_response_count"])
        or metadata["fallback_count"] != 0
        or metadata["invalid_response_count"] != 0
    ):
        raise ValueError("audit metadata count is invalid")


def _validate_audit_aggregate(aggregate: dict[str, object], metadata: dict[str, object]) -> None:
    if set(aggregate) != AUDIT_AGGREGATE_KEYS:
        raise ValueError("unknown or missing audit aggregate field")
    count = aggregate["sample_count"]
    if not _is_nonnegative_int(count):
        raise ValueError("audit sample_count is invalid")
    for key in ("hit_rate_at_10", "mrr", "efficiency", "recommended_technical_score"):
        if not _is_finite_number(aggregate[key], minimum=0.0, maximum=1.0):
            raise ValueError("audit aggregate scalar is invalid")
    mttc = aggregate["mttc"]
    if mttc is None:
        if count != 0:
            raise ValueError("audit mttc is invalid")
    elif not _is_finite_number(mttc, minimum=1.0, maximum=float(MAX_TURNS + 1)):
        raise ValueError("audit mttc is invalid")
    usage = aggregate["reported_token_usage"]
    if not isinstance(usage, dict) or set(usage) != TOKEN_USAGE_KEYS or not all(
        _is_nonnegative_int(usage[key]) for key in TOKEN_USAGE_KEYS
    ) or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise ValueError("audit token usage is invalid")
    scenarios = aggregate["scenario_metrics"]
    if not isinstance(scenarios, dict) or set(scenarios) != AUDIT_SCENARIOS:
        raise ValueError("audit scenario_metrics schema is invalid")
    scenario_counts = {name: _validate_metric_summary(scenarios[name]) for name in AUDIT_SCENARIOS}
    if sum(scenario_counts.values()) != count:
        raise ValueError("audit scenario sample counts are inconsistent")
    if count == 0:
        if aggregate["hit_rate_at_10"] != 0.0 or aggregate["mrr"] != 0.0 or mttc is not None:
            raise ValueError("audit empty aggregate metrics are invalid")
    else:
        weighted_metrics = {
            key: sum(
                float(scenarios[name][key]) * scenario_counts[name]
                for name in AUDIT_SCENARIOS
                if scenario_counts[name]
            ) / count
            for key in ("hit_rate_at_10", "mrr", "mttc")
        }
        if any(
            not math.isclose(float(aggregate[key]), weighted_metrics[key], rel_tol=0.0, abs_tol=1.000001e-6)
            for key in weighted_metrics
        ):
            raise ValueError("audit aggregate metrics disagree with scenario metrics")
    if (
        not _is_nonnegative_int(aggregate["invalid_response_count"])
        or aggregate["invalid_response_count"] != 0
        or aggregate["invalid_response_count"] != metadata["invalid_response_count"]
    ):
        raise ValueError("audit invalid response count is invalid")
    if count == 0:
        expected_efficiency = expected_score = 0.0
    else:
        raw_efficiency = max(0.0, min(1.0, (11.0 - float(mttc)) / 10.0))
        expected_efficiency = round(raw_efficiency, 6)
        expected_score = round(
            0.50 * float(aggregate["hit_rate_at_10"])
            + 0.30 * float(aggregate["mrr"])
            + 0.20 * raw_efficiency,
            6,
        )
    if (
        aggregate["efficiency"] != expected_efficiency
        or aggregate["recommended_technical_score"] != expected_score
    ):
        raise ValueError("audit aggregate metrics are inconsistent")


def _audit_payload(result: dict[str, object], metadata: dict[str, object]) -> dict[str, object]:
    aggregate = {key: result[key] for key in result if key != "sessions"}
    if _contains_audit_key(metadata, frozenset({"sessions"})) or _contains_audit_key(
        aggregate, frozenset({"sessions"})
    ):
        raise ValueError("sessions are forbidden in audit reports")
    unknown_result = set(result) - AUDIT_AGGREGATE_KEYS - {"sessions"}
    if unknown_result:
        raise ValueError("unknown audit aggregate field")
    aggregate = {key: result[key] for key in AUDIT_AGGREGATE_KEYS if key in result}
    if _contains_audit_key(metadata, SENSITIVE_AUDIT_KEYS) or _contains_audit_key(
        aggregate, SENSITIVE_AUDIT_KEYS
    ):
        raise ValueError("sensitive audit evidence is forbidden")
    _validate_audit_metadata(metadata)
    _validate_audit_aggregate(aggregate, metadata)
    return {**metadata, "aggregate": aggregate}


def _serialize_json(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _reject_existing_destination(destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(f"proxy output already exists: {destination}")


def _publish_exclusive(
    destination: Path, payload: dict[str, object], *, prepublish: Callable[[], None] | None = None
) -> None:
    """Atomically create a report, failing closed on collisions and link failures."""
    encoded = _serialize_json(payload)
    _reject_existing_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if prepublish is not None:
            prepublish()
        _reject_existing_destination(destination)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise
        except OSError as error:
            raise RuntimeError("exclusive report publication requires hardlink support") from error
    finally:
        temporary.unlink(missing_ok=True)


def write_audit_report(
    destination: Path,
    result: dict[str, object],
    metadata: dict[str, object],
    *,
    prepublish: Callable[[], None] | None = None,
) -> None:
    """Write a one-shot audit report that can never include session-level evidence."""
    _publish_exclusive(destination, _audit_payload(result, metadata), prepublish=prepublish)


def _load_agent(specification: str) -> type:
    if specification != "agent:Agent":
        raise ValueError("proxy evidence requires the default agent:Agent")
    repository = Path(__file__).resolve().parents[1]
    entrypoint = repository / "agent.py"
    spec = importlib.util.spec_from_file_location("_compasscart_sealed_agent", entrypoint)
    if spec is None or spec.loader is None:
        raise ValueError("default agent entrypoint is invalid")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_class = getattr(module, "Agent", None)
    source = inspect.getsourcefile(agent_class) if isinstance(agent_class, type) else None
    expected_source = repository / "src" / "compasscart" / "agent.py"
    if (
        not isinstance(agent_class, type)
        or agent_class.__module__ != "compasscart.agent"
        or source is None
        or Path(source).resolve() != expected_source
    ):
        raise ValueError("default agent source identity is invalid")
    return agent_class


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _config_hash(agent: object) -> str:
    return config_hash(getattr(agent, "config", None))


def _trace_details(
    agent: object, call_evidence: list[_CallEvidence]
) -> tuple[dict[str, object], int, dict[str, int]]:
    sink = getattr(agent, "traces", None)
    records = getattr(sink, "records", [])
    if not isinstance(records, list):
        raise ValueError("trace records must be a list")  # noqa: TRY004
    record_index = 0
    matched: list[dict[str, object]] = []
    for evidence in call_evidence:
        if not isinstance(evidence, _CallEvidence):
            raise ValueError("trace call evidence is invalid")  # noqa: TRY004
        record = records[record_index] if record_index < len(records) else None
        matches_call = isinstance(record, dict) and (
            record.get("session_id") == evidence.session_id and record.get("turn") == evidence.turn
        )
        if not matches_call:
            if evidence.trace_required:
                raise ValueError("trace records are stale, malformed, or out of order")
            continue
        elapsed_ms = record.get("elapsed_ms")
        if (
            not isinstance(elapsed_ms, (int, float))
            or isinstance(elapsed_ms, bool)
            or not math.isfinite(elapsed_ms)
            or elapsed_ms < 0
        ):
            raise ValueError("trace records are stale, malformed, or out of order")
        if not isinstance(record.get("fallbacks"), (list, tuple, set)):
            raise ValueError("trace records are stale, malformed, or out of order")  # noqa: TRY004
        matched.append(record)
        record_index += 1
    if record_index != len(records):
        raise ValueError("trace record count does not match evaluation")
    latency_values = [float(record["elapsed_ms"]) for record in matched]
    fallback_count = sum(bool(record["fallbacks"]) for record in matched)
    routes = Counter(str(record.get("route", "unknown")) for record in matched)
    return _latency_summary(latency_values), fallback_count, dict(sorted(routes.items()))


def _write_normal_report(destination: Path, report: dict[str, object]) -> None:
    _publish_exclusive(destination, report)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _audit_paths(proxy_root: Path, audit_label: str) -> tuple[Path, Path]:
    audit_dir = (proxy_root.resolve() / "audit").resolve()
    return audit_dir / f"{audit_label}.json", audit_dir / f"{audit_label}.lock"


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(getattr(info, "st_birthtime_ns", 0)),
        int(getattr(info, "st_file_attributes", 0)),
        int(getattr(info, "st_reparse_tag", 0)),
    )


def _matches_lock_identity(path: Path, identity: tuple[int, int, int, int, int]) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and _file_identity(info) == identity


def _verify_lock_owner(ownership: _AuditLockOwnership) -> bool:
    if not _matches_lock_identity(ownership.path, ownership.identity):
        return False
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(ownership.path, flags)
    except OSError:
        return False
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or _file_identity(info) != ownership.identity:
            return False
        expected = (ownership.token + os.linesep).encode("ascii")
        content = os.read(descriptor, len(expected) + 1)
        return content == expected
    except OSError:
        return False
    finally:
        os.close(descriptor)


def _release_audit_lock(ownership: _AuditLockOwnership) -> None:
    if not _verify_lock_owner(ownership):
        return
    try:
        ownership.path.unlink()
    except FileNotFoundError:
        return


def _cleanup_failed_reservation(
    path: Path, identity: tuple[int, int, int, int, int] | None
) -> None:
    if identity is None or not _matches_lock_identity(path, identity):
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _preflight_audit_publish(ownership: _AuditLockOwnership, output: Path) -> None:
    if not _verify_lock_owner(ownership):
        raise ValueError("audit lock ownership verification failed")
    _reject_existing_destination(output)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".probe-source", dir=output.parent
    )
    temporary = Path(temporary_name)
    probe = output.parent / f".{output.name}.{secrets.token_hex(16)}.probe"
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"audit hardlink preflight\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, probe)
        except OSError as error:
            raise RuntimeError("exclusive audit publication requires hardlink support") from error
        if not probe.is_file() or _file_identity(temporary.stat()) != _file_identity(probe.stat()):
            raise RuntimeError("exclusive audit publication hardlink preflight failed")
    finally:
        probe.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def _reserve_audit_output(proxy_root: Path, audit_label: str, output: Path) -> _AuditLockOwnership:
    expected, lock = _audit_paths(proxy_root, audit_label)
    if output.resolve() != expected:
        raise ValueError("audit output must use the canonical sealed audit path")
    _reject_existing_destination(expected)
    lock.parent.mkdir(parents=True, exist_ok=True)
    handle: object | None = None
    identity: tuple[int, int, int, int, int] | None = None
    token = secrets.token_hex(32)
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise FileExistsError("sealed audit reservation already exists") from error
    try:
        handle.write(token + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        identity = _file_identity(os.fstat(handle.fileno()))
        handle.close()
        handle = None
    except BaseException:
        if handle is not None:
            try:
                identity = identity or _file_identity(lock.lstat())
            except OSError:
                pass
            try:
                handle.close()
            except OSError:
                pass
        _cleanup_failed_reservation(lock, identity)
        raise
    if identity is None:
        raise RuntimeError("audit lock identity was not recorded")
    return _AuditLockOwnership(lock, token, identity)


def run_proxy(args: argparse.Namespace) -> None:
    proxy_root = Path(args.proxy_root)
    output = Path(args.output)
    audit_lock: _AuditLockOwnership | None = None
    execution_runtime_hash: str | None = None
    try:
        if args.agent != "agent:Agent":
            raise ValueError("proxy evidence requires the default agent:Agent")
        agent_class = _load_agent(args.agent)
        if args.audit_label is not None:
            audit_lock = _reserve_audit_output(proxy_root, args.audit_label, output)
            _preflight_audit_publish(audit_lock, output)
        else:
            if _is_within(output.resolve(), (proxy_root.resolve() / "audit").resolve()):
                raise ValueError("non-audit reports cannot target the sealed audit directory")
            _reject_existing_destination(output)
            execution_runtime_hash = runtime_hash()
        verified = _load_verified_proxy_suite(proxy_root, args.suite)
        rows = verified.rows
        selections = select_proxy_rows(rows, args.suite, args.folds, args.audit_label)
        catalog_ids, categories, products = catalog_index(args.catalog)
        manifest_hash = verified.manifest_hash
        dataset_hash = verified.dataset_hash
        fold_reports: list[dict[str, object]] = []
        config_hash = ""
        total_fallback_count = 0
        total_invalid_count = 0
        audit_result: dict[str, object] | None = None
        for fold, selected_rows in selections:
            agent: object | None = None
            try:
                agent = agent_class(args.catalog)
                fold_config_hash = _config_hash(agent)
                if config_hash and fold_config_hash != config_hash:
                    raise ValueError("agent config changed between folds")
                config_hash = config_hash or fold_config_hash
                call_evidence: list[_CallEvidence] = []
                result = evaluate_proxy(
                    agent, selected_rows, catalog_ids, categories, products,
                    call_evidence=call_evidence,
                )
                sessions = result["sessions"]
                if not isinstance(sessions, list):
                    raise ValueError("proxy evaluator did not return sessions")  # noqa: TRY004
                latency, fallback_count, routes = _trace_details(agent, call_evidence)
                if _config_hash(agent) != fold_config_hash:
                    raise ValueError("agent config changed during fold")
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
                    fold_report["sessions"] = sessions
                else:
                    audit_result = result
                fold_reports.append(fold_report)
            finally:
                if agent is not None:
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
            audit_metadata = {**metadata, "audit_label": args.audit_label}
            write_audit_report(
                output,
                audit_result,
                audit_metadata,
                prepublish=lambda: _preflight_audit_publish(audit_lock, output),
            )
            print(json.dumps(_audit_payload(audit_result, audit_metadata), sort_keys=True))
            return

        scores = [float(item["aggregate"]["recommended_technical_score"]) for item in fold_reports]
        selected_sample_count = sum(len(selected_rows) for _, selected_rows in selections)
        invalid_rate = total_invalid_count / max(selected_sample_count, 1)
        report = {
            **metadata,
            "runtime_hash": execution_runtime_hash,
            "folds": fold_reports,
            "mean_technical_score": round(statistics.fmean(scores), 6) if scores else 0.0,
            "std_technical_score": round(statistics.pstdev(scores), 6) if scores else 0.0,
            "selection_score": selection_score(scores, 0.0, invalid_rate),
            "api_cost_usd": 0.0,
        }
        if runtime_hash() != execution_runtime_hash:
            raise ValueError("runtime changed during normal proxy evidence execution")
        _write_normal_report(output, report)
        print(json.dumps({key: value for key, value in report.items() if key != "folds"}, sort_keys=True))
        if total_fallback_count or total_invalid_count:
            raise SystemExit(1)
    finally:
        if audit_lock is not None:
            _release_audit_lock(audit_lock)


def parse_proxy_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sealed CompassCart proxy suites")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--suite", choices=("representative", "stress"), required=True)
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--audit-label", choices=("baseline", "final"))
    parser.add_argument("--agent", default="agent:Agent")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    run_proxy(parse_proxy_args())


if __name__ == "__main__":
    main()
