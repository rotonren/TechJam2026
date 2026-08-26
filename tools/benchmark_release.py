"""Full-catalog, response-hash release benchmark for CompassCart."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import psutil

from evaluator.local_evaluator import normalize_recommendations
from tools.proxy_dataset import sha256_file, stable_int
from tools.run_cv import _latency_summary
from tools.run_proxy import ProxyDialogue, _load_verified_proxy_suite

_SESSION_COUNT = 200
_TURNS = (1, 2, 3, 4)
_RESPONSE_COUNT = _SESSION_COUNT * len(_TURNS)
_PROFILE_KEYS = frozenset({
    "average_prior_rating", "preference_tags", "purchase_frequency", "rating_style", "summary",
})
_SENSITIVE_KEYS = frozenset({
    "target", "targets", "ground_truth", "intent_card", "behavior", "recommendation",
    "recommendations", "hit", "hits",
})
_ROW_KEYS = frozenset({"session_id", "turn", "profile", "message"})
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_TRIAL_KEYS = frozenset({
    "init_ms", "peak_mib", "rss_mib", "peak_metric_source", "latencies_ms",
    "trace_latencies_ms", "instrumentation_delta_ms", "dense_available", "dense_status",
    "fallback_count", "response_hash", "transcript_hash", "catalog_hash", "catalog_snapshot_hash",
    "capture_provenance", "response_count", "platform",
})
_CAPTURE_METADATA_KEYS = frozenset({
    "catalog_hash", "proxy_manifest_hash", "representative_dataset_hash", "agent_class", "config_hash",
    "dense_available", "dense_status", "capture_seed", "session_count", "response_count", "cwd_mode", "platform",
})
_CAPTURE_MANIFEST_KEYS = _CAPTURE_METADATA_KEYS | frozenset({"schema_version", "transcript_hash", "created_at"})
_CAPTURE_PLATFORM_KEYS = frozenset({"python", "platform"})
_CAPTURE_PROVENANCE_KEYS = frozenset({
    "manifest_hash", "proxy_manifest_hash", "representative_dataset_hash", "agent_class", "config_hash",
    "capture_seed", "session_count", "response_count", "cwd_mode", "dense_available", "dense_status", "platform",
})
_TRIAL_PLATFORM_KEYS = frozenset({"os", "python", "processor", "onnxruntime", "psutil"})
_PARENT_PLATFORM_KEYS = frozenset({"os", "python", "processor", "cpu_logical", "cpu_physical", "ram_mib", "onnxruntime", "psutil"})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _contains_sensitive_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() in _SENSITIVE_KEYS or _contains_sensitive_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def _contains_sensitive_material(value: object) -> bool:
    if _contains_sensitive_key(value):
        return True
    if isinstance(value, dict):
        return any(_contains_sensitive_material(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_sensitive_material(item) for item in value)
    return False


def _safe_profile(profile: object) -> dict[str, object]:
    if not isinstance(profile, dict) or _contains_sensitive_material(profile):
        raise ValueError("profile is invalid or contains sensitive fields")
    result = {str(key): value for key, value in profile.items() if key in _PROFILE_KEYS}
    _validate_profile(result)
    return result


def _validate_profile(profile: object) -> None:
    if not isinstance(profile, dict):
        raise TypeError("profile must be an object")
    if not set(profile).issubset(_PROFILE_KEYS):
        raise ValueError("profile contains fields outside the allowlist")
    if _contains_sensitive_material(profile):
        raise ValueError("profile is invalid or contains sensitive fields")
    rating = profile.get("average_prior_rating")
    if rating is not None and (isinstance(rating, bool) or not isinstance(rating, (int, float)) or not math.isfinite(rating)):
        raise ValueError("average_prior_rating must be finite")
    tags = profile.get("preference_tags")
    if tags is not None and (not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags)):
        raise ValueError("preference_tags must be nonempty strings")
    for key in ("purchase_frequency", "rating_style", "summary"):
        value = profile.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a nonempty string")
    try:
        json.dumps(profile, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("profile is not JSON serializable") from error


def validate_transcript(rows: object) -> None:
    """Reject all transcript shapes other than exactly 200 contiguous four-turn sessions."""
    if not isinstance(rows, list) or len(rows) != _RESPONSE_COUNT:
        raise ValueError("transcript must contain exactly 800 rows")
    seen: set[str] = set()
    closed: set[str] = set()
    current: str | None = None
    expected_turn = 1
    sessions = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != _ROW_KEYS:
            raise ValueError("transcript rows must contain only allowlisted fields")
        session_id = row["session_id"]
        turn = row["turn"]
        message = row["message"]
        if (
            not isinstance(session_id, str)
            or len(session_id) != 10
            or not session_id.startswith("bench_")
            or not session_id[6:].isdigit()
        ):
            raise ValueError("session_id must be an opaque benchmark identifier")
        if isinstance(turn, bool) or not isinstance(turn, int):
            raise TypeError("turn must be an integer")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a nonempty string")
        _validate_profile(row["profile"])
        if current != session_id:
            if current is not None:
                if expected_turn != 5:
                    raise ValueError("session ended before its fourth turn")
                closed.add(current)
            if session_id in seen or session_id in closed:
                raise ValueError("session rows must be contiguous")
            current = session_id
            seen.add(session_id)
            sessions += 1
            expected_turn = 1
        if turn != expected_turn:
            raise ValueError("session turns must be exactly 1 through 4")
        expected_turn += 1
    if current is None or expected_turn != 5 or sessions != _SESSION_COUNT:
        raise ValueError("transcript must contain 200 complete sessions")


def _write_exclusive(destination: Path, payload: bytes) -> None:
    _publish_bundle([(destination, payload)])


def _same_inode(path: Path, expected: os.stat_result) -> bool:
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return (current.st_dev, current.st_ino) == (expected.st_dev, expected.st_ino)


def _unlink_if_same_inode(path: Path, expected: os.stat_result) -> None:
    if _same_inode(path, expected):
        path.unlink(missing_ok=True)


def _publish_bundle(entries: list[tuple[Path, bytes]]) -> None:
    """Publish a small related file set exclusively, rolling back partial links."""
    if len({path for path, _ in entries}) != len(entries):
        raise ValueError("publication destinations must be unique")
    for destination, _ in entries:
        if os.path.lexists(destination):
            raise FileExistsError(f"output already exists: {destination}")
    ordered = sorted(entries, key=lambda item: 0 if item[0].name.endswith(".manifest.json") else 1)
    staged: list[tuple[Path, Path, os.stat_result]] = []
    published: list[tuple[Path, os.stat_result]] = []
    try:
        for destination, payload in ordered:
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
            temporary = Path(temporary_name)
            temporary_stat = os.fstat(descriptor)
            staged.append((destination, temporary, temporary_stat))
            try:
                handle = os.fdopen(descriptor, "wb")
            except Exception:
                os.close(descriptor)
                raise
            with handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for destination, temporary, temporary_stat in staged:
            if os.path.lexists(destination):
                raise FileExistsError(f"output already exists: {destination}")
            try:
                os.link(temporary, destination)
            except OSError as error:
                if isinstance(error, FileExistsError):
                    raise
                raise RuntimeError("exclusive publication requires hardlink support") from error
            published.append((destination, temporary_stat))
    except Exception:
        for destination, temporary_stat in published:
            _unlink_if_same_inode(destination, temporary_stat)
        raise
    finally:
        for _, temporary, temporary_stat in staged:
            _unlink_if_same_inode(temporary, temporary_stat)


def _serialize_jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


@contextmanager
def _sealed_capture_environment() -> object:
    original_cwd = Path.cwd()
    previous_dense = os.environ.pop("COMPASSCART_DISABLE_DENSE", None)
    os.chdir(_repo_root())
    try:
        yield
    finally:
        os.chdir(original_cwd)
        if previous_dense is not None:
            os.environ["COMPASSCART_DISABLE_DENSE"] = previous_dense


def _capture_manifest(
    metadata: dict[str, object], *, transcript_hash: str,
) -> dict[str, object]:
    if set(metadata) != _CAPTURE_METADATA_KEYS:
        raise ValueError("capture metadata schema is invalid")
    for key in ("catalog_hash", "proxy_manifest_hash", "representative_dataset_hash", "config_hash"):
        _validate_hash(metadata[key], key)
    if not isinstance(metadata["agent_class"], str) or not metadata["agent_class"].strip():
        raise ValueError("capture agent class is invalid")
    if not isinstance(metadata["dense_available"], bool) or not isinstance(metadata["dense_status"], str) or not metadata["dense_status"].strip():
        raise ValueError("capture dense metadata is invalid")
    if metadata["capture_seed"] != 20260829 or metadata["session_count"] != _SESSION_COUNT or metadata["response_count"] != _RESPONSE_COUNT or metadata["cwd_mode"] != "root":
        raise ValueError("capture metadata is invalid")
    if not isinstance(metadata["platform"], dict) or set(metadata["platform"]) != _CAPTURE_PLATFORM_KEYS or _contains_sensitive_key(metadata["platform"]):
        raise ValueError("capture platform is invalid")
    if any(not isinstance(metadata["platform"][name], str) or not metadata["platform"][name].strip() for name in _CAPTURE_PLATFORM_KEYS):
        raise ValueError("capture platform is invalid")
    manifest = {
        "schema_version": 1, "transcript_hash": transcript_hash, **metadata,
        "created_at": datetime.now(UTC).isoformat(),
    }
    forbidden = _SENSITIVE_KEYS | {"message", "profile", "session_id", "sample_id"}
    if _contains_sensitive_key(manifest) or any(key in manifest for key in forbidden):
        raise ValueError("capture manifest contains sensitive fields")
    return manifest


def _capture_provenance(manifest: dict[str, object], manifest_hash: str) -> dict[str, object]:
    return {
        "manifest_hash": manifest_hash,
        "proxy_manifest_hash": manifest["proxy_manifest_hash"],
        "representative_dataset_hash": manifest["representative_dataset_hash"],
        "agent_class": manifest["agent_class"], "config_hash": manifest["config_hash"],
        "capture_seed": manifest["capture_seed"], "session_count": manifest["session_count"],
        "response_count": manifest["response_count"], "cwd_mode": manifest["cwd_mode"],
        "dense_available": manifest["dense_available"], "dense_status": manifest["dense_status"],
        "platform": manifest["platform"],
    }


def _validate_capture_provenance(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _CAPTURE_PROVENANCE_KEYS or _contains_sensitive_key(value):
        raise ValueError("capture provenance is invalid")
    for name in ("manifest_hash", "proxy_manifest_hash", "representative_dataset_hash", "config_hash"):
        _validate_hash(value[name], name)
    if not isinstance(value["agent_class"], str) or not value["agent_class"].strip():
        raise ValueError("capture provenance is invalid")
    if value["capture_seed"] != 20260829 or value["session_count"] != _SESSION_COUNT or value["response_count"] != _RESPONSE_COUNT:
        raise ValueError("capture provenance is invalid")
    if value["cwd_mode"] != "root" or not isinstance(value["dense_available"], bool) or not isinstance(value["dense_status"], str) or not value["dense_status"].strip():
        raise ValueError("capture provenance is invalid")
    capture_platform = value["platform"]
    if not isinstance(capture_platform, dict) or set(capture_platform) != _CAPTURE_PLATFORM_KEYS:
        raise ValueError("capture provenance is invalid")
    if any(not isinstance(capture_platform[name], str) or not capture_platform[name].strip() for name in _CAPTURE_PLATFORM_KEYS):
        raise ValueError("capture provenance is invalid")
    return value


def _load_capture_manifest(
    transcript: Path, *, transcript_hash: str, catalog_hash: str,
) -> tuple[dict[str, object], str]:
    manifest_path = Path(f"{transcript}.manifest.json")
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("capture manifest is invalid or missing") from error
    if not isinstance(manifest, dict) or set(manifest) != _CAPTURE_MANIFEST_KEYS:
        raise ValueError("capture manifest schema is invalid")
    if manifest.get("schema_version") != 1:
        raise ValueError("capture manifest version is invalid")
    if manifest.get("transcript_hash") != transcript_hash or manifest.get("catalog_hash") != catalog_hash:
        raise ValueError("capture manifest provenance is invalid")
    metadata = {name: manifest[name] for name in _CAPTURE_METADATA_KEYS}
    expected = _capture_manifest(metadata, transcript_hash=transcript_hash)
    for name in _CAPTURE_METADATA_KEYS:
        if manifest[name] != expected[name]:
            raise ValueError("capture manifest metadata is invalid")
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        raise ValueError("capture manifest timestamp is invalid")
    try:
        datetime.fromisoformat(created_at)
    except ValueError as error:
        raise ValueError("capture manifest timestamp is invalid") from error
    return manifest, hashlib.sha256(raw).hexdigest()


def _capture_payload(
    proxy_root: str | Path,
    catalog_path: str | Path,
    *,
    agent_class: type | Callable[[str], object] | None = None,
    suite_loader: Callable[[str | Path, str], object] | None = None,
) -> tuple[bytes, dict[str, object]]:
    """Build a transcript in the sealed capture environment without publishing it."""
    catalog = Path(catalog_path)
    if not catalog.is_absolute():
        catalog = _repo_root() / catalog
    catalog = catalog.resolve()
    with _sealed_capture_environment():
        loader = suite_loader or _load_verified_proxy_suite
        loaded = loader(proxy_root, "representative")
        rows = loaded.rows if hasattr(loaded, "rows") else loaded[1]
        eligible = [row for row in rows if row.get("proxy_fold") in {1, 2, 3, 4}]
        selected = sorted(eligible, key=lambda row: stable_int(20260829, str(row["sample_id"])))[:_SESSION_COUNT]
        if len(selected) != _SESSION_COUNT:
            raise ValueError("representative suite does not provide 200 eligible rows")
        catalog_hash_before = sha256_file(catalog)
        if agent_class is None:
            from agent import Agent as agent_class  # import only after sealed selection
        agent = agent_class(str(catalog))
        transcript: list[dict[str, object]] = []
        for index, row in enumerate(selected, 1):
            session_id = f"bench_{index:04d}"
            profile = _safe_profile(row.get("user_profile"))
            agent.reset(session_id, profile)
            dialogue = ProxyDialogue(row, row.get("dialogue_variant"))
            disclosed: set[str] = set()
            boundary_used = False
            override_applied = row.get("scenario_type") != "intent_override"
            message = dialogue.initial_message(str(row["category_bucket"]), disclosed)
            for turn in _TURNS:
                transcript.append({"session_id": session_id, "turn": turn, "profile": profile, "message": message})
                response = agent.respond(session_id, message, turn, 10)
                if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                    raise TypeError("agent response is not a valid response object")
                if turn == _TURNS[-1]:
                    continue
                override = row.get("behavior", {}).get("override", {})
                if not override_applied and turn + 1 == int(override.get("turn", 3)):
                    override_applied = True
                    value = str(override.get("new_value", ""))
                    if value:
                        disclosed.add(value)
                    message = str(override.get("message", "Actually, please ignore my earlier preference."))
                else:
                    message, boundary_used = dialogue.customer_reply(response.get("ask_attribute"), disclosed, boundary_used)
        if sha256_file(catalog) != catalog_hash_before:
            raise ValueError("catalog changed during transcript capture")
    validate_transcript(transcript)
    encoded = _serialize_jsonl(transcript)
    dense = getattr(agent, "dense", None)
    dense_available = bool(getattr(dense, "available", False))
    proxy_manifest_hash = getattr(loaded, "manifest_hash", None)
    representative_dataset_hash = getattr(loaded, "dataset_hash", None)
    if proxy_manifest_hash is None:
        proxy_manifest_hash = _canonical_hash(loaded[0] if isinstance(loaded, tuple) else {})
    if representative_dataset_hash is None:
        representative_dataset_hash = _canonical_hash(rows)
    metadata = {
        "catalog_hash": catalog_hash_before,
        "proxy_manifest_hash": proxy_manifest_hash,
        "representative_dataset_hash": representative_dataset_hash,
        "agent_class": f"{agent.__class__.__module__}:{agent.__class__.__qualname__}",
        "config_hash": hashlib.sha256(repr(getattr(agent, "config", None)).encode("utf-8")).hexdigest(),
        "dense_available": dense_available,
        "dense_status": getattr(dense, "status", "available" if dense_available else "unavailable"),
        "capture_seed": 20260829, "session_count": _SESSION_COUNT, "response_count": _RESPONSE_COUNT,
        "cwd_mode": "root", "platform": {"python": platform.python_version(), "platform": platform.platform()},
    }
    return encoded, metadata


def capture_transcript(
    proxy_root: str | Path, catalog_path: str | Path, output: str | Path, *,
    agent_class: type | Callable[[str], object] | None = None,
    suite_loader: Callable[[str | Path, str], object] | None = None,
) -> str:
    """Freeze and seal a response-independent 800-row transcript and sidecar."""
    destination = Path(output)
    manifest_path = Path(f"{destination}.manifest.json")
    if os.path.lexists(manifest_path):
        raise FileExistsError("transcript manifest already exists")
    encoded, metadata = _capture_payload(proxy_root, catalog_path, agent_class=agent_class, suite_loader=suite_loader)
    digest = hashlib.sha256(encoded).hexdigest()
    manifest = _capture_manifest(metadata, transcript_hash=digest)
    manifest_payload = _canonical_bytes(manifest) + b"\n"
    if os.path.lexists(destination):
        if destination.read_bytes() != encoded:
            raise ValueError("replayed transcript does not match existing bytes")
        _publish_bundle([(manifest_path, manifest_payload)])
    else:
        _publish_bundle([(destination, encoded), (manifest_path, manifest_payload)])
    return digest


def _finite(value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("metric must be finite")
    number = float(value)
    if number < 0 or (positive and number <= 0):
        raise ValueError("metric must be nonnegative")
    return number


def _finite_signed(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("metric must be finite")
    return float(value)


def _memory_mib(
    *, process: object | None = None, os_name: str | None = None, platform_name: str | None = None,
    resource_module: object | None = None,
) -> tuple[float, float, str]:
    """Return current RSS, genuine process high-water mark, and its unit source."""
    info = (process or psutil.Process()).memory_info()
    rss = float(info.rss) / (1024 * 1024)
    if (os_name or os.name) == "nt":
        peak = getattr(info, "peak_wset", None)
        if isinstance(peak, int) and peak >= 0:
            return rss, float(peak) / (1024 * 1024), "windows_peak_wset"
        raise RuntimeError("Windows peak working set is unavailable")
    if resource_module is None:
        try:
            import resource as resource_module
        except ImportError as error:
            raise RuntimeError("POSIX peak RSS is unavailable") from error
    usage = resource_module.getrusage(resource_module.RUSAGE_SELF)
    peak = getattr(usage, "ru_maxrss", None)
    if isinstance(peak, bool) or not isinstance(peak, (int, float)) or peak < 0:
        raise RuntimeError("POSIX peak RSS is unavailable")
    if (platform_name or sys.platform) == "darwin":
        return rss, float(peak) / (1024 * 1024), "posix_ru_maxrss_bytes"
    return rss, float(peak) / 1024, "posix_ru_maxrss_kib"


def _validate_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _validate_latency_values(value: object, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != _RESPONSE_COUNT:
        raise ValueError(f"{name} must contain exactly 800 values")
    return [_finite(item) for item in value]


def _validate_latency_diagnostic(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"count", "p50", "p95", "max"}:
        raise ValueError("instrumentation delta summary is invalid")
    if value["count"] != _RESPONSE_COUNT:
        raise ValueError("instrumentation delta count is invalid")
    for key in ("p50", "p95", "max"):
        _finite_signed(value[key])


def _validate_trial(trial: object) -> dict[str, object]:
    if not isinstance(trial, dict) or set(trial) != _TRIAL_KEYS:
        raise ValueError("worker trial schema is invalid")
    for name in ("init_ms", "peak_mib", "rss_mib"):
        _finite(trial[name])
    if trial["peak_metric_source"] not in {
        "windows_peak_wset", "posix_ru_maxrss_kib", "posix_ru_maxrss_bytes",
    }:
        raise ValueError("peak metric source is invalid")
    _validate_latency_values(trial["latencies_ms"], "latencies_ms")
    _validate_latency_values(trial["trace_latencies_ms"], "trace_latencies_ms")
    _validate_latency_diagnostic(trial["instrumentation_delta_ms"])
    expected_instrumentation = _latency_summary([
        wall - trace for wall, trace in zip(trial["latencies_ms"], trial["trace_latencies_ms"], strict=True)
    ])
    if trial["instrumentation_delta_ms"] != expected_instrumentation:
        raise ValueError("instrumentation delta summary does not match latency arrays")
    if not isinstance(trial["dense_available"], bool):
        raise TypeError("trial dense availability is invalid")
    if not isinstance(trial["dense_status"], str) or not trial["dense_status"].strip():
        raise ValueError("trial dense status is invalid")
    if isinstance(trial["fallback_count"], bool) or not isinstance(trial["fallback_count"], int) or trial["fallback_count"] < 0:
        raise ValueError("fallback count is invalid")
    if isinstance(trial["response_count"], bool) or trial["response_count"] != _RESPONSE_COUNT:
        raise ValueError("trial response count is invalid")
    for name in ("response_hash", "transcript_hash", "catalog_hash"):
        _validate_hash(trial[name], name)
    if trial["catalog_snapshot_hash"] != trial["catalog_hash"]:
        raise ValueError("trial catalog snapshot provenance is invalid")
    _validate_hash(trial["catalog_snapshot_hash"], "catalog_snapshot_hash")
    _validate_capture_provenance(trial["capture_provenance"])
    _validate_runtime_platform(trial["platform"])
    return trial


def run_worker(
    catalog_path: str | Path, transcript_path: str | Path, *, agent_class: type | None = None,
    clock: Callable[[], float] = time.perf_counter, catalog_hasher: Callable[[str | Path], str] = sha256_file,
) -> dict[str, object]:
    raw = Path(transcript_path).read_bytes()
    transcript_hash = hashlib.sha256(raw).hexdigest()
    try:
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("transcript is invalid JSONL") from error
    validate_transcript(rows)
    catalog_hash = catalog_hasher(catalog_path)
    _validate_hash(catalog_hash, "catalog_hash")
    manifest, manifest_hash = _load_capture_manifest(
        Path(transcript_path), transcript_hash=transcript_hash, catalog_hash=catalog_hash,
    )
    if agent_class is None:
        from agent import Agent as agent_class
    started = clock()
    agent = agent_class(str(Path(catalog_path).resolve()))
    init_ms = (clock() - started) * 1000
    _finite(init_ms)
    actual_agent_class = f"{agent.__class__.__module__}:{agent.__class__.__qualname__}"
    actual_config_hash = hashlib.sha256(repr(getattr(agent, "config", None)).encode("utf-8")).hexdigest()
    if manifest["agent_class"] != actual_agent_class or manifest["config_hash"] != actual_config_hash:
        raise ValueError("capture manifest agent provenance is invalid")
    catalog = getattr(agent, "catalog", None)
    catalog_ids = getattr(catalog, "valid_ids", None)
    if not isinstance(catalog_ids, set) or len(catalog_ids) != 50_000:
        raise ValueError("benchmark requires exactly 50,000 catalog IDs")
    trace_sink = getattr(agent, "traces", None)
    records = getattr(trace_sink, "records", None)
    if not isinstance(records, list):
        raise TypeError("agent must expose trace records")
    latency_values: list[float] = []
    trace_latency_values: list[float] = []
    normalized_responses: list[list[str]] = []
    fallback_count = 0
    for row in rows:
        session_id, turn = row["session_id"], row["turn"]
        if turn == 1:
            agent.reset(session_id, row["profile"])
        wall_started = clock()
        response = agent.respond(session_id, row["message"], turn, 10)
        wall_elapsed = (clock() - wall_started) * 1000
        if not math.isfinite(wall_elapsed) or wall_elapsed < 0:
            raise ValueError("response timing is invalid")
        if not isinstance(response, dict) or not isinstance(response.get("message"), str) or not isinstance(response.get("recommendations"), list):
            raise TypeError("agent response schema is invalid")
        normalized_responses.append(normalize_recommendations(response["recommendations"], catalog_ids))
        records = getattr(trace_sink, "records", None)
        if not isinstance(records, list) or not records or not isinstance(records[-1], dict):
            raise ValueError("trace record missing")
        record = records[-1]
        if record.get("session_id") != session_id or record.get("turn") != turn:
            raise ValueError("trace records are stale, malformed, or out of order")
        elapsed = _finite(record.get("elapsed_ms"))
        fallbacks = record.get("fallbacks")
        if not isinstance(fallbacks, (list, tuple, set)):
            raise TypeError("trace fallbacks are invalid")
        latency_values.append(round(wall_elapsed, 3))
        trace_latency_values.append(elapsed)
        fallback_count += int(bool(fallbacks))
    records = getattr(trace_sink, "records", None)
    if not isinstance(records, list) or len(records) != _RESPONSE_COUNT:
        raise ValueError("trace count does not match transcript")
    if catalog_hasher(catalog_path) != catalog_hash:
        raise ValueError("catalog changed during worker replay")
    dense = getattr(agent, "dense", None)
    dense_available = bool(getattr(dense, "available", False))
    dense_status = getattr(dense, "status", "available" if dense_available else "unavailable")
    if not isinstance(dense_status, str) or not dense_status:
        raise ValueError("dense status is invalid")
    rss_mib, peak_mib, peak_metric_source = _memory_mib()
    result: dict[str, object] = {
        "init_ms": round(init_ms, 3), "peak_mib": round(peak_mib, 3), "rss_mib": round(rss_mib, 3),
        "peak_metric_source": peak_metric_source, "latencies_ms": latency_values,
        "trace_latencies_ms": trace_latency_values,
        "instrumentation_delta_ms": _latency_summary([wall - trace for wall, trace in zip(latency_values, trace_latency_values, strict=True)]),
        "dense_available": dense_available, "dense_status": dense_status,
        "fallback_count": fallback_count, "response_hash": _canonical_hash(normalized_responses),
        "transcript_hash": transcript_hash, "catalog_hash": catalog_hash, "catalog_snapshot_hash": catalog_hash,
        "capture_provenance": _capture_provenance(manifest, manifest_hash), "response_count": _RESPONSE_COUNT,
        "platform": _runtime_platform_data(),
    }
    _validate_trial(result)
    return result


def aggregate_trials(trials: list[dict[str, object]]) -> dict[str, object]:
    if not trials:
        raise ValueError("at least one trial is required")
    values: defaultdict[str, list[float]] = defaultdict(list)
    combined_latencies: list[float] = []
    response_hashes: set[str] = set()
    transcript_hashes: set[str] = set()
    catalog_hashes: set[str] = set()
    snapshot_hashes: set[str] = set()
    capture_provenances: dict[str, dict[str, object]] = {}
    runtime_platforms: dict[str, dict[str, object]] = {}
    peak_sources: set[str] = set()
    dense_values: list[bool] = []
    statuses: set[str] = set()
    fallback_count = 0
    for trial in trials:
        _validate_trial(trial)
        for name in ("init_ms", "peak_mib", "rss_mib"):
            values[name].append(_finite(trial.get(name)))
        latencies = _validate_latency_values(trial["latencies_ms"], "latencies_ms")
        combined_latencies.extend(latencies)
        response_hash = trial["response_hash"]
        transcript_hash = trial["transcript_hash"]
        response_hashes.add(response_hash)
        transcript_hashes.add(transcript_hash)
        dense_values.append(trial["dense_available"])
        statuses.add(trial["dense_status"])
        fallback_count += trial["fallback_count"]
        catalog_hashes.add(trial["catalog_hash"])
        snapshot_hashes.add(trial["catalog_snapshot_hash"])
        capture_provenances[_canonical_hash(trial["capture_provenance"])] = trial["capture_provenance"]
        runtime_platforms[_canonical_hash(trial["platform"])] = trial["platform"]
        peak_sources.add(trial["peak_metric_source"])
    if len(response_hashes) != 1 or len(transcript_hashes) != 1 or len(catalog_hashes) != 1 or len(snapshot_hashes) != 1 or len(capture_provenances) != 1 or len(runtime_platforms) != 1 or len(peak_sources) != 1:
        raise ValueError("trials disagree on input or output hash")
    return {
        "trial_count": len(trials), "init_ms": statistics.median(values["init_ms"]),
        "peak_mib": statistics.median(values["peak_mib"]), "rss_mib": statistics.median(values["rss_mib"]),
        "latency_ms": _latency_summary(combined_latencies), "dense_available": all(dense_values),
        "dense_statuses": sorted(statuses), "fallback_count": fallback_count,
        "response_hash": response_hashes.pop(), "transcript_hash": transcript_hashes.pop(), "catalog_hash": catalog_hashes.pop(),
        "catalog_snapshot_hash": snapshot_hashes.pop(), "capture_provenance": next(iter(capture_provenances.values())),
        "runtime_platform": next(iter(runtime_platforms.values())), "peak_metric_source": peak_sources.pop(),
        "response_count": _RESPONSE_COUNT, "trials": trials,
    }


def _validate_aggregate_report(report: object) -> dict[str, object]:
    if not isinstance(report, dict):
        raise TypeError("aggregate report must be an object")
    aggregate_keys = {
        "trial_count", "init_ms", "peak_mib", "rss_mib", "latency_ms", "dense_available",
        "dense_statuses", "fallback_count", "response_hash", "transcript_hash", "catalog_hash",
        "catalog_snapshot_hash", "capture_provenance", "runtime_platform", "peak_metric_source", "response_count", "trials",
    }
    keys = set(report)
    parent_keys = aggregate_keys | {"cwd_mode", "platform"}
    if keys != parent_keys and keys != parent_keys | {"comparison"}:
        raise ValueError("aggregate report schema is invalid")
    trials = report.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("aggregate report trials are invalid")
    expected = aggregate_trials(trials)
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError("aggregate report does not match raw trial aggregate")
    if report.get("cwd_mode") not in {"root", "outside"}:
        raise ValueError("aggregate report cwd mode is invalid")
    platform_data = report["platform"]
    if not isinstance(platform_data, dict) or set(platform_data) != _PARENT_PLATFORM_KEYS or _contains_sensitive_key(platform_data):
        raise ValueError("aggregate report platform is invalid")
    for name in ("os", "python", "processor", "psutil"):
        if not isinstance(platform_data[name], str) or not platform_data[name].strip():
            raise ValueError("aggregate report platform is invalid")
    for name in ("cpu_logical", "cpu_physical"):
        if isinstance(platform_data[name], bool) or not isinstance(platform_data[name], int) or platform_data[name] < 0:
            raise ValueError("aggregate report platform is invalid")
    _finite(platform_data["ram_mib"])
    onnxruntime_version = platform_data["onnxruntime"]
    if onnxruntime_version is not None and (not isinstance(onnxruntime_version, str) or not onnxruntime_version.strip()):
        raise ValueError("aggregate report platform is invalid")
    runtime_platform = _validate_runtime_platform(report["runtime_platform"])
    if any(runtime_platform[name] != platform_data[name] for name in _TRIAL_PLATFORM_KEYS):
        raise ValueError("aggregate report runtime platform does not match parent platform")
    if "comparison" in report:
        comparison = report["comparison"]
        if not isinstance(comparison, dict) or set(comparison) != {"accepted", "same_output", "no_regression", "material_gain", "safe", "deltas"}:
            raise ValueError("aggregate report comparison is invalid")
        if any(not isinstance(comparison[name], bool) for name in ("accepted", "same_output", "no_regression", "material_gain", "safe")):
            raise ValueError("aggregate report comparison is invalid")
        deltas = comparison["deltas"]
        if not isinstance(deltas, dict) or set(deltas) != {"p95_pct", "init_pct", "peak_pct"}:
            raise ValueError("aggregate report comparison is invalid")
        for value in deltas.values():
            _finite_signed(value)
    return report


def compare_reports(
    candidate: dict[str, object], baseline: dict[str, object], *, require_dense: bool = True
) -> dict[str, object]:
    def metrics(report: dict[str, object], *, baseline_mode: bool) -> tuple[float, float, float, float]:
        latency = report.get("latency_ms")
        if not isinstance(latency, dict):
            raise TypeError("report latency is invalid")
        return (_finite(latency.get("p95"), positive=baseline_mode), _finite(latency.get("max")),
                _finite(report.get("init_ms"), positive=baseline_mode), _finite(report.get("peak_mib"), positive=baseline_mode))
    candidate = _validate_aggregate_report(candidate)
    baseline = _validate_aggregate_report(baseline)
    base_p95, _, base_init, base_peak = metrics(baseline, baseline_mode=True)
    cand_p95, cand_max, cand_init, cand_peak = metrics(candidate, baseline_mode=False)
    for report in (candidate, baseline):
        for name in ("response_hash", "transcript_hash", "catalog_hash"):
            _validate_hash(report.get(name), name)
    same_output = all(candidate[name] == baseline[name] for name in ("response_hash", "transcript_hash", "catalog_hash"))
    if candidate["cwd_mode"] != baseline["cwd_mode"]:
        raise ValueError("comparison cwd mode differs")
    if candidate["runtime_platform"] != baseline["runtime_platform"]:
        raise ValueError("comparison runtime platform differs")
    if candidate["peak_metric_source"] != baseline["peak_metric_source"]:
        raise ValueError("comparison peak metric source differs")
    if candidate["capture_provenance"] != baseline["capture_provenance"]:
        raise ValueError("comparison capture provenance differs")
    no_regression = cand_p95 <= base_p95 * 1.05 and cand_init <= base_init * 1.05 and cand_peak <= base_peak * 1.05
    material_gain = cand_p95 <= base_p95 * 0.90 or cand_init <= base_init * 0.95 or cand_peak <= base_peak * 0.95
    safe = cand_max < 1500 and candidate.get("fallback_count") == 0 and (
        not require_dense or candidate.get("dense_available") is True
    )
    return {"accepted": bool(same_output and no_regression and material_gain and safe), "same_output": same_output,
            "no_regression": no_regression, "material_gain": material_gain, "safe": safe,
            "deltas": {"p95_pct": (cand_p95 / base_p95 - 1) * 100, "init_pct": (cand_init / base_init - 1) * 100, "peak_pct": (cand_peak / base_peak - 1) * 100}}


def _onnxruntime_version() -> str | None:
    try:
        import onnxruntime
        return onnxruntime.__version__
    except ImportError:
        return None


def _runtime_platform_data() -> dict[str, object]:
    return {"os": platform.platform(), "python": platform.python_version(),
            "processor": platform.processor() or "unknown", "onnxruntime": _onnxruntime_version(),
            "psutil": psutil.__version__}


def _validate_runtime_platform(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _TRIAL_PLATFORM_KEYS or _contains_sensitive_key(value):
        raise ValueError("trial platform is invalid")
    for name in ("os", "python", "processor", "psutil"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise ValueError("trial platform is invalid")
    if value["onnxruntime"] is not None and (not isinstance(value["onnxruntime"], str) or not value["onnxruntime"].strip()):
        raise ValueError("trial platform is invalid")
    return value


def _platform_data() -> dict[str, object]:
    memory = psutil.virtual_memory()
    return {**_runtime_platform_data(), "cpu_logical": psutil.cpu_count(logical=True) or 0,
            "cpu_physical": psutil.cpu_count(logical=False) or 0,
            "ram_mib": round(memory.total / (1024 * 1024), 3)}


def _diagnostic_excerpt(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = "".join(
        "?" if ord(character) < 32 or 127 <= ord(character) <= 159 else character
        for character in str(value or "")
    )
    return text[-2_000:]


def _parse_worker_stdout(stdout: object) -> dict[str, object]:
    text = str(stdout or "")
    decoder = json.JSONDecoder()
    try:
        parsed, end = decoder.raw_decode(text.lstrip())
    except json.JSONDecodeError as error:
        raise ValueError(f"worker stdout is not one JSON document: {_diagnostic_excerpt(text)}") from error
    if text.lstrip()[end:].strip():
        raise ValueError(f"worker stdout contains extra output: {_diagnostic_excerpt(text)}")
    _validate_trial(parsed)
    return parsed


def _copy_catalog_snapshot(catalog: Path, destination: Path, expected_hash: str) -> Path:
    if sha256_file(catalog) != expected_hash:
        raise ValueError("catalog changed before snapshot")
    snapshot = destination / "catalog.snapshot.jsonl"
    with catalog.open("rb") as source, snapshot.open("xb") as target:
        shutil.copyfileobj(source, target)
        target.flush()
        os.fsync(target.fileno())
    source_hash_after = sha256_file(catalog)
    snapshot_hash = sha256_file(snapshot)
    if source_hash_after != expected_hash or snapshot_hash != expected_hash:
        raise ValueError("catalog changed while creating snapshot")
    return snapshot


def run_parent(
    catalog_path: str | Path, transcript_path: str | Path, *, trials: int = 3, cwd_mode: str = "outside",
    worker_timeout_seconds: float = 3600,
) -> dict[str, object]:
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be at least one")
    if cwd_mode not in {"root", "outside"}:
        raise ValueError("cwd_mode must be root or outside")
    if isinstance(worker_timeout_seconds, bool) or not isinstance(worker_timeout_seconds, (int, float)) or not math.isfinite(worker_timeout_seconds) or worker_timeout_seconds <= 0:
        raise ValueError("worker timeout must be positive")
    root = _repo_root()
    catalog = Path(catalog_path).resolve()
    transcript = Path(transcript_path).resolve()
    catalog_hash_before = sha256_file(catalog)
    transcript_hash_before = sha256_file(transcript)
    manifest, manifest_hash = _load_capture_manifest(
        transcript, transcript_hash=transcript_hash_before, catalog_hash=catalog_hash_before,
    )
    expected_provenance = _capture_provenance(manifest, manifest_hash)
    environment = os.environ.copy()
    environment.pop("COMPASSCART_DISABLE_DENSE", None)
    environment["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root)])
    result_trials: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="compasscart-benchmark-") as temporary:
        snapshot = _copy_catalog_snapshot(catalog, Path(temporary), catalog_hash_before)
        cwd = root if cwd_mode == "root" else Path(temporary)
        for _ in range(trials):
            if sha256_file(catalog) != catalog_hash_before:
                raise ValueError("catalog changed during benchmark")
            if sha256_file(transcript) != transcript_hash_before:
                raise ValueError("transcript changed during benchmark")
            if sha256_file(snapshot) != catalog_hash_before:
                raise ValueError("catalog snapshot changed during benchmark")
            command = [sys.executable, "-m", "tools.benchmark_release", "--worker", "--catalog", str(snapshot), "--transcript", str(transcript)]
            try:
                completed = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, check=False, timeout=worker_timeout_seconds)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"benchmark worker timed out after {worker_timeout_seconds}s; stdout={_diagnostic_excerpt(error.stdout)}; stderr={_diagnostic_excerpt(error.stderr)}"
                ) from error
            if completed.returncode != 0 or completed.stderr.strip():
                raise RuntimeError(
                    f"benchmark worker failed with return code {completed.returncode}; stdout={_diagnostic_excerpt(completed.stdout)}; stderr={_diagnostic_excerpt(completed.stderr)}"
                )
            parsed = _parse_worker_stdout(completed.stdout)
            if parsed["transcript_hash"] != transcript_hash_before:
                raise ValueError("worker transcript provenance mismatch")
            if parsed["catalog_hash"] != catalog_hash_before:
                raise ValueError("worker catalog provenance mismatch")
            if parsed["catalog_snapshot_hash"] != catalog_hash_before:
                raise ValueError("worker catalog snapshot provenance mismatch")
            if parsed["capture_provenance"] != expected_provenance:
                raise ValueError("worker capture provenance mismatch")
            result_trials.append(parsed)
    catalog_hash_after = sha256_file(catalog)
    transcript_hash_after = sha256_file(transcript)
    if catalog_hash_after != catalog_hash_before:
        raise ValueError("catalog changed during benchmark")
    if transcript_hash_after != transcript_hash_before:
        raise ValueError("transcript changed during benchmark")
    report = aggregate_trials(result_trials)
    report.update({"cwd_mode": cwd_mode, "platform": _platform_data()})
    return _validate_aggregate_report(report)


def write_report(destination: str | Path, report: dict[str, object]) -> None:
    try:
        encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("report is not JSON serializable") from error
    _write_exclusive(Path(destination), encoded)


def _preflight_destination(destination: str | Path) -> None:
    path = Path(destination)
    if os.path.lexists(path):
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".probe", dir=path.parent)
    temporary = Path(temporary_name)
    probe = temporary.with_suffix(".link")
    temporary_stat = os.fstat(descriptor)
    try:
        os.close(descriptor)
        os.link(temporary, probe)
    except OSError as error:
        raise RuntimeError("exclusive report publication requires hardlink support") from error
    finally:
        _unlink_if_same_inode(probe, temporary_stat)
        _unlink_if_same_inode(temporary, temporary_stat)


def load_report(path: str | Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("report is invalid") from error
    if not isinstance(value, dict):
        raise TypeError("report must be an object")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--capture-transcript", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--proxy-root")
    parser.add_argument("--transcript")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cwd-mode", choices=("root", "outside"), default="outside")
    parser.add_argument("--output")
    parser.add_argument("--compare")
    parser.add_argument("--allow-dense-unavailable", action="store_true")
    parser.add_argument("--worker-timeout-seconds", type=float, default=3600)
    args = parser.parse_args(raw_args)
    provided = {item.split("=", 1)[0] for item in raw_args if item.startswith("--")}
    if args.capture_transcript:
        allowed = {"--capture-transcript", "--catalog", "--proxy-root", "--output"}
        required = {"--proxy-root", "--output"}
    elif args.worker:
        allowed = {"--worker", "--catalog", "--transcript"}
        required = {"--transcript"}
    else:
        allowed = {"--catalog", "--transcript", "--trials", "--cwd-mode", "--output", "--compare", "--allow-dense-unavailable", "--worker-timeout-seconds"}
        required = {"--transcript", "--output"}
    invalid = provided - allowed
    missing = required - provided
    if invalid:
        parser.error(f"options are incompatible with this mode: {', '.join(sorted(invalid))}")
    if missing:
        parser.error(f"missing required options: {', '.join(sorted(missing))}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.capture_transcript:
        digest = capture_transcript(args.proxy_root, args.catalog, args.output)
        print(json.dumps({"responses": _RESPONSE_COUNT, "sha256": digest}, separators=(",", ":")))
        return
    if args.worker:
        print(json.dumps(run_worker(args.catalog, args.transcript), sort_keys=True, separators=(",", ":"), allow_nan=False))
        return
    _preflight_destination(args.output)
    baseline: dict[str, object] | None = None
    if args.compare:
        baseline = load_report(args.compare)
        _validate_aggregate_report(baseline)
    report = run_parent(
        args.catalog, args.transcript, trials=args.trials, cwd_mode=args.cwd_mode,
        worker_timeout_seconds=args.worker_timeout_seconds,
    )
    if args.compare:
        report["comparison"] = compare_reports(
            report, baseline, require_dense=not args.allow_dense_unavailable
        )
    write_report(args.output, report)
    latency = report["latency_ms"]
    failed = ((not args.allow_dense_unavailable and report["dense_available"] is not True) or report["fallback_count"] != 0 or not isinstance(latency, dict) or _finite(latency.get("max")) >= 1500 or (args.compare and not report["comparison"]["accepted"]))
    if failed:
        raise SystemExit("benchmark gate failed")


if __name__ == "__main__":
    main()
