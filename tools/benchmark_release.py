"""Full-catalog, response-hash release benchmark for CompassCart."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
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
_SENSITIVE_KEYS = frozenset({"target", "ground_truth", "intent_card", "behavior", "recommendations", "hits"})
_ROW_KEYS = frozenset({"session_id", "turn", "profile", "message"})


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
    if isinstance(value, str):
        lowered = value.lower()
        return any(key in lowered for key in _SENSITIVE_KEYS)
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
        if _contains_sensitive_material(message):
            raise ValueError("message contains sensitive material")
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
    if os.path.lexists(destination):
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(destination):
            raise FileExistsError(f"output already exists: {destination}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise
        except OSError as error:
            raise RuntimeError("exclusive publication requires hardlink support") from error
    finally:
        temporary.unlink(missing_ok=True)


def _serialize_jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def capture_transcript(
    proxy_root: str | Path,
    catalog_path: str | Path,
    output: str | Path,
    *,
    agent_class: type | Callable[[str], object] | None = None,
    suite_loader: Callable[[str | Path, str], object] | None = None,
) -> str:
    """Freeze a response-independent 800-row transcript from the verified representative suite."""
    destination = Path(output)
    if os.path.lexists(destination):
        raise FileExistsError(f"output already exists: {destination}")
    loader = suite_loader or _load_verified_proxy_suite
    loaded = loader(proxy_root, "representative")
    rows = loaded.rows if hasattr(loaded, "rows") else loaded[1]  # test seam retains verified default path
    eligible = [row for row in rows if row.get("proxy_fold") in {1, 2, 3, 4}]
    selected = sorted(eligible, key=lambda row: stable_int(20260829, str(row["sample_id"])))[:_SESSION_COUNT]
    if len(selected) != _SESSION_COUNT:
        raise ValueError("representative suite does not provide 200 eligible rows")
    if agent_class is None:
        from agent import Agent as agent_class  # import only after sealed selection
    agent = agent_class(str(Path(catalog_path).resolve()))
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
    validate_transcript(transcript)
    encoded = _serialize_jsonl(transcript)
    _write_exclusive(destination, encoded)
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("metric must be finite")
    number = float(value)
    if number < 0 or (positive and number <= 0):
        raise ValueError("metric must be nonnegative")
    return number


def _memory_mib() -> tuple[float, float]:
    info = psutil.Process().memory_info()
    rss = float(info.rss) / (1024 * 1024)
    peak_bytes = getattr(info, "peak_wset", info.rss) if os.name == "nt" else info.rss
    return float(peak_bytes) / (1024 * 1024), rss


def run_worker(
    catalog_path: str | Path, transcript_path: str | Path, *, agent_class: type | None = None
) -> dict[str, object]:
    raw = Path(transcript_path).read_bytes()
    transcript_hash = hashlib.sha256(raw).hexdigest()
    try:
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("transcript is invalid JSONL") from error
    validate_transcript(rows)
    if agent_class is None:
        from agent import Agent as agent_class
    started = time.perf_counter()
    agent = agent_class(str(Path(catalog_path).resolve()))
    init_ms = (time.perf_counter() - started) * 1000
    catalog = getattr(agent, "catalog", None)
    catalog_ids = getattr(catalog, "valid_ids", None)
    if not isinstance(catalog_ids, set) or len(catalog_ids) != 50_000:
        raise ValueError("benchmark requires exactly 50,000 catalog IDs")
    trace_sink = getattr(agent, "traces", None)
    records = getattr(trace_sink, "records", None)
    if not isinstance(records, list):
        raise TypeError("agent must expose trace records")
    latency_values: list[float] = []
    normalized_responses: list[list[str]] = []
    fallback_count = 0
    for row in rows:
        session_id, turn = row["session_id"], row["turn"]
        if turn == 1:
            agent.reset(session_id, row["profile"])
        wall_started = time.perf_counter()
        response = agent.respond(session_id, row["message"], turn, 10)
        wall_elapsed = (time.perf_counter() - wall_started) * 1000
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
        latency_values.append(elapsed)
        fallback_count += int(bool(fallbacks))
    records = getattr(trace_sink, "records", None)
    if not isinstance(records, list) or len(records) != _RESPONSE_COUNT:
        raise ValueError("trace count does not match transcript")
    dense = getattr(agent, "dense", None)
    dense_available = bool(getattr(dense, "available", False))
    dense_status = getattr(dense, "status", "available" if dense_available else "unavailable")
    if not isinstance(dense_status, str) or not dense_status:
        raise ValueError("dense status is invalid")
    peak_mib, rss_mib = _memory_mib()
    return {
        "init_ms": round(init_ms, 3), "peak_mib": round(peak_mib, 3), "rss_mib": round(rss_mib, 3),
        "latencies_ms": latency_values, "dense_available": dense_available, "dense_status": dense_status,
        "fallback_count": fallback_count, "response_hash": _canonical_hash(normalized_responses),
        "transcript_hash": transcript_hash, "response_count": _RESPONSE_COUNT,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
    }


def aggregate_trials(trials: list[dict[str, object]]) -> dict[str, object]:
    if not trials:
        raise ValueError("at least one trial is required")
    values: defaultdict[str, list[float]] = defaultdict(list)
    combined_latencies: list[float] = []
    response_hashes: set[str] = set()
    transcript_hashes: set[str] = set()
    dense_values: list[bool] = []
    statuses: set[str] = set()
    fallback_count = 0
    for trial in trials:
        if not isinstance(trial, dict):
            raise TypeError("trial must be an object")
        for name in ("init_ms", "peak_mib", "rss_mib"):
            values[name].append(_finite(trial.get(name)))
        latencies = trial.get("latencies_ms")
        if not isinstance(latencies, list) or not latencies:
            raise ValueError("trial latencies are invalid")
        combined_latencies.extend(_finite(item) for item in latencies)
        if trial.get("response_count") != _RESPONSE_COUNT:
            raise ValueError("trial response count is invalid")
        response_hash = trial.get("response_hash")
        transcript_hash = trial.get("transcript_hash")
        if not isinstance(response_hash, str) or not response_hash or not isinstance(transcript_hash, str) or not transcript_hash:
            raise ValueError("trial hashes are invalid")
        response_hashes.add(response_hash)
        transcript_hashes.add(transcript_hash)
        if not isinstance(trial.get("dense_available"), bool):
            raise TypeError("trial dense availability is invalid")
        dense_values.append(trial["dense_available"])
        status = trial.get("dense_status")
        if not isinstance(status, str) or not status:
            raise ValueError("trial dense status is invalid")
        statuses.add(status)
        fallback_count += int(_finite(trial.get("fallback_count")))
    if len(response_hashes) != 1 or len(transcript_hashes) != 1:
        raise ValueError("trials disagree on input or output hash")
    return {
        "trial_count": len(trials), "init_ms": statistics.median(values["init_ms"]),
        "peak_mib": statistics.median(values["peak_mib"]), "rss_mib": statistics.median(values["rss_mib"]),
        "latency_ms": _latency_summary(combined_latencies), "dense_available": all(dense_values),
        "dense_statuses": sorted(statuses), "fallback_count": fallback_count,
        "response_hash": response_hashes.pop(), "transcript_hash": transcript_hashes.pop(),
        "response_count": _RESPONSE_COUNT, "trials": trials,
    }


def compare_reports(
    candidate: dict[str, object], baseline: dict[str, object], *, require_dense: bool = True
) -> dict[str, object]:
    def metrics(report: dict[str, object], *, baseline_mode: bool) -> tuple[float, float, float, float]:
        latency = report.get("latency_ms")
        if not isinstance(latency, dict):
            raise TypeError("report latency is invalid")
        return (_finite(latency.get("p95"), positive=baseline_mode), _finite(latency.get("max")),
                _finite(report.get("init_ms"), positive=baseline_mode), _finite(report.get("peak_mib"), positive=baseline_mode))
    base_p95, _, base_init, base_peak = metrics(baseline, baseline_mode=True)
    cand_p95, cand_max, cand_init, cand_peak = metrics(candidate, baseline_mode=False)
    same_output = candidate.get("response_hash") == baseline.get("response_hash") and candidate.get("transcript_hash") == baseline.get("transcript_hash")
    no_regression = cand_p95 <= base_p95 * 1.05 and cand_init <= base_init * 1.05 and cand_peak <= base_peak * 1.05
    material_gain = cand_p95 <= base_p95 * 0.90 or cand_init <= base_init * 0.95 or cand_peak <= base_peak * 0.95
    safe = cand_max < 1500 and candidate.get("fallback_count") == 0 and (
        not require_dense or candidate.get("dense_available") is True
    )
    return {"accepted": bool(same_output and no_regression and material_gain and safe), "same_output": same_output,
            "no_regression": no_regression, "material_gain": material_gain, "safe": safe,
            "deltas": {"p95_pct": (cand_p95 / base_p95 - 1) * 100, "init_pct": (cand_init / base_init - 1) * 100, "peak_pct": (cand_peak / base_peak - 1) * 100}}


def _platform_data() -> dict[str, object]:
    memory = psutil.virtual_memory()
    try:
        import onnxruntime
        onnx_version: str | None = onnxruntime.__version__
    except ImportError:
        onnx_version = None
    return {"os": platform.platform(), "python": platform.python_version(), "processor": platform.processor(),
            "cpu_logical": psutil.cpu_count(logical=True), "cpu_physical": psutil.cpu_count(logical=False),
            "ram_mib": round(memory.total / (1024 * 1024), 3), "onnxruntime": onnx_version, "psutil": psutil.__version__}


def run_parent(catalog_path: str | Path, transcript_path: str | Path, *, trials: int = 3, cwd_mode: str = "outside") -> dict[str, object]:
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be at least one")
    if cwd_mode not in {"root", "outside"}:
        raise ValueError("cwd_mode must be root or outside")
    root = _repo_root()
    catalog = Path(catalog_path).resolve()
    transcript = Path(transcript_path).resolve()
    transcript_hash = sha256_file(transcript)
    environment = os.environ.copy()
    environment.pop("COMPASSCART_DISABLE_DENSE", None)
    environment["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root)])
    result_trials: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="compasscart-benchmark-") as temporary:
        cwd = root if cwd_mode == "root" else Path(temporary)
        for _ in range(trials):
            completed = subprocess.run([sys.executable, "-m", "tools.benchmark_release", "--worker", "--catalog", str(catalog), "--transcript", str(transcript)], cwd=cwd, env=environment, capture_output=True, text=True, check=False)
            if completed.returncode != 0 or completed.stderr.strip():
                raise RuntimeError("benchmark worker failed")
            try:
                parsed = json.loads(completed.stdout)
            except json.JSONDecodeError as error:
                raise ValueError("worker stdout must be one JSON document") from error
            if not isinstance(parsed, dict) or parsed.get("transcript_hash") != transcript_hash:
                raise ValueError("worker transcript provenance mismatch")
            result_trials.append(parsed)
    report = aggregate_trials(result_trials)
    report.update({"catalog_hash": sha256_file(catalog), "cwd_mode": cwd_mode, "platform": _platform_data()})
    return report


def write_report(destination: str | Path, report: dict[str, object]) -> None:
    try:
        encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("report is not JSON serializable") from error
    _write_exclusive(Path(destination), encoded)


def load_report(path: str | Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("report is invalid") from error
    if not isinstance(value, dict):
        raise TypeError("report must be an object")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.capture_transcript:
        if not args.proxy_root or not args.output:
            raise SystemExit("--capture-transcript requires --proxy-root and --output")
        digest = capture_transcript(args.proxy_root, args.catalog, args.output)
        print(json.dumps({"responses": _RESPONSE_COUNT, "sha256": digest}, separators=(",", ":")))
        return
    if args.worker:
        if not args.transcript:
            raise SystemExit("--worker requires --transcript")
        print(json.dumps(run_worker(args.catalog, args.transcript), sort_keys=True, separators=(",", ":"), allow_nan=False))
        return
    if not args.transcript or not args.output:
        raise SystemExit("benchmark requires --transcript and --output")
    report = run_parent(args.catalog, args.transcript, trials=args.trials, cwd_mode=args.cwd_mode)
    if args.compare:
        report["comparison"] = compare_reports(
            report, load_report(args.compare), require_dense=not args.allow_dense_unavailable
        )
    write_report(args.output, report)
    latency = report["latency_ms"]
    failed = ((not args.allow_dense_unavailable and report["dense_available"] is not True) or report["fallback_count"] != 0 or not isinstance(latency, dict) or _finite(latency.get("max")) >= 1500 or (args.compare and not report["comparison"]["accepted"]))
    if failed:
        raise SystemExit("benchmark gate failed")


if __name__ == "__main__":
    main()
