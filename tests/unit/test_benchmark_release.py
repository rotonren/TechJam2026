from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import benchmark_release as bench


def _trial(**overrides: object) -> dict[str, object]:
    trial: dict[str, object] = {
        "init_ms": 20.0, "peak_mib": 110.0, "rss_mib": 85.0,
        "peak_metric_source": "windows_peak_wset", "latencies_ms": [2.0] * 800,
        "trace_latencies_ms": [1.0] * 800, "instrumentation_delta_ms": {"count": 800, "p50": 1.0, "p95": 1.0, "max": 1.0},
        "dense_available": True, "dense_status": "available", "fallback_count": 0,
        "response_hash": "a" * 64, "transcript_hash": "b" * 64, "catalog_hash": "c" * 64,
        "response_count": 800, "platform": {"python": "test", "platform": "test"},
    }
    trial.update(overrides)
    return trial


def _rows(count: int = 200) -> list[dict[str, object]]:
    return [
        {"session_id": f"bench_{session:04d}", "turn": turn, "profile": {}, "message": "hello"}
        for session in range(1, count + 1)
        for turn in range(1, 5)
    ]


def _full_trial(**overrides: object) -> dict[str, object]:
    return _trial(**overrides)


def _aggregate_full(**overrides: object) -> dict[str, object]:
    latency_ms = overrides.pop("latency_ms", None)
    trial = _full_trial(**overrides)
    if latency_ms is not None:
        trial["latencies_ms"] = [float(latency_ms)] * 800
    report = bench.aggregate_trials([trial])
    report["catalog_hash"] = trial["catalog_hash"]
    return report


def _capture_metadata() -> dict[str, object]:
    return {
        "catalog_hash": "c" * 64, "proxy_manifest_hash": "a" * 64,
        "representative_dataset_hash": "b" * 64, "agent_class": "test:Agent",
        "config_hash": "d" * 64, "dense_available": True, "dense_status": "available",
        "capture_seed": 20260829, "session_count": 200, "response_count": 800,
        "cwd_mode": "root", "platform": {"python": "test"},
    }


def test_aggregate_trials_uses_medians_latency_summary_and_matching_hashes() -> None:
    report = bench.aggregate_trials([
        _trial(init_ms=10, peak_mib=100, rss_mib=80, latencies_ms=[1] * 800),
        _trial(init_ms=20, peak_mib=120, rss_mib=90, latencies_ms=[2] * 800),
        _trial(init_ms=30, peak_mib=110, rss_mib=85, latencies_ms=[4] * 800),
    ])

    assert report["init_ms"] == 20
    assert report["peak_mib"] == 110
    assert report["latency_ms"]["p95"] == 4
    assert report["dense_available"] is True
    assert report["response_hash"] == "a" * 64
    with pytest.raises(ValueError):
        bench.aggregate_trials([_trial(), _trial(response_hash="d" * 64)])
    with pytest.raises(ValueError):
        bench.aggregate_trials([])


def test_validate_transcript_rejects_invalid_or_sensitive_rows() -> None:
    rows = _rows()
    bench.validate_transcript(rows)
    invalid = [
        rows[:-1],
        _rows(199),
        [*rows[:4], *rows[8:12], *rows[4:8], *rows[8:12], *rows[12:]],
        [{**row, "turn": True} if row["turn"] == 1 else row for row in rows],
        [{key: value for key, value in row.items() if key != "profile"} if row["turn"] == 1 else row for row in rows],
        [{**row, "profile": []} if row["turn"] == 1 else row for row in rows],
        [{**row, "profile": {"arbitrary": "value"}} if row["turn"] == 1 else row for row in rows],
        [{**row, "message": " "} if row["turn"] == 1 else row for row in rows],
        [{**row, "session_id": "target-leak"} if row["session_id"] == "bench_0001" else row for row in rows],
        [{**row, "target": "leak"} if row["turn"] == 1 else row for row in rows],
        [{**row, "profile": {"intent_card": "leak"}} if row["turn"] == 1 else row for row in rows],
        [{**row, "profile": {"summary": "ground_truth"}} if row["turn"] == 1 else row for row in rows],
    ]
    for candidate in invalid:
        with pytest.raises((TypeError, ValueError)):
            bench.validate_transcript(candidate)


def test_run_worker_rejects_profile_extras_before_constructing_agent(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    rows = _rows()
    rows[0]["profile"] = {"hit": "leak"}
    transcript.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    constructed = False

    class MustNotConstruct:
        def __init__(self, _: str) -> None:
            nonlocal constructed
            constructed = True

    with pytest.raises(ValueError):
        bench.run_worker(tmp_path / "catalog.jsonl", transcript, agent_class=MustNotConstruct)
    assert constructed is False


def test_compare_reports_enforces_hash_output_performance_and_safety_gates() -> None:
    baseline = bench.aggregate_trials([_trial(init_ms=100, peak_mib=100, latencies_ms=[100] * 800)])
    candidate = bench.aggregate_trials([_trial(init_ms=95, peak_mib=100, latencies_ms=[90] * 800)])
    result = bench.compare_reports(candidate, baseline)
    assert result["accepted"] is True
    assert result["material_gain"] is True
    assert bench.compare_reports(
        bench.aggregate_trials([_trial(init_ms=106, peak_mib=100, latencies_ms=[90] * 800)]), baseline
    )["accepted"] is False
    assert bench.compare_reports(
        bench.aggregate_trials([_trial(init_ms=96, peak_mib=100, latencies_ms=[100] * 800)]), baseline
    )["accepted"] is False
    with pytest.raises(ValueError):
        bench.compare_reports(candidate, {**baseline, "init_ms": 0})


def test_main_allow_dense_unavailable_relaxes_only_dense_comparison_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = bench.aggregate_trials([_trial(init_ms=100, peak_mib=100, latencies_ms=[100] * 800)])
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    accepted = bench.aggregate_trials([_trial(dense_available=False, init_ms=95, peak_mib=100, latencies_ms=[90] * 800)])
    monkeypatch.setattr(bench, "run_parent", lambda *_args, **_kwargs: accepted)

    allowed_output = tmp_path / "allowed.json"
    bench.main(["--transcript", "input.jsonl", "--output", str(allowed_output), "--compare", str(baseline_path), "--allow-dense-unavailable"])
    assert json.loads(allowed_output.read_text(encoding="utf-8"))["comparison"]["accepted"] is True

    denied_output = tmp_path / "denied.json"
    with pytest.raises(SystemExit):
        bench.main(["--transcript", "input.jsonl", "--output", str(denied_output), "--compare", str(baseline_path)])
    assert denied_output.exists()

    no_gain = bench.aggregate_trials([_trial(dense_available=False, init_ms=96, peak_mib=100, latencies_ms=[100] * 800)])
    monkeypatch.setattr(bench, "run_parent", lambda *_args, **_kwargs: no_gain)
    rejected_output = tmp_path / "rejected.json"
    with pytest.raises(SystemExit):
        bench.main(["--transcript", "input.jsonl", "--output", str(rejected_output), "--compare", str(baseline_path), "--allow-dense-unavailable"])
    assert json.loads(rejected_output.read_text(encoding="utf-8"))["comparison"]["accepted"] is False


class _CaptureAgent:
    def __init__(self, _: str) -> None:
        self.reset_calls: list[tuple[str, dict]] = []
        self.calls: list[tuple[str, str, int]] = []

    def reset(self, session_id: str, profile: dict) -> None:
        self.reset_calls.append((session_id, profile))

    def respond(self, session_id: str, message: str, turn: int, _: int) -> dict:
        self.calls.append((session_id, message, turn))
        return {"message": "ok", "ask_attribute": "color", "recommendations": []}


def _proxy_row(index: int) -> dict[str, object]:
    return {
        "sample_id": f"sample-{index}", "proxy_fold": 1,
        "user_profile": {"average_prior_rating": 4.0, "preference_tags": ["x"]},
        "dialogue_variant": 0, "category_bucket": "home", "scenario_type": "browsing",
        "intent_card": {"hard_constraints": [], "soft_preferences": []}, "behavior": {},
    }


def test_capture_transcript_writes_allowlisted_four_turn_sessions_exclusively(tmp_path: Path) -> None:
    output = tmp_path / "transcript.jsonl"
    (tmp_path / "catalog.jsonl").write_text("catalog\n", encoding="utf-8")
    agent = _CaptureAgent("")
    digest = bench.capture_transcript(
        tmp_path, tmp_path / "catalog.jsonl", output, agent_class=lambda _: agent,
        suite_loader=lambda *_: ({}, [_proxy_row(index) for index in range(250)]),
    )
    payload = output.read_bytes()
    assert digest == hashlib.sha256(payload).hexdigest()
    rows = [json.loads(line) for line in payload.splitlines()]
    assert len(rows) == 800
    assert len(agent.calls) == 800
    assert {frozenset(row) for row in rows} == {frozenset({"session_id", "turn", "profile", "message"})}
    assert all("target" not in json.dumps(row).lower() for row in rows)
    manifest = json.loads((tmp_path / "transcript.jsonl.manifest.json").read_text(encoding="utf-8"))
    assert manifest["transcript_hash"] == digest
    assert manifest["response_count"] == 800
    assert set(manifest) == {
        "schema_version", "transcript_hash", "catalog_hash", "proxy_manifest_hash",
        "representative_dataset_hash", "agent_class", "config_hash", "dense_available",
        "dense_status", "capture_seed", "session_count", "response_count", "cwd_mode",
        "platform", "created_at",
    }
    assert "message" not in manifest and "profile" not in manifest
    with pytest.raises(FileExistsError):
        bench.capture_transcript(tmp_path, tmp_path / "catalog.jsonl", output, agent_class=lambda _: agent,
                                 suite_loader=lambda *_: ({}, [_proxy_row(index) for index in range(250)]))


def test_capture_seals_environment_and_rolls_back_on_bundle_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "transcript.jsonl"
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("catalog", encoding="utf-8")
    agent = _CaptureAgent("")
    monkeypatch.setenv("COMPASSCART_DISABLE_DENSE", "1")
    original_cwd = Path.cwd()
    suite = SimpleNamespace(rows=[_proxy_row(index) for index in range(250)], manifest_hash="a" * 64, dataset_hash="b" * 64)
    monkeypatch.setattr(bench, "_publish_bundle", lambda *_args: (_ for _ in ()).throw(RuntimeError("publish failed")))
    with pytest.raises(RuntimeError, match="publish failed"):
        bench.capture_transcript(tmp_path, catalog, output, agent_class=lambda _: agent, suite_loader=lambda *_: suite)
    assert not output.exists()
    assert not Path(str(output) + ".manifest.json").exists()
    assert os.environ["COMPASSCART_DISABLE_DENSE"] == "1"
    assert Path.cwd() == original_cwd


def test_capture_existing_transcript_only_publishes_sidecar_on_exact_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = tmp_path / "existing.jsonl"
    transcript.write_bytes(b"same")
    monkeypatch.setattr(bench, "_capture_payload", lambda *_args, **_kwargs: (b"different", _capture_metadata()))
    with pytest.raises(ValueError, match="does not match"):
        bench.capture_transcript(tmp_path, tmp_path / "catalog.jsonl", transcript)
    assert not Path(str(transcript) + ".manifest.json").exists()

    monkeypatch.setattr(bench, "_capture_payload", lambda *_args, **_kwargs: (b"same", _capture_metadata()))
    assert bench.capture_transcript(tmp_path, tmp_path / "catalog.jsonl", transcript) == hashlib.sha256(b"same").hexdigest()
    assert json.loads(Path(str(transcript) + ".manifest.json").read_text(encoding="utf-8"))["transcript_hash"] == hashlib.sha256(b"same").hexdigest()


class _TraceSink:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []


class _WorkerAgent:
    def __init__(self, _: str) -> None:
        self.catalog = type("Catalog", (), {"valid_ids": {f"id-{index}" for index in range(50_000)}})()
        self.dense = type("Dense", (), {"available": True})()
        self.traces = _TraceSink()
        self.resets: list[tuple[str, dict]] = []

    def reset(self, session_id: str, profile: dict) -> None:
        self.resets.append((session_id, profile))

    def respond(self, session_id: str, _: str, turn: int, __: int) -> dict:
        self.traces.records.append({"session_id": session_id, "turn": turn, "elapsed_ms": 1.5, "fallbacks": []})
        return {"message": "ok", "recommendations": ["id-1"]}


def test_run_worker_replays_validated_transcript_and_rejects_bad_agent_contract(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    (tmp_path / "catalog.jsonl").write_text("{}\n", encoding="utf-8")
    result = bench.run_worker(tmp_path / "catalog.jsonl", transcript, agent_class=_WorkerAgent)
    assert result["response_count"] == 800
    assert result["fallback_count"] == 0
    assert result["dense_available"] is True
    assert len(result["response_hash"]) == 64

    class BadCatalog(_WorkerAgent):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            self.catalog.valid_ids = set()
    with pytest.raises(ValueError):
        bench.run_worker(tmp_path / "catalog.jsonl", transcript, agent_class=BadCatalog)


def test_run_worker_supports_trace_sink_snapshots(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    (tmp_path / "catalog.jsonl").write_text("{}\n", encoding="utf-8")

    class SnapshotSink:
        def __init__(self) -> None:
            self._records: list[dict[str, object]] = []

        @property
        def records(self) -> list[dict[str, object]]:
            return list(self._records)

    class SnapshotAgent(_WorkerAgent):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            self.traces = SnapshotSink()

        def respond(self, session_id: str, message: str, turn: int, top_k: int) -> dict:
            self.traces._records.append({"session_id": session_id, "turn": turn, "elapsed_ms": 1.5, "fallbacks": []})
            return {"message": "ok", "recommendations": ["id-1"]}

    assert bench.run_worker(tmp_path / "catalog.jsonl", transcript, agent_class=SnapshotAgent)["response_count"] == 800


def test_run_worker_uses_wall_clock_latency_not_agent_trace(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    clock_values = iter(value / 1_000 for value in range(0, 3_204, 2))
    result = bench.run_worker(
        tmp_path / "catalog.jsonl", transcript, agent_class=_WorkerAgent,
        clock=lambda: next(clock_values), catalog_hasher=lambda _: "c" * 64,
    )
    assert result["latencies_ms"] == [2.0] * 800
    assert result["trace_latencies_ms"] == [1.5] * 800
    assert result["catalog_hash"] == "c" * 64


def test_run_worker_times_only_agent_construction_and_rechecks_catalog(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    clock_values = iter([0.0, 1.0, *[float(index) for index in range(2, 1_603)]])
    hashes = iter(["c" * 64, "c" * 64])
    result = bench.run_worker(
        tmp_path / "catalog.jsonl", transcript, agent_class=_WorkerAgent,
        clock=lambda: next(clock_values), catalog_hasher=lambda _: next(hashes),
    )
    assert result["init_ms"] == 1000.0

    changing_hashes = iter(["c" * 64, "d" * 64])
    with pytest.raises(ValueError, match="catalog changed"):
        bench.run_worker(tmp_path / "catalog.jsonl", transcript, agent_class=_WorkerAgent, catalog_hasher=lambda _: next(changing_hashes))


def test_memory_uses_platform_high_water_units(monkeypatch: pytest.MonkeyPatch) -> None:
    process = type("Process", (), {"memory_info": lambda self: type("Info", (), {"rss": 8 * 1024 * 1024, "peak_wset": 32 * 1024 * 1024})()})()
    assert bench._memory_mib(process=process, os_name="nt") == (8.0, 32.0, "windows_peak_wset")

    resource = type("Resource", (), {"RUSAGE_SELF": 0, "getrusage": lambda self, _: type("Usage", (), {"ru_maxrss": 16 * 1024})()})()
    assert bench._memory_mib(process=process, os_name="posix", platform_name="linux", resource_module=resource) == (8.0, 16.0, "posix_ru_maxrss_kib")
    resource.getrusage = lambda _: type("Usage", (), {"ru_maxrss": 16 * 1024 * 1024})()
    assert bench._memory_mib(process=process, os_name="posix", platform_name="darwin", resource_module=resource) == (8.0, 16.0, "posix_ru_maxrss_bytes")


def test_aggregate_trials_rejects_inexact_schema_and_catalog_disagreement() -> None:
    valid = _full_trial()
    with pytest.raises(ValueError):
        bench.aggregate_trials([{**valid, "catalog_hash": "a" * 64}, {**valid, "catalog_hash": "b" * 64}])
    with pytest.raises(ValueError):
        bench.aggregate_trials([{**valid, "latencies_ms": valid["latencies_ms"][:-1]}])
    with pytest.raises(ValueError):
        bench.aggregate_trials([{**valid, "fallback_count": True}])
    with pytest.raises(ValueError):
        bench.aggregate_trials([{**valid, "platform": {"recommendation": "leak"}}])


def test_compare_reports_requires_catalog_provenance_and_exact_boundaries() -> None:
    baseline = _aggregate_full(init_ms=100, peak_mib=100, latency_ms=100)
    candidate = _aggregate_full(init_ms=95, peak_mib=105, latency_ms=105)
    assert bench.compare_reports(candidate, baseline)["accepted"] is True
    assert bench.compare_reports({**candidate, "catalog_hash": "d" * 64}, baseline)["accepted"] is False
    assert bench.compare_reports(_aggregate_full(init_ms=95, peak_mib=100, latency_ms=1500), baseline)["accepted"] is False
    with pytest.raises(ValueError):
        bench.compare_reports({**candidate, "fallback_count": True}, baseline)


def test_run_parent_rejects_trial_toctou_and_reports_bounded_child_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    catalog = tmp_path / "catalog.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    catalog.write_text("catalog", encoding="utf-8")
    digest = hashlib.sha256(transcript.read_bytes()).hexdigest()
    trial = _full_trial(transcript_hash=digest, catalog_hash=hashlib.sha256(catalog.read_bytes()).hexdigest())
    monkeypatch.setattr(bench.subprocess, "run", lambda *_args, **_kwargs: type("Done", (), {"stdout": json.dumps(trial), "stderr": "", "returncode": 0})())
    hashes = iter([trial["catalog_hash"], digest, trial["catalog_hash"], digest, "f" * 64, digest])
    monkeypatch.setattr(bench, "sha256_file", lambda _: next(hashes))
    with pytest.raises(ValueError, match="catalog changed"):
        bench.run_parent(catalog, transcript, trials=1)

    monkeypatch.setattr(bench, "sha256_file", lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    monkeypatch.setattr(bench.subprocess, "run", lambda *_args, **_kwargs: type("Done", (), {"stdout": "x" * 3000, "stderr": "failed", "returncode": 7})())
    with pytest.raises(RuntimeError, match="return code 7") as error:
        bench.run_parent(catalog, transcript, trials=1)
    assert len(str(error.value)) < 2_300


def test_run_parent_rejects_extra_stdout_and_timeout_with_bounded_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    catalog = tmp_path / "catalog.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    catalog.write_text("catalog", encoding="utf-8")
    trial = _full_trial(
        transcript_hash=hashlib.sha256(transcript.read_bytes()).hexdigest(),
        catalog_hash=hashlib.sha256(catalog.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(bench.subprocess, "run", lambda *_args, **_kwargs: type("Done", (), {"stdout": json.dumps(trial) + " noise", "stderr": "", "returncode": 0})())
    with pytest.raises(ValueError, match="extra output"):
        bench.run_parent(catalog, transcript, trials=1)

    def timeout(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired("worker", 12, output="o" * 3_000, stderr="e" * 3_000)

    monkeypatch.setattr(bench.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out after 12") as error:
        bench.run_parent(catalog, transcript, trials=1, worker_timeout_seconds=12)
    assert len(str(error.value)) < 4_200


def test_cli_rejects_incompatible_options_and_invalid_baseline_before_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit):
        bench.main(["--worker", "--transcript", "input.jsonl", "--compare", "baseline.json"])
    with pytest.raises(SystemExit):
        bench.main(["--transcript", "input.jsonl", "--output", str(tmp_path / "report.json"), "--proxy-root", "proxy"])

    baseline = tmp_path / "bad.json"
    baseline.write_text("{", encoding="utf-8")
    called = False

    def no_worker(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return _aggregate_full()

    monkeypatch.setattr(bench, "run_parent", no_worker)
    with pytest.raises(ValueError):
        bench.main(["--transcript", "input.jsonl", "--output", str(tmp_path / "bad-report.json"), "--compare", str(baseline)])
    assert called is False


def test_parent_preflight_rejects_historical_or_forged_aggregate_before_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False
    monkeypatch.setattr(bench, "run_parent", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("worker ran")))
    output = tmp_path / "report.json"
    historical = tmp_path / "historical.json"
    historical.write_text(json.dumps({"latency_ms": {"p95": 1}}), encoding="utf-8")
    with pytest.raises(ValueError):
        bench.main(["--transcript", "input.jsonl", "--output", str(output), "--compare", str(historical)])
    assert called is False

    baseline = _aggregate_full()
    baseline["init_ms"] = 999.0
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(baseline), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate"):
        bench.main(["--transcript", "input.jsonl", "--output", str(tmp_path / "forged-output.json"), "--compare", str(forged)])


def test_run_parent_builds_isolated_child_command_and_parses_single_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    (tmp_path / "catalog.jsonl").write_text("{}\n", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, str], Path]] = []
    worker = _trial()
    worker["transcript_hash"] = hashlib.sha256(transcript.read_bytes()).hexdigest()
    worker["catalog_hash"] = hashlib.sha256((tmp_path / "catalog.jsonl").read_bytes()).hexdigest()

    def fake_run(command: list[str], **kwargs: object):
        calls.append((command, kwargs["env"], kwargs["cwd"]))
        return type("Done", (), {"stdout": json.dumps(worker), "stderr": "", "returncode": 0})()

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "hostile-parent-path")
    monkeypatch.setenv("COMPASSCART_DISABLE_DENSE", "1")
    report = bench.run_parent(tmp_path / "catalog.jsonl", transcript, trials=2, cwd_mode="outside")
    assert report["trial_count"] == 2
    assert len(calls) == 2
    assert calls[0][0][:3] == [sys.executable, "-m", "tools.benchmark_release"]
    assert calls[0][1]["PYTHONPATH"] == os.pathsep.join([str(bench._repo_root() / "src"), str(bench._repo_root())])
    assert "COMPASSCART_DISABLE_DENSE" not in calls[0][1]
    assert calls[0][2] != bench._repo_root()


def test_report_writer_and_cli_reject_malformed_or_conflicting_inputs(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    bench.write_report(path, {"ok": True})
    with pytest.raises(FileExistsError):
        bench.write_report(path, {"ok": True})
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        bench.load_report(tmp_path / "bad.json")
    with pytest.raises(SystemExit):
        bench.parse_args(["--worker", "--capture-transcript"])
    with pytest.raises(SystemExit):
        bench.parse_args(["--verify-captured-transcript", "--transcript", "input.jsonl", "--proxy-root", "proxy"])
    assert bench.parse_args(["--transcript", "input.jsonl", "--output", "report.json"]).worker is False
