from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from tools import benchmark_release as bench


def _trial(**overrides: object) -> dict[str, object]:
    trial: dict[str, object] = {
        "init_ms": 20.0,
        "peak_mib": 110.0,
        "rss_mib": 85.0,
        "latencies_ms": [1.0, 2.0],
        "dense_available": True,
        "dense_status": "available",
        "fallback_count": 0,
        "response_hash": "r" * 64,
        "transcript_hash": "t" * 64,
        "response_count": 800,
    }
    trial.update(overrides)
    return trial


def _rows(count: int = 200) -> list[dict[str, object]]:
    return [
        {"session_id": f"bench_{session:04d}", "turn": turn, "profile": {}, "message": "hello"}
        for session in range(1, count + 1)
        for turn in range(1, 5)
    ]


def test_aggregate_trials_uses_medians_latency_summary_and_matching_hashes() -> None:
    report = bench.aggregate_trials([
        _trial(init_ms=10, peak_mib=100, rss_mib=80, latencies_ms=[1, 2]),
        _trial(init_ms=20, peak_mib=120, rss_mib=90, latencies_ms=[2, 3]),
        _trial(init_ms=30, peak_mib=110, rss_mib=85, latencies_ms=[3, 4]),
    ])

    assert report["init_ms"] == 20
    assert report["peak_mib"] == 110
    assert report["latency_ms"]["p95"] == 4
    assert report["dense_available"] is True
    assert report["response_hash"] == "r" * 64
    with pytest.raises(ValueError):
        bench.aggregate_trials([_trial(), _trial(response_hash="x" * 64)])
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
        [{**row, "message": " "} if row["turn"] == 1 else row for row in rows],
        [{**row, "session_id": "target-leak"} if row["session_id"] == "bench_0001" else row for row in rows],
        [{**row, "target": "leak"} if row["turn"] == 1 else row for row in rows],
        [{**row, "profile": {"intent_card": "leak"}} if row["turn"] == 1 else row for row in rows],
        [{**row, "profile": {"summary": "ground_truth"}} if row["turn"] == 1 else row for row in rows],
    ]
    for candidate in invalid:
        with pytest.raises((TypeError, ValueError)):
            bench.validate_transcript(candidate)


def test_compare_reports_enforces_hash_output_performance_and_safety_gates() -> None:
    baseline = bench.aggregate_trials([_trial(init_ms=100, peak_mib=100, latencies_ms=[100] * 20)])
    candidate = bench.aggregate_trials([_trial(init_ms=95, peak_mib=100, latencies_ms=[90] * 20)])
    result = bench.compare_reports(candidate, baseline)
    assert result["accepted"] is True
    assert result["material_gain"] is True
    assert bench.compare_reports(
        bench.aggregate_trials([_trial(init_ms=106, peak_mib=100, latencies_ms=[90] * 20)]), baseline
    )["accepted"] is False
    assert bench.compare_reports(
        bench.aggregate_trials([_trial(init_ms=96, peak_mib=100, latencies_ms=[100] * 20)]), baseline
    )["accepted"] is False
    with pytest.raises(ValueError):
        bench.compare_reports(candidate, {**baseline, "init_ms": 0})


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
    with pytest.raises(FileExistsError):
        bench.capture_transcript(tmp_path, tmp_path / "catalog.jsonl", output, agent_class=lambda _: agent,
                                 suite_loader=lambda *_: ({}, [_proxy_row(index) for index in range(250)]))


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


def test_run_parent_builds_isolated_child_command_and_parses_single_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    (tmp_path / "catalog.jsonl").write_text("{}\n", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, str], Path]] = []
    worker = _trial()
    worker["transcript_hash"] = hashlib.sha256(transcript.read_bytes()).hexdigest()

    def fake_run(command: list[str], **kwargs: object):
        calls.append((command, kwargs["env"], kwargs["cwd"]))
        return type("Done", (), {"stdout": json.dumps(worker), "stderr": "", "returncode": 0})()

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    report = bench.run_parent(tmp_path / "catalog.jsonl", transcript, trials=2, cwd_mode="outside")
    assert report["trial_count"] == 2
    assert len(calls) == 2
    assert calls[0][0][:3] == [sys.executable, "-m", "tools.benchmark_release"]
    assert str(bench._repo_root() / "src") in calls[0][1]["PYTHONPATH"]
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
    assert bench.parse_args(["--transcript", "input.jsonl", "--output", "report.json"]).worker is False
