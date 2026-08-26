from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest

from tools import benchmark_release as bench


def _trial(**overrides: object) -> dict[str, object]:
    trial: dict[str, object] = {
        "init_ms": 20.0, "peak_mib": 110.0, "rss_mib": 85.0,
        "peak_metric_source": "windows_peak_wset", "latencies_ms": [2.0] * 800,
        "trace_latencies_ms": [1.0] * 800, "instrumentation_delta_ms": {"count": 800, "p50": 1.0, "p95": 1.0, "max": 1.0},
        "dense_available": True, "dense_status": "available", "fallback_count": 0,
        "response_hash": "a" * 64, "transcript_hash": "b" * 64, "catalog_hash": "c" * 64,
        "catalog_snapshot_hash": "c" * 64,
        "capture_provenance": {
            "manifest_hash": "e" * 64, "proxy_manifest_hash": "a" * 64,
            "representative_dataset_hash": "b" * 64, "agent_class": "test:Agent", "config_hash": "d" * 64,
            "capture_seed": 20260829, "session_count": 200, "response_count": 800,
            "cwd_mode": "root", "dense_available": True, "dense_status": "available",
            "platform": {"python": "test", "platform": "test"},
        },
        "response_count": 800,
        "platform": {"os": "Windows", "python": "3.12", "processor": "x86_64", "onnxruntime": "1.29.0", "psutil": "7.2.2"},
    }
    trial.update(overrides)
    if "instrumentation_delta_ms" not in overrides:
        trial["instrumentation_delta_ms"] = bench._latency_summary([
            wall - trace for wall, trace in zip(trial["latencies_ms"], trial["trace_latencies_ms"], strict=True)
        ])
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
        trial["instrumentation_delta_ms"] = bench._latency_summary([
            wall - trace for wall, trace in zip(trial["latencies_ms"], trial["trace_latencies_ms"], strict=True)
        ])
    report = bench.aggregate_trials([trial])
    report["catalog_hash"] = trial["catalog_hash"]
    return report


def _parent_report(**overrides: object) -> dict[str, object]:
    report = _aggregate_full(**overrides)
    report.update({
        "cwd_mode": "root",
        "platform": {
            "os": "Windows", "python": "3.12", "processor": "x86_64",
            "cpu_logical": 8, "cpu_physical": 4, "ram_mib": 16384.0,
            "onnxruntime": "1.29.0", "psutil": "7.2.2",
        },
    })
    return report


def _capture_metadata() -> dict[str, object]:
    return {
        "catalog_hash": "c" * 64, "proxy_manifest_hash": "a" * 64,
        "representative_dataset_hash": "b" * 64, "agent_class": "test:Agent",
        "config_hash": "d" * 64, "dense_available": True, "dense_status": "available",
        "capture_seed": 20260829, "session_count": 200, "response_count": 800,
        "cwd_mode": "root", "platform": {"python": "test", "platform": "test"},
    }


def _write_manifest(transcript: Path, catalog: Path, **overrides: object) -> dict[str, object]:
    metadata = _capture_metadata()
    metadata["catalog_hash"] = hashlib.sha256(catalog.read_bytes()).hexdigest()
    metadata.update(overrides)
    manifest = bench._capture_manifest(
        metadata, transcript_hash=hashlib.sha256(transcript.read_bytes()).hexdigest(),
    )
    Path(f"{transcript}.manifest.json").write_bytes(bench._canonical_bytes(manifest) + b"\n")
    return manifest


def test_publish_bundle_removes_registered_temp_when_fsync_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "transcript.jsonl"
    monkeypatch.setattr(bench.os, "fsync", lambda _: (_ for _ in ()).throw(OSError("disk failed")))

    with pytest.raises(OSError, match="disk failed"):
        bench._publish_bundle([(destination, b"private transcript")])

    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_publish_bundle_removes_registered_temp_when_write_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "transcript.jsonl"

    class FailingWriter:
        def __init__(self, descriptor: int) -> None:
            self.descriptor = descriptor

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            os.close(self.descriptor)

        def write(self, _: bytes) -> int:
            raise OSError("write failed")

    monkeypatch.setattr(bench.os, "fdopen", lambda descriptor, *_args, **_kwargs: FailingWriter(descriptor))
    with pytest.raises(OSError, match="write failed"):
        bench._publish_bundle([(destination, b"private transcript")])

    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_publish_bundle_removes_registered_temp_when_flush_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "transcript.jsonl"

    class FailingFlusher:
        def __init__(self, descriptor: int) -> None:
            self.descriptor = descriptor

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            os.close(self.descriptor)

        def write(self, value: bytes) -> int:
            return len(value)

        def flush(self) -> None:
            raise OSError("flush failed")

    monkeypatch.setattr(bench.os, "fdopen", lambda descriptor, *_args, **_kwargs: FailingFlusher(descriptor))
    with pytest.raises(OSError, match="flush failed"):
        bench._publish_bundle([(destination, b"private transcript")])

    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_publish_bundle_preserves_explicit_manifest_first_and_competing_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    manifest = Path(f"{transcript}.manifest.json")
    real_link = os.link
    calls: list[Path] = []

    def racing_link(source: str | Path, destination: str | Path, *_: object, **__: object) -> None:
        target = Path(destination)
        calls.append(target)
        if target == transcript:
            raise OSError("commit failed")
        real_link(source, target)
        target.unlink()
        target.write_bytes(b"competitor")

    monkeypatch.setattr(bench.os, "link", racing_link)
    with pytest.raises(RuntimeError, match="exclusive publication"):
        bench._publish_bundle([(manifest, b"manifest"), (transcript, b"transcript")])

    assert calls[0] == manifest
    assert manifest.read_bytes() == b"competitor"
    assert not transcript.exists()


def test_publish_bundle_never_reorders_explicit_commit_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar = tmp_path / "sidecar.json"
    commit_marker = tmp_path / "frozen.manifest.json"
    real_link = os.link
    destinations: list[Path] = []

    def spy_link(source: str | Path, destination: str | Path, *args: object, **kwargs: object) -> None:
        destinations.append(Path(destination))
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(bench.os, "link", spy_link)
    bench._publish_bundle([(sidecar, b"sidecar"), (commit_marker, b"transcript")])

    assert destinations == [sidecar, commit_marker]


def test_publish_bundle_keeps_published_sidecar_when_commit_marker_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar = tmp_path / "transcript.jsonl.manifest.json"
    commit_marker = tmp_path / "transcript.jsonl"
    real_link = os.link

    def fail_commit_link(source: str | Path, destination: str | Path, *args: object, **kwargs: object) -> None:
        if Path(destination) == commit_marker:
            raise OSError("commit link failed")
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(bench.os, "link", fail_commit_link)
    with pytest.raises(RuntimeError, match="exclusive publication"):
        bench._publish_bundle([(sidecar, b"sidecar"), (commit_marker, b"transcript")])

    assert sidecar.read_bytes() == b"sidecar"
    assert not commit_marker.exists()


def test_publish_bundle_fstat_failure_closes_descriptor_and_removes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fstat = os.fstat
    real_close = os.close
    descriptors: list[int] = []

    def fail_fstat(descriptor: int) -> object:
        descriptors.append(descriptor)
        raise OSError("fstat failed")

    monkeypatch.setattr(bench.os, "fstat", fail_fstat)
    with pytest.raises(OSError, match="fstat failed"):
        bench._publish_bundle([(tmp_path / "report.json", b"report")])

    leaked_paths = list(tmp_path.iterdir())
    leaked_descriptors: list[int] = []
    for descriptor in descriptors:
        try:
            real_fstat(descriptor)
        except OSError:
            continue
        leaked_descriptors.append(descriptor)
        real_close(descriptor)
    for path in leaked_paths:
        path.unlink(missing_ok=True)
    assert leaked_descriptors == []
    assert leaked_paths == []


def test_preflight_destination_uses_private_probe_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "report.json"
    observed_parent: Path | None = None

    def reject_link(source: str | Path, probe: str | Path, *__: object, **___: object) -> None:
        nonlocal observed_parent
        source_path = Path(source)
        probe_path = Path(probe)
        assert source_path.parent == probe_path.parent
        assert source_path.parent != tmp_path
        observed_parent = source_path.parent
        raise OSError("hardlink unavailable")

    monkeypatch.setattr(bench.os, "link", reject_link)
    with pytest.raises(RuntimeError, match="hardlink support"):
        bench._preflight_destination(destination)

    assert observed_parent is not None
    assert not observed_parent.exists()
    assert not list(tmp_path.iterdir())


def test_preflight_fstat_failure_closes_descriptor_and_removes_private_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fstat = os.fstat
    real_close = os.close
    descriptors: list[int] = []

    def fail_fstat(descriptor: int) -> object:
        descriptors.append(descriptor)
        raise OSError("fstat failed")

    monkeypatch.setattr(bench.os, "fstat", fail_fstat)
    with pytest.raises(OSError, match="fstat failed"):
        bench._preflight_destination(tmp_path / "report.json")

    leaked_paths = list(tmp_path.iterdir())
    leaked_descriptors: list[int] = []
    for descriptor in descriptors:
        try:
            real_fstat(descriptor)
        except OSError:
            continue
        leaked_descriptors.append(descriptor)
        real_close(descriptor)
    for path in leaked_paths:
        path.unlink(missing_ok=True)
    assert leaked_descriptors == []
    assert leaked_paths == []


def test_run_worker_rejects_missing_capture_manifest_before_agent_construction(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    catalog = tmp_path / "catalog.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    catalog.write_text("catalog", encoding="utf-8")
    constructed = False

    class MustNotConstruct:
        def __init__(self, _: str) -> None:
            nonlocal constructed
            constructed = True

    with pytest.raises(ValueError, match="manifest"):
        bench.run_worker(catalog, transcript, agent_class=MustNotConstruct)
    assert constructed is False


def test_run_worker_attests_complete_capture_provenance_from_sidecar(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    catalog = tmp_path / "catalog.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    catalog.write_text("catalog", encoding="utf-8")
    manifest = _write_manifest(
        transcript, catalog, agent_class=f"{_WorkerAgent.__module__}:{_WorkerAgent.__qualname__}",
        config_hash=hashlib.sha256(repr(None).encode("utf-8")).hexdigest(),
    )

    result = bench.run_worker(catalog, transcript, agent_class=_WorkerAgent)

    assert result["capture_provenance"] == {
        "manifest_hash": hashlib.sha256(Path(f"{transcript}.manifest.json").read_bytes()).hexdigest(),
        "proxy_manifest_hash": manifest["proxy_manifest_hash"],
        "representative_dataset_hash": manifest["representative_dataset_hash"],
        "agent_class": manifest["agent_class"],
        "config_hash": manifest["config_hash"],
        "capture_seed": 20260829,
        "session_count": 200,
        "response_count": 800,
        "cwd_mode": "root",
        "dense_available": True,
        "dense_status": "available",
        "platform": {"python": "test", "platform": "test"},
    }


def test_run_worker_rejects_sidecar_agent_or_config_mismatch(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    catalog = tmp_path / "catalog.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    catalog.write_text("catalog", encoding="utf-8")
    _write_manifest(
        transcript, catalog, agent_class=f"{_WorkerAgent.__module__}:{_WorkerAgent.__qualname__}",
        config_hash=hashlib.sha256(repr(None).encode("utf-8")).hexdigest(),
    )
    assert bench.run_worker(catalog, transcript, agent_class=_WorkerAgent)["response_count"] == 800

    _write_manifest(
        transcript, catalog, agent_class="other:Agent", config_hash=hashlib.sha256(repr(None).encode("utf-8")).hexdigest(),
    )
    with pytest.raises(ValueError, match="agent provenance"):
        bench.run_worker(catalog, transcript, agent_class=_WorkerAgent)


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
        [{**row, "profile": {"ground_truth": "leak"}} if row["turn"] == 1 else row for row in rows],
    ]
    for candidate in invalid:
        with pytest.raises((TypeError, ValueError)):
            bench.validate_transcript(candidate)
    rows[0]["message"] = "Could you show more recommendations?"
    bench.validate_transcript(rows)


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
    baseline = _parent_report(init_ms=100, peak_mib=100, latencies_ms=[100] * 800)
    candidate = _parent_report(init_ms=95, peak_mib=100, latencies_ms=[90] * 800)
    result = bench.compare_reports(candidate, baseline)
    assert result["accepted"] is True
    assert result["material_gain"] is True
    assert bench.compare_reports(
        _parent_report(init_ms=106, peak_mib=100, latencies_ms=[90] * 800), baseline
    )["accepted"] is False
    assert bench.compare_reports(
        _parent_report(init_ms=96, peak_mib=100, latencies_ms=[100] * 800), baseline
    )["accepted"] is False
    with pytest.raises(ValueError):
        bench.compare_reports(candidate, {**baseline, "init_ms": 0})


def test_main_allow_dense_unavailable_relaxes_only_dense_comparison_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _parent_report(init_ms=100, peak_mib=100, latencies_ms=[100] * 800)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    accepted = _parent_report(dense_available=False, init_ms=95, peak_mib=100, latencies_ms=[90] * 800)
    monkeypatch.setattr(bench, "run_parent", lambda *_args, **_kwargs: accepted)

    allowed_output = tmp_path / "allowed.json"
    bench.main(["--transcript", "input.jsonl", "--output", str(allowed_output), "--compare", str(baseline_path), "--allow-dense-unavailable"])
    assert json.loads(allowed_output.read_text(encoding="utf-8"))["comparison"]["accepted"] is True

    denied_output = tmp_path / "denied.json"
    with pytest.raises(SystemExit):
        bench.main(["--transcript", "input.jsonl", "--output", str(denied_output), "--compare", str(baseline_path)])
    assert denied_output.exists()

    no_gain = _parent_report(dense_available=False, init_ms=96, peak_mib=100, latencies_ms=[100] * 800)
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


def test_sealed_capture_environment_restores_dense_flag_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "COMPASSCART_DISABLE_DENSE"
    monkeypatch.delenv(key, raising=False)
    with bench._sealed_capture_environment():
        os.environ[key] = "set-inside"
    assert key not in os.environ

    monkeypatch.setenv(key, "")
    with bench._sealed_capture_environment():
        os.environ[key] = "changed-inside"
    assert os.environ[key] == ""


def test_capture_agent_reads_private_catalog_snapshot_during_restored_source_swap(tmp_path: Path) -> None:
    source_catalog = tmp_path / "catalog.jsonl"
    source_catalog.write_bytes(b"catalog-a")
    source_hash = hashlib.sha256(b"catalog-a").hexdigest()
    agent_paths: list[Path] = []
    observed_catalogs: list[bytes] = []

    class SnapshotReadingAgent(_CaptureAgent):
        def __init__(self, catalog_path: str) -> None:
            super().__init__(catalog_path)
            self.catalog_path = Path(catalog_path)
            agent_paths.append(self.catalog_path)

        def respond(self, session_id: str, message: str, turn: int, top_k: int) -> dict:
            if not self.calls:
                source_catalog.write_bytes(b"catalog-b")
            observed_catalogs.append(self.catalog_path.read_bytes())
            response = super().respond(session_id, message, turn, top_k)
            if len(self.calls) == 800:
                source_catalog.write_bytes(b"catalog-a")
            return response

    output = tmp_path / "transcript.jsonl"
    bench.capture_transcript(
        tmp_path, source_catalog, output, agent_class=SnapshotReadingAgent,
        suite_loader=lambda *_: ({}, [_proxy_row(index) for index in range(250)]),
    )

    manifest = json.loads(Path(f"{output}.manifest.json").read_text(encoding="utf-8"))
    assert agent_paths[0] != source_catalog.resolve()
    assert agent_paths[0].name == "catalog.snapshot.jsonl"
    assert not agent_paths[0].exists()
    assert observed_catalogs == [b"catalog-a"] * 800
    assert source_catalog.read_bytes() == b"catalog-a"
    assert manifest["catalog_hash"] == source_hash


def test_capture_rejects_source_catalog_left_changed(tmp_path: Path) -> None:
    source_catalog = tmp_path / "catalog.jsonl"
    source_catalog.write_bytes(b"catalog-a")

    class SourceMutatingAgent(_CaptureAgent):
        def respond(self, session_id: str, message: str, turn: int, top_k: int) -> dict:
            if not self.calls:
                source_catalog.write_bytes(b"catalog-b")
            return super().respond(session_id, message, turn, top_k)

    with pytest.raises(ValueError, match="catalog changed during transcript capture"):
        bench._capture_payload(
            tmp_path, source_catalog, agent_class=SourceMutatingAgent,
            suite_loader=lambda *_: ({}, [_proxy_row(index) for index in range(250)]),
        )


def test_catalog_snapshot_rejects_source_change_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_catalog = tmp_path / "catalog.jsonl"
    source_catalog.write_bytes(b"catalog-a")
    snapshot_directory = tmp_path / "snapshot"
    snapshot_directory.mkdir()
    expected_hash = hashlib.sha256(b"catalog-a").hexdigest()
    real_copy = bench.shutil.copyfileobj

    def mutate_during_copy(source: object, target: object) -> None:
        real_copy(source, target)
        source_catalog.write_bytes(b"catalog-b")

    monkeypatch.setattr(bench.shutil, "copyfileobj", mutate_during_copy)
    with pytest.raises(ValueError, match="catalog changed while creating snapshot"):
        bench._copy_catalog_snapshot(source_catalog, snapshot_directory, expected_hash)


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


def test_capture_publishes_sidecar_before_special_named_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = tmp_path / "frozen.manifest.json"
    sidecar = Path(f"{transcript}.manifest.json")
    real_link = os.link
    destinations: list[Path] = []

    def spy_link(source: str | Path, destination: str | Path, *args: object, **kwargs: object) -> None:
        destinations.append(Path(destination))
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(bench, "_capture_payload", lambda *_args, **_kwargs: (b"transcript", _capture_metadata()))
    monkeypatch.setattr(bench.os, "link", spy_link)
    bench.capture_transcript(tmp_path, tmp_path / "catalog.jsonl", transcript)

    assert destinations == [sidecar, transcript]


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
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("{}\n", encoding="utf-8")
    _write_manifest(transcript, catalog, agent_class=f"{_WorkerAgent.__module__}:{_WorkerAgent.__qualname__}",
                    config_hash=hashlib.sha256(repr(None).encode("utf-8")).hexdigest())
    result = bench.run_worker(catalog, transcript, agent_class=_WorkerAgent)
    assert result["response_count"] == 800
    assert result["fallback_count"] == 0
    assert result["dense_available"] is True
    assert len(result["response_hash"]) == 64

    class BadCatalog(_WorkerAgent):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            self.catalog.valid_ids = set()
    with pytest.raises(ValueError):
        bench.run_worker(catalog, transcript, agent_class=BadCatalog)


def test_run_worker_supports_trace_sink_snapshots(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("{}\n", encoding="utf-8")

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

    _write_manifest(transcript, catalog, agent_class=f"{SnapshotAgent.__module__}:{SnapshotAgent.__qualname__}",
                    config_hash=hashlib.sha256(repr(None).encode("utf-8")).hexdigest())
    assert bench.run_worker(catalog, transcript, agent_class=SnapshotAgent)["response_count"] == 800


def test_run_worker_uses_wall_clock_latency_not_agent_trace(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("{}\n", encoding="utf-8")
    _write_manifest(transcript, catalog, catalog_hash="c" * 64,
                    agent_class=f"{_WorkerAgent.__module__}:{_WorkerAgent.__qualname__}",
                    config_hash=hashlib.sha256(repr(None).encode("utf-8")).hexdigest())
    clock_values = iter(value / 1_000 for value in range(0, 3_204, 2))
    result = bench.run_worker(
        catalog, transcript, agent_class=_WorkerAgent,
        clock=lambda: next(clock_values), catalog_hasher=lambda _: "c" * 64,
    )
    assert result["latencies_ms"] == [2.0] * 800
    assert result["trace_latencies_ms"] == [1.5] * 800
    assert result["catalog_hash"] == "c" * 64


def test_run_worker_times_only_agent_construction_and_rechecks_catalog(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("{}\n", encoding="utf-8")
    _write_manifest(transcript, catalog, catalog_hash="c" * 64,
                    agent_class=f"{_WorkerAgent.__module__}:{_WorkerAgent.__qualname__}",
                    config_hash=hashlib.sha256(repr(None).encode("utf-8")).hexdigest())
    clock_values = iter([0.0, 1.0, *[float(index) for index in range(2, 1_603)]])
    hashes = iter(["c" * 64, "c" * 64])
    result = bench.run_worker(
        catalog, transcript, agent_class=_WorkerAgent,
        clock=lambda: next(clock_values), catalog_hasher=lambda _: next(hashes),
    )
    assert result["init_ms"] == 1000.0

    changing_hashes = iter(["c" * 64, "d" * 64])
    with pytest.raises(ValueError, match="catalog changed"):
        bench.run_worker(catalog, transcript, agent_class=_WorkerAgent, catalog_hasher=lambda _: next(changing_hashes))


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
    different_platform = {**valid, "platform": {**valid["platform"], "python": "3.13"}}
    assert bench._validate_trial(different_platform) == different_platform
    with pytest.raises(ValueError, match="trials disagree"):
        bench.aggregate_trials([valid, different_platform])
    with pytest.raises(ValueError, match="instrumentation"):
        bench.aggregate_trials([{**valid, "instrumentation_delta_ms": {"count": 800, "p50": 0, "p95": 0, "max": 0}}])


def test_compare_reports_requires_catalog_provenance_and_exact_boundaries() -> None:
    baseline = _parent_report(init_ms=100, peak_mib=100, latency_ms=100)
    candidate = _parent_report(init_ms=95, peak_mib=105, latency_ms=105)
    assert bench.compare_reports(candidate, baseline)["accepted"] is True
    with pytest.raises(ValueError, match="aggregate"):
        bench.compare_reports({**candidate, "catalog_hash": "d" * 64}, baseline)
    assert bench.compare_reports(_parent_report(init_ms=95, peak_mib=100, latency_ms=1500), baseline)["accepted"] is False
    with pytest.raises(ValueError):
        bench.compare_reports({**candidate, "fallback_count": True}, baseline)


def test_compare_reports_returns_structured_compatibility_mismatches() -> None:
    baseline = _parent_report(init_ms=100, peak_mib=100, latency_ms=100)
    compatibility_flags = {
        "same_cwd_mode", "same_runtime_platform", "same_parent_platform",
        "same_peak_metric_source", "same_capture_provenance", "same_catalog",
        "same_catalog_snapshot", "same_transcript", "same_output", "same_trial_count",
    }

    mismatches: list[tuple[dict[str, object], set[str], list[str]]] = []
    cwd_candidate = _parent_report(init_ms=95, peak_mib=100, latency_ms=90)
    cwd_candidate["cwd_mode"] = "outside"
    mismatches.append((cwd_candidate, {"same_cwd_mode"}, ["cwd_mode_mismatch"]))

    runtime_platform = {
        "os": "Linux", "python": "3.12", "processor": "x86_64",
        "onnxruntime": "1.29.0", "psutil": "7.2.2",
    }
    runtime_candidate = _parent_report(
        init_ms=95, peak_mib=100, latency_ms=90, platform=runtime_platform,
    )
    runtime_candidate["platform"] = {**runtime_candidate["platform"], **runtime_platform}
    mismatches.append((runtime_candidate, {"same_runtime_platform"}, ["runtime_platform_mismatch"]))

    parent_candidate = _parent_report(init_ms=95, peak_mib=100, latency_ms=90)
    parent_candidate["platform"] = {**parent_candidate["platform"], "cpu_logical": 16}
    mismatches.append((parent_candidate, {"same_parent_platform"}, ["parent_platform_mismatch"]))
    mismatches.extend([
        (
            _parent_report(
                init_ms=95, peak_mib=100, latency_ms=90,
                peak_metric_source="posix_ru_maxrss_kib",
            ),
            {"same_peak_metric_source"},
            ["peak_metric_source_mismatch"],
        ),
        (
            _parent_report(
                init_ms=95, peak_mib=100, latency_ms=90,
                capture_provenance={
                    **_trial()["capture_provenance"], "config_hash": "f" * 64,
                },
            ),
            {"same_capture_provenance"},
            ["capture_provenance_mismatch"],
        ),
        (
            _parent_report(
                init_ms=95, peak_mib=100, latency_ms=90,
                catalog_hash="d" * 64, catalog_snapshot_hash="d" * 64,
            ),
            {"same_catalog", "same_catalog_snapshot"},
            ["catalog_hash_mismatch", "catalog_snapshot_hash_mismatch"],
        ),
        (
            _parent_report(init_ms=95, peak_mib=100, latency_ms=90, transcript_hash="d" * 64),
            {"same_transcript"},
            ["transcript_hash_mismatch"],
        ),
        (
            _parent_report(init_ms=95, peak_mib=100, latency_ms=90, response_hash="d" * 64),
            {"same_output"},
            ["response_hash_mismatch"],
        ),
    ])
    trial_count_candidate = _parent_report(init_ms=95, peak_mib=100, latency_ms=90)
    trial_count_candidate["trials"] = trial_count_candidate["trials"] * 2
    trial_count_candidate.update(bench.aggregate_trials(trial_count_candidate["trials"]))
    mismatches.append((trial_count_candidate, {"same_trial_count"}, ["trial_count_mismatch"]))

    for candidate, false_flags, reasons in mismatches:
        result = bench.compare_reports(candidate, baseline)
        assert result["accepted"] is False
        assert result["compatible"] is False
        assert all(result[flag] is False for flag in false_flags)
        assert result["reasons"] == reasons
        assert all(result[flag] is True for flag in compatibility_flags - false_flags)


def test_main_writes_rejected_comparison_before_exiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _parent_report(init_ms=100, peak_mib=100, latency_ms=100)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate = _parent_report(init_ms=95, peak_mib=100, latency_ms=90)
    candidate["cwd_mode"] = "outside"
    worker_called = False

    def run_parent(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal worker_called
        worker_called = True
        return candidate

    monkeypatch.setattr(bench, "run_parent", run_parent)
    output = tmp_path / "rejected.json"

    with pytest.raises(SystemExit, match="benchmark gate failed"):
        bench.main([
            "--transcript", "input.jsonl", "--output", str(output),
            "--compare", str(baseline_path),
        ])

    assert worker_called is True
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["comparison"]["accepted"] is False
    assert written["comparison"]["same_cwd_mode"] is False
    assert written["comparison"]["reasons"] == ["cwd_mode_mismatch"]
    assert bench._validate_aggregate_report(written) == written


def test_diagnostic_excerpt_escapes_all_control_sequences_and_is_bounded() -> None:
    excerpt = bench._diagnostic_excerpt("a" * 3_000 + "\x1b[31mfail\x07\x85\x9b")
    assert len(excerpt) <= 2_000
    assert all(ord(character) >= 32 and not 127 <= ord(character) <= 159 for character in excerpt)


def test_run_parent_rejects_trial_toctou_and_reports_bounded_child_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    catalog = tmp_path / "catalog.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    catalog.write_text("catalog", encoding="utf-8")
    manifest = _write_manifest(transcript, catalog)
    digest = hashlib.sha256(transcript.read_bytes()).hexdigest()
    catalog_hash = hashlib.sha256(catalog.read_bytes()).hexdigest()
    trial = _full_trial(transcript_hash=digest, catalog_hash=catalog_hash, catalog_snapshot_hash=catalog_hash,
                        capture_provenance=bench._capture_provenance(manifest, hashlib.sha256(Path(f"{transcript}.manifest.json").read_bytes()).hexdigest()))
    monkeypatch.setattr(bench.subprocess, "run", lambda *_args, **_kwargs: type("Done", (), {"stdout": json.dumps(trial), "stderr": "", "returncode": 0})())
    hashes = iter([catalog_hash, digest, catalog_hash, catalog_hash, catalog_hash, "f" * 64])
    monkeypatch.setattr(bench, "sha256_file", lambda _: next(hashes))
    with pytest.raises(ValueError, match="catalog changed"):
        bench.run_parent(catalog, transcript, trials=1)

    monkeypatch.setattr(bench, "sha256_file", lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    monkeypatch.setattr(bench.subprocess, "run", lambda *_args, **_kwargs: type("Done", (), {"stdout": "x" * 3000, "stderr": "failed", "returncode": 7})())
    with pytest.raises(RuntimeError, match="return code 7") as error:
        bench.run_parent(catalog, transcript, trials=1)
    assert len(str(error.value)) < 2_300


def test_run_parent_requires_capture_manifest_before_spawning_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = tmp_path / "transcript.jsonl"
    catalog = tmp_path / "catalog.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    catalog.write_text("catalog", encoding="utf-8")
    monkeypatch.setattr(bench.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("worker spawned"))

    with pytest.raises(ValueError, match="manifest"):
        bench.run_parent(catalog, transcript, trials=1)


def test_run_parent_uses_one_private_catalog_snapshot_for_all_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = tmp_path / "transcript.jsonl"
    catalog = tmp_path / "catalog.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    catalog.write_text("catalog-v1", encoding="utf-8")
    manifest = _write_manifest(transcript, catalog)
    manifest_hash = hashlib.sha256(Path(f"{transcript}.manifest.json").read_bytes()).hexdigest()
    catalog_hash = hashlib.sha256(catalog.read_bytes()).hexdigest()
    worker = _full_trial(
        transcript_hash=hashlib.sha256(transcript.read_bytes()).hexdigest(), catalog_hash=catalog_hash,
        catalog_snapshot_hash=catalog_hash, capture_provenance=bench._capture_provenance(manifest, manifest_hash),
    )
    seen_snapshots: list[Path] = []

    def fake_run(command: list[str], **_kwargs: object) -> object:
        snapshot = Path(command[command.index("--catalog") + 1])
        seen_snapshots.append(snapshot)
        assert snapshot != catalog.resolve()
        assert snapshot.read_bytes() == b"catalog-v1"
        return type("Done", (), {"stdout": json.dumps(worker), "stderr": "", "returncode": 0})()

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    monkeypatch.setattr(bench, "_platform_data", lambda: _parent_report()["platform"])
    report = bench.run_parent(catalog, transcript, trials=2)

    assert report["catalog_hash"] == catalog_hash
    assert report["catalog_snapshot_hash"] == catalog_hash
    assert len({path for path in seen_snapshots}) == 1


def test_run_parent_rejects_worker_runtime_platform_that_disagrees_with_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = tmp_path / "transcript.jsonl"
    catalog = tmp_path / "catalog.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    catalog.write_text("catalog", encoding="utf-8")
    manifest = _write_manifest(transcript, catalog)
    catalog_hash = hashlib.sha256(catalog.read_bytes()).hexdigest()
    trial = _full_trial(
        transcript_hash=hashlib.sha256(transcript.read_bytes()).hexdigest(), catalog_hash=catalog_hash,
        catalog_snapshot_hash=catalog_hash, capture_provenance=bench._capture_provenance(
            manifest, hashlib.sha256(Path(f"{transcript}.manifest.json").read_bytes()).hexdigest(),
        ),
    )
    monkeypatch.setattr(bench.subprocess, "run", lambda *_args, **_kwargs: type("Done", (), {"stdout": json.dumps(trial), "stderr": "", "returncode": 0})())
    monkeypatch.setattr(bench, "_platform_data", lambda: {**_parent_report()["platform"], "os": "Linux"})

    with pytest.raises(ValueError, match="runtime platform"):
        bench.run_parent(catalog, transcript, trials=1)


def test_run_parent_rejects_extra_stdout_and_timeout_with_bounded_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    catalog = tmp_path / "catalog.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    catalog.write_text("catalog", encoding="utf-8")
    _write_manifest(transcript, catalog)
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

    baseline = _parent_report()
    baseline["init_ms"] = 999.0
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(baseline), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate"):
        bench.main(["--transcript", "input.jsonl", "--output", str(tmp_path / "forged-output.json"), "--compare", str(forged)])


@pytest.mark.parametrize(
    "field,value",
    [
        ("os", 3), ("python", 3), ("processor", 3), ("cpu_logical", "bad"),
        ("cpu_physical", "bad"), ("ram_mib", "bad"), ("ram_mib", float("nan")),
        ("onnxruntime", 123), ("psutil", 123),
    ],
)
def test_parent_preflight_requires_complete_typed_parent_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    report = _parent_report()
    report["platform"][field] = value
    path = tmp_path / f"bad-{field}.json"
    path.write_text(json.dumps(report, allow_nan=True), encoding="utf-8")
    monkeypatch.setattr(bench, "run_parent", lambda *_args, **_kwargs: pytest.fail("worker ran"))
    with pytest.raises((TypeError, ValueError)):
        bench.main(["--transcript", "input.jsonl", "--output", str(tmp_path / f"out-{field}.json"), "--compare", str(path)])


def test_parent_preflight_rejects_missing_parent_schema_and_validates_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    aggregate_only = _aggregate_full()
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate_only), encoding="utf-8")
    monkeypatch.setattr(bench, "run_parent", lambda *_args, **_kwargs: pytest.fail("worker ran"))
    with pytest.raises(ValueError, match="schema"):
        bench.main(["--transcript", "input.jsonl", "--output", str(tmp_path / "out.json"), "--compare", str(aggregate_path)])

    report = _parent_report()
    report["comparison"] = {"accepted": True}
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="comparison"):
        bench.main(["--transcript", "input.jsonl", "--output", str(tmp_path / "comparison-out.json"), "--compare", str(comparison_path)])


def test_run_parent_builds_isolated_child_command_and_parses_single_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("{}\n", encoding="utf-8")
    manifest = _write_manifest(transcript, catalog)
    calls: list[tuple[list[str], dict[str, str], Path]] = []
    worker = _trial()
    worker["transcript_hash"] = hashlib.sha256(transcript.read_bytes()).hexdigest()
    worker["catalog_hash"] = hashlib.sha256(catalog.read_bytes()).hexdigest()
    worker["catalog_snapshot_hash"] = worker["catalog_hash"]
    worker["capture_provenance"] = bench._capture_provenance(manifest, hashlib.sha256(Path(f"{transcript}.manifest.json").read_bytes()).hexdigest())

    def fake_run(command: list[str], **kwargs: object):
        calls.append((command, kwargs["env"], kwargs["cwd"]))
        return type("Done", (), {"stdout": json.dumps(worker), "stderr": "", "returncode": 0})()

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    monkeypatch.setattr(bench, "_platform_data", lambda: _parent_report()["platform"])
    monkeypatch.setenv("PYTHONPATH", "hostile-parent-path")
    monkeypatch.setenv("COMPASSCART_DISABLE_DENSE", "1")
    report = bench.run_parent(catalog, transcript, trials=2, cwd_mode="outside")
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
