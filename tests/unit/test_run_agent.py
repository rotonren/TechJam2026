from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import run_agent

MAX_TURNS = 10
EVIDENCE_KEYS = {
    "attempted_response_count",
    "trace_count",
    "trace_call_count_consistent",
    "fallback_count",
    "fallback_counts",
}


def _result() -> dict[str, Any]:
    return {
        "sample_count": 2,
        "hit_rate_at_10": 0.5,
        "mrr": 0.5,
        "mttc": 6.5,
        "efficiency": 0.45,
        "recommended_technical_score": 0.49,
        "reported_token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "scenario_metrics": {
            "buying": {
                "sample_count": 1,
                "hit_rate_at_10": 1.0,
                "mrr": 1.0,
                "mttc": 2.0,
            },
            "intent_override": {
                "sample_count": 1,
                "hit_rate_at_10": 0.0,
                "mrr": 0.0,
                "mttc": 11.0,
            },
        },
        "sessions": [
            {
                "sample_id": "private-sample-one",
                "scenario_type": "buying",
                "hit": True,
                "first_hit_turn": 2,
                "best_rank": 1,
                "reciprocal_rank": 1.0,
            },
            {
                "sample_id": "private-sample-two",
                "scenario_type": "intent_override",
                "hit": False,
                "first_hit_turn": None,
                "best_rank": None,
                "reciprocal_rank": 0.0,
            },
        ],
    }


def _valid_traces(result: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, session in enumerate(result["sessions"]):
        turns = session["first_hit_turn"] or MAX_TURNS
        records.extend(
            {
                "session_id": f"opaque-session-{index}",
                "turn": turn,
                "fallbacks": [],
            }
            for turn in range(1, turns + 1)
        )
    return records


def _arguments(catalog: Path, dataset: Path, raw: Path, evidence: Path) -> list[str]:
    return [
        "--catalog",
        str(catalog),
        "--dataset",
        str(dataset),
        "--output",
        str(raw),
        "--evidence-output",
        str(evidence),
    ]


def _invoke(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["run_agent", *arguments])
    run_agent.main()


def _install_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: dict[str, Any] | None = None,
    traces: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    official_result = _result() if result is None else result
    official_traces = _valid_traces(official_result) if traces is None else traces
    calls: dict[str, Any] = {
        "agent": 0,
        "evaluate": 0,
        "load_jsonl": 0,
        "catalog_index": 0,
        "agent_instances": [],
    }

    class FakeAgent:
        def __init__(self, catalog: str) -> None:
            calls["agent"] += 1
            calls["agent_catalog"] = catalog
            calls["agent_instances"].append(self)
            self.traces = SimpleNamespace(records=deepcopy(official_traces))

    samples = [{"sample_id": "loaded-public-row"}]
    catalog_data = ({"catalog-id"}, {"catalog-id": ["category"]}, {"catalog-id": {}})

    def fake_load_jsonl(dataset: str) -> list[dict[str, str]]:
        calls["load_jsonl"] += 1
        calls["dataset"] = dataset
        return samples

    def fake_catalog_index(
        catalog: str,
    ) -> tuple[set[str], dict[str, list[str]], dict[str, dict]]:
        calls["catalog_index"] += 1
        calls["indexed_catalog"] = catalog
        return catalog_data

    def fake_evaluate(*args: object) -> dict[str, Any]:
        calls["evaluate"] += 1
        calls["evaluate_args"] = args
        return official_result

    monkeypatch.setattr(run_agent, "Agent", FakeAgent)
    monkeypatch.setattr(run_agent, "load_jsonl", fake_load_jsonl)
    monkeypatch.setattr(run_agent, "catalog_index", fake_catalog_index)
    monkeypatch.setattr(run_agent, "evaluate", fake_evaluate)
    return calls


def _assert_no_evaluation(calls: dict[str, Any]) -> None:
    assert calls["agent"] == 0
    assert calls["evaluate"] == 0
    assert calls["load_jsonl"] == 0
    assert calls["catalog_index"] == 0


def _assert_no_temporary_entries(
    *directories: Path, allowed: set[Path] | None = None
) -> None:
    allowed = set() if allowed is None else {path.resolve() for path in allowed}
    for directory in directories:
        if directory.exists():
            assert {path.resolve() for path in directory.iterdir()} == {
                path for path in allowed if path.parent == directory.resolve()
            }


@pytest.mark.parametrize(
    "missing", ["--catalog", "--dataset", "--output", "--evidence-output"]
)
def test_cli_requires_every_input_and_output_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    calls = _install_evaluator(monkeypatch)
    arguments = _arguments(
        tmp_path / "catalog.jsonl",
        tmp_path / "dataset.jsonl",
        tmp_path / "raw.json",
        tmp_path / "evidence.json",
    )
    position = arguments.index(missing)
    del arguments[position : position + 2]

    with pytest.raises(SystemExit) as raised:
        _invoke(monkeypatch, arguments)

    assert raised.value.code == 2
    _assert_no_evaluation(calls)


def test_main_preserves_official_result_and_publishes_aggregate_evidence_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _result()
    original_result = deepcopy(result)
    calls = _install_evaluator(monkeypatch, result=result)
    raw = tmp_path / "published" / "raw.json"
    evidence = tmp_path / "published" / "evidence.json"
    real_link = os.link
    linked_destinations: list[Path] = []

    def spy_link(
        source: str | Path, destination: str | Path, *args: object, **kwargs: object
    ) -> None:
        linked_destinations.append(Path(destination).resolve())
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", spy_link)
    _invoke(
        monkeypatch,
        _arguments(
            tmp_path / "catalog.jsonl", tmp_path / "dataset.jsonl", raw, evidence
        ),
    )

    assert calls["agent"] == 1
    assert calls["evaluate"] == 1
    assert calls["agent_catalog"] == str(tmp_path / "catalog.jsonl")
    assert calls["dataset"] == str(tmp_path / "dataset.jsonl")
    assert calls["indexed_catalog"] == str(tmp_path / "catalog.jsonl")
    assert calls["evaluate_args"] == (
        calls["agent_instances"][0],
        [{"sample_id": "loaded-public-row"}],
        {"catalog-id"},
        {"catalog-id": ["category"]},
        {"catalog-id": {}},
    )
    assert result == original_result
    assert raw.read_bytes() == (json.dumps(original_result, indent=2) + "\n").encode(
        "utf-8"
    )
    assert json.loads(raw.read_text(encoding="utf-8")) == original_result

    evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert set(evidence_payload) == EVIDENCE_KEYS
    assert evidence_payload == {
        "attempted_response_count": 12,
        "trace_count": 12,
        "trace_call_count_consistent": True,
        "fallback_count": 0,
        "fallback_counts": {},
    }
    evidence_text = evidence.read_text(encoding="utf-8")
    for forbidden in (
        "sessions",
        "sample_id",
        "private-sample-one",
        "private-sample-two",
        "recommendations",
        "targets",
        "intent",
    ):
        assert forbidden not in evidence_text

    assert linked_destinations[-2:] == [evidence.resolve(), raw.resolve()]
    assert (
        len(linked_destinations) == 4
    )  # Two unique-parent probes, then two publications.
    expected_aggregate = {
        key: value for key, value in original_result.items() if key != "sessions"
    }
    assert capsys.readouterr().out == json.dumps(expected_aggregate, indent=2) + "\n"
    _assert_no_temporary_entries(raw.parent, allowed={raw, evidence})


def test_preflight_proves_hardlinks_in_each_unique_parent_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw-parent" / "raw.json"
    evidence = tmp_path / "evidence-parent" / "evidence.json"
    real_link = os.link
    probed_parents: set[Path] = set()

    def spy_link(
        source: str | Path, destination: str | Path, *args: object, **kwargs: object
    ) -> None:
        target = Path(destination).resolve()
        if target not in {raw.resolve(), evidence.resolve()}:
            for parent in (raw.parent.resolve(), evidence.parent.resolve()):
                if parent == target.parent or parent in target.parents:
                    probed_parents.add(parent)
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", spy_link)
    calls = _install_evaluator(monkeypatch)
    original_evaluate = run_agent.evaluate

    def assert_preflight_then_evaluate(*args: object) -> dict[str, Any]:
        assert probed_parents == {raw.parent.resolve(), evidence.parent.resolve()}
        return original_evaluate(*args)

    monkeypatch.setattr(run_agent, "evaluate", assert_preflight_then_evaluate)
    _invoke(
        monkeypatch,
        _arguments(
            tmp_path / "catalog.jsonl", tmp_path / "dataset.jsonl", raw, evidence
        ),
    )

    assert calls["evaluate"] == 1


def test_resolved_duplicate_outputs_are_rejected_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_evaluator(monkeypatch)
    raw = tmp_path / "nested" / ".." / "score.json"
    evidence = tmp_path / "score.json"

    with pytest.raises(ValueError, match="distinct"):
        _invoke(
            monkeypatch,
            _arguments(
                tmp_path / "catalog.jsonl", tmp_path / "dataset.jsonl", raw, evidence
            ),
        )

    _assert_no_evaluation(calls)
    assert not os.path.lexists(evidence)


@pytest.mark.parametrize("existing_name", ["raw", "evidence"])
def test_existing_output_is_rejected_without_touching_its_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_name: str,
) -> None:
    calls = _install_evaluator(monkeypatch)
    raw = tmp_path / "raw.json"
    evidence = tmp_path / "evidence.json"
    existing = raw if existing_name == "raw" else evidence
    other = evidence if existing_name == "raw" else raw
    existing.write_bytes(b"pre-existing bytes")

    with pytest.raises(FileExistsError, match="exists"):
        _invoke(
            monkeypatch,
            _arguments(
                tmp_path / "catalog.jsonl", tmp_path / "dataset.jsonl", raw, evidence
            ),
        )

    _assert_no_evaluation(calls)
    assert existing.read_bytes() == b"pre-existing bytes"
    assert not os.path.lexists(other)


def test_broken_output_symlink_is_rejected_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw.json"
    missing_target = tmp_path / "missing-target.json"
    simulated = False
    try:
        raw.symlink_to(missing_target)
    except OSError:
        simulated = True
        real_lexists = os.path.lexists
        monkeypatch.setattr(
            os.path,
            "lexists",
            lambda path: Path(path) == raw or real_lexists(path),
        )
    calls = _install_evaluator(monkeypatch)

    with pytest.raises(FileExistsError, match="exists"):
        _invoke(
            monkeypatch,
            _arguments(
                tmp_path / "catalog.jsonl",
                tmp_path / "dataset.jsonl",
                raw,
                tmp_path / "evidence.json",
            ),
        )

    _assert_no_evaluation(calls)
    assert os.path.lexists(raw)
    assert simulated or raw.is_symlink()
    assert not missing_target.exists()


def test_hardlink_preflight_failure_prevents_evaluation_and_cleans_probes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_evaluator(monkeypatch)
    raw = tmp_path / "raw-parent" / "raw.json"
    evidence = tmp_path / "evidence-parent" / "evidence.json"

    def fail_link(*_: object, **__: object) -> None:
        raise OSError("hardlinks disabled")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(RuntimeError, match="hardlink"):
        _invoke(
            monkeypatch,
            _arguments(
                tmp_path / "catalog.jsonl", tmp_path / "dataset.jsonl", raw, evidence
            ),
        )

    _assert_no_evaluation(calls)
    assert not os.path.lexists(raw)
    assert not os.path.lexists(evidence)
    _assert_no_temporary_entries(raw.parent, evidence.parent)


def test_hardlink_preflight_rejects_nonexclusive_collision_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_evaluator(monkeypatch)
    raw = tmp_path / "raw.json"
    evidence = tmp_path / "evidence.json"
    real_link = os.link

    def overwriting_link(source: str | Path, destination: str | Path) -> None:
        target = Path(destination)
        if os.path.lexists(target):
            target.unlink()
        real_link(source, target)

    monkeypatch.setattr(os, "link", overwriting_link)
    with pytest.raises(RuntimeError, match="exclusive"):
        _invoke(
            monkeypatch,
            _arguments(
                tmp_path / "catalog.jsonl", tmp_path / "dataset.jsonl", raw, evidence
            ),
        )

    _assert_no_evaluation(calls)
    assert not os.path.lexists(raw)
    assert not os.path.lexists(evidence)
    _assert_no_temporary_entries(tmp_path)


@pytest.mark.parametrize("trace_delta", [-1, 1])
def test_incomplete_or_extra_trace_evidence_publishes_neither_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    trace_delta: int,
) -> None:
    result = _result()
    traces = _valid_traces(result)
    traces = (
        traces[:trace_delta] if trace_delta < 0 else [*traces, deepcopy(traces[-1])]
    )
    calls = _install_evaluator(monkeypatch, result=result, traces=traces)
    raw = tmp_path / "raw.json"
    evidence = tmp_path / "evidence.json"

    with pytest.raises(ValueError, match="trace"):
        _invoke(
            monkeypatch,
            _arguments(
                tmp_path / "catalog.jsonl", tmp_path / "dataset.jsonl", raw, evidence
            ),
        )

    assert calls["agent"] == 1
    assert calls["evaluate"] == 1
    assert not os.path.lexists(raw)
    assert not os.path.lexists(evidence)
    assert capsys.readouterr().out == ""
    _assert_no_temporary_entries(tmp_path)


def test_duplicate_trace_for_one_turn_is_not_counted_as_complete_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result()
    traces = _valid_traces(result)
    traces[1] = deepcopy(traces[0])
    _install_evaluator(monkeypatch, result=result, traces=traces)

    with pytest.raises(ValueError, match="trace"):
        _invoke(
            monkeypatch,
            _arguments(
                tmp_path / "catalog.jsonl",
                tmp_path / "dataset.jsonl",
                tmp_path / "raw.json",
                tmp_path / "evidence.json",
            ),
        )

    assert not os.path.lexists(tmp_path / "raw.json")
    assert not os.path.lexists(tmp_path / "evidence.json")


@pytest.mark.parametrize("turn", [True, 1.0])
def test_noninteger_trace_turn_publishes_neither_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    turn: object,
) -> None:
    result = _result()
    traces = _valid_traces(result)
    traces[0]["turn"] = turn
    _install_evaluator(monkeypatch, result=result, traces=traces)

    with pytest.raises(ValueError, match="trace"):
        _invoke(
            monkeypatch,
            _arguments(
                tmp_path / "catalog.jsonl",
                tmp_path / "dataset.jsonl",
                tmp_path / "raw.json",
                tmp_path / "evidence.json",
            ),
        )

    assert not os.path.lexists(tmp_path / "raw.json")
    assert not os.path.lexists(tmp_path / "evidence.json")


def test_session_count_must_match_official_sample_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result()
    result["sample_count"] = 99
    _install_evaluator(monkeypatch, result=result)

    with pytest.raises(ValueError, match="session"):
        _invoke(
            monkeypatch,
            _arguments(
                tmp_path / "catalog.jsonl",
                tmp_path / "dataset.jsonl",
                tmp_path / "raw.json",
                tmp_path / "evidence.json",
            ),
        )

    assert not os.path.lexists(tmp_path / "raw.json")
    assert not os.path.lexists(tmp_path / "evidence.json")


@pytest.mark.parametrize("fallbacks", [["parser"], None])
def test_nonempty_or_missing_top_level_fallbacks_publish_neither_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fallbacks: list[str] | None,
) -> None:
    result = _result()
    traces = _valid_traces(result)
    if fallbacks is None:
        traces[0].pop("fallbacks")
    else:
        traces[0]["fallbacks"] = fallbacks
    calls = _install_evaluator(monkeypatch, result=result, traces=traces)
    raw = tmp_path / "raw.json"
    evidence = tmp_path / "evidence.json"

    with pytest.raises(ValueError, match="fallback"):
        _invoke(
            monkeypatch,
            _arguments(
                tmp_path / "catalog.jsonl", tmp_path / "dataset.jsonl", raw, evidence
            ),
        )

    assert calls["agent"] == 1
    assert calls["evaluate"] == 1
    assert not os.path.lexists(raw)
    assert not os.path.lexists(evidence)
    assert capsys.readouterr().out == ""
    _assert_no_temporary_entries(tmp_path)


def test_evidence_publication_race_never_overwrites_and_cleans_temps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_evaluator(monkeypatch)
    raw = tmp_path / "raw.json"
    evidence = tmp_path / "evidence.json"
    real_link = os.link

    def racing_link(
        source: str | Path, destination: str | Path, *args: object, **kwargs: object
    ) -> None:
        target = Path(destination).resolve()
        if target == evidence.resolve():
            evidence.write_bytes(b"competitor evidence")
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError):
        _invoke(
            monkeypatch,
            _arguments(
                tmp_path / "catalog.jsonl", tmp_path / "dataset.jsonl", raw, evidence
            ),
        )

    assert evidence.read_bytes() == b"competitor evidence"
    assert not os.path.lexists(raw)
    assert capsys.readouterr().out == ""
    _assert_no_temporary_entries(tmp_path, allowed={evidence})


def test_raw_completion_marker_race_keeps_evidence_and_competing_raw_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_evaluator(monkeypatch)
    raw = tmp_path / "raw.json"
    evidence = tmp_path / "evidence.json"
    real_link = os.link
    publications: list[Path] = []

    def racing_link(
        source: str | Path, destination: str | Path, *args: object, **kwargs: object
    ) -> None:
        target = Path(destination).resolve()
        if target in {raw.resolve(), evidence.resolve()}:
            publications.append(target)
        if target == raw.resolve():
            raw.write_bytes(b"competitor raw")
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError):
        _invoke(
            monkeypatch,
            _arguments(
                tmp_path / "catalog.jsonl", tmp_path / "dataset.jsonl", raw, evidence
            ),
        )

    assert publications == [evidence.resolve(), raw.resolve()]
    assert set(json.loads(evidence.read_text(encoding="utf-8"))) == EVIDENCE_KEYS
    assert raw.read_bytes() == b"competitor raw"
    assert capsys.readouterr().out == ""
    _assert_no_temporary_entries(tmp_path, allowed={raw, evidence})
