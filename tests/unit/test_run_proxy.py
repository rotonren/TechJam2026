from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.proxy_dataset import sha256_file


def _sample(scenario: str = "buying") -> dict:
    behavior = {}
    if scenario == "intent_override":
        behavior = {
            "override": {
                "turn": 3,
                "old_value": "soft blue",
                "new_value": "cotton",
                "message": "Actually, ignore my earlier preference. What I need is: cotton.",
            }
        }
    return {
        "sample_id": "proxy_sample_1",
        "scenario_type": scenario,
        "user_profile": {"summary": "safe"},
        "ground_truth": {"parent_asin": "A"},
        "intent_card": {
            "hard_constraints": ["cotton"],
            "soft_preferences": ["soft blue"],
        },
        "behavior": behavior,
    }


def test_proxy_dialogue_variant_is_deterministic_and_discloses_buying_hard_constraint():
    from tools.run_proxy import ProxyDialogue

    sample = _sample()
    first_disclosed: set[str] = set()
    second_disclosed: set[str] = set()
    first = ProxyDialogue(sample, 1).initial_message("shirts", first_disclosed)
    second = ProxyDialogue(sample, 1).initial_message("shirts", second_disclosed)

    assert "cotton" in first
    assert first == second
    assert first_disclosed == second_disclosed == {"cotton"}


def test_proxy_dialogue_variants_change_wording_not_disclosed_state():
    from tools.run_proxy import ProxyDialogue

    sample = _sample()
    messages: list[str] = []
    disclosed: list[set[str]] = []
    for variant in range(4):
        state: set[str] = set()
        messages.append(ProxyDialogue(sample, variant).initial_message("shirts", state))
        disclosed.append(state)

    assert len(set(messages)) == 4
    assert disclosed == [{"cotton"}] * 4


def test_proxy_dialogue_boundary_only_uses_no_preference_once():
    from tools.run_proxy import ProxyDialogue

    dialogue = ProxyDialogue(_sample("boundary"), 0)
    disclosed: set[str] = set()
    first, boundary_used = dialogue.customer_reply("material", disclosed, False)
    second, later_used = dialogue.customer_reply("color", disclosed, boundary_used)

    assert "don't have a preference" in first
    assert boundary_used is True
    assert "soft blue" in second
    assert later_used is True
    assert disclosed == {"soft blue"}


def test_proxy_dialogue_classifies_and_discloses_like_official_reply():
    from tools.run_proxy import ProxyDialogue

    dialogue = ProxyDialogue(_sample(), 2)
    disclosed: set[str] = set()
    material, used = dialogue.customer_reply("material", disclosed, False)
    other, used = dialogue.customer_reply("unknown", disclosed, used)

    assert "cotton" in material
    assert disclosed == {"cotton", "soft blue"}
    assert "soft blue" in other
    assert used is False


def test_override_initial_message_keeps_new_value_hidden():
    from tools.run_proxy import ProxyDialogue

    disclosed: set[str] = set()
    message = ProxyDialogue(_sample("intent_override"), 3).initial_message("shirts", disclosed)

    assert "shirts" in message
    assert "soft blue" in message
    assert "cotton" not in message
    assert disclosed == set()


def test_write_audit_report_is_aggregate_only_and_exclusive(tmp_path: Path):
    from tools.run_proxy import write_audit_report

    destination = tmp_path / "audit" / "baseline.json"
    result = _legal_audit_result()
    result["sessions"] = [{"sample_id": "secret"}]
    metadata = _legal_audit_metadata()
    write_audit_report(destination, result, metadata)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload == {**metadata, "aggregate": {key: value for key, value in result.items() if key != "sessions"}}
    with pytest.raises(FileExistsError):
        write_audit_report(destination, result, metadata)


@pytest.mark.parametrize(
    ("result", "metadata"),
    [
        ({"metric": {"sessions": ["secret"]}, "sessions": []}, {}),
        ({"metric": [{"nested": {"sessions": ["secret"]}}], "sessions": []}, {}),
        ({"sessions": []}, {"sessions": ["secret"]}),
        ({"sessions": []}, {"nested": {"sessions": ["secret"]}}),
        ({"metric": ({"sessions": ["secret"]},), "sessions": []}, {}),
        ({"sessions": []}, {"nested": ({"sessions": ["secret"]},)}),
    ],
)
def test_write_audit_report_rejects_nested_sessions_without_creating_destination(
    tmp_path: Path, result: dict, metadata: dict
):
    from tools.run_proxy import write_audit_report

    destination = tmp_path / "audit.json"

    with pytest.raises(ValueError, match="sessions"):
        write_audit_report(destination, result, metadata)

    assert not destination.exists()


def test_load_proxy_suite_validates_manifest_hash_and_count(tmp_path: Path):
    from tools.run_proxy import load_proxy_suite

    rows = [{"sample_id": "one"}, {"sample_id": "two"}]
    dataset = tmp_path / "representative.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "counts": {"representative": 2, "stress": 0},
        "output_hashes": {"representative.jsonl": sha256_file(dataset), "stress.jsonl": "unused"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded_manifest, loaded_rows = load_proxy_suite(tmp_path, "representative")
    assert loaded_manifest == manifest
    assert loaded_rows == rows
    dataset.write_text(dataset.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash|count"):
        load_proxy_suite(tmp_path, "representative")


def test_load_proxy_suite_rejects_matching_hash_with_wrong_manifest_count(tmp_path: Path):
    from tools.run_proxy import load_proxy_suite

    dataset = tmp_path / "representative.jsonl"
    dataset.write_text('{"sample_id":"one"}\n', encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "counts": {"representative": 2},
                "output_hashes": {"representative.jsonl": sha256_file(dataset)},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="count"):
        load_proxy_suite(tmp_path, "representative")


def test_select_proxy_rows_enforces_dev_audit_and_stress_guards():
    from tools.run_proxy import select_proxy_rows

    representative = [{"sample_id": f"r{fold}", "proxy_fold": fold} for fold in range(1, 6)]
    stress = [{"sample_id": "s1"}, {"sample_id": "s2"}]

    assert [fold for fold, _ in select_proxy_rows(representative, "representative", [1, 4], None)] == [1, 4]
    assert select_proxy_rows(representative, "representative", None, "baseline") == [(5, [representative[-1]])]
    assert select_proxy_rows(stress, "stress", [1], None) == [(None, stress)]
    with pytest.raises(ValueError):
        select_proxy_rows(representative, "representative", [5], None)
    with pytest.raises(ValueError):
        select_proxy_rows(stress, "stress", None, "baseline")
    with pytest.raises(ValueError):
        select_proxy_rows([], "stress", None, None)


@pytest.mark.parametrize("folds", ([5, 5], [5, 1], [1, 5], [5, 5, 5]))
def test_audit_folds_must_be_exactly_one_fold_five(folds: list[int]):
    from tools.run_proxy import select_proxy_rows

    rows = [{"sample_id": f"r{fold}", "proxy_fold": fold} for fold in range(1, 6)]

    with pytest.raises(ValueError, match="fold 5"):
        select_proxy_rows(rows, "representative", folds, "baseline")


def test_evaluate_proxy_matches_official_metrics_and_counts_invalid_responses():
    from tools.run_proxy import evaluate_proxy

    class ScriptedAgent:
        def __init__(self):
            self.calls = 0
            self.session_id = ""

        def reset(self, session_id: str, user_profile: dict) -> None:
            self.session_id = session_id

        def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> object:
            self.calls += 1
            if self.calls == 1:
                return "not a response"
            return {
                "message": "ok",
                "ask_attribute": None,
                "recommendations": ["A"],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }

    sample = _sample()
    agent = ScriptedAgent()
    result = evaluate_proxy(agent, [sample], {"A"}, {"A": ["Clothing", "Shirts"]}, {"A": {"parent_asin": "A"}})

    expected_id = "proxy_eval_" + hashlib.sha256(b"proxy_sample_1").hexdigest()[:20]
    assert agent.session_id == expected_id
    assert result["invalid_response_count"] == 1
    assert result["reported_token_usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert result["hit_rate_at_10"] == result["mrr"] == 1.0
    assert result["mttc"] == 2.0
    assert result["efficiency"] == 0.9
    assert result["recommended_technical_score"] == 0.98
    assert result["sessions"] == [{"sample_id": "proxy_sample_1", "scenario_type": "buying", "hit": True, "first_hit_turn": 2, "best_rank": 1, "reciprocal_rank": 1.0}]


def test_evaluate_proxy_counts_exception_once_and_preserves_later_usage():
    from tools.run_proxy import evaluate_proxy

    class ExceptionThenHitAgent:
        def reset(self, session_id: str, user_profile: dict) -> None:
            pass

        def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
            if turn == 1:
                raise RuntimeError("temporary failure")
            return {
                "message": "ok",
                "ask_attribute": None,
                "recommendations": ["A"],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    result = evaluate_proxy(
        ExceptionThenHitAgent(),
        [_sample()],
        {"A"},
        {"A": ["Clothing", "Shirts"]},
        {"A": {"parent_asin": "A"}},
    )

    assert result["invalid_response_count"] == 1
    assert result["reported_token_usage"] == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def test_evaluate_proxy_does_not_count_target_before_override_and_stops_after():
    from tools.run_proxy import evaluate_proxy

    sample = _sample("intent_override")
    messages: list[tuple[int, str]] = []

    class OverrideAgent:
        def reset(self, session_id: str, user_profile: dict) -> None:
            pass

        def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
            messages.append((turn, user_message))
            return {"message": "ok", "ask_attribute": None, "recommendations": ["A"]}

    result = evaluate_proxy(
        OverrideAgent(),
        [sample],
        {"A"},
        {"A": ["Clothing", "Shirts"]},
        {"A": {"parent_asin": "A"}},
    )

    assert [turn for turn, _ in messages] == [1, 2, 3]
    assert "soft blue" in messages[0][1]
    assert messages[2][1] == sample["behavior"]["override"]["message"]
    assert result["sessions"] == [{"sample_id": "proxy_sample_1", "scenario_type": "intent_override", "hit": True, "first_hit_turn": 3, "best_rank": 1, "reciprocal_rank": 1.0}]


def _run_args(tmp_path: Path, *, audit_label: str | None) -> SimpleNamespace:
    proxy_root = tmp_path / "proxy"
    return SimpleNamespace(
        catalog=tmp_path / "catalog.jsonl",
        proxy_root=proxy_root,
        suite="representative",
        folds=None,
        audit_label=audit_label,
        agent="test:Agent",
        output=(proxy_root / "audit" / f"{audit_label}.json") if audit_label else tmp_path / "report.json",
    )


def _patch_small_proxy_run(
    monkeypatch: pytest.MonkeyPatch, *, invalid_count: int, fallback_count: int = 0
) -> None:
    from tools import run_proxy

    result = _legal_audit_result()
    result.update(
        {
            "recommended_technical_score": 0.5,
            "invalid_response_count": invalid_count,
            "sessions": [{"sample_id": "one", "scenario_type": "buying"}],
        }
    )
    verified = run_proxy._VerifiedProxySuite(
        {}, [{"sample_id": "one"}], "manifest_hash", "dataset_hash"
    )
    monkeypatch.setattr(run_proxy, "_load_verified_proxy_suite", lambda root, suite: verified)
    monkeypatch.setattr(run_proxy, "select_proxy_rows", lambda rows, suite, folds, audit: [(1, rows)])
    monkeypatch.setattr(run_proxy, "catalog_index", lambda path: (set(), {}, {}))
    monkeypatch.setattr(run_proxy, "_load_agent", lambda spec: lambda catalog: object())
    monkeypatch.setattr(run_proxy, "evaluate_proxy", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        run_proxy, "_trace_details", lambda agent, sessions: ({"count": 0}, fallback_count, {})
    )
    monkeypatch.setattr(run_proxy, "_git_commit", lambda: "commit")
    monkeypatch.setattr(run_proxy, "_config_hash", lambda agent: "config")


@pytest.mark.parametrize(("invalid_count", "fallback_count"), [(1, 0), (0, 1)])
def test_run_proxy_audit_failure_does_not_create_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_count: int, fallback_count: int
):
    from tools.run_proxy import run_proxy

    _patch_small_proxy_run(
        monkeypatch, invalid_count=invalid_count, fallback_count=fallback_count
    )
    args = _run_args(tmp_path, audit_label="baseline")

    with pytest.raises(SystemExit, match="1"):
        run_proxy(args)

    assert not args.output.exists()


def test_run_proxy_non_audit_failure_writes_diagnostic_and_uses_sample_invalid_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools.run_cv import selection_score
    from tools.run_proxy import run_proxy

    _patch_small_proxy_run(monkeypatch, invalid_count=1)
    args = _run_args(tmp_path, audit_label=None)

    with pytest.raises(SystemExit, match="1"):
        run_proxy(args)

    report = json.loads(args.output.read_text(encoding="utf-8"))
    assert report["selection_score"] == selection_score([0.5], 0.0, 1.0)
    assert report["folds"][0]["sessions"] == [{"sample_id": "one", "scenario_type": "buying"}]


def test_frozen_evaluator_hash_is_unchanged():
    assert sha256_file(Path("evaluator/local_evaluator.py")) == "84ea899707452de249ca62abee77c4b40ab7a3139b5cc798ac30c9f521f91b30"


def _write_suite(root: Path, rows: list[dict], *, count: object | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "representative.jsonl"
    dataset.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "counts": {"representative": len(rows) if count is None else count},
                "output_hashes": {"representative.jsonl": sha256_file(dataset)},
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("rows", "count", "message"),
    [
        ([{"sample_id": "one"}], True, "count"),
        ([{"sample_id": "one"}], -1, "count"),
        ([{"sample_id": ""}], 1, "sample_id"),
        ([{"sample_id": "one"}, {"sample_id": "one"}], 2, "duplicate"),
    ],
)
def test_verified_proxy_suite_rejects_invalid_counts_and_sample_ids(
    tmp_path: Path, rows: list[dict], count: object, message: str
):
    from tools.run_proxy import _load_verified_proxy_suite

    _write_suite(tmp_path, rows, count=count)

    with pytest.raises(ValueError, match=message):
        _load_verified_proxy_suite(tmp_path, "representative")


def test_verified_proxy_suite_hashes_the_same_bytes_it_parses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.run_proxy import _load_verified_proxy_suite

    _write_suite(tmp_path, [{"sample_id": "one"}])
    dataset = tmp_path / "representative.jsonl"
    original_read_bytes = Path.read_bytes
    calls = 0

    def read_once_then_swap(path: Path) -> bytes:
        nonlocal calls
        value = original_read_bytes(path)
        if path == dataset:
            calls += 1
            dataset.write_bytes(b'{"sample_id":"swapped"}\n')
        return value

    monkeypatch.setattr(Path, "read_bytes", read_once_then_swap)
    verified = _load_verified_proxy_suite(tmp_path, "representative")

    assert calls == 1
    assert verified.rows == [{"sample_id": "one"}]
    assert verified.dataset_hash == hashlib.sha256(b'{"sample_id": "one"}\n').hexdigest()


def test_verified_proxy_suite_rejects_opaque_session_id_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import run_proxy

    _write_suite(tmp_path, [{"sample_id": "one"}, {"sample_id": "two"}])
    monkeypatch.setattr(run_proxy, "opaque_session_id", lambda sample_id: "collision")

    with pytest.raises(ValueError, match="collision"):
        run_proxy._load_verified_proxy_suite(tmp_path, "representative")


@pytest.mark.parametrize(
    ("result", "metadata", "message"),
    [
        ({"sessions": [], "targets": []}, {"suite": "representative"}, "unknown"),
        ({"sessions": [], "scenario_metrics": {"x": {"misses": 1}}}, {"suite": "representative"}, "sensitive"),
        ({"sessions": []}, {"suite": "representative", "extra": 1}, "unknown"),
        ({"sessions": []}, {"suite": "representative", "nested": {"TARGET": "x"}}, "sensitive"),
    ],
)
def test_audit_report_allowlist_rejects_unknown_and_sensitive_fields(
    tmp_path: Path, result: dict, metadata: dict, message: str
):
    from tools.run_proxy import write_audit_report

    with pytest.raises(ValueError, match=message):
        write_audit_report(tmp_path / "audit.json", result, metadata)


def test_audit_report_allowlist_keeps_only_approved_aggregate(tmp_path: Path):
    from tools.run_proxy import write_audit_report

    destination = tmp_path / "audit.json"
    result = _legal_audit_result()
    metadata = _legal_audit_metadata()
    write_audit_report(destination, result, metadata)

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        **metadata,
        "aggregate": {key: value for key, value in result.items() if key != "sessions"},
    }


def test_atomic_report_rejects_serialization_and_link_failures_without_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import run_proxy

    destination = tmp_path / "report.json"
    monkeypatch.setattr(run_proxy.json, "dumps", lambda value, **kwargs: (_ for _ in ()).throw(TypeError("bad")))
    with pytest.raises(TypeError, match="bad"):
        run_proxy._write_normal_report(destination, {"not": {"json": object()}})
    assert not destination.exists()

    monkeypatch.undo()
    monkeypatch.setattr(run_proxy.os, "link", lambda source, target: (_ for _ in ()).throw(OSError("no link")))
    with pytest.raises(RuntimeError, match="hardlink"):
        run_proxy._write_normal_report(destination, {"ok": True})
    assert not destination.exists()
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_trace_details_requires_exact_session_turn_alignment():
    from tools.run_proxy import _CallEvidence, _trace_details, opaque_session_id

    evidence = [
        _CallEvidence(opaque_session_id("one"), 1, True),
        _CallEvidence(opaque_session_id("one"), 2, True),
    ]
    expected = [
        {"session_id": opaque_session_id("one"), "turn": 1, "elapsed_ms": 1.0, "fallbacks": [], "route": "buying"},
        {"session_id": opaque_session_id("one"), "turn": 2, "elapsed_ms": 2.0, "fallbacks": [], "route": "buying"},
    ]
    agent = SimpleNamespace(traces=SimpleNamespace(records=expected))

    assert _trace_details(agent, evidence)[0] == {"count": 2, "p50": 1.5, "p95": 2.0, "max": 2.0}
    for broken in (expected[:-1], list(reversed(expected)), [{**expected[0], "session_id": "stale"}, expected[1]]):
        agent.traces.records = broken
        with pytest.raises(ValueError, match="trace"):
            _trace_details(agent, evidence)


def test_trace_details_rejects_truncated_large_trace_list():
    from tools.run_proxy import _CallEvidence, _trace_details, opaque_session_id

    evidence = [
        _CallEvidence(opaque_session_id(str(index)), turn, True)
        for index in range(501)
        for turn in range(1, 11)
    ]
    records = [
        {"session_id": opaque_session_id(str(index)), "turn": turn, "elapsed_ms": 1.0, "fallbacks": []}
        for index in range(501)
        for turn in range(1, 11)
    ][-5000:]
    agent = SimpleNamespace(traces=SimpleNamespace(records=records))

    with pytest.raises(ValueError, match="trace"):
        _trace_details(agent, evidence)


def test_parse_proxy_args_accepts_the_sealed_cli_contract(tmp_path: Path):
    from tools.run_proxy import parse_proxy_args

    args = parse_proxy_args(
        [
            "--proxy-root", str(tmp_path / "proxy"), "--suite", "representative",
            "--folds", "1", "2", "--output", str(tmp_path / "report.json"),
        ]
    )

    assert args.suite == "representative"
    assert args.folds == [1, 2]
    assert args.output == tmp_path / "report.json"


def test_run_proxy_successful_normal_report_is_exclusive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.run_proxy import run_proxy

    _patch_small_proxy_run(monkeypatch, invalid_count=0)
    args = _run_args(tmp_path, audit_label=None)

    run_proxy(args)

    assert args.output.exists()
    with pytest.raises(FileExistsError):
        run_proxy(args)


def test_run_proxy_successful_audit_uses_reservation_and_releases_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools.run_proxy import run_proxy

    _patch_small_proxy_run(monkeypatch, invalid_count=0)
    args = _run_args(tmp_path, audit_label="baseline")

    run_proxy(args)

    assert args.output.exists()
    assert not args.output.with_suffix(".lock").exists()


@pytest.mark.parametrize("reason", ("wrong-path", "existing-output", "existing-lock"))
def test_audit_reservation_rejects_before_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
):
    from tools import run_proxy

    _patch_small_proxy_run(monkeypatch, invalid_count=0)
    args = _run_args(tmp_path, audit_label="baseline")
    calls = 0

    def forbidden_evaluate(*args: object) -> dict:
        nonlocal calls
        calls += 1
        raise AssertionError("must not evaluate")

    monkeypatch.setattr(run_proxy, "evaluate_proxy", forbidden_evaluate)
    if reason == "wrong-path":
        args.output = tmp_path / "wrong.json"
    elif reason == "existing-output":
        args.output.parent.mkdir(parents=True)
        args.output.write_text("already", encoding="utf-8")
    else:
        args.output.parent.mkdir(parents=True)
        args.output.with_suffix(".lock").write_text("reserved", encoding="utf-8")

    with pytest.raises((ValueError, FileExistsError)):
        run_proxy.run_proxy(args)

    assert calls == 0


def test_non_audit_cannot_target_audit_directory_and_normal_collision_is_pre_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import run_proxy

    _patch_small_proxy_run(monkeypatch, invalid_count=0)
    args = _run_args(tmp_path, audit_label=None)
    args.output = args.proxy_root / "audit" / "not-audit.json"
    with pytest.raises(ValueError, match="audit"):
        run_proxy.run_proxy(args)

    args.output = tmp_path / "collision.json"
    args.output.write_text("already", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_proxy.run_proxy(args)


def test_audit_failure_releases_owned_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.run_proxy import run_proxy

    _patch_small_proxy_run(monkeypatch, invalid_count=1)
    args = _run_args(tmp_path, audit_label="baseline")

    with pytest.raises(SystemExit, match="1"):
        run_proxy(args)

    assert not args.output.exists()
    assert not args.output.with_suffix(".lock").exists()


def _legal_audit_result() -> dict:
    metrics = {"sample_count": 1, "hit_rate_at_10": 1.0, "mrr": 1.0, "mttc": 1.0}
    return {
        "sample_count": 4,
        "hit_rate_at_10": 1.0,
        "mrr": 1.0,
        "mttc": 1.0,
        "efficiency": 1.0,
        "recommended_technical_score": 1.0,
        "reported_token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "scenario_metrics": {
            "boundary": metrics.copy(), "browsing": metrics.copy(),
            "buying": metrics.copy(), "intent_override": metrics.copy(),
        },
        "invalid_response_count": 0,
        "sessions": [],
    }


def _legal_audit_metadata() -> dict:
    return {
        "created_at": "2026-08-27T00:00:00+00:00", "commit": "abc",
        "config_hash": "config", "manifest_hash": "manifest", "dataset_hash": "dataset",
        "suite": "representative", "fallback_count": 0, "invalid_response_count": 0,
        "audit_label": "baseline",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda result: result["scenario_metrics"]["buying"].update({"target_id": "secret"}),
        lambda result: result["scenario_metrics"].update({"other": {}}),
        lambda result: result["reported_token_usage"].update({"extra": 1}),
        lambda result: result.update({"sample_count": True}),
        lambda result: result["scenario_metrics"]["boundary"].update({"sample_count": 2}),
    ],
)
def test_audit_payload_requires_exact_nested_schema(mutate):
    from tools.run_proxy import _audit_payload

    result = _legal_audit_result()
    mutate(result)

    with pytest.raises(ValueError):
        _audit_payload(result, _legal_audit_metadata())


def test_audit_payload_requires_exact_metadata_schema():
    from tools.run_proxy import _audit_payload

    metadata = _legal_audit_metadata()
    metadata["fallback_count"] = True

    with pytest.raises(ValueError):
        _audit_payload(_legal_audit_result(), metadata)


def test_trace_evidence_allows_only_exception_call_to_be_missing():
    from tools.run_proxy import _CallEvidence, _trace_details, opaque_session_id

    session_id = opaque_session_id("one")
    evidence = [
        _CallEvidence(session_id, 1, True),
        _CallEvidence(session_id, 2, False),
        _CallEvidence(session_id, 3, True),
    ]
    records = [
        {"session_id": session_id, "turn": 1, "elapsed_ms": 1.0, "fallbacks": []},
        {"session_id": session_id, "turn": 3, "elapsed_ms": 1.0, "fallbacks": []},
    ]
    agent = SimpleNamespace(traces=SimpleNamespace(records=records))

    assert _trace_details(agent, evidence)[0]["count"] == 2
    agent.traces.records = [records[1]]
    with pytest.raises(ValueError, match="trace"):
        _trace_details(agent, evidence)
    agent.traces.records = [records[0], {**records[1], "turn": 4}, records[1]]
    with pytest.raises(ValueError, match="trace"):
        _trace_details(agent, evidence)


def test_evaluate_proxy_records_exception_and_invalid_response_call_evidence():
    from tools.run_proxy import evaluate_proxy, opaque_session_id

    calls: list = []

    class Agent:
        def reset(self, session_id: str, user_profile: dict) -> None:
            pass

        def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> object:
            if turn == 1:
                raise RuntimeError("no trace")
            return "invalid"

    evaluate_proxy(
        Agent(), [_sample()], {"A"}, {"A": ["Clothing", "Shirts"]},
        {"A": {"parent_asin": "A"}}, call_evidence=calls,
    )

    assert [(item.session_id, item.turn, item.trace_required) for item in calls] == [
        (opaque_session_id("proxy_sample_1"), 1, False),
        (opaque_session_id("proxy_sample_1"), 2, True),
        (opaque_session_id("proxy_sample_1"), 3, True),
        (opaque_session_id("proxy_sample_1"), 4, True),
        (opaque_session_id("proxy_sample_1"), 5, True),
        (opaque_session_id("proxy_sample_1"), 6, True),
        (opaque_session_id("proxy_sample_1"), 7, True),
        (opaque_session_id("proxy_sample_1"), 8, True),
        (opaque_session_id("proxy_sample_1"), 9, True),
        (opaque_session_id("proxy_sample_1"), 10, True),
    ]


def _patch_real_proxy_run(monkeypatch: pytest.MonkeyPatch, agent_class: type) -> None:
    from tools import run_proxy

    rows = [{**_sample(), "proxy_fold": 1}]
    verified = run_proxy._VerifiedProxySuite({}, rows, "manifest_hash", "dataset_hash")
    monkeypatch.setattr(run_proxy, "_load_verified_proxy_suite", lambda root, suite: verified)
    monkeypatch.setattr(run_proxy, "select_proxy_rows", lambda rows, suite, folds, audit: [(1, rows)])
    monkeypatch.setattr(
        run_proxy,
        "catalog_index",
        lambda path: ({"A"}, {"A": ["Clothing", "Shirts"]}, {"A": {"parent_asin": "A"}}),
    )
    monkeypatch.setattr(run_proxy, "_load_agent", lambda spec: agent_class)
    monkeypatch.setattr(run_proxy, "_git_commit", lambda: "commit")
    monkeypatch.setattr(run_proxy, "_config_hash", lambda agent: "config")


def test_run_proxy_tolerates_only_exception_calls_without_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools.run_proxy import run_proxy

    class ExceptionThenTraceAgent:
        def __init__(self, catalog: Path) -> None:
            self.traces = SimpleNamespace(records=[])

        def reset(self, session_id: str, user_profile: dict) -> None:
            pass

        def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
            if turn == 1:
                raise RuntimeError("no response and no trace")
            self.traces.records.append(
                {"session_id": session_id, "turn": turn, "elapsed_ms": 1.0, "fallbacks": []}
            )
            return {"message": "ok", "ask_attribute": None, "recommendations": ["A"]}

    _patch_real_proxy_run(monkeypatch, ExceptionThenTraceAgent)
    args = _run_args(tmp_path, audit_label=None)

    with pytest.raises(SystemExit, match="1"):
        run_proxy(args)

    report = json.loads(args.output.read_text(encoding="utf-8"))
    assert report["invalid_response_count"] == 1
    assert report["folds"][0]["latency_ms"]["count"] == 1


def test_run_proxy_requires_trace_for_invalid_returned_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools.run_proxy import run_proxy

    class InvalidWithoutTraceAgent:
        def __init__(self, catalog: Path) -> None:
            self.traces = SimpleNamespace(records=[])

        def reset(self, session_id: str, user_profile: dict) -> None:
            pass

        def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> object:
            return "invalid payload"

    _patch_real_proxy_run(monkeypatch, InvalidWithoutTraceAgent)
    args = _run_args(tmp_path, audit_label=None)

    with pytest.raises(ValueError, match="trace"):
        run_proxy(args)

    assert not args.output.exists()


@pytest.mark.parametrize("extra_record", (False, True))
def test_run_proxy_rejects_stale_and_extra_trace_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra_record: bool
):
    from tools.run_proxy import run_proxy

    class TraceAgent:
        def __init__(self, catalog: Path) -> None:
            self.traces = SimpleNamespace(records=[])

        def reset(self, session_id: str, user_profile: dict) -> None:
            pass

        def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
            self.traces.records.append(
                {
                    "session_id": session_id,
                    "turn": turn if extra_record else 0,
                    "elapsed_ms": 1.0,
                    "fallbacks": [],
                }
            )
            if extra_record:
                self.traces.records.append(
                    {"session_id": session_id, "turn": 2, "elapsed_ms": 1.0, "fallbacks": []}
                )
            return {"message": "ok", "ask_attribute": None, "recommendations": ["A"]}

    _patch_real_proxy_run(monkeypatch, TraceAgent)
    args = _run_args(tmp_path, audit_label=None)

    with pytest.raises(ValueError, match="trace"):
        run_proxy(args)

    assert not args.output.exists()


@pytest.mark.parametrize("failure", ("write", "flush", "fsync", "close"))
def test_reserve_audit_output_cleans_owned_lock_after_durable_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
):
    from tools import run_proxy

    output = tmp_path / "proxy" / "audit" / "baseline.json"
    original_open = Path.open

    class BrokenLock:
        def __init__(self, handle: object) -> None:
            self.handle = handle

        def write(self, value: str) -> int:
            if failure == "write":
                raise OSError("write failed")
            return self.handle.write(value)

        def flush(self) -> None:
            if failure == "flush":
                raise OSError("flush failed")
            self.handle.flush()

        def fileno(self) -> int:
            return self.handle.fileno()

        def close(self) -> None:
            self.handle.close()
            if failure == "close":
                raise OSError("close failed")

    def broken_open(path: Path, *args: object, **kwargs: object):
        handle = original_open(path, *args, **kwargs)
        return BrokenLock(handle)

    monkeypatch.setattr(Path, "open", broken_open)
    if failure == "fsync":
        monkeypatch.setattr(
            run_proxy.os,
            "fsync",
            lambda descriptor: (_ for _ in ()).throw(OSError("fsync failed")),
        )
    with pytest.raises(OSError, match=failure):
        run_proxy._reserve_audit_output(tmp_path / "proxy", "baseline", output)
    assert not output.with_suffix(".lock").exists()


def test_reserve_audit_output_preserves_preexisting_lock(tmp_path: Path):
    from tools.run_proxy import _reserve_audit_output

    output = tmp_path / "proxy" / "audit" / "baseline.json"
    lock = output.with_suffix(".lock")
    lock.parent.mkdir(parents=True)
    lock.write_text("other reservation\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="reservation"):
        _reserve_audit_output(tmp_path / "proxy", "baseline", output)

    assert lock.read_text(encoding="utf-8") == "other reservation\n"


def test_reserve_audit_output_cleans_owned_lock_after_post_create_file_exists_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import run_proxy

    output = tmp_path / "proxy" / "audit" / "baseline.json"
    original_open = Path.open

    class FileExistsWriteLock:
        def __init__(self, handle: object) -> None:
            self.handle = handle

        def write(self, value: str) -> int:
            raise FileExistsError("write collision")

        def close(self) -> None:
            self.handle.close()

    def file_exists_write_open(path: Path, *args: object, **kwargs: object):
        return FileExistsWriteLock(original_open(path, *args, **kwargs))

    monkeypatch.setattr(Path, "open", file_exists_write_open)
    with pytest.raises(FileExistsError, match="write collision"):
        run_proxy._reserve_audit_output(tmp_path / "proxy", "baseline", output)

    assert not output.with_suffix(".lock").exists()
