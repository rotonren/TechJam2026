from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import pytest

from compasscart_debug.agent_adapter import TurnObservation
from compasscart_debug.config import RuntimeIdentity
from compasscart_debug.errors import (
    ConflictError,
    DebugServiceError,
    NotReadyError,
    ReplayMismatchError,
    RequestMismatchError,
    UnresolvedTurnError,
    ValidationError,
)
from compasscart_debug.repository import (
    DebugRepository,
    RepositoryStorageError,
    TurnRecord,
)
from compasscart_debug.service import DebugService

IDENTITY = RuntimeIdentity("agent-v1", "catalog", "config", "assets")
NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


class FakeWorker:
    def __init__(self, identity: RuntimeIdentity = IDENTITY) -> None:
        self.current_identity = identity
        self.sessions: dict[str, list[tuple[int, str]]] = {}
        self.reset_calls: list[tuple[str, dict[str, object]]] = []
        self.has_session_calls: list[str] = []
        self.observe_calls: list[tuple[str, str, int]] = []
        self.rehydrate_calls: list[tuple[str, tuple[tuple[int, str], ...]]] = []
        self.events: list[str] = []
        self.fail_after_mutating_once = False
        self.replay_response_mismatch = False
        self.replay_state_mismatch = False
        self.has_session_error: DebugServiceError | None = None
        self.reply_prefix = "reply"
        self.fail_on_message: str | None = None

    @property
    def identity(self) -> RuntimeIdentity:
        return self.current_identity

    def reset_session(self, session_id: str, profile: dict[str, object]) -> None:
        self.events.append("reset")
        self.reset_calls.append((session_id, profile))
        self.sessions[session_id] = []

    def has_session(self, session_id: str) -> bool:
        self.has_session_calls.append(session_id)
        if self.has_session_error is not None:
            raise self.has_session_error
        return session_id in self.sessions

    def observe_turn(self, session_id: str, message: str, turn: int) -> TurnObservation:
        self.observe_calls.append((session_id, message, turn))
        self.sessions.setdefault(session_id, []).append((turn, message))
        if self.fail_after_mutating_once or message == self.fail_on_message:
            self.fail_after_mutating_once = False
            self.events.append("failed_observe")
            raise RuntimeError(r"C:\private\agent.py failed")
        self.events.append("observe")
        return self._observation(session_id, message, turn)

    def rehydrate(
        self,
        session_id: str,
        profile: dict[str, object],
        completed: list[TurnRecord],
    ) -> list[TurnObservation]:
        messages = tuple((item.turn, item.user_message) for item in completed)
        self.rehydrate_calls.append((session_id, messages))
        self.reset_session(session_id, profile)
        observations: list[TurnObservation] = []
        for turn, message in messages:
            self.sessions[session_id].append((turn, message))
            observations.append(self._observation(session_id, message, turn))
        if observations and self.replay_response_mismatch:
            first = observations[0]
            observations[0] = TurnObservation(
                {**first.response, "message": "different"},
                first.products,
                first.state,
                first.trace,
            )
        if observations and self.replay_state_mismatch:
            first = observations[0]
            observations[0] = TurnObservation(
                first.response,
                first.products,
                {**first.state, "candidate_count": 999},
                first.trace,
            )
        return observations

    def _observation(self, session_id: str, message: str, turn: int) -> TurnObservation:
        return TurnObservation(
            response={
                "message": f"{self.reply_prefix}:{message}",
                "ask_attribute": None,
                "recommendations": [],
                "usage": {},
            },
            products=[{"rank": 1, "parent_asin": "SHOE1"}],
            state={
                "session_id": session_id,
                "turn": turn,
                "candidate_count": 0,
                "history": [
                    {"turn": item_turn, "message": item_message}
                    for item_turn, item_message in self.sessions[session_id]
                ],
            },
            trace={"session_id": session_id, "turn": turn, "elapsed_ms": 1.0},
        )


class BlockingWorker(FakeWorker):
    def __init__(self) -> None:
        super().__init__()
        self.entered = Event()
        self.release = Event()

    def observe_turn(self, session_id: str, message: str, turn: int) -> TurnObservation:
        observation = super().observe_turn(session_id, message, turn)
        self.entered.set()
        assert self.release.wait(3)
        return observation


def _repository(path: Path) -> DebugRepository:
    repository = DebugRepository(path, clock=lambda: NOW)
    repository.initialize()
    return repository


def _service(
    tmp_path: Path,
    *,
    worker: FakeWorker | None = None,
    repository: DebugRepository | None = None,
    ids: tuple[str, ...] = ("session-1", "session-2", "session-3"),
) -> tuple[DebugService, DebugRepository, FakeWorker]:
    selected_repository = repository or _repository(tmp_path / "debug.sqlite3")
    selected_worker = worker or FakeWorker()
    session_ids = iter(ids)
    service = DebugService(
        selected_repository,
        selected_worker,
        uuid_factory=lambda: next(session_ids),
        clock=lambda: NOW,
    )
    return service, selected_repository, selected_worker


def _create(service: DebugService, name: str = "Test") -> str:
    return service.create_session(name, {})["session"]["session_id"]


def _stored_observation(session_id: str, message: str, turn: int) -> TurnObservation:
    worker = FakeWorker()
    worker.reset_session(session_id, {})
    return worker.observe_turn(session_id, message, turn)


def test_completed_request_is_idempotent_and_different_text_conflicts(
    tmp_path: Path,
) -> None:
    service, _, worker = _service(tmp_path)
    session_id = _create(service)

    first = service.send_message(session_id, "r1", "blue shoes")
    second = service.send_message(session_id, "r1", "blue shoes")

    assert first == second
    assert worker.observe_calls == [(session_id, "blue shoes", 1)]
    with pytest.raises(RequestMismatchError):
        service.send_message(session_id, "r1", "red shoes")


def test_inflight_duplicate_returns_pending_and_different_request_is_rejected(
    tmp_path: Path,
) -> None:
    worker = BlockingWorker()
    service, _, _ = _service(tmp_path, worker=worker)
    session_id = _create(service)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.send_message, session_id, "r1", "blue")
        assert worker.entered.wait(2)
        duplicate = service.send_message(session_id, "r1", "blue")
        with pytest.raises(UnresolvedTurnError):
            service.send_message(session_id, "r2", "red")
        worker.release.set()
        completed = first.result(timeout=3)

    assert duplicate["turn"]["status"] == "pending"
    assert completed["turn"]["status"] == "completed"
    assert worker.observe_calls == [(session_id, "blue", 1)]


def test_agent_error_marks_failed_and_retry_rehydrates_completed_only(
    tmp_path: Path,
) -> None:
    service, repository, worker = _service(tmp_path)
    session_id = _create(service)
    worker.fail_after_mutating_once = True

    with pytest.raises(DebugServiceError) as captured:
        service.send_message(session_id, "r1", "blue")

    assert captured.value.code == "agent_error"
    assert captured.value.retryable is True
    assert r"C:\private" not in captured.value.message
    assert repository.get_turn_by_request_id(session_id, "r1").status == "failed"
    assert repository.get_session(session_id).dirty is True

    retried = service.send_message(session_id, "r1", "blue")

    assert retried["turn"]["status"] == "completed"
    assert worker.events[:3] == ["reset", "failed_observe", "reset"]
    assert worker.rehydrate_calls == [(session_id, ())]
    assert worker.observe_calls[-1] == (session_id, "blue", 1)


def test_crash_pending_is_rehydrated_and_retried_with_same_turn(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "debug.sqlite3")
    first_service, _, _ = _service(tmp_path, repository=repository)
    session_id = _create(first_service)
    pending = repository.reserve_turn(session_id, "r1", "blue")
    restarted_worker = FakeWorker()
    restarted, _, _ = _service(
        tmp_path,
        worker=restarted_worker,
        repository=repository,
        ids=("unused",),
    )

    result = restarted.send_message(session_id, "r1", "blue")

    assert result["turn"]["turn"] == pending.turn == 1
    assert result["turn"]["status"] == "completed"
    assert restarted_worker.rehydrate_calls == [(session_id, ())]
    assert restarted_worker.observe_calls == [(session_id, "blue", 1)]


def test_completion_failure_never_returns_response_and_retry_rehydrates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, repository, worker = _service(tmp_path)
    session_id = _create(service)
    real_complete = repository.complete_turn
    calls = 0

    def fail_once(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RepositoryStorageError(r"failed at C:\private\debug.sqlite3")
        return real_complete(*args, **kwargs)

    monkeypatch.setattr(repository, "complete_turn", fail_once)

    with pytest.raises(DebugServiceError) as captured:
        service.send_message(session_id, "r1", "blue")

    assert captured.value.code == "snapshot_not_saved"
    assert repository.get_turn_by_request_id(session_id, "r1").status == "pending"
    assert repository.get_session(session_id).dirty is True

    result = service.send_message(session_id, "r1", "blue")

    assert result["turn"]["status"] == "completed"
    assert worker.rehydrate_calls == [(session_id, ())]
    assert worker.observe_calls == [
        (session_id, "blue", 1),
        (session_id, "blue", 1),
    ]


@pytest.mark.parametrize(
    ("failure_source", "expected_code"),
    [("repository", "repository_unavailable"), ("worker", "not_ready")],
)
def test_safe_hydration_failures_stay_pending_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_source: str,
    expected_code: str,
) -> None:
    service, repository, worker = _service(tmp_path)
    session_id = _create(service)
    if failure_source == "repository":
        monkeypatch.setattr(
            repository,
            "list_completed_turns",
            lambda _session_id: (_ for _ in ()).throw(
                RepositoryStorageError("private database path")
            ),
        )
        service._loaded_sessions.clear()
    else:
        worker.has_session_error = NotReadyError()

    with pytest.raises(DebugServiceError) as captured:
        service.send_message(session_id, "r1", "blue")

    assert captured.value.code == expected_code
    assert repository.get_turn_by_request_id(session_id, "r1").status == "pending"
    assert repository.get_session(session_id).dirty is False
    assert worker.observe_calls == []


def test_evicted_loaded_session_rehydrates_before_next_observation(
    tmp_path: Path,
) -> None:
    service, _, worker = _service(tmp_path)
    session_id = _create(service)
    worker.sessions.pop(session_id)

    service.send_message(session_id, "r1", "blue")

    assert worker.has_session_calls == [session_id]
    assert worker.rehydrate_calls == [(session_id, ())]
    assert worker.events[:2] == ["reset", "reset"]


def test_identity_mismatch_is_read_only_without_agent_operations(
    tmp_path: Path,
) -> None:
    service, repository, worker = _service(tmp_path)
    session_id = _create(service)
    worker.current_identity = RuntimeIdentity("agent-v2", "catalog", "config", "assets")
    before = (
        len(worker.reset_calls),
        len(worker.has_session_calls),
        len(worker.observe_calls),
        len(worker.rehydrate_calls),
    )

    with pytest.raises(DebugServiceError) as captured:
        service.send_message(session_id, "r1", "blue")

    assert captured.value.code == "runtime_incompatible"
    assert repository.get_session(session_id).read_only_reason == "runtime_incompatible"
    assert before == (
        len(worker.reset_calls),
        len(worker.has_session_calls),
        len(worker.observe_calls),
        len(worker.rehydrate_calls),
    )
    detail = service.get_session(session_id)
    assert (detail["continuation"], detail["can_send"]) == (
        "incompatible",
        False,
    )


@pytest.mark.parametrize("mismatch", ["response", "state"])
def test_replay_response_or_state_mismatch_becomes_read_only(
    tmp_path: Path, mismatch: str
) -> None:
    first, repository, _ = _service(tmp_path)
    session_id = _create(first)
    first.send_message(session_id, "r1", "blue")
    replay_worker = FakeWorker()
    setattr(replay_worker, f"replay_{mismatch}_mismatch", True)
    restarted, _, _ = _service(
        tmp_path,
        worker=replay_worker,
        repository=repository,
        ids=("unused",),
    )

    with pytest.raises(ReplayMismatchError):
        restarted.send_message(session_id, "r2", "red")

    assert repository.get_session(session_id).read_only_reason == "replay_mismatch"
    assert replay_worker.observe_calls == []
    detail = restarted.get_session(session_id)
    assert (detail["continuation"], detail["can_send"]) == (
        "incompatible",
        False,
    )


def test_detail_reports_all_continuation_states_and_turn_feedback(
    tmp_path: Path,
) -> None:
    service, repository, _ = _service(
        tmp_path,
        ids=("ready", "pending", "incompatible", "failed", "limit"),
    )
    ready = _create(service)
    pending = _create(service)
    repository.reserve_turn(pending, "r1", "pending")
    incompatible = _create(service)
    repository.set_read_only_reason(incompatible, "runtime_incompatible")
    failed = _create(service)
    failed_turn = repository.reserve_turn(failed, "r1", "failed")
    repository.fail_turn(failed, failed_turn.turn, {"code": "agent_error"})
    limit = _create(service)
    for turn in range(1, 11):
        row = repository.reserve_turn(limit, f"r{turn}", f"message {turn}")
        repository.complete_turn(
            limit,
            row.turn,
            _stored_observation(limit, f"message {turn}", turn),
        )

    expected = {
        ready: ("ready", True),
        pending: ("rehydrating", False),
        incompatible: ("incompatible", False),
        failed: ("blocked_failed", False),
        limit: ("turn_limit", False),
    }
    for session_id, state in expected.items():
        detail = service.get_session(session_id)
        assert (detail["continuation"], detail["can_send"]) == state
        assert all("feedback" in turn for turn in detail["turns"])


def test_create_resets_before_database_create(tmp_path: Path, monkeypatch) -> None:
    service, repository, worker = _service(tmp_path)
    events: list[str] = []
    real_reset = worker.reset_session
    real_create = repository.create_session

    def reset(*args: object, **kwargs: object) -> None:
        events.append("reset")
        real_reset(*args, **kwargs)

    def create_session(*args: object, **kwargs: object):
        events.append("create")
        return real_create(*args, **kwargs)

    monkeypatch.setattr(worker, "reset_session", reset)
    monkeypatch.setattr(repository, "create_session", create_session)

    result = service.create_session("Test", {"preference_tags": ["comfort"]})

    assert events == ["reset", "create"]
    assert result["session"]["profile"] == {"preference_tags": ["comfort"]}


def test_feedback_upserts_clears_and_is_returned_for_completed_duplicates(
    tmp_path: Path,
) -> None:
    service, _, _ = _service(tmp_path)
    session_id = _create(service)
    service.send_message(session_id, "r1", "shoes under $80")

    saved = service.set_feedback(
        session_id,
        1,
        "SHOE1",
        True,
        "over_budget",
        "Explicit budget was $80",
    )

    assert saved["reason"] == "over_budget"
    assert service.get_session(session_id)["turns"][0]["feedback"][0]["note"].endswith(
        "$80"
    )
    duplicate = service.send_message(session_id, "r1", "shoes under $80")
    assert duplicate["turn"]["feedback"] == [saved]
    assert service.set_feedback(session_id, 1, "SHOE1", False, None, "") is None
    assert service.get_session(session_id)["turns"][0]["feedback"] == []


def test_archive_scopes_block_mutation_until_unarchived(tmp_path: Path) -> None:
    service, _, _ = _service(
        tmp_path,
        ids=("archived-session", "active-session", "clone-session"),
    )
    archived = _create(service, "Archived")
    active = _create(service, "Active")

    patched = service.patch_session(archived, archived=True)

    assert patched["session"]["archived"] is True
    assert [item["session_id"] for item in service.list_sessions("active")] == [active]
    assert [item["session_id"] for item in service.list_sessions("archived")] == [
        archived
    ]
    assert {item["session_id"] for item in service.list_sessions("all")} == {
        active,
        archived,
    }
    assert service.get_session(archived)["can_send"] is False
    with pytest.raises(ConflictError):
        service.send_message(archived, "r1", "blue shoes")
    with pytest.raises(ConflictError):
        service.set_feedback(archived, 1, "SHOE1", False, None, "")
    with pytest.raises(ConflictError):
        service.clone_session(archived)

    service.patch_session(archived, archived=False)
    assert (
        service.send_message(archived, "r1", "blue shoes")["turn"]["status"]
        == "completed"
    )


def test_export_has_exact_envelope_safe_json_and_no_local_secrets(
    tmp_path: Path,
) -> None:
    service, _, _ = _service(tmp_path)
    session_id = service.create_session(
        "Export me",
        {
            "preference_tags": ["comfort"],
            "access_token": "secret-token-value",
            "database_path": r"C:\private\debug.sqlite3",
            "hostname": "demo.internal.example",
        },
    )["session"]["session_id"]
    service.send_message(session_id, "r1", "blue shoes")
    service.set_feedback(session_id, 1, "SHOE1", True, "other", "not for me")

    exported = service.export_session(session_id)
    encoded = json.dumps(exported, allow_nan=False)

    assert set(exported) == {
        "format",
        "schema_version",
        "exported_at",
        "session",
        "turns",
    }
    assert exported["format"] == "compasscart-debug-session"
    assert exported["schema_version"] == 1
    assert exported["exported_at"] == "2026-08-26T12:00:00Z"
    assert "secret-token-value" not in encoded
    assert r"C:\\private" not in encoded
    assert "demo.internal.example" not in encoded
    assert exported["turns"][0]["feedback"][0]["reason"] == "other"


def test_import_assigns_new_id_and_preserves_historical_observations(
    tmp_path: Path,
) -> None:
    service, _, worker = _service(tmp_path, ids=("source", "imported"))
    source = _create(service)
    service.send_message(source, "r1", "blue shoes")
    service.set_feedback(
        source,
        1,
        "SHOE1",
        True,
        "attribute_mismatch",
        "heel too high",
    )
    exported = service.export_session(source)
    reset_count = len(worker.reset_calls)

    imported = service.import_session(exported)

    assert imported["session"]["session_id"] == "imported"
    assert imported["session"]["session_id"] != exported["session"]["session_id"]
    assert imported["session"]["source_session_id"] == source
    assert imported["session"]["agent_version"] == exported["session"]["agent_version"]
    assert imported["turns"][0]["response"] == exported["turns"][0]["response"]
    assert imported["turns"][0]["feedback"][0]["note"] == "heel too high"
    assert imported["turns"][0]["request_id"] == "r1"
    assert len(worker.reset_calls) == reset_count
    with pytest.raises(ConflictError):
        service.send_message("imported", "new-request", "red shoes")
    with pytest.raises(ConflictError):
        service.set_feedback(
            "imported", 1, "SHOE1", True, "other", "must stay historical"
        )


def test_export_import_preserves_hostname_like_identifiers(tmp_path: Path) -> None:
    service, _, _ = _service(
        tmp_path, ids=("source.example.com", "imported.example.com")
    )
    source = _create(service)
    service.send_message(source, "request.example.com", "blue shoes")

    exported = service.export_session(source)
    imported = service.import_session(exported)

    assert exported["session"]["session_id"] == "source.example.com"
    assert exported["turns"][0]["request_id"] == "request.example.com"
    assert imported["turns"][0]["request_id"] == "request.example.com"


@pytest.mark.parametrize("status", ["pending", "failed"])
def test_import_accepts_safe_unresolved_historical_turns(
    tmp_path: Path, status: str
) -> None:
    service, repository, _ = _service(tmp_path / status, ids=("source", "imported"))
    source = _create(service)
    turn = repository.reserve_turn(source, "r1", "blue shoes")
    if status == "failed":
        repository.fail_turn(
            source,
            turn.turn,
            {"code": "agent_error", "message": "Historical failure."},
        )

    imported = service.import_session(service.export_session(source))

    assert imported["turns"][0]["status"] == status
    assert imported["turns"][0]["response"] is None


@pytest.mark.parametrize(
    "case",
    [
        "body",
        "format",
        "version",
        "order",
        "status",
        "completed_snapshot",
        "feedback_reason",
        "feedback_asin",
        "secret",
    ],
)
def test_import_rejects_malformed_history_without_partial_state(
    tmp_path: Path, case: str
) -> None:
    service, repository, _ = _service(tmp_path / case, ids=("source", "unused"))
    source = _create(service)
    service.send_message(source, "r1", "blue shoes")
    service.set_feedback(source, 1, "SHOE1", True, "other", "bad fit")
    malformed: object = copy.deepcopy(service.export_session(source))
    if case == "body":
        malformed = []
    elif case == "format":
        malformed["format"] = "not-compasscart"
    elif case == "version":
        malformed["schema_version"] = 999
    elif case == "order":
        malformed["turns"][0]["turn"] = 2
    elif case == "status":
        malformed["turns"][0]["status"] = "unknown"
    elif case == "completed_snapshot":
        malformed["turns"][0]["response"] = None
    elif case == "feedback_reason":
        malformed["turns"][0]["feedback"][0]["reason"] = "fit"
    elif case == "feedback_asin":
        malformed["turns"][0]["feedback"][0]["parent_asin"] = "MISSING"
    elif case == "secret":
        malformed["session"]["profile"]["access_token"] = "do-not-import"

    with pytest.raises(ValidationError) as captured:
        service.import_session(malformed)

    assert captured.value.field_errors
    assert [item.session_id for item in repository.list_sessions()] == [source]


def test_clone_replays_completed_prefix_with_current_identity_and_no_feedback(
    tmp_path: Path,
) -> None:
    service, _, worker = _service(
        tmp_path,
        ids=("source", "clone", "fresh-clone-request"),
    )
    source = _create(service, "Shopping")
    first = service.send_message(source, "source-r1", "blue shoes")
    service.send_message(source, "source-r2", "under $80")
    service.set_feedback(source, 1, "SHOE1", True, "other", "source only")
    worker.current_identity = RuntimeIdentity(
        "agent-v2", "catalog-v2", "config-v2", "assets-v2"
    )
    worker.reply_prefix = "current"

    clone = service.clone_session(source, through_turn=1)

    assert clone["session"]["session_id"] == "clone"
    assert clone["session"]["source_session_id"] == source
    assert clone["session"]["agent_version"] == "agent-v2"
    assert (
        clone["session"]["profile"] == service.get_session(source)["session"]["profile"]
    )
    assert len(clone["turns"]) == 1
    assert clone["turns"][0]["request_id"] != first["turn"]["request_id"]
    assert clone["turns"][0]["response"]["message"] == "current:blue shoes"
    assert clone["turns"][0]["state"]["session_id"] == "clone"
    assert clone["turns"][0]["feedback"] == []


def test_clone_failure_leaves_completed_prefix_and_failed_turn_viewable(
    tmp_path: Path,
) -> None:
    service, repository, worker = _service(
        tmp_path,
        ids=("source", "partial-clone", "fresh-r1", "fresh-r2"),
    )
    source = _create(service)
    service.send_message(source, "source-r1", "first")
    service.send_message(source, "source-r2", "second")
    worker.fail_on_message = "second"

    with pytest.raises(DebugServiceError):
        service.clone_session(source)

    partial = service.get_session("partial-clone")
    assert partial["session"]["source_session_id"] == source
    assert [turn["status"] for turn in partial["turns"]] == ["completed", "failed"]
    assert [turn.status for turn in repository.list_turns(source)] == [
        "completed",
        "completed",
    ]
