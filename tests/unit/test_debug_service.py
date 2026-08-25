from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import pytest

from compasscart_debug.agent_adapter import TurnObservation
from compasscart_debug.config import RuntimeIdentity
from compasscart_debug.errors import (
    DebugServiceError,
    NotReadyError,
    ReplayMismatchError,
    RequestMismatchError,
    UnresolvedTurnError,
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
        if self.fail_after_mutating_once:
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
                "message": f"reply:{message}",
                "ask_attribute": None,
                "recommendations": [],
                "usage": {},
            },
            products=[],
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
