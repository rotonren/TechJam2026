from __future__ import annotations

import importlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _repository_module():
    try:
        return importlib.import_module("compasscart_debug.repository")
    except ModuleNotFoundError:
        pytest.fail("The debug repository module has not been implemented.")


def _repository(path: Path):
    module = _repository_module()
    return module.DebugRepository(
        path,
        clock=lambda: datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc),
    )


def _create_session(repository, session_id: str = "session-1"):
    return repository.create_session(
        session_id=session_id,
        name="First session",
        profile={"budget": 100, "category": "shoes"},
        agent_version="debug-agent-1",
        catalog_sha256="c" * 64,
        config_sha256="d" * 64,
        assets_sha256="e" * 64,
    )


@dataclass
class _Observation:
    response: object
    products: object
    state: object
    trace: object


def _observation(parent_asin: str = "A1") -> _Observation:
    return _Observation(
        response={"message": "Found it"},
        products=[{"rank": 1, "parent_asin": parent_asin}],
        state={"route": "shopping"},
        trace={"duration_ms": 3},
    )


def test_initialize_creates_versioned_wal_database_and_reopens(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "debug.sqlite3"
    repository = _repository(path)

    repository.initialize()
    repository.initialize()

    assert path.is_file()
    assert repository.schema_version() == 1
    assert repository.health() is True
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        foreign_keys = connection.execute("PRAGMA foreign_key_list(turns)").fetchall()
    assert any(row[2] == "sessions" for row in foreign_keys)
    assert _repository(path).schema_version() == 1


@pytest.mark.parametrize("version", ["0", "2", "not-an-integer"])
def test_initialize_rejects_unsupported_schema_versions(
    tmp_path: Path, version: str
) -> None:
    path = tmp_path / "debug.sqlite3"
    repository = _repository(path)
    repository.initialize()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'", (version,)
        )

    with pytest.raises(_repository_module().RepositoryVersionError, match="schema"):
        _repository(path).initialize()


def test_repository_rejects_memory_database() -> None:
    with pytest.raises(ValueError, match="path-backed"):
        _repository_module().DebugRepository(":memory:")


def test_session_records_are_decoded_and_persisted_across_reopen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "debug.sqlite3"
    repository = _repository(path)
    repository.initialize()

    created = _create_session(repository)
    patched = repository.patch_session("session-1", name="Renamed", archived=True)
    dirty = repository.set_dirty("session-1")
    locked = repository.set_read_only_reason("session-1", "catalog changed")
    reopened = _repository(path)

    assert created.profile == {"budget": 100, "category": "shoes"}
    assert created.catalog_sha256 == "c" * 64
    assert patched.name == "Renamed"
    assert patched.archived is True
    assert dirty.dirty is True
    assert locked.read_only_reason == "catalog changed"
    assert reopened.get_session("session-1") == locked
    assert [record.session_id for record in reopened.list_sessions()] == ["session-1"]
    assert reopened.clear_dirty("session-1").dirty is False
    assert reopened.clear_read_only_reason("session-1").read_only_reason is None


class _TrackingConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()


def test_connections_close_after_success_and_exception(tmp_path: Path) -> None:
    connections: list[_TrackingConnection] = []

    def factory(*args, **kwargs) -> _TrackingConnection:
        connection = sqlite3.connect(*args, factory=_TrackingConnection, **kwargs)
        connections.append(connection)
        return connection

    repository = _repository_module().DebugRepository(
        tmp_path / "debug.sqlite3", connection_factory=factory
    )
    repository.initialize()
    connections.clear()

    assert repository.health() is True
    assert connections[-1].close_calls == 1
    with pytest.raises(_repository_module().NotFoundError):
        repository.get_session("missing")
    assert connections[-1].close_calls == 1


def test_reserve_turn_is_idempotent_for_the_same_request(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "debug.sqlite3")
    repository.initialize()
    _create_session(repository)

    first = repository.reserve_turn("session-1", "request-1", "show running shoes")
    again = repository.reserve_turn("session-1", "request-1", "show running shoes")

    assert first == again
    assert first.turn == 1
    assert first.status == "pending"
    assert len(repository.list_turns("session-1")) == 1


def test_reserve_turn_rejects_mismatched_or_unresolved_requests(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "debug.sqlite3")
    repository.initialize()
    _create_session(repository)
    pending = repository.reserve_turn("session-1", "request-1", "show shoes")
    errors = _repository_module()

    with pytest.raises(errors.RequestMismatchError):
        repository.reserve_turn("session-1", "request-1", "show jackets")
    with pytest.raises(errors.UnresolvedTurnError):
        repository.reserve_turn("session-1", "request-2", "show jackets")

    repository.fail_turn("session-1", pending.turn, {"code": "worker_failed"})
    with pytest.raises(errors.ConflictError):
        repository.reserve_turn("session-1", "request-1", "show shoes")
    with pytest.raises(errors.UnresolvedTurnError):
        repository.reserve_turn("session-1", "request-2", "show jackets")


def test_retry_failed_reopens_exact_request_and_completion_is_immutable(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "debug.sqlite3")
    repository.initialize()
    _create_session(repository)
    pending = repository.reserve_turn("session-1", "request-1", "show shoes")
    failed = repository.fail_turn("session-1", pending.turn, {"code": "worker_failed"})
    retried = repository.retry_failed("session-1", "request-1", "show shoes")
    completed = repository.complete_turn("session-1", pending.turn, _observation())
    errors = _repository_module()

    assert failed.error == {"code": "worker_failed"}
    assert retried.turn == failed.turn
    assert retried.created_at == failed.created_at
    assert retried.error is None
    assert completed.response == {"message": "Found it"}
    assert completed.products == [{"rank": 1, "parent_asin": "A1"}]
    assert repository.list_completed_turns("session-1") == [completed]
    with pytest.raises(errors.ConflictError):
        repository.complete_turn("session-1", pending.turn, _observation("A2"))
    with pytest.raises(errors.RequestMismatchError):
        repository.retry_failed("session-1", "request-1", "different message")


def test_reserve_turn_rejects_eleventh_turn(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "debug.sqlite3")
    repository.initialize()
    _create_session(repository)
    for number in range(1, 11):
        request_id = f"request-{number}"
        pending = repository.reserve_turn("session-1", request_id, f"message {number}")
        repository.complete_turn("session-1", pending.turn, _observation(f"A{number}"))

    with pytest.raises(_repository_module().TurnLimitError):
        repository.reserve_turn("session-1", "request-11", "message 11")


def test_completion_and_failure_decode_json_and_completion_rolls_back(
    tmp_path: Path,
) -> None:
    path = tmp_path / "debug.sqlite3"
    repository = _repository(path)
    repository.initialize()
    _create_session(repository)
    pending = repository.reserve_turn("session-1", "request-1", "show shoes")
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_observation BEFORE UPDATE OF response_json ON turns
            BEGIN SELECT RAISE(FAIL, 'observation rejected'); END;
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="observation rejected"):
        repository.complete_turn("session-1", pending.turn, _observation())
    pending = repository.get_turn_by_request_id("session-1", "request-1")
    failed = repository.fail_turn("session-1", pending.turn, {"detail": ["retry"]})

    assert pending.status == "pending"
    assert (
        pending.response is pending.products is pending.state is pending.trace is None
    )
    assert failed.status == "failed"
    assert failed.error == {"detail": ["retry"]}


def test_feedback_is_ranked_persistent_and_only_targets_completed_products(
    tmp_path: Path,
) -> None:
    path = tmp_path / "debug.sqlite3"
    repository = _repository(path)
    repository.initialize()
    _create_session(repository)
    pending = repository.reserve_turn("session-1", "request-1", "show shoes")
    errors = _repository_module()

    with pytest.raises(errors.ConflictError):
        repository.upsert_feedback("session-1", pending.turn, "A1", "fit", "too wide")
    repository.complete_turn(
        "session-1",
        pending.turn,
        _Observation(
            response={},
            products=[
                {"rank": 2, "parent_asin": "B2"},
                {"rank": 1, "parent_asin": "A1"},
            ],
            state={},
            trace={},
        ),
    )
    repository.upsert_feedback("session-1", 1, "B2", "price", "too expensive")
    updated = repository.upsert_feedback("session-1", 1, "A1", "fit", "too wide")
    repository.upsert_feedback("session-1", 1, "A1", "fit", "better now")

    assert updated.parent_asin == "A1"
    assert [item.parent_asin for item in repository.list_feedback("session-1", 1)] == [
        "A1",
        "B2",
    ]
    assert _repository(path).list_feedback("session-1", 1)[0].note == "better now"
    with pytest.raises(errors.ValidationError):
        repository.upsert_feedback("session-1", 1, "missing", "fit", "nope")
    repository.clear_feedback("session-1", 1, "B2")
    assert [item.parent_asin for item in repository.list_feedback("session-1", 1)] == [
        "A1"
    ]
    repository.clear_feedback("session-1", 1)
    assert repository.list_feedback("session-1", 1) == []


def test_reserve_completed_request_returns_the_exact_completed_turn(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "debug.sqlite3")
    repository.initialize()
    _create_session(repository)
    pending = repository.reserve_turn("session-1", "request-1", "show shoes")
    completed = repository.complete_turn("session-1", pending.turn, _observation())

    assert repository.reserve_turn("session-1", "request-1", "show shoes") == completed
    assert repository.list_turns("session-1") == [completed]


def test_retry_failed_rejects_pending_and_completed_turns(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "debug.sqlite3")
    repository.initialize()
    _create_session(repository)
    pending = repository.reserve_turn("session-1", "request-1", "show shoes")
    errors = _repository_module()

    with pytest.raises(errors.ConflictError):
        repository.retry_failed("session-1", "request-1", "show shoes")
    repository.complete_turn("session-1", pending.turn, _observation())
    with pytest.raises(errors.ConflictError):
        repository.retry_failed("session-1", "request-1", "show shoes")


def test_repeated_failure_and_completion_of_a_failed_turn_are_conflicts(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "debug.sqlite3")
    repository.initialize()
    _create_session(repository)
    pending = repository.reserve_turn("session-1", "request-1", "show shoes")
    repository.fail_turn("session-1", pending.turn, {"code": "worker_failed"})
    errors = _repository_module()

    with pytest.raises(errors.ConflictError):
        repository.fail_turn("session-1", pending.turn, {"code": "worker_failed"})
    with pytest.raises(errors.ConflictError):
        repository.complete_turn("session-1", pending.turn, _observation())


def test_failure_of_completed_turn_preserves_immutable_observation(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "debug.sqlite3")
    repository.initialize()
    _create_session(repository)
    pending = repository.reserve_turn("session-1", "request-1", "show shoes")
    completed = repository.complete_turn("session-1", pending.turn, _observation())

    with pytest.raises(_repository_module().ConflictError):
        repository.fail_turn("session-1", pending.turn, {"code": "late_failure"})

    after = repository.get_turn("session-1", pending.turn)
    assert (after.response, after.products, after.state, after.trace) == (
        completed.response,
        completed.products,
        completed.state,
        completed.trace,
    )


def test_concurrent_different_requests_leave_one_next_pending_turn(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "debug.sqlite3")
    repository.initialize()
    _create_session(repository)
    barrier = threading.Barrier(2)

    def reserve(request_id: str):
        barrier.wait()
        try:
            return repository.reserve_turn("session-1", request_id, request_id)
        except Exception as error:  # noqa: BLE001 - assert the concrete service error below.
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reserve, ["request-1", "request-2"]))

    records = [item for item in results if hasattr(item, "turn")]
    errors = [item for item in results if isinstance(item, Exception)]
    assert len(records) == 1
    assert records[0].turn == 1
    assert records[0].status == "pending"
    assert len(errors) == 1
    assert isinstance(errors[0], _repository_module().UnresolvedTurnError)
    assert [record.turn for record in repository.list_turns("session-1")] == [1]


def test_concurrent_identical_reservations_return_one_pending_turn(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "debug.sqlite3")
    repository.initialize()
    _create_session(repository)
    barrier = threading.Barrier(2)

    def reserve() -> object:
        barrier.wait()
        return repository.reserve_turn("session-1", "request-1", "show shoes")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: reserve(), range(2)))

    assert results[0] == results[1]
    assert results[0].turn == 1
    assert results[0].status == "pending"
    assert [record.turn for record in repository.list_turns("session-1")] == [1]
