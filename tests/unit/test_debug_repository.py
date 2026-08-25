from __future__ import annotations

import importlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
    assert _repository(path).health() is False


def _raw_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path)


def _schema_names(path: Path) -> list[str]:
    connection = _raw_connection(path)
    try:
        return [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
    finally:
        connection.close()


def _schema_objects(path: Path) -> list[tuple[str, str]]:
    connection = _raw_connection(path)
    try:
        return [
            (str(row[0]), str(row[1]))
            for row in connection.execute(
                """
                SELECT type, name FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            )
        ]
    finally:
        connection.close()


def test_initialize_rejects_newer_delete_journal_database_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v2.sqlite3"
    connection = _raw_connection(path)
    try:
        assert (
            connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
        )
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('schema_version', '2')"
        )
        connection.commit()
    finally:
        connection.close()
    before = _schema_names(path)

    with pytest.raises(_repository_module().RepositoryVersionError):
        _repository(path).initialize()

    connection = _raw_connection(path)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        connection.close()
    assert _schema_names(path) == before


def test_initialize_rejects_view_only_database_without_mutation(tmp_path: Path) -> None:
    path = tmp_path / "view.sqlite3"
    connection = _raw_connection(path)
    try:
        assert (
            connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
        )
        connection.execute("CREATE VIEW foreign_view AS SELECT 1 AS value")
        connection.commit()
    finally:
        connection.close()
    before = _schema_objects(path)

    with pytest.raises(_repository_module().RepositoryVersionError):
        _repository(path).initialize()

    connection = _raw_connection(path)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        connection.close()
    assert _schema_objects(path) == before
    assert _repository(path).health() is False


def test_initialize_rejects_nonempty_foreign_database_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "foreign.sqlite3"
    connection = _raw_connection(path)
    try:
        connection.execute("CREATE TABLE foreign_data (value TEXT)")
        connection.commit()
    finally:
        connection.close()
    before = _schema_names(path)

    with pytest.raises(_repository_module().RepositoryVersionError):
        _repository(path).initialize()

    assert _schema_names(path) == before


@pytest.mark.parametrize("table", ["sessions", "turns", "product_feedback"])
def test_initialize_and_health_reject_malformed_v1_schema(
    tmp_path: Path, table: str
) -> None:
    path = tmp_path / f"malformed-{table}.sqlite3"
    repository = _repository(path)
    repository.initialize()
    connection = _raw_connection(path)
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(f"DROP TABLE {table}")
        connection.execute(f"CREATE TABLE {table} (broken TEXT)")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(_repository_module().RepositoryVersionError):
        _repository(path).initialize()
    assert _repository(path).health() is False


def _replace_sessions_table(connection: sqlite3.Connection, foreign_key: str) -> None:
    connection.execute("PRAGMA foreign_keys=OFF")
    connection.execute("DROP TABLE sessions")
    connection.execute(
        f"""
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY, name TEXT NOT NULL, profile_json TEXT NOT NULL,
            agent_version TEXT NOT NULL, catalog_sha256 TEXT NOT NULL,
            config_sha256 TEXT NOT NULL, assets_sha256 TEXT, source_session_id TEXT,
            archived INTEGER NOT NULL DEFAULT 0 CHECK (archived IN (0,1)),
            dirty INTEGER NOT NULL DEFAULT 0 CHECK (dirty IN (0,1)),
            read_only_reason TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            {foreign_key}
        )
        """
    )


def test_initialize_rejects_sessions_without_required_default(tmp_path: Path) -> None:
    path = tmp_path / "missing-default.sqlite3"
    repository = _repository(path)
    repository.initialize()
    connection = _raw_connection(path)
    try:
        _replace_sessions_table(
            connection,
            "FOREIGN KEY (source_session_id) REFERENCES sessions(session_id)",
        )
        connection.execute("DROP TABLE sessions")
        connection.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY, name TEXT NOT NULL, profile_json TEXT NOT NULL,
                agent_version TEXT NOT NULL, catalog_sha256 TEXT NOT NULL,
                config_sha256 TEXT NOT NULL, assets_sha256 TEXT, source_session_id TEXT,
                archived INTEGER NOT NULL CHECK (archived IN (0,1)),
                dirty INTEGER NOT NULL DEFAULT 0 CHECK (dirty IN (0,1)),
                read_only_reason TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                FOREIGN KEY (source_session_id) REFERENCES sessions(session_id)
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(_repository_module().RepositoryVersionError):
        _repository(path).initialize()
    assert _repository(path).health() is False


def test_initialize_rejects_partial_unique_turn_index(tmp_path: Path) -> None:
    path = tmp_path / "partial-index.sqlite3"
    repository = _repository(path)
    repository.initialize()
    connection = _raw_connection(path)
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("DROP TABLE turns")
        connection.execute(
            """
            CREATE TABLE turns (
                session_id TEXT NOT NULL, turn INTEGER NOT NULL CHECK (turn BETWEEN 1 AND 10),
                request_id TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('pending','completed','failed')),
                user_message TEXT NOT NULL, response_json TEXT, products_json TEXT,
                state_json TEXT, trace_json TEXT, error_json TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY (session_id,turn),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX turns_session_request_partial
            ON turns(session_id, request_id) WHERE request_id IS NOT NULL
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(_repository_module().RepositoryVersionError):
        _repository(path).initialize()
    assert _repository(path).health() is False


def test_initialize_rejects_foreign_key_with_on_update_cascade(tmp_path: Path) -> None:
    path = tmp_path / "on-update.sqlite3"
    repository = _repository(path)
    repository.initialize()
    connection = _raw_connection(path)
    try:
        _replace_sessions_table(
            connection,
            """
            FOREIGN KEY (source_session_id) REFERENCES sessions(session_id)
            ON UPDATE CASCADE
            """,
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(_repository_module().RepositoryVersionError):
        _repository(path).initialize()
    assert _repository(path).health() is False


@pytest.mark.parametrize(
    "match_clause",
    ["MATCH FULL", "MATCH [FULL]", "MATCH `FULL`", "MATCH 'FULL'", "mAtCh\n fUlL"],
)
def test_initialize_rejects_explicit_foreign_key_match_without_mutation(
    tmp_path: Path, match_clause: str
) -> None:
    path = tmp_path / "match-full.sqlite3"
    repository = _repository(path)
    repository.initialize()
    connection = _raw_connection(path)
    try:
        _replace_sessions_table(
            connection,
            """
            FOREIGN KEY (source_session_id) REFERENCES sessions(session_id)
            """
            + match_clause,
        )
        connection.commit()
    finally:
        connection.close()
    before = _schema_objects(path)

    with pytest.raises(_repository_module().RepositoryVersionError):
        _repository(path).initialize()

    assert _schema_objects(path) == before
    assert _repository(path).health() is False


def test_health_and_schema_version_do_not_create_or_accept_unready_databases(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.sqlite3"
    empty = tmp_path / "empty.sqlite3"
    foreign = tmp_path / "foreign.sqlite3"
    corrupt = tmp_path / "corrupt.sqlite3"
    _raw_connection(empty).close()
    connection = _raw_connection(foreign)
    try:
        connection.execute("CREATE TABLE foreign_data (value TEXT)")
        connection.commit()
    finally:
        connection.close()
    corrupt.write_bytes(b"not a sqlite database")

    for path in (missing, empty, foreign, corrupt):
        repository = _repository(path)
        assert repository.health() is False
        with pytest.raises(_repository_module().RepositoryVersionError):
            repository.schema_version()

    assert not missing.exists()


def test_repository_rejects_memory_database() -> None:
    with pytest.raises(ValueError, match="path-backed"):
        _repository_module().DebugRepository(":memory:")


def test_repository_rejects_naive_clock_values(tmp_path: Path) -> None:
    repository = _repository_module().DebugRepository(
        tmp_path / "debug.sqlite3",
        clock=lambda: datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc).replace(
            tzinfo=None
        ),
    )
    repository.initialize()

    with pytest.raises(ValueError, match="timezone-aware"):
        _create_session(repository)


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


@dataclass
class _TransactionGate:
    armed: bool = False
    first_entered: threading.Event = field(default_factory=threading.Event)
    second_attempted: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    first_seen: bool = False


class _BlockingImmediateConnection(sqlite3.Connection):
    gate: _TransactionGate

    def execute(self, sql: str, parameters=()):
        if sql.strip().upper() != "BEGIN IMMEDIATE" or not self.gate.armed:
            return super().execute(sql, parameters)
        with self.gate.lock:
            first = not self.gate.first_seen
            self.gate.first_seen = True
            if not first:
                self.gate.second_attempted.set()
        result = super().execute(sql, parameters)
        if first:
            self.gate.first_entered.set()
            assert self.gate.release.wait(5), (
                "test did not release the held transaction"
            )
        return result


class _LockedImmediateConnection(sqlite3.Connection):
    def execute(self, sql: str, parameters=()):
        if sql.strip().upper() == "BEGIN IMMEDIATE":
            raise sqlite3.OperationalError("database is locked")
        return super().execute(sql, parameters)


def _blocking_repository(path: Path, gate: _TransactionGate):
    def factory(*args, **kwargs) -> _BlockingImmediateConnection:
        connection = sqlite3.connect(
            *args, factory=_BlockingImmediateConnection, **kwargs
        )
        connection.gate = gate
        return connection

    return _repository_module().DebugRepository(
        path,
        clock=lambda: datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc),
        connection_factory=factory,
    )


def test_feedback_upserts_overlap_after_first_transaction_acquires_lock(
    tmp_path: Path,
) -> None:
    path = tmp_path / "debug.sqlite3"
    gate = _TransactionGate()
    repository = _blocking_repository(path, gate)
    repository.initialize()
    _create_session(repository)
    pending = repository.reserve_turn("session-1", "request-1", "show shoes")
    repository.complete_turn(
        "session-1",
        pending.turn,
        _Observation(
            response={},
            products=[
                {"rank": 1, "parent_asin": "A1"},
                {"rank": 2, "parent_asin": "B2"},
            ],
            state={},
            trace={},
        ),
    )
    gate.armed = True

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            repository.upsert_feedback,
            "session-1",
            pending.turn,
            "A1",
            "other",
            "first",
        )
        assert gate.first_entered.wait(2)
        second = executor.submit(
            repository.upsert_feedback,
            "session-1",
            pending.turn,
            "B2",
            "other",
            "second",
        )
        assert gate.second_attempted.wait(2)
        gate.release.set()
        assert first.result(timeout=5).parent_asin == "A1"
        assert second.result(timeout=5).parent_asin == "B2"

    assert [item.parent_asin for item in repository.list_feedback("session-1", 1)] == [
        "A1",
        "B2",
    ]


def test_failed_turn_retries_race_after_first_transaction_acquires_lock(
    tmp_path: Path,
) -> None:
    path = tmp_path / "debug.sqlite3"
    gate = _TransactionGate()
    repository = _blocking_repository(path, gate)
    repository.initialize()
    _create_session(repository)
    pending = repository.reserve_turn("session-1", "request-1", "show shoes")
    repository.fail_turn("session-1", pending.turn, {"code": "worker_failed"})
    gate.armed = True

    def retry() -> object:
        try:
            return repository.retry_failed("session-1", "request-1", "show shoes")
        except Exception as error:  # noqa: BLE001 - assert the state-machine error below.
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(retry)
        assert gate.first_entered.wait(2)
        second = executor.submit(retry)
        assert gate.second_attempted.wait(2)
        gate.release.set()
        results = [first.result(timeout=5), second.result(timeout=5)]

    assert sum(hasattr(result, "turn") for result in results) == 1
    assert (
        sum(
            isinstance(result, _repository_module().ConflictError) for result in results
        )
        == 1
    )
    assert repository.get_turn("session-1", pending.turn).status == "pending"


def test_locked_storage_is_mapped_to_a_stable_busy_error(tmp_path: Path) -> None:
    path = tmp_path / "debug.sqlite3"
    repository = _repository(path)
    repository.initialize()
    _create_session(repository)

    def factory(*args, **kwargs) -> _LockedImmediateConnection:
        return sqlite3.connect(*args, factory=_LockedImmediateConnection, **kwargs)

    locked_repository = _repository_module().DebugRepository(
        path, connection_factory=factory
    )
    try:
        locked_repository.set_dirty("session-1")
    except Exception as error:  # noqa: BLE001 - require the repository boundary error.
        assert type(error).__name__ == "RepositoryBusyError"
    else:
        pytest.fail("A locked SQLite operation escaped the repository boundary.")


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
        repository.upsert_feedback(
            "session-1", pending.turn, "A1", "explicit_constraint", "too wide"
        )
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
    repository.upsert_feedback("session-1", 1, "B2", "over_budget", "too expensive")
    updated = repository.upsert_feedback(
        "session-1", 1, "A1", "attribute_mismatch", "too wide"
    )
    repository.upsert_feedback("session-1", 1, "A1", "attribute_mismatch", "better now")

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


def test_feedback_reasons_are_fixed_and_notes_may_be_empty(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "debug.sqlite3")
    repository.initialize()
    _create_session(repository)
    pending = repository.reserve_turn("session-1", "request-1", "show shoes")
    repository.complete_turn("session-1", pending.turn, _observation())

    accepted = repository.upsert_feedback("session-1", 1, "A1", "other", "")
    assert accepted.note == ""
    with pytest.raises(_repository_module().ValidationError):
        repository.upsert_feedback("session-1", 1, "A1", "fit", "not valid")
    with pytest.raises(_repository_module().ValidationError):
        repository.upsert_feedback("session-1", 1, "A1", 42, "not valid")


@pytest.mark.parametrize("payload", ["NaN", "{not-json"])
def test_tampered_turn_json_raises_stable_corruption_error(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "debug.sqlite3"
    repository = _repository(path)
    repository.initialize()
    _create_session(repository)
    pending = repository.reserve_turn("session-1", "request-1", "show shoes")
    repository.complete_turn("session-1", pending.turn, _observation())
    connection = _raw_connection(path)
    try:
        connection.execute(
            "UPDATE turns SET response_json = ? WHERE session_id = ? AND turn = ?",
            (payload, "session-1", pending.turn),
        )
        connection.commit()
    finally:
        connection.close()

    try:
        repository.get_turn("session-1", pending.turn)
    except Exception as error:  # noqa: BLE001 - require the repository boundary error.
        assert type(error).__name__ == "RepositoryCorruptionError"
    else:
        pytest.fail("Tampered JSON was returned instead of rejected.")


def test_tampered_session_profile_raises_stable_corruption_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "debug.sqlite3"
    repository = _repository(path)
    repository.initialize()
    _create_session(repository)
    connection = _raw_connection(path)
    try:
        connection.execute(
            "UPDATE sessions SET profile_json = 'NaN' WHERE session_id = 'session-1'"
        )
        connection.commit()
    finally:
        connection.close()

    try:
        repository.get_session("session-1")
    except Exception as error:  # noqa: BLE001 - require the repository boundary error.
        assert type(error).__name__ == "RepositoryCorruptionError"
    else:
        pytest.fail("Tampered session JSON was returned instead of rejected.")


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
    gate = _TransactionGate()
    repository = _blocking_repository(tmp_path / "debug.sqlite3", gate)
    repository.initialize()
    _create_session(repository)
    gate.armed = True

    def reserve(request_id: str):
        try:
            return repository.reserve_turn("session-1", request_id, request_id)
        except Exception as error:  # noqa: BLE001 - assert the concrete service error below.
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(reserve, "request-1")
        assert gate.first_entered.wait(2)
        second = executor.submit(reserve, "request-2")
        assert gate.second_attempted.wait(2)
        gate.release.set()
        results = [first.result(timeout=5), second.result(timeout=5)]

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
    gate = _TransactionGate()
    repository = _blocking_repository(tmp_path / "debug.sqlite3", gate)
    repository.initialize()
    _create_session(repository)
    gate.armed = True

    def reserve() -> object:
        return repository.reserve_turn("session-1", "request-1", "show shoes")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(reserve)
        assert gate.first_entered.wait(2)
        second = executor.submit(reserve)
        assert gate.second_attempted.wait(2)
        gate.release.set()
        results = [first.result(timeout=5), second.result(timeout=5)]

    assert results[0] == results[1]
    assert results[0].turn == 1
    assert results[0].status == "pending"
    assert [record.turn for record in repository.list_turns("session-1")] == [1]
