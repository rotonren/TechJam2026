from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .errors import (
    ConflictError,
    NotFoundError,
    RequestMismatchError,
    TurnLimitError,
    UnresolvedTurnError,
    ValidationError,
)

_SCHEMA_VERSION = 1
_UNSET = object()
_FEEDBACK_REASONS = frozenset(
    {
        "explicit_constraint",
        "wrong_category",
        "over_budget",
        "attribute_mismatch",
        "duplicate_or_too_similar",
        "other",
    }
)
_EXPLICIT_MATCH_CLAUSE = re.compile(
    r"\bmatch\b\s+(?:[a-z_][a-z0-9_]*|\"[^\"]+\")", re.IGNORECASE
)
_REQUIRED_TABLES = {"metadata", "sessions", "turns", "product_feedback"}
_TABLE_COLUMNS = {
    "metadata": (
        ("key", "TEXT", False, 1, None),
        ("value", "TEXT", True, 0, None),
    ),
    "sessions": (
        ("session_id", "TEXT", False, 1, None),
        ("name", "TEXT", True, 0, None),
        ("profile_json", "TEXT", True, 0, None),
        ("agent_version", "TEXT", True, 0, None),
        ("catalog_sha256", "TEXT", True, 0, None),
        ("config_sha256", "TEXT", True, 0, None),
        ("assets_sha256", "TEXT", False, 0, None),
        ("source_session_id", "TEXT", False, 0, None),
        ("archived", "INTEGER", True, 0, "0"),
        ("dirty", "INTEGER", True, 0, "0"),
        ("read_only_reason", "TEXT", False, 0, None),
        ("created_at", "TEXT", True, 0, None),
        ("updated_at", "TEXT", True, 0, None),
    ),
    "turns": (
        ("session_id", "TEXT", True, 1, None),
        ("turn", "INTEGER", True, 2, None),
        ("request_id", "TEXT", True, 0, None),
        ("status", "TEXT", True, 0, None),
        ("user_message", "TEXT", True, 0, None),
        ("response_json", "TEXT", False, 0, None),
        ("products_json", "TEXT", False, 0, None),
        ("state_json", "TEXT", False, 0, None),
        ("trace_json", "TEXT", False, 0, None),
        ("error_json", "TEXT", False, 0, None),
        ("created_at", "TEXT", True, 0, None),
        ("updated_at", "TEXT", True, 0, None),
    ),
    "product_feedback": (
        ("session_id", "TEXT", True, 1, None),
        ("turn", "INTEGER", True, 2, None),
        ("parent_asin", "TEXT", True, 3, None),
        ("reason", "TEXT", True, 0, None),
        ("note", "TEXT", True, 0, None),
        ("updated_at", "TEXT", True, 0, None),
    ),
}
_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY, name TEXT NOT NULL, profile_json TEXT NOT NULL,
        agent_version TEXT NOT NULL, catalog_sha256 TEXT NOT NULL,
        config_sha256 TEXT NOT NULL, assets_sha256 TEXT, source_session_id TEXT,
        archived INTEGER NOT NULL DEFAULT 0 CHECK (archived IN (0,1)),
        dirty INTEGER NOT NULL DEFAULT 0 CHECK (dirty IN (0,1)),
        read_only_reason TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        FOREIGN KEY (source_session_id) REFERENCES sessions(session_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS turns (
        session_id TEXT NOT NULL, turn INTEGER NOT NULL CHECK (turn BETWEEN 1 AND 10),
        request_id TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('pending','completed','failed')),
        user_message TEXT NOT NULL, response_json TEXT, products_json TEXT,
        state_json TEXT, trace_json TEXT, error_json TEXT,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY (session_id,turn), UNIQUE (session_id,request_id),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS product_feedback (
        session_id TEXT NOT NULL, turn INTEGER NOT NULL, parent_asin TEXT NOT NULL,
        reason TEXT NOT NULL, note TEXT NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY (session_id,turn,parent_asin),
        FOREIGN KEY (session_id,turn) REFERENCES turns(session_id,turn) ON DELETE CASCADE
    )
    """,
)


class RepositoryVersionError(RuntimeError):
    """Raised when a database needs a newer repository implementation."""


class RepositoryCorruptionError(RuntimeError):
    """Raised when persisted repository data cannot be decoded safely."""


class RepositoryBusyError(RuntimeError):
    """Raised when SQLite cannot acquire a repository write lock."""


class RepositoryStorageError(RuntimeError):
    """Raised when SQLite storage cannot safely complete an operation."""


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    name: str
    profile: Any
    agent_version: str
    catalog_sha256: str
    config_sha256: str
    assets_sha256: str | None
    source_session_id: str | None
    archived: bool
    dirty: bool
    read_only_reason: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class TurnRecord:
    session_id: str
    turn: int
    request_id: str
    status: str
    user_message: str
    response: Any | None
    products: Any | None
    state: Any | None
    trace: Any | None
    error: Any | None
    created_at: str
    updated_at: str


class TurnObservation(Protocol):
    response: Any
    products: Any
    state: Any
    trace: Any


@dataclass(frozen=True)
class FeedbackRecord:
    session_id: str
    turn: int
    parent_asin: str
    reason: str
    note: str
    updated_at: str


class DebugRepository:
    def __init__(
        self,
        path: Path | str,
        *,
        clock: Callable[[], datetime] | None = None,
        connection_factory: Callable[..., sqlite3.Connection] = sqlite3.connect,
    ) -> None:
        self.path = Path(path)
        if str(self.path) == ":memory:":
            raise ValueError("The debug repository requires a path-backed database.")
        self._clock = clock or (lambda: datetime.now(UTC))
        self._connection_factory = connection_factory

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect(mode="rwc") as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if not _user_objects(connection):
                    for statement in _SCHEMA_STATEMENTS:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                        (str(_SCHEMA_VERSION),),
                    )
                else:
                    _validate_existing_v1(connection)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            connection.execute("PRAGMA journal_mode=WAL")

    def schema_version(self) -> int:
        try:
            with self._connect(mode="ro") as connection:
                return _validate_existing_v1(connection)
        except RepositoryVersionError:
            raise
        except (RepositoryStorageError, sqlite3.DatabaseError) as error:
            raise RepositoryVersionError(
                "The database schema is unavailable."
            ) from error

    def health(self) -> bool:
        try:
            with self._connect(mode="ro") as connection:
                _validate_existing_v1(connection)
                result = connection.execute("PRAGMA quick_check").fetchone()
                return result is not None and result[0] == "ok"
        except (
            RepositoryVersionError,
            RepositoryStorageError,
            sqlite3.DatabaseError,
            OSError,
        ):
            return False

    def create_session(
        self,
        *,
        session_id: str,
        name: str,
        profile: Any,
        agent_version: str,
        catalog_sha256: str,
        config_sha256: str,
        assets_sha256: str | None = None,
        source_session_id: str | None = None,
    ) -> SessionRecord:
        now = self._timestamp()
        with self._connect() as connection:
            self._begin(connection)
            try:
                connection.execute(
                    """
                    INSERT INTO sessions(
                        session_id, name, profile_json, agent_version, catalog_sha256,
                        config_sha256, assets_sha256, source_session_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        name,
                        _encode_json(profile),
                        agent_version,
                        catalog_sha256,
                        config_sha256,
                        assets_sha256,
                        source_session_id,
                        now,
                        now,
                    ),
                )
                connection.commit()
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise ConflictError() from error
            except BaseException:
                connection.rollback()
                raise
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> SessionRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError()
        return _decode_session(row)

    def list_sessions(self) -> list[SessionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions ORDER BY created_at, session_id"
            ).fetchall()
        return [_decode_session(row) for row in rows]

    def patch_session(
        self,
        session_id: str,
        *,
        name: str | object = _UNSET,
        archived: bool | object = _UNSET,
    ) -> SessionRecord:
        updates: list[str] = []
        values: list[Any] = []
        if name is not _UNSET:
            updates.append("name = ?")
            values.append(name)
        if archived is not _UNSET:
            updates.append("archived = ?")
            values.append(int(bool(archived)))
        if not updates:
            return self.get_session(session_id)
        updates.append("updated_at = ?")
        values.extend((self._timestamp(), session_id))
        self._update_session(session_id, ", ".join(updates), values)
        return self.get_session(session_id)

    def set_dirty(self, session_id: str) -> SessionRecord:
        return self._set_session_value(session_id, "dirty", 1)

    def clear_dirty(self, session_id: str) -> SessionRecord:
        return self._set_session_value(session_id, "dirty", 0)

    def set_read_only_reason(self, session_id: str, reason: str) -> SessionRecord:
        if not reason:
            raise ValidationError({"reason": "A read-only reason is required."})
        return self._set_session_value(session_id, "read_only_reason", reason)

    def clear_read_only_reason(self, session_id: str) -> SessionRecord:
        return self._set_session_value(session_id, "read_only_reason", None)

    def get_turn(self, session_id: str, turn: int) -> TurnRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM turns WHERE session_id = ? AND turn = ?",
                (session_id, turn),
            ).fetchone()
        if row is None:
            raise NotFoundError()
        return _decode_turn(row)

    def get_turn_by_request_id(self, session_id: str, request_id: str) -> TurnRecord:
        record = self.find_turn_by_request_id(session_id, request_id)
        if record is None:
            raise NotFoundError()
        return record

    def find_turn_by_request_id(
        self, session_id: str, request_id: str
    ) -> TurnRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM turns WHERE session_id = ? AND request_id = ?",
                (session_id, request_id),
            ).fetchone()
        return _decode_turn(row) if row is not None else None

    def list_turns(self, session_id: str) -> list[TurnRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY turn", (session_id,)
            ).fetchall()
        return [_decode_turn(row) for row in rows]

    def list_completed_turns(self, session_id: str) -> list[TurnRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM turns
                WHERE session_id = ? AND status = 'completed'
                ORDER BY turn
                """,
                (session_id,),
            ).fetchall()
        return [_decode_turn(row) for row in rows]

    def reserve_turn(
        self, session_id: str, request_id: str, user_message: str
    ) -> TurnRecord:
        now = self._timestamp()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_session(connection, session_id)
                existing = connection.execute(
                    "SELECT * FROM turns WHERE session_id = ? AND request_id = ?",
                    (session_id, request_id),
                ).fetchone()
                if existing is not None:
                    if existing["user_message"] != user_message:
                        raise RequestMismatchError()
                    if existing["status"] == "failed":
                        raise ConflictError()
                    result = _decode_turn(existing)
                else:
                    unresolved = connection.execute(
                        """
                        SELECT 1 FROM turns
                        WHERE session_id = ? AND status IN ('pending', 'failed')
                        LIMIT 1
                        """,
                        (session_id,),
                    ).fetchone()
                    if unresolved is not None:
                        raise UnresolvedTurnError()
                    maximum = connection.execute(
                        "SELECT COALESCE(MAX(turn), 0) AS value FROM turns WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()["value"]
                    next_turn = int(maximum) + 1
                    if next_turn > 10:
                        raise TurnLimitError()
                    connection.execute(
                        """
                        INSERT INTO turns(
                            session_id, turn, request_id, status, user_message, created_at, updated_at
                        ) VALUES (?, ?, ?, 'pending', ?, ?, ?)
                        """,
                        (session_id, next_turn, request_id, user_message, now, now),
                    )
                    result = _decode_turn(
                        connection.execute(
                            "SELECT * FROM turns WHERE session_id = ? AND turn = ?",
                            (session_id, next_turn),
                        ).fetchone()
                    )
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

    def retry_failed(
        self, session_id: str, request_id: str, user_message: str
    ) -> TurnRecord:
        with self._connect() as connection:
            self._begin(connection)
            try:
                row = self._require_request(connection, session_id, request_id)
                if row["user_message"] != user_message:
                    raise RequestMismatchError()
                if row["status"] != "failed":
                    raise ConflictError()
                connection.execute(
                    """
                    UPDATE turns SET status = 'pending', error_json = NULL, updated_at = ?
                    WHERE session_id = ? AND request_id = ? AND status = 'failed'
                    """,
                    (self._timestamp(), session_id, request_id),
                )
                row = connection.execute(
                    "SELECT * FROM turns WHERE session_id = ? AND request_id = ?",
                    (session_id, request_id),
                ).fetchone()
                connection.commit()
                return _decode_turn(row)
            except BaseException:
                connection.rollback()
                raise

    def complete_turn(
        self, session_id: str, turn: int, observation: TurnObservation
    ) -> TurnRecord:
        response = _encode_json(observation.response)
        products = _encode_json(observation.products)
        state = _encode_json(observation.state)
        trace = _encode_json(observation.trace)
        with self._connect() as connection:
            self._begin(connection)
            try:
                row = self._require_turn(connection, session_id, turn)
                if row["status"] != "pending":
                    raise ConflictError()
                result = connection.execute(
                    """
                    UPDATE turns
                    SET status = 'completed', response_json = ?, products_json = ?,
                        state_json = ?, trace_json = ?, error_json = NULL, updated_at = ?
                    WHERE session_id = ? AND turn = ? AND status = 'pending'
                    """,
                    (
                        response,
                        products,
                        state,
                        trace,
                        self._timestamp(),
                        session_id,
                        turn,
                    ),
                )
                if result.rowcount != 1:
                    raise ConflictError()
                row = connection.execute(
                    "SELECT * FROM turns WHERE session_id = ? AND turn = ?",
                    (session_id, turn),
                ).fetchone()
                connection.commit()
                return _decode_turn(row)
            except BaseException:
                connection.rollback()
                raise

    def fail_turn(self, session_id: str, turn: int, error: Any) -> TurnRecord:
        error_json = _encode_json(error)
        with self._connect() as connection:
            self._begin(connection)
            try:
                row = self._require_turn(connection, session_id, turn)
                if row["status"] != "pending":
                    raise ConflictError()
                result = connection.execute(
                    """
                    UPDATE turns SET status = 'failed', error_json = ?, updated_at = ?
                    WHERE session_id = ? AND turn = ? AND status = 'pending'
                    """,
                    (error_json, self._timestamp(), session_id, turn),
                )
                if result.rowcount != 1:
                    raise ConflictError()
                row = connection.execute(
                    "SELECT * FROM turns WHERE session_id = ? AND turn = ?",
                    (session_id, turn),
                ).fetchone()
                connection.commit()
                return _decode_turn(row)
            except BaseException:
                connection.rollback()
                raise

    def upsert_feedback(
        self, session_id: str, turn: int, parent_asin: str, reason: str, note: str
    ) -> FeedbackRecord:
        if not isinstance(parent_asin, str):
            raise ValidationError({"parent_asin": "A product identifier is required."})
        if not isinstance(reason, str) or reason not in _FEEDBACK_REASONS:
            raise ValidationError({"reason": "Feedback reason is invalid."})
        if not isinstance(note, str):
            raise ValidationError({"note": "Feedback note must be text."})
        with self._connect() as connection:
            self._begin(connection)
            try:
                row = self._require_turn(connection, session_id, turn)
                if row["status"] != "completed":
                    raise ConflictError()
                if parent_asin not in _product_ranks(
                    _decode_json(row["products_json"])
                ):
                    raise ValidationError(
                        {"parent_asin": "Product is not present in this turn."}
                    )
                now = self._timestamp()
                connection.execute(
                    """
                    INSERT INTO product_feedback(
                        session_id, turn, parent_asin, reason, note, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, turn, parent_asin) DO UPDATE SET
                        reason = excluded.reason, note = excluded.note,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, turn, parent_asin, reason, note, now),
                )
                row = connection.execute(
                    """
                    SELECT * FROM product_feedback
                    WHERE session_id = ? AND turn = ? AND parent_asin = ?
                    """,
                    (session_id, turn, parent_asin),
                ).fetchone()
                connection.commit()
                return _decode_feedback(row)
            except BaseException:
                connection.rollback()
                raise

    def list_feedback(self, session_id: str, turn: int) -> list[FeedbackRecord]:
        with self._connect() as connection:
            turn_row = self._require_turn(connection, session_id, turn)
            rows = connection.execute(
                """
                SELECT * FROM product_feedback
                WHERE session_id = ? AND turn = ?
                ORDER BY parent_asin
                """,
                (session_id, turn),
            ).fetchall()
        ranks = _product_ranks(_decode_json(turn_row["products_json"]))
        records = [_decode_feedback(row) for row in rows]
        return sorted(
            records,
            key=lambda record: (
                ranks.get(record.parent_asin, 1_000_000),
                record.parent_asin,
            ),
        )

    def clear_feedback(
        self, session_id: str, turn: int, parent_asin: str | None = None
    ) -> None:
        with self._connect() as connection:
            self._begin(connection)
            try:
                self._require_turn(connection, session_id, turn)
                if parent_asin is None:
                    connection.execute(
                        "DELETE FROM product_feedback WHERE session_id = ? AND turn = ?",
                        (session_id, turn),
                    )
                else:
                    connection.execute(
                        """
                        DELETE FROM product_feedback
                        WHERE session_id = ? AND turn = ? AND parent_asin = ?
                        """,
                        (session_id, turn, parent_asin),
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _set_session_value(
        self, session_id: str, column: str, value: Any
    ) -> SessionRecord:
        self._update_session(
            session_id,
            f"{column} = ?, updated_at = ?",
            [value, self._timestamp(), session_id],
        )
        return self.get_session(session_id)

    def _update_session(
        self, session_id: str, assignments: str, values: list[Any]
    ) -> None:
        with self._connect() as connection:
            self._begin(connection)
            try:
                result = connection.execute(
                    f"UPDATE sessions SET {assignments} WHERE session_id = ?", values
                )
                if result.rowcount != 1:
                    raise NotFoundError()
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _require_session(connection: sqlite3.Connection, session_id: str) -> None:
        row = connection.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError()

    @staticmethod
    def _require_request(
        connection: sqlite3.Connection, session_id: str, request_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM turns WHERE session_id = ? AND request_id = ?",
            (session_id, request_id),
        ).fetchone()
        if row is None:
            raise NotFoundError()
        return row

    @staticmethod
    def _require_turn(
        connection: sqlite3.Connection, session_id: str, turn: int
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM turns WHERE session_id = ? AND turn = ?", (session_id, turn)
        ).fetchone()
        if row is None:
            raise NotFoundError()
        return row

    @contextmanager
    def _connect(self, *, mode: str = "rw") -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connection_factory(
                _sqlite_uri(self.path, mode), timeout=5, uri=True
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=5000")
            yield connection
        except sqlite3.IntegrityError:
            raise
        except sqlite3.OperationalError as error:
            if _is_busy_error(error):
                raise RepositoryBusyError("The repository is busy.") from error
            raise RepositoryStorageError(
                "The repository storage is unavailable."
            ) from error
        except sqlite3.DatabaseError as error:
            raise RepositoryStorageError(
                "The repository storage is unavailable."
            ) from error
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _begin(connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")

    def _timestamp(self) -> str:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "The repository clock must return a timezone-aware datetime."
            )
        value = value.astimezone(UTC).replace(microsecond=0)
        return value.isoformat().replace("+00:00", "Z")


def _encode_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _validate_schema_version(value: str) -> int:
    if value != str(_SCHEMA_VERSION):
        raise RepositoryVersionError("The database schema is incompatible.")
    return _SCHEMA_VERSION


def _sqlite_uri(path: Path, mode: str) -> str:
    return f"{path.resolve().as_uri()}?mode={mode}"


def _user_objects(connection: sqlite3.Connection) -> dict[str, tuple[str, str]]:
    rows = connection.execute(
        """
        SELECT type, name, sql FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        """
    ).fetchall()
    return {str(row["name"]): (str(row["type"]), str(row["sql"] or "")) for row in rows}


def _validate_existing_v1(connection: sqlite3.Connection) -> int:
    objects = _user_objects(connection)
    if set(objects) != _REQUIRED_TABLES or any(
        object_type != "table" for object_type, _ in objects.values()
    ):
        raise RepositoryVersionError("The database schema is incompatible.")
    tables = {name: sql for name, (_, sql) in objects.items()}
    for table, expected_columns in _TABLE_COLUMNS.items():
        rows = connection.execute(
            f"PRAGMA table_info({_quote_identifier(table)})"
        ).fetchall()
        actual_columns = tuple(
            (
                str(row["name"]),
                str(row["type"]).upper(),
                bool(row["notnull"]),
                row["pk"],
                _normalize_default(row["dflt_value"]),
            )
            for row in rows
        )
        if actual_columns != expected_columns:
            raise RepositoryVersionError("The database schema is incompatible.")
    row = connection.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        raise RepositoryVersionError("The database schema is incompatible.")
    _validate_schema_version(row["value"])
    _validate_schema_constraints(connection, tables)
    return _SCHEMA_VERSION


def _validate_schema_constraints(
    connection: sqlite3.Connection, tables: dict[str, str]
) -> None:
    sessions_sql = _normalize_sql(tables["sessions"])
    turns_sql = _normalize_sql(tables["turns"])
    if (
        "check(archivedin(0,1))" not in sessions_sql
        or "check(dirtyin(0,1))" not in sessions_sql
        or "check(turnbetween1and10)" not in turns_sql
        or "check(statusin('pending','completed','failed'))" not in turns_sql
    ):
        raise RepositoryVersionError("The database schema is incompatible.")
    if any(_EXPLICIT_MATCH_CLAUSE.search(sql) for sql in tables.values()):
        raise RepositoryVersionError("The database schema is incompatible.")
    if not _has_unique_index(connection, "turns", ("session_id", "request_id")):
        raise RepositoryVersionError("The database schema is incompatible.")
    if not _has_foreign_key(
        connection,
        "sessions",
        "sessions",
        (("source_session_id", "session_id"),),
        "NO ACTION",
    ):
        raise RepositoryVersionError("The database schema is incompatible.")
    if not _has_foreign_key(
        connection,
        "turns",
        "sessions",
        (("session_id", "session_id"),),
        "CASCADE",
    ):
        raise RepositoryVersionError("The database schema is incompatible.")
    if not _has_foreign_key(
        connection,
        "product_feedback",
        "turns",
        (("session_id", "session_id"), ("turn", "turn")),
        "CASCADE",
    ):
        raise RepositoryVersionError("The database schema is incompatible.")


def _has_unique_index(
    connection: sqlite3.Connection, table: str, expected: tuple[str, ...]
) -> bool:
    rows = connection.execute(
        f"PRAGMA index_list({_quote_identifier(table)})"
    ).fetchall()
    for row in rows:
        if not row["unique"] or row["origin"] != "u" or row["partial"] != 0:
            continue
        index_rows = connection.execute(
            f"PRAGMA index_info({_quote_identifier(str(row['name']))})"
        ).fetchall()
        columns = tuple(str(index_row["name"]) for index_row in index_rows)
        if columns == expected:
            return True
    return False


def _has_foreign_key(
    connection: sqlite3.Connection,
    table: str,
    target_table: str,
    expected: tuple[tuple[str, str], ...],
    on_delete: str,
) -> bool:
    rows = connection.execute(
        f"PRAGMA foreign_key_list({_quote_identifier(table)})"
    ).fetchall()
    grouped: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(int(row["id"]), []).append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row["seq"]))
        pairs = tuple((str(row["from"]), str(row["to"])) for row in group)
        if (
            str(group[0]["table"]) == target_table
            and pairs == expected
            and all(str(row["on_delete"]) == on_delete for row in group)
            and all(str(row["on_update"]) == "NO ACTION" for row in group)
            and all(str(row["match"]) == "NONE" for row in group)
        ):
            return True
    return False


def _normalize_sql(value: str) -> str:
    return "".join(value.lower().split())


def _normalize_default(value: object) -> str | None:
    return None if value is None else str(value).strip().lower()


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _is_busy_error(error: sqlite3.OperationalError) -> bool:
    message = str(error).lower()
    return "locked" in message or "busy" in message


def _decode_session(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        session_id=row["session_id"],
        name=row["name"],
        profile=_decode_json(row["profile_json"]),
        agent_version=row["agent_version"],
        catalog_sha256=row["catalog_sha256"],
        config_sha256=row["config_sha256"],
        assets_sha256=row["assets_sha256"],
        source_session_id=row["source_session_id"],
        archived=bool(row["archived"]),
        dirty=bool(row["dirty"]),
        read_only_reason=row["read_only_reason"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _decode_turn(row: sqlite3.Row) -> TurnRecord:
    return TurnRecord(
        session_id=row["session_id"],
        turn=row["turn"],
        request_id=row["request_id"],
        status=row["status"],
        user_message=row["user_message"],
        response=_decode_json(row["response_json"]),
        products=_decode_json(row["products_json"]),
        state=_decode_json(row["state_json"]),
        trace=_decode_json(row["trace_json"]),
        error=_decode_json(row["error_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _decode_json(value: str | None) -> Any | None:
    if value is None:
        return None
    try:
        return json.loads(value, parse_constant=_reject_nonstandard_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise RepositoryCorruptionError(
            "Persisted repository data is invalid."
        ) from error


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {value}")


def _decode_feedback(row: sqlite3.Row) -> FeedbackRecord:
    return FeedbackRecord(
        session_id=row["session_id"],
        turn=row["turn"],
        parent_asin=row["parent_asin"],
        reason=row["reason"],
        note=row["note"],
        updated_at=row["updated_at"],
    )


def _product_ranks(products: Any) -> dict[str, int]:
    if not isinstance(products, list):
        return {}
    ranks: dict[str, int] = {}
    for position, product in enumerate(products, start=1):
        if not isinstance(product, dict):
            continue
        parent_asin = product.get("parent_asin")
        if not isinstance(parent_asin, str):
            continue
        rank = product.get("rank")
        if not isinstance(rank, int):
            rank = position
        ranks[parent_asin] = min(ranks.get(parent_asin, rank), rank)
    return ranks
