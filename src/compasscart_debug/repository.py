from __future__ import annotations

import json
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
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN")
            try:
                for statement in _SCHEMA_STATEMENTS:
                    connection.execute(statement)
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                        (str(_SCHEMA_VERSION),),
                    )
                else:
                    _validate_schema_version(row["value"])
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def schema_version(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
        if row is None:
            raise RepositoryVersionError("The database schema is not initialized.")
        return _validate_schema_version(row["value"])

    def health(self) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1

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
        if not reason or not note:
            raise ValidationError({"feedback": "Reason and note are required."})
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
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = self._connection_factory(self.path, timeout=5)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=5000")
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _begin(connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN")

    def _timestamp(self) -> str:
        value = self._clock().astimezone(UTC).replace(microsecond=0)
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


def _decode_session(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        session_id=row["session_id"],
        name=row["name"],
        profile=json.loads(row["profile_json"]),
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
    return json.loads(value) if value is not None else None


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
