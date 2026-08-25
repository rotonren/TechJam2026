"""Persistent turn orchestration for the CompassCart debug runtime."""

from __future__ import annotations

import json
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar

from .agent_worker import AgentWorker
from .errors import (
    ConflictError,
    DebugServiceError,
    NotReadyError,
    ReplayMismatchError,
    RequestMismatchError,
    UnresolvedTurnError,
    ValidationError,
    WorkerBusyError,
)
from .repository import (
    FEEDBACK_REASONS,
    DebugRepository,
    FeedbackRecord,
    RepositoryBusyError,
    RepositoryCorruptionError,
    RepositoryStorageError,
    RepositoryVersionError,
    SessionRecord,
    TurnRecord,
)

_MAX_NAME_LENGTH = 120
_MAX_MESSAGE_LENGTH = 10_000
_MAX_REQUEST_ID_LENGTH = 128
_MAX_PREFERENCE_TAGS = 50
_MAX_PREFERENCE_TAG_LENGTH = 200
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
_EXPORT_FORMAT = "compasscart-debug-session"
_EXPORT_SCHEMA_VERSION = 1
_EXPORT_KEYS = frozenset(
    {"format", "schema_version", "exported_at", "session", "turns"}
)
_SESSION_KEYS = frozenset(
    {
        "session_id",
        "name",
        "profile",
        "agent_version",
        "catalog_sha256",
        "config_sha256",
        "assets_sha256",
        "source_session_id",
        "archived",
        "dirty",
        "read_only_reason",
        "created_at",
        "updated_at",
    }
)
_TURN_KEYS = frozenset(
    {
        "session_id",
        "turn",
        "request_id",
        "status",
        "user_message",
        "response",
        "products",
        "state",
        "trace",
        "feedback",
        "error",
        "created_at",
        "updated_at",
    }
)
_FEEDBACK_KEYS = frozenset(
    {"session_id", "turn", "parent_asin", "reason", "note", "updated_at"}
)
_TURN_STATUSES = frozenset({"pending", "completed", "failed"})
_SENSITIVE_EXPORT_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "database_path",
        "env",
        "environment",
        "exception",
        "host",
        "hostname",
        "password",
        "path",
        "repr",
        "secret",
        "token",
        "traceback",
    }
)
_IDENTIFIER_EXPORT_KEYS = frozenset(
    {
        "agent_version",
        "assets_sha256",
        "catalog_sha256",
        "config_sha256",
        "parent_asin",
        "request_id",
        "session_id",
        "source_session_id",
    }
)
_ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/(?:[^/\s]+/)+)")
_HOSTNAME = re.compile(
    r"\b(?:localhost|(?:\d{1,3}\.){3}\d{1,3}|[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+){2,})\b"
)
_EXCEPTION_REPR = re.compile(
    r"(?:Traceback \(most recent call last\)|\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\()"
)
_UNSET = object()
_T = TypeVar("_T")


@dataclass
class _InFlight:
    request_id: str
    user_message: str
    turn: TurnRecord | None = None


class _AgentOperationError(RuntimeError):
    """Private marker for an Agent call that may have changed memory."""


class DebugService:
    """Coordinate durable debug turns with the worker-owned Agent state."""

    def __init__(
        self,
        repository: DebugRepository,
        worker: AgentWorker,
        *,
        uuid_factory: Callable[[], object] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.worker = worker
        self.uuid_factory = uuid_factory or uuid.uuid4
        self.clock = clock or (lambda: datetime.now(UTC))
        self._loaded_sessions: set[str] = set()
        self._dirty_sessions: set[str] = set()
        self._coordination_lock = threading.RLock()
        self._inflight: dict[tuple[str, str], _InFlight] = {}

    def create_session(
        self, name: object, profile: object, *, source_session_id: str | None = None
    ) -> dict[str, Any]:
        clean_name = _validate_name(name)
        clean_profile = _validate_profile(profile)
        session_id = self._new_session_id()
        identity = self.worker.identity
        try:
            self.worker.reset_session(session_id, clean_profile)
        except DebugServiceError:
            self._mark_memory_dirty(session_id)
            raise
        except Exception as error:
            self._mark_memory_dirty(session_id)
            raise _agent_error() from error

        try:
            session = self._repository_call(
                self.repository.create_session,
                session_id=session_id,
                name=clean_name,
                profile=clean_profile,
                agent_version=identity.agent_version,
                catalog_sha256=identity.catalog_sha256,
                config_sha256=identity.config_sha256,
                assets_sha256=identity.assets_sha256,
                source_session_id=source_session_id,
            )
        except Exception:
            # reset() may already have installed an orphan in Agent memory.  It
            # is never considered loaded unless the durable create succeeds.
            self._mark_memory_dirty(session_id)
            raise

        with self._coordination_lock:
            self._loaded_sessions.add(session_id)
            self._dirty_sessions.discard(session_id)
        return {"session": _session_payload(session)}

    def get_session(self, session_id: str) -> dict[str, Any]:
        session = self._repository_call(self.repository.get_session, session_id)
        turns = self._repository_call(self.repository.list_turns, session_id)
        turn_payloads = [self._turn_with_feedback(turn) for turn in turns]
        continuation, can_send = self._continuation(session, turns)
        return {
            "session": _session_payload(session),
            "turns": turn_payloads,
            "continuation": continuation,
            "can_send": can_send,
        }

    def read_session(self, session_id: str) -> dict[str, Any]:
        """Stable spelling for callers that prefer read over get."""

        return self.get_session(session_id)

    def list_sessions(self, status: str = "active") -> list[dict[str, Any]]:
        if status not in {"active", "archived", "all"}:
            raise ValidationError({"status": "Session status is invalid."})
        sessions = self._repository_call(self.repository.list_sessions)
        if status == "active":
            sessions = [session for session in sessions if not session.archived]
        elif status == "archived":
            sessions = [session for session in sessions if session.archived]
        return [_session_payload(session) for session in sessions]

    def patch_session(
        self,
        session_id: str,
        *,
        name: object = _UNSET,
        archived: object = _UNSET,
    ) -> dict[str, Any]:
        values: dict[str, object] = {}
        if name is not _UNSET:
            values["name"] = _validate_name(name)
        if archived is not _UNSET:
            if not isinstance(archived, bool):
                raise ValidationError({"archived": "Archived must be a boolean."})
            values["archived"] = archived
        session = self._repository_call(
            self.repository.patch_session, session_id, **values
        )
        return {"session": _session_payload(session)}

    def set_feedback(
        self,
        session_id: str,
        turn: int,
        parent_asin: object,
        incorrect: object,
        reason: object,
        note: object,
    ) -> dict[str, Any] | None:
        if not isinstance(incorrect, bool):
            raise ValidationError({"incorrect": "Incorrect must be a boolean."})
        if not isinstance(parent_asin, str) or not parent_asin:
            raise ValidationError({"parent_asin": "A product identifier is required."})
        if not isinstance(note, str):
            raise ValidationError({"note": "Feedback note must be text."})
        session = self._repository_call(self.repository.get_session, session_id)
        if session.archived or session.read_only_reason == "imported_history":
            raise ConflictError()
        if not incorrect:
            self._repository_call(
                self.repository.clear_feedback, session_id, turn, parent_asin
            )
            return None
        if not isinstance(reason, str) or reason not in FEEDBACK_REASONS:
            raise ValidationError({"reason": "Feedback reason is invalid."})
        feedback = self._repository_call(
            self.repository.upsert_feedback,
            session_id,
            turn,
            parent_asin,
            reason,
            note,
        )
        return _feedback_payload(feedback)

    def export_session(self, session_id: str) -> dict[str, Any]:
        session = self._repository_call(self.repository.get_session, session_id)
        turns = self._repository_call(self.repository.list_turns, session_id)
        payload = {
            "format": _EXPORT_FORMAT,
            "schema_version": _EXPORT_SCHEMA_VERSION,
            "exported_at": _utc_timestamp(self.clock()),
            "session": _safe_export_value(_session_payload(session)),
            "turns": [
                _safe_export_value(self._turn_with_feedback(turn)) for turn in turns
            ],
        }
        return _json_copy(payload, "export")

    def import_session(self, payload: object) -> dict[str, Any]:
        session, turns = _validate_import_payload(payload)
        session_id = self._new_session_id()
        local_session_ids = {
            item.session_id
            for item in self._repository_call(self.repository.list_sessions)
        }
        source_session_id = (
            session["session_id"]
            if session["session_id"] in local_session_ids
            else None
        )
        self._repository_call(
            self.repository.import_session,
            session_id=session_id,
            name=session["name"],
            profile=session["profile"],
            agent_version=session["agent_version"],
            catalog_sha256=session["catalog_sha256"],
            config_sha256=session["config_sha256"],
            assets_sha256=session["assets_sha256"],
            source_session_id=source_session_id,
            archived=session["archived"],
            dirty=session["dirty"],
            read_only_reason="imported_history",
            created_at=session["created_at"],
            updated_at=session["updated_at"],
            turns=turns,
        )
        return self.get_session(session_id)

    def clone_session(
        self, source_session_id: str, through_turn: object = None
    ) -> dict[str, Any]:
        source = self._repository_call(self.repository.get_session, source_session_id)
        if source.archived:
            raise ConflictError()
        turns = self._repository_call(self.repository.list_turns, source_session_id)
        completed_prefix: list[TurnRecord] = []
        for turn in turns:
            if turn.status != "completed":
                break
            completed_prefix.append(turn)
        if through_turn is None:
            selected_count = len(completed_prefix)
        elif (
            not isinstance(through_turn, int)
            or isinstance(through_turn, bool)
            or through_turn < 0
            or through_turn > len(completed_prefix)
        ):
            raise ValidationError(
                {"through_turn": "Clone turn must select a completed prefix."}
            )
        else:
            selected_count = through_turn

        suffix = " (Clone)"
        clone_name = f"{source.name[: _MAX_NAME_LENGTH - len(suffix)].rstrip()}{suffix}"
        created = self.create_session(
            clone_name,
            source.profile,
            source_session_id=source_session_id,
        )
        clone_id = created["session"]["session_id"]
        source_request_ids = {turn.request_id for turn in turns}
        for turn in completed_prefix[:selected_count]:
            request_id = self._new_request_id(excluding=source_request_ids)
            source_request_ids.add(request_id)
            self.send_message(clone_id, request_id, turn.user_message)
        return self.get_session(clone_id)

    def send_message(
        self, session_id: str, request_id: object, user_message: object
    ) -> dict[str, Any]:
        clean_request_id = _validate_request_id(request_id)
        clean_message = _validate_message(user_message)
        session = self._repository_call(self.repository.get_session, session_id)
        if session.archived or session.read_only_reason == "imported_history":
            raise ConflictError()
        existing = self._repository_call(
            self.repository.find_turn_by_request_id, session_id, clean_request_id
        )
        if existing is not None and existing.user_message != clean_message:
            raise RequestMismatchError()
        if existing is not None and existing.status == "completed":
            return {"turn": self._turn_with_feedback(existing)}

        # Identity is checked before reserving a new durable row so an old
        # session cannot be made unresolved merely by viewing it under a new
        # runtime.  Reading the immutable identity does not call the Agent.
        self._assert_compatible_identity(session)
        if session.read_only_reason == "replay_mismatch":
            raise ReplayMismatchError()
        if session.read_only_reason not in (None, "runtime_incompatible"):
            raise _runtime_incompatible_error()

        key = (session_id, clean_request_id)
        with self._coordination_lock:
            local = self._inflight_for_session(session_id)
            if local is not None:
                _, marker = local
                if marker.request_id != clean_request_id:
                    raise UnresolvedTurnError()
                if marker.user_message != clean_message:
                    raise RequestMismatchError()
                if marker.turn is None:  # Defensive: reservation is lock-protected.
                    pending = self._repository_call(
                        self.repository.get_turn_by_request_id,
                        session_id,
                        clean_request_id,
                    )
                else:
                    pending = marker.turn
                return {"turn": _turn_payload(pending, feedback=[])}

            marker = _InFlight(clean_request_id, clean_message)
            self._inflight[key] = marker
            try:
                if existing is not None and existing.status == "failed":
                    pending = self._repository_call(
                        self.repository.retry_failed,
                        session_id,
                        clean_request_id,
                        clean_message,
                    )
                else:
                    pending = self._repository_call(
                        self.repository.reserve_turn,
                        session_id,
                        clean_request_id,
                        clean_message,
                    )
                marker.turn = pending
            except Exception:
                self._inflight.pop(key, None)
                raise

            if pending.status == "completed":
                self._inflight.pop(key, None)
                return {"turn": self._turn_with_feedback(pending)}

        retrying_unresolved = existing is not None and existing.status in {
            "pending",
            "failed",
        }
        try:
            try:
                self._ensure_hydrated(
                    session_id,
                    force=retrying_unresolved,
                )
                try:
                    observation = self.worker.observe_turn(
                        session_id, clean_message, pending.turn
                    )
                except (NotReadyError, WorkerBusyError):
                    raise
                except Exception as error:
                    raise _AgentOperationError() from error
            except ReplayMismatchError:
                self._fail_replay_attempt(session_id, pending.turn)
                raise
            except _AgentOperationError as error:
                self._record_agent_failure(session_id, pending.turn)
                raise _agent_error() from error
            except DebugServiceError:
                # A rejected worker submission or repository read did not
                # mutate Agent memory.  Keep the row pending for safe retry.
                raise
            except Exception as error:
                self._record_agent_failure(session_id, pending.turn)
                raise _agent_error() from error

            try:
                completed = self._repository_call(
                    self.repository.complete_turn,
                    session_id,
                    pending.turn,
                    observation,
                )
            except Exception as error:
                self._mark_dirty(session_id)
                raise DebugServiceError(
                    "snapshot_not_saved",
                    "The Agent responded, but the debug snapshot was not saved.",
                    503,
                    True,
                ) from error
            return {"turn": _turn_payload(completed, feedback=[])}
        finally:
            with self._coordination_lock:
                current = self._inflight.get(key)
                if current is marker:
                    self._inflight.pop(key, None)

    def _ensure_hydrated(self, session_id: str, *, force: bool = False) -> None:
        session = self._repository_call(self.repository.get_session, session_id)
        with self._coordination_lock:
            loaded = session_id in self._loaded_sessions
            memory_dirty = session_id in self._dirty_sessions
        needs_hydration = (
            force
            or not loaded
            or memory_dirty
            or session.dirty
            or session.read_only_reason is not None
        )

        # Always compare identity before any worker command, including the
        # has-session probe used to detect bounded-cache eviction.
        self._assert_compatible_identity(session)
        if not needs_hydration:
            try:
                needs_hydration = not self.worker.has_session(session_id)
            except (NotReadyError, WorkerBusyError):
                raise
            except Exception as error:
                raise _AgentOperationError() from error
        if not needs_hydration:
            return

        completed = self._repository_call(
            self.repository.list_completed_turns, session_id
        )
        # Re-read immediately before replay.  AgentWorker identities are
        # immutable, and this also catches an invalidate/restart race.
        self._assert_compatible_identity(session)
        try:
            replayed = self.worker.rehydrate(session_id, session.profile, completed)
        except (NotReadyError, WorkerBusyError):
            raise
        except Exception as error:
            raise _AgentOperationError() from error

        if len(replayed) != len(completed) or any(
            recorded.response != replay.response or recorded.state != replay.state
            for recorded, replay in zip(completed, replayed, strict=True)
        ):
            self._mark_memory_dirty(session_id)
            self._repository_call(
                self.repository.set_read_only_reason,
                session_id,
                "replay_mismatch",
            )
            raise ReplayMismatchError()

        try:
            if session.dirty:
                self._repository_call(self.repository.clear_dirty, session_id)
            if session.read_only_reason is not None:
                self._repository_call(
                    self.repository.clear_read_only_reason, session_id
                )
        except Exception:
            self._mark_dirty(session_id)
            raise
        with self._coordination_lock:
            self._loaded_sessions.add(session_id)
            self._dirty_sessions.discard(session_id)

    def _assert_compatible_identity(self, session: SessionRecord) -> None:
        identity = self.worker.identity
        if (
            session.agent_version,
            session.catalog_sha256,
            session.config_sha256,
            session.assets_sha256,
        ) == (
            identity.agent_version,
            identity.catalog_sha256,
            identity.config_sha256,
            identity.assets_sha256,
        ):
            return
        self._mark_memory_dirty(session.session_id)
        self._repository_call(
            self.repository.set_read_only_reason,
            session.session_id,
            "runtime_incompatible",
        )
        raise _runtime_incompatible_error()

    def _record_agent_failure(self, session_id: str, turn: int) -> None:
        try:
            self._repository_call(
                self.repository.fail_turn,
                session_id,
                turn,
                {
                    "code": "agent_error",
                    "message": "The Agent could not complete this turn.",
                },
            )
        except Exception:  # noqa: BLE001 - preserve the primary Agent failure.
            # If the failure snapshot cannot be saved, the pending row remains
            # crash residue and is still safe to retry after hydration.
            self._mark_memory_dirty(session_id)
        self._mark_dirty(session_id)

    def _fail_replay_attempt(self, session_id: str, turn: int) -> None:
        try:
            self._repository_call(
                self.repository.fail_turn,
                session_id,
                turn,
                {
                    "code": "replay_mismatch",
                    "message": "The replay does not match the recorded turn.",
                },
            )
        except Exception:  # noqa: BLE001 - preserve the replay mismatch error.
            self._mark_memory_dirty(session_id)
        self._mark_memory_dirty(session_id)

    def _mark_memory_dirty(self, session_id: str) -> None:
        with self._coordination_lock:
            self._loaded_sessions.discard(session_id)
            self._dirty_sessions.add(session_id)

    def _mark_dirty(self, session_id: str) -> None:
        self._mark_memory_dirty(session_id)
        try:
            self._repository_call(self.repository.set_dirty, session_id)
        except Exception:  # noqa: BLE001 - memory dirty state remains authoritative.
            # The in-memory flag is authoritative for this process.  A pending
            # or failed durable row forces replay after a restart as well.
            return

    def _continuation(
        self, session: SessionRecord, turns: list[TurnRecord]
    ) -> tuple[str, bool]:
        with self._coordination_lock:
            active = any(key[0] == session.session_id for key in self._inflight)
        if active or any(turn.status == "pending" for turn in turns):
            return "rehydrating", False
        if session.archived:
            return "archived", False
        if session.read_only_reason is not None:
            return "incompatible", False
        if any(turn.status == "failed" for turn in turns):
            return "blocked_failed", False
        if sum(turn.status == "completed" for turn in turns) >= 10:
            return "turn_limit", False
        return "ready", True

    def _turn_with_feedback(self, turn: TurnRecord) -> dict[str, Any]:
        feedback = self._repository_call(
            self.repository.list_feedback, turn.session_id, turn.turn
        )
        return _turn_payload(
            turn,
            feedback=[_feedback_payload(record) for record in feedback],
        )

    def _inflight_for_session(
        self, session_id: str
    ) -> tuple[tuple[str, str], _InFlight] | None:
        for key, marker in self._inflight.items():
            if key[0] == session_id:
                return key, marker
        return None

    def _new_session_id(self) -> str:
        value = str(self.uuid_factory())
        if not value or len(value) > 128:
            raise ValidationError({"session_id": "Session identifier is invalid."})
        return value

    def _new_request_id(self, *, excluding: set[str]) -> str:
        for _ in range(100):
            value = _validate_request_id(str(self.uuid_factory()))
            if value not in excluding:
                return value
        raise ValidationError({"request_id": "A fresh request identifier is required."})

    @staticmethod
    def _repository_call(
        callback: Callable[..., _T], *args: object, **kwargs: object
    ) -> _T:
        try:
            return callback(*args, **kwargs)
        except DebugServiceError:
            raise
        except RepositoryBusyError as error:
            raise DebugServiceError(
                "repository_busy",
                "The debug repository is busy.",
                503,
                True,
            ) from error
        except RepositoryCorruptionError as error:
            raise DebugServiceError(
                "repository_corrupt",
                "The debug repository contains invalid data.",
                500,
            ) from error
        except RepositoryVersionError as error:
            raise DebugServiceError(
                "repository_incompatible",
                "The debug repository version is incompatible.",
                500,
            ) from error
        except RepositoryStorageError as error:
            raise DebugServiceError(
                "repository_unavailable",
                "The debug repository is unavailable.",
                503,
                True,
            ) from error


def _validate_import_payload(
    payload: object,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    body = _exact_object(payload, _EXPORT_KEYS, "body")
    if _safe_export_value(body) != body:
        raise ValidationError(
            {"body": "Import data contains local or sensitive information."}
        )
    if body["format"] != _EXPORT_FORMAT:
        raise ValidationError({"format": "Import format is invalid."})
    if (
        not isinstance(body["schema_version"], int)
        or isinstance(body["schema_version"], bool)
        or body["schema_version"] != _EXPORT_SCHEMA_VERSION
    ):
        raise ValidationError({"schema_version": "Import version is invalid."})
    _validate_timestamp(body["exported_at"], "exported_at")
    session = _validate_import_session(body["session"])
    turns = _validate_import_turns(body["turns"], session["session_id"])
    return session, turns


def _validate_import_session(value: object) -> dict[str, Any]:
    session = _exact_object(value, _SESSION_KEYS, "session")
    source_session_id = _validate_identifier(
        session["session_id"], "session.session_id"
    )
    imported_source = session["source_session_id"]
    if imported_source is not None:
        imported_source = _validate_identifier(
            imported_source, "session.source_session_id"
        )
    assets_sha256 = session["assets_sha256"]
    if assets_sha256 is not None:
        assets_sha256 = _validate_required_text(assets_sha256, "session.assets_sha256")
    read_only_reason = session["read_only_reason"]
    if read_only_reason is not None and not isinstance(read_only_reason, str):
        raise ValidationError(
            {"session.read_only_reason": "Read-only reason must be text or null."}
        )
    if not isinstance(session["archived"], bool):
        raise ValidationError({"session.archived": "Archived must be a boolean."})
    if not isinstance(session["dirty"], bool):
        raise ValidationError({"session.dirty": "Dirty must be a boolean."})
    return {
        "session_id": source_session_id,
        "name": _validate_name(session["name"]),
        "profile": _validate_profile(session["profile"]),
        "agent_version": _validate_required_text(
            session["agent_version"], "session.agent_version"
        ),
        "catalog_sha256": _validate_required_text(
            session["catalog_sha256"], "session.catalog_sha256"
        ),
        "config_sha256": _validate_required_text(
            session["config_sha256"], "session.config_sha256"
        ),
        "assets_sha256": assets_sha256,
        "source_session_id": imported_source,
        "archived": session["archived"],
        "dirty": session["dirty"],
        "read_only_reason": read_only_reason,
        "created_at": _validate_timestamp(session["created_at"], "session.created_at"),
        "updated_at": _validate_timestamp(session["updated_at"], "session.updated_at"),
    }


def _validate_import_turns(
    value: object, source_session_id: str
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValidationError({"turns": "Turns must be a list."})
    if len(value) > 10:
        raise ValidationError({"turns": "At most 10 turns may be imported."})
    request_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for position, raw_turn in enumerate(value, start=1):
        field = f"turns[{position - 1}]"
        turn = _exact_object(raw_turn, _TURN_KEYS, field)
        if turn["session_id"] != source_session_id:
            raise ValidationError(
                {f"{field}.session_id": "Turn session identifier is invalid."}
            )
        if (
            not isinstance(turn["turn"], int)
            or isinstance(turn["turn"], bool)
            or turn["turn"] != position
        ):
            raise ValidationError(
                {f"{field}.turn": "Turns must be contiguous and ordered."}
            )
        request_id = _validate_request_id(turn["request_id"])
        if request_id in request_ids:
            raise ValidationError(
                {f"{field}.request_id": "Request identifiers must be unique."}
            )
        request_ids.add(request_id)
        status = turn["status"]
        if not isinstance(status, str) or status not in _TURN_STATUSES:
            raise ValidationError({f"{field}.status": "Turn status is invalid."})

        response = _json_copy(turn["response"], f"{field}.response")
        products = _json_copy(turn["products"], f"{field}.products")
        state = _json_copy(turn["state"], f"{field}.state")
        trace = _json_copy(turn["trace"], f"{field}.trace")
        error = _json_copy(turn["error"], f"{field}.error")
        feedback = _validate_import_feedback(
            turn["feedback"],
            field=field,
            source_session_id=source_session_id,
            turn_number=position,
            products=products,
        )
        snapshots = (response, products, state, trace)
        if status == "completed":
            if (
                not isinstance(response, dict)
                or not isinstance(products, list)
                or not isinstance(state, dict)
                or not isinstance(trace, dict)
                or error is not None
            ):
                raise ValidationError(
                    {field: "Completed turns require snapshots and no error."}
                )
        elif status == "failed":
            if any(item is not None for item in snapshots) or not isinstance(
                error, dict
            ):
                raise ValidationError(
                    {field: "Failed turns require only an error snapshot."}
                )
            if feedback:
                raise ValidationError(
                    {f"{field}.feedback": "Only completed turns may have feedback."}
                )
        elif any(item is not None for item in (*snapshots, error)) or feedback:
            raise ValidationError(
                {field: "Pending turns cannot contain snapshots or feedback."}
            )

        normalized.append(
            {
                "turn": position,
                "request_id": request_id,
                "status": status,
                "user_message": _validate_message(turn["user_message"]),
                "response": response,
                "products": products,
                "state": state,
                "trace": trace,
                "error": error,
                "created_at": _validate_timestamp(
                    turn["created_at"], f"{field}.created_at"
                ),
                "updated_at": _validate_timestamp(
                    turn["updated_at"], f"{field}.updated_at"
                ),
                "feedback": feedback,
            }
        )
    return normalized


def _validate_import_feedback(
    value: object,
    *,
    field: str,
    source_session_id: str,
    turn_number: int,
    products: Any,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValidationError({f"{field}.feedback": "Feedback must be a list."})
    product_asins = (
        {
            product.get("parent_asin")
            for product in products
            if isinstance(product, dict) and isinstance(product.get("parent_asin"), str)
        }
        if isinstance(products, list)
        else set()
    )
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw_feedback in enumerate(value):
        item_field = f"{field}.feedback[{index}]"
        feedback = _exact_object(raw_feedback, _FEEDBACK_KEYS, item_field)
        if feedback["session_id"] != source_session_id:
            raise ValidationError(
                {f"{item_field}.session_id": "Feedback session is invalid."}
            )
        if (
            not isinstance(feedback["turn"], int)
            or isinstance(feedback["turn"], bool)
            or feedback["turn"] != turn_number
        ):
            raise ValidationError({f"{item_field}.turn": "Feedback turn is invalid."})
        parent_asin = feedback["parent_asin"]
        if not isinstance(parent_asin, str) or not parent_asin:
            raise ValidationError(
                {f"{item_field}.parent_asin": "Feedback product is invalid."}
            )
        if parent_asin in seen:
            raise ValidationError(
                {f"{item_field}.parent_asin": "Feedback products must be unique."}
            )
        if parent_asin not in product_asins:
            raise ValidationError(
                {
                    f"{item_field}.parent_asin": (
                        "Feedback product is not present in the completed turn."
                    )
                }
            )
        seen.add(parent_asin)
        reason = feedback["reason"]
        if not isinstance(reason, str) or reason not in FEEDBACK_REASONS:
            raise ValidationError(
                {f"{item_field}.reason": "Feedback reason is invalid."}
            )
        note = feedback["note"]
        if not isinstance(note, str):
            raise ValidationError({f"{item_field}.note": "Feedback note must be text."})
        normalized.append(
            {
                "parent_asin": parent_asin,
                "reason": reason,
                "note": note,
                "updated_at": _validate_timestamp(
                    feedback["updated_at"], f"{item_field}.updated_at"
                ),
            }
        )
    return normalized


def _exact_object(
    value: object, expected_keys: frozenset[str], field: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError({field: "A JSON object is required."})
    if set(value) != expected_keys:
        raise ValidationError({field: "Object fields do not match the schema."})
    return value


def _validate_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValidationError({field: "Session identifier is invalid."})
    return value


def _validate_required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError({field: "Nonempty text is required."})
    return value


def _validate_timestamp(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValidationError({field: "A UTC timestamp is required."})
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValidationError({field: "A UTC timestamp is required."}) from error
    if parsed.tzinfo is None or _utc_timestamp(parsed) != value:
        raise ValidationError({field: "A canonical UTC timestamp is required."})
    return value


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("The service clock must return a timezone-aware datetime.")
    return (
        value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def _json_copy(value: object, field: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValidationError({field: "Value must contain valid JSON."}) from error
    if decoded != value:
        raise ValidationError({field: "Value must contain valid JSON."})
    return decoded


def _safe_export_value(value: Any, *, field: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            key: _safe_export_value(item, field=key)
            for key, item in value.items()
            if isinstance(key, str) and not _is_sensitive_export_key(key)
        }
    if isinstance(value, list):
        return [_safe_export_value(item, field=field) for item in value]
    if (
        isinstance(value, str)
        and field not in _IDENTIFIER_EXPORT_KEYS
        and (
            _ABSOLUTE_PATH.search(value)
            or _HOSTNAME.search(value)
            or _EXCEPTION_REPR.search(value)
        )
    ):
        return "[redacted]"
    return value


def _is_sensitive_export_key(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    return normalized in _SENSITIVE_EXPORT_KEYS or normalized.endswith(
        ("_path", "_token", "_secret", "_password", "_hostname")
    )


def _validate_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError({"name": "A session name is required."})
    clean = value.strip()
    if len(clean) > _MAX_NAME_LENGTH:
        raise ValidationError({"name": "The session name is too long."})
    return clean


def _validate_profile(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationError({"profile": "Profile must be an object."})
    if any(not isinstance(key, str) for key in value):
        raise ValidationError({"profile": "Profile keys must be text."})
    profile = dict(value)
    tags = profile.get("preference_tags", _UNSET)
    if tags is not _UNSET:
        if not isinstance(tags, (list, tuple)):
            raise ValidationError(
                {"preference_tags": "Preference tags must be a list of text."}
            )
        if len(tags) > _MAX_PREFERENCE_TAGS:
            raise ValidationError(
                {"preference_tags": "Too many preference tags were provided."}
            )
        clean_tags: list[str] = []
        for tag in tags:
            if (
                not isinstance(tag, str)
                or not tag.strip()
                or len(tag.strip()) > _MAX_PREFERENCE_TAG_LENGTH
            ):
                raise ValidationError(
                    {"preference_tags": "Preference tags must be nonempty text."}
                )
            clean_tags.append(tag.strip())
        profile["preference_tags"] = clean_tags
    try:
        encoded = json.dumps(profile, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            {"profile": "Profile must contain valid JSON."}
        ) from error
    if not isinstance(decoded, dict):  # Mapping input always encodes as an object.
        raise ValidationError({"profile": "Profile must be an object."})
    return decoded


def _validate_request_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_REQUEST_ID_LENGTH
        or _REQUEST_ID.fullmatch(value) is None
    ):
        raise ValidationError({"request_id": "Request identifier is invalid."})
    return value


def _validate_message(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError({"user_message": "A message is required."})
    if len(value) > _MAX_MESSAGE_LENGTH:
        raise ValidationError({"user_message": "The message is too long."})
    return value


def _session_payload(record: SessionRecord) -> dict[str, Any]:
    return {
        "session_id": record.session_id,
        "name": record.name,
        "profile": record.profile,
        "agent_version": record.agent_version,
        "catalog_sha256": record.catalog_sha256,
        "config_sha256": record.config_sha256,
        "assets_sha256": record.assets_sha256,
        "source_session_id": record.source_session_id,
        "archived": record.archived,
        "dirty": record.dirty,
        "read_only_reason": record.read_only_reason,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _turn_payload(
    record: TurnRecord, *, feedback: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "session_id": record.session_id,
        "turn": record.turn,
        "request_id": record.request_id,
        "status": record.status,
        "user_message": record.user_message,
        "response": record.response,
        "products": record.products,
        "state": record.state,
        "trace": record.trace,
        "feedback": feedback,
        "error": record.error,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _feedback_payload(record: FeedbackRecord) -> dict[str, Any]:
    return {
        "session_id": record.session_id,
        "turn": record.turn,
        "parent_asin": record.parent_asin,
        "reason": record.reason,
        "note": record.note,
        "updated_at": record.updated_at,
    }


def _agent_error() -> DebugServiceError:
    return DebugServiceError(
        "agent_error",
        "The Agent could not complete this turn.",
        500,
        True,
    )


def _runtime_incompatible_error() -> DebugServiceError:
    return DebugServiceError(
        "runtime_incompatible",
        "The recorded session is incompatible with the current runtime.",
        409,
    )
