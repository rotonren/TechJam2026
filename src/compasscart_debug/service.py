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
    DebugServiceError,
    NotReadyError,
    ReplayMismatchError,
    RequestMismatchError,
    UnresolvedTurnError,
    ValidationError,
    WorkerBusyError,
)
from .repository import (
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

    def send_message(
        self, session_id: str, request_id: object, user_message: object
    ) -> dict[str, Any]:
        clean_request_id = _validate_request_id(request_id)
        clean_message = _validate_message(user_message)
        session = self._repository_call(self.repository.get_session, session_id)
        existing = self._repository_call(
            self.repository.find_turn_by_request_id, session_id, clean_request_id
        )
        if existing is not None and existing.user_message != clean_message:
            raise RequestMismatchError()
        if existing is not None and existing.status == "completed":
            return {"turn": _turn_payload(existing, feedback=[])}

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
                return {"turn": _turn_payload(pending, feedback=[])}

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
