from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DebugServiceError(Exception):
    """A stable error that may be returned by the debug service."""

    code: str
    message: str
    http_status: int
    retryable: bool = False
    field_errors: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.field_errors:
            error["field_errors"] = dict(sorted(self.field_errors.items()))
        return {"error": error}


class RuntimeSetupError(RuntimeError):
    """A non-public runtime setup failure with a path-safe message."""


class AuthenticationError(DebugServiceError):
    def __init__(self) -> None:
        super().__init__("authentication_failed", "Authentication failed.", 401)


class ValidationError(DebugServiceError):
    def __init__(self, field_errors: Mapping[str, str] | None = None) -> None:
        super().__init__(
            "validation_failed", "Request validation failed.", 400, False, field_errors
        )


class ConflictError(DebugServiceError):
    def __init__(self) -> None:
        super().__init__("conflict", "The request conflicts with current state.", 409)


class RequestMismatchError(DebugServiceError):
    def __init__(self) -> None:
        super().__init__(
            "request_mismatch", "The request does not match the existing turn.", 409
        )


class NotReadyError(DebugServiceError):
    def __init__(self) -> None:
        super().__init__("not_ready", "The debug runtime is not ready.", 503, True)


class WorkerBusyError(DebugServiceError):
    def __init__(self) -> None:
        super().__init__("worker_busy", "The debug worker is busy.", 503, True)


class WorkerTimeoutError(DebugServiceError):
    def __init__(self) -> None:
        super().__init__("worker_timeout", "The debug worker timed out.", 504, True)


class ReplayMismatchError(DebugServiceError):
    def __init__(self) -> None:
        super().__init__(
            "replay_mismatch", "The replay does not match the recorded turn.", 409
        )


class UnresolvedTurnError(DebugServiceError):
    def __init__(self) -> None:
        super().__init__("unresolved_turn", "The turn has not completed.", 409, True)


class TurnLimitError(DebugServiceError):
    def __init__(self) -> None:
        super().__init__("turn_limit", "The session has reached its turn limit.", 400)


class NotFoundError(DebugServiceError):
    def __init__(self) -> None:
        super().__init__("not_found", "The requested resource was not found.", 404)
