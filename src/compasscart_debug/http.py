"""Small, dependency-free WSGI shell for the CompassCart debug service."""

from __future__ import annotations

import email.message
import hmac
import http.client
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote

from .agent_worker import AgentWorker
from .config import DebugConfig
from .errors import AuthenticationError, DebugServiceError
from .repository import DebugRepository
from .service import DebugService

JSON_CONTENT_TYPE = "application/json; charset=utf-8"
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'none'; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
_MUTATION_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_ASIN_RE = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
_TURN_RE = re.compile(r"(?:[1-9]|10)\Z")

# The URL-to-file relationship is deliberately literal.  The secondary basename
# entries are only for the temporary static-root fixtures used by unit tests.
STATIC_ALLOWLIST: dict[str, tuple[tuple[str, ...], str]] = {
    "/": (("index.html",), "text/html; charset=utf-8"),
    "/static/styles.css": (("styles.css",), "text/css; charset=utf-8"),
    "/static/js/api.js": (("js/api.js", "api.js"), "text/javascript; charset=utf-8"),
    "/static/js/dom.js": (("js/dom.js", "dom.js"), "text/javascript; charset=utf-8"),
    "/static/js/store.js": (
        ("js/store.js", "store.js"),
        "text/javascript; charset=utf-8",
    ),
    "/static/js/app.js": (("js/app.js", "app.js"), "text/javascript; charset=utf-8"),
}
STATIC_MAP = STATIC_ALLOWLIST


def _http_error(
    code: str,
    message: str,
    status: int,
    *,
    retryable: bool = False,
    field_errors: Mapping[str, str] | None = None,
) -> DebugServiceError:
    return DebugServiceError(code, message, status, retryable, field_errors)


def _not_found() -> DebugServiceError:
    return _http_error("not_found", "The requested resource was not found.", 404)


def _method_not_allowed() -> DebugServiceError:
    return _http_error("method_not_allowed", "The request method is not allowed.", 405)


def _authorization_values(environ: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key, raw in environ.items():
        normalized = str(key).upper().replace("-", "_")
        if normalized != "HTTP_AUTHORIZATION" and not normalized.startswith(
            "HTTP_AUTHORIZATION_"
        ):
            continue
        if isinstance(raw, (tuple, list)):
            values.extend(str(item) for item in raw)
        elif raw is not None:
            values.append(str(raw))
    return values


def check_bearer_token(environ: Mapping[str, Any], expected: str) -> bool:
    """Validate exactly one ``Authorization: Bearer`` value."""

    values = _authorization_values(environ)
    if len(values) != 1:
        return False
    pieces = values[0].strip().split()
    if len(pieces) != 2 or pieces[0].lower() != "bearer":
        return False
    candidate = pieces[1]
    # A comma is how many WSGI servers combine duplicate header lines.
    if (
        not candidate
        or "," in candidate
        or not isinstance(expected, str)
        or not expected
    ):
        return False
    return hmac.compare_digest(candidate, expected)


def _content_type_is_json(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parser = email.message.Message()
    try:
        parser["content-type"] = value
        if parser.get_content_type().lower() != "application/json":
            return False
        # Reject malformed unkeyed parameters while allowing charset and other
        # standards-compliant parameters.
        for parameter in value.split(";")[1:]:
            parameter = parameter.strip()
            if parameter and "=" not in parameter:
                return False
        return True
    except (TypeError, ValueError):
        return False


def _header_values(environ: Mapping[str, Any], name: str) -> list[str]:
    normalized_name = name.upper().replace("-", "_")
    values: list[str] = []
    for key, raw in environ.items():
        key_name = str(key).upper().replace("-", "_")
        if key_name != normalized_name and not key_name.startswith(
            normalized_name + "_"
        ):
            continue
        if isinstance(raw, (tuple, list)):
            values.extend(str(item) for item in raw)
        elif raw is not None:
            values.append(str(raw))
    return values


def _content_type_values(environ: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("CONTENT_TYPE", "HTTP_CONTENT_TYPE"):
        raw = environ.get(key)
        if isinstance(raw, (tuple, list)):
            values.extend(str(item) for item in raw)
        elif raw is not None:
            values.append(str(raw))
    return values


def parse_content_length(environ: Mapping[str, Any]) -> int | None:
    """Return a single non-negative Content-Length, or raise a safe HTTP error."""

    values: list[str] = []
    for key in ("CONTENT_LENGTH", "HTTP_CONTENT_LENGTH"):
        raw = environ.get(key)
        if isinstance(raw, (tuple, list)):
            values.extend(str(item) for item in raw)
        elif raw is not None:
            values.append(str(raw))
    if not values:
        return None
    if len(values) != 1:
        raise _http_error("invalid_content_length", "Content-Length is invalid.", 400)
    raw = values[0].strip()
    if not raw or not raw.isascii() or not raw.isdigit():
        raise _http_error("invalid_content_length", "Content-Length is invalid.", 400)
    try:
        value = int(raw, 10)
    except (TypeError, ValueError, OverflowError) as error:
        raise _http_error(
            "invalid_content_length", "Content-Length is invalid.", 400
        ) from error
    return value


def read_json_body(
    environ: Mapping[str, Any],
    *,
    max_bytes: int,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Read and validate one bounded JSON object from a WSGI request."""

    declared = parse_content_length(environ)
    if declared is not None and declared > max_bytes:
        raise _http_error("payload_too_large", "Request body is too large.", 413)
    stream = environ.get("wsgi.input")
    if stream is None or not hasattr(stream, "read"):
        raise _http_error("invalid_json", "Request body is not valid JSON.", 400)
    try:
        if declared is None:
            raw = stream.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise _http_error(
                    "payload_too_large", "Request body is too large.", 413
                )
        else:
            if declared == 0 and allow_empty:
                return {}
            raw = stream.read(declared)
            if not isinstance(raw, (bytes, bytearray)) or len(raw) != declared:
                raise _http_error(
                    "invalid_json", "Request body is not valid JSON.", 400
                )
    except DebugServiceError:
        raise
    except Exception as error:  # noqa: BLE001 - normalize all stream failures.
        del error
        raise _http_error("invalid_json", "Request body is not valid JSON.", 400)
    if not isinstance(raw, (bytes, bytearray)):
        raise _http_error("invalid_json", "Request body is not valid JSON.", 400)
    if not raw and allow_empty:
        return {}
    try:
        text = bytes(raw).decode("utf-8")
        value = json.loads(text, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        raise _http_error("invalid_json", "Request body is not valid JSON.", 400)
    if not isinstance(value, dict):
        raise _http_error(
            "validation_failed",
            "Request validation failed.",
            400,
            field_errors={"body": "A JSON object is required."},
        )
    return value


parse_json_body = read_json_body


def _reject_json_constant(value: str) -> None:
    del value
    raise ValueError("non-standard JSON constant")


def _decode_api_segments(path: str) -> list[str] | None:
    """Decode API path components while rejecting separator injection."""

    if not path.startswith("/"):
        return None
    raw_segments = path.split("/")
    if len(raw_segments) < 3 or raw_segments[0] != "":
        return None
    decoded: list[str] = []
    for raw in raw_segments[1:]:
        if not raw:
            return None
        try:
            value = unquote(raw, errors="strict")
        except (UnicodeDecodeError, ValueError):
            return None
        if not value or "/" in value or "\\" in value:
            return None
        if value in {".", ".."}:
            return None
        decoded.append(value)
    return decoded


def _is_clone_path(path: str) -> bool:
    segments = _decode_api_segments(path)
    return (
        segments is not None
        and len(segments) == 4
        and segments[1] == "sessions"
        and segments[3] == "clone"
    )


def _validated_session_id(value: str) -> str | None:
    if _SESSION_ID_RE.fullmatch(value) is None:
        return None
    return value


def _validated_turn(value: str) -> int | None:
    if _TURN_RE.fullmatch(value) is None:
        return None
    return int(value, 10)


def _validated_asin(value: str) -> str | None:
    if _ASIN_RE.fullmatch(value) is None:
        return None
    return value


def _require_body(body: dict[str, Any] | None) -> dict[str, Any]:
    if body is None:
        raise _http_error("validation_failed", "Request validation failed.", 400)
    return body


def _body_fields(
    body: dict[str, Any] | None,
    *,
    required: set[str],
    allowed: set[str],
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    values = _require_body(body)
    unknown = set(values).difference(allowed)
    if unknown:
        raise _http_error(
            "validation_failed",
            "Request validation failed.",
            400,
            field_errors={key: "Field is not recognized." for key in sorted(unknown)},
        )
    aliases = aliases or {}
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        target = aliases.get(key, key)
        if target in normalized and normalized[target] != value:
            raise _http_error(
                "validation_failed",
                "Request validation failed.",
                400,
                field_errors={target: "Conflicting values were provided."},
            )
        normalized[target] = value
    missing = required.difference(normalized)
    if missing:
        raise _http_error(
            "validation_failed",
            "Request validation failed.",
            400,
            field_errors={key: "Field is required." for key in sorted(missing)},
        )
    return normalized


def _parse_scope(query_string: str) -> str:
    try:
        parsed = parse_qs(query_string, keep_blank_values=True, strict_parsing=False)
    except ValueError as error:
        raise _http_error(
            "validation_failed", "Request validation failed.", 400
        ) from error
    if not parsed:
        return "active"
    if set(parsed) != {"scope"} or len(parsed["scope"]) != 1:
        raise _http_error(
            "validation_failed",
            "Request validation failed.",
            400,
            field_errors={"scope": "Scope must be active, archived, or all."},
        )
    scope = parsed["scope"][0]
    if scope not in {"active", "archived", "all"}:
        raise _http_error(
            "validation_failed",
            "Request validation failed.",
            400,
            field_errors={"scope": "Scope must be active, archived, or all."},
        )
    return scope


def _existing_turn_status(
    repository: Any, session_id: str, request_id: object
) -> str | None:
    if not isinstance(request_id, str):
        return None
    finder = getattr(repository, "find_turn_by_request_id", None)
    if not callable(finder):
        return None
    try:
        record = finder(session_id, request_id)
    except Exception:  # noqa: BLE001 - service remains authoritative for errors.
        return None
    if record is None:
        return None
    if isinstance(record, Mapping):
        status = record.get("status")
    else:
        status = getattr(record, "status", None)
    return status if isinstance(status, str) else None


def _message_status(payload: object, existing_status: str | None) -> int:
    turn: object = payload.get("turn") if isinstance(payload, Mapping) else None
    status = turn.get("status") if isinstance(turn, Mapping) else None
    if status == "pending":
        return 202
    if existing_status == "completed":
        return 200
    return 201


class DebugApplication:
    """WSGI callable containing only the debug shell's explicit dependencies."""

    def __init__(
        self,
        config: DebugConfig,
        repository: Any,
        worker: Any,
        service: Any,
        *,
        repository_error: BaseException | None = None,
        startup_error: BaseException | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.worker = worker
        self.service = service
        self.repository_error = repository_error
        self.startup_error = startup_error
        self._closed = False

    def __call__(self, environ: Mapping[str, Any], start_response: Callable[..., Any]):
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO", ""))
        body: dict[str, Any] | None = None
        request_id: str | None = None
        try:
            if path == "/api/health/live":
                if method != "GET":
                    raise _method_not_allowed()
                return self._json(start_response, 200, {"status": "live"})
            if path == "/api/health/ready":
                if method != "GET":
                    raise _method_not_allowed()
                return self._ready(start_response)
            if path in STATIC_ALLOWLIST:
                if method not in {"GET", "HEAD"}:
                    raise _method_not_allowed()
                return self._static(start_response, path, head=method == "HEAD")

            if path.startswith("/api/"):
                if not check_bearer_token(environ, self.config.access_token):
                    raise AuthenticationError()
                if method in _MUTATION_METHODS:
                    content_values = _content_type_values(environ)
                    content_type = content_values[0] if len(content_values) == 1 else ""
                    if not _content_type_is_json(content_type):
                        raise _http_error(
                            "unsupported_media_type",
                            "Content-Type must be application/json.",
                            415,
                        )
                    limit = (
                        self.config.max_import_bytes
                        if _decode_api_segments(path) == ["api", "import"]
                        else self.config.max_body_bytes
                    )
                    body = read_json_body(
                        environ,
                        max_bytes=limit,
                        allow_empty=_is_clone_path(path),
                    )
                    candidate_request_id = body.get("request_id")
                    if isinstance(candidate_request_id, str) and candidate_request_id:
                        request_id = candidate_request_id
                return self._dispatch_api(
                    start_response,
                    method,
                    path,
                    str(environ.get("QUERY_STRING", "")),
                    body,
                )
            raise _not_found()
        except DebugServiceError as error:
            return self._error(start_response, error, request_id=request_id)
        except Exception:  # noqa: BLE001 - never expose server internals.
            # Never serialize exception text, local paths, or object reprs.
            return self._error(
                start_response,
                _http_error(
                    "internal_error",
                    "The server could not complete the request.",
                    500,
                    retryable=True,
                ),
                request_id=request_id,
            )

    def _dispatch_api(
        self,
        start_response: Callable[..., Any],
        method: str,
        path: str,
        query_string: str,
        body: dict[str, Any] | None,
    ):
        """Dispatch one authenticated API request to the debug service."""

        segments = _decode_api_segments(path)
        if segments is None:
            raise _not_found()

        if segments == ["api", "import"]:
            if method != "POST":
                raise _method_not_allowed()
            payload = self.service.import_session(_require_body(body))
            return self._json(start_response, 201, payload)

        if len(segments) < 2 or segments[1] != "sessions":
            raise _not_found()

        if len(segments) == 2:
            if method == "GET":
                scope = _parse_scope(query_string)
                sessions = self.service.list_sessions(scope)
                return self._json(start_response, 200, {"sessions": sessions})
            if method == "POST":
                values = _body_fields(
                    body,
                    required={"name", "profile"},
                    allowed={"name", "profile"},
                )
                payload = self.service.create_session(values["name"], values["profile"])
                return self._json(start_response, 201, payload)
            raise _method_not_allowed()

        session_id = _validated_session_id(segments[2])
        if session_id is None:
            raise _not_found()

        if len(segments) == 3:
            if method == "GET":
                return self._json(
                    start_response, 200, self.service.get_session(session_id)
                )
            if method == "PATCH":
                values = _body_fields(
                    body,
                    required=set(),
                    allowed={"name", "archived"},
                )
                return self._json(
                    start_response,
                    200,
                    self.service.patch_session(session_id, **values),
                )
            raise _method_not_allowed()

        if len(segments) == 4 and segments[3] == "messages":
            if method != "POST":
                raise _method_not_allowed()
            values = _body_fields(
                body,
                required={"request_id", "user_message"},
                allowed={"request_id", "user_message"},
            )
            existing_status = _existing_turn_status(
                self.repository, session_id, values["request_id"]
            )
            payload = self.service.send_message(
                session_id, values["request_id"], values["user_message"]
            )
            status = _message_status(payload, existing_status)
            return self._json(start_response, status, payload)

        if len(segments) == 4 and segments[3] == "clone":
            if method != "POST":
                raise _method_not_allowed()
            values = _body_fields(
                body,
                required=set(),
                allowed={"through_turn"},
            )
            payload = self.service.clone_session(session_id, values.get("through_turn"))
            return self._json(start_response, 201, payload)

        if len(segments) == 4 and segments[3] == "export":
            if method != "GET":
                raise _method_not_allowed()
            return self._json(
                start_response, 200, self.service.export_session(session_id)
            )

        if len(segments) == 7 and segments[3] == "turns" and segments[5] == "feedback":
            turn = _validated_turn(segments[4])
            asin = _validated_asin(segments[6])
            if turn is None or asin is None:
                raise _not_found()
            if method != "PUT":
                raise _method_not_allowed()
            values = _body_fields(
                body,
                required={"incorrect"},
                allowed={"incorrect", "is_inaccurate", "reason", "note"},
                aliases={"is_inaccurate": "incorrect"},
            )
            feedback = self.service.set_feedback(
                session_id,
                turn,
                asin,
                values["incorrect"],
                values.get("reason"),
                values.get("note", ""),
            )
            return self._json(start_response, 200, {"feedback": feedback})

        raise _not_found()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.worker, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - cleanup must be best effort.
                # Cleanup must not expose worker internals to callers.
                return

    def _ready(self, start_response: Callable[..., Any]):
        if self.startup_error is not None:
            return self._json(start_response, 503, {"status": "fatal"})
        try:
            state = str(getattr(self.worker, "state", "stopped"))
        except Exception:  # noqa: BLE001 - readiness must fail closed.
            state = "fatal"
        if state != "ready":
            return self._json(start_response, 503, {"status": state})
        try:
            healthy = self.repository_error is None and bool(self.repository.health())
        except Exception:  # noqa: BLE001 - readiness must fail closed.
            healthy = False
        if not healthy:
            return self._json(start_response, 503, {"status": "database_unavailable"})
        return self._json(start_response, 200, {"status": "ready"})

    def _static(self, start_response: Callable[..., Any], path: str, *, head: bool):
        relative_paths, content_type = STATIC_ALLOWLIST[path]
        root = Path(self.config.static_root).resolve()
        data: bytes | None = None
        try:
            for relative in relative_paths:
                candidate = (root / relative).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    continue
                if not candidate.is_file():
                    continue
                data = candidate.read_bytes()
                break
        except (OSError, ValueError):
            data = None
        if data is None:
            raise _not_found()
        return self._bytes(start_response, 200, data, content_type, head=head)

    def _json(self, start_response: Callable[..., Any], status: int, payload: Any):
        try:
            data = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError):
            data = b'{"error":{"code":"internal_error","message":"The server could not complete the request.","retryable":true}}'
            status = 500
        return self._bytes(start_response, status, data, JSON_CONTENT_TYPE)

    def _error(
        self,
        start_response: Callable[..., Any],
        error: DebugServiceError,
        *,
        request_id: str | None = None,
    ):
        payload = error.to_payload()
        if request_id:
            payload["request_id"] = request_id
        return self._json(start_response, error.http_status, payload)

    @staticmethod
    def _bytes(
        start_response: Callable[..., Any],
        status: int,
        data: bytes,
        content_type: str,
        *,
        head: bool = False,
    ):
        reason = http.client.responses.get(status, "")
        headers = [
            ("Content-Type", content_type),
            ("Content-Length", str(len(data))),
            ("X-Content-Type-Options", "nosniff"),
            ("Referrer-Policy", "no-referrer"),
            ("Cache-Control", "no-store"),
            ("Content-Security-Policy", CSP),
            ("X-Frame-Options", "DENY"),
        ]
        start_response(f"{status} {reason}", headers)
        return [b""] if head else [data]


# Public aliases make the small shell easy to discover without coupling callers
# to a particular class spelling.
WSGIApplication = DebugApplication
DebugWSGIApplication = DebugApplication


def create_application(
    config: DebugConfig | None = None,
    *,
    repository: Any | None = None,
    worker: Any | None = None,
    service: Any | None = None,
    agent_factory: Callable[..., Any] | None = None,
) -> DebugApplication:
    """Build the shell, initialize SQLite, and start Agent work asynchronously."""

    if config is None:
        config = DebugConfig.from_env()
    selected_repository = (
        repository if repository is not None else DebugRepository(config.database_path)
    )
    repository_error: BaseException | None = None
    initialize = getattr(selected_repository, "initialize", None)
    if callable(initialize):
        try:
            initialize()
        except Exception as error:  # noqa: BLE001 - preserve a safe readiness state.
            repository_error = error

    selected_worker = worker
    if selected_worker is None:
        selected_worker = AgentWorker(agent_factory, config=config)
    selected_service = (
        service
        if service is not None
        else DebugService(selected_repository, selected_worker)
    )
    startup_error: BaseException | None = None
    start = getattr(selected_worker, "start", None)
    if callable(start):
        try:
            start()
        except Exception as error:  # noqa: BLE001 - preserve a safe startup state.
            startup_error = error
    return DebugApplication(
        config,
        selected_repository,
        selected_worker,
        selected_service,
        repository_error=repository_error,
        startup_error=startup_error,
    )


__all__ = [
    "CSP",
    "STATIC_ALLOWLIST",
    "STATIC_MAP",
    "DebugApplication",
    "DebugWSGIApplication",
    "WSGIApplication",
    "check_bearer_token",
    "create_application",
    "parse_content_length",
    "parse_json_body",
    "read_json_body",
]
