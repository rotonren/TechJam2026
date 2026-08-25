"""Small, dependency-free WSGI shell for the CompassCart debug service."""

from __future__ import annotations

import email.message
import hmac
import http.client
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

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
                        if path == "/api/import"
                        else self.config.max_body_bytes
                    )
                    read_json_body(environ, max_bytes=limit)
                # Task 7 adds the service routes.  Authentication and body
                # protections intentionally remain active for unknown routes.
                raise _not_found()
            raise _not_found()
        except DebugServiceError as error:
            return self._error(start_response, error)
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
            )

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

    def _error(self, start_response: Callable[..., Any], error: DebugServiceError):
        return self._json(start_response, error.http_status, error.to_payload())

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
