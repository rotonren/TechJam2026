from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from compasscart_debug.config import DebugConfig
from compasscart_debug.http import create_application

TOKEN = "t" * 43


class FakeRepository:
    def __init__(
        self, healthy: bool = True, *, initialize_error: BaseException | None = None
    ):
        self.healthy = healthy
        self.initialize_error = initialize_error
        self.initialize_calls = 0
        self.health_calls = 0

    def initialize(self) -> None:
        self.initialize_calls += 1
        if self.initialize_error is not None:
            raise self.initialize_error

    def health(self) -> bool:
        self.health_calls += 1
        return self.healthy


class FakeWorker:
    def __init__(self, state: str = "ready"):
        self.state = state
        self.start_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def close(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.close_calls += 1


class StartFailingWorker(FakeWorker):
    def start(self) -> None:
        self.start_calls += 1
        raise RuntimeError("startup secret details")


class FakeService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        def method(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.calls.append(name)
            raise AssertionError(f"service unexpectedly called: {name}")

        return method


class SentinelInput(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.read_calls = 0

    def read(self, *args: Any, **kwargs: Any) -> bytes:
        self.read_calls += 1
        return super().read(*args, **kwargs)


def _config(tmp_path: Path, *, max_body_bytes: int = 64) -> DebugConfig:
    static_root = tmp_path / "static"
    static_root.mkdir()
    return DebugConfig(
        catalog_path=tmp_path / "catalog.jsonl",
        database_path=tmp_path / "debug.sqlite3",
        static_root=static_root,
        asset_root=tmp_path / "assets",
        access_token=TOKEN,
        max_body_bytes=max_body_bytes,
    )


def _make_app(
    tmp_path: Path,
    *,
    worker: FakeWorker | None = None,
    repository: FakeRepository | None = None,
    service: FakeService | None = None,
    max_body_bytes: int = 64,
):
    repo = repository or FakeRepository()
    selected_worker = worker or FakeWorker()
    selected_service = service or FakeService()
    app = create_application(
        _config(tmp_path, max_body_bytes=max_body_bytes),
        repository=repo,
        worker=selected_worker,
        service=selected_service,
        agent_factory=lambda: object(),
    )
    return app, repo, selected_worker, selected_service


def _request(
    app: Any,
    path: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes = b"",
    input_stream: io.BytesIO | None = None,
):
    environ: dict[str, Any] = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": "",
        "SERVER_NAME": "localhost",
        "SERVER_PORT": "80",
        "SERVER_PROTOCOL": "HTTP/1.1",
        "wsgi.url_scheme": "http",
        "wsgi.input": input_stream or io.BytesIO(body),
        "wsgi.errors": io.StringIO(),
        "wsgi.version": (1, 0),
        "wsgi.multithread": False,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
        "CONTENT_LENGTH": str(len(body)),
    }
    for key, value in (headers or {}).items():
        environ["HTTP_" + key.upper().replace("-", "_")] = value
    captured: dict[str, Any] = {}

    def start_response(
        status: str, response_headers: list[tuple[str, str]], exc_info=None
    ):
        del exc_info
        captured["status"] = status
        captured["headers"] = dict(response_headers)

    chunks = app(environ, start_response)
    payload = b"".join(chunks)
    close = getattr(chunks, "close", None)
    if close is not None:
        close()
    status_code = int(captured["status"].split(" ", 1)[0])
    content_type = captured["headers"].get("Content-Type", "")
    parsed: Any = None
    if content_type.startswith("application/json") and payload:
        parsed = json.loads(payload.decode("utf-8"))
    return status_code, captured["headers"], payload, parsed


def test_live_is_exact_and_never_touches_service_or_repository(tmp_path: Path) -> None:
    app, repository, worker, service = _make_app(tmp_path)
    try:
        status, headers, body, parsed = _request(app, "/api/health/live")
    finally:
        app.close()
    assert (status, parsed) == (200, {"status": "live"})
    assert body == b'{"status":"live"}'
    assert service.calls == []
    assert repository.health_calls == 0
    assert worker.start_calls == 1
    assert headers["Content-Type"] == "application/json; charset=utf-8"


@pytest.mark.parametrize(
    ("state", "status", "payload"),
    [
        ("initializing", 503, {"status": "initializing"}),
        ("setup_required", 503, {"status": "setup_required"}),
        ("fatal", 503, {"status": "fatal"}),
        ("stopped", 503, {"status": "stopped"}),
        ("ready", 200, {"status": "ready"}),
    ],
)
def test_ready_reports_worker_state_without_agent_calls(
    tmp_path: Path, state: str, status: int, payload: dict[str, str]
) -> None:
    worker = FakeWorker(state)
    app, repository, _, service = _make_app(tmp_path, worker=worker)
    try:
        actual_status, _, _, actual_payload = _request(app, "/api/health/ready")
    finally:
        app.close()
    assert (actual_status, actual_payload) == (status, payload)
    assert service.calls == []
    assert repository.health_calls == (1 if state == "ready" else 0)


def test_ready_maps_unhealthy_database_to_stable_status(tmp_path: Path) -> None:
    app, repository, _, _ = _make_app(tmp_path, repository=FakeRepository(False))
    try:
        status, _, _, payload = _request(app, "/api/health/ready")
    finally:
        app.close()
    assert (status, payload) == (503, {"status": "database_unavailable"})
    assert repository.health_calls == 1


def test_ready_fails_closed_when_worker_start_raises_with_ready_state(
    tmp_path: Path,
) -> None:
    worker = StartFailingWorker("ready")
    repository = FakeRepository(True)
    app, _, _, _ = _make_app(tmp_path, worker=worker, repository=repository)
    try:
        status, _, _, payload = _request(app, "/api/health/ready")
    finally:
        app.close()
    assert (status, payload) == (503, {"status": "fatal"})
    assert repository.health_calls == 0


def test_security_headers_are_present_on_static_and_error_responses(
    tmp_path: Path,
) -> None:
    app, _, _, _ = _make_app(tmp_path)
    (Path(app.config.static_root) / "index.html").write_text(
        "<html>ok</html>", encoding="utf-8"
    )
    try:
        responses = [
            _request(app, "/"),
            _request(app, "/missing"),
            _request(app, "/api/sessions"),
        ]
    finally:
        app.close()
    expected = {
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-store",
        "Content-Security-Policy": (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'none'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        ),
    }
    for _, headers, _, _ in responses:
        for name, value in expected.items():
            assert headers[name] == value
        assert "Access-Control-Allow-Origin" not in headers


@pytest.mark.parametrize(
    "authorization",
    [None, "", "Basic abc", f"Bearer {TOKEN}, Bearer {TOKEN}", "Bearer wrong"],
)
def test_protected_api_requires_exactly_one_bearer_token(
    tmp_path: Path, authorization: str | None
) -> None:
    headers = {} if authorization is None else {"Authorization": authorization}
    app, _, _, _ = _make_app(tmp_path)
    try:
        status, _, _, payload = _request(app, "/api/unknown", headers=headers)
    finally:
        app.close()
    assert status == 401
    assert payload == {
        "error": {
            "code": "authentication_failed",
            "message": "Authentication failed.",
            "retryable": False,
        }
    }


def test_authentication_uses_constant_time_compare(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from compasscart_debug import http

    calls: list[tuple[str, str]] = []
    original = http.hmac.compare_digest

    def compare(left: str, right: str) -> bool:
        calls.append((left, right))
        return original(left, right)

    monkeypatch.setattr(http.hmac, "compare_digest", compare)
    app, _, _, _ = _make_app(tmp_path)
    try:
        status, _, _, _ = _request(
            app, "/api/unknown", headers={"Authorization": f"Bearer {TOKEN}"}
        )
    finally:
        app.close()
    assert status == 404
    assert calls == [(TOKEN, TOKEN)]


def test_query_and_cookie_tokens_do_not_authenticate(tmp_path: Path) -> None:
    app, _, _, _ = _make_app(tmp_path)
    try:
        status_query, _, _, _ = _request(app, f"/api/unknown?token={quote(TOKEN)}")
        status_cookie, _, _, _ = _request(
            app, "/api/unknown", headers={"Cookie": f"token={TOKEN}"}
        )
    finally:
        app.close()
    assert (status_query, status_cookie) == (401, 401)


def test_mutations_require_json_content_type(tmp_path: Path) -> None:
    app, _, _, _ = _make_app(tmp_path)
    try:
        status, _, _, payload = _request(
            app,
            "/api/unknown",
            method="POST",
            headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "text/plain"},
            body=b"{}",
        )
    finally:
        app.close()
    assert status == 415
    assert payload["error"]["code"] == "unsupported_media_type"


def test_oversized_content_length_is_rejected_before_reading_body(
    tmp_path: Path,
) -> None:
    sentinel = SentinelInput(b"{}")
    app, _, _, _ = _make_app(tmp_path, max_body_bytes=1)
    try:
        status, _, _, payload = _request(
            app,
            "/api/unknown",
            method="POST",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
            },
            body=b"{}",
            input_stream=sentinel,
        )
    finally:
        app.close()
    assert status == 413
    assert payload["error"]["code"] == "payload_too_large"
    assert sentinel.read_calls == 0


def test_malformed_json_returns_stable_validation_error(tmp_path: Path) -> None:
    app, _, _, _ = _make_app(tmp_path)
    try:
        status, _, _, payload = _request(
            app,
            "/api/unknown",
            method="POST",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json; charset=utf-8",
            },
            body=b"{not-json",
        )
    finally:
        app.close()
    assert status == 400
    assert payload["error"]["code"] == "invalid_json"
    assert "path" not in json.dumps(payload).lower()


@pytest.mark.parametrize(
    ("path", "filename", "mime"),
    [
        ("/", "index.html", "text/html; charset=utf-8"),
        ("/static/styles.css", "styles.css", "text/css; charset=utf-8"),
        ("/static/js/api.js", "api.js", "text/javascript; charset=utf-8"),
        ("/static/js/dom.js", "dom.js", "text/javascript; charset=utf-8"),
        ("/static/js/store.js", "store.js", "text/javascript; charset=utf-8"),
        ("/static/js/app.js", "app.js", "text/javascript; charset=utf-8"),
    ],
)
def test_static_paths_use_exact_allowlist_and_mime(
    tmp_path: Path, path: str, filename: str, mime: str
) -> None:
    app, _, _, _ = _make_app(tmp_path)
    target = Path(app.config.static_root) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(filename, encoding="utf-8")
    try:
        status, headers, body, _ = _request(app, path)
    finally:
        app.close()
    assert status == 200
    assert headers["Content-Type"] == mime
    assert body == filename.encode()


@pytest.mark.parametrize(
    "path",
    [
        "/static/secret.txt",
        "/static/../secret.txt",
        "/static/%2e%2e/secret.txt",
        "/static/%2E%2E%2Fsecret.txt",
        "/static/js/../secret.txt",
    ],
)
def test_static_traversal_and_unknown_paths_are_404(tmp_path: Path, path: str) -> None:
    app, _, _, _ = _make_app(tmp_path)
    (Path(app.config.static_root).parent / "secret.txt").write_text(
        "secret", encoding="utf-8"
    )
    try:
        status, _, _, payload = _request(app, path)
    finally:
        app.close()
    assert status == 404
    assert payload["error"]["code"] == "not_found"
    assert "secret" not in json.dumps(payload).lower()


def test_blocking_agent_factory_does_not_block_application_startup(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def factory():
        entered.set()
        release.wait(2.0)
        return object()

    config = _config(tmp_path)
    repository = FakeRepository()
    started = time.monotonic()
    app = create_application(config, repository=repository, agent_factory=factory)
    elapsed = time.monotonic() - started
    try:
        assert elapsed < 0.5
        assert entered.wait(1.0)
        live_status, _, _, live_payload = _request(app, "/api/health/live")
        ready_status, _, _, ready_payload = _request(app, "/api/health/ready")
        assert (live_status, live_payload) == (200, {"status": "live"})
        assert (ready_status, ready_payload) == (503, {"status": "initializing"})
    finally:
        app.close()
        release.set()


def test_error_envelopes_do_not_expose_secret_or_exception_details(
    tmp_path: Path,
) -> None:
    app, _, _, _ = _make_app(tmp_path)
    try:
        status, _, _, payload = _request(
            app,
            "/api/unknown",
            method="POST",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
            },
            body=b"not-json",
        )
    finally:
        app.close()
    serialized = json.dumps(payload)
    assert status == 400
    assert TOKEN not in serialized
    assert "Traceback" not in serialized
    assert "ValueError" not in serialized
    assert str(tmp_path) not in serialized
