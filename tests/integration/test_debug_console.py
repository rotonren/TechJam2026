from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from compasscart_debug.config import DebugConfig
from compasscart_debug.http import create_application

TOKEN = "i" * 43


def _config(catalog: Path, database: Path, tmp_path: Path) -> DebugConfig:
    static_root = tmp_path / "static"
    static_root.mkdir(parents=True, exist_ok=True)
    return DebugConfig(
        catalog_path=catalog,
        database_path=database,
        static_root=static_root,
        asset_root=tmp_path / "assets",
        access_token=TOKEN,
        max_body_bytes=1_048_576,
        max_import_bytes=4_194_304,
        command_timeout_seconds=10.0,
    )


def _request(
    app: Any,
    path: str,
    *,
    method: str = "GET",
    payload: object | None = None,
) -> tuple[int, dict[str, str], Any]:
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    environ: dict[str, Any] = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": "",
        "SERVER_NAME": "localhost",
        "SERVER_PORT": "80",
        "SERVER_PROTOCOL": "HTTP/1.1",
        "wsgi.url_scheme": "http",
        "wsgi.input": io.BytesIO(body),
        "wsgi.errors": io.StringIO(),
        "wsgi.version": (1, 0),
        "wsgi.multithread": False,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
        "CONTENT_LENGTH": str(len(body)),
    }
    environ["HTTP_AUTHORIZATION"] = f"Bearer {TOKEN}"
    if method in {"POST", "PUT", "PATCH", "DELETE"}:
        environ["CONTENT_TYPE"] = "application/json"

    captured: dict[str, Any] = {}

    def start_response(status: str, headers: list[tuple[str, str]], exc_info=None):
        del exc_info
        captured["status"] = int(status.split(" ", 1)[0])
        captured["headers"] = dict(headers)

    data = b"".join(app(environ, start_response))
    parsed = json.loads(data.decode("utf-8")) if data else None
    return captured["status"], captured["headers"], parsed


def _start(catalog: Path, database: Path, tmp_path: Path):
    app = create_application(_config(catalog, database, tmp_path))
    assert app.worker.wait_until_ready(10.0), getattr(app.worker, "startup_error", None)
    return app


def test_real_fixture_console_persists_rehydrates_and_shares_history(
    fixture_catalog_path: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("COMPASSCART_DISABLE_DENSE", "1")
    database = tmp_path / "debug.sqlite3"

    first = _start(fixture_catalog_path, database, tmp_path / "first")
    try:
        status, _, created = _request(
            first,
            "/api/sessions",
            method="POST",
            payload={"name": "Running shoes", "profile": {"preference_tags": []}},
        )
        assert status == 201
        session_id = created["session"]["session_id"]

        first_turn_status, _, first_turn = _request(
            first,
            f"/api/sessions/{session_id}/messages",
            method="POST",
            payload={
                "request_id": "r1",
                "user_message": "I need running shoes under $80",
            },
        )
        second_turn_status, _, second_turn = _request(
            first,
            f"/api/sessions/{session_id}/messages",
            method="POST",
            payload={"request_id": "r2", "user_message": "blue mesh and wide fit"},
        )
        assert (first_turn_status, second_turn_status) == (201, 201)
        for payload in (first_turn, second_turn):
            products = payload["turn"]["products"]
            assert [item["rank"] for item in products] == list(
                range(1, len(products) + 1)
            )

        asin = first_turn["turn"]["products"][0]["parent_asin"]
        feedback_status, _, feedback = _request(
            first,
            f"/api/sessions/{session_id}/turns/1/feedback/{asin}",
            method="PUT",
            payload={
                "is_inaccurate": True,
                "reason": "attribute_mismatch",
                "note": "wide fit was not clear",
            },
        )
        assert feedback_status == 200
        assert feedback["feedback"]["parent_asin"] == asin

        exported_status, _, exported = _request(
            first, f"/api/sessions/{session_id}/export"
        )
        assert exported_status == 200
        assert exported["format"] == "compasscart-debug-session"

        clone_status, _, clone = _request(
            first,
            f"/api/sessions/{session_id}/clone",
            method="POST",
            payload={"through_turn": 1},
        )
        assert clone_status == 201
        assert clone["session"]["source_session_id"] == session_id
        assert [turn["turn"] for turn in clone["turns"]] == [1]

        archived_status, _, _ = _request(
            first,
            f"/api/sessions/{session_id}",
            method="PATCH",
            payload={"archived": True},
        )
        assert archived_status == 200
    finally:
        first.close()

    second = _start(fixture_catalog_path, database, tmp_path / "second")
    try:
        detail_status, _, detail = _request(second, f"/api/sessions/{session_id}")
        assert detail_status == 200
        assert [turn["turn"] for turn in detail["turns"]] == [1, 2]
        assert detail["turns"][0]["feedback"][0]["parent_asin"] == asin
        assert detail["session"]["archived"] is True

        blocked_status, _, blocked = _request(
            second,
            f"/api/sessions/{session_id}/messages",
            method="POST",
            payload={"request_id": "archived-r3", "user_message": "more shoes"},
        )
        assert blocked_status == 409
        assert blocked["error"]["code"] == "conflict"
    finally:
        second.close()

    imported_app = _start(fixture_catalog_path, database, tmp_path / "imported")
    try:
        _, _, source_export = _request(
            imported_app, f"/api/sessions/{session_id}/export"
        )
        import_status, _, imported = _request(
            imported_app,
            "/api/import",
            method="POST",
            payload=source_export,
        )
        assert import_status == 201
        assert imported["session"]["session_id"] != session_id
        assert len(imported["turns"]) == 2
    finally:
        imported_app.close()
