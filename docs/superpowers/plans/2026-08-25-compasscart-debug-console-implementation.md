# CompassCart Persistent Debug Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable, persistent three-pane debugging console that calls the current CompassCart Agent unchanged, records reproducible multi-turn observations and human feedback, and can run locally or as a single-instance Docker/Render service.

**Architecture:** A WSGI process serves the UI and short-lived SQLite repository operations. Exactly one dedicated worker thread constructs and owns the existing `agent.Agent`; every reset, response, catalog lookup, state read, and trace read is serialized through that worker to preserve the catalog SQLite thread affinity. The official submission allowlist remains unchanged, while the debug companion adds its own requirements, static UI, runtime-asset installer, Docker configuration, and deployment documentation.

**Tech Stack:** Python 3.12, standard-library WSGI/SQLite/threading/zipfile, existing CompassCart Agent and ONNX runtime, native HTML/CSS/ES modules, Gunicorn gthread for Linux containers, Docker Compose, Render Blueprint, pytest, Node built-in tests, Browser/IAB verification.

---

### Task 4: Orchestrate Idempotent Turns and Dirty-State Recovery

**Files:**
- Create: `src/compasscart_debug/service.py`
- Create: `tests/unit/test_debug_service.py`

- [ ] **Step 1: Write failing create/send/idempotency tests**

```python
def test_create_session_resets_once_and_persists_profile(service, worker):
    detail = service.create_session(
        name="Blue shoes", profile={"preference_tags": ["comfort"]}
    )
    assert worker.reset_calls == [(detail["session"]["session_id"], {
        "preference_tags": ["comfort"]
    })]
    assert detail["turns"] == []


def test_send_message_returns_completed_idempotent_result(service, worker):
    session = service.create_session(name="Test", profile={})["session"]
    first = service.send_message(
        session["session_id"], request_id="req-1", message="blue shoes"
    )
    second = service.send_message(
        session["session_id"], request_id="req-1", message="blue shoes"
    )
    assert first == second
    assert worker.observe_calls == [(session["session_id"], "blue shoes", 1)]
```

- [ ] **Step 2: Run service tests and verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_service.py -v
```

Expected: collection fails because `DebugService` is missing.

- [ ] **Step 3: Implement create, list, detail, patch, and message orchestration**

`DebugService` owns `DebugRepository`, `AgentWorker`, `RuntimeIdentity`, UUID/time callables, and a process-local set of dirty/loaded session IDs. Implement this exact send order:

```python
def send_message(self, session_id: str, *, request_id: str, message: str):
    message = message.strip()
    if not message:
        raise ValidationError({"user_message": "required"})
    existing = self.repository.get_turn_by_request(session_id, request_id)
    if existing and existing.user_message != message:
        raise ConflictError("request_id_message_mismatch")
    if existing and existing.status == "completed":
        return self._session_and_turn(session_id, existing.turn)
    if existing and existing.status == "failed":
        pending = self.repository.retry_failed(session_id, request_id, message)
    elif existing and existing.status == "pending":
        pending = existing
    else:
        pending = self.repository.reserve_turn(session_id, request_id, message)

    self._ensure_hydrated(session_id)
    try:
        observation = self.worker.observe_turn(session_id, message, pending.turn)
    except Exception as error:
        safe = {"code": "agent_error", "message": "The Agent could not complete this turn."}
        self.repository.fail_turn(session_id, pending.turn, safe)
        self._mark_dirty(session_id)
        raise DebugServiceError("agent_error", safe["message"], 500, True) from error
    try:
        completed = self.repository.complete_turn(session_id, pending.turn, observation)
    except Exception:
        self._mark_dirty(session_id)
        raise DebugServiceError(
            "snapshot_not_saved",
            "The Agent responded, but the debug snapshot was not saved.",
            503,
            True,
        )
    self.repository.mark_dirty(session_id, False)
    self._dirty_sessions.discard(session_id)
    return self._session_and_turn(session_id, completed.turn)
```

If `complete_turn()` fails and the repository can still be reached, leave the durable row pending. Do not return the unsaved Agent response as a successful turn. `create_session()` calls worker reset before creating the DB record; a DB failure may leave an unused in-memory session but no visible partial record.

- [ ] **Step 4: Write failing mutation-before-error and DB-failure recovery tests**

```python
def test_uncaught_agent_error_marks_failed_and_rehydrates_before_retry(service):
    session_id = service.create_session(name="x", profile={})["session"]["session_id"]
    service.worker.fail_after_mutating_once = True
    with pytest.raises(DebugServiceError) as captured:
        service.send_message(session_id, request_id="r1", message="blue")
    assert captured.value.code == "agent_error"
    result = service.send_message(session_id, request_id="r1", message="blue")
    assert result["turn"]["status"] == "completed"
    assert service.worker.events[:3] == ["reset", "failed_observe", "reset"]


def test_completion_commit_failure_never_continues_advanced_memory(service):
    session_id = service.create_session(name="x", profile={})["session"]["session_id"]
    service.repository.fail_next_completion = True
    with pytest.raises(DebugServiceError, match="not saved"):
        service.send_message(session_id, request_id="r1", message="blue")
    service.repository.fail_next_completion = False
    service.send_message(session_id, request_id="r1", message="blue")
    assert service.worker.reset_count == 2
```

- [ ] **Step 5: Implement full canonical rehydration checks**

Before any continuation, `_ensure_hydrated()` checks all four stored/current identity fields and whether `worker.has_session(session_id)` remains true. It then replays only completed turns when the process has not loaded the session, the worker evicted it, the DB/session is dirty, or a pending/failed retry is occurring.

Canonical equality includes the full response and state but removes trace timing:

```python
def replay_matches(stored: TurnRecord, fresh: TurnObservation) -> bool:
    return (
        canonical_json(stored.response) == canonical_json(fresh.response)
        and canonical_json(stored.state) == canonical_json(fresh.state)
    )
```

On mismatch, set `read_only_reason="replay_mismatch"` and raise `ReplayMismatchError`; never accept a new message on that historical session. Rehydration trace records are discarded because the adapter returns them only to the comparison call.

- [ ] **Step 6: Run service, repository, and worker tests; commit**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_service.py tests/unit/test_debug_repository.py tests/unit/test_debug_worker.py -v
git add src/compasscart_debug/service.py tests/unit/test_debug_service.py
git commit -m "feat: recover persistent debug conversations"
```

Expected: PASS, including a retry after an Agent mutation and a retry after DB completion failure.

---

### Task 5: Add Clone, Feedback, Import, Export, and Archive Workflows

**Files:**
- Modify: `src/compasscart_debug/repository.py`
- Modify: `src/compasscart_debug/service.py`
- Modify: `tests/unit/test_debug_repository.py`
- Modify: `tests/unit/test_debug_service.py`

- [ ] **Step 1: Write failing feedback and archive tests**

```python
def test_feedback_can_be_saved_updated_and_cleared(service, completed_session):
    sid = completed_session["session"]["session_id"]
    saved = service.set_feedback(
        sid, 1, "SHOE1", is_inaccurate=True,
        reason="over_budget", note="Explicit budget was $80"
    )
    assert saved["reason"] == "over_budget"
    detail = service.get_session(sid)
    assert detail["turns"][0]["feedback"][0]["parent_asin"] == "SHOE1"
    assert service.set_feedback(
        sid, 1, "SHOE1", is_inaccurate=False, reason=None, note=""
    ) is None


def test_archived_session_is_listed_and_can_be_restored(service, completed_session):
    sid = completed_session["session"]["session_id"]
    service.patch_session(sid, archived=True)
    assert all(row["session_id"] != sid for row in service.list_sessions("active"))
    assert any(row["session_id"] == sid for row in service.list_sessions("archived"))
    service.patch_session(sid, archived=False)
    assert any(row["session_id"] == sid for row in service.list_sessions("active"))
```

Feedback reasons are exactly `explicit_constraint`, `wrong_category`, `over_budget`, `attribute_mismatch`, `duplicate_or_too_similar`, and `other`. Reject feedback for an ASIN not present in the selected completed turn.

- [ ] **Step 2: Write failing export/import schema tests**

```python
def test_export_import_assigns_new_id_and_preserves_immutable_history(service, completed_session):
    sid = completed_session["session"]["session_id"]
    payload = service.export_session(sid)
    assert payload["format"] == "compasscart-debug-session"
    assert payload["schema_version"] == 1
    assert "access_token" not in json.dumps(payload)
    imported = service.import_session(payload)
    assert imported["session"]["session_id"] != sid
    assert imported["turns"][0]["response"] == payload["turns"][0]["response"]


@pytest.mark.parametrize("payload", [
    {},
    {"format": "wrong", "schema_version": 1},
    {"format": "compasscart-debug-session", "schema_version": 99},
])
def test_import_rejects_invalid_envelopes(service, payload):
    with pytest.raises(ValidationError):
        service.import_session(payload)
```

Export contains `format`, `schema_version`, `exported_at`, `session`, and `turns`; it omits access tokens, DB paths, runtime paths, exception reprs, and hostnames. Import validates all field types, turn ordering, status transitions, feedback ASIN membership, and a maximum of 10 turns. Imported observations remain immutable and receive a new session ID.

- [ ] **Step 3: Write failing current-version clone tests**

```python
def test_clone_replays_messages_and_keeps_new_observations(service, completed_session):
    original = completed_session
    clone = service.clone_session(
        original["session"]["session_id"], through_turn=1, name="Retest current version"
    )
    assert clone["session"]["source_session_id"] == original["session"]["session_id"]
    assert clone["turns"][0]["user_message"] == original["turns"][0]["user_message"]
    assert clone["turns"][0]["request_id"] != original["turns"][0]["request_id"]
    assert clone["turns"][0]["feedback"] == []
    assert service.worker.observe_calls[-1][0] == clone["session"]["session_id"]
```

- [ ] **Step 4: Implement clone with newly observed snapshots**

`clone_session(source_id, through_turn=None, name=None)` creates a new session using the source profile and current runtime identity, records `source_session_id`, then calls the normal `send_message()` path for each completed source message up to the requested turn with new UUID request IDs. It never copies old response/state/trace/products or feedback. If a replayed new turn fails, return the partial clone with a safe error on its failed turn.

- [ ] **Step 5: Run all service/repository tests and commit**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_repository.py tests/unit/test_debug_service.py -v
git add src/compasscart_debug/repository.py src/compasscart_debug/service.py tests/unit/test_debug_repository.py tests/unit/test_debug_service.py
git commit -m "feat: share and annotate debug sessions"
```

Expected: PASS; imported and cloned IDs are unique and old observations remain unchanged.

---

### Task 6: Build the Secure WSGI Shell and Health Endpoints

**Files:**
- Create: `src/compasscart_debug/http.py`
- Create: `src/compasscart_debug/wsgi.py`
- Create: `tools/run_debug_server.py`
- Create: `tests/unit/test_debug_http.py`

- [ ] **Step 1: Write failing liveness/readiness tests**

```python
def test_liveness_is_minimal_and_never_touches_agent(wsgi_client, fake_service):
    response = wsgi_client.get("/api/health/live")
    assert response.status_code == 200
    assert response.json == {"status": "live"}
    assert fake_service.calls == []


@pytest.mark.parametrize("state,status", [
    ("initializing", 503), ("setup_required", 503),
    ("fatal", 503), ("ready", 200),
])
def test_readiness_status_codes(wsgi_factory, state, status):
    response = wsgi_factory(worker_state=state).get("/api/health/ready")
    assert response.status_code == status
    assert response.json == {"status": state}
```

- [ ] **Step 2: Write failing auth, body-limit, and security-header tests**

```python
def test_protected_api_requires_constant_time_bearer(wsgi_client):
    assert wsgi_client.get("/api/sessions").status_code == 401
    assert wsgi_client.get(
        "/api/sessions", headers={"Authorization": f"Bearer {VALID_TOKEN}"}
    ).status_code == 200


def test_every_response_has_restrictive_security_headers(wsgi_client):
    response = wsgi_client.get("/")
    assert response.headers["Content-Security-Policy"] == (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'none'; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    )
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"
    assert "Access-Control-Allow-Origin" not in response.headers


def test_json_body_limit_is_checked_before_reading(wsgi_factory):
    response = wsgi_factory(max_body=16).post(
        "/api/import", body=b"x" * 17,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {VALID_TOKEN}"},
    )
    assert response.status_code == 413
    assert response.json["error"]["code"] == "body_too_large"
```

- [ ] **Step 3: Run HTTP tests and verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_http.py -v
```

Expected: collection fails because the WSGI modules are absent.

- [ ] **Step 4: Implement the WSGI response, auth, and literal static map**

Use a fixed route map; never resolve arbitrary request paths against the filesystem:

```python
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/static/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/static/js/api.js": ("js/api.js", "text/javascript; charset=utf-8"),
    "/static/js/dom.js": ("js/dom.js", "text/javascript; charset=utf-8"),
    "/static/js/store.js": ("js/store.js", "text/javascript; charset=utf-8"),
    "/static/js/conversation.js": ("js/conversation.js", "text/javascript; charset=utf-8"),
    "/static/js/recommendations.js": ("js/recommendations.js", "text/javascript; charset=utf-8"),
    "/static/js/inspector.js": ("js/inspector.js", "text/javascript; charset=utf-8"),
    "/static/js/dialogs.js": ("js/dialogs.js", "text/javascript; charset=utf-8"),
    "/static/js/app.js": ("js/app.js", "text/javascript; charset=utf-8"),
}
```

`authenticate(environ, expected)` extracts exactly one `Bearer ` prefix and calls `hmac.compare_digest(provided, expected)`. Mutation routes require `Content-Type: application/json`. JSON errors always use the stable envelope and UTF-8; they never contain raw exception messages. All responses get the headers asserted above.

- [ ] **Step 5: Implement async Agent startup wiring and local threaded server**

```python
def create_application(config: DebugConfig, *, agent_factory=None):
    repository = DebugRepository(config.database_path)
    repository.initialize()
    if agent_factory is None:
        from agent import Agent
        runtime = config.runtime_config()
        agent_factory = lambda: Agent(config.catalog_path, config=runtime)
    worker = AgentWorker(
        agent_factory,
        queue_size=config.command_queue_size,
        timeout=config.command_timeout_seconds,
    )
    service = DebugService(repository, worker, build_runtime_identity(config))
    worker.start()
    return DebugApplication(config, service, worker)
```

The function returns immediately after starting the worker, so `/api/health/live` works during Agent cold start. `tools/run_debug_server.py` combines `ThreadingMixIn` and `WSGIServer`, binds `config.host/config.port`, logs the local URL without printing the token, and closes the worker on `KeyboardInterrupt`.

- [ ] **Step 6: Run HTTP tests and commit**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_http.py -v
git add src/compasscart_debug/http.py src/compasscart_debug/wsgi.py tools/run_debug_server.py tests/unit/test_debug_http.py
git commit -m "feat: expose secure debug service shell"
```

Expected: PASS; liveness remains 200 while the fake Agent factory is blocked.

---

## Scope and File Map

Do not modify `agent.py`, `src/compasscart/*.py`, evaluator code, or the official `requirements.txt` while implementing this plan.

### Debug backend

- `src/compasscart_debug/__init__.py`: package marker and debug schema version.
- `src/compasscart_debug/errors.py`: stable service errors and HTTP-safe error codes.
- `src/compasscart_debug/config.py`: environment parsing, token validation, runtime paths, and behavioral identity fingerprints.
- `src/compasscart_debug/snapshots.py`: JSON-safe response, product, state, and exact-last-trace snapshots.
- `src/compasscart_debug/repository.py`: schema migration and short-lived SQLite connections for sessions, turns, and feedback.
- `src/compasscart_debug/agent_adapter.py`: synchronous read-only adapter around one current Agent instance.
- `src/compasscart_debug/agent_worker.py`: bounded command queue and the sole Agent-owning thread.
- `src/compasscart_debug/service.py`: turn reservation, idempotency, recovery, replay, clone, import, and export orchestration.
- `src/compasscart_debug/http.py`: WSGI routing, auth, request limits, response headers, and a literal static-file map.
- `src/compasscart_debug/wsgi.py`: app factory and asynchronous Agent startup wiring.

### Static UI

- `src/compasscart_debug/static/index.html`: semantic login and three-pane application shell.
- `src/compasscart_debug/static/styles.css`: design tokens, desktop columns, inspector drawer, mobile tabs, and states.
- `src/compasscart_debug/static/js/api.js`: Bearer API client, JSON import, and Blob export.
- `src/compasscart_debug/static/js/dom.js`: safe text-only DOM helpers and formatters.
- `src/compasscart_debug/static/js/store.js`: small observable UI state container.
- `src/compasscart_debug/static/js/conversation.js`: transcript, turn selection, composer, and retry behavior.
- `src/compasscart_debug/static/js/recommendations.js`: ranked products and feedback editor.
- `src/compasscart_debug/static/js/inspector.js`: trace, ledger, raw snapshots, and unavailable-score state.
- `src/compasscart_debug/static/js/dialogs.js`: login, new session, import, archive, and clone dialogs with focus management.
- `src/compasscart_debug/static/js/app.js`: bootstrap and cross-module event coordination.

### Runtime and deployment

- `src/compasscart_debug/runtime_bundle.py`: deterministic private runtime package, safe installation, activation, and SQLite backup helpers.
- `tools/run_debug_server.py`: local threaded WSGI entry point.
- `tools/package_debug_runtime.py`: private catalog/assets bundle CLI.
- `tools/install_debug_runtime.py`: checksum-verified persistent runtime installer CLI.
- `tools/backup_debug_data.py`: online-safe SQLite backup CLI.
- `requirements-debug.txt`: official runtime requirements plus Gunicorn only.
- `requirements-debug-dev.txt`: browser-test dependencies only.
- `deploy/gunicorn.conf.py`: one process, four HTTP threads, one in-app Agent worker.
- `Dockerfile.debug`, `.dockerignore`, `compose.debug.yaml`, `.env.debug.example`, `render.yaml`: portable and fixed-host deployment.
- `docs/debug-console-deployment.md`: local, Docker, Render, backup, restore, and host migration guide.

### Tests

- `tests/unit/test_debug_config.py`
- `tests/unit/test_debug_snapshots.py`
- `tests/unit/test_debug_repository.py`
- `tests/unit/test_debug_worker.py`
- `tests/unit/test_debug_service.py`
- `tests/unit/test_debug_http.py`
- `tests/unit/test_debug_runtime_bundle.py`
- `tests/unit/test_debug_backup.py`
- `tests/integration/test_debug_console.py`
- `tests/browser/test_debug_console_ui.py`
- `tests/js/debug_store.test.mjs`
- `tests/deployment/test_debug_deployment.py`
- Modify `tests/contract/test_submission_package.py`

## Stable API Shapes

All implementations and tests use these exact resource envelopes:

```json
{
  "error": {
    "code": "turn_limit",
    "message": "This session has reached 10 completed turns.",
    "retryable": false,
    "field_errors": {"user_message": "required"}
  },
  "request_id": "client-generated-id"
}
```

```json
{
  "session_id": "uuid",
  "name": "Blue running shoes",
  "archived": false,
  "created_at": "2026-08-25T14:00:00Z",
  "updated_at": "2026-08-25T14:00:01Z",
  "completed_turn_count": 1,
  "max_turns": 10,
  "agent_version": "git-or-build-version",
  "source_session_id": null,
  "continuation": {"status": "ready", "can_send": true}
}
```

```json
{
  "turn": 1,
  "request_id": "client-generated-id",
  "status": "completed",
  "user_message": "I need blue running shoes under $80",
  "response": {
    "message": "Here are the closest matches I found.",
    "ask_attribute": null,
    "recommendations": [{"parent_asin": "SHOE1"}],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0}
  },
  "products": [],
  "state": {},
  "trace": {},
  "feedback": [],
  "error": null,
  "created_at": "2026-08-25T14:00:01Z"
}
```

---

### Task 1: Add Debug Configuration, Errors, and JSON-Safe Snapshots

**Files:**
- Create: `src/compasscart_debug/__init__.py`
- Create: `src/compasscart_debug/errors.py`
- Create: `src/compasscart_debug/config.py`
- Create: `src/compasscart_debug/snapshots.py`
- Create: `tests/unit/test_debug_config.py`
- Create: `tests/unit/test_debug_snapshots.py`

- [ ] **Step 1: Write failing configuration and token tests**

```python
def test_debug_config_requires_generated_token(monkeypatch, tmp_path):
    monkeypatch.setenv("COMPASSCART_CATALOG_PATH", str(tmp_path / "catalog.jsonl"))
    monkeypatch.setenv("COMPASSCART_DEBUG_DATABASE", str(tmp_path / "debug.sqlite3"))
    monkeypatch.setenv("COMPASSCART_DEBUG_TOKEN", "change-me")
    with pytest.raises(ValueError, match="43 characters"):
        DebugConfig.from_env()


def test_runtime_identity_ignores_host_path_spelling(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text('{"parent_asin":"A"}\n', encoding="utf-8")
    first = RuntimeIdentity.build(
        agent_version="abc123",
        catalog_path=catalog,
        runtime_config=RuntimeConfig(dense_model_dir=Path("C:/first/model")),
        asset_manifest_path=None,
        dense_disabled=True,
    )
    second = RuntimeIdentity.build(
        agent_version="abc123",
        catalog_path=catalog,
        runtime_config=RuntimeConfig(dense_model_dir=Path("/second/model")),
        asset_manifest_path=None,
        dense_disabled=True,
    )
    assert first.config_sha256 == second.config_sha256
    assert first.catalog_sha256 == second.catalog_sha256
```

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_config.py -v
```

Expected: collection fails because `compasscart_debug.config` does not exist.

- [ ] **Step 3: Implement configuration, stable errors, and behavioral identity**

Use these exact public types and validation rules:

```python
@dataclass(frozen=True)
class DebugConfig:
    catalog_path: Path
    database_path: Path
    static_root: Path
    host: str
    port: int
    access_token: str
    asset_root: Path
    max_body_bytes: int = 1_048_576
    max_import_bytes: int = 16_777_216
    command_queue_size: int = 32
    command_timeout_seconds: float = 180.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "DebugConfig":
        values = dict(os.environ if env is None else env)
        token = values.get("COMPASSCART_DEBUG_TOKEN", "")
        rejected = {"change-me", "replace-me", "secret", "password"}
        if len(token) < 43 or token.lower() in rejected:
            raise ValueError("COMPASSCART_DEBUG_TOKEN must be at least 43 characters")
        root = Path(__file__).resolve().parent
        return cls(
            catalog_path=Path(values.get("COMPASSCART_CATALOG_PATH", "data/catalog.jsonl")),
            database_path=Path(values.get("COMPASSCART_DEBUG_DATABASE", "var/debug/compasscart-debug.sqlite3")),
            static_root=root / "static",
            host=values.get("HOST", "127.0.0.1"),
            port=int(values.get("PORT", "8765")),
            access_token=token,
            asset_root=Path(values.get("COMPASSCART_ASSET_ROOT", "assets")),
            max_body_bytes=int(values.get("COMPASSCART_DEBUG_MAX_BODY", "1048576")),
            max_import_bytes=int(values.get("COMPASSCART_DEBUG_MAX_IMPORT", "16777216")),
            command_queue_size=int(values.get("COMPASSCART_DEBUG_QUEUE_SIZE", "32")),
            command_timeout_seconds=float(values.get("COMPASSCART_DEBUG_TIMEOUT", "180")),
        )

    def runtime_config(self) -> RuntimeConfig:
        return RuntimeConfig(
            dense_model_dir=self.asset_root / "model",
            dense_vector_dir=self.asset_root / "product_vectors",
            dense_manifest_path=self.asset_root / "SHA256SUMS",
        )


@dataclass(frozen=True)
class RuntimeIdentity:
    agent_version: str
    catalog_sha256: str
    config_sha256: str
    assets_sha256: str | None

    @classmethod
    def build(cls, *, agent_version, catalog_path, runtime_config,
              asset_manifest_path, dense_disabled):
        behavior = dataclasses.asdict(runtime_config)
        behavior.pop("dense_model_dir", None)
        behavior.pop("dense_vector_dir", None)
        behavior.pop("dense_manifest_path", None)
        behavior["dense_disabled"] = bool(dense_disabled)
        return cls(
            agent_version=agent_version,
            catalog_sha256=sha256_file(catalog_path),
            config_sha256=sha256_json(behavior),
            assets_sha256=(
                sha256_file(asset_manifest_path)
                if asset_manifest_path and Path(asset_manifest_path).is_file()
                else None
            ),
        )
```

`errors.py` defines `DebugServiceError(code, message, http_status, retryable=False, field_errors=None)` plus concrete `AuthenticationError`, `ValidationError`, `ConflictError`, `NotReadyError`, `WorkerBusyError`, `ReplayMismatchError`, and `TurnLimitError`. Error messages must not contain exception reprs or filesystem paths.

- [ ] **Step 4: Run the configuration tests and verify GREEN**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Write failing snapshot tests**

```python
def test_snapshot_state_serializes_every_field_and_sorts_sets():
    state = SessionState("s1", turn=2, route="buying", intent_version=2)
    state.no_preference_attributes = {"size", "brand"}
    state.constraints = [
        Constraint("color", "red", 1.0, True, "message", 1, 1, "superseded"),
        Constraint("color", "blue", 1.0, True, "message", 2, 2, "active"),
    ]
    result = snapshot_state(state)
    assert result["no_preference_attributes"] == ["brand", "size"]
    assert [item["status"] for item in result["constraints"]] == [
        "superseded", "active"
    ]
    json.dumps(result, allow_nan=False)


def test_products_follow_response_order_and_never_invent_scores(fake_agent):
    response = {"recommendations": [
        {"parent_asin": "B"}, {"parent_asin": "A"}
    ]}
    products = snapshot_products(fake_agent, response)
    assert [item["parent_asin"] for item in products] == ["B", "A"]
    assert [item["rank"] for item in products] == [1, 2]
    assert all("score" not in item and "source_scores" not in item for item in products)


def test_capture_trace_never_reuses_an_older_record():
    records = [{"session_id": "s1", "turn": 1, "elapsed_ms": 2.0}]
    assert capture_exact_trace(records, "s1", 2) is None
```

- [ ] **Step 6: Run snapshot tests and verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_snapshots.py -v
```

Expected: FAIL because the snapshot functions are missing.

- [ ] **Step 7: Implement recursive JSON normalization and snapshots**

```python
def json_safe(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def snapshot_state(state: SessionState) -> dict[str, object]:
    result = json_safe(state)
    if not isinstance(result, dict):
        raise TypeError("SessionState snapshot must be a mapping")
    return result


def capture_exact_trace(records, session_id: str, turn: int):
    if not records:
        return None
    latest = records[-1]
    if latest.get("session_id") != session_id or latest.get("turn") != turn:
        return None
    result = json_safe(latest)
    return result if isinstance(result, dict) else None
```

`snapshot_response()` requires the four contract keys and normalizes them without adding fields. `snapshot_products()` iterates only `response["recommendations"]`, reads `agent.catalog.product(id)` and `agent.catalog.attributes[id]`, and returns `rank`, `parent_asin`, `title`, `price`, `average_rating`, `rating_number`, `store`, `categories`, `features`, `details`, `attributes`, and `metadata_missing`. Missing catalog rows preserve rank/ASIN and set all metadata to `None`, empty lists/maps, and `metadata_missing=True`.

- [ ] **Step 8: Run both focused files and commit**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_config.py tests/unit/test_debug_snapshots.py -v
git add src/compasscart_debug tests/unit/test_debug_config.py tests/unit/test_debug_snapshots.py
git commit -m "feat: define debug runtime snapshots"
```

Expected: PASS; commit contains no `src/compasscart/` changes.

---

### Task 2: Build the Versioned SQLite Debug Repository

**Files:**
- Create: `src/compasscart_debug/repository.py`
- Create: `tests/unit/test_debug_repository.py`

- [ ] **Step 1: Write failing schema and reopen tests**

```python
def test_repository_initializes_and_reopens_schema(tmp_path):
    path = tmp_path / "debug.sqlite3"
    DebugRepository(path).initialize()
    reopened = DebugRepository(path)
    reopened.initialize()
    assert reopened.schema_version() == 1


def test_repository_rejects_newer_schema(tmp_path):
    path = tmp_path / "debug.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO metadata VALUES ('schema_version', '99')")
    with pytest.raises(RepositoryVersionError, match="newer"):
        DebugRepository(path).initialize()
```

- [ ] **Step 2: Run the schema tests and verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_repository.py -v
```

Expected: collection fails because `DebugRepository` is missing.

- [ ] **Step 3: Implement connection policy and schema v1**

Every public repository method opens and closes its own connection:

```python
@contextmanager
def _connection(self):
    self.path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(self.path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    try:
        yield connection
    finally:
        connection.close()
```

`initialize()` enables WAL and creates these exact constraints:

```sql
CREATE TABLE sessions (
  session_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  profile_json TEXT NOT NULL,
  agent_version TEXT NOT NULL,
  catalog_sha256 TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  assets_sha256 TEXT,
  source_session_id TEXT,
  archived INTEGER NOT NULL DEFAULT 0 CHECK (archived IN (0,1)),
  dirty INTEGER NOT NULL DEFAULT 0 CHECK (dirty IN (0,1)),
  read_only_reason TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (source_session_id) REFERENCES sessions(session_id)
);
CREATE TABLE turns (
  session_id TEXT NOT NULL,
  turn INTEGER NOT NULL CHECK (turn BETWEEN 1 AND 10),
  request_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending','completed','failed')),
  user_message TEXT NOT NULL,
  response_json TEXT,
  products_json TEXT,
  state_json TEXT,
  trace_json TEXT,
  error_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (session_id, turn),
  UNIQUE (session_id, request_id),
  FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
CREATE TABLE product_feedback (
  session_id TEXT NOT NULL,
  turn INTEGER NOT NULL,
  parent_asin TEXT NOT NULL,
  reason TEXT NOT NULL,
  note TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (session_id, turn, parent_asin),
  FOREIGN KEY (session_id, turn) REFERENCES turns(session_id, turn) ON DELETE CASCADE
);
```

- [ ] **Step 4: Write failing turn state-machine tests**

```python
def test_reserve_turn_is_idempotent_and_blocks_different_message(repository, session):
    first = repository.reserve_turn(session.session_id, "req-1", "blue shoes")
    same = repository.reserve_turn(session.session_id, "req-1", "blue shoes")
    assert first.turn == same.turn == 1
    assert same.status == "pending"
    with pytest.raises(RequestConflictError):
        repository.reserve_turn(session.session_id, "req-1", "red shoes")


def test_failed_turn_cannot_be_skipped(repository, session):
    pending = repository.reserve_turn(session.session_id, "req-1", "blue shoes")
    repository.fail_turn(session.session_id, pending.turn, {"code": "agent_error"})
    with pytest.raises(UnresolvedTurnError):
        repository.reserve_turn(session.session_id, "req-2", "next message")
    retried = repository.retry_failed(session.session_id, "req-1", "blue shoes")
    assert retried.turn == 1
    assert retried.status == "pending"


def test_repository_rejects_eleventh_turn_before_agent_call(repository, session):
    for number in range(1, 11):
        row = repository.reserve_turn(session.session_id, f"req-{number}", str(number))
        repository.complete_turn(session.session_id, row.turn, observation(number))
    with pytest.raises(TurnLimitError):
        repository.reserve_turn(session.session_id, "req-11", "eleven")
```

- [ ] **Step 5: Implement transactional CRUD and state transitions**

`reserve_turn()` uses `BEGIN IMMEDIATE`, returns an existing matching request unchanged, rejects the same request ID with different text, rejects any other pending/failed row, calculates `turn = COALESCE(MAX(turn), 0) + 1`, and rejects values above 10. `retry_failed()` accepts only the same request ID and original message and performs `failed -> pending`. `complete_turn()` accepts only pending and writes response/products/state/trace in one transaction. `fail_turn()` accepts only pending and writes a safe error envelope. No update method can alter a completed observation.

Expose these typed methods with JSON decoded on output:

```python
create_session(record: SessionRecord) -> SessionRecord
list_sessions(scope: Literal["active", "archived", "all"]) -> list[SessionRecord]
get_session(session_id: str) -> SessionRecord
patch_session(session_id: str, *, name: str | None, archived: bool | None) -> SessionRecord
mark_dirty(session_id: str, dirty: bool) -> None
mark_read_only(session_id: str, reason: str | None) -> None
reserve_turn(session_id: str, request_id: str, message: str) -> TurnRecord
retry_failed(session_id: str, request_id: str, message: str) -> TurnRecord
complete_turn(session_id: str, turn: int, observation: TurnObservation) -> TurnRecord
fail_turn(session_id: str, turn: int, error: Mapping[str, object]) -> TurnRecord
list_turns(session_id: str) -> list[TurnRecord]
get_turn_by_request(session_id: str, request_id: str) -> TurnRecord | None
completed_turns(session_id: str) -> list[TurnRecord]
upsert_feedback(session_id: str, turn: int, parent_asin: str,
                reason: str, note: str) -> FeedbackRecord
clear_feedback(session_id: str, turn: int, parent_asin: str) -> None
```

- [ ] **Step 6: Add persistence, feedback, and rollback tests**

Test that feedback survives a new repository instance, only attaches to a completed product ASIN from that turn, can be updated and cleared, and is returned separately from the immutable product snapshot. Monkeypatch the final `UPDATE turns` to raise `sqlite3.OperationalError` and assert the row remains pending rather than partially completed.

- [ ] **Step 7: Run repository tests and commit**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_repository.py -v
git add src/compasscart_debug/repository.py tests/unit/test_debug_repository.py
git commit -m "feat: persist debug sessions and turns"
```

Expected: PASS with no skipped repository tests.

---

### Task 3: Serialize All Agent Work Through One Dedicated Thread

**Files:**
- Create: `src/compasscart_debug/agent_adapter.py`
- Create: `src/compasscart_debug/agent_worker.py`
- Create: `tests/unit/test_debug_worker.py`

- [ ] **Step 1: Write failing thread-affinity and reset tests**

```python
def test_agent_is_constructed_and_used_only_on_worker_thread():
    seen: list[int] = []

    class FakeAgent:
        def __init__(self):
            seen.append(threading.get_ident())
            self.traces = SimpleNamespace(records=[])
            self.sessions = FakeSessions()
            self.catalog = FakeCatalog()

        def reset(self, session_id, profile):
            seen.append(threading.get_ident())

        def respond(self, session_id, message, turn, top_k):
            seen.append(threading.get_ident())
            return valid_response("A")

    worker = AgentWorker(lambda: FakeAgent(), queue_size=2)
    worker.start()
    worker.reset_session("s1", {})
    worker.observe_turn("s1", "first", 1)
    worker.observe_turn("s1", "second", 2)
    worker.close()
    assert len(set(seen)) == 1
    assert seen[0] != threading.get_ident()


def test_normal_second_turn_does_not_reset_agent(fake_worker):
    fake_worker.reset_session("s1", {})
    fake_worker.observe_turn("s1", "one", 1)
    fake_worker.observe_turn("s1", "two", 2)
    assert fake_worker.fake_agent.reset_calls == [("s1", {})]
```

- [ ] **Step 2: Run worker tests and verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_worker.py -v
```

Expected: FAIL because worker and adapter modules do not exist.

- [ ] **Step 3: Implement `AgentAdapter` without internal algorithm calls**

```python
@dataclass(frozen=True)
class TurnObservation:
    response: dict[str, object]
    products: list[dict[str, object]]
    state: dict[str, object]
    trace: dict[str, object] | None


class AgentAdapter:
    def __init__(self, agent: object) -> None:
        self.agent = agent

    def reset_session(self, session_id: str, profile: dict[str, object]) -> None:
        self.agent.reset(session_id, profile)

    def has_session(self, session_id: str) -> bool:
        return self.agent.sessions.get(session_id) is not None

    def observe_turn(self, session_id: str, message: str, turn: int) -> TurnObservation:
        response = self.agent.respond(session_id, message, turn, 10)
        state = self.agent.sessions.get(session_id)
        if state is None:
            raise RuntimeError("Agent session disappeared after respond")
        records = self.agent.traces.records
        return TurnObservation(
            response=snapshot_response(response),
            products=snapshot_products(self.agent, response),
            state=snapshot_state(state),
            trace=capture_exact_trace(records, session_id, turn),
        )
```

Do not import `compasscart.parser`, `router`, `retrieval`, `ranker`, or `question_policy` in this package.

- [ ] **Step 4: Implement the bounded worker and observable startup states**

```python
class AgentWorker:
    def __init__(self, factory, *, queue_size: int = 32, timeout: float = 180.0):
        self._factory = factory
        self._queue = queue.Queue(maxsize=queue_size)
        self._timeout = timeout
        self._state = "stopped"
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="compasscart-agent", daemon=True)

    def start(self) -> None:
        with self._state_lock:
            if self._state != "stopped":
                return
            self._state = "initializing"
        self._thread.start()

    def _run(self) -> None:
        try:
            adapter = AgentAdapter(self._factory())
        except FileNotFoundError:
            self._set_state("setup_required")
            return
        except Exception:
            self._set_state("fatal")
            return
        self._adapter = adapter
        self._set_state("ready")
        while True:
            command = self._queue.get()
            if command is _STOP:
                return
            try:
                command.future.set_result(command.call(adapter))
            except BaseException as error:
                command.future.set_exception(error)

    def _call(self, operation):
        if self.state != "ready":
            raise NotReadyError(self.state)
        future = concurrent.futures.Future()
        try:
            self._queue.put_nowait(_Command(operation, future))
        except queue.Full as error:
            raise WorkerBusyError() from error
        return future.result(timeout=self._timeout)
```

Public methods are `reset_session`, `has_session`, `observe_turn`, `rehydrate`, `invalidate_session`, `state`, `start`, and `close`. `rehydrate(session_id, profile, completed_turns)` performs one reset and ordered observation calls, returns fresh observations for comparison, and never persists replay traces itself.

- [ ] **Step 5: Add serialization, queue-full, setup-required, and missing-trace tests**

Use a blocking fake command to fill a queue of size one and assert the next caller gets `WorkerBusyError`. Make the factory raise `FileNotFoundError` and assert `worker.state == "setup_required"` while the caller thread remains responsive. Verify a mismatched trace returns `None`. Verify concurrent callers all execute one at a time on the same worker thread.

- [ ] **Step 6: Run worker and snapshot tests, then commit**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_worker.py tests/unit/test_debug_snapshots.py -v
git add src/compasscart_debug/agent_adapter.py src/compasscart_debug/agent_worker.py tests/unit/test_debug_worker.py
git commit -m "feat: serialize debug agent observations"
```

Expected: PASS; the tests prove Agent construction occurs inside the worker thread.

---
