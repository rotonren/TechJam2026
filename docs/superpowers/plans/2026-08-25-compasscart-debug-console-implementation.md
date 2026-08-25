# CompassCart Persistent Debug Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable, persistent three-pane debugging console that calls the current CompassCart Agent unchanged, records reproducible multi-turn observations and human feedback, and can run locally or as a single-instance Docker/Render service.

**Architecture:** A WSGI process serves static UI/API traffic and uses short-lived SQLite repository connections. Exactly one dedicated worker thread constructs and owns the existing `agent.Agent`; every reset, response, catalog lookup, state read, and trace read is serialized through that worker so the catalog SQLite connection never crosses threads. Debug code and deployment dependencies stay outside the official submission allowlist.

**Tech Stack:** Python 3.12, standard-library WSGI/SQLite/threading/zipfile, existing CompassCart Agent, native HTML/CSS/ES modules, Gunicorn gthread, Docker Compose, Render Blueprint, pytest, Node built-in tests, Browser/IAB verification.

---

## File Map and Fixed Boundaries

Do not modify `agent.py`, any file under `src/compasscart/`, evaluator code, `requirements.txt`, or `tools/package_submission.py`.

- `src/compasscart_debug/__init__.py`: debug package/schema version.
- `src/compasscart_debug/errors.py`: stable service errors and safe HTTP codes.
- `src/compasscart_debug/config.py`: environment parsing, runtime paths, token validation, fingerprints.
- `src/compasscart_debug/snapshots.py`: JSON-safe response/product/state/exact-trace snapshots.
- `src/compasscart_debug/repository.py`: schema and per-operation SQLite connections.
- `src/compasscart_debug/agent_adapter.py`: calls only current Agent `reset/respond`, then reads catalog/state/trace.
- `src/compasscart_debug/agent_worker.py`: bounded queue and sole Agent-owning thread.
- `src/compasscart_debug/service.py`: idempotency, recovery, replay, clone, feedback, import/export.
- `src/compasscart_debug/http.py`: WSGI routing, auth, limits, headers, literal static map.
- `src/compasscart_debug/wsgi.py`: app factory and asynchronous Agent startup.
- `src/compasscart_debug/static/index.html`, `styles.css`, `js/*.js`, `js/package.json`: approved A layout and an isolated ES-module boundary for Node tests.
- `src/compasscart_debug/runtime_bundle.py`: deterministic private runtime package/install/backup logic.
- `tools/run_debug_server.py`, `package_debug_runtime.py`, `install_debug_runtime.py`, `backup_debug_data.py`: CLIs.
- `requirements-debug.txt`, `requirements-debug-dev.txt`, `Dockerfile.debug`, `.dockerignore`, `compose.debug.yaml`, `.env.debug.example`, `render.yaml`, `deploy/gunicorn.conf.py`: isolated deployment surface.
- `docs/debug-console-deployment.md`: operation, migration, backup, Render setup.

Stable API rules:

- Errors: `{"error":{"code":str,"message":str,"retryable":bool,"field_errors"?:object},"request_id"?:str}`.
- A completed turn contains `turn`, `request_id`, `status`, `user_message`, `response`, `products`, `state`, `trace`, `feedback`, `error`, and `created_at`.
- Products contain `rank`, `parent_asin`, metadata fields, and `metadata_missing`; they never contain invented `score` or `source_scores`.
- Session continuation is one of `ready`, `rehydrating`, `incompatible`, `blocked_failed`, or `turn_limit` plus `can_send`.
- Message requests are `{request_id, user_message}`. Same ID/same text is idempotent; same ID/different text is 409.
- Feedback reasons are `explicit_constraint`, `wrong_category`, `over_budget`, `attribute_mismatch`, `duplicate_or_too_similar`, or `other`.

---

### Task 1: Configuration, Errors, Identity, and JSON-Safe Snapshots

**Files:**
- Create: `src/compasscart_debug/__init__.py`
- Create: `src/compasscart_debug/errors.py`
- Create: `src/compasscart_debug/config.py`
- Create: `src/compasscart_debug/snapshots.py`
- Test: `tests/unit/test_debug_config.py`
- Test: `tests/unit/test_debug_snapshots.py`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_debug_config_requires_generated_token(tmp_path):
    env = {
        "COMPASSCART_CATALOG_PATH": str(tmp_path / "catalog.jsonl"),
        "COMPASSCART_DEBUG_DATABASE": str(tmp_path / "debug.sqlite3"),
        "COMPASSCART_DEBUG_TOKEN": "change-me",
    }
    with pytest.raises(ValueError, match="43 characters"):
        DebugConfig.from_env(env)


def test_identity_ignores_absolute_dense_paths(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text('{"parent_asin":"A"}\n', encoding="utf-8")
    one = RuntimeIdentity.build(
        "v1", catalog, RuntimeConfig(dense_model_dir=Path("C:/one")), None, True
    )
    two = RuntimeIdentity.build(
        "v1", catalog, RuntimeConfig(dense_model_dir=Path("/two")), None, True
    )
    assert one.catalog_sha256 == two.catalog_sha256
    assert one.config_sha256 == two.config_sha256
```

- [ ] **Step 2: Verify RED**

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_config.py -v
```

Expected: import failure for `compasscart_debug.config`.

- [ ] **Step 3: Implement configuration and safe errors**

```python
@dataclass(frozen=True)
class DebugConfig:
    catalog_path: Path
    database_path: Path
    static_root: Path
    asset_root: Path
    runtime_root: Path | None
    host: str
    port: int
    access_token: str
    max_body_bytes: int = 1_048_576
    max_import_bytes: int = 16_777_216
    command_queue_size: int = 32
    command_timeout_seconds: float = 180.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "DebugConfig":
        values = dict(os.environ if env is None else env)
        token = values.get("COMPASSCART_DEBUG_TOKEN", "")
        if len(token) < 43 or token.lower() in {"change-me", "replace-me", "password"}:
            raise ValueError("COMPASSCART_DEBUG_TOKEN must be at least 43 characters")
        package_root = Path(__file__).resolve().parent
        return cls(
            catalog_path=Path(values.get("COMPASSCART_CATALOG_PATH", "data/catalog.jsonl")),
            database_path=Path(values.get("COMPASSCART_DEBUG_DATABASE", "var/debug/compasscart-debug.sqlite3")),
            static_root=package_root / "static",
            asset_root=Path(values.get("COMPASSCART_ASSET_ROOT", "assets")),
            runtime_root=(
                Path(values["COMPASSCART_RUNTIME_ROOT"])
                if values.get("COMPASSCART_RUNTIME_ROOT")
                else None
            ),
            host=values.get("HOST", "127.0.0.1"),
            port=int(values.get("PORT", "8765")),
            access_token=token,
        )

    def resolve_runtime_paths(self) -> RuntimePaths:
        if self.runtime_root is None:
            return RuntimePaths(self.catalog_path, self.asset_root)
        runtime_id = (self.runtime_root / "CURRENT").read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", runtime_id):
            raise FileNotFoundError("installed debug runtime is unavailable")
        release = (self.runtime_root / "releases" / runtime_id).resolve()
        releases = (self.runtime_root / "releases").resolve()
        if releases not in release.parents:
            raise FileNotFoundError("installed debug runtime is unavailable")
        return RuntimePaths(release / "catalog.jsonl", release / "assets")

    def runtime_config(self, paths: RuntimePaths) -> RuntimeConfig:
        return RuntimeConfig(
            dense_model_dir=paths.asset_root / "model",
            dense_vector_dir=paths.asset_root / "product_vectors",
            dense_manifest_path=paths.asset_root / "SHA256SUMS",
        )
```

Add frozen `RuntimePaths(catalog_path, asset_root)` and `RuntimeIdentity(agent_version, catalog_sha256, config_sha256, assets_sha256)`. `resolve_runtime_paths()` is called only by the Agent factory inside the dedicated worker thread; a missing/invalid `CURRENT` pointer therefore becomes `setup_required` instead of preventing the WSGI process from starting. Its config fingerprint canonicalizes behavioral `RuntimeConfig` fields, removes absolute dense paths, and adds the effective dense-disabled flag. `errors.py` defines `DebugServiceError(code, message, http_status, retryable=False, field_errors=None)`, an internal `RuntimeSetupError`, and concrete authentication, validation, conflict, not-ready, busy, replay-mismatch, and turn-limit errors; no safe message includes exception reprs or paths.

- [ ] **Step 4: Write failing snapshot tests**

```python
def test_state_snapshot_preserves_constraints_and_sorts_sets():
    state = SessionState("s", turn=2, route="buying", intent_version=2)
    state.no_preference_attributes = {"size", "brand"}
    state.constraints = [
        Constraint("color", "red", 1.0, True, "message", 1, 1, "superseded"),
        Constraint("color", "blue", 1.0, True, "message", 2, 2, "active"),
    ]
    result = snapshot_state(state)
    assert result["no_preference_attributes"] == ["brand", "size"]
    assert [row["status"] for row in result["constraints"]] == ["superseded", "active"]
    json.dumps(result, allow_nan=False)


def test_product_order_comes_from_response_and_has_no_fake_score(fake_agent):
    products = snapshot_products(fake_agent, {
        "recommendations": [{"parent_asin": "B"}, {"parent_asin": "A"}]
    })
    assert [(row["rank"], row["parent_asin"]) for row in products] == [(1, "B"), (2, "A")]
    assert all("score" not in row and "source_scores" not in row for row in products)


def test_mismatched_latest_trace_is_unavailable():
    records = [{"session_id": "s", "turn": 1}]
    assert capture_exact_trace(records, "s", 2) is None
```

- [ ] **Step 5: Implement JSON normalization and snapshot functions**

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


def capture_exact_trace(records, session_id: str, turn: int):
    if not records:
        return None
    latest = records[-1]
    if latest.get("session_id") != session_id or latest.get("turn") != turn:
        return None
    return json_safe(latest)
```

`snapshot_response()` validates and preserves the four Agent contract keys: `message`, `ask_attribute`, `recommendations`, and `usage`. `snapshot_state()` includes every `SessionState` field. `snapshot_products()` iterates only response recommendations and returns rank, ASIN, title, price, rating, rating count, store, categories, features, details, normalized attributes, and `metadata_missing`; missing rows retain rank/ASIN and explicit null/empty metadata.

- [ ] **Step 6: Verify GREEN and commit**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_config.py tests/unit/test_debug_snapshots.py -v
git add src/compasscart_debug tests/unit/test_debug_config.py tests/unit/test_debug_snapshots.py
git commit -m "feat: define debug runtime snapshots"
```

Expected: PASS and no changes under `src/compasscart/`.

---

### Task 2: Versioned SQLite Repository and Turn State Machine

**Files:**
- Create: `src/compasscart_debug/repository.py`
- Test: `tests/unit/test_debug_repository.py`

- [ ] **Step 1: Write failing schema/state tests**

```python
def test_repository_initializes_and_reopens(tmp_path):
    path = tmp_path / "debug.sqlite3"
    DebugRepository(path).initialize()
    reopened = DebugRepository(path)
    reopened.initialize()
    assert reopened.schema_version() == 1


def test_failed_turn_cannot_be_skipped(repository, session_record):
    turn = repository.reserve_turn(session_record.session_id, "r1", "blue")
    repository.fail_turn(session_record.session_id, turn.turn, {"code": "agent_error"})
    with pytest.raises(UnresolvedTurnError):
        repository.reserve_turn(session_record.session_id, "r2", "next")
    retried = repository.retry_failed(session_record.session_id, "r1", "blue")
    assert (retried.turn, retried.status) == (1, "pending")


def test_same_request_different_text_conflicts(repository, session_record):
    repository.reserve_turn(session_record.session_id, "r1", "blue")
    with pytest.raises(RequestConflictError):
        repository.reserve_turn(session_record.session_id, "r1", "red")
```

- [ ] **Step 2: Verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_repository.py -v
```

Expected: import failure for `DebugRepository`.

- [ ] **Step 3: Implement per-operation connections and schema v1**

Every method opens a fresh `sqlite3.connect(path, timeout=5)`, sets row factory, `foreign_keys=ON`, and `busy_timeout=5000`; initialization enables WAL. Schema:

```sql
CREATE TABLE sessions (
 session_id TEXT PRIMARY KEY, name TEXT NOT NULL, profile_json TEXT NOT NULL,
 agent_version TEXT NOT NULL, catalog_sha256 TEXT NOT NULL,
 config_sha256 TEXT NOT NULL, assets_sha256 TEXT, source_session_id TEXT,
 archived INTEGER NOT NULL DEFAULT 0 CHECK (archived IN (0,1)),
 dirty INTEGER NOT NULL DEFAULT 0 CHECK (dirty IN (0,1)),
 read_only_reason TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 FOREIGN KEY (source_session_id) REFERENCES sessions(session_id)
);
CREATE TABLE turns (
 session_id TEXT NOT NULL, turn INTEGER NOT NULL CHECK (turn BETWEEN 1 AND 10),
 request_id TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('pending','completed','failed')),
 user_message TEXT NOT NULL, response_json TEXT, products_json TEXT,
 state_json TEXT, trace_json TEXT, error_json TEXT,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 PRIMARY KEY (session_id,turn), UNIQUE (session_id,request_id),
 FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
CREATE TABLE product_feedback (
 session_id TEXT NOT NULL, turn INTEGER NOT NULL, parent_asin TEXT NOT NULL,
 reason TEXT NOT NULL, note TEXT NOT NULL, updated_at TEXT NOT NULL,
 PRIMARY KEY (session_id,turn,parent_asin),
 FOREIGN KEY (session_id,turn) REFERENCES turns(session_id,turn) ON DELETE CASCADE
);
```

Store schema version in `metadata`; reject a DB with a version newer than the code.

- [ ] **Step 4: Implement atomic reservation and immutable completion**

`reserve_turn()` uses `BEGIN IMMEDIATE`, returns an existing same-ID/same-text row, rejects an unresolved different request, selects `MAX(turn)+1`, and raises before turn 11. Legal transitions are `pending->completed`, `pending->failed`, and same-request `failed->pending`. Completed response/product/state/trace JSON cannot be edited.

Implement typed CRUD for session create/list/get/patch, dirty/read-only flags, turn lookup/list/completed list, reserve/retry/complete/fail, feedback upsert/clear, and decoded records. `complete_turn()` writes all observation fields in one transaction; an injected update error must leave the row pending.

- [ ] **Step 5: Add feedback/persistence/concurrency tests**

Test reopen persistence, feedback update/clear, rejection of feedback for an ASIN absent from the completed turn, turn-10 enforcement, and rollback on completion failure. For two concurrent different requests against one ready session, assert exactly one `BEGIN IMMEDIATE` reservation creates the next pending turn and the other receives `UnresolvedTurnError`; for two concurrent identical request IDs/messages, assert both callers observe the same single reserved row. No concurrency case may create duplicate turn numbers or skip an unresolved turn. Also test newer-schema rejection.

- [ ] **Step 6: Verify GREEN and commit**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_repository.py -v
git add src/compasscart_debug/repository.py tests/unit/test_debug_repository.py
git commit -m "feat: persist debug sessions and turns"
```

Expected: PASS with no skipped tests.

---

### Task 3: Dedicated Agent Adapter and Worker Thread

**Files:**
- Create: `src/compasscart_debug/agent_adapter.py`
- Create: `src/compasscart_debug/agent_worker.py`
- Test: `tests/unit/test_debug_worker.py`

- [ ] **Step 1: Write failing thread-affinity/reset tests**

```python
def test_agent_factory_and_all_calls_share_worker_thread(fake_agent_factory):
    worker = AgentWorker(fake_agent_factory, queue_size=2)
    worker.start()
    worker.reset_session("s", {})
    worker.observe_turn("s", "one", 1)
    worker.observe_turn("s", "two", 2)
    worker.close()
    assert len(set(fake_agent_factory.thread_ids)) == 1
    assert fake_agent_factory.thread_ids[0] != threading.get_ident()
    assert fake_agent_factory.agent.reset_calls == [("s", {})]


def test_factory_file_missing_becomes_setup_required(fake_agent_factory):
    fake_agent_factory.error = FileNotFoundError("catalog")
    worker = AgentWorker(fake_agent_factory)
    worker.start()
    wait_for_state(worker, "setup_required")
    assert worker.state == "setup_required"


def test_factory_checksum_failure_becomes_setup_required(fake_agent_factory):
    fake_agent_factory.error = RuntimeSetupError("asset checksum mismatch")
    worker = AgentWorker(fake_agent_factory)
    worker.start()
    wait_for_state(worker, "setup_required")
    assert worker.state == "setup_required"
```

- [ ] **Step 2: Verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_worker.py -v
```

Expected: imports fail for adapter/worker.

- [ ] **Step 3: Implement synchronous read-only adapter**

```python
@dataclass(frozen=True)
class TurnObservation:
    response: dict[str, object]
    products: list[dict[str, object]]
    state: dict[str, object]
    trace: dict[str, object] | None


class AgentAdapter:
    def __init__(self, agent, identity):
        self.agent = agent
        self.identity = identity

    def reset_session(self, session_id, profile):
        self.agent.reset(session_id, profile)

    def has_session(self, session_id):
        return self.agent.sessions.get(session_id) is not None

    def observe_turn(self, session_id, message, turn):
        response = self.agent.respond(session_id, message, turn, 10)
        state = self.agent.sessions.get(session_id)
        if state is None:
            raise RuntimeError("Agent session disappeared after respond")
        return TurnObservation(
            snapshot_response(response),
            snapshot_products(self.agent, response),
            snapshot_state(state),
            capture_exact_trace(self.agent.traces.records, session_id, turn),
        )
```

This package must not import parser/router/retriever/ranker/question-policy modules.

- [ ] **Step 4: Implement bounded worker**

The worker factory runs inside `_run()`, never in the constructor/caller, and returns an `AgentAdapter` containing both the Agent and immutable `RuntimeIdentity`. Queue commands carry a callable and `concurrent.futures.Future`. `queue.Full` maps to `WorkerBusyError`; future timeout maps to a retryable timeout. Expected missing/corrupt runtime failures are normalized by the factory to `RuntimeSetupError` and map to `setup_required`; unexpected construction errors map to `fatal`. Startup states are `stopped`, `initializing`, `ready`, `setup_required`, and `fatal`. Public calls are reset, has-session, observe, rehydrate, invalidate, identity, state, start, and close. `identity` is readable only after state becomes ready and exposes no mutable Agent object. Rehydrate performs one reset then ordered completed-message observations; replay traces are returned only for comparison and never persisted.

- [ ] **Step 5: Add serialization/bounded-queue/trace tests**

Use a blocking fake call to fill a size-one queue and prove the next submission fails fast. Prove concurrent callers execute serially on one worker thread. Prove a mismatched latest trace yields `None`, and normal second turn never calls reset. Prove the worker detects an evicted Agent session via `has_session()`.

- [ ] **Step 6: Verify GREEN and commit**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_worker.py tests/unit/test_debug_snapshots.py -v
git add src/compasscart_debug/agent_adapter.py src/compasscart_debug/agent_worker.py tests/unit/test_debug_worker.py
git commit -m "feat: serialize debug agent observations"
```

Expected: PASS; all fake Agent thread IDs are identical.

---

### Task 4: Service Turn Orchestration and Dirty-State Recovery

**Files:**
- Create: `src/compasscart_debug/service.py`
- Test: `tests/unit/test_debug_service.py`

- [ ] **Step 1: Write failing create/idempotency/recovery tests**

```python
def test_same_completed_request_returns_without_second_agent_call(service):
    session = service.create_session("Test", {})["session"]
    first = service.send_message(session["session_id"], "r1", "blue shoes")
    second = service.send_message(session["session_id"], "r1", "blue shoes")
    assert first == second
    assert service.worker.observe_calls == [(session["session_id"], "blue shoes", 1)]


def test_agent_mutation_then_error_rehydrates_before_retry(service):
    sid = service.create_session("Test", {})["session"]["session_id"]
    service.worker.fail_after_mutating_once = True
    with pytest.raises(DebugServiceError) as captured:
        service.send_message(sid, "r1", "blue")
    assert captured.value.code == "agent_error"
    service.send_message(sid, "r1", "blue")
    assert service.worker.events[:3] == ["reset", "failed_observe", "reset"]


def test_duplicate_request_while_inflight_returns_pending(service, blocking_worker):
    sid = service.create_session("Test", {})["session"]["session_id"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.send_message, sid, "r1", "blue")
        blocking_worker.wait_until_observing()
        duplicate = service.send_message(sid, "r1", "blue")
        assert duplicate["turn"]["status"] == "pending"
        assert blocking_worker.observe_calls == [(sid, "blue", 1)]
        blocking_worker.release()
        first.result()
```

- [ ] **Step 2: Verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_service.py -v
```

Expected: import failure for `DebugService`.

- [ ] **Step 3: Implement session create/read/list/patch and message order**

`DebugService` owns repository, worker, UUID/time callables, loaded/dirty sets, and a lock-protected in-flight map keyed by session/request ID; it reads the current immutable identity from the ready worker immediately before creating, cloning, or rehydrating a session. New session order is worker reset, then DB create. Message order is: validate text; return completed duplicate; under the coordination lock return the existing pending row for an identical in-flight request or reject a different in-flight request; register this request; reserve or retry its row; ensure hydrated; worker observe; atomically complete row; remove the in-flight marker in `finally`. A durable pending row with no process-local in-flight marker is crash residue and is rehydrated/retried with the same request. On uncaught Agent error, mark the row failed and session dirty. On final DB write failure, leave the durable row pending when possible, mark memory dirty, and never return the unsaved response as success.

```python
try:
    observation = self.worker.observe_turn(session_id, message, pending.turn)
except Exception as error:
    self.repository.fail_turn(session_id, pending.turn, {
        "code": "agent_error",
        "message": "The Agent could not complete this turn.",
    })
    self._mark_dirty(session_id)
    raise DebugServiceError("agent_error", "The Agent could not complete this turn.", 500, True) from error
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
```

- [ ] **Step 4: Implement canonical rehydration**

Before continuation, compare stored/current agent version, catalog checksum, behavioral config fingerprint, and assets checksum. Rehydrate when the process has not loaded the session, worker session was evicted, session is dirty, or retrying unresolved work. Replay only completed messages. Compare full response and full canonical state (including all constraint fields, asked/pending attributes, query history, no-preference set, prior recommendations, candidate count); exclude trace time. Mismatch sets `read_only_reason="replay_mismatch"` and raises 409.

Every session detail also computes `{continuation, can_send}` from durable state: `rehydrating/false` for an in-flight or crash-pending turn, `incompatible/false` for identity/replay mismatch, `blocked_failed/false` for a failed turn, `turn_limit/false` after ten completed turns, otherwise `ready/true`. Retrying the same failed request is the only mutation allowed while `blocked_failed`.

- [ ] **Step 5: Add DB-completion-failure, eviction, and fingerprint tests**

Assert an Agent response followed by DB failure cannot proceed on advanced memory. Assert retry resets and replays completed turns first. Assert concurrent duplicate requests execute the Agent once, while a pending row found after simulated process restart is retried. Assert an evicted session rehydrates even if present in the service loaded set. Assert config/catalog/version mismatch is read-only and never calls the Agent. Assert all five continuation states and `can_send` values.

- [ ] **Step 6: Verify GREEN and commit**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_service.py tests/unit/test_debug_repository.py tests/unit/test_debug_worker.py -v
git add src/compasscart_debug/service.py tests/unit/test_debug_service.py
git commit -m "feat: recover persistent debug conversations"
```

Expected: PASS, including both dirty-state failure modes.

---

### Task 5: Feedback, Archive, Import/Export, and Current-Version Clone

**Files:**
- Modify: `src/compasscart_debug/repository.py`
- Modify: `src/compasscart_debug/service.py`
- Modify: `tests/unit/test_debug_repository.py`
- Modify: `tests/unit/test_debug_service.py`

- [ ] **Step 1: Write failing feedback/archive tests**

```python
def test_feedback_persists_updates_and_clears(service, completed_session):
    sid = completed_session["session"]["session_id"]
    saved = service.set_feedback(
        sid, 1, "SHOE1", True, "over_budget", "Explicit budget was $80"
    )
    assert saved["reason"] == "over_budget"
    assert service.get_session(sid)["turns"][0]["feedback"][0]["note"].endswith("$80")
    assert service.set_feedback(sid, 1, "SHOE1", False, None, "") is None


def test_archive_scope_can_restore_session(service, completed_session):
    sid = completed_session["session"]["session_id"]
    service.patch_session(sid, archived=True)
    assert sid not in {row["session_id"] for row in service.list_sessions("active")}
    assert sid in {row["session_id"] for row in service.list_sessions("archived")}
    service.patch_session(sid, archived=False)
    assert sid in {row["session_id"] for row in service.list_sessions("active")}
```

- [ ] **Step 2: Write failing import/export/clone tests**

```python
def test_export_import_assigns_new_id_without_secrets(service, completed_session):
    source_id = completed_session["session"]["session_id"]
    payload = service.export_session(source_id)
    assert payload["format"] == "compasscart-debug-session"
    assert payload["schema_version"] == 1
    assert "token" not in json.dumps(payload).lower()
    imported = service.import_session(payload)
    assert imported["session"]["session_id"] != source_id
    assert imported["turns"][0]["response"] == payload["turns"][0]["response"]


def test_clone_replays_messages_and_does_not_copy_feedback(service, completed_session):
    source = completed_session
    clone = service.clone_session(source["session"]["session_id"], through_turn=1)
    assert clone["session"]["source_session_id"] == source["session"]["session_id"]
    assert clone["turns"][0]["request_id"] != source["turns"][0]["request_id"]
    assert clone["turns"][0]["feedback"] == []
```

- [ ] **Step 3: Implement closed feedback and export schemas**

Reject feedback for an ASIN absent from that completed turn. Export exact keys `format`, `schema_version`, `exported_at`, `session`, and `turns`; omit token, paths, hostname, and raw exceptions. Import validates types, ordering, status, at most 10 turns, feedback membership, and assigns a new ID. Invalid format/version/body raises `ValidationError`, not `KeyError`.

- [ ] **Step 4: Implement clone using new Agent observations**

Clone copies only profile/name/source relationship, creates a new ID with current identity, then sends each source completed user message through normal `send_message()` using new request IDs. Never copy old response/products/state/trace/feedback. A clone failure returns the partial new session with its own failed turn.

- [ ] **Step 5: Verify GREEN and commit**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_repository.py tests/unit/test_debug_service.py -v
git add src/compasscart_debug/repository.py src/compasscart_debug/service.py tests/unit/test_debug_repository.py tests/unit/test_debug_service.py
git commit -m "feat: share and annotate debug sessions"
```

Expected: PASS; imports and clones have unique IDs and immutable historical snapshots.

---

### Task 6: Secure WSGI Shell, Static Allowlist, and Health Semantics

**Files:**
- Create: `src/compasscart_debug/http.py`
- Create: `src/compasscart_debug/wsgi.py`
- Create: `tools/run_debug_server.py`
- Test: `tests/unit/test_debug_http.py`

- [ ] **Step 1: Write failing health/auth/header tests**

```python
def test_live_never_touches_agent(wsgi_client, fake_service):
    response = wsgi_client.get("/api/health/live")
    assert (response.status_code, response.json) == (200, {"status": "live"})
    assert fake_service.calls == []


@pytest.mark.parametrize("state,status", [
    ("initializing", 503), ("setup_required", 503), ("fatal", 503), ("ready", 200)
])
def test_ready_status(wsgi_factory, state, status):
    response = wsgi_factory(worker_state=state).get("/api/health/ready")
    assert response.status_code == status
    assert response.json == {"status": state}


def test_ready_requires_database_probe(wsgi_factory):
    response = wsgi_factory(worker_state="ready", repository_ready=False).get(
        "/api/health/ready"
    )
    assert (response.status_code, response.json) == (
        503,
        {"status": "database_unavailable"},
    )


def test_security_headers_and_no_cors(wsgi_client):
    response = wsgi_client.get("/")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"
    assert "Access-Control-Allow-Origin" not in response.headers
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]
```

- [ ] **Step 2: Verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_http.py -v
```

Expected: import failure for WSGI modules.

- [ ] **Step 3: Implement literal static map, JSON limits, auth, and headers**

Map `/`, `/static/styles.css`, and each named JS module to a fixed relative file and content type. Never resolve arbitrary URL paths. Protected API requires one `Authorization: Bearer` value and `hmac.compare_digest`; health/static remain public. Mutations require `application/json`. Reject oversized `Content-Length` before reading. Send UTF-8 stable error envelopes only.

Use this exact CSP:

```text
default-src 'self'; script-src 'self'; style-src 'self'; img-src 'none'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'
```

- [ ] **Step 4: Implement app factory and local threaded server**

`create_application()` initializes only DebugRepository synchronously, constructs `AgentWorker(factory)`, then calls `worker.start()` and immediately returns the WSGI app. The factory itself runs inside the worker: it calls `config.resolve_runtime_paths()`, imports root `Agent`, builds the explicit `RuntimeConfig`, constructs the Agent, computes `RuntimeIdentity`, and returns `AgentAdapter(agent, identity)`. Missing runtime files are caught by the worker as `setup_required`, so liveness and the login shell remain available. Readiness probes both a short repository `SELECT 1` and worker state without touching catalog contents. `wsgi.py` exports a module-level `application = create_application()` for Gunicorn. The local CLI combines `ThreadingMixIn` with WSGI server, prints URL without token, and closes worker on interrupt.

- [ ] **Step 5: Add body/path/startup tests**

Test 413 before body read, wrong content type 415, path traversal 404, all protected routes 401, constant-time helper use, public health minimality, and liveness 200 while a blocking factory leaves readiness 503 initializing.

- [ ] **Step 6: Verify GREEN and commit**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_http.py -v
git add src/compasscart_debug/http.py src/compasscart_debug/wsgi.py tools/run_debug_server.py tests/unit/test_debug_http.py
git commit -m "feat: expose secure debug service shell"
```

Expected: PASS; liveness does not access Agent/catalog.

---

### Task 7: Complete the JSON API and Real Fixture Integration

**Files:**
- Modify: `src/compasscart_debug/http.py`
- Modify: `tests/unit/test_debug_http.py`
- Create: `tests/integration/test_debug_console.py`

- [ ] **Step 1: Write failing route-contract tests**

```python
def test_session_message_feedback_and_export_routes(api_client):
    created = api_client.post("/api/sessions", json={
        "name": "Blue shoes", "profile": {"preference_tags": ["comfort"]}
    })
    assert created.status_code == 201
    sid = created.json["session"]["session_id"]
    turn = api_client.post(f"/api/sessions/{sid}/messages", json={
        "request_id": "r1", "user_message": "blue running shoes under $80"
    })
    assert turn.status_code == 201
    asin = turn.json["turn"]["products"][0]["parent_asin"]
    feedback = api_client.put(f"/api/sessions/{sid}/turns/1/feedback/{asin}", json={
        "is_inaccurate": True, "reason": "over_budget", "note": "too expensive"
    })
    assert feedback.status_code == 200
    exported = api_client.get(f"/api/sessions/{sid}/export")
    assert exported.json["format"] == "compasscart-debug-session"


def test_same_pending_request_returns_202(api_client, blocked_service):
    response = api_client.post("/api/sessions/s/messages", json={
        "request_id": "r1", "user_message": "blue"
    })
    assert response.status_code == 202
```

- [ ] **Step 2: Verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_http.py -v
```

Expected: 404 for new API routes.

- [ ] **Step 3: Implement exact method/path mapping**

```text
GET    /api/sessions?scope=active|archived|all
POST   /api/sessions
GET    /api/sessions/{id}
PATCH  /api/sessions/{id}
POST   /api/sessions/{id}/messages
POST   /api/sessions/{id}/clone
PUT    /api/sessions/{id}/turns/{turn}/feedback/{asin}
GET    /api/sessions/{id}/export
POST   /api/import
```

Return 201 for new sessions/completed new turns/clones/imports, 200 for reads/patches/feedback/idempotent completed responses, 202 for an existing pending request, 400 for field validation, 401 auth, 404 missing resources, 409 request mismatch/replay mismatch/unresolved failure/turn limit, 413 size limit, 415 type mismatch, and 503 not-ready/busy. URL-decode IDs then validate against conservative ID/ASIN regexes. The API never exposes exception reprs or local paths.

- [ ] **Step 4: Write real-Agent fixture integration tests**

```python
def test_real_debug_console_two_turns_persist_and_rehydrate(
    fixture_catalog_path, tmp_path, valid_debug_token
):
    first = running_debug_app(fixture_catalog_path, tmp_path, valid_debug_token)
    session = first.client.create_session({"preference_tags": []})
    first.client.send(session.id, "r1", "I need running shoes under $80")
    first.client.send(session.id, "r2", "blue mesh and wide fit")
    first.close()

    second = running_debug_app(fixture_catalog_path, tmp_path, valid_debug_token)
    detail = second.client.get_session(session.id)
    assert [turn["turn"] for turn in detail["turns"]] == [1, 2]
    continued = second.client.send(session.id, "r3", "no brand preference")
    assert continued["turn"]["turn"] == 3
```

Also assert response recommendation order equals product rank order, feedback survives restart, export/import works, and an injected retriever failure remains a completed turn whose trace contains `fallbacks=["retriever"]` rather than a service error.

- [ ] **Step 5: Run API and integration tests; commit**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_http.py tests/integration/test_debug_console.py -v
git add src/compasscart_debug/http.py tests/unit/test_debug_http.py tests/integration/test_debug_console.py
git commit -m "feat: expose debug conversation API"
```

Expected: PASS with the real fixture Agent on turns 1, 2, and recovered turn 3.

---

### Task 8: Implement the Approved Three-Pane UI Shell and Safe Client State

**Files:**
- Create: `src/compasscart_debug/static/index.html`
- Create: `src/compasscart_debug/static/styles.css`
- Create: `src/compasscart_debug/static/js/api.js`
- Create: `src/compasscart_debug/static/js/dom.js`
- Create: `src/compasscart_debug/static/js/store.js`
- Create: `src/compasscart_debug/static/js/app.js`
- Create: `src/compasscart_debug/static/js/package.json`
- Create: `tests/js/debug_store.test.mjs`
- Modify: `tests/unit/test_debug_http.py`

- [ ] **Step 1: Write failing static/CSP/text-only tests**

```python
def test_index_has_semantic_three_pane_shell(wsgi_client):
    html = wsgi_client.get("/").text
    assert '<main id="debug-workspace"' in html
    assert 'id="conversation-pane"' in html
    assert 'id="recommendations-pane"' in html
    assert 'id="inspector-pane"' in html
    assert '<script type="module" src="/static/js/app.js"></script>' in html
    assert "<script>" not in html and " style=" not in html


def test_frontend_never_uses_dynamic_inner_html(project_root):
    scripts = (project_root / "src/compasscart_debug/static/js").glob("*.js")
    assert "innerHTML" not in "\n".join(path.read_text(encoding="utf-8") for path in scripts)
```

```javascript
import test from "node:test";
import assert from "node:assert/strict";
import { createStore } from "../../src/compasscart_debug/static/js/store.js";

test("selecting a turn updates recommendations and inspector atomically", () => {
  const store = createStore({ currentSession: { turns: [{ turn: 1 }, { turn: 2 }] } });
  store.selectTurn(2);
  assert.equal(store.getState().selectedTurn, 2);
  assert.equal(store.getState().selectedObservation.turn, 2);
});
```

- [ ] **Step 2: Verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_http.py -v
node --test tests/js/debug_store.test.mjs
```

Expected: static 404/import failure.

- [ ] **Step 3: Build semantic shell and external-only assets**

`index.html` contains a login view, app header, session selector, new/import/export/archive controls, left conversation pane, center recommendation pane, right inspector, `dialog` elements, live region, and reusable templates. It contains no inline script/style, no external CDN, no marketing copy, and no product image placeholder.

`src/compasscart_debug/static/js/package.json` contains exactly `{"type":"module","private":true}`. This gives Node's built-in test runner an ES-module boundary for the browser `.js` modules without changing unrelated repository scripts; it is not added to the HTTP static allowlist.

CSS tokens follow the approved mockup: light gray canvas, white panes, charcoal text, restrained teal/blue, thin borders, 8px radius, dense readable spacing, and no gradients. Desktop `>=1280px` uses `minmax(320px,.85fr) minmax(480px,1.4fr) minmax(320px,.9fr)`. Medium uses two panes plus inspector drawer. Mobile `<768px` uses Conversation/Recommendations/Diagnostics tabs and sticky composer. Add `:focus-visible`, non-color status cues, `prefers-reduced-motion`, `overflow-wrap:anywhere`, and horizontally scrolling raw `<pre>`.

- [ ] **Step 4: Implement API, DOM, and store primitives**

`api.js` reads/writes only `sessionStorage["compasscart_debug_token"]`, adds Bearer auth, parses safe errors, and handles JSON Blob download/File text import. `dom.js` creates nodes with `document.createElement` and assigns untrusted values with `textContent`; it formats nullable price/rating and stable pretty JSON. `store.js` owns auth, service status, sessions, current session, selected turn, mobile tab, inspector state, and busy flag. It does not duplicate server turn/business rules.

- [ ] **Step 5: Implement login/bootstrap and run tests**

Correct token loads health/sessions; 401 clears only the debug token and returns to login. Initializing/setup-required keep historical browsing available but disable send. Store subscriptions render stable pane states. Run:

```powershell
node --test tests/js/debug_store.test.mjs
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_http.py -v
git add src/compasscart_debug/static tests/js/debug_store.test.mjs tests/unit/test_debug_http.py
git commit -m "feat: add three-pane debug console shell"
```

Expected: Node and pytest PASS; no `innerHTML` or inline executable content.

---

### Task 9: Complete Conversation, Product, Diagnostics, and Browser Workflows

**Files:**
- Create: `src/compasscart_debug/static/js/conversation.js`
- Create: `src/compasscart_debug/static/js/recommendations.js`
- Create: `src/compasscart_debug/static/js/inspector.js`
- Create: `src/compasscart_debug/static/js/dialogs.js`
- Modify: `src/compasscart_debug/static/js/app.js`
- Create: `requirements-debug-dev.txt`
- Create: `tests/browser/test_debug_console_ui.py`

- [ ] **Step 1: Write failing browser tests for the primary flow**

```python
def test_create_send_select_and_annotate(page, debug_server, debug_token):
    page.goto(debug_server.url)
    page.get_by_label("访问口令").fill(debug_token)
    page.get_by_role("button", name="进入调试台").click()
    page.get_by_role("button", name="新建会话").click()
    page.get_by_label("偏好标签").fill("comfort, running")
    page.get_by_role("button", name="创建会话").click()
    page.get_by_label("消息").fill("I need blue running shoes under $80")
    page.get_by_role("button", name="发送").click()
    expect(page.get_by_text("Blue Mesh Running Shoes")).to_be_visible()
    expect(page.get_by_text("Buying")).to_be_visible()
    page.get_by_role("button", name=re.compile("标记.*不准确")).first.click()
    page.get_by_label("原因").select_option("over_budget")
    page.get_by_label("备注").fill("Budget mismatch")
    page.get_by_role("button", name="保存标记").click()
    page.reload()
    expect(page.get_by_text("Budget mismatch")).to_be_visible()
```

- [ ] **Step 2: Write failing keyboard/history/XSS tests**

Test Enter sends one request, Shift+Enter and IME composition do not send, busy blocks double submit, selected historical turn switches product and diagnostics together, and `<img src=x onerror=...>` in message/title/note renders literally without DOM execution. Test failed retry reuses request ID/message; turn 10 replaces composer with new/clone actions.

- [ ] **Step 3: Implement conversation and product modules**

`conversation.js` renders user/Agent messages and ask_attribute, selectable turns, composer keyboard semantics, pending/failed/retry states, and auto-selects latest completion. Preference tags exist only in the new-session dialog and become read-only once the first message is sent; the UI never offers profile mutation for an existing conversation. `recommendations.js` strictly uses `turn.products` rank order, shows explicit missing metadata, expands one feedback editor, supports save/update/clear, and gives mobile icon buttons at least 44px. No code infers scores or source contributions.

- [ ] **Step 4: Implement inspector and accessible dialogs**

Inspector displays route, intent version, candidate count, elapsed time, ask attribute, fallbacks, all constraint statuses/details, asked/pending/no-preference/query-history, raw response/state/trace, and a permanent `当前 Agent 未记录最终分数与召回源贡献` notice. Dialogs cover login/new/import/archive/clone, trap focus, close on Escape, and restore focus. Mobile tabs implement WAI-ARIA roles, roving tabindex, ArrowLeft/ArrowRight/Home/End.

- [ ] **Step 5: Run browser suite at required viewports**

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-debug-dev.txt
& .\.venv\Scripts\python.exe -m playwright install chromium
& .\.venv\Scripts\python.exe -m pytest tests/browser/test_debug_console_ui.py -v
```

Run cases at 1440x900, 1024x768, 390x844, and 320x568. Assert no page-level horizontal overflow, no overlapping primary controls, drawer focus returns, mobile tabs work, no console error/warn, and raw JSON cannot widen the page.

- [ ] **Step 6: Commit frontend behavior**

```powershell
git add src/compasscart_debug/static requirements-debug-dev.txt tests/browser/test_debug_console_ui.py
git commit -m "feat: make debug console interactive"
```

Expected: complete create/send/history/feedback/import/export/archive/clone UI flow passes.

---

### Task 10: Private Runtime Bundle, Atomic Installer, and Online Backup

**Files:**
- Create: `src/compasscart_debug/runtime_bundle.py`
- Create: `tools/package_debug_runtime.py`
- Create: `tools/install_debug_runtime.py`
- Create: `tools/backup_debug_data.py`
- Test: `tests/unit/test_debug_runtime_bundle.py`
- Test: `tests/unit/test_debug_backup.py`

- [ ] **Step 1: Write failing deterministic bundle tests**

```python
EXPECTED_RUNTIME_FILES = {
    "catalog.jsonl",
    "assets/SHA256SUMS",
    "assets/model/model.int8.onnx",
    "assets/model/tokenizer.json",
    "assets/product_vectors/product_ids.npy",
    "assets/product_vectors/scales.npy",
    "assets/product_vectors/vectors.int8.npy",
}


def test_runtime_bundle_is_deterministic_and_allowlisted(runtime_fixture, tmp_path):
    first = build_runtime_bundle(runtime_fixture, tmp_path / "first.zip")
    second = build_runtime_bundle(runtime_fixture, tmp_path / "second.zip")
    assert first.sha256 == second.sha256
    with zipfile.ZipFile(first.path) as archive:
        assert set(archive.namelist()) == EXPECTED_RUNTIME_FILES | {"runtime-manifest.json"}
        manifest = json.loads(archive.read("runtime-manifest.json"))
    assert manifest["schema_version"] == 1
    assert {row["path"] for row in manifest["files"]} == EXPECTED_RUNTIME_FILES
```

- [ ] **Step 2: Write failing installer attack tests**

```python
@pytest.mark.parametrize("member", ["../escape", "/absolute", "C:\\escape", "a\\..\\escape"])
def test_installer_rejects_escaping_paths(tmp_path, member):
    archive = malicious_zip(tmp_path / "bad.zip", member)
    with pytest.raises(RuntimeBundleError):
        install_runtime_bundle(archive, tmp_path / "runtime")


def test_failed_install_keeps_previous_current(tmp_path, valid_runtime_zip):
    root = tmp_path / "runtime"
    first = install_runtime_bundle(valid_runtime_zip, root)
    corrupted = corrupt_member(valid_runtime_zip, tmp_path / "corrupt.zip")
    with pytest.raises(RuntimeBundleError):
        install_runtime_bundle(corrupted, root)
    assert (root / "CURRENT").read_text(encoding="utf-8").strip() == first.runtime_id
```

- [ ] **Step 3: Verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_runtime_bundle.py -v
```

Expected: runtime bundle import failure.

- [ ] **Step 4: Implement deterministic package and safe install**

Manifest v1 contains schema version, runtime ID, and sorted `{path,size_bytes,sha256}` entries. Runtime ID is SHA256 of canonical manifest content without runtime ID. ZIP entries use fixed `(2026,1,1,0,0,0)` timestamp and mode. Packaging rejects missing/hash-mismatched quantized assets and excludes unquantized model/config/vocab files.

Installer rejects absolute/drive/`..`/backslash ambiguity, duplicates, symlinks, non-allowlisted files, more than 16 files, per-file size above 256 MiB, total expanded size above 512 MiB, and suspicious compression ratio above 200. Extract to `runtime/releases/<id>.staging-<random>`, stream-check size/hash, rename to `releases/<id>`, then atomically replace a text `CURRENT` file containing the 64-character runtime ID. Failure never changes CURRENT. `activate_runtime(root,id)` verifies the selected release before replacing CURRENT. On the next process start, `DebugConfig.resolve_runtime_paths()` reads that pointer and resolves `catalog.jsonl` plus `assets/` from the selected release.

- [ ] **Step 5: Write and implement WAL-safe backup tests**

```python
def test_backup_uses_sqlite_backup_api_while_wal_has_uncheckpointed_rows(tmp_path):
    source = tmp_path / "debug.sqlite3"
    repository = seeded_wal_repository(source)
    result = backup_debug_database(source, tmp_path / "backups")
    with sqlite3.connect(result.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    assert hashlib.sha256(result.database_path.read_bytes()).hexdigest() == result.sha256
```

Use `sqlite3.Connection.backup()` into a timestamped temporary DB, run `PRAGMA integrity_check`, hash it, write canonical metadata JSON, then atomically rename both. Never copy a live WAL DB file directly.

- [ ] **Step 6: Add exact CLIs, run tests, and commit**

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_debug_runtime_bundle.py tests/unit/test_debug_backup.py -v
& .\.venv\Scripts\python.exe -m tools.package_debug_runtime --help
& .\.venv\Scripts\python.exe -m tools.install_debug_runtime --help
& .\.venv\Scripts\python.exe -m tools.backup_debug_data --help
git add src/compasscart_debug/runtime_bundle.py tools/package_debug_runtime.py tools/install_debug_runtime.py tools/backup_debug_data.py tests/unit/test_debug_runtime_bundle.py tests/unit/test_debug_backup.py
git commit -m "feat: package private debug runtime assets"
```

Expected: deterministic fixture ZIP, all attack fixtures rejected, backup integrity passes.

---

### Task 11: Docker Compose and Render Persistent Deployment

**Files:**
- Create: `requirements-debug.txt`
- Modify: `requirements-debug-dev.txt`
- Create: `deploy/gunicorn.conf.py`
- Create: `Dockerfile.debug`
- Create: `.dockerignore`
- Create: `compose.debug.yaml`
- Create: `.env.debug.example`
- Create: `render.yaml`
- Create: `tests/deployment/test_debug_deployment.py`

- [ ] **Step 1: Write failing deployment-configuration tests**

```python
def test_render_is_single_paid_instance_with_persistent_disk(project_root):
    config = yaml.safe_load((project_root / "render.yaml").read_text(encoding="utf-8"))
    assert len(config["services"]) == 1
    service = config["services"][0]
    assert service["runtime"] == "docker"
    assert service["plan"] == "starter"
    assert service["numInstances"] == 1
    assert service["dockerfilePath"] == "./Dockerfile.debug"
    assert service["healthCheckPath"] == "/api/health/live"
    assert service["disk"]["mountPath"] == "/var/data"
    token = next(row for row in service["envVars"] if row["key"] == "COMPASSCART_DEBUG_TOKEN")
    assert token == {"key": "COMPASSCART_DEBUG_TOKEN", "sync": False}


def test_debug_image_never_copies_private_or_runtime_data(project_root):
    dockerfile = (project_root / "Dockerfile.debug").read_text(encoding="utf-8")
    assert "COPY data" not in dockerfile
    assert "COPY assets" not in dockerfile
    assert "COPY var" not in dockerfile
    assert "COPY .env" not in dockerfile
    assert "--workers 1" in (project_root / "deploy/gunicorn.conf.py").read_text(encoding="utf-8")
```

- [ ] **Step 2: Verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/deployment/test_debug_deployment.py -v
```

Expected: missing deployment files.

- [ ] **Step 3: Implement isolated requirements and one-process Gunicorn**

`requirements-debug.txt` contains only `-r requirements.txt` and `gunicorn>=23,<24`. `requirements-debug-dev.txt` contains `-r requirements-dev.txt`, `playwright>=1.56,<2`, `pytest-playwright>=0.7,<1`, and `PyYAML>=6,<7`. Gunicorn config binds `${HOST:-0.0.0.0}:${PORT:-8000}`, uses `workers=1`, `threads=4`, `worker_class="gthread"`, logs to stdout/stderr, timeout 240, and never preloads the app.

- [ ] **Step 4: Implement Dockerfile, ignore rules, and Compose**

Dockerfile uses `python:3.12-slim`, copies only requirements, root Agent entry, `src/compasscart`, `src/compasscart_debug`, required debug CLIs, and Gunicorn config, then runs as an unprivileged app user. `.dockerignore` excludes `.git`, worktrees, venv, env files, `var`, `dist`, catalog/public set, all assets, caches, logs, and tests.

Compose builds `Dockerfile.debug`, binds host only to `127.0.0.1:${COMPASSCART_DEBUG_PORT:-8000}`, mounts catalog and assets read-only, mounts `./var/debug:/var/data` writable, requires token with `${COMPASSCART_DEBUG_TOKEN:?}`, sets explicit DB/catalog/asset paths, `restart: unless-stopped`, `read_only: true`, `/tmp` tmpfs, `no-new-privileges`, drops all capabilities, and uses Python urllib for `/api/health/live`.

- [ ] **Step 5: Implement Render Blueprint**

One Docker web service uses `plan: starter`, `numInstances: 1`, `dockerfilePath: ./Dockerfile.debug`, live health, and a 1 GiB or larger `/var/data` disk. Set `HOST=0.0.0.0`, `COMPASSCART_DEBUG_DATABASE=/var/data/compasscart-debug.sqlite3`, `COMPASSCART_RUNTIME_ROOT=/var/data/runtime`, and `COMPASSCART_DEBUG_TOKEN: {sync:false}`. Do not hardcode PORT, secret, catalog, or asset paths. A fresh disk has no CURRENT pointer, so it stays setup-required until the verified runtime bundle is SCP-uploaded and installed.

- [ ] **Step 6: Run static and conditional container checks; commit**

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-debug-dev.txt
& .\.venv\Scripts\python.exe -m pytest tests/deployment/test_debug_deployment.py -v
if (Get-Command docker -ErrorAction SilentlyContinue) {
  docker compose -f compose.debug.yaml config
  docker build -f Dockerfile.debug -t compasscart-debug:test .
} else {
  Write-Output "Docker engine unavailable: static deployment tests completed; runtime build remains externally unverified."
}
git add requirements-debug.txt requirements-debug-dev.txt deploy/gunicorn.conf.py Dockerfile.debug .dockerignore compose.debug.yaml .env.debug.example render.yaml tests/deployment/test_debug_deployment.py
git commit -m "ops: make debug console deployable"
```

Expected on this workstation: pytest PASS and an explicit Docker-unavailable record; do not claim a successful image build without an engine.

---

### Task 12: Documentation, Submission Isolation, and Final Browser Verification

**Files:**
- Create: `docs/debug-console-deployment.md`
- Modify: `README.md`
- Modify: `reports/final/architecture.md`
- Modify: `reports/final/demo-script.md`
- Modify: `tests/contract/test_submission_package.py`
- Create: `docs/superpowers/assets/2026-08-25-debug-console-a-concept.png`
- Create during QA then remove: `var/debug/qa/`

- [ ] **Step 1: Write failing official-package isolation assertions**

```python
def test_official_zip_excludes_debug_companion(submission_zip):
    with zipfile.ZipFile(submission_zip) as archive:
        names = set(archive.namelist())
        requirements = archive.read("requirements.txt").decode("utf-8")
    forbidden = (
        "src/compasscart_debug/", "requirements-debug", "Dockerfile.debug",
        "compose.debug.yaml", "render.yaml", "deploy/", "debug.sqlite",
        "package_debug_runtime.py", "install_debug_runtime.py", "backup_debug_data.py",
    )
    assert not any(marker in name for marker in forbidden for name in names)
    assert "gunicorn" not in requirements.lower()
```

- [ ] **Step 2: Write complete local/Docker/Render operations documentation**

Deployment guide includes:

1. Generate token with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
2. Native local server command and explicit catalog/asset/DB variables.
3. Compose `.env` creation, `var/debug` creation, config/up/health/log/stop/start commands.
4. Docker Desktop/daemon auto-start caveat.
5. Runtime bundle packaging and encrypted Render SCP upload without a hardcoded account hostname.
6. Shell install command, ready check, rollback activation, backup command, restore steps.
7. New-computer migration: repository, `var/debug`, authorized catalog/assets, regenerated local env.
8. Render paid-instance/persistent-disk prerequisite; free service is not described as always-on.
9. Explicit statement that the debug companion is excluded from and not required by official scoring.

README gives a compact entry and links to the guide. Architecture report shows the read-only boundary. Demo script shows one Browsing and one Intent Override flow, feedback marking, export, restart recovery, and offline Agent independence.

- [ ] **Step 3: Run full automated verification**

```powershell
$env:PYTHONPATH = "src;."
& .\.venv\Scripts\python.exe -m pytest -q
node --test tests/js/debug_store.test.mjs
& .\.venv\Scripts\python.exe -m ruff check agent.py src tools tests
& .\.venv\Scripts\python.exe -m ruff format --check agent.py src tools tests
& .\.venv\Scripts\python.exe -m tools.package_submission
& .\.venv\Scripts\python.exe -m pytest tests/contract/test_submission_package.py -v
```

Expected: all tests/checks PASS; official ZIP contains no debug files/dependencies/secrets.

- [ ] **Step 4: Start the real debug app and verify with Browser/IAB first**

```powershell
$env:PYTHONPATH = "src;."
$env:COMPASSCART_DEBUG_TOKEN = & .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
$env:COMPASSCART_CATALOG_PATH = "tests/fixtures/catalog.jsonl"
$env:COMPASSCART_DISABLE_DENSE = "1"
$env:COMPASSCART_DEBUG_DATABASE = "var/debug/qa/debug.sqlite3"
& .\.venv\Scripts\python.exe -m tools.run_debug_server
```

Use Browser/IAB to execute login, new session, two turns, historical selection, feedback save/update/clear, export/import, archive/unarchive, clone, failed retry, and turn-10 state. Verify 1440x900, 1024x768, 390x844, and 320x568. Capture latest implementation screenshot under `var/debug/qa/`.

- [ ] **Step 5: Perform visual fidelity comparison**

Capture the accepted A mockup as `docs/superpowers/assets/2026-08-25-debug-console-a-concept.png`. Use `view_image` on both that concept and the latest browser screenshot. Record at least five comparisons: three-column proportions, toolbar density, chat hierarchy, ranked product anatomy, diagnostic ledger hierarchy, plus mobile collapse. Check visible copy, typography, palette, spacing, missing metadata, focus states, overflow, and no invented score/source data. Fix every material mismatch before continuing.

- [ ] **Step 6: Verify persistence and deployment evidence proportionately**

Stop and restart the native service and prove sessions/feedback remain. If Docker becomes available, run Compose stop/start and force-recreate persistence checks; if it remains unavailable, keep the limitation explicit and report only static Blueprint/Dockerfile verification. Do not create a paid Render service or transmit catalog/assets until the user signs into Render and confirms the specific paid deployment/upload action.

- [ ] **Step 7: Remove temporary QA artifacts and commit release documentation**

Delete only files created under the exact project directory `var/debug/qa/`; verify the resolved path before removal. Do not delete `var/debug` user data outside QA. Then:

```powershell
git add README.md docs/debug-console-deployment.md docs/superpowers/assets/2026-08-25-debug-console-a-concept.png reports/final/architecture.md reports/final/demo-script.md tests/contract/test_submission_package.py
git commit -m "docs: document persistent debug console"
git status --short
```

Expected: only pre-existing unrelated user files remain untracked/modified.

---

## Plan Completion Checks

- Every new Python function is introduced by a failing test and observed RED before implementation.
- Agent construction and every Agent/catalog/state/trace operation occur on the same dedicated thread.
- Normal turns never call reset; recovery calls one reset then completed-turn replay.
- Failed/pending turns cannot be skipped, and DB/Agent non-atomic failures force rehydration.
- Replay checks runtime identity plus full deterministic response and SessionState.
- Product order always follows response recommendations; no score/source explanation is fabricated.
- All untrusted text is assigned with `textContent`; CSP forbids inline and third-party code.
- SQLite feedback and sessions survive process restart; export/import and clone semantics stay distinct.
- Runtime bundle installation is checksum-verified, bounded, path-safe, atomic, and rollback-capable.
- Docker/Render use one process/one Agent worker, a persistent disk, secret token, and live-vs-ready health split.
- Official submission remains offline and excludes debug code, DB, deployment config, and Gunicorn.
- Browser verification covers desktop, medium, mobile, narrow mobile, accessibility, errors, and the full primary workflow.
