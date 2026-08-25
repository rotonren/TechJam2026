from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from compasscart_debug.agent_adapter import AgentAdapter, TurnObservation
from compasscart_debug.agent_worker import AgentWorker
from compasscart_debug.config import RuntimeIdentity
from compasscart_debug.errors import (
    NotReadyError,
    RuntimeSetupError,
    WorkerBusyError,
    WorkerTimeoutError,
)

IDENTITY = RuntimeIdentity("agent-v1", "catalog", "config", None)


@dataclass
class FakeState:
    session_id: str
    turn: int = 0


class FakeAgent:
    def __init__(self) -> None:
        self.sessions: dict[str, FakeState] = {}
        self.traces = SimpleNamespace(records=[])
        self.catalog = SimpleNamespace(
            products={
                "A": {
                    "title": "A product",
                    "price": 2,
                    "average_rating": 4.5,
                    "rating_number": 4,
                    "store": "Store",
                    "categories": ["demo"],
                    "features": [],
                    "details": {},
                }
            },
            attributes={"A": {"color": ("blue",)}},
        )
        self.thread_ids: list[int] = []
        self.reset_calls: list[tuple[str, dict[str, object]]] = []
        self.respond_calls: list[tuple[str, str, int, int]] = []
        self.evict_after_respond = False
        self.trace_mismatch = False

    def _record_thread(self) -> None:
        self.thread_ids.append(threading.get_ident())

    def reset(self, session_id: str, profile: dict[str, object]) -> None:
        self._record_thread()
        self.reset_calls.append((session_id, profile))
        self.sessions[session_id] = FakeState(session_id)

    def respond(
        self, session_id: str, message: str, turn: int, top_k: int
    ) -> dict[str, object]:
        self._record_thread()
        self.respond_calls.append((session_id, message, turn, top_k))
        state = self.sessions[session_id]
        state.turn = turn
        trace_turn = turn - 1 if self.trace_mismatch else turn
        self.traces.records.append({"session_id": session_id, "turn": trace_turn})
        if self.evict_after_respond:
            self.sessions.pop(session_id, None)
        return {
            "message": f"reply:{message}",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": "A", "score": 0.5}],
            "usage": {},
        }


@dataclass
class Factory:
    agent: FakeAgent = field(default_factory=FakeAgent)
    error: BaseException | None = None
    calls: int = 0
    thread_ids: list[int] = field(default_factory=list)

    def __call__(self) -> AgentAdapter:
        self.calls += 1
        self.thread_ids.append(threading.get_ident())
        if self.error is not None:
            raise self.error
        return AgentAdapter(self.agent, IDENTITY)


@dataclass
class BlockingFactory:
    agent: FakeAgent = field(default_factory=FakeAgent)
    error: BaseException | None = None
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    calls: int = 0

    def __call__(self) -> AgentAdapter:
        self.calls += 1
        self.entered.set()
        self.release.wait(2.0)
        self.finished.set()
        if self.error is not None:
            raise self.error
        return AgentAdapter(self.agent, IDENTITY)


def _wait(worker: AgentWorker, expected: str, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker.state == expected:
            return
        time.sleep(0.005)
    raise AssertionError(f"worker state is {worker.state!r}, expected {expected!r}")


def test_factory_and_all_agent_calls_share_one_worker_thread() -> None:
    factory = Factory()
    worker = AgentWorker(factory, command_timeout=1.0)
    worker.start()
    assert worker.wait_until_ready(2.0)

    worker.reset_session("s", {})
    first = worker.observe_turn("s", "one", 1)
    second = worker.observe_turn("s", "two", 2)
    worker.close()

    assert isinstance(first, TurnObservation)
    assert second.response["message"] == "reply:two"
    assert len(set(factory.thread_ids + factory.agent.thread_ids)) == 1
    assert factory.thread_ids[0] != threading.get_ident()
    assert factory.agent.reset_calls == [("s", {})]
    assert factory.agent.respond_calls == [
        ("s", "one", 1, 10),
        ("s", "two", 2, 10),
    ]


def test_normal_observe_does_not_reset_and_rehydrate_resets_once_in_order() -> None:
    factory = Factory()
    worker = AgentWorker(factory, command_timeout=1.0)
    worker.start()
    assert worker.wait_until_ready(2.0)

    worker.reset_session("s", {"color": "blue"})
    worker.observe_turn("s", "one", 1)
    worker.observe_turn("s", "two", 2)
    replay = worker.rehydrate(
        "s",
        {"color": "blue"},
        [(1, "one"), (2, "two")],
    )
    worker.close()

    assert len(replay) == 2
    assert factory.agent.reset_calls == [
        ("s", {"color": "blue"}),
        ("s", {"color": "blue"}),
    ]
    assert factory.agent.respond_calls[-2:] == [
        ("s", "one", 1, 10),
        ("s", "two", 2, 10),
    ]


def test_adapter_returns_no_trace_when_latest_record_does_not_match() -> None:
    agent = FakeAgent()
    adapter = AgentAdapter(agent, IDENTITY)
    adapter.reset_session("s", {})
    agent.trace_mismatch = True

    observation = adapter.observe_turn("s", "one", 1)

    assert observation.trace is None
    assert observation.response["recommendations"] == [{"parent_asin": "A"}]


def test_worker_detects_evicted_session() -> None:
    factory = Factory()
    worker = AgentWorker(factory, command_timeout=1.0)
    worker.start()
    assert worker.wait_until_ready(2.0)
    worker.reset_session("s", {})
    # Simulate the bounded SessionStore evicting a least-recently-used entry.
    factory.agent.sessions.pop("s")

    assert worker.has_session("s") is False
    worker.close()


@pytest.mark.parametrize(
    "error", [FileNotFoundError("catalog"), RuntimeSetupError("bad setup")]
)
def test_expected_factory_setup_failures_enter_setup_required(
    error: BaseException,
) -> None:
    factory = Factory(error=error)
    worker = AgentWorker(factory)
    worker.start()
    _wait(worker, "setup_required")
    assert worker.state == "setup_required"
    worker.close()


def test_unexpected_factory_failure_enters_fatal() -> None:
    factory = Factory(error=ValueError("boom"))
    worker = AgentWorker(factory)
    worker.start()
    _wait(worker, "fatal")
    assert worker.state == "fatal"
    worker.close()


def test_factory_returning_non_adapter_enters_fatal() -> None:
    worker = AgentWorker(lambda: object())  # type: ignore[arg-type]
    worker.start()

    _wait(worker, "fatal")

    assert worker.state == "fatal"
    worker.close()


def test_start_can_retry_after_setup_failure() -> None:
    factory = Factory(error=FileNotFoundError("catalog"))
    worker = AgentWorker(factory)
    worker.start()
    _wait(worker, "setup_required")

    factory.error = None
    worker.start()

    assert worker.wait_until_ready(2.0)
    assert factory.calls == 2
    worker.close()


def test_close_wins_over_a_late_factory_failure() -> None:
    factory = BlockingFactory(error=FileNotFoundError("catalog"))
    worker = AgentWorker(factory, command_timeout=0.1)
    worker.start()
    assert factory.entered.wait(1.0)

    worker.close(timeout=0.01)
    factory.release.set()
    assert factory.finished.wait(1.0)
    time.sleep(0.01)

    assert worker.state == "stopped"


def test_invalidate_wins_over_a_late_factory_success() -> None:
    factory = BlockingFactory()
    worker = AgentWorker(factory, command_timeout=0.1)
    worker.start()
    assert factory.entered.wait(1.0)

    worker.invalidate()
    factory.release.set()
    assert factory.finished.wait(1.0)
    time.sleep(0.01)

    assert worker.state == "stopped"


def test_invalidate_during_startup_can_restart_after_late_factory_failure() -> None:
    factory = BlockingFactory(error=RuntimeSetupError("bad setup"))
    worker = AgentWorker(factory, command_timeout=0.1)
    worker.start()
    assert factory.entered.wait(1.0)

    worker.invalidate()
    factory.release.set()
    assert factory.finished.wait(1.0)
    time.sleep(0.01)
    assert worker.state == "stopped"

    factory.error = None
    factory.entered.clear()
    factory.release.clear()
    worker.start()
    assert factory.entered.wait(1.0)
    factory.release.set()
    assert worker.wait_until_ready(1.0)
    assert factory.calls == 2
    worker.close()


def test_invalidate_drops_identity_and_allows_rebuild() -> None:
    factory = Factory()
    worker = AgentWorker(factory)
    worker.start()
    assert worker.wait_until_ready(2.0)
    assert worker.identity == IDENTITY

    worker.invalidate()

    assert worker.state == "stopped"
    with pytest.raises(NotReadyError):
        _ = worker.identity
    worker.start()
    assert worker.wait_until_ready(2.0)
    assert factory.calls == 2
    worker.close()


def test_identity_is_unavailable_until_ready_and_is_immutable() -> None:
    factory = Factory()
    worker = AgentWorker(factory)

    with pytest.raises(NotReadyError):
        _ = worker.identity

    worker.start()
    assert worker.wait_until_ready(2.0)
    assert worker.identity == IDENTITY
    with pytest.raises((AttributeError, TypeError)):
        worker.identity.agent_version = "changed"  # type: ignore[misc]
    worker.close()


def test_bounded_queue_fails_fast_and_timeout_is_stable() -> None:
    factory = Factory()
    worker = AgentWorker(factory, queue_size=1, command_timeout=0.05)
    worker.start()
    assert worker.wait_until_ready(2.0)
    worker.reset_session("s", {})

    entered = threading.Event()
    release = threading.Event()

    def block() -> None:
        entered.set()
        release.wait(2.0)

    worker._submit(block, wait=False)  # type: ignore[attr-defined]
    assert entered.wait(1.0)
    queued = worker._submit(lambda: None, wait=False)  # type: ignore[attr-defined]
    assert queued is None
    started = time.monotonic()
    with pytest.raises(WorkerBusyError):
        worker._submit(lambda: None, wait=False)  # type: ignore[attr-defined]
    assert time.monotonic() - started < 0.2
    release.set()
    worker.close()


def test_command_timeout_maps_to_worker_timeout() -> None:
    factory = Factory()
    worker = AgentWorker(factory, command_timeout=0.02)
    worker.start()
    assert worker.wait_until_ready(2.0)
    entered = threading.Event()
    release = threading.Event()

    def block() -> None:
        entered.set()
        release.wait(1.0)

    with pytest.raises(WorkerTimeoutError):
        worker._submit(block)  # type: ignore[attr-defined]
    assert entered.is_set()
    release.set()
    worker.close()


def test_close_rejects_new_commands_without_hanging() -> None:
    factory = Factory()
    worker = AgentWorker(factory)
    worker.start()
    assert worker.wait_until_ready(2.0)
    worker.close()

    with pytest.raises(NotReadyError):
        worker.has_session("s")


def test_close_unblocks_a_caller_waiting_on_a_running_command() -> None:
    factory = Factory()
    worker = AgentWorker(factory, command_timeout=5.0)
    worker.start()
    assert worker.wait_until_ready(2.0)
    entered = threading.Event()
    release = threading.Event()
    outcome: list[BaseException] = []

    def block() -> None:
        entered.set()
        release.wait(2.0)

    def call() -> None:
        try:
            worker._submit(block)  # type: ignore[attr-defined]
        except Exception as error:  # noqa: BLE001 - capture the worker outcome.
            outcome.append(error)

    caller = threading.Thread(target=call)
    caller.start()
    assert entered.wait(1.0)
    worker.close(timeout=0.01)
    caller.join(0.5)
    release.set()
    assert not caller.is_alive()
    assert outcome and isinstance(outcome[0], NotReadyError)


def test_close_default_join_budget_is_bounded_for_a_blocked_command() -> None:
    factory = Factory()
    worker = AgentWorker(factory, command_timeout=2.0)
    worker.start()
    assert worker.wait_until_ready(2.0)
    entered = threading.Event()
    release = threading.Event()
    outcome: list[BaseException] = []

    def block() -> None:
        entered.set()
        release.wait(2.0)

    def call() -> None:
        try:
            worker._submit(block)  # type: ignore[attr-defined]
        except Exception as error:  # noqa: BLE001 - capture the worker outcome.
            outcome.append(error)

    caller = threading.Thread(target=call)
    caller.start()
    assert entered.wait(1.0)
    started = time.monotonic()
    worker.close()
    elapsed = time.monotonic() - started
    release.set()
    caller.join(0.5)

    assert elapsed < 0.5
    assert outcome and isinstance(outcome[0], NotReadyError)
