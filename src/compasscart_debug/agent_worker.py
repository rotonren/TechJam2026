"""Single-threaded, bounded execution boundary for the debug Agent."""

from __future__ import annotations

import inspect
import json
import os
import queue
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import TypeAlias

from .agent_adapter import AgentAdapter, TurnObservation
from .config import DebugConfig, RuntimeIdentity
from .errors import (
    NotReadyError,
    RuntimeSetupError,
    WorkerBusyError,
    WorkerTimeoutError,
)

WorkerState: TypeAlias = str
_STOP = object()
_MAX_CLOSE_JOIN_SECONDS = 0.25


@dataclass
class _Command:
    callback: Callable[[AgentAdapter], object]
    future: Future[object]


class AgentWorker:
    """Own an Agent on one dedicated thread and serialize every operation."""

    def __init__(
        self,
        factory: Callable[..., AgentAdapter] | None = None,
        *,
        config: DebugConfig | None = None,
        queue_size: int | None = None,
        command_timeout: float | None = None,
        command_timeout_seconds: float | None = None,
        timeout: float | None = None,
        daemon: bool = True,
    ) -> None:
        self._config = config or DebugConfig()
        selected_size = (
            self._config.command_queue_size if queue_size is None else queue_size
        )
        if (
            not isinstance(selected_size, int)
            or isinstance(selected_size, bool)
            or selected_size < 1
        ):
            raise ValueError("queue_size must be positive")
        selected_timeout = command_timeout
        if selected_timeout is None:
            selected_timeout = command_timeout_seconds
        if selected_timeout is None:
            selected_timeout = timeout
        if selected_timeout is None:
            selected_timeout = self._config.command_timeout_seconds
        if selected_timeout <= 0:
            raise ValueError("command_timeout must be positive")

        self._queue: queue.Queue[_Command | object] = queue.Queue(maxsize=selected_size)
        self._timeout = float(selected_timeout)
        self._daemon = bool(daemon)
        self._factory = (
            factory
            if factory is not None
            else lambda: _build_default_adapter(self._config)
        )
        self._state_lock = threading.RLock()
        self._state_value: WorkerState = "stopped"
        self._adapter: AgentAdapter | None = None
        self._identity_value: RuntimeIdentity | None = None
        self._thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._active_futures: set[Future[object]] = set()
        self._closed = False
        self._startup_error: BaseException | None = None

    @property
    def state(self) -> WorkerState:
        with self._state_lock:
            return self._state_value

    @property
    def identity(self) -> RuntimeIdentity:
        with self._state_lock:
            if self._state_value != "ready" or self._identity_value is None:
                raise NotReadyError()
            return self._identity_value

    @property
    def startup_error(self) -> BaseException | None:
        with self._state_lock:
            return self._startup_error

    def start(self) -> None:
        """Start initialization and return without waiting for the factory."""

        with self._state_lock:
            if self._closed:
                raise NotReadyError()
            if self._thread is not None and self._thread.is_alive():
                return
            self._clear_queue_locked()
            self._state_value = "initializing"
            self._adapter = None
            self._identity_value = None
            self._startup_error = None
            self._ready_event.clear()
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._run,
                name="compasscart-agent-worker",
                daemon=self._daemon,
            )
            self._thread = thread
            thread.start()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Wait for initialization to finish; return whether it became ready."""

        self._ready_event.wait(timeout)
        return self.state == "ready"

    def reset_session(self, session_id: str, profile: Mapping[str, object]) -> None:
        self._submit(lambda adapter: adapter.reset_session(session_id, profile))

    def has_session(self, session_id: str) -> bool:
        result = self._submit(lambda adapter: adapter.has_session(session_id))
        return bool(result)

    def observe_turn(self, session_id: str, message: str, turn: int) -> TurnObservation:
        result = self._submit(
            lambda adapter: adapter.observe_turn(session_id, message, turn)
        )
        return result  # type: ignore[return-value]

    def rehydrate(
        self,
        session_id: str,
        profile: Mapping[str, object],
        completed_messages: Sequence[object],
    ) -> list[TurnObservation]:
        result = self._submit(
            lambda adapter: adapter.rehydrate(session_id, profile, completed_messages)
        )
        return result  # type: ignore[return-value]

    def invalidate(self) -> None:
        """Discard the Agent and stop the current worker so it can be rebuilt."""

        with self._state_lock:
            state = self._state_value
            thread = self._thread
        if state == "ready" and thread is not None and thread.is_alive():
            self._submit(self._invalidate_on_worker)
            return
        with self._state_lock:
            if self._closed:
                return
            self._adapter = None
            self._identity_value = None
            self._stop_event.set()
            self._state_value = "stopped"
            self._ready_event.set()
        self._wake_and_reject_pending()

    def close(self, timeout: float | None = None) -> None:
        """Reject new work, wake the worker, and return within a bounded join."""

        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            self._adapter = None
            self._identity_value = None
            self._state_value = "stopped"
            self._ready_event.set()
            self._reject_active_locked()
            thread = self._thread
        self._wake_and_reject_pending()
        if thread is not None and thread is not threading.current_thread():
            join_timeout = (
                min(self._timeout, _MAX_CLOSE_JOIN_SECONDS)
                if timeout is None
                else max(0.0, timeout)
            )
            thread.join(join_timeout)

    def _submit(
        self,
        callback: Callable[[AgentAdapter], object] | Callable[[], object],
        *,
        wait: bool = True,
    ) -> object | None:
        with self._state_lock:
            if (
                self._closed
                or self._state_value != "ready"
                or self._adapter is None
                or self._thread is None
                or not self._thread.is_alive()
            ):
                raise NotReadyError()
        future: Future[object] = Future()

        def invoke(adapter: AgentAdapter) -> object:
            # The public callbacks all accept the adapter.  A zero-argument
            # callback remains useful for focused queue/timeout tests.
            try:
                signature = inspect.signature(callback)
            except (TypeError, ValueError):
                return callback(adapter)  # type: ignore[misc]
            positional = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            accepts_varargs = any(
                parameter.kind is inspect.Parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            )
            if not positional and not accepts_varargs:
                return callback()  # type: ignore[call-arg]
            return callback(adapter)  # type: ignore[misc]

        command = _Command(invoke, future)
        with self._state_lock:
            if (
                self._closed
                or self._state_value != "ready"
                or self._adapter is None
                or self._thread is None
                or not self._thread.is_alive()
            ):
                raise NotReadyError()
            try:
                self._queue.put_nowait(command)
            except queue.Full as error:
                raise WorkerBusyError() from error
        if not wait:
            return None
        try:
            return future.result(timeout=self._timeout)
        except FutureTimeoutError as error:
            raise WorkerTimeoutError() from error

    def _invalidate_on_worker(self, adapter: AgentAdapter) -> None:
        del adapter
        with self._state_lock:
            self._adapter = None
            self._identity_value = None
            self._stop_event.set()
            self._state_value = "stopped"
            self._ready_event.set()

    def _run(self) -> None:
        try:
            adapter = self._construct_adapter()
            identity = adapter.identity
        except Exception as error:  # noqa: BLE001 - classify startup failures safely.
            setup_failure = _is_setup_failure(error)
            self._wake_and_reject_pending()
            with self._state_lock:
                if self._closed or self._stop_event.is_set():
                    # close()/invalidate() cancelled startup; a late factory
                    # result must not overwrite the caller's stopped state.
                    self._state_value = "stopped"
                    self._adapter = None
                    self._identity_value = None
                    self._ready_event.set()
                    self._thread = None
                    return
                self._startup_error = error
                self._state_value = "setup_required" if setup_failure else "fatal"
                self._adapter = None
                self._identity_value = None
                self._ready_event.set()
                self._thread = None
            return

        with self._state_lock:
            if self._stop_event.is_set() or self._closed:
                self._adapter = None
                self._identity_value = None
                self._state_value = "stopped"
                self._ready_event.set()
                self._thread = None
                return
            self._adapter = adapter
            self._identity_value = identity
            self._state_value = "ready"
            self._ready_event.set()

        try:
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is _STOP:
                    break
                command = item
                if not isinstance(command, _Command):
                    continue
                if command.future.cancelled():
                    continue
                current = self._adapter
                if current is None:
                    if not command.future.done():
                        command.future.set_exception(NotReadyError())
                    continue
                with self._state_lock:
                    if self._stop_event.is_set() or self._closed:
                        if not command.future.done():
                            command.future.set_exception(NotReadyError())
                        continue
                    self._active_futures.add(command.future)
                try:
                    result = command.callback(current)
                except Exception as error:  # noqa: BLE001 - propagate through Future.
                    if not command.future.done():
                        command.future.set_exception(error)
                else:
                    if not command.future.done():
                        command.future.set_result(result)
                finally:
                    with self._state_lock:
                        self._active_futures.discard(command.future)
        finally:
            self._wake_and_reject_pending()
            with self._state_lock:
                if self._state_value == "ready" or self._stop_event.is_set():
                    self._state_value = "stopped"
                    self._adapter = None
                    self._identity_value = None
                    self._ready_event.set()
                self._thread = None

    def _construct_adapter(self) -> AgentAdapter:
        factory = self._factory
        # Supporting a one-argument injected factory keeps configuration
        # explicit while preserving the common no-argument test factory.
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            return factory()  # type: ignore[call-arg,return-value]
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        required = [
            parameter
            for parameter in positional
            if parameter.default is inspect.Parameter.empty
        ]
        if required:
            return factory(self._config)  # type: ignore[call-arg,return-value]
        return factory()  # type: ignore[call-arg,return-value]

    def _wake_and_reject_pending(self) -> None:
        with self._state_lock:
            self._clear_queue_locked()
            try:
                self._queue.put_nowait(_STOP)
            except queue.Full:
                # A command can only occupy the queue while the worker is
                # running; the stop event still guarantees loop exit.
                pass

    def _clear_queue_locked(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, _Command) and not item.future.done():
                item.future.set_exception(NotReadyError())

    def _reject_active_locked(self) -> None:
        for future in tuple(self._active_futures):
            if not future.done():
                future.set_exception(NotReadyError())


def _is_setup_failure(error: BaseException) -> bool:
    if isinstance(
        error,
        (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            NotADirectoryError,
            RuntimeSetupError,
            OSError,
            UnicodeError,
            EOFError,
            json.JSONDecodeError,
        ),
    ):
        return True
    name = type(error).__name__.lower()
    message = str(error).lower()
    return any(token in name or token in message for token in ("checksum", "setup"))


def _build_default_adapter(config: DebugConfig) -> AgentAdapter:
    """Construct the current public Agent and its immutable runtime identity."""

    paths = config.resolve_runtime_paths()
    runtime_config = config.runtime_config(paths)
    # Importing the root entry point here ensures no Agent/catalog work occurs
    # in the caller thread.
    from agent import Agent

    agent = Agent(paths.catalog_path, config=runtime_config)
    dense_disabled = os.environ.get("COMPASSCART_DISABLE_DENSE") == "1"
    identity = RuntimeIdentity.build(
        os.environ.get("COMPASSCART_VERSION", "development"),
        paths.catalog_path,
        runtime_config,
        runtime_config.dense_manifest_path,
        dense_disabled,
    )
    return AgentAdapter(agent, identity)
