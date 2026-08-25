"""Read-only bridge from the debug worker to the public CompassCart Agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import RuntimeIdentity
from .snapshots import (
    capture_exact_trace,
    snapshot_products,
    snapshot_response,
    snapshot_state,
)


@dataclass(frozen=True)
class TurnObservation:
    """The serializable result of one Agent turn."""

    response: dict[str, object]
    products: list[dict[str, object]]
    state: dict[str, object]
    trace: dict[str, object] | None


class AgentAdapter:
    """Keep all reads of a mutable Agent behind the worker thread.

    The adapter deliberately does not expose the Agent as a public attribute.
    It is created and used by :class:`AgentWorker` on one dedicated thread.
    """

    __slots__ = ("_agent", "_identity")

    def __init__(self, agent: object, identity: RuntimeIdentity) -> None:
        self._agent = agent
        self._identity = identity

    @property
    def identity(self) -> RuntimeIdentity:
        return self._identity

    def reset_session(self, session_id: str, profile: Mapping[str, object]) -> None:
        agent = self._agent
        agent.reset(session_id, profile)  # type: ignore[attr-defined]

    def has_session(self, session_id: str) -> bool:
        agent = self._agent
        sessions = agent.sessions  # type: ignore[attr-defined]
        return sessions.get(session_id) is not None

    def observe_turn(self, session_id: str, message: str, turn: int) -> TurnObservation:
        agent = self._agent
        # Keep this call exactly on the worker and use the public Agent API.
        response = agent.respond(session_id, message, turn, 10)  # type: ignore[attr-defined]
        state = agent.sessions.get(session_id)  # type: ignore[attr-defined]
        if state is None:
            raise RuntimeError("Agent session disappeared after respond")

        response_snapshot = snapshot_response(response)
        products_snapshot = snapshot_products(agent, response)
        state_snapshot = snapshot_state(state)
        trace_snapshot = capture_exact_trace(
            agent.traces.records,
            session_id,
            turn,  # type: ignore[attr-defined]
        )
        return TurnObservation(
            response=response_snapshot,
            products=products_snapshot,
            state=state_snapshot,
            trace=trace_snapshot,
        )

    def rehydrate(
        self,
        session_id: str,
        profile: Mapping[str, object],
        completed_messages: Sequence[object],
    ) -> list[TurnObservation]:
        """Reset once and replay completed messages in their supplied order."""

        self.reset_session(session_id, profile)
        observations: list[TurnObservation] = []
        for index, item in enumerate(completed_messages, start=1):
            turn, message = _message_turn(item, index)
            observations.append(self.observe_turn(session_id, message, turn))
        return observations


def _message_turn(item: object, default_turn: int) -> tuple[int, str]:
    """Accept repository records and the small tuple/string test form.

    The repository exposes ``TurnRecord(turn, user_message)`` while a caller
    may also pass ``(turn, message)`` or just messages.  This normalization
    keeps replay ordering explicit without requiring a repository import here.
    """

    if isinstance(item, str):
        return default_turn, item

    if isinstance(item, Mapping):
        raw_turn = item.get("turn", default_turn)
        raw_message = item.get("user_message", item.get("message"))
        return _validated_turn_message(raw_turn, raw_message, default_turn)

    if isinstance(item, (tuple, list)) and len(item) == 2:
        first, second = item
        if isinstance(first, int) and isinstance(second, str):
            return _validated_turn_message(first, second, default_turn)
        if isinstance(first, str) and isinstance(second, int):
            return _validated_turn_message(second, first, default_turn)

    raw_turn = getattr(item, "turn", default_turn)
    raw_message = getattr(item, "user_message", getattr(item, "message", None))
    return _validated_turn_message(raw_turn, raw_message, default_turn)


def _validated_turn_message(
    raw_turn: Any, raw_message: Any, default_turn: int
) -> tuple[int, str]:
    turn = default_turn if raw_turn is None else raw_turn
    if not isinstance(turn, int) or isinstance(turn, bool) or turn < 1:
        raise ValueError("Completed replay turn is invalid.")
    if not isinstance(raw_message, str):
        raise TypeError("Completed replay message is invalid.")
    return turn, raw_message
