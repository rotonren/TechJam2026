from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

from .models import Constraint, SessionState
from .normalization import normalize_value
from .parser import MessageParser, ParsedConstraint

_PROFILE_FEATURES = {
    "comfort": "comfortable",
    "durability": "durable",
    "warmth": "warm",
    "weather": "weatherproof",
}


class SessionStore:
    def __init__(self, parser: MessageParser, max_sessions: int = 1_000) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self._parser = parser
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._last_updates: dict[str, tuple[int, str, SessionState]] = {}

    def reset(self, session_id: str, profile: dict[str, object]) -> SessionState:
        state = SessionState(session_id=session_id)
        for parsed in self._profile_constraints(profile):
            state.constraints.append(
                self._to_constraint(parsed, turn=0, intent_version=1)
            )
        self._sessions[session_id] = state
        self._sessions.move_to_end(session_id)
        self._last_updates.pop(session_id, None)
        self._trim()
        return state

    def update(
        self,
        session_id: str,
        message: str,
        turn: int,
        expected_attribute: str | None = None,
    ) -> SessionState:
        prior_update = self._last_updates.get(session_id)
        if prior_update and prior_update[:2] == (turn, message):
            return prior_update[2]

        state = self._sessions.get(session_id)
        if state is None:
            state = self.reset(session_id, {})

        result = self._parser.parse(message, turn, expected_attribute)
        incoming = list(result.constraints)
        is_override = result.is_override or self._has_explicit_conflict(state, incoming)

        if is_override:
            self._begin_new_intent(state, incoming)

        if result.no_preference_attribute:
            state.no_preference_attributes.add(result.no_preference_attribute)
            self._reject_attribute(state, result.no_preference_attribute)

        for parsed in incoming:
            self._merge_constraint(state, parsed, turn)

        state.turn = turn
        if result.route_hint:
            state.route = result.route_hint
        self._sessions.move_to_end(session_id)
        self._last_updates[session_id] = (turn, message, state)
        return state

    @staticmethod
    def _has_explicit_conflict(
        state: SessionState, incoming: list[ParsedConstraint]
    ) -> bool:
        new_values = {(item.attribute, item.value) for item in incoming if item.is_hard}
        return any(
            old.is_hard
            and old.source != "profile"
            and any(
                attribute == old.attribute and value != old.value
                for attribute, value in new_values
            )
            for old in state.active_constraints()
        )

    def _begin_new_intent(
        self, state: SessionState, incoming: list[ParsedConstraint]
    ) -> None:
        old_version = state.intent_version
        state.intent_version += 1
        incoming_by_attribute: dict[str, set[str]] = {}
        for item in incoming:
            incoming_by_attribute.setdefault(item.attribute, set()).add(item.value)

        retained: list[Constraint] = []
        for index, old in enumerate(state.constraints):
            if old.status != "active" or old.intent_version != old_version:
                continue
            conflicts = (
                old.attribute in incoming_by_attribute
                and old.value not in incoming_by_attribute[old.attribute]
            )
            if conflicts:
                state.constraints[index] = replace(old, status="superseded")
            elif (
                old.is_hard and old.source != "profile" and old.attribute != "category"
            ):
                retained.append(replace(old, intent_version=state.intent_version))
        state.constraints.extend(retained)

    def _merge_constraint(
        self, state: SessionState, parsed: ParsedConstraint, turn: int
    ) -> None:
        active = state.active_constraints()
        if any(
            item.attribute == parsed.attribute and item.value == parsed.value
            for item in active
        ):
            return

        for index, old in enumerate(state.constraints):
            if old.status != "active" or old.attribute != parsed.attribute:
                continue
            if old.source == "profile" and old.value != parsed.value:
                state.constraints[index] = replace(old, status="rejected")
            elif parsed.is_hard and old.value != parsed.value:
                state.constraints[index] = replace(old, status="superseded")

        state.constraints.append(
            self._to_constraint(parsed, turn, state.intent_version)
        )

    @staticmethod
    def _reject_attribute(state: SessionState, attribute: str) -> None:
        for index, old in enumerate(state.constraints):
            if old.status == "active" and old.attribute == attribute:
                status = "rejected" if old.source == "profile" else "superseded"
                state.constraints[index] = replace(old, status=status)

    @staticmethod
    def _to_constraint(
        parsed: ParsedConstraint, turn: int, intent_version: int
    ) -> Constraint:
        return Constraint(
            attribute=parsed.attribute,
            value=parsed.value,
            confidence=parsed.confidence,
            is_hard=parsed.is_hard,
            source=parsed.source,
            created_turn=turn,
            intent_version=intent_version,
        )

    @staticmethod
    def _profile_constraints(profile: dict[str, object]) -> list[ParsedConstraint]:
        raw_tags = profile.get("preference_tags")
        if not isinstance(raw_tags, list):
            return []
        result: list[ParsedConstraint] = []
        parser = MessageParser()
        for raw_tag in raw_tags:
            tag = normalize_value(raw_tag)
            if not tag:
                continue
            parsed = parser.parse(tag, turn=0).constraints
            if parsed:
                for item in parsed:
                    result.append(
                        replace(item, confidence=0.25, is_hard=False, source="profile")
                    )
                continue
            value = _PROFILE_FEATURES.get(tag, tag)
            result.append(ParsedConstraint("feature", value, 0.25, False, "profile"))
        return result

    def _trim(self) -> None:
        while len(self._sessions) > self._max_sessions:
            session_id, _ = self._sessions.popitem(last=False)
            self._last_updates.pop(session_id, None)
