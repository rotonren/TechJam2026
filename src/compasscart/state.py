from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

from .models import Constraint, SessionState
from .normalization import normalize_category_value, normalize_value
from .parser import MessageParser, ParsedConstraint

_PROFILE_FEATURES = {
    "comfort": "comfortable",
    "comfortable": "comfortable",
    "fit": "comfortable",
    "durability": "durable",
    "durable": "durable",
    "warmth": "warm",
    "warm": "warm",
    "weather": "weatherproof",
    "weatherproof": "weatherproof",
    "lightweight": "lightweight",
    "breathable": "breathable",
    "waterproof": "waterproof",
    "stretch": "stretch",
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

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

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
        state.override_scope = "none"
        state.continuation_requested = result.is_continuation
        # Keep an explicit route context across short clarification turns.  A
        # later explicit hint still replaces it; a goal-level override without
        # a hint starts from the prior route as a safe fallback.
        if result.route_hint is not None:
            state.route_hint = result.route_hint
        incoming = list(result.constraints)
        if result.is_override and self._is_new_goal(state, incoming):
            state.override_scope = "goal"
            if result.route_hint is None:
                state.route_hint = None
            self._begin_new_goal(state)
            state.query_history.clear()
        elif result.is_override:
            state.override_scope = "attribute"
            if result.replace_preferences:
                self._replace_prior_preferences(state)
                state.query_history.clear()

        if result.no_preference_attribute:
            state.no_preference_attributes.add(result.no_preference_attribute)
            self._reject_attribute(state, result.no_preference_attribute)

        for parsed in incoming:
            self._merge_constraint(state, parsed, turn)

        if result.has_substantive_evidence and message.strip():
            state.query_history.append(message.strip())
            state.query_history[:] = state.query_history[-4:]

        state.turn = turn
        if result.route_hint:
            state.route = result.route_hint
        self._sessions.move_to_end(session_id)
        self._last_updates[session_id] = (turn, message, state)
        return state

    @staticmethod
    def _is_new_goal(state: SessionState, incoming: list[ParsedConstraint]) -> bool:
        incoming_categories = {
            normalize_category_value(value)
            for item in incoming
            if item.attribute == "category" and item.is_hard
            for value in (item.alternatives or (item.value,))
        }
        if not incoming_categories:
            return False
        active_categories = {
            normalize_category_value(value)
            for item in state.active_constraints()
            if item.attribute == "category" and item.is_hard
            for value in (item.alternatives or (item.value,))
        }
        return not incoming_categories.issubset(active_categories)

    @staticmethod
    def _begin_new_goal(state: SessionState) -> None:
        state.intent_version += 1
        for index, old in enumerate(state.constraints):
            if old.status != "active":
                continue
            if old.source == "profile":
                state.constraints[index] = replace(old, intent_version=state.intent_version)
            else:
                state.constraints[index] = replace(old, status="superseded")
        state.asked_attributes.clear()
        state.pending_attribute = None
        state.no_preference_attributes.clear()

    @staticmethod
    def _replace_prior_preferences(state: SessionState) -> None:
        for index, old in enumerate(state.constraints):
            if (
                old.status == "active"
                and old.attribute != "category"
                and old.source != "profile"
            ):
                state.constraints[index] = replace(old, status="superseded")
        state.asked_attributes.clear()
        state.pending_attribute = None
        state.no_preference_attributes.clear()

    def _merge_constraint(
        self, state: SessionState, parsed: ParsedConstraint, turn: int
    ) -> None:
        active = state.active_constraints()
        if any(
            item.is_hard == parsed.is_hard and self._same_semantics(item, parsed)
            for item in active
        ):
            return

        for index, old in enumerate(state.constraints):
            if old.status != "active" or old.attribute != parsed.attribute:
                continue
            if old.source == "profile" and (
                old.value != parsed.value or parsed.is_hard
            ):
                state.constraints[index] = replace(old, status="rejected")
            elif parsed.is_hard and not self._same_semantics(old, parsed):
                state.constraints[index] = replace(old, status="superseded")

        state.constraints.append(
            self._to_constraint(parsed, turn, state.intent_version)
        )

    @staticmethod
    def _same_semantics(item: Constraint, parsed: ParsedConstraint) -> bool:
        if item.attribute != parsed.attribute or item.operator != parsed.operator:
            return False
        normalizer = (
            normalize_category_value
            if item.attribute == "category"
            else normalize_value
        )
        if normalizer(item.value) != normalizer(parsed.value):
            return False
        if tuple(sorted(normalizer(value) for value in item.values())) != tuple(
            sorted(normalizer(value) for value in parsed.alternatives or (parsed.value,))
        ):
            return False
        return normalizer(item.upper_value or "") == normalize_value(
            parsed.upper_value or ""
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
            operator=parsed.operator,
            upper_value=parsed.upper_value,
            alternatives=parsed.alternatives,
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
            mapped = _PROFILE_FEATURES.get(tag)
            if mapped:
                result.append(
                    ParsedConstraint("feature", mapped, 0.25, False, "profile")
                )
                continue
            parsed = parser.parse(tag, turn=0).constraints
            if parsed:
                for item in parsed:
                    result.append(
                        replace(item, confidence=0.25, is_hard=False, source="profile")
                    )
                continue
            result.append(ParsedConstraint("feature", tag, 0.25, False, "profile"))
        return result

    def _trim(self) -> None:
        while len(self._sessions) > self._max_sessions:
            session_id, _ = self._sessions.popitem(last=False)
            self._last_updates.pop(session_id, None)
