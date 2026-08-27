from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ConstraintSource = Literal["message", "profile", "clarification", "inferred"]
ConstraintStatus = Literal["active", "superseded", "rejected"]
ConstraintOperator = Literal["eq", "in", "not_in", "lte", "gte", "between"]
OverrideScope = Literal["none", "goal", "attribute"]
Route = Literal["buying", "browsing"]
RouteReason = Literal["explicit_browsing", "explicit_buying", "specificity_fallback"]


@dataclass(frozen=True)
class Constraint:
    attribute: str
    value: str
    confidence: float
    is_hard: bool
    source: ConstraintSource
    created_turn: int
    intent_version: int
    status: ConstraintStatus = "active"
    operator: ConstraintOperator = "eq"
    upper_value: str | None = None
    alternatives: tuple[str, ...] = ()

    def values(self) -> tuple[str, ...]:
        return self.alternatives or (self.value,)


@dataclass
class SessionState:
    session_id: str
    turn: int = 0
    route: Route = "browsing"
    intent_version: int = 1
    constraints: list[Constraint] = field(default_factory=list)
    asked_attributes: list[str] = field(default_factory=list)
    pending_attribute: str | None = None
    query_history: list[str] = field(default_factory=list)
    no_preference_attributes: set[str] = field(default_factory=set)
    previous_recommendations: list[str] = field(default_factory=list)
    candidate_count: int = 0
    continuation_requested: bool = False
    override_scope: OverrideScope = "none"
    route_hint: Route | None = None

    def active_constraints(self) -> list[Constraint]:
        return [
            item
            for item in self.constraints
            if item.status == "active"
            and (
                item.intent_version == self.intent_version
                or item.attribute == "category"
            )
        ]


@dataclass(frozen=True)
class RetrievalPlan:
    route: Route
    query_text: str
    hard_filters: dict[str, tuple[str, ...]] = field(default_factory=dict)
    soft_preferences: dict[str, tuple[str, ...]] = field(default_factory=dict)
    excluded_values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    candidate_limit: int = 500
    source_weights: tuple[tuple[str, float], ...] = ()
    is_override: bool = False
    hard_constraints: tuple[Constraint, ...] = ()
    route_reason: RouteReason = "specificity_fallback"

    def effective_hard_constraints(self) -> tuple[Constraint, ...]:
        """Return explicit constraints, or preserve legacy hard-filter semantics."""
        if self.hard_constraints:
            return self.hard_constraints

        constraints: list[Constraint] = []
        for attribute, values in self.hard_filters.items():
            if not values:
                continue
            if attribute == "budget":
                value = max(values, key=_numeric_sort_key)
                operator: ConstraintOperator = "lte"
                alternatives: tuple[str, ...] = ()
            else:
                value = values[0]
                operator = "eq" if len(values) == 1 else "in"
                alternatives = values if len(values) > 1 else ()
            constraints.append(
                Constraint(
                    attribute=attribute,
                    value=value,
                    confidence=1.0,
                    is_hard=True,
                    source="inferred",
                    created_turn=0,
                    intent_version=0,
                    operator=operator,
                    alternatives=alternatives,
                )
            )
        return tuple(constraints)


@dataclass
class Candidate:
    parent_asin: str
    product: dict[str, object] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    violations: tuple[str, ...] = ()
    relaxed: bool = False
    source_ranks: dict[str, int] = field(default_factory=dict)
    pre_rank: int | None = None


def _numeric_sort_key(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("-inf")


@dataclass(frozen=True)
class QuestionDecision:
    ask_attribute: str | None
    utility: float = 0.0
