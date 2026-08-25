from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ConstraintSource = Literal["message", "profile", "clarification", "inferred"]
ConstraintStatus = Literal["active", "superseded", "rejected"]
Route = Literal["buying", "browsing"]


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


@dataclass
class SessionState:
    session_id: str
    turn: int = 0
    route: Route = "browsing"
    intent_version: int = 1
    constraints: list[Constraint] = field(default_factory=list)
    asked_attributes: list[str] = field(default_factory=list)
    no_preference_attributes: set[str] = field(default_factory=set)
    previous_recommendations: list[str] = field(default_factory=list)
    candidate_count: int = 0

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


@dataclass
class Candidate:
    parent_asin: str
    product: dict[str, object] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)
    score: float = 0.0


@dataclass(frozen=True)
class QuestionDecision:
    ask_attribute: str | None
    utility: float = 0.0
