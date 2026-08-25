from __future__ import annotations

from collections import defaultdict

from .config import RuntimeConfig
from .models import RetrievalPlan, SessionState
from .normalization import terms


class RoutePlanner:
    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()

    def build_plan(
        self,
        state: SessionState,
        query_text: str,
        *,
        is_override: bool = False,
    ) -> RetrievalPlan:
        hard: dict[str, list[str]] = defaultdict(list)
        soft: dict[str, list[str]] = defaultdict(list)
        excluded: dict[str, list[str]] = defaultdict(list)
        active_constraints = state.active_constraints()
        hard_constraints = tuple(
            item
            for item in active_constraints
            if item.is_hard and item.confidence >= 0.9
        )
        for item in active_constraints:
            target = hard if item.is_hard and item.confidence >= 0.9 else soft
            if item.operator == "not_in":
                for value in item.values():
                    if value not in excluded[item.attribute]:
                        excluded[item.attribute].append(value)
                continue
            for value in item.values():
                if value not in target[item.attribute]:
                    target[item.attribute].append(value)

        hard_count = sum(len(values) for values in hard.values())
        explicit_fields = {"budget", "size", "material"}.intersection(hard)
        query_specificity = min(len(terms(query_text)), 8) / 8.0
        specificity = hard_count + len(explicit_fields) + query_specificity
        if state.route_hint == "browsing":
            route = "browsing"
            route_reason = "explicit_browsing"
        elif state.route_hint == "buying":
            route = "buying"
            route_reason = "explicit_buying"
        else:
            route = "buying" if specificity >= 1.25 else state.route
            route_reason = "specificity_fallback"

        if is_override:
            source_weights = self.config.override_route_weights
        elif route == "buying":
            source_weights = self.config.buying_route_weights
        else:
            source_weights = self.config.browsing_route_weights

        return RetrievalPlan(
            route=route,
            query_text=query_text,
            hard_filters={key: tuple(values) for key, values in hard.items()},
            soft_preferences={key: tuple(values) for key, values in soft.items()},
            excluded_values={key: tuple(values) for key, values in excluded.items()},
            candidate_limit=self.config.candidate_limit,
            source_weights=source_weights,
            is_override=is_override,
            hard_constraints=hard_constraints,
            route_reason=route_reason,
        )
