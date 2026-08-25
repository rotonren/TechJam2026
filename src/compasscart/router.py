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
        for item in state.active_constraints():
            target = hard if item.is_hard and item.confidence >= 0.9 else soft
            if item.value not in target[item.attribute]:
                target[item.attribute].append(item.value)

        hard_count = sum(len(values) for values in hard.values())
        explicit_fields = {"budget", "size", "material"}.intersection(hard)
        query_specificity = min(len(terms(query_text)), 8) / 8.0
        specificity = hard_count + len(explicit_fields) + query_specificity
        route = "buying" if specificity >= 1.25 else state.route

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
            candidate_limit=self.config.candidate_limit,
            source_weights=source_weights,
            is_override=is_override,
        )
