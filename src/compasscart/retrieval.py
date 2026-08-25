from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Protocol

from .catalog import CatalogIndex
from .models import Candidate, RetrievalPlan, SessionState


class DenseRetriever(Protocol):
    @property
    def available(self) -> bool: ...

    def search(self, text: str, limit: int) -> list[Candidate]: ...


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    weights: Mapping[str, float],
    k: int = 60,
) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for source, identifiers in rankings.items():
        weight = max(float(weights.get(source, 0.0)), 0.0)
        seen: set[str] = set()
        for rank, identifier in enumerate(identifiers, start=1):
            if identifier in seen:
                continue
            seen.add(identifier)
            scores[identifier] += weight / (k + rank)
    return sorted(scores, key=lambda identifier: (-scores[identifier], identifier))


class HybridRetriever:
    def __init__(
        self,
        catalog: CatalogIndex,
        dense: DenseRetriever | None = None,
        *,
        rrf_k: int = 60,
    ) -> None:
        self.catalog = catalog
        self.dense = dense
        self.rrf_k = rrf_k

    def retrieve(
        self, plan: RetrievalPlan, state: SessionState | None = None
    ) -> list[Candidate]:
        component_limit = min(150, plan.candidate_limit)
        lexical = self.catalog.search_lexical(plan, limit=component_limit)
        attribute_ids = self._attribute_candidates(plan, component_limit)
        profile_ids = self._profile_candidates(plan, component_limit)
        dense = self._dense_candidates(plan.query_text, component_limit)

        rankings = {
            "lexical": [item.parent_asin for item in lexical],
            "attribute": attribute_ids,
            "profile": profile_ids,
            "dense": [item.parent_asin for item in dense],
        }
        weights = dict(plan.source_weights) or self._default_weights(plan.route)
        fused_ids = reciprocal_rank_fusion(rankings, weights=weights, k=self.rrf_k)
        fused_ids = [item for item in fused_ids if item in self.catalog.valid_ids]

        desired = min(10, len(self.catalog.valid_ids), plan.candidate_limit)
        if len(fused_ids) < desired:
            for identifier in self._fallback_ids(plan):
                if identifier not in fused_ids:
                    fused_ids.append(identifier)
                if len(fused_ids) >= desired:
                    break

        fused_ids = fused_ids[: plan.candidate_limit]
        source_ranks = {
            source: {identifier: rank for rank, identifier in enumerate(ids, start=1)}
            for source, ids in rankings.items()
        }
        candidates: list[Candidate] = []
        for identifier in fused_ids:
            contributions = {
                source: weights.get(source, 0.0) / (self.rrf_k + ranks[identifier])
                for source, ranks in source_ranks.items()
                if identifier in ranks and weights.get(source, 0.0) > 0
            }
            candidates.append(
                Candidate(
                    parent_asin=identifier,
                    product=self.catalog.product(identifier),
                    source_scores=contributions,
                    score=sum(contributions.values()),
                )
            )
        return candidates

    def _attribute_candidates(self, plan: RetrievalPlan, limit: int) -> list[str]:
        groups: list[set[str]] = []
        for attribute, values in plan.hard_filters.items():
            if attribute == "budget":
                budgets = [float(value) for value in values]
                groups.append(
                    {
                        identifier
                        for identifier, product in self.catalog.products.items()
                        if self._price(product) <= max(budgets)
                    }
                )
                continue
            matches: set[str] = set()
            for value in values:
                matches.update(self.catalog.attribute_ids(attribute, value))
            groups.append(matches)
        if not groups:
            return []
        identifiers = set.intersection(*groups)
        return sorted(
            identifiers,
            key=lambda item: (-self.catalog.quality[item], item),
        )[:limit]

    def _profile_candidates(self, plan: RetrievalPlan, limit: int) -> list[str]:
        scores: dict[str, int] = defaultdict(int)
        for attribute, values in plan.soft_preferences.items():
            for value in values:
                for identifier in self.catalog.attribute_ids(attribute, value):
                    scores[identifier] += 1
        return sorted(
            scores,
            key=lambda item: (-scores[item], -self.catalog.quality[item], item),
        )[:limit]

    def _dense_candidates(self, text: str, limit: int) -> list[Candidate]:
        if self.dense is None or not self.dense.available:
            return []
        try:
            return self.dense.search(text, limit)
        except Exception:  # noqa: BLE001 - optional backend cannot break retrieval.
            return []

    def _fallback_ids(self, plan: RetrievalPlan) -> list[str]:
        category_ids: set[str] = set()
        for value in plan.hard_filters.get("category", ()):
            category_ids.update(self.catalog.attribute_ids("category", value))
        ordered_category = sorted(
            category_ids, key=lambda item: (-self.catalog.quality[item], item)
        )
        return ordered_category + self.catalog.popular_ids(plan.candidate_limit)

    @staticmethod
    def _price(product: dict[str, object]) -> float:
        try:
            return float(product.get("price", float("inf")))
        except (TypeError, ValueError):
            return float("inf")

    @staticmethod
    def _default_weights(route: str) -> dict[str, float]:
        if route == "buying":
            return {"attribute": 0.45, "lexical": 0.35, "dense": 0.20}
        return {"dense": 0.45, "lexical": 0.30, "profile": 0.25}
