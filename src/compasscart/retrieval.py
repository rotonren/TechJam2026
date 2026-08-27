from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence

from .catalog import CatalogIndex
from .dense import DenseBackend
from .models import Candidate, Constraint, RetrievalPlan, SessionState
from .normalization import normalize_value


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
        dense: DenseBackend | None = None,
        *,
        rrf_k: int = 60,
    ) -> None:
        self.catalog = catalog
        self.dense = dense
        self.rrf_k = rrf_k

    def retrieve(
        self,
        plan: RetrievalPlan,
        state: SessionState | None = None,
        *,
        exclude_ids: Collection[str] | None = None,
    ) -> list[Candidate]:
        # The caller normally supplies the exclusion set explicitly so ordinary
        # refinements can reuse a previous result.  Keep the state-based fallback
        # for direct component callers that set the continuation flag themselves.
        excluded = set(exclude_ids or ())
        if state is not None and state.continuation_requested and exclude_ids is None:
            excluded.update(state.previous_recommendations)
        component_limit = min(150, plan.candidate_limit)
        lexical = self.catalog.search_lexical(plan, limit=component_limit)
        attribute_ids = self._attribute_candidates(plan, component_limit)
        profile_ids = self._profile_candidates(plan, component_limit)
        dense = self._dense_candidates(plan.query_text, component_limit)

        rankings = {
            "lexical": self._exact_ids(
                [item.parent_asin for item in lexical], plan, excluded
            ),
            "attribute": self._exact_ids(attribute_ids, plan, excluded),
            "profile": self._exact_ids(profile_ids, plan, excluded),
            "dense": self._exact_ids(
                [item.parent_asin for item in dense], plan, excluded
            ),
        }
        weights = dict(plan.source_weights) or self._default_weights(plan.route)
        fused_ids = reciprocal_rank_fusion(rankings, weights=weights, k=self.rrf_k)
        fused_ids = [item for item in fused_ids if item in self.catalog.valid_ids]

        desired = min(10, len(self.catalog.valid_ids), plan.candidate_limit)
        fallback_ids: list[str] | None = None

        def fallback() -> list[str]:
            nonlocal fallback_ids
            if fallback_ids is None:
                fallback_ids = self._fallback_ids(plan)
            return fallback_ids

        if len(fused_ids) < desired:
            for identifier in self._exact_ids(fallback(), plan, excluded):
                if identifier not in fused_ids:
                    fused_ids.append(identifier)
                if len(fused_ids) >= desired:
                    break

        exact_ids = fused_ids[: plan.candidate_limit]
        relaxed_ids: list[str] = []
        if len(exact_ids) < desired:
            for identifier in fallback():
                if (
                    identifier in excluded
                    or identifier in exact_ids
                    or identifier in relaxed_ids
                ):
                    continue
                if self._violations(identifier, plan):
                    relaxed_ids.append(identifier)
                if len(exact_ids) + len(relaxed_ids) >= desired:
                    break

        # If a caller asks for more pages than the catalog contains, the
        # exclusion set can consume every eligible ID.  Preserve the official
        # valid-ID guarantee as a last resort, after all unshown IDs were tried.
        if not exact_ids and not relaxed_ids and excluded:
            for identifier in fallback():
                if identifier not in self.catalog.valid_ids:
                    continue
                relaxed_ids.append(identifier)
                if len(relaxed_ids) >= desired:
                    break
        source_ranks = {
            source: {identifier: rank for rank, identifier in enumerate(ids, start=1)}
            for source, ids in rankings.items()
        }
        candidates: list[Candidate] = []
        for identifier in exact_ids:
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
        for identifier in relaxed_ids:
            candidates.append(
                Candidate(
                    parent_asin=identifier,
                    product=self.catalog.product(identifier),
                    violations=self._violations(identifier, plan),
                    relaxed=True,
                )
            )
        return candidates

    def _attribute_candidates(self, plan: RetrievalPlan, limit: int) -> list[str]:
        constraints = plan.effective_hard_constraints()
        if constraints:
            groups = [self._ids_for_constraint(item) for item in constraints]
            if not groups or any(not group for group in groups):
                return []
            identifiers = set.intersection(*groups)
            return sorted(
                identifiers,
                key=lambda item: (-self.catalog.quality[item], item),
            )[:limit]
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

    def _ids_for_constraint(self, constraint: Constraint) -> set[str]:
        if constraint.attribute == "budget":
            return {
                identifier
                for identifier, product in self.catalog.products.items()
                if self._price_matches(product.get("price"), constraint)
            }

        if constraint.attribute == "category":
            matched = set().union(
                *(self.catalog.category_ids(value) for value in constraint.values())
            )
            if constraint.operator == "not_in":
                postings = self.catalog.category_term_inverted.values()
                present = (
                    set().union(*postings)
                    if self.catalog.category_term_inverted
                    else set()
                )
                return present - matched
            if constraint.operator in {"eq", "in"}:
                return matched
            return set()

        attribute = constraint.attribute
        normalizer = normalize_value
        inverted = self.catalog.attribute_inverted.get(attribute, {})

        def ids_for(value: str) -> set[str]:
            wanted = normalizer(value)
            return {
                identifier
                for catalog_value, identifiers in inverted.items()
                if normalizer(catalog_value) == wanted
                for identifier in identifiers
            }

        matched = set().union(*(ids_for(value) for value in constraint.values()))
        if constraint.operator == "not_in":
            present = set().union(*inverted.values()) if inverted else set()
            return present - matched
        if constraint.operator in {"eq", "in"}:
            return matched
        return set()

    @classmethod
    def _price_matches(cls, raw_price: object, constraint: Constraint) -> bool:
        price = cls._finite_positive(raw_price)
        lower = cls._finite_positive(constraint.value)
        if price is None or lower is None:
            return False
        if constraint.operator == "lte":
            return price <= lower
        if constraint.operator == "gte":
            return price >= lower
        if constraint.operator == "between":
            upper = cls._finite_positive(constraint.upper_value)
            return upper is not None and lower <= price <= upper
        if constraint.operator in {"eq", "in"}:
            values = {
                number
                for value in constraint.values()
                if (number := cls._finite_positive(value)) is not None
            }
            return price in values
        if constraint.operator == "not_in":
            values = {
                number
                for value in constraint.values()
                if (number := cls._finite_positive(value)) is not None
            }
            return price not in values
        return False

    @staticmethod
    def _finite_positive(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 and math.isfinite(number) else None

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

    def _violations(self, identifier: str, plan: RetrievalPlan) -> tuple[str, ...]:
        constraints = plan.effective_hard_constraints()
        if not constraints:
            return ()
        return self.catalog.violations(identifier, constraints)

    def _exact_ids(
        self,
        identifiers: Iterable[str],
        plan: RetrievalPlan,
        excluded: Collection[str] = (),
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for identifier in identifiers:
            if (
                identifier not in seen
                and identifier not in excluded
                and identifier in self.catalog.valid_ids
                and not self._violations(identifier, plan)
            ):
                seen.add(identifier)
                result.append(identifier)
        return result

    def _dense_candidates(self, text: str, limit: int) -> list[Candidate]:
        if self.dense is None or not self.dense.available:
            return []
        try:
            return self.dense.search(text, limit)
        except Exception:  # noqa: BLE001 - optional backend cannot break retrieval.
            return []

    def _fallback_ids(self, plan: RetrievalPlan) -> list[str]:
        category_groups = [
            self._ids_for_constraint(constraint)
            for constraint in plan.effective_hard_constraints()
            if constraint.attribute == "category"
            and constraint.operator in {"eq", "in"}
        ]
        category_ids = set.intersection(*category_groups) if category_groups else set()
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
