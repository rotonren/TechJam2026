from __future__ import annotations

from .catalog import CatalogIndex
from .models import Candidate, Constraint, SessionState
from .normalization import terms


class ConstraintRanker:
    def __init__(self, catalog: CatalogIndex, *, mmr_lambda: float = 0.85) -> None:
        self.catalog = catalog
        self.mmr_lambda = mmr_lambda

    def rank(
        self,
        candidates: list[Candidate],
        state: SessionState,
        *,
        top_k: int | None = None,
    ) -> list[Candidate]:
        if not candidates:
            return []
        lexical = self._normalized_source(candidates, "lexical")
        dense = self._normalized_source(candidates, "dense")
        hard = [item for item in state.active_constraints() if item.is_hard]
        soft = [
            item
            for item in state.active_constraints()
            if not item.is_hard and item.source != "profile"
        ]
        profile = [
            item for item in state.active_constraints() if item.source == "profile"
        ]

        scored: list[Candidate] = []
        for candidate in candidates:
            identifier = candidate.parent_asin
            hard_coverage = self._coverage(identifier, hard)
            soft_coverage = self._coverage(identifier, soft)
            profile_affinity = self._coverage(identifier, profile)
            category_constraints = [
                item for item in hard + soft if item.attribute == "category"
            ]
            category_match = self._coverage(identifier, category_constraints)
            conflict = float(any(self._conflicts(identifier, item) for item in hard))
            score = (
                0.30 * hard_coverage
                + 0.20 * lexical[identifier]
                + 0.20 * dense[identifier]
                + 0.10 * category_match
                + 0.10 * soft_coverage
                + 0.05 * profile_affinity
                + 0.05 * self.catalog.quality.get(identifier, 0.0)
                - 0.60 * conflict
            )
            scored.append(
                Candidate(
                    parent_asin=identifier,
                    product=candidate.product,
                    source_scores=dict(candidate.source_scores),
                    score=score,
                    violations=candidate.violations,
                    relaxed=candidate.relaxed,
                )
            )

        scored.sort(key=lambda item: (item.relaxed, -item.score, item.parent_asin))
        limit = len(scored) if top_k is None else min(max(top_k, 0), len(scored))
        if state.route == "browsing" and limit:
            exact = [item for item in scored if not item.relaxed]
            relaxed = [item for item in scored if item.relaxed]
            return (self._diverse_order(exact) + self._diverse_order(relaxed))[:limit]
        return scored[:limit]

    def _coverage(self, identifier: str, constraints: list[Constraint]) -> float:
        if not constraints:
            return 0.0
        return sum(self._matches(identifier, item) for item in constraints) / len(
            constraints
        )

    def _matches(self, identifier: str, constraint: Constraint) -> bool:
        return self.catalog.matches(identifier, constraint)

    def _conflicts(self, identifier: str, constraint: Constraint) -> bool:
        return not self._matches(identifier, constraint)

    def _diverse_order(self, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        diverse = self._mmr(candidates, min(len(candidates), 10))
        selected = {item.parent_asin for item in diverse}
        return diverse + [item for item in candidates if item.parent_asin not in selected]

    @staticmethod
    def _normalized_source(
        candidates: list[Candidate], source: str
    ) -> dict[str, float]:
        values = {
            item.parent_asin: max(item.source_scores.get(source, 0.0), 0.0)
            for item in candidates
        }
        if source == "lexical" and not any(values.values()):
            fusion = {item.parent_asin: max(item.score, 0.0) for item in candidates}
            minimum = min(fusion.values(), default=0.0)
            maximum = max(fusion.values(), default=0.0)
            scale = maximum - minimum
            if scale > 0:
                return {
                    identifier: (value - minimum) / scale
                    for identifier, value in fusion.items()
                }
        maximum = max(values.values(), default=0.0)
        if maximum <= 0:
            return {identifier: 0.0 for identifier in values}
        return {identifier: value / maximum for identifier, value in values.items()}

    def _mmr(self, candidates: list[Candidate], limit: int) -> list[Candidate]:
        selected: list[Candidate] = []
        remaining = list(candidates)
        maximum = max((item.score for item in candidates), default=1.0)
        minimum = min((item.score for item in candidates), default=0.0)
        scale = maximum - minimum or 1.0
        while remaining and len(selected) < limit:
            best = max(
                remaining,
                key=lambda item: (
                    self.mmr_lambda * ((item.score - minimum) / scale)
                    - (1.0 - self.mmr_lambda)
                    * max(
                        (
                            self._similarity(item.parent_asin, prior.parent_asin)
                            for prior in selected
                        ),
                        default=0.0,
                    ),
                    -ord(item.parent_asin[0]) if item.parent_asin else 0,
                ),
            )
            selected.append(best)
            remaining.remove(best)
        return selected

    def _similarity(self, left: str, right: str) -> float:
        left_terms = self._diversity_terms(left)
        right_terms = self._diversity_terms(right)
        union = left_terms | right_terms
        if not union:
            return 0.0
        return len(left_terms & right_terms) / len(union)

    def _diversity_terms(self, identifier: str) -> set[str]:
        attributes = self.catalog.attributes.get(identifier, {})
        values = (
            *attributes.get("category", ()),
            *attributes.get("material", ()),
            *attributes.get("style", ()),
            *attributes.get("use_case", ()),
        )
        return {token for value in values for token in terms(value)}
