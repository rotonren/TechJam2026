from __future__ import annotations

import time
from collections import OrderedDict

from .catalog import CatalogIndex
from .models import Candidate, Constraint, SessionState
from .normalization import terms


class ConstraintRanker:
    def __init__(
        self,
        catalog: CatalogIndex,
        *,
        fusion_weight: float = 0.0,
        attribute_weight: float = 0.0,
        consensus_bonus: float = 0.0,
        boundary_bonus: float = 0.0,
        mmr_lambda: float = 0.85,
        adaptive_browsing_mmr: bool = False,
    ) -> None:
        if fusion_weight not in {0.0, 0.10, 0.15}:
            raise ValueError("fusion_weight must be one of 0.0, 0.10, or 0.15")
        if attribute_weight not in {0.0, 0.05, 0.10}:
            raise ValueError("attribute_weight must be one of 0.0, 0.05, or 0.10")
        if consensus_bonus not in {0.0, 0.025, 0.05}:
            raise ValueError("consensus_bonus must be one of 0.0, 0.025, or 0.05")
        if boundary_bonus not in {0.0, 0.025}:
            raise ValueError("boundary_bonus must be one of 0.0 or 0.025")
        if fusion_weight + attribute_weight > 0.40:
            raise ValueError("fusion and attribute weights must not exceed 0.40")
        if mmr_lambda != 0.85:
            raise ValueError("mmr_lambda must be 0.85")
        if not isinstance(adaptive_browsing_mmr, bool):
            raise TypeError("adaptive_browsing_mmr must be a bool")
        self.catalog = catalog
        self.fusion_weight = fusion_weight
        self.attribute_weight = attribute_weight
        self.consensus_bonus = min(consensus_bonus, 0.05)
        self.boundary_bonus = min(boundary_bonus, 0.025)
        self.mmr_lambda = mmr_lambda
        self.adaptive_browsing_mmr = adaptive_browsing_mmr
        self._diversity_cache: OrderedDict[str, frozenset[str]] = OrderedDict()

    def rank(
        self,
        candidates: list[Candidate],
        state: SessionState,
        *,
        top_k: int | None = None,
        deadline: float | None = None,
        diagnostics: list[str] | None = None,
    ) -> list[Candidate]:
        if not candidates:
            return []
        lexical = self._normalized_source(
            candidates,
            "lexical",
            allow_score_fallback=self.fusion_weight == 0.0,
        )
        dense = self._normalized_source(candidates, "dense")
        attribute = self._normalized_source(candidates, "attribute")
        fusion = self._normalized_scores(candidates)
        source_weight = (
            0.40 - self.fusion_weight - self.attribute_weight
        ) / 2.0
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
            consensus = self._has_consensus(candidate)
            consensus_evidence = self.consensus_bonus if consensus else 0.0
            boundary_evidence = (
                self.boundary_bonus
                if consensus
                and candidate.pre_rank is not None
                and candidate.pre_rank <= 10
                and not candidate.relaxed
                and not conflict
                else 0.0
            )
            score = (
                0.30 * hard_coverage
                + source_weight * lexical[identifier]
                + source_weight * dense[identifier]
                + self.fusion_weight * fusion[identifier]
                + self.attribute_weight * attribute[identifier]
                + 0.10 * category_match
                + 0.10 * soft_coverage
                + 0.05 * profile_affinity
                + 0.05 * self.catalog.quality.get(identifier, 0.0)
                + consensus_evidence
                + boundary_evidence
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
                    source_ranks=dict(candidate.source_ranks),
                    pre_rank=candidate.pre_rank,
                )
            )

        scored.sort(key=lambda item: (item.relaxed, -item.score, item.parent_asin))
        limit = len(scored) if top_k is None else min(max(top_k, 0), len(scored))
        if state.route == "browsing" and limit:
            if deadline is None or time.perf_counter() < deadline:
                exact = [item for item in scored if not item.relaxed]
                relaxed = [item for item in scored if item.relaxed]
                return (
                    self._browsing_order(exact) + self._browsing_order(relaxed)
                )[:limit]
            if diagnostics is not None and "mmr_budget" not in diagnostics:
                diagnostics.append("mmr_budget")
        return scored[:limit]

    @staticmethod
    def _has_consensus(candidate: Candidate) -> bool:
        sources = {
            source
            for source, rank in candidate.source_ranks.items()
            if source != "profile" and isinstance(rank, int) and rank > 0
        }
        return len(sources) >= 2 and bool(sources & {"lexical", "attribute"})

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

    def _browsing_order(self, candidates: list[Candidate]) -> list[Candidate]:
        if not self.adaptive_browsing_mmr:
            return self._diverse_order(candidates)
        if not self._adaptive_mmr_eligible(candidates):
            return list(candidates)
        return self._diverse_order(candidates)

    def _adaptive_mmr_eligible(self, candidates: list[Candidate]) -> bool:
        if len(candidates) < 11:
            return False
        maximum = max(item.score for item in candidates)
        minimum = min(item.score for item in candidates)
        scale = maximum - minimum
        normalized_gap = (
            abs(candidates[9].score - candidates[10].score) / scale
            if scale > 0
            else 0.0
        )
        if normalized_gap > 0.025:
            return False
        similar_pairs = 0
        first_ten = candidates[:10]
        for left_index, left in enumerate(first_ten):
            for right in first_ten[left_index + 1 :]:
                if self._similarity(left.parent_asin, right.parent_asin) >= 0.60:
                    similar_pairs += 1
                    if similar_pairs >= 3:
                        return True
        return False

    @staticmethod
    def _normalized_source(
        candidates: list[Candidate],
        source: str,
        *,
        allow_score_fallback: bool = True,
    ) -> dict[str, float]:
        values = {
            item.parent_asin: max(item.source_scores.get(source, 0.0), 0.0)
            for item in candidates
        }
        if source == "lexical" and allow_score_fallback and not any(values.values()):
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

    @staticmethod
    def _normalized_scores(candidates: list[Candidate]) -> dict[str, float]:
        values = {
            item.parent_asin: max(float(item.score), 0.0) for item in candidates
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

    def _diversity_terms(self, identifier: str) -> frozenset[str]:
        cached = self._diversity_cache.pop(identifier, None)
        if cached is not None:
            self._diversity_cache[identifier] = cached
            return cached
        attributes = self.catalog.attributes.get(identifier, {})
        values = (
            *attributes.get("category", ()),
            *attributes.get("material", ()),
            *attributes.get("style", ()),
            *attributes.get("use_case", ()),
        )
        result = frozenset(token for value in values for token in terms(value))
        self._diversity_cache[identifier] = result
        if len(self._diversity_cache) > 4_096:
            self._diversity_cache.popitem(last=False)
        return result
