from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from .models import Candidate, QuestionDecision, SessionState
from .normalization import extract_attributes, normalize_value

_ATTRIBUTES = (
    "material",
    "color",
    "size",
    "style",
    "budget",
    "feature",
    "use_case",
)
_HARD_REQUIREMENT_ATTRIBUTES = frozenset({"material", "size", "budget"})
_RESPONSE_LIKELIHOOD = {
    "category": 0.95,
    "material": 0.90,
    "color": 0.90,
    "size": 0.85,
    "style": 0.80,
    "brand": 0.65,
    "budget": 0.90,
    "feature": 0.70,
    "use_case": 0.85,
}

SupportCallback = Callable[[str, str], bool]


def _always_supported(_attribute: str, _value: str) -> bool:
    return True


def _never_supported(_attribute: str, _value: str) -> bool:
    return False


class QuestionPolicy:
    def __init__(
        self,
        attribute_lookup=None,
        parser_support: SupportCallback | None = None,
        retrieval_support: SupportCallback | None = None,
    ) -> None:
        self.attribute_lookup = attribute_lookup
        if parser_support is None and retrieval_support is None:
            self.parser_support = _always_supported
            self.retrieval_support = _always_supported
        else:
            self.parser_support = (
                parser_support if parser_support is not None else _never_supported
            )
            self.retrieval_support = (
                retrieval_support
                if retrieval_support is not None
                else _never_supported
            )

    def choose(
        self, candidates: list[Candidate], state: SessionState
    ) -> QuestionDecision:
        if state.continuation_requested or len(candidates) <= 10 or state.turn >= 10:
            return QuestionDecision(None)

        probabilities = self._probabilities(candidates)
        candidate_attributes = [self._attributes(candidate) for candidate in candidates]
        candidate_ids = [candidate.parent_asin for candidate in candidates]
        blocked = set(state.asked_attributes) | state.no_preference_attributes
        # An attribute explicitly constrained by the user is already answered;
        # asking for it again would contradict the turn's hard semantics (for
        # example, asking for material after "black leather belt").
        blocked.update(
            item.attribute
            for item in state.active_constraints()
            if item.is_hard
        )
        utilities = {
            attribute: self._utility(
                attribute,
                candidate_attributes,
                probabilities,
                state.turn,
                candidate_ids,
            )
            for attribute in _ATTRIBUTES
            if attribute not in blocked
        }
        if not utilities:
            return QuestionDecision(None)

        prefer_hard_requirement = (
            state.route == "buying" or state.override_scope != "none"
        )
        attribute, utility = max(
            utilities.items(),
            key=lambda item: (
                item[1],
                int(
                    prefer_hard_requirement
                    and item[0] in _HARD_REQUIREMENT_ATTRIBUTES
                ),
                -_ATTRIBUTES.index(item[0]),
            ),
        )
        threshold = 0.15 if state.turn == 9 else 0.10 if state.turn == 8 else 0.0
        if utility > threshold:
            return QuestionDecision(attribute, utility)

        return QuestionDecision(None)

    def _attributes(self, candidate: Candidate) -> dict[str, tuple[str, ...]]:
        if self.attribute_lookup is not None:
            attributes = self.attribute_lookup.get(candidate.parent_asin)
            if attributes is not None:
                return attributes
        return extract_attributes(candidate.product)

    def _utility(
        self,
        attribute: str,
        candidate_attributes: list[dict[str, tuple[str, ...]]],
        probabilities: list[float],
        turn: int,
        candidate_ids: list[str] | None = None,
    ) -> float:
        partitions: dict[str, list[int]] = defaultdict(list)
        for index, attributes in enumerate(candidate_attributes):
            values = attributes.get(attribute, ())
            for value in dict.fromkeys(normalize_value(item) for item in values):
                if value:
                    partitions[value].append(index)

        parser_values = tuple(
            value
            for value in partitions
            if self.parser_support(attribute, value)
        )
        retrieval_values = tuple(
            value
            for value in partitions
            if self.retrieval_support(attribute, value)
        )
        parser_value_set = set(parser_values)
        retrieval_value_set = set(retrieval_values)
        identifiers = candidate_ids or [str(index) for index in range(len(probabilities))]
        meaningful_partitions = {
            value: indices
            for value, indices in partitions.items()
            if value in parser_value_set
            and value in retrieval_value_set
            and (
                len({identifiers[index] for index in indices}) >= 2
                or sum(probabilities[index] for index in indices) >= 0.05
            )
        }
        if len(meaningful_partitions) < 2:
            return 0.0

        covered = {
            index
            for indices in meaningful_partitions.values()
            for index in indices
        }
        coverage_mass = sum(probabilities[index] for index in sorted(covered))
        if coverage_mass <= 0:
            return 0.0
        current_known_top10 = sum(
            probabilities[index]
            for index in range(min(10, len(candidate_attributes)))
            if index in covered
        )
        current_conditional = current_known_top10 / coverage_mass
        post_answer_indices: set[int] = set()
        for indices in meaningful_partitions.values():
            ranked_unique = list(dict.fromkeys(indices))
            post_answer_indices.update(ranked_unique[:10])
        post_answer_mass = sum(
            probabilities[index] for index in sorted(post_answer_indices)
        )
        post_conditional = min(post_answer_mass / coverage_mass, 1.0)
        candidate_reduction = max(post_conditional - current_conditional, 0.0)
        parser_covered = {
            index for value in parser_values for index in partitions[value]
        }
        retrieval_covered = {
            index for value in retrieval_values for index in partitions[value]
        }
        raw_covered = {
            index for indices in partitions.values() for index in indices
        }
        raw_partition_mass = sum(
            probabilities[index] for index in sorted(raw_covered)
        )
        parser_support = (
            sum(probabilities[index] for index in sorted(parser_covered))
            / raw_partition_mass
        )
        retrieval_support = (
            sum(probabilities[index] for index in sorted(retrieval_covered))
            / raw_partition_mass
        )
        answerability = _RESPONSE_LIKELIHOOD[attribute]
        remaining_turn_value = max((11 - max(turn, 1)) / 10.0, 0.1)
        no_preference_risk = 0.05 * (1.0 - answerability)
        return (
            candidate_reduction
            * answerability
            * parser_support
            * retrieval_support
            * remaining_turn_value
            - no_preference_risk
        )

    @staticmethod
    def _probabilities(candidates: list[Candidate]) -> list[float]:
        scores = [float(candidate.score) for candidate in candidates]
        minimum = min(scores, default=0.0)
        if minimum <= 0:
            scores = [score - minimum + 1e-9 for score in scores]
        total = sum(scores)
        if total <= 0:
            return [1.0 / len(candidates)] * len(candidates)
        return [score / total for score in scores]
