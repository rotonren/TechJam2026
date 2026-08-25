from __future__ import annotations

from collections import defaultdict

from .models import Candidate, QuestionDecision, SessionState
from .normalization import extract_attributes

_ATTRIBUTES = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
)
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


class QuestionPolicy:
    def choose(
        self, candidates: list[Candidate], state: SessionState
    ) -> QuestionDecision:
        if len(candidates) <= 10:
            return QuestionDecision(None)

        probabilities = self._probabilities(candidates)
        candidate_attributes = [
            extract_attributes(candidate.product) for candidate in candidates
        ]
        blocked = set(state.asked_attributes) | state.no_preference_attributes
        utilities = {
            attribute: self._utility(
                attribute, candidate_attributes, probabilities, state.turn
            )
            for attribute in _ATTRIBUTES
            if attribute not in blocked
        }
        if not utilities:
            return QuestionDecision(None)

        attribute, utility = max(
            utilities.items(), key=lambda item: (item[1], -_ATTRIBUTES.index(item[0]))
        )
        threshold = 0.15 if state.turn >= 8 else 0.0
        if utility > threshold:
            return QuestionDecision(attribute, utility)

        if (
            state.turn < 8
            and len(candidates) > 200
            and utility < 0.03
            and "other" not in blocked
        ):
            return QuestionDecision("other", max(utility, 0.01))
        return QuestionDecision(None)

    def _utility(
        self,
        attribute: str,
        candidate_attributes: list[dict[str, tuple[str, ...]]],
        probabilities: list[float],
        turn: int,
    ) -> float:
        partitions: dict[str, list[int]] = defaultdict(list)
        for index, attributes in enumerate(candidate_attributes):
            values = attributes.get(attribute, ())
            for value in values:
                partitions[value].append(index)

        covered = {index for indices in partitions.values() for index in indices}
        coverage_mass = sum(probabilities[index] for index in covered)
        if coverage_mass <= 0 or len(partitions) <= 1:
            return 0.0

        current_known_top10 = sum(
            probabilities[index]
            for index in range(min(10, len(candidate_attributes)))
            if index in covered
        )
        current_conditional = current_known_top10 / coverage_mass
        post_answer_mass = 0.0
        for indices in partitions.values():
            ranked_unique = list(dict.fromkeys(indices))
            post_answer_mass += sum(
                probabilities[index] for index in ranked_unique[:10]
            )
        post_conditional = min(post_answer_mass / coverage_mass, 1.0)
        gain = max(post_conditional - current_conditional, 0.0)
        remaining_turn_factor = max((11 - max(turn, 1)) / 10.0, 0.1)
        return (
            gain
            * coverage_mass
            * _RESPONSE_LIKELIHOOD[attribute]
            * remaining_turn_factor
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
