from __future__ import annotations

from collections import defaultdict

from .evolution import DEFAULT_RESPONSE_LIKELIHOOD, PolicyMemory
from .models import Candidate, QuestionDecision, SessionState
from .normalization import extract_attributes

_ATTRIBUTES = (
    "material",
    "color",
    "size",
    "style",
    "budget",
    "feature",
    "use_case",
)
_RESPONSE_LIKELIHOOD = DEFAULT_RESPONSE_LIKELIHOOD


class QuestionPolicy:
    def __init__(self, attribute_lookup=None, memory: PolicyMemory | None = None) -> None:
        self.attribute_lookup = attribute_lookup
        # Without a memory the policy uses the hand-written table unchanged,
        # which is the behaviour every prior release measured.
        self.memory = memory or PolicyMemory(enabled=False)

    def choose(
        self, candidates: list[Candidate], state: SessionState
    ) -> QuestionDecision:
        if state.continuation_requested:
            return QuestionDecision(None)
        if len(candidates) <= 10:
            return QuestionDecision(None)

        # An explicit override invalidates earlier preference evidence. Ask
        # once for any additional distinguishing detail when the replacement
        # still leaves an overloaded pool. Override scope is turn-local, so
        # this cannot form a repeated generic-question loop.
        if state.override_scope != "none" and "other" not in state.asked_attributes:
            return QuestionDecision("other", 1.0)

        probabilities = self._probabilities(candidates)
        candidate_attributes = [self._attributes(candidate) for candidate in candidates]
        blocked = (
            set(state.asked_attributes)
            | state.no_preference_attributes
            | state.unproductive_attributes
        )
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
                state.profile_segment,
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
        segment: str = "",
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
        # The response likelihood is the estimate the memory refines: it starts
        # at the hand-written prior and moves only with observed evidence.
        likelihood = self.memory.likelihood(attribute, segment=segment)
        return gain * coverage_mass * likelihood * remaining_turn_factor

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
