from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping

from .attribute_schema import ALLOWED_ASK_ATTRIBUTES
from .models import Candidate, QuestionDecision

ALLOWED_ATTRIBUTES = set(ALLOWED_ASK_ATTRIBUTES)
_QUESTIONS = {
    "category": "What type of product are you looking for?",
    "material": "Do you have a material preference?",
    "color": "Which color would you prefer?",
    "size": "What size or fit should I prioritize?",
    "style": "Which style do you prefer?",
    "brand": "Do you have a preferred brand?",
    "budget": "What budget should I stay within?",
    "feature": "Which feature matters most to you?",
    "use_case": "What will you mainly use it for?",
    "other": "What other detail should I prioritize?",
}


class ResponseBuilder:
    def __init__(self, valid_ids: set[str], fallback_ids: Iterable[str]) -> None:
        self.valid_ids = set(valid_ids)
        self.fallback_ids = tuple(dict.fromkeys(fallback_ids))

    def build(
        self,
        ranked: list[Candidate],
        question: QuestionDecision | None,
        *,
        top_k: int,
        relaxed: bool | None = None,
        relaxed_constraints: Iterable[str] = (),
        excluded_ids: Collection[str] = (),
        usage: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        limit = min(max(int(top_k), 0), 10)
        identifiers: list[str] = []
        selected: list[Candidate] = []
        seen: set[str] = set()
        for candidate in ranked:
            identifier = candidate.parent_asin
            if (
                identifier in self.valid_ids
                and identifier not in excluded_ids
                and identifier not in seen
            ):
                identifiers.append(identifier)
                selected.append(candidate)
                seen.add(identifier)
            if len(identifiers) >= limit:
                break
        if len(identifiers) < limit:
            for identifier in self.fallback_ids:
                if (
                    identifier in self.valid_ids
                    and identifier not in excluded_ids
                    and identifier not in seen
                ):
                    identifiers.append(identifier)
                    seen.add(identifier)
                if len(identifiers) >= limit:
                    break
        # If the user requested more pages than the catalog can provide, use a
        # previously shown ID only as a final contract-preserving fallback.
        if len(identifiers) < min(limit, len(self.valid_ids)) and excluded_ids:
            for identifier in self.fallback_ids:
                if identifier in self.valid_ids and identifier not in seen:
                    identifiers.append(identifier)
                    seen.add(identifier)
                if len(identifiers) >= limit:
                    break

        attribute = question.ask_attribute if question else None
        if attribute not in ALLOWED_ATTRIBUTES:
            attribute = None
        message = (
            _QUESTIONS[attribute]
            if attribute
            else "Here are the closest matches I found."
        )
        constraints = list(dict.fromkeys(str(item) for item in relaxed_constraints if item))
        for candidate in selected:
            if candidate.relaxed:
                constraints.extend(
                    item
                    for item in candidate.violations
                    if item and item not in constraints
                )
        is_relaxed = (
            relaxed
            if relaxed is not None
            else bool(constraints) or any(item.relaxed for item in selected)
        )
        if is_relaxed:
            detail = ", ".join(constraints)
            if detail:
                message = f"{message} These are close alternatives after relaxing {detail}."
            else:
                message = f"{message} These are close alternatives after relaxing a strict preference."
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": [
                {"parent_asin": identifier} for identifier in identifiers
            ],
            "usage": _token_usage(usage),
        }


def _token_usage(usage: Mapping[str, object] | None) -> dict[str, int]:
    """Report what a model actually consumed, defaulting to nothing.

    The contract requires non-negative integers, so a backend that reports a
    malformed or negative count is treated as having reported none rather than
    being allowed to emit an invalid response.
    """
    reported: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
    if not usage:
        return reported
    for field in reported:
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            continue
        reported[field] = value
    return reported
