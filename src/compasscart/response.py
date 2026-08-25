from __future__ import annotations

from collections.abc import Iterable

from .models import Candidate, QuestionDecision

ALLOWED_ATTRIBUTES = {
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
}
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
    ) -> dict[str, object]:
        limit = min(max(int(top_k), 0), 10)
        identifiers: list[str] = []
        seen: set[str] = set()
        for candidate in ranked:
            identifier = candidate.parent_asin
            if identifier in self.valid_ids and identifier not in seen:
                identifiers.append(identifier)
                seen.add(identifier)
            if len(identifiers) >= limit:
                break
        if len(identifiers) < limit:
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
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": [
                {"parent_asin": identifier} for identifier in identifiers
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
