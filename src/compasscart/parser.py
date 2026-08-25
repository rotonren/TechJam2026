from __future__ import annotations

import re
from dataclasses import dataclass

from .models import ConstraintSource, Route
from .normalization import (
    COLORS,
    FEATURES,
    MATERIALS,
    STYLES,
    USE_CASES,
    normalize_value,
    terms,
)

RouteHint = Route | None

_OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead|ignore (?:my )?(?:earlier|previous|old)?|"
    r"change (?:it|that|my mind)|what i need is|rather than)\b",
    re.IGNORECASE,
)
_NO_PREFERENCE_RE = re.compile(
    r"\b(?:don['’]?t|do not|no)\s+(?:have\s+)?(?:an?\s+)?(?:additional\s+)?"
    r"preference\b|\b(?:doesn['’]?t|does not) matter\b|\buse your judgment\b",
    re.IGNORECASE,
)
_BROWSING_RE = re.compile(
    r"\b(?:still exploring|just browsing|open to|not sure|show me options|"
    r"looking around|any suggestions)\b",
    re.IGNORECASE,
)
_BUYING_RE = re.compile(
    r"\b(?:key requirement|must have|need|under|at most|no more than|budget)\b|[$£€]\s*\d",
    re.IGNORECASE,
)
_BUDGET_RE = re.compile(
    r"(?:[$£€]\s*|\b(?:under|below|less than|at most|no more than|up to|budget(?: around| of)?)[\s:$£€]*)"
    r"(?P<amount>\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_CATEGORY_RE = re.compile(
    r"\b(?:looking for|need|want|shopping for)\s+(?:an?\s+|some\s+)?"
    r"(?P<category>[a-z][a-z\s-]{0,48}?)(?=\s*(?:[,.!;]|\b(?:but|with|that)\b|$))",
    re.IGNORECASE,
)
_KNOWN_CATEGORIES = {
    "belt",
    "belts",
    "boot",
    "boots",
    "coat",
    "coats",
    "dress",
    "dresses",
    "jacket",
    "jackets",
    "jeans",
    "pants",
    "shirt",
    "shirts",
    "shoe",
    "shoes",
    "shorts",
    "skirt",
    "skirts",
    "sneaker",
    "sneakers",
    "sweater",
    "sweaters",
    "top",
    "tops",
}


@dataclass(frozen=True)
class ParsedConstraint:
    attribute: str
    value: str
    confidence: float = 1.0
    is_hard: bool = True
    source: ConstraintSource = "message"


@dataclass(frozen=True)
class ParseResult:
    constraints: tuple[ParsedConstraint, ...] = ()
    route_hint: RouteHint = None
    is_override: bool = False
    no_preference_attribute: str | None = None


class MessageParser:
    """Deterministic parser for the evaluator's shopping language."""

    def parse(
        self,
        message: str,
        turn: int,
        expected_attribute: str | None = None,
    ) -> ParseResult:
        del turn  # The parser is stateless; turn-aware behavior belongs to the ledger.
        text = normalize_value(message)
        if not text:
            return ParseResult()
        is_override = bool(_OVERRIDE_RE.search(text))
        if is_override:
            expected_attribute = None

        if _NO_PREFERENCE_RE.search(text):
            attribute = expected_attribute or self._mentioned_attribute(text)
            return ParseResult(
                route_hint="browsing",
                no_preference_attribute=attribute,
                is_override=is_override,
            )

        source: ConstraintSource = "clarification" if expected_attribute else "message"
        extracted: list[ParsedConstraint] = []

        if expected_attribute:
            extracted.extend(self._extract_expected(text, expected_attribute, source))
        else:
            extracted.extend(self._extract_known_values(text, source))
            extracted.extend(self._extract_category(text, source))

        route_hint: RouteHint = None
        if _BROWSING_RE.search(text):
            route_hint = "browsing"
        elif extracted or _BUYING_RE.search(text):
            route_hint = "buying"

        return ParseResult(
            constraints=tuple(self._deduplicate(extracted)),
            route_hint=route_hint,
            is_override=is_override,
        )

    def _extract_expected(
        self, text: str, attribute: str, source: ConstraintSource
    ) -> list[ParsedConstraint]:
        attribute = normalize_value(attribute)
        known = self._extract_known_values(text, source)
        matches = [item for item in known if item.attribute == attribute]
        if matches:
            return matches

        if attribute == "category":
            categories = self._extract_category(text, source)
            if categories:
                return categories

        cleaned = re.sub(r"^for that,?\s*(?:what matters is:?)?\s*", "", text)
        cleaned = cleaned.strip(" .;:")
        if cleaned and attribute in {
            "brand",
            "category",
            "feature",
            "other",
            "size",
            "style",
            "use_case",
        }:
            return [ParsedConstraint(attribute, cleaned, 1.0, True, source)]
        return []

    def _extract_known_values(
        self, text: str, source: ConstraintSource
    ) -> list[ParsedConstraint]:
        tokens = terms(text)
        result: list[ParsedConstraint] = []
        vocabularies = (
            ("color", COLORS),
            ("material", MATERIALS),
            ("style", STYLES),
            ("feature", FEATURES),
            ("use_case", USE_CASES),
        )
        for attribute, vocabulary in vocabularies:
            for value in tokens:
                if value in vocabulary:
                    canonical = "gray" if value == "grey" else value
                    result.append(
                        ParsedConstraint(attribute, canonical, 1.0, True, source)
                    )

        for match in _BUDGET_RE.finditer(text):
            result.append(
                ParsedConstraint(
                    "budget", f"{float(match.group('amount')):.2f}", 1.0, True, source
                )
            )
        return result

    def _extract_category(
        self, text: str, source: ConstraintSource
    ) -> list[ParsedConstraint]:
        match = _CATEGORY_RE.search(text)
        if not match:
            return []
        category_tokens = terms(match.group("category"))
        known = [token for token in category_tokens if token in _KNOWN_CATEGORIES]
        value = " ".join(known or category_tokens)
        if not value:
            return []
        return [ParsedConstraint("category", value, 1.0, True, source)]

    @staticmethod
    def _mentioned_attribute(text: str) -> str | None:
        for attribute in (
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
        ):
            if re.search(rf"\b{re.escape(attribute.replace('_', ' '))}\b", text):
                return attribute
        return None

    @staticmethod
    def _deduplicate(items: list[ParsedConstraint]) -> list[ParsedConstraint]:
        result: list[ParsedConstraint] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            key = (item.attribute, item.value)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result
