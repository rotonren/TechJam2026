from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from .models import ConstraintOperator, ConstraintSource, Route
from .normalization import (
    COLORS,
    FEATURES,
    MATERIALS,
    STYLES,
    USE_CASES,
    normalize_category_value,
    normalize_value,
    terms,
)

RouteHint = Route | None

_OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead|ignore (?:my )?(?:earlier|previous|old)?|"
    r"change (?:it|that|my mind)|changed (?:my )?mind|"
    r"i(?:'|’)ve changed my mind|i have changed my mind|"
    r"what i need is|rather than)\b",
    re.IGNORECASE,
)
_PREFERENCE_RESET_RE = re.compile(
    r"\b(?:ignore (?:my )?(?:(?:earlier|previous|old)\s+)?preferences?|"
    r"changed (?:my )?mind|i(?:'|’)ve changed my mind|"
    r"i have changed my mind|what i need is)\b",
    re.IGNORECASE,
)
_OVERRIDE_PAYLOAD_RE = re.compile(
    r"\bwhat i need is\s*:?[\s-]*(?P<value>[^.!?]{2,160})",
    re.IGNORECASE,
)
_CONTINUATION_RE = re.compile(
    r"\b(?:show me more|more options|different choices)\b", re.IGNORECASE
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
_AMOUNT = r"[$£€]?\s*(?P<amount>\d+(?:\.\d{1,2})?)"
_BUDGET_BETWEEN_RE = re.compile(
    r"\b(?:between|from)\s*"
    + _AMOUNT
    + r"\s*(?:and|to)\s*[$£€]?\s*(?P<upper>\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_BUDGET_DIRECTION_RE = re.compile(
    r"\b(?P<direction>under|below|less than|at most|no more than|up to|"
    r"over|above|more than|at least|from)\b\s*(?:a\s+)?(?:budget\s+(?:of|around)\s+)?"
    + _AMOUNT,
    re.IGNORECASE,
)
_BUDGET_DEFAULT_RE = re.compile(
    r"(?:[$£€]\s*|\bbudget(?:\s+(?:around|of))?[\s:$£€]*)"
    r"(?P<amount>\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_CATEGORY_RE = re.compile(
    r"\b(?:looking for|need|want|shopping for)\s+(?:an?\s+|some\s+)?"
    r"(?P<category>[a-z][a-z\s-]{0,48}?)(?=\s*(?:[,.!;]|\b(?:but|with|that)\b|$))",
    re.IGNORECASE,
)
_AMAZON_ROOT_TAXONOMY_RE = re.compile(
    r"\b(?:clothing\s*,?\s+)?shoes?\s+(?:&|and)\s+jewelry"
    r"(?:\s+(?:men|women))?\b",
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
_ALIAS_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "brand",
    "by",
    "for",
    "from",
    "key",
    "my",
    "new",
    "of",
    "on",
    "or",
    "requirement",
    "style",
    "the",
    "to",
    "want",
}
_NON_BRAND_TERMS = _KNOWN_CATEGORIES | {
    "accessory",
    "accessories",
    "clothing",
    "jewelry",
    "pants",
    "sweatpants",
}
_TOKEN_SPAN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_EXPECTED_FILLER_TERMS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "can",
    "do",
    "for",
    "have",
    "i",
    "id",
    "im",
    "is",
    "it",
    "just",
    "like",
    "me",
    "my",
    "need",
    "of",
    "one",
    "please",
    "prefer",
    "the",
    "that",
    "to",
    "want",
    "what",
    "would",
}


@dataclass(frozen=True)
class ParsedConstraint:
    attribute: str
    value: str
    confidence: float = 1.0
    is_hard: bool = True
    source: ConstraintSource = "message"
    operator: ConstraintOperator = "eq"
    upper_value: str | None = None
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParseResult:
    constraints: tuple[ParsedConstraint, ...] = ()
    route_hint: RouteHint = None
    is_override: bool = False
    no_preference_attribute: str | None = None
    is_continuation: bool = False
    replace_preferences: bool = False


class MessageParser:
    """Deterministic parser for the evaluator's shopping language."""

    def __init__(self, vocabulary: Mapping[str, tuple[str, ...]] | None = None) -> None:
        fixed: dict[str, set[str]] = {
            "color": set(COLORS),
            "material": set(MATERIALS),
            "style": set(STYLES),
            "feature": set(FEATURES),
            "use_case": set(USE_CASES),
        }
        base_fixed = {attribute: set(values) for attribute, values in fixed.items()}
        for attribute, values in (vocabulary or {}).items():
            if attribute not in {
                "brand",
                "category",
                "color",
                "feature",
                "material",
                "size",
                "style",
                "use_case",
            }:
                continue
            fixed.setdefault(attribute, set()).update(
                normalize_value(value) for value in values if normalize_value(value)
            )
        self._vocabulary = {
            attribute: tuple(
                sorted(
                    {
                        "gray" if value == "grey" else normalize_value(value)
                        for value in values
                        if normalize_value(value)
                    },
                    key=lambda value: (-len(value.split()), -len(value), value),
                )
            )
            for attribute, values in fixed.items()
        }
        self._fixed_values = {
            attribute: {
                normalize_value(value) for value in base_fixed.get(attribute, ())
            }
            for attribute in self._vocabulary
        }
        self._semantic_fixed_values = set().union(*self._fixed_values.values())
        # Catalog vocabularies can contain tens of thousands of brands and
        # noisy category labels.  Store token n-grams once at construction so
        # each message is matched by a bounded dictionary scan rather than by
        # compiling one regular expression per catalog value.
        self._phrase_lookup: dict[str, dict[tuple[str, ...], str]] = {}
        self._max_phrase_words = 1
        for attribute, values in self._vocabulary.items():
            lookup: dict[tuple[str, ...], str] = {}
            for value in values:
                value_terms = tuple(terms(value))
                if not value_terms:
                    continue
                self._max_phrase_words = max(
                    self._max_phrase_words, len(value_terms)
                )
                for alias in self._phrase_aliases(value_terms, attribute):
                    # Values are sorted longest-first in `_vocabulary`; retain
                    # the first canonical value for deterministic collisions.
                    lookup.setdefault(alias, value)
            self._phrase_lookup[attribute] = lookup

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
        replace_preferences = bool(_PREFERENCE_RESET_RE.search(text))
        is_continuation = bool(_CONTINUATION_RE.search(text))
        if is_override:
            expected_attribute = None

        if _NO_PREFERENCE_RE.search(text):
            attribute = expected_attribute or self._mentioned_attribute(text)
            return ParseResult(
                no_preference_attribute=attribute,
                is_override=is_override,
                is_continuation=is_continuation,
                replace_preferences=replace_preferences,
            )

        # Continuation commands are control input, not answers to a pending
        # attribute question.  Returning before catalog-vocabulary matching
        # also prevents noisy aliases such as "show me more" from becoming a
        # hard feature/use-case constraint.
        if is_continuation:
            return ParseResult(
                is_override=is_override,
                is_continuation=True,
                replace_preferences=replace_preferences,
            )

        source: ConstraintSource = "clarification" if expected_attribute else "message"
        extracted: list[ParsedConstraint] = []

        known = self._extract_known_values(text, source, expected_attribute)
        categories = (
            []
            if any(item.attribute == "category" for item in known)
            else self._extract_category(text, source)
        )
        extracted.extend(known)
        extracted.extend(categories)
        if is_override and replace_preferences:
            extracted.extend(self._extract_override_payload(text, extracted))
        if (
            expected_attribute
            and not any(
                item.attribute == normalize_value(expected_attribute)
                for item in extracted
            )
            and not any(item.operator == "not_in" for item in extracted)
            and self._has_unrecognized_expected_text(
                text, extracted, expected_attribute
            )
        ):
            extracted.extend(self._extract_expected(text, expected_attribute, source))

        route_hint: RouteHint = None
        if _BUYING_RE.search(text):
            route_hint = "buying"
        elif _BROWSING_RE.search(text):
            route_hint = "browsing"

        return ParseResult(
            constraints=tuple(self._deduplicate(extracted)),
            route_hint=route_hint,
            is_override=is_override,
            is_continuation=is_continuation,
            replace_preferences=replace_preferences,
        )

    @staticmethod
    def _extract_override_payload(
        text: str, extracted: list[ParsedConstraint]
    ) -> list[ParsedConstraint]:
        """Retain an explicit replacement requirement as soft text evidence."""
        match = _OVERRIDE_PAYLOAD_RE.search(text)
        if match is None:
            return []
        value = normalize_value(match.group("value"))
        if not value:
            return []
        payload_terms = {token for token in terms(value) if not token.isdigit()}
        represented_terms = {
            token
            for item in extracted
            for raw_value in (*item.alternatives, item.value, item.upper_value or "")
            for token in terms(raw_value)
            if not token.isdigit()
        }
        if payload_terms and payload_terms.issubset(represented_terms):
            return []
        return [ParsedConstraint("feature", value, 0.6, False, "clarification")]

    def _has_unrecognized_expected_text(
        self,
        text: str,
        extracted: list[ParsedConstraint],
        expected_attribute: str | None,
    ) -> bool:
        """Avoid assigning a known answer to an unrelated pending slot.

        A clarification such as ``Blue.`` can arrive while a use-case question
        is pending.  Known aliases are already represented in ``extracted``;
        only residual, non-filler words should be offered to the pending slot.
        Character spans are used so multi-word catalog aliases and budget
        phrases are removed without changing the original query evidence.
        """
        covered: list[tuple[int, int]] = [
            (start, end)
            for start, end, _, _ in self._vocabulary_matches(
                text, expected_attribute
            )
        ]
        for pattern in (_BUDGET_BETWEEN_RE, _BUDGET_DIRECTION_RE, _BUDGET_DEFAULT_RE):
            covered.extend(match.span() for match in pattern.finditer(text))
        category_match = _CATEGORY_RE.search(text)
        if category_match:
            covered.append(category_match.span("category"))

        # The fixed lexical category list is useful even when a lightweight
        # parser was constructed without a catalog vocabulary. A style answer
        # is the exception: terms such as "sneaker" can describe a look, and
        # should survive as soft text evidence rather than disappear merely
        # because they are also category words.
        if normalize_value(expected_attribute or "") != "style":
            for match in re.finditer(
                r"\b(?:"
                + "|".join(sorted(_KNOWN_CATEGORIES, key=len, reverse=True))
                + r")\b",
                text,
                re.IGNORECASE,
            ):
                covered.append(match.span())

        residual_chars = [
            character
            if not any(start <= index < end for start, end in covered)
            else " "
            for index, character in enumerate(text)
        ]
        residual_tokens = terms("".join(residual_chars))
        return any(token not in _EXPECTED_FILLER_TERMS for token in residual_tokens)

    def _extract_expected(
        self, text: str, attribute: str, source: ConstraintSource
    ) -> list[ParsedConstraint]:
        attribute = normalize_value(attribute)
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
            return [ParsedConstraint(attribute, cleaned, 0.6, False, source)]
        return []

    def _extract_known_values(
        self,
        text: str,
        source: ConstraintSource,
        expected_attribute: str | None = None,
    ) -> list[ParsedConstraint]:
        matches = self._vocabulary_matches(text, expected_attribute)
        result: list[ParsedConstraint] = []
        index = 0
        while index < len(matches):
            start, end, attribute, value = matches[index]
            alternatives = [value]
            next_index = index + 1
            while next_index < len(matches):
                other_start, other_end, other_attribute, other_value = matches[next_index]
                connector = text[end:other_start]
                if (
                    other_attribute != attribute
                    or not re.fullmatch(r"\s*,?\s*or\s*", connector, re.IGNORECASE)
                ):
                    break
                alternatives.append(other_value)
                end = other_end
                next_index += 1

            operator: ConstraintOperator = "eq"
            values: tuple[str, ...] = ()
            if len(alternatives) > 1:
                operator = "not_in" if self._is_negative_value(text, start) else "in"
                values = tuple(dict.fromkeys(alternatives))
            elif self._is_negative_value(text, start):
                operator = "not_in"
                values = (value,)
            result.append(
                ParsedConstraint(
                    attribute,
                    value,
                    1.0,
                    True,
                    source,
                    operator=operator,
                    alternatives=values,
                )
            )
            index = next_index

        return [*result, *self._extract_budget(text, source)]

    def _vocabulary_matches(
        self, text: str, expected_attribute: str | None = None
    ) -> list[tuple[int, int, str, str]]:
        match_text = re.sub(r"['’]s\b", "  ", text)
        token_spans = [
            (match.group(0).lower(), match.start(), match.end())
            for match in _TOKEN_SPAN_RE.finditer(match_text)
        ]
        raw_candidates: list[tuple[int, int, str, str]] = []
        for start_index, (_, start, _) in enumerate(token_spans):
            max_length = min(self._max_phrase_words, len(token_spans) - start_index)
            for length in range(max_length, 0, -1):
                end_index = start_index + length - 1
                key = tuple(
                    token_spans[index][0] for index in range(start_index, end_index + 1)
                )
                end = token_spans[end_index][2]
                for attribute, lookup in self._phrase_lookup.items():
                    value = lookup.get(key)
                    if value is None:
                        continue
                    raw_candidates.append((start, end, attribute, value))

        root_taxonomy_spans = tuple(
            match.span() for match in _AMAZON_ROOT_TAXONOMY_RE.finditer(text)
        )
        category_spans = tuple(
            (start, end)
            for start, end, attribute, value in raw_candidates
            if attribute == "category"
            and self._alias_allowed(
                attribute,
                value,
                text,
                start,
                end,
                (),
                root_taxonomy_spans,
                expected_attribute,
            )
            and not self._is_negative_value(text, start)
        )
        candidates = [
            (start, end, attribute, value)
            for start, end, attribute, value in raw_candidates
            if self._alias_allowed(
                attribute,
                value,
                text,
                start,
                end,
                category_spans,
                root_taxonomy_spans,
                expected_attribute,
            )
            and not (
                attribute == "category" and self._is_negative_value(text, start)
            )
        ]

        # Prefer the longest phrase at a start position and prevent overlapping
        # aliases for the same attribute while allowing the same text to map to
        # different represented attributes (for example `casual` style/use_case).
        ordered = sorted(
            candidates,
            key=lambda item: (item[0], -(item[1] - item[0]), item[2], item[3]),
        )
        result: list[tuple[int, int, str, str]] = []
        occupied: dict[str, list[tuple[int, int]]] = {}
        for candidate in ordered:
            start, end, attribute, _ = candidate
            if any(
                start < prior_end and prior_start < end
                for prior_start, prior_end in occupied.get(attribute, ())
            ):
                continue
            occupied.setdefault(attribute, []).append((start, end))
            result.append(candidate)
        return sorted(result, key=lambda item: (item[0], item[1], item[2], item[3]))

    def _alias_allowed(
        self,
        attribute: str,
        value: str,
        text: str,
        start: int,
        end: int,
        category_spans: tuple[tuple[int, int], ...],
        root_taxonomy_spans: tuple[tuple[int, int], ...],
        expected_attribute: str | None,
    ) -> bool:
        normalized = normalize_value(value)
        value_terms = terms(normalized)
        if not value_terms or any(len(token) < 2 for token in value_terms):
            return False
        if len(value_terms) == 1 and value_terms[0] in _ALIAS_STOPWORDS:
            return False
        expected = normalize_value(expected_attribute or "")
        fixed_value = normalized in self._fixed_values.get(attribute, set())
        explicit_cue = attribute == expected or self._has_explicit_alias_cue(
            attribute,
            normalized,
            text,
            start,
            end,
        )
        # A material followed by a component role describes that component,
        # not necessarily the whole product. During broad clarification keep
        # the complete phrase as soft text evidence instead of replacing the
        # user's primary material constraint.
        scoped_material = re.match(
            r"\s+(?:sole|upper|outer|inner|lining)\b", text[end:]
        )
        if (
            attribute == "material"
            and scoped_material is not None
            and expected in {"feature", "other"}
        ):
            return False
        # Catalog-derived style labels often come from free-form product text.
        # During a style clarification they are query evidence, not a safe
        # structured filter. Returning False lets `_extract_expected` retain
        # the complete answer as a soft clarification constraint.
        if expected == "style" and attribute == "style" and not fixed_value:
            return False
        # Long catalog descriptions can incidentally contain another product's
        # category or size vocabulary (for example "dress" or "free"). Keep
        # fixed semantic values and concise multi-attribute answers, but require
        # a nearby cue before a dynamic alias escapes the pending slot.
        if (
            expected
            and attribute != expected
            and not fixed_value
            and not self._is_concise_clarification(text)
            and not explicit_cue
        ):
            return False
        if fixed_value:
            return True
        if attribute == "category" and self._overlaps_any(
            start, end, root_taxonomy_spans
        ):
            return False
        if attribute in {"brand", "style"}:
            if self._overlaps_any(start, end, category_spans) and not explicit_cue:
                return False
            if not explicit_cue:
                return False
        if attribute == "brand" and (
            normalized in _NON_BRAND_TERMS
            or (
                normalized in self._semantic_fixed_values
                and not explicit_cue
            )
            or normalized in {
                normalize_value(item) for item in self._vocabulary.get("category", ())
            }
        ):
            return False
        if attribute == "category" and (
            normalized in COLORS
            or normalized in MATERIALS
            or normalized in STYLES
            or normalized in FEATURES
            or normalized in USE_CASES
        ):
            return False
        if attribute == "category" and re.search(
            r"\b(?:under|over|above|below|less\s+than|more\s+than|"
            r"at\s+least|at\s+most|up\s+to|no\s+more\s+than)\s*\d",
            normalized,
        ):
            return False

        prefix = text[:start].lower()
        boundary = max(prefix.rfind("."), prefix.rfind(":"), prefix.rfind(";"))
        clause = prefix[boundary + 1 :]
        if attribute == "category" and normalize_category_value(normalized) in {
            normalize_category_value(item) for item in _KNOWN_CATEGORIES
        }:
            return True
        if attribute == "category":
            return bool(
                re.search(
                    r"\b(?:looking\s+for|shopping\s+for|need|want|find|show\s+me)\b",
                    clause,
                )
            )
        if attribute == "brand":
            return True
        if attribute == "size":
            return bool(re.search(r"\b(?:size|sized|wear|in)\b", clause))
        if attribute == "style":
            return True
        if attribute == "feature":
            return bool(
                re.search(
                    r"\b(?:feature|need|want|must\s+have|with|looking\s+for|prefer)\b",
                    clause,
                )
            )
        return True

    @staticmethod
    def _overlaps_any(
        start: int, end: int, spans: tuple[tuple[int, int], ...]
    ) -> bool:
        return any(
            start < other_end and other_start < end
            for other_start, other_end in spans
        )

    @staticmethod
    def _has_explicit_alias_cue(
        attribute: str, value: str, text: str, start: int, end: int
    ) -> bool:
        if attribute == "brand":
            if "brand" in terms(value) or re.search(r"['’]s\b", text[start:end]):
                return True
            left = text[max(0, start - 24) : start]
            right = text[end : min(len(text), end + 16)]
            return bool(
                re.search(
                    r"\b(?:brand|by|from)(?:\s+is)?\s*[:=-]?\s*$", left
                )
                or re.match(r"\s*(?:brand)\b", right)
            )
        if attribute == "style":
            left = text[max(0, start - 24) : start]
            right = text[end : min(len(text), end + 16)]
            return bool(
                re.search(
                    r"\b(?:style|styled|look|design)(?:\s+is)?\s*[:=-]?\s*$",
                    left,
                )
                or re.match(
                    r"\s*[,;:=-]?\s*(?:style|styled|look|design)\b", right
                )
            )
        left = text[max(0, start - 32) : start]
        if attribute == "category":
            return bool(
                re.search(
                    r"\b(?:category|type|kind|looking\s+for|shopping\s+for|"
                    r"need|want|find|show\s+me)\b[^.!;:]{0,20}$",
                    left,
                )
            )
        if attribute == "size":
            return bool(
                re.search(r"\b(?:size|sized|width)\b[^.!;:]{0,12}$", left)
                or re.search(r"\bwear(?:\s+an?)?\s*$", left)
            )
        if attribute == "feature":
            return bool(
                re.search(
                    r"\b(?:feature|with|must\s+have|prefer)\b[^.!;:]{0,20}$",
                    left,
                )
            )
        if attribute == "use_case":
            return bool(re.search(r"\bfor\b[^.!;:]{0,20}$", left))
        return False

    @staticmethod
    def _is_concise_clarification(text: str) -> bool:
        cleaned = re.sub(r"^for that,?\s*(?:what matters is:?)?\s*", "", text)
        return len(terms(cleaned)) <= 4

    @staticmethod
    def _phrase_aliases(
        value_terms: tuple[str, ...], attribute: str
    ) -> tuple[tuple[str, ...], ...]:
        if not value_terms:
            return ()
        last = value_terms[-1]
        variants = {last}
        if attribute == "category":
            variants.add(normalize_category_value(last))
        # Match ordinary singular/plural user phrasing for all catalog values,
        # while keeping the catalog's original spelling as the returned value.
        if len(last) > 3 and last.endswith("ies"):
            variants.add(last[:-3] + "y")
        elif len(last) > 3 and last.endswith("ses"):
            variants.add(last[:-2])
        elif len(last) > 2 and last.endswith("s") and not last.endswith("ss"):
            variants.add(last[:-1])
        elif len(last) > 2:
            variants.add(last + "s")
        return tuple(
            (*value_terms[:-1], variant) for variant in variants if variant
        )

    @staticmethod
    def _phrase_pattern(value: str) -> re.Pattern[str]:
        words = value.split()
        if not words:
            return re.compile(r"(?!)")
        phrase = r"\s+".join(
            [*(re.escape(word) for word in words[:-1]), MessageParser._word_pattern(words[-1])]
        )
        return re.compile(rf"(?<![a-z0-9]){phrase}(?![a-z0-9])", re.IGNORECASE)

    @staticmethod
    def _word_pattern(word: str) -> str:
        if len(word) > 3 and word.endswith("ies"):
            return rf"(?:{re.escape(word[:-3] + 'y')}|{re.escape(word)})"
        if len(word) > 3 and word.endswith("ses"):
            return rf"{re.escape(word[:-2])}(?:es)?"
        if len(word) > 2 and word.endswith("s"):
            return rf"{re.escape(word[:-1])}s?"
        if len(word) > 2 and word.endswith("y") and word[-2] not in "aeiou":
            return rf"(?:{re.escape(word)}|{re.escape(word[:-1])}ies)"
        if len(word) > 2 and word.endswith(("ch", "sh", "x", "z")):
            return rf"{re.escape(word)}(?:es)?"
        return rf"{re.escape(word)}s?"

    @staticmethod
    def _is_negative_value(text: str, start: int) -> bool:
        return bool(
            re.search(
                r"\b(?:not|without|no|don['’]?t\s+want|do\s+not\s+want|"
                r"doesn['’]?t\s+include|does\s+not\s+include)\s*$",
                text[:start],
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _extract_budget(text: str, source: ConstraintSource) -> list[ParsedConstraint]:
        result: list[ParsedConstraint] = []
        covered: list[tuple[int, int]] = []
        for match in _BUDGET_BETWEEN_RE.finditer(text):
            covered.append(match.span())
            result.append(
                ParsedConstraint(
                    "budget",
                    f"{float(match.group('amount')):.2f}",
                    1.0,
                    True,
                    source,
                    operator="between",
                    upper_value=f"{float(match.group('upper')):.2f}",
                )
            )

        directions: dict[str, ConstraintOperator] = {
            "under": "lte",
            "below": "lte",
            "less than": "lte",
            "at most": "lte",
            "no more than": "lte",
            "up to": "lte",
            "over": "gte",
            "above": "gte",
            "more than": "gte",
            "at least": "gte",
            "from": "gte",
        }
        for match in _BUDGET_DIRECTION_RE.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in covered):
                continue
            covered.append(match.span())
            result.append(
                ParsedConstraint(
                    "budget",
                    f"{float(match.group('amount')):.2f}",
                    1.0,
                    True,
                    source,
                    operator=directions[normalize_value(match.group("direction"))],
                )
            )

        for match in _BUDGET_DEFAULT_RE.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in covered):
                continue
            result.append(
                ParsedConstraint(
                    "budget",
                    f"{float(match.group('amount')):.2f}",
                    1.0,
                    True,
                    source,
                    operator="lte",
                )
            )
        return result

    def _extract_category(
        self, text: str, source: ConstraintSource
    ) -> list[ParsedConstraint]:
        match = _CATEGORY_RE.search(text)
        if not match:
            return []
        if self._overlaps_any(
            *match.span("category"),
            tuple(item.span() for item in _AMAZON_ROOT_TAXONOMY_RE.finditer(text)),
        ):
            return []
        category_tokens = terms(match.group("category"))
        known = [token for token in category_tokens if token in _KNOWN_CATEGORIES]
        if not known:
            return []
        return [ParsedConstraint("category", " ".join(known), 1.0, True, source)]

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
        seen: set[tuple[str, str, ConstraintOperator, str | None, tuple[str, ...]]] = set()
        for item in items:
            key = (
                item.attribute,
                normalize_category_value(item.value)
                if item.attribute == "category"
                else item.value,
                item.operator,
                item.upper_value,
                tuple(
                    sorted(
                        normalize_category_value(value)
                        if item.attribute == "category"
                        else value
                        for value in item.alternatives
                    )
                ),
            )
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result
