from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Mapping

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")

COLORS = {
    "black",
    "white",
    "blue",
    "red",
    "pink",
    "green",
    "brown",
    "gray",
    "grey",
    "purple",
    "yellow",
    "orange",
}
MATERIALS = {
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
    "mesh",
    "denim",
    "linen",
    "rubber",
}
USE_CASES = {
    "running",
    "gym",
    "outdoor",
    "hiking",
    "work",
    "winter",
    "casual",
    "formal",
    "party",
    "travel",
}
STYLES = {
    "classic",
    "casual",
    "formal",
    "athletic",
    "vintage",
    "modern",
    "puffer",
    "a-line",
    "slim",
    "relaxed",
}
FEATURES = {
    "breathable",
    "comfortable",
    "durable",
    "insulated",
    "lightweight",
    "stretch",
    "warm",
    "waterproof",
    "weatherproof",
}
GENERIC_CATEGORIES = {
    "accessories",
    "clothing",
    "clothing shoes jewelry",
    "clothing shoes and jewelry",
    "men",
    "shoes and jewelry",
    "women",
}
CATEGORY_LINK_TERMS = frozenset({"and", "for", "of"})
CATEGORY_PLURAL_EXCEPTIONS = {
    "booties": "bootie",
    "hoodies": "hoodie",
    "panties": "panty",
    "ties": "tie",
}


def flatten_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return " ".join(
            f"{flatten_text(key)} {flatten_text(item)}"
            for key, item in value.items()
        )
    if isinstance(value, Iterable):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def normalize_value(value: object) -> str:
    normalized = WHITESPACE_RE.sub(" ", flatten_text(value)).strip().lower()
    return normalized.strip(" -;,.")


def normalize_category_value(value: object) -> str:
    """Normalize category words while accepting ordinary singular/plural variants."""
    return " ".join(_singular_category_term(token) for token in terms(value))


def category_term_set(value: object) -> frozenset[str]:
    return frozenset(
        _singular_category_term(token)
        for token in terms(value)
        if token not in CATEGORY_LINK_TERMS
    )


def _singular_category_term(token: str) -> str:
    if token in CATEGORY_PLURAL_EXCEPTIONS:
        return CATEGORY_PLURAL_EXCEPTIONS[token]
    if len(token) > 3 and token.endswith("ies"):
        singular = f"{token[:-3]}y"
    elif (len(token) > 3 and token.endswith("ses")) or (
        len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes"))
    ):
        singular = token[:-2]
    elif len(token) > 2 and token.endswith("s") and not token.endswith("ss"):
        singular = token[:-1]
    else:
        singular = token
    return sys.intern(singular)


def terms(text: object) -> list[str]:
    return list(
        dict.fromkeys(
            sys.intern(token.lower()) for token in TOKEN_RE.findall(flatten_text(text))
        )
    )


def _ordered_matches(tokens: Iterable[str], vocabulary: set[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(token for token in tokens if token in vocabulary))


def _detail_values(product: dict[str, object], key_fragment: str) -> list[str]:
    details = product.get("details")
    if not isinstance(details, dict):
        return []
    return [
        normalize_value(value)
        for key, value in details.items()
        if key_fragment in normalize_value(key) and normalize_value(value)
    ]


def _categories(product: dict[str, object]) -> tuple[str, ...]:
    raw_categories = product.get("categories")
    if not isinstance(raw_categories, list):
        return ()
    result: list[str] = []
    for category in raw_categories:
        for part in re.split(r"[,>]", str(category)):
            value = normalize_value(part.replace("&", "and"))
            compact = " ".join(terms(value))
            if compact and compact not in GENERIC_CATEGORIES:
                result.append(compact)
    return tuple(dict.fromkeys(result))


def extract_attributes(product: dict[str, object]) -> dict[str, tuple[str, ...]]:
    corpus = " ".join(searchable_fields(product))
    corpus_terms = terms(corpus)

    colors = list(_ordered_matches(corpus_terms, COLORS))
    colors = ["gray" if value == "grey" else value for value in colors]
    materials = _ordered_matches(corpus_terms, MATERIALS)
    use_cases = _ordered_matches(corpus_terms, USE_CASES)
    features = _ordered_matches(corpus_terms, FEATURES)

    size_values = _detail_values(product, "size")
    if "wide" in corpus_terms:
        size_values.append("wide")
    if "narrow" in corpus_terms:
        size_values.append("narrow")

    style_values = _detail_values(product, "style")
    style_values.extend(_ordered_matches(corpus_terms, STYLES))

    brand_values = _detail_values(product, "brand")
    store = normalize_value(product.get("store"))
    if store:
        brand_values.append(store)

    price = product.get("price")
    budget: tuple[str, ...] = ()
    if isinstance(price, (int, float)):
        budget = (f"{float(price):.2f}",)

    return {
        "category": _categories(product),
        "material": materials,
        "color": tuple(dict.fromkeys(colors)),
        "size": tuple(dict.fromkeys(size_values)),
        "style": tuple(dict.fromkeys(style_values)),
        "brand": tuple(dict.fromkeys(brand_values)),
        "budget": budget,
        "feature": features,
        "use_case": use_cases,
    }


def searchable_fields(product: dict[str, object]) -> tuple[str, ...]:
    return (
        flatten_text(product.get("title")),
        flatten_text(product.get("categories")),
        flatten_text(product.get("features")),
        flatten_text(product.get("details")),
        flatten_text(product.get("store")),
        flatten_text(product.get("description")),
    )


def searchable_term_set(product: dict[str, object]) -> frozenset[str]:
    return frozenset(terms(searchable_fields(product)))
