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
    "synthetic",
}
DISCOVERED_MATERIALS = frozenset(MATERIALS)
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
_DETAIL_ATTRIBUTE_ALIASES = {
    "age range description": "audience",
    "brand": "brand",
    "brand name": "brand",
    "closure type": "closure",
    "color": "color",
    "colour": "color",
    "country of origin": "origin",
    "department": "audience",
    "fabric type": "material",
    "fit type": "fit",
    "inner material": "inner_material",
    "lining material": "inner_material",
    "manufacturer": "brand",
    "material": "material",
    "material type": "material",
    "neck style": "neckline",
    "occasion": "occasion",
    "outer material": "outer_material",
    "pattern": "pattern",
    "product care instructions": "care",
    "recommended uses for product": "use_case",
    "shape": "shape",
    "size": "size",
    "sleeve type": "sleeve",
    "sole material": "sole_material",
    "special feature": "feature",
    "special features": "feature",
    "sport": "use_case",
    "sport type": "use_case",
    "style": "style",
    "suggested users": "audience",
    "target audience": "audience",
    "theme": "theme",
}
_NON_DISCOVERY_DETAIL_TERMS = (
    "batter",
    "best seller",
    "date first available",
    "dimension",
    "item model",
    "model name",
    "model number",
    "model year",
    "part number",
    "rank",
    "upc",
    "weight",
)
_ENUMERATED_DETAIL_ATTRIBUTES = {
    "audience",
    "feature",
    "material",
    "occasion",
    "use_case",
}
_GLOBAL_LAYER_ATTRIBUTES = frozenset({"category", "brand", "budget"})
_CATEGORY_LAYER_CORE_ATTRIBUTES = frozenset({"material", "color", "size", "style"})
_DYNAMIC_LAYER_CORE_ATTRIBUTES = frozenset({"feature", "use_case"})
_DYNAMIC_DETAIL_ATTRIBUTES = frozenset({"occasion", "theme"})
_CATEGORY_DETAIL_ATTRIBUTES_BY_SCOPE = {
    "apparel": frozenset(
        {
            "audience",
            "closure",
            "fit",
            "neckline",
            "pattern",
            "shape",
            "sleeve",
        }
    ),
    "footwear": frozenset(
        {
            "audience",
            "closure",
            "fit",
            "inner_material",
            "outer_material",
            "pattern",
            "shape",
            "sole_material",
        }
    ),
    "jewelry": frozenset({"audience", "pattern", "shape"}),
    "accessories": frozenset({"audience", "closure", "pattern", "shape"}),
    "other": frozenset({"audience", "closure", "fit", "pattern", "shape"}),
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
    elif len(token) > 4 and token.endswith(
        ("sses", "ches", "shes", "xes", "zes")
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


def _discovered_detail_attributes(
    product: dict[str, object],
) -> dict[str, list[str]]:
    details = product.get("details")
    if not isinstance(details, dict):
        return {}
    result: dict[str, list[str]] = {}
    for raw_key, raw_value in details.items():
        attribute = _detail_attribute_name(raw_key)
        if attribute is None:
            continue
        values = _detail_attribute_values(attribute, raw_value)
        if values:
            result.setdefault(attribute, []).extend(values)
    return result


def _detail_attribute_name(raw_key: object) -> str | None:
    normalized = " ".join(terms(raw_key))
    if not normalized or any(term in normalized for term in _NON_DISCOVERY_DETAIL_TERMS):
        return None
    return _DETAIL_ATTRIBUTE_ALIASES.get(normalized)


def _detail_attribute_values(attribute: str, raw_value: object) -> list[str]:
    normalized = normalize_value(raw_value)
    if not normalized or len(normalized) > 160 or len(terms(normalized)) > 16:
        return []
    values = [normalized]
    if attribute in _ENUMERATED_DETAIL_ATTRIBUTES:
        values.extend(
            normalize_value(part)
            for part in re.split(r"[,;/|]", normalized)
            if normalize_value(part)
        )
    return list(dict.fromkeys(values))


def _detail_values(product: dict[str, object], key_fragment: str) -> list[str]:
    """Return legacy core values without widening the scoring-time schema."""
    details = product.get("details")
    if not isinstance(details, dict):
        return []
    return [
        normalize_value(value)
        for key, value in details.items()
        if key_fragment in normalize_value(key) and normalize_value(value)
    ]


def _scoped_material_attributes(corpus: str) -> dict[str, list[str]]:
    """Extract material roles without collapsing every role into ``material``."""
    normalized = normalize_value(corpus)
    scopes = {
        "sole_material": ("sole",),
        "outer_material": ("upper", "outer", "shell"),
        "inner_material": ("inner", "lining"),
    }
    result: dict[str, list[str]] = {}
    for attribute, labels in scopes.items():
        for material in DISCOVERED_MATERIALS:
            material_pattern = re.escape(material)
            if any(
                re.search(
                    rf"\b(?:{material_pattern})\b(?:\s+[a-z0-9-]+){{0,2}}"
                    rf"\s+\b{re.escape(label)}\b"
                    rf"|\b{re.escape(label)}(?:\s+material)?\b"
                    rf"\s*(?:(?:is|made\s+of)\s+|[:=-]\s*)?"
                    rf"\b(?:{material_pattern})\b",
                    normalized,
                )
                for label in labels
            ):
                result.setdefault(attribute, []).append(material)
    return result


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


def infer_category_scope(product: dict[str, object]) -> str:
    """Map the catalog taxonomy to a small, reusable commercial schema scope."""
    taxonomy = set(category_term_set(_categories(product)))
    if taxonomy.intersection(
        {
            "boot",
            "clog",
            "flat",
            "footwear",
            "loafer",
            "pump",
            "sandal",
            "shoe",
            "slipper",
            "sneaker",
        }
    ):
        return "footwear"
    if taxonomy.intersection(
        {
            "bracelet",
            "earring",
            "jewelry",
            "necklace",
            "ring",
            "watch",
        }
    ):
        return "jewelry"
    if taxonomy.intersection(
        {
            "accessory",
            "bag",
            "belt",
            "glove",
            "handbag",
            "hat",
            "purse",
            "scarf",
            "tie",
            "wallet",
        }
    ):
        return "accessories"
    if taxonomy.intersection(
        {
            "blouse",
            "coat",
            "dress",
            "hoodie",
            "jacket",
            "jean",
            "pant",
            "shirt",
            "short",
            "skirt",
            "sweater",
            "top",
        }
    ):
        return "apparel"
    return "other"


def extract_layered_attributes(
    product: dict[str, object],
    *,
    core_attributes: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Build isolated global, category and dynamic attribute layers.

    The legacy extractor remains the stable scoring-time contract. The two
    discovered layers are separate so noisy catalog metadata can be inspected
    or enabled selectively without silently changing hard-filter behavior.
    """
    core = dict(core_attributes or extract_attributes(product))
    details = _discovered_detail_attributes(product)
    for attribute, values in _scoped_material_attributes(
        " ".join(searchable_fields(product))
    ).items():
        details.setdefault(attribute, []).extend(values)

    scope = infer_category_scope(product)
    category_names = _CATEGORY_DETAIL_ATTRIBUTES_BY_SCOPE[scope]
    global_layer = {
        name: core[name]
        for name in _GLOBAL_LAYER_ATTRIBUTES
        if core.get(name)
    }
    category_layer = {
        name: core[name]
        for name in _CATEGORY_LAYER_CORE_ATTRIBUTES
        if core.get(name)
    }
    category_layer.update(
        {
            name: tuple(dict.fromkeys(details[name]))
            for name in category_names
            if details.get(name)
        }
    )
    dynamic_layer = {
        name: core[name]
        for name in _DYNAMIC_LAYER_CORE_ATTRIBUTES
        if core.get(name)
    }
    dynamic_layer.update(
        {
            name: tuple(dict.fromkeys(details[name]))
            for name in _DYNAMIC_DETAIL_ATTRIBUTES
            if details.get(name)
        }
    )
    return {
        "global": global_layer,
        "category": category_layer,
        "dynamic": dynamic_layer,
    }


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
