from collections import deque
from types import MappingProxyType

import pytest

from compasscart.normalization import (
    category_term_set,
    extract_attributes,
    normalize_category_value,
    searchable_fields,
    searchable_term_set,
    terms,
)


def test_extracts_material_color_budget_and_use_case():
    product = {
        "title": "Blue mesh running shoes",
        "price": 79.99,
        "features": ["Breathable mesh for gym and outdoor running"],
        "details": {"Color": "Navy Blue", "Material": "Mesh"},
    }

    attributes = extract_attributes(product)

    assert "blue" in attributes["color"]
    assert "mesh" in attributes["material"]
    assert "running" in attributes["use_case"]
    assert attributes["budget"] == ("79.99",)


def test_terms_are_normalized_and_deduplicated():
    assert terms("Blue BLUE running-shoes") == ["blue", "running", "shoes"]


def test_category_term_set_merges_values_and_drops_link_terms():
    assert category_term_set(("Shoes", "Athletic")) == frozenset(
        {"shoe", "athletic"}
    )


@pytest.mark.parametrize(
    "value",
    (
        frozenset({"Shoes", "Athletic"}),
        deque(("Shoes", "Athletic")),
        (item for item in ("Shoes", "Athletic")),
    ),
)
def test_category_term_set_structurally_flattens_general_iterables(value):
    assert category_term_set(value) == frozenset({"shoe", "athletic"})


def test_category_term_set_structurally_flattens_mappings_and_decodes_bytes():
    key = (item for item in ("Department",))
    value = MappingProxyType({key: b"Watches"})

    assert category_term_set(value) == frozenset({"department", "watch"})
    assert category_term_set("Clothing and Shoes for Men") == frozenset(
        {"clothing", "shoe", "men"}
    )


@pytest.mark.parametrize(
    ("plural", "singular"),
    (
        ("booties", "bootie"),
        ("hoodies", "hoodie"),
        ("panties", "panty"),
        ("ties", "tie"),
    ),
)
def test_category_plural_exceptions_are_preserved(plural, singular):
    assert normalize_category_value(plural) == singular


def test_watch_and_watches_share_a_category_term():
    assert normalize_category_value("watch") == "watch"
    assert normalize_category_value("watches") == "watch"
    assert category_term_set(("Watch", "Watches")) == frozenset({"watch"})


def test_searchable_term_set_uses_all_searchable_fields():
    product = {
        "title": "Everyday Hoodie",
        "features": ["Machine washable"],
        "description": ["A durable layer"],
    }

    assert searchable_term_set(product) == frozenset(terms(searchable_fields(product)))
