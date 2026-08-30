from collections import deque
from types import MappingProxyType

import pytest

from compasscart.normalization import (
    category_term_set,
    extract_attributes,
    extract_layered_attributes,
    infer_category_scope,
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


def test_material_can_be_represented_in_core_and_component_specific_layers():
    product = {
        "title": "Running shoes",
        "categories": ["Shoes", "Athletic"],
        "features": ["100% Synthetic upper"],
    }
    attributes = extract_attributes(product)
    layers = extract_layered_attributes(product, core_attributes=attributes)

    assert "synthetic" in attributes["material"]
    assert layers["category"]["outer_material"] == ("synthetic",)


def test_extracts_catalog_detail_attributes_without_a_fixed_product_schema():
    product = {
        "title": "Trail shoe with a rubber sole and synthetic upper",
        "categories": ["Shoes", "Athletic"],
        "details": {
            "Closure Type": "Zipper",
            "Fit Type": "Regular Fit",
            "Pattern": "Geometric",
            "Date First Available": "January 1, 2026",
        },
    }
    core = extract_attributes(product)
    layers = extract_layered_attributes(product, core_attributes=core)
    attributes = layers["category"]

    assert attributes["closure"] == ("zipper",)
    assert attributes["fit"] == ("regular fit",)
    assert attributes["pattern"] == ("geometric",)
    assert attributes["sole_material"] == ("rubber",)
    assert attributes["outer_material"] == ("synthetic",)
    assert "date_first_available" not in attributes


def test_category_specific_fields_are_scoped_by_product_family():
    footwear = {
        "categories": ["Shoes"],
        "details": {"Sole Material": "Rubber", "Sleeve Type": "Long Sleeve"},
    }
    apparel = {
        "categories": ["Clothing", "Shirts"],
        "details": {"Sole Material": "Rubber", "Sleeve Type": "Long Sleeve"},
    }

    assert infer_category_scope(footwear) == "footwear"
    assert infer_category_scope(apparel) == "apparel"
    assert "sole_material" in extract_layered_attributes(footwear)["category"]
    assert "sleeve" not in extract_layered_attributes(footwear)["category"]
    assert "sleeve" in extract_layered_attributes(apparel)["category"]
    assert "sole_material" not in extract_layered_attributes(apparel)["category"]


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


@pytest.mark.parametrize(
    ("singular", "plural", "expected"),
    (
        ("watch", "watches", "watch"),
        ("blouse", "blouses", "blouse"),
        ("case", "cases", "case"),
        ("purse", "purses", "purse"),
        ("bootie", "booties", "bootie"),
        ("hoodie", "hoodies", "hoodie"),
        ("tie", "ties", "tie"),
    ),
)
def test_category_singular_and_plural_forms_share_a_term(
    singular, plural, expected
):
    assert normalize_category_value(singular) == expected
    assert normalize_category_value(plural) == expected


def test_searchable_term_set_uses_all_searchable_fields():
    product = {
        "title": "Everyday Hoodie",
        "features": ["Machine washable"],
        "description": ["A durable layer"],
    }

    assert searchable_term_set(product) == frozenset(terms(searchable_fields(product)))
