import pytest

import compasscart.catalog as catalog_module
from compasscart.catalog import CatalogIndex
from compasscart.models import Constraint, RetrievalPlan
from compasscart.normalization import searchable_fields, searchable_term_set, terms


def test_catalog_returns_valid_unique_matches(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    plan = RetrievalPlan(route="buying", query_text="blue running shoes")

    matches = index.search_lexical(plan, limit=10)

    assert matches[0].parent_asin == "SHOE1"
    assert len({item.parent_asin for item in matches}) == len(matches)
    assert all(item.parent_asin in index.valid_ids for item in matches)


@pytest.mark.parametrize("enable_fts", (True, False))
def test_catalog_does_not_build_a_duplicate_field_term_index(
    fixture_catalog_path, enable_fts
):
    index = CatalogIndex(fixture_catalog_path, enable_fts=enable_fts)

    assert index.field_terms == {}


def test_attribute_lookup_filters_material(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)

    assert index.attribute_ids("material", "leather") == {"BELT1"}


@pytest.mark.parametrize("enable_fts", (True, False))
def test_catalog_builds_shared_semantic_caches_independent_of_fts(
    fixture_catalog_path, enable_fts
):
    index = CatalogIndex(fixture_catalog_path, enable_fts=enable_fts)
    product = index.product("SHOE1")

    assert index.category_terms["SHOE1"] == frozenset({"shoe", "athletic"})
    assert index.searchable_terms["SHOE1"] == searchable_term_set(product)
    assert index.category_term_inverted["shoe"] == {"SHOE1"}
    assert index.category_ids("athletic shoes") == {"SHOE1"}


def test_catalog_searchable_cache_reuses_token_objects(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)

    shoe_token = next(
        token for token in index.searchable_terms["SHOE1"] if token == "clothing"
    )
    dress_token = next(
        token for token in index.searchable_terms["DRESS1"] if token == "clothing"
    )
    assert shoe_token is dress_token


def test_catalog_searchable_cache_uses_compact_term_ids(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    cached = index.searchable_terms["SHOE1"]

    assert cached == searchable_term_set(index.product("SHOE1"))
    assert cached.storage_bytes == len(cached) * 4


def test_catalog_searchable_cache_preserves_frozenset_interface(
    fixture_catalog_path,
):
    index = CatalogIndex(fixture_catalog_path)
    cached = index.searchable_terms["SHOE1"]
    expected = searchable_term_set(index.product("SHOE1"))
    other = frozenset({"blue", "dress", "missing"})

    assert "blue" in cached
    assert "dress" in index.searchable_terms["DRESS1"]
    assert "dress" not in cached
    assert "missing" not in cached
    assert frozenset(iter(cached)) == expected
    assert len(cached) == len(expected)
    assert cached == expected
    assert expected == cached

    binary_results = (
        (cached & other, expected & other),
        (cached | other, expected | other),
        (cached - other, expected - other),
        (cached ^ other, expected ^ other),
    )
    for actual, reference in binary_results:
        assert type(actual) is frozenset
        assert actual == reference

    assert cached.isdisjoint({"dress", "missing"})
    assert cached.issubset(expected | {"missing"})
    assert cached.issuperset({"blue", "mesh"})

    named_results = (
        (cached.union(other), expected.union(other)),
        (cached.intersection(other), expected.intersection(other)),
        (cached.difference(other), expected.difference(other)),
        (
            cached.symmetric_difference(other),
            expected.symmetric_difference(other),
        ),
    )
    for actual, reference in named_results:
        assert type(actual) is frozenset
        assert actual == reference


def test_catalog_matches_and_violations_use_shared_semantics(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    matching = Constraint("category", "athletic shoes", 1.0, True, "message", 1, 1)
    missing = Constraint("category", "formal shoes", 1.0, True, "message", 1, 1)

    assert index.matches("SHOE1", matching)
    assert index.violations("SHOE1", (matching, missing)) == (
        "category=formal shoes",
    )


def test_python_fallback_preserves_best_match(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path, enable_fts=False)
    plan = RetrievalPlan(route="browsing", query_text="waterproof winter jacket")

    matches = index.search_lexical(plan, limit=10)

    assert matches[0].parent_asin == "JACKET1"


def test_python_fallback_uses_compact_masks_without_retokenizing_products(
    fixture_catalog_path, monkeypatch
):
    index = CatalogIndex(fixture_catalog_path, enable_fts=False)
    plan = RetrievalPlan(
        route="buying",
        query_text="blue running shoes",
        hard_filters={"color": ("blue",)},
    )
    query = set(terms(plan.query_text))
    weights = (6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
    reference: list[tuple[float, float, str]] = []
    for parent_asin, product in index.products.items():
        if not index._matches_hard(parent_asin, plan):
            continue
        score = sum(
            weight * len(query.intersection(terms(field)))
            for weight, field in zip(weights, searchable_fields(product), strict=True)
        )
        if score:
            reference.append((score, index.quality[parent_asin], parent_asin))
    reference.sort(key=lambda item: (-item[0], -item[1], item[2]))

    assert isinstance(index.field_masks["SHOE1"], bytes)
    assert len(index.field_masks["SHOE1"]) == len(index.searchable_terms["SHOE1"])
    monkeypatch.setattr(
        catalog_module,
        "searchable_fields",
        lambda product: pytest.fail("fallback retokenized product fields"),
    )

    matches = index.search_lexical(plan, limit=10)

    assert [item.parent_asin for item in matches] == [
        parent_asin for _, _, parent_asin in reference[:10]
    ]


def test_parser_vocabulary_is_normalized_read_only_and_filters_generic_categories(
    fixture_catalog_path,
):
    vocabulary = CatalogIndex(fixture_catalog_path).parser_vocabulary()

    assert vocabulary["brand"] == tuple(sorted(set(vocabulary["brand"])))
    assert {"acme fashion", "fleet shoes"} <= set(vocabulary["brand"])
    assert {"medium", "10 wide"} <= set(vocabulary["size"])
    assert "dresses" in vocabulary["category"]
    assert not {"clothing", "men", "women", "shoes and jewelry"}.intersection(
        vocabulary["category"]
    )
    with pytest.raises(TypeError):
        vocabulary["brand"] = ()
