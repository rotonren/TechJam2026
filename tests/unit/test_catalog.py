import pytest

from compasscart.catalog import CatalogIndex
from compasscart.models import RetrievalPlan


def test_catalog_returns_valid_unique_matches(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    plan = RetrievalPlan(route="buying", query_text="blue running shoes")

    matches = index.search_lexical(plan, limit=10)

    assert matches[0].parent_asin == "SHOE1"
    assert len({item.parent_asin for item in matches}) == len(matches)
    assert all(item.parent_asin in index.valid_ids for item in matches)


def test_catalog_only_builds_python_term_index_when_fts_is_unavailable(
    fixture_catalog_path,
):
    index = CatalogIndex(fixture_catalog_path)

    assert bool(index.field_terms) is not index._fts_enabled


def test_attribute_lookup_filters_material(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)

    assert index.attribute_ids("material", "leather") == {"BELT1"}


def test_python_fallback_preserves_best_match(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path, enable_fts=False)
    plan = RetrievalPlan(route="browsing", query_text="waterproof winter jacket")

    matches = index.search_lexical(plan, limit=10)

    assert matches[0].parent_asin == "JACKET1"


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
