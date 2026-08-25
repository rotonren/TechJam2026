from compasscart.catalog import CatalogIndex
from compasscart.models import RetrievalPlan


def test_catalog_returns_valid_unique_matches(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    plan = RetrievalPlan(route="buying", query_text="blue running shoes")

    matches = index.search_lexical(plan, limit=10)

    assert matches[0].parent_asin == "SHOE1"
    assert len({item.parent_asin for item in matches}) == len(matches)
    assert all(item.parent_asin in index.valid_ids for item in matches)


def test_attribute_lookup_filters_material(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)

    assert index.attribute_ids("material", "leather") == {"BELT1"}


def test_python_fallback_preserves_best_match(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path, enable_fts=False)
    plan = RetrievalPlan(route="browsing", query_text="waterproof winter jacket")

    matches = index.search_lexical(plan, limit=10)

    assert matches[0].parent_asin == "JACKET1"
