from compasscart.catalog import CatalogIndex
from compasscart.models import Constraint, RetrievalPlan, SessionState
from compasscart.retrieval import HybridRetriever, reciprocal_rank_fusion


def test_rrf_fuses_and_deduplicates_candidates():
    fused = reciprocal_rank_fusion(
        {"lexical": ["A", "B"], "attribute": ["B", "C"]},
        weights={"lexical": 0.35, "attribute": 0.45},
        k=60,
    )

    assert set(fused) == {"A", "B", "C"}
    assert fused[0] == "B"


def test_hybrid_retrieval_uses_attributes_and_returns_catalog_fallbacks(
    fixture_catalog_path,
):
    index = CatalogIndex(fixture_catalog_path)
    state = SessionState(
        "s1",
        route="buying",
        constraints=[Constraint("material", "leather", 1.0, True, "message", 1, 1)],
    )
    plan = RetrievalPlan(
        route="buying",
        query_text="durable work accessory",
        hard_filters={"material": ("leather",)},
        source_weights=(("attribute", 0.45), ("lexical", 0.35)),
    )

    candidates = HybridRetriever(index).retrieve(plan, state)

    assert candidates[0].parent_asin == "BELT1"
    assert len(candidates) == len(index.valid_ids)
    assert len({item.parent_asin for item in candidates}) == len(candidates)


def test_retrieval_relaxes_impossible_filter_to_valid_products(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    state = SessionState(
        "s1",
        constraints=[Constraint("color", "purple", 1.0, True, "message", 1, 1)],
    )
    plan = RetrievalPlan(
        route="buying",
        query_text="purple shoes",
        hard_filters={"color": ("purple",)},
        source_weights=(("attribute", 0.45), ("lexical", 0.35)),
    )

    candidates = HybridRetriever(index).retrieve(plan, state)

    assert {item.parent_asin for item in candidates} == index.valid_ids


def test_budget_filter_uses_price_ceiling_for_lexical_search(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    plan = RetrievalPlan(
        route="buying",
        query_text="running shoes",
        hard_filters={"budget": ("80.00",)},
    )

    candidates = index.search_lexical(plan, limit=10)

    assert candidates[0].parent_asin == "SHOE1"
    assert all(float(item.product["price"]) <= 80.0 for item in candidates)
