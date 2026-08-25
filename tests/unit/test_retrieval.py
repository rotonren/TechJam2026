from compasscart.catalog import CatalogIndex
from compasscart.models import Candidate, Constraint, RetrievalPlan, SessionState
from compasscart.retrieval import HybridRetriever, reciprocal_rank_fusion


def _hard_constraint(
    attribute: str, value: str, *, operator: str = "eq", upper_value: str | None = None
) -> Constraint:
    return Constraint(
        attribute,
        value,
        1.0,
        True,
        "message",
        1,
        1,
        operator=operator,
        upper_value=upper_value,
    )


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


def test_legacy_hard_filters_mark_nonmatching_fallbacks_as_relaxed(
    fixture_catalog_path,
):
    index = CatalogIndex(fixture_catalog_path)
    plan = RetrievalPlan(
        route="buying",
        query_text="durable work accessory",
        hard_filters={"material": ("leather",)},
        source_weights=(("attribute", 0.45), ("lexical", 0.35)),
    )

    candidates = HybridRetriever(index).retrieve(plan)
    exact = [candidate for candidate in candidates if not candidate.relaxed]
    relaxed = [candidate for candidate in candidates if candidate.relaxed]

    assert [candidate.parent_asin for candidate in exact] == ["BELT1"]
    assert relaxed
    assert all(candidate.violations == ("material=leather",) for candidate in relaxed)


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


def test_retrieval_filters_dense_candidates_then_appends_relaxed_alternatives(
    fixture_catalog_path,
):
    class DenseOnly:
        available = True

        def search(self, _text, _limit):
            return [Candidate("JACKET1")]

    index = CatalogIndex(fixture_catalog_path)
    budget = _hard_constraint("budget", "50", operator="lte")
    plan = RetrievalPlan(
        route="buying",
        query_text="jacket",
        hard_filters={"budget": ("50",)},
        hard_constraints=(budget,),
        source_weights=(("dense", 1.0),),
    )

    candidates = HybridRetriever(index, DenseOnly()).retrieve(plan)
    exact = [item for item in candidates if not item.relaxed]
    relaxed = [item for item in candidates if item.relaxed]

    assert exact
    assert all(float(item.product["price"]) <= 50.0 for item in exact)
    assert all(item.violations for item in relaxed)
    assert all(item.relaxed for item in candidates[len(exact) :])
    assert "JACKET1" not in {item.parent_asin for item in exact}


def test_retrieval_excludes_requested_ids_before_filling_exact_results(
    fixture_catalog_path,
):
    index = CatalogIndex(fixture_catalog_path)
    plan = RetrievalPlan(
        route="buying",
        query_text="shoes",
        hard_constraints=(),
        source_weights=(("attribute", 1.0),),
        candidate_limit=10,
    )

    candidates = HybridRetriever(index).retrieve(
        plan, exclude_ids={"SHOE1"}
    )

    assert candidates
    assert "SHOE1" not in {item.parent_asin for item in candidates}


def test_attribute_candidate_lookup_does_not_scan_every_catalog_item(
    fixture_catalog_path,
):
    index = CatalogIndex(fixture_catalog_path)
    retriever = HybridRetriever(index)
    plan = RetrievalPlan(
        route="buying",
        query_text="blue shoes",
        hard_constraints=(_hard_constraint("color", "blue"),),
        hard_filters={"color": ("blue",)},
    )

    def fail_on_full_scan(*_args, **_kwargs):
        raise AssertionError("attribute lookup should use inverted indexes")

    retriever._exact_ids = fail_on_full_scan

    assert retriever._attribute_candidates(plan, 10) == ["SHOE1"]
