import time

import pytest

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


def test_retrieval_skips_fallback_when_dense_results_fill_desired_candidates(
    fixture_catalog_path, monkeypatch
):
    class DenseAll:
        available = True

        def search(self, _text, _limit):
            return [Candidate(identifier) for identifier in index.valid_ids]

    index = CatalogIndex(fixture_catalog_path)
    retriever = HybridRetriever(index, DenseAll())
    plan = RetrievalPlan(
        route="buying",
        query_text="",
        source_weights=(("dense", 1.0),),
        candidate_limit=4,
    )
    monkeypatch.setattr(
        retriever,
        "_fallback_ids",
        lambda _plan: (_ for _ in ()).throw(AssertionError("fallback was called")),
    )

    candidates = retriever.retrieve(plan)

    assert {candidate.parent_asin for candidate in candidates} == index.valid_ids


def test_retrieval_preserves_extra_exact_fallback_above_desired() -> None:
    identifiers = [f"P{index:02d}" for index in range(11)]

    class Catalog:
        def __init__(self) -> None:
            self.valid_ids = set(identifiers)
            self.quality = {identifier: 1.0 for identifier in identifiers}

        def search_lexical(self, _plan, *, limit):
            return [Candidate(identifier) for identifier in identifiers[:limit]][:10]

        def popular_ids(self, _limit):
            return [identifiers[-1]]

        def product(self, _identifier):
            return {}

    plan = RetrievalPlan(
        route="buying",
        query_text="query",
        source_weights=(("lexical", 1.0),),
        candidate_limit=11,
    )

    candidates = HybridRetriever(Catalog()).retrieve(plan)

    assert [candidate.parent_asin for candidate in candidates] == identifiers


@pytest.mark.parametrize(
    ("plan", "exclude_ids", "expected_ids"),
    (
        (
            RetrievalPlan(route="buying", query_text="unmatched query"),
            (),
            ("SHOE1", "JACKET1", "DRESS1", "BELT1"),
        ),
        (
            RetrievalPlan(
                route="buying",
                query_text="purple shoes",
                hard_constraints=(_hard_constraint("color", "purple"),),
            ),
            (),
            ("SHOE1", "JACKET1", "DRESS1", "BELT1"),
        ),
        (
            RetrievalPlan(route="buying", query_text="unmatched query"),
            ("SHOE1", "JACKET1", "DRESS1", "BELT1"),
            ("SHOE1", "JACKET1", "DRESS1", "BELT1"),
        ),
    ),
    ids=("fallback", "relaxation", "exhausted-exclusion"),
)
def test_retrieval_computes_fallback_once_without_changing_result_order(
    fixture_catalog_path, monkeypatch, plan, exclude_ids, expected_ids
):
    index = CatalogIndex(fixture_catalog_path)
    retriever = HybridRetriever(index)
    fallback_ids = retriever._fallback_ids
    calls = 0

    def counted_fallback(current_plan):
        nonlocal calls
        calls += 1
        return fallback_ids(current_plan)

    monkeypatch.setattr(retriever, "_fallback_ids", counted_fallback)

    candidates = retriever.retrieve(plan, exclude_ids=exclude_ids)

    assert calls == 1
    assert tuple(candidate.parent_asin for candidate in candidates) == expected_ids


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


def test_expired_deadline_skips_available_dense_backend_and_keeps_candidates(
    fixture_catalog_path,
):
    class CountingDense:
        available = True

        def __init__(self) -> None:
            self.calls = 0

        def search(self, _text, _limit):
            self.calls += 1
            return [Candidate("JACKET1")]

    index = CatalogIndex(fixture_catalog_path)
    dense = CountingDense()
    diagnostics: list[str] = []
    candidates = HybridRetriever(index, dense).retrieve(
        RetrievalPlan(route="browsing", query_text="shoes"),
        deadline=time.perf_counter() - 1,
        diagnostics=diagnostics,
    )

    assert dense.calls == 0
    assert candidates
    assert {item.parent_asin for item in candidates} <= index.valid_ids
    assert diagnostics == ["dense_budget"]


@pytest.mark.parametrize("deadline", (None, float("inf")))
def test_available_dense_backend_runs_without_an_expired_deadline(
    fixture_catalog_path, deadline
):
    class CountingDense:
        available = True

        def __init__(self) -> None:
            self.calls = 0

        def search(self, _text, _limit):
            self.calls += 1
            return [Candidate("JACKET1")]

    index = CatalogIndex(fixture_catalog_path)
    dense = CountingDense()
    diagnostics: list[str] = []

    HybridRetriever(index, dense).retrieve(
        RetrievalPlan(route="browsing", query_text="shoes"),
        deadline=deadline,
        diagnostics=diagnostics,
    )

    assert dense.calls == 1
    assert diagnostics == []


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


def test_category_filter_candidates_and_fallback_share_semantics(
    fixture_catalog_path, monkeypatch
):
    index = CatalogIndex(fixture_catalog_path)
    retriever = HybridRetriever(index)
    category = _hard_constraint("category", "athletic shoes")
    plan = RetrievalPlan(
        route="buying",
        query_text="athletic shoes",
        hard_filters={"category": ("athletic shoes",)},
        hard_constraints=(category,),
    )
    monkeypatch.setattr(index, "popular_ids", lambda _limit: ["DRESS1", "SHOE1"])

    assert retriever._exact_ids(["SHOE1", "DRESS1"], plan) == ["SHOE1"]
    assert retriever._attribute_candidates(plan, 10) == ["SHOE1"]
    assert retriever._fallback_ids(plan)[0] == "SHOE1"


def test_category_not_in_candidate_ids_use_semantic_category_union(
    fixture_catalog_path,
):
    index = CatalogIndex(fixture_catalog_path)
    retriever = HybridRetriever(index)
    excluded = _hard_constraint("category", "athletic shoes", operator="not_in")

    assert retriever._ids_for_constraint(excluded) == {
        "BELT1",
        "DRESS1",
        "JACKET1",
    }


def test_retrieval_records_positive_weight_deduplicated_source_ranks_and_pre_rank(
    fixture_catalog_path, monkeypatch
):
    class DenseResults:
        available = True

        def search(self, _text, _limit):
            return [Candidate("SHOE1"), Candidate("DRESS1"), Candidate("SHOE1")]

    index = CatalogIndex(fixture_catalog_path)
    retriever = HybridRetriever(index, DenseResults())
    monkeypatch.setattr(
        index,
        "search_lexical",
        lambda _plan, *, limit: [
            Candidate("DRESS1"),
            Candidate("DRESS1"),
            Candidate("SHOE1"),
        ],
    )
    plan = RetrievalPlan(
        route="browsing",
        query_text="shoes",
        source_weights=(("lexical", 0.30), ("dense", 0.45), ("profile", 0.0)),
        candidate_limit=4,
    )

    candidates = retriever.retrieve(plan)
    by_id = {item.parent_asin: item for item in candidates}

    assert by_id["SHOE1"].source_ranks == {"lexical": 2, "dense": 1}
    assert by_id["DRESS1"].source_ranks == {"lexical": 1, "dense": 2}
    assert "profile" not in by_id["SHOE1"].source_ranks
    assert [item.pre_rank for item in candidates[:2]] == [1, 2]


def test_dense_rescue_skips_dense_when_positive_non_dense_evidence_exists(
    fixture_catalog_path, monkeypatch
):
    class DenseMustNotRun:
        available = True

        def search(self, _text, _limit):
            raise AssertionError("dense must not run when lexical evidence exists")

    index = CatalogIndex(fixture_catalog_path)
    retriever = HybridRetriever(index, DenseMustNotRun(), dense_rescue_only=True)
    monkeypatch.setattr(
        index,
        "search_lexical",
        lambda _plan, *, limit: [Candidate("DRESS1")],
    )
    plan = RetrievalPlan(
        route="browsing",
        query_text="dress",
        source_weights=(("lexical", 0.30), ("dense", 0.45)),
        candidate_limit=4,
    )

    candidates = retriever.retrieve(plan)

    assert candidates[0].parent_asin == "DRESS1"
    assert candidates[0].source_ranks == {"lexical": 1}


def test_dense_rescue_runs_dense_when_all_positive_non_dense_sources_are_empty(
    fixture_catalog_path, monkeypatch
):
    calls = 0

    class DenseRescue:
        available = True

        def search(self, _text, _limit):
            nonlocal calls
            calls += 1
            return [Candidate("SHOE1")]

    index = CatalogIndex(fixture_catalog_path)
    retriever = HybridRetriever(index, DenseRescue(), dense_rescue_only=True)
    monkeypatch.setattr(index, "search_lexical", lambda _plan, *, limit: [])
    plan = RetrievalPlan(
        route="browsing",
        query_text="unseen semantic phrase",
        source_weights=(("lexical", 0.30), ("dense", 0.45)),
        candidate_limit=4,
    )

    candidates = retriever.retrieve(plan)

    assert calls == 1
    assert candidates[0].parent_asin == "SHOE1"
    assert candidates[0].source_ranks == {"dense": 1}


def test_dense_rescue_ignores_zero_weight_non_dense_evidence(
    fixture_catalog_path, monkeypatch
):
    calls = 0

    class DenseRescue:
        available = True

        def search(self, _text, _limit):
            nonlocal calls
            calls += 1
            return [Candidate("SHOE1")]

    index = CatalogIndex(fixture_catalog_path)
    retriever = HybridRetriever(index, DenseRescue(), dense_rescue_only=True)
    monkeypatch.setattr(
        index,
        "search_lexical",
        lambda _plan, *, limit: [Candidate("DRESS1")],
    )
    plan = RetrievalPlan(
        route="browsing",
        query_text="semantic phrase",
        source_weights=(("lexical", 0.0), ("dense", 1.0)),
        candidate_limit=4,
    )

    candidates = retriever.retrieve(plan)

    assert calls == 1
    assert candidates[0].parent_asin == "SHOE1"


def test_retrieval_keeps_zero_weight_rrf_behavior_without_recording_evidence(
    fixture_catalog_path, monkeypatch
):
    index = CatalogIndex(fixture_catalog_path)
    retriever = HybridRetriever(index)
    monkeypatch.setattr(
        index,
        "search_lexical",
        lambda _plan, *, limit: [Candidate("DRESS1"), Candidate("SHOE1")],
    )
    monkeypatch.setattr(
        retriever, "_attribute_candidates", lambda _plan, _limit: ["SHOE1"]
    )
    plan = RetrievalPlan(
        route="buying",
        query_text="shoes",
        source_weights=(("lexical", 1.0), ("attribute", 0.0)),
        candidate_limit=4,
    )

    candidates = retriever.retrieve(plan)
    by_id = {item.parent_asin: item for item in candidates}

    assert by_id["DRESS1"].source_ranks == {"lexical": 1}
    assert by_id["SHOE1"].source_ranks == {"lexical": 2}
    assert by_id["SHOE1"].pre_rank == 2


def test_retrieval_fallback_candidates_have_no_rank_evidence(fixture_catalog_path):
    candidates = HybridRetriever(CatalogIndex(fixture_catalog_path)).retrieve(
        RetrievalPlan(route="buying", query_text="unmatched query")
    )

    fallback = [item for item in candidates if not item.source_scores]
    assert fallback
    assert all(item.source_ranks == {} for item in fallback)
    assert all(item.pre_rank is None for item in fallback)
