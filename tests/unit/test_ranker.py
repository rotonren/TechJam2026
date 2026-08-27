import time

import pytest

from compasscart.catalog import CatalogIndex
from compasscart.models import Candidate, Constraint, SessionState
from compasscart.normalization import terms
from compasscart.ranker import ConstraintRanker


def _candidates(index: CatalogIndex, identifiers: list[str]) -> list[Candidate]:
    return [
        Candidate(parent_asin=item, product=index.product(item), score=1.0)
        for item in identifiers
    ]


def test_explicit_hard_match_beats_conflicting_product(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    state = SessionState(
        "s1",
        route="buying",
        constraints=[Constraint("color", "blue", 1.0, True, "message", 1, 1)],
    )

    ranked = ConstraintRanker(index).rank(
        _candidates(index, ["DRESS1", "SHOE1"]), state
    )

    assert ranked[0].parent_asin == "SHOE1"


def test_semantic_category_match_beats_conflicting_product(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    index.quality.update({"DRESS1": 0.0, "SHOE1": 0.0})
    index.attributes["SHOE1"]["category"] = ()
    state = SessionState(
        "s1",
        route="buying",
        constraints=[
            Constraint("category", "athletic shoes", 1.0, True, "message", 1, 1)
        ],
    )

    ranked = ConstraintRanker(index).rank(
        _candidates(index, ["DRESS1", "SHOE1"]), state
    )

    assert ranked[0].parent_asin == "SHOE1"


def test_message_constraint_outweighs_profile_preference(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    state = SessionState(
        "s1",
        route="buying",
        constraints=[
            Constraint("color", "red", 0.25, False, "profile", 0, 1),
            Constraint("color", "blue", 1.0, True, "message", 1, 1),
        ],
    )

    ranked = ConstraintRanker(index).rank(
        _candidates(index, ["DRESS1", "SHOE1"]), state
    )

    assert ranked[0].parent_asin == "SHOE1"


def test_browsing_mmr_adds_category_diversity(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    candidates = _candidates(index, ["DRESS1", "SHOE1", "BELT1", "JACKET1"])
    for position, candidate in enumerate(candidates):
        candidate.score = 1.0 - position * 0.01
    state = SessionState("s1", route="browsing")

    ranked = ConstraintRanker(index).rank(candidates, state, top_k=3)

    assert ranked[0].parent_asin == "DRESS1"
    assert (
        len({index.attributes[item.parent_asin]["category"][-1] for item in ranked})
        == 3
    )


def test_browsing_only_applies_mmr_to_final_top10(fixture_catalog_path):
    class CountingRanker(ConstraintRanker):
        calls = 0

        def _similarity(self, left: str, right: str) -> float:
            self.calls += 1
            return super()._similarity(left, right)

    index = CatalogIndex(fixture_catalog_path)
    candidates = [
        Candidate(parent_asin=f"FAKE{position:02d}", score=40.0 - position)
        for position in range(40)
    ]
    ranker = CountingRanker(index)

    ranked = ranker.rank(candidates, SessionState("s1", route="browsing"))

    assert len(ranked) == 40
    assert ranker.calls < 2_000


def test_expired_deadline_skips_browsing_mmr_without_mixing_relaxed_candidates(
    fixture_catalog_path, monkeypatch
):
    index = CatalogIndex(fixture_catalog_path)
    ranker = ConstraintRanker(index)
    candidates = [
        Candidate("JACKET1", product=index.product("JACKET1"), score=10.0, relaxed=True),
        Candidate("BELT1", product=index.product("BELT1"), score=0.1),
    ]
    diagnostics: list[str] = []
    monkeypatch.setattr(
        ranker,
        "_mmr",
        lambda *_args: (_ for _ in ()).throw(AssertionError("MMR was called")),
    )

    ranked = ranker.rank(
        candidates,
        SessionState("s1", route="browsing"),
        top_k=2,
        deadline=time.perf_counter() - 1,
        diagnostics=diagnostics,
    )

    assert [item.parent_asin for item in ranked] == ["BELT1", "JACKET1"]
    assert diagnostics == ["mmr_budget"]


def test_expired_deadline_does_not_record_mmr_skip_for_buying(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    diagnostics: list[str] = []

    ranked = ConstraintRanker(index).rank(
        _candidates(index, ["DRESS1", "SHOE1"]),
        SessionState("s1", route="buying"),
        deadline=time.perf_counter() - 1,
        diagnostics=diagnostics,
    )

    assert ranked
    assert diagnostics == []


def test_diversity_terms_cache_reuses_values_and_evicts_least_recently_used(
    fixture_catalog_path,
):
    catalog = CatalogIndex(fixture_catalog_path)
    ranker = ConstraintRanker(catalog)

    initial = ranker._diversity_terms("SHOE1")
    attributes = catalog.attributes["SHOE1"]
    expected = frozenset(
        token
        for field in ("category", "material", "style", "use_case")
        for value in attributes[field]
        for token in terms(value)
    )
    assert isinstance(initial, frozenset)
    assert initial == expected
    assert ranker._diversity_terms("SHOE1") is initial

    ranker._diversity_terms("DRESS1")
    ranker._diversity_terms("SHOE1")
    assert list(ranker._diversity_cache)[-1] == "SHOE1"

    for index in range(4_094):
        ranker._diversity_terms(f"MISSING{index}")

    assert len(ranker._diversity_cache) == 4_096
    ranker._diversity_terms("OVERFLOW")
    assert "DRESS1" not in ranker._diversity_cache
    assert "SHOE1" in ranker._diversity_cache


def test_exact_candidates_rank_before_higher_scoring_relaxed_candidates(
    fixture_catalog_path,
):
    index = CatalogIndex(fixture_catalog_path)
    state = SessionState("s1", route="buying")
    candidates = [
        Candidate("JACKET1", product=index.product("JACKET1"), score=10.0, relaxed=True),
        Candidate("BELT1", product=index.product("BELT1"), score=0.1),
    ]

    ranked = ConstraintRanker(index).rank(candidates, state)

    assert [item.parent_asin for item in ranked] == ["BELT1", "JACKET1"]
    assert ranked[1].relaxed is True


def test_fusion_reorders_exact_candidates_without_promoting_relaxed(
    fixture_catalog_path,
):
    index = CatalogIndex(fixture_catalog_path)
    index.quality["DRESS1"] = 0.0
    index.quality["SHOE1"] = 0.0
    index.quality["JACKET1"] = 0.0
    candidates = [
        Candidate(
            "DRESS1",
            index.product("DRESS1"),
            {"lexical": 0.08, "attribute": 0.8},
            0.9,
        ),
        Candidate(
            "SHOE1", index.product("SHOE1"), {"lexical": 0.1}, 0.1
        ),
        Candidate(
            "JACKET1",
            index.product("JACKET1"),
            {"lexical": 0.1, "attribute": 0.9},
            1.0,
            relaxed=True,
        ),
    ]
    state = SessionState("s1", route="buying")

    baseline = ConstraintRanker(index, fusion_weight=0.0).rank(candidates, state)
    experiment = ConstraintRanker(index, fusion_weight=0.10).rank(candidates, state)

    assert [item.parent_asin for item in baseline] == [
        "SHOE1",
        "DRESS1",
        "JACKET1",
    ]
    assert [item.parent_asin for item in experiment] == [
        "DRESS1",
        "SHOE1",
        "JACKET1",
    ]
    assert experiment[0].score > experiment[1].score
    assert experiment[-1].relaxed is True


@pytest.mark.parametrize("fusion_weight", [-0.01, 0.05, 0.11, 0.16])
def test_ranker_rejects_out_of_range_fusion_weight(
    fixture_catalog_path, fusion_weight
):
    index = CatalogIndex(fixture_catalog_path)

    with pytest.raises(ValueError, match="fusion_weight"):
        ConstraintRanker(index, fusion_weight=fusion_weight)


@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("attribute_weight", True),
        ("attribute_weight", 0.025),
        ("attribute_weight", float("nan")),
        ("consensus_bonus", True),
        ("consensus_bonus", 0.01),
        ("consensus_bonus", float("inf")),
        ("boundary_bonus", True),
        ("boundary_bonus", 0.01),
        ("boundary_bonus", float("-inf")),
    ),
)
def test_ranker_rejects_non_predeclared_feature_values(
    fixture_catalog_path, argument, value
):
    with pytest.raises((TypeError, ValueError), match=argument):
        ConstraintRanker(
            CatalogIndex(fixture_catalog_path), **{argument: value}
        )


@pytest.mark.parametrize(
    "argument",
    (
        "fusion_weight",
        "attribute_weight",
        "consensus_bonus",
        "boundary_bonus",
    ),
)
def test_ranker_rejects_false_as_a_numeric_weight(fixture_catalog_path, argument):
    with pytest.raises((TypeError, ValueError), match=argument):
        ConstraintRanker(CatalogIndex(fixture_catalog_path), **{argument: False})


@pytest.mark.parametrize("value", (True, 0.0, 1.0, float("nan")))
def test_ranker_has_no_mmr_lambda_one_path(fixture_catalog_path, value):
    with pytest.raises((TypeError, ValueError), match="mmr_lambda"):
        ConstraintRanker(CatalogIndex(fixture_catalog_path), mmr_lambda=value)


def test_ranker_preserves_candidate_evidence_fields(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    candidate = Candidate(
        "SHOE1",
        product=index.product("SHOE1"),
        source_scores={"lexical": 0.2},
        source_ranks={"lexical": 2},
        pre_rank=3,
    )

    ranked = ConstraintRanker(index).rank(
        [candidate], SessionState("s1", route="buying")
    )

    assert ranked[0].source_ranks == {"lexical": 2}
    assert ranked[0].pre_rank == 3


def test_attribute_evidence_uses_normalized_source_and_exact_source_budget(
    fixture_catalog_path,
):
    index = CatalogIndex(fixture_catalog_path)
    index.quality.update({"DRESS1": 0.0, "SHOE1": 0.0})
    candidates = [
        Candidate(
            "DRESS1",
            index.product("DRESS1"),
            {"lexical": 1.0, "dense": 1.0},
            1.0,
        ),
        Candidate(
            "SHOE1",
            index.product("SHOE1"),
            {"lexical": 0.0, "dense": 0.0, "attribute": 2.0},
            0.0,
        ),
    ]

    ranked = ConstraintRanker(
        index,
        fusion_weight=0.10,
        attribute_weight=0.10,
    ).rank(candidates, SessionState("s1", route="buying"))

    assert ranked[0].parent_asin == "DRESS1"
    scores = {item.parent_asin: item.score for item in ranked}
    assert scores["DRESS1"] == pytest.approx(0.30)
    assert scores["SHOE1"] == pytest.approx(0.10)


def test_consensus_requires_two_positive_non_profile_sources_with_lexical_or_attribute(
    fixture_catalog_path,
):
    index = CatalogIndex(fixture_catalog_path)
    for identifier in index.valid_ids:
        index.quality[identifier] = 0.0
    candidates = [
        Candidate("BELT1", source_ranks={"dense": 1, "profile": 1}),
        Candidate("DRESS1", source_ranks={"dense": 1, "other": 1}),
        Candidate("JACKET1", source_ranks={"lexical": 1, "profile": 1}),
        Candidate("SHOE1", source_ranks={"lexical": 1, "dense": 1}),
    ]

    ranked = ConstraintRanker(
        index,
        fusion_weight=0.10,
        consensus_bonus=0.05,
    ).rank(candidates, SessionState("s1", route="buying"))
    scores = {item.parent_asin: item.score for item in ranked}

    assert scores["SHOE1"] == pytest.approx(0.05)
    assert scores["BELT1"] == pytest.approx(0.0)
    assert scores["DRESS1"] == pytest.approx(0.0)
    assert scores["JACKET1"] == pytest.approx(0.0)


def test_consensus_and_boundary_bonuses_are_independent_and_capped(
    fixture_catalog_path,
):
    index = CatalogIndex(fixture_catalog_path)
    index.quality["SHOE1"] = 0.0
    candidate = Candidate(
        "SHOE1",
        source_ranks={"lexical": 1, "dense": 2, "attribute": 3},
        pre_rank=10,
    )

    ranked = ConstraintRanker(
        index,
        fusion_weight=0.10,
        consensus_bonus=0.05,
        boundary_bonus=0.025,
    ).rank([candidate], SessionState("s1", route="buying"))

    assert ranked[0].score == pytest.approx(0.075)


def test_boundary_feature_can_lift_base_rank_11_only_with_top10_pre_rank(
    fixture_catalog_path,
):
    index = CatalogIndex(fixture_catalog_path)
    identifiers = [f"P{position:02d}" for position in range(1, 12)]
    index.valid_ids.update(identifiers)
    index.quality.update({identifier: 0.0 for identifier in identifiers})
    candidates = [
        Candidate(identifier, score=0.50 - position * 0.01, pre_rank=position)
        for position, identifier in enumerate(identifiers[:10], start=1)
    ]
    candidates.append(
        Candidate(
            identifiers[10],
            score=0.395,
            source_ranks={"lexical": 1, "dense": 1},
            pre_rank=10,
        )
    )

    ranked = ConstraintRanker(
        index,
        fusion_weight=0.10,
        consensus_bonus=0.0,
        boundary_bonus=0.025,
    ).rank(candidates, SessionState("s1", route="buying"), top_k=10)

    assert identifiers[10] in [item.parent_asin for item in ranked]
    assert identifiers[9] not in [item.parent_asin for item in ranked]


def test_boundary_requires_exact_consensus_no_hard_conflict_and_pre_rank_at_most_10(
    fixture_catalog_path,
):
    index = CatalogIndex(fixture_catalog_path)
    index.quality.update({"BELT1": 0.0, "DRESS1": 0.0, "JACKET1": 0.0, "SHOE1": 0.0})
    state = SessionState(
        "s1",
        route="buying",
        constraints=[Constraint("color", "blue", 1.0, True, "message", 1, 1)],
    )
    candidates = [
        Candidate("SHOE1", source_ranks={"lexical": 1, "dense": 1}, pre_rank=10),
        Candidate("DRESS1", source_ranks={"lexical": 1, "dense": 1}, pre_rank=10),
        Candidate("BELT1", source_ranks={"lexical": 1, "dense": 1}, pre_rank=11),
        Candidate(
            "JACKET1",
            source_ranks={"lexical": 1, "dense": 1},
            pre_rank=10,
            relaxed=True,
        ),
    ]

    ranked = ConstraintRanker(
        index,
        fusion_weight=0.10,
        consensus_bonus=0.0,
        boundary_bonus=0.025,
    ).rank(candidates, state)
    scores = {item.parent_asin: item.score for item in ranked}

    assert scores["SHOE1"] == pytest.approx(0.325)
    assert scores["DRESS1"] == pytest.approx(-0.60)
    assert scores["BELT1"] == pytest.approx(-0.60)
    assert scores["JACKET1"] == pytest.approx(-0.60)
    assert ranked[-1].relaxed is True


def test_equal_scores_end_with_parent_asin_ascending(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    index.quality.update({"DRESS1": 0.0, "SHOE1": 0.0})

    ranked = ConstraintRanker(index, fusion_weight=0.10).rank(
        [Candidate("SHOE1"), Candidate("DRESS1")],
        SessionState("s1", route="buying"),
    )

    assert [item.parent_asin for item in ranked] == ["DRESS1", "SHOE1"]


def test_adaptive_mmr_false_preserves_current_default_browsing_mmr(
    fixture_catalog_path, monkeypatch
):
    index = CatalogIndex(fixture_catalog_path)
    ranker = ConstraintRanker(index, adaptive_browsing_mmr=False)
    calls: list[tuple[int, int]] = []

    def record(candidates, limit):
        calls.append((len(candidates), limit))
        return list(candidates[:limit])

    monkeypatch.setattr(ranker, "_mmr", record)
    candidates = [Candidate(f"P{position:02d}", score=20.0 - position) for position in range(12)]

    ranker.rank(candidates, SessionState("s1", route="browsing"), top_k=10)

    assert calls == [(12, 10)]


def test_adaptive_mmr_true_requires_all_fixed_gates(fixture_catalog_path, monkeypatch):
    index = CatalogIndex(fixture_catalog_path)
    ranker = ConstraintRanker(
        index, fusion_weight=0.10, adaptive_browsing_mmr=True
    )
    candidates = [
        Candidate(f"P{position:02d}", score=100.0 - position * 10.0)
        for position in range(9)
    ]
    candidates.extend([Candidate("P09", score=1.0), Candidate("P10", score=0.0)])
    calls: list[float] = []
    monkeypatch.setattr(ranker, "_similarity", lambda _left, _right: 0.60)

    def record(items, limit):
        calls.append(ranker.mmr_lambda)
        return list(reversed(items[:limit]))

    monkeypatch.setattr(ranker, "_mmr", record)

    first = ranker.rank(candidates, SessionState("s1", route="browsing"), top_k=10)
    second = ranker.rank(candidates, SessionState("s1", route="browsing"), top_k=10)

    assert [item.parent_asin for item in first] == [item.parent_asin for item in second]
    assert calls == [0.85, 0.85]


def test_adaptive_mmr_uses_only_attribute_term_similarity(
    fixture_catalog_path, monkeypatch
):
    index = CatalogIndex(fixture_catalog_path)
    ranker = ConstraintRanker(
        index, fusion_weight=0.10, adaptive_browsing_mmr=True
    )
    candidates = [
        Candidate(f"P{position:02d}", score=100.0 - position * 10.0)
        for position in range(9)
    ]
    candidates.extend([Candidate("P09", score=1.0), Candidate("P10", score=0.0)])
    monkeypatch.setattr(
        ranker,
        "_diversity_terms",
        lambda _identifier: frozenset({"shared", "attribute"}),
    )
    calls = 0

    def record(items, limit):
        nonlocal calls
        calls += 1
        return list(items[:limit])

    monkeypatch.setattr(ranker, "_mmr", record)

    ranker.rank(candidates, SessionState("s1", route="browsing"), top_k=10)

    assert calls == 1


def test_adaptive_mmr_gates_off_when_min_max_normalized_gap_is_too_large(
    fixture_catalog_path, monkeypatch
):
    index = CatalogIndex(fixture_catalog_path)
    ranker = ConstraintRanker(
        index, fusion_weight=0.10, adaptive_browsing_mmr=True
    )
    candidates = [
        Candidate(f"P{position:02d}", score=100.0 - position)
        for position in range(11)
    ]
    monkeypatch.setattr(ranker, "_similarity", lambda _left, _right: 0.60)
    calls = 0

    def record(items, limit):
        nonlocal calls
        calls += 1
        return list(items[:limit])

    monkeypatch.setattr(ranker, "_mmr", record)

    ranker.rank(candidates, SessionState("s1", route="browsing"), top_k=10)

    assert calls == 0


def test_adaptive_mmr_gates_on_at_min_max_normalized_gap_boundary(
    fixture_catalog_path, monkeypatch
):
    index = CatalogIndex(fixture_catalog_path)
    ranker = ConstraintRanker(
        index, fusion_weight=0.10, adaptive_browsing_mmr=True
    )
    candidates = [
        Candidate(f"P{position:02d}", score=140.0 - 4.0 * position)
        for position in range(9)
    ]
    candidates.extend([Candidate("P09", score=101.0), Candidate("P10", score=100.0)])
    monkeypatch.setattr(ranker, "_similarity", lambda _left, _right: 0.60)
    calls = 0

    def record(items, limit):
        nonlocal calls
        calls += 1
        return list(items[:limit])

    monkeypatch.setattr(ranker, "_mmr", record)

    ranker.rank(candidates, SessionState("s1", route="browsing"), top_k=10)

    assert calls == 1


@pytest.mark.parametrize(
    "candidates",
    (
        [Candidate(f"SHORT{position:02d}", score=1.0 - position * 0.001) for position in range(10)],
        [Candidate(f"GAP{position:02d}", score=1.0 if position < 10 else 0.0) for position in range(11)],
    ),
)
def test_adaptive_mmr_returns_base_order_when_size_or_score_gate_fails(
    fixture_catalog_path, monkeypatch, candidates
):
    index = CatalogIndex(fixture_catalog_path)
    ranker = ConstraintRanker(index, adaptive_browsing_mmr=True)
    monkeypatch.setattr(
        ranker,
        "_mmr",
        lambda *_args: (_ for _ in ()).throw(AssertionError("MMR was called")),
    )

    ranked = ranker.rank(candidates, SessionState("s1", route="browsing"))

    assert [item.parent_asin for item in ranked] == sorted(
        [item.parent_asin for item in candidates],
        key=lambda identifier: next(
            (-item.score, item.parent_asin)
            for item in candidates
            if item.parent_asin == identifier
        ),
    )


def test_adaptive_mmr_returns_base_order_when_duplicate_gate_fails(
    fixture_catalog_path, monkeypatch
):
    index = CatalogIndex(fixture_catalog_path)
    ranker = ConstraintRanker(index, adaptive_browsing_mmr=True)
    candidates = [Candidate(f"P{position:02d}", score=1.0 - position * 0.001) for position in range(11)]
    monkeypatch.setattr(ranker, "_similarity", lambda _left, _right: 0.59)
    monkeypatch.setattr(
        ranker,
        "_mmr",
        lambda *_args: (_ for _ in ()).throw(AssertionError("MMR was called")),
    )

    ranked = ranker.rank(candidates, SessionState("s1", route="browsing"))

    assert [item.parent_asin for item in ranked] == [item.parent_asin for item in candidates]


def test_adaptive_mmr_never_crosses_exact_and_relaxed_segments(
    fixture_catalog_path, monkeypatch
):
    index = CatalogIndex(fixture_catalog_path)
    ranker = ConstraintRanker(index, adaptive_browsing_mmr=True)
    exact = [Candidate(f"E{position:02d}", score=1.0 - position * 0.001) for position in range(11)]
    relaxed = [
        Candidate(f"R{position:02d}", score=2.0 - position * 0.001, relaxed=True)
        for position in range(11)
    ]
    monkeypatch.setattr(ranker, "_similarity", lambda _left, _right: 0.60)
    monkeypatch.setattr(ranker, "_mmr", lambda items, limit: list(reversed(items[:limit])))

    ranked = ranker.rank(exact + relaxed, SessionState("s1", route="browsing"))

    assert all(not item.relaxed for item in ranked[:11])
    assert all(item.relaxed for item in ranked[11:])
