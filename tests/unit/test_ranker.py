import pytest

from compasscart.catalog import CatalogIndex
from compasscart.models import Candidate, Constraint, SessionState
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


def test_diversity_terms_cache_reuses_values_and_evicts_least_recently_used(
    fixture_catalog_path,
):
    ranker = ConstraintRanker(CatalogIndex(fixture_catalog_path))

    initial = ranker._diversity_terms("SHOE1")
    assert ranker._diversity_terms("SHOE1") is initial

    ranker._diversity_terms("DRESS1")
    ranker._diversity_terms("SHOE1")
    assert list(ranker._diversity_cache)[-1] == "SHOE1"

    for index in range(4_096):
        ranker._diversity_terms(f"MISSING{index}")

    assert len(ranker._diversity_cache) == 4_096
    assert "SHOE1" not in ranker._diversity_cache


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


@pytest.mark.parametrize("fusion_weight", [-0.01, 0.11])
def test_ranker_rejects_out_of_range_fusion_weight(
    fixture_catalog_path, fusion_weight
):
    index = CatalogIndex(fixture_catalog_path)

    with pytest.raises(ValueError, match="fusion_weight"):
        ConstraintRanker(index, fusion_weight=fusion_weight)
