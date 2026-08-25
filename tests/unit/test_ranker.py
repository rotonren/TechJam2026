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
