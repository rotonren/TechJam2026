from compasscart.models import Candidate, SessionState
from compasscart.question_policy import QuestionPolicy


def _candidate(
    identifier: str,
    score: float,
    *,
    material: str | None = None,
    color: str | None = None,
) -> Candidate:
    details = {}
    if material:
        details["Material"] = material
    if color:
        details["Color"] = color
    return Candidate(
        parent_asin=identifier,
        product={"parent_asin": identifier, "title": "item", "details": details},
        score=score,
    )


def test_policy_chooses_attribute_with_largest_expected_top10_gain():
    candidates = [
        _candidate(
            f"P{index:02d}",
            12.0 - index,
            material="cotton" if index < 6 else "leather",
            color="blue" if index < 2 else None,
        )
        for index in range(12)
    ]
    state = SessionState("s1", turn=2, candidate_count=len(candidates))

    decision = QuestionPolicy().choose(candidates, state)

    assert decision.ask_attribute == "material"
    assert decision.utility > 0


def test_policy_never_repeats_or_asks_rejected_attribute():
    candidates = [
        _candidate(
            f"P{index:02d}",
            12.0 - index,
            material="cotton" if index < 6 else "leather",
            color="blue" if index % 2 else "red",
        )
        for index in range(12)
    ]
    state = SessionState(
        "s1",
        turn=3,
        candidate_count=len(candidates),
        asked_attributes=["color"],
        no_preference_attributes={"size"},
    )

    decision = QuestionPolicy().choose(candidates, state)

    assert decision.ask_attribute not in {"color", "size"}


def test_policy_stops_asking_when_candidate_set_fits_top10():
    candidates = [
        _candidate(f"P{index:02d}", 10.0 - index, material="cotton")
        for index in range(10)
    ]

    decision = QuestionPolicy().choose(candidates, SessionState("s1", turn=2))

    assert decision.ask_attribute is None
    assert decision.utility == 0.0


def test_late_turn_requires_meaningful_utility():
    candidates = [
        _candidate(f"P{index:02d}", 12.0 - index, material="cotton")
        for index in range(12)
    ]

    decision = QuestionPolicy().choose(candidates, SessionState("s1", turn=9))

    assert decision.ask_attribute is None


def test_policy_prefers_answerable_attribute_over_category_or_brand():
    candidates = []
    for index in range(12):
        candidate = _candidate(
            f"P{index:02d}",
            12.0 - index,
            material="cotton" if index < 6 else "leather",
        )
        candidate.product["categories"] = [f"Category {index}"]
        candidate.product["store"] = f"Brand {index}"
        candidates.append(candidate)

    decision = QuestionPolicy().choose(candidates, SessionState("s1", turn=2))

    assert decision.ask_attribute == "material"
