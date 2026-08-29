from compasscart import question_policy
from compasscart.models import Candidate, Constraint, SessionState
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


def test_policy_reuses_catalog_attributes_without_extracting_products(monkeypatch):
    candidates = [
        Candidate(parent_asin=f"P{index:02d}", product={}, score=12.0 - index)
        for index in range(12)
    ]
    lookup = {
        candidate.parent_asin: {
            "material": ("cotton",) if index < 6 else ("leather",)
        }
        for index, candidate in enumerate(candidates)
    }

    def should_not_extract(product):
        raise AssertionError("catalog attributes should be reused")

    monkeypatch.setattr(question_policy, "extract_attributes", should_not_extract)

    decision = QuestionPolicy(lookup).choose(candidates, SessionState("s1", turn=2))

    assert decision.ask_attribute == "material"


def test_policy_falls_back_to_product_extraction_when_catalog_id_is_missing():
    candidates = [
        _candidate(
            f"P{index:02d}",
            12.0 - index,
            material="cotton" if index < 6 else "leather",
        )
        for index in range(12)
    ]

    decision = QuestionPolicy({}).choose(candidates, SessionState("s1", turn=2))

    assert decision.ask_attribute == "material"


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


def test_policy_does_not_invent_an_other_question_for_undifferentiated_results():
    candidates = [_candidate(f"P{index:03d}", 300.0 - index) for index in range(250)]

    decision = QuestionPolicy().choose(candidates, SessionState("s1", turn=2))

    assert decision.ask_attribute is None


def test_policy_asks_for_distinguishing_detail_immediately_after_override():
    candidates = [_candidate(f"P{index:03d}", 300.0 - index) for index in range(250)]
    state = SessionState("s1", turn=3, override_scope="attribute")

    decision = QuestionPolicy().choose(candidates, state)

    assert decision.ask_attribute == "other"
    assert decision.utility == 1.0


def test_policy_does_not_ask_a_follow_up_when_more_results_are_requested():
    candidates = [
        _candidate(
            f"P{index:02d}",
            12.0 - index,
            material="cotton" if index < 6 else "leather",
        )
        for index in range(12)
    ]
    state = SessionState("s1", turn=2, continuation_requested=True)

    decision = QuestionPolicy().choose(candidates, state)

    assert decision.ask_attribute is None


def test_policy_does_not_ask_for_an_explicitly_constrained_attribute():
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
        turn=2,
        constraints=[
            Constraint("material", "leather", 1.0, True, "message", 1, 1)
        ],
    )

    decision = QuestionPolicy().choose(candidates, state)

    assert decision.ask_attribute != "material"
