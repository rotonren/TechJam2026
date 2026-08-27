import pytest

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


def _lookup_candidates(
    rows: list[dict[str, tuple[str, ...]]],
    *,
    scores: list[float] | None = None,
) -> tuple[list[Candidate], dict[str, dict[str, tuple[str, ...]]]]:
    candidates = [
        Candidate(
            parent_asin=f"P{index:02d}",
            product={},
            score=scores[index] if scores is not None else 1.0,
        )
        for index in range(len(rows))
    ]
    return candidates, {
        candidate.parent_asin: rows[index]
        for index, candidate in enumerate(candidates)
    }


def _policy_utility(
    policy: QuestionPolicy,
    candidates: list[Candidate],
    lookup: dict[str, dict[str, tuple[str, ...]]],
    attribute: str,
    *,
    turn: int = 1,
) -> float:
    probabilities = policy._probabilities(candidates)
    return policy._utility(
        attribute,
        [lookup[candidate.parent_asin] for candidate in candidates],
        probabilities,
        turn,
        [candidate.parent_asin for candidate in candidates],
    )


def _choose_with_utilities(monkeypatch, state, utilities):
    candidates, lookup = _lookup_candidates([{} for _ in range(11)])
    policy = QuestionPolicy(lookup)
    monkeypatch.setattr(
        policy,
        "_utility",
        lambda attribute, *_args: utilities.get(attribute, 0.0),
    )
    return policy.choose(candidates, state)


def _strict_material_then_color_candidates():
    rows = [
        {
            "material": ("cotton" if index < 10 else "leather",),
            "color": ("blue" if index < 15 else "red",),
        }
        for index in range(20)
    ]
    return _lookup_candidates(rows)


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


def test_policy_skips_strictly_highest_utility_asked_attribute():
    candidates, lookup = _strict_material_then_color_candidates()
    policy = QuestionPolicy(lookup)

    unblocked = policy.choose(candidates, SessionState("open", turn=1))
    blocked = policy.choose(
        candidates,
        SessionState("asked", turn=1, asked_attributes=["material"]),
    )

    assert unblocked.ask_attribute == "material"
    assert unblocked.utility == pytest.approx(0.445)
    assert blocked.ask_attribute == "color"
    assert blocked.utility == pytest.approx(0.22)


def test_policy_skips_strictly_highest_utility_no_preference_attribute():
    candidates, lookup = _strict_material_then_color_candidates()
    policy = QuestionPolicy(lookup)

    unblocked = policy.choose(candidates, SessionState("open", turn=1))
    blocked = policy.choose(
        candidates,
        SessionState(
            "rejected", turn=1, no_preference_attributes={"material"}
        ),
    )

    assert unblocked.ask_attribute == "material"
    assert unblocked.utility == pytest.approx(0.445)
    assert blocked.ask_attribute == "color"
    assert blocked.utility == pytest.approx(0.22)


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


def test_policy_prefers_lower_reduction_attribute_when_high_reduction_is_unsupported():
    rows = [
        {
            "material": ("cotton" if index < 10 else "leather",),
            "color": ("blue" if index < 8 else "red",),
        }
        for index in range(20)
    ]
    candidates, lookup = _lookup_candidates(rows)
    policy = QuestionPolicy(
        lookup,
        parser_support=lambda attribute, value: not (
            attribute == "material" and value == "leather"
        ),
        retrieval_support=lambda _attribute, _value: True,
    )

    decision = policy.choose(candidates, SessionState("s1", turn=1))

    assert decision.ask_attribute == "color"


@pytest.mark.parametrize("unsupported_callback", ("parser", "retrieval"))
def test_policy_requires_both_callbacks_for_each_partition(unsupported_callback):
    rows = [
        {"material": ("cotton" if index < 6 else "leather",)}
        for index in range(12)
    ]
    candidates, lookup = _lookup_candidates(rows)

    def support(attribute, value):
        return not (attribute == "material" and value == "leather")

    policy = QuestionPolicy(
        lookup,
        parser_support=support if unsupported_callback == "parser" else lambda *_: True,
        retrieval_support=(
            support if unsupported_callback == "retrieval" else lambda *_: True
        ),
    )

    decision = policy.choose(candidates, SessionState("s1", turn=1))

    assert decision.ask_attribute is None


@pytest.mark.parametrize("provided_callback", ("parser_support", "retrieval_support"))
def test_policy_does_not_treat_a_missing_support_callback_as_evidence(
    provided_callback,
):
    rows = [
        {"material": ("cotton" if index < 10 else "leather",)}
        for index in range(20)
    ]
    candidates, lookup = _lookup_candidates(rows)

    decision = QuestionPolicy(
        lookup, **{provided_callback: lambda _attribute, _value: True}
    ).choose(candidates, SessionState("s1", turn=1))

    assert decision.ask_attribute is None


def test_policy_ignores_supported_partition_below_meaningful_size_and_mass():
    rows = [
        {"material": ("cotton" if index < 10 else "leather",)}
        for index in range(11)
    ]
    candidates, lookup = _lookup_candidates(rows, scores=[1.0] * 10 + [0.01])

    decision = QuestionPolicy(lookup).choose(
        candidates, SessionState("s1", turn=1)
    )

    assert decision.ask_attribute is None


def test_two_distinct_candidate_ids_make_low_mass_partitions_meaningful():
    attributes = [
        *({} for _ in range(16)),
        {"material": ("cotton",)},
        {"material": ("cotton",)},
        {"material": ("leather",)},
        {"material": ("leather",)},
    ]
    probabilities = [0.06] * 16 + [0.01] * 4
    candidate_ids = [f"P{index:02d}" for index in range(20)]

    utility = QuestionPolicy()._utility(
        "material", attributes, probabilities, 1, candidate_ids
    )

    assert utility == pytest.approx(0.895)


def test_duplicate_candidate_rows_do_not_satisfy_unique_id_boundary():
    attributes = [
        *({} for _ in range(16)),
        {"material": ("cotton",)},
        {"material": ("cotton",)},
        {"material": ("leather",)},
        {"material": ("leather",)},
    ]
    probabilities = [0.06] * 16 + [0.01] * 4
    candidate_ids = [
        *(f"P{index:02d}" for index in range(16)),
        "COTTON",
        "COTTON",
        "LEATHER",
        "LEATHER",
    ]

    utility = QuestionPolicy()._utility(
        "material", attributes, probabilities, 1, candidate_ids
    )

    assert utility == 0.0


def test_probability_mass_boundary_is_inclusive_at_exactly_point_zero_five():
    attributes = [
        *({} for _ in range(10)),
        {"material": ("cotton",)},
        {"material": ("leather",)},
    ]
    probabilities = [0.09] * 10 + [0.05, 0.05]

    utility = QuestionPolicy()._utility(
        "material",
        attributes,
        probabilities,
        1,
        [f"P{index:02d}" for index in range(12)],
    )

    assert utility == pytest.approx(0.895)


def test_policy_accepts_single_candidate_partition_with_meaningful_mass():
    rows = [
        {"material": ("cotton" if index < 10 else "leather",)}
        for index in range(11)
    ]
    candidates, lookup = _lookup_candidates(rows, scores=[1.0] * 10 + [10.0])

    decision = QuestionPolicy(lookup).choose(
        candidates, SessionState("s1", turn=1)
    )

    assert decision.ask_attribute == "material"


def test_policy_uses_raw_partition_mass_as_support_fraction_denominator():
    values = ["cotton"] * 5 + ["wool"] * 5 + ["nylon"] * 5 + ["leather"] * 5
    candidates, lookup = _lookup_candidates(
        [{"material": (value,)} for value in values] + [{} for _ in range(10)]
    )
    policy = QuestionPolicy(
        lookup,
        parser_support=lambda _attribute, value: value != "nylon",
        retrieval_support=lambda _attribute, value: value != "wool",
    )

    decision = policy.choose(candidates, SessionState("s1", turn=1))

    expected = 0.5 * 0.90 * 0.75 * 0.75 * 1.0 - 0.05 * (1.0 - 0.90)
    assert decision.ask_attribute == "material"
    assert decision.utility == pytest.approx(expected)


@pytest.mark.parametrize(
    ("parser_values", "retrieval_values"),
    (
        ({"a", "b"}, {"a", "b", "c"}),
        ({"a", "b", "c"}, {"a", "b"}),
    ),
)
def test_policy_uses_unequal_unioned_support_fractions_for_overlapping_partitions(
    parser_values, retrieval_values
):
    rows = []
    for index in range(20):
        values = []
        if index < 10:
            values.append("a")
        if 5 <= index < 15:
            values.append("b")
        if index >= 10:
            values.append("c")
        rows.append({"material": tuple(values)})
    candidates, lookup = _lookup_candidates(rows)
    policy = QuestionPolicy(
        lookup,
        parser_support=lambda _attribute, value: value in parser_values,
        retrieval_support=lambda _attribute, value: value in retrieval_values,
    )

    utility = _policy_utility(policy, candidates, lookup, "material")

    expected = (1.0 / 3.0) * 0.90 * 0.75 * 1.0 - 0.05 * (1.0 - 0.90)
    assert utility == pytest.approx(expected)


def test_policy_unions_candidate_indices_for_support_mass():
    rows = []
    for index in range(20):
        if index < 5:
            values = ("cotton",)
        elif index < 15:
            values = ("cotton", "leather")
        else:
            values = ("leather",)
        rows.append({"material": values})
    candidates, lookup = _lookup_candidates(rows)

    decision = QuestionPolicy(
        lookup,
        parser_support=lambda *_: True,
        retrieval_support=lambda *_: True,
    ).choose(candidates, SessionState("s1", turn=1))

    expected = 0.25 * 0.90 * 1.0 * 1.0 * 1.0 - 0.05 * (1.0 - 0.90)
    assert decision.ask_attribute == "material"
    assert decision.utility == pytest.approx(expected)


def test_policy_does_not_claim_reduction_for_identical_overlapping_partitions():
    candidates, lookup = _lookup_candidates(
        [{"material": ("cotton", "leather")} for _ in range(20)]
    )

    decision = QuestionPolicy(lookup).choose(
        candidates, SessionState("s1", turn=1)
    )

    assert decision.ask_attribute is None


@pytest.mark.parametrize(
    ("attribute", "answerability"),
    (
        ("material", 0.90),
        ("color", 0.90),
        ("size", 0.85),
        ("style", 0.80),
        ("budget", 0.90),
        ("feature", 0.70),
        ("use_case", 0.85),
    ),
)
def test_policy_uses_exact_response_likelihood_and_no_preference_risk(
    attribute, answerability
):
    rows = [
        {attribute: ("first" if index < 10 else "second",)}
        for index in range(20)
    ]
    candidates, lookup = _lookup_candidates(rows)
    policy = QuestionPolicy(lookup)

    utility = _policy_utility(policy, candidates, lookup, attribute)

    expected = 0.5 * answerability - 0.05 * (1.0 - answerability)
    assert utility == pytest.approx(expected)


@pytest.mark.parametrize(
    "state,candidate_count",
    (
        (SessionState("continuation", turn=1, continuation_requested=True), 20),
        (SessionState("last", turn=10), 20),
        (SessionState("empty", turn=1), 0),
        (SessionState("single", turn=1), 1),
        (SessionState("nine", turn=1), 9),
        (SessionState("ten", turn=1), 10),
    ),
)
def test_policy_short_circuits_before_support_callbacks(state, candidate_count):
    candidates, lookup = _lookup_candidates(
        [
            {"material": ("cotton" if index % 2 else "leather",)}
            for index in range(candidate_count)
        ]
    )

    def unexpected_callback(*_args):
        raise AssertionError("support callback should not run")

    decision = QuestionPolicy(
        lookup,
        parser_support=unexpected_callback,
        retrieval_support=unexpected_callback,
    ).choose(candidates, state)

    assert decision.ask_attribute is None


@pytest.mark.parametrize(
    ("turn", "utility", "expected_attribute"),
    (
        (7, 0.000001, "material"),
        (8, 0.099999, None),
        (8, 0.100000, None),
        (8, 0.100001, "material"),
        (9, 0.149999, None),
        (9, 0.150000, None),
        (9, 0.150001, "material"),
    ),
)
def test_policy_uses_exact_late_turn_thresholds(
    monkeypatch, turn, utility, expected_attribute
):
    decision = _choose_with_utilities(
        monkeypatch,
        SessionState(f"turn-{turn}-{utility}", turn=turn),
        {"material": utility},
    )

    assert decision.ask_attribute == expected_attribute


@pytest.mark.parametrize(
    "state",
    (
        SessionState("buying", turn=1, route="buying"),
        SessionState(
            "override", turn=1, route="browsing", override_scope="attribute"
        ),
    ),
)
def test_hard_requirement_preference_cannot_override_strict_utility(
    monkeypatch, state
):
    decision = _choose_with_utilities(
        monkeypatch,
        state,
        {
            "color": 0.200001,
            "material": 0.200000,
            "size": 0.200000,
            "budget": 0.200000,
        },
    )

    assert decision.ask_attribute == "color"


@pytest.mark.parametrize(
    ("state", "hard_attribute"),
    tuple(
        (state, hard_attribute)
        for state in (
            SessionState("buying", turn=1, route="buying"),
            SessionState(
                "override", turn=1, route="browsing", override_scope="attribute"
            ),
        )
        for hard_attribute in ("budget", "size", "material")
    ),
)
def test_buying_and_override_prefer_every_hard_requirement_on_equal_utility(
    monkeypatch, state, hard_attribute
):
    decision = _choose_with_utilities(
        monkeypatch,
        state,
        {"color": 0.20, hard_attribute: 0.20},
    )

    assert decision.ask_attribute == hard_attribute


@pytest.mark.parametrize(
    ("state", "utilities", "expected_attribute"),
    (
        (
            SessionState("browsing", turn=1, route="browsing"),
            {"color": 0.20, "style": 0.20},
            "color",
        ),
        (
            SessionState("buying", turn=1, route="buying"),
            {"size": 0.20, "budget": 0.20},
            "size",
        ),
        (
            SessionState(
                "override", turn=1, route="browsing", override_scope="attribute"
            ),
            {"size": 0.20, "budget": 0.20},
            "size",
        ),
    ),
)
def test_fixed_attribute_order_is_the_final_equal_utility_tie_break(
    monkeypatch, state, utilities, expected_attribute
):
    decision = _choose_with_utilities(monkeypatch, state, utilities)

    assert question_policy._ATTRIBUTES == (
        "material",
        "color",
        "size",
        "style",
        "budget",
        "feature",
        "use_case",
    )
    assert question_policy._HARD_REQUIREMENT_ATTRIBUTES == {
        "budget",
        "size",
        "material",
    }
    assert decision.ask_attribute == expected_attribute
