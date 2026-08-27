import pytest

from agent import Agent
from compasscart.models import Constraint
from compasscart.parser import MessageParser
from compasscart.state import SessionStore


def test_override_supersedes_conflict_and_retains_other_hard_constraints():
    store = SessionStore(MessageParser())
    store.reset("s1", {"preference_tags": []})
    store.update("s1", "I need a red cotton dress", 1)

    state = store.update("s1", "Actually, ignore red. I need blue.", 3)

    active = {(item.attribute, item.value) for item in state.active_constraints()}
    assert ("color", "blue") in active
    assert ("color", "red") not in active
    assert ("material", "cotton") in active
    assert any(
        item.value == "red" and item.status == "superseded"
        for item in state.constraints
    )


def test_user_constraint_rejects_conflicting_profile_preference():
    store = SessionStore(MessageParser())
    state = store.reset("s1", {"preference_tags": ["red"]})
    assert ("color", "red") in {
        (item.attribute, item.value) for item in state.active_constraints()
    }

    state = store.update("s1", "I need blue shoes", 1)

    assert ("color", "blue") in {
        (item.attribute, item.value) for item in state.active_constraints()
    }
    assert any(
        item.value == "red" and item.status == "rejected" for item in state.constraints
    )


def test_update_is_idempotent_for_same_turn_and_message():
    store = SessionStore(MessageParser())
    store.reset("s1", {"preference_tags": []})

    first = store.update("s1", "I need blue shoes", 1)
    second = store.update("s1", "I need blue shoes", 1)

    assert second is first
    assert len(second.constraints) == len(first.constraints)


def test_no_preference_is_recorded_and_not_added_as_constraint():
    store = SessionStore(MessageParser())
    state = store.reset("s1", {"preference_tags": []})
    state.asked_attributes.append("size")

    state = store.update(
        "s1", "I don't have a preference for size; please use your judgment.", 2
    )

    assert "size" in state.no_preference_attributes
    assert all(item.attribute != "size" for item in state.active_constraints())


def test_no_preference_keeps_existing_buying_route():
    store = SessionStore(MessageParser())
    store.reset("s1", {"preference_tags": []})
    store.update("s1", "I need blue shoes", 1)

    state = store.update(
        "s1", "No preference.", 2, expected_attribute="size"
    )

    assert state.route == "buying"
    assert state.route_hint == "buying"


def test_query_history_keeps_evidence_but_resets_on_override():
    store = SessionStore(MessageParser())
    store.reset("s1", {"preference_tags": []})
    store.update("s1", "I need a red cotton dress with buckle closure", 1)
    state = store.update(
        "s1", "I don't have a preference for size; use your judgment.", 2
    )
    assert state.query_history == [
        "I need a red cotton dress with buckle closure",
    ]

    state = store.update("s1", "Actually, what I need is blue leather shoes.", 3)

    assert state.query_history == ["Actually, what I need is blue leather shoes."]


def test_update_copies_parser_semantics_to_constraint():
    store = SessionStore(MessageParser())
    store.reset("s1", {})

    state = store.update("s1", "between $50 and $80", 1)

    budget = state.active_constraints()[0]
    assert (budget.operator, budget.value, budget.upper_value, budget.alternatives) == (
        "between",
        "50.00",
        "80.00",
        (),
    )


def test_update_keeps_unknown_second_message_as_bounded_query_evidence():
    store = SessionStore(MessageParser())
    store.reset("s1", {})
    store.update("s1", "I need blue shoes", 1)

    state = store.update("s1", "with a magnetic clasp", 2)

    assert state.query_history == ["I need blue shoes", "with a magnetic clasp"]
    assert Agent._query_text("with a magnetic clasp", state).count(
        "with a magnetic clasp"
    ) == 1


def test_unrecognized_clarification_stays_query_evidence_without_hard_constraint():
    store = SessionStore(MessageParser({"feature": ("Waterproof",)}))
    store.reset("s1", {})

    state = store.update(
        "s1", "with a magnetic clasp", 2, expected_attribute="feature"
    )

    assert state.query_history == ["with a magnetic clasp"]
    assert Agent._query_text("with a magnetic clasp", state).count(
        "with a magnetic clasp"
    ) == 1
    clarification_constraints = [
        item for item in state.constraints if item.source == "clarification"
    ]
    assert [
        (item.attribute, item.value, item.is_hard, item.confidence)
        for item in clarification_constraints
    ] == [("feature", "with a magnetic clasp", False, 0.6)]


def test_catalog_category_remains_specific_in_session_state():
    store = SessionStore(
        MessageParser(
            {
                "category": ("Fashion Sneakers", "Boy Shorts"),
                "style": ("Sneaker",),
                "brand": ("Boy",),
            }
        )
    )

    for session_id, message, expected in (
        ("fashion", "I want Fashion Sneakers", "fashion sneakers"),
        ("boy", "I want Boy Shorts", "boy shorts"),
    ):
        store.reset(session_id, {})
        state = store.update(session_id, message, 1)
        assert [
            item.value
            for item in state.active_constraints()
            if item.attribute == "category"
        ] == [expected]


def test_update_marks_and_clears_continuation_requests_and_keeps_last_four_messages():
    store = SessionStore(MessageParser())
    store.reset("s1", {})
    for turn, message in enumerate(("one", "two", "three", "four", "five"), start=1):
        state = store.update("s1", message, turn)

    state = store.update("s1", "show me more", 6)
    assert state.continuation_requested is True
    assert state.query_history == ["two", "three", "four", "five"]

    state = store.update("s1", "blue", 7)
    assert state.continuation_requested is False
    assert state.override_scope == "none"


def test_control_only_message_does_not_replace_substantive_history():
    store = SessionStore(MessageParser())
    store.reset("s1", {})
    store.update("s1", "I need a belt", 1)

    state = store.update("s1", "Here are the closest matches I found.", 2)

    assert state.query_history == ["I need a belt"]


@pytest.mark.parametrize(
    "message",
    ("Thanks", "search", "Here are the closest matches I found."),
)
def test_pending_feature_does_not_treat_control_only_text_as_evidence(message):
    store = SessionStore(MessageParser({"feature": ("waterproof",)}))
    store.reset("s1", {})

    state = store.update("s1", message, 1, expected_attribute="feature")

    assert state.query_history == []
    assert state.active_constraints() == []
    assert message not in Agent._query_text(message, state)


@pytest.mark.parametrize(
    "message",
    (
        "Show me more, please.",
        "Could you show me more?",
        "Please, show me more.",
        "Could you, please show me more?",
    ),
)
def test_pending_feature_ignores_polite_continuation_wrappers(message):
    store = SessionStore(MessageParser({"feature": ("waterproof",)}))
    state = store.reset("s1", {})
    state.pending_attribute = "feature"

    state = store.update("s1", message, 1, expected_attribute="feature")

    assert state.continuation_requested is True
    assert state.query_history == []
    assert state.active_constraints() == []


@pytest.mark.parametrize(
    "message",
    (
        "No preference, thanks.",
        "No preference please.",
        "I don't have a preference for feature, thank you.",
        "Thanks, no preference please.",
    ),
)
def test_pending_feature_ignores_polite_no_preference_wrappers(message):
    store = SessionStore(MessageParser({"feature": ("waterproof",)}))
    state = store.reset("s1", {})
    state.pending_attribute = "feature"

    state = store.update("s1", message, 1, expected_attribute="feature")

    assert "feature" in state.no_preference_attributes
    assert state.query_history == []
    assert state.active_constraints() == []


def test_mixed_no_preference_keeps_attribute_signal_and_raw_query_history():
    store = SessionStore(MessageParser())
    store.reset("s1", {})
    store.update("s1", "I need red hiking boots", 1)

    message = "I have no preference for color, but I need waterproof hiking boots."
    state = store.update("s1", message, 2, expected_attribute="feature")

    assert "color" in state.no_preference_attributes
    assert {(item.attribute, item.value) for item in state.active_constraints()} >= {
        ("feature", "waterproof"),
        ("category", "boots"),
    }
    assert message in state.query_history


def test_goal_override_replaces_prior_hard_constraints_and_question_state():
    store = SessionStore(MessageParser())
    state = store.reset("s1", {"preference_tags": ["comfort"]})
    state.asked_attributes.append("size")
    state.pending_attribute = "material"
    state.no_preference_attributes.add("brand")
    store.update("s1", "I need a red cotton dress", 1)

    state = store.update(
        "s1", "Actually, I need a black leather belt", 2
    )

    assert state.override_scope == "goal"
    assert {(item.attribute, item.value) for item in state.active_constraints()} == {
        ("color", "black"),
        ("material", "leather"),
        ("category", "belt"),
        ("feature", "comfortable"),
    }
    assert state.asked_attributes == []
    assert state.pending_attribute is None
    assert state.no_preference_attributes == set()
    assert state.query_history == ["Actually, I need a black leather belt"]


def test_attribute_override_replaces_only_mentioned_attribute():
    store = SessionStore(MessageParser())
    store.reset("s1", {})
    store.update("s1", "I need a red cotton dress", 1)

    state = store.update("s1", "Actually, make it blue", 2)

    assert state.override_scope == "attribute"
    assert {(item.attribute, item.value) for item in state.active_constraints()} == {
        ("color", "blue"),
        ("material", "cotton"),
        ("category", "dress"),
    }


def test_explicit_preference_reset_keeps_category_but_clears_old_preferences():
    store = SessionStore(MessageParser())
    store.reset("s1", {})
    prior = store.update("s1", "I need a red cotton dress", 1)
    prior.asked_attributes.append("brand")
    prior.pending_attribute = "size"
    prior.no_preference_attributes.add("feature")

    state = store.update(
        "s1",
        "Actually, ignore my earlier preference. What I need is leather.",
        2,
    )

    assert state.override_scope == "attribute"
    assert {(item.attribute, item.value) for item in state.active_constraints()} == {
        ("category", "dress"),
        ("material", "leather"),
    }
    assert state.query_history == [
        "Actually, ignore my earlier preference. What I need is leather."
    ]
    assert state.asked_attributes == []
    assert state.pending_attribute is None
    assert state.no_preference_attributes == set()


def test_goal_override_allows_an_attribute_rejected_under_the_old_goal():
    store = SessionStore(MessageParser())
    state = store.reset("s1", {})
    state.asked_attributes.append("size")
    state.no_preference_attributes.add("size")
    store.update("s1", "I need a red dress", 1)

    state = store.update("s1", "Actually, I need blue shoes", 2)

    assert state.override_scope == "goal"
    assert "size" not in state.asked_attributes
    assert "size" not in state.no_preference_attributes


def test_profile_constraints_are_soft_and_never_override_user_color():
    store = SessionStore(MessageParser())
    state = store.reset(
        "s1",
        {
            "preference_tags": [
                "comfort",
                "fit",
                "durable",
                "warm",
                "weatherproof",
                "lightweight",
                "breathable",
                "waterproof",
                "stretch",
                "red",
            ],
            "purchase_frequency": 3,
            "average_prior_rating": 4.8,
            "rating_style": "high",
        },
    )
    state = store.update("s1", "I need blue shoes", 1)

    active = state.active_constraints()
    assert all(not item.is_hard for item in active if item.source == "profile")
    assert ("color", "blue") in {(item.attribute, item.value) for item in active}
    assert ("color", "red") not in {(item.attribute, item.value) for item in active}
    assert {item.value for item in active if item.source == "profile"} >= {
        "comfortable",
        "durable",
        "warm",
        "weatherproof",
        "lightweight",
        "breathable",
        "waterproof",
        "stretch",
    }
    assert not {
        "3",
        "4.8",
        "high",
    } & {item.value for item in active}


def test_negative_constraint_replaces_same_value_positive_constraint():
    store = SessionStore(MessageParser())
    store.reset("s1", {})
    store.update("s1", "I need leather shoes", 1)

    state = store.update("s1", "I don't want leather", 2)

    material = [item for item in state.active_constraints() if item.attribute == "material"]
    assert len(material) == 1
    assert material[0].operator == "not_in"
    assert material[0].value == "leather"


def test_goal_override_compares_category_singular_plural_forms():
    parser = MessageParser({"category": ("Dresses", "Belts")})
    store = SessionStore(parser)
    store.reset("s1", {})
    store.update("s1", "I need a dress", 1)

    state = store.update("s1", "I've changed my mind: belt", 2)

    assert state.override_scope == "goal"
    assert {item.value for item in state.active_constraints() if item.attribute == "category"} == {
        "belts"
    }


def test_goal_override_drops_non_profile_soft_category_constraints():
    store = SessionStore(MessageParser())
    state = store.reset("s1", {})
    state.constraints.append(
        Constraint("category", "dress", 0.5, False, "inferred", 1, 1)
    )

    state = store.update("s1", "Actually, I need a black belt", 2)

    assert ("category", "dress") not in {
        (item.attribute, item.value) for item in state.active_constraints()
    }
    assert any(
        item.attribute == "category"
        and item.value == "dress"
        and item.status == "superseded"
        for item in state.constraints
    )


def test_goal_override_supersedes_old_non_profile_soft_constraints():
    store = SessionStore(MessageParser())
    state = store.reset("s1", {})
    state.constraints.append(
        Constraint("feature", "warm", 0.5, False, "inferred", 1, 1)
    )

    state = store.update("s1", "Actually, I need a black belt", 2)

    assert any(
        item.attribute == "feature"
        and item.value == "warm"
        and item.status == "superseded"
        for item in state.constraints
    )
