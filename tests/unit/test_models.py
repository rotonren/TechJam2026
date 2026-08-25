from compasscart.models import Candidate, Constraint, SessionState


def test_session_state_returns_only_current_active_constraints():
    state = SessionState(session_id="s1", intent_version=2)
    state.constraints.extend(
        [
            Constraint("color", "red", 1.0, True, "message", 1, 1, "superseded"),
            Constraint("color", "blue", 1.0, True, "message", 3, 2, "active"),
        ]
    )

    assert [(item.attribute, item.value) for item in state.active_constraints()] == [
        ("color", "blue")
    ]


def test_session_state_keeps_retained_category_from_earlier_intent():
    state = SessionState(session_id="s1", intent_version=2)
    state.constraints.append(
        Constraint("category", "shoes", 0.9, True, "message", 1, 1, "active")
    )

    assert [(item.attribute, item.value) for item in state.active_constraints()] == [
        ("category", "shoes")
    ]


def test_constraint_and_candidate_defaults_preserve_existing_call_shapes():
    constraint = Constraint("color", "blue", 1.0, True, "message", 1, 1)
    candidate = Candidate("P1")

    assert constraint.operator == "eq"
    assert constraint.upper_value is None
    assert constraint.alternatives == ()
    assert constraint.values() == ("blue",)
    assert candidate.violations == ()
    assert candidate.relaxed is False


def test_constraint_values_prefers_alternatives_over_legacy_value():
    constraint = Constraint(
        "color",
        "stale",
        1.0,
        True,
        "message",
        1,
        1,
        alternatives=("black", "blue"),
    )

    assert constraint.values() == ("black", "blue")


def test_session_state_route_hint_does_not_shift_existing_positional_fields():
    state = SessionState("s1", 3, "buying", 7)

    assert state.intent_version == 7
    assert state.route_hint is None


def test_session_state_new_fields_do_not_shift_legacy_positional_arguments():
    constraints = [Constraint("color", "blue", 1.0, True, "message", 1, 1)]
    state = SessionState(
        "s1",
        3,
        "buying",
        7,
        constraints,
        ["color"],
        "size",
        ["blue shoes"],
        {"brand"},
        ["P1"],
        4,
    )

    assert state.constraints is constraints
    assert state.asked_attributes == ["color"]
    assert state.pending_attribute == "size"
    assert state.query_history == ["blue shoes"]
    assert state.no_preference_attributes == {"brand"}
    assert state.previous_recommendations == ["P1"]
    assert state.candidate_count == 4
    assert state.continuation_requested is False
    assert state.override_scope == "none"
    assert state.route_hint is None
