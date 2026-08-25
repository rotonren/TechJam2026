from compasscart.models import Constraint, SessionState


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
