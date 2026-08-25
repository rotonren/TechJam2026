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
