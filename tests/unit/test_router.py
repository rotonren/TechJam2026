from compasscart.models import Constraint, SessionState
from compasscart.router import RoutePlanner


def _constraint(attribute: str, value: str) -> Constraint:
    return Constraint(attribute, value, 1.0, True, "message", 1, 1)


def test_concrete_hard_requirement_routes_to_buying():
    state = SessionState("s1", constraints=[_constraint("color", "blue")])

    plan = RoutePlanner().build_plan(state, "I need blue running shoes")

    assert plan.route == "buying"
    assert plan.hard_filters["color"] == ("blue",)
    assert dict(plan.source_weights)["attribute"] == 0.45


def test_vague_exploration_routes_to_browsing():
    state = SessionState("s1", route="browsing")

    plan = RoutePlanner().build_plan(state, "I'm still exploring shoes")

    assert plan.route == "browsing"
    assert dict(plan.source_weights)["dense"] == 0.45


def test_override_uses_balanced_source_weights():
    state = SessionState("s1", constraints=[_constraint("material", "leather")])

    plan = RoutePlanner().build_plan(
        state, "Actually, I need leather", is_override=True
    )

    assert dict(plan.source_weights) == {
        "lexical": 0.35,
        "dense": 0.35,
        "attribute": 0.30,
    }
    assert plan.is_override is True
