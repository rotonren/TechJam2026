from __future__ import annotations

from agent import Agent
from compasscart.constraints import hard_constraint_violations


def _ids(response: dict[str, object]) -> list[str]:
    return [item["parent_asin"] for item in response["recommendations"]]


def _assert_contract(response: dict[str, object], valid_ids: set[str]) -> None:
    assert set(response) == {
        "message",
        "ask_attribute",
        "recommendations",
        "usage",
    }
    identifiers = _ids(response)
    assert identifiers
    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers) <= valid_ids


def test_functional_edge_conversations_preserve_constraints_and_routes(
    fixture_catalog_path,
):
    agent = Agent(fixture_catalog_path)

    agent.reset("alternatives", {})
    alternatives = agent.respond(
        "alternatives", "I need black or blue running shoes under $80.", 1, 10
    )
    _assert_contract(alternatives, agent.catalog.valid_ids)
    state = agent.sessions.get("alternatives")
    assert state is not None
    active = {item.attribute: item for item in state.active_constraints()}
    assert active["color"].operator == "in"
    assert active["color"].alternatives == ("black", "blue")
    assert active["budget"].operator == "lte"
    for identifier in _ids(alternatives):
        violations = hard_constraint_violations(
            agent.catalog.product(identifier),
            agent.catalog.attributes[identifier],
            [item for item in state.active_constraints() if item.is_hard],
        )
        # The exact first result must satisfy every hard condition.  Relaxed
        # results, when needed, are disclosed in the response message.
        if identifier == _ids(alternatives)[0]:
            assert not violations

    agent.reset("browsing", {})
    first = agent.respond("browsing", "I'm still exploring shoes.", 1, 10)
    second = agent.respond("browsing", "cotton", 2, 10)
    _assert_contract(first, agent.catalog.valid_ids)
    _assert_contract(second, agent.catalog.valid_ids)
    browsing_trace = agent.traces.records[-1]
    assert browsing_trace["route"] == "browsing"
    assert browsing_trace["route_reason"] == "explicit_browsing"
    assert any(
        item.attribute == "material" and item.value == "cotton"
        for item in agent.sessions.get("browsing").active_constraints()
    )

    agent.reset("override", {})
    agent.respond("override", "I need a red cotton dress.", 1, 10)
    changed = agent.respond(
        "override", "Actually, I need a black leather belt for work.", 2, 10
    )
    _assert_contract(changed, agent.catalog.valid_ids)
    override_state = agent.sessions.get("override")
    assert override_state is not None
    assert override_state.override_scope == "goal"
    assert not {
        ("color", "red"),
        ("material", "cotton"),
        ("category", "dresses"),
    } & {
        (item.attribute, item.value) for item in override_state.active_constraints()
    }

    agent.reset("negative", {})
    negative = agent.respond("negative", "I need shoes.", 1, 10)
    negative = agent.respond("negative", "I don't want leather.", 2, 10)
    _assert_contract(negative, agent.catalog.valid_ids)
    negative_state = agent.sessions.get("negative")
    assert negative_state is not None
    leather = next(
        item for item in negative_state.active_constraints() if item.attribute == "material"
    )
    assert leather.operator == "not_in"
    assert all(item.attribute != "use_case" for item in negative_state.active_constraints())

    agent.reset("evidence", {})
    agent.respond("evidence", "with a magnetic clasp", 1, 10)
    evidence = agent.respond("evidence", "I need a belt", 2, 10)
    _assert_contract(evidence, agent.catalog.valid_ids)
    evidence_state = agent.sessions.get("evidence")
    assert evidence_state is not None
    assert evidence_state.query_history == [
        "with a magnetic clasp",
        "I need a belt",
    ]
    assert any(item.attribute == "category" for item in evidence_state.active_constraints())
