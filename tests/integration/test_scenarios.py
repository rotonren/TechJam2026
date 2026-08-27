from __future__ import annotations

import json

from agent import Agent
from compasscart.config import RuntimeConfig


def _ids(response: dict) -> list[str]:
    return [item["parent_asin"] for item in response["recommendations"]]


def test_buying_and_browsing_always_return_ranked_products(fixture_catalog_path):
    agent = Agent(fixture_catalog_path)

    agent.reset("buy", {"preference_tags": []})
    buying = agent.respond(
        "buy", "I'm looking for running shoes. A key requirement is: under $80.", 1, 10
    )
    agent.reset("browse", {"preference_tags": []})
    browsing = agent.respond(
        "browse", "I'm looking for shoes, but I'm still exploring.", 1, 10
    )

    assert _ids(buying)[0] == "SHOE1"
    assert _ids(browsing)
    assert buying["recommendations"] and browsing["recommendations"]


def test_intent_override_replaces_conflicting_old_constraint(fixture_catalog_path):
    agent = Agent(fixture_catalog_path)
    agent.reset("override", {"preference_tags": []})
    first = agent.respond("override", "I need a red cotton dress", 1, 10)

    changed = agent.respond(
        "override",
        "Actually, ignore red cotton dresses. What I need is blue mesh running shoes.",
        3,
        10,
    )

    assert _ids(first)[0] == "DRESS1"
    assert _ids(changed)[0] == "SHOE1"
    state = agent.sessions.get("override")
    assert state is not None
    assert ("color", "red") not in {
        (item.attribute, item.value) for item in state.active_constraints()
    }
    assert state.override_scope == "goal"
    assert not {
        ("color", "red"),
        ("material", "cotton"),
        ("category", "dress"),
    } & {(item.attribute, item.value) for item in state.active_constraints()}


def test_boundary_reply_prevents_repeated_question(tmp_path):
    catalog_path = tmp_path / "catalog.jsonl"
    products = []
    for index in range(12):
        products.append(
            {
                "parent_asin": f"P{index:02d}",
                "title": f"Training shoe {index}",
                "features": ["breathable"],
                "details": {
                    "Material": "Cotton" if index < 6 else "Leather",
                    "Color": "Blue" if index % 2 else "Red",
                },
                "categories": ["Clothing", "Shoes"],
                "price": 30 + index,
                "average_rating": 4.5,
                "rating_number": 100 + index,
            }
        )
    catalog_path.write_text(
        "".join(json.dumps(product) + "\n" for product in products), encoding="utf-8"
    )
    agent = Agent(catalog_path)
    agent.reset("boundary", {"preference_tags": []})

    first = agent.respond(
        "boundary", "I'm looking for shoes, but I'm still exploring.", 1, 10
    )
    asked = first["ask_attribute"]
    assert asked is not None
    second = agent.respond(
        "boundary",
        f"I don't have a preference for {asked}; please use your judgment.",
        2,
        10,
    )

    assert first["recommendations"] and second["recommendations"]
    assert second["ask_attribute"] != asked
    state = agent.sessions.get("boundary")
    assert state is not None
    assert state.pending_attribute == second["ask_attribute"]


def test_browsing_context_survives_a_follow_up_attribute_answer(fixture_catalog_path):
    agent = Agent(fixture_catalog_path)
    agent.reset("browse-follow-up", {})

    first = agent.respond("browse-follow-up", "I'm still exploring shoes.", 1, 4)
    second = agent.respond("browse-follow-up", "cotton", 2, 4)

    assert first["recommendations"] and second["recommendations"]
    trace = agent.traces.records[-1]
    assert trace["route"] == "browsing"
    assert trace["route_reason"] == "explicit_browsing"


def test_agent_wires_bounded_rank_calibration_without_changing_response_contract(
    fixture_catalog_path,
):
    config = RuntimeConfig(
        rank_attribute_weight=0.05,
        rank_consensus_bonus=0.025,
        rank_boundary_bonus=0.025,
        rank_fusion_weight=0.15,
        adaptive_browsing_mmr=True,
    )
    agent = Agent(fixture_catalog_path, config=config)
    agent.reset("rank-config", {"preference_tags": []})

    response = agent.respond("rank-config", "running shoes", 1, 4)

    assert response["recommendations"]
    assert set(response) == {"message", "ask_attribute", "recommendations", "usage"}
    assert agent.ranker.attribute_weight == 0.05
    assert agent.ranker.consensus_bonus == 0.025
    assert agent.ranker.boundary_bonus == 0.025
    assert agent.ranker.fusion_weight == 0.15
    assert agent.ranker.adaptive_browsing_mmr is True
