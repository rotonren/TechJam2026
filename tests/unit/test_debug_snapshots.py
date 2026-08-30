from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import Agent
from compasscart.models import Constraint, SessionState
from compasscart_debug.snapshots import (
    capture_exact_trace,
    json_safe,
    snapshot_products,
    snapshot_response,
    snapshot_state,
)


def test_snapshot_state_preserves_all_fields_as_strict_json() -> None:
    state = SessionState(
        session_id="session-1",
        turn=3,
        route="buying",
        intent_version=2,
        constraints=[
            Constraint("color", "red", 0.75, True, "message", 1, 1, "superseded"),
            Constraint("size", "large", 0.6, False, "profile", 0, 2, "active"),
        ],
        asked_attributes=["color"],
        pending_attribute="brand",
        query_history=["red shoes"],
        no_preference_attributes={"material", "brand"},
        previous_recommendations=["A", "B"],
        candidate_count=12,
        profile_segment="comfort|fit",
        unproductive_attributes={"style", "budget"},
        stall_count=2,
        disclosure_count=1,
    )

    snapshot = snapshot_state(state)

    assert snapshot == {
        "session_id": "session-1",
        "turn": 3,
        "route": "buying",
        "intent_version": 2,
        "constraints": [
            {
                "attribute": "color",
                "value": "red",
                "confidence": 0.75,
                "is_hard": True,
                "source": "message",
                    "created_turn": 1,
                    "intent_version": 1,
                    "status": "superseded",
                    "operator": "eq",
                    "upper_value": None,
                    "alternatives": [],
                },
            {
                "attribute": "size",
                "value": "large",
                "confidence": 0.6,
                "is_hard": False,
                "source": "profile",
                    "created_turn": 0,
                    "intent_version": 2,
                    "status": "active",
                    "operator": "eq",
                    "upper_value": None,
                    "alternatives": [],
                },
        ],
        "asked_attributes": ["color"],
        "pending_attribute": "brand",
        "query_history": ["red shoes"],
        "continuation_requested": False,
        "override_scope": "none",
        "no_preference_attributes": ["brand", "material"],
        "previous_recommendations": ["A", "B"],
        "candidate_count": 12,
        "route_hint": None,
        "profile_segment": "comfort|fit",
        "unproductive_attributes": ["budget", "style"],
        "stall_count": 2,
        "disclosure_count": 1,
    }
    json.dumps(snapshot, allow_nan=False)


def test_agent_snapshots_use_real_response_contract_and_catalog_attributes(
    fixture_catalog_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COMPASSCART_DISABLE_DENSE", "1")
    agent = Agent(fixture_catalog_path)
    agent.reset("session-1", {})
    response = agent.respond("session-1", "blue shoes", 1, 10)

    response_snapshot = snapshot_response(response)
    product_snapshot = snapshot_products(agent, response)
    identifiers = [item["parent_asin"] for item in response["recommendations"]]

    assert response_snapshot == response
    assert set(response_snapshot) == {
        "message",
        "ask_attribute",
        "recommendations",
        "usage",
    }
    assert [(item["rank"], item["parent_asin"]) for item in product_snapshot] == [
        (rank, identifier) for rank, identifier in enumerate(identifiers, start=1)
    ]
    assert product_snapshot[0]["normalized_attributes"] == json_safe(
        agent.catalog.attributes[identifiers[0]]
    )
    assert all(
        "score" not in item and "source_scores" not in item for item in product_snapshot
    )


def test_snapshot_products_keeps_rank_for_missing_metadata() -> None:
    class AgentWithoutMetadata:
        def __init__(self) -> None:
            self.catalog = SimpleNamespace(
                products={}, attributes={"MISSING": {"color": ("red",)}}
            )

    snapshot = snapshot_products(
        AgentWithoutMetadata(), {"recommendations": [{"parent_asin": "MISSING"}]}
    )

    assert snapshot == [
        {
            "rank": 1,
            "parent_asin": "MISSING",
            "title": None,
            "price": None,
            "rating": None,
            "rating_count": None,
            "store": None,
            "categories": [],
            "features": [],
            "details": {},
            "normalized_attributes": {},
            "metadata_missing": True,
        }
    ]


def test_capture_exact_trace_uses_only_the_last_record() -> None:
    records = [
        {"session_id": "s", "turn": 2, "value": "old"},
        {"session_id": "other", "turn": 2, "value": "new"},
    ]

    assert capture_exact_trace(records, "s", 2) is None
    assert capture_exact_trace([*records, {"session_id": "s", "turn": 2}], "s", 2) == {
        "session_id": "s",
        "turn": 2,
    }


def test_snapshot_response_keeps_only_official_agent_response_fields() -> None:
    response = {
        "message": "Matches",
        "recommendations": [{"parent_asin": "A"}],
        "ask_attribute": None,
        "usage": {"prompt_tokens": 0},
    }

    assert snapshot_response(response) == {
        "message": "Matches",
        "recommendations": [{"parent_asin": "A"}],
        "ask_attribute": None,
        "usage": {"prompt_tokens": 0},
    }


def test_snapshot_response_strips_unofficial_recommendation_fields() -> None:
    response = {
        "message": "Matches",
        "recommendations": [
            {"parent_asin": "A", "score": 0.9, "source_scores": {"dense": 0.9}},
            {"parent_asin": "B", "unexpected": "discard"},
        ],
        "ask_attribute": None,
        "usage": {"prompt_tokens": 0},
    }

    assert snapshot_response(response)["recommendations"] == [
        {"parent_asin": "A"},
        {"parent_asin": "B"},
    ]


@pytest.mark.parametrize(
    "response", [{}, {"message": "only"}, {"message": object(), "recommendations": []}]
)
def test_snapshot_response_rejects_invalid_shapes_safely(response: object) -> None:
    with pytest.raises((TypeError, ValueError), match="response"):
        snapshot_response(response)


def test_snapshot_response_rejects_malformed_recommendation_entries() -> None:
    response = {
        "message": "Matches",
        "recommendations": [{"parent_asin": 42}],
        "ask_attribute": None,
        "usage": {"prompt_tokens": 0},
    }

    with pytest.raises((TypeError, ValueError), match="response"):
        snapshot_response(response)


def test_json_safe_normalizes_supported_values_deterministically() -> None:
    value = {"set": {3, "2"}, "path": Path("assets/model"), "nan": float("nan")}

    assert json_safe(value) == {
        "nan": None,
        "path": str(Path("assets/model")),
        "set": ["2", 3],
    }
    json.dumps(json_safe(value), allow_nan=False)


def test_json_safe_rejects_colliding_stringified_mapping_keys() -> None:
    for value in ({1: "integer", "1": "string"}, {"1": "string", 1: "integer"}):
        with pytest.raises(TypeError, match="collide"):
            json_safe(value)


def test_json_safe_sorts_normal_mixed_mapping_keys_deterministically() -> None:
    first = json_safe({2: "integer", "10": "string"})
    second = json_safe({"10": "string", 2: "integer"})

    assert first == second == {"2": "integer", "10": "string"}
    assert list(first) == ["2", "10"]
