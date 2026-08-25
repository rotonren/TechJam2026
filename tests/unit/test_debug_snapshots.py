from __future__ import annotations

import json
from pathlib import Path

import pytest

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
            },
        ],
        "asked_attributes": ["color"],
        "pending_attribute": "brand",
        "query_history": ["red shoes"],
        "no_preference_attributes": ["brand", "material"],
        "previous_recommendations": ["A", "B"],
        "candidate_count": 12,
    }
    json.dumps(snapshot, allow_nan=False)


def test_snapshot_products_keeps_response_order_and_current_catalog_fields() -> None:
    response = {"recommendations": [{"parent_asin": "B"}, {"parent_asin": "A"}]}
    products = {
        "A": {
            "title": "Alpha",
            "price": 10.0,
            "average_rating": 4.5,
            "rating_number": 7,
            "store": "Store A",
            "categories": ["shoes"],
            "features": ["light"],
            "details": {"color": "red"},
            "normalized_attributes": {"color": ["red"]},
        },
        "B": {
            "title": "Beta",
            "price": 20.0,
            "average_rating": 4.0,
            "rating_number": 4,
            "store": "Store B",
            "categories": ["boots"],
            "features": ["warm"],
            "details": {"color": "black"},
            "normalized_attributes": {"color": ["black"]},
        },
    }

    snapshot = snapshot_products(response, products)

    assert [(item["rank"], item["parent_asin"]) for item in snapshot] == [
        (1, "B"),
        (2, "A"),
    ]
    assert snapshot[0]["rating"] == 4.0
    assert snapshot[0]["rating_count"] == 4
    assert "score" not in snapshot[0]
    assert "source_scores" not in snapshot[0]


def test_snapshot_products_keeps_rank_for_missing_metadata() -> None:
    snapshot = snapshot_products({"recommendations": [{"parent_asin": "MISSING"}]}, {})

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
        "conversation_summary": "summary",
        "usage": {"prompt_tokens": 0},
    }

    assert snapshot_response(response) == {
        "message": "Matches",
        "recommendations": [{"parent_asin": "A"}],
        "ask_attribute": None,
        "conversation_summary": "summary",
    }


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
        "conversation_summary": None,
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
