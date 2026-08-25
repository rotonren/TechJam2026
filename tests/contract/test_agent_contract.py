from __future__ import annotations

import pytest

from agent import Agent
from compasscart.models import Candidate, QuestionDecision
from compasscart.response import ALLOWED_ATTRIBUTES, ResponseBuilder
from compasscart.tracing import TraceSink


def test_agent_response_matches_official_contract(fixture_catalog_path):
    agent = Agent(fixture_catalog_path)
    agent.reset("s1", {"preference_tags": ["comfort"]})

    response = agent.respond("s1", "I need blue running shoes under $80", 1, 10)

    assert set(response) == {"message", "ask_attribute", "recommendations", "usage"}
    assert isinstance(response["message"], str)
    assert response["ask_attribute"] in ALLOWED_ATTRIBUTES | {None}
    identifiers = [item["parent_asin"] for item in response["recommendations"]]
    assert 0 < len(identifiers) <= 10
    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers) <= agent.catalog.valid_ids
    assert response["usage"] == {"prompt_tokens": 0, "completion_tokens": 0}


def test_agent_requires_reset(fixture_catalog_path):
    agent = Agent(fixture_catalog_path)

    with pytest.raises(RuntimeError, match="reset"):
        agent.respond("missing", "shoes", 1, 10)


def test_empty_message_returns_safe_recommendations(fixture_catalog_path):
    agent = Agent(fixture_catalog_path)
    agent.reset("s1", {})

    response = agent.respond("s1", "", 1, 10)

    assert response["ask_attribute"] == "category"
    assert response["recommendations"]


def test_component_exception_is_contained_for_every_valid_turn(fixture_catalog_path):
    agent = Agent(fixture_catalog_path)
    agent.reset("s1", {})

    def fail(*_args, **_kwargs):
        raise ValueError("injected parser failure")

    agent.sessions._parser.parse = fail
    for turn in range(1, 11):
        response = agent.respond("s1", "show me shoes", turn, 10)
        assert response["recommendations"]
        assert response["ask_attribute"] in ALLOWED_ATTRIBUTES | {None}


def test_response_builder_removes_invalid_duplicates_and_fills():
    builder = ResponseBuilder({"A", "B", "C"}, ["C", "B", "A"])
    ranked = [
        Candidate("A", score=0.9),
        Candidate("bad", score=0.8),
        Candidate("A", score=0.7),
    ]

    response = builder.build(ranked, QuestionDecision("material", 0.2), top_k=3)

    assert [item["parent_asin"] for item in response["recommendations"]] == [
        "A",
        "C",
        "B",
    ]
    assert response["ask_attribute"] == "material"


def test_trace_sink_is_bounded():
    sink = TraceSink(max_entries=2)
    sink.record({"turn": 1})
    sink.record({"turn": 2})
    sink.record({"turn": 3})

    assert [item["turn"] for item in sink.records] == [2, 3]
