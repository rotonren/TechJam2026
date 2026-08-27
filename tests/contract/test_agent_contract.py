from __future__ import annotations

import pytest

from agent import Agent
from compasscart.models import Candidate, Constraint, QuestionDecision, SessionState
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


def test_agent_trace_accepts_legacy_dense_backend_without_status(
    fixture_catalog_path,
    monkeypatch,
):
    class LegacyDenseBackend:
        available = True

        def search(self, _text: str, _limit: int) -> list[Candidate]:
            return []

    monkeypatch.setenv("COMPASSCART_DISABLE_DENSE", "1")
    agent = Agent(fixture_catalog_path)
    legacy_dense = LegacyDenseBackend()
    agent.dense = legacy_dense
    agent.retriever.dense = legacy_dense
    agent.reset("legacy-dense", {})

    response = agent.respond("legacy-dense", "running shoes", turn=1, top_k=3)

    assert set(response) == {"message", "ask_attribute", "recommendations", "usage"}
    assert agent.traces.records[-1]["dense_status"] == "unknown"


def test_agent_requires_reset(fixture_catalog_path):
    agent = Agent(fixture_catalog_path)

    with pytest.raises(RuntimeError, match="reset"):
        agent.respond("missing", "shoes", 1, 10)


def test_agent_question_policy_wires_catalog_and_parser_support(fixture_catalog_path):
    agent = Agent(fixture_catalog_path)

    assert agent.question_policy.attribute_lookup is agent.catalog.attributes
    assert agent.question_policy.parser_support("material", "cotton") is True
    assert agent.question_policy.retrieval_support("material", "cotton") is True
    assert agent.question_policy.retrieval_support("material", "unobtainium") is False


def test_agent_applies_bare_catalog_size_answer_as_a_hard_clarification(
    fixture_catalog_path,
    monkeypatch,
):
    monkeypatch.setenv("COMPASSCART_DISABLE_DENSE", "1")
    agent = Agent(fixture_catalog_path)
    decisions = iter(
        (QuestionDecision("size", 0.2), QuestionDecision(None))
    )
    monkeypatch.setattr(
        agent.question_policy,
        "choose",
        lambda *_args: next(decisions),
    )
    agent.reset("size-clarification", {})

    question = agent.respond(
        "size-clarification", "I need running shoes", turn=1, top_k=1
    )
    response = agent.respond(
        "size-clarification", "10 wide", turn=2, top_k=1
    )

    state = agent.sessions.get("size-clarification")
    assert question["ask_attribute"] == "size"
    assert state is not None
    assert [
        (item.value, item.is_hard, item.source)
        for item in state.active_constraints()
        if item.attribute == "size"
    ] == [("10 wide", True, "clarification")]
    assert [
        item["parent_asin"] for item in response["recommendations"]
    ] == ["SHOE1"]


def test_empty_message_returns_safe_recommendations(fixture_catalog_path):
    agent = Agent(fixture_catalog_path)
    agent.reset("s1", {})

    response = agent.respond("s1", "", 1, 10)

    assert response["ask_attribute"] == "category"
    assert response["recommendations"]


def test_empty_message_on_last_turn_returns_recommendations_without_question(
    fixture_catalog_path,
):
    agent = Agent(fixture_catalog_path)
    agent.reset("last-empty", {})

    response = agent.respond("last-empty", "", 10, 10)

    assert response["ask_attribute"] is None
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


def test_agent_query_uses_bounded_dialog_evidence():
    from compasscart.models import SessionState

    state = SessionState("s1")
    state.query_history = ["alloy buckle closure", "black belt"]

    query = Agent._query_text("current message", state)

    assert "alloy buckle closure" in query
    assert "black belt" in query
    assert query.count("current message") == 1


def test_agent_query_preserves_all_alternative_constraint_values():
    state = SessionState("s1")
    state.constraints = [
        Constraint(
            "color",
            "black",
            1.0,
            True,
            "message",
            1,
            1,
            operator="in",
            alternatives=("black", "blue"),
        )
    ]

    query = Agent._query_text("", state)

    assert "black" in query
    assert "blue" in query
