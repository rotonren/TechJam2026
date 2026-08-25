from __future__ import annotations

from agent import Agent
from compasscart.models import Candidate


def _assert_valid(response: dict, valid_ids: set[str]) -> None:
    identifiers = [item["parent_asin"] for item in response["recommendations"]]
    assert response["message"]
    assert identifiers
    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers) <= valid_ids


def test_parser_ranker_and_empty_retrieval_failures_are_contained(
    fixture_catalog_path,
):
    agent = Agent(fixture_catalog_path)
    agent.reset("parser", {})
    agent.sessions._parser.parse = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        ValueError("parser")
    )
    _assert_valid(
        agent.respond("parser", "show me shoes", 1, 10), agent.catalog.valid_ids
    )

    agent = Agent(fixture_catalog_path)
    agent.reset("ranker", {})
    agent.ranker.rank = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        TimeoutError("ranker")
    )
    _assert_valid(
        agent.respond("ranker", "show me shoes", 1, 10), agent.catalog.valid_ids
    )

    agent = Agent(fixture_catalog_path)
    agent.reset("empty", {})
    agent.retriever.retrieve = lambda *_args, **_kwargs: []
    _assert_valid(
        agent.respond("empty", "show me shoes", 1, 10), agent.catalog.valid_ids
    )


def test_fts_error_and_duplicate_invalid_rankings_are_sanitized(
    fixture_catalog_path,
):
    class BrokenConnection:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("fts failed")

    agent = Agent(fixture_catalog_path)
    agent.reset("s1", {})
    agent.catalog._fts_enabled = True
    agent.catalog.connection = BrokenConnection()
    agent.ranker.rank = lambda *_args, **_kwargs: [
        Candidate("bad"),
        Candidate("SHOE1"),
        Candidate("SHOE1"),
    ]

    response = agent.respond("s1", "blue running shoes", 1, 10)

    _assert_valid(response, agent.catalog.valid_ids)
    assert response["recommendations"][0]["parent_asin"] == "SHOE1"


def test_lost_known_session_is_rebuilt_from_reset_profile(fixture_catalog_path):
    agent = Agent(fixture_catalog_path)
    agent.reset("s1", {"preference_tags": ["comfort"]})
    agent.sessions._sessions.pop("s1")

    response = agent.respond("s1", "show me shoes", 1, 10)

    _assert_valid(response, agent.catalog.valid_ids)
    assert agent.sessions.get("s1") is not None


def test_trace_storage_failure_does_not_break_response(fixture_catalog_path):
    class BrokenRecords:
        def append(self, _payload):
            raise OSError("trace unavailable")

    agent = Agent(fixture_catalog_path)
    agent.reset("s1", {})
    agent.traces._records = BrokenRecords()

    response = agent.respond("s1", "show me shoes", 1, 10)

    _assert_valid(response, agent.catalog.valid_ids)
    assert agent.traces.enabled is False
