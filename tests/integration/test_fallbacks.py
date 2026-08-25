from __future__ import annotations

import json

from agent import Agent
from compasscart.models import Candidate


def _assert_valid(response: dict, valid_ids: set[str]) -> None:
    identifiers = [item["parent_asin"] for item in response["recommendations"]]
    assert response["message"]
    assert identifiers
    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers) <= valid_ids


def _ids(response: dict) -> list[str]:
    return [item["parent_asin"] for item in response["recommendations"]]


def _twelve_product_catalog(tmp_path):
    path = tmp_path / "twelve-products.jsonl"
    prices = (40.0, 42.0, 45.0, 49.0, 50.0, 65.0, 80.0, 100.0, 101.0, 125.0, 150.0, 175.0)
    products = [
        {
            "parent_asin": f"P{index:02d}",
            "title": f"{color.title()} running shoe {index}",
            "features": ["breathable"],
            "details": {"Color": color},
            "categories": ["Clothing", "Shoes"],
            "price": price,
            "average_rating": 4.0,
            "rating_number": 100 + index,
        }
        for index, (price, color) in enumerate(
            zip(prices, ("black", "blue", "red") * 4, strict=True)
        )
    ]
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in products), encoding="utf-8"
    )
    return path


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


def test_agent_trace_exposes_route_and_relaxation_evidence(fixture_catalog_path):
    agent = Agent(fixture_catalog_path)
    agent.reset("s1", {})

    response = agent.respond("s1", "still exploring shoes under $50", 1, 10)

    _assert_valid(response, agent.catalog.valid_ids)
    assert "relaxing budget<=50.00" in response["message"]
    trace = agent.traces.records[-1]
    assert trace["route_reason"] == "explicit_buying"
    assert trace["relaxed_count"] > 0
    assert "budget<=50.00" in trace["relaxed_constraints"]


def test_first_category_query_returns_the_singular_plural_exact_match(
    fixture_catalog_path,
):
    agent = Agent(fixture_catalog_path)
    agent.reset("belt", {})

    response = agent.respond("belt", "I need a belt", 1, 1)

    assert _ids(response) == ["BELT1"]
    assert agent.traces.records[-1]["relaxed_count"] == 0


def test_constraint_matrix_keeps_exact_prices_and_colors_ahead_of_alternatives(tmp_path):
    agent = Agent(_twelve_product_catalog(tmp_path))
    agent.reset("under", {})

    under = agent.respond("under", "I need shoes under $50", 1, 10)
    under_ids = _ids(under)
    under_prices = [float(agent.catalog.product(identifier)["price"]) for identifier in under_ids]
    assert all(price <= 50.0 for price in under_prices[:5])
    assert all(price > 50.0 for price in under_prices[5:])

    state = agent.sessions.get("under")
    assert state is not None
    plan = agent.router.build_plan(state, "I need shoes under $50")
    ranked = agent.ranker.rank(agent.retriever.retrieve(plan, state), state)
    assert all(not candidate.relaxed for candidate in ranked[:5])
    assert all(candidate.relaxed for candidate in ranked[5:])

    agent.reset("between", {})
    between = agent.respond("between", "I need shoes between $50 and $100", 1, 4)
    between_prices = [
        float(agent.catalog.product(identifier)["price"])
        for identifier in _ids(between)
    ]
    assert all(50.0 <= price <= 100.0 for price in between_prices)

    agent.reset("colors", {})
    colors = agent.respond("colors", "I need black or blue shoes", 1, 8)
    selected_colors = {
        agent.catalog.attributes[identifier]["color"][0]
        for identifier in _ids(colors)
    }
    assert selected_colors <= {"black", "blue"}


def test_show_me_more_excludes_previous_batch_but_refinement_does_not(
    fixture_catalog_path,
):
    agent = Agent(fixture_catalog_path)
    agent.reset("more", {})

    first = agent.respond("more", "show me shoes", 1, 2)
    first_ids = _ids(first)
    more = agent.respond("more", "show me more", 2, 2)
    more_ids = _ids(more)

    assert first_ids
    assert more_ids
    assert set(first_ids).isdisjoint(more_ids)
    state = agent.sessions.get("more")
    assert state is not None
    assert state.continuation_requested is False

    refined = agent.respond("more", "blue shoes", 3, 2)
    refined_ids = _ids(refined)
    assert set(refined_ids) & set(first_ids)


def test_empty_retrieval_uses_constraint_aware_fallback(fixture_catalog_path):
    agent = Agent(fixture_catalog_path)
    agent.reset("empty-hard", {})
    agent.retriever.retrieve = lambda *_args, **_kwargs: []

    response = agent.respond("empty-hard", "I need shoes under $80", 1, 10)

    prices = [
        float(agent.catalog.product(identifier)["price"])
        for identifier in _ids(response)
    ]
    # SHOE1 is the exact fixture match; any additional IDs must be disclosed
    # alternatives, but no silent over-budget item may be returned as exact.
    assert prices[0] <= 80.0
    assert "relaxing" in response["message"]
