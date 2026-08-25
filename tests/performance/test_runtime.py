from __future__ import annotations

import statistics
import time

from agent import Agent


def test_800_sessions_are_bounded_valid_and_fast(fixture_catalog_path, monkeypatch):
    monkeypatch.setenv("COMPASSCART_DISABLE_DENSE", "1")
    agent = Agent(fixture_catalog_path)
    latencies = []

    for index in range(800):
        session_id = f"session-{index}"
        agent.reset(session_id, {"preference_tags": ["comfort"]})
        started = time.perf_counter()
        response = agent.respond(session_id, "show me running shoes", 1, 10)
        latencies.append((time.perf_counter() - started) * 1_000)
        identifiers = [item["parent_asin"] for item in response["recommendations"]]
        assert identifiers
        assert set(identifiers) <= agent.catalog.valid_ids

    p95 = sorted(latencies)[int(len(latencies) * 0.95) - 1]
    assert p95 < 2_000
    assert len(agent.sessions._sessions) <= 1_000
    assert len(agent.traces.records) <= 5_000
    assert statistics.fmean(latencies) < 500
