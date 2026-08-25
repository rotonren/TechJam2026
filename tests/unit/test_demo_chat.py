from __future__ import annotations

import io
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from compasscart.models import Constraint, SessionState
from tools.demo_chat import (
    DEFAULT_PROFILE,
    SCENARIOS,
    build_parser,
    format_product,
    format_response,
    format_trace,
    interactive_session,
    main,
    run_scenario,
)


@dataclass
class _State:
    intent_version: int = 2


class _Sessions:
    def get(self, _session_id: str) -> _State:
        return _State()


class _Catalog:
    products: ClassVar[dict[str, dict[str, object]]] = {
        "A1": {
            "title": "Blue Trail Running Shoe",
            "price": 79.99,
            "average_rating": 4.5,
        },
        "A2": {"title": "A very long " + "product " * 20},
    }

    def product(self, parent_asin: str) -> dict[str, object]:
        return self.products[parent_asin]


class _Dense:
    available = True


class _Traces:
    records: ClassVar[list[dict[str, object]]] = [
        {
            "route": "buying",
            "route_reason": "explicit budget constraint",
            "intent_version": 2,
            "active_constraints": [("color", "black"), ("material", "leather")],
            "candidate_count": 42,
            "ask_attribute": "size",
            "relaxed_count": 1,
            "relaxed_constraints": ["budget<=80"],
            "elapsed_ms": 123.456,
            "fallbacks": [],
        }
    ]


class _Agent:
    catalog = _Catalog()
    dense = _Dense()
    sessions = _Sessions()
    traces = _Traces()


class _FakeAgent:
    def __init__(self, _catalog: str | Path) -> None:
        self.catalog = _Catalog()
        self.dense = _Dense()
        self.sessions = _Sessions()
        self.traces = _Traces()
        self.reset_calls: list[tuple[str, dict[str, object]]] = []
        self.respond_calls: list[tuple[str, str, int, int]] = []

    def reset(self, session_id: str, profile: dict[str, object]) -> None:
        self.reset_calls.append((session_id, profile))

    def respond(
        self, session_id: str, message: str, turn: int, top_k: int
    ) -> dict[str, object]:
        self.respond_calls.append((session_id, message, turn, top_k))
        return {
            "message": "Which material would you prefer?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "A1"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


def test_format_product_includes_readable_metadata():
    text = format_product(1, "A1", _Catalog.products["A1"])

    assert text == "  1. Blue Trail Running Shoe | $79.99 | 4.5/5 | A1"


def test_format_product_handles_missing_values_and_long_titles():
    text = format_product(2, "A2", _Catalog.products["A2"], title_width=36)

    assert "Price unavailable" in text
    assert "Rating unavailable" in text
    assert "..." in text
    assert len(text.split(" | ")[0]) <= 41


def test_format_response_shows_question_products_and_zero_usage():
    response = {
        "message": "Which material would you prefer?",
        "ask_attribute": "material",
        "recommendations": [{"parent_asin": "A1"}, {"parent_asin": "A2"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }

    text = format_response(_Agent(), response, display_limit=1)

    assert "CompassCart > Which material would you prefer?" in text
    assert "Structured question: material" in text
    assert "Blue Trail Running Shoe" in text
    assert "A very long" not in text
    assert "Tokens: prompt=0, completion=0" in text


def test_format_trace_exposes_demo_evidence():
    text = format_trace(_Agent(), "session-1")

    assert "route=buying" in text
    assert "route_reason=explicit budget constraint" in text
    assert "intent=v2" in text
    assert "color=black" in text
    assert "material=leather" in text
    assert "candidates=42" in text
    assert "ask=size" in text
    assert "relaxed=1" in text
    assert "relaxed_constraints=budget<=80" in text
    assert "latency=123.456 ms" in text
    assert "fallbacks=none" in text
    assert "dense=on" in text


def test_format_trace_preserves_constraint_operators():
    state = SessionState(
        "session-1",
        constraints=[
            Constraint("budget", "50.00", 1.0, True, "message", 1, 1, operator="gte"),
            Constraint(
                "material",
                "leather",
                1.0,
                True,
                "message",
                1,
                1,
                operator="not_in",
                alternatives=("leather",),
            ),
        ],
    )
    agent = SimpleNamespace(
        traces=_Traces(),
        sessions=SimpleNamespace(get=lambda _session_id: state),
        dense=_Dense(),
    )

    text = format_trace(agent, "session-1")

    assert "budget>=50.00" in text
    assert "material not in (leather)" in text


def test_guided_scenarios_cover_four_judging_behaviors():
    assert set(SCENARIOS) == {"browsing", "buying", "override", "boundary"}
    assert all(SCENARIOS[name] for name in SCENARIOS)
    assert "Actually" in " ".join(SCENARIOS["override"])
    assert "no preference" in " ".join(SCENARIOS["boundary"]).lower()
    assert SCENARIOS["override"] == (
        "I need running shoes.",
        "Blue.",
        "Actually, ignore my earlier preference. What I need is a black leather belt for work.",
    )
    assert DEFAULT_PROFILE["preference_tags"] == []


def test_parser_exposes_live_demo_options():
    args = build_parser().parse_args(
        ["--catalog", "catalog.jsonl", "--scenario", "override", "--lexical", "--top", "3"]
    )

    assert args.catalog == Path("catalog.jsonl")
    assert args.scenario == "override"
    assert args.lexical is True
    assert args.top == 3


def test_demo_module_help_runs_without_pythonpath():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "-m", "tools.demo_chat", "--help"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Interactive CompassCart product-search demonstration." in completed.stdout


def test_run_scenario_sends_each_turn_and_prints_trace():
    agent = _FakeAgent("unused")
    output = io.StringIO()

    run_scenario(agent, "override", output=output, display_limit=2, show_trace=True)

    assert [call[1] for call in agent.respond_calls] == list(SCENARIOS["override"])
    assert [call[2] for call in agent.respond_calls] == [1, 2, 3]
    assert "User > I need running shoes." in output.getvalue()
    assert "Demo evidence" in output.getvalue()


def test_interactive_commands_reset_toggle_trace_and_quit():
    agent = _FakeAgent("unused")
    inputs = iter(["blue shoes", "/trace", "leather", "/new", "/quit"])
    output = io.StringIO()

    code = interactive_session(
        agent,
        input_fn=lambda _prompt: next(inputs),
        output=output,
        display_limit=5,
        show_trace=True,
    )

    assert code == 0
    assert len(agent.reset_calls) == 2
    assert [call[2] for call in agent.respond_calls] == [1, 2]
    assert output.getvalue().count("Demo evidence |") == 1
    assert "New shopping session started." in output.getvalue()


def test_interactive_session_handles_eof_and_ten_turn_limit():
    agent = _FakeAgent("unused")
    inputs = iter([*[f"message {index}" for index in range(1, 12)], "/quit"])
    output = io.StringIO()

    interactive_session(
        agent,
        input_fn=lambda _prompt: next(inputs),
        output=output,
        display_limit=5,
        show_trace=False,
    )

    assert len(agent.respond_calls) == 10
    assert "Turn 10 reached" in output.getvalue()

    eof_output = io.StringIO()
    assert (
        interactive_session(
            _FakeAgent("unused"),
            input_fn=lambda _prompt: (_ for _ in ()).throw(EOFError),
            output=eof_output,
            display_limit=5,
            show_trace=False,
        )
        == 0
    )
    assert "Demo ended." in eof_output.getvalue()


def test_main_reports_missing_catalog_without_traceback(tmp_path):
    output = io.StringIO()

    code = main(
        ["--catalog", str(tmp_path / "missing.jsonl")],
        output=output,
        agent_class=_FakeAgent,
    )

    assert code == 2
    assert "Catalog not found" in output.getvalue()
