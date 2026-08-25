from __future__ import annotations

import argparse
import os
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO


SCENARIOS: dict[str, tuple[str, ...]] = {
    "browsing": (
        "Any suggestions?",
    ),
    "buying": (
        "I need blue running shoes under $80.",
    ),
    "override": (
        "I need running shoes.",
        "Blue.",
        "Actually, ignore my earlier preference. What I need is a black leather belt for work.",
    ),
    "boundary": (
        "Any suggestions?",
        "I have no preference; please use your judgment.",
        "I still have no preference; please use your judgment.",
    ),
}

DEFAULT_PROFILE: dict[str, object] = {
    "purchase_frequency": "unknown",
    "average_prior_rating": None,
    "rating_style": "unknown",
    "preference_tags": [],
    "summary": "Neutral demo profile; all constraints come from the live conversation.",
}

HELP_TEXT = """Commands:
  /help   Show this help
  /new    Start a new shopping session
  /trace  Show or hide route, constraints, latency, and fallback evidence
  /quit   Exit the demo

Enter shopping requests in English. Examples:
  Any suggestions?
  I need blue running shoes under $80.
  Actually, ignore my earlier preference. What I need is a black leather belt.
"""


def _compact_text(value: object, *, width: int) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "Untitled product"
    if len(text) <= width:
        return text
    return text[: max(width - 3, 1)].rstrip() + "..."


def _number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_product(
    index: int,
    parent_asin: str,
    product: Mapping[str, object],
    *,
    title_width: int = 68,
) -> str:
    title = _compact_text(product.get("title"), width=title_width)
    price = _number(product.get("price"))
    rating = _number(product.get("average_rating"))
    price_text = f"${price:.2f}" if price is not None and price > 0 else "Price unavailable"
    rating_text = f"{rating:g}/5" if rating is not None and rating > 0 else "Rating unavailable"
    return f"{index:>3}. {title} | {price_text} | {rating_text} | {parent_asin}"


def format_response(agent: object, response: Mapping[str, object], *, display_limit: int = 5) -> str:
    lines = [f"CompassCart > {response.get('message', '')}"]
    ask_attribute = response.get("ask_attribute")
    lines.append(f"Structured question: {ask_attribute or 'none'}")
    lines.append("Recommendations:")
    recommendations = response.get("recommendations")
    if isinstance(recommendations, list):
        for index, item in enumerate(recommendations[:display_limit], start=1):
            if not isinstance(item, Mapping):
                continue
            parent_asin = str(item.get("parent_asin", ""))
            if not parent_asin:
                continue
            product = agent.catalog.product(parent_asin)
            lines.append(format_product(index, parent_asin, product))
    if len(lines) == 3:
        lines.append("  No recommendations returned.")
    usage = response.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    lines.append(
        "Tokens: "
        f"prompt={usage.get('prompt_tokens', 0)}, "
        f"completion={usage.get('completion_tokens', 0)}"
    )
    return "\n".join(lines)


def format_trace(agent: object, session_id: str) -> str:
    records = agent.traces.records
    trace = records[-1] if records else {}
    state = agent.sessions.get(session_id)
    intent = getattr(state, "intent_version", "?")
    constraints = trace.get("active_constraints") or []
    constraint_text = ", ".join(f"{key}={value}" for key, value in constraints) or "none"
    fallbacks = trace.get("fallbacks") or []
    fallback_text = ",".join(str(item) for item in fallbacks) or "none"
    dense_text = "on" if bool(getattr(agent.dense, "available", False)) else "off"
    latency = _number(trace.get("elapsed_ms"))
    latency_text = f"{latency:.3f} ms" if latency is not None else "unknown"
    return (
        "Demo evidence | "
        f"route={trace.get('route', 'unknown')} | "
        f"intent=v{intent} | "
        f"constraints={constraint_text} | "
        f"candidates={trace.get('candidate_count', 'unknown')} | "
        f"latency={latency_text} | "
        f"fallbacks={fallback_text} | "
        f"dense={dense_text}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive CompassCart product-search demonstration."
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--scenario", choices=sorted(SCENARIOS))
    parser.add_argument(
        "--lexical",
        action="store_true",
        help="Disable the local dense model and demonstrate deterministic fallback.",
    )
    parser.add_argument(
        "--no-trace",
        action="store_false",
        dest="show_trace",
        default=True,
        help="Hide technical evidence lines.",
    )
    parser.add_argument(
        "--top",
        type=int,
        choices=range(1, 11),
        default=5,
        metavar="N",
        help="Number of recommendations to display (1-10, default: 5).",
    )
    return parser


def _write(output: TextIO, text: str = "") -> None:
    print(text, file=output, flush=True)


def _new_session(agent: object) -> str:
    session_id = f"demo-{uuid.uuid4().hex[:10]}"
    agent.reset(session_id, dict(DEFAULT_PROFILE))
    return session_id


def _show_turn(
    agent: object,
    session_id: str,
    response: Mapping[str, object],
    *,
    output: TextIO,
    display_limit: int,
    show_trace: bool,
) -> None:
    _write(output, format_response(agent, response, display_limit=display_limit))
    if show_trace:
        _write(output, format_trace(agent, session_id))
    _write(output)


def run_scenario(
    agent: object,
    scenario: str,
    *,
    output: TextIO = sys.stdout,
    display_limit: int = 5,
    show_trace: bool = True,
) -> int:
    session_id = _new_session(agent)
    _write(output, f"Guided scenario: {scenario}")
    _write(output, "=" * 72)
    for turn, message in enumerate(SCENARIOS[scenario], start=1):
        _write(output, f"User > {message}")
        response = agent.respond(session_id, message, turn, 10)
        _show_turn(
            agent,
            session_id,
            response,
            output=output,
            display_limit=display_limit,
            show_trace=show_trace,
        )
    _write(output, "Scenario complete.")
    return 0


def interactive_session(
    agent: object,
    *,
    input_fn=input,
    output: TextIO = sys.stdout,
    display_limit: int = 5,
    show_trace: bool = True,
) -> int:
    session_id = _new_session(agent)
    turn = 0
    trace_enabled = show_trace
    _write(output, "Interactive demo ready. Type /help for commands.")
    _write(output, "Use English shopping requests for reliable attribute extraction.")
    _write(output)

    while True:
        try:
            message = input_fn("You > ").strip()
        except (EOFError, KeyboardInterrupt, StopIteration):
            _write(output, "\nDemo ended.")
            return 0

        if not message:
            continue
        command = message.lower()
        if command == "/quit":
            _write(output, "Demo ended.")
            return 0
        if command == "/help":
            _write(output, HELP_TEXT.rstrip())
            continue
        if command == "/new":
            session_id = _new_session(agent)
            turn = 0
            _write(output, "New shopping session started.")
            continue
        if command == "/trace":
            trace_enabled = not trace_enabled
            status = "shown" if trace_enabled else "hidden"
            _write(output, f"Demo evidence is now {status}.")
            continue
        if command.startswith("/"):
            _write(output, "Unknown command. Type /help for available commands.")
            continue
        if turn >= 10:
            _write(output, "Turn 10 reached. Type /new to start another session.")
            continue

        next_turn = turn + 1
        try:
            response = agent.respond(session_id, message, next_turn, 10)
        except Exception as exc:  # noqa: BLE001 - keep the live demo recoverable.
            _write(output, f"CompassCart could not answer this turn: {exc}")
            continue
        turn = next_turn
        _show_turn(
            agent,
            session_id,
            response,
            output=output,
            display_limit=display_limit,
            show_trace=trace_enabled,
        )
        if turn == 10:
            _write(output, "Turn 10 reached. Type /new to start another session.")


def main(
    argv: list[str] | None = None,
    *,
    output: TextIO = sys.stdout,
    agent_class=None,
) -> int:
    args = build_parser().parse_args(argv)
    if not args.catalog.is_file():
        _write(output, f"Catalog not found: {args.catalog}")
        _write(output, "Place the frozen catalog at data/catalog.jsonl and try again.")
        return 2
    if args.lexical:
        os.environ["COMPASSCART_DISABLE_DENSE"] = "1"

    if agent_class is None:
        from agent import Agent

        agent_class = Agent

    _write(output, "Loading the 50,000-product catalog. First start may take 15 seconds...")
    try:
        agent = agent_class(args.catalog)
    except Exception as exc:  # noqa: BLE001 - provide a concise demo startup error.
        _write(output, f"Could not start CompassCart: {exc}")
        return 2

    dense_status = "on" if bool(getattr(agent.dense, "available", False)) else "off"
    _write(output, f"CompassCart loaded. Dense retrieval: {dense_status}. Runtime API cost: $0.")
    _write(output)
    if args.scenario:
        return run_scenario(
            agent,
            args.scenario,
            output=output,
            display_limit=args.top,
            show_trace=args.show_trace,
        )
    return interactive_session(
        agent,
        output=output,
        display_limit=args.top,
        show_trace=args.show_trace,
    )


if __name__ == "__main__":
    raise SystemExit(main())
