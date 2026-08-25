from __future__ import annotations

import time
from pathlib import Path

from .catalog import CatalogIndex
from .config import RuntimeConfig
from .dense import load_dense_backend
from .models import Candidate, QuestionDecision, RetrievalPlan, SessionState
from .parser import MessageParser
from .question_policy import QuestionPolicy
from .ranker import ConstraintRanker
from .response import ResponseBuilder
from .retrieval import HybridRetriever
from .router import RoutePlanner
from .state import SessionStore
from .tracing import TraceSink


class CompassCartAgent:
    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.catalog = CatalogIndex(catalog_path)
        self.parser = MessageParser()
        self.sessions = SessionStore(self.parser)
        self.router = RoutePlanner(self.config)
        self.dense = load_dense_backend(
            self.config.dense_model_dir,
            self.config.dense_vector_dir,
            self.config.dense_manifest_path,
        )
        self.retriever = HybridRetriever(
            self.catalog, self.dense, rrf_k=self.config.rrf_k
        )
        self.ranker = ConstraintRanker(self.catalog)
        self.question_policy = QuestionPolicy()
        self.response_builder = ResponseBuilder(
            self.catalog.valid_ids,
            self.catalog.popular_ids(self.config.max_recommendations),
        )
        self.traces = TraceSink()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict[str, object]:
        state = self.sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        if not 1 <= turn <= 10:
            raise ValueError("turn must be between 1 and 10")

        started = time.perf_counter()
        fallbacks: list[str] = []
        old_version = state.intent_version
        expected_attribute = (
            state.asked_attributes[-1] if state.asked_attributes else None
        )
        try:
            state = self.sessions.update(
                session_id,
                user_message,
                turn,
                expected_attribute=expected_attribute,
            )
        except Exception:  # noqa: BLE001 - preserve the last valid dialog state.
            fallbacks.append("parser")
            state.turn = turn

        query_text = self._query_text(user_message, state)
        is_override = state.intent_version > old_version
        try:
            plan = self.router.build_plan(state, query_text, is_override=is_override)
            state.route = plan.route
        except Exception:  # noqa: BLE001 - route failure uses lexical browsing.
            fallbacks.append("router")
            plan = RetrievalPlan(route="browsing", query_text=query_text)

        try:
            candidates = self.retriever.retrieve(plan, state)
        except Exception:  # noqa: BLE001 - popular products are the retrieval fallback.
            fallbacks.append("retriever")
            candidates = self._popular_candidates()
        state.candidate_count = len(candidates)

        try:
            ranked = self.ranker.rank(candidates, state)
        except Exception:  # noqa: BLE001 - retain deterministic RRF order.
            fallbacks.append("ranker")
            ranked = candidates

        if not user_message.strip():
            question = QuestionDecision("category")
        else:
            try:
                question = self.question_policy.choose(ranked, state)
            except Exception:  # noqa: BLE001 - recommendations do not require a question.
                fallbacks.append("question")
                question = QuestionDecision(None)
        if (
            question.ask_attribute
            and question.ask_attribute not in state.asked_attributes
        ):
            state.asked_attributes.append(question.ask_attribute)
        state.previous_recommendations = [
            candidate.parent_asin for candidate in ranked[: min(top_k, 10)]
        ]

        response = self.response_builder.build(ranked, question, top_k=top_k)
        elapsed_ms = round((time.perf_counter() - started) * 1_000, 3)
        self.traces.record(
            {
                "session_id": session_id,
                "turn": turn,
                "route": state.route,
                "active_constraints": [
                    (item.attribute, item.value) for item in state.active_constraints()
                ],
                "candidate_count": len(candidates),
                "ask_attribute": question.ask_attribute,
                "fallbacks": fallbacks,
                "elapsed_ms": elapsed_ms,
            }
        )
        return response

    def _popular_candidates(self) -> list[Candidate]:
        return [
            Candidate(
                parent_asin=identifier,
                product=self.catalog.product(identifier),
                source_scores={"fallback": 1.0 / rank},
                score=1.0 / rank,
            )
            for rank, identifier in enumerate(
                self.catalog.popular_ids(self.config.candidate_limit), start=1
            )
        ]

    @staticmethod
    def _query_text(message: str, state: SessionState) -> str:
        values = [item.value for item in state.active_constraints()]
        return " ".join([message, *values]).strip()
