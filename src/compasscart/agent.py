from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Collection
from pathlib import Path

from .catalog import CatalogIndex
from .config import RuntimeConfig
from .constraints import hard_constraint_violations
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
        self.parser = MessageParser(self.catalog.parser_vocabulary())
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
        self._profiles: OrderedDict[str, dict[str, object]] = OrderedDict()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._profiles[session_id] = dict(user_profile)
        self._profiles.move_to_end(session_id)
        while len(self._profiles) > 1_000:
            self._profiles.popitem(last=False)
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
            profile = self._profiles.get(session_id)
            if profile is None:
                raise RuntimeError("reset must be called before respond")
            state = self.sessions.reset(session_id, profile)
        if not 1 <= turn <= 10:
            raise ValueError("turn must be between 1 and 10")

        started = time.perf_counter()
        fallbacks: list[str] = []
        old_version = state.intent_version
        expected_attribute = state.pending_attribute
        previous_recommendations = set(state.previous_recommendations)
        state.pending_attribute = None
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

        continuation_requested = state.continuation_requested
        excluded_ids = previous_recommendations if continuation_requested else set()

        query_text = self._query_text(user_message, state)
        is_override = state.intent_version > old_version or state.override_scope != "none"
        try:
            plan = self.router.build_plan(state, query_text, is_override=is_override)
            state.route = plan.route
        except Exception:  # noqa: BLE001 - route failure uses lexical browsing.
            fallbacks.append("router")
            plan = RetrievalPlan(route="browsing", query_text=query_text)

        try:
            try:
                candidates = self.retriever.retrieve(
                    plan, state, exclude_ids=excluded_ids
                )
            except TypeError as error:
                # Preserve compatibility with lightweight test/demonstration
                # retrievers that still expose the original two-argument call.
                if "exclude_ids" not in str(error):
                    raise
                candidates = self.retriever.retrieve(plan, state)
        except Exception:  # noqa: BLE001 - popular products are the retrieval fallback.
            fallbacks.append("retriever")
            candidates = self._popular_candidates(plan, excluded_ids)
        if not candidates:
            fallbacks.append("empty_retrieval")
            candidates = self._popular_candidates(plan, excluded_ids)
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
        state.pending_attribute = question.ask_attribute
        state.previous_recommendations = [
            candidate.parent_asin for candidate in ranked[: min(top_k, 10)]
        ]

        response = self.response_builder.build(
            ranked, question, top_k=top_k, excluded_ids=excluded_ids
        )
        # A continuation marker belongs only to this turn.  Clearing it after
        # response construction prevents a later refinement from inheriting the
        # previous-page exclusion behavior.
        state.continuation_requested = False
        elapsed_ms = round((time.perf_counter() - started) * 1_000, 3)
        self.traces.record(
            {
                "session_id": session_id,
                "turn": turn,
                "route": state.route,
                "route_reason": plan.route_reason,
                "intent_version": state.intent_version,
                "override_scope": state.override_scope,
                "active_constraints": [
                    (item.attribute, item.value) for item in state.active_constraints()
                ],
                "candidate_count": len(candidates),
                "relaxed_count": sum(item.relaxed for item in ranked[:top_k]),
                "relaxed_constraints": list(
                    dict.fromkeys(
                        violation
                        for item in ranked[:top_k]
                        if item.relaxed
                        for violation in item.violations
                    )
                ),
                "ask_attribute": question.ask_attribute,
                "fallbacks": fallbacks,
                "elapsed_ms": elapsed_ms,
            }
        )
        return response

    def _popular_candidates(
        self,
        plan: RetrievalPlan | None = None,
        excluded_ids: Collection[str] = (),
    ) -> list[Candidate]:
        excluded = set(excluded_ids)
        identifiers = [
            identifier
            for identifier in self.catalog.popular_ids(self.config.candidate_limit)
            if identifier not in excluded
        ]
        if plan is None:
            return [
                Candidate(
                    parent_asin=identifier,
                    product=self.catalog.product(identifier),
                    source_scores={"fallback": 1.0 / rank},
                    score=1.0 / rank,
                )
                for rank, identifier in enumerate(identifiers, start=1)
            ]

        exact: list[Candidate] = []
        relaxed: list[Candidate] = []
        for identifier in identifiers:
            violations = hard_constraint_violations(
                self.catalog.product(identifier),
                self.catalog.attributes[identifier],
                plan.effective_hard_constraints(),
            )
            target = relaxed if violations else exact
            target.append(
                Candidate(
                    parent_asin=identifier,
                    product=self.catalog.product(identifier),
                    source_scores={"fallback": 1.0 / (len(exact) + len(relaxed) + 1)},
                    score=1.0 / (len(exact) + len(relaxed) + 1),
                    violations=violations,
                    relaxed=bool(violations),
                )
            )
        if not exact and not relaxed and excluded:
            # Last-resort valid IDs when every popular item was excluded.
            for identifier in self.catalog.popular_ids(self.config.candidate_limit):
                violations = hard_constraint_violations(
                    self.catalog.product(identifier),
                    self.catalog.attributes[identifier],
                    plan.effective_hard_constraints(),
                )
                relaxed.append(
                    Candidate(
                        parent_asin=identifier,
                        product=self.catalog.product(identifier),
                        violations=violations,
                        relaxed=bool(violations),
                    )
                )
        return [*exact, *relaxed]

    @staticmethod
    def _query_text(message: str, state: SessionState) -> str:
        evidence = [item.strip() for item in state.query_history[-4:] if item.strip()]
        current = message.strip()
        if current and current not in evidence:
            evidence.append(current)
        values: list[str] = []
        for item in state.active_constraints():
            for value in item.values():
                if value and value not in evidence and value not in values:
                    values.append(value)
            if (
                item.upper_value
                and item.upper_value not in evidence
                and item.upper_value not in values
            ):
                values.append(item.upper_value)
        return " ".join([*evidence, *values]).strip()[:512].rstrip()
