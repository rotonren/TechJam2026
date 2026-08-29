from __future__ import annotations

import inspect
import time
from collections import OrderedDict
from collections.abc import Collection
from pathlib import Path

from .catalog import CatalogIndex
from .config import RuntimeConfig
from .constraints import hard_constraint_violations
from .dense import NullDenseBackend, load_dense_backend
from .evolution import PolicyMemory, profile_segment
from .models import Candidate, QuestionDecision, RetrievalPlan, SessionState
from .orchestration import StrategyDecision, StrategySelector
from .parser import MessageParser
from .question_policy import QuestionPolicy
from .ranker import ConstraintRanker
from .rerank import RerankStage, load_rerank_backend
from .response import ResponseBuilder
from .retrieval import HybridRetriever
from .router import RoutePlanner
from .state import SessionStore
from .tracing import TraceSink

SUBMISSION_ROOT = Path(__file__).resolve().parents[2]


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
        try:
            dense_paths = self.config.resolve_dense_paths(SUBMISSION_ROOT)
        except ValueError:
            self.dense = NullDenseBackend("layout_invalid")
        else:
            self.dense = load_dense_backend(*dense_paths)
        self.retriever = HybridRetriever(
            self.catalog,
            self.dense,
            rrf_k=self.config.rrf_k,
            dense_rescue_only=self.config.dense_rescue_only,
        )
        self.ranker = ConstraintRanker(
            self.catalog,
            fusion_weight=self.config.rank_fusion_weight,
            attribute_weight=self.config.rank_attribute_weight,
            consensus_bonus=self.config.rank_consensus_bonus,
            boundary_bonus=self.config.rank_boundary_bonus,
            mmr_lambda=self.config.mmr_lambda,
            adaptive_browsing_mmr=self.config.adaptive_browsing_mmr,
        )
        rerank_assets = self.config.resolve_rerank_asset_dir(SUBMISSION_ROOT)

        def rerank_backend(name: str):
            return load_rerank_backend(
                enabled=self.config.rerank_enabled,
                backend=name,
                asset_dir=rerank_assets,
                max_length=self.config.rerank_max_length,
                structured_prompt=self.config.rerank_structured_prompt,
            )

        browsing_backend = rerank_backend(self.config.rerank_backend)
        self.reranker = RerankStage(
            browsing_backend,
            window=self.config.rerank_window,
            buying_window=self.config.rerank_buying_window,
            weight=self.config.rerank_weight,
            buying_weight=self.config.rerank_buying_weight,
            buying_requires_override=self.config.rerank_buying_requires_override,
            buying_backend=(
                rerank_backend(self.config.rerank_buying_backend)
                if self.config.rerank_buying_backend
                else browsing_backend
            ),
        )
        self.memory = PolicyMemory(enabled=self.config.evolution_enabled)
        self.strategy = StrategySelector(enabled=self.config.strategy_enabled)
        self.question_policy = QuestionPolicy(self.catalog.attributes, self.memory)
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
        state = self.sessions.reset(session_id, user_profile)
        state.profile_segment = profile_segment(user_profile)

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
            state.profile_segment = profile_segment(profile)
        if not 1 <= turn <= 10:
            raise ValueError("turn must be between 1 and 10")

        started = time.perf_counter()
        deadline = started + max(self.config.component_timeout_ms, 0) / 1_000.0
        fallbacks: list[str] = []
        budget_skips: list[str] = []
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        old_version = state.intent_version
        expected_attribute = state.pending_attribute
        previous_recommendations = set(state.previous_recommendations)
        refusals_before = set(state.no_preference_attributes)
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

        self._observe_question_outcome(state, expected_attribute, refusals_before)
        continuation_requested = state.continuation_requested
        excluded_ids = previous_recommendations if continuation_requested else set()

        query_text = self._query_text(user_message, state)
        is_override = state.intent_version > old_version or state.override_scope != "none"
        strategy = StrategyDecision()
        try:
            plan = self.router.build_plan(state, query_text, is_override=is_override)
            state.route = plan.route
        except Exception:  # noqa: BLE001 - route failure uses lexical browsing.
            fallbacks.append("router")
            plan = RetrievalPlan(route="browsing", query_text=query_text)

        try:
            retrieve_keywords = self._supported_keywords(
                self.retriever.retrieve,
                ("exclude_ids", "deadline", "diagnostics"),
            )
            retrieve_args: dict[str, object] = {}
            if "exclude_ids" in retrieve_keywords:
                retrieve_args["exclude_ids"] = excluded_ids
            if "deadline" in retrieve_keywords:
                retrieve_args["deadline"] = deadline
            if "diagnostics" in retrieve_keywords:
                retrieve_args["diagnostics"] = budget_skips
            candidates = self.retriever.retrieve(plan, state, **retrieve_args)
        except Exception:  # noqa: BLE001 - popular products are the retrieval fallback.
            fallbacks.append("retriever")
            candidates = self._popular_candidates(plan, excluded_ids)
        if not candidates:
            fallbacks.append("empty_retrieval")
            candidates = self._popular_candidates(plan, excluded_ids)
        state.candidate_count = len(candidates)

        try:
            rank_keywords = self._supported_keywords(
                self.ranker.rank, ("deadline", "diagnostics")
            )
            rank_args: dict[str, object] = {}
            if "deadline" in rank_keywords:
                rank_args["deadline"] = deadline
            if "diagnostics" in rank_keywords:
                rank_args["diagnostics"] = budget_skips
            ranked = self.ranker.rank(candidates, state, **rank_args)
        except Exception:  # noqa: BLE001 - retain deterministic RRF order.
            fallbacks.append("ranker")
            ranked = candidates

        try:
            ranked = self.reranker.apply(
                ranked,
                state,
                deadline=deadline,
                diagnostics=budget_skips,
                usage=usage,
            )
        except Exception:  # noqa: BLE001 - keep the constraint ranker's order.
            fallbacks.append("reranker")

        if not user_message.strip():
            question = QuestionDecision("category")
        else:
            try:
                question = self.question_policy.choose(ranked, state)
            except Exception:  # noqa: BLE001 - recommendations do not require a question.
                fallbacks.append("question")
                question = QuestionDecision(None)
            try:
                strategy = self.strategy.select(
                    state, structured_question=question.ask_attribute
                )
                if strategy.open_question:
                    question = QuestionDecision("other", 1.0)
            except Exception:  # noqa: BLE001 - keep the structured decision.
                fallbacks.append("strategy")
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
            ranked, question, top_k=top_k, excluded_ids=excluded_ids, usage=usage
        )
        # A continuation marker belongs only to this turn.  Clearing it after
        # response construction prevents a later refinement from inheriting the
        # previous-page exclusion behavior.
        state.continuation_requested = False
        elapsed_ms = round((time.perf_counter() - started) * 1_000, 3)
        dense_status = getattr(self.dense, "status", "unknown")
        if not isinstance(dense_status, str) or not dense_status.strip():
            dense_status = "unknown"
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
                "dense_status": dense_status,
                "rerank_status": getattr(
                    self.reranker.backend_for(state), "status", "unknown"
                ),
                "strategy": strategy.name,
                "strategy_reason": strategy.reason,
                "usage": dict(usage),
                "stall_count": state.stall_count,
                "unproductive_attributes": sorted(state.unproductive_attributes),
                "fallbacks": fallbacks,
                "budget_skips": budget_skips,
                "elapsed_ms": elapsed_ms,
            }
        )
        return response

    def _observe_question_outcome(
        self,
        state: SessionState,
        asked_attribute: str | None,
        refusals_before: set[str],
    ) -> None:
        """Score the previous turn's question against what the reply carried.

        The reply to a clarification either states a requirement or refuses the
        attribute, and that is exactly the quantity the response-likelihood
        table estimates. Recording it here is what lets the memory correct a
        prior we guessed.

        Refusal is the signal, not whether a structured constraint came out of
        it. A requirement stated as free text often parses to nothing yet still
        reaches retrieval and the rerank stage as query evidence, so counting
        parsed constraints would measure the parser rather than the shopper.
        """
        refused = bool(
            asked_attribute
            and asked_attribute in state.no_preference_attributes - refusals_before
        )
        disclosed = bool(asked_attribute) and not refused
        # A stall is the shopper pushing back or having nothing to add; a turn
        # where no question was pending is simply not evidence either way.
        if refused or state.continuation_requested:
            state.stall_count += 1
        elif disclosed:
            state.stall_count = 0
            state.disclosure_count += 1
        if not asked_attribute:
            return
        if not disclosed:
            # Distilled negative evidence: this attribute had nothing to give in
            # this session, so a later turn should spend its question elsewhere.
            state.unproductive_attributes.add(asked_attribute)
        try:
            self.memory.observe(
                asked_attribute,
                disclosed,
                context=state.route,
                segment=state.profile_segment,
            )
        except Exception:  # noqa: BLE001 - learning must never break a turn.
            # Stop learning rather than risk a partially updated estimate; the
            # policy falls back to the hand-written prior from here on.
            self.memory.enabled = False

    @staticmethod
    def _supported_keywords(callable_: object, names: tuple[str, ...]) -> set[str]:
        try:
            parameters = inspect.signature(callable_).parameters.values()
        except (TypeError, ValueError):
            return set(names)
        if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters):
            return set(names)
        return {
            item.name
            for item in parameters
            if item.name in names
            and item.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }

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
