"""Semantic rerank stage applied to the head of the ranked candidate list.

Retrieval already places the target product inside the candidate pool for the
overwhelming majority of sessions; what costs score is its *position*.  The
fusion and constraint stages both reason at term level, so a requirement that
the shopper stated as a contiguous phrase ("water resistant rubber outsole")
scores no better than a candidate that merely mentions the same words far
apart.  This stage rescores the head of the list with a phrase-aware model and
reorders it, leaving the tail and the exact/relaxed partition untouched.

Adjacency is deliberately the only signal this stage adds.  Term rarity is
already priced in upstream by BM25, and an experiment that re-weighted these
scores by window-local inverse document frequency measured `-0.011`
TechnicalScore against this design: the rerank window is selected *by* the
query, so the shopper's own terms are common inside it and a local statistic
penalizes exactly the evidence that matters.

The backend is pluggable.  `PhraseMatchBackend` is the default because it needs
no model asset, no optional dependency, and no network, so it cannot weaken the
offline guarantee.  A cross-encoder or hosted-model backend can replace it
behind the same protocol without touching the orchestrator.
"""

from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import Candidate, Constraint, SessionState
from .normalization import TOKEN_RE, flatten_text, searchable_fields

# Function words carry no retrieval signal and would otherwise inflate the
# coverage denominator of a conversational phrase.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "has", "have", "i", "in", "is", "it", "its", "me", "my", "of", "on",
        "or", "please", "s", "should", "so", "some", "that", "the", "then",
        "there", "these", "they", "this", "to", "was", "we", "what", "with",
        "would", "you", "your",
    }
)
_MAX_PRODUCT_TOKENS = 600
_MAX_EVIDENCE_TOKENS = 48
# A contiguous run this long already identifies a product; longer runs should
# not keep growing the score and crowding out coverage evidence.
_RUN_SATURATION = 6
_EVIDENCE_WEIGHTS = {
    "hard": 1.0,
    "message": 0.8,
    "soft": 0.6,
    "profile": 0.2,
}

Evidence = tuple[tuple[tuple[str, ...], float], ...]


@dataclass(frozen=True)
class RerankContext:
    """The dialog state grouped by the role each requirement plays.

    `Evidence` is deliberately lossy: it flattens the ledger into deduplicated
    tokens, which is what a lexical scorer wants and is exactly what a model
    cannot use. Which requirements bind, which are merely preferred, what the
    shopper ruled out, and what only came from a profile all disappear - a
    budget of `80.00` even survives flattening as the two tokens `80` and `00`.
    Backends that can reason about structure take this instead.
    """

    hard: tuple[str, ...] = ()
    stated: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    inferred: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()
    is_override: bool = False

    def is_empty(self) -> bool:
        return not (self.hard or self.stated or self.excluded or self.messages)


def describe_constraint(item: Constraint) -> str:
    """Render one ledger entry the way a shopper would have stated it."""
    money = item.attribute == "budget"

    def amount(value: str) -> str:
        return f"${value}" if money else value

    values = item.values()
    if item.operator == "lte":
        detail = f"at most {amount(item.value)}"
    elif item.operator == "gte":
        detail = f"at least {amount(item.value)}"
    elif item.operator == "between" and item.upper_value:
        detail = f"between {amount(item.value)} and {amount(item.upper_value)}"
    elif len(values) > 1:
        detail = "one of " + ", ".join(values)
    else:
        detail = amount(item.value)
    return f"{item.attribute}: {detail}"


def session_context(state: SessionState) -> RerankContext:
    """Group the active ledger so a model can tell the roles apart.

    Grouping by `is_hard` alone conflates two different things: a preference
    the agent inferred weakly, and a requirement the shopper stated in words
    the parser could not turn into a filter. The second is not optional. It is
    what an override turn produces - the replacement requirement arrives as
    free text and is stored soft so it cannot over-filter retrieval - so
    grouping it as "helpful, not required" tells the model the opposite of what
    the shopper just said.
    """
    grouped: dict[str, list[str]] = {
        "hard": [],
        "stated": [],
        "excluded": [],
        "inferred": [],
    }
    for item in state.active_constraints():
        if item.operator == "not_in":
            role = "excluded"
        elif item.source in {"profile", "inferred"}:
            role = "inferred"
        elif item.is_hard:
            role = "hard"
        else:
            role = "stated"
        grouped[role].append(describe_constraint(item))
    return RerankContext(
        hard=tuple(dict.fromkeys(grouped["hard"])),
        stated=tuple(dict.fromkeys(grouped["stated"])),
        excluded=tuple(dict.fromkeys(grouped["excluded"])),
        inferred=tuple(dict.fromkeys(grouped["inferred"])),
        messages=tuple(
            message.strip() for message in state.query_history[-3:] if message.strip()
        ),
        is_override=state.intent_version > 1 or state.override_scope != "none",
    )


def render_requirements(context: RerankContext, query: str) -> str:
    """Format the context as the requirements half of a model prompt."""
    if context.is_empty():
        return f"Requirements: {query}"
    blocks = (
        ("Hard requirements (a product must satisfy all):", context.hard, "- {}"),
        ("Must NOT have:", context.excluded, "- {}"),
        (
            "Also stated by the shopper, phrased loosely but still wanted:",
            context.stated,
            "- {}",
        ),
        ("What the shopper said, most recent last:", context.messages, '- "{}"'),
        (
            "Weak signal inferred, not stated by the shopper:",
            context.inferred,
            "- {}",
        ),
    )
    sections = [
        heading + "\n" + "\n".join(bullet.format(item) for item in items)
        for heading, items, bullet in blocks
        if items
    ]
    return "\n\n".join(sections)


class RerankBackend(Protocol):
    @property
    def available(self) -> bool: ...

    @property
    def status(self) -> str: ...

    def scores(
        self,
        evidence: Evidence,
        candidates: Sequence[Candidate],
        *,
        context: RerankContext | None = None,
    ) -> list[float]: ...


class NullRerankBackend:
    available = False

    def __init__(self, status: str = "disabled") -> None:
        self.status = status

    def scores(
        self,
        evidence: Evidence,
        candidates: Sequence[Candidate],
        *,
        context: RerankContext | None = None,
    ) -> list[float]:
        del evidence, context
        return [0.0] * len(candidates)


def token_sequence(text: object, limit: int | None = None) -> tuple[str, ...]:
    """Tokenize while preserving order and repetition.

    `normalization.terms` deduplicates, which destroys the adjacency this stage
    depends on, so phrase matching needs its own tokenizer.
    """
    tokens = [
        sys.intern(token.lower()) for token in TOKEN_RE.findall(flatten_text(text))
    ]
    return tuple(tokens if limit is None else tokens[:limit])


def longest_contiguous_run(
    evidence: Sequence[str], positions: dict[str, list[int]]
) -> int:
    """Length of the longest run of `evidence` appearing contiguously."""
    best = 0
    previous: dict[int, int] = {}
    for token in evidence:
        current: dict[int, int] = {}
        for index in positions.get(token, ()):
            length = previous.get(index - 1, 0) + 1
            current[index] = length
            best = max(best, length)
        previous = current
    return best


class PhraseMatchBackend:
    """Score candidates by phrase adjacency plus content-term coverage."""

    status = "phrase"
    available = True

    def __init__(
        self,
        *,
        coverage_weight: float = 0.4,
        run_weight: float = 0.6,
        cache_size: int = 4096,
    ) -> None:
        if not 0.0 <= coverage_weight <= 1.0:
            raise ValueError("coverage_weight must be within [0, 1]")
        if not 0.0 <= run_weight <= 1.0:
            raise ValueError("run_weight must be within [0, 1]")
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.coverage_weight = coverage_weight
        self.run_weight = run_weight
        self._cache_size = cache_size
        self._tokens: OrderedDict[str, tuple[str, ...]] = OrderedDict()

    def scores(
        self,
        evidence: Evidence,
        candidates: Sequence[Candidate],
        *,
        context: RerankContext | None = None,
    ) -> list[float]:
        del context  # A lexical scorer wants the tokens, not the roles.
        if not evidence or not candidates:
            return [0.0] * len(candidates)
        total_weight = sum(weight for _, weight in evidence)
        if total_weight <= 0:
            return [0.0] * len(candidates)

        prepared = [
            (tokens, frozenset(tokens) - _STOPWORDS, weight)
            for tokens, weight in evidence
        ]
        results: list[float] = []
        for candidate in candidates:
            product_tokens = self._product_tokens(candidate)
            positions: dict[str, list[int]] = {}
            for index, token in enumerate(product_tokens):
                positions.setdefault(token, []).append(index)
            present = positions.keys()
            accumulated = 0.0
            for tokens, content, weight in prepared:
                coverage = (
                    len(content.intersection(present)) / len(content) if content else 0.0
                )
                run = min(
                    longest_contiguous_run(tokens, positions), _RUN_SATURATION
                ) / _RUN_SATURATION
                accumulated += weight * (
                    self.coverage_weight * coverage + self.run_weight * run
                )
            results.append(accumulated / total_weight)
        return results

    def _product_tokens(self, candidate: Candidate) -> tuple[str, ...]:
        identifier = candidate.parent_asin
        cached = self._tokens.get(identifier)
        if cached is not None:
            self._tokens.move_to_end(identifier)
            return cached
        tokens = token_sequence(
            " ".join(searchable_fields(candidate.product)), _MAX_PRODUCT_TOKENS
        )
        self._tokens[identifier] = tokens
        self._tokens.move_to_end(identifier)
        while len(self._tokens) > self._cache_size:
            self._tokens.popitem(last=False)
        return tokens


def compact_document(product: dict[str, object], limit: int) -> str:
    """Build the shortest product text that still identifies it.

    The full searchable text runs to about 1100 characters, most of it
    marketing description that costs sequence length without adding evidence.
    Title first, then the structured fields the shopper actually states
    requirements about.
    """
    parts: list[str] = []
    budget = limit
    for value in searchable_fields(product):
        text = value.strip()
        if not text:
            continue
        parts.append(text[:budget])
        budget -= len(parts[-1])
        if budget <= 0:
            break
    return " | ".join(parts)


def evidence_query(evidence: Evidence, limit: int) -> str:
    """Flatten weighted evidence into one query, strongest phrase first."""
    ordered = sorted(evidence, key=lambda item: -item[1])
    tokens: list[str] = []
    seen: set[str] = set()
    for phrase, _ in ordered:
        for token in phrase:
            if token in seen:
                continue
            seen.add(token)
            tokens.append(token)
            if len(tokens) >= limit:
                return " ".join(tokens)
    return " ".join(tokens)


class CrossEncoderBackend:
    """Score (query, product) pairs with a quantized ONNX cross-encoder.

    Unlike the phrase backend this judges semantic relevance, so it can rank a
    product that satisfies a stated requirement without repeating its wording.
    Any load or inference failure marks the backend unavailable, which returns
    the stage to the constraint ranker's order rather than failing the turn.
    """

    def __init__(
        self,
        session: object,
        tokenizer: object,
        *,
        max_length: int = 128,
        document_chars: int = 400,
        query_tokens: int = 40,
        failure_limit: int = 3,
    ) -> None:
        if max_length < 16:
            raise ValueError("max_length must be at least 16")
        if document_chars < 32:
            raise ValueError("document_chars must be at least 32")
        if query_tokens < 4:
            raise ValueError("query_tokens must be at least 4")
        self._session = session
        self._tokenizer = tokenizer
        self.max_length = max_length
        self.document_chars = document_chars
        self.query_tokens = query_tokens
        self._failure_limit = failure_limit
        self._failures = 0
        self._available = True
        self.status = "cross_encoder"

    @property
    def available(self) -> bool:
        return self._available

    def scores(
        self,
        evidence: Evidence,
        candidates: Sequence[Candidate],
        *,
        context: RerankContext | None = None,
    ) -> list[float]:
        del context  # The cross-encoder scores a flat query against text.
        if not self._available or not evidence or not candidates:
            return [0.0] * len(candidates)
        query = evidence_query(evidence, self.query_tokens)
        if not query:
            return [0.0] * len(candidates)
        try:
            import numpy as np

            pairs = [
                (query, compact_document(candidate.product, self.document_chars))
                for candidate in candidates
            ]
            encoded = self._tokenizer.encode_batch(pairs)
            feed = {
                "input_ids": np.array(
                    [item.ids for item in encoded], dtype=np.int64
                ),
                "attention_mask": np.array(
                    [item.attention_mask for item in encoded], dtype=np.int64
                ),
                "token_type_ids": np.array(
                    [item.type_ids for item in encoded], dtype=np.int64
                ),
            }
            logits = self._session.run(None, feed)[0]
            return [float(value) for value in np.asarray(logits).reshape(-1)]
        except Exception:  # noqa: BLE001 - degrade instead of failing the turn.
            self._failures += 1
            if self._failures >= self._failure_limit:
                self._available = False
                self.status = "inference_failed"
            return [0.0] * len(candidates)


_LLM_SYSTEM_PROMPT = (
    "You rank shopping candidates against a customer's stated requirements. "
    'Reply with JSON only, in the form {"ranking": [...]}. '
    "The list must contain EVERY candidate index exactly once, ordered most to "
    "least relevant. Never omit an index, even if it matches poorly."
)


class LlmRerankBackend:
    """Rank the window with a hosted model over an OpenAI-compatible endpoint.

    This is the one component that can leave the machine, so it is built to be
    removable: credentials come from the environment and are never read from
    the repository, the call uses only the standard library, and any failure -
    no credentials, a timeout, a refused request, an unparseable reply, a reply
    that is not a permutation of the window - marks the backend unavailable so
    the stage falls back to the lexical backend rather than degrading the turn.

    `submission_rules.md` warns that official scoring may run without network
    access, and forbids shipping credentials. The offline path is therefore the
    one that must always work; this is an upgrade over it, never a requirement.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 8.0,
        document_chars: int = 220,
        query_tokens: int = 48,
        failure_limit: int = 3,
        cache_size: int = 2048,
        prompt_style: str = "flat",
    ) -> None:
        if not base_url or not api_key or not model:
            raise ValueError("base_url, api_key and model are all required")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.document_chars = document_chars
        self.query_tokens = query_tokens
        self._failure_limit = failure_limit
        self._failures = 0
        self._available = True
        self.status = "llm"
        # Measured: grouping the ledger by role is better on ordinary Buying
        # turns (MRR 0.477 -> 0.522) and worse on Intent Override (HitRate
        # 0.9333 -> 0.9000). `adaptive` takes each where it wins; `flat` and
        # `structured` keep both single-style measurements reproducible.
        if prompt_style not in {"flat", "structured", "adaptive"}:
            raise ValueError(
                'prompt_style must be "flat", "structured" or "adaptive"'
            )
        self.prompt_style = prompt_style
        self.last_usage: dict[str, int] = {}
        self._cache: OrderedDict[tuple[str, tuple[str, ...]], list[float]] = (
            OrderedDict()
        )
        self._cache_size = cache_size

    @property
    def available(self) -> bool:
        return self._available

    def scores(
        self,
        evidence: Evidence,
        candidates: Sequence[Candidate],
        *,
        context: RerankContext | None = None,
    ) -> list[float]:
        self.last_usage = {}
        if not self._available or not candidates:
            return [0.0] * len(candidates)
        query = evidence_query(evidence, self.query_tokens)
        grouped = context or RerankContext()
        if self.prompt_style == "structured":
            use_structure = True
        elif self.prompt_style == "adaptive":
            # A replacement requirement arrives as free text with almost no
            # ledger behind it yet, and the grouped form measured worse there.
            use_structure = not grouped.is_override
        else:
            use_structure = False
        requirements = (
            render_requirements(grouped, query)
            if use_structure
            else f"Requirements: {query}"
        )
        if not requirements.strip():
            return [0.0] * len(candidates)

        key = (requirements, tuple(item.parent_asin for item in candidates))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return list(cached)

        try:
            ranking, usage = self._request(requirements, candidates)
        except Exception:  # noqa: BLE001 - the offline backend is the guarantee.
            self._failures += 1
            if self._failures >= self._failure_limit:
                self._available = False
                self.status = "llm_unavailable"
            return [0.0] * len(candidates)

        self.last_usage = usage
        # Highest score first, matching the order the model returned.
        count = len(candidates)
        scores = [0.0] * count
        for position, index in enumerate(ranking):
            scores[index] = float(count - position)
        self._cache[key] = list(scores)
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return scores

    def _request(
        self, requirements: str, candidates: Sequence[Candidate]
    ) -> tuple[list[int], dict[str, int]]:
        import json
        import urllib.request

        listing = "\n".join(
            f"{index}. {compact_document(item.product, self.document_chars)}"
            for index, item in enumerate(candidates)
        )
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{requirements}\n\n"
                            f"Candidates ({len(candidates)} total):\n{listing}"
                        ),
                    },
                ],
                "response_format": {"type": "json_object"},
                # Zero temperature keeps a replayed session on the same path.
                "temperature": 0,
                "max_tokens": 16 * len(candidates) + 64,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read())
        content = payload["choices"][0]["message"]["content"]
        ranking = json.loads(content)["ranking"]
        if sorted(ranking) != list(range(len(candidates))):
            # A partial or duplicated ranking would silently drop candidates.
            raise ValueError("model did not return a permutation of the window")
        reported = payload.get("usage") or {}
        usage = {
            field: int(reported[field])
            for field in ("prompt_tokens", "completion_tokens")
            if isinstance(reported.get(field), int)
        }
        return [int(index) for index in ranking], usage


def load_llm_backend(*, prompt_style: str = "flat") -> RerankBackend:
    """Build the hosted backend from the environment, or report why not."""
    base_url = os.environ.get("COMPASSCART_LLM_BASE_URL", "").strip()
    api_key = os.environ.get("COMPASSCART_LLM_API_KEY", "").strip()
    model = os.environ.get("COMPASSCART_LLM_MODEL", "").strip()
    if not base_url or not api_key or not model:
        return NullRerankBackend("llm_credentials_missing")
    try:
        return LlmRerankBackend(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt_style=prompt_style,
        )
    except Exception:  # noqa: BLE001 - optional by construction.
        return NullRerankBackend("llm_construction_failed")


def session_evidence(state: SessionState) -> Evidence:
    """Collect the phrases the shopper has actually stated.

    Raw messages are kept alongside parsed constraints because a requirement is
    often disclosed as free text that no structured attribute captures.
    """
    collected: list[tuple[tuple[str, ...], float]] = []
    seen: set[tuple[str, ...]] = set()

    def add(text: object, kind: str) -> None:
        tokens = token_sequence(text, _MAX_EVIDENCE_TOKENS)
        if not tokens or tokens in seen:
            return
        seen.add(tokens)
        collected.append((tokens, _EVIDENCE_WEIGHTS[kind]))

    for constraint in state.active_constraints():
        if constraint.source == "profile":
            kind = "profile"
        elif constraint.is_hard:
            kind = "hard"
        else:
            kind = "soft"
        for value in constraint.values():
            add(value, kind)
        if constraint.upper_value:
            add(constraint.upper_value, kind)
    for message in state.query_history:
        add(message, "message")
    return tuple(collected)


def load_cross_encoder_backend(
    asset_dir: Path, *, max_length: int = 128
) -> RerankBackend:
    """Load the ONNX cross-encoder, or report why it is unavailable."""
    model_path = asset_dir / "model.int8.onnx"
    tokenizer_path = asset_dir / "tokenizer.json"
    if not model_path.is_file() or not tokenizer_path.is_file():
        return NullRerankBackend("assets_missing")
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=max_length)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        return CrossEncoderBackend(session, tokenizer, max_length=max_length)
    except Exception:  # noqa: BLE001 - an optional accelerator, never required.
        return NullRerankBackend("load_failed")


def load_rerank_backend(
    *,
    enabled: bool = True,
    backend: str = "phrase",
    asset_dir: Path | None = None,
    max_length: int = 128,
    prompt_style: str = "flat",
) -> RerankBackend:
    """Select the rerank backend, honoring the ablation environment switch."""
    if os.environ.get("COMPASSCART_DISABLE_RERANK") == "1":
        return NullRerankBackend("disabled_by_environment")
    if not enabled:
        return NullRerankBackend("disabled_by_config")
    try:
        if backend == "llm":
            loaded = load_llm_backend(prompt_style=prompt_style)
            # No credentials in the scoring environment is the expected case,
            # not an error; the lexical backend is what must always work.
            return loaded if loaded.available else PhraseMatchBackend()
        if backend == "cross_encoder":
            if asset_dir is None:
                return NullRerankBackend("assets_missing")
            loaded = load_cross_encoder_backend(asset_dir, max_length=max_length)
            # The lexical backend is the documented offline guarantee, so a
            # missing or broken model asset degrades to it rather than to no
            # reranking at all.
            return loaded if loaded.available else PhraseMatchBackend()
        return PhraseMatchBackend()
    except Exception:  # noqa: BLE001 - the stage is optional by construction.
        return NullRerankBackend("construction_failed")


class RerankStage:
    """Reorder the head of the ranked list without changing its membership."""

    def __init__(
        self,
        backend: RerankBackend | None = None,
        *,
        window: int = 50,
        weight: float = 0.6,
        buying_weight: float | None = None,
        buying_backend: RerankBackend | None = None,
        buying_window: int | None = None,
        buying_requires_override: bool = False,
    ) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        if buying_window is not None and buying_window < 2:
            raise ValueError("buying_window must be at least 2")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be within [0, 1]")
        if buying_weight is not None and not 0.0 <= buying_weight <= 1.0:
            raise ValueError("buying_weight must be within [0, 1]")
        self.backend = backend or NullRerankBackend()
        self.window = window
        self.weight = weight
        # A Buying turn already carries explicit hard constraints, so the
        # constraint ranker is well informed and reordering it mostly adds
        # noise; a Browsing turn has vague evidence and gains the most.  The
        # measured split on the public set was `+0.038` Browsing HitRate
        # against `-0.025` Buying, which is what this separate weight exists
        # to stop paying for.
        self.buying_weight = weight if buying_weight is None else buying_weight
        # Buying and Browsing can use different rerankers: phrase adjacency
        # suits verbatim quotation, and a model suits judging whether a hard
        # requirement is actually satisfied.
        self.buying_backend = buying_backend or self.backend
        # The window is per route as well, because the two backends have
        # different budgets: a lexical scorer is happy with fifty candidates,
        # while a hosted model pays prompt tokens for every one of them.
        self.buying_window = window if buying_window is None else buying_window
        # Measured: the model's gain concentrates on Intent Override
        # (+0.0060 weighted) while ordinary Buying turns are flat (-0.0006),
        # so spending a call on every Buying turn buys mostly nothing.
        self.buying_requires_override = buying_requires_override

    def backend_for(self, state: SessionState) -> RerankBackend:
        return self.buying_backend if state.route == "buying" else self.backend

    def window_for(self, state: SessionState) -> int:
        return self.buying_window if state.route == "buying" else self.window

    def runs_for(self, state: SessionState) -> bool:
        """Whether this turn is worth spending the Buying backend on."""
        if state.route != "buying" or not self.buying_requires_override:
            return True
        # An override persists for the rest of the session through the intent
        # version; `override_scope` only marks the turn it arrived on.
        return state.intent_version > 1 or state.override_scope != "none"

    @property
    def available(self) -> bool:
        browsing_ready = (
            bool(getattr(self.backend, "available", False)) and self.weight > 0.0
        )
        buying_ready = (
            bool(getattr(self.buying_backend, "available", False))
            and self.buying_weight > 0.0
        )
        return browsing_ready or buying_ready

    def weight_for(self, state: SessionState) -> float:
        return self.buying_weight if state.route == "buying" else self.weight

    def apply(
        self,
        candidates: list[Candidate],
        state: SessionState,
        *,
        deadline: float | None = None,
        diagnostics: list[str] | None = None,
        usage: dict[str, int] | None = None,
    ) -> list[Candidate]:
        if not self.available or len(candidates) < 2:
            return candidates
        weight = self.weight_for(state)
        backend = self.backend_for(state)
        if weight <= 0.0 or not getattr(backend, "available", False):
            return candidates
        if not self.runs_for(state):
            return candidates
        if deadline is not None and time.perf_counter() >= deadline:
            if diagnostics is not None and "rerank_budget" not in diagnostics:
                diagnostics.append("rerank_budget")
            return candidates
        evidence = session_evidence(state)
        if not evidence:
            return candidates

        window = self.window_for(state)
        head = candidates[:window]
        tail = candidates[window:]
        scores = backend.scores(evidence, head, context=session_context(state))
        if usage is not None:
            # A backend that calls a model reports what the turn cost; the
            # offline backends report nothing and leave the totals at zero.
            for field, value in getattr(backend, "last_usage", {}).items():
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    usage[field] = usage.get(field, 0) + value
        if len(scores) != len(head):
            raise ValueError("rerank backend returned a mismatched score count")

        lowest = min(scores)
        span = max(scores) - lowest
        if span <= 0:
            # No candidate in the window is distinguishable, so the stage has
            # no signal to contribute.  Returning early keeps the ranker's
            # order instead of collapsing the blend onto the identifier
            # tie-break, which at weight 1.0 would be an alphabetical shuffle.
            return candidates
        divisor = len(head) - 1 if len(head) > 1 else 1
        blended: list[tuple[bool, float, str, Candidate]] = []
        for position, (candidate, score) in enumerate(zip(head, scores)):
            # The base term preserves whatever order the ranker produced,
            # including its browsing diversity pass, without depending on the
            # scale of its raw score.
            base = 1.0 - position / divisor
            final = (1.0 - weight) * base + weight * ((score - lowest) / span)
            blended.append((candidate.relaxed, -final, candidate.parent_asin, candidate))
        blended.sort(key=lambda item: item[:3])
        return [item[3] for item in blended] + tail
