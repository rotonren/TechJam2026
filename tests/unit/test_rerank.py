import json
import time

import pytest

from compasscart.models import Candidate, Constraint, SessionState
from compasscart.normalization import terms
from compasscart.rerank import (
    CrossEncoderBackend,
    LlmRerankBackend,
    NullRerankBackend,
    PhraseMatchBackend,
    RerankStage,
    compact_document,
    evidence_query,
    load_cross_encoder_backend,
    load_llm_backend,
    load_rerank_backend,
    longest_contiguous_run,
    session_evidence,
    token_sequence,
)


def _product(text: str) -> dict[str, object]:
    return {"title": text, "features": [], "details": {}, "description": ""}


def _candidate(identifier: str, text: str, *, relaxed: bool = False) -> Candidate:
    return Candidate(
        parent_asin=identifier, product=_product(text), score=1.0, relaxed=relaxed
    )


def _state(*constraints: Constraint, messages: tuple[str, ...] = ()) -> SessionState:
    return SessionState(
        "s1", constraints=list(constraints), query_history=list(messages)
    )


def test_token_sequence_keeps_order_and_repetition():
    text = "water resistant water"

    assert token_sequence(text) == ("water", "resistant", "water")
    # The shared tokenizer deduplicates, which is why this stage needs its own.
    assert terms(text) == ["water", "resistant"]


def test_token_sequence_honors_the_limit():
    assert token_sequence("a b c d", 2) == ("a", "b")


@pytest.mark.parametrize(
    ("evidence", "document", "expected"),
    (
        (("water", "resistant"), ("a", "water", "resistant", "b"), 2),
        (("water", "resistant"), ("water", "b", "resistant"), 1),
        (("a", "b", "c"), ("x", "a", "b", "c", "y"), 3),
        (("missing",), ("a", "b"), 0),
        ((), ("a", "b"), 0),
    ),
)
def test_longest_contiguous_run(evidence, document, expected):
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(document):
        positions.setdefault(token, []).append(index)

    assert longest_contiguous_run(evidence, positions) == expected


def test_phrase_adjacency_outscores_scattered_terms():
    backend = PhraseMatchBackend()
    evidence = ((("water", "resistant", "leather", "boot"), 1.0),)
    adjacent = _candidate("A", "water resistant leather boot")
    scattered = _candidate("B", "leather bag water bottle resistant band boot rack")

    adjacent_score, scattered_score = backend.scores(evidence, [adjacent, scattered])

    assert adjacent_score > scattered_score


def test_backend_returns_zero_without_evidence_or_candidates():
    backend = PhraseMatchBackend()

    assert backend.scores((), [_candidate("A", "anything")]) == [0.0]
    assert backend.scores(((("a",), 1.0),), []) == []


def test_session_evidence_weights_hard_above_soft_above_profile():
    state = _state(
        Constraint("material", "leather", 1.0, True, "message", 1, 1),
        Constraint("color", "navy", 0.6, False, "message", 1, 1),
        Constraint("feature", "comfortable", 0.5, False, "profile", 0, 1),
    )

    weights = {tokens[0]: weight for tokens, weight in session_evidence(state)}

    assert weights["leather"] > weights["navy"] > weights["comfortable"]


def test_session_evidence_includes_raw_messages_and_deduplicates():
    state = _state(
        Constraint("material", "leather", 1.0, True, "message", 1, 1),
        messages=("leather", "water resistant sole"),
    )

    collected = [tokens for tokens, _ in session_evidence(state)]

    assert ("water", "resistant", "sole") in collected
    assert collected.count(("leather",)) == 1


def test_stage_reorders_the_head_without_changing_membership():
    state = _state(Constraint("feature", "water resistant", 1.0, True, "message", 1, 1))
    candidates = [
        _candidate("A", "cotton tote bag"),
        _candidate("B", "wool scarf"),
        _candidate("C", "water resistant hiking boot"),
    ]
    stage = RerankStage(PhraseMatchBackend(), window=50, weight=1.0)

    reranked = stage.apply(candidates, state)

    assert next(item.parent_asin for item in reranked) == "C"
    assert {item.parent_asin for item in reranked} == {"A", "B", "C"}


def test_stage_keeps_exact_candidates_ahead_of_relaxed_ones():
    state = _state(Constraint("feature", "water resistant", 1.0, True, "message", 1, 1))
    candidates = [
        _candidate("A", "cotton tote bag"),
        _candidate("B", "water resistant hiking boot", relaxed=True),
    ]
    stage = RerankStage(PhraseMatchBackend(), window=50, weight=1.0)

    reranked = stage.apply(candidates, state)

    # B scores far higher but is a disclosed relaxation, so it must stay last.
    assert [item.parent_asin for item in reranked] == ["A", "B"]


def test_stage_leaves_candidates_beyond_the_window_untouched():
    state = _state(Constraint("feature", "water resistant", 1.0, True, "message", 1, 1))
    candidates = [
        _candidate("A", "cotton tote bag"),
        _candidate("B", "wool scarf"),
        _candidate("C", "water resistant hiking boot"),
    ]
    stage = RerankStage(PhraseMatchBackend(), window=2, weight=1.0)

    reranked = stage.apply(candidates, state)

    assert reranked[2].parent_asin == "C"


def test_zero_weight_and_null_backend_are_identity():
    state = _state(Constraint("feature", "water resistant", 1.0, True, "message", 1, 1))
    candidates = [_candidate("A", "cotton tote"), _candidate("B", "water resistant")]

    zero = RerankStage(PhraseMatchBackend(), weight=0.0)
    null = RerankStage(NullRerankBackend(), weight=1.0)

    assert zero.apply(candidates, state) is candidates
    assert null.apply(candidates, state) is candidates
    assert zero.available is False
    assert null.available is False


def test_expired_deadline_skips_rerank_and_records_a_diagnostic():
    state = _state(Constraint("feature", "water resistant", 1.0, True, "message", 1, 1))
    candidates = [_candidate("A", "cotton tote"), _candidate("B", "water resistant")]
    stage = RerankStage(PhraseMatchBackend(), weight=1.0)
    diagnostics: list[str] = []

    result = stage.apply(
        candidates, state, deadline=time.perf_counter() - 1.0, diagnostics=diagnostics
    )

    assert result is candidates
    assert diagnostics == ["rerank_budget"]


def test_empty_evidence_leaves_the_order_unchanged():
    stage = RerankStage(PhraseMatchBackend(), weight=1.0)
    candidates = [_candidate("A", "cotton tote"), _candidate("B", "water resistant")]

    assert stage.apply(candidates, _state()) is candidates


def test_indistinguishable_candidates_keep_the_ranker_order():
    # Every candidate scores identically, so a naive blend would fall through
    # to the identifier tie-break and shuffle the list alphabetically.
    state = _state(Constraint("material", "leather", 1.0, True, "message", 1, 1))
    candidates = [
        _candidate("Z", "cotton tote"),
        _candidate("A", "cotton tote"),
    ]
    stage = RerankStage(PhraseMatchBackend(), weight=1.0)

    result = stage.apply(candidates, state)

    assert [item.parent_asin for item in result] == ["Z", "A"]


def test_buying_route_uses_its_own_weight():
    stage = RerankStage(PhraseMatchBackend(), weight=1.0, buying_weight=0.0)
    browsing = _state(Constraint("feature", "water resistant", 1.0, True, "message", 1, 1))
    buying = _state(Constraint("feature", "water resistant", 1.0, True, "message", 1, 1))
    buying.route = "buying"
    candidates = [
        _candidate("A", "cotton tote bag"),
        _candidate("B", "water resistant hiking boot"),
    ]

    assert stage.weight_for(browsing) == 1.0
    assert stage.weight_for(buying) == 0.0
    # Browsing reorders; Buying is left exactly as the constraint ranker had it.
    assert next(item.parent_asin for item in stage.apply(candidates, browsing)) == "B"
    assert stage.apply(candidates, buying) is candidates


def test_buying_weight_defaults_to_the_shared_weight():
    stage = RerankStage(PhraseMatchBackend(), weight=0.6)

    assert stage.buying_weight == 0.6


def test_stage_is_available_when_only_one_route_is_weighted():
    stage = RerankStage(PhraseMatchBackend(), weight=0.8, buying_weight=0.0)

    assert stage.available is True


def test_stage_rejects_an_out_of_range_buying_weight():
    with pytest.raises(ValueError):
        RerankStage(PhraseMatchBackend(), weight=0.8, buying_weight=1.5)


def test_mismatched_backend_score_count_is_rejected():
    class BrokenBackend:
        available = True
        status = "broken"

        def scores(self, evidence, candidates):
            del evidence, candidates
            return [1.0]

    state = _state(Constraint("feature", "water", 1.0, True, "message", 1, 1))
    stage = RerankStage(BrokenBackend(), weight=1.0)

    with pytest.raises(ValueError):
        stage.apply([_candidate("A", "a"), _candidate("B", "b")], state)


def test_compact_document_stays_within_its_budget():
    product = {
        "title": "T" * 500,
        "features": ["F" * 500],
        "details": {},
        "description": "D" * 500,
    }

    text = compact_document(product, 120)

    assert len(text) <= 120 + len(" | ")
    assert text.startswith("T")


def test_evidence_query_orders_by_weight_and_deduplicates():
    evidence = (
        (("navy", "cotton"), 0.6),
        (("leather", "cotton"), 1.0),
    )

    assert evidence_query(evidence, 10) == "leather cotton navy"
    assert evidence_query(evidence, 2) == "leather cotton"


def test_cross_encoder_scores_are_returned_per_candidate():
    class FakeTokenizer:
        def encode_batch(self, pairs):
            return [
                type("E", (), {"ids": [1], "attention_mask": [1], "type_ids": [0]})()
                for _ in pairs
            ]

    class FakeSession:
        def run(self, _outputs, feed):
            import numpy as np

            return [np.array([[2.0], [1.0]], dtype="float32")]

    backend = CrossEncoderBackend(FakeSession(), FakeTokenizer())
    evidence = ((("leather",), 1.0),)

    scores = backend.scores(evidence, [_candidate("A", "a"), _candidate("B", "b")])

    assert scores == [2.0, 1.0]
    assert backend.available is True


def test_cross_encoder_marks_itself_unavailable_after_repeated_failures():
    class BrokenSession:
        def run(self, _outputs, feed):
            raise RuntimeError("boom")

    class FakeTokenizer:
        def encode_batch(self, pairs):
            return [
                type("E", (), {"ids": [1], "attention_mask": [1], "type_ids": [0]})()
                for _ in pairs
            ]

    backend = CrossEncoderBackend(
        BrokenSession(), FakeTokenizer(), failure_limit=2
    )
    evidence = ((("leather",), 1.0),)
    candidates = [_candidate("A", "a")]

    assert backend.scores(evidence, candidates) == [0.0]
    assert backend.available is True
    assert backend.scores(evidence, candidates) == [0.0]
    assert backend.available is False
    assert backend.status == "inference_failed"


@pytest.mark.parametrize(
    "kwargs",
    ({"max_length": 8}, {"document_chars": 4}, {"query_tokens": 1}),
)
def test_cross_encoder_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CrossEncoderBackend(object(), object(), **kwargs)


def test_missing_cross_encoder_assets_report_their_reason(tmp_path):
    backend = load_cross_encoder_backend(tmp_path)

    assert backend.available is False
    assert backend.status == "assets_missing"


def test_cross_encoder_request_degrades_to_the_lexical_backend(tmp_path, monkeypatch):
    monkeypatch.delenv("COMPASSCART_DISABLE_RERANK", raising=False)

    backend = load_rerank_backend(backend="cross_encoder", asset_dir=tmp_path)

    # The offline lexical path is the guarantee; a missing model must not turn
    # reranking off entirely.
    assert isinstance(backend, PhraseMatchBackend)
    assert backend.available is True


def test_environment_switch_disables_the_backend(monkeypatch):
    monkeypatch.setenv("COMPASSCART_DISABLE_RERANK", "1")

    backend = load_rerank_backend(enabled=True)

    assert backend.available is False
    assert backend.status == "disabled_by_environment"


def test_disabled_by_config_reports_its_reason(monkeypatch):
    monkeypatch.delenv("COMPASSCART_DISABLE_RERANK", raising=False)

    backend = load_rerank_backend(enabled=False)

    assert backend.available is False
    assert backend.status == "disabled_by_config"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"window": 1}, ValueError),
        ({"weight": -0.1}, ValueError),
        ({"weight": 1.1}, ValueError),
    ),
)
def test_stage_rejects_invalid_configuration(kwargs, error):
    with pytest.raises(error):
        RerankStage(PhraseMatchBackend(), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    ({"coverage_weight": -0.1}, {"run_weight": 1.5}, {"cache_size": 0}),
)
def test_backend_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PhraseMatchBackend(**kwargs)


def test_product_token_cache_evicts_and_still_scores():
    backend = PhraseMatchBackend(cache_size=1)
    evidence = ((("leather",), 1.0),)
    first = _candidate("A", "leather boot")
    second = _candidate("B", "cotton tote")

    backend.scores(evidence, [first, second])
    repeated = backend.scores(evidence, [first, second])

    assert repeated[0] > repeated[1]


class _StubResponse:
    """Minimal stand-in for the object urlopen returns as a context manager."""

    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _llm_payload(ranking, *, prompt=100, completion=20):
    return {
        "choices": [{"message": {"content": json.dumps({"ranking": ranking})}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def _llm_backend(**kwargs):
    return LlmRerankBackend(
        base_url="https://example.invalid", api_key="k", model="m", **kwargs
    )


def _patch_urlopen(monkeypatch, handler):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", handler)


def test_llm_backend_turns_a_ranking_into_descending_scores(monkeypatch):
    _patch_urlopen(monkeypatch, lambda *a, **k: _StubResponse(_llm_payload([2, 0, 1])))
    backend = _llm_backend()
    candidates = [_candidate(c, c.lower()) for c in ("A", "B", "C")]

    scores = backend.scores(((("leather",), 1.0),), candidates)

    assert scores == [2.0, 1.0, 3.0]
    assert backend.last_usage == {"prompt_tokens": 100, "completion_tokens": 20}


def test_llm_backend_caches_an_identical_window(monkeypatch):
    calls = []

    def handler(*args, **kwargs):
        calls.append(1)
        return _StubResponse(_llm_payload([1, 0]))

    _patch_urlopen(monkeypatch, handler)
    backend = _llm_backend()
    evidence = ((("leather",), 1.0),)
    candidates = [_candidate("A", "a"), _candidate("B", "b")]

    first = backend.scores(evidence, candidates)
    second = backend.scores(evidence, candidates)

    assert first == second
    assert len(calls) == 1


@pytest.mark.parametrize(
    "payload",
    (
        {"choices": [{"message": {"content": "not json"}}]},
        {"choices": [{"message": {"content": '{"ranking": [0]}'}}]},
        {"choices": [{"message": {"content": '{"ranking": [0, 0]}'}}]},
        {"choices": [{"message": {"content": '{"ranking": [0, 5]}'}}]},
    ),
)
def test_llm_backend_rejects_a_reply_that_is_not_a_permutation(monkeypatch, payload):
    # A partial or duplicated ranking would silently drop candidates.
    _patch_urlopen(monkeypatch, lambda *a, **k: _StubResponse(payload))
    backend = _llm_backend()

    scores = backend.scores(
        ((("leather",), 1.0),), [_candidate("A", "a"), _candidate("B", "b")]
    )

    assert scores == [0.0, 0.0]


def test_llm_backend_gives_up_after_repeated_failures(monkeypatch):
    def handler(*args, **kwargs):
        raise TimeoutError("no route to host")

    _patch_urlopen(monkeypatch, handler)
    backend = _llm_backend(failure_limit=2)
    evidence = ((("leather",), 1.0),)
    candidates = [_candidate("A", "a"), _candidate("B", "b")]

    backend.scores(evidence, candidates)
    assert backend.available is True
    backend.scores(evidence, candidates)
    assert backend.available is False
    assert backend.status == "llm_unavailable"


@pytest.mark.parametrize(
    "kwargs",
    (
        {"base_url": "", "api_key": "k", "model": "m"},
        {"base_url": "u", "api_key": "", "model": "m"},
        {"base_url": "u", "api_key": "k", "model": ""},
        {"base_url": "u", "api_key": "k", "model": "m", "timeout_s": 0},
    ),
)
def test_llm_backend_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        LlmRerankBackend(**kwargs)


def test_missing_credentials_report_their_reason(monkeypatch):
    for name in (
        "COMPASSCART_LLM_BASE_URL",
        "COMPASSCART_LLM_API_KEY",
        "COMPASSCART_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    backend = load_llm_backend()

    assert backend.available is False
    assert backend.status == "llm_credentials_missing"


def test_llm_request_degrades_to_the_lexical_backend_without_credentials(monkeypatch):
    monkeypatch.delenv("COMPASSCART_DISABLE_RERANK", raising=False)
    monkeypatch.delenv("COMPASSCART_LLM_API_KEY", raising=False)

    backend = load_rerank_backend(backend="llm")

    # Official scoring may run with no network and no credentials; that is the
    # expected case, not a failure, so reranking must still happen.
    assert isinstance(backend, PhraseMatchBackend)
    assert backend.available is True


def test_stage_uses_a_different_backend_per_route():
    browsing = PhraseMatchBackend()
    buying = PhraseMatchBackend()
    stage = RerankStage(browsing, buying_backend=buying, weight=0.8, buying_weight=0.8)
    state = _state()

    assert stage.backend_for(state) is browsing
    state.route = "buying"
    assert stage.backend_for(state) is buying


def test_stage_is_available_when_only_the_buying_backend_is():
    stage = RerankStage(
        NullRerankBackend(),
        buying_backend=PhraseMatchBackend(),
        weight=0.8,
        buying_weight=0.8,
    )

    assert stage.available is True
