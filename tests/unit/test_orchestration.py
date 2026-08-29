import pytest

from compasscart.models import SessionState
from compasscart.orchestration import StrategyDecision, StrategySelector


def _state(**fields) -> SessionState:
    state = SessionState("s1")
    for name, value in fields.items():
        setattr(state, name, value)
    return state


def test_default_strategy_is_the_existing_pipeline():
    decision = StrategySelector().select(
        _state(candidate_count=500), structured_question="material"
    )

    assert decision == StrategyDecision()
    assert decision.name == "probe"


def test_no_structured_question_becomes_an_open_question():
    state = _state(candidate_count=200)

    decision = StrategySelector().select(state, structured_question=None)

    assert decision.name == "open_probe"
    assert decision.open_question is True


def test_a_structured_question_is_left_alone():
    state = _state(candidate_count=200)

    decision = StrategySelector().select(state, structured_question="material")

    assert decision.open_question is False


@pytest.mark.parametrize(
    "fields",
    (
        {"asked_attributes": ["other"]},
        {"no_preference_attributes": {"other"}},
    ),
)
def test_an_open_question_is_never_repeated(fields):
    state = _state(candidate_count=200, **fields)

    decision = StrategySelector().select(state, structured_question=None)

    assert decision.open_question is False


def test_a_small_candidate_pool_stops_asking():
    state = _state(candidate_count=6)

    decision = StrategySelector(exploit_candidates=10).select(
        state, structured_question=None
    )

    assert decision.name == "exploit"
    assert decision.open_question is False


def test_disabled_selector_always_returns_the_default():
    state = _state(stall_count=9, candidate_count=1)

    assert StrategySelector(enabled=False).select(
        state, structured_question=None
    ) == StrategyDecision()


def test_selection_is_deterministic():
    state = _state(candidate_count=200)
    selector = StrategySelector()

    first = selector.select(state, structured_question=None)
    second = selector.select(state, structured_question=None)

    assert first == second


def test_selector_rejects_an_invalid_exploit_threshold():
    with pytest.raises(ValueError):
        StrategySelector(exploit_candidates=0)
