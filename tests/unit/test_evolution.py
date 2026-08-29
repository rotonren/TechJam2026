import pytest

from compasscart.evolution import (
    DEFAULT_RESPONSE_LIKELIHOOD,
    PolicyMemory,
    YieldStat,
    profile_segment,
)


def test_yield_stat_accumulates_and_reports_its_rate():
    stat = YieldStat().observe(True).observe(False).observe(True)

    assert (stat.asked, stat.disclosed) == (3, 2)
    assert stat.rate == pytest.approx(2 / 3)
    assert YieldStat().rate == 0.0


def test_cold_memory_returns_the_hand_written_prior():
    memory = PolicyMemory()

    for attribute, prior in DEFAULT_RESPONSE_LIKELIHOOD.items():
        assert memory.likelihood(attribute) == prior


def test_unknown_attribute_falls_back_to_a_neutral_prior():
    assert PolicyMemory().likelihood("not-an-attribute") == 0.70


def test_evidence_moves_the_estimate_toward_what_was_observed():
    memory = PolicyMemory(prior_strength=4.0)
    prior = DEFAULT_RESPONSE_LIKELIHOOD["feature"]

    # The hand-written prior underrates this attribute; every observation says so.
    for _ in range(20):
        memory.observe("feature", True)

    posterior = memory.likelihood("feature")
    assert posterior > prior
    assert posterior < 1.0


def test_evidence_can_also_lower_an_overrated_prior():
    memory = PolicyMemory(prior_strength=4.0)
    prior = DEFAULT_RESPONSE_LIKELIHOOD["material"]

    for _ in range(20):
        memory.observe("material", False)

    assert memory.likelihood("material") < prior


def test_a_single_observation_barely_moves_a_confident_prior():
    memory = PolicyMemory(prior_strength=8.0)
    prior = DEFAULT_RESPONSE_LIKELIHOOD["size"]

    memory.observe("size", False)

    assert abs(memory.likelihood("size") - prior) < 0.1


def test_disabled_memory_never_leaves_the_prior():
    memory = PolicyMemory(enabled=False)

    for _ in range(50):
        memory.observe("feature", True)

    assert memory.likelihood("feature") == DEFAULT_RESPONSE_LIKELIHOOD["feature"]
    assert memory.snapshot()["observations"] == 0


def test_environment_switch_disables_learning(monkeypatch):
    monkeypatch.setenv("COMPASSCART_DISABLE_EVOLUTION", "1")

    memory = PolicyMemory()
    memory.observe("feature", True)

    assert memory.enabled is False
    assert memory.likelihood("feature") == DEFAULT_RESPONSE_LIKELIHOOD["feature"]


def test_segment_is_ignored_until_it_has_enough_evidence():
    memory = PolicyMemory(prior_strength=4.0, segment_floor=10)
    for _ in range(4):
        memory.observe("size", True, segment="fit")
    thin = memory.likelihood("size", segment="fit")

    assert thin == memory.likelihood("size")

    for _ in range(10):
        memory.observe("size", True, segment="fit")

    assert memory.likelihood("size", segment="fit") > thin


def test_route_context_refines_the_pooled_estimate():
    memory = PolicyMemory(prior_strength=4.0, segment_floor=10)
    # The same question behaves differently depending on where it is asked.
    for _ in range(20):
        memory.observe("size", True, context="browsing")
        memory.observe("size", False, context="buying")

    browsing = memory.likelihood("size", context="browsing")
    buying = memory.likelihood("size", context="buying")
    pooled = memory.likelihood("size")

    assert browsing > pooled > buying


def test_thin_route_evidence_does_not_override_the_pool():
    memory = PolicyMemory(prior_strength=4.0, segment_floor=10)
    for _ in range(20):
        memory.observe("size", True)
    memory.observe("size", False, context="buying")

    assert memory.likelihood("size", context="buying") == memory.likelihood("size")


def test_route_and_segment_refine_in_order():
    memory = PolicyMemory(prior_strength=4.0, segment_floor=4)
    for _ in range(10):
        memory.observe("size", False, context="buying", segment="fit")
    for _ in range(10):
        memory.observe("size", True, context="buying", segment="warmth")

    # Both segments sit inside the same route, so the route figure is their
    # average while each segment pulls away from it in its own direction.
    route = memory.likelihood("size", context="buying")
    assert memory.likelihood("size", context="buying", segment="fit") < route
    assert memory.likelihood("size", context="buying", segment="warmth") > route


def test_snapshot_reports_route_contexts():
    memory = PolicyMemory()
    memory.observe("size", True, context="browsing")

    contexts = memory.snapshot()["contexts"]

    assert contexts["browsing"]["size"] == {
        "asked": 1,
        "disclosed": 1,
        "observed_rate": 1.0,
    }


def test_segment_table_is_bounded():
    memory = PolicyMemory(max_segments=2)
    for index in range(5):
        memory.observe("size", True, segment=f"segment-{index}")

    assert memory.snapshot()["segment_count"] == 2
    # The global estimate still saw every observation.
    assert memory.snapshot()["attributes"]["size"]["asked"] == 5


def test_snapshot_reports_prior_and_posterior_without_session_data():
    memory = PolicyMemory(prior_strength=4.0)
    for _ in range(10):
        memory.observe("feature", True, segment="fit")

    snapshot = memory.snapshot()

    assert snapshot["enabled"] is True
    assert snapshot["observations"] == 10
    entry = snapshot["attributes"]["feature"]
    assert entry["asked"] == 10
    assert entry["disclosed"] == 10
    assert entry["prior"] == DEFAULT_RESPONSE_LIKELIHOOD["feature"]
    assert entry["posterior"] > entry["prior"]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"prior_strength": 0}, ValueError),
        ({"segment_floor": 0}, ValueError),
        ({"max_segments": 0}, ValueError),
    ),
)
def test_memory_rejects_invalid_configuration(kwargs, error):
    with pytest.raises(error):
        PolicyMemory(**kwargs)


@pytest.mark.parametrize(
    ("profile", "expected"),
    (
        ({"preference_tags": ["fit", "comfort"]}, "comfort|fit"),
        ({"preference_tags": ["Comfort", " fit "]}, "comfort|fit"),
        ({"preference_tags": []}, ""),
        ({"preference_tags": "fit"}, ""),
        ({}, ""),
        (None, ""),
    ),
)
def test_profile_segment_is_stable_and_tolerant(profile, expected):
    assert profile_segment(profile) == expected


def test_profile_segment_is_capped():
    profile = {"preference_tags": ["a", "b", "c", "d", "e", "f"]}

    assert profile_segment(profile).count("|") == 3
