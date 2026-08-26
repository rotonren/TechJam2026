from compasscart.constraints import (
    display_constraint,
    hard_constraint_violations,
    matches_constraint,
)
from compasscart.models import Constraint


def _constraint(
    attribute: str,
    value: str,
    *,
    operator: str = "eq",
    upper_value: str | None = None,
    alternatives: tuple[str, ...] = (),
    is_hard: bool = True,
    confidence: float = 1.0,
    source: str = "message",
) -> Constraint:
    return Constraint(
        attribute,
        value,
        confidence,
        is_hard,
        source,
        1,
        1,
        operator=operator,
        upper_value=upper_value,
        alternatives=alternatives,
    )


def test_eq_normalizes_attribute_values_before_matching():
    constraint = _constraint("color", "Light Blue")

    assert matches_constraint({}, {"color": (" light   blue ",)}, constraint)


def test_category_matching_normalizes_singular_and_plural_forms():
    constraint = _constraint("category", "belt")

    assert matches_constraint({}, {"category": ("belts",)}, constraint)


def test_category_matching_merges_available_terms_across_values():
    constraint = _constraint("category", "athletic shoes")

    assert matches_constraint(
        {}, {"category": ("Shoes", "Athletic")}, constraint
    )
    assert not matches_constraint(
        {}, {"category": ("Shoes", "Formal")}, constraint
    )


def test_category_not_in_negates_matches_but_empty_availability_never_matches():
    excluded = _constraint("category", "athletic shoes", operator="not_in")

    assert not matches_constraint(
        {}, {"category": ("Shoes", "Athletic")}, excluded
    )
    assert matches_constraint({}, {"category": ("Shoes", "Formal")}, excluded)
    assert not matches_constraint({}, {"category": ()}, excluded)


def test_soft_clarification_matches_open_text_but_hard_message_stays_structured():
    product = {
        "features": ["Machine washable with reinforced seams"],
        "description": ["A durable everyday layer"],
    }
    clarification = _constraint(
        "feature",
        "machine washable with reinforced seams",
        is_hard=False,
        confidence=0.6,
        source="clarification",
    )
    message = _constraint(
        "feature",
        "machine washable with reinforced seams",
        is_hard=True,
    )

    assert matches_constraint(product, {"feature": ()}, clarification)
    assert not matches_constraint(product, {"feature": ()}, message)


def test_in_and_not_in_match_normalized_alternatives():
    attributes = {"color": ("Blue",)}

    assert matches_constraint(
        {}, attributes, _constraint("color", "red", operator="in", alternatives=("BLUE",))
    )
    assert not matches_constraint(
        {}, attributes, _constraint("color", "red", operator="not_in", alternatives=("BLUE",))
    )


def test_not_in_matches_a_present_non_excluded_value():
    assert matches_constraint(
        {}, {"color": ("blue",)}, _constraint("color", "red", operator="not_in")
    )


def test_numeric_budget_bounds_and_between_are_inclusive():
    product = {"price": "80.00"}

    assert matches_constraint(product, {}, _constraint("budget", "80", operator="lte"))
    assert matches_constraint(product, {}, _constraint("budget", "80", operator="gte"))
    assert matches_constraint(
        product, {}, _constraint("budget", "60", operator="between", upper_value="80")
    )
    assert not matches_constraint(
        product, {}, _constraint("budget", "79", operator="lte")
    )
    assert not matches_constraint(
        {"price": "75"}, {}, _constraint("budget", "80", operator="gte")
    )


def test_numeric_operators_only_apply_to_budget():
    assert not matches_constraint(
        {}, {"size": ("80",)}, _constraint("size", "80", operator="lte")
    )


def test_budget_eq_and_not_in_compare_numeric_prices():
    product = {"price": "80.00"}

    assert matches_constraint(product, {}, _constraint("budget", "80"))
    assert matches_constraint(product, {}, _constraint("budget", "79", operator="not_in"))


def test_invalid_budget_prices_and_bounds_do_not_match():
    for invalid in ("-1", "0", "NaN", "Infinity", "-Infinity"):
        assert not matches_constraint(
            {"price": invalid}, {}, _constraint("budget", "80", operator="lte")
        )

    assert not matches_constraint(
        {"price": "80"}, {}, _constraint("budget", "-1", operator="gte")
    )
    assert not matches_constraint(
        {"price": "80"}, {}, _constraint("budget", "NaN", operator="lte")
    )
    assert not matches_constraint(
        {"price": "80"}, {}, _constraint("budget", "Infinity", operator="lte")
    )


def test_missing_or_non_numeric_values_fail_hard_constraints():
    constraints = (
        _constraint("color", "blue"),
        _constraint("budget", "80", operator="lte"),
        _constraint("size", "large", is_hard=False),
    )

    assert not matches_constraint({}, {}, constraints[1])
    assert hard_constraint_violations({"price": "unknown"}, {}, constraints) == (
        "color=blue",
        "budget<=80",
    )


def test_display_constraint_is_stable_and_readable():
    assert display_constraint(_constraint("color", "red")) == "color=red"
    assert (
        display_constraint(
            _constraint("size", "stale", operator="in", alternatives=("medium", "large"))
        )
        == "size in (medium,large)"
    )
    assert (
        display_constraint(
            _constraint("color", "red", operator="not_in", alternatives=("blue",))
        )
        == "color not in (blue)"
    )
    assert (
        display_constraint(_constraint("budget", "50", operator="between", upper_value="80"))
        == "budget between 50 and 80"
    )
