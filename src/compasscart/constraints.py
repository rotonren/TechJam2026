from __future__ import annotations

import math
from collections.abc import Collection, Iterable

from .models import Constraint
from .normalization import (
    category_term_set,
    normalize_value,
    searchable_term_set,
    terms,
)


def matches_constraint(
    product: dict[str, object],
    attributes: dict[str, tuple[str, ...]],
    constraint: Constraint,
    *,
    category_terms: Collection[str] | None = None,
    searchable_terms: Collection[str] | None = None,
) -> bool:
    """Return whether a product satisfies one normalized constraint."""
    if constraint.operator in {"lte", "gte", "between"}:
        return constraint.attribute == "budget" and _matches_budget_range(
            product, constraint
        )
    if constraint.attribute == "budget":
        return _matches_budget_equality(product, constraint)

    if constraint.attribute == "category":
        available = frozenset(
            category_terms
            if category_terms is not None
            else category_term_set(attributes.get("category", ()))
        )
        match = any(
            wanted and wanted.issubset(available)
            for value in constraint.values()
            if (wanted := category_term_set(value))
        )
        return _apply_set_operator(match, bool(available), constraint)

    if constraint.source == "clarification" and not constraint.is_hard:
        available = frozenset(
            searchable_terms
            if searchable_terms is not None
            else searchable_term_set(product)
        )
        match = any(
            wanted and wanted.issubset(available)
            for value in constraint.values()
            if (wanted := frozenset(terms(value)))
        )
        return _apply_set_operator(match, bool(available), constraint)

    values = _product_values(attributes, constraint.attribute)
    if not values:
        return False
    wanted = {normalize_value(value) for value in constraint.values()}
    normalized_values = {normalize_value(value) for value in values}
    if constraint.operator in {"eq", "in"}:
        return bool(normalized_values & wanted)
    if constraint.operator == "not_in":
        return not bool(normalized_values & wanted)
    return False


def hard_constraint_violations(
    product: dict[str, object],
    attributes: dict[str, tuple[str, ...]],
    constraints: Iterable[Constraint],
    *,
    category_terms: Collection[str] | None = None,
    searchable_terms: Collection[str] | None = None,
) -> tuple[str, ...]:
    return tuple(
        display_constraint(constraint)
        for constraint in constraints
        if constraint.is_hard
        and not matches_constraint(
            product,
            attributes,
            constraint,
            category_terms=category_terms,
            searchable_terms=searchable_terms,
        )
    )


def _apply_set_operator(
    match: bool, has_available: bool, constraint: Constraint
) -> bool:
    if not has_available:
        return False
    if constraint.operator in {"eq", "in"}:
        return match
    if constraint.operator == "not_in":
        return not match
    return False


def display_constraint(constraint: Constraint) -> str:
    values = ",".join(constraint.values())
    if constraint.operator == "eq":
        return f"{constraint.attribute}={constraint.value}"
    if constraint.operator == "in":
        return f"{constraint.attribute} in ({values})"
    if constraint.operator == "not_in":
        return f"{constraint.attribute} not in ({values})"
    if constraint.operator == "lte":
        return f"{constraint.attribute}<={constraint.value}"
    if constraint.operator == "gte":
        return f"{constraint.attribute}>={constraint.value}"
    if constraint.operator == "between":
        return f"{constraint.attribute} between {constraint.value} and {constraint.upper_value}"
    return f"{constraint.attribute} {constraint.operator} {values}"


def _product_values(
    attributes: dict[str, tuple[str, ...]], attribute: str
) -> tuple[object, ...]:
    raw_values = attributes.get(attribute, ())
    if isinstance(raw_values, str):
        return (raw_values,)
    return tuple(raw_values)


def _matches_budget_range(product: dict[str, object], constraint: Constraint) -> bool:
    price = _positive_finite_number(product.get("price"))
    lower = _positive_finite_number(constraint.value)
    if price is None or lower is None:
        return False
    if constraint.operator == "lte":
        return price <= lower
    if constraint.operator == "gte":
        return price >= lower
    upper = _positive_finite_number(constraint.upper_value)
    return upper is not None and lower <= price <= upper


def _matches_budget_equality(product: dict[str, object], constraint: Constraint) -> bool:
    price = _positive_finite_number(product.get("price"))
    expected = {
        number
        for value in constraint.values()
        if (number := _positive_finite_number(value)) is not None
    }
    if price is None or not expected:
        return False
    match = price in expected
    if constraint.operator in {"eq", "in"}:
        return match
    if constraint.operator == "not_in":
        return not match
    return False


def _positive_finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and math.isfinite(number) else None
