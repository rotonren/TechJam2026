from __future__ import annotations

from typing import get_type_hints

import pytest

from tools.proxy_dataset import (
    ProxyProduct,
    representative_ids,
    stable_int,
    stress_ids,
)


def make_products(count: int = 40) -> list[ProxyProduct]:
    return [
        ProxyProduct(
            parent_asin=f"P{index:03d}",
            category=f"category-{index % 4}",
            price_bin=f"price-{index % 3}",
            popularity_bin=f"popularity-{index % 5}",
            completeness_bin=f"completeness-{index % 2}",
            difficulty=("easy", "medium", "hard")[index % 3],
        )
        for index in range(count)
    ]


def test_stable_int_is_deterministic_and_seeded() -> None:
    assert stable_int(20260826, "P001") == stable_int(20260826, "P001")
    assert stable_int(20260826, "P001") != stable_int(20260827, "P001")


def test_proxy_product_difficulty_is_a_string_bucket() -> None:
    assert get_type_hints(ProxyProduct)["difficulty"] is str


def test_proxy_product_dimensions_expose_sampling_marginals() -> None:
    product = ProxyProduct(
        parent_asin="P001",
        category="audio",
        price_bin="mid",
        popularity_bin="high",
        completeness_bin="complete",
        difficulty="hard",
    )

    assert product.dimensions() == (
        ("category", "audio"),
        ("price", "mid"),
        ("popularity", "high"),
        ("completeness", "complete"),
    )


def test_representative_ids_are_order_independent_unique_and_cover_categories() -> None:
    records = make_products()

    selected = representative_ids(records, count=20, seed=20260826)
    reordered = representative_ids(list(reversed(records)), count=20, seed=20260826)

    assert selected == reordered
    assert len(selected) == 20
    assert len(set(selected)) == 20
    categories = {record.category for record in records if record.parent_asin in selected}
    assert categories == {"category-0", "category-1", "category-2", "category-3"}


def test_representative_ids_reject_duplicate_parent_asins() -> None:
    records = make_products(4)
    records.append(records[0])

    with pytest.raises(ValueError, match="duplicate parent_asin"):
        representative_ids(records, count=4, seed=20260826)


def test_stress_ids_are_disjoint_unique_and_include_a_rare_category() -> None:
    records = [
        ProxyProduct(
            parent_asin=f"P{index:03d}",
            category="common" if index < 35 else "rare",
            price_bin="mid",
            popularity_bin="high",
            completeness_bin="complete",
            difficulty="medium",
        )
        for index in range(40)
    ]
    excluded = set(representative_ids(records, count=20, seed=20260826))

    selected = stress_ids(records, count=8, seed=20260826, excluded=excluded)

    assert len(selected) == 8
    assert len(set(selected)) == 8
    assert set(selected).isdisjoint(excluded)
    assert any(record.parent_asin in selected and record.category == "rare" for record in records)


def test_stress_ids_validate_available_population() -> None:
    records = make_products(2)

    with pytest.raises(ValueError, match="count"):
        stress_ids(records, count=3, seed=20260826, excluded=set())
