"""Deterministic sampling helpers for proxy product datasets."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class ProxyProduct:
    parent_asin: str
    category: str
    price_bin: str
    popularity_bin: str
    completeness_bin: str
    difficulty: str

    def dimensions(self) -> tuple[tuple[str, str], ...]:
        return (
            ("category", self.category),
            ("price", self.price_bin),
            ("popularity", self.popularity_bin),
            ("completeness", self.completeness_bin),
        )


def stable_int(seed: int, value: str) -> int:
    """Return a stable, seed-specific integer derived from SHA-256."""
    payload = f"{seed}\0{value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")


def representative_ids(
    records: list[ProxyProduct], *, count: int, seed: int
) -> list[str]:
    """Select a deterministic sample that covers each category and marginal."""
    _validate_unique_ids(records)
    categories = {record.category for record in records}
    _validate_representative_count(count, len(records), len(categories))

    selected = {
        min(
            (record for record in records if record.category == category),
            key=lambda record: _stable_record_key(seed, record),
        ).parent_asin
        for category in categories
    }
    quotas = _marginal_quotas(records, count, seed)
    selected_records = {
        record.parent_asin: record for record in records if record.parent_asin in selected
    }

    while len(selected) < count:
        selected_counts = Counter(
            dimension for record in selected_records.values() for dimension in record.dimensions()
        )
        candidates = (record for record in records if record.parent_asin not in selected)
        next_record = min(
            candidates,
            key=lambda record: (
                -sum(
                    max(0, quotas[dimension] - selected_counts[dimension])
                    for dimension in record.dimensions()
                ),
                *_stable_record_key(seed, record),
            ),
        )
        selected.add(next_record.parent_asin)
        selected_records[next_record.parent_asin] = next_record

    return sorted(selected, key=lambda parent_asin: (stable_int(seed, parent_asin), parent_asin))


def stress_ids(
    records: list[ProxyProduct], *, count: int, seed: int, excluded: set[str]
) -> list[str]:
    """Select a deterministic sample biased toward sparse marginal values."""
    _validate_unique_ids(records)
    candidates = [record for record in records if record.parent_asin not in excluded]
    if count < 0 or count > len(candidates):
        raise ValueError("count must be between zero and the available population")

    frequencies = Counter(dimension for record in records for dimension in record.dimensions())
    denominator = (1 << 256) + 1

    def ordering_key(record: ProxyProduct) -> tuple[float, int, str]:
        record_hash = stable_int(seed, record.parent_asin)
        unit = (record_hash + 1) / denominator
        weight = sum(1 / frequencies[dimension] for dimension in record.dimensions())
        return (-math.log(unit) / weight, record_hash, record.parent_asin)

    return [record.parent_asin for record in sorted(candidates, key=ordering_key)[:count]]


def _marginal_quotas(
    records: list[ProxyProduct], count: int, seed: int
) -> Counter[tuple[str, str]]:
    frequencies = Counter(dimension for record in records for dimension in record.dimensions())
    quotas: Counter[tuple[str, str]] = Counter()
    population = len(records)

    for dimension_name in ("category", "price", "popularity", "completeness"):
        values = [
            dimension
            for dimension in frequencies
            if dimension[0] == dimension_name
        ]
        exact = {
            dimension: count * frequencies[dimension] / population for dimension in values
        }
        floors = {dimension: math.floor(quota) for dimension, quota in exact.items()}
        remaining = count - sum(floors.values())
        remainders = sorted(
            values,
            key=lambda dimension: (
                -(exact[dimension] - floors[dimension]),
                stable_int(seed, f"{dimension[0]}\0{dimension[1]}"),
                dimension[1],
            ),
        )
        quotas.update(floors)
        quotas.update({dimension: 1 for dimension in remainders[:remaining]})

    return quotas


def _stable_record_key(seed: int, record: ProxyProduct) -> tuple[int, str]:
    return stable_int(seed, record.parent_asin), record.parent_asin


def _validate_unique_ids(records: list[ProxyProduct]) -> None:
    identifiers = [record.parent_asin for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate parent_asin values are not allowed")


def _validate_representative_count(count: int, population: int, categories: int) -> None:
    if count < categories or count > population:
        raise ValueError("count must be between the category count and population")
