"""Deterministic sampling helpers for proxy product datasets."""

from __future__ import annotations

import hashlib
import heapq
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
    prepared, frequencies, category_best = _prepare_records(records, seed)
    _validate_representative_count(count, len(prepared), len(category_best))

    selected_indices = set(category_best.values())
    quotas = _marginal_quotas(frequencies, count, seed)
    deficits = quotas.copy()
    for index in selected_indices:
        for dimension in prepared[index][1]:
            if deficits[dimension] > 0:
                deficits[dimension] -= 1

    cells: dict[tuple[tuple[str, str], ...], list[int]] = {}
    for index, (_, dimensions, _) in enumerate(prepared):
        cells.setdefault(dimensions, []).append(index)
    for indices in cells.values():
        indices.sort(key=lambda index: (prepared[index][2], prepared[index][0]))

    positions = {dimensions: 0 for dimensions in cells}
    heap: list[tuple[int, int, str, tuple[tuple[str, str], ...]]] = []

    def enqueue(dimensions: tuple[tuple[str, str], ...]) -> None:
        indices = cells[dimensions]
        position = positions[dimensions]
        while position < len(indices) and indices[position] in selected_indices:
            position += 1
        positions[dimensions] = position
        if position < len(indices):
            parent_asin, _, record_hash = prepared[indices[position]]
            score = sum(deficits[dimension] for dimension in dimensions)
            heapq.heappush(heap, (-score, record_hash, parent_asin, dimensions))

    for dimensions in cells:
        enqueue(dimensions)

    while len(selected_indices) < count:
        negative_score, record_hash, parent_asin, dimensions = heapq.heappop(heap)
        index = cells[dimensions][positions[dimensions]]
        score = sum(deficits[dimension] for dimension in dimensions)
        if score != -negative_score or (record_hash, parent_asin) != (
            prepared[index][2],
            prepared[index][0],
        ):
            enqueue(dimensions)
            continue

        selected_indices.add(index)
        for dimension in dimensions:
            if deficits[dimension] > 0:
                deficits[dimension] -= 1
        enqueue(dimensions)

    selected = (prepared[index] for index in selected_indices)
    return [
        parent_asin
        for parent_asin, _, _ in sorted(selected, key=lambda item: (item[2], item[0]))
    ]


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
    frequencies: Counter[tuple[str, str]], count: int, seed: int
) -> Counter[tuple[str, str]]:
    quotas: Counter[tuple[str, str]] = Counter()
    population = sum(frequency for dimension, frequency in frequencies.items() if dimension[0] == "category")

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


def _prepare_records(
    records: list[ProxyProduct], seed: int
) -> tuple[
    list[tuple[str, tuple[tuple[str, str], ...], int]],
    Counter[tuple[str, str]],
    dict[str, int],
]:
    prepared: list[tuple[str, tuple[tuple[str, str], ...], int]] = []
    frequencies: Counter[tuple[str, str]] = Counter()
    category_best: dict[str, int] = {}
    seen_ids: set[str] = set()

    for index, record in enumerate(records):
        if record.parent_asin in seen_ids:
            raise ValueError("duplicate parent_asin values are not allowed")
        seen_ids.add(record.parent_asin)
        dimensions = record.dimensions()
        record_hash = stable_int(seed, record.parent_asin)
        prepared.append((record.parent_asin, dimensions, record_hash))
        frequencies.update(dimensions)
        best_index = category_best.get(record.category)
        if best_index is None or (record_hash, record.parent_asin) < (
            prepared[best_index][2],
            prepared[best_index][0],
        ):
            category_best[record.category] = index

    return prepared, frequencies, category_best


def _validate_unique_ids(records: list[ProxyProduct]) -> None:
    identifiers = [record.parent_asin for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate parent_asin values are not allowed")


def _validate_representative_count(count: int, population: int, categories: int) -> None:
    if count < categories or count > population:
        raise ValueError("count must be between the category count and population")
