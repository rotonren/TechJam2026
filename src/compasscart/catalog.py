from __future__ import annotations

import json
import math
import sqlite3
from array import array
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from pathlib import Path
from types import MappingProxyType

from .constraints import hard_constraint_violations, matches_constraint
from .models import Candidate, Constraint, RetrievalPlan
from .normalization import (
    GENERIC_CATEGORIES,
    category_term_set,
    extract_attributes,
    normalize_value,
    searchable_fields,
    terms,
)

FIELD_WEIGHTS = (6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
FIELD_MASK_SCORES = tuple(
    sum(weight for index, weight in enumerate(FIELD_WEIGHTS) if mask & (1 << index))
    for mask in range(1 << len(FIELD_WEIGHTS))
)


class _CompactTermSet(AbstractSet[str]):
    __slots__ = ("_ids", "_term_ids", "_terms")

    def __init__(
        self,
        identifiers: array[int],
        term_ids: dict[str, int],
        terms_by_id: list[str],
    ) -> None:
        self._ids = identifiers
        self._term_ids = term_ids
        self._terms = terms_by_id

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, str):
            return False
        identifier = self._term_ids.get(value)
        if identifier is None:
            return False
        position = bisect_left(self._ids, identifier)
        return position < len(self._ids) and self._ids[position] == identifier

    def __iter__(self):
        return (self._terms[identifier] for identifier in self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    @classmethod
    def _from_iterable(cls, iterable: Iterable[str]) -> frozenset[str]:
        return frozenset(iterable)

    def issubset(self, other: Iterable[str]) -> bool:
        return frozenset(self).issubset(other)

    def issuperset(self, other: Iterable[str]) -> bool:
        return frozenset(self).issuperset(other)

    def union(self, *others: Iterable[str]) -> frozenset[str]:
        return frozenset(self).union(*others)

    def intersection(self, *others: Iterable[str]) -> frozenset[str]:
        return frozenset(self).intersection(*others)

    def difference(self, *others: Iterable[str]) -> frozenset[str]:
        return frozenset(self).difference(*others)

    def symmetric_difference(self, other: Iterable[str]) -> frozenset[str]:
        return frozenset(self).symmetric_difference(other)

    @property
    def storage_bytes(self) -> int:
        return len(self._ids) * self._ids.itemsize


class CatalogIndex:
    def __init__(self, catalog_path: str | Path, *, enable_fts: bool = True) -> None:
        self.catalog_path = Path(catalog_path)
        # An empty filename gives SQLite an automatically deleted temporary database.
        self.connection = sqlite3.connect("")
        self.products: dict[str, dict[str, object]] = {}
        self.valid_ids: set[str] = set()
        self.attributes: dict[str, dict[str, tuple[str, ...]]] = {}
        self.category_terms: dict[str, frozenset[str]] = {}
        self.searchable_terms: dict[str, _CompactTermSet] = {}
        self._search_term_ids: dict[str, int] = {}
        self._search_terms: list[str] = []
        self.category_term_inverted: dict[str, set[str]] = defaultdict(set)
        self.attribute_inverted: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.field_terms: dict[str, tuple[set[str], ...]] = {}
        self.field_masks: dict[str, bytes] = {}
        self.quality: dict[str, float] = {}
        self._fts_enabled = False
        self._load(enable_fts=enable_fts)

    def _load(self, *, enable_fts: bool) -> None:
        cursor = self.connection.cursor()
        if enable_fts:
            try:
                cursor.execute(
                    "CREATE VIRTUAL TABLE products USING fts5("
                    "parent_asin UNINDEXED, title, categories, features, details, store, "
                    "description, tokenize='unicode61 remove_diacritics 2')"
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False

        fts_batch: list[tuple[str, ...]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                fields = searchable_fields(product)
                attributes = extract_attributes(product)
                category_terms = category_term_set(attributes.get("category", ()))
                searchable_terms, field_masks = _compact_searchable_terms(
                    fields, self._search_term_ids, self._search_terms
                )

                self.products[parent_asin] = product
                self.valid_ids.add(parent_asin)
                self.attributes[parent_asin] = attributes
                self.category_terms[parent_asin] = category_terms
                self.searchable_terms[parent_asin] = searchable_terms
                self.field_masks[parent_asin] = field_masks
                for term in category_terms:
                    self.category_term_inverted[term].add(parent_asin)
                for attribute, values in attributes.items():
                    for value in values:
                        self.attribute_inverted[attribute][value].add(parent_asin)
                if self._fts_enabled:
                    fts_batch.append((parent_asin, *fields))
                    if len(fts_batch) >= 1_000:
                        cursor.executemany(
                            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)",
                            fts_batch,
                        )
                        fts_batch.clear()

        if self._fts_enabled:
            if fts_batch:
                cursor.executemany(
                    "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", fts_batch
                )
            self.connection.commit()

        max_reviews = max(
            (
                self._number(product.get("rating_number"))
                for product in self.products.values()
            ),
            default=1.0,
        )
        review_scale = max(math.log1p(max_reviews), 1.0)
        for parent_asin, product in self.products.items():
            rating = min(max(self._number(product.get("average_rating")), 0.0), 5.0)
            reviews = max(self._number(product.get("rating_number")), 0.0)
            self.quality[parent_asin] = 0.7 * (rating / 5.0) + 0.3 * (
                math.log1p(reviews) / review_scale
            )

    @staticmethod
    def _number(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def attribute_ids(self, attribute: str, value: str) -> set[str]:
        if attribute == "category":
            return self.category_ids(value)
        return set(
            self.attribute_inverted.get(attribute, {}).get(
                normalize_value(value), set()
            )
        )

    def category_ids(self, value: object) -> set[str]:
        desired = category_term_set(value)
        if not desired:
            return set()
        postings = iter(desired)
        matches = set(self.category_term_inverted.get(next(postings), set()))
        for term in postings:
            matches.intersection_update(self.category_term_inverted.get(term, set()))
            if not matches:
                break
        return matches

    def matches(self, parent_asin: str, constraint: Constraint) -> bool:
        return matches_constraint(
            self.products[parent_asin],
            self.attributes[parent_asin],
            constraint,
            category_terms=self.category_terms[parent_asin],
            searchable_terms=self.searchable_terms[parent_asin],
        )

    def violations(
        self, parent_asin: str, constraints: Iterable[Constraint]
    ) -> tuple[str, ...]:
        return hard_constraint_violations(
            self.products[parent_asin],
            self.attributes[parent_asin],
            constraints,
            category_terms=self.category_terms[parent_asin],
            searchable_terms=self.searchable_terms[parent_asin],
        )

    def parser_vocabulary(self) -> Mapping[str, tuple[str, ...]]:
        attributes = ("brand", "size", "category", "material", "style", "feature", "use_case")
        vocabulary: dict[str, tuple[str, ...]] = {}
        for attribute in attributes:
            values = {
                normalize_value(value)
                for value in self.attribute_inverted.get(attribute, {})
                if normalize_value(value)
            }
            if attribute == "category":
                values.difference_update(GENERIC_CATEGORIES)
            vocabulary[attribute] = tuple(sorted(values))
        return MappingProxyType(vocabulary)

    def product(self, parent_asin: str) -> dict[str, object]:
        return self.products[parent_asin]

    def popular_ids(self, limit: int = 10) -> list[str]:
        return sorted(self.valid_ids, key=lambda item: (-self.quality[item], item))[
            :limit
        ]

    def search_lexical(self, plan: RetrievalPlan, *, limit: int) -> list[Candidate]:
        query_terms = terms(plan.query_text)
        if not query_terms:
            return self._as_candidates(self.popular_ids(limit), source="lexical")

        if self._fts_enabled:
            try:
                expression = " OR ".join(f'"{token}"' for token in query_terms[:40])
                rows = self.connection.execute(
                    "SELECT parent_asin FROM products WHERE products MATCH ? "
                    "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) "
                    "LIMIT ?",
                    (expression, max(limit * 3, limit)),
                ).fetchall()
                ranked = [str(row[0]) for row in rows]
                ranked = [item for item in ranked if self._matches_hard(item, plan)]
                return self._as_candidates(ranked[:limit], source="lexical")
            except sqlite3.OperationalError:
                self._fts_enabled = False

        return self._fallback_search(plan, query_terms, limit)

    def _matches_hard(self, parent_asin: str, plan: RetrievalPlan) -> bool:
        constraints = plan.effective_hard_constraints()
        if constraints:
            return not self.violations(parent_asin, constraints)
        attributes = self.attributes[parent_asin]
        for attribute, values in plan.hard_filters.items():
            if attribute == "budget":
                ceilings = [self._number(value) for value in values]
                price = self._number(self.products[parent_asin].get("price"))
                if not ceilings or price <= 0 or price > max(ceilings):
                    return False
            elif not any(value in attributes.get(attribute, ()) for value in values):
                return False
        return True

    def _fallback_search(
        self, plan: RetrievalPlan, query_terms: list[str], limit: int
    ) -> list[Candidate]:
        query_ids = frozenset(
            identifier
            for term in query_terms
            if (identifier := self._search_term_ids.get(term)) is not None
        )
        if not query_ids:
            return []
        scored: list[tuple[float, float, str]] = []
        for parent_asin, searchable_terms in self.searchable_terms.items():
            if not self._matches_hard(parent_asin, plan):
                continue
            score = sum(
                FIELD_MASK_SCORES[mask]
                for term_id, mask in zip(
                    searchable_terms._ids,
                    self.field_masks[parent_asin],
                    strict=True,
                )
                if term_id in query_ids
            )
            if score > 0:
                scored.append((score, self.quality[parent_asin], parent_asin))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return self._as_candidates(
            [parent_asin for _, _, parent_asin in scored[:limit]], source="lexical"
        )

    def _as_candidates(
        self, parent_asins: list[str], *, source: str
    ) -> list[Candidate]:
        return [
            Candidate(
                parent_asin=parent_asin,
                product=self.products[parent_asin],
                source_scores={source: 1.0 / rank},
                score=1.0 / rank,
            )
            for rank, parent_asin in enumerate(dict.fromkeys(parent_asins), start=1)
        ]


def _compact_searchable_terms(
    fields: tuple[str, ...],
    term_ids: dict[str, int],
    terms_by_id: list[str],
) -> tuple[_CompactTermSet, bytes]:
    masks: dict[str, int] = {}
    for index, field in enumerate(fields):
        bit = 1 << index
        for term in terms(field):
            masks[term] = masks.get(term, 0) | bit

    encoded: list[tuple[int, int]] = []
    for term, mask in masks.items():
        identifier = term_ids.get(term)
        if identifier is None:
            identifier = len(terms_by_id)
            if identifier >= 1 << 32:
                raise OverflowError("searchable vocabulary exceeds 32-bit term IDs")
            term_ids[term] = identifier
            terms_by_id.append(term)
        encoded.append((identifier, mask))
    encoded.sort()
    identifiers = array("I", (identifier for identifier, _ in encoded))
    return (
        _CompactTermSet(identifiers, term_ids, terms_by_id),
        bytes(mask for _, mask in encoded),
    )
