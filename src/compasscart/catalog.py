from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from .constraints import hard_constraint_violations
from .models import Candidate, RetrievalPlan
from .normalization import (
    GENERIC_CATEGORIES,
    extract_attributes,
    normalize_value,
    searchable_fields,
    terms,
)


class CatalogIndex:
    def __init__(self, catalog_path: str | Path, *, enable_fts: bool = True) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.products: dict[str, dict[str, object]] = {}
        self.valid_ids: set[str] = set()
        self.attributes: dict[str, dict[str, tuple[str, ...]]] = {}
        self.attribute_inverted: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.field_terms: dict[str, tuple[set[str], ...]] = {}
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

                self.products[parent_asin] = product
                self.valid_ids.add(parent_asin)
                self.attributes[parent_asin] = attributes
                if not self._fts_enabled:
                    self.field_terms[parent_asin] = tuple(
                        set(terms(field)) for field in fields
                    )
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
        return set(
            self.attribute_inverted.get(attribute, {}).get(
                normalize_value(value), set()
            )
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

    def _ensure_field_terms(self) -> None:
        if self.field_terms:
            return
        self.field_terms = {
            parent_asin: tuple(set(terms(field)) for field in searchable_fields(product))
            for parent_asin, product in self.products.items()
        }

    def _matches_hard(self, parent_asin: str, plan: RetrievalPlan) -> bool:
        constraints = plan.effective_hard_constraints()
        if constraints:
            return not hard_constraint_violations(
                self.products[parent_asin],
                self.attributes[parent_asin],
                constraints,
            )
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
        self._ensure_field_terms()
        query = set(query_terms)
        field_weights = (6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
        scored: list[tuple[float, float, str]] = []
        for parent_asin, fields in self.field_terms.items():
            if not self._matches_hard(parent_asin, plan):
                continue
            score = sum(
                weight * len(query.intersection(field))
                for weight, field in zip(field_weights, fields, strict=True)
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
