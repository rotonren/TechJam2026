from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .normalization import GENERIC_CATEGORIES, normalize_value, terms

CORE_ATTRIBUTES = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
)
CORE_QUESTION_ATTRIBUTES = (
    "material",
    "color",
    "size",
    "style",
    "budget",
    "feature",
    "use_case",
)
PARSER_ATTRIBUTES = (
    "brand",
    "size",
    "category",
    "material",
    "style",
    "feature",
    "use_case",
)
GLOBAL_ATTRIBUTES = frozenset({"category", "brand", "budget"})
CATEGORY_ATTRIBUTES = frozenset({"material", "color", "size", "style"})
DYNAMIC_ATTRIBUTES = frozenset({"feature", "use_case"})
ALLOWED_ASK_ATTRIBUTES = frozenset((*CORE_ATTRIBUTES, "other"))
_CORE_RESPONSE_LIKELIHOOD = {
    "category": 0.95,
    "material": 0.90,
    "color": 0.90,
    "size": 0.85,
    "style": 0.80,
    "brand": 0.65,
    "budget": 0.90,
    "feature": 0.70,
    "use_case": 0.85,
}
_ATTRIBUTE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
AttributeLayer = Literal["global", "category", "dynamic"]


@dataclass(frozen=True)
class AttributeSpec:
    """One catalog-backed attribute recorded by the layered schema."""

    name: str
    ask_attribute: str
    response_likelihood: float
    values: tuple[str, ...] = ()
    catalog_coverage: float = 0.0
    value_count: int = 0
    layer: AttributeLayer = "global"
    source: str = "core"
    question_eligible: bool = False
    cues: tuple[str, ...] = ()
    category_scopes: tuple[str, ...] = ()


class AttributeSchema:
    """Immutable three-layer schema learned from normalized catalog attributes.

    The evaluator permits only a stable set of ``ask_attribute`` values.  The
    schema records catalog-specific fields for inspection and controlled
    experiments, while its default parser vocabulary exposes only the stable
    public attributes. This prevents discovered metadata from silently
    changing hard-filter behavior.
    """

    def __init__(self, specifications: Mapping[str, AttributeSpec]) -> None:
        self._specifications = MappingProxyType(dict(specifications))

    @classmethod
    def from_catalog(
        cls,
        attribute_inverted: Mapping[str, Mapping[str, set[str]]],
        product_count: int,
    ) -> AttributeSchema:
        """Build a layered schema from a legacy flat index."""
        layers: dict[str, dict[str, Mapping[str, set[str]]]] = {
            "global": {},
            "category": {},
            "dynamic": {},
        }
        for attribute, inverted in attribute_inverted.items():
            if attribute in GLOBAL_ATTRIBUTES:
                layer = "global"
            elif attribute in CATEGORY_ATTRIBUTES or attribute not in CORE_ATTRIBUTES:
                layer = "category"
            else:
                layer = "dynamic"
            layers[layer][attribute] = inverted
        return cls.from_layers(layers, product_count=product_count)

    @classmethod
    def from_layers(
        cls,
        layers: Mapping[
            str, Mapping[str, Mapping[str, set[str]]]
        ],
        *,
        product_count: int,
        category_scopes: Mapping[str, set[str]] | None = None,
    ) -> AttributeSchema:
        specifications: dict[str, AttributeSpec] = {}
        attribute_names = dict.fromkeys(
            (
                *CORE_ATTRIBUTES,
                *(name for layer in layers.values() for name in layer),
            )
        )
        minimum_dynamic_coverage = (
            2 if product_count < 100 else max(12, math.ceil(product_count * 0.001))
        )
        minimum_value_support = 1 if product_count < 100 else 2

        for attribute in attribute_names:
            if not _ATTRIBUTE_NAME_RE.fullmatch(attribute):
                continue
            layer = cls._layer_for(attribute, layers)
            inverted = layers.get(layer, {}).get(attribute, {})
            covered_ids = set().union(*inverted.values()) if inverted else set()
            is_core = attribute in CORE_ATTRIBUTES
            values = tuple(
                sorted(
                    {
                        normalized
                        for value, identifiers in inverted.items()
                        if (normalized := normalize_value(value))
                        and (is_core or cls._safe_parser_value(normalized))
                        and (is_core or len(identifiers) >= minimum_value_support)
                        and not (
                            attribute == "category"
                            and normalized in GENERIC_CATEGORIES
                        )
                    }
                )
            )
            coverage = len(covered_ids) / product_count if product_count else 0.0
            value_count = len(inverted)
            question_eligible = (
                not is_core
                and len(covered_ids) >= minimum_dynamic_coverage
                and 2 <= value_count <= 128
                and len(values) >= 2
            )
            specifications[attribute] = AttributeSpec(
                name=attribute,
                ask_attribute=(attribute if attribute in ALLOWED_ASK_ATTRIBUTES else "other"),
                response_likelihood=_CORE_RESPONSE_LIKELIHOOD.get(attribute, 0.65),
                values=values,
                catalog_coverage=coverage,
                value_count=value_count,
                layer=layer,
                source="core" if is_core else "catalog",
                question_eligible=question_eligible,
                cues=cls._attribute_cues(attribute),
                category_scopes=tuple(
                    sorted((category_scopes or {}).get(attribute, set()))
                ),
            )

        return cls(specifications)

    @staticmethod
    def _layer_for(
        attribute: str,
        layers: Mapping[str, Mapping[str, Mapping[str, set[str]]]],
    ) -> AttributeLayer:
        if attribute in GLOBAL_ATTRIBUTES:
            return "global"
        if attribute in CATEGORY_ATTRIBUTES:
            return "category"
        if attribute in DYNAMIC_ATTRIBUTES:
            return "dynamic"
        for layer in ("global", "category", "dynamic"):
            if attribute in layers.get(layer, {}):
                return layer  # type: ignore[return-value]
        return "category"

    @staticmethod
    def _safe_parser_value(value: str) -> bool:
        value_terms = terms(value)
        return bool(
            value_terms
            and len(value) <= 80
            and len(value_terms) <= 8
            and not all(token.isdigit() for token in value_terms)
            and not re.search(r"(?:https?://|www\.)", value)
        )

    @staticmethod
    def _attribute_cues(attribute: str) -> tuple[str, ...]:
        defaults = {attribute.replace("_", " ")}
        defaults.update(
            {
                "audience": ("audience", "department", "target audience"),
                "care": ("care", "care instructions"),
                "closure": ("closure", "closure type"),
                "fit": ("fit", "fit type"),
                "neckline": ("neck", "neckline", "neck style"),
                "outer_material": ("outer material", "upper", "upper material"),
                "sole_material": ("sole", "sole material"),
                "inner_material": ("inner material", "lining", "lining material"),
                "sleeve": ("sleeve", "sleeve type"),
            }.get(attribute, ())
        )
        return tuple(sorted(defaults, key=lambda cue: (-len(cue), cue)))

    @property
    def specifications(self) -> Mapping[str, AttributeSpec]:
        return self._specifications

    @property
    def dynamic_attributes(self) -> frozenset[str]:
        return frozenset(
            name for name, spec in self._specifications.items() if spec.layer == "dynamic"
        )

    @property
    def discovered_attributes(self) -> frozenset[str]:
        return frozenset(
            name for name, spec in self._specifications.items() if spec.source == "catalog"
        )

    def specifications_for_layer(
        self, layer: AttributeLayer
    ) -> tuple[AttributeSpec, ...]:
        return tuple(
            spec for spec in self._specifications.values() if spec.layer == layer
        )

    def parser_vocabulary(
        self, *, include_discovered: bool = False
    ) -> Mapping[str, tuple[str, ...]]:
        names = (
            tuple(self._specifications)
            if include_discovered
            else PARSER_ATTRIBUTES
        )
        return MappingProxyType(
            {
                name: self._specifications[name].values
                for name in names
                if name in self._specifications and self._specifications[name].values
            }
        )

    def question_specs(
        self, *, include_dynamic: bool = False
    ) -> tuple[AttributeSpec, ...]:
        core = tuple(
            self._specifications[name]
            for name in CORE_QUESTION_ATTRIBUTES
            if name in self._specifications
        )
        dynamic = tuple(
            sorted(
                (
                    spec
                    for spec in self._specifications.values()
                    if spec.question_eligible
                ),
                key=lambda spec: (
                    -spec.catalog_coverage,
                    spec.value_count,
                    spec.name,
                ),
            )
        )
        return (*core, *dynamic) if include_dynamic else core

    def cues_for(self, attribute: str) -> tuple[str, ...]:
        spec = self._specifications.get(attribute)
        return spec.cues if spec else self._attribute_cues(attribute)
