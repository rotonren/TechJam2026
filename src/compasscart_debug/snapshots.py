from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

_RESPONSE_FIELDS = (
    "message",
    "ask_attribute",
    "recommendations",
    "usage",
)


def json_safe(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        snapshot: dict[str, object] = {}
        for key in sorted(value, key=_mapping_sort_key):
            json_key = _mapping_key(key)
            if json_key in snapshot:
                raise TypeError("Mapping keys collide after JSON conversion.")
            snapshot[json_key] = json_safe(value[key])
        return snapshot
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value, key=_set_sort_key)]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return _safe_string(value)


def snapshot_state(state: object) -> dict[str, object]:
    if not dataclasses.is_dataclass(state) or isinstance(state, type):
        raise TypeError("State snapshot is invalid.")
    snapshot = json_safe(state)
    if not isinstance(snapshot, dict):
        raise TypeError("State snapshot is invalid.")
    return snapshot


def snapshot_products(
    agent: object, response: Mapping[str, object]
) -> list[dict[str, object]]:
    catalog = getattr(agent, "catalog", None)
    products = getattr(catalog, "products", None)
    attributes = getattr(catalog, "attributes", None)
    if not isinstance(products, Mapping) or not isinstance(attributes, Mapping):
        raise TypeError("Agent catalog metadata is unavailable.")
    recommendations = response.get("recommendations")
    if not isinstance(recommendations, list):
        raise TypeError("Response recommendations are invalid.")
    snapshots: list[dict[str, object]] = []
    for rank, recommendation in enumerate(recommendations, start=1):
        identifier = _recommendation_identifier(recommendation)
        product = products.get(identifier)
        metadata_missing = not isinstance(product, Mapping)
        product = product if isinstance(product, Mapping) else {}
        normalized_attributes = (
            {} if metadata_missing else attributes.get(identifier, {})
        )
        if not isinstance(normalized_attributes, Mapping):
            normalized_attributes = {}
        snapshots.append(
            {
                "rank": rank,
                "parent_asin": identifier,
                "title": json_safe(product.get("title")),
                "price": json_safe(product.get("price")),
                "rating": json_safe(product.get("average_rating")),
                "rating_count": json_safe(product.get("rating_number")),
                "store": json_safe(product.get("store")),
                "categories": json_safe(product.get("categories", [])),
                "features": json_safe(product.get("features", [])),
                "details": json_safe(product.get("details", {})),
                "normalized_attributes": json_safe(normalized_attributes),
                "metadata_missing": metadata_missing,
            }
        )
    return snapshots


def capture_exact_trace(
    records: Sequence[Mapping[str, object]], session_id: str, turn: int
) -> dict[str, object] | None:
    if not records:
        return None
    record = records[-1]
    if record.get("session_id") != session_id or record.get("turn") != turn:
        return None
    snapshot = json_safe(record)
    return snapshot if isinstance(snapshot, dict) else None


def snapshot_response(response: object) -> dict[str, object]:
    if not isinstance(response, Mapping):
        raise TypeError("Agent response is invalid.")
    if any(name not in response for name in _RESPONSE_FIELDS):
        raise ValueError("Agent response is invalid.")
    if not isinstance(response["message"], str):
        raise TypeError("Agent response is invalid.")
    if not isinstance(response["recommendations"], list):
        raise TypeError("Agent response is invalid.")
    recommendations: list[dict[str, str]] = []
    for recommendation in response["recommendations"]:
        try:
            identifier = _recommendation_identifier(recommendation)
        except (TypeError, ValueError) as error:
            raise ValueError("Agent response is invalid.") from error
        recommendations.append({"parent_asin": identifier})
    if response["ask_attribute"] is not None and not isinstance(
        response["ask_attribute"], str
    ):
        raise TypeError("Agent response is invalid.")
    if not isinstance(response["usage"], Mapping):
        raise TypeError("Agent response is invalid.")
    return {
        "message": response["message"],
        "ask_attribute": response["ask_attribute"],
        "recommendations": recommendations,
        "usage": json_safe(response["usage"]),
    }


def _recommendation_identifier(recommendation: object) -> str:
    if not isinstance(recommendation, Mapping):
        raise TypeError("Response recommendations are invalid.")
    identifier = recommendation.get("parent_asin")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Response recommendations are invalid.")
    return identifier


def _mapping_sort_key(value: object) -> tuple[str, str, str]:
    value_type = type(value)
    return (value_type.__module__, value_type.__qualname__, _mapping_key(value))


def _mapping_key(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return str(value)
    text = _safe_string(value)
    if text is None:
        raise TypeError("Mapping key is not JSON-safe.")
    return text


def _set_sort_key(value: object) -> tuple[str, str]:
    return (_mapping_key(value), type(value).__name__)


def _safe_string(value: object) -> str | None:
    try:
        text = str(value)
    except Exception:  # noqa: BLE001 - snapshots must stay available.
        return None
    return None if " object at 0x" in text else text
