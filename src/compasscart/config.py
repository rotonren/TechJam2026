from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    candidate_limit: int = 500
    rrf_k: int = 60
    max_recommendations: int = 10
    component_timeout_ms: int = 800
    dense_model_dir: Path = Path("assets/model")
    dense_vector_dir: Path = Path("assets/product_vectors")
    dense_manifest_path: Path = Path("assets/SHA256SUMS")
    bm25_field_weights: tuple[float, ...] = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
    buying_route_weights: tuple[tuple[str, float], ...] = (
        ("attribute", 0.45),
        ("lexical", 0.35),
        ("dense", 0.20),
    )
    browsing_route_weights: tuple[tuple[str, float], ...] = (
        ("dense", 0.45),
        ("lexical", 0.30),
        ("profile", 0.25),
    )
    override_route_weights: tuple[tuple[str, float], ...] = (
        ("lexical", 0.35),
        ("dense", 0.35),
        ("attribute", 0.30),
    )
