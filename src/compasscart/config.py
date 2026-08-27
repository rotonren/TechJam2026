from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    candidate_limit: int = 500
    rrf_k: int = 60
    max_recommendations: int = 10
    component_timeout_ms: int = 800
    dense_model_dir: str | Path = Path("assets/model")
    dense_vector_dir: str | Path = Path("assets/product_vectors")
    dense_manifest_path: str | Path = Path("assets/SHA256SUMS")
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
    rank_fusion_weight: float = 0.10
    rank_attribute_weight: float = 0.0
    rank_consensus_bonus: float = 0.0
    rank_boundary_bonus: float = 0.0
    mmr_lambda: float = 0.85
    adaptive_browsing_mmr: bool = True

    def __post_init__(self) -> None:
        self._validate_choice(
            "rank_fusion_weight", self.rank_fusion_weight, {0.10, 0.15}
        )
        self._validate_choice(
            "rank_attribute_weight", self.rank_attribute_weight, {0.0, 0.05, 0.10}
        )
        self._validate_choice(
            "rank_consensus_bonus", self.rank_consensus_bonus, {0.0, 0.025, 0.05}
        )
        self._validate_choice(
            "rank_boundary_bonus", self.rank_boundary_bonus, {0.0, 0.025}
        )
        self._validate_choice("mmr_lambda", self.mmr_lambda, {0.85})
        if not isinstance(self.adaptive_browsing_mmr, bool):
            raise TypeError("adaptive_browsing_mmr must be a bool")
        if self.rank_fusion_weight + self.rank_attribute_weight > 0.40:
            raise ValueError(
                "rank_fusion_weight + rank_attribute_weight must not exceed 0.40"
            )

    @staticmethod
    def _validate_choice(name: str, value: float, allowed: set[float]) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite number")
        if not math.isfinite(float(value)) or value not in allowed:
            raise ValueError(f"{name} must be one of {sorted(allowed)}")

    def resolve_dense_paths(
        self, submission_root: Path
    ) -> tuple[Path, Path, Path]:
        root = submission_root.resolve()
        if not (root / "agent.py").is_file() or not (root / "assets").is_dir():
            raise ValueError("submission root must contain agent.py and assets")

        def resolve(path: str | Path) -> Path:
            resolved_path = Path(path)
            return resolved_path if resolved_path.is_absolute() else root / resolved_path

        return (
            resolve(self.dense_model_dir),
            resolve(self.dense_vector_dir),
            resolve(self.dense_manifest_path),
        )
