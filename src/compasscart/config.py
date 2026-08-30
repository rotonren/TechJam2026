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
    rank_fusion_weight: float = 0.25
    rank_attribute_weight: float = 0.0
    rank_consensus_bonus: float = 0.0
    rank_boundary_bonus: float = 0.0
    mmr_lambda: float = 0.85
    adaptive_browsing_mmr: bool = False
    dense_rescue_only: bool = True
    rerank_enabled: bool = True
    rerank_window: int = 50
    rerank_buying_window: int | None = None
    rerank_weight: float = 0.8
    # Measured: the rerank stage is worth +0.038 Browsing HitRate and -0.025 on
    # Buying, where explicit hard constraints already inform the ranker.
    rerank_buying_weight: float = 0.0
    rerank_backend: str = "phrase"
    rerank_buying_backend: str | None = None
    rerank_buying_requires_override: bool = False
    rerank_prompt_style: str = "flat"
    evolution_enabled: bool = True
    strategy_enabled: bool = True
    rerank_max_length: int = 128
    rerank_asset_dir: str | Path = Path("assets/reranker")

    def __post_init__(self) -> None:
        self._validate_choice(
            "rank_fusion_weight", self.rank_fusion_weight, {0.10, 0.15, 0.25}
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
        if not isinstance(self.dense_rescue_only, bool):
            raise TypeError("dense_rescue_only must be a bool")
        if not isinstance(self.strategy_enabled, bool):
            raise TypeError("strategy_enabled must be a bool")
        if not isinstance(self.evolution_enabled, bool):
            raise TypeError("evolution_enabled must be a bool")
        if not isinstance(self.rerank_enabled, bool):
            raise TypeError("rerank_enabled must be a bool")
        if isinstance(self.rerank_window, bool) or not isinstance(
            self.rerank_window, int
        ):
            raise TypeError("rerank_window must be an int")
        if self.rerank_window not in {20, 50, 100}:
            raise ValueError("rerank_window must be one of 20, 50, or 100")
        if self.rerank_buying_window is not None and (
            self.rerank_buying_window not in {20, 50, 100}
        ):
            raise ValueError(
                "rerank_buying_window must be one of 20, 50, or 100"
            )
        self._validate_choice(
            "rerank_weight", self.rerank_weight, {0.0, 0.3, 0.45, 0.6, 0.8, 1.0}
        )
        self._validate_choice(
            "rerank_buying_weight",
            self.rerank_buying_weight,
            {0.0, 0.3, 0.45, 0.6, 0.8, 1.0},
        )
        if self.rerank_prompt_style not in {"flat", "structured", "adaptive"}:
            raise ValueError(
                'rerank_prompt_style must be "flat", "structured" or "adaptive"'
            )
        if not isinstance(self.rerank_buying_requires_override, bool):
            raise TypeError("rerank_buying_requires_override must be a bool")
        allowed_backends = {"phrase", "cross_encoder", "llm"}
        if self.rerank_backend not in allowed_backends:
            raise ValueError(f"rerank_backend must be one of {sorted(allowed_backends)}")
        if (
            self.rerank_buying_backend is not None
            and self.rerank_buying_backend not in allowed_backends
        ):
            raise ValueError(
                f"rerank_buying_backend must be one of {sorted(allowed_backends)}"
            )
        if isinstance(self.rerank_max_length, bool) or not isinstance(
            self.rerank_max_length, int
        ):
            raise TypeError("rerank_max_length must be an int")
        if self.rerank_max_length not in {96, 128, 192, 256}:
            raise ValueError(
                "rerank_max_length must be one of 96, 128, 192, or 256"
            )
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

    def resolve_rerank_asset_dir(self, submission_root: Path) -> Path:
        """Resolve the rerank asset directory against the installed package."""
        path = Path(self.rerank_asset_dir)
        return path if path.is_absolute() else submission_root.resolve() / path

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
