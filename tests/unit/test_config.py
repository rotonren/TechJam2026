from __future__ import annotations

from pathlib import Path

import pytest

import compasscart.agent as agent_module
from compasscart.agent import CompassCartAgent
from compasscart.config import RuntimeConfig
from compasscart.dense import NullDenseBackend


def _write_submission_layout(root: Path) -> None:
    (root / "agent.py").write_text("# submission entry point\n", encoding="utf-8")
    (root / "assets").mkdir()


def test_resolve_dense_paths_uses_submission_root_for_defaults(tmp_path: Path):
    submission_root = tmp_path / "release"
    submission_root.mkdir()
    _write_submission_layout(submission_root)

    paths = RuntimeConfig().resolve_dense_paths(submission_root)

    assert paths == (
        submission_root / "assets" / "model",
        submission_root / "assets" / "product_vectors",
        submission_root / "assets" / "SHA256SUMS",
    )


def test_resolve_dense_paths_preserves_absolute_overrides(tmp_path: Path):
    submission_root = tmp_path / "release"
    submission_root.mkdir()
    _write_submission_layout(submission_root)
    model_dir = tmp_path / "external" / "model"
    vector_dir = tmp_path / "external" / "vectors"
    manifest_path = tmp_path / "external" / "SHA256SUMS"
    config = RuntimeConfig(
        dense_model_dir=model_dir,
        dense_vector_dir=vector_dir,
        dense_manifest_path=manifest_path,
    )

    assert config.resolve_dense_paths(submission_root) == (
        model_dir,
        vector_dir,
        manifest_path,
    )


def test_resolve_dense_paths_accepts_relative_and_absolute_strings(tmp_path: Path):
    submission_root = tmp_path / "release"
    submission_root.mkdir()
    _write_submission_layout(submission_root)
    absolute_model = tmp_path / "external" / "model"
    config = RuntimeConfig(
        dense_model_dir=str(absolute_model),
        dense_vector_dir="custom/vectors",
        dense_manifest_path="custom/SHA256SUMS",
    )

    assert config.resolve_dense_paths(submission_root) == (
        absolute_model,
        submission_root / "custom" / "vectors",
        submission_root / "custom" / "SHA256SUMS",
    )


def test_agent_initializes_with_string_dense_paths(
    fixture_catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    loaded_paths: list[tuple[Path, Path, Path]] = []

    def capture_loader(
        model_dir: Path, vector_dir: Path, manifest_path: Path
    ) -> NullDenseBackend:
        loaded_paths.append((model_dir, vector_dir, manifest_path))
        return NullDenseBackend("asset_missing")

    monkeypatch.setattr(agent_module, "load_dense_backend", capture_loader)
    config = RuntimeConfig(
        dense_model_dir="assets/model",
        dense_vector_dir="assets/product_vectors",
        dense_manifest_path="assets/SHA256SUMS",
    )

    agent = CompassCartAgent(fixture_catalog_path, config=config)

    assert agent.dense.status == "asset_missing"
    assert loaded_paths == [
        (
            agent_module.SUBMISSION_ROOT / "assets" / "model",
            agent_module.SUBMISSION_ROOT / "assets" / "product_vectors",
            agent_module.SUBMISSION_ROOT / "assets" / "SHA256SUMS",
        )
    ]


@pytest.mark.parametrize("missing", ["agent.py", "assets"])
def test_resolve_dense_paths_rejects_invalid_submission_layout(
    tmp_path: Path, missing: str
):
    submission_root = tmp_path / "release"
    submission_root.mkdir()
    _write_submission_layout(submission_root)
    target = submission_root / missing
    if target.is_dir():
        target.rmdir()
    else:
        target.unlink()

    with pytest.raises(ValueError, match="submission root"):
        RuntimeConfig().resolve_dense_paths(submission_root)


def test_agent_reports_layout_invalid_without_trying_cwd_assets(
    tmp_path: Path,
    fixture_catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_if_called(*_args: object) -> None:
        raise AssertionError("dense loader must not run for an invalid layout")

    monkeypatch.setattr(agent_module, "SUBMISSION_ROOT", tmp_path)
    monkeypatch.setattr(agent_module, "load_dense_backend", fail_if_called)

    agent = CompassCartAgent(fixture_catalog_path)
    agent.reset("layout", {})
    response = agent.respond("layout", "running shoes", turn=1, top_k=3)

    assert isinstance(agent.dense, NullDenseBackend)
    assert agent.dense.status == "layout_invalid"
    assert agent.traces.records[-1]["dense_status"] == "layout_invalid"
    assert set(response) == {"message", "ask_attribute", "recommendations", "usage"}


def test_rank_calibration_neutral_values_and_agent_wiring_are_neutral(
    fixture_catalog_path: Path,
):
    config = RuntimeConfig(
        rank_fusion_weight=0.10,
        rank_attribute_weight=0.0,
        rank_consensus_bonus=0.0,
        rank_boundary_bonus=0.0,
        adaptive_browsing_mmr=False,
        dense_rescue_only=True,
    )
    agent = CompassCartAgent(fixture_catalog_path, config=config)

    assert config.rank_attribute_weight == 0.0
    assert config.rank_consensus_bonus == 0.0
    assert config.rank_boundary_bonus == 0.0
    assert config.rank_fusion_weight == 0.10
    assert config.adaptive_browsing_mmr is False
    assert config.dense_rescue_only is True
    assert agent.ranker.fusion_weight == 0.10
    assert agent.ranker.attribute_weight == 0.0
    assert agent.ranker.consensus_bonus == 0.0
    assert agent.ranker.boundary_bonus == 0.0
    assert agent.ranker.adaptive_browsing_mmr is False
    assert agent.retriever.dense_rescue_only is True


@pytest.mark.parametrize(
    ("field", "values"),
    (
        ("rank_attribute_weight", (0.0, 0.05, 0.10)),
        ("rank_consensus_bonus", (0.0, 0.025, 0.05)),
        ("rank_boundary_bonus", (0.0, 0.025)),
        ("rank_fusion_weight", (0.10, 0.15, 0.25)),
    ),
)
def test_rank_calibration_accepts_only_predeclared_numeric_values(field, values):
    for value in values:
        config = RuntimeConfig(**{field: value})
        assert getattr(config, field) == value


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("rank_attribute_weight", True),
        ("rank_attribute_weight", -0.05),
        ("rank_attribute_weight", 0.025),
        ("rank_attribute_weight", float("nan")),
        ("rank_consensus_bonus", True),
        ("rank_consensus_bonus", -0.025),
        ("rank_consensus_bonus", 0.01),
        ("rank_consensus_bonus", float("inf")),
        ("rank_boundary_bonus", True),
        ("rank_boundary_bonus", -0.025),
        ("rank_boundary_bonus", 0.05),
        ("rank_boundary_bonus", float("-inf")),
        ("rank_fusion_weight", True),
        ("rank_fusion_weight", -0.10),
        ("rank_fusion_weight", 0.0),
        ("rank_fusion_weight", 0.20),
        ("rank_fusion_weight", float("nan")),
    ),
)
def test_rank_calibration_rejects_invalid_numeric_values(field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        RuntimeConfig(**{field: value})


@pytest.mark.parametrize("value", (0, 1, "true", None))
def test_adaptive_browsing_mmr_requires_a_real_bool(value):
    with pytest.raises(TypeError, match="adaptive_browsing_mmr"):
        RuntimeConfig(adaptive_browsing_mmr=value)


@pytest.mark.parametrize("value", (0, 1, "true", None))
def test_dense_rescue_only_requires_a_real_bool(value):
    with pytest.raises(TypeError, match="dense_rescue_only"):
        RuntimeConfig(dense_rescue_only=value)


@pytest.mark.parametrize(
    "value", (True, 0.0, 1.0, -0.85, float("nan"), float("inf"))
)
def test_runtime_config_keeps_mmr_lambda_fixed_at_point_85(value):
    with pytest.raises((TypeError, ValueError), match="mmr_lambda"):
        RuntimeConfig(mmr_lambda=value)


def test_rank_source_budget_is_exact_for_every_allowed_combination():
    for fusion_weight in (0.10, 0.15, 0.25):
        for attribute_weight in (0.0, 0.05, 0.10):
            config = RuntimeConfig(
                rank_fusion_weight=fusion_weight,
                rank_attribute_weight=attribute_weight,
            )
            source_weight = (
                0.40 - config.rank_fusion_weight - config.rank_attribute_weight
            ) / 2.0

            assert (
                2 * source_weight
                + config.rank_fusion_weight
                + config.rank_attribute_weight
                == pytest.approx(0.40)
            )
