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
