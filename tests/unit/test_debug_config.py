from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from compasscart.config import RuntimeConfig
from compasscart_debug.config import DebugConfig, RuntimeIdentity, RuntimeSetupError


@pytest.mark.parametrize("token", ["change-me", "x" * 42])
def test_from_env_rejects_placeholder_or_short_access_tokens(token: str) -> None:
    with pytest.raises(ValueError, match="access token"):
        DebugConfig.from_env({"COMPASSCART_DEBUG_TOKEN": token})


def test_from_env_preserves_explicit_catalog_and_asset_paths(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    assets = tmp_path / "assets"

    config = DebugConfig.from_env(
        {
            "COMPASSCART_DEBUG_TOKEN": "t" * 43,
            "COMPASSCART_CATALOG_PATH": str(catalog),
            "COMPASSCART_ASSET_ROOT": str(assets),
        }
    )

    assert config.catalog_path == catalog
    assert config.asset_root == assets


def test_runtime_identity_ignores_absolute_dense_paths(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text('{"parent_asin":"A"}\n', encoding="utf-8")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text("assets", encoding="utf-8")
    first = RuntimeConfig(
        dense_model_dir=tmp_path / "one" / "model",
        dense_vector_dir=tmp_path / "one" / "vectors",
        dense_manifest_path=tmp_path / "one" / "SHA256SUMS",
    )
    second = replace(
        first,
        dense_model_dir=tmp_path / "two" / "model",
        dense_vector_dir=tmp_path / "two" / "vectors",
        dense_manifest_path=tmp_path / "two" / "SHA256SUMS",
    )

    left = RuntimeIdentity.build("1.0", catalog, first, manifest, dense_disabled=False)
    right = RuntimeIdentity.build(
        "1.0", catalog, second, manifest, dense_disabled=False
    )

    assert left.config_sha256 == right.config_sha256
    assert left.catalog_sha256 == right.catalog_sha256
    assert left.assets_sha256 == right.assets_sha256


def test_resolve_runtime_paths_reads_current_later_and_builds_release_paths(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    config = DebugConfig.from_env(
        {
            "COMPASSCART_DEBUG_TOKEN": "t" * 43,
            "COMPASSCART_RUNTIME_ROOT": str(runtime_root),
        }
    )
    (runtime_root / "CURRENT").write_text("a" * 64, encoding="utf-8")

    paths = config.resolve_runtime_paths()

    assert (
        paths.catalog_path == runtime_root / "releases" / ("a" * 64) / "catalog.jsonl"
    )
    assert paths.asset_root == runtime_root / "releases" / ("a" * 64) / "assets"


@pytest.mark.parametrize("pointer", ["A" * 64, "../release", "a" * 63])
def test_resolve_runtime_paths_rejects_invalid_release_pointer(
    tmp_path: Path, pointer: str
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "CURRENT").write_text(pointer, encoding="utf-8")
    config = DebugConfig.from_env(
        {
            "COMPASSCART_DEBUG_TOKEN": "t" * 43,
            "COMPASSCART_RUNTIME_ROOT": str(runtime_root),
        }
    )

    with pytest.raises(RuntimeSetupError) as error:
        config.resolve_runtime_paths()

    assert str(runtime_root) not in str(error.value)


def test_resolve_runtime_paths_fails_safely_for_missing_pointer(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    config = DebugConfig.from_env(
        {
            "COMPASSCART_DEBUG_TOKEN": "t" * 43,
            "COMPASSCART_RUNTIME_ROOT": str(runtime_root),
        }
    )

    with pytest.raises(FileNotFoundError) as error:
        config.resolve_runtime_paths()

    assert str(runtime_root) not in str(error.value)
