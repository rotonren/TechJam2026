from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from compasscart.config import RuntimeConfig
from compasscart_debug import config as debug_config
from compasscart_debug.config import DebugConfig, RuntimeIdentity, RuntimeSetupError
from compasscart_debug.errors import TurnLimitError, ValidationError


@pytest.mark.parametrize(
    "token", ["change-me", "CHANGE-ME", "replace-me", "PASSWORD", "x" * 42]
)
def test_from_env_rejects_placeholder_or_short_access_tokens(token: str) -> None:
    with pytest.raises(ValueError, match="43 characters"):
        DebugConfig.from_env({"COMPASSCART_DEBUG_TOKEN": token})


def test_from_env_uses_deployment_env_names_and_fixed_static_root(
    tmp_path: Path,
) -> None:
    config = DebugConfig.from_env(
        {
            "COMPASSCART_DEBUG_TOKEN": "t" * 43,
            "COMPASSCART_DEBUG_DATABASE": str(tmp_path / "debug.sqlite3"),
            "HOST": "0.0.0.0",
            "PORT": "9123",
            "COMPASSCART_DEBUG_STATIC_ROOT": str(tmp_path / "untrusted-static"),
        }
    )

    assert config.database_path == tmp_path / "debug.sqlite3"
    assert config.host == "0.0.0.0"
    assert config.port == 9123
    assert config.static_root == Path(debug_config.__file__).resolve().parent / "static"


def test_from_env_uses_fixed_deployment_defaults() -> None:
    config = DebugConfig.from_env({"COMPASSCART_DEBUG_TOKEN": "t" * 43})

    assert config.database_path == Path("var/debug/compasscart-debug.sqlite3")
    assert config.host == "127.0.0.1"
    assert config.port == 8765
    assert config.static_root.name == "static"
    assert config.static_root.parent.name == "compasscart_debug"


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


@pytest.mark.parametrize("manifest", [None, Path("does-not-exist")])
def test_runtime_identity_allows_missing_optional_manifest(
    tmp_path: Path, manifest: Path | None
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text('{"parent_asin":"A"}\n', encoding="utf-8")
    if manifest is not None:
        manifest = tmp_path / manifest

    identity = RuntimeIdentity.build(
        "1.0", catalog, RuntimeConfig(), manifest, dense_disabled=False
    )

    assert identity.assets_sha256 is None


def test_runtime_identity_skips_assets_when_dense_is_disabled(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    manifest = tmp_path / "SHA256SUMS"
    catalog.write_text('{"parent_asin":"A"}\n', encoding="utf-8")
    manifest.write_text("assets", encoding="utf-8")

    identity = RuntimeIdentity.build(
        "1.0", catalog, RuntimeConfig(), manifest, dense_disabled=True
    )

    assert identity.assets_sha256 is None


def test_stable_error_payload_sorts_fields_and_turn_limit_conflicts() -> None:
    payload = ValidationError({"z": "last", "a": "first"}).to_payload()

    assert payload == {
        "error": {
            "code": "validation_failed",
            "message": "Request validation failed.",
            "retryable": False,
            "field_errors": {"a": "first", "z": "last"},
        }
    }
    assert TurnLimitError().http_status == 409


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
