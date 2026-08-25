from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compasscart.config import RuntimeConfig

from .errors import RuntimeSetupError

_MIN_TOKEN_LENGTH = 43
_RUNTIME_ID = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class RuntimePaths:
    catalog_path: Path
    asset_root: Path


@dataclass(frozen=True)
class DebugConfig:
    catalog_path: Path = Path("data/catalog.jsonl")
    database_path: Path = Path("var/debug/compasscart-debug.sqlite3")
    static_root: Path = Path(__file__).resolve().parent / "static"
    asset_root: Path = Path("assets")
    runtime_root: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    access_token: str = ""
    max_body_bytes: int = 1_048_576
    max_import_bytes: int = 16_777_216
    command_queue_size: int = 32
    command_timeout_seconds: float = 180.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> DebugConfig:
        values = os.environ if env is None else env
        token = values.get("COMPASSCART_DEBUG_TOKEN", "")
        if (
            token.lower() in {"change-me", "replace-me", "password"}
            or len(token) < _MIN_TOKEN_LENGTH
        ):
            raise ValueError("COMPASSCART_DEBUG_TOKEN must be at least 43 characters.")
        runtime_root = values.get("COMPASSCART_RUNTIME_ROOT")
        return cls(
            catalog_path=Path(
                values.get("COMPASSCART_CATALOG_PATH", "data/catalog.jsonl")
            ),
            database_path=Path(
                values.get(
                    "COMPASSCART_DEBUG_DATABASE", "var/debug/compasscart-debug.sqlite3"
                )
            ),
            static_root=Path(__file__).resolve().parent / "static",
            asset_root=Path(values.get("COMPASSCART_ASSET_ROOT", "assets")),
            runtime_root=Path(runtime_root) if runtime_root else None,
            host=values.get("HOST", "127.0.0.1"),
            port=_positive_int(values, "PORT", 8765),
            access_token=token,
            max_body_bytes=_positive_int(
                values, "COMPASSCART_DEBUG_MAX_BODY_BYTES", 1_048_576
            ),
            max_import_bytes=_positive_int(
                values, "COMPASSCART_DEBUG_MAX_IMPORT_BYTES", 16_777_216
            ),
            command_queue_size=_positive_int(
                values, "COMPASSCART_DEBUG_COMMAND_QUEUE_SIZE", 32
            ),
            command_timeout_seconds=_positive_float(
                values, "COMPASSCART_DEBUG_COMMAND_TIMEOUT_SECONDS", 180.0
            ),
        )

    def resolve_runtime_paths(self) -> RuntimePaths:
        if self.runtime_root is None:
            return RuntimePaths(self.catalog_path, self.asset_root)
        pointer = self.runtime_root / "CURRENT"
        try:
            runtime_id = pointer.read_text(encoding="utf-8").strip()
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "Runtime release pointer is unavailable."
            ) from error
        except OSError as error:
            raise RuntimeSetupError(
                "Runtime release pointer is unavailable."
            ) from error
        if not _RUNTIME_ID.fullmatch(runtime_id):
            raise RuntimeSetupError("Runtime release pointer is invalid.")

        releases = (self.runtime_root / "releases").resolve()
        release = (releases / runtime_id).resolve()
        try:
            release.relative_to(releases)
        except ValueError as error:
            raise RuntimeSetupError("Runtime release is invalid.") from error
        return RuntimePaths(release / "catalog.jsonl", release / "assets")

    @staticmethod
    def runtime_config(paths: RuntimePaths) -> RuntimeConfig:
        return RuntimeConfig(
            dense_model_dir=paths.asset_root / "model",
            dense_vector_dir=paths.asset_root / "product_vectors",
            dense_manifest_path=paths.asset_root / "SHA256SUMS",
        )


@dataclass(frozen=True)
class RuntimeIdentity:
    agent_version: str
    catalog_sha256: str
    config_sha256: str
    assets_sha256: str | None

    @classmethod
    def build(
        cls,
        agent_version: str,
        catalog_path: Path,
        runtime_config: RuntimeConfig,
        manifest_path: Path | None,
        dense_disabled: bool,
    ) -> RuntimeIdentity:
        config_payload = {
            field.name: _canonical_config_value(
                field.name, getattr(runtime_config, field.name)
            )
            for field in dataclasses.fields(runtime_config)
        }
        config_payload["dense_disabled"] = bool(dense_disabled)
        return cls(
            agent_version=agent_version,
            catalog_sha256=_sha256_file(catalog_path),
            config_sha256=_sha256_json(config_payload),
            assets_sha256=None
            if dense_disabled
            else _optional_sha256_file(manifest_path),
        )


def _positive_int(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _positive_float(values: Mapping[str, str], name: str, default: float) -> float:
    raw = values.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number.") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_sha256_file(path: Path | None) -> str | None:
    return _sha256_file(path) if path is not None and path.is_file() else None


def _canonical_config_value(name: str, value: Any) -> Any:
    if isinstance(value, Path):
        if name.startswith("dense_") and value.is_absolute():
            return None
        return value.as_posix()
    if isinstance(value, tuple):
        return [_canonical_config_value(name, item) for item in value]
    return value


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
