"""Deterministic fingerprints for CompassCart runtime source and configuration."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Final

from compasscart.config import RuntimeConfig

_CHUNK_SIZE: Final = 1024 * 1024


def _repo_root(root: str | Path | None = None) -> Path:
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]


def runtime_hash(root: str | Path | None = None) -> str:
    """Hash the production entry point and package sources in a stable order."""
    repository = _repo_root(root)
    paths = [repository / "agent.py", *sorted((repository / "src" / "compasscart").rglob("*.py"))]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise ValueError(f"runtime source is missing: {path.relative_to(repository).as_posix()}")
        digest.update(path.relative_to(repository).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            while chunk := source.read(_CHUNK_SIZE):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _json_value(value: object) -> object:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("configuration numbers must be finite")
        return value
    if isinstance(value, Path):
        if value.is_absolute():
            raise ValueError("configuration paths must be relative")
        return value.as_posix()
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("configuration mappings require string keys")
        return {key: _json_value(value[key]) for key in sorted(value)}
    raise TypeError(f"configuration value is not JSON safe: {type(value).__name__}")


def canonical_config(config: RuntimeConfig | None) -> object:
    """Return the canonical JSON-safe representation used by all runtime tools."""
    if config is None:
        return None
    if not isinstance(config, RuntimeConfig) or not is_dataclass(config):
        raise TypeError("runtime configuration must be RuntimeConfig")
    return {item.name: _json_value(getattr(config, item.name)) for item in fields(config)}


def config_hash(config: RuntimeConfig | None) -> str:
    if config is None:
        return hashlib.sha256(repr(None).encode("utf-8")).hexdigest()
    payload = json.dumps(canonical_config(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
