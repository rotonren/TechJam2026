from __future__ import annotations

from pathlib import Path

import pytest


def test_runtime_hash_uses_sorted_relative_paths_and_file_bytes(tmp_path: Path) -> None:
    from tools.runtime_fingerprint import runtime_hash

    (tmp_path / "src" / "compasscart").mkdir(parents=True)
    (tmp_path / "agent.py").write_bytes(b"agent\r\n")
    (tmp_path / "src" / "compasscart" / "z.py").write_bytes(b"z")
    (tmp_path / "src" / "compasscart" / "a.py").write_bytes(b"a")

    first = runtime_hash(tmp_path)
    (tmp_path / "src" / "compasscart" / "z.py").write_bytes(b"changed")

    assert len(first) == 64
    assert first != runtime_hash(tmp_path)


def test_config_hash_canonicalizes_runtime_config_paths_and_tuples() -> None:
    from compasscart.config import RuntimeConfig
    from tools.runtime_fingerprint import config_hash

    assert config_hash(RuntimeConfig()) == config_hash(
        RuntimeConfig(dense_model_dir="assets/model", bm25_field_weights=list(RuntimeConfig().bm25_field_weights))  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("value", [object(), {"bad": {1, 2}}, float("nan"), True])
def test_config_hash_rejects_non_deterministic_or_non_json_safe_config(value: object) -> None:
    from tools.runtime_fingerprint import config_hash

    with pytest.raises((TypeError, ValueError)):
        config_hash(value)
