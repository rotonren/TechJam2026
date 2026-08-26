from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.release_audit import (
    load_release_results,
    sha256_file,
    verify_release_fingerprints,
)


def test_sha256_file_streams_stable_digest(tmp_path: Path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"compasscart\n" * 100)

    assert sha256_file(path) == (
        "9d9e0dc28564fb3e82f3c212b5521d1e90d5dc8003833cfbafac8c3c7cdd852e"
    )


def test_load_release_results_requires_current_schema(tmp_path: Path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema version 2"):
        load_release_results(path)


def test_release_fingerprints_reject_changed_inputs(tmp_path: Path):
    data = tmp_path / "data"
    evaluator = tmp_path / "evaluator"
    data.mkdir()
    evaluator.mkdir()
    catalog = data / "catalog.jsonl"
    public = data / "public_set.jsonl"
    local_evaluator = evaluator / "local_evaluator.py"
    catalog.write_text("catalog\n", encoding="utf-8")
    public.write_text("public\n", encoding="utf-8")
    local_evaluator.write_text("# evaluator\n", encoding="utf-8")
    results = {
        "reproducibility": {
            "catalog_jsonl_sha256": sha256_file(catalog),
            "public_set_sha256": sha256_file(public),
            "evaluator_sha256": sha256_file(local_evaluator),
        }
    }

    assert verify_release_fingerprints(
        results, root=tmp_path, catalog_path=catalog
    )["catalog_jsonl"] == sha256_file(catalog)

    public.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="public_set fingerprint mismatch"):
        verify_release_fingerprints(results, root=tmp_path, catalog_path=catalog)
