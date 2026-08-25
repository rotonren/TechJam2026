from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tools.package_submission import build_submission

REQUIRED_ENTRIES = {
    "agent.py",
    "requirements.txt",
    "README.md",
    "DATA_ATTRIBUTION.md",
    "licenses/all-MiniLM-L6-v2-APACHE-2.0.txt",
    "assets/SHA256SUMS",
    "assets/model/model.int8.onnx",
    "assets/model/tokenizer.json",
    "assets/product_vectors/product_ids.npy",
    "assets/product_vectors/scales.npy",
    "assets/product_vectors/vectors.int8.npy",
    "reports/final/devpost.md",
    "reports/final/final-results.json",
    "reports/final/release-checklist.md",
    "src/compasscart/agent.py",
}


@pytest.fixture()
def submission_zip(tmp_path: Path) -> Path:
    destination = tmp_path / "compasscart-submission.zip"
    build_submission(destination)
    return destination


def test_submission_contains_runtime_allowlist(submission_zip: Path):
    with zipfile.ZipFile(submission_zip) as archive:
        names = set(archive.namelist())

    assert REQUIRED_ENTRIES <= names
    assert not any(
        forbidden in name.lower()
        for name in names
        for forbidden in (
            ".env",
            "__pycache__",
            ".pyc",
            "public_set",
            "evaluator/",
            "organizer/",
            "reports/experiments/",
            "model.onnx",
        )
    )


def test_submission_archive_is_reproducible(tmp_path: Path):
    first = build_submission(tmp_path / "first.zip")
    second = build_submission(tmp_path / "second.zip")

    assert first.sha256 == second.sha256
    assert first.file_count == second.file_count
    assert first.size_bytes == second.size_bytes
