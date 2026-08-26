from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "reports" / "final" / "final-results.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_metadata_identifies_review_candidate_and_runtime():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    candidate = results["candidate"]

    assert results["schema_version"] == 2
    assert candidate["status"] == "owner_review"
    assert candidate["proposed_tag"] == "compasscart-v3-candidate"
    assert candidate["tag"] is None
    assert candidate["runtime_commit"] == (
        "54b2a626878a81c997f5299b39193f9741a120f7"
    )


def test_release_fingerprints_match_tracked_evidence():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    fingerprints = results["reproducibility"]

    assert fingerprints["public_set_sha256"] == _sha256(
        PROJECT_ROOT / "data" / "public_set.jsonl"
    )
    assert fingerprints["evaluator_sha256"] == _sha256(
        PROJECT_ROOT / "evaluator" / "local_evaluator.py"
    )


def test_primary_release_documents_use_canonical_metrics():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    public = results["official_public_evaluation"]
    development = results["development_cv"]
    documents = {
        name: (PROJECT_ROOT / path).read_text(encoding="utf-8")
        for name, path in {
            "readme": "README.md",
            "architecture": "reports/final/architecture.md",
            "devpost": "reports/final/devpost.md",
            "demo": "reports/final/demo-script.md",
            "checklist": "reports/final/release-checklist.md",
        }.items()
    }

    assert f'{public["recommended_technical_score"]:.6f}' in documents["readme"]
    assert f'{public["recommended_technical_score"]:.6f}' in documents["devpost"]
    assert f'{public["recommended_technical_score"]:.6f}' in documents["demo"]
    assert f'{development["selection_score"]:.6f}' in documents["architecture"]
    assert "54b2a62" in documents["readme"]
    assert "54b2a62" in documents["checklist"]
    assert "compasscart-v3-candidate" in documents["checklist"]

    for name, text in documents.items():
        assert "0.518309" not in text, f"stale v2 public score in {name}"


def test_public_evaluation_declares_dense_mode_and_exact_environment():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    assert results["official_public_evaluation"]["mode"] == "dense"
    assert results["reproducibility"]["dense_available"] is True
    assert results["reproducibility"]["catalog_rows"] == 50_000
    assert results["reproducibility"]["onnxruntime"] == "1.29.0"


def test_runtime_dependencies_match_recorded_release_environment():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    reproducibility = results["reproducibility"]
    requirements = {
        line.strip()
        for line in (PROJECT_ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert requirements == {
        f'numpy=={reproducibility["numpy"]}',
        f'onnxruntime=={reproducibility["onnxruntime"]}',
        f'tokenizers=={reproducibility["tokenizers"]}',
    }
