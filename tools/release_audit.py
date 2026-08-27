from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from agent import Agent
from tools.package_submission import PROJECT_ROOT, build_submission
from tools.run_cv import _config_hash

RESULTS_PATH = PROJECT_ROOT / "reports" / "final" / "final-results.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical_text(path: Path) -> str:
    """Hash UTF-8 text after normalizing checkout-specific line endings."""
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def load_release_results(path: Path = RESULTS_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("release results must use schema version 2")
    return payload


def verify_release_fingerprints(
    results: dict[str, Any],
    *,
    root: Path,
    catalog_path: Path,
) -> dict[str, str]:
    expected = results["reproducibility"]
    binary_inputs = {
        "catalog_jsonl": (
            catalog_path,
            str(expected["catalog_jsonl_sha256"]),
        ),
    }
    text_inputs = {
        "public_set": (
            root / "data" / "public_set.jsonl",
            str(expected["public_set_canonical_sha256"]),
        ),
        "evaluator": (
            root / "evaluator" / "local_evaluator.py",
            str(expected["evaluator_canonical_sha256"]),
        ),
    }
    actual: dict[str, str] = {}
    for name, (path, wanted) in binary_inputs.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != wanted:
            raise ValueError(f"{name} fingerprint mismatch")
        actual[name] = digest
    for name, (path, wanted) in text_inputs.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_canonical_text(path)
        if digest != wanted:
            raise ValueError(f"{name} fingerprint mismatch")
        actual[name] = digest
    return actual


def verify_runtime_versions(results: dict[str, Any]) -> dict[str, str]:
    expected = results["reproducibility"]
    packages = {
        "numpy": "numpy",
        "onnxruntime": "onnxruntime",
        "tokenizers": "tokenizers",
    }
    actual = {name: version(distribution) for name, distribution in packages.items()}
    for name, installed in actual.items():
        if installed != str(expected[name]):
            raise ValueError(
                f"{name} version mismatch: expected {expected[name]}, got {installed}"
            )
    return actual


def audit_release(
    catalog_path: Path,
    *,
    destination: Path,
    root: Path = PROJECT_ROOT,
    require_dense: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    catalog_path = catalog_path.resolve()
    results = load_release_results(root / "reports" / "final" / "final-results.json")
    fingerprints = verify_release_fingerprints(
        results, root=root, catalog_path=catalog_path
    )
    versions = verify_runtime_versions(results)

    agent = Agent(catalog_path)
    config_hash = _config_hash(agent)
    expected_config_hash = str(results["candidate"]["config_hash"])
    if config_hash != expected_config_hash:
        raise ValueError(
            f"runtime config fingerprint mismatch: expected {expected_config_hash}, "
            f"got {config_hash}"
        )
    dense_available = bool(agent.dense.available)
    if require_dense and not dense_available:
        raise RuntimeError("dense runtime is required but unavailable")
    agent.reset("release-audit", {"preference_tags": []})
    response = agent.respond(
        "release-audit", "I need blue running shoes", turn=1, top_k=10
    )
    recommendations = response.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        raise RuntimeError("release smoke returned no recommendations")

    package = build_submission(destination, root=root)
    return {
        "candidate_status": results["candidate"]["status"],
        "proposed_tag": results["candidate"]["proposed_tag"],
        "runtime_commit": results["candidate"]["runtime_commit"],
        "config_hash": config_hash,
        "catalog_count": len(agent.catalog.valid_ids),
        "dense_available": dense_available,
        "runtime_versions": versions,
        "fingerprints": fingerprints,
        "smoke_recommendation_count": len(recommendations),
        "package": {
            "path": str(package.destination),
            "file_count": package.file_count,
            "size_bytes": package.size_bytes,
            "sha256": package.sha256,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/compasscart-submission.zip"),
    )
    parser.add_argument(
        "--allow-lexical",
        action="store_true",
        help="Allow the audit to pass when the dense runtime is unavailable.",
    )
    args = parser.parse_args()
    report = audit_release(
        args.catalog,
        destination=args.output,
        require_dense=not args.allow_lexical,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
