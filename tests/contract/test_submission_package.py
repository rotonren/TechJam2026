from __future__ import annotations

import json
import os
import subprocess
import sys
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


def test_package_module_imports_from_repository_root_without_pythonpath():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "-c", "import tools.package_submission"],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ModuleNotFoundError" not in completed.stderr


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


def test_extracted_submission_loads_dense_assets_outside_cwd(
    submission_zip: Path,
    fixture_catalog_path: Path,
    tmp_path: Path,
):
    release_root = tmp_path / "release"
    outside_cwd = tmp_path / "outside"
    release_root.mkdir()
    outside_cwd.mkdir()
    with zipfile.ZipFile(submission_zip) as archive:
        archive.extractall(release_root)

    environment = os.environ.copy()
    environment.pop("COMPASSCART_DISABLE_DENSE", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(release_root / "src"), str(release_root))
    )
    script = """
import json
import sys
import agent as agent_module
import compasscart.agent as compasscart_agent_module
from agent import Agent

agent = Agent(sys.argv[1])
agent.reset("package", {})
response = agent.respond("package", "running shoes", turn=1, top_k=3)
print(json.dumps({
    "agent_file": agent_module.__file__,
    "compasscart_agent_file": compasscart_agent_module.__file__,
    "dense_available": agent.dense.available,
    "dense_status": agent.dense.status,
    "trace_dense_status": agent.traces.records[-1]["dense_status"],
    "response_keys": sorted(response),
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(fixture_catalog_path.resolve())],
        cwd=outside_cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )

    payload = json.loads(completed.stdout)
    assert {
        key: payload[key]
        for key in (
            "dense_available",
            "dense_status",
            "trace_dense_status",
            "response_keys",
        )
    } == {
        "dense_available": True,
        "dense_status": "available",
        "trace_dense_status": "available",
        "response_keys": ["ask_attribute", "message", "recommendations", "usage"],
    }
    assert Path(payload["agent_file"]).resolve() == (release_root / "agent.py").resolve()
    assert Path(payload["compasscart_agent_file"]).resolve() == (
        release_root / "src" / "compasscart" / "agent.py"
    ).resolve()
