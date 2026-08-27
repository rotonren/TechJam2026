from __future__ import annotations

import hashlib
import importlib
import json
import zipfile
from pathlib import Path

import pytest

from tools.package_submission import ASSET_FILES, ZIP_TIMESTAMP, PackageResult


def _package_source():
    return importlib.import_module("tools.package_source")


def _write(root: Path, relative: str, content: str | bytes = "fixture\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)
    return path


@pytest.fixture()
def source_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    for relative in (
        ".gitattributes",
        ".gitignore",
        "agent.py",
        "requirements.txt",
        "requirements-assets.txt",
        "requirements-dev.txt",
        "README.md",
        "DATA_ATTRIBUTION.md",
        "MODEL_ATTRIBUTION.md",
        "licenses/all-MiniLM-L6-v2-APACHE-2.0.txt",
        "src/compasscart/__init__.py",
        "src/compasscart/agent.py",
        "tests/__init__.py",
        "tests/unit/test_package_source.py",
        "tests/fixtures/catalog.jsonl",
        "tools/__init__.py",
        "tools/package_source.py",
        "tools/run_agent.py",
        "tools/download_kit.ps1",
        "docs/agent_api_contract.json",
        "docs/baseline_results.json",
        "docs/compasscart-operation-guide.docx",
        "docs/competition_specification.md",
        "docs/evaluation_config.json",
        "docs/submission_rules.md",
        "docs/superpowers/plans/source-plan.md",
        "docs/superpowers/specs/source-design.md",
        "reports/final/architecture.md",
    ):
        _write(root, relative)

    _write(
        root,
        "reports/final/final-results.json",
        json.dumps(
            {
                "technical_score": 0.7,
                "scenario_metrics": {"intent_override": {"hit_rate_at_10": 0.8}},
                "evaluated_sessions": 200,
            }
        ),
    )
    _write(
        root,
        "reports/final/score-results-safe-2026-08-27.json",
        json.dumps({"technical_score": 0.71, "sample_count": 200}),
    )
    _write(root, "reports/final/score-results-safe-2026-08-27.md")

    manifest_lines = []
    for index, relative in enumerate(ASSET_FILES[1:], start=1):
        payload = f"asset-{index}".encode()
        _write(root, relative, payload)
        asset_relative = relative.removeprefix("assets/")
        manifest_lines.append(f"{hashlib.sha256(payload).hexdigest()}  {asset_relative}")
    _write(root, ASSET_FILES[0], "\n".join(manifest_lines) + "\n")

    for relative in (
        ".git/config",
        ".venv/secret.txt",
        "src/compasscart/__pycache__/agent.pyc",
        "tests/.pytest_cache/state",
        "tools/__pycache__/run_agent.pyc",
        "var/raw-score.json",
        "dist/old.zip",
        "evaluator/local_evaluator.py",
        "data/public_set.jsonl",
        "data/catalog.jsonl",
        "organizer/data/catalog.jsonl",
        "reports/experiments/cv-raw.json",
        "reports/final/raw-session-score.json",
        "reports/final/raw-session-score.md",
        "results.json",
    ):
        _write(root, relative)

    _write(root, "docs/compasscart-agent-architecture-analysis.docx", b"protected")
    _write(
        root,
        "reports/final/score-results-b641ff9-2026-08-26.json",
        json.dumps({"sessions": [{"sample_id": "public_0001"}]}),
    )
    _write(root, "reports/final/score-results-b641ff9-2026-08-26.md", "protected")
    return root


def test_source_bundle_contains_allowlisted_project_files_only(
    source_root: Path, tmp_path: Path
) -> None:
    package_source = _package_source()
    destination = tmp_path / "output" / "source.zip"

    result = package_source.build_source(destination, root=source_root)

    with zipfile.ZipFile(destination) as archive:
        names = set(archive.namelist())
    assert {
        "agent.py",
        "requirements-dev.txt",
        "README.md",
        "DATA_ATTRIBUTION.md",
        "licenses/all-MiniLM-L6-v2-APACHE-2.0.txt",
        "src/compasscart/agent.py",
        "tests/unit/test_package_source.py",
        "tests/fixtures/catalog.jsonl",
        "tools/package_source.py",
        "tools/download_kit.ps1",
        "docs/competition_specification.md",
        "docs/submission_rules.md",
        "docs/superpowers/plans/source-plan.md",
        "docs/superpowers/specs/source-design.md",
        "reports/final/architecture.md",
        "reports/final/final-results.json",
        "reports/final/score-results-safe-2026-08-27.json",
        "reports/final/score-results-safe-2026-08-27.md",
        "assets/model/model.int8.onnx",
    } <= names
    assert {
        ".git/config",
        ".venv/secret.txt",
        "src/compasscart/__pycache__/agent.pyc",
        "tests/.pytest_cache/state",
        "tools/__pycache__/run_agent.pyc",
        "var/raw-score.json",
        "dist/old.zip",
        "evaluator/local_evaluator.py",
        "data/public_set.jsonl",
        "data/catalog.jsonl",
        "organizer/data/catalog.jsonl",
        "reports/experiments/cv-raw.json",
        "reports/final/raw-session-score.json",
        "reports/final/raw-session-score.md",
        "results.json",
        "docs/compasscart-agent-architecture-analysis.docx",
        "reports/final/score-results-b641ff9-2026-08-26.json",
        "reports/final/score-results-b641ff9-2026-08-26.md",
    }.isdisjoint(names)
    assert result.destination == destination
    assert result.file_count == len(names)
    assert result.size_bytes == destination.stat().st_size
    assert result.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "forbidden_key",
    (
        "sessions",
        "sample_id",
        "sampleIds",
        "recommendations",
        "target_id",
        "targetASINs",
        "intent",
        "intent_data",
    ),
)
def test_packaged_score_reports_reject_nested_sensitive_keys(
    source_root: Path, tmp_path: Path, forbidden_key: str
) -> None:
    package_source = _package_source()
    _write(
        source_root,
        "reports/final/score-results-safe-2026-08-27.json",
        json.dumps({"aggregate": {"nested": {forbidden_key: []}}}),
    )
    destination = tmp_path / f"rejected-{forbidden_key}.zip"

    with pytest.raises(ValueError, match="forbidden report key"):
        package_source.build_source(destination, root=source_root)

    assert not destination.exists()


def test_source_bundle_rejects_secrets_in_development_files(
    source_root: Path, tmp_path: Path
) -> None:
    package_source = _package_source()
    fake_token = "a" * 16
    _write(source_root, "tools/leak.ps1", f'access_token = "{fake_token}"\n')
    destination = tmp_path / "secret.zip"

    with pytest.raises(ValueError, match="possible embedded secret"):
        package_source.build_source(destination, root=source_root)

    assert not destination.exists()


def test_source_bundle_rejects_allowlist_path_traversal(
    source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_source = _package_source()
    _write(source_root.parent, "escape.py")
    monkeypatch.setattr(
        package_source,
        "ROOT_FILES",
        (*package_source.ROOT_FILES, "../escape.py"),
    )

    with pytest.raises(ValueError, match="unsafe source path"):
        package_source.build_source(tmp_path / "traversal.zip", root=source_root)


def test_source_bundle_rejects_symlinks(
    source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_source = _package_source()
    link = source_root / "src" / "compasscart" / "linked.py"
    _write(source_root, "src/compasscart/linked.py")
    real_is_link = package_source._is_link
    monkeypatch.setattr(
        package_source,
        "_is_link",
        lambda path: path == link or real_is_link(path),
    )

    with pytest.raises(ValueError, match="symlink"):
        package_source.build_source(tmp_path / "symlink.zip", root=source_root)


def test_source_bundle_preserves_an_existing_destination(
    source_root: Path, tmp_path: Path
) -> None:
    package_source = _package_source()
    destination = _write(tmp_path, "published/source.zip", b"keep-me")

    with pytest.raises(FileExistsError):
        package_source.build_source(destination, root=source_root)

    assert destination.read_bytes() == b"keep-me"
    assert list(destination.parent.iterdir()) == [destination]


def test_source_bundle_has_reproducible_bytes_and_zip_metadata(
    source_root: Path, tmp_path: Path
) -> None:
    package_source = _package_source()
    first_path = tmp_path / "first.zip"
    second_path = tmp_path / "second.zip"

    first = package_source.build_source(first_path, root=source_root)
    second = package_source.build_source(second_path, root=source_root)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.sha256 == second.sha256
    with zipfile.ZipFile(first_path) as archive:
        entries = archive.infolist()
    assert [entry.filename for entry in entries] == sorted(
        entry.filename for entry in entries
    )
    assert all(entry.date_time == ZIP_TIMESTAMP for entry in entries)
    assert all(entry.external_attr >> 16 == 0o100644 for entry in entries)


def test_source_package_cli_uses_default_only_when_output_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_source = _package_source()
    destinations: list[Path | None] = []
    result = PackageResult(tmp_path / "result.zip", 1, 2, "0" * 64)

    def fake_build_source(destination=None):
        destinations.append(destination)
        return result

    monkeypatch.setattr(package_source, "build_source", fake_build_source)

    package_source.main([])
    package_source.main(["--output", str(tmp_path / "explicit.zip")])

    assert destinations == [None, tmp_path / "explicit.zip"]
