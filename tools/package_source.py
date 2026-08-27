from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from tools.package_submission import (
    ASSET_FILES,
    PROJECT_ROOT,
    SECRET_PATTERN,
    TEXT_SUFFIXES,
    ZIP_TIMESTAMP,
    PackageResult,
    _scan_for_secrets,
    _verify_assets,
)

DEFAULT_OUTPUT = Path("dist/compasscart-source.zip")

# This is deliberately independent from the competition-submission allowlist.
ROOT_FILES = (
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
)
DOCUMENT_FILES = (
    "docs/agent_api_contract.json",
    "docs/baseline_results.json",
    "docs/compasscart-operation-guide.docx",
    "docs/competition_specification.md",
    "docs/evaluation_config.json",
    "docs/submission_rules.md",
)
TREE_ALLOWLIST = (
    ("src/compasscart", frozenset({".py"})),
    (
        "tests",
        frozenset(
            {".css", ".html", ".js", ".json", ".jsonl", ".md", ".mjs", ".py", ".txt"}
        ),
    ),
    ("tools", frozenset({".json", ".md", ".ps1", ".py", ".sh", ".txt"})),
    ("docs/superpowers/plans", frozenset({".md"})),
    ("docs/superpowers/specs", frozenset({".md"})),
)
PROTECTED_FILES = frozenset(
    {
        "docs/compasscart-agent-architecture-analysis.docx",
        "reports/final/score-results-b641ff9-2026-08-26.json",
        "reports/final/score-results-b641ff9-2026-08-26.md",
    }
)
REPORT_MARKDOWN_NAMES = frozenset(
    {
        "ablation-template.md",
        "architecture.md",
        "balanced-hardening-foundation-results.md",
        "balanced-score-hardening-results.md",
        "demo-script.md",
        "devpost.md",
        "release-checklist.md",
        "score-optimization-results.md",
    }
)
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__", "node_modules"}
)
SOURCE_TEXT_SUFFIXES = frozenset(
    {
        "",
        ".cfg",
        ".css",
        ".html",
        ".ini",
        ".js",
        ".jsonl",
        ".mjs",
        ".ps1",
        ".rst",
        ".sh",
        ".toml",
        ".yaml",
        ".yml",
        *TEXT_SUFFIXES,
    }
)
FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "groundtruth",
        "groundtruthids",
        "intent",
        "intentdata",
        "intents",
        "intenttext",
        "rawintent",
        "recommendation",
        "recommendations",
        "recommendedids",
        "sampleid",
        "sampleids",
        "session",
        "sessionid",
        "sessionids",
        "sessions",
        "targetasin",
        "targetasins",
        "targetid",
        "targetids",
        "targetproductid",
        "targetproductids",
        "userintent",
    }
)
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:$")


def _validate_relative_path(relative: str) -> PurePosixPath:
    normalized = relative.replace("\\", "/")
    raw_parts = normalized.split("/")
    if (
        not normalized
        or any(part in {"", ".", ".."} for part in raw_parts)
        or _WINDOWS_DRIVE.fullmatch(raw_parts[0])
    ):
        raise ValueError(f"unsafe source path: {relative}")
    path = PurePosixPath(normalized)
    if path.is_absolute():
        raise ValueError(f"unsafe source path: {relative}")
    return path


def _is_link(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (is_junction is not None and is_junction())


def _resolve_allowed_path(root: Path, relative: str, *, directory: bool) -> Path:
    relative_path = _validate_relative_path(relative)
    path = root
    for part in relative_path.parts:
        path /= part
        if _is_link(path):
            raise ValueError(f"symlink is not allowed in source bundle: {relative}")
    expected = path.is_dir() if directory else path.is_file()
    if not expected:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"source {kind} missing: {relative}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"source path escapes root: {relative}") from error
    return path


def _tree_files(root: Path, relative: str, suffixes: frozenset[str]) -> list[Path]:
    directory = _resolve_allowed_path(root, relative, directory=True)
    files: list[Path] = []

    def visit(current: Path) -> None:
        for child in sorted(current.iterdir(), key=lambda path: path.name):
            child_relative = child.relative_to(root).as_posix()
            if _is_link(child):
                raise ValueError(
                    f"symlink is not allowed in source bundle: {child_relative}"
                )
            if child.is_dir():
                if child.name not in EXCLUDED_DIRECTORY_NAMES:
                    visit(child)
                continue
            if child.is_file() and child.suffix.lower() in suffixes:
                files.append(child)

    visit(directory)
    return files


def _report_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _validate_report_value(value: Any, report: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _report_key(str(key)) in FORBIDDEN_REPORT_KEYS:
                raise ValueError(f"forbidden report key in {report}: {key}")
            _validate_report_value(nested, report)
    elif isinstance(value, list):
        for nested in value:
            _validate_report_value(nested, report)


def _validate_report(path: Path, root: Path) -> None:
    relative = path.relative_to(root).as_posix()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid aggregate report JSON: {relative}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"aggregate report must be an object: {relative}")
    _validate_report_value(payload, relative)


def _report_files(root: Path) -> list[Path]:
    candidates = [
        path
        for path in _tree_files(
            root, "reports/final", frozenset({".json", ".md"})
        )
        if path.relative_to(root).as_posix() not in PROTECTED_FILES
    ]
    files: list[Path] = []
    aggregate_score_json: set[str] = set()
    for path in candidates:
        if path.suffix.lower() != ".json":
            continue
        if path.name == "final-results.json" or (
            path.name.startswith("score-results-") and path.name.endswith(".json")
        ):
            _validate_report(path, root)
            files.append(path)
            if path.name.startswith("score-results-"):
                aggregate_score_json.add(path.name)
    for path in candidates:
        if path.suffix.lower() != ".md":
            continue
        paired_json = path.with_suffix(".json").name
        if path.name in REPORT_MARKDOWN_NAMES or paired_json in aggregate_score_json:
            files.append(path)
    return files


def _source_files(root: Path) -> list[Path]:
    files = [
        _resolve_allowed_path(root, relative, directory=False)
        for relative in (*ROOT_FILES, *DOCUMENT_FILES, *ASSET_FILES)
    ]
    for relative, suffixes in TREE_ALLOWLIST:
        files.extend(_tree_files(root, relative, suffixes))
    files.extend(_report_files(root))
    unique = {path.relative_to(root).as_posix(): path for path in files}
    return [unique[relative] for relative in sorted(unique)]


def _scan_source_for_secrets(files: list[Path]) -> None:
    _scan_for_secrets(files)
    for path in files:
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES or suffix not in SOURCE_TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        if SECRET_PATTERN.search(text):
            raise ValueError(f"possible embedded secret: {path.name}")


def _write_archive(staged: Path, root: Path, files: list[Path]) -> None:
    with zipfile.ZipFile(staged, "w") as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def build_source(
    destination: str | Path | None = None,
    *,
    root: Path = PROJECT_ROOT,
) -> PackageResult:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    destination = DEFAULT_OUTPUT if destination is None else Path(destination)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)

    files = _source_files(root)
    _verify_assets(root)
    _scan_source_for_secrets(files)

    descriptor, staged_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    staged = Path(staged_name)
    try:
        _write_archive(staged, root, files)
        with staged.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.link(staged, destination)
    finally:
        staged.unlink(missing_ok=True)

    payload = destination.read_bytes()
    return PackageResult(
        destination=destination,
        file_count=len(files),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the CompassCart source bundle")
    parser.add_argument("--output", type=Path, help="archive destination")
    arguments = parser.parse_args(argv)
    result = build_source(arguments.output)
    size_mib = result.size_bytes / (1024 * 1024)
    print(f"Package: {result.destination}")
    print(f"Files: {result.file_count}")
    print(f"Size: {size_mib:.2f} MiB")
    print(f"SHA256: {result.sha256}")


if __name__ == "__main__":
    main()
