from __future__ import annotations

import hashlib
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from compasscart.integrity import sha256_file

ASSET_FILES = (
    "assets/SHA256SUMS",
    "assets/model/model.int8.onnx",
    "assets/model/tokenizer.json",
    "assets/product_vectors/product_ids.npy",
    "assets/product_vectors/scales.npy",
    "assets/product_vectors/vectors.int8.npy",
)
ROOT_FILES = (
    "agent.py",
    "requirements.txt",
    "README.md",
    "DATA_ATTRIBUTION.md",
    "MODEL_ATTRIBUTION.md",
    "licenses/all-MiniLM-L6-v2-APACHE-2.0.txt",
)
OPTIONAL_REPORTS = (
    "reports/final/architecture.md",
    "reports/final/demo-script.md",
    "reports/final/devpost.md",
    "reports/final/final-results.json",
    "reports/final/release-checklist.md",
)
TEXT_SUFFIXES = {".json", ".md", ".py", ".txt"}
SECRET_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
    r"\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}"
)
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class PackageResult:
    destination: Path
    file_count: int
    size_bytes: int
    sha256: str


def _submission_files(root: Path) -> list[Path]:
    relative_paths = [
        *ROOT_FILES,
        *ASSET_FILES,
        "tools/__init__.py",
        "tools/run_agent.py",
    ]
    relative_paths.extend(
        path.relative_to(root).as_posix()
        for path in sorted((root / "src" / "compasscart").glob("*.py"))
    )
    relative_paths.extend(path for path in OPTIONAL_REPORTS if (root / path).is_file())
    files = [root / relative for relative in relative_paths]
    missing = [
        path.relative_to(root).as_posix() for path in files if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"submission files missing: {', '.join(missing)}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _verify_assets(root: Path) -> None:
    asset_root = (root / "assets").resolve()
    manifest = asset_root / "SHA256SUMS"
    verified: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        normalized = relative.strip().replace("\\", "/")
        target = (asset_root / normalized).resolve()
        if asset_root not in target.parents:
            raise ValueError(f"asset path escapes root: {relative}")
        actual = sha256_file(target)
        if actual.lower() != digest.lower():
            raise ValueError(f"asset checksum mismatch: {relative}")
        verified.add(normalized)
    expected = {path.removeprefix("assets/") for path in ASSET_FILES[1:]}
    if verified != expected:
        raise ValueError("asset manifest does not match the runtime allowlist")


def _scan_for_secrets(files: list[Path]) -> None:
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        if SECRET_PATTERN.search(text):
            raise ValueError(f"possible embedded secret: {path.name}")


def _smoke_test_agent(root: Path) -> None:
    from agent import Agent

    variable = "COMPASSCART_DISABLE_DENSE"
    previous = os.environ.get(variable)
    os.environ[variable] = "1"
    try:
        agent = Agent(root / "tests" / "fixtures" / "catalog.jsonl")
        agent.reset("package-smoke", {"preference_tags": []})
        response = agent.respond(
            "package-smoke", "I need blue running shoes", turn=1, top_k=10
        )
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous
    if not isinstance(response.get("message"), str):
        raise TypeError("submission Agent smoke test returned an invalid response")
    recommendations = response.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        raise ValueError("submission Agent smoke test returned no recommendations")


def build_submission(
    destination: str | Path = "dist/compasscart-submission.zip",
    *,
    root: Path = PROJECT_ROOT,
) -> PackageResult:
    root = root.resolve()
    destination = Path(destination)
    if not destination.is_absolute():
        destination = root / destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    files = _submission_files(root)
    _verify_assets(root)
    _scan_for_secrets(files)
    _smoke_test_agent(root)

    with zipfile.ZipFile(destination, "w") as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())

    payload = destination.read_bytes()
    return PackageResult(
        destination=destination,
        file_count=len(files),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def main() -> None:
    result = build_submission()
    size_mib = result.size_bytes / (1024 * 1024)
    print(f"Package: {result.destination}")
    print(f"Files: {result.file_count}")
    print(f"Size: {size_mib:.2f} MiB")
    print(f"SHA256: {result.sha256}")


if __name__ == "__main__":
    main()
