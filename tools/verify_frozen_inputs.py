"""Fail-closed integrity preflight for inputs sealed by the hardening plan."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Final

_HASH_RE: Final = re.compile(r"^[0-9a-f]{64}$")
FROZEN_INPUTS: Final[dict[str, str]] = {
    "evaluator/local_evaluator.py": "79a5ea06f9a1b8c5036f30efa85dc1f36b8f6b06eb8feb8f545dfa767bc45564",
    "data/public_set.jsonl": "857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579",
    "data/catalog.jsonl": "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67",
    "assets/SHA256SUMS": "eafb2068d73e217af2949b5dd5a87b36fcf25b316b1bde9429cb2d40af52ee51",
    "assets/model/model.int8.onnx": "3013f5cdb68ea6b6a271ab8fef96c5e6721669c2c2be3f83ec1be07486133892",
    "assets/model/tokenizer.json": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
    "assets/product_vectors/product_ids.npy": "e5ab6608c15dd0b51dd2f63db088705613efdfea85859462c2d514752fe8d7c9",
    "assets/product_vectors/scales.npy": "3eb26371cb15a3e2af5d287a290cd338c12c3a3f9e606bdd911c53e6d4064d53",
    "assets/product_vectors/vectors.int8.npy": "ccaf43034103312788ddde27890861c6f5d93052dbc930b0b1bff56acf0c4d63",
}
_FROZEN_PATHS: Final = frozenset({
    "evaluator/local_evaluator.py", "data/public_set.jsonl", "data/catalog.jsonl",
    "assets/SHA256SUMS", "assets/model/model.int8.onnx", "assets/model/tokenizer.json",
    "assets/product_vectors/product_ids.npy", "assets/product_vectors/scales.npy",
    "assets/product_vectors/vectors.int8.npy",
})


def _validate_manifest(manifest: Mapping[str, str]) -> None:
    if not isinstance(manifest, Mapping) or not manifest:
        raise ValueError("manifest is invalid")
    for relative, expected in manifest.items():
        path = Path(relative)
        if (
            not isinstance(relative, str)
            or "\\" in relative
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != relative
            or not isinstance(expected, str)
            or _HASH_RE.fullmatch(expected) is None
        ):
            raise ValueError("manifest is invalid")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: str | Path, manifest: Mapping[str, str]) -> int:
    _validate_manifest(manifest)
    repository = Path(root).resolve()
    status = 0
    for relative in sorted(manifest):
        target = (repository / relative).resolve()
        try:
            target.relative_to(repository)
        except ValueError as error:
            raise ValueError("manifest is invalid") from error
        if not target.is_file():
            print(f"{relative} missing")
            status = 1
        elif _sha256_file(target) != manifest[relative]:
            print(f"{relative} mismatch")
            status = 1
        else:
            print(f"{relative} ok")
    return status


def main() -> None:
    if set(FROZEN_INPUTS) != _FROZEN_PATHS:
        print("manifest anomaly")
        raise SystemExit(1)
    try:
        status = verify_manifest(Path(__file__).resolve().parents[1], FROZEN_INPUTS)
    except ValueError:
        print("manifest anomaly")
        raise SystemExit(1) from None
    raise SystemExit(status)


if __name__ == "__main__":
    main()
