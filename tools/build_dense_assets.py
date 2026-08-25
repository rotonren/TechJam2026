from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compasscart.normalization import searchable_fields


def _enable_system_trust() -> None:
    import truststore

    truststore.inject_into_ssl()


def _quantize_vectors(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-9)
    scales = np.max(np.abs(normalized), axis=1) / 127.0
    scales = np.maximum(scales, np.finfo(np.float32).eps).astype(np.float32)
    vectors = np.clip(np.rint(normalized / scales[:, None]), -127, 127).astype(np.int8)
    return vectors, scales


def _write_manifest(root: Path, paths: list[Path]) -> Path:
    root = root.resolve()
    entries: list[tuple[str, str]] = []
    for path in paths:
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError("asset path escapes manifest root")
        relative = resolved.relative_to(root).as_posix()
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        entries.append((relative, digest))
    manifest = root / "SHA256SUMS"
    manifest.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in sorted(entries)),
        encoding="utf-8",
    )
    return manifest


def _cleanup_export(source_model: Path, quantized_model: Path) -> None:
    if source_model.resolve() != quantized_model.resolve() and source_model.exists():
        source_model.unlink()


def _load_catalog(catalog_path: Path) -> tuple[list[str], list[str]]:
    identifiers: list[str] = []
    texts: list[str] = []
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            identifiers.append(str(product["parent_asin"]))
            texts.append(" ".join(searchable_fields(product)))
    if not identifiers:
        raise ValueError("catalog is empty")
    return identifiers, texts


def build_assets(
    catalog_path: Path,
    asset_root: Path,
    *,
    model_name: str,
    batch_size: int,
) -> Path:
    _enable_system_trust()
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    model_dir = asset_root / "model"
    vector_dir = asset_root / "product_vectors"
    model_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_dir)
    exported = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    exported.save_pretrained(model_dir)
    source_model = next(model_dir.glob("*.onnx"))
    quantized_model = model_dir / "model.int8.onnx"
    quantize_dynamic(source_model, quantized_model, weight_type=QuantType.QInt8)
    _cleanup_export(source_model, quantized_model)

    identifiers, texts = _load_catalog(catalog_path)
    encoder = SentenceTransformer(model_name, device="cpu")
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors, scales = _quantize_vectors(embeddings)
    max_length = max(len(identifier) for identifier in identifiers)
    product_ids = np.asarray(identifiers, dtype=f"<U{max_length}")

    product_ids_path = vector_dir / "product_ids.npy"
    vectors_path = vector_dir / "vectors.int8.npy"
    scales_path = vector_dir / "scales.npy"
    np.save(product_ids_path, product_ids, allow_pickle=False)
    np.save(vectors_path, vectors, allow_pickle=False)
    np.save(scales_path, scales, allow_pickle=False)
    return _write_manifest(
        asset_root,
        [
            quantized_model,
            model_dir / "tokenizer.json",
            product_ids_path,
            vectors_path,
            scales_path,
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--asset-root", type=Path, default=Path("assets"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    manifest = build_assets(
        args.catalog,
        args.asset_root,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    print(f"Dense assets ready: {manifest}")


if __name__ == "__main__":
    main()
