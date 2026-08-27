from __future__ import annotations

import builtins
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from compasscart.catalog import CatalogIndex
from compasscart.dense import (
    NullDenseBackend,
    OnnxDenseBackend,
    _verify_manifest,
    load_dense_backend,
)
from compasscart.integrity import sha256_file
from compasscart.models import Candidate, RetrievalPlan, SessionState
from compasscart.retrieval import HybridRetriever
from tools.build_dense_assets import (
    _cleanup_export,
    _enable_system_trust,
    _quantize_vectors,
    _write_manifest,
)


@dataclass
class _Encoding:
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


class _Tokenizer:
    def encode(self, _text: str) -> _Encoding:
        return _Encoding(ids=[1, 2], attention_mask=[1, 1], type_ids=[0, 0])


class _Session:
    def get_inputs(self):
        return [
            type("Input", (), {"name": "input_ids"})(),
            type("Input", (), {"name": "attention_mask"})(),
        ]

    def run(self, _outputs, _inputs):
        return [np.array([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32)]


def _write_loader_assets(root: Path) -> tuple[Path, Path, Path]:
    model_dir = root / "model"
    vector_dir = root / "vectors"
    model_dir.mkdir()
    vector_dir.mkdir()
    paths = (
        model_dir / "model.int8.onnx",
        model_dir / "tokenizer.json",
        vector_dir / "product_ids.npy",
        vector_dir / "vectors.int8.npy",
        vector_dir / "scales.npy",
    )
    for path in paths:
        path.write_bytes(b"fixture")
    manifest = root / "SHA256SUMS"
    manifest.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(root).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    return model_dir, vector_dir, manifest


def test_missing_assets_return_null_backend(tmp_path):
    backend = load_dense_backend(
        tmp_path / "model", tmp_path / "vectors", tmp_path / "SHA256SUMS"
    )

    assert isinstance(backend, NullDenseBackend)
    assert backend.available is False
    assert backend.status == "asset_missing"
    assert backend.search("shoes", 10) == []


def test_checksum_mismatch_returns_null_backend(tmp_path):
    target = tmp_path / "vectors.npy"
    target.write_bytes(b"corrupt")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(f"{'0' * 64}  vectors.npy\n", encoding="utf-8")

    backend = load_dense_backend(tmp_path, tmp_path, manifest)

    assert isinstance(backend, NullDenseBackend)
    assert backend.status == "asset_invalid"


def test_manifest_verification_streams_without_path_read_bytes(tmp_path, monkeypatch):
    path = tmp_path / "asset.bin"
    payload = b"streamed asset" * 1_000
    path.write_bytes(payload)
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(
        f"{hashlib.sha256(payload).hexdigest()}  asset.bin\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: pytest.fail("manifest verification must stream file content"),
    )

    _verify_manifest(manifest)


def test_sha256_file_matches_hashlib_and_streams(tmp_path, monkeypatch):
    path = tmp_path / "asset.bin"
    payload = b"streamed asset" * 1_000
    path.write_bytes(payload)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: pytest.fail("sha256_file must stream file content"),
    )

    assert sha256_file(path, chunk_size=17) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_rejects_non_positive_chunk_size(tmp_path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"asset")

    with pytest.raises(ValueError):
        sha256_file(path, chunk_size=0)


def test_environment_disable_has_priority_over_missing_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPASSCART_DISABLE_DENSE", "1")

    backend = load_dense_backend(
        tmp_path / "missing-model",
        tmp_path / "missing-vectors",
        tmp_path / "missing-manifest",
    )

    assert isinstance(backend, NullDenseBackend)
    assert backend.status == "disabled_by_environment"


def test_dependency_import_failure_returns_reason(tmp_path, monkeypatch):
    model_dir, vector_dir, manifest = _write_loader_assets(tmp_path)
    real_import = builtins.__import__

    def fail_onnxruntime_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("dependency unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_onnxruntime_import)

    backend = load_dense_backend(model_dir, vector_dir, manifest)

    assert isinstance(backend, NullDenseBackend)
    assert backend.status == "dependency_missing"


def test_unexpected_initialization_failure_returns_reason(tmp_path, monkeypatch):
    import onnxruntime

    model_dir, vector_dir, manifest = _write_loader_assets(tmp_path)

    def fail_session(*_args, **_kwargs):
        raise RuntimeError("session failed")

    monkeypatch.setattr(onnxruntime, "InferenceSession", fail_session)

    backend = load_dense_backend(model_dir, vector_dir, manifest)

    assert isinstance(backend, NullDenseBackend)
    assert backend.status == "initialization_failed"


def test_local_backend_ranks_by_mean_pooled_cosine():
    backend = OnnxDenseBackend(
        _Session(),
        _Tokenizer(),
        product_ids=np.array(["A", "B"]),
        vectors=np.array([[127, 0], [0, 127]], dtype=np.int8),
        scales=np.array([1 / 127, 1 / 127], dtype=np.float32),
    )

    results = backend.search("query", 2)

    assert backend.status == "available"
    assert [item.parent_asin for item in results] == ["A", "B"]
    assert results[0].score > results[1].score


def test_dense_backend_preserves_unicode_product_ids_and_array_storage(tmp_path):
    product_ids_path = tmp_path / "product_ids.npy"
    vectors_path = tmp_path / "vectors.npy"
    scales_path = tmp_path / "scales.npy"
    np.save(product_ids_path, np.array(["A", "B"]))
    np.save(vectors_path, np.array([[127, 0], [0, 127]], dtype=np.int8))
    np.save(scales_path, np.array([1 / 127, 1 / 127], dtype=np.float32))
    product_ids = np.load(product_ids_path, mmap_mode="r", allow_pickle=False)
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    scales = np.load(scales_path, mmap_mode="r", allow_pickle=False)
    backend = OnnxDenseBackend(
        _Session(),
        _Tokenizer(),
        product_ids=product_ids,
        vectors=vectors,
        scales=scales,
    )

    assert isinstance(backend.product_ids, np.memmap)
    assert isinstance(backend.vectors, np.memmap)
    assert isinstance(backend.scales, np.memmap)
    assert np.shares_memory(backend.product_ids, product_ids)
    assert np.shares_memory(backend.vectors, vectors)
    assert np.shares_memory(backend.scales, scales)


def test_dense_backend_rejects_non_unicode_product_ids():
    with pytest.raises(ValueError, match="product IDs"):
        OnnxDenseBackend(
            _Session(),
            _Tokenizer(),
            product_ids=np.array([b"A"]),
            vectors=np.array([[127, 0]], dtype=np.int8),
            scales=np.array([1 / 127], dtype=np.float32),
        )


@pytest.mark.parametrize(
    ("product_ids", "vectors", "scales", "message"),
    (
        (
            np.array([["A"]]),
            np.array([[127]], dtype=np.int8),
            np.array([1.0], dtype=np.float32),
            "product IDs must be a non-empty one-dimensional Unicode array",
        ),
        (
            np.array([], dtype="U1"),
            np.empty((0, 1), dtype=np.int8),
            np.array([], dtype=np.float32),
            "product IDs must be a non-empty one-dimensional Unicode array",
        ),
        (
            np.array(["A"]),
            np.array([[127]], dtype=np.int16),
            np.array([1.0], dtype=np.float32),
            "dense vectors must use int8 dtype",
        ),
        (
            np.array(["A"]),
            np.array([127], dtype=np.int8),
            np.array([1.0], dtype=np.float32),
            "dense vectors must be a non-empty two-dimensional array",
        ),
        (
            np.array(["A"]),
            np.empty((1, 0), dtype=np.int8),
            np.array([1.0], dtype=np.float32),
            "dense vectors must be a non-empty two-dimensional array",
        ),
        (
            np.array(["A"]),
            np.array([[127]], dtype=np.int8),
            np.array([1.0], dtype=np.float64),
            "dense vector scales must use float32 dtype",
        ),
        (
            np.array(["A"]),
            np.array([[127]], dtype=np.int8),
            np.array([[1.0]], dtype=np.float32),
            "dense vector scales have an incompatible shape",
        ),
    ),
)
def test_dense_backend_rejects_invalid_mmap_array_shapes_and_dtypes(
    product_ids, vectors, scales, message
):
    with pytest.raises(ValueError, match=message):
        OnnxDenseBackend(
            _Session(),
            _Tokenizer(),
            product_ids=product_ids,
            vectors=vectors,
            scales=scales,
        )


def test_dense_backend_keeps_valid_array_instances_without_casting():
    product_ids = np.array(["A"])
    vectors = np.array([[127]], dtype=np.int8)
    scales = np.array([1.0], dtype=np.float32)

    backend = OnnxDenseBackend(
        _Session(),
        _Tokenizer(),
        product_ids=product_ids,
        vectors=vectors,
        scales=scales,
    )

    assert backend.product_ids is product_ids
    assert backend.vectors is vectors
    assert backend.scales is scales


def test_inference_exception_is_tolerated_until_failure_limit():
    class BrokenSession(_Session):
        def run(self, _outputs, _inputs):
            raise RuntimeError("inference failed")

    backend = OnnxDenseBackend(
        BrokenSession(),
        _Tokenizer(),
        product_ids=np.array(["A"]),
        vectors=np.array([[127, 0]], dtype=np.int8),
        scales=np.array([1 / 127], dtype=np.float32),
    )

    assert backend.search("query", 1) == []
    assert backend.available is True
    assert backend.status == "available"


def test_dense_backend_resets_failure_count_after_success():
    class FlakySession(_Session):
        calls = 0

        def run(self, outputs, inputs):
            self.calls += 1
            if self.calls in (1, 3, 4):
                raise RuntimeError("inference failed")
            return super().run(outputs, inputs)

    backend = OnnxDenseBackend(
        FlakySession(),
        _Tokenizer(),
        product_ids=np.array(["A"]),
        vectors=np.array([[127, 0]], dtype=np.int8),
        scales=np.array([1 / 127], dtype=np.float32),
    )

    assert backend.search("query", 1) == []
    assert backend.search("query", 1)[0].parent_asin == "A"
    assert backend.search("query", 1) == []
    assert backend.search("query", 1) == []
    assert backend.available is True


def test_dense_backend_opens_circuit_after_failure_limit():
    class BrokenSession(_Session):
        def run(self, _outputs, _inputs):
            raise RuntimeError("inference failed")

    backend = OnnxDenseBackend(
        BrokenSession(),
        _Tokenizer(),
        product_ids=np.array(["A"]),
        vectors=np.array([[127, 0]], dtype=np.int8),
        scales=np.array([1 / 127], dtype=np.float32),
        failure_limit=3,
    )

    assert backend.search("query", 1) == []
    assert backend.search("query", 1) == []
    assert backend.available is True
    assert backend.search("query", 1) == []
    assert backend.available is False
    assert backend.status == "inference_failed"


def test_dense_candidates_participate_in_rrf(fixture_catalog_path):
    class FakeDense:
        available = True

        def search(self, _text: str, _limit: int) -> list[Candidate]:
            return [Candidate("JACKET1", score=1.0)]

    index = CatalogIndex(fixture_catalog_path)
    plan = RetrievalPlan(
        route="browsing",
        query_text="zzzz",
        source_weights=(("dense", 1.0),),
    )

    candidates = HybridRetriever(index, FakeDense()).retrieve(plan, SessionState("s1"))

    assert candidates[0].parent_asin == "JACKET1"
    assert candidates[0].source_scores["dense"] > 0


def test_asset_quantization_preserves_normalized_direction():
    embeddings = np.array([[3.0, 4.0], [-2.0, 0.5]], dtype=np.float32)

    vectors, scales = _quantize_vectors(embeddings)
    restored = vectors.astype(np.float32) * scales[:, None]

    assert vectors.dtype == np.int8
    assert np.allclose(np.linalg.norm(restored, axis=1), 1.0, atol=0.02)


def test_asset_manifest_is_sorted_and_verifiable(tmp_path):
    second = tmp_path / "b.bin"
    first = tmp_path / "a.bin"
    second.write_bytes(b"second")
    first.write_bytes(b"first")

    manifest = _write_manifest(tmp_path, [second, first])

    assert [line.split()[-1] for line in manifest.read_text().splitlines()] == [
        "a.bin",
        "b.bin",
    ]


def test_asset_builder_enables_windows_system_trust(monkeypatch):
    called = []
    fake = type("TrustStore", (), {"inject_into_ssl": lambda: called.append(True)})
    monkeypatch.setitem(__import__("sys").modules, "truststore", fake)

    _enable_system_trust()

    assert called == [True]


def test_asset_builder_removes_unquantized_export(tmp_path):
    source = tmp_path / "model.onnx"
    quantized = tmp_path / "model.int8.onnx"
    source.write_bytes(b"large intermediate")
    quantized.write_bytes(b"small release model")

    _cleanup_export(source, quantized)

    assert source.exists() is False
    assert quantized.exists() is True
