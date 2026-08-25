from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compasscart.catalog import CatalogIndex
from compasscart.dense import NullDenseBackend, OnnxDenseBackend, load_dense_backend
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


def test_missing_assets_return_null_backend(tmp_path):
    backend = load_dense_backend(
        tmp_path / "model", tmp_path / "vectors", tmp_path / "SHA256SUMS"
    )

    assert isinstance(backend, NullDenseBackend)
    assert backend.available is False
    assert backend.search("shoes", 10) == []


def test_checksum_mismatch_returns_null_backend(tmp_path):
    target = tmp_path / "vectors.npy"
    target.write_bytes(b"corrupt")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(f"{'0' * 64}  vectors.npy\n", encoding="utf-8")

    backend = load_dense_backend(tmp_path, tmp_path, manifest)

    assert isinstance(backend, NullDenseBackend)


def test_local_backend_ranks_by_mean_pooled_cosine():
    backend = OnnxDenseBackend(
        _Session(),
        _Tokenizer(),
        product_ids=np.array(["A", "B"]),
        vectors=np.array([[127, 0], [0, 127]], dtype=np.int8),
        scales=np.array([1 / 127, 1 / 127], dtype=np.float32),
    )

    results = backend.search("query", 2)

    assert [item.parent_asin for item in results] == ["A", "B"]
    assert results[0].score > results[1].score


def test_inference_exception_disables_backend():
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
    assert backend.available is False


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
