from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

import numpy as np

from .models import Candidate


class DenseBackend(Protocol):
    @property
    def available(self) -> bool: ...

    @property
    def status(self) -> str: ...

    def search(self, text: str, limit: int) -> list[Candidate]: ...


class NullDenseBackend:
    available = False

    def __init__(self, status: str = "unavailable") -> None:
        self.status = status

    def search(self, text: str, limit: int) -> list[Candidate]:
        del text, limit
        return []


class OnnxDenseBackend:
    status = "available"

    def __init__(
        self,
        session: object,
        tokenizer: object,
        *,
        product_ids: np.ndarray,
        vectors: np.ndarray,
        scales: np.ndarray,
    ) -> None:
        if vectors.ndim != 2 or len(product_ids) != vectors.shape[0]:
            raise ValueError("dense IDs and vectors have incompatible shapes")
        if scales.shape != (vectors.shape[0],):
            raise ValueError("dense vector scales have an incompatible shape")
        self.session = session
        self.tokenizer = tokenizer
        self.product_ids = product_ids.astype(str)
        self.vectors = vectors.astype(np.int8, copy=False)
        self.scales = scales.astype(np.float32, copy=False)
        self._input_names = {item.name for item in session.get_inputs()}
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def search(self, text: str, limit: int) -> list[Candidate]:
        if not self._available or limit <= 0 or not text.strip():
            return []
        try:
            query = self._embed(text)
            scores = (self.vectors @ query) * self.scales
            count = min(limit, len(scores))
            if count == len(scores):
                indices = np.arange(len(scores))
            else:
                indices = np.argpartition(scores, -count)[-count:]
            ranked = sorted(
                indices.tolist(),
                key=lambda index: (-float(scores[index]), self.product_ids[index]),
            )
            return [
                Candidate(
                    parent_asin=str(self.product_ids[index]),
                    source_scores={"dense": float(scores[index])},
                    score=float(scores[index]),
                )
                for index in ranked
            ]
        except Exception:  # noqa: BLE001 - corrupt optional inference disables dense.
            self._available = False
            self.status = "inference_failed"
            return []

    def _embed(self, text: str) -> np.ndarray:
        encoding = self.tokenizer.encode(text)
        input_ids = np.asarray([encoding.ids], dtype=np.int64)
        attention_mask = np.asarray([encoding.attention_mask], dtype=np.int64)
        inputs: dict[str, np.ndarray] = {"input_ids": input_ids}
        if "attention_mask" in self._input_names:
            inputs["attention_mask"] = attention_mask
        if "token_type_ids" in self._input_names:
            inputs["token_type_ids"] = np.asarray([encoding.type_ids], dtype=np.int64)
        output = np.asarray(self.session.run(None, inputs)[0], dtype=np.float32)
        if output.ndim == 3:
            mask = attention_mask[..., None].astype(np.float32)
            pooled = (output * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
            vector = pooled[0]
        elif output.ndim == 2:
            vector = output[0]
        else:
            raise ValueError("unexpected dense model output shape")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("dense model returned a zero vector")
        return vector / norm


def load_dense_backend(
    model_dir: str | Path,
    vector_dir: str | Path,
    manifest_path: str | Path,
) -> DenseBackend:
    if os.environ.get("COMPASSCART_DISABLE_DENSE") == "1":
        return NullDenseBackend("disabled_by_environment")
    model_dir = Path(model_dir)
    vector_dir = Path(vector_dir)
    manifest_path = Path(manifest_path)
    try:
        _verify_manifest(manifest_path)
        model_path = model_dir / "model.int8.onnx"
        tokenizer_path = model_dir / "tokenizer.json"
        product_ids_path = vector_dir / "product_ids.npy"
        vectors_path = vector_dir / "vectors.int8.npy"
        scales_path = vector_dir / "scales.npy"
        required = (
            model_path,
            tokenizer_path,
            product_ids_path,
            vectors_path,
            scales_path,
        )
        if not all(path.is_file() for path in required):
            raise FileNotFoundError("dense assets are incomplete")

        import onnxruntime as ort
        from tokenizers import Tokenizer

        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return OnnxDenseBackend(
            session,
            tokenizer,
            product_ids=np.load(product_ids_path, allow_pickle=False),
            vectors=np.load(vectors_path, allow_pickle=False),
            scales=np.load(scales_path, allow_pickle=False),
        )
    except FileNotFoundError:
        return NullDenseBackend("asset_missing")
    except ImportError:
        return NullDenseBackend("dependency_missing")
    except ValueError:
        return NullDenseBackend("asset_invalid")
    except Exception:  # noqa: BLE001 - optional dense failures remain lexical-only.
        return NullDenseBackend("initialization_failed")


def _verify_manifest(manifest_path: Path) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    root = manifest_path.parent.resolve()
    entries = 0
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, relative = raw_line.split(maxsplit=1)
        target = (root / relative.strip()).resolve()
        if root not in target.parents:
            raise ValueError("dense manifest path escapes asset root")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual.lower() != digest.lower():
            raise ValueError(f"dense checksum mismatch: {relative}")
        entries += 1
    if entries == 0:
        raise ValueError("dense manifest is empty")
