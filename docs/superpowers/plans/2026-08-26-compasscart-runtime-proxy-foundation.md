# CompassCart Runtime and Proxy Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立不泄漏公开答案的代理评测与全量性能基线，修复任意工作目录下 Dense 资产失效，并在保持推荐顺序等价的前提下降低官方环境的延迟、内存和 fallback 风险。

**Architecture:** 第一部分只新增开发工具，冻结代表性/压力代理集、audit 和 800-response benchmark，不触碰评分生产路径。第二部分先完成 R0 资产解析，再逐项实施 P0 等价输出优化；每项使用失败先行测试、独立提交和同一 R0 benchmark 比较。官方 evaluator、public set、catalog、Dense 模型/向量和评分公式保持字节不变。

**Tech Stack:** CPython 3.12、pytest、Ruff、标准库 JSON/hashlib/subprocess、NumPy memmap、ONNX Runtime、SQLite FTS5、psutil（仅开发 benchmark）。

---

## Scope Guard

执行前记录并在最终再次核对以下 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `evaluator/local_evaluator.py` | `84EA899707452DE249CA62ABEE77C4B40AB7A3139B5CC798AC30C9F521F91B30` |
| `data/public_set.jsonl` | `571359A8A69014C43FC30D39C996C4A28E875DCCC249DFFC707358757BEB16C0` |
| `data/catalog.jsonl` | `DA979B05A68AF864CB0DCF9EE6A81C010C7E66A57978AD286C7A2E005FC69A67` |
| `assets/SHA256SUMS` | `2857869C2A872CCEA9D93BB043B8CB45EEE07CB1EFC1F943B401C1919982D86E` |
| `assets/model/model.int8.onnx` | `3013F5CDB68EA6B6A271AB8FEF96C5E6721669C2C2BE3F83EC1BE07486133892` |
| `assets/model/tokenizer.json` | `DA0E79933B9ED51798A3AE27893D3C5FA4A201126CEF75586296DF9B4D2C62A0` |
| `assets/product_vectors/product_ids.npy` | `E5AB6608C15DD0B51DD2F63DB088705613EFDFEA85859462C2D514752FE8D7C9` |
| `assets/product_vectors/scales.npy` | `3EB26371CB15A3E2AF5D287A290CD338C12C3A3F9E606BDD911C53E6D4064D53` |
| `assets/product_vectors/vectors.int8.npy` | `CCAF43034103312788DDDE27890861C6F5D93052DBC930B0B1BFF56ACF0C4D63` |

禁止修改或暂存上述文件。禁止在 `src/` 中出现 `public_`、公开 ASIN、ground truth 或代理 target ID。所有代理数据、transcript、CV、audit 和 benchmark 输出只写入 ignored `var/balanced-hardening/`。

## File Map

**Create**

- `tools/proxy_dataset.py`: 冻结输入校验、目录分层、代理样本/profile/manifest 生成。
- `tools/run_proxy.py`: 代理对话扰动、评分、开发折、压力套件和不可覆盖 audit 输出。
- `tools/benchmark_release.py`: transcript 捕获、独立子进程 benchmark、内存/延迟聚合。
- `src/compasscart/integrity.py`: 流式 SHA-256 公共运行时 helper。
- `tests/unit/test_proxy_dataset.py`: 代理抽样、隔离、比例、hash 和 profile 测试。
- `tests/unit/test_run_proxy.py`: 对话扰动、报告脱敏和 audit 写保护测试。
- `tests/unit/test_benchmark_release.py`: transcript、聚合和 worker schema 测试。
- `tests/unit/test_config.py`: submission root 和相对/绝对资产路径测试。
- `reports/final/balanced-hardening-foundation-results.md`: R0/P0 聚合证据和进入 S1-S5 的门禁结论。

**Modify**

- `requirements-dev.txt`: 添加仅 benchmark 使用的 psutil。
- `src/compasscart/config.py:8`: 解析相对 Dense 路径，不改变配置字段或 repr。
- `src/compasscart/agent.py:24`: 使用稳定 submission root，记录 Dense 状态和协作式预算跳过。
- `src/compasscart/dense.py:20`: 状态原因、三次连续失败 circuit breaker、流式 hash 和 mmap。
- `src/compasscart/catalog.py:87`: 缓存 popularity、FTS 连续失败计数。
- `src/compasscart/retrieval.py:43`: lazy fallback 和 Dense 预算检查。
- `src/compasscart/ranker.py:24`: 有界 diversity cache 和 MMR 预算检查。
- `src/compasscart/question_policy.py:30`: 复用 CatalogIndex 已规范化属性。
- `tools/package_submission.py:71`: 使用流式 hash。
- `tests/unit/test_dense.py:42`: 状态、mmap、hash 和 circuit breaker 回归。
- `tests/unit/test_catalog.py:1`: popularity 与 FTS circuit breaker 回归。
- `tests/unit/test_retrieval.py:1`: lazy fallback 与预算回归。
- `tests/unit/test_ranker.py:1`: diversity cache 与预算回归。
- `tests/unit/test_question_policy.py:24`: attribute lookup 等价回归。
- `tests/contract/test_submission_package.py:36`: 仓库外 CWD 的 extracted Dense smoke。
- `tests/integration/test_fallbacks.py:27`: Agent 预算和状态 trace。
- `README.md:8`: 任意 CWD 资产行为与 benchmark 命令。

### Task 1: Deterministic Proxy Sampling Primitives

**Files:**
- Create: `tools/proxy_dataset.py`
- Create: `tests/unit/test_proxy_dataset.py`

- [ ] **Step 1: Write failing sampling tests**

在 `tests/unit/test_proxy_dataset.py` 添加：

```python
from __future__ import annotations

from tools.proxy_dataset import (
    ProxyProduct,
    representative_ids,
    stable_int,
    stress_ids,
)


def _record(index: int, category: str, price_bin: str, popularity: str, completeness: str) -> ProxyProduct:
    return ProxyProduct(
        parent_asin=f"P{index:03d}",
        category=category,
        price_bin=price_bin,
        popularity_bin=popularity,
        completeness_bin=completeness,
        difficulty="medium",
    )


def test_stable_int_is_seeded_and_order_independent():
    assert stable_int(20260826, "P001") == stable_int(20260826, "P001")
    assert stable_int(20260826, "P001") != stable_int(20260827, "P001")


def test_representative_selection_is_deterministic_and_covers_categories():
    rows = [
        _record(index, f"cat-{index % 4}", f"q{index % 4 + 1}", f"q{index % 4 + 1}", "5+")
        for index in range(40)
    ]
    first = representative_ids(rows, count=20, seed=20260826)
    second = representative_ids(list(reversed(rows)), count=20, seed=20260826)

    assert first == second
    assert len(first) == len(set(first)) == 20
    selected_categories = {row.category for row in rows if row.parent_asin in first}
    assert selected_categories == {"cat-0", "cat-1", "cat-2", "cat-3"}


def test_stress_selection_is_disjoint_and_prefers_rare_strata():
    rows = [
        _record(index, "common" if index < 35 else "rare", "q1", "q1", "0-2")
        for index in range(40)
    ]
    representative = set(representative_ids(rows, count=10, seed=20260826))
    stress = stress_ids(rows, count=8, seed=20260827, excluded=representative)

    assert representative.isdisjoint(stress)
    assert len(stress) == len(set(stress)) == 8
    assert any(row.category == "rare" and row.parent_asin in stress for row in rows)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_proxy_dataset.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'tools.proxy_dataset'`.

- [ ] **Step 3: Implement the sampling primitives**

创建 `tools/proxy_dataset.py`，先实现以下完整边界；CLI 和 JSON 输出留给 Task 2：

```python
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ProxyProduct:
    parent_asin: str
    category: str
    price_bin: str
    popularity_bin: str
    completeness_bin: str
    difficulty: str

    def dimensions(self) -> tuple[tuple[str, str], ...]:
        return (
            ("category", self.category),
            ("price", self.price_bin),
            ("popularity", self.popularity_bin),
            ("completeness", self.completeness_bin),
        )


def stable_int(seed: int, value: str) -> int:
    payload = f"{seed}\0{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _apportion(values: list[str], count: int) -> dict[str, int]:
    frequencies = Counter(values)
    total = sum(frequencies.values())
    if count < 0 or count > total:
        raise ValueError("sample count must fit the available population")
    raw = {key: count * value / total for key, value in frequencies.items()}
    quotas = {key: min(frequencies[key], math.floor(raw[key])) for key in frequencies}
    remaining = count - sum(quotas.values())
    order = sorted(frequencies, key=lambda key: (-(raw[key] - quotas[key]), key))
    while remaining:
        progressed = False
        for key in order:
            if quotas[key] < frequencies[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise ValueError("unable to apportion proxy sample")
    return quotas


def representative_ids(records: list[ProxyProduct], *, count: int, seed: int) -> list[str]:
    unique = {record.parent_asin: record for record in records}
    if len(unique) != len(records):
        raise ValueError("proxy products must have unique parent_asin values")
    by_category: dict[str, list[ProxyProduct]] = defaultdict(list)
    for record in records:
        by_category[record.category].append(record)
    if count < len(by_category) or count > len(records):
        raise ValueError("representative count cannot cover every category")

    selected: list[ProxyProduct] = []
    selected_ids: set[str] = set()
    for category in sorted(by_category):
        record = min(by_category[category], key=lambda item: stable_int(seed, item.parent_asin))
        selected.append(record)
        selected_ids.add(record.parent_asin)

    dimensions = ("category", "price", "popularity", "completeness")
    quotas = {
        dimension: _apportion(
            [dict(record.dimensions())[dimension] for record in records], count
        )
        for dimension in dimensions
    }
    selected_counts = {
        dimension: Counter(dict(record.dimensions())[dimension] for record in selected)
        for dimension in dimensions
    }
    remaining = [record for record in records if record.parent_asin not in selected_ids]
    while len(selected) < count:
        def priority(record: ProxyProduct) -> tuple[int, int]:
            values = dict(record.dimensions())
            deficit = sum(
                max(quotas[dimension].get(values[dimension], 0) - selected_counts[dimension][values[dimension]], 0)
                for dimension in dimensions
            )
            return deficit, -stable_int(seed, record.parent_asin)

        record = max(remaining, key=priority)
        remaining.remove(record)
        selected.append(record)
        selected_ids.add(record.parent_asin)
        for dimension, value in record.dimensions():
            selected_counts[dimension][value] += 1
    return [record.parent_asin for record in sorted(selected, key=lambda item: stable_int(seed, item.parent_asin))]


def stress_ids(
    records: list[ProxyProduct],
    *,
    count: int,
    seed: int,
    excluded: set[str],
) -> list[str]:
    candidates = [record for record in records if record.parent_asin not in excluded]
    if count < 0 or count > len(candidates):
        raise ValueError("stress count must fit the remaining population")
    frequencies = Counter(pair for record in records for pair in record.dimensions())

    def weighted_key(record: ProxyProduct) -> tuple[float, int]:
        weight = sum(1.0 / frequencies[pair] for pair in record.dimensions())
        unit = (stable_int(seed, record.parent_asin) + 1) / ((1 << 256) + 1)
        return -math.log(unit) / weight, stable_int(seed, record.parent_asin)

    return [record.parent_asin for record in sorted(candidates, key=weighted_key)[:count]]
```

- [ ] **Step 4: Run focused tests and Ruff**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_proxy_dataset.py -q
& .\.venv\Scripts\python.exe -m ruff check tools/proxy_dataset.py tests/unit/test_proxy_dataset.py
```

Expected: `3 passed`; Ruff prints `All checks passed!`.

- [ ] **Step 5: Commit**

```powershell
git add tools/proxy_dataset.py tests/unit/test_proxy_dataset.py
git commit -m "test: add deterministic proxy sampling"
```

### Task 2: Frozen Proxy Bundle and Manifest

**Files:**
- Modify: `tools/proxy_dataset.py`
- Modify: `tests/unit/test_proxy_dataset.py`

- [ ] **Step 1: Add failing bundle tests**

追加一个临时 catalog/public-set 测试，调用 `build_proxy_bundle(..., enforce_frozen=False)` 两次，并断言：

```python
import json
from pathlib import Path

from tools.proxy_dataset import build_proxy_bundle


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_proxy_bundle_is_reproducible_excludes_public_targets_and_sanitizes_profiles(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    public = tmp_path / "public.jsonl"
    products = [
        {
            "parent_asin": f"P{index:03d}",
            "title": f"Product {index}",
            "features": ["comfortable", "durable"],
            "details": {"Color": "blue" if index % 2 else "black"},
            "categories": ["Clothing", f"Category {index % 5}"],
            "price": float(index + 10),
            "average_rating": 4.0,
            "rating_number": index + 1,
        }
        for index in range(50)
    ]
    public_rows = [
        {
            "sample_id": "public_0001",
            "scenario_type": "buying",
            "ground_truth": {"parent_asin": "P000"},
            "user_profile": {
                "average_prior_rating": 5.0,
                "preference_tags": ["fit", "comfort"],
                "purchase_frequency": "3-4 prior purchases",
                "rating_style": "usually positive",
                "summary": "source text must not be copied",
            },
        }
    ]
    _write_jsonl(catalog, products)
    _write_jsonl(public, public_rows)

    first = build_proxy_bundle(
        catalog,
        public,
        tmp_path / "first",
        representative_count=20,
        stress_count=8,
        enforce_frozen=False,
    )
    second = build_proxy_bundle(
        catalog,
        public,
        tmp_path / "second",
        representative_count=20,
        stress_count=8,
        enforce_frozen=False,
    )

    assert first["output_hashes"] == second["output_hashes"]
    representative = [json.loads(line) for line in (tmp_path / "first" / "representative.jsonl").read_text().splitlines()]
    assert "P000" not in {row["ground_truth"]["parent_asin"] for row in representative}
    assert {row["scenario_type"] for row in representative} == {"buying", "browsing", "intent_override", "boundary"}
    assert all(row["user_profile"]["summary"] != "source text must not be copied" for row in representative)
    assert {row["proxy_fold"] for row in representative} == {1, 2, 3, 4, 5}
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_proxy_dataset.py::test_proxy_bundle_is_reproducible_excludes_public_targets_and_sanitizes_profiles -q
```

Expected: FAIL because `build_proxy_bundle` is not defined.

- [ ] **Step 3: Implement frozen inputs, records, profiles, scenarios and manifest**

在 `tools/proxy_dataset.py` 增加以下常量和 API，并使用 `evaluator.local_evaluator.intent_card`、`behavior_for`、`coarse_category` 生成隐藏字段：

```python
import argparse
import json
import random
from pathlib import Path

from evaluator.local_evaluator import behavior_for, coarse_category, intent_card


FROZEN_SHA256 = {
    "evaluator/local_evaluator.py": "84ea899707452de249ca62abee77c4b40ab7a3139b5cc798ac30c9f521f91b30",
    "data/public_set.jsonl": "571359a8a69014c43fc30d39c996c4a28e875dccc249dffc707358757beb16c0",
    "data/catalog.jsonl": "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67",
    "assets/SHA256SUMS": "2857869c2a872ccea9d93bb043b8cb45eee07cb1efc1f943b401c1919982d86e",
    "assets/model/model.int8.onnx": "3013f5cdb68ea6b6a271ab8fef96c5e6721669c2c2be3f83ec1be07486133892",
    "assets/model/tokenizer.json": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
    "assets/product_vectors/product_ids.npy": "e5ab6608c15dd0b51dd2f63db088705613efdfea85859462c2d514752fe8d7c9",
    "assets/product_vectors/scales.npy": "3eb26371cb15a3e2af5d287a290cd338c12c3a3f9e606bdd911c53e6d4064d53",
    "assets/product_vectors/vectors.int8.npy": "ccaf43034103312788ddde27890861c6f5d93052dbc930b0b1bff56acf0c4d63",
}
SCENARIO_WEIGHTS = {"buying": 0.40, "browsing": 0.40, "intent_override": 0.15, "boundary": 0.05}
GENERATOR_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_inputs(root: Path) -> dict[str, str]:
    actual = {relative: sha256_file(root / relative) for relative in FROZEN_SHA256}
    mismatches = {relative: value for relative, value in actual.items() if value != FROZEN_SHA256[relative]}
    if mismatches:
        raise ValueError(f"frozen competition input mismatch: {sorted(mismatches)}")
    return actual


def _safe_profile(profile: dict[str, object]) -> dict[str, object]:
    tags = [str(value) for value in profile.get("preference_tags", []) if str(value).strip()]
    rating_style = str(profile.get("rating_style", "unspecified"))
    emphasis = ", ".join(tags) if tags else "general preferences"
    return {
        "average_prior_rating": float(profile.get("average_prior_rating", 0.0) or 0.0),
        "preference_tags": tags,
        "purchase_frequency": str(profile.get("purchase_frequency", "unspecified")),
        "rating_style": rating_style,
        "summary": f"Prior purchases emphasize {emphasis}; ratings are {rating_style}.",
    }


def _scenario_schedule(count: int, seed: int) -> list[str]:
    raw = {name: count * weight for name, weight in SCENARIO_WEIGHTS.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    for name in sorted(raw, key=lambda key: (-(raw[key] - counts[key]), key))[: count - sum(counts.values())]:
        counts[name] += 1
    values = [
        (name, ordinal)
        for name in sorted(counts)
        for ordinal in range(counts[name])
    ]
    ordered = sorted(
        values,
        key=lambda item: (
            stable_int(seed, f"{item[0]}:{item[1]}"),
            item[0],
            item[1],
        ),
    )
    return [name for name, _ordinal in ordered]


def _assign_proxy_folds(rows: list[dict], seed: int) -> None:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scenario_type"]), str(row["difficulty_bucket"]))].append(row)
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda row: stable_int(seed, str(row["sample_id"])))
        for position, row in enumerate(ordered):
            row["proxy_fold"] = position % 5 + 1


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _number(value: object) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quartile_thresholds(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    ordered = sorted(values)
    return tuple(
        ordered[min(math.ceil(len(ordered) * fraction) - 1, len(ordered) - 1)]
        for fraction in (0.25, 0.50, 0.75)
    )


def _quartile(value: float, thresholds: tuple[float, float, float]) -> str:
    for index, threshold in enumerate(thresholds, start=1):
        if value <= threshold:
            return f"q{index}"
    return "q4"


def _present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _product_records(
    products: list[dict[str, object]],
    excluded: set[str],
) -> tuple[list[ProxyProduct], dict[str, dict[str, object]]]:
    products_by_id: dict[str, dict[str, object]] = {}
    for product in products:
        identifier = str(product.get("parent_asin", "")).strip()
        if not identifier or identifier in products_by_id:
            raise ValueError("catalog parent_asin values must be non-empty and unique")
        products_by_id[identifier] = product

    price_values = [
        number
        for product in products
        if (number := _number(product.get("price"))) is not None
    ]
    popularity_values = [
        _number(product.get("rating_number")) or 0.0 for product in products
    ]
    price_thresholds = _quartile_thresholds(price_values)
    popularity_thresholds = _quartile_thresholds(popularity_values)
    searchable_fields = (
        "title",
        "features",
        "details",
        "description",
        "categories",
        "store",
    )
    records: list[ProxyProduct] = []
    for identifier in sorted(products_by_id):
        if identifier in excluded:
            continue
        product = products_by_id[identifier]
        price = _number(product.get("price"))
        completeness = sum(_present(product.get(field)) for field in searchable_fields)
        completeness_bin = "0-2" if completeness <= 2 else "3-4" if completeness <= 4 else "5+"
        popularity_bin = _quartile(
            _number(product.get("rating_number")) or 0.0,
            popularity_thresholds,
        )
        difficulty = (
            "hard"
            if completeness_bin == "0-2" or price is None
            else "easy"
            if completeness_bin == "5+" and popularity_bin == "q4"
            else "medium"
        )
        records.append(
            ProxyProduct(
                parent_asin=identifier,
                category=coarse_category(
                    [str(value) for value in product.get("categories") or []]
                ),
                price_bin="missing" if price is None else _quartile(price, price_thresholds),
                popularity_bin=popularity_bin,
                completeness_bin=completeness_bin,
                difficulty=difficulty,
            )
        )
    return records, products_by_id
```

实现 `build_proxy_bundle()` 时：

```python
def build_proxy_bundle(
    catalog_path: Path,
    public_path: Path,
    output_dir: Path,
    *,
    representative_count: int = 2000,
    stress_count: int = 800,
    enforce_frozen: bool = True,
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    catalog_path = catalog_path.resolve()
    public_path = public_path.resolve()
    output_dir = output_dir.resolve()
    if enforce_frozen:
        if catalog_path != (root / "data/catalog.jsonl").resolve():
            raise ValueError("frozen proxy generation requires data/catalog.jsonl")
        if public_path != (root / "data/public_set.jsonl").resolve():
            raise ValueError("frozen proxy generation requires data/public_set.jsonl")
        if not output_dir.is_relative_to((root / "var").resolve()):
            raise ValueError("frozen proxy output must stay under ignored var/")
        input_hashes = verify_frozen_inputs(root)
    else:
        input_hashes = {
            "catalog": sha256_file(catalog_path),
            "public": sha256_file(public_path),
        }
    products = [json.loads(line) for line in catalog_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    public_rows = [json.loads(line) for line in public_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    excluded = {str(row["ground_truth"]["parent_asin"]) for row in public_rows}
    profiles = [_safe_profile(dict(row["user_profile"])) for row in public_rows]
    if not profiles:
        raise ValueError("public profile pool must not be empty")
    records, products_by_id = _product_records(products, excluded)
    records_by_id = {record.parent_asin: record for record in records}
    representative = representative_ids(records, count=representative_count, seed=20260826)
    stress = stress_ids(records, count=stress_count, seed=20260827, excluded=set(representative))

    def materialize(identifiers: list[str], suite: str, seed: int) -> list[dict]:
        scenarios = _scenario_schedule(len(identifiers), seed)
        ordered_profiles = sorted(profiles, key=lambda value: stable_int(20260828, _canonical_hash(value)))
        rows: list[dict] = []
        for index, (identifier, scenario) in enumerate(zip(identifiers, scenarios, strict=True), start=1):
            product = products_by_id[identifier]
            card = intent_card(product)
            sample_id = f"proxy_{suite}_{index:04d}"
            rows.append({
                "sample_id": sample_id,
                "scenario_type": scenario,
                "category_bucket": coarse_category([str(value) for value in product.get("categories") or []]),
                "difficulty_bucket": records_by_id[identifier].difficulty,
                "ground_truth": {"parent_asin": identifier},
                "user_profile": ordered_profiles[(index - 1) % len(ordered_profiles)],
                "intent_card": card,
                "behavior": behavior_for(scenario, card, random.Random(f"{seed}:{sample_id}")),
                "dialogue_variant": stable_int(seed, sample_id) % 4,
                "proxy_suite": suite,
            })
        if suite == "representative":
            _assign_proxy_folds(rows, seed)
        return rows

    representative_rows = materialize(representative, "representative", 20260826)
    stress_rows = materialize(stress, "stress", 20260827)
    if set(representative) & set(stress):
        raise ValueError("representative and stress proxy targets must be disjoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "representative.jsonl", representative_rows)
    _write_jsonl(output_dir / "stress.jsonl", stress_rows)
    generator_config = {
        "scenario_weights": SCENARIO_WEIGHTS,
        "dimensions": ["category", "price", "popularity", "completeness"],
        "price_bins": ["missing", "q1", "q2", "q3", "q4"],
        "popularity_bins": ["q1", "q2", "q3", "q4"],
        "completeness_bins": ["0-2", "3-4", "5+"],
        "representative_count": representative_count,
        "stress_count": stress_count,
        "profile_summary_version": 1,
        "fold_count": 5,
    }
    manifest = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "generator_config": generator_config,
        "generator_config_hash": _canonical_hash(generator_config),
        "seeds": {"representative": 20260826, "stress": 20260827, "profile": 20260828},
        "input_hashes": input_hashes,
        "excluded_target_hash": _canonical_hash(sorted(excluded)),
        "target_hashes": {
            "representative": _canonical_hash(representative),
            "stress": _canonical_hash(stress),
        },
        "output_hashes": {
            "representative.jsonl": sha256_file(output_dir / "representative.jsonl"),
            "stress.jsonl": sha256_file(output_dir / "stress.jsonl"),
        },
        "counts": {"representative": len(representative_rows), "stress": len(stress_rows)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
```

在文件底部添加固定 CLI；除非显式传 `--no-enforce-frozen`（只供单元测试临时数据），真实生成始终执行冻结 hash 和 `var/` 路径检查：

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen CompassCart proxy suites")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public-set", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representative-count", type=int, default=2000)
    parser.add_argument("--stress-count", type=int, default=800)
    parser.add_argument("--no-enforce-frozen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest = build_proxy_bundle(
        args.catalog,
        args.public_set,
        args.output_dir,
        representative_count=args.representative_count,
        stress_count=args.stress_count,
        enforce_frozen=not args.no_enforce_frozen,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and create the frozen real bundle**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_proxy_dataset.py -q
& .\.venv\Scripts\python.exe -m ruff check tools/proxy_dataset.py tests/unit/test_proxy_dataset.py
& .\.venv\Scripts\python.exe -m tools.proxy_dataset --catalog data/catalog.jsonl --public-set data/public_set.jsonl --output-dir var/balanced-hardening/proxy-v1
```

Expected: tests pass; manifest reports representative `2000`, stress `800`, disjoint target hashes, and all frozen input hashes match. Do not open generated target/session rows except for schema validation commands in this plan.

- [ ] **Step 5: Commit**

```powershell
git add tools/proxy_dataset.py tests/unit/test_proxy_dataset.py
git commit -m "test: freeze catalog-derived proxy suites"
```

### Task 3: Proxy Dialogue Evaluator and Audit Guard

**Files:**
- Create: `tools/run_proxy.py`
- Create: `tests/unit/test_run_proxy.py`
- Modify: `tests/unit/test_experiment_tools.py`

- [ ] **Step 1: Write failing dialogue and audit tests**

测试必须证明同一 sample/turn 的模板稳定、不同 `dialogue_variant` 会改变文本但不改变 disclosed constraint，并证明 audit 报告不含 sessions 且不能覆盖：

```python
import json

import pytest

from tools.run_proxy import ProxyDialogue, write_audit_report


def test_proxy_dialogue_variants_preserve_disclosed_constraint():
    sample = {
        "sample_id": "proxy_representative_0001",
        "scenario_type": "buying",
        "dialogue_variant": 1,
        "intent_card": {"hard_constraints": ["cotton"], "soft_preferences": ["blue"]},
        "behavior": {},
    }
    dialogue = ProxyDialogue()
    disclosed: set[str] = set()

    message = dialogue.initial_message(sample, "shirts", disclosed)

    assert "cotton" in message
    assert disclosed == {"cotton"}
    assert message == dialogue.initial_message(sample, "shirts", set())


def test_audit_report_omits_sessions_and_refuses_overwrite(tmp_path):
    result = {"recommended_technical_score": 0.5, "sessions": [{"sample_id": "secret"}]}
    destination = tmp_path / "baseline.json"

    write_audit_report(destination, result, metadata={"audit_label": "baseline"})

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert "sessions" not in payload["aggregate"]
    with pytest.raises(FileExistsError):
        write_audit_report(destination, result, metadata={"audit_label": "baseline"})
```

- [ ] **Step 2: Verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_run_proxy.py -q
```

Expected: `ModuleNotFoundError: No module named 'tools.run_proxy'`.

- [ ] **Step 3: Implement deterministic proxy dialogue**

创建 `tools/run_proxy.py`。从 `evaluator.local_evaluator` 只导入评分/规范 helper 和常量，不修改 evaluator。模块使用以下导入和固定模板：

```python
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from evaluator.local_evaluator import (
    ALLOWED_ATTRIBUTES,
    MAX_TURNS,
    TOP_K,
    catalog_index,
    classify_constraint,
    coarse_category,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)
from tools.proxy_dataset import sha256_file
from tools.run_cv import _latency_summary, selection_score


INITIAL_BUYING = (
    "I'm looking for {category}. A key requirement is: {constraint}.",
    "Please find {category}; {constraint} is essential.",
    "I need {category}, and it must have: {constraint}.",
    "Help me choose {category}. Prioritize {constraint}.",
)
INITIAL_BROWSING = (
    "I'm looking for {category}, but I'm still exploring.",
    "I'm browsing {category} and have not decided on the details.",
    "Show me some {category}; I'm open to options.",
    "I want to explore {category} before narrowing it down.",
)
INITIAL_OVERRIDE = (
    "I'm looking for {category}. {old_value}",
    "Please help me find {category}. Initially, {old_value}",
    "I want {category}; for now, {old_value}",
    "Show me {category}. My earlier preference is: {old_value}",
)
NO_PREFERENCE = (
    "I don't have a preference for {attribute}; please use your judgment.",
    "No preference on {attribute}; choose what fits best.",
    "I'm flexible about {attribute}.",
    "Any {attribute} is fine with me.",
)
CONTROL_REPLY = (
    "Those options are not quite right yet. Ask me about one specific attribute.",
    "Please narrow this down by asking one concrete question.",
    "I need another direction; ask about a specific preference.",
    "Keep searching and ask me for one useful detail.",
)
DISCLOSURE_REPLY = (
    "For that, what matters is: {values}.",
    "Please prioritize {values}.",
    "My preference there is {values}.",
    "Use this requirement: {values}.",
)
NO_ADDITIONAL = (
    "I don't have an additional preference for {attribute}.",
    "Nothing else for {attribute}.",
    "I have no further requirement for {attribute}.",
    "No additional {attribute} preference.",
)
```

`ProxyDialogue` 使用 `dialogue_variant % 4`，实现与官方 `initial_message()` / `customer_reply()` 相同的 disclosed、Boundary 和 attribute 分类状态变化，只替换措辞：

```python
class ProxyDialogue:
    @staticmethod
    def _variant(sample: dict[str, object]) -> int:
        return int(sample.get("dialogue_variant", 0)) % 4

    def initial_message(
        self,
        sample: dict[str, object],
        category: str,
        disclosed: set[str],
    ) -> str:
        variant = self._variant(sample)
        scenario = str(sample["scenario_type"])
        card = dict(sample["intent_card"])
        hard = [str(value) for value in card.get("hard_constraints", [])]
        if scenario == "buying" and hard:
            disclosed.add(hard[0])
            return INITIAL_BUYING[variant].format(
                category=category,
                constraint=hard[0],
            )
        if scenario == "intent_override":
            override = dict(dict(sample["behavior"])["override"])
            return INITIAL_OVERRIDE[variant].format(
                category=category,
                old_value=str(override["old_value"]),
            )
        return INITIAL_BROWSING[variant].format(category=category)

    def customer_reply(
        self,
        sample: dict[str, object],
        ask_attribute: object,
        disclosed: set[str],
        boundary_used: bool,
    ) -> tuple[str, bool]:
        variant = self._variant(sample)
        attribute = ask_attribute if isinstance(ask_attribute, str) else None
        if (
            sample["scenario_type"] == "boundary"
            and not boundary_used
            and attribute
        ):
            return NO_PREFERENCE[variant].format(attribute=attribute), True
        if not attribute:
            return CONTROL_REPLY[variant], boundary_used
        if attribute not in ALLOWED_ATTRIBUTES:
            attribute = "other"
        card = dict(sample["intent_card"])
        constraints = [
            *[str(value) for value in card.get("hard_constraints", [])],
            *[str(value) for value in card.get("soft_preferences", [])],
        ]
        matches = [
            value
            for value in constraints
            if value not in disclosed
            and (attribute == "other" or classify_constraint(value) == attribute)
        ][:2]
        if not matches:
            return NO_ADDITIONAL[variant].format(attribute=attribute), boundary_used
        disclosed.update(matches)
        return (
            DISCLOSURE_REPLY[variant].format(values="; ".join(matches)),
            boundary_used,
        )


def evaluate_proxy(
    agent: object,
    samples: list[dict[str, object]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, object]],
) -> dict[str, object]:
    dialogue = ProxyDialogue()
    sessions: list[dict[str, object]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    invalid_response_count = 0
    for sample in samples:
        session_id = "proxy_eval_" + hashlib.sha256(
            str(sample["sample_id"]).encode("utf-8")
        ).hexdigest()[:20]
        agent.reset(session_id, dict(sample["user_profile"]))
        target = str(dict(sample["ground_truth"])["parent_asin"])
        effective_card, effective_behavior = materialize_hidden_fields(sample, products)
        effective_sample = {
            **sample,
            "intent_card": effective_card,
            "behavior": effective_behavior,
        }
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = dialogue.initial_message(
            effective_sample,
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        hit_turn: int | None = None
        best_rank: int | None = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:  # noqa: BLE001 - mirror official evaluator behavior.
                response = {"message": "", "ask_attribute": None, "recommendations": []}
                invalid_response_count += 1
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
                invalid_response_count += 1
            usage = response.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                    total_prompt_tokens += prompt_tokens
                if isinstance(completion_tokens, int) and completion_tokens >= 0:
                    total_completion_tokens += completion_tokens
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = dict(dict(effective_sample.get("behavior", {})).get("override") or {})
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(
                    override.get(
                        "message",
                        "Actually, please ignore my earlier preference.",
                    )
                )
            else:
                user_message, boundary_used = dialogue.customer_reply(
                    effective_sample,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )
        sessions.append(
            {
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "hit": hit_turn is not None,
                "first_hit_turn": hit_turn,
                "best_rank": best_rank,
                "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            }
        )

    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical_score = (
        0.50 * overall["hit_rate_at_10"]
        + 0.30 * overall["mrr"]
        + 0.20 * efficiency
    )
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "reported_token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        "scenario_metrics": {
            name: metric_summary(grouped[name]) for name in sorted(grouped)
        },
        "invalid_response_count": invalid_response_count,
        "sessions": sessions,
    }
```

保持上面 TechnicalScore 三项权重和 `efficiency` 表达式与官方 evaluator 完全相同；不要抽象出新的评分公式。

`write_audit_report()` 必须使用 exclusive create：

```python
def write_audit_report(destination: Path, result: dict, *, metadata: dict) -> None:
    payload = {
        **metadata,
        "aggregate": {key: value for key, value in result.items() if key != "sessions"},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
```

CLI 参数固定为 `--catalog`、`--proxy-root`、`--suite representative|stress`、`--folds`、`--audit-label baseline|final`、`--output`。新增以下加载和选择边界：

```python
def load_proxy_suite(proxy_root: Path, suite: str) -> tuple[dict, list[dict]]:
    manifest_path = proxy_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_name = f"{suite}.jsonl"
    dataset_path = proxy_root / dataset_name
    if sha256_file(dataset_path) != manifest["output_hashes"][dataset_name]:
        raise ValueError(f"proxy dataset hash mismatch: {dataset_name}")
    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest["counts"][suite]):
        raise ValueError(f"proxy dataset count mismatch: {suite}")
    return manifest, rows


def select_proxy_rows(
    rows: list[dict],
    suite: str,
    folds: list[int] | None,
    audit_label: str | None,
) -> list[tuple[int | None, list[dict]]]:
    if suite == "stress":
        if audit_label is not None:
            raise ValueError("stress suite cannot be used as audit")
        return [(None, rows)]
    if audit_label is not None:
        if folds not in (None, [5]):
            raise ValueError("audit is sealed to representative fold 5")
        selected = [row for row in rows if int(row["proxy_fold"]) == 5]
        if not selected:
            raise ValueError("representative audit fold is empty")
        return [(5, selected)]
    chosen = sorted(set(folds or [1, 2, 3, 4]))
    if not chosen or any(fold not in {1, 2, 3, 4} for fold in chosen):
        raise ValueError("development runs only allow representative folds 1-4")
    selected = [
        (fold, [row for row in rows if int(row["proxy_fold"]) == fold])
        for fold in chosen
    ]
    if any(not fold_rows for _fold, fold_rows in selected):
        raise ValueError("selected representative fold is empty")
    return selected


def write_report(destination: Path, payload: dict[str, object]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_agent(specification: str):
    module_name, class_name = specification.split(":", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _config_hash(agent: object) -> str:
    return hashlib.sha256(repr(getattr(agent, "config", None)).encode()).hexdigest()


def _fold_report(
    fold: int | None,
    agent: object,
    result: dict[str, object],
) -> dict[str, object]:
    trace_sink = getattr(agent, "traces", None)
    records = list(trace_sink.records) if hasattr(trace_sink, "records") else []
    return {
        "fold": fold,
        "sample_count": int(result["sample_count"]),
        "aggregate": {key: value for key, value in result.items() if key != "sessions"},
        "latency_ms": _latency_summary(
            [float(record.get("elapsed_ms", 0.0)) for record in records]
        ),
        "fallback_count": sum(bool(record.get("fallbacks")) for record in records),
        "route_distribution": dict(
            sorted(Counter(str(record.get("route", "unknown")) for record in records).items())
        ),
        "sessions": result["sessions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run guarded CompassCart proxy evaluation")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--suite", choices=("representative", "stress"), required=True)
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--audit-label", choices=("baseline", "final"))
    parser.add_argument("--agent", default="agent:Agent")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest, rows = load_proxy_suite(args.proxy_root, args.suite)
    selected = select_proxy_rows(rows, args.suite, args.folds, args.audit_label)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent_class = _load_agent(args.agent)
    fold_reports: list[dict[str, object]] = []
    config_hash = ""
    for fold, fold_rows in selected:
        agent = agent_class(args.catalog)
        config_hash = config_hash or _config_hash(agent)
        result = evaluate_proxy(agent, fold_rows, catalog_ids, categories, products)
        fold_reports.append(_fold_report(fold, agent, result))

    fallback_count = sum(int(report["fallback_count"]) for report in fold_reports)
    invalid_count = sum(
        int(dict(report["aggregate"])["invalid_response_count"])
        for report in fold_reports
    )
    metadata = {
        "commit": _git_commit(),
        "config_hash": config_hash,
        "manifest_hash": sha256_file(args.proxy_root / "manifest.json"),
        "dataset_hash": manifest["output_hashes"][f"{args.suite}.jsonl"],
        "suite": args.suite,
        "fallback_count": fallback_count,
        "invalid_response_count": invalid_count,
    }
    if fallback_count or invalid_count:
        raise RuntimeError("proxy run must have zero fallback and invalid responses")
    if args.audit_label is not None:
        result = dict(fold_reports[0]["aggregate"])
        write_audit_report(
            args.output,
            result,
            metadata={**metadata, "audit_label": args.audit_label},
        )
        return

    technical_scores = [
        float(dict(report["aggregate"])["recommended_technical_score"])
        for report in fold_reports
    ]
    sample_count = sum(int(report["sample_count"]) for report in fold_reports)
    report = {
        **metadata,
        "folds": fold_reports,
        "mean_technical_score": round(statistics.fmean(technical_scores), 6),
        "std_technical_score": round(statistics.pstdev(technical_scores), 6),
        "selection_score": selection_score(
            technical_scores,
            0.0,
            invalid_count / max(sample_count, 1),
        ),
    }
    write_report(args.output, report)


if __name__ == "__main__":
    main()
```

每折创建新的 Agent；stress 作为一个 `fold=None` 报告并保留 800 sessions。`--audit-label` 只走 exclusive-create `write_audit_report()`。不要增加读取公开 intent card、公开 scenario answer 或已知失败列表的代码路径。

- [ ] **Step 4: Run focused equivalence tests**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_run_proxy.py tests/unit/test_experiment_tools.py tests/test_evaluator.py -q
& .\.venv\Scripts\python.exe -m ruff check tools/run_proxy.py tests/unit/test_run_proxy.py
```

Expected: all pass; unchanged `tests/test_evaluator.py` confirms official metric behavior.

- [ ] **Step 5: Record development, stress and sealed baseline audit**

```powershell
& .\.venv\Scripts\python.exe -m tools.run_proxy --catalog data/catalog.jsonl --proxy-root var/balanced-hardening/proxy-v1 --suite representative --folds 1 2 3 4 --output var/balanced-hardening/proxy-v1/dev-baseline.json
& .\.venv\Scripts\python.exe -m tools.run_proxy --catalog data/catalog.jsonl --proxy-root var/balanced-hardening/proxy-v1 --suite stress --output var/balanced-hardening/proxy-v1/stress-baseline.json
& .\.venv\Scripts\python.exe -m tools.run_proxy --catalog data/catalog.jsonl --proxy-root var/balanced-hardening/proxy-v1 --suite representative --audit-label baseline --output var/balanced-hardening/proxy-v1/audit/baseline.json
```

Expected: dev has four folds and sessions; stress has 800 sessions; audit contains aggregate only. Do not run audit again and do not inspect per-target proxy misses.

- [ ] **Step 6: Commit**

```powershell
git add tools/run_proxy.py tests/unit/test_run_proxy.py tests/unit/test_experiment_tools.py
git commit -m "test: add guarded proxy evaluator"
```

### Task 4: Full-Catalog Benchmark Harness and Pre-R0 Evidence

**Files:**
- Create: `tools/benchmark_release.py`
- Create: `tests/unit/test_benchmark_release.py`
- Modify: `requirements-dev.txt`

- [ ] **Step 1: Add psutil and failing report tests**

在 `requirements-dev.txt` 添加 `psutil>=6,<8`，安装后创建测试：

```python
from tools.benchmark_release import aggregate_trials, validate_transcript


def test_trial_aggregation_uses_medians_and_nearest_rank_p95():
    trials = [
        {"init_ms": 10.0, "peak_mib": 100.0, "rss_mib": 80.0, "latencies_ms": [1.0, 2.0], "dense_available": True, "dense_status": "available", "fallback_count": 0, "response_hash": "same"},
        {"init_ms": 20.0, "peak_mib": 120.0, "rss_mib": 90.0, "latencies_ms": [2.0, 3.0], "dense_available": True, "dense_status": "available", "fallback_count": 0, "response_hash": "same"},
        {"init_ms": 30.0, "peak_mib": 110.0, "rss_mib": 85.0, "latencies_ms": [3.0, 4.0], "dense_available": True, "dense_status": "available", "fallback_count": 0, "response_hash": "same"},
    ]

    report = aggregate_trials(trials)

    assert report["median_init_ms"] == 20.0
    assert report["median_peak_mib"] == 110.0
    assert report["latency_ms"]["p95"] == 4.0
    assert report["dense_available"] is True


def test_transcript_requires_exactly_800_ordered_responses():
    rows = [{"session_id": f"S{index // 4}", "turn": index % 4 + 1, "profile": {}, "message": "query"} for index in range(800)]
    validate_transcript(rows)
```

- [ ] **Step 2: Verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_benchmark_release.py -q
```

Expected: import fails because `tools.benchmark_release` does not exist.

- [ ] **Step 3: Implement worker and parent modes**

创建 `tools/benchmark_release.py`，使用以下完整核心 API。worker 只打印一个 JSON 对象，parent 才写报告：

```python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import psutil

from evaluator.local_evaluator import catalog_index, normalize_recommendations
from tools.proxy_dataset import sha256_file, stable_int
from tools.run_cv import _latency_summary
from tools.run_proxy import ProxyDialogue, load_proxy_suite


def validate_transcript(rows: list[dict[str, object]]) -> None:
    if len(rows) != 800:
        raise ValueError("benchmark transcript must contain exactly 800 responses")
    grouped: dict[str, list[int]] = defaultdict(list)
    active_session: str | None = None
    closed: set[str] = set()
    for row in rows:
        session_id = row.get("session_id")
        turn = row.get("turn")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("transcript session_id must be a non-empty string")
        if not isinstance(turn, int):
            raise ValueError("transcript turn must be an integer")
        if not isinstance(row.get("profile"), dict):
            raise ValueError("transcript profile must be an object")
        if not isinstance(row.get("message"), str) or not str(row["message"]).strip():
            raise ValueError("transcript message must be a non-empty string")
        if active_session != session_id:
            if session_id in closed:
                raise ValueError("transcript sessions must be contiguous")
            if active_session is not None:
                closed.add(active_session)
            active_session = session_id
        grouped[session_id].append(turn)
    if len(grouped) != 200:
        raise ValueError("benchmark transcript must contain exactly 200 sessions")
    if any(turns != [1, 2, 3, 4] for turns in grouped.values()):
        raise ValueError("every benchmark session must contain ordered turns 1-4")


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _memory_mib() -> tuple[float, float]:
    info = psutil.Process().memory_info()
    rss = float(getattr(info, "rss")) / (1024 * 1024)
    peak = float(getattr(info, "peak_wset", getattr(info, "rss"))) / (1024 * 1024)
    return rss, peak


def aggregate_trials(trials: list[dict[str, object]]) -> dict[str, object]:
    if not trials:
        raise ValueError("at least one benchmark trial is required")
    response_hashes = {str(trial["response_hash"]) for trial in trials}
    if len(response_hashes) != 1:
        raise ValueError("benchmark trials produced different responses")
    latencies = [
        float(value)
        for trial in trials
        for value in list(trial["latencies_ms"])
    ]
    return {
        "trial_count": len(trials),
        "median_init_ms": round(
            statistics.median(float(trial["init_ms"]) for trial in trials), 3
        ),
        "median_peak_mib": round(
            statistics.median(float(trial["peak_mib"]) for trial in trials), 3
        ),
        "median_rss_mib": round(
            statistics.median(float(trial["rss_mib"]) for trial in trials), 3
        ),
        "latency_ms": _latency_summary(latencies),
        "dense_available": all(bool(trial["dense_available"]) for trial in trials),
        "dense_statuses": sorted({str(trial["dense_status"]) for trial in trials}),
        "fallback_count": sum(int(trial["fallback_count"]) for trial in trials),
        "response_hash": response_hashes.pop(),
        "trials": trials,
    }


def compare_reports(candidate: dict, baseline: dict) -> dict[str, object]:
    same_output = candidate["response_hash"] == baseline["response_hash"]
    candidate_p95 = float(dict(candidate["latency_ms"])["p95"])
    baseline_p95 = float(dict(baseline["latency_ms"])["p95"])
    candidate_max = float(dict(candidate["latency_ms"])["max"])
    metrics = {
        "p95": (candidate_p95, baseline_p95),
        "init": (float(candidate["median_init_ms"]), float(baseline["median_init_ms"])),
        "peak": (float(candidate["median_peak_mib"]), float(baseline["median_peak_mib"])),
    }
    no_regression = all(current <= previous * 1.05 for current, previous in metrics.values())
    material_gain = (
        candidate_p95 <= baseline_p95 * 0.90
        or float(candidate["median_init_ms"]) <= float(baseline["median_init_ms"]) * 0.95
        or float(candidate["median_peak_mib"]) <= float(baseline["median_peak_mib"]) * 0.95
    )
    accepted = same_output and no_regression and material_gain and candidate_max < 1500.0
    return {
        "accepted": accepted,
        "same_output": same_output,
        "no_metric_regressed_over_5_percent": no_regression,
        "material_gain": material_gain,
        "max_below_1500_ms": candidate_max < 1500.0,
    }
```

transcript 捕获固定为代表性开发折中 stable-hash 最小的 200 个 session；每个 session 无论推荐是否命中都捕获四轮，不能根据结果提前终止：

```python
def capture_transcript(proxy_root: Path, catalog_path: Path, output: Path) -> str:
    _manifest, rows = load_proxy_suite(proxy_root, "representative")
    eligible = [row for row in rows if int(row["proxy_fold"]) in {1, 2, 3, 4}]
    selected = sorted(
        eligible,
        key=lambda row: stable_int(20260829, str(row["sample_id"])),
    )[:200]
    if len(selected) != 200:
        raise ValueError("benchmark capture requires 200 development sessions")

    from agent import Agent

    agent = Agent(catalog_path)
    dialogue = ProxyDialogue()
    transcript: list[dict[str, object]] = []
    for position, sample in enumerate(selected, start=1):
        session_id = f"bench_{position:04d}"
        profile = dict(sample["user_profile"])
        agent.reset(session_id, profile)
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = dialogue.initial_message(
            sample,
            str(sample["category_bucket"]),
            disclosed,
        )
        for turn in range(1, 5):
            transcript.append(
                {
                    "session_id": session_id,
                    "turn": turn,
                    "profile": profile,
                    "message": message,
                }
            )
            response = agent.respond(session_id, message, turn, 10)
            if turn == 4:
                continue
            override = dict(dict(sample.get("behavior", {})).get("override") or {})
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(override["message"])
            else:
                message, boundary_used = dialogue.customer_reply(
                    sample,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )
    validate_transcript(transcript)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in transcript:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return sha256_file(output)
```

worker 断言 catalog 恰好 50,000 IDs；Agent 初始化计时包含完整构造，响应计时读取 trace，响应哈希只包含规范化 ID 顺序：

```python
def run_worker(catalog_path: Path, transcript_path: Path) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_transcript(rows)
    catalog_ids, _categories, _products = catalog_index(catalog_path)
    if len(catalog_ids) != 50_000:
        raise ValueError("release benchmark requires the full 50,000-product catalog")

    from agent import Agent

    started = time.perf_counter()
    agent = Agent(catalog_path)
    init_ms = (time.perf_counter() - started) * 1_000
    latencies: list[float] = []
    responses: list[list[str]] = []
    fallback_count = 0
    for row in rows:
        session_id = str(row["session_id"])
        turn = int(row["turn"])
        if turn == 1:
            agent.reset(session_id, dict(row["profile"]))
        response = agent.respond(session_id, str(row["message"]), turn, 10)
        responses.append(normalize_recommendations(response.get("recommendations"), catalog_ids))
        trace = agent.traces.records[-1]
        latencies.append(float(trace["elapsed_ms"]))
        fallback_count += int(bool(trace.get("fallbacks")))
    rss_mib, peak_mib = _memory_mib()
    return {
        "init_ms": round(init_ms, 3),
        "peak_mib": round(peak_mib, 3),
        "rss_mib": round(rss_mib, 3),
        "latencies_ms": latencies,
        "dense_available": bool(agent.dense.available),
        "dense_status": str(getattr(agent.dense, "status", "unknown")),
        "fallback_count": fallback_count,
        "response_hash": _canonical_hash(responses),
    }


def run_parent(
    catalog_path: Path,
    transcript_path: Path,
    *,
    trials: int,
    cwd_mode: str,
) -> dict[str, object]:
    if trials < 1:
        raise ValueError("trials must be positive")
    root = Path(__file__).resolve().parents[1]
    catalog_path = catalog_path.resolve()
    transcript_path = transcript_path.resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="compasscart-benchmark-") as outside:
        cwd = root if cwd_mode == "root" else Path(outside)
        for _trial in range(trials):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.benchmark_release",
                    "--worker",
                    "--catalog",
                    str(catalog_path),
                    "--transcript",
                    str(transcript_path),
                ],
                cwd=cwd,
                env=environment,
                check=True,
                text=True,
                capture_output=True,
            )
            results.append(json.loads(completed.stdout))
    return {
        **aggregate_trials(results),
        "catalog_hash": sha256_file(catalog_path),
        "transcript_hash": sha256_file(transcript_path),
        "cwd_mode": cwd_mode,
        "platform": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
    }
```

三种互斥模式使用以下入口。parent 先 exclusive-create 写完整证据，再执行资源门禁：

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a CompassCart release")
    parser.add_argument("--capture-transcript", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--proxy-root", type=Path)
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cwd-mode", choices=("root", "outside"), default="outside")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--allow-dense-unavailable", action="store_true")
    args = parser.parse_args()
    if args.capture_transcript and args.worker:
        parser.error("capture and worker modes are mutually exclusive")
    if args.capture_transcript:
        if args.proxy_root is None or args.output is None:
            parser.error("capture mode requires --proxy-root and --output")
        digest = capture_transcript(args.proxy_root, args.catalog, args.output)
        print(json.dumps({"responses": 800, "sha256": digest}, sort_keys=True))
        return
    if args.worker:
        if args.transcript is None:
            parser.error("worker mode requires --transcript")
        print(
            json.dumps(
                run_worker(args.catalog.resolve(), args.transcript.resolve()),
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return
    if args.transcript is None or args.output is None:
        parser.error("parent mode requires --transcript and --output")

    report = run_parent(
        args.catalog,
        args.transcript,
        trials=args.trials,
        cwd_mode=args.cwd_mode,
    )
    if args.compare is not None:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        report["comparison"] = compare_reports(report, baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    failures: list[str] = []
    if not report["dense_available"] and not args.allow_dense_unavailable:
        failures.append("dense unavailable")
    if int(report["fallback_count"]):
        failures.append("fallback count is non-zero")
    if float(dict(report["latency_ms"])["max"]) >= 1500.0:
        failures.append("response max is not below 1.5 seconds")
    if args.compare is not None and not dict(report["comparison"])["accepted"]:
        failures.append("comparison gate rejected candidate")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
```

`--allow-dense-unavailable` 只允许 Dense 状态门禁失效，不放宽 fallback、响应哈希或时延门禁。

- [ ] **Step 4: Run focused tests and capture transcript**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_benchmark_release.py -q
& .\.venv\Scripts\python.exe -m ruff check tools/benchmark_release.py tests/unit/test_benchmark_release.py
& .\.venv\Scripts\python.exe -m tools.benchmark_release --capture-transcript --proxy-root var/balanced-hardening/proxy-v1 --catalog data/catalog.jsonl --output var/balanced-hardening/benchmark-transcript.jsonl
```

Expected: tests pass; transcript validator reports 800 responses and a SHA-256.

- [ ] **Step 5: Capture pre-R0 root and expected outside-CWD evidence**

```powershell
& .\.venv\Scripts\python.exe -m tools.benchmark_release --catalog data/catalog.jsonl --transcript var/balanced-hardening/benchmark-transcript.jsonl --trials 3 --cwd-mode root --output var/balanced-hardening/benchmark-pre-r0-root.json
& .\.venv\Scripts\python.exe -m tools.benchmark_release --catalog data/catalog.jsonl --transcript var/balanced-hardening/benchmark-transcript.jsonl --trials 1 --cwd-mode outside --allow-dense-unavailable --output var/balanced-hardening/benchmark-pre-r0-outside.json
```

Expected: root report has `dense_available=true`; outside report has `dense_available=false`. If outside is already true, stop and reconcile the design assumption before R0.

- [ ] **Step 6: Commit**

```powershell
git add requirements-dev.txt tools/benchmark_release.py tests/unit/test_benchmark_release.py
git commit -m "perf: add full-catalog release benchmark"
```

### Task 5: R0 CWD-Independent Dense Assets and Status Reasons

**Files:**
- Create: `tests/unit/test_config.py`
- Modify: `src/compasscart/config.py:8`
- Modify: `src/compasscart/agent.py:24`
- Modify: `src/compasscart/dense.py:20`
- Modify: `tests/unit/test_dense.py:42`
- Modify: `tests/contract/test_submission_package.py:36`

- [ ] **Step 1: Write failing config, status and extracted-CWD tests**

`tests/unit/test_config.py`：

```python
from pathlib import Path

import pytest

from compasscart.config import RuntimeConfig


def test_relative_dense_paths_resolve_from_valid_submission_root(tmp_path):
    (tmp_path / "agent.py").write_text("", encoding="utf-8")
    (tmp_path / "assets").mkdir()

    paths = RuntimeConfig().resolve_dense_paths(tmp_path)

    assert paths == (
        tmp_path / "assets/model",
        tmp_path / "assets/product_vectors",
        tmp_path / "assets/SHA256SUMS",
    )


def test_absolute_dense_paths_are_preserved(tmp_path):
    absolute = (tmp_path / "custom").resolve()
    config = RuntimeConfig(dense_model_dir=absolute, dense_vector_dir=absolute, dense_manifest_path=absolute / "SHA256SUMS")
    root = tmp_path / "submission"
    root.mkdir()
    (root / "agent.py").write_text("", encoding="utf-8")
    (root / "assets").mkdir()

    assert config.resolve_dense_paths(root) == (absolute, absolute, absolute / "SHA256SUMS")


def test_invalid_submission_layout_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="submission root"):
        RuntimeConfig().resolve_dense_paths(tmp_path)
```

更新 Dense 测试，要求 Null backend 的 `status` 分别为 `asset_missing`、`asset_invalid`、`disabled_by_environment`；Onnx backend 为 `available`。

在 submission contract 添加 subprocess 测试：解压 ZIP 到 `tmp_path / "release"`，CWD 设为另一个 `tmp_path / "outside"`，`PYTHONPATH` 只含解压 root/src 和 root，使用绝对 fixture catalog 构造 Agent，打印 JSON，并断言 `dense_available=true`、`dense_status="available"`。

- [ ] **Step 2: Verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_config.py tests/unit/test_dense.py tests/contract/test_submission_package.py -q
```

Expected: config method/status assertions fail；extracted outside-CWD smoke reports Dense unavailable.

- [ ] **Step 3: Implement path resolution and classified backend status**

在 `RuntimeConfig` 添加：

```python
def resolve_dense_paths(self, submission_root: Path) -> tuple[Path, Path, Path]:
    root = submission_root.resolve()
    if not (root / "agent.py").is_file() or not (root / "assets").is_dir():
        raise ValueError("submission root must contain agent.py and assets")

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    return (
        resolve(self.dense_model_dir),
        resolve(self.dense_vector_dir),
        resolve(self.dense_manifest_path),
    )
```

在 `dense.py`：

```python
class NullDenseBackend:
    available = False

    def __init__(self, status: str = "unavailable") -> None:
        self.status = status

    def search(self, text: str, limit: int) -> list[Candidate]:
        del text, limit
        return []
```

`OnnxDenseBackend.status = "available"`。`load_dense_backend()` 按顺序分类：环境禁用、FileNotFoundError=`asset_missing`、ImportError=`dependency_missing`、ValueError=`asset_invalid`、其他=`initialization_failed`。

在 `agent.py` 顶层定义：

```python
SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
```

Agent 初始化时调用 `self.config.resolve_dense_paths(SUBMISSION_ROOT)`；layout ValueError 时使用 `NullDenseBackend("layout_invalid")`，不能尝试 CWD。Trace 增加 `dense_status`，不改变官方 response schema。

- [ ] **Step 4: Run focused and full tests**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_config.py tests/unit/test_dense.py tests/contract/test_submission_package.py tests/contract/test_agent_contract.py -q
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m ruff check src tests tools agent.py
```

Expected: full suite passes, extracted subprocess Dense available, Ruff clean.

- [ ] **Step 5: Commit R0**

```powershell
git add src/compasscart/config.py src/compasscart/agent.py src/compasscart/dense.py tests/unit/test_config.py tests/unit/test_dense.py tests/contract/test_submission_package.py
git commit -m "fix: resolve dense assets from submission root"
```

- [ ] **Step 6: Capture the formal R0 benchmark baseline**

```powershell
& .\.venv\Scripts\python.exe -m tools.benchmark_release --catalog data/catalog.jsonl --transcript var/balanced-hardening/benchmark-transcript.jsonl --trials 3 --cwd-mode outside --output var/balanced-hardening/benchmark-r0.json
```

Expected: all three trials `dense_available=true`, fallback count 0, max response below 1.5 s. This file is the only P0 resource baseline.

### Task 6: Cached Popularity and Lazy Fallback

**Files:**
- Modify: `src/compasscart/catalog.py:87,243`
- Modify: `src/compasscart/retrieval.py:43`
- Modify: `tests/unit/test_catalog.py`
- Modify: `tests/unit/test_retrieval.py`

- [ ] **Step 1: Write RED tests for no repeated sort and no unnecessary fallback**

```python
def test_popular_ids_use_the_frozen_catalog_order(fixture_catalog_path):
    index = CatalogIndex(fixture_catalog_path)
    expected = index.popular_ids(4)

    class NoReadQuality(dict):
        def __getitem__(self, key):
            raise AssertionError(f"quality was re-read for {key}")

    index.quality = NoReadQuality(index.quality)
    assert index.popular_ids(4) == expected


def test_retrieval_skips_fallback_when_fused_exact_ids_fill_desired(fixture_catalog_path, monkeypatch):
    class AllDense:
        available = True

        def search(self, _text, _limit):
            return [Candidate(identifier) for identifier in sorted(index.valid_ids)]

    index = CatalogIndex(fixture_catalog_path)
    retriever = HybridRetriever(index, AllDense())
    plan = RetrievalPlan(route="browsing", query_text="query", source_weights=(("dense", 1.0),))
    monkeypatch.setattr(retriever, "_fallback_ids", lambda _plan: (_ for _ in ()).throw(AssertionError("fallback should be lazy")))

    assert len(retriever.retrieve(plan)) == len(index.valid_ids)
```

- [ ] **Step 2: Verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_catalog.py::test_popular_ids_use_the_frozen_catalog_order tests/unit/test_retrieval.py::test_retrieval_skips_fallback_when_fused_exact_ids_fill_desired -q
```

Expected: both fail on current eager behavior.

- [ ] **Step 3: Implement exact-output caching/laziness**

在 Catalog `_load()` 完成 quality 后一次性设置：

```python
self._popular_order = tuple(
    sorted(self.valid_ids, key=lambda item: (-self.quality[item], item))
)
```

并替换：

```python
def popular_ids(self, limit: int = 10) -> list[str]:
    return list(self._popular_order[: max(int(limit), 0)])
```

Retriever 将 fallback 改为惰性闭包，只在 `len(fused_ids) < desired`、relaxation 或 exhausted-exclusion 路径调用，且每个 retrieve 最多计算一次：

```python
fallback_ids: list[str] | None = None

def fallback() -> list[str]:
    nonlocal fallback_ids
    if fallback_ids is None:
        fallback_ids = self._fallback_ids(plan)
    return fallback_ids
```

- [ ] **Step 4: Verify output equivalence**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_catalog.py tests/unit/test_retrieval.py tests/integration/test_scenarios.py tests/integration/test_fallbacks.py -q
& .\.venv\Scripts\python.exe -m ruff check src/compasscart/catalog.py src/compasscart/retrieval.py tests/unit/test_catalog.py tests/unit/test_retrieval.py
```

Expected: pass; no recommendation ordering test changes.

- [ ] **Step 5: Commit**

```powershell
git add src/compasscart/catalog.py src/compasscart/retrieval.py tests/unit/test_catalog.py tests/unit/test_retrieval.py
git commit -m "perf: cache popularity and defer fallback"
```

### Task 7: Bounded MMR Cache and Catalog Attribute Reuse

**Files:**
- Modify: `src/compasscart/ranker.py:182`
- Modify: `src/compasscart/question_policy.py:30`
- Modify: `src/compasscart/agent.py:48`
- Modify: `tests/unit/test_ranker.py`
- Modify: `tests/unit/test_question_policy.py`

- [ ] **Step 1: Write RED cache/reuse tests**

Ranker 测试断言同一 ID 第二次返回缓存对象且实例缓存最多 4096 项；QuestionPolicy 测试给 product 空 details、lookup 含 material，且 monkeypatch `extract_attributes` 抛错，仍选择 material。

```python
def test_diversity_terms_use_a_bounded_cache(fixture_catalog_path):
    ranker = ConstraintRanker(CatalogIndex(fixture_catalog_path))
    first = ranker._diversity_terms("SHOE1")
    second = ranker._diversity_terms("SHOE1")

    assert first is second
    assert list(ranker._diversity_cache) == ["SHOE1"]
    for index in range(4096):
        ranker._diversity_terms(f"MISSING{index:04d}")
    assert len(ranker._diversity_cache) == 4096
    assert "SHOE1" not in ranker._diversity_cache


def test_policy_reuses_catalog_attributes(monkeypatch):
    lookup = {
        f"P{index:02d}": {"material": ("cotton" if index < 6 else "leather",)}
        for index in range(12)
    }
    candidates = [Candidate(f"P{index:02d}", product={}, score=12.0 - index) for index in range(12)]
    monkeypatch.setattr("compasscart.question_policy.extract_attributes", lambda _product: (_ for _ in ()).throw(AssertionError("must reuse catalog attributes")))

    decision = QuestionPolicy(lookup).choose(candidates, SessionState("s1", turn=2))

    assert decision.ask_attribute == "material"
```

- [ ] **Step 2: Verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_ranker.py::test_diversity_terms_use_a_bounded_cache tests/unit/test_question_policy.py::test_policy_reuses_catalog_attributes -q
```

Expected: `_diversity_cache` is missing and constructor rejects lookup.

- [ ] **Step 3: Implement bounded reuse**

在 Ranker 添加 `from collections import OrderedDict`；构造时添加
`self._diversity_cache: OrderedDict[str, frozenset[str]] = OrderedDict()`，并修改：

```python
def _diversity_terms(self, identifier: str) -> frozenset[str]:
    if identifier in self._diversity_cache:
        cached = self._diversity_cache.pop(identifier)
        self._diversity_cache[identifier] = cached
        return cached
    attributes = self.catalog.attributes.get(identifier, {})
    values = (
        *attributes.get("category", ()),
        *attributes.get("material", ()),
        *attributes.get("style", ()),
        *attributes.get("use_case", ()),
    )
    result = frozenset(token for value in values for token in terms(value))
    self._diversity_cache[identifier] = result
    if len(self._diversity_cache) > 4096:
        self._diversity_cache.popitem(last=False)
    return result
```

QuestionPolicy constructor接收可选只读 lookup；ID 不存在时保持旧 `extract_attributes(product)` fallback：

```python
def __init__(self, attribute_lookup=None) -> None:
    self.attribute_lookup = attribute_lookup

def _candidate_attributes(self, candidate: Candidate):
    if self.attribute_lookup is not None:
        attributes = self.attribute_lookup.get(candidate.parent_asin)
        if attributes is not None:
            return attributes
    return extract_attributes(candidate.product)
```

Agent 改为 `self.question_policy = QuestionPolicy(self.catalog.attributes)`。

- [ ] **Step 4: Run equivalence tests**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_ranker.py tests/unit/test_question_policy.py tests/contract/test_agent_contract.py -q
& .\.venv\Scripts\python.exe -m ruff check src/compasscart/ranker.py src/compasscart/question_policy.py src/compasscart/agent.py tests/unit/test_ranker.py tests/unit/test_question_policy.py
```

Expected: pass; existing QuestionPolicy tests using no lookup remain unchanged.

- [ ] **Step 5: Commit**

```powershell
git add src/compasscart/ranker.py src/compasscart/question_policy.py src/compasscart/agent.py tests/unit/test_ranker.py tests/unit/test_question_policy.py
git commit -m "perf: reuse bounded ranking semantics"
```

### Task 8: Streaming Integrity, Dense mmap, and Backend Circuit Breakers

**Files:**
- Create: `src/compasscart/integrity.py`
- Modify: `src/compasscart/dense.py:28,105,150`
- Modify: `src/compasscart/catalog.py:109,247`
- Modify: `tools/package_submission.py:71`
- Modify: `tests/unit/test_dense.py`
- Modify: `tests/unit/test_catalog.py`

- [ ] **Step 1: Write RED integrity/mmap/failure tests**

测试 `sha256_file()` 与 hashlib 一致且不调用 `Path.read_bytes()`；构造三次失败/成功重置 session；用 `np.load(..., mmap_mode="r")` 临时数组断言 backend shares memory；FTS 连续三次 OperationalError 后才禁用。

```python
def test_single_dense_failure_does_not_permanently_disable_backend():
    class FlakySession(_Session):
        calls = 0

        def run(self, outputs, inputs):
            self.calls += 1
            if self.calls in {1, 3, 4, 5}:
                raise RuntimeError("transient")
            return super().run(outputs, inputs)

    backend = OnnxDenseBackend(
        FlakySession(),
        _Tokenizer(),
        product_ids=np.array(["A"]),
        vectors=np.array([[127, 0]], dtype=np.int8),
        scales=np.array([1 / 127], dtype=np.float32),
        failure_limit=3,
    )

    assert backend.search("query", 1) == []
    assert backend.available is True
    assert backend.search("query", 1)
    assert backend.search("query", 1) == []
    assert backend.search("query", 1) == []
    assert backend.search("query", 1) == []
    assert backend.available is False
```

- [ ] **Step 2: Verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_dense.py tests/unit/test_catalog.py -q
```

Expected: new tests fail because runtime uses eager reads/copies and first-failure disable.

- [ ] **Step 3: Implement streaming hash and mmap**

创建 `integrity.py`：

```python
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

Dense `_verify_manifest()` 和 package `_verify_assets()` 使用该 helper。`load_dense_backend()` 使用：

```python
product_ids=np.load(product_ids_path, mmap_mode="r", allow_pickle=False)
vectors=np.load(vectors_path, mmap_mode="r", allow_pickle=False)
scales=np.load(scales_path, mmap_mode="r", allow_pickle=False)
```

Onnx backend 不再 `product_ids.astype(str)`；先校验 dtype kind 为 `U`，保留 memmap。

- [ ] **Step 4: Implement three-consecutive-failure breakers**

Onnx backend 初始化 `_failure_limit` 和 `_consecutive_failures`；成功设 0，异常加一，达到上限才 `_available=False`。Catalog 初始化 `_fts_failures=0`；FTS 查询成功清零，OperationalError 加一，达到 3 才 `_fts_enabled=False`，当前调用始终转 fallback search。

- [ ] **Step 5: Run focused/full tests and package contract**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_dense.py tests/unit/test_catalog.py tests/contract/test_submission_package.py -q
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m ruff check src tests tools agent.py
```

Expected: all pass; assets hash unchanged; package reproducible.

- [ ] **Step 6: Commit**

```powershell
git add src/compasscart/integrity.py src/compasscart/dense.py src/compasscart/catalog.py tools/package_submission.py tests/unit/test_dense.py tests/unit/test_catalog.py
git commit -m "perf: stream assets and tolerate transient backends"
```

### Task 9: Cooperative Runtime Budget

**Files:**
- Modify: `src/compasscart/agent.py:67`
- Modify: `src/compasscart/retrieval.py:43`
- Modify: `src/compasscart/ranker.py:24`
- Modify: `tests/unit/test_retrieval.py`
- Modify: `tests/unit/test_ranker.py`
- Modify: `tests/integration/test_fallbacks.py`

- [ ] **Step 1: Write RED budget tests**

Retriever 使用已过期 deadline 时不得调用 Dense，但必须返回合法 lexical/attribute 候选；Browsing Ranker 在已过期 deadline 时不调用 `_mmr`；Agent trace 必须记录 `budget_skips`。

```python
def test_expired_budget_skips_dense_without_losing_valid_candidates(fixture_catalog_path):
    class FailingDense:
        available = True

        def search(self, text, limit):
            raise AssertionError("dense should be skipped")

    diagnostics: list[str] = []
    index = CatalogIndex(fixture_catalog_path)
    plan = RetrievalPlan(route="buying", query_text="blue shoes")
    candidates = HybridRetriever(index, FailingDense()).retrieve(
        plan,
        deadline=0.0,
        diagnostics=diagnostics,
    )

    assert candidates
    assert "dense_budget" in diagnostics
```

- [ ] **Step 2: Verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_retrieval.py::test_expired_budget_skips_dense_without_losing_valid_candidates tests/unit/test_ranker.py tests/integration/test_fallbacks.py -q
```

Expected: `retrieve()` rejects deadline/diagnostics and trace lacks key.

- [ ] **Step 3: Implement cooperative checks without background threads**

Agent 在 `started` 后计算：

```python
deadline = started + max(self.config.component_timeout_ms, 0) / 1000.0
budget_skips: list[str] = []
```

真实 HybridRetriever 调用传 `deadline`/`diagnostics`；legacy injected retriever 保持现有兼容路径。Retriever 完成 lexical/attribute/profile 后，只有 `time.perf_counter() < deadline` 才执行 Dense，否则追加 `dense_budget`。Ranker 完成确定性 score/sort 后，只有 deadline 未过期才运行 Browsing MMR，否则返回未多样化 exact+relaxed 顺序并追加 `mmr_budget`。Trace 增加 `budget_skips`；这些跳过不计入错误 `fallbacks`。

- [ ] **Step 4: Run budget, contract and performance tests**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_retrieval.py tests/unit/test_ranker.py tests/integration/test_fallbacks.py tests/contract/test_agent_contract.py tests/performance/test_runtime.py -q
& .\.venv\Scripts\python.exe -m ruff check src/compasscart/agent.py src/compasscart/retrieval.py src/compasscart/ranker.py tests/unit/test_retrieval.py tests/unit/test_ranker.py tests/integration/test_fallbacks.py
```

Expected: pass; normal tests have no budget skip; forced-expiry path remains contract-valid.

- [ ] **Step 5: Commit**

```powershell
git add src/compasscart/agent.py src/compasscart/retrieval.py src/compasscart/ranker.py tests/unit/test_retrieval.py tests/unit/test_ranker.py tests/integration/test_fallbacks.py
git commit -m "perf: enforce cooperative retrieval budget"
```

### Task 10: P0 Benchmark Gate, Frozen Input Audit, and Foundation Report

**Files:**
- Create: `reports/final/balanced-hardening-foundation-results.md`
- Modify: `README.md:8`

- [ ] **Step 1: Run all static and behavioral verification**

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m ruff check src tests tools agent.py
git diff --check
```

Expected: all tests pass, Ruff clean, diff check clean.

- [ ] **Step 2: Re-run frozen input hashes and fail on any mismatch**

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'evaluator\local_evaluator.py','data\public_set.jsonl','data\catalog.jsonl','assets\SHA256SUMS','assets\model\model.int8.onnx','assets\model\tokenizer.json','assets\product_vectors\product_ids.npy','assets\product_vectors\scales.npy','assets\product_vectors\vectors.int8.npy'
```

Expected: exactly the nine hashes in Scope Guard. Any mismatch blocks completion.

- [ ] **Step 3: Run P0 benchmark and compare with R0**

```powershell
& .\.venv\Scripts\python.exe -m tools.benchmark_release --catalog data/catalog.jsonl --transcript var/balanced-hardening/benchmark-transcript.jsonl --trials 3 --cwd-mode outside --output var/balanced-hardening/benchmark-p0.json --compare var/balanced-hardening/benchmark-r0.json
```

Expected: Dense true and fallback 0 in all trials; output response IDs equal R0 transcript snapshot; P95 improves at least 10% or peak/init improves at least 5%; no other median regresses more than 5%; max response below 1.5 s. If gate fails, revert only the P0 commit that failed its isolated comparison and rerun.

- [ ] **Step 4: Re-run proxy development and stress regression, not audit**

```powershell
& .\.venv\Scripts\python.exe -m tools.run_proxy --catalog data/catalog.jsonl --proxy-root var/balanced-hardening/proxy-v1 --suite representative --folds 1 2 3 4 --output var/balanced-hardening/proxy-v1/dev-p0.json
& .\.venv\Scripts\python.exe -m tools.run_proxy --catalog data/catalog.jsonl --proxy-root var/balanced-hardening/proxy-v1 --suite stress --output var/balanced-hardening/proxy-v1/stress-p0.json
```

Expected: exact same aggregate and session outputs as baseline because R0/P0 does not change healthy-host ranking; no fallback. Do not run `--audit-label final` in this foundation plan.

- [ ] **Step 5: Build and independently smoke the competition package**

```powershell
& .\.venv\Scripts\python.exe -m tools.package_submission
& .\.venv\Scripts\python.exe -m pytest tests/contract/test_submission_package.py -q
```

Expected: deterministic ZIP, secret scan clean, extracted arbitrary-CWD Dense available, no evaluator/public_set/catalog in package.

- [ ] **Step 6: Write the foundation evidence report**

`reports/final/balanced-hardening-foundation-results.md` 必须包含：runtime commit、九个冻结 hash、proxy manifest/output hashes、dev/stress aggregate、audit baseline aggregate（无 sessions）、pre-R0 outside failure、R0/P0 三次原始值与中位数、等价输出 hash、pytest/Ruff/package 结果、Windows 环境和 `macOS verification pending`。结论只能是 `Foundation accepted` 或列出失败门槛；不得声称分数已提高。

README 增加任意 CWD 资产解析说明和 benchmark 命令，明确 benchmark/psutil 不进入比赛运行依赖。

- [ ] **Step 7: Commit evidence and documentation**

```powershell
git add reports/final/balanced-hardening-foundation-results.md README.md
git commit -m "docs: record hardening foundation evidence"
```

- [ ] **Step 8: Review first-plan completion before S1-S5**

Run:

```powershell
git status --short
git log --oneline --decorate -12
```

Expected: only the pre-existing user DOCX and separately generated score JSON/Markdown remain untracked; no proxy/benchmark files are tracked. Request code review. Only after review approves and foundation report says `Foundation accepted`, write the second implementation plan for S1-S5 score behavior.
