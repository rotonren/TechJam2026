"""Deterministic sampling helpers for proxy product datasets."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import random
import tempfile
from collections import Counter
from dataclasses import dataclass
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
SCENARIO_WEIGHTS = {
    "buying": 0.40,
    "browsing": 0.40,
    "intent_override": 0.15,
    "boundary": 0.05,
}
GENERATOR_VERSION = 1
REPRESENTATIVE_SEED = 20260826
STRESS_SEED = 20260827
PROFILE_SEED = 20260828
DIMENSION_NAMES = ("category", "price", "popularity", "completeness")


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
    """Return a stable, seed-specific integer derived from SHA-256."""
    payload = f"{seed}\0{value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")


def representative_ids(
    records: list[ProxyProduct], *, count: int, seed: int
) -> list[str]:
    """Select a deterministic sample that covers each category and marginal."""
    prepared, frequencies, category_best = _prepare_records(records, seed)
    _validate_representative_count(count, len(prepared), len(category_best))

    selected_indices = set(category_best.values())
    quotas = _marginal_quotas(frequencies, count, seed)
    deficits = quotas.copy()
    for index in selected_indices:
        for dimension in prepared[index][1]:
            if deficits[dimension] > 0:
                deficits[dimension] -= 1

    cells: dict[tuple[tuple[str, str], ...], list[int]] = {}
    for index, (_, dimensions, _) in enumerate(prepared):
        cells.setdefault(dimensions, []).append(index)
    for indices in cells.values():
        indices.sort(key=lambda index: (prepared[index][2], prepared[index][0]))

    positions = {dimensions: 0 for dimensions in cells}
    heap: list[tuple[int, int, str, tuple[tuple[str, str], ...]]] = []

    def enqueue(dimensions: tuple[tuple[str, str], ...]) -> None:
        indices = cells[dimensions]
        position = positions[dimensions]
        while position < len(indices) and indices[position] in selected_indices:
            position += 1
        positions[dimensions] = position
        if position < len(indices):
            parent_asin, _, record_hash = prepared[indices[position]]
            score = sum(deficits[dimension] for dimension in dimensions)
            heapq.heappush(heap, (-score, record_hash, parent_asin, dimensions))

    for dimensions in cells:
        enqueue(dimensions)

    while len(selected_indices) < count:
        negative_score, record_hash, parent_asin, dimensions = heapq.heappop(heap)
        index = cells[dimensions][positions[dimensions]]
        score = sum(deficits[dimension] for dimension in dimensions)
        if score != -negative_score or (record_hash, parent_asin) != (
            prepared[index][2],
            prepared[index][0],
        ):
            enqueue(dimensions)
            continue

        selected_indices.add(index)
        for dimension in dimensions:
            if deficits[dimension] > 0:
                deficits[dimension] -= 1
        enqueue(dimensions)

    selected = (prepared[index] for index in selected_indices)
    return [
        parent_asin
        for parent_asin, _, _ in sorted(selected, key=lambda item: (item[2], item[0]))
    ]


def stress_ids(
    records: list[ProxyProduct], *, count: int, seed: int, excluded: set[str]
) -> list[str]:
    """Select a deterministic sample biased toward sparse marginal values."""
    _validate_unique_ids(records)
    candidates = [record for record in records if record.parent_asin not in excluded]
    if count < 0 or count > len(candidates):
        raise ValueError("count must be between zero and the available population")

    frequencies = Counter(dimension for record in records for dimension in record.dimensions())
    denominator = (1 << 256) + 1

    def ordering_key(record: ProxyProduct) -> tuple[float, int, str]:
        record_hash = stable_int(seed, record.parent_asin)
        unit = (record_hash + 1) / denominator
        weight = sum(1 / frequencies[dimension] for dimension in record.dimensions())
        return (-math.log(unit) / weight, record_hash, record.parent_asin)

    return [record.parent_asin for record in sorted(candidates, key=ordering_key)[:count]]


def _marginal_quotas(
    frequencies: Counter[tuple[str, str]], count: int, seed: int
) -> Counter[tuple[str, str]]:
    quotas: Counter[tuple[str, str]] = Counter()
    population = sum(frequency for dimension, frequency in frequencies.items() if dimension[0] == "category")

    for dimension_name in ("category", "price", "popularity", "completeness"):
        values = [
            dimension
            for dimension in frequencies
            if dimension[0] == dimension_name
        ]
        exact = {
            dimension: count * frequencies[dimension] / population for dimension in values
        }
        floors = {dimension: math.floor(quota) for dimension, quota in exact.items()}
        remaining = count - sum(floors.values())
        remainders = sorted(
            values,
            key=lambda dimension: (
                -(exact[dimension] - floors[dimension]),
                stable_int(seed, f"{dimension[0]}\0{dimension[1]}"),
                dimension[1],
            ),
        )
        quotas.update(floors)
        quotas.update({dimension: 1 for dimension in remainders[:remaining]})

    return quotas


def _prepare_records(
    records: list[ProxyProduct], seed: int
) -> tuple[
    list[tuple[str, tuple[tuple[str, str], ...], int]],
    Counter[tuple[str, str]],
    dict[str, int],
]:
    prepared: list[tuple[str, tuple[tuple[str, str], ...], int]] = []
    frequencies: Counter[tuple[str, str]] = Counter()
    category_best: dict[str, int] = {}
    seen_ids: set[str] = set()

    for index, record in enumerate(records):
        if record.parent_asin in seen_ids:
            raise ValueError("duplicate parent_asin values are not allowed")
        seen_ids.add(record.parent_asin)
        dimensions = record.dimensions()
        record_hash = stable_int(seed, record.parent_asin)
        prepared.append((record.parent_asin, dimensions, record_hash))
        frequencies.update(dimensions)
        best_index = category_best.get(record.category)
        if best_index is None or (record_hash, record.parent_asin) < (
            prepared[best_index][2],
            prepared[best_index][0],
        ):
            category_best[record.category] = index

    return prepared, frequencies, category_best


def _validate_unique_ids(records: list[ProxyProduct]) -> None:
    identifiers = [record.parent_asin for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate parent_asin values are not allowed")


def _validate_representative_count(count: int, population: int, categories: int) -> None:
    if count < categories or count > population:
        raise ValueError("count must be between the category count and population")


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_inputs(root: str | Path) -> dict[str, str]:
    """Confirm every immutable evaluator input matches its approved digest."""
    root_path = Path(root).resolve()
    actual: dict[str, str] = {}
    mismatches: list[str] = []
    for relative_path, expected in FROZEN_SHA256.items():
        path = root_path / relative_path
        try:
            observed = sha256_file(path)
        except OSError as error:
            mismatches.append(f"{relative_path}: unreadable ({error})")
            continue
        actual[relative_path] = observed
        if observed != expected:
            mismatches.append(f"{relative_path}: expected {expected}, got {observed}")
    if mismatches:
        raise ValueError("frozen input hash mismatches: " + "; ".join(mismatches))
    return actual


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stage_payload(path: str | Path, write: object) -> Path:
    destination = Path(path)
    if destination.is_dir():
        raise ValueError(f"proxy output destination is a directory: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            writer = write
            if not callable(writer):
                raise TypeError("staged payload writer must be callable")
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _stage_jsonl(path: str | Path, rows: list[dict]) -> Path:
    return _stage_payload(
        path,
        lambda handle: handle.writelines(_canonical_json(row) + "\n" for row in rows),
    )


def _stage_text(path: str | Path, text: str) -> Path:
    return _stage_payload(path, lambda handle: handle.write(text))


@dataclass
class _StagedEntry:
    destination: Path
    temporary: Path
    existed: bool = False
    backup: Path | None = None
    moved: bool = False
    published: bool = False


def _entry_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _remove_entry(path: Path) -> None:
    if not _entry_exists(path):
        return
    if path.is_dir():
        raise IsADirectoryError(f"proxy output entry is a directory: {path}")
    path.unlink()


def _backup_path(destination: Path) -> Path:
    descriptor, backup_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".bak", dir=destination.parent
    )
    os.close(descriptor)
    return Path(backup_name)


def _rollback_staged(entries: list[_StagedEntry]) -> list[Exception]:
    errors: list[Exception] = []
    for entry in entries:
        if entry.published:
            try:
                _remove_entry(entry.destination)
            except OSError as error:
                errors.append(error)
    for entry in entries:
        if entry.moved and entry.backup is not None:
            try:
                _remove_entry(entry.destination)
                os.replace(entry.backup, entry.destination)
                entry.moved = False
            except OSError as error:
                errors.append(error)
    for entry in entries:
        if not entry.existed:
            try:
                _remove_entry(entry.destination)
            except OSError as error:
                errors.append(error)
    return errors


def _cleanup_staged(entries: list[_StagedEntry]) -> list[Exception]:
    errors: list[Exception] = []
    for entry in entries:
        for path in (entry.temporary, entry.backup):
            if path is None:
                continue
            try:
                _remove_entry(path)
            except OSError as error:
                errors.append(error)
    return errors


def _replace_staged(staged: list[tuple[Path, Path]]) -> None:
    entries = [_StagedEntry(destination, temporary) for destination, temporary in staged]
    if len({entry.destination for entry in entries}) != len(entries):
        raise ValueError("proxy output destinations must be unique")
    try:
        for entry in entries:
            if entry.destination.is_dir() or entry.temporary.is_dir():
                raise ValueError(f"proxy output destination is a directory: {entry.destination}")
            entry.existed = _entry_exists(entry.destination)
        for entry in entries:
            if not entry.existed:
                continue
            entry.backup = _backup_path(entry.destination)
            try:
                os.replace(entry.destination, entry.backup)
            except Exception:
                entry.moved = not _entry_exists(entry.destination) and _entry_exists(entry.backup)
                raise
            entry.moved = True
        for entry in entries:
            try:
                os.replace(entry.temporary, entry.destination)
            except Exception:
                entry.published = not _entry_exists(entry.temporary) and _entry_exists(entry.destination)
                raise
            entry.published = True
    except Exception as original_error:
        rollback_errors = _rollback_staged(entries)
        cleanup_errors = _cleanup_staged(entries)
        if rollback_errors or cleanup_errors:
            details = "; ".join(str(error) for error in [*rollback_errors, *cleanup_errors])
            raise RuntimeError(
                f"proxy output replacement failed: {original_error}; rollback failed: {details}"
            ) from original_error
        raise
    cleanup_errors = _cleanup_staged(entries)
    if cleanup_errors:
        details = "; ".join(str(error) for error in cleanup_errors)
        raise RuntimeError(f"proxy output replacement succeeded but cleanup failed: {details}")


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    destination = Path(path)
    temporary = _stage_jsonl(destination, rows)
    _replace_staged([(destination, temporary)])


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("quartiles require at least one value")
    ordered = sorted(values)
    return tuple(ordered[math.ceil(len(ordered) * fraction) - 1] for fraction in (0.25, 0.50, 0.75))  # type: ignore[return-value]


def _quartile_bin(value: float, thresholds: tuple[float, float, float]) -> str:
    for index, threshold in enumerate(thresholds, start=1):
        if value <= threshold:
            return f"q{index}"
    return "q4"


def _has_content(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_content(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_content(item) for item in value)
    return value is not None


def _categories(product: dict) -> list[str]:
    values = product.get("categories")
    if not isinstance(values, list):
        return []
    return [value.strip() for value in values if isinstance(value, str) and value.strip()]


def _safe_profile(source: object) -> dict:
    profile = source if isinstance(source, dict) else {}
    source_tags = profile.get("preference_tags")
    tags = [value.strip() for value in source_tags if isinstance(value, str) and value.strip()] if isinstance(source_tags, list) else []
    purchase_frequency = profile.get("purchase_frequency")
    rating_style = profile.get("rating_style")
    rating = _finite_number(profile.get("average_prior_rating"))
    frequency = purchase_frequency.strip() if isinstance(purchase_frequency, str) and purchase_frequency.strip() else "unspecified"
    style = rating_style.strip() if isinstance(rating_style, str) and rating_style.strip() else "unspecified"
    preference_text = ", ".join(tags) if tags else "general preferences"
    return {
        "average_prior_rating": 0.0 if rating is None else rating,
        "preference_tags": tags,
        "purchase_frequency": frequency,
        "rating_style": style,
        "summary": f"Prior purchases emphasize {preference_text}; ratings are {style}.",
    }


def _read_catalog(path: Path) -> dict[str, dict]:
    products: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            product = json.loads(line)
            if not isinstance(product, dict):
                raise TypeError(f"catalog line {line_number} must be an object")
            raw_parent_asin = product.get("parent_asin")
            if not isinstance(raw_parent_asin, str) or not raw_parent_asin.strip():
                raise ValueError(f"catalog line {line_number} has an empty parent_asin")
            parent_asin = raw_parent_asin.strip()
            if parent_asin in products:
                raise ValueError("duplicate parent_asin values are not allowed")
            products[parent_asin] = {**product, "parent_asin": parent_asin}
    if not products:
        raise ValueError("catalog must contain at least one product")
    return products


def _read_public_restrictions(path: Path) -> tuple[set[str], list[dict]]:
    excluded: set[str] = set()
    profiles: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            ground_truth = row.get("ground_truth")
            if isinstance(ground_truth, dict):
                parent_asin = str(ground_truth.get("parent_asin", "")).strip()
                if parent_asin:
                    excluded.add(parent_asin)
            if isinstance(row.get("user_profile"), dict):
                profiles.append(_safe_profile(row["user_profile"]))
    if not profiles:
        raise ValueError("public set must contain at least one user profile")
    return excluded, profiles


def _build_records(products: dict[str, dict], excluded: set[str]) -> list[ProxyProduct]:
    ordered = [products[parent_asin] for parent_asin in sorted(products)]
    prices = [number for product in ordered if (number := _finite_number(product.get("price"))) is not None]
    price_thresholds = _quartiles(prices) if prices else None
    popularity_values = [_finite_number(product.get("rating_number")) or 0.0 for product in ordered]
    popularity_thresholds = _quartiles(popularity_values)
    records: list[ProxyProduct] = []
    for product in ordered:
        parent_asin = product["parent_asin"]
        if parent_asin in excluded:
            continue
        price = _finite_number(product.get("price"))
        price_bin = "missing" if price is None else _quartile_bin(price, price_thresholds)  # type: ignore[arg-type]
        popularity_bin = _quartile_bin(_finite_number(product.get("rating_number")) or 0.0, popularity_thresholds)
        completeness = sum(_has_content(product.get(field)) for field in ("title", "features", "details", "description", "categories", "store"))
        completeness_bin = "0-2" if completeness <= 2 else "3-4" if completeness <= 4 else "5+"
        difficulty = "hard" if completeness <= 2 or price is None else "easy" if completeness >= 5 and popularity_bin == "q4" else "medium"
        records.append(ProxyProduct(parent_asin, coarse_category(_categories(product)), price_bin, popularity_bin, completeness_bin, difficulty))
    return records


def scenario_schedule(count: int, seed: int) -> list[str]:
    """Produce an exact-proportion, stably shuffled scenario sequence."""
    if count < 0:
        raise ValueError("count must be non-negative")
    exact = {name: count * weight for name, weight in SCENARIO_WEIGHTS.items()}
    counts = {name: math.floor(value) for name, value in exact.items()}
    remaining = count - sum(counts.values())
    for name in sorted(SCENARIO_WEIGHTS, key=lambda item: (-(exact[item] - counts[item]), item))[:remaining]:
        counts[name] += 1
    scheduled = [(name, ordinal) for name in sorted(counts) for ordinal in range(counts[name])]
    return [name for name, ordinal in sorted(scheduled, key=lambda item: (stable_int(seed, f"{item[0]}:{item[1]}"), item[0], item[1]))]


def _materialize_suite(
    suite: str,
    target_ids: list[str],
    records_by_id: dict[str, ProxyProduct],
    products_by_id: dict[str, dict],
    profiles: list[dict],
    seed: int,
) -> list[dict]:
    schedule = scenario_schedule(len(target_ids), seed)
    profile_order = sorted(
        profiles,
        key=lambda profile: (
            stable_int(PROFILE_SEED, _canonical_hash(profile)),
            _canonical_json(profile),
        ),
    )
    rows: list[dict] = []
    for index, (target_id, scenario_type) in enumerate(zip(target_ids, schedule), start=1):
        sample_id = f"proxy_{suite}_{index:04d}"
        record = records_by_id[target_id]
        product = products_by_id[target_id]
        card = intent_card(product)
        rows.append({
            "sample_id": sample_id,
            "scenario_type": scenario_type,
            "category_bucket": record.category,
            "difficulty_bucket": record.difficulty,
            "ground_truth": {"parent_asin": target_id},
            "user_profile": profile_order[(index - 1) % len(profile_order)],
            "intent_card": card,
            "behavior": behavior_for(scenario_type, card, random.Random(f"{seed}:{sample_id}")),
            "dialogue_variant": stable_int(seed, sample_id) % 4,
            "proxy_suite": suite,
        })
    return rows


def _assign_folds(rows: list[dict], seed: int) -> None:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["scenario_type"], row["difficulty_bucket"]), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: (stable_int(seed, row["sample_id"]), row["sample_id"]))
        for position, row in enumerate(group):
            row["proxy_fold"] = position % 5 + 1


def _under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def build_proxy_bundle(
    catalog_path: str | Path,
    public_path: str | Path,
    output_dir: str | Path,
    representative_count: int = 2000,
    stress_count: int = 800,
    enforce_frozen: bool = True,
    overwrite: bool = False,
) -> dict:
    """Build deterministic, catalog-derived proxy suites and their manifest."""
    root = Path(__file__).resolve().parents[1]
    catalog = Path(catalog_path).resolve()
    public = Path(public_path).resolve()
    output = Path(output_dir).resolve()
    if enforce_frozen:
        if catalog != (root / "data" / "catalog.jsonl").resolve():
            raise ValueError("enforced frozen generation requires data/catalog.jsonl")
        if public != (root / "data" / "public_set.jsonl").resolve():
            raise ValueError("enforced frozen generation requires data/public_set.jsonl")
        if not _under(output, (root / "var").resolve()):
            raise ValueError("enforced frozen generation requires an output directory under var")
        input_hashes = verify_frozen_inputs(root)
    else:
        input_hashes = {"catalog": sha256_file(catalog), "public_set": sha256_file(public)}

    output_files = (output / "representative.jsonl", output / "stress.jsonl", output / "manifest.json")
    if any(path.exists() or path.is_symlink() for path in output_files) and not overwrite:
        raise ValueError("output directory already contains a proxy bundle; rerun with --force to replace it")
    products_by_id = _read_catalog(catalog)
    excluded, profiles = _read_public_restrictions(public)
    records = _build_records(products_by_id, excluded)
    if representative_count + stress_count > len(records):
        raise ValueError("insufficient post-exclusion population for requested proxy suites")
    representative_targets = representative_ids(records, count=representative_count, seed=REPRESENTATIVE_SEED)
    stress_targets = stress_ids(records, count=stress_count, seed=STRESS_SEED, excluded=set(representative_targets))
    if not set(representative_targets).isdisjoint(stress_targets):
        raise AssertionError("representative and stress targets must be disjoint")
    records_by_id = {record.parent_asin: record for record in records}
    representative = _materialize_suite("representative", representative_targets, records_by_id, products_by_id, profiles, REPRESENTATIVE_SEED)
    stress = _materialize_suite("stress", stress_targets, records_by_id, products_by_id, profiles, STRESS_SEED)
    _assign_folds(representative, REPRESENTATIVE_SEED)

    output.mkdir(parents=True, exist_ok=True)
    representative_path, stress_path, manifest_path = output_files
    staged: list[tuple[Path, Path]] = []
    try:
        staged.append((representative_path, _stage_jsonl(representative_path, representative)))
        staged.append((stress_path, _stage_jsonl(stress_path, stress)))
        output_hashes = {
            "representative.jsonl": sha256_file(staged[0][1]),
            "stress.jsonl": sha256_file(staged[1][1]),
        }
        generator_config = {
            "scenario_weights": SCENARIO_WEIGHTS,
            "dimensions": list(DIMENSION_NAMES),
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
            "seeds": {
                "representative": REPRESENTATIVE_SEED,
                "stress": STRESS_SEED,
                "profile": PROFILE_SEED,
            },
            "input_hashes": input_hashes,
            "excluded_target_hash": _canonical_hash(sorted(excluded)),
            "target_hashes": {
                "representative": _canonical_hash(representative_targets),
                "stress": _canonical_hash(stress_targets),
            },
            "output_hashes": output_hashes,
            "counts": {"representative": len(representative), "stress": len(stress)},
        }
        manifest_text = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        staged.append((manifest_path, _stage_text(manifest_path, manifest_text)))
        _replace_staged(staged)
    except Exception:
        for _destination, temporary in staged:
            temporary.unlink(missing_ok=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen catalog-derived proxy suites")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public-set", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representative-count", type=int, default=2000)
    parser.add_argument("--stress-count", type=int, default=800)
    parser.add_argument("--no-enforce-frozen", action="store_false", dest="enforce_frozen", help=argparse.SUPPRESS)
    parser.add_argument(
        "--force",
        action="store_true",
        dest="overwrite",
        help="replace an existing proxy bundle after staging and rollback protection",
    )
    args = parser.parse_args()
    manifest = build_proxy_bundle(
        args.catalog, args.public_set, args.output_dir,
        representative_count=args.representative_count,
        stress_count=args.stress_count,
        enforce_frozen=args.enforce_frozen,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
