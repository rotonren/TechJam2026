from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import get_type_hints

import pytest

from tools import proxy_dataset
from tools.proxy_dataset import (
    ProxyProduct,
    _build_records,
    build_proxy_bundle,
    representative_ids,
    scenario_schedule,
    stable_int,
    stress_ids,
    verify_frozen_inputs,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_proxy_bundle_is_deterministic_and_excludes_public_targets(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    public_set = tmp_path / "public.jsonl"
    products = [
        {
            "parent_asin": f"P{index:03d}",
            "title": f"Product {index}",
            "features": ["comfortable fit", f"feature {index % 3}"],
            "details": {"material": "cotton"},
            "categories": ["Clothing", "Shoes", "Running" if index % 2 else "Walking"],
            "price": float(index + 1),
            "average_rating": 4.0,
            "rating_number": index * 10,
        }
        for index in range(50)
    ]
    write_jsonl(catalog, products)
    write_jsonl(
        public_set,
        [{
            "ground_truth": {"parent_asin": "P000"},
            "user_profile": {
                "average_prior_rating": 5,
                "preference_tags": ["fit", "comfort"],
                "purchase_frequency": "monthly",
                "rating_style": "generous",
                "summary": "source text must not be copied",
            },
        }],
    )

    first = build_proxy_bundle(
        catalog, public_set, tmp_path / "one", representative_count=20, stress_count=8,
        enforce_frozen=False,
    )
    second = build_proxy_bundle(
        catalog, public_set, tmp_path / "two", representative_count=20, stress_count=8,
        enforce_frozen=False,
    )
    representative = [json.loads(line) for line in (tmp_path / "one" / "representative.jsonl").read_text(encoding="utf-8").splitlines()]
    stress = [json.loads(line) for line in (tmp_path / "one" / "stress.jsonl").read_text(encoding="utf-8").splitlines()]

    assert first["output_hashes"] == second["output_hashes"]
    assert first["output_hashes"] == {
        "representative.jsonl": proxy_dataset.sha256_file(tmp_path / "one" / "representative.jsonl"),
        "stress.jsonl": proxy_dataset.sha256_file(tmp_path / "one" / "stress.jsonl"),
    }
    assert set(first) == {
        "schema_version",
        "generator_version",
        "generator_config",
        "generator_config_hash",
        "seeds",
        "input_hashes",
        "excluded_target_hash",
        "target_hashes",
        "output_hashes",
        "counts",
    }
    assert "P000" not in {row["ground_truth"]["parent_asin"] for row in representative}
    assert {row["scenario_type"] for row in representative} == {
        "buying", "browsing", "intent_override", "boundary",
    }
    assert all(row["user_profile"]["summary"] != "source text must not be copied" for row in representative)
    assert {row["proxy_fold"] for row in representative} == {1, 2, 3, 4, 5}
    assert {row["ground_truth"]["parent_asin"] for row in representative}.isdisjoint(
        {row["ground_truth"]["parent_asin"] for row in stress}
    )
    assert first["generator_config_hash"]
    assert first["generator_config_hash"] == proxy_dataset._canonical_hash(first["generator_config"])
    assert first["seeds"] == {
        "representative": 20260826,
        "stress": 20260827,
        "profile": 20260828,
    }
    assert "dimensions" in first["generator_config"]
    assert "dimension_names" not in first["generator_config"]
    assert set(first["generator_config"]) == {
        "scenario_weights",
        "dimensions",
        "price_bins",
        "popularity_bins",
        "completeness_bins",
        "representative_count",
        "stress_count",
        "profile_summary_version",
        "fold_count",
    }
    assert representative[0]["sample_id"] == "proxy_representative_0001"
    assert stress[0]["sample_id"] == "proxy_stress_0001"
    assert first["target_hashes"] == second["target_hashes"]
    assert "P000" not in {row["ground_truth"]["parent_asin"] for row in stress}


def test_scenario_schedule_uses_largest_remainder_and_is_deterministic() -> None:
    schedule = scenario_schedule(20, 20260826)

    assert schedule == scenario_schedule(20, 20260826)
    assert {name: schedule.count(name) for name in set(schedule)} == {
        "buying": 8,
        "browsing": 8,
        "intent_override": 3,
        "boundary": 1,
    }
    assert scenario_schedule(0, 20260826) == []


def test_build_records_uses_global_quartiles_and_missing_data_difficulty() -> None:
    products = {
        "A": {"parent_asin": "A", "categories": ["Clothing", "Shoes"], "price": 10, "rating_number": 1},
        "B": {"parent_asin": "B", "title": "B", "features": ["f"], "details": {"d": "x"}, "description": "d", "categories": ["Clothing", "Shoes"], "store": "s", "price": 20, "rating_number": 2},
        "C": {"parent_asin": "C", "title": "C", "features": ["f"], "details": {"d": "x"}, "description": "d", "categories": ["Clothing", "Shoes"], "store": "s", "price": 30, "rating_number": 3},
        "D": {"parent_asin": "D", "title": "D", "features": ["f"], "details": {"d": "x"}, "description": "d", "categories": ["Clothing", "Shoes"], "store": "s", "price": 40, "rating_number": 4},
        "E": {"parent_asin": "E", "title": "E", "categories": ["Clothing", "Shoes"], "rating_number": "bad"},
    }
    records = {record.parent_asin: record for record in _build_records(products, set())}

    assert records["A"].price_bin == "q1"
    assert records["D"].popularity_bin == "q4"
    assert records["D"].completeness_bin == "5+"
    assert records["D"].difficulty == "easy"
    assert records["A"].difficulty == "hard"
    assert records["E"].price_bin == "missing"
    assert records["E"].popularity_bin == "q1"
    assert records["E"].difficulty == "hard"


def test_finite_number_accepts_numeric_strings_and_rejects_invalid_values() -> None:
    assert proxy_dataset._finite_number("12.5") == 12.5
    assert proxy_dataset._finite_number(" 3 ") == 3.0
    assert proxy_dataset._finite_number(7) == 7.0
    assert all(proxy_dataset._finite_number(value) is None for value in (True, "", "nope", "nan", "inf"))


def test_build_records_uses_nearest_rank_boundaries_for_numeric_strings() -> None:
    products = {
        f"P{index}": {
            "parent_asin": f"P{index}",
            "title": "title",
            "features": ["feature"],
            "details": {"detail": "value"},
            "categories": ["Shoes"],
            "price": str(index),
            "rating_number": str(index),
        }
        for index in range(1, 9)
    }
    records = {record.parent_asin: record for record in _build_records(products, set())}

    assert [records[f"P{index}"].price_bin for index in range(1, 9)] == [
        "q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4",
    ]
    assert [records[f"P{index}"].popularity_bin for index in range(1, 9)] == [
        "q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4",
    ]


def test_build_records_keeps_excluded_products_in_full_population_statistics() -> None:
    products = {
        "A": {"parent_asin": "A", "categories": ["Shoes"], "price": "1", "rating_number": "1"},
        "B": {"parent_asin": "B", "categories": ["Shoes"], "price": "2", "rating_number": "2"},
        "C": {"parent_asin": "C", "categories": ["Shoes"], "price": "3", "rating_number": "3"},
        "included": {"parent_asin": "included", "categories": ["Shoes"], "price": "100", "rating_number": "100"},
    }
    records = _build_records(products, {"A", "B", "C"})

    assert [record.parent_asin for record in records] == ["included"]
    assert records[0].price_bin == "q4"
    assert records[0].popularity_bin == "q4"


def test_build_records_classifies_completeness_and_difficulty_buckets() -> None:
    products = {
        "hard": {"parent_asin": "hard", "title": "title", "categories": ["Shoes"], "price": 1, "rating_number": 1},
        "medium": {"parent_asin": "medium", "title": "title", "features": ["feature"], "details": {"detail": "x"}, "categories": ["Shoes"], "price": 2, "rating_number": 2},
        "filler": {"parent_asin": "filler", "title": "title", "features": ["feature"], "details": {"detail": "x"}, "description": "description", "categories": ["Shoes"], "store": "store", "price": 3, "rating_number": 3},
        "easy": {"parent_asin": "easy", "title": "title", "features": ["feature"], "details": {"detail": "x"}, "description": "description", "categories": ["Shoes"], "store": "store", "price": 4, "rating_number": 4},
    }
    records = {record.parent_asin: record for record in _build_records(products, set())}

    assert (records["hard"].completeness_bin, records["hard"].difficulty) == ("0-2", "hard")
    assert (records["medium"].completeness_bin, records["medium"].difficulty) == ("3-4", "medium")
    assert (records["easy"].completeness_bin, records["easy"].difficulty) == ("5+", "easy")


def test_safe_profile_has_exact_allowlist_and_defaults() -> None:
    profile = proxy_dataset._safe_profile({
        "average_prior_rating": "not numeric",
        "preference_tags": [" fit ", "", 3],
        "purchase_frequency": None,
        "rating_style": "",
        "summary": "do not retain this",
        "target": "do not retain this either",
    })

    assert profile == {
        "average_prior_rating": 0.0,
        "preference_tags": ["fit"],
        "purchase_frequency": "unspecified",
        "rating_style": "unspecified",
        "summary": "Prior purchases emphasize fit; ratings are unspecified.",
    }


def test_assign_proxy_folds_groups_orders_and_round_robins() -> None:
    rows = [
        {"sample_id": f"proxy_representative_{index:04d}", "scenario_type": "buying", "difficulty_bucket": "hard"}
        for index in range(1, 8)
    ] + [
        {"sample_id": "proxy_representative_0008", "scenario_type": "browsing", "difficulty_bucket": "hard"},
    ]

    proxy_dataset._assign_folds(rows, 20260826)
    buying = sorted(
        rows[:7],
        key=lambda row: (stable_int(20260826, row["sample_id"]), row["sample_id"]),
    )

    assert [row["proxy_fold"] for row in buying] == [1, 2, 3, 4, 5, 1, 2]
    assert rows[7]["proxy_fold"] == 1
    assert all("fold" not in row for row in rows)


def test_verify_frozen_inputs_reports_mismatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_dataset, "FROZEN_SHA256", {"one.txt": "0" * 64})
    (tmp_path / "one.txt").write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="one.txt"):
        verify_frozen_inputs(tmp_path)


def test_frozen_hash_constants_match_checked_out_project_files() -> None:
    project_root = Path(__file__).resolve().parents[2]

    assert verify_frozen_inputs(project_root) == proxy_dataset.FROZEN_SHA256


def test_enforced_frozen_paths_reject_unapproved_inputs_before_reading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="data/catalog.jsonl"):
        build_proxy_bundle(tmp_path / "missing-catalog", tmp_path / "missing-public", tmp_path / "output")

    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="data/public_set.jsonl"):
        build_proxy_bundle(root / "data/catalog.jsonl", tmp_path / "missing-public", root / "var/proxy-test")
    with pytest.raises(ValueError, match="under var"):
        build_proxy_bundle(root / "data/catalog.jsonl", root / "data/public_set.jsonl", tmp_path / "output")


def test_bundle_rejects_empty_profiles_duplicate_ids_and_insufficient_population(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    public_set = tmp_path / "public.jsonl"
    write_jsonl(catalog, [{"parent_asin": "A", "categories": ["Shoes"], "price": 1, "rating_number": 1}])
    write_jsonl(public_set, [{"ground_truth": {"parent_asin": "A"}}])
    with pytest.raises(ValueError, match="user profile"):
        build_proxy_bundle(catalog, public_set, tmp_path / "empty", representative_count=0, stress_count=0, enforce_frozen=False)

    write_jsonl(catalog, [{"parent_asin": "A"}, {"parent_asin": "A"}])
    write_jsonl(public_set, [{"user_profile": {}}])
    with pytest.raises(ValueError, match="duplicate parent_asin"):
        build_proxy_bundle(catalog, public_set, tmp_path / "duplicates", representative_count=0, stress_count=0, enforce_frozen=False)

    write_jsonl(catalog, [{"parent_asin": "A", "categories": ["Shoes"], "price": 1, "rating_number": 1}])
    with pytest.raises(ValueError, match="insufficient post-exclusion"):
        build_proxy_bundle(catalog, public_set, tmp_path / "insufficient", representative_count=1, stress_count=1, enforce_frozen=False)


def test_bundle_is_independent_of_catalog_line_order(tmp_path: Path) -> None:
    products = [
        {"parent_asin": f"P{index:03d}", "title": "t", "features": ["f"], "details": {"x": "y"}, "categories": ["Shoes", str(index % 3)], "price": index + 1, "rating_number": index}
        for index in range(30)
    ]
    forward, reverse = tmp_path / "forward.jsonl", tmp_path / "reverse.jsonl"
    public_forward, public_reverse = tmp_path / "public-forward.jsonl", tmp_path / "public-reverse.jsonl"
    write_jsonl(forward, products)
    write_jsonl(reverse, list(reversed(products)))
    profiles = [
        {"user_profile": {"preference_tags": ["fit"], "rating_style": "brief"}},
        {"user_profile": {"preference_tags": ["comfort"], "rating_style": "detailed"}},
    ]
    write_jsonl(public_forward, profiles)
    write_jsonl(public_reverse, list(reversed(profiles)))

    first = build_proxy_bundle(forward, public_forward, tmp_path / "forward", 15, 5, enforce_frozen=False)
    second = build_proxy_bundle(reverse, public_reverse, tmp_path / "reverse", 15, 5, enforce_frozen=False)

    assert first["target_hashes"] == second["target_hashes"]
    assert first["output_hashes"] == second["output_hashes"]


def test_bundle_overwrite_only_replaces_known_proxy_output_files(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    public_set = tmp_path / "public.jsonl"
    output = tmp_path / "output"
    write_jsonl(catalog, [
        {"parent_asin": "A", "categories": ["Shoes"], "price": 1, "rating_number": 1},
        {"parent_asin": "B", "categories": ["Shoes"], "price": 2, "rating_number": 2},
        {"parent_asin": "C", "categories": ["Shoes"], "price": 3, "rating_number": 3},
    ])
    write_jsonl(public_set, [{"user_profile": {}}])
    first = build_proxy_bundle(catalog, public_set, output, 2, 1, enforce_frozen=False)
    (output / "unrelated.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="rerun with --force"):
        build_proxy_bundle(catalog, public_set, output, 2, 1, enforce_frozen=False)
    second = build_proxy_bundle(catalog, public_set, output, 2, 1, enforce_frozen=False, overwrite=True)

    assert second["output_hashes"] == first["output_hashes"]
    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "preserve"


def make_existing_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    catalog = tmp_path / "catalog.jsonl"
    public_set = tmp_path / "public.jsonl"
    output = tmp_path / "output"
    write_jsonl(catalog, [
        {"parent_asin": "A", "categories": ["Shoes"], "price": 1, "rating_number": 1},
        {"parent_asin": "B", "categories": ["Shoes"], "price": 2, "rating_number": 2},
        {"parent_asin": "C", "categories": ["Shoes"], "price": 3, "rating_number": 3},
    ])
    write_jsonl(public_set, [{"user_profile": {}}])
    build_proxy_bundle(catalog, public_set, output, 2, 1, enforce_frozen=False)
    return catalog, public_set, output


@pytest.mark.parametrize("filename", ["representative.jsonl", "stress.jsonl", "manifest.json"])
def test_bundle_overwrite_replaces_hardlinks_without_mutating_external_file(tmp_path: Path, filename: str) -> None:
    catalog, public_set, output = make_existing_bundle(tmp_path)
    destination = output / filename
    external = tmp_path / f"external-{filename}"
    external.write_text("external content", encoding="utf-8")
    destination.unlink()
    try:
        os.link(external, destination)
    except OSError as error:
        pytest.skip(f"hard links unavailable on this platform: {error}")

    build_proxy_bundle(catalog, public_set, output, 2, 1, enforce_frozen=False, overwrite=True)

    assert external.read_text(encoding="utf-8") == "external content"
    assert destination.read_text(encoding="utf-8") != "external content"


@pytest.mark.parametrize("filename", ["representative.jsonl", "stress.jsonl", "manifest.json"])
def test_bundle_overwrite_replaces_symlinks_without_mutating_external_file(tmp_path: Path, filename: str) -> None:
    catalog, public_set, output = make_existing_bundle(tmp_path)
    destination = output / filename
    external = tmp_path / f"external-{filename}"
    external.write_text("external content", encoding="utf-8")
    destination.unlink()
    try:
        destination.symlink_to(external)
    except OSError as error:
        pytest.skip(f"symlinks unavailable on this platform: {error}")

    build_proxy_bundle(catalog, public_set, output, 2, 1, enforce_frozen=False, overwrite=True)

    assert external.read_text(encoding="utf-8") == "external content"
    assert not destination.is_symlink()


@pytest.mark.parametrize("filename", ["representative.jsonl", "stress.jsonl", "manifest.json"])
def test_bundle_overwrite_replaces_broken_symlinks(tmp_path: Path, filename: str) -> None:
    catalog, public_set, output = make_existing_bundle(tmp_path)
    destination = output / filename
    destination.unlink()
    try:
        destination.symlink_to(tmp_path / "missing-target")
    except OSError as error:
        pytest.skip(f"symlinks unavailable on this platform: {error}")

    build_proxy_bundle(catalog, public_set, output, 2, 1, enforce_frozen=False, overwrite=True)

    assert destination.is_file()
    assert not destination.is_symlink()


@pytest.mark.parametrize("filename", ["representative.jsonl", "stress.jsonl", "manifest.json"])
def test_bundle_overwrite_rejects_directory_destinations_and_cleans_staging(tmp_path: Path, filename: str) -> None:
    catalog, public_set, output = make_existing_bundle(tmp_path)
    destination = output / filename
    destination.unlink()
    destination.mkdir()

    with pytest.raises(ValueError, match="destination is a directory"):
        build_proxy_bundle(catalog, public_set, output, 2, 1, enforce_frozen=False, overwrite=True)

    assert list(output.glob(".*.tmp")) == []


def staged_bundle(output: Path, prefix: str) -> list[tuple[Path, Path]]:
    return [
        (output / filename, proxy_dataset._stage_text(output / filename, f"{prefix}-{filename}"))
        for filename in ("representative.jsonl", "stress.jsonl", "manifest.json")
    ]


@pytest.mark.parametrize("failure_position", range(1, 7))
def test_replace_staged_rolls_back_every_backup_and_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_position: int,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    filenames = ("representative.jsonl", "stress.jsonl", "manifest.json")
    old = {filename: f"old-{filename}" for filename in filenames}
    for filename, content in old.items():
        (output / filename).write_text(content, encoding="utf-8")
    (output / "unrelated.txt").write_text("preserve", encoding="utf-8")
    staged = staged_bundle(output, "new")
    actual_replace = proxy_dataset.os.replace
    calls = 0

    def fail_at_position(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_position:
            raise OSError(f"fault at replacement {failure_position}")
        actual_replace(source, destination)

    monkeypatch.setattr(proxy_dataset.os, "replace", fail_at_position)

    with pytest.raises(OSError, match=f"replacement {failure_position}"):
        proxy_dataset._replace_staged(staged)

    assert {filename: (output / filename).read_text(encoding="utf-8") for filename in filenames} == old
    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "preserve"
    assert list(output.glob(".*.tmp")) == []
    assert list(output.glob(".*.bak")) == []


@pytest.mark.parametrize("failure_position", range(1, 4))
def test_replace_staged_failure_in_fresh_directory_leaves_no_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_position: int,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    staged = staged_bundle(output, "new")
    actual_replace = proxy_dataset.os.replace
    calls = 0

    def fail_at_position(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_position:
            raise OSError(f"fault at publication {failure_position}")
        actual_replace(source, destination)

    monkeypatch.setattr(proxy_dataset.os, "replace", fail_at_position)

    with pytest.raises(OSError, match=f"publication {failure_position}"):
        proxy_dataset._replace_staged(staged)

    assert not any((output / filename).exists() for filename in ("representative.jsonl", "stress.jsonl", "manifest.json"))
    assert list(output.glob(".*.tmp")) == []
    assert list(output.glob(".*.bak")) == []


def test_replace_staged_publishes_manifest_last_and_replaces_all_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    filenames = ("representative.jsonl", "stress.jsonl", "manifest.json")
    for filename in filenames:
        (output / filename).write_text(f"old-{filename}", encoding="utf-8")
    staged = staged_bundle(output, "new")
    actual_replace = proxy_dataset.os.replace
    published: list[str] = []

    def record_publications(source: object, destination: object) -> None:
        if str(source).endswith(".tmp"):
            published.append(Path(destination).name)
        actual_replace(source, destination)

    monkeypatch.setattr(proxy_dataset.os, "replace", record_publications)

    proxy_dataset._replace_staged(staged)

    assert published == list(filenames)
    assert {filename: (output / filename).read_text(encoding="utf-8") for filename in filenames} == {
        filename: f"new-{filename}" for filename in filenames
    }
    assert list(output.glob(".*.tmp")) == []
    assert list(output.glob(".*.bak")) == []


def test_replace_staged_rollback_preserves_hardlink_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("old-representative", encoding="utf-8")
    representative = output / "representative.jsonl"
    try:
        os.link(external, representative)
    except OSError as error:
        pytest.skip(f"hard links unavailable on this platform: {error}")
    (output / "stress.jsonl").write_text("old-stress", encoding="utf-8")
    (output / "manifest.json").write_text("old-manifest", encoding="utf-8")
    staged = staged_bundle(output, "new")
    actual_replace = proxy_dataset.os.replace
    calls = 0

    def fail_first_publication(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("fault at first publication")
        actual_replace(source, destination)

    monkeypatch.setattr(proxy_dataset.os, "replace", fail_first_publication)

    with pytest.raises(OSError, match="first publication"):
        proxy_dataset._replace_staged(staged)

    assert external.read_text(encoding="utf-8") == "old-representative"
    assert os.path.samefile(external, representative)


@pytest.mark.parametrize("restore_position", range(1, 4))
def test_replace_staged_retains_original_backup_when_each_restore_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_position: int,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    filenames = ("representative.jsonl", "stress.jsonl", "manifest.json")
    old = {filename: f"old-{filename}" for filename in filenames}
    for filename, content in old.items():
        (output / filename).write_text(content, encoding="utf-8")
    (output / "unrelated.txt").write_text("preserve", encoding="utf-8")
    staged = staged_bundle(output, "new")
    actual_replace = proxy_dataset.os.replace
    calls = 0

    def fail_publication_and_restore(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls in {4, 4 + restore_position}:
            raise OSError(f"fault at replacement {calls}")
        actual_replace(source, destination)

    monkeypatch.setattr(proxy_dataset.os, "replace", fail_publication_and_restore)

    with pytest.raises(RuntimeError) as error:
        proxy_dataset._replace_staged(staged)

    failed_filename = filenames[restore_position - 1]
    retained = list(output.glob(".*.bak"))
    assert len(retained) == 1
    assert retained[0].read_text(encoding="utf-8") == old[failed_filename]
    assert str(retained[0].resolve()) in str(error.value)
    assert not os.path.lexists(output / failed_filename)
    assert {
        filename: (output / filename).read_text(encoding="utf-8")
        for filename in filenames
        if filename != failed_filename
    } == {filename: old[filename] for filename in filenames if filename != failed_filename}
    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "preserve"
    assert list(output.glob(".*.tmp")) == []


def test_replace_staged_retains_every_backup_after_multiple_restore_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    filenames = ("representative.jsonl", "stress.jsonl", "manifest.json")
    old = {filename: f"old-{filename}" for filename in filenames}
    for filename, content in old.items():
        (output / filename).write_text(content, encoding="utf-8")
    staged = staged_bundle(output, "new")
    actual_replace = proxy_dataset.os.replace
    calls = 0

    def fail_publication_and_two_restores(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls in {4, 5, 6}:
            raise OSError(f"fault at replacement {calls}")
        actual_replace(source, destination)

    monkeypatch.setattr(proxy_dataset.os, "replace", fail_publication_and_two_restores)

    with pytest.raises(RuntimeError) as error:
        proxy_dataset._replace_staged(staged)

    retained = list(output.glob(".*.bak"))
    assert len(retained) == 2
    assert {path.read_text(encoding="utf-8") for path in retained} == {
        old["representative.jsonl"], old["stress.jsonl"],
    }
    assert all(str(path.resolve()) in str(error.value) for path in retained)
    assert not os.path.lexists(output / "representative.jsonl")
    assert not os.path.lexists(output / "stress.jsonl")
    assert (output / "manifest.json").read_text(encoding="utf-8") == old["manifest.json"]
    assert list(output.glob(".*.tmp")) == []


def test_replace_staged_mixed_existing_and_absent_destinations_preserves_recovery_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "representative.jsonl").write_text("old-representative", encoding="utf-8")
    (output / "manifest.json").write_text("old-manifest", encoding="utf-8")
    staged = staged_bundle(output, "new")
    actual_replace = proxy_dataset.os.replace
    calls = 0

    def fail_publication_and_first_restore(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls in {3, 4}:
            raise OSError(f"fault at replacement {calls}")
        actual_replace(source, destination)

    monkeypatch.setattr(proxy_dataset.os, "replace", fail_publication_and_first_restore)

    with pytest.raises(RuntimeError) as error:
        proxy_dataset._replace_staged(staged)

    retained = list(output.glob(".*.bak"))
    assert len(retained) == 1
    assert retained[0].read_text(encoding="utf-8") == "old-representative"
    assert str(retained[0].resolve()) in str(error.value)
    assert not os.path.lexists(output / "representative.jsonl")
    assert not os.path.lexists(output / "stress.jsonl")
    assert (output / "manifest.json").read_text(encoding="utf-8") == "old-manifest"
    assert list(output.glob(".*.tmp")) == []


def test_retained_backup_details_use_no_follow_absolute_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "destination.jsonl"
    backup = tmp_path / ".destination.jsonl.recovery.bak"
    backup.write_text("original", encoding="utf-8")
    entry = proxy_dataset._StagedEntry(
        destination=destination,
        temporary=tmp_path / ".destination.jsonl.staged.tmp",
        backup=backup,
        backup_holds_original=True,
    )
    expected = (
        f"destination {os.path.abspath(os.fspath(destination))} "
        f"retained original at {os.path.abspath(os.fspath(backup))}"
    )

    def resolve_must_not_run(self: Path, *args: object, **kwargs: object) -> Path:
        raise AssertionError(f"resolve unexpectedly called for {self}")

    monkeypatch.setattr(Path, "resolve", resolve_must_not_run)

    assert proxy_dataset._retained_backup_details([entry]) == [expected]


def test_restore_failure_retains_symlink_backup_entry_and_reports_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("original", encoding="utf-8")
    representative = output / "representative.jsonl"
    try:
        representative.symlink_to(external)
    except OSError as error:
        pytest.skip(f"symlinks unavailable on this platform: {error}")
    (output / "stress.jsonl").write_text("old-stress", encoding="utf-8")
    (output / "manifest.json").write_text("old-manifest", encoding="utf-8")
    staged = staged_bundle(output, "new")
    actual_replace = proxy_dataset.os.replace
    calls = 0

    def fail_publication_and_first_restore(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls in {4, 5}:
            raise OSError(f"fault at replacement {calls}")
        actual_replace(source, destination)

    monkeypatch.setattr(proxy_dataset.os, "replace", fail_publication_and_first_restore)

    with pytest.raises(RuntimeError) as error:
        proxy_dataset._replace_staged(staged)

    retained = list(output.glob(".*.bak"))
    assert len(retained) == 1
    assert retained[0].is_symlink()
    assert os.path.samefile(os.readlink(retained[0]), external)
    assert os.path.abspath(os.fspath(retained[0])) in str(error.value)
    assert external.read_text(encoding="utf-8") == "original"


def test_cli_help_exposes_force_and_keeps_frozen_bypass_hidden(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["proxy_dataset", "--help"])

    with pytest.raises(SystemExit, match="0"):
        proxy_dataset.main()

    help_text = capsys.readouterr().out
    assert "--force" in help_text
    assert "replace an existing proxy bundle after staging and rollback protection" in " ".join(help_text.split())
    assert "--no-enforce-frozen" not in help_text


def make_products(count: int = 40) -> list[ProxyProduct]:
    return [
        ProxyProduct(
            parent_asin=f"P{index:03d}",
            category=f"category-{index % 4}",
            price_bin=f"price-{index % 3}",
            popularity_bin=f"popularity-{index % 5}",
            completeness_bin=f"completeness-{index % 2}",
            difficulty=("easy", "medium", "hard")[index % 3],
        )
        for index in range(count)
    ]


def test_stable_int_is_deterministic_and_seeded() -> None:
    assert stable_int(20260826, "P001") == stable_int(20260826, "P001")
    assert stable_int(20260826, "P001") != stable_int(20260827, "P001")
    assert stable_int(20260826, "P001") == (
        111489689599360011317610825598156399003291339275809538316951117850236407355197
    )


def test_proxy_product_difficulty_is_a_string_bucket() -> None:
    assert get_type_hints(ProxyProduct)["difficulty"] is str


def test_proxy_product_dimensions_expose_sampling_marginals() -> None:
    product = ProxyProduct(
        parent_asin="P001",
        category="audio",
        price_bin="mid",
        popularity_bin="high",
        completeness_bin="complete",
        difficulty="hard",
    )

    assert product.dimensions() == (
        ("category", "audio"),
        ("price", "mid"),
        ("popularity", "high"),
        ("completeness", "complete"),
    )


def test_representative_ids_are_order_independent_unique_and_cover_categories() -> None:
    records = make_products()

    selected = representative_ids(records, count=20, seed=20260826)
    reordered = representative_ids(list(reversed(records)), count=20, seed=20260826)

    assert selected == reordered
    assert len(selected) == 20
    assert len(set(selected)) == 20
    categories = {record.category for record in records if record.parent_asin in selected}
    assert categories == {"category-0", "category-1", "category-2", "category-3"}


def test_representative_ids_reject_duplicate_parent_asins() -> None:
    records = make_products(4)
    records.append(records[0])

    with pytest.raises(ValueError, match="duplicate parent_asin"):
        representative_ids(records, count=4, seed=20260826)


def test_representative_ids_validate_population_and_coverage_boundaries() -> None:
    records = make_products(4)

    assert representative_ids([], count=0, seed=20260826) == []

    with pytest.raises(ValueError, match="count"):
        representative_ids(records, count=-1, seed=20260826)

    with pytest.raises(ValueError, match="count"):
        representative_ids(records, count=3, seed=20260826)

    with pytest.raises(ValueError, match="count"):
        representative_ids(records, count=5, seed=20260826)


def test_representative_ids_match_quota_greedy_selection_and_stable_order() -> None:
    records = [
        ProxyProduct("A0", "a", "low", "high", "full", "easy"),
        ProxyProduct("A1", "a", "high", "low", "sparse", "medium"),
        ProxyProduct("B0", "b", "low", "low", "full", "hard"),
        ProxyProduct("B1", "b", "high", "high", "sparse", "easy"),
        ProxyProduct("B2", "b", "high", "high", "full", "medium"),
    ]

    # The category seeds are A0 and B2. Quotas then select B1; IDs are hash-ordered.
    assert representative_ids(records, count=3, seed=20260826) == ["A0", "B2", "B1"]


def test_representative_ids_scales_to_large_proxy_populations() -> None:
    records = [
        ProxyProduct(
            parent_asin=f"L{index:05d}",
            category=f"category-{index % 50}",
            price_bin=f"price-{index % 7}",
            popularity_bin=f"popularity-{index % 11}",
            completeness_bin=f"completeness-{index % 5}",
            difficulty=("easy", "medium", "hard")[index % 3],
        )
        for index in range(50_000)
    ]

    started_at = time.perf_counter()
    selected = representative_ids(records, count=2_000, seed=20260826)
    elapsed = time.perf_counter() - started_at

    assert len(selected) == 2_000
    # The prior repeated scan projects to about 200 seconds; 15 seconds leaves ample CI headroom.
    assert elapsed < 15


def test_stress_ids_are_disjoint_unique_and_include_a_rare_category() -> None:
    records = [
        ProxyProduct(
            parent_asin=f"P{index:03d}",
            category="common" if index < 35 else "rare",
            price_bin="mid",
            popularity_bin="high",
            completeness_bin="complete",
            difficulty="medium",
        )
        for index in range(40)
    ]
    excluded = set(representative_ids(records, count=20, seed=20260826))

    selected = stress_ids(records, count=8, seed=20260826, excluded=excluded)

    assert len(selected) == 8
    assert len(set(selected)) == 8
    assert set(selected).isdisjoint(excluded)
    assert any(record.parent_asin in selected and record.category == "rare" for record in records)


def test_stress_ids_use_full_population_frequencies_and_are_order_independent() -> None:
    records = [
        ProxyProduct("R0", "c2", "p2", "o2", "m2", "easy"),
        ProxyProduct("R1", "c1", "p2", "o0", "m2", "easy"),
        ProxyProduct("R2", "c1", "p0", "o2", "m0", "easy"),
        ProxyProduct("R3", "c1", "p1", "o2", "m1", "easy"),
        ProxyProduct("R4", "c1", "p1", "o0", "m1", "easy"),
        ProxyProduct("R5", "c0", "p1", "o0", "m2", "easy"),
        ProxyProduct("R6", "c0", "p1", "o1", "m1", "easy"),
        ProxyProduct("R7", "c1", "p1", "o0", "m2", "easy"),
        ProxyProduct("R8", "c1", "p2", "o2", "m0", "easy"),
        ProxyProduct("R9", "c2", "p0", "o1", "m2", "easy"),
    ]
    excluded = {"R0", "R1", "R2", "R3"}

    selected = stress_ids(records, count=6, seed=20260826, excluded=excluded)

    # Candidate-only frequencies would incorrectly order R4 before R5.
    assert selected == ["R6", "R5", "R4", "R7", "R9", "R8"]
    assert (
        stress_ids(list(reversed(records)), count=6, seed=20260826, excluded=excluded)
        == selected
    )


def test_stress_ids_reject_duplicate_parent_asins() -> None:
    records = make_products(2)
    records.append(records[0])

    with pytest.raises(ValueError, match="duplicate parent_asin"):
        stress_ids(records, count=1, seed=20260826, excluded=set())


def test_stress_ids_validate_available_population() -> None:
    records = make_products(2)

    assert stress_ids(records, count=0, seed=20260826, excluded=set()) == []

    with pytest.raises(ValueError, match="count"):
        stress_ids(records, count=-1, seed=20260826, excluded=set())

    with pytest.raises(ValueError, match="count"):
        stress_ids(records, count=3, seed=20260826, excluded=set())

    with pytest.raises(ValueError, match="count"):
        stress_ids(records, count=1, seed=20260826, excluded={"P000", "P001"})
