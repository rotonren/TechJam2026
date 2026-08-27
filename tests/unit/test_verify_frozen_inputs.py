from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def test_verify_manifest_streams_valid_files_and_prints_only_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from tools.verify_frozen_inputs import verify_manifest

    target = tmp_path / "data" / "frozen.bin"
    target.parent.mkdir()
    target.write_bytes(b"frozen")
    manifest = {"data/frozen.bin": hashlib.sha256(b"frozen").hexdigest()}

    assert verify_manifest(tmp_path, manifest) == 0
    assert capsys.readouterr().out == "data/frozen.bin ok\n"


@pytest.mark.parametrize("relative_path", ["../outside", "/absolute", "data/../escape"])
def test_verify_manifest_rejects_paths_outside_root(tmp_path: Path, relative_path: str) -> None:
    from tools.verify_frozen_inputs import verify_manifest

    with pytest.raises(ValueError, match="manifest"):
        verify_manifest(tmp_path, {relative_path: "a" * 64})


def test_verify_manifest_reports_missing_and_mismatch_without_other_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from tools.verify_frozen_inputs import verify_manifest

    target = tmp_path / "data" / "frozen.bin"
    target.parent.mkdir()
    target.write_bytes(b"wrong")
    status = verify_manifest(tmp_path, {
        "data/frozen.bin": hashlib.sha256(b"expected").hexdigest(),
        "data/missing.bin": "b" * 64,
    })

    assert status == 1
    assert capsys.readouterr().out.splitlines() == ["data/frozen.bin mismatch", "data/missing.bin missing"]


def test_verify_manifest_rejects_an_invalid_or_noncanonical_manifest(tmp_path: Path) -> None:
    from tools.verify_frozen_inputs import verify_manifest

    with pytest.raises(ValueError, match="manifest"):
        verify_manifest(tmp_path, {"data/file": "UPPER"})
    with pytest.raises(ValueError, match="manifest"):
        verify_manifest(tmp_path, {"data\\file": "a" * 64})


def test_main_rejects_a_manifest_with_replaced_frozen_path(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from tools import verify_frozen_inputs as frozen

    altered = dict(frozen.FROZEN_INPUTS)
    altered["other/file"] = altered.pop("data/catalog.jsonl")
    monkeypatch.setattr(frozen, "FROZEN_INPUTS", altered)

    with pytest.raises(SystemExit, match="1"):
        frozen.main()

    assert capsys.readouterr().out == "manifest anomaly\n"
