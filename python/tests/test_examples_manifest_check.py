"""Every reader of a pre-generated corpus refuses an examples file its manifest does not describe
(mistake M18: ``tools/rung0_real_run.py`` took ``--examples`` and ``--manifest-in`` independently
and never hashed the examples against ``examples_sha256``).

The check has one owner, ``qd_data.defect_class.check_examples_manifest``, called by the shared
reader ``qd_train.mutate_adapter.read_example_objects``. Every test fails on the pre-change code:
the reader neither takes ``manifest`` nor looks beside the file.

Torch-free.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qd_data.config import DataConfig
from qd_data.defect_class import check_examples_manifest
from qd_train.mutate_adapter import MalformedExample, read_example_objects

ROW = {"id": "a", "context": "x"}


def _corpus(tmp: Path, *, sha: str | None, name: str = "manifest.json") -> tuple[Path, Path]:
    examples = tmp / "examples.jsonl"
    examples.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    manifest = tmp / name
    body: dict[str, object] = {"data_snapshot_hash": "d" * 64}
    if sha is not None:
        body["examples_sha256"] = sha
    manifest.write_text(json.dumps(body), encoding="utf-8")
    return examples, manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(examples: Path, tmp: Path, **kw: object) -> list[dict[str, object]]:
    return read_example_objects(examples, config=DataConfig(), repo_root=tmp, **kw)  # type: ignore[arg-type]


def test_a_matching_manifest_is_read(tmp_path: Path) -> None:
    examples, manifest = _corpus(tmp_path, sha=None)
    manifest.write_text(json.dumps({"examples_sha256": _sha(examples)}), encoding="utf-8")
    assert _read(examples, tmp_path, manifest=manifest) == [ROW]
    assert _read(examples, tmp_path) == [ROW], "the manifest beside the file is checked too"


def test_an_explicit_manifest_that_describes_other_bytes_is_refused(tmp_path: Path) -> None:
    examples, manifest = _corpus(tmp_path, sha="0" * 64, name="elsewhere.json")
    with pytest.raises(MalformedExample, match="hashes to"):
        _read(examples, tmp_path, manifest=manifest)


def test_the_manifest_beside_the_file_is_checked_without_being_named(tmp_path: Path) -> None:
    examples, _ = _corpus(tmp_path, sha="0" * 64)
    with pytest.raises(MalformedExample, match="hashes to"):
        _read(examples, tmp_path)


def test_a_manifest_that_records_no_digest_is_refused_not_waved_through(tmp_path: Path) -> None:
    examples, manifest = _corpus(tmp_path, sha=None)
    with pytest.raises(MalformedExample, match="records no examples_sha256"):
        _read(examples, tmp_path, manifest=manifest)


def test_a_named_manifest_that_is_absent_is_refused(tmp_path: Path) -> None:
    examples, _ = _corpus(tmp_path, sha="0" * 64)
    with pytest.raises(MalformedExample, match="does not exist"):
        _read(examples, tmp_path, manifest=tmp_path / "absent.json")


def test_no_manifest_at_all_is_reported_as_unchecked_not_as_checked(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    examples.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    assert check_examples_manifest(examples) is False
    (tmp_path / "b").mkdir()
    examples, manifest = _corpus(tmp_path / "b", sha=None)
    manifest.write_text(json.dumps({"examples_sha256": _sha(examples)}), encoding="utf-8")
    assert check_examples_manifest(examples) is True
