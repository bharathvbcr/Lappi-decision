"""The tracked VAL_ROWS writer (GAP-W9B-NO-TRACKED-VAL-ROWS-FILE-WRITER-2026-10-18): determinism,
the binding to the shard set, the split refusals, and an empty val split as NotRun."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from data_fixtures import small_corpus

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.errors import HeldOutViolation
from qd_data.manifest import Manifest, build_manifests
from qd_data.mixture import build_mixture
from qd_data.rows import DataRow
from qd_data.split import split
from qd_data.val_requests import (
    ValRowsNotRun,
    binding_path,
    check_binding,
    read_val_rows,
    write_val_rows,
)

CONFIG = DataConfig.legacy_v5()


def _shard_set(root: Path) -> tuple[Path, Path, list[DataRow]]:
    """A shard set's ``data/pool`` as the pipeline writes it, and its val rows."""
    mixture = build_mixture(small_corpus(24), config=CONFIG)
    dd = dedupe(list(mixture.rows), config=CONFIG)
    report = split(dd, config=CONFIG)
    manifests = build_manifests(
        config=CONFIG, mixture=mixture, dedupe_report=dd, split_report=report,
        provenance={"built_at_utc": "2026-01-01T00:00:00+00:00"},
    )
    pool = root / "data" / "pool"
    pool.mkdir(parents=True)
    for name in ("train", "val"):
        manifests[name].write(pool / f"{name}.json", allow_not_run=True)
    return pool / "train.json", pool / "val.json", list(report.rows_by_split["val"])


def _write(tmp: Path, name: str = "val_rows.jsonl", **kw: object) -> tuple[Path, Path, Path]:
    train, val, rows = _shard_set(tmp / "set")
    out = tmp / name
    write_val_rows(
        kw.get("rows", rows), val_manifest=val, train_manifest=train, out=out,  # type: ignore[arg-type]
        config=CONFIG, repo_root=tmp / "set",
        **{k: v for k, v in kw.items() if k != "rows"},  # type: ignore[arg-type]
    )
    return train, val, out


def test_rows_are_written_in_manifest_order_and_read_back(tmp_path: Path) -> None:
    _, val, out = _write(tmp_path)
    ids = [e["row_id"] for e in json.loads(val.read_text(encoding="utf-8"))["entries"]]
    rows = read_val_rows(out)
    assert [r.row_id for r in rows] == ids and rows
    assert {r.split for r in rows} == {"val"}


def test_written_twice_is_byte_identical_whatever_the_input_order(tmp_path: Path) -> None:
    train, val, rows = _shard_set(tmp_path / "set")
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    kw = {"val_manifest": val, "train_manifest": train, "config": CONFIG,
          "repo_root": tmp_path / "set"}
    write_val_rows(rows, out=a, **kw)  # type: ignore[arg-type]
    write_val_rows(list(reversed(rows)), out=b, **kw)  # type: ignore[arg-type]
    assert a.read_bytes() == b.read_bytes()
    assert binding_path(a).read_bytes() == binding_path(b).read_bytes()


def test_the_binding_records_both_hashes_and_matches_its_shard_set(tmp_path: Path) -> None:
    train, val, out = _write(tmp_path)
    binding = json.loads(binding_path(out).read_text(encoding="utf-8"))
    assert binding["val_manifest_sha256"] == hashlib.sha256(val.read_bytes()).hexdigest()
    assert binding["train_manifest_sha256"] == hashlib.sha256(train.read_bytes()).hexdigest()
    assert binding["data_snapshot_hash"] == json.loads(val.read_text())["data_snapshot_hash"]
    assert binding["rows_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    check_binding(out, train_manifest_sha256=binding["train_manifest_sha256"])


def test_a_mismatched_binding_is_refused(tmp_path: Path) -> None:
    _, _, out = _write(tmp_path)
    with pytest.raises(ValueError, match="train_manifest_sha256"):
        check_binding(out, train_manifest_sha256="ab" * 32)
    out.write_bytes(out.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="rows_sha256"):
        check_binding(out, train_manifest_sha256=json.loads(
            binding_path(out).read_text())["train_manifest_sha256"])


def test_a_hand_made_file_with_no_binding_is_not_run(tmp_path: Path) -> None:
    hand = tmp_path / "hand.jsonl"
    hand.write_text("", encoding="utf-8")
    with pytest.raises(ValRowsNotRun, match="binding"):
        check_binding(hand, train_manifest_sha256="ab" * 32)


def test_a_train_manifest_as_the_val_manifest_is_refused(tmp_path: Path) -> None:
    train, _, rows = _shard_set(tmp_path / "set")
    with pytest.raises(ValueError, match="not 'val'"):
        write_val_rows(rows, val_manifest=train, train_manifest=train, out=tmp_path / "o.jsonl",
                       config=CONFIG, repo_root=tmp_path / "set")
    assert not (tmp_path / "o.jsonl").exists()


def test_a_held_out_path_is_refused_before_any_byte_is_read(tmp_path: Path) -> None:
    train, _, rows = _shard_set(tmp_path / "set")
    missing = tmp_path / "set" / "data" / "heldout" / "val.json"  # does not exist: never opened
    with pytest.raises(HeldOutViolation):
        write_val_rows(rows, val_manifest=missing, train_manifest=train,
                       out=tmp_path / "o.jsonl", config=CONFIG, repo_root=tmp_path / "set")
    assert not (tmp_path / "o.jsonl").exists()


def test_a_heldout_declaring_manifest_is_refused(tmp_path: Path) -> None:
    train, val, rows = _shard_set(tmp_path / "set")
    raw = json.loads(val.read_text(encoding="utf-8"))
    raw["split"] = "heldout"
    moved = val.with_name("pool_x.json")
    moved.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        write_val_rows(rows, val_manifest=moved, train_manifest=train, out=tmp_path / "o.jsonl",
                       config=CONFIG, repo_root=tmp_path / "set")


def test_rows_that_are_not_the_manifests_are_refused(tmp_path: Path) -> None:
    train, val, rows = _shard_set(tmp_path / "set")
    with pytest.raises(ValueError, match="ids differ"):
        write_val_rows(rows[1:], val_manifest=val, train_manifest=train,
                       out=tmp_path / "o.jsonl", config=CONFIG, repo_root=tmp_path / "set")


def test_the_row_cap_fails_closed(tmp_path: Path) -> None:
    train, val, rows = _shard_set(tmp_path / "set")
    assert len(rows) > 1
    with pytest.raises(ValueError, match="cap"):
        write_val_rows(rows, val_manifest=val, train_manifest=train, out=tmp_path / "o.jsonl",
                       config=CONFIG, repo_root=tmp_path / "set", max_rows=len(rows) - 1)
    assert not (tmp_path / "o.jsonl").exists()


def test_an_empty_val_split_is_not_run_not_an_empty_pass(tmp_path: Path) -> None:
    train, val, _ = _shard_set(tmp_path / "set")
    m = Manifest.read(val)
    empty = Manifest(**{**{f: getattr(m, f) for f in m.__dataclass_fields__}, "entries": ()})
    empty.write(val, allow_not_run=True)
    with pytest.raises(ValRowsNotRun, match="empty"):
        write_val_rows([], val_manifest=val, train_manifest=train, out=tmp_path / "o.jsonl",
                       config=CONFIG, repo_root=tmp_path / "set")
    assert not (tmp_path / "o.jsonl").exists()
