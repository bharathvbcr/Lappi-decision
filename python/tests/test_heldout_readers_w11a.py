"""Rule 3 at the readers that sat outside the shard door (GAP-HELDOUT-DATA-READERS-OUTSIDE-
SHARDREADER-AUDIT-2026-10-19): the train/val pool manifests ``split_cache.read_manifests`` opens,
and the corpus, record and manifest files ``ft_linear_control`` hashes for a control's recipe.

Each is refused, through ``qd_train.data_access.assert_path_not_held_out``, when the given path
spells a held-out marker or when a link in it resolves into a held-out root, and refused before
any file is opened: the read hooks below record every ``Path`` read and the tests assert none
happened. The symlink target holds a perfectly valid file, so before the door a reader accepted it.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import ft_linear_control as ftc  # noqa: E402
import split_cache as sc  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.errors import HeldOutViolation  # noqa: E402

MARKERS = ["heldout", "held_out", "held-out"]


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every ``Path.read_text`` / ``read_bytes`` / ``open`` from here on, by path."""
    seen: list[str] = []
    for name in ("read_text", "read_bytes", "open"):
        real = getattr(Path, name)

        def hook(self: Path, *a: object, _real=real, _name=name, **k: object):
            seen.append(f"{_name}:{self}")
            return _real(self, *a, **k)

        monkeypatch.setattr(Path, name, hook)
    return seen


def _config(root: Path) -> DataConfig:
    """Held-out root ``<root>/sealed``: its name carries no marker, so only the root layer can
    refuse a link into it (the marker layer is exercised by the given-path tests)."""
    return dataclasses.replace(DataConfig(), held_out_roots=(root / "sealed",))


def _manifest_body(name: str) -> str:
    entries = [{"row_id": f"{name}-1", "content_hash": "c" * 64}]
    return json.dumps({"n_rows": 1, "entries": entries, "data_snapshot_hash": "d" * 64})


def _held_out_file(repo: Path, name: str, body: str) -> Path:
    target = repo / "sealed" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


def _pool(out: Path) -> Path:
    pool = out / "data" / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    return pool


# --- split_cache.read_manifests (reaches the train and val pool manifests) -----------------------


def test_read_manifests_reads_a_clean_pool(tmp_path: Path) -> None:
    out = tmp_path / "out"
    for name in ("train", "val"):
        (_pool(out) / f"{name}.json").write_text(_manifest_body(name), encoding="utf-8")
    got = sc.read_manifests(out, config=_config(tmp_path), repo=tmp_path)
    assert (got.n_train, got.n_val) == (1, 1)


@pytest.mark.parametrize("name", ["train", "val"])
def test_read_manifests_refuses_a_member_linked_into_a_held_out_root(
    tmp_path: Path, reads: list[str], name: str
) -> None:
    out = tmp_path / "out"
    other = "val" if name == "train" else "train"
    (_pool(out) / f"{other}.json").write_text(_manifest_body(other), encoding="utf-8")
    target = _held_out_file(tmp_path, f"{name}.json", _manifest_body(name))
    (_pool(out) / f"{name}.json").symlink_to(target)
    reads.clear()
    with pytest.raises(HeldOutViolation):
        sc.read_manifests(out, config=_config(tmp_path), repo=tmp_path)
    assert [r for r in reads if "sealed" in r or f"{name}.json" in r] == []


@pytest.mark.parametrize("marker", MARKERS)
def test_read_manifests_refuses_a_held_out_marker_in_the_given_path(
    tmp_path: Path, reads: list[str], marker: str
) -> None:
    out = tmp_path / marker / "out"
    for name in ("train", "val"):
        (_pool(out) / f"{name}.json").write_text(_manifest_body(name), encoding="utf-8")
    reads.clear()
    with pytest.raises(HeldOutViolation):
        sc.read_manifests(out, config=_config(tmp_path), repo=tmp_path)
    assert [r for r in reads if r.endswith(".json")] == []


# --- ft_linear_control.check_shard_set (the same manifests, by --split-cache-shards) -------------


def test_check_shard_set_refuses_a_linked_manifest_before_reading_it(
    tmp_path: Path, reads: list[str]
) -> None:
    shards = tmp_path / "shards"
    (_pool(shards) / "val.json").write_text(_manifest_body("val"), encoding="utf-8")
    target = _held_out_file(tmp_path, "train.json", _manifest_body("train"))
    (_pool(shards) / "train.json").symlink_to(target)
    row = SimpleNamespace(protocol=SimpleNamespace(data_snapshot_hash="d" * 64), row_id="r")
    reads.clear()
    with pytest.raises(HeldOutViolation):
        ftc.check_shard_set(shards, row, config=_config(tmp_path))
    assert [r for r in reads if r.endswith(".json")] == []


# --- ft_linear_control's recipe inputs: a record and a corpus's manifest ---------------------


def test_read_input_bytes_reads_a_clean_file(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(b"abc")
    assert ftc.read_input_bytes(path, config=_config(tmp_path)) == b"abc"


def test_read_input_bytes_refuses_a_link_into_a_held_out_root(
    tmp_path: Path, reads: list[str]
) -> None:
    target = _held_out_file(tmp_path, "record.json", "{}")
    link = tmp_path / "record.json"
    link.symlink_to(target)
    reads.clear()
    with pytest.raises(HeldOutViolation):
        ftc.read_input_bytes(link, config=_config(tmp_path))
    assert reads == []


@pytest.mark.parametrize("marker", MARKERS)
def test_read_input_bytes_refuses_a_held_out_marker_in_the_given_path(
    tmp_path: Path, reads: list[str], marker: str
) -> None:
    path = tmp_path / marker / "record.json"
    path.parent.mkdir()
    path.write_bytes(b"abc")
    reads.clear()
    with pytest.raises(HeldOutViolation):
        ftc.read_input_bytes(path, config=_config(tmp_path))
    assert reads == []


def test_corpus_examples_sha256_refuses_a_linked_manifest(
    tmp_path: Path, reads: list[str]
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    target = _held_out_file(tmp_path, "manifest.json", json.dumps({"examples_sha256": "e" * 64}))
    (corpus / "manifest.json").symlink_to(target)
    reads.clear()
    with pytest.raises(HeldOutViolation):
        ftc.corpus_examples_sha256(corpus, config=_config(tmp_path))
    assert reads == []


def test_corpus_examples_sha256_reads_a_clean_manifest(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text(json.dumps({"examples_sha256": "e" * 64}))
    assert ftc.corpus_examples_sha256(corpus, config=_config(tmp_path)) == "e" * 64
