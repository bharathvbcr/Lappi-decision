"""Rule 3 at the ``tools/real_ft_run.py`` readers that sat outside the shard door
(GAP-HELDOUT-DATA-READERS-OUTSIDE-SHARDREADER-AUDIT-2026-10-19): the train manifest
``corpus_facts`` and ``check_defect_source`` read by path, and the ``probe.json`` and corpus
files ``open_candidate_set`` read under a directory that had only a directory check.

A link into a held-out root, or a held-out marker in the given path, is refused before any byte of
the file is read. The link's target holds a valid file, so before the door each reader took it.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import real_ft_run as rft  # noqa: E402
from test_real_ft_arm_b_probe import _candidate_world, _open  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.errors import HeldOutViolation  # noqa: E402

MARKERS = ["heldout", "held_out", "held-out"]
BODY = json.dumps({
    "data_snapshot_hash": "d" * 64, "status": {"state": "ran", "passed": True},
    "mixture": {"n_input": {"bigcode/commitpackft": 5}},
    "config_fingerprint": DataConfig().fingerprint(),
})


def _config(root: Path) -> DataConfig:
    """Held-out root ``<root>/sealed``: no marker in its name, so only the root layer can refuse
    a link into it."""
    return dataclasses.replace(DataConfig(), held_out_roots=(root / "sealed",))


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    for name in ("read_text", "read_bytes", "open"):
        real = getattr(Path, name)

        def hook(self: Path, *a: object, _real=real, _name=name, **k: object):
            seen.append(f"{_name}:{self}")
            return _real(self, *a, **k)

        monkeypatch.setattr(Path, name, hook)
    return seen


def _linked_train_manifest(out: Path, root: Path) -> Path:
    target = root / "sealed" / "train.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(BODY, encoding="utf-8")
    link = out / rft.TRAIN_MANIFEST
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    return link


def _facts(out: Path, config: DataConfig) -> rft.CorpusFacts:
    return rft.corpus_facts(out, data_snapshot_hash="d" * 64, repo_history=True,
                            commitpackft=None, config=config)


def test_corpus_facts_reads_a_clean_manifest(tmp_path: Path) -> None:
    path = tmp_path / "out" / rft.TRAIN_MANIFEST
    path.parent.mkdir(parents=True)
    path.write_text(BODY, encoding="utf-8")
    assert _facts(tmp_path / "out", _config(tmp_path)) is not None


def test_corpus_facts_refuses_a_train_manifest_linked_into_a_held_out_root(
    tmp_path: Path, reads: list[str]
) -> None:
    out = tmp_path / "out"
    _linked_train_manifest(out, tmp_path)
    reads.clear()
    with pytest.raises(HeldOutViolation):
        _facts(out, _config(tmp_path))
    assert [r for r in reads if r.endswith("train.json")] == []


def test_check_defect_source_refuses_a_train_manifest_linked_into_a_held_out_root(
    tmp_path: Path, reads: list[str]
) -> None:
    out = tmp_path / "out"
    _linked_train_manifest(out, tmp_path)
    reads.clear()
    with pytest.raises(HeldOutViolation):
        rft.check_defect_source(out, defect_class=None, config=_config(tmp_path))
    assert [r for r in reads if r.endswith("train.json")] == []


@pytest.mark.parametrize("marker", MARKERS)
def test_both_train_manifest_readers_refuse_a_held_out_marker_in_the_out_path(
    tmp_path: Path, reads: list[str], marker: str
) -> None:
    out = tmp_path / marker / "out"
    path = out / rft.TRAIN_MANIFEST
    path.parent.mkdir(parents=True)
    path.write_text(BODY, encoding="utf-8")
    reads.clear()
    with pytest.raises(HeldOutViolation):
        _facts(out, _config(tmp_path))
    with pytest.raises(HeldOutViolation):
        rft.check_defect_source(out, defect_class=None, config=_config(tmp_path))
    assert [r for r in reads if r.endswith("train.json")] == []


# --- open_candidate_set: probe.json and the corpus's two files ---------------------------------


def _link_member(w, path: Path) -> None:
    """Replace ``path`` by a link to a byte-identical copy under ``<tmp>/sealed``."""
    target = w.tmp / "sealed" / f"{path.parent.name}-{path.name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)


@pytest.mark.parametrize("member", ["probe", "examples", "manifest"])
def test_open_candidate_set_refuses_a_member_linked_into_a_held_out_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, member: str
) -> None:
    w = _candidate_world(tmp_path, monkeypatch)
    path = {"probe": w.set_out / "probe.json", "examples": w.corpus / "examples.jsonl",
            "manifest": w.corpus / "manifest.json"}[member]
    _link_member(w, path)
    seen: list[str] = []
    real = Path.read_bytes
    real_text = Path.read_text

    def spy_bytes(self: Path):
        seen.append(str(self))
        return real(self)

    def spy_text(self: Path, *a: object, **k: object):
        seen.append(str(self))
        return real_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_bytes", spy_bytes)
    monkeypatch.setattr(Path, "read_text", spy_text)
    with pytest.raises(SystemExit, match="held-out"):
        _open(w, config=_config(tmp_path))
    assert [s for s in seen if "sealed" in s or s == str(path)] == []


def test_open_candidate_set_still_opens_a_clean_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = _candidate_world(tmp_path, monkeypatch)
    assert _open(w, config=_config(tmp_path)) is not None
