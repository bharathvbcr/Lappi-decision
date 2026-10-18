"""``tools/real_ft_run.py``'s ft row records the train manifest it read.

``corpus_facts`` reads ``data/pool/train.json`` beside the shards once, hashes those bytes,
and hands the digest (and the manifest's ``val_without_train`` block) to ``main``, which
stages them for every ft row. An eval or verdict row states nothing about the manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import real_ft_run as rft  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_train.data_access import _file_sha256  # noqa: E402
from qd_train.ledger import Ledger  # noqa: E402

BLOCK = {"count": 1, "families": {"synth.general/access_route": 3}}


def _manifest(out: Path, **extra: object) -> Path:
    path = out / rft.TRAIN_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "data_snapshot_hash": "d" * 64, "status": {"state": "ran", "passed": True},
        "mixture": {"n_input": {"bigcode/commitpackft": 5}},
        "config_fingerprint": DataConfig().fingerprint(), **extra,
    }), encoding="utf-8")
    return path


def _facts(out: Path) -> rft.CorpusFacts:
    return rft.corpus_facts(out, data_snapshot_hash="d" * 64, repo_history=True,
                            commitpackft=None, config=DataConfig())


def _write_ft_row(tmp_path: Path, run_kind: str = "ft") -> dict[str, object]:
    reader = SimpleNamespace(header=SimpleNamespace(
        data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64))
    recipe: dict[str, object] = {"device": "cpu", "hidden": 8, "heads": 2}
    ledger = Ledger(tmp_path / "ledger.jsonl")
    recorder = rft._recorder(
        ledger, reader=reader, seed=0, recipe=recipe, run_kind=run_kind,  # type: ignore[arg-type]
        quick_reasons=[], notes="test", wall_clock_s=1.0, cost=None,
    )
    with recorder:
        pass
    ledger.verify_chain()
    return json.loads(ledger.raw_lines()[-1])


@pytest.fixture(autouse=True)
def _clean_staging():
    rft._RUN_TRAIN_MANIFEST.clear()
    yield
    rft._RUN_TRAIN_MANIFEST.clear()


def test_corpus_facts_hashes_the_bytes_it_read(tmp_path: Path) -> None:
    path = _manifest(tmp_path, val_without_train=BLOCK)
    facts = _facts(tmp_path)
    assert facts.train_manifest_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert facts.train_manifest_sha256 == _file_sha256(path)  # TrainingData's definition
    assert facts.val_without_train == BLOCK


def test_an_edited_manifest_gives_a_different_digest(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    before = _facts(tmp_path).train_manifest_sha256
    path.write_bytes(path.read_bytes() + b"\n")  # same JSON, different file
    after = _facts(tmp_path).train_manifest_sha256
    assert before is not None and after is not None and before != after


def test_a_manifest_without_the_block_records_it_as_not_recorded(tmp_path: Path) -> None:
    _manifest(tmp_path)
    assert _facts(tmp_path).val_without_train is None


def test_the_ft_row_carries_what_main_staged(tmp_path: Path) -> None:
    path = _manifest(tmp_path, val_without_train=BLOCK)
    facts = _facts(tmp_path)
    rft._RUN_TRAIN_MANIFEST.update(
        train_manifest_sha256=facts.train_manifest_sha256,
        val_without_train=facts.val_without_train,
    )
    line = _write_ft_row(tmp_path)
    assert line["train_manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert line["val_without_train"] == BLOCK


def test_a_row_that_is_not_an_ft_row_states_nothing_about_the_manifest(tmp_path: Path) -> None:
    rft._RUN_TRAIN_MANIFEST.update(train_manifest_sha256="a" * 64, val_without_train=BLOCK)
    line = _write_ft_row(tmp_path, run_kind="eval")
    assert "train_manifest_sha256" not in line and "val_without_train" not in line


def test_nothing_staged_means_not_recorded(tmp_path: Path) -> None:
    line = _write_ft_row(tmp_path)
    assert "train_manifest_sha256" not in line and "val_without_train" not in line
