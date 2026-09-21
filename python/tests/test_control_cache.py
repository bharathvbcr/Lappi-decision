"""The linear-control cache: what it must refuse to serve.

A cache that serves the wrong verdict is worse than no cache. The run would record a paired
margin against an opponent it never faced, and the number would look exactly like a measured
one -- the failure this repository keeps finding, in a new place. So most of these tests are
about misses, not hits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_train import control_cache  # noqa: E402
from qd_train.control_cache import (  # noqa: E402
    control_key,
    load_control,
    store_control,
)

BASE = {
    "train_docs": ["def a(): pass", "def b(): pass", "fn c() {}"],
    "train_labels": ["x", "y", "x"],
    "val_docs": ["def d(): pass", "fn e() {}"],
    "seed": 0,
    "max_iter": 500,
    "hasher_params": (3, 5, 65536),
    "l2_grid": (1e-4, 1e-3),
    "tol": 1e-4,
    "lr": 0.05,
}


def _store(tmp_path, key, correct):
    return store_control(
        tmp_path, key, np.asarray(correct, dtype=bool),
        fitted_s=1.5, n_train=3, l2=1e-3, iterations=42, final_grad_norm=1e-5,
    )


def test_a_stored_verdict_round_trips(tmp_path):
    key = control_key(**BASE)
    _store(tmp_path, key, [True, False])
    hit = load_control(tmp_path, key, expected_n=2)
    assert hit is not None
    assert hit.correct.tolist() == [True, False]
    assert hit.n_train == 3
    assert hit.l2 == 1e-3
    assert hit.iterations == 42


def test_a_missing_entry_is_a_miss_not_an_error(tmp_path):
    assert load_control(tmp_path, control_key(**BASE), expected_n=2) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("train_docs", ["def a(): pass", "def b(): pass", "fn DIFFERENT() {}"]),
        ("train_labels", ["x", "y", "y"]),
        ("val_docs", ["def d(): pass", "fn DIFFERENT() {}"]),
        ("seed", 1),
        ("max_iter", 6000),
        ("hasher_params", (3, 4, 65536)),
        ("l2_grid", (1e-4, 1e-2)),
        ("tol", 1e-5),
        ("lr", 0.1),
    ],
)
def test_every_input_that_changes_the_answer_changes_the_key(field, value):
    """If one of these did not move the key, a run would be served a verdict fitted under
    different conditions -- which is indistinguishable, in the ledger, from a real one."""
    assert control_key(**{**BASE, field: value}) != control_key(**BASE)


def test_the_documents_cannot_collide_by_concatenation():
    """["ab", "c"] and ["a", "bc"] join to the same bytes and must not share a key.

    Without a length prefix they would, and two different corpora would share a verdict.
    """
    a = control_key(**{**BASE, "train_docs": ["ab", "c"], "train_labels": ["x", "y"]})
    b = control_key(**{**BASE, "train_docs": ["a", "bc"], "train_labels": ["x", "y"]})
    assert a != b


def test_editing_the_baseline_implementation_invalidates_every_entry(tmp_path, monkeypatch):
    """A verdict produced by code that no longer exists must never be served.

    This is `GAP-CODE-COMMIT-DIRTY-DOES-NOT-PIN-WHAT-RAN` in miniature: the fix is to hash
    the bytes that ran, not the version that was nominally checked out.
    """
    key_before = control_key(**BASE)
    _store(tmp_path, key_before, [True, False])

    monkeypatch.setattr(control_cache, "_digest_source", lambda: "a-different-baseline")
    key_after = control_key(**BASE)

    assert key_after != key_before
    assert load_control(tmp_path, key_after, expected_n=2) is None


def test_a_verdict_of_the_wrong_length_is_refused(tmp_path):
    """Length disagreement means it cannot be paired with this validation set.

    Serving it would score the model's predictions against another split's answers, which
    is the pairing hazard the paired margin exists to avoid.
    """
    key = control_key(**BASE)
    _store(tmp_path, key, [True, False])
    assert load_control(tmp_path, key, expected_n=3) is None
    assert load_control(tmp_path, key, expected_n=2) is not None


def test_a_corrupt_entry_is_a_miss_rather_than_a_crash(tmp_path):
    key = control_key(**BASE)
    path = _store(tmp_path, key, [True, False])
    path.write_text("{not json", encoding="utf-8")
    assert load_control(tmp_path, key, expected_n=2) is None


def test_a_truncated_entry_is_a_miss(tmp_path):
    key = control_key(**BASE)
    path = _store(tmp_path, key, [True, False])
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["correct"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_control(tmp_path, key, expected_n=2) is None


def test_the_write_leaves_no_temporary_behind(tmp_path):
    """The write is atomic via rename; a leftover .tmp would mean it was not."""
    _store(tmp_path, control_key(**BASE), [True, False])
    assert list(tmp_path.glob("*.tmp")) == []
    assert len(list(tmp_path.glob("linear-control-*.json"))) == 1


def test_storing_creates_the_directory(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    _store(nested, control_key(**BASE), [True, False])
    assert nested.is_dir()
