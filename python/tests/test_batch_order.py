"""``tools/real_ft_run.py --batch-order seed``: v5 varies the batch order with the training seed.

Fable's seed-order ruling (AUDIT/post-f-2026-10-02/fable-seed-order-ruling.md, sections 1 and
4): F's seeds all trained on one order, because the epoch arm planned once at
``DataConfig().seed``. Pinned here:

* without the flag, every seed plans at ``DataConfig().seed`` -- the same order, the same
  ``train.consumed_digest`` -- and the ft row's recipe and ``recipe_hash`` are the ones the
  tool wrote before this change (:data:`BASE_REF`, the commit below it, run on the same toy
  corpus in the same test);
* with it, seed s plans at s: the orders and consumed digests differ by seed, the recipe gains
  only ``batch_order = "seed"`` (the constant, so the seeds share one ``recipe_hash``), and both
  arms train on their seed's plan;
* every ft row carries ``corpus.plan_seed``, ``corpus.plan_order_digest`` (sha256 over the
  plan's ``(bucket, rows)``, :func:`real_ft_run.plan_order_digest`) and ``train.consumed_digest``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
from test_real_ft_rungd_flags import (  # noqa: E402, F401
    REV,
    _base_module,
    _deterministic_restored,
    _patch_module,
    _run,
    corpus,
)

from qd_data.config import DataConfig  # noqa: E402
from qd_train.ledger import Ledger, LedgerRow  # noqa: E402
from qd_train.shards import ShardReader  # noqa: E402

#: The commit below ``--batch-order``: the tool whose non-flag rows this change must reproduce.
#: A differential against a fixed ref: once the tool's non-flag recipe legitimately moves,
#: refresh it to the commit below that change.
BASE_REF = "f168576"
CONFIG = DataConfig()
SHORT = ("--seeds", "0", "1", "--max-steps", "3")


def _value(row: LedgerRow, name: str) -> Any:
    return row.metrics[name].to_json()["value"]


def _ft(rows: list[LedgerRow]) -> list[LedgerRow]:
    found = [r for r in rows if r.run_kind == "ft"]
    assert [r.protocol.seed for r in found] == [0, 1], [r.protocol.seed for r in found]
    return found


def _reader(corpus) -> ShardReader:  # noqa: F811 - the fixture imported above
    return ShardReader(corpus.out / "shards" / "train", config=CONFIG, repo_root=corpus.out,
                       expect_rev=REV)


# --- the pieces -------------------------------------------------------------------------------


def test_the_plan_seed_is_the_training_seed_only_under_the_flag():
    assert rft.plan_seed_for(None, seed=3, config=CONFIG) == CONFIG.seed
    assert rft.plan_seed_for(rft.BATCH_ORDER_SEED, seed=3, config=CONFIG) == 3
    with pytest.raises(ValueError, match="batch_order"):
        rft.plan_seed_for("shuffled", seed=3, config=CONFIG)


_OFF = {"lower_layers_n": 0, "lower_lr_scale": 1.0, "beta2": rft.DEFAULT_BETA2,
        "permutation": None, "replay": None}


def test_the_recipe_names_the_constant_only_when_the_flag_is_given():
    assert rft._recipe_pieces(**_OFF) == {}
    assert rft._recipe_pieces(**_OFF, batch_order="seed") == {"batch_order": "seed"}
    assert "batch_order" in rft.RECIPE_PIECE_KEYS
    with pytest.raises(ValueError, match="batch_order"):
        rft._recipe_pieces(**_OFF, batch_order="0")


def test_the_order_digest_is_the_plans_bucket_and_rows_in_order(corpus):  # noqa: F811
    reader = _reader(corpus)
    tokens = int(max(reader.header.buckets))
    plans = reader._plan(batch_tokens=tokens, seed=7, epoch=0)
    body = json.dumps([[p.bucket, list(p.rows)] for p in plans], separators=(",", ":"))
    want = hashlib.sha256(body.encode()).hexdigest()
    assert rft.plan_order_digest(reader, batch_tokens=tokens, plan_seed=7) == want
    assert rft.plan_order_digest(reader, batch_tokens=tokens, plan_seed=8) != want
    narrow = min(int(p.width) for p in plans)
    kept = [[p.bucket, list(p.rows)] for p in plans if p.width <= narrow]
    assert rft.plan_order_digest(
        reader, batch_tokens=tokens, plan_seed=7, max_width=narrow
    ) == hashlib.sha256(json.dumps(kept, separators=(",", ":")).encode()).hexdigest()


def test_batch_order_is_refused_where_nothing_trains(tmp_path):
    with pytest.raises(SystemExit, match="--batch-order"):
        rft.main([
            "--out", str(tmp_path), "--rev", REV, "--batch-order", "seed",
            "--score-checkpoint", str(tmp_path / "epoch-seed0-cpu.json"),
            "--ft-ledger", str(tmp_path / "ft.jsonl"),
        ])


# --- end to end: the epoch arm, three runs on one toy corpus ---------------------------------


def test_the_flag_varies_the_order_by_seed_and_without_it_the_rows_are_the_base_commits(
    corpus, tmp_path, monkeypatch  # noqa: F811
):
    base = _base_module(BASE_REF, tmp_path, monkeypatch)
    before = _ft(_run(corpus, tmp_path / "base.jsonl", *SHORT, module=base, short=True))
    plain = _ft(_run(corpus, tmp_path / "plain.jsonl", *SHORT, short=True))
    flag = _ft(_run(corpus, tmp_path / "flag.jsonl", *SHORT, "--batch-order", "seed",
                    short=True))

    # Absent: the base commit's recipe and recipe_hash, and its order (the consumed digest).
    for old, new in zip(before, plain, strict=True):
        assert new.recipe == old.recipe and "batch_order" not in new.recipe
        assert new.protocol.recipe_hash == old.protocol.recipe_hash
        assert _value(new, "train.consumed_digest") == _value(old, "train.consumed_digest")
    reader = _reader(corpus)
    # No --batch-tokens: the widest bucket, which adds no recipe key (_resolve_batch_tokens).
    assert "batch_tokens" not in plain[0].recipe
    tokens = int(max(reader.header.buckets))
    one_order = rft.plan_order_digest(reader, batch_tokens=tokens, plan_seed=CONFIG.seed)
    assert [_value(r, "corpus.plan_seed") for r in plain] == [CONFIG.seed, CONFIG.seed]
    assert [_value(r, "corpus.plan_order_digest") for r in plain] == [one_order, one_order]
    assert len({_value(r, "train.consumed_digest") for r in plain}) == 1, "one order, F's case"

    # Given: seed s plans at s, so the orders and what was consumed differ by seed...
    assert [_value(r, "corpus.plan_seed") for r in flag] == [0, 1]
    assert [_value(r, "corpus.plan_order_digest") for r in flag] == [
        rft.plan_order_digest(reader, batch_tokens=tokens, plan_seed=s) for s in (0, 1)
    ]
    assert len({_value(r, "corpus.plan_order_digest") for r in flag}) == 2
    assert len({_value(r, "train.consumed_digest") for r in flag}) == 2
    # ...while the recipe is the plain one plus the constant, one recipe_hash for both seeds.
    for p, f in zip(plain, flag, strict=True):
        assert f.recipe == {**p.recipe, "batch_order": "seed"}
    assert flag[0].protocol.recipe_hash == flag[1].protocol.recipe_hash
    assert flag[0].protocol.recipe_hash != plain[0].protocol.recipe_hash


# --- end to end: arm 2 under the flag ---------------------------------------------------------


def test_arm_2_trains_each_seed_on_its_own_plans_cut(corpus, tmp_path):  # noqa: F811
    """The memorise arm reads plan_small from the seed's plan, not the first seed's. It is run
    for one pass: its floor claims are not under test here, only which batches it trained.
    ``--max-width`` is the narrowest bucket's width, so arm 2's plan is a cut of the epoch's."""
    reader = _reader(corpus)
    tokens = int(max(reader.header.buckets))
    width = min(int(p.width) for p in reader._plan(batch_tokens=tokens, seed=0, epoch=0))
    ledger = tmp_path / "m.jsonl"
    argv = [
        "--out", str(corpus.out), "--rev", REV, "--devices", "cpu", "--seeds", "0", "1",
        "--passes", "1", "--ledger", str(ledger), "--batch-order", "seed",
        "--max-width", str(width),
    ]
    with (
        pytest.MonkeyPatch.context() as mp,
        _deterministic_restored(),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        _patch_module(mp, rft, corpus)
        rft.main(argv)
    rows = [r for r in Ledger(ledger).rows() if r.run_kind == "ft"]
    assert [r.recipe["tag"] for r in rows] == ["memorise", "memorise"]
    assert [_value(r, "corpus.plan_seed") for r in rows] == [0, 1]
    assert [_value(r, "corpus.plan_order_digest") for r in rows] == [
        rft.plan_order_digest(reader, batch_tokens=tokens, plan_seed=s, max_width=width)
        for s in (0, 1)
    ]
    assert len({_value(r, "train.consumed_digest") for r in rows}) == 2
