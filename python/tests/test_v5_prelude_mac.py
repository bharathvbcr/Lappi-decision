"""campaign/post-f-queue/v5_prelude_mac.py: v5's Mac prelude, on L-v5-train's toy corpus.

The prelude stops ``real_ft_run.main`` at ``Ledger(args.ledger)`` and prints, from the tool's own
code, the order digest of every plan seed 0-4, the seed-invariant plan shape, the noul-weight
position count and (with C1) the permutation sentence. Pinned here, on the corpus and module
patch ``test_real_ft_rungd_flags`` already uses (CPU, no tower is loaded):

* the record's digests are ``plan_order_digest`` at plan seeds 0-4, five distinct ones, and
  ``seed_arms(...).shape()`` is one value across them; main never opens its ledger;
* a run without ``--batch-order seed``, a shape that moves with the seed, or two seeds sharing a
  digest fails with no record;
* the record satisfies the waiter's own check, ``v5_prelude_check`` in ``v5_common.sh``;
* the CLI refuses an argv that is not v5 seed 0's.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
from test_real_ft_rungd_flags import (  # noqa: E402, F401
    REV,
    _deterministic_restored,
    _patch_module,
    corpus,
)

from qd_data.config import DataConfig  # noqa: E402
from qd_train.shards import ShardReader  # noqa: E402

#: QD_V5_QUEUE_DIR points the tests at another copy of the queue directory (the fail-first run
#: uses a tree without the prelude), as in test_v5_queue_scripts.py.
QUEUE = Path(os.environ.get("QD_V5_QUEUE_DIR", str(REPO / "campaign" / "post-f-queue")))
PRELUDE = QUEUE / "v5_prelude_mac.py"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
#: v5 seed 0's recipe for the default decisions (v5_common.sh v5_recipe; the waiter's tests pin it).
V5_RECIPE = [
    "--optimizer",
    "master",
    "--lr",
    "1e-5",
    "--epoch",
    "--no-memorise",
    "--batch-tokens",
    "35403",
    "--lower-layers-n",
    "8",
    "--lower-layers-lr-scale",
    "0.1",
    "--checkpoint-skip-layers",
    "6",
    "--min-lr",
    "0",
    "--batch-order",
    "seed",
]


def _prelude():
    spec = importlib.util.spec_from_file_location("v5_prelude_mac", PRELUDE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _toy_argv(corpus, tmp_path: Path, *extra: str) -> list[str]:  # noqa: F811
    return [
        "--out",
        str(corpus.out),
        "--rev",
        REV,
        "--devices",
        "cpu",
        "--seeds",
        "0",
        "--epoch",
        "--no-memorise",
        "--ledger",
        str(tmp_path / "never.jsonl"),
        "--real-backbone",
        str(corpus.snapshot),
        *extra,
    ]


def _run(corpus, argv: list[str], mp: pytest.MonkeyPatch, **kw) -> dict:  # noqa: F811
    prelude = _prelude()
    with _deterministic_restored(), contextlib.redirect_stdout(io.StringIO()):
        _patch_module(mp, rft, corpus)
        return prelude.run_prelude(rft, argv, commit=COMMIT, noul_weight=4.0, **kw)


def _reader(corpus) -> ShardReader:  # noqa: F811
    return ShardReader(
        corpus.out / "shards" / "train", config=DataConfig(), repo_root=corpus.out, expect_rev=REV
    )


def test_the_record_carries_every_plan_seeds_order_and_one_shape(corpus, tmp_path):  # noqa: F811
    argv = _toy_argv(corpus, tmp_path, "--batch-order", "seed")
    real_ledger = rft.Ledger
    with pytest.MonkeyPatch.context() as mp:
        record = _run(corpus, argv, mp)
    assert rft.Ledger is real_ledger, "the prelude put Ledger back"
    assert not (tmp_path / "never.jsonl").exists(), "main opened its ledger"
    reader = _reader(corpus)
    tokens = int(max(reader.header.buckets))
    want = {
        str(s): rft.plan_order_digest(reader, batch_tokens=tokens, plan_seed=s) for s in range(5)
    }
    assert record["plan_order_digest"] == want
    assert len(set(want.values())) == 5
    assert record["batch_order"] == "seed" and record["plan_seeds"] == [0, 1, 2, 3, 4]
    assert record["shape_equal_across_seeds"] is True and record["ok"] is True
    assert record["plan_batches"] == len(list(reader._plan(batch_tokens=tokens, seed=0, epoch=0)))
    assert record["shard_hash"] == reader.header.shard_hash()
    assert record["code_commit"] == COMMIT and record["argv"] == argv
    assert record["option_permutation_seed"] is None and record["sentence"] is None
    assert record["noul_weight"] == 4.0
    assert isinstance(record["noul_weighted_positions"], int)
    assert record["noul_supervised_positions"] >= record["noul_weighted_positions"]


def test_a_run_without_batch_order_seed_fails(corpus, tmp_path):  # noqa: F811
    prelude = _prelude()
    with (
        pytest.MonkeyPatch.context() as mp,
        pytest.raises(prelude.PreludeFailed) as failed,
        _deterministic_restored(),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        _patch_module(mp, rft, corpus)
        prelude.run_prelude(rft, _toy_argv(corpus, tmp_path), commit=COMMIT, noul_weight=4.0)
    assert failed.value.code == 3 and "--batch-order None" in str(failed.value)


def test_a_shape_that_moves_with_the_seed_fails(corpus, tmp_path):  # noqa: F811
    prelude = _prelude()
    real_shape = rft.SeedArms.shape

    def moving(self):
        return (*real_shape(self)[:-1], real_shape(self)[-1] + self.plan_seed)

    with pytest.MonkeyPatch.context() as mp, pytest.raises(prelude.PreludeFailed) as failed:
        mp.setattr(rft.SeedArms, "shape", moving)
        with _deterministic_restored(), contextlib.redirect_stdout(io.StringIO()):
            _patch_module(mp, rft, corpus)
            prelude.run_prelude(
                rft,
                _toy_argv(corpus, tmp_path, "--batch-order", "seed"),
                commit=COMMIT,
                noul_weight=4.0,
            )
    assert failed.value.code == 5 and "plan seed 1's shape" in str(failed.value)


def test_two_plan_seeds_sharing_a_digest_fail(corpus, tmp_path):  # noqa: F811
    prelude = _prelude()
    real_digest = rft.plan_order_digest

    def collide(reader, *, batch_tokens, plan_seed, max_width=None):
        return real_digest(
            reader, batch_tokens=batch_tokens, plan_seed=min(plan_seed, 3), max_width=max_width
        )

    with pytest.MonkeyPatch.context() as mp, pytest.raises(prelude.PreludeFailed) as failed:
        mp.setattr(rft, "plan_order_digest", collide)
        with _deterministic_restored(), contextlib.redirect_stdout(io.StringIO()):
            _patch_module(mp, rft, corpus)
            prelude.run_prelude(
                rft,
                _toy_argv(corpus, tmp_path, "--batch-order", "seed"),
                commit=COMMIT,
                noul_weight=4.0,
            )
    assert failed.value.code == 5
    assert "share an order digest" in str(failed.value) or "SeedArms digests" in str(failed.value)


def test_the_record_passes_the_waiters_own_check(corpus, tmp_path):  # noqa: F811
    """v5_common.sh v5_prelude_check on a record the prelude wrote. The toy run cannot carry
    v5's real recipe (an F-shaped recipe needs the 2B tower's layers) and its tiny snapshot has
    no tokenizer.json, so the record's tokenizer sha is None and the waiter refuses it as it
    stands. Two fields are then supplied -- argv (v5 seed 0's recipe) and the tokenizer sha (a
    stand-in file's) -- and every other field is the prelude's."""
    prelude = _prelude()
    with pytest.MonkeyPatch.context() as mp:
        record = _run(corpus, _toy_argv(corpus, tmp_path, "--batch-order", "seed"), mp)
    record["argv"] = ["--out", "/x", *V5_RECIPE, "--seeds", "0"]
    path = tmp_path / "record.json"
    tok = tmp_path / "tokenizer.json"
    tok.write_text('{"model": "stand-in"}')
    tok_sha = hashlib.sha256(tok.read_bytes()).hexdigest()
    assert (corpus.out / "shards" / "train" / "header.json").is_file()

    def check(rec: dict) -> list[str]:
        if path.exists():
            path.unlink()
        prelude.write_record(path, rec)
        script = f"""
source "{QUEUE}/post_f_common.sh" || exit 90
Q="{tmp_path}"; PF="{tmp_path}"
source "{QUEUE}/idle_common.sh" || exit 91
source "{QUEUE}/v5_common.sh" || exit 92
PY="{sys.executable}"; V5_PRELUDE="{path}"; V5_DATA="{corpus.out}"; V5_LANE_AT={COMMIT}
TOKENIZER_SHA256={tok_sha}
V5_C1=off; V5_C2A=off; V5_C2B=off; V5_LOWER=keep; V5_LRSET=f
v5_recipe; v5_prelude_check; echo "RC=$?"
"""
        r = subprocess.run(
            ["/bin/bash", "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        )
        return [*re.findall(r"^RC=(\d+)$", r.stdout, re.M), r.stdout + r.stderr]

    assert record["tokenizer_json_sha256"] is None
    assert check(record)[0] == "1", "a record with no tokenizer sha must not pass"
    record["tokenizer_json_sha256"] = tok_sha
    got = check(record)
    assert got[0] == "0", got[-1]
    record["plan_order_digest"]["4"] = record["plan_order_digest"]["0"]
    assert check(record)[0] == "1", "two plan seeds sharing a digest must not pass"


V5_ARGV = [
    "--out",
    "/data/v5",
    "--rev",
    "a" * 40,
    *V5_RECIPE,
    "--devices",
    "cuda",
    "--seeds",
    "0",
    "--score-val",
    "--needle",
    "--ood",
    "--wall-clock-cap-s",
    "32400",
]


@pytest.mark.parametrize(
    ("argv", "ok"),
    [
        (V5_ARGV, True),
        ([a for a in V5_ARGV if a not in ("--batch-order", "seed")], False),
        (
            [
                a if a != "0" or i != V5_ARGV.index("--seeds") + 1 else "1"
                for i, a in enumerate(V5_ARGV)
            ],
            False,
        ),
        ([*V5_ARGV, "--noul-weight", "4"], False),
        ([*V5_ARGV, "--shuffled-label", "x"], False),
        ([a for a in V5_ARGV if a != "--score-val"], False),
        ([*V5_ARGV, "--min-lr", "0"], False),
    ],
)
def test_check_argv_wants_v5_seed_0s_argv(argv: list[str], ok: bool) -> None:
    prelude = _prelude()
    if ok:
        prelude.check_argv(argv)
        return
    with pytest.raises(prelude.PreludeFailed) as failed:
        prelude.check_argv(argv)
    assert failed.value.code == 2


@pytest.mark.parametrize(
    "ours",
    [
        ["--root", ".", "--record", "r.json"],
        ["--root", ".", "--record", "r.json", "--noul-weight", "0"],
        ["--root", ".", "--record", "r.json", "--noul-weight", "nan"],
        ["--root", ".", "--record", "r.json", "--noul-weight", "4", "--root", "."],
    ],
)
def test_parse_refuses_its_own_options(ours: list[str]) -> None:
    prelude = _prelude()
    with pytest.raises(prelude.PreludeFailed) as failed:
        prelude.parse([*ours, "--", *V5_ARGV])
    assert failed.value.code == 2


def test_the_cli_writes_nothing_on_a_usage_error(tmp_path) -> None:
    rec = tmp_path / "rec.json"
    r = subprocess.run(
        [
            sys.executable,
            str(PRELUDE),
            "--root",
            str(REPO),
            "--record",
            str(rec),
            "--noul-weight",
            "4",
            "--",
            *V5_ARGV[:-2],
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert r.returncode == 2, r.stdout + r.stderr
    assert "PRELUDE USAGE: the argv has --wall-clock-cap-s 0 time(s)" in r.stdout
    assert not rec.exists()
