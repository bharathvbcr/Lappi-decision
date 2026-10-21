"""Adversarial tests for lane L3's packed ``native_minhash`` (adv/ft-57fc7484-l3).

Not part of the lane: every test here is an attack on ``tools/real_tokenizer_pipeline.py``'s
``native_minhash`` as of ``lane/ft-57fc7484-l3``. Tests named ``test_BREAK_*`` assert the behaviour
a reviewer would expect and FAIL on the lane; the rest pin attacks that held.

The oracle for parity is (a) the pure-Python reference path (no block) and (b) the lane's parent
``d09222d``'s own ``native_minhash``, ``git show``n into a scratch module, so the old and the new
packed implementation are compared on the same seeded random corpora.
"""

from __future__ import annotations

import os
import random
import struct
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_composed_file_near_duplicates import CONFIG, _repo_in, _row
from test_native_minhash_packed import _decisions, _fixture

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import real_tokenizer_pipeline as pipeline  # noqa: E402

import qd_data.dedupe as dedupe_module  # noqa: E402
import qd_data.split as split_module  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.minhash import MinHasher, exact_jaccard, shingle  # noqa: E402
from qd_data.rows import DataRow  # noqa: E402

OLD_REV = "d09222d"


@pytest.fixture
def prep(qd_prep_bin: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(qd_prep_bin))
    return qd_prep_bin


def _old_pipeline() -> types.ModuleType:
    done = subprocess.run(
        ["git", "show", f"{OLD_REV}:tools/real_tokenizer_pipeline.py"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    if done.returncode != 0:
        pytest.skip(f"git show {OLD_REV} failed: {done.stderr.strip()}")
    name = "real_tokenizer_pipeline_old_oracle"
    module = types.ModuleType(name)
    module.__file__ = str(REPO / "tools" / "real_tokenizer_pipeline.py")
    sys.modules[name] = module
    exec(compile(done.stdout, module.__file__, "exec"), module.__dict__)  # noqa: S102
    return module


def _outcome(rows: list[DataRow]) -> tuple[Any, ...]:
    try:
        return ("ok", _decisions(rows))
    except BaseException as exc:  # noqa: BLE001 -- the comparison is of outcomes, raised or not
        return ("raised", type(exc).__name__)


def _in_block(rows: list[DataRow], batch: int, body: Any) -> dict[str, Any]:
    """Run ``body`` inside the block, then ``dedupe``+``split`` so the block's own read and band
    checks are satisfied; return what ``body`` returned, or ``{"refused": message}`` when the
    block refused anywhere. A refusal that only says the block was never read or banded is a
    harness fault, not a verdict, and raises."""
    box: dict[str, Any] = {}
    try:
        with pipeline.native_minhash(rows, config=CONFIG, batch_sets=batch):
            try:
                box.update(body())
            except SystemExit as exc:
                box["refused"] = str(exc)
            _decisions(rows)
    except SystemExit as exc:
        box["refused"] = str(exc)
    msg = box.get("refused", "")
    assert "was installed and never" not in msg and "never banded" not in msg, msg
    return box


_REPOS: dict[str, list[str]] = {}


def _repos(side: str) -> list[str]:
    if side not in _REPOS:
        _REPOS[side] = [_repo_in(side, f"org/{side[0]}{j}-") for j in range(2)]
    return _REPOS[side]


def _random_corpus(seed: int) -> list[DataRow]:
    rnd = random.Random(seed)
    vocab = [f"w{i}" for i in range(16)] + ["é", "中文", "a\x00b", "\U0001f9ea"]
    bases = [
        [rnd.choice(vocab) for _ in range(rnd.choice([1, 2, 4, 5, 6, 9, 25, 60]))]
        for _ in range(rnd.randint(2, 6))
    ]
    rows: list[DataRow] = []
    for i in range(rnd.randint(3, 22)):
        tokens = list(rnd.choice(bases))
        op = rnd.randrange(6)
        if op == 1 and tokens:
            tokens[rnd.randrange(len(tokens))] = rnd.choice(vocab)
        elif op == 2 and len(tokens) > 1:
            tokens.pop()
        elif op == 3:
            tokens.append(rnd.choice(vocab))
        elif op == 4:
            tokens = [rnd.choice(vocab) for _ in range(rnd.randint(1, 30))]
        seps = [" ", "  ", "\n", "\t "]
        text = "".join(t + rnd.choice(seps) for t in tokens).rstrip() or "x"
        repo = rnd.choice(_repos(rnd.choice(["train", "train", "val"])))
        rows.append(_row(f"r{i:03d}", repo=repo, text=text))
    return rows


def test_old_and_new_native_agree_with_the_reference_on_seeded_random_corpora(
    prep: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _old_pipeline()
    bad: list[str] = []
    compared = clusters = 0
    for seed in range(60):
        rows = _random_corpus(seed)
        monkeypatch.delenv(pipeline.PREP_BIN_ENV)
        want = _outcome(rows)
        monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(prep))
        with old.native_minhash(rows, config=CONFIG):
            old_got = _outcome(rows)
        batch = random.Random(seed).choice([1, 2, 3, 7, pipeline.NATIVE_MINHASH_BATCH_SETS])
        try:
            with pipeline.native_minhash(rows, config=CONFIG, batch_sets=batch):
                new_got = _outcome(rows)
        except BaseException as exc:  # noqa: BLE001
            new_got = ("block raised", type(exc).__name__, str(exc)[:120])
        if not (want == old_got == new_got):
            bad.append(f"seed {seed} batch {batch}: ref {want[:1]} old {old_got[:1]} new {new_got}")
        compared += 1
        clusters += bool(want[0] == "ok" and want[1][1])
    assert not bad, bad
    assert clusters >= 5, f"only {clusters} of {compared} corpora held a duplicate cluster"


@pytest.mark.parametrize("n_extra", [0, 1, 2, 3])
def test_identical_sets_across_every_batch_boundary(
    prep: Path, monkeypatch: pytest.MonkeyPatch, n_extra: int,
) -> None:
    """Sets that repeat at positions straddling each batch edge (batch_sets 1, 2, 3)."""
    train, val = _repos("train")[0], _repos("val")[0]
    base = [" ".join(f"s{j}u{i}" for i in range(12)) for j in range(6)]
    texts = []
    for j in range(6):
        texts.append(base[j])
        if j % 2:
            texts.append(base[(j + n_extra) % 6].replace(" ", "  "))
    rows = [_row(f"b{i:02d}", repo=(train, val)[i % 2], text=t) for i, t in enumerate(texts)]
    monkeypatch.delenv(pipeline.PREP_BIN_ENV)
    want = _decisions(rows)
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(prep))
    for batch in (1, 2, 3, 4, 6, 7):
        with pipeline.native_minhash(rows, config=CONFIG, batch_sets=batch):
            assert _decisions(rows) == want, batch


@pytest.mark.parametrize("bad", [None, "5", 5.0, np.int64(5), [5], 1_000_001, 0, -3, False])
def test_batch_sets_validation_matrix(prep: Path, bad: Any) -> None:
    with (
        pytest.raises(SystemExit, match="batch_sets"),
        pipeline.native_minhash(_fixture(), config=CONFIG, batch_sets=bad),
    ):
        raise AssertionError("opened")
    assert dedupe_module.MinHasher is MinHasher and split_module.MinHasher is MinHasher


def test_batch_sets_at_both_legal_edges_opens(prep: Path) -> None:
    for ok in (1, pipeline.MAX_NATIVE_MINHASH_BATCH_SETS):
        with pipeline.native_minhash(_fixture(), config=CONFIG, batch_sets=ok):
            _decisions(_fixture())


# --- collisions ------------------------------------------------------------------------------


def _two_sets_sharing_all_but_one_shingle() -> tuple[list[DataRow], bytes, bytes]:
    """Texts A and B of six tokens each: two shingles, the first shared, the second different."""
    a = "p0 p1 p2 p3 p4 qa"
    b = "p0 p1 p2 p3 p4 qb"
    sa = shingle(a, k=CONFIG.shingle_size).shingles
    sb = shingle(b, k=CONFIG.shingle_size).shingles
    assert len(sa) == len(sb) == 2 and len(sa & sb) == 1
    (xa,) = sa - sb
    (xb,) = sb - sa
    train, val = _repos("train")[0], _repos("val")[0]
    return [_row("A", repo=train, text=a), _row("B", repo=val, text=b)], xa, xb


def test_BREAK_a_collision_between_two_different_sets_is_not_silently_merged(
    prep: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two different shingles (one per set) hash alike. The docstring says a collision between
    two sets "is checked by the block's canaries"; here set B is never a new set (its packed hash
    row equals A's), so it is never a canary, and nothing refuses. Expected: a refusal, or B's
    own signature and Jaccard. Actual: B is answered with A's signature and exact_jaccard(A, B)
    is 1.0 instead of 1/3."""
    rows, xa, xb = _two_sets_sharing_all_but_one_shingle()
    real_hash = hash
    monkeypatch.setattr(
        pipeline, "hash", lambda b: real_hash(xa) if b == xb else real_hash(b), raising=False
    )
    plain_b = shingle(rows[1].dedupe_text, k=CONFIG.shingle_size).shingles
    plain_a = shingle(rows[0].dedupe_text, k=CONFIG.shingle_size).shingles
    reference = MinHasher(num_perm=CONFIG.num_perm, seed=CONFIG.seed)

    def body() -> dict[str, Any]:
        pa = dedupe_module.shingle(rows[0].dedupe_text, k=CONFIG.shingle_size).shingles
        pb = dedupe_module.shingle(rows[1].dedupe_text, k=CONFIG.shingle_size).shingles
        hasher = dedupe_module.MinHasher(num_perm=CONFIG.num_perm, seed=CONFIG.seed)
        return {"sig": tuple(hasher.signature(pb)),
                "jac": dedupe_module.exact_jaccard(pa, pb)}

    got = _in_block(rows, 1, body)
    if "refused" in got:
        return  # a loud refusal is the acceptable outcome
    assert got["sig"] == reference.signature(plain_b), "B was signed as A (silent merge)"
    assert got["jac"] == exact_jaccard(plain_a, plain_b), f"exact_jaccard {got['jac']} != 1/3"


def test_a_collision_inside_one_set_is_refused_with_a_real_two_shingle_set(
    prep: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, xa, _xb = _two_sets_sharing_all_but_one_shingle()
    monkeypatch.setattr(pipeline, "hash", lambda _b: 3, raising=False)
    with pytest.raises(SystemExit, match="same 64-bit hash"), pipeline.native_minhash(
        rows, config=CONFIG, batch_sets=1
    ):
        raise AssertionError("opened")


def test_BREAK_a_packed_set_from_an_earlier_block_is_signed_by_row_number_not_content(
    prep: Path,
) -> None:
    """``NativeMinHasher.signature`` trusts ``PackedShingles.row`` without checking it names the
    set it was handed. A packed set kept from block 1 and passed to block 2's hasher returns
    block 2's row of the same number: another document's signature, no error. The pre-change
    code looked the set up by content and refused an unknown one."""
    first = _fixture()
    kept: dict[str, Any] = {}

    def grab() -> dict[str, Any]:
        kept["stale"] = dedupe_module.shingle(
            first[4].dedupe_text, k=CONFIG.shingle_size).shingles
        return {}

    _in_block(first, 2, grab)
    other = [
        _row(f"o{i}", repo=_repos("train")[0], text=" ".join(f"z{i}q{j}" for j in range(30)))
        for i in range(10)
    ]
    reference = MinHasher(num_perm=CONFIG.num_perm, seed=CONFIG.seed)

    def sign_stale() -> dict[str, Any]:
        hasher = dedupe_module.MinHasher(num_perm=CONFIG.num_perm, seed=CONFIG.seed)
        return {"sig": tuple(hasher.signature(kept["stale"]))}

    got = _in_block(other, 2, sign_stale)
    if "refused" in got:
        return
    plain = shingle(first[4].dedupe_text, k=CONFIG.shingle_size).shingles
    assert got["sig"] == reference.signature(plain), "a stale packed set got another document's row"


# --- partial failure ---------------------------------------------------------------------------

_WRAPPER = """#!{python}
import os, struct, subprocess, sys
real, mode, at, counter = {real!r}, {mode!r}, {at}, {counter!r}
n = (int(open(counter).read()) if os.path.exists(counter) else 0) + 1
open(counter, "w").write(str(n))
argv = sys.argv[1:]
rc = subprocess.run([real, *argv]).returncode
if n == at and rc == 0:
    out = argv[argv.index("--output") + 1]
    raw = open(out, "rb").read()
    magic = {magic!r}
    hdr = len(magic) + 12
    k, cnt = struct.unpack_from("<IQ", raw, len(magic))
    if mode == "garbage":
        raw = b"junk"
    elif mode == "empty":
        raw = b""
    elif mode == "cut8":
        raw = raw[:-8]
    elif mode == "fewer":
        raw = raw[:len(magic)] + struct.pack("<IQ", k, cnt - 1) + raw[hdr:hdr + (cnt - 1) * k * 8]
    elif mode == "more":
        raw = raw[:len(magic)] + struct.pack("<IQ", k, cnt + 1) + raw[hdr:] + raw[hdr:hdr + k * 8]
    elif mode == "dup_rows":
        row0 = raw[hdr:hdr + k * 8]
        raw = raw[:hdr] + row0 * cnt
    elif mode == "swap01" and cnt >= 2:
        r = [raw[hdr + i * k * 8: hdr + (i + 1) * k * 8] for i in range(cnt)]
        r[0], r[1] = r[1], r[0]
        raw = raw[:hdr] + b"".join(r)
    elif mode == "wrong_k":
        raw = raw[:len(magic)] + struct.pack("<IQ", k - 1, cnt) + raw[hdr:]
    open(out, "wb").write(raw)
sys.exit(rc)
"""


def _wrapper(tmp_path: Path, prep: Path, mode: str, at: int) -> Path:
    path = tmp_path / f"qd-prep-{mode}-{at}"
    path.write_text(_WRAPPER.format(
        python=sys.executable, real=str(prep), mode=mode, at=at,
        counter=str(tmp_path / f"count-{mode}-{at}"), magic=pipeline._PREP_REPLY_MAGIC,
    ), encoding="utf-8")
    path.chmod(0o755)
    return path


def _wide_rows(n: int) -> list[DataRow]:
    return [
        _row(f"w{i:03d}", repo=_repos("train")[i % 2],
             text=" ".join(f"x{i}y{j}" for j in range(20)))
        for i in range(n)
    ]


@pytest.mark.parametrize("mode", ["garbage", "empty", "cut8", "fewer", "more", "wrong_k"])
@pytest.mark.parametrize("at", [1, 2, 3])
def test_a_bad_reply_in_any_batch_is_refused_and_nothing_stays_installed(
    prep: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, at: int,
) -> None:
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(_wrapper(tmp_path, prep, mode, at)))
    with pytest.raises(SystemExit), pipeline.native_minhash(
        _wide_rows(12), config=CONFIG, batch_sets=4
    ):
        raise AssertionError("opened on a bad reply")
    assert dedupe_module.MinHasher is MinHasher and split_module.MinHasher is MinHasher
    assert dedupe_module.exact_jaccard is exact_jaccard


@pytest.mark.parametrize("mode", ["dup_rows", "swap01"])
def test_BREAK_inherited_a_wrong_but_well_formed_middle_batch_is_not_caught(
    prep: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    """INHERITED, not a regression: canaries are the first and last 4 distinct sets only, as
    before. A middle batch whose reply is well formed but wrong (every row row 0's; rows 0 and 1
    swapped) is accepted and its signatures used for dedupe and split. Expected: a refusal."""
    rows = _wide_rows(40)
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(_wrapper(tmp_path, prep, mode, 3)))
    got = _in_block(rows, 8, dict)
    assert "refused" in got, f"{mode}: a wrong middle batch (sets 16..23 of 40) was accepted"


# --- memory --------------------------------------------------------------------------------------


def test_per_row_python_objects_after_the_build_stay_small(prep: Path) -> None:
    """3,000 small rows: tracemalloc bytes held after the build, per distinct set. The lane's
    own bound is for 60 large rows; this is the many-small-sets shape the v6 corpus also has."""
    import tracemalloc

    rows = [
        _row(f"m{i:05d}", repo=_repos("train")[i % 2],
             text=" ".join(f"m{i}k{j}" for j in range(12)))
        for i in range(3000)
    ]
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        with pipeline.native_minhash(rows, config=CONFIG, batch_sets=500):
            held = tracemalloc.get_traced_memory()[0] - before
            dedupe(list(rows), config=CONFIG)
    finally:
        tracemalloc.stop()
    per = held / len(rows)
    print(f"held {held / (1 << 20):.2f} MiB, {per:.0f} bytes per row")
    assert per < 2500, f"{per:.0f} bytes held per distinct set after the build"


def test_env_unset_is_a_refusal_not_a_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(pipeline.PREP_BIN_ENV, raising=False)
    with pytest.raises(SystemExit, match="unset"), pipeline.native_minhash(
        _fixture(), config=CONFIG, batch_sets=1
    ):
        raise AssertionError("opened")
    assert os.environ.get(pipeline.PREP_BIN_ENV) is None
    assert struct.calcsize("<I") == 4
