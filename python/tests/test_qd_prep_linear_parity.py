"""``crates/qd-prep`` featurises and fits the FT linear control as ``qd_train.baseline`` does.

``tools/ft_linear_control.py`` on one full-mixture eval row ran 2.3 h+ on the GH200
(2026-10-01) without finishing; its engine -- ``CharNGramHasher.transform`` and
``LinearBaseline.fit`` -- now runs in ``qd-prep ngrams`` / ``qd-prep linfit`` through
``tools/linear_control_native.py``. ``qd_train.baseline`` is unchanged and is the oracle.

**Parity, as defined and measured here.**

1. Hashing: ``indptr``, ``indices`` and ``data`` byte for byte, on adversarial documents, this
   repository's sources, FT request prompts and (when on disk) the real defect corpus's.
2. The fit, against the oracle's SPARSE operand (``dense_budget_bytes=0``): bit for bit --
   weights, bias, loss history, iteration count, convergence, final gradient norm, every grid
   point's iterations/gradient norm/validation accuracy, the selected L2, evaluation logits and
   predictions. This is the arithmetic the binary is written to reproduce.
3. The fit, against the oracle's DEFAULT operand (dense BLAS where it fits, which the box used
   below 24 GB): (a) the same convergence flag, iteration counts within 1% (at least 1), (b) the
   same selected L2, (c) the same predictions except rows whose top-2 logit gap is below 1e-9
   relative (counted, printed), (d) ``||W_native - W_dense|| / ||W_dense||`` (Frobenius) at
   most ``DENSE_WEIGHT_RTOL``, with the largest elementwise difference printed beside it.
   Because 2. holds, every number in 3. is the oracle's OWN dense-vs-sparse spread -- the
   operand ``--dense-budget-gb`` already picks per task. Its two operands differ by rounding
   (``test_the_fit_is_unchanged_by_densification``), and over hundreds of Adam steps that
   rounding grows: Adam's normalised step moves a rarely-active feature's weight by about
   ``lr`` whatever the size of its gradient, so the largest ELEMENTWISE difference is not a
   tolerance worth stating, while the weight vector as a whole and the logits stay close.

The bit-for-bit claims rest on numpy's float64 ``exp``/``log``/``**`` being the platform libm's
and on numpy's pairwise summation order, both measured on this Mac (numpy 2.5.0); on another
host they are re-measured by running this file there, never assumed.

The binary is built with ``cargo build --release -p qd-prep``; without cargo every test that
needs it is SKIPPED with that reason, never passed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from data_fixtures import small_corpus

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import linear_control_native as native  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_train.baseline import (  # noqa: E402
    CharNGramHasher,
    ContextLengthFeatures,
    LinearBaseline,
    request_texts,
)

CONFIG = DataConfig()
DEFECT_DIR = REPO / "data" / "pool" / "commitpackft-corpus-v2"
DEFECT_DOWNLOAD = REPO / "data" / "pool" / "commitpackft"
#: (d) above, on the Frobenius norm. Measured on this Mac (the reference's dense-vs-sparse
#: spread, since the port is the sparse operand bit for bit): ~1e-16 on the synthetic fixtures;
#: on real defect-corpus prompts the largest elementwise difference was 6.3e-7 at 400 docs (777
#: iterations) and 1.08e-3 at 10,800 docs (850 iterations), where the Frobenius difference was
#: 1.54e-4, the held-out logits differed by at most 5.9e-6 relative and no prediction moved
#: (smallest top-2 gap 2.3e-3). The bound sits above the worst measured, not above a guess.
DENSE_WEIGHT_RTOL = 1e-3
#: (c) above: a row whose top-2 logits are closer than this (relative) may flip on rounding.
NEAR_TIE_REL = 1e-9


def test_the_adapter_and_the_fixture_name_one_variable() -> None:
    import real_tokenizer_pipeline as pipeline

    assert native.PREP_BIN_ENV == pipeline.PREP_BIN_ENV == "QD_PREP_BIN"


# --- hashing ---------------------------------------------------------------------------

ADVERSARIAL_DOCS = [
    "",
    "a",
    "ab",  # shorter than n_min
    "abc",  # exactly n_min
    "abcde",  # exactly n_max
    "a" * 5000,  # one gram per order, counted thousands of times
    "crlf one\r\ncrlf two\r\n\r\n",
    "nul\x00inside\x00and\x7fdelete",
    chr(0xFEFF) + "bom and " + chr(0xFFFD) + " replacement",
    "combining e" + chr(0x301) * 3 + " and a" + chr(0x308),
    "astral \U0001f9ea\U0001f9ec\U0010ffff plane \U00020000",
    "rtl " + "".join(map(chr, (0x5E9, 0x5DC, 0x5D5, 0x5DD))) + " mixed "
    + "".join(map(chr, (0x627, 0x644, 0x639, 0x631, 0x628, 0x64A, 0x629))),
    "non" + chr(0xA0) + "breaking" + chr(0x2028) + "line" + chr(0x3000) + "ideographic",
    "\t\t   \n\n  ",
    "\U0001f9ea",  # one astral code point: four bytes, still shorter than n_min
    "\U0001f9ea\U0001f9ea\U0001f9ea",  # three astral code points: one 12-byte gram
    "".join(chr(c) for c in range(0x20, 0x2000) if not 0xD800 <= c < 0xE000),
    "x" * 200_001 + chr(0xE9) + "y" * 50_000,  # a quarter-megabyte document
]

HASHERS = [
    CharNGramHasher(),
    CharNGramHasher(n_min=1, n_max=1, dim=16),
    CharNGramHasher(n_min=2, n_max=7, dim=2**10),
    CharNGramHasher(n_min=5, n_max=5, dim=2**20),
]


def _assert_hash_parity(binary: Path, docs: list[str], hasher: CharNGramHasher) -> int:
    want = hasher.transform(docs)
    got = native.ngrams(binary, docs, hasher)
    assert got.shape == want.shape
    for field in ("indptr", "indices", "data", "rows"):
        a, b = getattr(want, field), getattr(got, field)
        assert a.dtype == b.dtype, field
        assert a.tobytes() == b.tobytes(), (
            f"{field} differs ({hasher.n_min}..{hasher.n_max}, {hasher.dim})"
        )
    return int(want.indices.size)


@pytest.mark.parametrize("hasher", HASHERS, ids=lambda h: f"{h.n_min}-{h.n_max}-{h.dim}")
def test_adversarial_documents_hash_byte_for_byte(qd_prep_bin: Path, hasher) -> None:
    assert _assert_hash_parity(qd_prep_bin, ADVERSARIAL_DOCS, hasher) > 0


def test_this_repositorys_sources_hash_byte_for_byte(qd_prep_bin: Path) -> None:
    """Real code that is always present. Every eighth file in sorted order: the whole tree is
    ~6 MB, which the Python reference takes most of a minute to hash."""
    paths = sorted(
        p for p in [*REPO.glob("python/**/*.py"), *REPO.glob("tools/*.py"),
                    *REPO.glob("crates/*/src/**/*.rs")]
        if "target" not in p.parts
    )[::8]
    docs = [p.read_text(encoding="utf-8", errors="strict") for p in paths]
    assert len(docs) > 30
    assert _assert_hash_parity(qd_prep_bin, docs, CharNGramHasher()) > 100_000


def _fixture_request_docs():
    rows = build_mixture(small_corpus(30), config=CONFIG).rows
    docs, excluded = request_texts(rows, seed=CONFIG.seed)
    assert not excluded
    return docs


def test_ft_request_prompts_hash_byte_for_byte(qd_prep_bin: Path) -> None:
    docs = _fixture_request_docs()
    assert len({d.task for d in docs}) >= 3
    assert _assert_hash_parity(qd_prep_bin, [d.text for d in docs], CharNGramHasher()) > 0
    # The raw request contexts too (bytes on these rows), as the model's byte window decodes them.
    contexts = [
        c.decode("utf-8", errors="replace") if isinstance(c, bytes) else c
        for c in (d.context for d in docs)
    ]
    assert _assert_hash_parity(qd_prep_bin, contexts, CharNGramHasher()) > 0


def _defect_request_docs(max_rows: int):
    from qd_data.defect_class import DEFECT_SOURCE_ID, load_defect_rows

    load = load_defect_rows(
        DEFECT_DIR, download_root=DEFECT_DOWNLOAD, config=CONFIG, repo_root=REPO,
        max_rows=max_rows,
    )
    rows = build_mixture({DEFECT_SOURCE_ID: list(load.rows)}, config=CONFIG).rows
    docs, _ = request_texts(rows, seed=CONFIG.seed)
    return docs


@pytest.mark.skipif(
    not (DEFECT_DIR / "examples.jsonl").is_file(), reason=f"{DEFECT_DIR} is not on disk"
)
def test_real_defect_corpus_prompts_hash_byte_for_byte(qd_prep_bin: Path) -> None:
    docs = _defect_request_docs(300)
    assert len(docs) > 200
    assert _assert_hash_parity(qd_prep_bin, [d.text for d in docs], CharNGramHasher()) > 0


def test_a_lone_surrogate_is_refused_as_the_reference_refuses_it(qd_prep_bin: Path) -> None:
    doc = "abc\ud800def"
    with pytest.raises(UnicodeEncodeError):
        CharNGramHasher().transform([doc])
    with pytest.raises(UnicodeEncodeError):
        native.ngrams(qd_prep_bin, [doc], CharNGramHasher())


# --- the fit: bit for bit against the sparse operand -------------------------------------


def _keyword_docs(n: int, n_classes: int, seed: int) -> tuple[list[str], list[str]]:
    """Short code-ish documents whose label a few keywords mostly decide."""
    rng = np.random.default_rng(seed)
    words = ["def", "return", "self", "fix", "bug", "add", "test", "docs", chr(0xFC) + "ml",
             "\U0001f9ea", "x = 1", "\r\n", "\t", "if", "else"]
    marks = [f"mark{c} " * (1 + c % 3) for c in range(n_classes)]
    docs, labels = [], []
    for i in range(n):
        c = i % n_classes
        body = " ".join(rng.choice(words, int(rng.integers(2, 30))))
        if rng.random() < 0.75:
            body += " " + marks[c]
        docs.append(body)
        labels.append(f"class-{c:02d}")
    return docs, labels


def _largest_fittable_task(docs) -> str:
    """The task with the most docs among those with two classes or more; ties go to the name
    sorted first, because ``max`` keeps the first of equal counts and set order varies per
    process."""
    tasks = sorted(t for t in {d.task for d in docs}
                   if len({d.value for d in docs if d.task == t}) >= 2)
    return max(tasks, key=lambda t: sum(d.task == t for d in docs))


def _fixtures() -> dict[str, tuple[object, list, list, list, int, int]]:
    """name -> (featurizer, docs, labels, eval docs, seed, max_iter)."""
    out: dict[str, tuple[object, list, list, list, int, int]] = {}
    docs, labels = _keyword_docs(240, 3, seed=1)
    out["keywords-k3-converges"] = (CharNGramHasher(), docs, labels, docs[:40], 0, 2000)
    docs, labels = _keyword_docs(180, 9, seed=2)
    out["keywords-k9-pairwise-block"] = (
        CharNGramHasher(dim=2**10), docs, labels, docs[::7], 3, 400,
    )
    docs, labels = _keyword_docs(150, 12, seed=8)
    out["keywords-k12-unconverged"] = (CharNGramHasher(dim=2**12), docs, labels, docs[:9], 1, 60)
    docs, labels = _keyword_docs(40, 2, seed=3)
    out["max-iter-zero"] = (CharNGramHasher(), docs, labels, docs[:5], 0, 0)
    # Non-separable: the same texts under both labels, empty and short documents among them.
    noisy = ["", "ab", "same text", "same text", "other", "other", "", "x y z"] * 6
    noisy_labels = ["p" if i % 2 else "q" for i in range(len(noisy))]
    out["conflicting-empty-short"] = (CharNGramHasher(), noisy, noisy_labels, noisy[:6], 4, 500)
    out["four-rows-two-classes"] = (
        CharNGramHasher(), ["fix the bug", "add a test", "fix it", "add it"],
        ["fix", "add", "fix", "add"], ["fix something", "add something"], 0, 6000,
    )
    rng = np.random.default_rng(9)
    ctx = ["x" * int(rng.integers(1, 4000)) + "\n" * int(rng.integers(0, 80)) for _ in range(200)]
    out["length-features"] = (
        ContextLengthFeatures(), ctx, ["long" if len(c) > 2000 else "short" for c in ctx],
        ctx[:25], 2, 1500,
    )
    request = _fixture_request_docs()
    in_task = [d for d in request if d.task == _largest_fittable_task(request)]
    out["ft-request-prompts"] = (
        CharNGramHasher(), [d.text for d in in_task], [d.value for d in in_task],
        [d.text for d in in_task[:10]], 0, 6000,
    )
    return out


FIXTURES = _fixtures()


def _model(featurizer, seed: int, max_iter: int, *, dense_budget_bytes: int | None):
    return LinearBaseline(
        hasher=featurizer, seed=seed, max_iter=max_iter, dense_budget_bytes=dense_budget_bytes
    )


def _reference_grid(model: LinearBaseline, docs, labels):
    """The oracle's grid, fit by fit: ``_train_once`` on its own carve, scored as ``fit``
    scores it. ``fit`` keeps only the winner; this keeps all four for comparison."""
    classes = tuple(sorted(set(labels)))
    y = np.array([classes.index(label) for label in labels], dtype=np.int64)
    x = model.hasher.transform(list(docs))
    order = np.random.default_rng(model.seed).permutation(len(docs))
    n_val = max(1, int(len(docs) * 0.2))
    val_idx, tr_idx = order[:n_val], order[n_val:]
    x_tr, x_val = x.select(tr_idx), x.select(val_idx)
    out = []
    for l2 in model.l2_grid:
        w, b, conv, it, gn, _ = model._train_once(x_tr, y[tr_idx], len(classes), l2)
        acc = float((np.argmax(x_val.matmul(w) + b, axis=1) == y[val_idx]).mean())
        out.append((l2, acc, conv, it, gn))
    return out


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_the_fit_is_the_sparse_reference_bit_for_bit(qd_prep_bin: Path, name: str) -> None:
    featurizer, docs, labels, eval_docs, seed, max_iter = FIXTURES[name]
    reference = _model(featurizer, seed, max_iter, dense_budget_bytes=0)
    want = reference.fit(list(docs), list(labels))
    got = native.fit(
        qd_prep_bin, _model(featurizer, seed, max_iter, dense_budget_bytes=0),
        docs, labels, eval_docs,
    )
    assert (got.fit.l2, got.fit.converged, got.fit.iterations) == (
        want.l2, want.converged, want.iterations
    )
    assert got.fit.classes == want.classes
    assert (np.float64(got.fit.final_grad_norm).tobytes()
            == np.float64(want.final_grad_norm).tobytes())
    assert got.fit.weights.tobytes() == want.weights.tobytes()
    assert got.fit.bias.tobytes() == want.bias.tobytes()
    assert np.asarray(got.fit.loss_history).tobytes() == np.asarray(want.loss_history).tobytes()
    for g, (l2, acc, conv, it, gn) in zip(got.grid, _reference_grid(reference, docs, labels),
                                         strict=True):
        assert (g.l2, g.accuracy, g.converged, g.iterations) == (l2, acc, conv, it)
        assert np.float64(g.grad_norm).tobytes() == np.float64(gn).tobytes()
    x_eval = featurizer.transform(list(eval_docs))
    assert got.eval_logits.tobytes() == (x_eval.matmul(want.weights) + want.bias).tobytes()
    assert got.predictions == reference.predict(list(eval_docs))


def test_the_fixtures_cover_what_the_parity_claim_needs() -> None:
    """Converged and unconverged fits, k >= 8 (numpy's eight-accumulator row sum), no
    iterations at all, and both featurizers -- so the claim above is not a claim about one
    easy case."""
    ks = {name: len(set(f[2])) for name, f in FIXTURES.items()}
    assert max(ks.values()) >= 8
    assert any(f[5] == 0 for f in FIXTURES.values())
    assert any(isinstance(f[0], ContextLengthFeatures) for f in FIXTURES.values())


def test_the_fit_does_not_depend_on_the_thread_count_or_the_run(qd_prep_bin: Path) -> None:
    featurizer, docs, labels, eval_docs, seed, max_iter = FIXTURES["keywords-k9-pairwise-block"]

    def once(threads: int | None):
        r = native.fit(qd_prep_bin, _model(featurizer, seed, max_iter, dense_budget_bytes=0),
                       docs, labels, eval_docs, threads=threads)
        return (r.fit.weights.tobytes(), r.fit.bias.tobytes(), r.eval_logits.tobytes(),
                r.grid, r.fit.loss_history, r.fit.iterations)

    first = once(1)
    for threads in (2, 3, 5, 18, None, None):
        assert once(threads) == first, f"threads={threads}"


# --- the fit: within tolerance of the dense operand -------------------------------------


def _top2_rel_gap(logits: np.ndarray) -> np.ndarray:
    top = np.sort(logits, axis=1)[:, -2:]
    scale = np.maximum(np.abs(top).max(axis=1), 1e-300)
    return (top[:, 1] - top[:, 0]) / scale


def _dense_comparison(binary: Path, featurizer, docs, labels, eval_docs, seed, max_iter) -> dict:
    reference = _model(featurizer, seed, max_iter, dense_budget_bytes=None)
    budget = 64 * 1024**3  # dense whenever it fits in RAM, as --dense-budget-gb 24 is on the box
    reference.dense_budget_bytes = budget
    want = reference.fit(list(docs), list(labels))
    got = native.fit(binary, _model(featurizer, seed, max_iter, dense_budget_bytes=0),
                     docs, labels, eval_docs)
    want_logits = featurizer.transform(list(eval_docs)).matmul(want.weights) + want.bias
    disagree = np.nonzero(np.argmax(want_logits, axis=1) != np.argmax(got.eval_logits, axis=1))[0]
    near_tie = _top2_rel_gap(want_logits) < NEAR_TIE_REL
    return {
        "converged": (want.converged, got.fit.converged),
        "iterations": (want.iterations, got.fit.iterations),
        "l2": (want.l2, got.fit.l2),
        "prediction_disagreements": int(disagree.size),
        "disagreements_not_near_tie": int(np.sum(~near_tie[disagree])),
        "near_ties": int(near_tie.sum()),
        "eval_rows": len(eval_docs),
        **_weight_spread(want.weights, got.fit.weights),
    }


def _weight_spread(want: np.ndarray, got: np.ndarray) -> dict[str, float]:
    """(d): the Frobenius difference the bound is on, and the elementwise one beside it."""
    diff = want - got
    return {
        "frobenius_weight_rel_diff": float(np.linalg.norm(diff))
        / max(float(np.linalg.norm(want)), 1e-300),
        "max_weight_rel_diff": float(np.abs(diff).max())
        / max(float(np.abs(want).max()), 1e-300),
    }


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_the_fit_is_within_tolerance_of_the_dense_reference(qd_prep_bin: Path, name: str) -> None:
    measured = _dense_comparison(qd_prep_bin, *FIXTURES[name])
    print(json.dumps({"fixture": name, **measured}))
    (dense_conv, conv), (dense_it, it) = measured["converged"], measured["iterations"]
    assert dense_conv == conv
    assert abs(dense_it - it) <= max(1, int(0.01 * dense_it))
    assert measured["l2"][0] == measured["l2"][1]
    assert measured["disagreements_not_near_tie"] == 0
    assert measured["frobenius_weight_rel_diff"] <= DENSE_WEIGHT_RTOL


@pytest.mark.skipif(
    not (DEFECT_DIR / "examples.jsonl").is_file(), reason=f"{DEFECT_DIR} is not on disk"
)
def test_real_defect_prompts_fit_bit_for_bit_and_within_tolerance_of_dense(
    qd_prep_bin: Path,
) -> None:
    """The phase-3 task itself, at a size the sparse oracle can fit in a test."""
    docs = _defect_request_docs(400)
    texts, labels = [d.text for d in docs], [d.value for d in docs]
    featurizer = CharNGramHasher()
    reference = _model(featurizer, 0, 6000, dense_budget_bytes=0)
    want = reference.fit(texts, labels)
    got = native.fit(qd_prep_bin, _model(featurizer, 0, 6000, dense_budget_bytes=0),
                     texts, labels, texts[:50])
    assert (got.fit.l2, got.fit.converged, got.fit.iterations) == (
        want.l2, want.converged, want.iterations
    )
    assert got.fit.weights.tobytes() == want.weights.tobytes()
    assert np.asarray(got.fit.loss_history).tobytes() == np.asarray(want.loss_history).tobytes()
    assert got.predictions == reference.predict(texts[:50])
    measured = _dense_comparison(qd_prep_bin, featurizer, texts, labels, texts[:200], 0, 6000)
    print(json.dumps({"fixture": "defect-corpus-400-rows", "docs": len(texts), **measured}))
    assert measured["converged"][0] == measured["converged"][1]
    assert measured["l2"][0] == measured["l2"][1]
    assert measured["disagreements_not_near_tie"] == 0
    assert measured["frobenius_weight_rel_diff"] <= DENSE_WEIGHT_RTOL


# --- the adapter's refusals -------------------------------------------------------------


def _script(tmp_path: Path, name: str, body: str) -> Path:
    """A stand-in binary; argv is ``<command> --input REQ --output REPLY``."""
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_without_the_variable_nothing_is_fitted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(native.PREP_BIN_ENV, raising=False)
    with pytest.raises(SystemExit, match=r"QD_PREP_BIN is unset.*not a fallback"):
        native.prep_binary()


@pytest.mark.parametrize("named", ["relative/qd-prep", "/no/such/qd-prep"])
def test_a_binary_that_is_not_an_absolute_file_is_refused(
    named: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(native.PREP_BIN_ENV, named)
    with pytest.raises(SystemExit, match="cargo build --release -p qd-prep"):
        native.prep_binary()


def test_an_unrunnable_file_a_failure_a_hang_and_a_wrong_reply_are_refused(
    tmp_path: Path,
) -> None:
    featurizer, docs, labels, eval_docs, seed, max_iter = FIXTURES["four-rows-two-classes"]
    not_a_program = tmp_path / "not-a-program"
    not_a_program.write_bytes(b"\x7fELF\x02\x01\x01garbage")
    not_a_program.chmod(0o755)
    cases = [
        (not_a_program, "could not be run", native.DEFAULT_FIT_TIMEOUT_S),
        (_script(tmp_path, "fails", "echo boom >&2; exit 3"), "exited 3: boom",
         native.DEFAULT_FIT_TIMEOUT_S),
        (_script(tmp_path, "hangs", "sleep 30"), "ran past", 0.5),
        (_script(tmp_path, "garbage", 'printf garbage-garbage > "$5"'), "is not a", 60.0),
        (_script(tmp_path, "short", 'printf QDPLFOK1 > "$5"'), "ends inside a field", 60.0),
    ]
    for binary, match, timeout_s in cases:
        with pytest.raises(SystemExit, match=match):
            native.fit(binary, _model(featurizer, seed, max_iter, dense_budget_bytes=0),
                       docs, labels, eval_docs, timeout_s=timeout_s)


def _tampered(monkeypatch: pytest.MonkeyPatch, edit) -> None:
    """Run the real binary, then hand the adapter an edited reply."""
    real = native._run

    def run(binary, command, request, *, timeout_s, threads):
        reply, summary = real(binary, command, request, timeout_s=timeout_s, threads=threads)
        return edit(bytearray(reply)), summary

    monkeypatch.setattr(native, "_run", run)


def test_a_reply_with_the_wrong_numbers_is_caught_by_the_canaries(
    qd_prep_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    featurizer, docs, labels, eval_docs, seed, max_iter = FIXTURES["keywords-k3-converges"]

    def flip_last_logit(reply: bytearray) -> bytes:
        value = np.frombuffer(bytes(reply[-8:]), dtype="<f8")[0]
        reply[-8:] = np.float64(np.nextafter(value, np.inf)).tobytes()
        return bytes(reply)

    _tampered(monkeypatch, flip_last_logit)
    with pytest.raises(SystemExit, match="logits for evaluation rows"):
        native.fit(qd_prep_bin, _model(featurizer, seed, max_iter, dense_budget_bytes=0),
                   docs, labels, eval_docs)


def test_a_reply_whose_selection_is_not_the_references_rule_is_refused(
    qd_prep_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    featurizer, docs, labels, eval_docs, seed, max_iter = FIXTURES["keywords-k3-converges"]
    grid_end = 8 + 12 + 4 * 37  # magic, three u32s, four grid records

    def select_another(reply: bytearray) -> bytes:
        (selected,) = np.frombuffer(bytes(reply[grid_end:grid_end + 4]), dtype="<u4")
        reply[grid_end:grid_end + 4] = np.uint32((int(selected) + 1) % 4).tobytes()
        return bytes(reply)

    _tampered(monkeypatch, select_another)
    with pytest.raises(SystemExit, match="the reference's rule"):
        native.fit(qd_prep_bin, _model(featurizer, seed, max_iter, dense_budget_bytes=0),
                   docs, labels, eval_docs)


def test_the_references_refusals_are_the_adapters(qd_prep_bin: Path) -> None:
    model = _model(CharNGramHasher(), 0, 10, dense_budget_bytes=0)
    for docs, labels, match in [
        (["a", "b", "c", "d"], ["x", "y", "x"], "differ in length"),
        (["a", "b", "c"], ["x", "y", "x"], "at least 4 examples"),
        (["a", "b", "c", "d"], ["x"] * 4, "at least 2 classes"),
    ]:
        with pytest.raises(ValueError, match=match):
            LinearBaseline(hasher=CharNGramHasher(), max_iter=10).fit(docs, labels)
        with pytest.raises(ValueError, match=match):
            native.fit(qd_prep_bin, model, docs, labels, [])


# --- benchmark --------------------------------------------------------------------------

#: Rounds of each A/B; each round runs the reference, then the binary.
BENCH_ROUNDS = 3


def _min_of_n(label: str, reference, binary, rounds: int, check) -> dict:
    """``rounds`` alternating runs, reference first; ``check(want, got)`` after each pair,
    outside the timers, returns what it measured about the pair."""
    a: list[float] = []
    b: list[float] = []
    checked: list[object] = []
    for _ in range(rounds):
        started = time.perf_counter()
        want = reference()
        a.append(time.perf_counter() - started)
        started = time.perf_counter()
        got = binary()
        b.append(time.perf_counter() - started)
        checked.append(check(want, got))
    return {
        "stage": label, "rounds": rounds,
        "reference_s": [round(x, 3) for x in a], "qd_prep_s": [round(x, 3) for x in b],
        "reference_min_s": round(min(a), 3), "qd_prep_min_s": round(min(b), 3),
        "speedup_min_over_min": round(min(a) / min(b), 1), "checked": checked,
    }


def _bench(binary: Path, name: str, texts: list[str], labels: list[str], rounds: int) -> None:
    """Hashing and the fit, each A/B'd separately. The fit's reference arm runs at the box's
    ``--dense-budget-gb 24`` -- dense GEMM wherever it fits, which is what the box ran -- so its
    answer is compared by (a)-(d) of the module docstring, not bit for bit. Training is the
    first 90% of the documents; the last 10% are scored by both fits."""
    hasher = CharNGramHasher()
    cut = int(len(texts) * 0.9)
    train, train_labels, held = texts[:cut], labels[:cut], texts[cut:]

    def hash_ref():
        x = hasher.transform(texts)
        return x.indptr.tobytes(), x.indices.tobytes(), x.data.tobytes()

    def hash_native():
        x = native.ngrams(binary, texts, hasher)
        return x.indptr.tobytes(), x.indices.tobytes(), x.data.tobytes()

    def same_bytes(want, got) -> str:
        assert got == want, "hashing: the arms disagree"
        return "byte-identical"

    def fit_ref():
        f = _model(hasher, 0, 6000, dense_budget_bytes=24 * 1024**3).fit(train, train_labels)
        return f, hasher.transform(held).matmul(f.weights) + f.bias

    def fit_native():
        f = native.fit(binary, _model(hasher, 0, 6000, dense_budget_bytes=0),
                       train, train_labels, held)
        return f.fit, f.eval_logits

    def within_tolerance(want, got) -> dict:
        (wf, wl), (gf, gl) = want, got
        disagree = np.nonzero(np.argmax(wl, axis=1) != np.argmax(gl, axis=1))[0]
        near_tie = _top2_rel_gap(wl) < NEAR_TIE_REL
        measured = {
            "l2": [wf.l2, gf.l2], "converged": [wf.converged, gf.converged],
            "iterations": [wf.iterations, gf.iterations],
            "held_rows": len(held), "prediction_disagreements": int(disagree.size),
            "disagreements_not_near_tie": int(np.sum(~near_tie[disagree])),
            "min_top2_gap": float(_top2_rel_gap(wl).min()) if len(held) else None,
            "max_held_logit_rel_diff": float(np.abs(wl - gl).max())
            / max(float(np.abs(wl).max()), 1e-300) if len(held) else None,
            **_weight_spread(wf.weights, gf.weights),
        }
        assert wf.l2 == gf.l2 and wf.converged == gf.converged, measured
        assert abs(wf.iterations - gf.iterations) <= max(1, int(0.01 * wf.iterations)), measured
        assert measured["disagreements_not_near_tie"] == 0, measured
        assert measured["frobenius_weight_rel_diff"] <= DENSE_WEIGHT_RTOL, measured
        return measured

    nnz = int(hasher.transform(texts[:500]).indices.size)
    print(json.dumps({
        "benchmark": "linear_control", "fixture": name, "docs": len(texts),
        "classes": len(set(labels)), "chars_mean": round(float(np.mean([len(t) for t in texts]))),
        "nnz_per_doc_first_500": round(nnz / min(500, len(texts))), "host_cpus": os.cpu_count(),
        "hashing": _min_of_n("hashing", hash_ref, hash_native, rounds, same_bytes),
        "fit": _min_of_n("fit (grid + refit, max_iter 6000)", fit_ref, fit_native, rounds,
                         within_tolerance),
    }))


@pytest.mark.skipif(
    os.environ.get("QD_PREP_BENCH") != "1",
    reason="the reference-vs-qd-prep benchmark runs only with QD_PREP_BENCH=1",
)
def test_benchmark_small_real_shaped_fixture_interleaved_min_of_n(qd_prep_bin: Path) -> None:
    """The committed A/B behind the speedup claim, hashing and fit timed separately.

        QD_PREP_BENCH=1 pytest -s python/tests/test_qd_prep_linear_parity.py -k benchmark
    """
    docs = _fixture_request_docs()
    task = _largest_fittable_task(docs)
    in_task = [d for d in docs if d.task == task]
    _bench(qd_prep_bin, f"ft-request-prompts:{task}", [d.text for d in in_task],
           [d.value for d in in_task], BENCH_ROUNDS)


#: Rows of the real defect corpus the realistic benchmark loads (its letter docs follow).
BENCH_DEFECT_ROWS = int(os.environ.get("QD_PREP_BENCH_DEFECT_ROWS", "12000"))


@pytest.mark.skipif(
    os.environ.get("QD_PREP_BENCH") != "1",
    reason="the reference-vs-qd-prep benchmark runs only with QD_PREP_BENCH=1",
)
@pytest.mark.skipif(
    not (DEFECT_DIR / "examples.jsonl").is_file(), reason=f"{DEFECT_DIR} is not on disk"
)
def test_benchmark_real_defect_prompts_interleaved_min_of_n(qd_prep_bin: Path) -> None:
    """The phase-3 task's own prompts at a realistic size, hashing and fit separately.

        QD_PREP_BENCH=1 QD_PREP_BENCH_DEFECT_ROWS=12000 pytest -s \\
          python/tests/test_qd_prep_linear_parity.py -k real_defect_prompts_interleaved
    """
    docs = [d for d in _defect_request_docs(BENCH_DEFECT_ROWS) if d.kind == "choice"]
    _bench(qd_prep_bin, f"defect-corpus:{BENCH_DEFECT_ROWS}-rows", [d.text for d in docs],
           [d.value for d in docs], int(os.environ.get("QD_PREP_BENCH_ROUNDS", "2")))
