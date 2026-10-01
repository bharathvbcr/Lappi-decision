"""The FT linear control's featurisation and fit, run by ``qd-prep linfit`` (crates/qd-prep).

## Why

``tools/ft_linear_control.py`` on one eval row of the full mixture (150,225 training docs,
11,013 val, about a dozen tasks) ran 2.3 h+ at ~26 cores on the GH200 on 2026-10-01 without
finishing. Its engine was ``qd_train.baseline``: ``CharNGramHasher.transform`` hashes every
character n-gram in an interpreted byte loop, and ``LinearBaseline.fit`` runs five Adam fits
(the four-value L2 grid on a validation carve, then the refit) of up to 6,000 iterations, on a
dense BLAS operand where ``n * d * 8`` fits ``--dense-budget-gb`` and on a numpy gather/bincount
operand where it does not -- and at 65,536 columns, any task above ~49k training rows does not.

## What moved, and what did not

``qd_train.baseline`` is the reference oracle and is **not edited**: ``control_cache`` folds its
sha256 into every cache key, so one changed byte would orphan every cached control on the box.
Everything that is a *choice* stays defined there and is read from it: the featurizer and its
parameters, the classes (``sorted(set(labels))``), the validation carve (``fit``'s own
``default_rng(seed).permutation`` and ``max(1, int(n * val_frac))``), the initial weights
(``_train_once``'s ``default_rng(seed).normal(0, 0.01, (d, k))``, identical for every fit
because ``_train_once`` reseeds), the L2 grid, ``tol``, ``lr`` and ``max_iter``. The two RNG
draws are made here with numpy, exactly as the reference makes them, and handed to the binary --
reproducing PCG64 and numpy's ziggurat in Rust would buy nothing (they cost milliseconds) and
would be a second implementation of the seed's meaning.

The binary owns the arithmetic, written operation for operation as the reference runs it on its
SPARSE operand (``CSR.matmul``/``rmatmul``): see ``crates/qd-prep/src/linfit.rs``. On this Mac it
is that path bit for bit -- weights, bias, loss history, iteration count, gradient norm, grid
accuracies, logits (``python/tests/test_qd_prep_linear_parity.py``). Against the reference's
DENSE operand (a BLAS GEMM) it differs by rounding, as the reference's own two operands differ
(``test_the_fit_is_unchanged_by_densification``); the parity test measures that too.

Predictions are made here, from the binary's logits, by the reference's own expression
(``LinearBaseline._softmax`` then ``argmax``).

## What is checked on every call (no Python fallback)

Without :data:`PREP_BIN_ENV` the caller refuses before fitting anything. A binary that is not an
absolute file, cannot run, exits non-zero, runs past its timeout, or replies with anything but
exactly the shape asked for is a refusal. The reply's grid must be the request's grid, its
selected L2 must be the one the reference's rule (first strictly best validation accuracy)
picks from the reply's own accuracies, and the first and last :data:`NATIVE_LOGIT_CANARIES`
evaluation rows are re-featurised by the reference and scored with ``CSR.matmul(W) + b``; their
logits must equal the binary's bit for bit. That covers the binary's hashing and its matmul on
the run's own documents without re-fitting anything in Python.
"""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from qd_train.baseline import (  # noqa: E402
    CSR,
    BaselineFit,
    CharNGramHasher,
    LinearBaseline,
)

#: The variable naming the ``qd-prep`` binary; the same one ``real_tokenizer_pipeline``'s
#: native MinHash reads (``test_qd_prep_linear_parity`` asserts the two agree). An absolute
#: path, never searched for: the aarch64 box has no Rust toolchain.
PREP_BIN_ENV: Final[str] = "QD_PREP_BIN"
#: What the ledger row records as the engine that fitted the control.
ENGINE_NAME: Final[str] = "qd-prep linfit"
#: Seconds one ``qd-prep linfit`` call may take when the caller names no cap. Six hours is
#: three times the Python engine's 2.3 h+ on the full mixture: past it, the binary is hung.
DEFAULT_FIT_TIMEOUT_S: Final[float] = 6 * 3600.0
#: Evaluation rows re-scored by the reference from each end of every reply.
NATIVE_LOGIT_CANARIES: Final[int] = 4
#: Seconds one ``qd-prep ngrams`` call may take.
NGRAMS_TIMEOUT_S: Final[float] = 3600.0

_NGRAMS_REQUEST: Final[bytes] = b"QDPNGIN1"
_NGRAMS_REPLY: Final[bytes] = b"QDPNGOK1"
_LINFIT_REQUEST: Final[bytes] = b"QDPLFIN1"
_LINFIT_REPLY: Final[bytes] = b"QDPLFOK1"
_FEATURES_DOCS: Final[int] = 1
_FEATURES_ROWS: Final[int] = 2
_BUILD: Final[str] = (
    "build it with `cargo build --release -p qd-prep` (target/release/qd-prep) on the Mac, or "
    "cross-build it for the aarch64 box with `CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="
    "/Users/bharath/qd-campaign/sysroot-aarch64-linux-gnu/link.sh "
    "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS='-C linker-flavor=gcc' cargo build "
    "--release -p qd-prep --bin qd-prep --target aarch64-unknown-linux-gnu --target-dir "
    f"/Users/bharath/qd-campaign/target-aarch64-linux-lc`, and set {PREP_BIN_ENV} to its "
    "absolute path"
)


def prep_binary() -> Path:
    """The binary :data:`PREP_BIN_ENV` names, or a refusal. Read before any work is done."""
    named = os.environ.get(PREP_BIN_ENV, "")
    if not named:
        raise SystemExit(
            f"{PREP_BIN_ENV} is unset. The linear control is featurised and fitted by "
            f"crates/qd-prep; qd_train.baseline is its parity oracle, not a fallback. {_BUILD}"
        )
    binary = Path(named)
    if not binary.is_absolute() or not binary.is_file():
        raise SystemExit(f"{PREP_BIN_ENV}={binary} is not an absolute path to a file; {_BUILD}")
    return binary


def engine_sha256(binary: Path) -> str:
    """sha256 of the binary's bytes: which engine fitted the control, for the ledger."""
    return hashlib.sha256(binary.read_bytes()).hexdigest()


def _run(binary: Path, command: str, request: bytes, *, timeout_s: float,
         threads: int | None) -> tuple[bytes, str]:
    """``binary command --input REQ --output REPLY``; the reply's bytes and the summary line."""
    if not binary.is_absolute() or not binary.is_file():
        raise SystemExit(f"{PREP_BIN_ENV}={binary} is not an absolute path to a file; {_BUILD}")
    if not timeout_s > 0:
        raise SystemExit(f"a qd-prep timeout of {timeout_s} s would never let it run")
    with tempfile.TemporaryDirectory(prefix=f"qd-prep-{command}-") as tmp:
        req_path, reply_path = Path(tmp) / "request.bin", Path(tmp) / "reply.bin"
        req_path.write_bytes(request)
        cmd = [str(binary), command, "--input", str(req_path), "--output", str(reply_path)]
        if threads is not None:
            cmd += ["--threads", str(threads)]
        try:
            done = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise SystemExit(
                f"{binary} {command} ran past {timeout_s:.0f} s and was killed"
            ) from exc
        except OSError as exc:
            raise SystemExit(
                f"{binary} could not be run ({exc}) -- built for another platform?; {_BUILD}"
            ) from exc
        if done.returncode != 0:
            raise SystemExit(
                f"{binary} {command} exited {done.returncode}: {done.stderr.strip()[-2000:]}"
            )
        return reply_path.read_bytes(), done.stdout.strip()


def _docs_block(docs: Sequence[str]) -> list[bytes]:
    encoded = []
    for i, doc in enumerate(docs):
        if not isinstance(doc, str):
            raise TypeError(f"document {i} is {type(doc).__name__}, not str")
        # Strict: a lone surrogate has no UTF-8 form, and the reference's own encode of any
        # gram containing it raises too. Refusing here is never a silent difference.
        encoded.append(doc.encode("utf-8"))
    return [
        struct.pack("<Q", len(encoded)),
        np.asarray([len(e) for e in encoded], dtype="<u8").tobytes(),
        b"".join(encoded),
    ]


def _csr_block(x: CSR) -> list[bytes]:
    return [
        struct.pack("<Q", x.shape[0]),
        np.ascontiguousarray(x.indptr, dtype="<u8").tobytes(),
        np.ascontiguousarray(x.indices, dtype="<u4").tobytes(),
        np.ascontiguousarray(x.data, dtype="<f8").tobytes(),
    ]


class _Reader:
    """Sequential little-endian reads that refuse to run past the reply."""

    def __init__(self, buf: bytes, binary: Path) -> None:
        self.buf, self.at, self.binary = buf, 0, binary

    def take(self, n: int) -> bytes:
        if n < 0 or self.at + n > len(self.buf):
            raise SystemExit(
                f"{self.binary} wrote a {len(self.buf)}-byte reply that ends inside a field "
                f"at offset {self.at}; it is not this request's reply"
            )
        out = self.buf[self.at : self.at + n]
        self.at += n
        return out

    def unpack(self, fmt: str) -> tuple:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def array(self, dtype: str, count: int) -> np.ndarray:
        width = np.dtype(dtype).itemsize
        return np.frombuffer(self.take(width * count), dtype=dtype).copy()

    def end(self) -> None:
        if self.at != len(self.buf):
            raise SystemExit(
                f"{self.binary} wrote {len(self.buf) - self.at} byte(s) past the reply's end; "
                "it is not this request's reply"
            )


def ngrams(binary: Path, docs: Sequence[str], hasher: CharNGramHasher, *,
           threads: int | None = None) -> CSR:
    """``hasher.transform(docs)``, computed by ``qd-prep ngrams``. Bit-identical by design;
    used by the parity test and the benchmark, which compare it with the reference."""
    request = b"".join([
        _NGRAMS_REQUEST, struct.pack("<III", hasher.n_min, hasher.n_max, hasher.dim),
        *_docs_block(docs),
    ])
    reply, _ = _run(binary, "ngrams", request, timeout_s=NGRAMS_TIMEOUT_S, threads=threads)
    r = _Reader(reply, binary)
    if r.take(len(_NGRAMS_REPLY)) != _NGRAMS_REPLY:
        raise SystemExit(f"{binary} wrote a reply that is not a {_NGRAMS_REPLY!r} file")
    n_rows, n_cols, nnz = r.unpack("<QIQ")
    if (n_rows, n_cols) != (len(docs), hasher.dim):
        raise SystemExit(
            f"{binary} replied {n_rows} rows x {n_cols} columns; {len(docs)} x {hasher.dim} "
            "were asked for"
        )
    indptr = r.array("<u8", n_rows + 1).astype(np.int64)
    indices = r.array("<u4", nnz).astype(np.int64)
    data = r.array("<f8", nnz).astype(np.float64)
    r.end()
    if indptr[0] != 0 or indptr[-1] != nnz or np.any(np.diff(indptr) < 0):
        raise SystemExit(f"{binary} replied an indptr that does not run from 0 to {nnz}")
    lengths = indptr[1:] - indptr[:-1]
    return CSR(
        indptr=indptr, indices=indices, data=data,
        rows=np.repeat(np.arange(len(docs)), lengths), shape=(len(docs), hasher.dim),
    )


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One L2 value's fit on the training carve, scored on the validation carve."""

    l2: float
    val_correct: int
    val_total: int
    converged: bool
    iterations: int
    grad_norm: float

    @property
    def accuracy(self) -> float:
        """The reference's ``float((argmax(...) == y_val).mean())``."""
        return float(np.float64(self.val_correct) / np.float64(self.val_total))


@dataclass(frozen=True)
class NativeFit:
    """What ``LinearBaseline.fit`` would leave in ``fit_``, and the evaluation rows' answers."""

    fit: BaselineFit
    grid: tuple[GridPoint, ...]
    selected: int
    eval_logits: np.ndarray
    predictions: list[str]
    summary: str


def _reference_selection(grid: Sequence[GridPoint]) -> int:
    """``LinearBaseline.fit``'s rule: the first L2 whose accuracy beats every earlier one."""
    best: tuple[float, int] | None = None
    for i, g in enumerate(grid):
        acc = g.accuracy
        if best is None or acc > best[0]:
            best = (acc, i)
    if best is None:
        raise SystemExit("the reply holds no grid value to select")
    return best[1]


def fit(
    binary: Path,
    model: LinearBaseline,
    docs: Sequence[str],
    labels: Sequence[str],
    eval_docs: Sequence[str],
    *,
    val_frac: float = 0.2,
    timeout_s: float = DEFAULT_FIT_TIMEOUT_S,
    threads: int | None = None,
) -> NativeFit:
    """``model.fit(docs, labels)`` and the evaluation documents' logits, by ``qd-prep linfit``.

    ``model`` supplies every choice -- featurizer, seed, grid, tol, lr, max_iter -- and is not
    fitted itself; the caller may install ``NativeFit.fit`` as its ``fit_`` so that
    ``model.convergence()`` speaks for the result exactly as it would for a Python fit.
    """
    # LinearBaseline.fit's refusals, verbatim, in its order.
    if len(docs) != len(labels):
        raise ValueError(f"docs and labels differ in length: {len(docs)} vs {len(labels)}")
    if len(docs) < 4:
        raise ValueError(f"need at least 4 examples to fit and validate, got {len(docs)}")
    classes = tuple(sorted(set(labels)))
    if len(classes) < 2:
        raise ValueError(f"need at least 2 classes, got {classes}")
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[label] for label in labels], dtype=np.int64)
    hasher = model.hasher
    d, k = hasher.dim, len(classes)

    # The reference's two draws, made exactly as it makes them.
    order = np.random.default_rng(model.seed).permutation(len(docs))
    n_val = max(1, int(len(docs) * val_frac))
    if len(docs) - n_val <= 0:
        raise ValueError("validation split consumed the whole training set")
    w0 = np.random.default_rng(model.seed).normal(0.0, 0.01, size=(d, k)).astype(np.float64)

    # The n-gram hasher is evaluated in the binary; any other featurizer (the length control's
    # five dense features) is evaluated here by the reference and its rows are sent.
    if type(hasher) is CharNGramHasher:
        header = struct.pack("<BIII", _FEATURES_DOCS, hasher.n_min, hasher.n_max, hasher.dim)
        train_block, eval_block = _docs_block(docs), _docs_block(eval_docs)
        eval_rows: CSR | None = None
    else:
        header = struct.pack("<BI", _FEATURES_ROWS, d)
        eval_rows = hasher.transform(list(eval_docs))
        train_block, eval_block = _csr_block(hasher.transform(list(docs))), _csr_block(eval_rows)
    grid_in = tuple(float(v) for v in model.l2_grid)
    request = b"".join([
        _LINFIT_REQUEST, header,
        struct.pack("<IIdd", k, model.max_iter, model.tol, model.lr),
        struct.pack(f"<I{len(grid_in)}d", len(grid_in), *grid_in),
        *train_block,
        np.ascontiguousarray(y, dtype="<u4").tobytes(),
        struct.pack("<Q", n_val),
        np.ascontiguousarray(order, dtype="<u8").tobytes(),
        np.ascontiguousarray(w0, dtype="<f8").tobytes(),
        *eval_block,
    ])
    reply, summary = _run(binary, "linfit", request, timeout_s=timeout_s, threads=threads)

    r = _Reader(reply, binary)
    if r.take(len(_LINFIT_REPLY)) != _LINFIT_REPLY:
        raise SystemExit(f"{binary} wrote a reply that is not a {_LINFIT_REPLY!r} file")
    got_k, got_d, n_grid = r.unpack("<III")
    if (got_k, got_d, n_grid) != (k, d, len(grid_in)):
        raise SystemExit(
            f"{binary} replied {got_k} classes x {got_d} columns over {n_grid} L2 values; "
            f"{k} x {d} over {len(grid_in)} were asked for"
        )
    grid = tuple(
        GridPoint(l2, correct, total, bool(conv), iters, gn)
        for l2, correct, total, conv, iters, gn in (r.unpack("<dQQBId") for _ in range(n_grid))
    )
    (selected,) = r.unpack("<I")
    conv_b, iterations, grad_norm, n_hist = r.unpack("<BIdI")
    history = r.array("<f8", n_hist)
    weights = r.array("<f8", d * k).reshape(d, k)
    bias = r.array("<f8", k)
    (n_eval,) = r.unpack("<Q")
    logits = r.array("<f8", n_eval * k).reshape(n_eval, k)
    r.end()

    if tuple(g.l2 for g in grid) != grid_in or any(g.val_total != n_val for g in grid):
        raise SystemExit(f"{binary} replied a grid that is not the request's: {grid}")
    if selected != _reference_selection(grid):
        raise SystemExit(
            f"{binary} selected L2 {grid[selected].l2 if selected < len(grid) else selected}, "
            f"but the reference's rule over its own accuracies selects "
            f"{grid[_reference_selection(grid)].l2}"
        )
    if n_eval != len(eval_docs):
        raise SystemExit(f"{binary} scored {n_eval} evaluation rows of {len(eval_docs)}")
    if n_hist != iterations or iterations > model.max_iter:
        raise SystemExit(
            f"{binary} reports {iterations} iterations with {n_hist} losses under a budget of "
            f"{model.max_iter}"
        )
    _check_canaries(binary, hasher, eval_docs, eval_rows, weights, bias, logits)

    probabilities = LinearBaseline._softmax(logits)
    predictions = [classes[i] for i in np.argmax(probabilities, axis=1)] if n_eval else []
    baseline_fit = BaselineFit(
        weights=weights, bias=bias, classes=classes, l2=grid[selected].l2,
        converged=bool(conv_b), iterations=int(iterations), final_grad_norm=float(grad_norm),
        loss_history=tuple(float(v) for v in history),
    )
    return NativeFit(baseline_fit, grid, selected, logits, predictions, summary)


def _check_canaries(
    binary: Path, hasher: object, eval_docs: Sequence[str], eval_rows: CSR | None,
    weights: np.ndarray, bias: np.ndarray, logits: np.ndarray,
) -> None:
    """The first and last evaluation rows, featurised and scored by the reference."""
    n = len(eval_docs)
    picks = sorted(set(range(min(NATIVE_LOGIT_CANARIES, n))) |
                   set(range(max(0, n - NATIVE_LOGIT_CANARIES), n)))
    if not picks:
        return
    idx = np.asarray(picks, dtype=np.int64)
    if eval_rows is None:
        if not isinstance(hasher, CharNGramHasher):  # pragma: no cover - the caller's branch
            raise SystemExit("evaluation rows were neither hashed here nor sent as rows")
        rows = hasher.transform([eval_docs[i] for i in picks])
    else:
        rows = eval_rows.select(idx)
    want = rows.matmul(weights) + bias
    if want.tobytes() != np.ascontiguousarray(logits[idx]).tobytes():
        raise SystemExit(
            f"{binary}'s logits for evaluation rows {picks} differ from the reference's "
            "featurisation scored with the binary's own weights; its features or its matmul "
            "are not the reference's"
        )
