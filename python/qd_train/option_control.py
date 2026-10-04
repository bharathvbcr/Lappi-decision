"""The per-option linear control: one binary scorer over each row's own shown options.

## Why a second control

``qd_train.baseline``'s control is a classifier over one class set, and
:func:`~qd_train.baseline.control_label_space` labels a task whose rows offer their own options
(CommonsenseQA, MMLU, CLINC's sampled and per-domain intent sets) by the gold LETTER: the one
label every row shares. That is honest and it is the general families' control of record, but
a bag of char n-grams can tie a letter to an option's content only through letter-prefixed
n-grams, so on J1's docs it sits at chance (CSQA 222/1,197, MMLU 368/1,485). A margin against
a control that cannot lose is not a check (``GAP-GENERAL-LETTER-CONTROL-CANNOT-READ-PER-ROW-
OPTIONS``; Fable H, 2026-10-01).

This control reads the options instead. It is **separately named and report-only**: its rows
carry ``linear_option_control.*`` and ``paired_margin_vs_linear_option.*``, never the letter
control's keys, and adopting it as a gate's comparator is the human's decision under G1.

## What it is

* **Rows.** Only tasks whose rows offer their own options -- ``control_label_space`` answers
  ``"letter"`` -- are scored; the caller decides that. A row's options are ``RequestDoc.offered``
  (the rendering's ``letter_to_value``, the same source ``control_label_space`` reads), and they
  are read only if the prompt the model saw shows exactly those options: its option block must
  equal, byte for byte, the lines the renderer makes from them. The question is the prompt's
  context region (what the row asks: the CSQA/MMLU question, the CLINC utterance). Neither is
  ever taken from the gold: the gold only says which shown option is the positive.
* **Examples.** Every shown option of every row, noul included -- it is shown, and the model
  may answer it. Its text is the option *as rendered* (``escape_inline``), which is what the
  model read.
* **Features.** Two blocks of hashed char n-grams, each built exactly as
  :class:`~qd_train.baseline.CharNGramHasher` builds one (and so each L2-normalised on its
  own): the pair text ``question + "\\n" + option`` in columns ``[0, dim)``, and the option text
  alone in ``[dim, 2*dim)``.
* **Scorer.** One binary logistic regression shared by every option: the gold option is 1,
  the others 0. It is fitted by :meth:`LinearBaseline._train_once` with two classes -- the
  letter control's own Adam loop, reseeded per fit -- so there is one optimizer, not two. An
  option's score is its log-odds ``z[1] - z[0]``; a row's answer is the first option of
  greatest score (numpy's ``argmax`` tie rule).
* **Grid, selection, budget: the letter control's.** The same four-value L2 grid, ``tol``,
  ``lr`` and ``max_iter``; a 20% validation carve of the TRAINING data drawn by
  ``default_rng(seed).permutation``; the first L2 whose carve accuracy is strictly greater than
  every earlier one; a refit on all the training data. Two things follow from scoring rows
  rather than labels, and both keep the mirror exact rather than loose. The carve is of ROWS --
  a row's options never straddle it, or its question would be on both sides. And the carve
  accuracy is the accuracy of the decision the control is scored on (the best option of each
  carve row), as the letter control's is its own top-1; per-example binary accuracy would be
  dominated by the negatives and select L2 on a quantity nothing is scored by.

This module is the oracle. ``qd-prep linfit``'s option-pairs request (``crates/qd-prep``,
through ``tools/linear_control_native.fit_options``) reproduces it bit for bit on its sparse
operand, as it does the letter control's. It lives outside ``baseline.py`` on purpose:
``control_cache`` hashes that file into every cached letter control, and this control must not
orphan them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qd_data.render import M_ANSWER, M_CTX_BEGIN, M_CTX_END, M_OPT_BEGIN, M_OPT_END, escape_inline

from .baseline import (
    CSR,
    DENSE_OPERAND_BUDGET_BYTES,
    DENSE_SECONDS_PER_CELL_CLASS,
    MIN_FIT_EXAMPLES,
    SPARSE_SECONDS_PER_NONZERO_CLASS,
    CharNGramHasher,
    LinearBaseline,
    RequestDoc,
)
from .tristate import NotRun, Ran, TriState

__all__ = [
    "OptionFit",
    "OptionGridPoint",
    "OptionPairFeatures",
    "OptionRow",
    "OptionScorer",
    "best_option",
    "examples",
    "hstack",
    "pair_doc",
    "read_option_row",
    "read_option_rows",
]

#: Every marker that delimits what this control reads, each of which a readable prompt holds
#: exactly once. ``escape_block``/``escape_inline`` guarantee the context and the options
#: cannot contain ``<|``, so a second copy means the text is not one rendered prompt.
_DELIMITERS = (M_CTX_BEGIN, M_CTX_END, M_OPT_BEGIN, M_OPT_END, M_ANSWER)


@dataclass(frozen=True, slots=True)
class OptionRow:
    """One row of a per-row-option task, as the per-option control reads it."""

    doc: RequestDoc
    #: The prompt's context region: what the row asks, escaped as the model saw it.
    question: str
    #: The shown options' texts, as rendered, in the order shown; noul is among them.
    options: tuple[str, ...]
    #: Index into ``options`` of the gold letter's option.
    gold: int


def read_option_row(doc: RequestDoc) -> OptionRow | str:
    """``doc``'s question and shown options, or the reason they cannot be read.

    The options are ``doc.offered``, accepted only when the prompt's option block is exactly the
    lines the renderer makes from them (``f"{letter}. {escape_inline(value)}"``, noul's line
    last, then the end marker and the answer marker). A doc built by anything but
    :func:`~qd_train.baseline.request_texts` -- or a prompt that does not show what ``offered``
    says it shows -- is refused here, never scored against options the model was not shown.
    """
    if not doc.offered or not doc.letter:
        return (
            f"{doc.row_id}: the doc carries no rendering (no offered options or gold letter); "
            "build it with request_texts"
        )
    text = doc.text
    repeated = [m for m in _DELIMITERS if text.count(m) != 1]
    if repeated:
        return (
            f"{doc.row_id}: the prompt does not hold each delimiter exactly once "
            f"({', '.join(f'{m} x{text.count(m)}' for m in repeated)})"
        )
    letters = [letter for letter, _ in doc.offered]
    if len(set(letters)) != len(letters):
        return f"{doc.row_id}: the offered letters repeat: {letters}"
    if doc.letter not in letters:
        return f"{doc.row_id}: the gold letter {doc.letter!r} is not among the shown {letters}"
    options = tuple(escape_inline(value) for _, value in doc.offered)
    block = (
        f"{M_OPT_BEGIN}\n"
        + "".join(f"{letter}. {shown}\n" for letter, shown in zip(letters, options, strict=True))
        + f"{M_OPT_END}\n{M_ANSWER}"
    )
    if not text.endswith(block):
        return (
            f"{doc.row_id}: the prompt's option block is not the {len(options)} offered "
            "option(s) as the renderer lays them out, so what the model was shown cannot be "
            "read from it"
        )
    # RenderedPrompt.context_region's bounds, read off the text.
    opened = text.find(M_CTX_BEGIN + "\n")
    start = opened + len(M_CTX_BEGIN) + 1
    end = text.rfind("\n" + M_CTX_END)
    if opened < 0 or end < start:
        return f"{doc.row_id}: the prompt has no context region between its delimiters"
    return OptionRow(
        doc=doc, question=text[start:end], options=options, gold=letters.index(doc.letter)
    )


def read_option_rows(
    docs: Sequence[RequestDoc],
) -> tuple[list[OptionRow], list[tuple[RequestDoc, str]]]:
    """``(readable, unreadable)``: every doc read, or kept with the reason it could not be."""
    readable: list[OptionRow] = []
    unreadable: list[tuple[RequestDoc, str]] = []
    for doc in docs:
        got = read_option_row(doc)
        if isinstance(got, str):
            unreadable.append((doc, got))
        else:
            readable.append(got)
    return readable, unreadable


def pair_doc(question: str, option: str) -> str:
    """The pair text one example's first feature block hashes."""
    return f"{question}\n{option}"


def examples(rows: Sequence[OptionRow]) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray]:
    """``(docs, labels, sizes)``: every shown option of every row, rows in order, options in
    shown order. ``docs[i]`` is ``(pair text, option text)``; ``labels[i]`` is 1 for the gold
    option and 0 otherwise; ``sizes[r]`` is row ``r``'s option count."""
    docs = [(pair_doc(r.question, o), o) for r in rows for o in r.options]
    labels = np.asarray(
        [int(j == r.gold) for r in rows for j in range(len(r.options))], dtype=np.int64
    )
    sizes = np.asarray([len(r.options) for r in rows], dtype=np.int64)
    return docs, labels, sizes


def hstack(left: CSR, right: CSR) -> CSR:
    """``[left | right]``: each row's ``left`` entries, then its ``right`` entries at columns
    shifted by ``left``'s width. Both are column-sorted, so the result is too."""
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"cannot stack {left.shape[0]} rows beside {right.shape[0]}")
    n, width = left.shape[0], left.shape[1]
    len_l = left.indptr[1:] - left.indptr[:-1]
    len_r = right.indptr[1:] - right.indptr[:-1]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(len_l + len_r, out=indptr[1:])
    # Where each entry lands: its row's start, plus its offset inside its own block, plus (for
    # the right block) the row's left length.
    at_l = indptr[:-1][left.rows] + (np.arange(left.indices.size) - left.indptr[:-1][left.rows])
    at_r = (
        indptr[:-1][right.rows] + len_l[right.rows]
        + (np.arange(right.indices.size) - right.indptr[:-1][right.rows])
    )
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    data = np.empty(int(indptr[-1]), dtype=np.float64)
    indices[at_l], data[at_l] = left.indices, left.data
    indices[at_r], data[at_r] = right.indices + width, right.data
    return CSR(
        indptr=indptr, indices=indices, data=data,
        rows=np.repeat(np.arange(n), len_l + len_r), shape=(n, width + right.shape[1]),
    )


class OptionPairFeatures:
    """The two feature blocks: n-grams of the pair text, then n-grams of the option alone."""

    def __init__(self, hasher: CharNGramHasher | None = None) -> None:
        self.hasher = hasher or CharNGramHasher()
        self.dim = 2 * self.hasher.dim

    def transform(self, docs: Sequence[tuple[str, str]]) -> CSR:
        return hstack(
            self.hasher.transform([p for p, _ in docs]),
            self.hasher.transform([o for _, o in docs]),
        )

    def nnz_bound(self, docs: Sequence[tuple[str, str]]) -> int:
        return self.hasher.nnz_bound([p for p, _ in docs]) + self.hasher.nnz_bound(
            [o for _, o in docs]
        )


def best_option(logits: np.ndarray, sizes: np.ndarray) -> list[int]:
    """Each row's answer: the index, within the row, of its first option of greatest log-odds
    ``logits[:, 1] - logits[:, 0]``."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"a binary scorer's logits are (n, 2), got {logits.shape}")
    if int(sizes.sum()) != logits.shape[0] or np.any(sizes < 1):
        raise ValueError(
            f"{logits.shape[0]} scored options do not divide into rows of sizes summing to "
            f"{int(sizes.sum())} (every row needs an option)"
        )
    score = logits[:, 1] - logits[:, 0]
    starts = np.cumsum(sizes) - sizes
    return [int(np.argmax(score[s : s + k])) for s, k in zip(starts, sizes, strict=True)]


@dataclass(frozen=True, slots=True)
class OptionGridPoint:
    """One L2 value's fit on the training carve, scored on the carve's rows."""

    l2: float
    val_correct: int
    val_total: int
    converged: bool
    iterations: int
    grad_norm: float


@dataclass(frozen=True, slots=True)
class OptionFit:
    weights: np.ndarray
    bias: np.ndarray
    l2: float
    converged: bool
    iterations: int
    final_grad_norm: float
    loss_history: tuple[float, ...]
    grid: tuple[OptionGridPoint, ...]
    selected: int


class OptionScorer:
    """The per-option control: a binary scorer over each row's shown options."""

    def __init__(
        self,
        *,
        hasher: CharNGramHasher | None = None,
        l2_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
        max_iter: int = 500,
        tol: float = 1e-4,
        lr: float = 0.05,
        seed: int = 0,
        val_frac: float = 0.2,
        dense_budget_bytes: int | None = None,
    ) -> None:
        self.features = OptionPairFeatures(hasher)
        self.l2_grid = l2_grid
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.seed = seed
        self.val_frac = val_frac
        self.dense_budget_bytes = dense_budget_bytes
        self.fit_: OptionFit | None = None

    def _optimizer(self) -> LinearBaseline:
        """The letter control's Adam loop, configured as this control is. Only its
        ``_train_once`` is used: it reseeds, so every fit starts from the same ``W``."""
        return LinearBaseline(
            hasher=self.features.hasher, l2_grid=self.l2_grid, max_iter=self.max_iter,
            tol=self.tol, lr=self.lr, seed=self.seed, dense_budget_bytes=self.dense_budget_bytes,
        )

    def carve(self, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(val_idx, tr_idx, val_sizes)``: the carve's examples and the rest, by row.

        Rows are drawn as ``LinearBaseline.fit`` draws documents -- ``default_rng(seed)
        .permutation(n)``, the first ``max(1, int(n * val_frac))`` to validation -- and each
        row's options follow it, in shown order, keeping the permutation's row order.
        """
        n = len(sizes)
        if n < MIN_FIT_EXAMPLES:
            raise ValueError(f"need at least {MIN_FIT_EXAMPLES} rows to fit and validate, got {n}")
        order = np.random.default_rng(self.seed).permutation(n)
        n_val = max(1, int(n * self.val_frac))
        val_rows, tr_rows = order[:n_val], order[n_val:]
        if len(tr_rows) == 0:
            raise ValueError("validation split consumed the whole training set")
        starts = np.cumsum(sizes) - sizes

        def spread(rows: np.ndarray) -> np.ndarray:
            return np.concatenate(
                [np.arange(starts[r], starts[r] + sizes[r]) for r in rows]
            ).astype(np.int64)

        return spread(val_rows), spread(tr_rows), sizes[val_rows]

    def projected_fit_seconds(self, rows: Sequence[OptionRow]) -> float:
        """``LinearBaseline.projected_fit_seconds``'s worst-case bound, for two classes over the
        option examples and both feature blocks."""
        docs, _, _ = examples(rows)
        n, d = len(docs), self.features.dim
        limit = (
            DENSE_OPERAND_BUDGET_BYTES if self.dense_budget_bytes is None
            else self.dense_budget_bytes
        )
        if n * d * 8 <= limit:
            per_iteration = DENSE_SECONDS_PER_CELL_CLASS * n * d * 2
        else:
            per_iteration = SPARSE_SECONDS_PER_NONZERO_CLASS * self.features.nnz_bound(docs) * 2
        return per_iteration * self.max_iter * (len(self.l2_grid) + 1)

    def fit(self, rows: Sequence[OptionRow]) -> OptionFit:
        """Fit, selecting L2 on a carve of the training rows by the best option per row."""
        docs, y, sizes = examples(rows)
        if np.any(sizes < 2):
            raise ValueError("a row with fewer than two shown options has nothing to choose")
        val_idx, tr_idx, val_sizes = self.carve(sizes)
        X = self.features.transform(docs)
        X_tr, X_val = X.select(tr_idx), X.select(val_idx)
        y_val = y[val_idx]
        val_starts = np.cumsum(val_sizes) - val_sizes
        optimizer = self._optimizer()

        grid: list[OptionGridPoint] = []
        best: tuple[float, int] | None = None
        for i, l2 in enumerate(self.l2_grid):
            W, b, conv, it, gn, hist = optimizer._train_once(X_tr, y[tr_idx], 2, l2)
            picks = best_option(X_val.matmul(W) + b, val_sizes)
            correct = sum(
                int(y_val[s + p] == 1) for s, p in zip(val_starts, picks, strict=True)
            )
            acc = float(correct / len(val_sizes))
            if best is None or acc > best[0]:
                best = (acc, i)
            grid.append(OptionGridPoint(l2, correct, len(val_sizes), conv, it, gn))
        assert best is not None  # l2_grid is non-empty by construction

        selected = best[1]
        l2 = self.l2_grid[selected]
        W, b, conv, it, gn, hist = optimizer._train_once(X, y, 2, l2)
        self.fit_ = OptionFit(
            weights=W, bias=b, l2=l2, converged=conv, iterations=it, final_grad_norm=gn,
            loss_history=tuple(hist), grid=tuple(grid), selected=selected,
        )
        return self.fit_

    def logits(self, rows: Sequence[OptionRow]) -> np.ndarray:
        if self.fit_ is None:
            raise RuntimeError("fit() before logits()")
        docs, _, _ = examples(rows)
        return self.features.transform(docs).matmul(self.fit_.weights) + self.fit_.bias

    def predict(self, rows: Sequence[OptionRow]) -> list[int]:
        """Each row's answer, as an index into its shown options."""
        _, _, sizes = examples(rows)
        return best_option(self.logits(rows), sizes)

    def convergence(self) -> TriState:
        """``LinearBaseline.convergence``'s verdict on this control's refit."""
        if self.fit_ is None:
            return NotRun(reason="the per-option control was never fitted")
        if not self.fit_.converged:
            return NotRun(
                reason=(
                    f"the per-option control did not converge in {self.fit_.iterations} "
                    f"iterations (final grad norm {self.fit_.final_grad_norm:.3e} > tol "
                    f"{self.tol:.1e}); an unconverged control cannot be 'beaten'"
                )
            )
        return Ran(
            passed=True,
            value=self.fit_.final_grad_norm,
            detail=f"converged in {self.fit_.iterations} iterations at l2={self.fit_.l2}",
        )
