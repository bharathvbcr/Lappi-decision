"""The char-n-gram linear baseline: the control arm the whole decision rests on.

The plan drops the transformer control arm; this is the only control. The 2B ships
only if it beats this by a paired margin on three seeds on the natural held-out set.

That makes the baseline's *strength* a correctness property, not an implementation
detail. A weak baseline is not a conservative choice — it manufactures a win for the
model and corrupts the one gate that decides the program. So:

- Convergence is checked and reported, never assumed. A baseline that did not
  converge is `NotRun`, not a low score: a model cannot beat a control that never
  finished training, and reporting it as beaten is the failure this guards.
- Regularization is selected on a validation split, not fixed at a guess, so the
  baseline is the best linear model on these features rather than the first one.

Implemented on numpy rather than scikit-learn to avoid adding a dependency. The
optimizer is Adam with an explicit convergence criterion; see `LOGISTIC_NOTE` in the
handoff for the argument that this is not weaker than sklearn's lbfgs for this
problem, and for the check that was run to establish it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from qd_data.errors import QdRefusal
from qd_data.render import DEFAULT_CAPS, RenderCaps, render
from qd_data.rows import DataRow

from .artifacts import SLOT_CHOICE, SLOT_SCORE, line_start_indices
from .mutate_adapter import MUTATION_CLASSES
from .shards import UnencodableGold, answer_letter, training_texts
from .tristate import NotRun, Ran, TriState

__all__ = [
    "CSR",
    "DENSE_OPERAND_BUDGET_BYTES",
    "LETTER_KINDS",
    "BaselineFit",
    "CharNGramHasher",
    "ContextLengthFeatures",
    "DenseOperand",
    "Featurizer",
    "LabelSpace",
    "LinearBaseline",
    "RequestDoc",
    "context_texts",
    "control_label",
    "control_label_space",
    "fit_budget_refusal",
    "request_texts",
]


def context_texts(decisions) -> tuple[list[str], list[str]]:
    """Exactly the bytes the model sees, as text, with the gold class name.

    ``EncodedContext.ids`` are byte values and ``ids[i] == raw[i]`` for every kept byte, so
    this is the model's own input rather than a re-read of the source file -- the same
    truncation, the same window. Comparing against a control that saw *more* of the file
    would be comparing two different tasks.

    **Here because the paired margin depends on there being exactly one of it.** It lived
    in ``tools/rung0_linear_control.py``, which imports ``tools/rung0_real_run.py`` -- so
    the run could not import it back at module scope without a cycle, and did it inside a
    function instead. Three tools now render the control's input, and a second copy of this
    would be free to drift into scoring a different task while still producing a margin
    that looked measured. It belongs with the control that consumes it, which is this
    module. ``GAP-CONTEXT-TEXTS-HAS-NO-OWNER``.
    """
    docs: list[str] = []
    labels: list[str] = []
    for d in decisions:
        # errors="replace" only where a multi-byte character was split by the window. The
        # model sees those same split bytes; the replacement affects the control's
        # tokenisation of one character at the boundary, not what either model was given.
        docs.append(bytes(d.context.ids).decode("utf-8", errors="replace"))
        labels.append(MUTATION_CLASSES[d.gold_option])
    return docs, labels


def fit_budget_refusal(projected_s: float, max_fit_minutes: float | None) -> str | None:
    """The refusal to print and exit on, or ``None`` when the fit may start.

    This module already knew how long the fit would take -- ``projected_fit_seconds`` is
    computed and printed one line before the fit begins -- and did nothing with it. A
    caller that wrapped the fit in a shorter ``timeout`` therefore got the worst of both:
    the box saturated for the length of the cap, the process killed before it converged,
    and nothing written to the cache. The arm that reads the cache then reports
    ``paired_margin_vs_linear: not_run`` exactly as it would have if the fit had never
    been launched, so the wasted hour leaves no trace distinguishing it from doing nothing.

    That already happened once in this repository -- 5fd0ea8, *"The linear control could
    not finish inside the cap it was launched under"*.

    So the projection becomes a precondition rather than a progress message. Refusing
    costs the caller nothing it would otherwise have had, and it turns a silent hour into
    an immediate, legible error naming both numbers.

    **The projection is a worst-case bound, and the cap must be read against it as one.**
    ``projected_fit_seconds`` prices ``max_iter`` iterations. The optimiser stops at ``tol``
    instead, usually far earlier: measured 2026-09-22 on 37,385 training documents, the
    projection was 101.4 minutes and the fit converged in 443 iterations and **345.7s** --
    an overshoot of 17.6x. Six sibling fits landed between 204.2s and 351.7s against the
    same projection.

    Two things follow, and the second is easy to get backwards. A cap must be set above the
    PROJECTION, not above observed times, or this refuses fits that would have finished
    comfortably -- a 60-minute cap would reject a six-minute fit. And a projection must
    never be quoted as a cost: doing that turned a $0.91 job into a documented $22 one and
    routed it to a human as a spending decision it did not need to be.

    ``None`` for ``max_fit_minutes`` means the caller accepts any duration, which is the
    right default for an interactive fit that owns its own terminal.
    """
    if max_fit_minutes is None:
        return None
    if projected_s <= max_fit_minutes * 60:
        return None
    return (
        f"refusing to start: the fit projects to {projected_s / 60:.1f} minute(s) but "
        f"--max-fit-minutes is {max_fit_minutes:g}. It would be killed before it "
        "converged and NOTHING would be cached, which the arm that reads this cache "
        "cannot tell apart from a fit that was never launched.\n"
        "Raise the caller's cap above the projection, or fit a smaller training set. "
        "Do not lower --max-iter to fit inside the cap: that weakens the opponent the "
        "model is measured against, which is a promotion decision and not this tool's."
    )


#: The slot kinds the linear control can stand opposite: the letter channel. A span slot's
#: answer is a pair of line numbers chosen by the pointer head, which a bag of n-grams has no
#: way to produce, so it is excluded by name rather than scored as a constant.
LETTER_KINDS: dict[int, str] = {SLOT_CHOICE: "choice", SLOT_SCORE: "score"}


@dataclass(frozen=True, slots=True)
class RequestDoc:
    """One letter slot of one FT row, as the linear control sees it.

    ``text`` is ``render(...).prompt_for(slot)`` -- the prefix and the slot's suffix, which is
    every byte the model conditions on for that slot's answer. ``value`` is the gold VALUE,
    ``letter`` the gold letter in this rendering, and ``offered`` every ``(letter, value)``
    pair the rendering shows, noul included. Which of ``value`` and ``letter`` the control is
    labelled by is a per-task decision, :func:`control_label_space`; either way correctness
    is the same question the model's ``top == gold_row`` answers, because on one row the
    letter and the value name the same option.
    """

    row_id: str
    slot_name: str
    kind: str
    #: ``family_id/slot_name``: the task. One control per task, because a ``change_scope``
    #: bin is not a candidate answer to ``commit_intent``.
    task: str
    text: str
    value: str
    letter: str
    offered: tuple[tuple[str, str], ...]
    metadata: Mapping[str, str] = field(default_factory=dict)
    #: ``row.request.context`` alone -- what :class:`ContextLengthFeatures` measures. Empty
    #: for a doc built without it, which the length control reports as not run rather than
    #: scoring as a zero-length context. ``str`` or ``bytes``, as the request carries it.
    context: str | bytes = ""


def request_texts(
    rows: Iterable[DataRow], *, seed: int, caps: RenderCaps = DEFAULT_CAPS
) -> tuple[list[RequestDoc], list[str]]:
    """``(docs, excluded)``: every letter slot of every row, rendered the way FT renders it.

    The FT counterpart of :func:`context_texts`, and here for the same reason: the paired
    margin depends on there being exactly one rendering of the control's input. The row
    admission is the writer's -- ``training_texts`` raises ``UnencodableGold`` or a
    ``QdRefusal`` for exactly the rows ``write_shards`` drops, and a row dropped there has no
    model verdict to pair with -- so the same ``except`` drops it here and names it.

    ``seed`` must be the ``DataConfig.seed`` the shards were written at. Rendered at any
    other seed the choice options come out in a different order and the prompt text is not
    the one the model read.
    """
    docs: list[RequestDoc] = []
    excluded: list[str] = []
    for row in sorted(rows, key=lambda r: r.row_id):
        staged: list[RequestDoc] = []
        try:
            rendered = render(row.request, caps=caps, seed=seed)
            for spec in training_texts(row, seed=seed, caps=caps):
                kind = LETTER_KINDS.get(spec.slot_kind)
                if kind is None:
                    continue
                slot = rendered.slot(spec.slot_name)
                letter = answer_letter(row, spec.slot_name, slot.letter_to_value)
                staged.append(
                    RequestDoc(
                        row_id=row.row_id,
                        slot_name=spec.slot_name,
                        kind=kind,
                        task=f"{row.family_id}/{spec.slot_name}",
                        text=rendered.prompt_for(spec.slot_name),
                        value=slot.letter_to_value[letter],
                        letter=letter,
                        offered=tuple(slot.letter_to_value.items()),
                        metadata=dict(row.metadata),
                        context=row.request.context,
                    )
                )
        except (UnencodableGold, QdRefusal) as exc:
            excluded.append(f"{row.row_id}: {type(exc).__name__}: {exc}")
            continue
        docs.extend(staged)
    return docs, excluded


#: What a task's control is labelled by. See :func:`control_label_space`.
LabelSpace = Literal["value", "letter"]


def control_label_space(train: Sequence[RequestDoc], val: Sequence[RequestDoc]) -> LabelSpace:
    """``"value"`` when every row of the task offers one option set, else ``"letter"``.

    The control is a classifier over one class set, so its labels have to mean the same
    thing on every row. Which label does depends on where the options come from:

    * **One option set on every row** (``code.defect_class``'s classes, yes/no, a score's
      bins, CLINC's domains). The value is the class. The letter is not: ``qd_data.render``
      shuffles a choice slot's options per example, so ``'A'`` names a different class on
      every row, and a control asked to predict it from n-grams would be an artificially
      weak opponent -- which manufactures a win.
    * **The row's own options** (CommonsenseQA, MMLU, CLINC's sampled and per-domain intent
      sets). The value is not a class: it is one of THIS question's options, so values barely
      recur across rows (k=4,956 on 9,619 CSQA train rows, k=12,080 on 14,200 MMLU rows, on
      the 2026-09-29 general record without the replay partition), and where the split is by
      label -- CLINC's ``repo_key`` is the intent -- no val gold is a training class at all
      (0/1,500 on ``intent.classification``). A control over those classes
      cannot answer the question it is scored on; J1's CSQA control selected its L2 at
      4/1635 on a five-way task. The letter is the one label every row shares, and the
      model answers in letters.

    Decided from the option sets the rows OFFER -- the prompt -- never from their golds, so
    no val label is read to configure the control. A train split that offers one set while
    val offers another is refused: the control's classes would not cover what val asks, and
    that is a split defect to fix, not a label space to guess. So is a doc without its
    rendering's options, or docs from more than one task.
    """
    docs = [*train, *val]
    tasks = sorted({d.task for d in docs})
    if len(tasks) > 1:
        raise ValueError(f"a control's label space is one task's; these docs span {tasks}")
    bare = [d.row_id for d in docs if not d.offered or not d.letter]
    if bare:
        raise ValueError(
            f"{len(bare)} doc(s) carry no rendering (no offered options or gold letter), "
            f"first {bare[:3]}: build them with request_texts, which records what the model "
            "was shown"
        )
    train_sets = {frozenset(v for _, v in d.offered) for d in train}
    val_sets = {frozenset(v for _, v in d.offered) for d in val}
    if len(train_sets | val_sets) == 1:
        return "value"
    if len(train_sets) == 1:
        (fixed,) = train_sets
        differing = sorted(sorted(s) for s in val_sets if s != fixed)
        raise ValueError(
            f"task {tasks[0]}: every training row offers the option set {sorted(fixed)} and "
            f"{len(differing)} val option set(s) differ from it, first {differing[0]}; the "
            "control's classes would not cover what val asks"
        )
    return "letter"


def control_label(doc: RequestDoc, space: LabelSpace) -> str:
    """``doc``'s label in ``space``: its gold value, or its gold letter in this rendering."""
    if space == "value":
        return doc.value
    if space == "letter":
        return doc.letter
    raise ValueError(f"unknown label space {space!r}")


@dataclass(frozen=True, slots=True)
class CSR:
    """Sparse rows: `data[indptr[i]:indptr[i+1]]` at columns `indices[...]`.

    Char n-grams are extremely sparse — a 2 KB diff touches a few thousand of 262144
    columns. A dense encoding is not merely slower: at the real scale of ~400K
    training examples it is ~26 GB of mostly zeros, so a dense baseline could not be
    fitted on the full set at all, and a control arm that cannot see the whole
    training set is a weak control.
    """

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    rows: np.ndarray  # row index per nonzero; precomputed, used by both matmuls
    shape: tuple[int, int]

    def matmul(self, W: np.ndarray) -> np.ndarray:
        """X @ W -> (n, k). One bincount per class, each O(nnz)."""
        n, k = self.shape[0], W.shape[1]
        out = np.zeros((n, k), dtype=np.float64)
        if self.indices.size:
            gathered = W[self.indices]  # (nnz, k)
            for c in range(k):
                out[:, c] = np.bincount(self.rows, weights=self.data * gathered[:, c], minlength=n)
        return out

    def rmatmul(self, D: np.ndarray) -> np.ndarray:
        """X.T @ D -> (d, k). One bincount per class, each O(nnz + d)."""
        d, k = self.shape[1], D.shape[1]
        out = np.zeros((d, k), dtype=np.float64)
        if self.indices.size:
            spread = D[self.rows]  # (nnz, k)
            for c in range(k):
                out[:, c] = np.bincount(self.indices, weights=self.data * spread[:, c], minlength=d)
        return out

    def select(self, idx: np.ndarray) -> CSR:
        """Row subset, preserving sparsity (used for the L2 validation split)."""
        lengths = (self.indptr[1:] - self.indptr[:-1])[idx]
        take = np.concatenate(
            [np.arange(self.indptr[i], self.indptr[i + 1]) for i in idx]
        ) if idx.size else np.empty(0, dtype=np.int64)
        new_indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        return CSR(
            indptr=new_indptr,
            indices=self.indices[take],
            data=self.data[take],
            rows=np.repeat(np.arange(len(idx)), lengths),
            shape=(len(idx), self.shape[1]),
        )

    def as_operand(self, *, budget_bytes: int | None = None) -> CSR | DenseOperand:
        """`self`, or a dense equivalent when one fits inside the budget.

        Both matmuls above are gather-then-scatter: each allocates an `(nnz, k)`
        temporary and runs `k` bincounts over it. That is correct and it is what makes
        the full-scale set fittable at all, but it moves memory in a random-access
        pattern that no BLAS can help with. Measured on the rung-0 corpus -- 774 docs,
        `nnz` 6,325,171, `d` 65,536, so 12.5% dense, which is not sparse in the sense
        the CSR was built for -- it costs 0.379s per iteration. Across the 4-value L2
        grid plus the refit at `max_iter=6000` that is 30,000 iterations and 3.2 hours,
        which is how a run launched under a 9000s cap spent ten minutes in this function
        and wrote zero ledger rows.

        Dense does ~8x more arithmetic and is still far quicker, because a GEMM streams
        and threads where a gather does neither.

        The sparse path is not replaced. `CSR`'s docstring is right that at ~400K
        examples a dense encoding is ~26 GB and could not be fitted at all, and a
        control arm that cannot see the whole training set is a weak control. So the
        choice is made once, by an explicit byte budget, and above it nothing changes.

        `budget_bytes` is a parameter because the right answer depends on who is asking.
        Inside a training run the default is deliberately small: that process owns a GPU,
        and a multi-gigabyte allocation there buys nothing a cache would not. A dedicated
        fit that owns the machine passes a real one -- the commitpackft corpus is 20.9 GB
        dense on a box with 406 GB free, which is the difference between the control being
        fittable on the corpus the plan names and not.
        """
        n, d = self.shape
        # Resolved here rather than as a default argument so the module-level budget stays
        # patchable; a default is bound once at definition and would ignore it.
        limit = DENSE_OPERAND_BUDGET_BYTES if budget_bytes is None else budget_bytes
        if n * d * self.data.dtype.itemsize > limit:
            return self
        # Densifying by assignment is only equal to the sparse path when each
        # (row, column) appears once; a duplicate would be overwritten rather than
        # summed, and the two paths would silently disagree. `transform` accumulates
        # into a per-row dict and sorts, and `select` preserves that, so duplicates
        # are adjacent if they exist at all -- which makes the check O(nnz).
        if self.indices.size > 1 and np.any(
            (np.diff(self.rows) == 0) & (np.diff(self.indices) == 0)
        ):
            raise ValueError(
                "CSR holds a duplicate (row, column) entry, so it cannot be "
                "densified by assignment without changing the value it represents"
            )
        dense = np.zeros(self.shape, dtype=self.data.dtype)
        dense[self.rows, self.indices] = self.data
        return DenseOperand(dense=dense, shape=self.shape)


#: Above this, `CSR.as_operand` keeps the sparse path. 512 MB admits the rung-0 corpus
#: (774 x 65,536 float64 = 406 MB) and refuses the full-scale set by two orders of
#: magnitude, which is the intent: make the affordable case fast without pretending the
#: unaffordable one has become affordable.
DENSE_OPERAND_BUDGET_BYTES = 512 * 1024 * 1024

#: Per-iteration cost, calibrated on the GH200 box on 2026-09-21 against the rung-0 corpus
#: (n=774, d=65,536, k=4, nnz=6,325,171): the sparse path measured 0.3647s and the dense
#: path 0.0042s per iteration. Both paths are memory-bound and scale with the quantity
#: named, so one measured point fixes each constant. They exist to let
#: `projected_fit_seconds` answer "how long would this take" WITHOUT running it -- a
#: control that silently takes 150 hours and a control that reports it cannot be fitted are
#: very different records, and only the second one is honest.
SPARSE_SECONDS_PER_NONZERO_CLASS = 0.3647 / (6_325_171 * 4)
DENSE_SECONDS_PER_CELL_CLASS = 0.0042 / (774 * 65_536 * 4)


@dataclass(frozen=True, slots=True)
class DenseOperand:
    """A `CSR` materialised dense, exposing the same two matmuls so the training loop
    does not branch. One loop, two representations -- not two loops that can drift."""

    dense: np.ndarray
    shape: tuple[int, int]

    def matmul(self, W: np.ndarray) -> np.ndarray:
        """X @ W -> (n, k)."""
        return self.dense @ W

    def rmatmul(self, D: np.ndarray) -> np.ndarray:
        """X.T @ D -> (d, k)."""
        return self.dense.T @ D


class CharNGramHasher:
    """Hashed char n-gram features. Deterministic, dependency-free, no vocabulary.

    Hashing rather than a learned vocabulary because the vocabulary would have to be
    fitted on training text and is one more thing that can leak across a split.
    """

    def __init__(self, *, n_min: int = 3, n_max: int = 5, dim: int = 2**16) -> None:
        if n_min < 1 or n_max < n_min:
            raise ValueError(f"require 1 <= n_min <= n_max, got {n_min}, {n_max}")
        if dim < 16 or (dim & (dim - 1)) != 0:
            raise ValueError(f"dim must be a power of two >= 16, got {dim}")
        self.n_min, self.n_max, self.dim = n_min, n_max, dim

    def nnz_bound(self, docs: list[str]) -> int:
        """An upper bound on ``transform(docs)``'s nonzeros: at most one column per n-gram,
        and at most ``dim`` per document. High is the safe direction for a fit-time guard."""
        orders = self.n_max - self.n_min + 1
        return sum(min(orders * len(doc), self.dim) for doc in docs)

    def _hash(self, gram: str) -> int:
        # FNV-1a, 64-bit. Stable across processes and platforms, unlike hash(),
        # which is salted per interpreter and would make features irreproducible.
        h = 0xCBF29CE484222325
        for byte in gram.encode("utf-8"):
            h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return h & (self.dim - 1)

    def transform(self, docs: list[str]) -> CSR:
        indptr = np.zeros(len(docs) + 1, dtype=np.int64)
        all_idx: list[np.ndarray] = []
        all_val: list[np.ndarray] = []

        for i, doc in enumerate(docs):
            counts: dict[int, float] = {}
            for n in range(self.n_min, self.n_max + 1):
                if len(doc) < n:
                    continue
                for j in range(len(doc) - n + 1):
                    col = self._hash(doc[j : j + n])
                    counts[col] = counts.get(col, 0.0) + 1.0
            if counts:
                cols = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
                vals = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
                norm = float(np.sqrt(np.sum(vals * vals)))
                if norm > 0:
                    vals = vals / norm  # L2; a zero row stays absent, never NaN
                order = np.argsort(cols)
                all_idx.append(cols[order])
                all_val.append(vals[order])
                indptr[i + 1] = indptr[i] + len(cols)
            else:
                indptr[i + 1] = indptr[i]

        indices = np.concatenate(all_idx) if all_idx else np.empty(0, dtype=np.int64)
        data = np.concatenate(all_val) if all_val else np.empty(0, dtype=np.float64)
        lengths = indptr[1:] - indptr[:-1]
        return CSR(
            indptr=indptr,
            indices=indices,
            data=data,
            rows=np.repeat(np.arange(len(docs)), lengths),
            shape=(len(docs), self.dim),
        )


class Featurizer(Protocol):
    """What ``LinearBaseline`` needs from a feature map: a width, a transform, and a bound on
    the transform's nonzeros for ``projected_fit_seconds``."""

    dim: int

    def transform(self, docs: list[str]) -> CSR: ...

    def nnz_bound(self, docs: list[str]) -> int: ...


class ContextLengthFeatures:
    """The context's size and nothing else: a control for labels that length predicts.

    ``GAP-A3-CLEAN-DIFFS-ARE-LONGER-THAN-MUTATION-DIFFS``: in ``code.defect_class`` a clean
    row is a real commit's diff and a mutated row one synthetic edit, so clean diffs are
    longer (p50 739 bytes against 286-410) and a model can beat the majority rate by reading
    size alone. Fitted by :class:`LinearBaseline` exactly as the n-gram control is -- same
    L2 grid, same convergence check -- on these five dense features of the CONTEXT (not the
    rendered prompt, whose fixed scaffolding would only add a constant):

    ``b = log2(1 + utf8 bytes) / 20`` and ``l = log2(1 + lines) / 16`` -- lines by
    :func:`qd_train.artifacts.line_start_indices`, the system's one line rule -- then
    ``b, l, b*b, l*l, b*l``. The quadratic terms let a linear model put a class in a middle
    band of length rather than only at one end, which is the stronger opponent. The fixed
    scales keep every feature near ``[0, 1]`` without fitting a normaliser on anything.
    """

    dim: int = 5

    def features(self, doc: str | bytes) -> tuple[float, float, float, float, float]:
        # A request context is str or bytes; bytes are counted on UTF-8 either way, and
        # line_start_indices counts the same lines in both (a newline is one byte, one char).
        raw = doc.encode("utf-8") if isinstance(doc, str) else bytes(doc)
        b = float(np.log2(1 + len(raw))) / 20.0
        lines = float(np.log2(1 + len(line_start_indices(raw)))) / 16.0
        return (b, lines, b * b, lines * lines, b * lines)

    def nnz_bound(self, docs: Sequence[str | bytes]) -> int:
        return len(docs) * self.dim

    def transform(self, docs: Sequence[str | bytes]) -> CSR:
        n = len(docs)
        data = np.asarray([v for doc in docs for v in self.features(doc)], dtype=np.float64)
        return CSR(
            indptr=np.arange(0, n * self.dim + 1, self.dim, dtype=np.int64),
            indices=np.tile(np.arange(self.dim, dtype=np.int64), n),
            data=data,
            rows=np.repeat(np.arange(n), self.dim),
            shape=(n, self.dim),
        )


@dataclass(frozen=True, slots=True)
class BaselineFit:
    weights: np.ndarray
    bias: np.ndarray
    classes: tuple[str, ...]
    l2: float
    converged: bool
    iterations: int
    final_grad_norm: float
    loss_history: tuple[float, ...]


class LinearBaseline:
    """Multinomial logistic regression on hashed char n-grams, via Adam + L2."""

    def __init__(
        self,
        *,
        hasher: Featurizer | None = None,
        l2_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
        max_iter: int = 500,
        tol: float = 1e-4,
        lr: float = 0.05,
        seed: int = 0,
        dense_budget_bytes: int | None = None,
    ) -> None:
        self.hasher: Featurizer = hasher or CharNGramHasher()
        self.l2_grid = l2_grid
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.seed = seed
        self.dense_budget_bytes = dense_budget_bytes
        self.fit_: BaselineFit | None = None

    # -- internals -------------------------------------------------------

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _train_once(
        self, X: CSR, y: np.ndarray, n_classes: int, l2: float
    ) -> tuple[np.ndarray, np.ndarray, bool, int, float, list[float]]:
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        # Chosen once, outside the loop: the loop below is written against the two
        # matmuls and never learns which representation answered them.
        ops = X.as_operand(budget_bytes=self.dense_budget_bytes)
        W = rng.normal(0.0, 0.01, size=(d, n_classes)).astype(np.float64)
        b = np.zeros(n_classes, dtype=np.float64)
        Y = np.zeros((len(y), n_classes), dtype=np.float64)
        Y[np.arange(len(y)), y] = 1.0

        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        mb = np.zeros_like(b)
        vb = np.zeros_like(b)
        b1, b2, eps = 0.9, 0.999, 1e-8
        history: list[float] = []
        converged, grad_norm, it = False, float("inf"), 0

        for it in range(1, self.max_iter + 1):
            P = self._softmax(ops.matmul(W) + b)
            loss = float(-np.sum(Y * np.log(np.clip(P, 1e-12, None))) / len(y) + l2 * np.sum(W * W))
            history.append(loss)

            diff = (P - Y) / len(y)
            gW = ops.rmatmul(diff) + 2.0 * l2 * W
            gb = diff.sum(axis=0)
            grad_norm = float(np.sqrt(np.sum(gW * gW) + np.sum(gb * gb)))
            if grad_norm < self.tol:
                converged = True
                break

            for p, g, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
                m *= b1
                m += (1 - b1) * g
                v *= b2
                v += (1 - b2) * (g * g)
                mhat = m / (1 - b1**it)
                vhat = v / (1 - b2**it)
                p -= self.lr * mhat / (np.sqrt(vhat) + eps)

        return W, b, converged, it, grad_norm, history

    # -- API -------------------------------------------------------------

    def projected_fit_seconds(self, docs: list[str], *, n_classes: int) -> float:
        """What `fit` would cost on these documents, without hashing or fitting them.

        The rung-0 corpus is 774 documents and fits in 38s. The commitpackft corpus the
        plan names is 39,946 training documents: `n * d * 8` is 20.9 GB, so `as_operand`
        keeps the sparse path, and the sparse path scales with `nnz` -- about 320M here
        against 6.3M -- which projects to roughly 18s per iteration and 150 hours for the
        grid plus refit. Launching into that is indistinguishable from a hang, and the run
        of 2026-09-21 has already established what that costs to diagnose.

        So the caller asks first. `nnz` is an upper bound rather than the true count: a
        document contributes at most one column per n-gram and at most `dim` columns
        overall, and duplicates only reduce it. Bounding it high is the safe direction for
        a guard -- it can refuse a fit that would have been affordable, which is visible
        and fixable, but it will not admit one that is not, which is the failure that
        burns a run.
        """
        d = self.hasher.dim
        n = len(docs)
        nnz = self.hasher.nnz_bound(docs)
        limit = (
            DENSE_OPERAND_BUDGET_BYTES
            if self.dense_budget_bytes is None
            else self.dense_budget_bytes
        )
        if n * d * 8 <= limit:
            per_iteration = DENSE_SECONDS_PER_CELL_CLASS * n * d * n_classes
        else:
            per_iteration = SPARSE_SECONDS_PER_NONZERO_CLASS * nnz * n_classes
        # The L2 grid is one fit per value, plus the refit on all the training data.
        return per_iteration * self.max_iter * (len(self.l2_grid) + 1)

    def fit(self, docs: list[str], labels: list[str], *, val_frac: float = 0.2) -> BaselineFit:
        """Fit, selecting L2 on a held-out slice of the training data.

        The validation slice is carved from *training* data only. Touching the real
        held-out set to tune the baseline would be the same leak the controls exist
        to catch, committed by the control itself.
        """
        if len(docs) != len(labels):
            raise ValueError(f"docs and labels differ in length: {len(docs)} vs {len(labels)}")
        if len(docs) < 4:
            raise ValueError(f"need at least 4 examples to fit and validate, got {len(docs)}")

        classes = tuple(sorted(set(labels)))
        if len(classes) < 2:
            raise ValueError(f"need at least 2 classes, got {classes}")
        index = {c: i for i, c in enumerate(classes)}
        y = np.array([index[label] for label in labels], dtype=np.int64)
        X = self.hasher.transform(docs)

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(docs))
        n_val = max(1, int(len(docs) * val_frac))
        val_idx, tr_idx = order[:n_val], order[n_val:]
        if len(tr_idx) == 0:
            raise ValueError("validation split consumed the whole training set")

        X_tr, X_val = X.select(tr_idx), X.select(val_idx)
        best: (
            tuple[float, float, np.ndarray, np.ndarray, bool, int, float, list[float]] | None
        ) = None
        for l2 in self.l2_grid:
            W, b, conv, it, gn, hist = self._train_once(X_tr, y[tr_idx], len(classes), l2)
            acc = float((np.argmax(X_val.matmul(W) + b, axis=1) == y[val_idx]).mean())
            if best is None or acc > best[0]:
                best = (acc, l2, W, b, conv, it, gn, hist)
        assert best is not None  # l2_grid is non-empty by construction

        # Refit the winning L2 on all the training data.
        _, l2, *_ = best
        W, b, conv, it, gn, hist = self._train_once(X, y, len(classes), l2)
        self.fit_ = BaselineFit(
            weights=W, bias=b, classes=classes, l2=l2,
            converged=conv, iterations=it, final_grad_norm=gn, loss_history=tuple(hist),
        )
        return self.fit_

    def predict_proba(self, docs: list[str]) -> np.ndarray:
        if self.fit_ is None:
            raise RuntimeError("fit() before predict_proba()")
        X = self.hasher.transform(docs)
        return self._softmax(X.matmul(self.fit_.weights) + self.fit_.bias)

    def predict(self, docs: list[str]) -> list[str]:
        if self.fit_ is None:
            raise RuntimeError("fit() before predict()")
        idx = np.argmax(self.predict_proba(docs), axis=1)
        return [self.fit_.classes[i] for i in idx]

    def convergence(self) -> TriState:
        """Did the control arm actually train?

        `NotRun` when it did not converge — *not* a low accuracy. A model cannot
        beat a baseline that never finished training, and scoring it as though it
        had is how a weak control manufactures a win.
        """
        if self.fit_ is None:
            return NotRun(reason="baseline was never fitted")
        if not self.fit_.converged:
            return NotRun(
                reason=(
                    f"baseline did not converge in {self.fit_.iterations} iterations "
                    f"(final grad norm {self.fit_.final_grad_norm:.3e} > tol {self.tol:.1e}); "
                    "an unconverged control cannot be 'beaten'"
                )
            )
        return Ran(
            passed=True,
            value=self.fit_.final_grad_norm,
            detail=f"converged in {self.fit_.iterations} iterations at l2={self.fit_.l2}",
        )
