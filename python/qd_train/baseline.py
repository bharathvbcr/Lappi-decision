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

from dataclasses import dataclass

import numpy as np

from .tristate import NotRun, Ran, TriState

__all__ = [
    "CSR",
    "DENSE_OPERAND_BUDGET_BYTES",
    "BaselineFit",
    "CharNGramHasher",
    "DenseOperand",
    "LinearBaseline",
]


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

    def as_operand(self) -> CSR | DenseOperand:
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
        """
        n, d = self.shape
        if n * d * self.data.dtype.itemsize > DENSE_OPERAND_BUDGET_BYTES:
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
        hasher: CharNGramHasher | None = None,
        l2_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
        max_iter: int = 500,
        tol: float = 1e-4,
        lr: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.hasher = hasher or CharNGramHasher()
        self.l2_grid = l2_grid
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.seed = seed
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
        ops = X.as_operand()
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
