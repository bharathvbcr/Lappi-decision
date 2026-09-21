"""The dense operand: the linear control has to be affordable to be a control at all.

`CSR.matmul`/`rmatmul` are gather-then-scatter. On the rung-0 corpus that is 0.379s per
iteration, and the 4-value L2 grid plus the refit at `max_iter=6000` is 30,000 of them --
3.2 hours, against a run capped at 9000s. A run launched on 2026-09-21 spent ten minutes
inside `LinearBaseline.fit` with the GPU at 0% and wrote zero ledger rows before it was
terminated. A gate whose opponent cannot be scored inside the budget is a gate that never
reports, which is the shape this repository keeps finding.

So these tests hold two things at once:

* the dense path exists and is **exactly** the sparse path's arithmetic, to floating-point
  reassociation only -- otherwise the speedup would have quietly changed the control's
  score, and a control that moved because it got faster is not a control;
* the budget still refuses the full-scale case, because `CSR`'s reason for existing is that
  ~400K examples dense is ~26 GB and a control that cannot see the whole training set is a
  weak control.

Every test here fails against the pre-fix module, where `as_operand` does not exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_train.baseline import (  # noqa: E402
    CSR,
    DENSE_OPERAND_BUDGET_BYTES,
    CharNGramHasher,
    DenseOperand,
    LinearBaseline,
)

DOCS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
    "fn main() { let x = 1; }\n",
    "fn other() { let y = 2; }\n",
    "class Thing:\n    pass\n",
    "package main\nfunc f() {}\n",
    "const x = 1;\nexport default x;\n",
    "SELECT * FROM t WHERE id = 1;\n",
]
LABELS = ["a", "b", "a", "b", "a", "b", "a", "b"]


@pytest.fixture
def small_csr():
    """A real hashed matrix, small enough that the budget admits it."""
    return CharNGramHasher(dim=2**8).transform(DOCS)


def test_the_dense_operand_is_chosen_when_it_fits(small_csr):
    assert isinstance(small_csr.as_operand(), DenseOperand)


def test_matmul_agrees_between_the_two_representations(small_csr):
    rng = np.random.default_rng(0)
    W = rng.normal(size=(small_csr.shape[1], 3))
    np.testing.assert_allclose(
        small_csr.as_operand().matmul(W), small_csr.matmul(W), rtol=0, atol=1e-12
    )


def test_rmatmul_agrees_between_the_two_representations(small_csr):
    rng = np.random.default_rng(1)
    D = rng.normal(size=(small_csr.shape[0], 3))
    np.testing.assert_allclose(
        small_csr.as_operand().rmatmul(D), small_csr.rmatmul(D), rtol=0, atol=1e-12
    )


def test_the_two_representations_agree_after_a_row_subset(small_csr):
    """`fit` carves the L2 validation slice with `select`, then densifies the result.

    `select` rebuilds `rows` and `indptr`; densifying a mis-rebuilt subset would still
    produce a plausible matrix, so the agreement is asserted after the subset too.
    """
    subset = small_csr.select(np.array([0, 2, 5, 7]))
    rng = np.random.default_rng(2)
    W = rng.normal(size=(subset.shape[1], 3))
    np.testing.assert_allclose(
        subset.as_operand().matmul(W), subset.matmul(W), rtol=0, atol=1e-12
    )


def test_the_dense_operand_reports_the_same_shape(small_csr):
    assert small_csr.as_operand().shape == small_csr.shape


def test_the_budget_refuses_a_matrix_that_would_not_fit():
    """Above the budget, `as_operand` returns the CSR itself and nothing changes.

    `dim=2**22` over eight docs is 8 x 4,194,304 x 8 bytes = 268 MB -- under the budget --
    so the refusal is asserted at a width that genuinely exceeds it rather than at a token
    large number that might not.
    """
    wide = CharNGramHasher(dim=2**24).transform(DOCS)
    assert wide.shape[0] * wide.shape[1] * 8 > DENSE_OPERAND_BUDGET_BYTES
    assert wide.as_operand() is wide


def test_a_duplicate_entry_is_refused_rather_than_silently_overwritten(small_csr):
    """Densifying by assignment is only equal to the sparse path on unique entries.

    `transform` cannot produce a duplicate, so this constructs one directly: the guard
    has to hold against a CSR built by some future path, not merely against this one.
    """
    duplicated = CSR(
        indptr=np.array([0, 2], dtype=np.int64),
        indices=np.array([5, 5], dtype=np.int64),
        data=np.array([1.0, 2.0], dtype=np.float64),
        rows=np.array([0, 0], dtype=np.int64),
        shape=(1, 16),
    )
    with pytest.raises(ValueError, match="duplicate"):
        duplicated.as_operand()


def test_a_duplicate_would_actually_have_changed_the_answer():
    """The guard is not decorative: without it the two paths disagree by 1.0 here."""
    duplicated = CSR(
        indptr=np.array([0, 2], dtype=np.int64),
        indices=np.array([5, 5], dtype=np.int64),
        data=np.array([1.0, 2.0], dtype=np.float64),
        rows=np.array([0, 0], dtype=np.int64),
        shape=(1, 16),
    )
    W = np.zeros((16, 1))
    W[5, 0] = 1.0
    # The sparse path sums both entries; assignment would have kept only the last.
    assert duplicated.matmul(W)[0, 0] == pytest.approx(3.0)


def test_the_fit_is_unchanged_by_densification(monkeypatch):
    """The same data, fitted both ways, produces the same weights.

    This is the property that matters: the speedup must not move the control's score.
    Forcing the budget to zero is what selects the sparse path, so both arms of the
    comparison run the same code with the same seed.
    """
    dense_fit = LinearBaseline(
        hasher=CharNGramHasher(dim=2**8), seed=0, max_iter=40
    ).fit(DOCS, LABELS)

    monkeypatch.setattr("qd_train.baseline.DENSE_OPERAND_BUDGET_BYTES", 0)
    sparse_fit = LinearBaseline(
        hasher=CharNGramHasher(dim=2**8), seed=0, max_iter=40
    ).fit(DOCS, LABELS)

    assert dense_fit.l2 == sparse_fit.l2
    assert dense_fit.iterations == sparse_fit.iterations
    assert dense_fit.converged == sparse_fit.converged
    np.testing.assert_allclose(dense_fit.weights, sparse_fit.weights, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(dense_fit.bias, sparse_fit.bias, rtol=1e-9, atol=1e-11)


def test_the_sparse_path_is_still_reachable(monkeypatch):
    """Guards the guard: if the budget stopped selecting sparse, the test above would be
    comparing the dense path against itself and would pass no matter what changed."""
    monkeypatch.setattr("qd_train.baseline.DENSE_OPERAND_BUDGET_BYTES", 0)
    csr = CharNGramHasher(dim=2**8).transform(DOCS)
    assert csr.as_operand() is csr
