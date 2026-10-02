"""The linear control's step schedule, against the fit it was made for (2026-10-02).

F seed 0's intent.domain n-gram control (ledger row c89b89a1) ended UNCONVERGED at its
6,000-iteration budget, grad norm 1.118e-2 against a tolerance of 1e-4, and the paired-margin
gate it opposes was NOT RUN. The budget was not what ran out: Adam at a constant step settled
into a limit cycle from about iteration 1,000 (HANDOFF/prep-containment-2026-10-02.md records
the curve), so no budget at that step converges. ``qd_train.baseline.step_size`` and
``crates/qd-prep/src/linfit.rs``'s ``step_size`` keep the step for the first
``LR_CONSTANT_ITERS`` iterations -- every fit that converged before is unchanged bit for bit
-- and halve it every ``LR_HALVING_PERIOD`` after.

The fixture (``crates/qd-prep/tests/fixtures/linfit-intent-domain-300.jsonl``) is 300 of that
control's own train documents, on which a constant step never meets the tolerance either.
Both tests below fail against the pre-fix engines: the constant-step fit ends unconverged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import ft_linear_control as ftc  # noqa: E402
import linear_control_native as native  # noqa: E402

from qd_train.baseline import (  # noqa: E402
    LR_CONSTANT_ITERS,
    LR_HALVING_PERIOD,
    CharNGramHasher,
    LinearBaseline,
    step_size,
)

FIXTURE = REPO / "crates" / "qd-prep" / "tests" / "fixtures" / "linfit-intent-domain-300.jsonl"
#: The L2 the control selected on the full task (c89b89a1's refit), and the only one fitted here.
L2 = 1e-4


def _fixture() -> tuple[list[str], list[str]]:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    rows = [json.loads(line) for line in lines[1:]]
    assert header["rows"] == len(rows) == 300
    assert header["source_task"] == "intent.domain/domain"
    return [r["text"] for r in rows], [r["label"] for r in rows]


def _model(*, dense_budget_bytes: int | None = None) -> LinearBaseline:
    return LinearBaseline(hasher=CharNGramHasher(), seed=0, max_iter=ftc.DEFAULT_MAX_ITER,
                          l2_grid=(L2,), dense_budget_bytes=dense_budget_bytes)


def test_the_step_is_constant_through_the_old_budget_then_halves_every_period() -> None:
    lr = 0.05
    assert [step_size(lr, t) for t in (1, 2, 5_999, 6_000)] == [lr] * 4
    assert step_size(lr, 6_001) == step_size(lr, 6_500) == lr / 2
    assert step_size(lr, 6_501) == step_size(lr, 7_000) == lr / 4
    assert step_size(lr, 12_000) == lr / 4096
    assert step_size(1.0, LR_CONSTANT_ITERS + LR_HALVING_PERIOD * 1_074) == 5e-324
    assert step_size(1.0, LR_CONSTANT_ITERS + LR_HALVING_PERIOD * 1_075) == 0.0


def test_the_budget_leaves_room_past_the_constant_step() -> None:
    """A budget at or under ``LR_CONSTANT_ITERS`` would never reach the first halving."""
    assert ftc.DEFAULT_MAX_ITER >= LR_CONSTANT_ITERS + 4 * LR_HALVING_PERIOD


@pytest.mark.slow
def test_the_intent_domain_limit_cycle_converges_in_the_native_engine(
    qd_prep_bin: Path,
) -> None:
    """The control's whole fit -- the one-value grid on its carve, then the refit -- by
    ``qd-prep linfit``. About 90 s: two fits of ~6,000 iterations at 65,536 x 11 weights."""
    docs, labels = _fixture()
    got = native.fit(qd_prep_bin, _model(), docs, labels, docs[:4])
    assert got.fit.l2 == L2
    assert got.fit.converged, (
        f"unconverged in {got.fit.iterations} iterations, grad norm {got.fit.final_grad_norm:.3e}"
    )
    assert LR_CONSTANT_ITERS < got.fit.iterations <= ftc.DEFAULT_MAX_ITER
    assert got.fit.final_grad_norm < 1e-4


@pytest.mark.slow
def test_the_intent_domain_limit_cycle_converges_in_the_reference() -> None:
    """The oracle's own refit on the fixture, on its dense operand (the quicker of its two;
    the sparse one, which the native engine mirrors bit for bit, is crossed past the halving
    in test_qd_prep_linear_parity). About two minutes."""
    docs, labels = _fixture()
    reference = _model()
    classes = tuple(sorted(set(labels)))
    y = np.array([classes.index(label) for label in labels], dtype=np.int64)
    _w, _b, converged, iterations, grad_norm, history = reference._train_once(
        reference.hasher.transform(docs), y, len(classes), L2
    )
    assert converged, f"unconverged in {iterations} iterations, grad norm {grad_norm:.3e}"
    assert LR_CONSTANT_ITERS < iterations <= ftc.DEFAULT_MAX_ITER
    assert len(history) == iterations
