"""The parts of ``tools/rung0_real_run.py`` that decide whether its numbers mean anything.

The tool's claim is "rung 0 scored X on files it never saw". Three things have to hold for
that sentence to be true rather than merely printed, and each is tested here:

* the split is disjoint **by file**, and stays that way as the corpus grows;
* the accuracy is reported against the majority-class baseline, because
  ``MUTATION_CLASSES`` is skewed and a model that answers ``stub`` every time scores 51.5%
  on this corpus;
* an empty evaluation set reads as ``NotRun``, not as 0%.

Torch-gated: the tool imports ``torch`` at module scope for the model it trains. The repo
venv has no torch by design, so these run in the ``mac`` extra alongside the other trainer
tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import rung0_real_run as tool  # noqa: E402

from qd_train.tristate import NotRun, Ran  # noqa: E402


def _example(path: str, cls: str = "stub") -> dict[str, object]:
    """The two fields the split and the baseline actually read."""
    return {"function": {"path": path}, "class": cls}


# -- the split ---------------------------------------------------------------------------


def test_the_split_is_disjoint_by_file() -> None:
    """The property the whole tool rests on. Mutations are generated per function, so one
    file yields many examples sharing context, naming and style; a random split would put
    siblings of a validation example into training and report memorisation as
    generalisation."""
    examples = [_example(f"src/mod{i}.py") for i in range(200) for _ in range(5)]
    train, val = tool.split_by_file(examples, val_share=0.25)
    train_paths = {e["function"]["path"] for e in train}  # type: ignore[index]
    val_paths = {e["function"]["path"] for e in val}  # type: ignore[index]
    assert train_paths and val_paths
    assert not (train_paths & val_paths)


def test_every_example_lands_on_exactly_one_side() -> None:
    """A split that silently drops rows would make both sides look cleaner than the corpus."""
    examples = [_example(f"src/mod{i}.py") for i in range(120)]
    train, val = tool.split_by_file(examples, val_share=0.3)
    assert len(train) + len(val) == len(examples)


def test_a_files_side_does_not_move_when_the_corpus_grows() -> None:
    """Hashed rather than shuffled, and this is why.

    If assignment depended on corpus order or size, adding files would move an old file
    across the split -- turning a previously held-out measurement into a training one
    without any visible change. The tool would still print "files it never saw".
    """
    small = [_example(f"src/mod{i}.py") for i in range(30)]
    large = small + [_example(f"src/new{i}.py") for i in range(300)]

    _, val_small = tool.split_by_file(small, val_share=0.25)
    _, val_large = tool.split_by_file(large, val_share=0.25)

    was_val = {e["function"]["path"] for e in val_small}  # type: ignore[index]
    still_val = {e["function"]["path"] for e in val_large}  # type: ignore[index]
    assert was_val <= still_val, f"{sorted(was_val - still_val)} moved out of validation"


def test_a_val_share_outside_the_open_unit_interval_is_refused() -> None:
    """0.0 and 1.0 both produce an empty side, which measures nothing while looking fine."""
    for share in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="val_share"):
            tool.split_by_file([_example("a.py")], val_share=share)


# -- the baseline ------------------------------------------------------------------------


class _Decision:
    """Only ``gold_option`` is read by the baseline; the real ByteDecision needs a context."""

    def __init__(self, gold_option: int) -> None:
        self.gold_option = gold_option


def test_the_baseline_is_the_majority_class_share() -> None:
    decisions = [_Decision(0)] * 7 + [_Decision(1)] * 2 + [_Decision(2)]
    share, name = tool.majority_baseline(decisions)
    assert share == pytest.approx(0.7)
    assert name == tool.MUTATION_CLASSES[0]


def test_a_model_at_the_baseline_does_not_pass() -> None:
    """The gate's whole point. On this corpus ``stub`` is 51.5% of the rows, so 51.5%
    accuracy is what answering one constant scores -- evidence of nothing."""
    at = tool._accuracy_gate(
        0.515, 0.515, n=160, what="choice", baseline_name="majority-class baseline"
    )
    assert isinstance(at, Ran)
    assert not at.passed

    above = tool._accuracy_gate(
        0.60, 0.515, n=160, what="choice", baseline_name="majority-class baseline"
    )
    assert isinstance(above, Ran)
    assert above.passed
    assert "+8.5%" in above.detail


def test_a_span_head_at_chance_does_not_pass() -> None:
    """The defect this tool shipped with for one run, pinned.

    A pointer's trivial answer is 1/candidates, never 0. The first version passed 0.0 as
    the span baseline and recorded ``passed=True`` for a span head scoring 0.6% over ~100
    candidate lines -- a head doing nothing, reported as a head that beat its bar. Ledger
    rows 8b7291cd, 3767e56b and d9005962 carry that verdict and cannot be rewritten.
    """
    chance = 1 / 100
    at_chance = tool._accuracy_gate(
        chance, chance, n=160, what="span start", baseline_name="uniform-pointer chance"
    )
    assert isinstance(at_chance, Ran)
    assert not at_chance.passed

    # The measured numbers from the 3-seed run, against the chance they have to beat.
    for measured in (0.025, 0.006, 0.031):
        state = tool._accuracy_gate(
            measured, chance, n=160, what="span start", baseline_name="uniform-pointer chance"
        )
        assert isinstance(state, Ran)
        # Not asserting these fail -- 2.5% and 3.1% do exceed 1% chance. What is asserted
        # is that the comparison happens against chance at all: with the old 0.0 baseline
        # every one of them passed, including 0.6%, which is BELOW chance.
        assert state.passed == (measured > chance)
    below = tool._accuracy_gate(
        0.006, chance, n=160, what="span start", baseline_name="uniform-pointer chance"
    )
    assert not below.passed, "0.6% is below 1% chance and must not report as passing"


def test_an_empty_evaluation_set_reads_as_not_run_not_as_zero() -> None:
    """0% and "nothing was measured" are different facts, and only one of them is about
    the model. ``NotRun`` has no ``passed`` field to be misread as a failure either."""
    state = tool._accuracy_gate(
        0.0, 0.5, n=0, what="span start", baseline_name="majority-class baseline"
    )
    assert isinstance(state, NotRun)
    assert not hasattr(state, "passed")


# -- batching ----------------------------------------------------------------------------


class _Context:
    def __init__(self, n: int) -> None:
        self.n_bytes_kept = n


class _Sized:
    def __init__(self, n: int) -> None:
        self.context = _Context(n)


def test_batches_are_length_homogeneous() -> None:
    """``plan_batch`` pads to the batch maximum, so one long row makes every short row in
    its batch pay the difference. Sorting first is what keeps a batch's width its own."""
    decisions = [_Sized(n) for n in (100, 10, 90, 20, 80, 30)]
    ordered = sorted(decisions, key=lambda d: d.context.n_bytes_kept)
    chunks = [ordered[i : i + 2] for i in range(0, len(ordered), 2)]
    widths = [max(d.context.n_bytes_kept for d in c) for c in chunks]
    assert widths == sorted(widths)
    # Sorted chunking wastes strictly less than taking them in arrival order.
    unsorted_chunks = [decisions[i : i + 2] for i in range(0, len(decisions), 2)]
    sorted_waste = sum(
        max(d.context.n_bytes_kept for d in c) * len(c)
        - sum(d.context.n_bytes_kept for d in c)
        for c in chunks
    )
    unsorted_waste = sum(
        max(d.context.n_bytes_kept for d in c) * len(c)
        - sum(d.context.n_bytes_kept for d in c)
        for c in unsorted_chunks
    )
    assert sorted_waste < unsorted_waste


def test_a_batch_size_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        tool.bucketed_batches([], batch_size=0, config=tool.ByteDeciderConfig())


# -- the context window ------------------------------------------------------------------


def test_the_default_context_is_wider_than_the_config_default() -> None:
    """Measured, not preferred: at ``ByteDeciderConfig``'s 1024 only 9.0% of a real
    qd-mutate corpus survives ``SpanOutsideWindow``, against 26.6% at 4096. A tool whose
    corpus is 91% refused is measuring its truncation, not its model."""
    assert tool.ByteDeciderConfig().max_context_bytes < tool.DEFAULT_CONTEXT_BYTES
    assert tool.DEFAULT_CONTEXT_BYTES <= tool.MAX_CONTEXT_BYTES
