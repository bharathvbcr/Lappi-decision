"""The control that existed and had never once been evaluated.

``qd_train.eval_harness.shuffled_label_control`` has been in this repository the whole
time. Across **988 ledger rows** -- every chain file, every lane, CPU, MPS and GH200 --
``controls["shuffled_label"]`` reads ``not_run`` on all of them, and so do
``degenerate_head``, ``privileged_hunk`` and ``transfer_gate``. The reason is one line in
the harness:

    controls["shuffled_label"] = NotRun(reason="no shuffled-label model was evaluated")

The check was never broken. It needs the held-out accuracy of a model trained on destroyed
labels, and nothing in this repository ever trained one. A control that cannot be reached
is indistinguishable, in every row it appears in, from a control that was reached and
passed -- which is this project's founding defect, sitting inside the machinery built to
detect it.

``tools/rung0_real_run.py --shuffle-train-labels`` is that model. These tests pin the
permutation it performs, because a shuffle that quietly failed to destroy the signal would
make the control PASS for the wrong reason -- and a passing control is exactly the result
nobody re-examines.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

from qd_train.eval_harness import shuffled_label_control  # noqa: E402
from qd_train.tristate import Ran  # noqa: E402


@dataclass(frozen=True, slots=True)
class FakeDecision:
    """Enough of `ByteDecision` for the permutation: the label and its option count.

    Frozen, because the real one is -- a shuffle that mutated in place would pass against a
    mutable stand-in and raise on the real type at the top of a GPU run.
    """

    example_id: str
    options: tuple[bytes, ...]
    gold_option: int


def _decisions(labels: list[int], *, arity: int = 4) -> list[FakeDecision]:
    options = tuple(f"opt{i}".encode() for i in range(arity))
    return [FakeDecision(f"e{i}", options, label) for i, label in enumerate(labels)]


def _shuffle():
    """Imported lazily: `rung0_real_run` imports torch, which the repo venv lacks."""
    pytest.importorskip("torch", reason="rung0_real_run imports torch at module scope")
    from rung0_real_run import shuffle_train_labels

    return shuffle_train_labels


# --------------------------------------------------------------------------
# 1 -- the permutation
# --------------------------------------------------------------------------


def test_the_shuffle_preserves_the_label_distribution_exactly():
    """A permutation, not fresh random labels, and the control depends on the difference.

    `shuffled_label_control` compares the shuffled model against the MAJORITY-CLASS rate.
    Drawing fresh labels would move the marginal and move that ceiling with it, so the
    control would be measured against a bar that does not apply to the real arms.
    """
    shuffle = _shuffle()
    labels = [0, 0, 0, 1, 1, 2, 3, 3, 3, 3]
    out = shuffle(_decisions(labels), seed=0)
    assert sorted(d.gold_option for d in out) == sorted(labels), (
        "the shuffle changed the label multiset, so it is not a permutation and the "
        "majority rate the control compares against no longer describes these labels"
    )


def test_the_shuffle_actually_moves_labels():
    """The one failure that would make the control pass for the wrong reason.

    A shuffle that returned its input would leave the signal intact, the 'control' model
    would score like the real one, and -- because the control asks for accuracy at or BELOW
    chance -- it would report LEAKAGE rather than silently passing. That is the safer
    direction, but it is still a lie about what was measured, so it is pinned.
    """
    shuffle = _shuffle()
    labels = [i % 4 for i in range(200)]
    out = shuffle(_decisions(labels), seed=0)
    moved = sum(1 for before, after in zip(labels, out, strict=True) if before != after.gold_option)
    assert moved > 100, (
        f"only {moved} of 200 labels moved; a permutation that leaves most labels in place "
        "has not destroyed the signal the control exists to destroy"
    )


def test_the_shuffle_never_produces_an_invalid_gold_option():
    """`ByteDecision.__post_init__` refuses a gold_option outside its own options.

    Decisions with different option counts must be permuted within their own group. A
    global permutation raises partway through a run -- and only on a corpus where the
    counts differ, which is not today's, so the failure would arrive later and elsewhere.
    """
    shuffle = _shuffle()
    mixed = _decisions([0, 1, 2, 3], arity=4) + _decisions([0, 1], arity=2)
    out = shuffle(mixed, seed=1)
    for decision in out:
        assert 0 <= decision.gold_option < len(decision.options), (
            f"{decision.example_id}: gold_option {decision.gold_option} is outside its "
            f"{len(decision.options)} options -- the real ByteDecision would have raised"
        )


def test_the_shuffle_does_not_mutate_its_input():
    shuffle = _shuffle()
    original = _decisions([0, 1, 2, 3] * 10)
    before = [d.gold_option for d in original]
    shuffle(original, seed=0)
    assert [d.gold_option for d in original] == before, "the shuffle mutated its input"


def test_the_same_seed_gives_the_same_permutation():
    """So a control run is reproducible, and so --shuffle-seed means something."""
    shuffle = _shuffle()
    labels = [i % 4 for i in range(50)]
    a = shuffle(_decisions(labels), seed=7)
    b = shuffle(_decisions(labels), seed=7)
    c = shuffle(_decisions(labels), seed=8)
    assert [d.gold_option for d in a] == [d.gold_option for d in b]
    assert [d.gold_option for d in a] != [d.gold_option for d in c], (
        "two different --shuffle-seed values produced the same permutation"
    )


# --------------------------------------------------------------------------
# 2 -- the runner reaches the control at all
# --------------------------------------------------------------------------


def test_the_runner_records_the_control_only_under_the_flag():
    """Both halves matter, and the second is the one that keeps the ledger honest.

    Under the flag the row must carry a real `shuffled_label`. WITHOUT it the row must keep
    saying `not_run`, because a run trained on true labels has not evaluated this control
    and a row that claimed otherwise would be the defect this file is about, inverted.
    """
    source = (REPO / "tools" / "rung0_real_run.py").read_text(encoding="utf-8")
    assert 'recorder.control(\n                    "shuffled_label"' in source, (
        "rung0_real_run.py never calls recorder.control('shuffled_label'), so every row it "
        "writes reports the control as not_run -- which is the state all 988 existing rows "
        "are in"
    )
    marker = source.index('recorder.control(\n                    "shuffled_label"')
    preceding = source[:marker]
    assert "if args.shuffle_train_labels:" in preceding[-400:], (
        "the control is recorded unconditionally; a run on TRUE labels would then claim to "
        "have evaluated a shuffled-label control it never ran"
    )


def test_the_control_and_the_arm_it_controls_for_cannot_share_a_recipe_hash():
    """Everything else about them is identical by design, so the recipe must separate them.

    Without this a control run and a real run hash alike, and anything pooling by protocol
    averages a model trained on destroyed labels into the measurement it exists to check.
    """
    source = (REPO / "tools" / "rung0_real_run.py").read_text(encoding="utf-8")
    recipe_start = source.index("    recipe = {")
    recipe_block = source[recipe_start : recipe_start + 2500]
    assert '"shuffle_train_labels"' in recipe_block, (
        "the recipe does not record whether labels were shuffled, so the control and the "
        "arm it controls for produce the same recipe_hash"
    )


# --------------------------------------------------------------------------
# 3 -- the harness end, on the numbers a run will actually hand it
# --------------------------------------------------------------------------


def test_a_shuffled_model_at_chance_passes_and_above_the_ceiling_fails():
    """The control's own verdict, on the two cases that matter.

    Chance is the majority rate, 60% here, not 1/k = 25%. A model scoring 58% has learned
    nothing beyond the prior; one scoring 85% on labels that were destroyed in training can
    only have read the answer off the split.
    """
    labels = [0] * 60 + [1] * 20 + [2] * 15 + [3] * 5

    at_chance = shuffled_label_control(0.58, labels, n_eval=len(labels))
    assert isinstance(at_chance, Ran) and at_chance.passed, at_chance.detail

    leaking = shuffled_label_control(0.85, labels, n_eval=len(labels))
    assert isinstance(leaking, Ran) and not leaking.passed
    assert "LEAKAGE" in leaking.detail, leaking.detail


if __name__ == "__main__":  # pragma: no cover - convenience only
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
