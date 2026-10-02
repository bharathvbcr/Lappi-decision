"""The length-only control: can the context's size alone predict the label?

``GAP-A3-CLEAN-DIFFS-ARE-LONGER-THAN-MUTATION-DIFFS``: ``code.defect_class``'s clean rows are
the real commit's diff and the mutated rows are one synthetic edit, so clean diffs are longer
(p50 739 bytes against 286-410). A model can then score above the majority rate by reading
the diff's size and nothing else. The go/no-go has to show the 2B beats a control that sees
ONLY that -- byte length and line count of the context -- as well as the char-n-gram one.
The length control reuses ``LinearBaseline``'s fitting (L2 grid, convergence check) with a
different featurizer, and ``tools/ft_linear_control.py`` reports it beside the n-gram gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import ft_linear_control as ftc  # noqa: E402

from qd_data.defect_class import DEFECT_CLASSES  # noqa: E402
from qd_data.schema import NOUL, NOUL_LETTER, OPTION_LETTERS  # noqa: E402
from qd_train.baseline import (  # noqa: E402
    ContextLengthFeatures,
    LinearBaseline,
    RequestDoc,
)
from qd_train.tristate import NotRun, Ran  # noqa: E402

TASK = "code.defect_class/defect_class"


def _context(i: int, *, lines: int) -> str:
    # Same line content for every class: only the count differs.
    return "".join(f"+    x{(i + k) % 7} = y\n" for k in range(lines))


#: What every code.defect_class row offers, in letter order: one option set, so the control
#: is labelled by value (``qd_train.baseline.control_label_space``).
OFFERED = (*zip(OPTION_LETTERS, DEFECT_CLASSES, strict=False), (NOUL_LETTER, NOUL))


def _doc(i: int, value: str, *, context: str) -> RequestDoc:
    # The prompt text says nothing about the class, so the n-gram control cannot read it.
    return RequestDoc(
        row_id=f"r{i:04d}", slot_name="defect_class", kind="choice", task=TASK,
        text=f"<|qd_context|>\n{context}<|qd_context_end|> which class?", value=value,
        letter=next(k for k, v in OFFERED if v == value), offered=OFFERED, context=context,
    )


def _split() -> tuple[list[RequestDoc], list[RequestDoc]]:
    docs = [
        _doc(i, "clean" if i % 2 else "stub", context=_context(i, lines=24 if i % 2 else 3))
        for i in range(80)
    ]
    return docs[:60], docs[60:]


def test_the_features_are_the_context_size_and_nothing_else() -> None:
    feats = ContextLengthFeatures()
    a = feats.transform(["ab\ncd\n", "zz\nqq\n"])
    b = feats.transform(["ab\ncd\n", "ab\ncd\nef\n"])
    # Two contexts of equal byte length and line count are one point, whatever they say.
    assert a.data[: feats.dim].tolist() == a.data[feats.dim :].tolist()
    assert b.data[: feats.dim].tolist() != b.data[feats.dim :].tolist()
    assert a.shape == (2, feats.dim)


def test_a_length_only_label_is_learned_by_the_length_control() -> None:
    train, val = _split()
    model = LinearBaseline(hasher=ContextLengthFeatures(), seed=0, max_iter=6_000)
    model.fit([d.context for d in train], [d.value for d in train])
    conv = model.convergence()
    assert isinstance(conv, Ran) and conv.passed, conv
    predicted = model.predict([d.context for d in val])
    assert predicted == [d.value for d in val]


def test_the_tool_reports_the_length_control_beside_the_ngram_gate(qd_prep: Path) -> None:
    train, val = _split()
    keys = [(d.row_id, d.slot_name) for d in val]
    verdicts = ftc.Verdicts(
        eval_row_id="e", seed=0, correct=dict.fromkeys(keys, True),
        kind_of=dict.fromkeys(keys, "choice"), by_slot_name=True, span_rows=0,
    )
    result = ftc.score_against_control(train, val, verdicts, seed=0)
    length_acc = result.metrics[f"length_control_top1.{TASK}"]
    assert isinstance(length_acc, Ran) and length_acc.value == 1.0
    assert isinstance(result.metrics[f"length_control_convergence.{TASK}"], Ran)
    margin = result.metrics["paired_margin_vs_length_control"]
    # The model is right everywhere and so is the length control: a zero margin.
    assert isinstance(margin, Ran) and margin.value == pytest.approx(0.0)
    # Reported separately: the n-gram arm's gate and metrics are still its own.
    assert f"linear_control_top1.{TASK}" in result.metrics


def test_docs_without_a_context_leave_the_length_control_not_run(qd_prep: Path) -> None:
    """A RequestDoc built without its context (every caller before this field) must not be
    scored as a zero-length context: that control would be constant and easy to beat."""
    train, val = _split()

    def bare(docs: list[RequestDoc]) -> list[RequestDoc]:
        return [
            RequestDoc(row_id=d.row_id, slot_name=d.slot_name, kind=d.kind, task=d.task,
                       text=d.text, value=d.value, letter=d.letter, offered=d.offered)
            for d in docs
        ]

    bare_train, bare_val = bare(train), bare(val)
    keys = [(d.row_id, d.slot_name) for d in bare_val]
    verdicts = ftc.Verdicts(
        eval_row_id="e", seed=0, correct=dict.fromkeys(keys, True),
        kind_of=dict.fromkeys(keys, "choice"), by_slot_name=True, span_rows=0,
    )
    result = ftc.score_against_control(bare_train, bare_val, verdicts, seed=0)
    assert isinstance(result.metrics["paired_margin_vs_length_control"], NotRun)
    assert "no context" in result.metrics["paired_margin_vs_length_control"].reason
