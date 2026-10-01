"""The FT linear control's label space: one per task, and one every row of the task shares.

The control is multinomial logistic regression over a fixed class set. ``request_texts`` used
to label every doc by its gold VALUE -- right for a task that offers the same option set on
every row (``code.defect_class``'s five classes, yes/no, a score's bins, CLINC's ten domains),
because the letter is a per-row shuffle artefact there. It is wrong wherever the options are
the row's own: CommonsenseQA and MMLU answer with the text of one of THIS question's options,
so the values are near-unique across rows (measured on the 2026-09-29 general record: k=4,956
on 9,619 CSQA train rows, k=12,080 on 14,200 MMLU rows), and CLINC's sampled-intent tasks are
split by intent, so no val gold is a training class at all (0/1,500). A control over those
classes cannot answer the question it is scored on: J1's CSQA control picked its L2 at 4/1635
on a five-way task (GH200, 2026-10-01). For a task whose rows offer different option sets the
letter is the one label every row shares, and the model answers in letters.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import pytest
from data_fixtures import INTENT_VOCABULARY, clinc_row, commitpackft_row

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import ft_linear_control as ftc  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.general import CSQA_FAMILY, ClincDomainMap  # noqa: E402
from qd_data.loaders import CsqaRow, MmluRow  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.schema import NOUL_LETTER, OPTION_LETTERS  # noqa: E402
from qd_train.baseline import (  # noqa: E402
    LinearBaseline,
    RequestDoc,
    control_label,
    control_label_space,
    request_texts,
)
from qd_train.tristate import NotRun, Ran  # noqa: E402

CSQA_TASK = f"{CSQA_FAMILY}/answer"
#: Every label a letter-space control can see: the slot's letters and the noul letter.
LETTER_ALPHABET = frozenset(OPTION_LETTERS) | {NOUL_LETTER}


def _csqa_raw(i: int) -> CsqaRow:
    """A CommonsenseQA-shaped question: five options that are this question's own text.

    Every option names the row, so no answer text recurs across rows -- as in the real set,
    where 9,619 training questions have 4,956 distinct answers. The gold is always the
    ``shelf`` option, so a control that can see which LETTER sits beside ``shelf`` learns it;
    one that has to name the gold TEXT has never seen it before.
    """
    texts = (f"drawer {i}", f"shelf {i}", f"garage {i}", f"attic {i}", f"car {i}")
    return CsqaRow(
        qid=f"c{i:04d}", question=f"Where would you most likely keep item number {i}?",
        concept=f"k{i % 7}", labels=("A", "B", "C", "D", "E"), texts=texts, answer_key="B",
        upstream_split="train",
    )


def _mmlu_raw(i: int) -> MmluRow:
    return MmluRow(
        subject=f"s{i % 5}", question=f"What follows {i}?",
        choices=(str(i + 1), str(i + 2), str(i + 3), str(i + 4)), answer_index=0,
        upstream_split="test",
    )


def _csqa_docs(n_train: int, n_val: int) -> tuple[list[RequestDoc], list[RequestDoc]]:
    config = DataConfig()
    mixture = build_mixture(
        {"tau/commonsense_qa": [_csqa_raw(i) for i in range(n_train + n_val)]}, config=config
    )
    assert mixture.status.passed is True, mixture.status
    docs, excluded = request_texts(mixture.rows, seed=config.seed)
    assert excluded == []
    docs = sorted(docs, key=lambda d: d.row_id)
    assert {d.task for d in docs} == {CSQA_TASK}
    return docs[:n_train], docs[n_train:]


def _fit(
    task: str, train: list[RequestDoc], val: list[RequestDoc], engine: Path
) -> ftc.TaskControl:
    return ftc.fit_task(
        task, train, val, by_slot_name=True, seed=0, max_iter=ftc.DEFAULT_MAX_ITER,
        dense_budget_bytes=0, max_fit_minutes=None, cache_dir=None, engine=engine,
    )


# --- the defect: a control that has to name a never-seen answer text ---------------------


def test_a_csqa_shaped_control_beats_chance(qd_prep: Path) -> None:
    """Pre-fix this scored 0/50: 120 training rows were 120 classes, and no val gold was one.

    Chance is 1/5. The gold's letter is learnable from the prompt (the 5-gram ``"B. sh"``
    binds a letter to the gold's text), so a control in the right label space clears chance
    by a wide margin; one in the wrong space cannot answer any val row at all.
    """
    train, val = _csqa_docs(120, 50)
    got = _fit(CSQA_TASK, train, val, qd_prep)
    assert isinstance(got.convergence, Ran), got.convergence
    assert isinstance(got.accuracy, Ran), got.accuracy
    assert got.accuracy.n_total == 50
    assert got.accuracy.value > 0.5, got.accuracy


class _Stop(Exception):
    """Raised by the stand-in engine once it has seen what the control was handed."""


def _labels_handed_to_the_engine(
    monkeypatch: pytest.MonkeyPatch, task: str, train: list[RequestDoc], val: list[RequestDoc]
) -> list[str]:
    seen: list[list[str]] = []

    def capture(
        _engine: Path, _model: LinearBaseline, _docs: Sequence[str], labels: Sequence[str],
        _val_docs: Sequence[str], **_kw: object,
    ) -> NoReturn:
        seen.append(list(labels))
        raise _Stop

    monkeypatch.setattr(ftc.native, "fit", capture)
    with pytest.raises(_Stop):
        ftc.fit_task(
            task, train, val, by_slot_name=True, seed=0, max_iter=ftc.DEFAULT_MAX_ITER,
            dense_budget_bytes=0, max_fit_minutes=None, cache_dir=None, engine=Path("/x"),
        )
    assert len(seen) == 1
    return seen[0]


def _general_docs() -> list[RequestDoc]:
    vocab = sorted(set(INTENT_VOCABULARY))
    q = len(vocab) // 4
    domains = ClincDomainMap(domains={f"d{k}": tuple(vocab[k * q:(k + 1) * q]) for k in range(4)})
    config = DataConfig()
    mixture = build_mixture(
        {
            "tau/commonsense_qa": [_csqa_raw(i) for i in range(40)],
            "cais/mmlu": [_mmlu_raw(i) for i in range(40)],
            "clinc/clinc_oos": [clinc_row(i) for i in range(60)],
            "bigcode/commitpackft": [commitpackft_row(i) for i in range(40)],
        },
        config=config, clinc_domain_map=domains,
    )
    assert mixture.status.passed is True, mixture.status
    docs, excluded = request_texts(mixture.rows, seed=config.seed)
    assert excluded == []
    return docs


def test_no_task_hands_the_engine_more_classes_than_a_row_can_offer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model answers one of at most 16 letters plus noul on every row, so a control with
    more classes than that is predicting something no row asks. Pre-fix the CSQA and MMLU
    tasks here handed the engine one class per row (40 each)."""
    docs = _general_docs()
    tasks = sorted({d.task for d in docs})
    assert {CSQA_TASK, "knowledge.multiple_choice/answer", "intent.classification/intent",
            "intent.within_domain/intent", "intent.domain/domain"} <= set(tasks)
    for task in tasks:
        mine = [d for d in docs if d.task == task]
        if len({d.value for d in mine}) < 2:
            continue  # a single-class task never reaches the engine
        labels = _labels_handed_to_the_engine(monkeypatch, task, mine, mine[:4])
        assert len(set(labels)) <= len(LETTER_ALPHABET), (task, len(set(labels)))


# --- the rule ---------------------------------------------------------------------------


def test_one_option_set_is_labelled_by_value_and_per_row_options_by_letter() -> None:
    docs = _general_docs()
    by_task: dict[str, list[RequestDoc]] = {}
    for d in docs:
        by_task.setdefault(d.task, []).append(d)
    want = {
        # The row's own options: the value is not a class any other row has.
        CSQA_TASK: "letter",
        "knowledge.multiple_choice/answer": "letter",
        # 16 intents sampled per row, and the split is by intent.
        "intent.classification/intent": "letter",
        # A domain's own intents: ten different option sets.
        "intent.within_domain/intent": "letter",
        # One option set on every row.
        "intent.domain/domain": "value",
        "intent.in_scope/in_scope": "value",
        "code.language_id/language": "value",
    }
    for task, space in want.items():
        assert task in by_task, task
        rows = by_task[task]
        assert control_label_space(rows, []) == space, task
        for d in rows:
            label = control_label(d, space)
            if space == "value":
                assert label == d.value
            else:
                assert label == d.letter and label in LETTER_ALPHABET
                # The letter is THIS rendering's: it names the gold value on this row.
                assert dict(d.offered)[label] == d.value


def test_a_fixed_option_set_task_hands_the_engine_its_values_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``code.defect_class``, yes/no and score tasks keep the labels they were fitted on."""
    docs = [d for d in _general_docs() if d.task == "code.language_id/language"]
    labels = _labels_handed_to_the_engine(monkeypatch, docs[0].task, docs, docs[:4])
    assert labels == [d.value for d in docs]


def test_a_val_split_that_asks_another_question_is_refused() -> None:
    """Train offers one option set and val another: the control's classes would not include
    an option val offers. That is a broken split, not a label space; it is not guessed at."""
    docs = [d for d in _general_docs() if d.task == "intent.within_domain/intent"]
    by_set: dict[frozenset[str], list[RequestDoc]] = {}
    for d in docs:
        by_set.setdefault(frozenset(v for _, v in d.offered), []).append(d)
    assert len(by_set) >= 2
    one, other = list(by_set.values())[:2]
    with pytest.raises(ValueError, match="option set"):
        control_label_space(one, other)


def test_the_refused_split_makes_the_task_not_run(qd_prep: Path) -> None:
    docs = [d for d in _general_docs() if d.task == "intent.within_domain/intent"]
    by_set: dict[frozenset[str], list[RequestDoc]] = {}
    for d in docs:
        by_set.setdefault(frozenset(v for _, v in d.offered), []).append(d)
    one, other = list(by_set.values())[:2]
    got = _fit("intent.within_domain/intent", one * 3, other, qd_prep)
    assert isinstance(got.convergence, NotRun)
    assert "option set" in got.convergence.reason


def test_docs_without_their_rendering_are_refused() -> None:
    bare = RequestDoc(
        row_id="r", slot_name="s", kind="choice", task="t/s", text="x", value="v",
        letter="", offered=(),
    )
    with pytest.raises(ValueError, match="rendering"):
        control_label_space([bare], [])


def test_docs_of_two_tasks_are_refused() -> None:
    docs = _general_docs()
    a = next(d for d in docs if d.task == CSQA_TASK)
    b = next(d for d in docs if d.task == "intent.domain/domain")
    with pytest.raises(ValueError, match="one task"):
        control_label_space([a], [b])


# --- parity: the native engine against the reference, in letter space --------------------


def test_a_letter_space_control_answers_row_for_row_as_the_python_engine_does(
    qd_prep: Path,
) -> None:
    """The perf lane's parity definition, on a per-row-option task: identical convergence,
    iterations and selected L2 (one ``detail`` string carries all three) and identical
    per-row verdicts. The reference is fitted on the same labels by the same rule."""
    train, val = _csqa_docs(120, 50)
    got = _fit(CSQA_TASK, train, val, qd_prep)
    space = control_label_space(train, val)
    assert space == "letter"
    reference = LinearBaseline(seed=0, max_iter=ftc.DEFAULT_MAX_ITER, dense_budget_bytes=0)
    reference.fit([d.text for d in train], [control_label(d, space) for d in train])
    assert isinstance(got.convergence, Ran)
    assert got.convergence.detail == reference.convergence().detail  # type: ignore[union-attr]
    predicted = reference.predict([d.text for d in val])
    want = {ftc.doc_key(d, by_slot_name=True): p == control_label(d, space)
            for d, p in zip(val, predicted, strict=True)}
    assert got.correct == want
