"""A4: ``paired_margin_vs_linear`` for FT eval rows, via ``tools/ft_linear_control.py``.

Before this, every FT eval row carried the gate as ``not_run`` with no tool able to change
that (``ledger/gh200-ft-commitpackft-2026-09-22.jsonl``). These tests pin the pieces that
make a post-hoc verdict trustworthy rather than merely present: the control reads the
model's own prompt, the two arms are paired by key over an identical population, a
verdicts file that belongs to another row is refused, and the operator-holdout arm refuses
to be a silent no-op.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
#: A full sha: the tool writes a ledger row, so it refuses a revision named by HEAD or a branch.
REV = "0632f693d3b765b726499e7b4bf19c67959b75cb"
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ft_linear_control as ftc  # noqa: E402
from data_fixtures import commitpackft_row, squad_row  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.render import DEFAULT_CAPS, render  # noqa: E402
from qd_train.baseline import RequestDoc, request_texts  # noqa: E402
from qd_train.eval_harness import paired_margin_by_key, paired_margin_test  # noqa: E402
from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402


def _rows(n: int = 40):
    config = DataConfig()
    mixture = build_mixture(
        {
            "bigcode/commitpackft": [commitpackft_row(i) for i in range(n)],
            "rajpurkar/squad_v2": [squad_row(i) for i in range(6)],
        },
        config=config,
    )
    return list(mixture.rows), config


# --- request_texts --------------------------------------------------------------------


def test_request_texts_is_the_prompt_the_model_reads_labelled_by_value() -> None:
    rows, config = _rows()
    docs, excluded = request_texts(rows, seed=config.seed)
    assert excluded == []
    assert docs, "fixture rows produced no letter slots"
    by_id = {r.row_id: r for r in rows}
    for d in docs:
        row = by_id[d.row_id]
        rendered = render(row.request, caps=DEFAULT_CAPS, seed=config.seed)
        assert d.text == rendered.prompt_for(d.slot_name)
        gold = next(g for g in row.gold if g.slot_name == d.slot_name)
        assert d.value == str(gold.value)
        assert d.kind in ("choice", "score")
        # The rendering's own options and gold letter, which the letter-space control reads.
        assert d.offered == tuple(rendered.slot(d.slot_name).letter_to_value.items())
        assert dict(d.offered)[d.letter] == d.value
    # Span slots have no letter answer and are never scored by the control.
    assert not any(d.task.startswith("qa.answer_span") for d in docs)


def test_request_texts_depends_on_the_render_seed() -> None:
    """Rendered at another seed the choice options move, so the text is not the model's."""
    rows, config = _rows()
    a, _ = request_texts(rows, seed=config.seed)
    b, _ = request_texts(rows, seed=config.seed + 1)
    choice_a = [d.text for d in a if d.kind == "choice"]
    choice_b = [d.text for d in b if d.kind == "choice"]
    assert choice_a != choice_b


# --- paired_margin_by_key -------------------------------------------------------------


def test_pairing_by_key_ignores_order_and_matches_positional() -> None:
    rng = np.random.default_rng(3)
    m = rng.random(200) > 0.4
    b = rng.random(200) > 0.5
    keys = [f"r{i}" for i in range(200)]
    model = dict(zip(keys, m.tolist(), strict=True))
    control = dict(zip(reversed(keys), reversed(b.tolist()), strict=True))
    got = paired_margin_by_key(model, control, seed=0)
    order = sorted(keys, key=repr)
    want = paired_margin_test(
        np.array([model[k] for k in order]), np.array([control[k] for k in order]), seed=0
    )
    assert isinstance(got, Ran) and isinstance(want, Ran)
    assert got.value == pytest.approx(want.value)
    assert got.detail == want.detail


def test_pairing_refuses_different_populations() -> None:
    got = paired_margin_by_key({"a": True, "b": False}, {"a": True, "c": True})
    assert isinstance(got, NotRun)
    assert "different rows" in got.reason


# --- verdicts -------------------------------------------------------------------------


def _write_verdicts(path: Path, lines: list[dict]) -> Path:
    path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    return path


def test_missing_verdicts_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ftc.Refused, match="no verdicts file"):
        ftc.load_verdicts(tmp_path / "absent.jsonl")


@pytest.mark.parametrize(
    ("lines", "match"),
    [
        (
            [{"eval_row_id": "e1", "seed": 0, "row_id": "r", "kind": "choice", "correct": 1}],
            "JSON boolean",
        ),
        (
            [
                {"eval_row_id": "e1", "seed": 0, "row_id": "r", "kind": "choice", "correct": True},
                {"eval_row_id": "e2", "seed": 0, "row_id": "s", "kind": "choice", "correct": True},
            ],
            "mixes eval rows",
        ),
        (
            [
                {"eval_row_id": "e1", "seed": 0, "row_id": "r", "kind": "choice", "correct": True},
                {"eval_row_id": "e1", "seed": 0, "row_id": "r", "kind": "choice", "correct": False},
            ],
            "share key",
        ),
    ],
)
def test_verdicts_that_could_mispair_are_refused(
    tmp_path: Path, lines: list[dict], match: str
) -> None:
    with pytest.raises(ftc.Refused, match=match):
        ftc.load_verdicts(_write_verdicts(tmp_path / "v.jsonl", lines))


def test_slot_name_disambiguates_two_slots_of_one_kind(tmp_path: Path) -> None:
    lines = [
        {"eval_row_id": "e", "seed": 0, "row_id": "r", "kind": "choice", "slot_name": "a",
         "correct": True},
        {"eval_row_id": "e", "seed": 0, "row_id": "r", "kind": "choice", "slot_name": "b",
         "correct": False},
        {"eval_row_id": "e", "seed": 0, "row_id": "q", "kind": "span", "slot_name": "s",
         "correct": False},
    ]
    v = ftc.load_verdicts(_write_verdicts(tmp_path / "v.jsonl", lines))
    assert v.by_slot_name and v.correct == {("r", "a"): True, ("r", "b"): False}
    assert v.span_rows == 1


def _span(row: str, correct: bool, expected: bool | None) -> dict:
    line = {"eval_row_id": "e", "seed": 0, "row_id": row, "kind": "span",
            "slot_name": "defect_span", "correct": correct}
    if expected is not None:
        line["expected_abstain"] = expected
    return line


def test_span_rows_are_scored_against_always_abstaining(tmp_path: Path) -> None:
    """The n-gram control cannot produce a line pair, so the span rows had no opponent at
    all. Always abstaining is right exactly where the gold abstains: a span head that only
    learned to abstain would tie it, and one that points correctly beats it."""
    rows = [_span(f"q{i}", True, i % 2 == 0) for i in range(200)]
    v = ftc.load_verdicts(_write_verdicts(tmp_path / "v.jsonl", rows))
    got = ftc.abstain_constant_margin(v, seed=0)
    assert isinstance(got, Ran) and got.value == pytest.approx(0.5) and got.passed
    constant = [_span(f"q{i}", i % 2 == 0, i % 2 == 0) for i in range(200)]
    tie = ftc.abstain_constant_margin(
        ftc.load_verdicts(_write_verdicts(tmp_path / "w.jsonl", constant)), seed=0
    )
    assert isinstance(tie, Ran) and tie.value == 0.0 and not tie.passed


def test_a_file_without_expected_abstain_leaves_the_span_margin_not_run(tmp_path: Path) -> None:
    rows = [_span("q0", True, True), _span("q1", True, None)]
    v = ftc.load_verdicts(_write_verdicts(tmp_path / "v.jsonl", rows))
    got = ftc.abstain_constant_margin(v, seed=0)
    assert isinstance(got, NotRun) and "predates" in got.reason
    with pytest.raises(ftc.Refused, match="JSON boolean"):
        ftc.load_verdicts(_write_verdicts(tmp_path / "w.jsonl", [
            {**_span("q0", True, True), "expected_abstain": 1}
        ]))


# --- the control ----------------------------------------------------------------------


#: One option set offered on every row of every stand-in task, in letter order: these docs
#: stand for a fixed-class task, so the control is labelled by value
#: (``qd_train.baseline.control_label_space``).
_OFFERED = (*zip("ABCDEF", ("a", "logic", "no", "only", "stub", "yes"), strict=True), ("Z", "noul"))


def _doc(i: int, value: str, *, task: str = "fam/slot", op: str | None = None) -> RequestDoc:
    # Signal in the text: the value's own token appears, plus row-specific noise.
    return RequestDoc(
        row_id=f"r{i:04d}", slot_name="slot", kind="choice", task=task,
        text=f"change {i} noise {i * 7919 % 97} marker_{value} end",
        value=value, letter=next(k for k, v in _OFFERED if v == value), offered=_OFFERED,
        metadata={"operator": op} if op else {},
    )


def _verdicts_for(val: list[RequestDoc], hits: list[bool], *, span: int = 0) -> ftc.Verdicts:
    keys = [ftc.doc_key(d, by_slot_name=True) for d in val]
    return ftc.Verdicts(
        eval_row_id="e", seed=0, correct=dict(zip(keys, hits, strict=True)),
        kind_of={k: "choice" for k in keys}, by_slot_name=True, span_rows=span,
    )


def test_gate_is_the_paired_margin_over_every_letter_row(qd_prep: Path) -> None:
    train = [_doc(i, "yes" if i % 2 else "no") for i in range(60)]
    val = [_doc(1000 + i, "yes" if i % 2 else "no") for i in range(40)]
    hits = [i % 4 != 0 for i in range(40)]  # the model: 75%
    result = ftc.score_against_control(train, val, _verdicts_for(val, hits), seed=0)
    assert isinstance(result.gate, Ran), result.gate
    control_acc = result.metrics["linear_control_top1.fam/slot"]
    assert isinstance(control_acc, Ran)
    # The control reads the value marker, so it is perfect and the model trails it.
    assert control_acc.value == 1.0
    assert result.gate.value == pytest.approx(0.75 - 1.0)
    assert result.gate.passed is False
    assert isinstance(result.metrics["paired_margin_vs_linear.span"], NotRun)
    assert isinstance(result.metrics["paired_margin_vs_abstain_constant.span"], NotRun)


def test_a_task_whose_control_cannot_fit_makes_the_gate_not_run(qd_prep: Path) -> None:
    """A pooled margin over only the tasks that fitted is a capped sample, not the gate."""
    train = [_doc(i, "yes" if i % 2 else "no") for i in range(40)]
    train += [_doc(500 + i, "only", task="other/slot") for i in range(10)]
    val = [_doc(1000 + i, "yes" if i % 2 else "no") for i in range(10)]
    val += [_doc(2000 + i, "only", task="other/slot") for i in range(5)]
    result = ftc.score_against_control(train, val, _verdicts_for(val, [True] * 15), seed=0)
    assert isinstance(result.gate, NotRun)
    assert "fewer than two classes" in result.gate.reason


# --- operator holdout -----------------------------------------------------------------


def test_holdout_refuses_to_be_a_silent_no_op() -> None:
    train = [_doc(i, "a") for i in range(4)]
    with pytest.raises(ftc.Refused, match="no training row carries"):
        ftc.apply_holdout(train, ftc.HoldOut("operator", "stub.panic"))
    tagged = [_doc(i, "a", op="logic.x") for i in range(4)]
    with pytest.raises(ftc.Refused, match="no training row has"):
        ftc.apply_holdout(tagged, ftc.HoldOut("operator", "stub.panic"))


def test_holdout_removes_the_operator_and_reports_own_and_siblings(qd_prep: Path) -> None:
    def cls(i: int) -> str:
        return "stub" if i % 2 else "logic"

    def op(i: int) -> str:
        return ("stub.panic" if i % 4 == 1 else "stub.todo") if i % 2 else "logic.widen"

    train = [_doc(i, cls(i), op=op(i)) for i in range(80)]
    val = [_doc(1000 + i, cls(i), op=op(i)) for i in range(40)]
    hold = ftc.HoldOut("operator", "stub.panic")
    result = ftc.score_against_control(
        train, val, _verdicts_for(val, [True] * 40), seed=0, hold=hold
    )
    assert result.removed_by_holdout == 20
    assert result.train_rows == 60
    own = result.metrics["holdout.own.model_top1"]
    sib = result.metrics["holdout.siblings.model_top1"]
    assert isinstance(own, Ran) and own.n_total == 10  # val rows i%4==1
    assert isinstance(sib, Ran) and sib.n_total == 10  # stub rows from stub.todo
    assert isinstance(result.metrics["paired_margin_vs_linear.holdout.own"], Ran)


# --- the tool end to end ---------------------------------------------------------------


def _eval_row(ledger_path: Path, *, choice: tuple[int, int], score: tuple[int, int]) -> str:
    """An FT eval row shaped like real_ft_run's _record_score output."""
    protocol = Protocol(
        data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64, backbone_commit="standin",
        recipe_hash="r" * 64, seed=0,
    )
    with RunRecorder(
        Ledger(ledger_path), entry_point=REPO / "tools" / "ft_linear_control.py",
        protocol=protocol, run_kind="eval", repo=REPO, env=Environment.detect(device="cpu"),
        wall_clock_s=1.0, cost=None, quick=True, quick_reason="test fixture",
        recipe={"tool": "tools/real_ft_run.py", "tag": "epoch-score-val"},
    ) as rec:
        for kind, (n, total) in (("choice", choice), ("score", score)):
            rec.metric(
                f"val_top1.{kind}",
                Ran(passed=True, value=n / total, n=n, n_total=total) if total
                else NotRun(reason=f"the val set holds no {kind} row to score"),
            )
    assert rec.row is not None
    return rec.row.row_id


def _fake_runner(
    monkeypatch: pytest.MonkeyPatch, train, val, *, with_fn: bool = True,
    calls: list[dict[str, object]] | None = None,
) -> None:
    module = types.ModuleType("real_ft_run")
    if with_fn:
        # The real signature: the defect-class corpus is part of how the run built its split.
        # The general record, its per-file cap and the replay partition too (phase 4).
        def ft_split_rows(*, commitpackft, max_pairs, rev, config, defect_class=None,
                          defect_download=None, defect_max_rows=None, repo_history=True,
                          general_record=None, general_max_rows=None,
                          replay_partition=False, defect_noul=None):
            if calls is not None:
                calls.append({"defect_class": defect_class, "defect_download": defect_download,
                              "defect_max_rows": defect_max_rows,
                              "repo_history": repo_history, "rev": rev,
                              "general_record": general_record,
                              "general_max_rows": general_max_rows,
                              "replay_partition": replay_partition,
                              "defect_noul": defect_noul})
            return train, val
        module.ft_split_rows = ft_split_rows  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "real_ft_run", module)


def _split(rows):
    train = [r for i, r in enumerate(rows) if i % 4]
    val = [r for i, r in enumerate(rows) if not i % 4]
    return train, val


def test_the_tool_writes_the_gate_as_a_new_eval_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qd_prep: Path
) -> None:
    rows, config = _rows(80)
    # change_scope is single-class in the fixture, which the control refuses (tested above).
    rows = [r for r in rows if r.family_id == "code.commit_intent"]
    train, val = _split(rows)
    val_docs, _ = request_texts(val, seed=config.seed)
    hits = [i % 3 != 0 for i in range(len(val_docs))]
    kinds = {"choice": [0, 0], "score": [0, 0]}
    for d, h in zip(val_docs, hits, strict=True):
        kinds[d.kind][0] += h
        kinds[d.kind][1] += 1
    ledger = tmp_path / "ledger.jsonl"
    eval_id = _eval_row(ledger, choice=tuple(kinds["choice"]), score=(0, 0))
    verdicts = _write_verdicts(tmp_path / "v.jsonl", [
        {"eval_row_id": eval_id, "seed": 0, "row_id": d.row_id, "kind": d.kind,
         "correct": h} for d, h in zip(val_docs, hits, strict=True)
    ])
    _fake_runner(monkeypatch, train, val)
    code = ftc.main([
        # No --eval-row: the verdicts file names its eval row, which is how a campaign config
        # can point at a file before the row it scores exists.
        "--ledger", str(ledger), "--verdicts", str(verdicts),
        "--max-pairs", "80", "--rev", REV,
    ])
    rows_out = Ledger(ledger).rows()
    assert len(rows_out) == 2
    gate = rows_out[-1].gates["paired_margin_vs_linear"]
    assert isinstance(gate, Ran), gate
    assert code == 0
    assert rows_out[-1].protocol == rows_out[0].protocol  # joins the seed family
    assert rows_out[-1].metrics["scored_eval_row_id"].value == eval_id  # type: ignore[union-attr]
    # Before this tool the gate on the FT row was never evaluated; the new row is.
    assert isinstance(rows_out[0].gates["paired_margin_vs_linear"], NotRun)


def _tool_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                  calls: list[dict[str, object]] | None = None) -> list[str]:
    """The argv, ledger and verdicts of ``test_the_tool_writes_the_gate_as_a_new_eval_row``."""
    rows, config = _rows(80)
    rows = [r for r in rows if r.family_id == "code.commit_intent"]
    train, val = _split(rows)
    val_docs, _ = request_texts(val, seed=config.seed)
    hits = [i % 3 != 0 for i in range(len(val_docs))]
    ledger = tmp_path / "ledger.jsonl"
    eval_id = _eval_row(
        ledger, choice=(sum(h for d, h in zip(val_docs, hits, strict=True) if d.kind == "choice"),
                        sum(d.kind == "choice" for d in val_docs)), score=(0, 0),
    )
    verdicts = _write_verdicts(tmp_path / "v.jsonl", [
        {"eval_row_id": eval_id, "seed": 0, "row_id": d.row_id, "kind": d.kind,
         "correct": h} for d, h in zip(val_docs, hits, strict=True)
    ])
    _fake_runner(monkeypatch, train, val, calls=calls)
    return ["--ledger", str(ledger), "--verdicts", str(verdicts), "--max-pairs", "80",
            "--rev", REV]


def test_the_row_names_the_engine_and_the_bytes_that_fitted_the_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qd_prep: Path
) -> None:
    """The Python engine wrote no engine key; a row fitted by qd-prep says so, by sha256."""
    argv = _tool_fixture(tmp_path, monkeypatch)
    assert ftc.main(argv) == 0
    recipe = Ledger(tmp_path / "ledger.jsonl").rows()[-1].recipe or {}
    assert recipe["control_engine"] == "qd-prep linfit"
    assert recipe["control_engine_sha256"] == hashlib.sha256(qd_prep.read_bytes()).hexdigest()


def test_without_the_engine_the_tool_refuses_before_rebuilding_the_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No Python fallback, and no split rebuild -- most of a run's wall clock -- for nothing."""
    calls: list[dict[str, object]] = []
    argv = _tool_fixture(tmp_path, monkeypatch, calls=calls)
    monkeypatch.delenv("QD_PREP_BIN", raising=False)
    with pytest.raises(SystemExit, match="QD_PREP_BIN is unset"):
        ftc.main(argv)
    assert calls == []
    assert len(Ledger(tmp_path / "ledger.jsonl").rows()) == 1  # the eval row alone


def test_without_the_engine_nothing_is_scored(monkeypatch: pytest.MonkeyPatch) -> None:
    train = [_doc(i, "yes" if i % 2 else "no") for i in range(20)]
    val = [_doc(1000 + i, "yes" if i % 2 else "no") for i in range(6)]
    monkeypatch.delenv("QD_PREP_BIN", raising=False)
    with pytest.raises(SystemExit, match="QD_PREP_BIN is unset"):
        ftc.score_against_control(train, val, _verdicts_for(val, [True] * 6), seed=0)


def _task_docs():
    rows, config = _rows(80)
    train, val = _split(rows)
    train_docs, _ = request_texts(train, seed=config.seed)
    val_docs, _ = request_texts(val, seed=config.seed)
    return train_docs, val_docs


@pytest.mark.parametrize("arm", [ftc.NGRAM_ARM, ftc.LENGTH_ARM], ids=lambda a: a.name)
def test_every_tasks_control_answers_row_for_row_as_the_python_engine_did(
    qd_prep: Path, arm: ftc.ControlArm
) -> None:
    """Old engine vs new, through the tool's own ``fit_task``: the Python ``LinearBaseline``
    on its sparse operand (what ``fit_task`` ran before) and qd-prep agree on every val row of
    every task, and on convergence, iterations and the selected L2."""
    from qd_train.baseline import LinearBaseline, control_label, control_label_space

    train, val = _task_docs()
    tasks = sorted({d.task for d in val})
    compared = 0
    for task in tasks:
        t_train = [d for d in train if d.task == task]
        t_val = [d for d in val if d.task == task]
        got = ftc.fit_task(
            task, t_train, t_val, by_slot_name=True, seed=0, max_iter=ftc.DEFAULT_MAX_ITER,
            dense_budget_bytes=0, max_fit_minutes=None, cache_dir=None, engine=qd_prep, arm=arm,
        )
        space = control_label_space(t_train, t_val)
        labels = [control_label(d, space) for d in t_train]
        if len(set(labels)) < 2:
            assert isinstance(got.convergence, NotRun)
            continue
        reference = LinearBaseline(
            hasher=arm.make_features(), seed=0, max_iter=ftc.DEFAULT_MAX_ITER,
            dense_budget_bytes=0,
        )
        fit = reference.fit([arm.text_of(d) for d in t_train], labels)
        assert isinstance(got.convergence, Ran) == fit.converged, task
        if not fit.converged:
            continue
        assert got.convergence.detail == reference.convergence().detail  # type: ignore[union-attr]
        predicted = reference.predict([arm.text_of(d) for d in t_val])
        want = {ftc.doc_key(d, by_slot_name=True): p == control_label(d, space)
                for d, p in zip(t_val, predicted, strict=True)}
        assert got.correct == want, task
        compared += 1
    assert compared >= 1


def test_a_native_fit_is_cached_and_read_back_under_the_unchanged_key(
    tmp_path: Path, qd_prep: Path
) -> None:
    """A native fit is stored and read back under one key. The key folds in baseline.py's
    sha256, so the label-space change there (``control_label_space``, 2026-10-01) orphaned
    every entry written before it -- deliberately: a general task's cached verdicts were
    scored in the wrong label space."""
    train = [_doc(i, "yes" if i % 2 else "no") for i in range(40)]
    val = [_doc(1000 + i, "yes" if i % 2 else "no") for i in range(12)]
    kwargs = dict(by_slot_name=True, seed=0, max_iter=ftc.DEFAULT_MAX_ITER,
                  dense_budget_bytes=None, max_fit_minutes=None, cache_dir=tmp_path,
                  engine=qd_prep)
    first = ftc.fit_task("fam/slot", train, val, **kwargs)  # type: ignore[arg-type]
    second = ftc.fit_task("fam/slot", train, val, **kwargs)  # type: ignore[arg-type]
    assert (first.cached, second.cached) == (False, True)
    assert second.correct == first.correct
    assert len(list(tmp_path.glob("linear-control-*.json"))) == 1


def test_the_defect_class_corpus_reaches_the_runs_own_split_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qd_prep: Path
) -> None:
    """Phase 3's go/no-go scores a model trained on code.defect_class. Without the corpus the
    control rebuilt the split with no defect rows, so the run's verdicts could never pair and
    the gate could never pass -- the campaign would stop on a tool gap, not on the model."""
    rows, config = _rows(80)
    rows = [r for r in rows if r.family_id == "code.commit_intent"]
    train, val = _split(rows)
    val_docs, _ = request_texts(val, seed=config.seed)
    kinds = {"choice": [0, 0], "score": [0, 0]}
    for d in val_docs:
        kinds[d.kind][0] += 1
        kinds[d.kind][1] += 1
    ledger = tmp_path / "ledger.jsonl"
    eval_id = _eval_row(ledger, choice=tuple(kinds["choice"]), score=(0, 0))
    verdicts = _write_verdicts(tmp_path / "v.jsonl", [
        {"eval_row_id": eval_id, "seed": 0, "row_id": d.row_id, "kind": d.kind, "correct": True}
        for d in val_docs
    ])
    calls: list[dict[str, object]] = []
    _fake_runner(monkeypatch, train, val, calls=calls)
    corpus, download = tmp_path / "corpus-v2", tmp_path / "download"
    ftc.main([
        "--ledger", str(ledger), "--verdicts", str(verdicts), "--max-pairs", "80",
        "--rev", REV, "--defect-class", str(corpus), "--defect-download", str(download),
        "--defect-max-rows", "40",
    ])
    assert calls == [{"defect_class": corpus, "defect_download": download,
                      "defect_max_rows": 40, "repo_history": True, "rev": REV,
                      "general_record": None, "general_max_rows": None,
                      "replay_partition": False, "defect_noul": None}]
    assert Ledger(ledger).rows()[-1].recipe["defect_class"] == "corpus-v2"
    assert "defect_noul_examples_sha256" not in Ledger(ledger).rows()[-1].recipe


def test_a_set_built_with_noul_rows_is_rebuilt_with_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qd_prep: Path
) -> None:
    """Phase 4's v3 set (89b619d9) was built with --defect-noul: the noul rows are in its
    train and val splits, so a control that rebuilt the split without them would pair a
    different val set. The row names the noul corpus by its manifest's examples sha256, as
    the pipeline's and real_ft_run's recipes do."""
    ledger, verdicts, train, val = _scorable(tmp_path)
    noul = tmp_path / "defect-noul-v1"
    noul.mkdir()
    (noul / "manifest.json").write_text(json.dumps({"examples_sha256": "ab" * 32}))
    calls: list[dict[str, object]] = []
    _fake_runner(monkeypatch, train, val, calls=calls)
    ftc.main(["--ledger", str(ledger), "--verdicts", str(verdicts), "--rev", REV,
              "--max-pairs", "80", "--defect-noul", str(noul)])
    assert [c["defect_noul"] for c in calls] == [noul]
    assert Ledger(ledger).rows()[-1].recipe["defect_noul_examples_sha256"] == "ab" * 32


def _scorable(tmp_path: Path) -> tuple[Path, Path, list, list]:
    rows, config = _rows(80)
    rows = [r for r in rows if r.family_id == "code.commit_intent"]
    train, val = _split(rows)
    val_docs, _ = request_texts(val, seed=config.seed)
    ledger = tmp_path / "ledger.jsonl"
    eval_id = _eval_row(ledger, choice=(len(val_docs), len(val_docs)), score=(0, 0))
    verdicts = _write_verdicts(tmp_path / "v.jsonl", [
        {"eval_row_id": eval_id, "seed": 0, "row_id": d.row_id, "kind": d.kind, "correct": True}
        for d in val_docs
    ])
    return ledger, verdicts, train, val


def test_a_set_built_without_repository_history_is_rebuilt_without_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qd_prep: Path
) -> None:
    """Phase 3 trains on code.defect_class alone (--no-repo-history). The control has
    to rebuild that split, not one with this repository's history added back."""
    ledger, verdicts, train, val = _scorable(tmp_path)
    calls: list[dict[str, object]] = []
    _fake_runner(monkeypatch, train, val, calls=calls)
    ftc.main(["--ledger", str(ledger), "--verdicts", str(verdicts), "--rev", REV,
              "--no-repo-history", "--defect-class", str(tmp_path / "corpus-v2")])
    assert [c["repo_history"] for c in calls] == [False]
    assert Ledger(ledger).rows()[-1].recipe["repo_history"] is False


def test_a_general_record_set_is_rebuilt_with_the_record_and_its_replay_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qd_prep: Path
) -> None:
    """Phase 4 trains on the full mixture (--general-record, --replay-shards). The control
    has to rebuild that split -- the general families in, the replay-only rows out of the
    gold train split -- and its row names the record by sha256, only when one was used."""
    ledger, verdicts, train, val = _scorable(tmp_path)
    record = tmp_path / "fetch-record.json"
    record.write_text("[]", encoding="utf-8")
    calls: list[dict[str, object]] = []
    _fake_runner(monkeypatch, train, val, calls=calls)
    ftc.main(["--ledger", str(ledger), "--verdicts", str(verdicts), "--rev", REV,
              "--max-pairs", "80", "--general-record", str(record),
              "--general-max-rows", "123", "--replay-partition"])
    assert [(c["general_record"], c["general_max_rows"], c["replay_partition"])
            for c in calls] == [(record, 123, True)]
    recipe = Ledger(ledger).rows()[-1].recipe
    assert recipe["general_record_sha256"] == hashlib.sha256(b"[]").hexdigest()
    assert recipe["general_max_rows"] == 123
    assert recipe["replay_partition"] is True


def test_the_general_row_cap_is_recorded_resolved_not_as_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qd_prep: Path
) -> None:
    import real_tokenizer_pipeline as pipeline

    ledger, verdicts, train, val = _scorable(tmp_path)
    record = tmp_path / "fetch-record.json"
    record.write_text("[]", encoding="utf-8")
    _fake_runner(monkeypatch, train, val)
    ftc.main(["--ledger", str(ledger), "--verdicts", str(verdicts), "--rev", REV,
              "--max-pairs", "80", "--general-record", str(record)])
    recipe = Ledger(ledger).rows()[-1].recipe
    assert recipe["general_max_rows"] == pipeline.DEFAULT_GENERAL_MAX_ROWS
    assert "replay_partition" not in recipe


def test_a_control_without_a_general_record_hashes_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qd_prep: Path
) -> None:
    ledger, verdicts, train, val = _scorable(tmp_path)
    calls: list[dict[str, object]] = []
    _fake_runner(monkeypatch, train, val, calls=calls)
    ftc.main(["--ledger", str(ledger), "--verdicts", str(verdicts), "--rev", REV,
              "--max-pairs", "80"])
    assert [(c["general_record"], c["general_max_rows"], c["replay_partition"])
            for c in calls] == [(None, None, False)]
    recipe = Ledger(ledger).rows()[-1].recipe
    assert not {"general_record_sha256", "general_max_rows", "replay_partition"} & set(recipe)


@pytest.mark.parametrize("flags", [["--general-max-rows", "5"], ["--replay-partition"]])
def test_general_options_without_the_record_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    flags: list[str],
) -> None:
    _fake_runner(monkeypatch, [], [])
    with pytest.raises(SystemExit) as exc:
        ftc.main(["--ledger", str(tmp_path / "l.jsonl"), "--verdicts", str(tmp_path / "v"),
                  "--max-pairs", "80", "--rev", REV, *flags])
    assert exc.value.code == 2
    assert "without --general-record read nothing" in capsys.readouterr().err


def test_max_pairs_is_refused_where_it_bounds_nothing_and_required_where_it_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, verdicts, train, val = _scorable(tmp_path)
    _fake_runner(monkeypatch, train, val)
    base = ["--ledger", str(ledger), "--verdicts", str(verdicts), "--rev", REV]
    with pytest.raises(ftc.Refused, match="bounds nothing"):
        ftc.main([*base, "--no-repo-history", "--max-pairs", "80"])
    with pytest.raises(ftc.Refused, match="--max-pairs is required"):
        ftc.main(base)
    assert len(Ledger(ledger).rows()) == 1, "a refused control writes no gate row"


@pytest.mark.parametrize("rev", ["HEAD", "main", "0632f69"])
def test_a_revision_named_by_anything_but_its_full_sha_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rev: str
) -> None:
    """The gate row names the split by its revision; HEAD is a different split tomorrow."""
    ledger, verdicts, train, val = _scorable(tmp_path)
    _fake_runner(monkeypatch, train, val)
    with pytest.raises(ftc.Refused, match="not a full 40-character commit sha"):
        ftc.main(["--ledger", str(ledger), "--verdicts", str(verdicts), "--max-pairs", "80",
                  "--rev", rev])
    assert len(Ledger(ledger).rows()) == 1


def test_defect_options_without_the_corpus_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_runner(monkeypatch, [], [])
    with pytest.raises(SystemExit) as exc:
        ftc.main(["--ledger", str(tmp_path / "l.jsonl"), "--verdicts", str(tmp_path / "v"),
                  "--max-pairs", "80", "--rev", REV, "--defect-max-rows", "40"])
    assert exc.value.code == 2
    assert "without --defect-class read nothing" in capsys.readouterr().err


def test_verdicts_from_another_decode_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    eval_id = _eval_row(ledger, choice=(5, 10), score=(1, 1))
    verdicts = _write_verdicts(tmp_path / "v.jsonl", [
        {"eval_row_id": eval_id, "seed": 0, "row_id": f"r{i}", "kind": "choice",
         "correct": True} for i in range(10)
    ] + [{"eval_row_id": eval_id, "seed": 0, "row_id": "s", "kind": "score", "correct": True}])
    _fake_runner(monkeypatch, [], [])
    with pytest.raises(ftc.Refused, match="not that run's decode"):
        ftc.main(["--ledger", str(ledger), "--eval-row", eval_id, "--verdicts",
                  str(verdicts), "--max-pairs", "1", "--rev", REV])


def test_a_runner_without_ft_split_rows_is_refused_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runner(monkeypatch, [], [], with_fn=False)
    with pytest.raises(ftc.Refused, match="has no ft_split_rows"):
        ftc.split_rows_function()


def test_the_iteration_budget_matches_the_rung0_control() -> None:
    pytest.importorskip("torch")
    import rung0_linear_control
    import rung0_real_run

    assert ftc.DEFAULT_MAX_ITER == rung0_linear_control.DEFAULT_MAX_ITER
    assert ftc.DEFAULT_MAX_ITER == rung0_real_run.LINEAR_CONTROL_MAX_ITER


def test_held_out_families_are_refused_before_any_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qd_prep: Path
) -> None:
    """Rule 3: a control fitted on a held-out family sets the bar with data the model may
    never see. ``code.language_id`` is in DEFAULT_HELD_OUT_FAMILIES."""
    rows, config = _rows(20)
    intent = [r for r in rows if r.family_id == "code.commit_intent"]
    held = [r for r in rows if r.family_id == "code.language_id"]
    assert held and config.is_held_out_family("code.language_id")
    val_docs, _ = request_texts(intent[:5], seed=config.seed)
    ledger = tmp_path / "ledger.jsonl"
    eval_id = _eval_row(ledger, choice=(0, len(val_docs)), score=(0, 0))
    verdicts = _write_verdicts(tmp_path / "v.jsonl", [
        {"eval_row_id": eval_id, "seed": 0, "row_id": d.row_id, "kind": d.kind,
         "correct": False} for d in val_docs
    ])
    _fake_runner(monkeypatch, intent[5:] + held, intent[:5])
    with pytest.raises(ftc.Refused, match="held-out"):
        ftc.main(["--ledger", str(ledger), "--verdicts", str(verdicts),
                  "--max-pairs", "1", "--rev", REV])
    assert len(Ledger(ledger).rows()) == 1, "a refused control writes no gate row"
