"""A4: ``paired_margin_vs_linear`` for FT eval rows, via ``tools/ft_linear_control.py``.

Before this, every FT eval row carried the gate as ``not_run`` with no tool able to change
that (``ledger/gh200-ft-commitpackft-2026-09-22.jsonl``). These tests pin the pieces that
make a post-hoc verdict trustworthy rather than merely present: the control reads the
model's own prompt, the two arms are paired by key over an identical population, a
verdicts file that belongs to another row is refused, and the operator-holdout arm refuses
to be a silent no-op.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
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


# --- the control ----------------------------------------------------------------------


def _doc(i: int, value: str, *, task: str = "fam/slot", op: str | None = None) -> RequestDoc:
    # Signal in the text: the value's own token appears, plus row-specific noise.
    return RequestDoc(
        row_id=f"r{i:04d}", slot_name="slot", kind="choice", task=task,
        text=f"change {i} noise {i * 7919 % 97} marker_{value} end",
        value=value, metadata={"operator": op} if op else {},
    )


def _verdicts_for(val: list[RequestDoc], hits: list[bool], *, span: int = 0) -> ftc.Verdicts:
    keys = [ftc.doc_key(d, by_slot_name=True) for d in val]
    return ftc.Verdicts(
        eval_row_id="e", seed=0, correct=dict(zip(keys, hits, strict=True)),
        kind_of={k: "choice" for k in keys}, by_slot_name=True, span_rows=span,
    )


def test_gate_is_the_paired_margin_over_every_letter_row() -> None:
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


def test_a_task_whose_control_cannot_fit_makes_the_gate_not_run() -> None:
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


def test_holdout_removes_the_operator_and_reports_own_and_siblings() -> None:
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


def _fake_runner(monkeypatch: pytest.MonkeyPatch, train, val, *, with_fn: bool = True) -> None:
    module = types.ModuleType("real_ft_run")
    if with_fn:
        def ft_split_rows(*, commitpackft, max_pairs, rev, config):
            return train, val
        module.ft_split_rows = ft_split_rows  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "real_ft_run", module)


def _split(rows):
    train = [r for i, r in enumerate(rows) if i % 4]
    val = [r for i, r in enumerate(rows) if not i % 4]
    return train, val


def test_the_tool_writes_the_gate_as_a_new_eval_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        "--max-pairs", "80", "--rev", "HEAD",
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
                  str(verdicts), "--max-pairs", "1", "--rev", "HEAD"])


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
                  "--max-pairs", "1", "--rev", "HEAD"])
    assert len(Ledger(ledger).rows()) == 1, "a refused control writes no gate row"
