"""The per-option linear control (Fable H, 2026-10-01): ``qd_train.option_control``, its native
engine (``qd-prep linfit`` features 3, through ``linear_control_native.fit_options``) and its
report-only row (``tools/ft_linear_control.py --option-control``).

The letter control labels a per-row-option task by the gold letter and sits at chance there (J1:
CSQA 222/1,197, MMLU 368/1,485), because a bag of n-grams cannot tie a letter to a row's own
option text. This control scores each SHOWN option with one binary scorer and answers the best.
These tests pin: options are read from the prompt the model saw, never from the gold; the carve
and the L2 selection mirror the letter control's by row; the native engine is the oracle bit for
bit; and the row it writes carries only its own keys, so the letter control's gate, the ledger's
join and promotion are untouched by it.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
REV = "0632f693d3b765b726499e7b4bf19c67959b75cb"
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ft_linear_control as ftc  # noqa: E402
import linear_control_native as native  # noqa: E402
from data_fixtures import commitpackft_row  # noqa: E402
from test_control_label_space import (  # noqa: E402
    CSQA_TASK,
    _csqa_docs,
    _csqa_raw,
    _general_docs,
)
from test_ft_linear_control import _fake_runner, _write_verdicts  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.render import M_OPT_BEGIN, escape_block, escape_inline  # noqa: E402
from qd_train.baseline import CharNGramHasher, RequestDoc, request_texts  # noqa: E402
from qd_train.ledger import (  # noqa: E402
    CODE_THAT_RAN,
    REQUIRED_CONTROLS,
    REQUIRED_GATES,
    Environment,
    Ledger,
    Protocol,
    RunRecorder,
    _joined,
    _promotion_units,
)
from qd_train.option_control import (  # noqa: E402
    OptionPairFeatures,
    OptionScorer,
    best_option,
    examples,
    hstack,
    read_option_row,
    read_option_rows,
)
from qd_train.tristate import NotRun, Ran  # noqa: E402

PER_ROW_TASKS = (
    CSQA_TASK, "knowledge.multiple_choice/answer", "intent.classification/intent",
    "intent.within_domain/intent",
)
ONE_SET_TASKS = ("intent.domain/domain", "intent.in_scope/in_scope", "code.language_id/language")


def _dense(x) -> np.ndarray:
    out = np.zeros(x.shape)
    out[x.rows, x.indices] = x.data
    return out


# --- reading a row ------------------------------------------------------------------------


def test_a_rows_question_and_options_are_read_from_the_prompt_it_was_shown() -> None:
    docs = [d for d in _general_docs() if d.task in PER_ROW_TASKS]
    rows, unreadable = read_option_rows(docs)
    assert unreadable == []
    assert {r.doc.task for r in rows} == set(PER_ROW_TASKS)
    for r in rows:
        d = r.doc
        # Every shown option, noul included, as rendered, in the order shown.
        assert r.options == tuple(escape_inline(v) for _, v in d.offered)
        assert r.options[-1] == "noul"
        assert all(f"\n{letter}. {o}\n" in d.text
                   for (letter, _), o in zip(d.offered, r.options, strict=True))
        # The gold names the positive and nothing else.
        assert dict(d.offered)[d.letter] == d.value
        assert r.options[r.gold] == escape_inline(d.value)
        # The question is the context region, as the model read it.
        assert r.question == escape_block(
            d.context if isinstance(d.context, str) else d.context.decode("utf-8")
        )
        assert r.question in d.text and M_OPT_BEGIN not in r.question


def _first_csqa() -> RequestDoc:
    return next(d for d in _general_docs() if d.task == CSQA_TASK)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda d: dataclasses.replace(d, offered=(), letter=""), "no rendering"),
        (lambda d: dataclasses.replace(d, letter="Q"), "not among the shown"),
        # The prompt shows options other than the ones the doc says it shows.
        (lambda d: dataclasses.replace(
            d, offered=(*d.offered[:-2], (d.offered[-2][0], "swapped"), d.offered[-1])
        ), "option block"),
        (lambda d: dataclasses.replace(d, text=d.text + M_OPT_BEGIN), "exactly once"),
        (lambda d: dataclasses.replace(d, text=d.text.replace("\n<|qd_context_end|>", "", 1)),
         "exactly once"),
    ],
)
def test_a_row_whose_shown_options_cannot_be_read_is_refused_with_the_reason(
    change, reason: str
) -> None:
    doc = change(_first_csqa())
    got = read_option_row(doc)
    assert isinstance(got, str), got
    assert reason in got and doc.row_id in got


# --- the oracle's pieces ------------------------------------------------------------------


def test_the_two_blocks_are_the_pair_text_then_the_option_alone() -> None:
    rows, _ = read_option_rows(_csqa_docs(3, 0)[0])
    docs, labels, sizes = examples(rows)
    assert docs[0] == (f"{rows[0].question}\n{rows[0].options[0]}", rows[0].options[0])
    assert sizes.tolist() == [len(r.options) for r in rows]
    assert labels.sum() == len(rows)  # one positive per row
    hasher = CharNGramHasher()
    x = OptionPairFeatures(hasher).transform(docs)
    assert x.shape == (len(docs), 2 * hasher.dim)
    want = np.hstack([
        _dense(hasher.transform([p for p, _ in docs])),
        _dense(hasher.transform([o for _, o in docs])),
    ])
    assert np.array_equal(_dense(x), want)
    # Column-sorted within each row, as CSR.matmul's accumulation order needs.
    same_row = np.diff(x.rows) == 0
    assert np.all(np.diff(x.indices)[same_row] > 0)
    # And empty blocks stack.
    empty = hasher.transform(["", "ab"])
    assert np.array_equal(_dense(hstack(empty, empty)), np.zeros((2, 2 * hasher.dim)))


def test_a_rows_answer_is_its_first_option_of_greatest_log_odds() -> None:
    logits = np.array([[0.0, 1.0], [0.5, 1.5], [0.0, 0.0],  # row 0: tie at 1.0 -> option 0
                       [2.0, 0.0], [0.0, 0.5]])             # row 1 -> option 1
    assert best_option(logits, np.array([3, 2])) == [0, 1]
    assert best_option(np.zeros((0, 2)), np.array([], dtype=np.int64)) == []
    with pytest.raises(ValueError):
        best_option(logits, np.array([3, 3]))


def test_the_carve_is_whole_rows_drawn_as_the_letter_control_draws_documents() -> None:
    sizes = np.array([3, 5, 2, 4, 6, 3, 2, 5, 4, 3])
    val_idx, tr_idx, val_sizes = OptionScorer(seed=7).carve(sizes)
    order = np.random.default_rng(7).permutation(len(sizes))
    val_rows, tr_rows = order[:2], order[2:]  # max(1, int(10 * 0.2))
    starts = np.cumsum(sizes) - sizes
    assert val_idx.tolist() == [i for r in val_rows for i in range(starts[r], starts[r] + sizes[r])]
    assert tr_idx.tolist() == [i for r in tr_rows for i in range(starts[r], starts[r] + sizes[r])]
    assert val_sizes.tolist() == sizes[val_rows].tolist()
    with pytest.raises(ValueError, match="at least 4 rows"):
        OptionScorer().carve(np.array([2, 2, 2]))


# --- the native engine is the oracle ------------------------------------------------------


def test_the_native_option_fit_is_the_oracle_bit_for_bit(qd_prep: Path) -> None:
    """Production settings (max_iter 6,000, the four-value grid) on a CSQA-shaped task: the
    binary's convergence, iterations, every grid point, selected L2, gradient norm, loss
    history, weights, bias and logits are the oracle's, bit for bit, and so are the answers."""
    train, val = _csqa_docs(120, 50)
    tr, _ = read_option_rows(train)
    va, _ = read_option_rows(val)
    got = native.fit_options(qd_prep, OptionScorer(seed=0, max_iter=ftc.DEFAULT_MAX_ITER), tr, va)
    ref = OptionScorer(seed=0, max_iter=ftc.DEFAULT_MAX_ITER, dense_budget_bytes=0)
    want = ref.fit(tr)
    assert got.fit.grid == want.grid
    assert (got.fit.selected, got.fit.l2) == (want.selected, want.l2)
    assert (got.fit.converged, got.fit.iterations) == (want.converged, want.iterations)
    assert float(got.fit.final_grad_norm).hex() == float(want.final_grad_norm).hex()
    assert got.fit.loss_history == want.loss_history
    assert got.fit.weights.tobytes() == want.weights.tobytes()
    assert got.fit.bias.tobytes() == want.bias.tobytes()
    assert got.eval_logits.tobytes() == ref.logits(va).tobytes()
    assert got.predictions == ref.predict(va)
    # The grid counts carve ROWS: 20% of 120.
    assert all(g.val_total == 24 for g in got.fit.grid)
    assert want.converged


def test_a_reply_whose_grid_is_not_counted_by_row_is_refused(qd_prep: Path) -> None:
    """The binary's reply is checked against the request: a grid whose validation totals are
    not the carve's rows (here, as if it had counted the carve's 36 option examples, which is
    what a per-example selection would report) is not this request's reply."""
    train, val = _csqa_docs(30, 5)
    tr, _ = read_option_rows(train)
    va, _ = read_option_rows(val)
    real = native._read_reply

    def as_if_examples(reply: bytes, binary: Path, **kw: object):
        return real(reply, binary, **{**kw, "val_total": 6 * 6})  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(native, "_read_reply", as_if_examples)
        with pytest.raises(SystemExit, match="grid that is not the request's"):
            native.fit_options(qd_prep, OptionScorer(seed=0, max_iter=50), tr, va)


# --- the tool: which tasks, which keys ----------------------------------------------------


def test_per_row_option_tasks_are_scored_and_one_set_tasks_are_not(qd_prep: Path) -> None:
    docs = _general_docs()
    by_task: dict[str, list[RequestDoc]] = {}
    for d in docs:
        by_task.setdefault(d.task, []).append(d)
    for task in (*PER_ROW_TASKS, *ONE_SET_TASKS):
        mine = by_task[task]
        cut = max(4, len(mine) * 3 // 4)
        got = ftc.fit_option_task(
            task, mine[:cut], mine[cut:], by_slot_name=True, seed=0,
            max_iter=ftc.DEFAULT_MAX_ITER, max_fit_minutes=None, engine=qd_prep,
        )
        if task in ONE_SET_TASKS:
            assert got.applies is False, task
            assert isinstance(got.accuracy, NotRun) and "one option set" in got.accuracy.reason
            assert got.correct == {}
        else:
            assert got.applies is True, task
            assert isinstance(got.accuracy, Ran), (task, got.accuracy)
            assert got.accuracy.n_total == len(mine) - cut
            assert isinstance(got.chance, Ran)


def test_a_csqa_shaped_task_is_answered_from_its_options(qd_prep: Path) -> None:
    """The gold is always the ``shelf`` option, and only reading the options can see that."""
    train, val = _csqa_docs(120, 50)
    got = ftc.fit_option_task(
        CSQA_TASK, train, val, by_slot_name=True, seed=0, max_iter=ftc.DEFAULT_MAX_ITER,
        max_fit_minutes=None, engine=qd_prep,
    )
    assert isinstance(got.convergence, Ran) and got.convergence.passed
    assert isinstance(got.accuracy, Ran) and isinstance(got.chance, Ran)
    assert got.chance.value == pytest.approx(1 / 6)  # five options and noul
    assert got.accuracy.value > 0.9, got.accuracy


def _verdicts(val: list[RequestDoc], hits: list[bool]) -> ftc.Verdicts:
    keys = [ftc.doc_key(d, by_slot_name=True) for d in val]
    return ftc.Verdicts(
        eval_row_id="e", seed=0, correct=dict(zip(keys, hits, strict=True)),
        kind_of={k: d.kind for k, d in zip(keys, val, strict=True)}, by_slot_name=True,
        span_rows=0,
    )


def _mixed_split() -> tuple[list[RequestDoc], list[RequestDoc]]:
    docs = [d for d in _general_docs()
            if d.task in (CSQA_TASK, "intent.domain/domain", "code.language_id/language")]
    train = [d for i, d in enumerate(docs) if i % 4]
    val = [d for i, d in enumerate(docs) if not i % 4]
    return train, val


def test_the_option_row_carries_only_its_own_keys(qd_prep: Path) -> None:
    train, val = _mixed_split()
    hits = [i % 3 != 0 for i in range(len(val))]
    got = ftc.score_against_option_control(
        train, val, _verdicts(val, hits), seed=0, max_iter=ftc.DEFAULT_MAX_ITER, engine=qd_prep,
    )
    for key in got.metrics:
        assert key.startswith((f"{ftc.OPTION_ARM}.", f"{ftc.OPTION_MARGIN}.")), key
        assert key not in REQUIRED_GATES and key not in REQUIRED_CONTROLS
        # Never the letter control's names, nor one a prefix match on them would take.
        assert not key.startswith(("linear_control", "length_control", f"{ftc.GATE}.",
                                   "paired_margin_vs_length", "paired_margin_vs_abstain"))
    pooled = got.metrics[f"{ftc.OPTION_MARGIN}.choice"]
    assert isinstance(pooled, Ran), pooled
    csqa_val = [d for d in val if d.task == CSQA_TASK]
    assert pooled.n_total == len(csqa_val), "only the per-row-option task's rows"
    assert "per-row-option tasks" in (pooled.detail or "")
    assert isinstance(got.metrics[f"{ftc.OPTION_MARGIN}.choice.{CSQA_TASK}"], Ran)
    for task in ("intent.domain/domain", "code.language_id/language"):
        assert isinstance(got.metrics[f"{ftc.OPTION_ARM}.top1.{task}"], NotRun)
        assert f"{ftc.OPTION_MARGIN}.choice.{task}" not in got.metrics


def test_an_unreadable_val_row_leaves_the_margins_not_run_never_scored_wrong(
    qd_prep: Path,
) -> None:
    train, val = _mixed_split()
    i = next(j for j, d in enumerate(val) if d.task == CSQA_TASK)
    val[i] = dataclasses.replace(val[i], text=val[i].text.replace("A. ", "A) ", 1))
    got = ftc.score_against_option_control(
        train, val, _verdicts(val, [True] * len(val)), seed=0, max_iter=ftc.DEFAULT_MAX_ITER,
        engine=qd_prep,
    )
    top1 = got.metrics[f"{ftc.OPTION_ARM}.top1.{CSQA_TASK}"]
    assert isinstance(top1, Ran)
    assert top1.n_total == sum(d.task == CSQA_TASK for d in val) - 1
    assert "1 val row(s) whose shown options could not be read are NOT scored" in (
        top1.detail or ""
    )
    for key in (f"{ftc.OPTION_MARGIN}.choice", f"{ftc.OPTION_MARGIN}.choice.{CSQA_TASK}"):
        assert isinstance(got.metrics[key], NotRun), key


def test_a_split_with_no_per_row_option_task_scores_nothing(qd_prep: Path) -> None:
    train, val = _mixed_split()
    train = [d for d in train if d.task != CSQA_TASK]
    val = [d for d in val if d.task != CSQA_TASK]
    got = ftc.score_against_option_control(
        train, val, _verdicts(val, [True] * len(val)), seed=0, engine=qd_prep,
    )
    pooled = got.metrics[f"{ftc.OPTION_MARGIN}.choice"]
    assert isinstance(pooled, NotRun) and "no task" in pooled.reason


# --- the row, the ledger's join and promotion --------------------------------------------


def _eval_row(ledger: Path, val: list[RequestDoc], hits: list[bool]) -> str:
    protocol = Protocol(
        data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64, recipe_hash="r" * 64,
        backbone_commit="b" * 40, seed=0,
    )
    with RunRecorder(
        Ledger(ledger), entry_point=REPO / "tools" / "real_ft_run.py", protocol=protocol,
        run_kind="eval", repo=REPO, env=Environment.detect(device="cpu"), wall_clock_s=1.0,
        cost=None, quick=False, recipe={"tool": "tools/real_ft_run.py"},
    ) as rec:
        for kind in ("choice", "score"):
            n = sum(h for d, h in zip(val, hits, strict=True) if d.kind == kind)
            total = sum(d.kind == kind for d in val)
            rec.metric(
                f"val_top1.{kind}",
                Ran(passed=True, value=n / total, n=n, n_total=total) if total
                else NotRun(reason=f"the val set holds no {kind} row to score"),
            )
    assert rec.row is not None
    return rec.row.row_id


def test_the_option_row_changes_nothing_the_ledger_joins_or_promotion_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qd_prep: Path
) -> None:
    """Eval row + the letter control's row, then + the per-option row: every gate and control
    the unit is judged by joins to the same state, and promotion's reasons are the same but
    for the unit's label naming one more row."""
    config = DataConfig()
    mixture = build_mixture(
        {"tau/commonsense_qa": [_csqa_raw(i) for i in range(80)],
         "bigcode/commitpackft": [commitpackft_row(i) for i in range(80)]},
        config=config,
    )
    rows = [r for r in mixture.rows
            if r.family_id in ("commonsense.multiple_choice", "code.commit_intent")]
    train = [r for i, r in enumerate(rows) if i % 4]
    val = [r for i, r in enumerate(rows) if not i % 4]
    val_docs, _ = request_texts(val, seed=config.seed)
    hits = [i % 3 != 0 for i in range(len(val_docs))]
    ledger = tmp_path / "ledger.jsonl"
    eval_id = _eval_row(ledger, val_docs, hits)
    verdicts = _write_verdicts(tmp_path / "v.jsonl", [
        {"eval_row_id": eval_id, "seed": 0, "row_id": d.row_id, "kind": d.kind,
         "slot_name": d.slot_name, "correct": h}
        for d, h in zip(val_docs, hits, strict=True)
    ])
    _fake_runner(monkeypatch, train, val)
    argv = ["--ledger", str(ledger), "--verdicts", str(verdicts), "--max-pairs", "80",
            "--rev", REV]
    assert ftc.main(argv) == 0  # the letter control's gate row
    family = Ledger(ledger).rows()[0].protocol.hash_without_seed()
    before_rows = Ledger(ledger).rows()
    before = Ledger(ledger).promotion_verdict(family)

    assert ftc.main([*argv, "--option-control"]) == 0
    after_rows = Ledger(ledger).rows()
    assert len(after_rows) == len(before_rows) + 1
    option = after_rows[-1]
    recipe = option.recipe or {}
    assert recipe["eval_row_id"] == eval_id and recipe["control"] == ftc.OPTION_ARM
    assert recipe["report_only"] is True and "G1" in str(recipe["comparator_decision"])
    assert "report-only" in (option.notes or "")
    assert option.quick is False  # inherited from the eval row; never a promotion blocker
    # Metrics only: every gate and control on it is the recorder's own never-evaluated fill.
    assert all(isinstance(g, NotRun) and "never evaluated" in g.reason
               for g in option.gates.values())
    assert all(isinstance(c, NotRun) for c in option.controls.values())
    # Beside the recorder's own provenance (`code_that_ran`, on every row of every tool).
    foreign = [k for k in option.metrics if k != CODE_THAT_RAN
               and not k.startswith((f"{ftc.OPTION_ARM}.", f"{ftc.OPTION_MARGIN}."))]
    assert foreign == []
    assert isinstance(option.metrics[f"{ftc.OPTION_MARGIN}.choice"], Ran)
    letter = before_rows[-1]
    assert not any(k.startswith(ftc.OPTION_MARGIN) or k.startswith(ftc.OPTION_ARM)
                   for k in letter.metrics)

    def joined(rows):
        units, refusals = _promotion_units(rows)
        assert refusals == []
        ((unit, sups),) = units
        assert unit.row_id == eval_id
        return (
            {g: _joined(unit.gates.get(g), [s.gates.get(g) for s in sups])
             for g in REQUIRED_GATES},
            {c: _joined(unit.controls.get(c), [s.controls.get(c) for s in sups])
             for c in REQUIRED_CONTROLS},
        )

    assert joined(after_rows) == joined(before_rows)
    assert isinstance(joined(after_rows)[0][ftc.GATE], Ran)  # the letter row's gate
    after = Ledger(ledger).promotion_verdict(family)
    assert after.promoted == before.promoted
    label_tail = f" + {option.row_id}"
    assert [r.replace(label_tail, "") for r in after.reasons] == list(before.reasons)


def test_the_option_control_refuses_the_flags_it_would_ignore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = ["--ledger", str(tmp_path / "l.jsonl"), "--verdicts", str(tmp_path / "v.jsonl"),
            "--max-pairs", "1", "--rev", REV, "--option-control"]
    with pytest.raises(ftc.Refused, match="operator-holdout"):
        ftc.main([*base, "--hold-out-operator", "swap_args"])
    with pytest.raises(ftc.Refused, match="not cached"):
        ftc.main([*base, "--control-cache", str(tmp_path / "cache")])


