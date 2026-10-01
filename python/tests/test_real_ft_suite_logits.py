"""``--suite-logits``: a scored checkpoint's per-case scores on its suite-verdict lines.

Fable I diagnostic (2026-10-01): why does the noul row lose on OOD cases -- by how much, and
to which row -- for each seed, the average and the 3-seed ensemble? The OOD lines already
carry both passes' letter rows, the noul row included (``row_logits_1``/``row_logits_2``).
The needle lines carried only the pointer's argmax; under the flag they also carry its
scores over the runtime's rows (``start_logits``/``end_logits``, the abstention at
``noul_row``). Off, every line is what it always was.

The flag is refused wherever it would be dropped: without a checkpoint scorer and without
the file it writes into. The gate row's needle suite is decoded by the worker process from
the suite its scoring process handed over; under the flag the worker's verdicts carry the
pointer scores too, and the scoring process refuses a worker verdict that does not
(GAP-SUITE-LOGITS-NEEDLE-WORKER-2026-10-01, closed in the merge of main f45a646).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
from test_needle_ft_contract import _decoded, _suite  # noqa: E402
from test_needle_handoff import _plan, _reader, _write  # noqa: E402
from test_needle_handoff import _suite as _handoff_suite  # noqa: E402
from test_real_ft_score_checkpoint import (  # noqa: E402
    _average,
    _avg_control_args,
    _avg_row,
    _ledger,
    _run_avg_needle_control,
)

from qd_train.needle import build_suite  # noqa: E402


def _pointer(i: int) -> dict[str, list[float]]:
    return {"start_logits": [float(i), 0.5, -1.0], "end_logits": [0.0, float(i), 2.0]}


def _decode_recording(calls: list[bool], cases):
    """A stand-in ``_decode`` for one needle case per call, recording what it was asked."""

    def decode(step, batches, labels_for, letter_id, *, pointer_scores):
        i = len(calls)
        calls.append(pointer_scores)
        verdict = {"kind": "span", "row_id": cases[i].case_id, "top": [1, 2], "noul_row": 2,
                   "rows": 3}
        return {"verdicts": [{**verdict, **(_pointer(i) if pointer_scores else {})}]}

    return decode


def test_needle_lines_carry_the_pointer_scores_only_when_asked(monkeypatch):
    cases = build_suite(target_tokens=1024, cases_per_depth=1, seed=0)
    suite = _suite(cases)
    monkeypatch.setattr(rft, "release_device_cache", lambda: None)

    calls: list[bool] = []
    monkeypatch.setattr(rft, "_decode", _decode_recording(calls, cases))
    off = rft.needle_predictions(None, suite, {})  # type: ignore[arg-type]
    assert calls == [False] * len(cases)
    for line in off.verdicts:
        assert set(line) == set(rft.NEEDLE_VERDICT_FIELDS), "off, a line is what it was"

    calls.clear()
    on = rft.needle_predictions(None, suite, {}, logits=True)  # type: ignore[arg-type]
    assert calls == [True] * len(cases)
    assert on.predictions == off.predictions, "the scores ride along; the verdict is unmoved"
    for i, line in enumerate(on.verdicts):
        assert set(line) == {*rft.NEEDLE_VERDICT_FIELDS, *rft.SUITE_LOGIT_KEYS}
        assert {k: line[k] for k in rft.SUITE_LOGIT_KEYS} == _pointer(i)
        assert len(line["start_logits"]) == line["noul_row"] + 1  # type: ignore[arg-type]

    # score_needle decodes in-process (no worker's predictions) with the same switch.
    calls.clear()
    _, _, lines = rft.score_needle(None, suite, {}, logits=True)  # type: ignore[arg-type]
    assert calls == [True] * len(cases) and all("start_logits" in v for v in lines)


def test_a_needle_control_row_asks_for_the_scores_under_the_flag(tmp_path, monkeypatch):
    """--needle-control decodes in this process, so its lines carry the scores the flag
    asks for -- for every control length."""
    avg = _average(tmp_path)
    ledger = _ledger(tmp_path, *(_avg_row(s) for s in (0, 1, 2)))
    asked: list[object] = []

    def predictions(step, s, letter_id, **kw):
        asked.append(kw.get("logits"))
        decoded = _decoded(s.cases, hit_every=1, abstain_every=2)
        return rft.NeedleDecoded(
            decoded.predictions,
            tuple({**v, **(_pointer(0) if kw.get("logits") else {})} for v in decoded.verdicts),
        )

    for flag in (False, True):
        asked.clear()
        args = _avg_control_args(avg, ledger)
        args.suite_logits = flag
        (_, lines, _), _, n_cases = _run_avg_needle_control(
            monkeypatch, args, predictions=predictions
        )
        assert asked == [flag] * 3, "one decode per control length, each told the flag"
        assert all(("start_logits" in v) is flag for v in lines) and len(lines) == 3 * n_cases


_BASE = ["--rev", "0" * 40, "--score-val", "--real-backbone", "/x",
         "--devices", "cuda", "--instance", "gh200", "--usd-per-hour", "2.29"]
_ONE = ["--score-checkpoint", "/c/epoch-seed0-cuda.json", "--ft-row-id", "aaaaaaaa",
        "--ft-ledger", "/l", "--seeds", "0"]


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--suite-logits"], "needs --score-checkpoint"),
        (["--epoch", "--needle", "--suite-logits", "--suite-verdicts-out", "s.jsonl"],
         "needs --score-checkpoint"),
        ([*_ONE, "--ood", "--suite-logits"], "without that file"),
    ],
)
def test_suite_logits_is_refused_where_it_would_be_dropped(tmp_path, extra, match):
    argv = ["--out", str(tmp_path), *_BASE,
            *[str(tmp_path / x) if x.endswith(".jsonl") else x for x in extra]]
    with pytest.raises(SystemExit, match=match):
        rft.main(argv)


@pytest.mark.parametrize(
    ("needle", "ood", "control", "plan"),
    [(False, True, None, None), (True, False, (1024,), None), (True, False, None, None),
     (True, True, None, None), (True, True, None, Path("/plan.json"))],
)
def test_suite_logits_is_accepted_where_it_is_written(needle, ood, control, plan):
    """Every needle decode keeps the scores: --needle-control and a --score-plan decode
    in-process, and a single --score-checkpoint gate row's worker writes them into the
    verdicts it hands back (the --needle rows here were refused before the worker did)."""
    args = argparse.Namespace(
        suite_logits=True,
        score_checkpoint=None if plan else Path("/c/epoch-seed0-cuda.json"),
        suite_verdicts_out=Path("/s.jsonl"), needle=needle, ood=ood, needle_control=control,
        score_plan=plan,
    )
    rft._check_suite_logits_flags(args)
    rft._check_suite_logits_flags(SimpleNamespace(suite_logits=False))  # off checks nothing


def _worker_argv(tmp_path: Path, handoff: Path, out: Path, *extra: str) -> list[str]:
    """main()'s argv as the needle worker, as run_needle_worker builds it: the scoring
    process's own argv, then the handoff and the predictions file."""
    backbone = tmp_path / "snapshot"
    backbone.mkdir(exist_ok=True)
    return [
        "--out", str(tmp_path), "--rev", "0" * 40, "--no-repo-history",
        "--score-checkpoint", str(tmp_path / "epoch-seed0-cpu.json"),
        "--ft-ledger", str(tmp_path / "ft.jsonl"), "--ft-row-id", "abcdefgh", "--seeds", "0",
        "--real-backbone", str(backbone), "--score-val", "--needle", "--devices", "cpu",
        *extra, "--needle-handoff", str(handoff), "--needle-predictions-out", str(out),
    ]


def test_the_needle_worker_writes_the_pointer_scores_under_the_flag(tmp_path, monkeypatch):
    """GAP-SUITE-LOGITS-NEEDLE-WORKER-2026-10-01: the gate row's needle suite is decoded by
    the worker, from the suite its scoring process handed over. Under --suite-logits every
    verdict it writes carries the pointer's scores, as the in-process decodes' do; off, none.
    The real needle_predictions runs; only the step and its decode are stood in."""
    suite = _handoff_suite()
    handoff = _write(tmp_path, suite)
    monkeypatch.setattr(rft, "resolve_rev", lambda repo, rev: rev)
    monkeypatch.setattr(rft, "ShardReader", lambda *a, **k: _reader("t" * 64))
    monkeypatch.setattr(rft, "open_val_reader", lambda *a, **k: _reader("v" * 64))
    monkeypatch.setattr(rft, "val_plan", lambda reader, config: _plan())
    monkeypatch.setattr(rft, "_checkpoint_step", lambda args, **kw: ("step", {}, {}, 0, {}))
    monkeypatch.setattr(rft, "release_device_cache", lambda: None)
    suite_out = tmp_path / "suite.jsonl"
    for flag in (False, True):
        calls: list[bool] = []
        monkeypatch.setattr(rft, "_decode", _decode_recording(calls, suite.cases))
        out = tmp_path / f"predictions-{flag}.json"
        extra = ("--suite-logits", "--suite-verdicts-out", str(suite_out)) if flag else ()
        assert rft.main(_worker_argv(tmp_path, handoff, out, *extra)) == 0
        assert calls == [flag] * len(suite.cases), "every case decoded, each told the flag"
        verdicts = json.loads(out.read_text(encoding="utf-8"))["verdicts"]
        assert [v["case_id"] for v in verdicts] == [c.case_id for c in suite.cases]
        for i, line in enumerate(verdicts):
            if flag:
                assert {k: line[k] for k in rft.SUITE_LOGIT_KEYS} == _pointer(i)
            else:
                assert set(line) == set(rft.NEEDLE_VERDICT_FIELDS), "off, a line is unchanged"
    assert not suite_out.exists(), "the worker writes no suite lines; its scoring process does"


def test_the_scoring_process_keeps_a_workers_scores_and_refuses_their_absence():
    """Under the flag, a worker's verdicts reach the suite lines with their scores, and one
    without them is refused -- never written as a line indistinguishable from a run that was
    not asked for scores. Off, the worker's verdicts are taken as they are."""
    cases = build_suite(target_tokens=1024, cases_per_depth=1, seed=0)
    suite = _suite(cases)
    plain = _decoded(cases, hit_every=1, abstain_every=2)
    scored = rft.NeedleDecoded(
        plain.predictions, tuple({**v, **_pointer(i)} for i, v in enumerate(plain.verdicts))
    )
    _, _, lines = rft.score_needle(None, suite, {}, decoded=scored, logits=True)  # type: ignore[arg-type]
    assert [{k: v[k] for k in rft.SUITE_LOGIT_KEYS} for v in lines] == [
        _pointer(i) for i in range(len(cases))
    ]
    _, _, lines = rft.score_needle(None, suite, {}, decoded=plain, logits=False)  # type: ignore[arg-type]
    assert lines == plain.verdicts
    for drop in rft.SUITE_LOGIT_KEYS:
        partial = rft.NeedleDecoded(
            scored.predictions,
            tuple({k: x for k, x in v.items() if not (i == 1 and k == drop)}
                  for i, v in enumerate(scored.verdicts)),
        )
        with pytest.raises(SystemExit, match=f"lacks {drop}"):
            rft.score_needle(None, suite, {}, decoded=partial, logits=True)  # type: ignore[arg-type]
    with pytest.raises(SystemExit, match="pointer scores"):
        rft.score_needle(None, suite, {}, decoded=plain, logits=True)  # type: ignore[arg-type]


def test_run_needle_worker_hands_the_flag_over_and_the_scores_back(tmp_path, monkeypatch):
    """The scoring process starts its worker with its own argv, --suite-logits included, and
    takes the scores in the worker's verdicts back unchanged."""
    suite = _handoff_suite()
    seen: dict[str, object] = {}

    def run(cmd, *, timeout, check):
        seen["cmd"] = list(cmd)
        out = Path(cmd[cmd.index("--needle-predictions-out") + 1])
        verdicts = [
            {**{k: None for k in rft.NEEDLE_VERDICT_FIELDS}, "case_id": c.case_id,
             "predicted_hunk": None, **_pointer(i)}
            for i, c in enumerate(suite.cases)
        ]
        out.write_text(json.dumps({
            "digest": suite.digest, "predictions": {c.case_id: None for c in suite.cases},
            "verdicts": verdicts,
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rft.subprocess, "run", run)
    argv = ["--score-checkpoint", "/c/epoch-seed0-cuda.json", "--needle", "--suite-logits"]
    decoded = rft.run_needle_worker(
        argv, suite, letter_id={"Z": 57, "A": 32}, eval_widths=[1152, 300, 301],
        train=_reader("t" * 64),  # type: ignore[arg-type]
        val=SimpleNamespace(reader=_reader("v" * 64), plan=_plan()),  # type: ignore[arg-type]
    )
    cmd = seen["cmd"]
    assert isinstance(cmd, list) and cmd[2 : 2 + len(argv)] == argv
    assert [{k: v[k] for k in rft.SUITE_LOGIT_KEYS} for v in decoded.verdicts] == [
        _pointer(i) for i in range(len(suite.cases))
    ]
