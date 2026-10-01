"""``--suite-logits``: a scored checkpoint's per-case scores on its suite-verdict lines.

Fable I diagnostic (2026-10-01): why does the noul row lose on OOD cases -- by how much, and
to which row -- for each seed, the average and the 3-seed ensemble? The OOD lines already
carry both passes' letter rows, the noul row included (``row_logits_1``/``row_logits_2``).
The needle lines carried only the pointer's argmax; under the flag they also carry its
scores over the runtime's rows (``start_logits``/``end_logits``, the abstention at
``noul_row``). Off, every line is what it always was.

The flag is refused wherever it would be dropped: without a checkpoint scorer, without the
file it writes into, and beside the gate row's needle suite, which the worker process
decodes without pointer scores.
"""

from __future__ import annotations

import argparse
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
        ([*_ONE, "--needle", "--suite-logits", "--suite-verdicts-out", "s.jsonl"],
         "needle worker process"),
        ([*_ONE, "--needle", "--ood", "--suite-logits", "--suite-verdicts-out", "s.jsonl"],
         "needle worker process"),
    ],
)
def test_suite_logits_is_refused_where_it_would_be_dropped(tmp_path, extra, match):
    argv = ["--out", str(tmp_path), *_BASE,
            *[str(tmp_path / x) if x.endswith(".jsonl") else x for x in extra]]
    with pytest.raises(SystemExit, match=match):
        rft.main(argv)


@pytest.mark.parametrize(
    ("needle", "ood", "control"),
    [(False, True, None), (True, False, (1024,)), (True, False, None)],
)
def test_suite_logits_is_accepted_where_it_is_written(needle, ood, control):
    args = argparse.Namespace(
        suite_logits=True, score_checkpoint=Path("/c/epoch-seed0-cuda.json"),
        suite_verdicts_out=Path("/s.jsonl"), needle=needle, ood=ood, needle_control=control,
    )
    if needle and control is None:
        with pytest.raises(SystemExit, match="needle worker process"):
            rft._check_suite_logits_flags(args)
        return
    rft._check_suite_logits_flags(args)
    rft._check_suite_logits_flags(SimpleNamespace(suite_logits=False))  # off checks nothing
