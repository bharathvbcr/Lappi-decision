"""``needle_hunk_recall`` on the FT eval, under the contract the human approved 2026-09-30.

GAP-NEEDLE-HUNK-RECALL-HAS-NO-FT-EVAL-CONTRACT: the gate was implemented and unwired, and
the defect val set cannot score it (every row is one hunk). These pin the contract's three
numbers, that a needle case becomes a defect row whose gold is the needle hunk, and the hit
rule: the hunk of the predicted START line, a header line or an abstention being a miss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_data.config import DataConfig  # noqa: E402
from qd_data.defect_class import CONTEXT_HEADER_LINES  # noqa: E402
from qd_train import needle  # noqa: E402
from qd_train.needle import build_suite, hunk_of_context_line, needle_defect_row  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402


def _cases():
    return build_suite(target_tokens=1024, cases_per_depth=2, seed=3)


def test_the_contract_is_the_one_the_human_approved():
    """Rule 2: thresholds are read-only to an agent. A change here is a human decision."""
    assert needle.NEEDLE_MIN_RECALL == 0.95
    assert needle.NEEDLE_CASES_PER_DEPTH == 60
    assert needle.NEEDLE_TARGET_TOKENS == 8192
    assert needle.NEEDLE_HIT_RULE == "hunk-of-predicted-start-line"


def test_every_case_becomes_a_defect_row_whose_gold_is_the_needle_hunk():
    for case in _cases():
        row = needle_defect_row(case, config=DataConfig())
        assert row.family_id == "code.defect_class"
        gold = {g.slot_name: g for g in row.gold}
        assert gold["defect_class"].value == "stub"
        start, end = gold["defect_span"].value
        # 1-based over the rendered context, which is the header plus the haystack.
        assert hunk_of_context_line(case, start - 1) == case.needle_index
        assert hunk_of_context_line(case, end - 1) == case.needle_index
        assert row.metadata["language"] == case.language


def test_a_header_line_belongs_to_no_hunk_and_a_line_past_the_end_is_refused():
    case = _cases()[0]
    for line in range(CONTEXT_HEADER_LINES):
        assert hunk_of_context_line(case, line) is None
    assert hunk_of_context_line(case, CONTEXT_HEADER_LINES) == 0, "the haystack opens on @@"
    n = len(case.context.split("\n"))
    with pytest.raises(ValueError, match="past the context"):
        hunk_of_context_line(case, CONTEXT_HEADER_LINES + n)
    with pytest.raises(ValueError, match="negative"):
        hunk_of_context_line(case, -1)


def test_every_line_of_a_hunk_maps_to_that_hunk():
    case = _cases()[4]
    body = case.context.split("\n")
    seen = -1
    for j, text in enumerate(body[:-1]):
        if text.startswith("@@"):
            seen += 1
        assert hunk_of_context_line(case, CONTEXT_HEADER_LINES + j) == seen
    assert seen == case.n_hunks - 1


torch = pytest.importorskip("torch", reason="real_ft_run imports torch at module scope")
sys.path.insert(0, str(REPO / "tools"))
import real_ft_run as rft  # noqa: E402


def _suite(cases, width: int = 8832):
    from types import SimpleNamespace

    import numpy as np

    return rft.NeedleSuite(
        cases=cases,
        batches=[SimpleNamespace(tokens=np.zeros((1, width))) for _ in cases],  # type: ignore[misc]
        labels_for={i: [] for i in range(len(cases))},
        token_lengths=[8000 + i for i in range(len(cases))],
    )


def _line_in_hunk(case, hunk: int) -> int:
    body = case.context.split("\n")
    seen = -1
    for j, text in enumerate(body):
        if text.startswith("@@"):
            seen += 1
            if seen == hunk:
                return CONTEXT_HEADER_LINES + j + 1  # a line inside, not the header
    raise AssertionError(hunk)


def _score(monkeypatch, cases, tops):
    verdicts = [
        {"kind": "span", "row_id": c.case_id, "top": list(t), "noul_row": 999}
        for c, t in zip(cases, tops, strict=True)
    ]
    monkeypatch.setattr(rft, "_decode", lambda *a, **k: {"verdicts": verdicts})
    return rft.score_needle(None, _suite(cases), {})  # type: ignore[arg-type]


def test_pointing_inside_the_needle_hunk_is_a_hit_whatever_the_end_line(monkeypatch):
    cases = build_suite(target_tokens=1024, cases_per_depth=5, seed=1)
    tops = [(_line_in_hunk(c, c.needle_index), 0) for c in cases]
    gate, metrics, _ = _score(monkeypatch, cases, tops)
    assert isinstance(gate, Ran) and gate.passed and gate.value == 1.0
    assert set(metrics) >= {"needle_hunk_recall.depth.0-20%", "needle_suite_tokens"}


def test_the_next_hunk_an_abstention_and_a_header_line_are_misses(monkeypatch):
    cases = build_suite(target_tokens=1024, cases_per_depth=5, seed=1)
    tops = []
    for i, c in enumerate(cases):
        if i % 3 == 0:
            other = c.needle_index + 1 if c.needle_index + 1 < c.n_hunks else c.needle_index - 1
            tops.append((_line_in_hunk(c, other), 0))
        elif i % 3 == 1:
            tops.append((999, 999))  # the abstention row
        else:
            tops.append((0, 0))  # the `file:` header
    gate, metrics, _ = _score(monkeypatch, cases, tops)
    assert isinstance(gate, Ran) and gate.value == 0.0 and not gate.passed
    assert "abstained" in metrics["needle_suite_tokens"].detail


def test_without_the_flag_the_gate_is_not_run_and_the_recipe_is_untouched():
    suite = rft.NeedleSuite([], [], {}, [], not_run="--needle was not given")
    gate, metrics, lines = rft.score_needle(None, suite, {})  # type: ignore[arg-type]
    assert isinstance(gate, NotRun) and metrics == {}
    assert rft.needle_gate((gate, metrics, lines), suite).recipe is None, (
        "a gate that did not run must not move the recipe hash"
    )
    ran = rft.NeedleSuite([], [], {}, [8000], seed=4)
    recipe = rft.needle_gate((gate, metrics, lines), ran).recipe
    assert recipe is not None and recipe["min_recall"] == 0.95 and recipe["suite_seed"] == 4


def test_needle_argv_needs_score_val_and_the_real_backbone(tmp_path):
    with pytest.raises(SystemExit, match="--needle scores the model"):
        rft.main(["--out", str(tmp_path), "--rev", "0" * 40, "--needle"])


def test_a_step_too_narrow_for_the_suite_is_refused_before_any_decode(monkeypatch):
    """2026-09-30: the step was bounded by the val plan (1,110 tokens), and the needle
    suite's first 8,329-token case was refused by the step itself -- after a full val pass.
    Now refused before anything is decoded, and both paths size the step for the suite."""
    import inspect
    from types import SimpleNamespace

    cases = build_suite(target_tokens=1024, cases_per_depth=1, seed=0)

    def never(*a, **k):
        raise AssertionError("decoded with a step that cannot hold the suite")

    monkeypatch.setattr(rft, "_decode", never)
    with pytest.raises(SystemExit, match="bounded at 1110"):
        rft.score_needle(SimpleNamespace(max_width=1110), _suite(cases), {})  # type: ignore[arg-type]
    assert "max_width=max((width, *eval_widths))" in inspect.getsource(rft._real_step)
    assert "eval_widths=suite_widths(needle_suite, ood_suite)" in inspect.getsource(
        rft._score_checkpoint
    )
    assert "eval_widths=suite_widths(needle_suite, ood_suite)" in inspect.getsource(rft.main)
    empty = rft.OodSuite([], None, None, not_run="x")
    assert rft.suite_widths(_suite(cases), empty) == [8832] * len(cases), (
        "the step is bounded by the PADDED width every case is decoded at"
    )


def test_the_suite_is_decoded_one_case_at_a_time_at_one_width(monkeypatch):
    """MPS keeps a compiled graph per distinct shape (probe, 2026-09-30: six cases at one
    width held driver memory flat at 18.81 GiB; a new length added 3.9 GiB), so every case
    is decoded at the suite's one padded width, one batch per call."""
    cases = build_suite(target_tokens=1024, cases_per_depth=2, seed=0)
    widths: list[int] = []

    def decode(step, batches, labels_for, letter_id):
        widths.extend(int(b.tokens.shape[1]) for b in batches)
        assert len(batches) == 1
        return {"verdicts": []}

    monkeypatch.setattr(rft, "_decode", decode)
    monkeypatch.setattr(rft, "release_device_cache", lambda: None)
    rft.score_needle(None, _suite(cases), {})  # type: ignore[arg-type]
    assert widths == [8832] * len(cases)


def test_a_workers_predictions_are_scored_without_decoding(monkeypatch):
    cases = build_suite(target_tokens=1024, cases_per_depth=5, seed=1)

    def never(*a, **k):
        raise AssertionError("decoded although the worker's predictions were given")

    monkeypatch.setattr(rft, "_decode", never)
    gate, _, _ = rft.score_needle(
        None, _suite(cases), {},  # type: ignore[arg-type]
        decoded=rft.NeedleDecoded({c.case_id: c.needle_index for c in cases}),
    )
    assert isinstance(gate, Ran) and gate.value == 1.0


def _worker(monkeypatch, payload, returncode=0):
    """A worker process that answers ``payload``. The handoff it is given is stubbed here:
    ``test_needle_handoff.py`` holds what is written and read."""
    import json

    def fake_run(cmd, timeout, check):
        assert cmd[-2] == "--needle-predictions-out" and timeout == rft.NEEDLE_WORKER_TIMEOUT_S
        assert cmd[-4] == "--needle-handoff" and Path(cmd[-3]).is_file()
        Path(cmd[-1]).write_text(json.dumps(payload), encoding="utf-8")
        from types import SimpleNamespace

        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(rft.subprocess, "run", fake_run)
    monkeypatch.setattr(
        rft, "write_needle_handoff", lambda path, suite, **kw: path.write_bytes(b"handoff")
    )


def _run_worker(suite):
    return rft.run_needle_worker(
        ["--x"], suite, letter_id={}, eval_widths=[], train=None, val=None,  # type: ignore[arg-type]
    )


def _raw(cases, predictions):
    return [
        {**{k: 0 for k in rft.NEEDLE_VERDICT_FIELDS}, "suite": "needle", "case_id": c.case_id,
         "predicted_hunk": predictions[c.case_id]}
        for c in cases
    ]


def test_score_needle_keeps_each_cases_raw_pointer(monkeypatch):
    """2026-09-30: 235 of 300 cases abstained and ~63 missed, and only the hunk index was
    kept -- so a miss in the adjacent hunk (a mapping defect) and one in a far filler (the
    model) could not be told apart. The pointer itself is now kept per case."""
    cases = build_suite(target_tokens=1024, cases_per_depth=2, seed=1)
    tops = [
        (999, 999) if i == 0 else (_line_in_hunk(c, c.needle_index), 0)
        for i, c in enumerate(cases)
    ]
    _, _, lines = _score(monkeypatch, cases, tops)
    assert [v["case_id"] for v in lines] == [c.case_id for c in cases]
    for v, c, (start, end) in zip(lines, cases, tops, strict=True):
        assert set(rft.NEEDLE_VERDICT_FIELDS) <= set(v)
        assert (v["start"], v["end"], v["noul_row"]) == (start, end, 999)
        assert v["needle_index"] == c.needle_index and v["depth_bucket"] == c.depth_bucket
    assert lines[0]["abstained"] and lines[0]["predicted_hunk"] is None and not lines[0]["hit"]
    assert all(v["hit"] and not v["abstained"] for v in lines[1:])
    gate = rft.needle_gate(_score(monkeypatch, cases, tops), _suite(cases))
    assert gate.verdicts == lines


def test_a_worker_without_matching_raw_verdicts_is_refused(monkeypatch):
    cases = build_suite(target_tokens=1024, cases_per_depth=1, seed=0)
    suite = rft.NeedleSuite(cases, [], {}, [1] * len(cases), digest="d" * 64)
    good = {c.case_id: c.needle_index for c in cases}
    raw = _raw(cases, good)
    wrong_hunk = [{**raw[0], "predicted_hunk": None}, *raw[1:]]
    short = [{k: v for k, v in raw[0].items() if k != "start"}, *raw[1:]]
    for verdicts, match in (
        (None, "no raw verdicts"),
        (raw[1:], "raw verdicts do not name exactly"),
        (wrong_hunk, "names hunk None and its prediction"),
        (short, "lacks \\['start'\\]"),
    ):
        _worker(monkeypatch, {"digest": "d" * 64, "predictions": good, "verdicts": verdicts})
        with pytest.raises(SystemExit, match=match):
            _run_worker(suite)


def test_the_needle_worker_is_trusted_only_for_this_exact_suite(monkeypatch):
    cases = build_suite(target_tokens=1024, cases_per_depth=1, seed=0)
    suite = rft.NeedleSuite(cases, [], {}, [1] * len(cases), digest="d" * 64)
    good = {c.case_id: (None if i % 2 else c.needle_index) for i, c in enumerate(cases)}
    raw = _raw(cases, good)
    _worker(monkeypatch, {"digest": "d" * 64, "predictions": good, "verdicts": raw})
    got = _run_worker(suite)
    assert got.predictions == good and list(got.verdicts) == raw
    for payload, code, match in (
        ({"digest": "e" * 64, "predictions": good, "verdicts": raw}, 0, "different suite"),
        ({"digest": "d" * 64, "predictions": good, "verdicts": raw}, 3, "exited 3"),
        ({"digest": "d" * 64, "predictions": dict(list(good.items())[1:]), "verdicts": raw},
         0, "exactly"),
        ({"digest": "d" * 64, "predictions": {**good, cases[0].case_id: "2"}, "verdicts": raw},
         0, "is '2'"),
    ):
        _worker(monkeypatch, payload, code)
        with pytest.raises(SystemExit, match=match):
            _run_worker(suite)


def test_the_worker_flag_only_means_something_as_the_needle_worker(tmp_path):
    with pytest.raises(SystemExit, match="is the --needle worker"):
        rft.main(["--out", str(tmp_path), "--rev", "0" * 40,
                  "--needle-predictions-out", str(tmp_path / "p.json")])


def test_repadding_moves_nothing_but_the_padding():
    import numpy as np

    from qd_train.artifacts import SLOT_SPAN, SPAN_ABSTAIN
    from qd_train.shards import assemble_batch

    ids = np.arange(10, 30, dtype=np.int32)
    batch = assemble_batch(
        [ids], kinds=np.asarray([SLOT_SPAN]), target_index=np.asarray([18]),
        spans=np.asarray([[SPAN_ABSTAIN, SPAN_ABSTAIN]]), candidates=[(0, 4, 9)],
        width=20, bucket=20, index=7,
    )
    wide = rft._repad(batch, 64)
    assert wide.tokens.shape == (1, 64) and wide.index == 7
    assert wide.tokens[0, :20].tolist() == ids.tolist()
    assert int(wide.lengths[0]) == 20 and int(wide.target_index[0]) == 18
    assert np.flatnonzero(wide.line_starts[0]).tolist() == [0, 4, 9]
    assert wide.span_target.tolist() == batch.span_target.tolist()


def test_a_span_only_batch_skips_the_vocabulary_head():
    import inspect

    src = inspect.getsource(rft._decode)
    assert "if any(label.slot_kind != SLOT_SPAN for label in labels_for[b])" in src


# --- --needle-control: the gate's rule at other lengths, on a row of its own -----------------


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("2048,1024", "distinct and ascending"),
        ("1024,1024", "distinct and ascending"),
        ("128,1024", "at least 256"),
        ("1k,2k", "comma-separated"),
        (",".join(str(256 * (i + 1)) for i in range(9)), "at most 8 arms"),
    ],
)
def test_needle_control_lengths_are_refused_unless_well_formed(text, match):
    with pytest.raises(SystemExit, match=match):
        rft.parse_needle_control(text)


def test_needle_control_lengths_parse():
    assert rft.parse_needle_control("1024,2048,4096,8192") == (1024, 2048, 4096, 8192)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ([], "needs --score-checkpoint and --needle"),
        (["--needle", "--score-checkpoint", "/c.json", "--ood"], "--ood would record nothing"),
    ],
)
def test_needle_control_argv_is_refused_before_anything_loads(tmp_path, extra, match):
    with pytest.raises(SystemExit, match=match):
        rft.main(["--out", str(tmp_path), "--rev", "0" * 40, "--needle-control", "1024",
                  *extra])


def _decoded(cases, hit_every: int, abstain_every: int):
    predictions = {}
    for i, c in enumerate(cases):
        if i % abstain_every == 0:
            predictions[c.case_id] = None
        elif i % hit_every == 0:
            predictions[c.case_id] = c.needle_index
        else:
            predictions[c.case_id] = c.needle_index + 1
    return rft.NeedleDecoded(predictions, tuple(_raw(cases, predictions)))


def test_an_arm_is_recorded_under_its_control_key_and_never_the_gate():
    cases = build_suite(target_tokens=1024, cases_per_depth=5, seed=1)
    suite = _suite(cases)
    metrics = rft.needle_control_metrics(
        1024, suite, _decoded(cases, hit_every=1, abstain_every=5), trained_width=1625
    )
    assert "needle_hunk_recall" not in metrics
    assert all(
        k.startswith(("needle_hunk_recall.control.1024", "needle_suite_tokens.control.1024"))
        for k in metrics
    )
    arm = metrics["needle_hunk_recall.control.1024"]
    assert isinstance(arm, Ran) and "NOT the gate" in arm.detail
    abstained = metrics["needle_hunk_recall.control.1024.abstained"]
    assert (abstained.n, abstained.n_total) == (5, 25)
    assert "beyond the trained width 1625" in metrics["needle_suite_tokens.control.1024"].detail, (
        "_suite() reports 8000+ real tokens per case, past the trained width"
    )
    assert {f"needle_hunk_recall.control.1024.depth.{b}"
            for b in ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")} <= set(metrics)


def test_the_gate_length_arm_must_be_the_gate_suite(monkeypatch):
    from types import SimpleNamespace

    cases = build_suite(target_tokens=1024, cases_per_depth=1, seed=0)
    built = rft.NeedleSuite(cases, [], {}, [1] * len(cases), digest="b" * 64)
    monkeypatch.setattr(rft, "prepare_needle", lambda *a, **k: built)

    def never(*a, **k):
        raise AssertionError("a checkpoint was loaded for a refused sweep")

    monkeypatch.setattr(rft, "_checkpoint_step", never)
    args = SimpleNamespace(needle_control=(1024, rft.NEEDLE_TARGET_TOKENS))
    val = SimpleNamespace(reader=None, letter_id={})
    gate = rft.NeedleSuite(cases, [], {}, [1] * len(cases), digest="a" * 64)
    with pytest.raises(SystemExit, match="not the gate suite's"):
        rft.run_needle_control(
            args, reader=None, val=val, device="cpu", ledger=None,  # type: ignore[arg-type]
            config=DataConfig(), gate_suite=gate, reasons_for=lambda *a: [],
        )


def test_a_control_row_is_its_own_quick_family_and_names_no_eval_row(monkeypatch):
    """The promotion join treats a row naming recipe.eval_row_id as a supplement of that
    eval row, and a quick row in a family refuses the whole family. So the sweep's row has
    a recipe of its own and is quick by construction."""
    from types import SimpleNamespace

    import numpy as np

    cases = build_suite(target_tokens=1024, cases_per_depth=1, seed=0)
    suite = rft.NeedleSuite(
        cases, [SimpleNamespace(tokens=np.zeros((1, 1088)))] * len(cases), {},
        [900] * len(cases), digest="a" * 64,
    )
    monkeypatch.setattr(rft, "prepare_needle", lambda *a, **k: suite)
    ft = {"row_id": "ft1", "metrics": {"corpus.plan_max_width": {"value": 1625},
                                       "train.termination": {"value": "steps_exhausted"}}}
    meta = {"sidecar": {"digest": "c" * 16}}
    monkeypatch.setattr(rft, "_checkpoint_step", lambda *a, **k: (None, ft, {}, 0, meta))
    monkeypatch.setattr(
        rft, "needle_predictions",
        lambda step, s, letter_id: _decoded(s.cases, hit_every=1, abstain_every=2),
    )
    captured: dict[str, object] = {}

    class Recorder:
        row = SimpleNamespace(row_id="row-1")
        noul_rate = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def metric(self, name, state):
            captured.setdefault("metrics", {})[name] = state  # type: ignore[union-attr]

        def gate(self, name, state):
            raise AssertionError(f"the control recorded the gate {name}")

    def recorder(ledger, **kw):
        captured.update(kw)
        return Recorder()

    monkeypatch.setattr(rft, "_recorder", recorder)
    monkeypatch.setattr(rft, "_cost", lambda **k: None)
    args = SimpleNamespace(
        needle_control=(1024,), score_dtype="fp32",
        score_checkpoint=Path("epoch-seed0-cuda.json"), usd_per_hour=2.29,
        usd_per_gpu_hour=None, instance="gh200", wall_clock_cap_s=600.0,
    )
    reader = SimpleNamespace(header=SimpleNamespace(shard_hash=lambda: "s" * 64))
    val = SimpleNamespace(reader=reader, letter_id={})
    row_id, lines, seed = rft.run_needle_control(
        args, reader=reader, val=val, device="cuda", ledger=None,  # type: ignore[arg-type]
        config=DataConfig(), gate_suite=suite, reasons_for=lambda *a: [],
    )
    recipe = captured["recipe"]
    assert row_id == "row-1" and captured["run_kind"] == "eval"
    assert seed == 0, "one seed's control row is at its checkpoint's seed"
    assert recipe["scored_checkpoint"] == f"epoch-seed0-cuda.json:{'c' * 16}"
    assert captured["metrics"]["ft_run_row_id"].value == "ft1"
    assert "ft_run_row_ids" not in captured["metrics"] and "averaged" not in recipe
    assert "eval_row_id" not in recipe, "a control naming the eval row would join its family"
    assert recipe["tag"] == "epoch-needle-length-control" and "needle" not in recipe
    assert recipe["needle_control"]["target_tokens"] == [1024]
    assert rft.NEEDLE_CONTROL_QUICK_REASON in captured["quick_reasons"]
    assert "needle_hunk_recall.control.1024" in captured["metrics"]
    assert len(lines) == len(cases) and all(v["target_tokens"] == 1024 for v in lines)
