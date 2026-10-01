"""The report-only composed slice as a ``--score-plan`` pass: ``composed``.

Fable G5(ii)'s slice ruling (relayed by the compose and longctx lanes, 2026-10-01): the slice
is scored for a report, never as a gate. The pass opens the slice before any model loads and
refuses it unless its header says ``report_only`` and refuse-gold, its shard hash is the one
build row ``dafe86af`` names, its exclusions are the measured list and its dropped rows agree
with that row; then, per kind, one decode, conditions 6 and 8's tables on a quick row, and
per-row lines kept apart from the kind's gate lines.

The tables themselves are ``test_composed_slice.py``'s. The real slice is opened by the last
test here when ``QD_SLICE_*`` name it (see :func:`_real_slice_env`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
from test_composed_slice import FIXTURE, _candidates  # noqa: E402
from test_real_ft_score_checkpoint import _CapturedRecorder  # noqa: E402

from qd_train import composed_slice as cs  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

SEED0 = {"name": "seed0", "checkpoints": ["/c/epoch-seed0-cuda.json"],
         "ft_row_ids": ["aaaaaaaa"], "seeds": [0], "passes": ["composed"]}
AVG = {"name": "avg", "checkpoints": ["/c/avg.safetensors"], "seeds": [0, 1, 2],
       "passes": ["composed"]}
ENS3 = {"name": "ens3", "checkpoints": [f"/c/epoch-seed{s}-cuda.json" for s in (0, 1, 2)],
        "ft_row_ids": ["aaaaaaaa", "bbbbbbbb", "cccccccc"], "seeds": [0, 1, 2],
        "passes": ["gates", "composed"]}


def _write_plan(tmp_path: Path, kinds: list[dict]) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"kinds": kinds}), encoding="utf-8")
    return path


def _argv(tmp_path: Path, plan: Path | None, *extra: str, device: str = "cuda") -> list[str]:
    return ["--out", str(tmp_path), "--rev", "0" * 40, "--score-val", "--real-backbone", "/x",
            "--ft-ledger", "/l", "--devices", device, "--instance", "gh200",
            "--usd-per-hour", "2.29",
            *(["--score-plan", str(plan)] if plan is not None else []), *extra]


# --- the plan and its argv ---------------------------------------------------------------------


def test_a_plan_kind_may_run_the_composed_pass_for_one_checkpoint_an_average_and_an_ensemble(
    tmp_path,
):
    plan = rft.read_score_plan(_write_plan(tmp_path, [SEED0, AVG, ENS3]))
    assert [k.passes for k in plan.kinds] == [("composed",), ("composed",), ("gates", "composed")]
    assert plan.wants("composed") and "composed" in rft.PLAN_PASSES


@pytest.mark.parametrize(
    ("kinds", "extra", "device", "match"),
    [
        ([SEED0], [], "cuda", "give both or neither"),
        ([dict(SEED0, passes=["ood"])], ["--ood", "--ood-general-record", "/g.json",
                                         "--composed-slice", "/s"], "cuda",
         "give both or neither"),
        ([SEED0], ["--composed-slice", "/s"], "mps", "score the slice on cuda"),
        ([SEED0], ["--composed-slice", "/s"], "cuda", None),
        ([AVG], ["--composed-slice", "/s"], "cuda", None),
        ([ENS3], ["--composed-slice", "/s"], "cuda", None),
    ],
)
def test_composed_plan_argv_is_checked_before_anything_loads(tmp_path, kinds, extra, device,
                                                            match):
    plan = _write_plan(tmp_path, kinds)
    if match is None:
        # Every argv check passed; the first disk read is resolving --rev, which no
        # repository here has.
        with pytest.raises(subprocess.CalledProcessError, match="rev-parse"):
            rft.main(_argv(tmp_path, plan, *extra, device=device))
        return
    with pytest.raises(SystemExit, match=match):
        rft.main(_argv(tmp_path, plan, *extra, device=device))


def test_composed_slice_is_read_only_by_a_plan(tmp_path):
    argv = _argv(tmp_path, None, "--score-checkpoint", "/c/epoch-seed0-cuda.json",
                 "--ft-row-id", "aaaaaaaa", "--seeds", "0", "--composed-slice", "/s")
    with pytest.raises(SystemExit, match="read only by a --score-plan 'composed' pass"):
        rft.main(argv)


def test_a_kinds_slice_lines_have_a_file_of_their_own_and_it_is_never_overwritten(tmp_path):
    out = tmp_path / "suite.jsonl"
    assert rft.plan_output(out, "ens3") == tmp_path / "suite-ens3.jsonl"
    assert rft.plan_output(out, "ens3", "composed") == tmp_path / "suite-ens3.composed.jsonl"
    # A kind name holds no dot, so no kind's own file is another kind's part.
    assert rft.PLAN_KIND_NAME.fullmatch("ens3.composed") is None
    plan = _write_plan(tmp_path, [ENS3])
    rft.plan_output(out, "ens3", "composed").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"suite-ens3\.composed\.jsonl already exists"):
        rft.main(_argv(tmp_path, plan, "--composed-slice", "/s", "--suite-verdicts-out",
                       str(out)))


# --- the pinned build ------------------------------------------------------------------------


def test_the_pinned_build_row_names_the_pinned_set_and_corpus():
    row = rft.composed_slice_build(
        rft.COMPOSED_SLICE_LEDGER, rft.COMPOSED_SLICE_ROW,
        shard_hash=rft.COMPOSED_SLICE_SHARD_HASH,
    )
    reached = row.metrics["report_only_slice_rows"]
    assert isinstance(reached, Ran) and (reached.n, reached.n_total) == (1550, 1550)
    manifest = (rft.COMPOSED_SLICE_CORPUS / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == row.recipe["report_only_slice_manifest_sha256"]
    with pytest.raises(SystemExit, match="does not name the slice's shard hash"):
        rft.composed_slice_build(rft.COMPOSED_SLICE_LEDGER, rft.COMPOSED_SLICE_ROW,
                                 shard_hash="0" * 64)
    with pytest.raises(SystemExit, match="0 times, not once"):
        rft.composed_slice_build(rft.COMPOSED_SLICE_LEDGER, "not-a-row",
                                 shard_hash=rft.COMPOSED_SLICE_SHARD_HASH)


# --- the scorer's verdicts, as the slice's ------------------------------------------------------


def _fixture_slice(*, dropped: bool = True) -> rft.ComposedSlice:
    """The three fixture rows as a slice: the stub decoded on both slots, the clean row's span
    slot excluded (a gold collision in the measured list's bucket), the diag row dropped
    before write (or decoded, ``dropped=False``)."""
    cases = cs.load_cases([FIXTURE])
    by_class = {c.mutation_class: c for c in cases.values()}
    stub = next(c for c in cases.values() if c.mutation_class == "stub")
    clean = next(c for c in cases.values() if c.is_clean)
    diag = next(c for c in cases.values() if c.diag_half is not None)
    assert len({stub.row_id, clean.row_id, diag.row_id}) == 3, by_class

    def seq(case, length):
        cands = _candidates(case)
        gold = len(cands) if case.gold_line is None else case.gold_line
        return cs.SpanSequence(tuple(cands), gold, length)

    span = {stub.row_id: seq(stub, 3000)}
    choice = {stub.row_id: 2990, clean.row_id: 880}
    rows: tuple[tuple[cs.ComposedCase, str], ...] = ()
    if dropped:
        rows = ((diag, "dropped_before_write"),)
    else:
        span[diag.row_id] = seq(diag, 5000)
        choice[diag.row_id] = 4990
    return rft.ComposedSlice(
        reader=None, plan=[SimpleNamespace(tokens=SimpleNamespace(shape=(1, 7801)))],  # type: ignore[arg-type,list-item]
        labels_for={}, letter_id={}, cases=cases, span=span, choice_length=choice,
        slot_excluded={(clean.row_id, "defect_span"): "gold_shares_token"},
        row_exclusions=rows, recipe={"build_row": "b" * 36, "shard_hash": "s" * 64},
    )


def _scorer_verdicts(composed: rft.ComposedSlice, *, logits: bool = False) -> list[dict]:
    """``_decode``'s verdicts for every written slot: the span pointer on the gold."""
    out: list[dict] = []
    for row_id, seq in composed.span.items():
        noul = seq.head_rows
        out.append({"row_id": row_id, "slot_name": "defect_span", "kind": "span",
                    "top": [seq.gold_head_row, seq.gold_head_row], "noul_row": noul,
                    "correct": True,
                    **({"start_logits": [0.5] * (noul + 1), "end_logits": [0.25] * (noul + 1)}
                       if logits else {})})
    for row_id in composed.choice_length:
        out.append({"row_id": row_id, "slot_name": "defect_class", "kind": "choice",
                    "top": 0, "gold_row": 0, "correct": True, "row_logits": [1.0, 0.0]})
    return out


def test_every_row_is_one_slice_verdict_and_one_line():
    composed = _fixture_slice()
    verdicts, lines = rft.composed_slice_verdicts(
        composed, _scorer_verdicts(composed), logits=False
    )
    assert len(verdicts) == len(lines) == 2, "the dropped row is counted, not decoded"
    by_id = {line["case_id"]: line for line in lines}
    stub, clean = (next(c for c in composed.cases.values() if p(c))
                   for p in (lambda c: c.mutation_class == "stub", lambda c: c.is_clean))
    assert by_id[stub.row_id]["hunk_hit"] is True
    assert by_id[stub.row_id]["predicted_lines"] == [stub.gold_line]
    assert by_id[clean.row_id]["span_excluded"] == "gold_shares_token"
    assert by_id[clean.row_id]["choice_row_logits"] == [1.0, 0.0]
    assert "start_logits" not in by_id[stub.row_id], "off, no pointer scores"
    m = cs.slice_metrics(verdicts, row_exclusions=composed.row_exclusions, corpus=composed.cases)
    assert m["composed.diag.seen_filler.rows_excluded.dropped_before_write"].value == 1 or (
        m["composed.diag.unseen.rows_excluded.dropped_before_write"].value == 1
    )


def test_the_pointer_scores_ride_on_the_line_under_the_flag_and_are_refused_absent():
    composed = _fixture_slice(dropped=False)
    _, lines = rft.composed_slice_verdicts(
        composed, _scorer_verdicts(composed, logits=True), logits=True
    )
    span_lines = [line for line in lines if line["span_excluded"] is None]
    assert len(span_lines) == 2 and all("start_logits" in line for line in span_lines)
    with pytest.raises(SystemExit, match="lacks start_logits, end_logits under --suite-logits"):
        rft.composed_slice_verdicts(composed, _scorer_verdicts(composed), logits=True)


def test_a_verdict_twice_or_for_no_slice_row_or_missing_is_refused():
    composed = _fixture_slice()
    good = _scorer_verdicts(composed)
    with pytest.raises(SystemExit, match="decoded twice"):
        rft.composed_slice_verdicts(composed, [*good, good[0]], logits=False)
    diag = next(c for c, _ in composed.row_exclusions)
    stray = {**good[-1], "row_id": diag.row_id}
    with pytest.raises(SystemExit, match="verdicts for no slice row"):
        rft.composed_slice_verdicts(composed, [*good, stray], logits=False)
    with pytest.raises(SystemExit, match="never neither or both"):
        rft.composed_slice_verdicts(composed, good[1:], logits=False)


def _run_args() -> SimpleNamespace:
    return SimpleNamespace(
        score_checkpoint=Path("epoch-seed0-cuda.json"), score_dtype="fp32", usd_per_hour=2.29,
        usd_per_gpu_hour=None, instance="gh200", wall_clock_cap_s=600.0, suite_logits=False,
    )


def test_the_slice_row_is_quick_holds_the_tables_and_no_gate(monkeypatch):
    composed = _fixture_slice()
    asked: dict[str, object] = {}

    def decode(step, plan, labels_for, letter_id, *, pointer_scores):
        asked.update(step=step, plan=plan, pointer_scores=pointer_scores)
        return {"verdicts": _scorer_verdicts(composed), "rows_not_decoded": 0,
                "letters_without_id": []}

    captured: dict[str, object] = {}

    def recorder(ledger, **kw):
        captured.update(kw)
        captured["recorder"] = _CapturedRecorder(captured)
        return captured["recorder"]

    monkeypatch.setattr(rft, "_decode", decode)
    monkeypatch.setattr(rft, "_recorder", recorder)
    monkeypatch.setattr(rft, "_cost", lambda **k: None)
    reader = SimpleNamespace(header=SimpleNamespace(shard_hash=lambda: "t" * 64))
    ft = {"row_id": "ft0", "metrics": {"train.termination": {"value": "steps_exhausted"}}}
    loaded = ("step", ft, {"lr": 1e-5}, 0, {"sidecar": {"digest": "d" * 64}})
    row_id, lines, seed = rft.run_composed_slice(
        _run_args(), loaded=loaded, reader=reader, composed=composed,  # type: ignore[arg-type]
        device="cuda", ledger=None, reasons_for=lambda tag, device, termination=None: [],  # type: ignore[arg-type]
        plan_note="Scored as kind 'seed0'.",
    )
    assert (row_id, seed, len(lines)) == ("control-row", 0, 2)
    assert asked == {"step": "step", "plan": composed.plan, "pointer_scores": False}
    metrics = captured["metrics"]
    assert metrics["ft_run_row_id"].value == "ft0"
    assert all(k == "ft_run_row_id" or k.startswith("composed.") for k in metrics)
    assert not any(k.startswith(("needle_hunk_recall", "ood_abstain", "val_top1")) for k in metrics)
    recipe = captured["recipe"]
    assert recipe["tag"] == "epoch-composed-slice" and recipe["composed_slice"] == composed.recipe
    assert rft.COMPOSED_SLICE_QUICK_REASON in captured["quick_reasons"]
    assert captured["run_kind"] == "eval" and "Scored as kind 'seed0'." in captured["notes"]
    assert "Report-only" in captured["notes"]
    assert isinstance(captured["recorder"].noul_rate, NotRun)  # type: ignore[union-attr]

    def undecoded(*a, **k):
        return {"verdicts": [], "rows_not_decoded": 2, "letters_without_id": ["E"]}

    monkeypatch.setattr(rft, "_decode", undecoded)
    with pytest.raises(SystemExit, match="2 slice sequences were not decoded"):
        rft.run_composed_slice(
            _run_args(), loaded=loaded, reader=reader, composed=composed,  # type: ignore[arg-type]
            device="cuda", ledger=None, reasons_for=lambda *a, **k: [],  # type: ignore[arg-type]
        )


def test_a_plan_writes_each_kinds_slice_lines_apart_from_its_gate_lines(tmp_path, monkeypatch):
    composed = _fixture_slice()
    plan = rft.ScorePlan(path=Path("p.json"), sha256="0" * 64, kinds=(
        rft.PlanKind("seed0", (Path("/c/epoch-seed0-cuda.json"),), ("aaaaaaaa",), (0,),
                     ("composed",)),
        rft.PlanKind("ens3", tuple(Path(f"/c/epoch-seed{s}-cuda.json") for s in (0, 1, 2)),
                     ("aaaaaaaa", "bbbbbbbb", "cccccccc"), (0, 1, 2), ("gates", "composed")),
    ))
    suite = tmp_path / "s.jsonl"
    widths_seen: list[list[int]] = []

    def load(args, **k):
        widths_seen.append(list(k["eval_widths"]))
        return ("model", len(widths_seen))

    def score(args, **k):
        gate = rft.SuiteGate("needle_hunk_recall", NotRun(reason="x"), {}, "needle", None,
                             ({"suite": "needle", "case_id": "c0"},))
        return "gate-row", {"verdicts": []}, None, [gate], 0

    slice_calls: list[object] = []

    def run_slice(args, *, loaded, composed, plan_note, **k):
        slice_calls.append((loaded, composed, plan_note))
        return f"slice-row-{len(slice_calls)}", ({"suite": "composed_slice", "case_id": "x"},), 3

    monkeypatch.setattr(rft, "_checkpoint_step", load)
    monkeypatch.setattr(rft, "_score_checkpoint", score)
    monkeypatch.setattr(rft, "run_composed_slice", run_slice)
    monkeypatch.setattr(rft, "release_device_cache", lambda: None)
    args = SimpleNamespace(verdicts_out=None, suite_verdicts_out=suite, score_plan=plan.path,
                           score_checkpoint=None, ft_row_id=None, seeds=[0, 1, 2])
    common = dict(
        reader=None, val=None, device="cuda", ledger=None, reasons_for=lambda *a: [],
        second_pass=None, needle_suite=rft.NeedleSuite([], [], {}, [], not_run="x"),
        ood_suite=rft.OodSuite([], None, None, not_run="x"), suite_seed=0,
    )
    recorded = rft.run_score_plan(args, plan, composed=composed, **common)  # type: ignore[arg-type]
    assert recorded == [("seed0", "composed", "slice-row-1"), ("ens3", "gates", "gate-row"),
                        ("ens3", "composed", "slice-row-2")]
    assert widths_seen == [[7801], [7801]], "the step is sized for the slice's widest batch"
    assert [c[0] for c in slice_calls] == [("model", 1), ("model", 2)]
    assert all(c[1] is composed for c in slice_calls)
    assert not rft.plan_output(suite, "seed0").exists(), "a composed-only kind has no gate lines"
    (line,) = [json.loads(x) for x in rft.plan_output(suite, "seed0", "composed").read_text(
        encoding="utf-8").splitlines()]
    assert line == {"eval_row_id": "slice-row-1", "seed": 3, "gate": rft.COMPOSED_SLICE_GATE,
                    "score_kind": "seed0", "suite": "composed_slice", "case_id": "x"}
    gate_lines = [json.loads(x) for x in rft.plan_output(suite, "ens3").read_text(
        encoding="utf-8").splitlines()]
    assert [g["gate"] for g in gate_lines] == ["needle_hunk_recall"], "no slice line among them"
    assert rft.plan_output(suite, "ens3", "composed").is_file()
    with pytest.raises(SystemExit, match="needs the slice"):
        rft.run_score_plan(args, plan, composed=None, **common)  # type: ignore[arg-type]


# --- the real slice (opt-in: it reads ~1.5 GB from the campaign directories) --------------------


def _real_slice_env() -> dict[str, Path] | None:
    """``QD_SLICE_OUT`` (the slice build's --out), ``QD_SLICE_REPO_ROOT`` (a checkout whose
    data/pool holds the slice corpus's examples, its base corpus, pool and licences) and
    ``QD_SLICE_TRAIN_OUT`` (v4's out dir, whose train set's remap the slice shares)."""
    names = ("QD_SLICE_OUT", "QD_SLICE_REPO_ROOT", "QD_SLICE_TRAIN_OUT")
    if not all(os.environ.get(n) for n in names):
        return None
    return {n: Path(os.environ[n]) for n in names}


@pytest.mark.skipif(_real_slice_env() is None, reason="QD_SLICE_* do not name the real slice")
def test_the_real_slice_opens_aligns_and_scores_its_gold_perfectly():
    from qd_data.config import DataConfig
    from qd_train.trainer import ft_supervision

    env = _real_slice_env()
    assert env is not None
    config = DataConfig()
    train = rft.ShardReader(env["QD_SLICE_TRAIN_OUT"] / "shards" / "train", config=config,
                            repo_root=env["QD_SLICE_TRAIN_OUT"])
    pool = env["QD_SLICE_REPO_ROOT"] / "data" / "pool"
    composed = rft.open_composed_slice(
        env["QD_SLICE_OUT"], corpus_dir=pool / "commitpackft-composed-slice-v1",
        download_root=pool / "commitpackft", repo_root=env["QD_SLICE_REPO_ROOT"],
        config=config, train=train, letter_id={},
    )
    assert len(composed.cases) == 1550 and len(composed.reader) == 3099
    assert len(composed.span) == 1549 and len(composed.choice_length) == 1550
    shared = sum(len(set(s.candidates)) != len(s.candidates) for s in composed.span.values())
    assert len(composed.span) - shared == 461, "the build's refuse-any survival count"
    assert composed.slot_excluded == dict(cs.MEASURED_EXCLUSIONS)
    assert composed.row_exclusions == ()
    assert max(int(b.tokens.shape[1]) for b in composed.plan) == 7801
    # The scorer answering every slot with the shard's own gold: every table at 1.0, so the
    # corpus's hunks, the shard's candidates and the head's rows are one mapping.
    verdicts = []
    for b, batch in enumerate(composed.plan):
        sup = ft_supervision(batch)
        heads = None if sup.span is None else rft.plan_span_batch(sup.span)
        span_k = {} if sup.span is None else {int(r): k for k, r in enumerate(sup.span.rows)}
        for r, label in enumerate(composed.labels_for[b]):
            if label.slot_kind == rft.SLOT_SPAN:
                assert heads is not None
                k = span_k[r]
                g = [int(heads.gold_start[k]), int(heads.gold_end[k])]
                verdicts.append({"row_id": label.row_id, "slot_name": label.slot_name,
                                 "top": g, "noul_row": int(heads.n_candidates[k]),
                                 "correct": True})
            else:
                verdicts.append({"row_id": label.row_id, "slot_name": label.slot_name,
                                 "correct": True, "row_logits": [0.0]})
    sv, lines = rft.composed_slice_verdicts(composed, verdicts, logits=False)
    m = cs.slice_metrics(sv, row_exclusions=composed.row_exclusions, corpus=composed.cases)
    assert len(sv) == len(lines) == 1550
    for s, n_any in (("val", 377), ("diag.seen_filler", 45), ("diag.unseen", 39)):
        assert m[f"composed.{s}.refuse_any.span_sequences"].value == n_any
        for pop in cs.POPULATIONS:
            assert m[f"composed.{s}.{pop}.all.all.hunk_hit"].value == 1.0
            assert m[f"composed.{s}.{pop}.all.all.span_top1"].value == 1.0
        assert m[f"composed.{s}.both_policies.all.all.choice_top1"].value == 1.0
    assert m["composed.diag.unseen.span_excluded.gold_shares_token"].value == 1
    assert dataclasses.asdict(cs.SpanSequence((1,), 0, 1)) == {
        "candidates": (1,), "gold_head_row": 0, "length_tokens": 1}
