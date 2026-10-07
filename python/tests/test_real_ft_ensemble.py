"""``tools/real_ft_run.py --score-checkpoint A.json B.json C.json``: a 3-seed LOGIT ensemble.

Fable I (2026-10-01): the candidate is a three-seed artifact, either the weight average or a
logit ensemble of the three seeds, scored on identical rows. The ensemble decodes every row
from the mean of the towers' log-probabilities over the slot's own rows -- letters and span
pointers alike -- and applies the existing rules to that mean: the abstention (the noul row
winning), the permuted second pass, the needle pointer and the OOD rule.

What is pinned here:

* the combine rule, in closed form (:func:`real_ft_run.combine_readouts`);
* an ensemble of one tower three times decodes exactly as that tower;
* the ORACLE: three separately scored seeds' own ``row_logits`` (and pointer scores), mean
  log-softmaxed in this test, give the ensemble's top row on every val row, in the first
  pass AND the permuted second pass -- run end to end through ``main()`` on three tiny
  ``--optimizer master`` checkpoints;
* the ensemble's eval row: tag ``ens3-score-val``, its ``ensemble`` block (seeds, every ft
  row, each tower's checkpoint and digests, the combine rule), protocol seed = suite seed,
  ``ft_run_row_ids``, quick only for the CPU, never promotable by itself;
* every refusal, before any tower is built.
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
import torch  # noqa: E402
from test_real_ft_score_checkpoint import (  # noqa: E402
    AVG_IDS,
    STEPS,
    SUITE_SEED,
    _avg_row,
    _ledger,
    _reader,
    _schedule,
    tiny_master_checkpoints,
)

from qd_data.config import DataConfig  # noqa: E402

# --- the combine rule ------------------------------------------------------------------------


def _readout(letters, starts=None, ends=None, plan_rows=(4,)):
    plan = None if starts is None else SimpleNamespace(runtime_rows=list(plan_rows))
    return rft.BatchReadout(
        letters={r: torch.tensor(v, dtype=torch.float32) for r, v in letters.items()},
        plan=plan,
        starts=None if starts is None else [torch.tensor(s) for s in starts],
        ends=None if ends is None else [torch.tensor(e) for e in ends],
        span_index={} if starts is None else {1: 0},
    )


def test_the_ensemble_is_the_mean_of_per_tower_log_softmax_over_the_slots_rows():
    a = _readout({0: [2.0, 0.0, 1.0]}, starts=[[0.0, 3.0, 1.0, 0.5]], ends=[[1.0, 0.0, 0.0, 2.0]])
    b = _readout({0: [0.0, 2.5, 1.0]}, starts=[[2.0, 0.0, 0.0, 0.0]], ends=[[0.0, 0.0, 0.0, 1.0]])
    out = rft.combine_readouts([a, b])

    def mean_lsm(*rows):
        stacked = torch.stack([torch.log_softmax(torch.tensor(r, dtype=torch.float64), -1)
                               for r in rows])
        return stacked.mean(0)

    assert out.letters[0].dtype == torch.float64
    assert torch.allclose(out.letters[0], mean_lsm([2.0, 0.0, 1.0], [0.0, 2.5, 1.0]))
    assert torch.allclose(out.starts[0], mean_lsm([0.0, 3.0, 1.0, 0.5], [2.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(out.ends[0], mean_lsm([1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0]))
    # The argmax of the mean log-probabilities is the argmax of the mean raw scores: each
    # tower's log-softmax is its scores minus one constant.
    raw = (torch.tensor([2.0, 0.0, 1.0]) + torch.tensor([0.0, 2.5, 1.0])) / 2
    assert int(out.letters[0].argmax()) == int(raw.argmax()) == 1
    assert out.span_index == {1: 0}


def test_towers_that_read_one_batch_differently_are_refused():
    base = _readout({0: [1.0, 0.0]})
    with pytest.raises(ValueError, match="two or more"):
        rft.combine_readouts([base])
    with pytest.raises(ValueError, match="different rows"):
        rft.combine_readouts([base, _readout({1: [1.0, 0.0]})])
    with pytest.raises(ValueError, match="different row counts"):
        rft.combine_readouts([base, _readout({0: [1.0, 0.0, 2.0]})])
    spans = _readout({}, starts=[[0.0, 1.0]], ends=[[0.0, 1.0]], plan_rows=(2,))
    with pytest.raises(ValueError, match="planned"):
        rft.combine_readouts([
            spans, _readout({}, starts=[[0.0, 1.0]], ends=[[0.0, 1.0]], plan_rows=(3,)),
        ])


def _device_plan(device: str, counts, *, gold=None, requires_grad: bool = False):
    """A real :class:`qd_train.heads.SpanPlan` whose tensors live on ``device``, as
    ``plan_span_batch(..., device=step.device)`` builds it for a tower on the GPU."""
    from qd_train.heads import SpanPlan

    k, width = len(counts), max(counts)

    def t(values, dtype):
        return torch.as_tensor(values, dtype=dtype, device=device)

    pos = t([[3 * j for j in range(width)] for _ in range(k)], torch.float32 if requires_grad
            else torch.int64)
    if requires_grad:
        pos.requires_grad_(True)
    return SpanPlan(
        candidate_pos=pos,
        candidate_valid=t([[j < c for j in range(width)] for c in counts], torch.bool),
        n_candidates=t(list(counts), torch.int64),
        query_index=t([7] * k, torch.int64),
        gold_start=t(list(gold or [0] * k), torch.int64),
        gold_end=t(list(gold or [0] * k), torch.int64),
        abstaining=t([False] * k, torch.bool),
    )


def _device_readout(device: str, plan) -> rft.BatchReadout:
    rows = int(plan.runtime_rows[0])
    return rft.BatchReadout(
        letters={0: torch.tensor([1.0, 0.0], device=device)}, plan=plan,
        starts=[torch.arange(rows, dtype=torch.float32, device=device)],
        ends=[torch.arange(rows, dtype=torch.float32, device=device)], span_index={1: 0},
    )


def _accelerator() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.skipif(_accelerator() is None, reason="neither CUDA nor MPS is available here")
def test_towers_on_a_gpu_compare_their_span_plans_on_the_host():
    """J4's ens3 gate row on the GH200 (2026-10-01) died in combine_readouts: np.asarray of the
    plan's runtime_rows, a cuda tensor. An MPS tensor refuses np.asarray the same way, so this
    runs the failing comparison on whichever accelerator this host has."""
    device = _accelerator()
    assert device is not None
    a = _device_readout(device, _device_plan(device, [3]))
    b = _device_readout(device, _device_plan(device, [3]))
    out = rft.combine_readouts([a, b])
    # The mean is taken on the host in float64 (MPS has no float64), from the towers' own
    # values: two identical towers average to their own log-softmax.
    assert out.plan is a.plan and out.starts is not None
    assert out.starts[0].dtype == torch.float64 and out.starts[0].device.type == "cpu"
    expected = torch.log_softmax(torch.arange(4, dtype=torch.float64), dim=-1)
    assert torch.allclose(out.starts[0], expected)
    with pytest.raises(ValueError, match="planned"):
        rft.combine_readouts([a, _device_readout(device, _device_plan(device, [4]))])
    with pytest.raises(ValueError, match="planned"):
        rft.combine_readouts([a, _device_readout(device, _device_plan(device, [3], gold=[2]))])


def test_span_plans_are_compared_field_for_field_on_any_tensor():
    """Device-independent: a plan tensor numpy cannot read directly (one that requires grad
    raises from np.asarray on any device) is read on the host; and two plans of one batch
    are the same plan in every field, not only in their row counts -- the same counts with
    a different gold row are a different batch."""
    a = _device_readout("cpu", _device_plan("cpu", [3], requires_grad=True))
    b = _device_readout("cpu", _device_plan("cpu", [3], requires_grad=True))
    rft.combine_readouts([a, b])
    same_rows = _device_readout("cpu", _device_plan("cpu", [3], gold=[1]))
    plain = _device_readout("cpu", _device_plan("cpu", [3]))
    with pytest.raises(ValueError, match="planned"):
        rft.combine_readouts([plain, same_rows])
    assert rft.same_span_plan(None, None) and not rft.same_span_plan(plain.plan, None)


def test_a_tower_ensemble_is_bounded_by_its_narrowest_tower_and_one_device():
    steps = [SimpleNamespace(device="cpu", max_width=w) for w in (512, 256, 1024)]
    assert rft.TowerEnsemble(steps).max_width == 256
    with pytest.raises(ValueError, match="two or more"):
        rft.TowerEnsemble(steps[:1])
    with pytest.raises(ValueError, match="different devices"):
        rft.TowerEnsemble([SimpleNamespace(device="cpu", max_width=8),
                           SimpleNamespace(device="cuda", max_width=8)])


# --- end to end on three tiny master checkpoints ---------------------------------------------


def _score_main(tiny, tmp_path: Path, label: str, extra: list[str]) -> tuple[Path, Path]:
    from test_real_ft_shuffled_label import REV

    ledger = tmp_path / f"eval-{label}.jsonl"
    verdicts = tmp_path / f"verdicts-{label}.jsonl"
    assert rft.main([
        "--out", str(tiny.out), "--rev", REV, "--devices", "cpu", "--score-val",
        "--real-backbone", str(tiny.snapshot), "--ft-ledger", str(tiny.ft_ledger),
        "--ledger", str(ledger), "--tokenizer-json", str(tiny.tokenizer),
        "--verdicts-out", str(verdicts), *extra,
    ]) == 0
    return ledger, verdicts


def _lines(path: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = json.loads(raw)
        out[(line["row_id"], line["slot_name"])] = line
    return out


def _mean_log_softmax(rows: list[list[float]]) -> torch.Tensor:
    return torch.stack([
        torch.log_softmax(torch.tensor(r, dtype=torch.float64), -1) for r in rows
    ]).mean(0)


def _val_and_steps(tiny):
    """The val set, its permuted second pass and the three loaded towers, built the way main
    builds them, for decodes this test makes directly."""
    from test_real_ft_shuffled_label import REV

    from qd_train.shards import ShardReader

    config = DataConfig()
    reader = ShardReader(tiny.out / "shards" / "train", config=config, repo_root=tiny.out,
                         expect_rev=REV)
    labels, _ = rft._labels(tiny.train, config=config)
    labels = rft.pair_labels(reader, labels, require_index=False)
    letter_id = rft.vocab_letter_ids(
        reader, tokenizer_json=tiny.tokenizer, letter_id=rft._letter_ids(reader, labels)
    )
    val = rft.open_val_set(tiny.out, config=config, rev=REV, rows=list(tiny.val),
                           train=reader, letter_id=letter_id)
    second = rft.prepare_second_pass(val, reader=val.reader, tokenizer_json=tiny.tokenizer,
                                     seed=config.seed)
    steps = []
    for seed, path, row_id in zip((0, 1, 2), tiny.paths, tiny.ids, strict=True):
        args = argparse.Namespace(
            score_checkpoint=path, ft_row_id=row_id, seeds=[seed], ft_ledger=tiny.ft_ledger,
            real_backbone=tiny.snapshot, score_dtype="fp32",
        )
        step, *_ = rft._checkpoint_step(args, reader=reader, val=val, device="cpu",
                                        eval_widths=[], suite_seed=config.seed)
        steps.append(step)
    return val, second, steps


def test_every_span_verdict_carries_both_pointer_distributions_and_their_gold(
    tmp_path, monkeypatch
):
    """GAP-CALIB-SPAN-NOT-FITTABLE-FROM-VERDICTS: a span entry is fitted from the verdict file,
    so every span line ``--verdicts-out`` writes carries what the fit reads -- the start and
    end pointers' scores over the runtime's rows and each pointer's gold row -- with no flag,
    as every letter line carries ``row_logits`` and ``gold_row``."""
    tiny = tiny_master_checkpoints(tmp_path, monkeypatch)
    val, _, steps = _val_and_steps(tiny)
    scored = rft._decode(steps[0], val.plan, val.labels_for, val.letter_id)
    spans = [v for v in scored["verdicts"] if v["kind"] == "span"]
    assert spans, "the tiny val set holds span rows"
    lines = {
        line["row_id"]: line
        for line in rft._verdict_lines(scored, eval_row_id="e1", seed=0)
        if line["kind"] == "span"
    }
    for v in spans:
        rows, noul = int(v["rows"]), int(v["noul_row"])
        assert noul == rows - 1
        for key in ("start_logits", "end_logits"):
            assert len(v[key]) == rows, (key, v["row_id"])
        gold = (int(v["gold_start"]), int(v["gold_end"]))
        if v["expected_abstain"]:
            assert gold == (noul, noul)
        else:
            assert 0 <= gold[0] <= gold[1] < noul
        tops = [max(range(rows), key=v[key].__getitem__) for key in ("start_logits", "end_logits")]
        assert tops == list(v["top"]), "the written scores are the ones the verdict read"
        line = lines[v["row_id"]]
        for key in ("start_logits", "end_logits", "gold_start", "gold_end", "rows", "noul_row"):
            assert line[key] == v[key], key


def test_an_ensemble_of_one_tower_three_times_decodes_as_that_tower(tmp_path, monkeypatch):
    tiny = tiny_master_checkpoints(tmp_path, monkeypatch)
    val, _, steps = _val_and_steps(tiny)
    one = rft._decode(steps[0], val.plan, val.labels_for, val.letter_id)
    ens = rft._decode(
        rft.TowerEnsemble([steps[0]] * 3), val.plan, val.labels_for, val.letter_id
    )
    assert len(one["verdicts"]) == len(ens["verdicts"]) > 0
    kinds = set()
    for a, b in zip(one["verdicts"], ens["verdicts"], strict=True):
        kinds.add(a["kind"])
        for key in ("row_id", "kind", "top", "correct", "abstain_correct", "noul_row", "rows"):
            assert a[key] == b[key], (key, a["row_id"])
        if a["kind"] == "span":
            for key in ("start_logits", "end_logits"):
                assert torch.allclose(torch.tensor(b[key], dtype=torch.float64),
                                      _mean_log_softmax([a[key]]))
            continue
        assert b["noul_probability"] == pytest.approx(a["noul_probability"], abs=1e-6)
        # One tower's log-softmax: its raw row_logits shifted by one constant.
        assert torch.allclose(torch.tensor(b["row_logits"], dtype=torch.float64),
                              _mean_log_softmax([a["row_logits"]]))
    assert kinds == {"choice", "span"}


def test_three_separately_scored_seeds_predict_the_ensembles_every_verdict(
    tmp_path, monkeypatch
):
    """The oracle. Each seed is scored on its own through main(); this test combines their
    own row_logits and pointer scores by the stated rule and must land on the ensemble's top
    row for every val row, in the first pass and in the permuted second pass. A decode that
    applied the abstention or the second pass to anything but the averaged log-probabilities
    -- one tower's scores, raw-logit means of the permuted pass, the first tower's pointer --
    fails here."""
    tiny = tiny_master_checkpoints(tmp_path, monkeypatch)
    singles = [
        _lines(_score_main(tiny, tmp_path, f"seed{s}", [
            "--seeds", str(s), "--score-checkpoint", str(tiny.paths[s]),
            "--ft-row-id", tiny.ids[s],
        ])[1])
        for s in (0, 1, 2)
    ]
    ledger, verdicts = _score_main(tiny, tmp_path, "ens3", [
        "--seeds", "0", "1", "2", "--score-checkpoint", *map(str, tiny.paths),
        "--ft-row-id", *tiny.ids,
    ])
    ens = _lines(verdicts)
    assert set(ens) == set(singles[0]) and ens
    letter_rows = 0
    for key, line in ens.items():
        if line["kind"] == "span":
            continue
        letter_rows += 1
        mean = _mean_log_softmax([s[key]["row_logits"] for s in singles])
        assert line["top"] == int(mean.argmax()), key
        assert torch.allclose(torch.tensor(line["row_logits"], dtype=torch.float64), mean)
        assert line["noul_probability"] == pytest.approx(
            float(torch.softmax(mean, -1)[line["noul_row"]]), abs=1e-12
        )
        assert line["correct"] == (line["top"] == line["gold_row"])
    assert letter_rows > 0
    # Agreement across seeds would make the oracle vacuous: on this fixture the seeds differ.
    tops = {k: tuple(s[k]["top"] for s in singles) for k in ens if ens[k]["kind"] != "span"}
    assert any(len(set(t)) > 1 for t in tops.values()) or any(
        s[k]["row_logits"] != singles[0][k]["row_logits"] for s in singles[1:] for k in tops
    )

    # The permuted second pass and the span pointers: the ensemble's own decode (what main
    # ran, as its top_permuted shows) against the seeds' decodes combined here.
    val, second, steps = _val_and_steps(tiny)
    assert second.not_run is None and second.perms
    per_seed = [
        rft._decode(s, second.batches, second.labels_for, val.letter_id) for s in steps
    ]
    ens_second = rft._decode(rft.TowerEnsemble(steps), second.batches, second.labels_for,
                             val.letter_id)
    by_key = [{(v["row_id"], v["slot_name"]): v for v in d["verdicts"]} for d in per_seed]
    asked = 0
    for v in ens_second["verdicts"]:
        key = (v["row_id"], v["slot_name"])
        mean = _mean_log_softmax([d[key]["row_logits"] for d in by_key])
        assert v["top"] == int(mean.argmax()), key
        if key in second.perms:
            asked += 1
            assert ens[key]["top_permuted"] == v["top"], key
    assert asked == len(second.perms) > 0
    pointer = [rft._decode(s, val.plan, val.labels_for, val.letter_id) for s in steps]
    pointer_by_key = [{(v["row_id"], v["slot_name"]): v for v in d["verdicts"]} for d in pointer]
    spans = 0
    for key, line in ens.items():
        if line["kind"] != "span":
            continue
        spans += 1
        start = _mean_log_softmax([p[key]["start_logits"] for p in pointer_by_key])
        end = _mean_log_softmax([p[key]["end_logits"] for p in pointer_by_key])
        assert line["top"] == [int(start.argmax()), int(end.argmax())], key
    assert spans > 0

    # The row itself.
    from qd_train.ledger import Ledger
    from qd_train.run_control import Checkpoint

    (row,) = Ledger(ledger).rows()
    recipe = dict(row.recipe or {})
    assert recipe["tag"] == "ens3-score-val" and row.run_kind == "eval"
    block = recipe["ensemble"]
    bodies = [Checkpoint.read_body(p) for p in tiny.paths]
    assert block == {
        "seeds": [0, 1, 2], "ft_row_ids": tiny.ids, "n_towers": 3,
        "towers": [
            {"checkpoint": p.name, "seed": s, "sidecar_digest": b["sidecar"]["digest"],
             "payload_digest": b["payload_digest"]}
            for s, p, b in zip((0, 1, 2), tiny.paths, bodies, strict=True)
        ],
        "combine": rft.ENSEMBLE_COMBINE, "protocol_seed": SUITE_SEED, "suite_seed": SUITE_SEED,
    }
    assert recipe["scored_checkpoint"] == ",".join(
        f"{p.name}:{b['sidecar']['digest']}" for p, b in zip(tiny.paths, bodies, strict=True)
    )
    assert "averaged" not in recipe and recipe["score_dtype"] == "fp32"
    assert row.protocol.seed == SUITE_SEED
    assert row.metrics["ft_run_row_ids"].value == ",".join(tiny.ids)
    assert "ft_run_row_id" not in row.metrics
    only_the_device = rft.quick_reasons(
        tag="epoch", device="cpu", real_backbone=True,
        corpus=rft.CorpusFacts(snapshot_not_run=None, history_rows={}),
    )
    assert row.quick and row.quick_reason == "; ".join(only_the_device)
    assert "human's decision" in row.notes and "LOGIT ENSEMBLE" in row.notes
    assert {"permutation_consistency", "ece", "needle_hunk_recall", "ood_abstain"} <= set(
        row.gates
    )
    lines = [json.loads(x) for x in verdicts.read_text(encoding="utf-8").splitlines()]
    assert {(v["eval_row_id"], v["seed"]) for v in lines} == {(row.row_id, SUITE_SEED)}


def test_an_ensemble_of_two_seeds_is_quick_for_its_seed_count():
    model = rft.ScoredModel(
        tag="ens2", scored_checkpoint="a:b,c:d", terminations=("steps_exhausted",) * 2,
        ft_row_ids=("r0", "r1"), block_key="ensemble", block={}, described="x",
    )
    assert model.seed_reasons() == [
        "an ensemble of 2 seeds: rule 8 marks fewer than 3 seeds quick"
    ]
    assert model.ft_row_metric()[0] == "ft_run_row_ids"


# --- refusals, before any tower is built -----------------------------------------------------


def _tower_bodies(tmp_path: Path, seeds=(0, 1, 2), *, device: str = "cuda",
                  steps: int = STEPS, sidecar: bool = True, arm: str = "epoch") -> list[Path]:
    """One JSON body (and a two-number sidecar) per seed, written by ``Checkpoint.write``."""
    import struct

    from qd_train.run_control import Checkpoint, LossLog, Position, TensorRef

    paths = []
    for s in seeds:
        path = tmp_path / "ckpt" / f"{arm}-seed{s}-{device}.json"
        state: dict[str, object] = {"span_weight": 1.0, "vocab_size": 8}
        if sidecar:
            state["tower"] = {"w": TensorRef("float32", (2,), struct.pack("<2f", s, 1.0))}
            state["span_head"] = {"b": TensorRef("float32", (1,), struct.pack("<f", s))}
        Checkpoint(
            position=Position(epoch=0, index=steps), optimizer_step=steps, seed=s,
            schedule=_schedule(), loss_log=LossLog().snapshot(),
            consumed_digest=f"{s + 1:064x}", model_state=state,
        ).write(path)
        paths.append(path)
    return paths


def _ens_args(paths, ledger, ids=None, seeds=(0, 1, 2)) -> argparse.Namespace:
    return argparse.Namespace(
        score_checkpoint=list(paths), ft_row_id=[AVG_IDS[s] for s in (0, 1, 2)] if ids is None
        else list(ids), seeds=list(seeds), ft_ledger=ledger,
        real_backbone=Path("/models/snap"), score_dtype="fp32",
    )


class _Reached(Exception):
    """Raised in place of building a tower: every pairing check before it passed."""


def _score(args, monkeypatch, *, reached=None):
    import numpy as np

    def never(**kwargs):
        raise AssertionError("a tower was built for an ensemble that cannot be paired")

    monkeypatch.setattr(rft, "_real_step", reached or never)
    val = SimpleNamespace(plan=[SimpleNamespace(tokens=np.zeros((1, 64)))])
    return rft._score_checkpoint(
        args, reader=_reader(), val=val, device="cuda", ledger=None,  # type: ignore[arg-type]
        reasons_for=lambda *a, **k: [], second_pass=None,  # type: ignore[arg-type]
        needle_suite=rft.NeedleSuite([], [], {}, [], not_run="no"),
        ood_suite=rft.OodSuite([], None, None, not_run="no"), suite_seed=SUITE_SEED,
    )


def test_a_well_formed_ensemble_passes_every_pairing_check_and_reaches_its_towers(
    tmp_path, monkeypatch
):
    """The positive control for every refusal below: the same fixture, unaltered, reads each
    tower's verified weights and builds a step for each, at the protocol seed."""
    paths = _tower_bodies(tmp_path)
    ledger = _ledger(tmp_path, *(_avg_row(s) for s in (0, 1, 2)))
    built: list[dict] = []

    def reached(**kwargs):
        built.append(kwargs)
        if len(built) == 3:
            raise _Reached
        return SimpleNamespace(load_weights=lambda w: None, device="cuda", max_width=64), None, None

    with pytest.raises(_Reached):
        _score(_ens_args(paths, ledger), monkeypatch, reached=reached)
    assert [k["seed"] for k in built] == [SUITE_SEED] * 3
    assert {k["total_steps"] for k in built} == {STEPS}


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("one_tower", "two or more checkpoints"),
        ("an_average", "per-seed epoch checkpoints"),
        ("same_path", "one checkpoint twice"),
        ("too_few_ids", "names 2 ft row"),
        ("same_seed", "one seed twice"),
        ("seeds_out_of_order", "--seeds says"),
        ("missing_row", "0 rows match"),
        ("quick_row", "quick"),
        ("same_row", "one row twice"),
        ("other_recipe", "recipe_hash"),
        ("other_snapshot", "data_snapshot_hash"),
        ("other_steps", "optimizer_steps"),
        ("other_shard_set", "shard_hash"),
        ("rows_out_of_order", "protocol seed"),
        ("memorise_arm", "only an epoch checkpoint"),
        ("body_at_other_step", "optimizer step"),
        ("no_sidecar", "declares no tensor sidecar"),
    ],
)
def test_an_ensemble_that_cannot_be_paired_is_refused_before_any_tower(
    tmp_path, monkeypatch, case, match
):
    rows = {s: _avg_row(s) for s in (0, 1, 2)}
    paths = _tower_bodies(
        tmp_path, steps=STEPS - 1 if case == "body_at_other_step" else STEPS,
        sidecar=case != "no_sidecar", arm="memorise" if case == "memorise_arm" else "epoch",
    )
    ids: list[str] | None = None
    seeds = (0, 1, 2)
    if case == "one_tower":
        paths, ids, seeds = paths[:1], [AVG_IDS[0]], (0,)
    elif case == "an_average":
        paths = [*paths[:2], tmp_path / "avg.safetensors"]
    elif case == "same_path":
        paths = [paths[0], paths[1], paths[0]]
    elif case == "too_few_ids":
        ids = [AVG_IDS[0], AVG_IDS[1]]
    elif case == "same_seed":
        paths = [*paths[:2], _tower_bodies(tmp_path, (0,), device="mps")[0]]
    elif case == "seeds_out_of_order":
        seeds = (1, 0, 2)
    elif case == "missing_row":
        del rows[2]
    elif case == "quick_row":
        rows[1] = _avg_row(1, quick=True)
    elif case == "same_row":
        ids = [AVG_IDS[0], AVG_IDS[0], AVG_IDS[2]]
    elif case == "other_recipe":
        rows[2] = _avg_row(2, recipe_hash="f" * 64)
    elif case == "other_snapshot":
        rows[2] = _avg_row(2, snapshot="9" * 64)
    elif case == "other_steps":
        rows[1] = _avg_row(1, steps=STEPS - 1)
    elif case == "other_shard_set":
        rows = {s: _avg_row(s, shard_hash="b" * 64) for s in (0, 1, 2)}
    elif case == "rows_out_of_order":
        ids = [AVG_IDS[1], AVG_IDS[0], AVG_IDS[2]]
    ledger = _ledger(tmp_path, *rows.values())
    args = _ens_args(paths, ledger, ids=ids, seeds=seeds)
    with pytest.raises(SystemExit, match=match):
        if case == "one_tower":
            rft._ensemble_inputs(args, reader=_reader(), suite_seed=SUITE_SEED)
        else:
            _score(args, monkeypatch)


def test_a_tower_that_changes_between_pairing_and_read_is_refused(tmp_path, monkeypatch):
    from qd_train.run_control import Checkpoint

    paths = _tower_bodies(tmp_path)
    ledger = _ledger(tmp_path, *(_avg_row(s) for s in (0, 1, 2)))
    real_read = Checkpoint.read_weights

    def read_then_swap(path, *, subtrees):
        weights, header = real_read(path, subtrees=subtrees)
        return weights, {**header, "payload_digest": "0" * 64}

    monkeypatch.setattr(Checkpoint, "read_weights", read_then_swap)
    with pytest.raises(SystemExit, match="changed between its pairing and its read"):
        _score(_ens_args(paths, ledger), monkeypatch)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--score-checkpoint", "/c/epoch-seed0-cuda.json", "/c/epoch-seed1-cuda.json",
          "--ft-row-id", "aaaaaaaa", "--seeds", "0", "1"], "one --ft-row-id and one --seeds"),
        (["--score-checkpoint", "/c/epoch-seed0-cuda.json", "/c/epoch-seed1-cuda.json",
          "--ft-row-id", "aaaaaaaa", "bbbbbbbb", "--seeds", "0"],
         "one --ft-row-id and one --seeds"),
        (["--score-checkpoint", "/c/epoch-seed0-cuda.json", "/c/avg.safetensors",
          "--ft-row-id", "aaaaaaaa", "bbbbbbbb", "--seeds", "0", "1"], "per-seed epoch"),
        (["--score-checkpoint", "/c/epoch-seed0-cuda.json",
          "--ft-row-id", "aaaaaaaa", "bbbbbbbb", "--seeds", "0"], "one --ft-row-id"),
    ],
)
def test_ensemble_argv_is_refused_before_anything_loads(tmp_path, extra, match):
    argv = ["--out", str(tmp_path), "--rev", "0" * 40, "--score-val", "--real-backbone", "/x",
            "--ft-ledger", "/l", "--devices", "cuda", "--instance", "gh200",
            "--usd-per-hour", "2.29", *extra]
    with pytest.raises(SystemExit, match=match):
        rft.main(argv)


def test_an_ensembles_needle_control_row_names_every_tower(tmp_path, monkeypatch):
    """--needle-control on an ensemble: the row names the ensemble as its score row does --
    every ft row, every tower's digests, tag ens3-needle-length-control -- and stays quick."""
    from test_real_ft_score_checkpoint import _avg_control_args, _run_avg_needle_control

    paths = _tower_bodies(tmp_path)
    ledger = _ledger(tmp_path, *(_avg_row(s) for s in (0, 1, 2)))
    args = _avg_control_args(paths[0], ledger)
    args.score_checkpoint = list(paths)
    args.ft_row_id = [AVG_IDS[s] for s in (0, 1, 2)]

    def step_reached(**kwargs):
        step = SimpleNamespace(load_weights=lambda w: None, device="cuda", max_width=4096)
        return step, None, None

    (_, lines, seed), captured, n_cases = _run_avg_needle_control(
        monkeypatch, args, step_reached=step_reached
    )
    recipe = captured["recipe"]
    assert recipe["tag"] == "ens3-needle-length-control" and seed == SUITE_SEED
    assert recipe["ensemble"]["ft_row_ids"] == [AVG_IDS[s] for s in (0, 1, 2)]
    assert [t["checkpoint"] for t in recipe["ensemble"]["towers"]] == [p.name for p in paths]
    assert "averaged" not in recipe
    assert captured["metrics"]["ft_run_row_ids"].value == ",".join(AVG_IDS[s] for s in (0, 1, 2))
    assert rft.NEEDLE_CONTROL_QUICK_REASON in captured["quick_reasons"]
    assert "LOGIT ENSEMBLE" in captured["notes"]
    assert len(lines) == 3 * n_cases
