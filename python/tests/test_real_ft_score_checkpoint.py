"""``tools/real_ft_run.py --score-checkpoint``: score a saved epoch checkpoint, train nothing.

The three phase-3 weight sets of 2026-09-30 were saved but never scored outside the process
that trained them. Scoring them later means pairing a file with the ft row that wrote it,
and every way that pairing could silently put the wrong weights under the wrong row is
refused before a tower loads.

J7 (phase 6) scores an AVERAGE of three seeds' weights, written by ``tools/ckpt_average.py``
as a ``.safetensors`` and its ``.manifest.json``. The pairing is then one file with three
ft rows, and the same refusal discipline holds: every row the manifest names must be in
the ledger, completed and not quick; the rows must be one configuration at one step; and
each source checkpoint's JSON body on disk must hash to what the manifest recorded.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import ckpt_average  # noqa: E402
import real_ft_run  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402

SHARD_HASH = "a" * 64
#: The suite and protocol seed every --score-checkpoint run builds its suites at.
SUITE_SEED = DataConfig().seed
_NO_NEEDLE = real_ft_run.NeedleSuite([], [], {}, [], not_run="--needle was not given")
_NO_OOD = real_ft_run.OodSuite([], None, None, not_run="--ood was not given")


def _ft_row(row_id: str = "7f2c11db-3eb8-4361-9620-6b164ec37f1e", **recipe_over) -> dict:
    recipe = {
        "tag": "epoch", "device": "cuda", "shard_hash": SHARD_HASH,
        "backbone_snapshot": "snap", "attn_implementation": "sdpa", "lr": 1e-5,
        "span_weight": 1.0, "optimizer_recipe": "master",
    }
    recipe.update(recipe_over)
    return {
        "row_id": row_id, "run_kind": "ft", "status": "completed", "recipe": recipe,
        "protocol": {"seed": 0},
        "metrics": {"train.optimizer_steps": {"value": 1505}},
    }


def _ledger(tmp_path: Path, *rows: dict) -> Path:
    path = tmp_path / "ft.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_the_ft_row_is_found_by_a_unique_prefix_and_refused_otherwise(tmp_path):
    ledger = _ledger(
        tmp_path, _ft_row(), _ft_row("7f2c11db-ffff-0000"),
        {**_ft_row("99999999-eval"), "run_kind": "eval"},
    )
    assert real_ft_run._ft_row(ledger, "7f2c11db-3eb8")["row_id"].startswith("7f2c11db-3eb8")
    with pytest.raises(SystemExit, match="at least 8"):
        real_ft_run._ft_row(ledger, "7f2c11d")
    with pytest.raises(SystemExit, match="2 rows match"):
        real_ft_run._ft_row(ledger, "7f2c11db")
    with pytest.raises(SystemExit, match="0 rows match"):
        real_ft_run._ft_row(ledger, "deadbeef")
    with pytest.raises(SystemExit, match="scores the model a completed ft row"):
        real_ft_run._ft_row(ledger, "99999999")


def _args(tmp_path: Path, ledger: Path, name: str = "epoch-seed0-cuda.json", seeds=(0,)):
    return argparse.Namespace(
        score_checkpoint=tmp_path / name, seeds=list(seeds), ft_ledger=ledger,
        ft_row_id="7f2c11db", real_backbone=Path("/models/snap"), score_dtype="fp32",
    )


def _reader(shard_hash: str = SHARD_HASH) -> SimpleNamespace:
    return SimpleNamespace(header=SimpleNamespace(shard_hash=lambda: shard_hash))


@pytest.mark.parametrize(
    ("row", "match"),
    [
        (_ft_row(device="mps"), "recipe device"),
        (_ft_row(shard_hash="b" * 64), "shard_hash"),
        (_ft_row(backbone_snapshot="other"), "backbone_snapshot"),
        (_ft_row(tag="memorise"), "recipe tag"),
    ],
)
def test_a_checkpoint_paired_with_the_wrong_row_is_refused_before_any_read(tmp_path, row, match):
    """No checkpoint file exists here: the refusal must come from the pairing alone."""
    args = _args(tmp_path, _ledger(tmp_path, row))
    with pytest.raises(SystemExit, match=match):
        real_ft_run._score_checkpoint(
            args, reader=_reader(), val=None, device="mps", ledger=None,  # type: ignore[arg-type]
            reasons_for=lambda *a, **k: [], second_pass=None,  # type: ignore[arg-type]
            needle_suite=_NO_NEEDLE, ood_suite=_NO_OOD, suite_seed=SUITE_SEED,
        )


def test_the_seed_in_the_filename_must_be_the_seed_asked_for(tmp_path):
    args = _args(tmp_path, _ledger(tmp_path, _ft_row()), seeds=(1,))
    with pytest.raises(SystemExit, match="is seed 0"):
        real_ft_run._score_checkpoint(
            args, reader=_reader(), val=None, device="mps", ledger=None,  # type: ignore[arg-type]
            reasons_for=lambda *a, **k: [], second_pass=None,  # type: ignore[arg-type]
            needle_suite=_NO_NEEDLE, ood_suite=_NO_OOD, suite_seed=SUITE_SEED,
        )


def test_only_an_epoch_checkpoint_is_scored(tmp_path):
    args = _args(tmp_path, _ledger(tmp_path, _ft_row()), name="memorise-seed0-cuda.json")
    with pytest.raises(SystemExit, match="only an epoch checkpoint"):
        real_ft_run._score_checkpoint(
            args, reader=_reader(), val=None, device="mps", ledger=None,  # type: ignore[arg-type]
            reasons_for=lambda *a, **k: [], second_pass=None,  # type: ignore[arg-type]
            needle_suite=_NO_NEEDLE, ood_suite=_NO_OOD, suite_seed=SUITE_SEED,
        )


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--score-val"], "--score-checkpoint needs"),
        (["--score-val", "--real-backbone", "/x", "--ft-ledger", "/l", "--ft-row-id", "abcdefgh",
          "--devices", "mps", "cpu"], "exactly one"),
        (["--score-val", "--real-backbone", "/x", "--ft-ledger", "/l", "--ft-row-id", "abcdefgh",
          "--devices", "mps", "--seeds", "0", "--epoch"], "trains nothing"),
    ],
)
def test_score_checkpoint_argv_is_refused_before_anything_loads(tmp_path, extra, match):
    argv = ["--out", str(tmp_path), "--rev", "0" * 40, "--score-checkpoint",
            str(tmp_path / "epoch-seed0-cuda.json"), *extra]
    if "--seeds" not in extra:
        argv += ["--seeds", "0"]
    with pytest.raises(SystemExit, match=match):
        real_ft_run.main(argv)


def test_ft_row_flags_alone_are_refused(tmp_path):
    with pytest.raises(SystemExit, match="only mean something"):
        real_ft_run.main(["--out", str(tmp_path), "--rev", "0" * 40, "--ft-row-id", "abcdefgh"])


def test_the_scored_checkpoint_keys_reach_the_recipe_only_when_present():
    """A training run's score rows must keep the recipe hash they always had."""
    src = (REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8")
    assert "**{k: run[k] for k in SCORED_CHECKPOINT_KEYS if k in run}" in src
    # "averaged" joined for J7: the provenance block of a scored average (its seeds, ft
    # rows, manifest sha256 and source), on an averaged score row and on no other.
    # "ensemble" joined for J7' (Fable I): a logit ensemble's seeds, ft rows, towers and
    # combine rule, on an ensemble's score row and on no other.
    # "trained_by" joined for human ask 7 of Fable's ojas advice (2026-10-01): the trainer,
    # device, source and manifest of a Metal export, on a row scoring one and on no other.
    assert set(real_ft_run.SCORED_CHECKPOINT_KEYS) == {
        "score_dtype", "scored_checkpoint", "averaged", "ensemble", "trained_by",
    }


def test_a_bf16_score_of_a_master_checkpoint_gets_the_master_layout():
    """2026-09-30, GH200: --score-dtype bf16 on the master-trained phase-3 checkpoints was
    given ADAMW_BF16, and build_optimizer refused it -- bf16 moments do not survive the
    1,505-step schedule -- before a single val row was scored. Scoring takes no step, but
    the step is built like training's, so it is built from the same recipe."""
    import torch

    from qd_train.memory import ADAMW_BF16, ADAMW_FP32
    from qd_train.optim import DEFAULT_BETA2, moment_settling

    assert not moment_settling(dtype=torch.bfloat16, beta2=DEFAULT_BETA2).survives(1505)
    master = real_ft_run.optimizer_spec("bf16", "master")
    assert master.keeps_fp32_master and master.state_bytes == 4
    assert real_ft_run.optimizer_spec("bf16", "bf16") is ADAMW_BF16
    assert real_ft_run.optimizer_spec("fp32", "master") is ADAMW_FP32
    with pytest.raises(ValueError, match="no optimizer layout"):
        real_ft_run.optimizer_spec("fp16", "master")


# --- an averaged checkpoint (tools/ckpt_average.py), J7 --------------------------------------

STEPS = 1505
RECIPE_HASH = "c" * 64
AVG_IDS = {s: f"5eed000{s}-ab12-4c34-8d56-00000000000{s}" for s in (0, 1, 2)}


def _schedule():
    from qd_train.run_control import LRSchedule

    return LRSchedule(peak_lr=1e-5, total_steps=STEPS, warmup_steps=10, min_lr=1e-6)


def _sources(tmp_path: Path, seeds=(0, 1, 2)) -> list[ckpt_average.InputRecord]:
    """One JSON body per seed, written by ``Checkpoint.write`` -- what the manifest's
    ``payload_digest`` is checked against. No tensors: only the body is ever read."""
    from qd_train.run_control import Checkpoint, LossLog, Position

    records = []
    for s in seeds:
        path = tmp_path / "ckpt" / f"epoch-seed{s}-cuda.json"
        Checkpoint(
            position=Position(epoch=0, index=STEPS), optimizer_step=STEPS, seed=s,
            schedule=_schedule(), loss_log=LossLog().snapshot(),
            consumed_digest=f"{s + 1:064x}", model_state={"span_weight": 1.0, "vocab_size": 8},
        ).write(path)
        digest = json.loads(path.read_text(encoding="utf-8"))["payload_digest"]
        records.append(ckpt_average.InputRecord(path=path, seed=s, payload_digest=digest))
    return records


def _average(tmp_path: Path, *, seeds=(0, 1, 2), ft_ids=None) -> Path:
    """An average written by the tool's own writer, over tiny stand-in tensors."""
    import torch

    records = _sources(tmp_path, seeds)
    out = tmp_path / "avg" / "avg.safetensors"
    tensors = {"tower.w": torch.zeros(4, dtype=torch.bfloat16), "span_head.b": torch.zeros(2)}
    ckpt_average.write(
        tensors, out, source="masters", inputs=records, optimizer_step=STEPS,
        schedule=_schedule().to_json(), scalars={"vocab_size": 8, "span_weight": 1.0},
        ft_row_ids=[AVG_IDS[s] for s in seeds] if ft_ids is None else list(ft_ids),
        tensor_sources={"tower.w": "master", "span_head.b": "master"},
        master_index={"tower.w": 0, "span_head.b": 1},
    )
    return out


def _avg_row(seed: int, *, quick: bool = False, recipe_hash: str = RECIPE_HASH,
             steps: int = STEPS, snapshot: str = "d" * 64, **recipe_over) -> dict:
    row = _ft_row(AVG_IDS[seed], **recipe_over)
    row["protocol"] = {
        "seed": seed, "recipe_hash": recipe_hash, "data_snapshot_hash": snapshot,
        "tokenizer_hash": "e" * 64, "backbone_commit": "snap:vocab248320",
    }
    row["quick"] = quick
    row["metrics"] = {
        "train.optimizer_steps": {"value": steps},
        "train.termination": {"value": "steps_exhausted"},
    }
    return row


def _avg_args(avg: Path, ledger: Path, seeds=(0, 1, 2)) -> argparse.Namespace:
    return argparse.Namespace(
        score_checkpoint=avg, seeds=list(seeds), ft_ledger=ledger, ft_row_id=None,
        real_backbone=Path("/models/snap"), score_dtype="fp32",
    )


def _score_avg(args: argparse.Namespace, *, val: object = None):
    return real_ft_run._score_checkpoint(
        args, reader=_reader(), val=val, device="cuda", ledger=None,  # type: ignore[arg-type]
        reasons_for=lambda *a, **k: [], second_pass=None,  # type: ignore[arg-type]
        needle_suite=_NO_NEEDLE, ood_suite=_NO_OOD, suite_seed=SUITE_SEED,
    )


class _Reached(Exception):
    """Raised in place of building the tower: every pairing check before it passed."""


def test_a_well_formed_average_passes_every_pairing_check_and_reaches_the_tower(
    tmp_path, monkeypatch
):
    """The positive control for every refusal below: the same fixture, unaltered, gets as
    far as building the step -- at the protocol seed, the rows' schedule and --score-dtype
    -- with the averaged weights already read and verified."""
    import numpy as np

    avg = _average(tmp_path)
    ledger = _ledger(tmp_path, *(_avg_row(s) for s in (0, 1, 2)))
    seen: dict[str, object] = {}

    def reached(**kwargs):
        seen.update(kwargs)
        raise _Reached

    monkeypatch.setattr(real_ft_run, "_real_step", reached)
    val = SimpleNamespace(plan=[SimpleNamespace(tokens=np.zeros((1, 64)))])
    with pytest.raises(_Reached):
        _score_avg(_avg_args(avg, ledger), val=val)
    assert seen["seed"] == SUITE_SEED and seen["total_steps"] == STEPS
    assert seen["dtype"] == "fp32" and seen["span_weight"] == 1.0


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing_row", "0 rows match"),
        ("quick_row", "quick"),
        ("other_recipe", "recipe_hash"),
        ("other_steps", "optimizer_steps"),
        ("other_snapshot", "data_snapshot_hash"),
        ("rows_out_of_order", "seed"),
        ("other_shard_set", "shard_hash"),
    ],
)
def test_an_average_its_ft_rows_do_not_describe_is_refused_before_the_tower(
    tmp_path, case, match
):
    rows = {s: _avg_row(s) for s in (0, 1, 2)}
    ft_ids = None
    if case == "missing_row":
        del rows[2]
    elif case == "quick_row":
        rows[1] = _avg_row(1, quick=True)
    elif case == "other_recipe":
        rows[2] = _avg_row(2, recipe_hash="f" * 64)
    elif case == "other_steps":
        rows[1] = _avg_row(1, steps=STEPS - 1)
    elif case == "other_snapshot":
        rows[2] = _avg_row(2, snapshot="9" * 64)
    elif case == "rows_out_of_order":
        ft_ids = [AVG_IDS[1], AVG_IDS[0], AVG_IDS[2]]
    elif case == "other_shard_set":
        rows = {s: _avg_row(s, shard_hash="b" * 64) for s in (0, 1, 2)}
    avg = _average(tmp_path, ft_ids=ft_ids)
    ledger = _ledger(tmp_path, *rows.values())
    with pytest.raises(SystemExit, match=match):
        _score_avg(_avg_args(avg, ledger))


def test_an_average_whose_source_checkpoint_changed_is_refused(tmp_path):
    """The manifest recorded each input's payload_digest; a body on disk that hashes to
    something else is not the checkpoint the average was made from."""
    from qd_train.run_control import Checkpoint, LossLog, Position

    avg = _average(tmp_path)
    Checkpoint(
        position=Position(epoch=0, index=STEPS), optimizer_step=STEPS, seed=1,
        schedule=_schedule(), loss_log=LossLog().snapshot(), consumed_digest=f"{9:064x}",
        model_state={"span_weight": 1.0, "vocab_size": 8},
    ).write(tmp_path / "ckpt" / "epoch-seed1-cuda.json")
    ledger = _ledger(tmp_path, *(_avg_row(s) for s in (0, 1, 2)))
    with pytest.raises(SystemExit, match="payload_digest"):
        _score_avg(_avg_args(avg, ledger))


def test_an_average_whose_source_checkpoint_is_gone_is_refused(tmp_path):
    """Fail closed: a digest that cannot be checked is not a digest that matched. Only the
    JSON body is needed (a few hundred KB), never the sidecar."""
    avg = _average(tmp_path)
    (tmp_path / "ckpt" / "epoch-seed2-cuda.json").unlink()
    ledger = _ledger(tmp_path, *(_avg_row(s) for s in (0, 1, 2)))
    with pytest.raises(SystemExit, match="not on disk"):
        _score_avg(_avg_args(avg, ledger))


def test_an_average_naming_no_ft_rows_is_refused(tmp_path):
    avg = _average(tmp_path, ft_ids=[])
    ledger = _ledger(tmp_path, *(_avg_row(s) for s in (0, 1, 2)))
    with pytest.raises(SystemExit, match="ft_row_ids"):
        _score_avg(_avg_args(avg, ledger))


def test_an_average_scored_under_other_seeds_is_refused(tmp_path):
    avg = _average(tmp_path)
    ledger = _ledger(tmp_path, *(_avg_row(s) for s in (0, 1, 2)))
    with pytest.raises(SystemExit, match="--seeds"):
        _score_avg(_avg_args(avg, ledger, seeds=(0, 1)))


def test_altered_averaged_weights_are_refused(tmp_path):
    avg = _average(tmp_path)
    avg.write_bytes(avg.read_bytes() + b" ")
    ledger = _ledger(tmp_path, *(_avg_row(s) for s in (0, 1, 2)))
    with pytest.raises(SystemExit, match="safetensors_sha256"):
        _score_avg(_avg_args(avg, ledger))


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--seeds", "0", "1", "2", "--ft-row-id", "5eed0000"], "its manifest names"),
        (["--seeds", "0"], "two or more"),
        # Fable G (2026-10-01) lets an average take --needle-control; the flags that clash
        # with it still clash on an average.
        (["--seeds", "0", "1", "2", "--needle", "--needle-control", "1024", "--ood"],
         "--ood would record nothing"),
        (["--seeds", "0", "1", "2", "--needle", "--needle-control", "1024",
          "--verdicts-out", "/v.jsonl"], "--verdicts-out would record nothing"),
        (["--seeds", "0", "--needle", "--needle-control", "1024"], "two or more"),
    ],
)
def test_average_argv_is_refused_before_anything_loads(tmp_path, extra, match):
    argv = ["--out", str(tmp_path), "--rev", "0" * 40, "--score-checkpoint",
            str(tmp_path / "avg.safetensors"), "--score-val", "--real-backbone", "/x",
            "--ft-ledger", "/l", "--devices", "cuda", "--instance", "gh200",
            "--usd-per-hour", "2.29", *extra]
    with pytest.raises(SystemExit, match=match):
        real_ft_run.main(argv)


# --- --needle-control on an average (Fable G, 2026-10-01) ----------------------------------


def test_average_needle_control_argv_is_accepted(tmp_path, monkeypatch):
    """An average with --needle-control gets past every argv check: the first thing after
    them, resolving --rev, is reached."""

    def reached(*a, **k):
        raise _Reached

    monkeypatch.setattr(real_ft_run, "resolve_rev", reached)
    with pytest.raises(_Reached):
        real_ft_run.main([
            "--out", str(tmp_path), "--rev", "0" * 40, "--score-checkpoint",
            str(tmp_path / "avg.safetensors"), "--score-val", "--real-backbone", "/x",
            "--ft-ledger", "/l", "--devices", "cuda", "--instance", "gh200",
            "--usd-per-hour", "2.29", "--wall-clock-cap-s", "5400", "--seeds", "0", "1", "2",
            "--needle", "--needle-control", "1024,2048,4096",
            "--suite-verdicts-out", str(tmp_path / "suite.jsonl"),
        ])


class _CapturedRecorder:
    """Stands in for ``_recorder``'s RunRecorder: keeps what the row would be written with."""

    def __init__(self, captured: dict[str, object]):
        self.captured = captured
        self.row = SimpleNamespace(row_id="control-row")
        self.noul_rate = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def metric(self, name, state):
        self.captured.setdefault("metrics", {})[name] = state  # type: ignore[union-attr]

    def gate(self, name, state):
        raise AssertionError(f"a needle control recorded the gate {name}")


def _run_avg_needle_control(monkeypatch, args, *, step_reached=None, predictions=None):
    """``run_needle_control`` on an average, with ``_averaged_weights``' pairing checks run as
    written. Only the tower (``_real_step``), the suite build and the decode
    (``predictions``, standing in for ``needle_predictions``) are stood in for: no GPU, no
    backbone."""
    import numpy as np
    from test_needle_ft_contract import _decoded

    from qd_train.needle import build_suite

    cases = build_suite(target_tokens=1024, cases_per_depth=1, seed=0)
    suite = real_ft_run.NeedleSuite(
        cases, [SimpleNamespace(tokens=np.zeros((1, 1088)))] * len(cases), {},
        [900] * len(cases), digest="a" * 64,
    )
    monkeypatch.setattr(real_ft_run, "prepare_needle", lambda *a, **k: suite)
    if predictions is None:
        def predictions(step, s, letter_id, **kw):
            return _decoded(s.cases, hit_every=1, abstain_every=2)
    monkeypatch.setattr(real_ft_run, "needle_predictions", predictions)
    if step_reached is None:
        def step_reached(**kwargs):
            return SimpleNamespace(load_weights=lambda weights: None), None, None
    monkeypatch.setattr(real_ft_run, "_real_step", step_reached)
    captured: dict[str, object] = {}

    def recorder(ledger, **kw):
        captured.update(kw)
        return _CapturedRecorder(captured)

    monkeypatch.setattr(real_ft_run, "_recorder", recorder)
    monkeypatch.setattr(real_ft_run, "_cost", lambda **k: None)
    reader = _reader()
    val = SimpleNamespace(
        reader=reader, letter_id={}, plan=[SimpleNamespace(tokens=np.zeros((1, 64)))]
    )
    result = real_ft_run.run_needle_control(
        args, reader=reader, val=val, device="cuda", ledger=None,  # type: ignore[arg-type]
        config=DataConfig(), gate_suite=suite,
        reasons_for=lambda tag, device, termination=None: [f"{tag} on {device}: {termination}"],
    )
    return result, captured, len(cases)


def _avg_control_args(avg: Path, ledger: Path, seeds=(0, 1, 2)) -> argparse.Namespace:
    return argparse.Namespace(
        **vars(_avg_args(avg, ledger, seeds)), needle_control=(1024, 2048, 4096),
        usd_per_hour=2.29, usd_per_gpu_hour=None, instance="gh200", wall_clock_cap_s=5400.0,
        suite_logits=False,
    )


def test_an_averages_needle_control_row_pairs_with_every_input_ft_row(tmp_path, monkeypatch):
    """The average's length-control row names the same model the average's score row does:
    all three ft rows its manifest names (each checked against --ft-ledger by
    _averaged_weights), its manifest sha256 and its safetensors sha256 -- never one input's
    ft row -- at the protocol seed, and it stays quick."""
    avg = _average(tmp_path)
    ledger = _ledger(tmp_path, *(_avg_row(s) for s in (0, 1, 2)))
    manifest_file = ckpt_average.manifest_path(avg)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    (row_id, lines, seed), captured, n_cases = _run_avg_needle_control(
        monkeypatch, _avg_control_args(avg, ledger)
    )
    recipe = captured["recipe"]
    assert row_id == "control-row" and captured["run_kind"] == "eval"
    assert seed == SUITE_SEED and captured["seed"] == SUITE_SEED
    assert recipe["tag"] == "avg-needle-length-control"
    assert recipe["averaged"] == {
        "seeds": [0, 1, 2], "ft_row_ids": [AVG_IDS[s] for s in (0, 1, 2)],
        "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "source": "masters", "n_inputs": 3, "protocol_seed": SUITE_SEED,
        "suite_seed": SUITE_SEED,
    }
    assert recipe["scored_checkpoint"] == f"avg.safetensors:{manifest['safetensors_sha256']}"
    assert recipe["needle_control"]["target_tokens"] == [1024, 2048, 4096]
    assert "eval_row_id" not in recipe, "a control naming the eval row would join its family"
    metrics = captured["metrics"]
    assert metrics["ft_run_row_ids"].value == ",".join(AVG_IDS[s] for s in (0, 1, 2))
    assert "ft_run_row_id" not in metrics, "that names ONE row; an average is three"
    assert {f"needle_hunk_recall.control.{n}" for n in (1024, 2048, 4096)} <= set(metrics)
    reasons = captured["quick_reasons"]
    assert real_ft_run.NEEDLE_CONTROL_QUICK_REASON in reasons
    # Every input's termination is read, and one reason shared by three rows is one reason.
    assert reasons.count("epoch on cuda: steps_exhausted") == 1
    assert all(AVG_IDS[s] in captured["notes"] for s in (0, 1, 2))
    assert len(lines) == 3 * n_cases


def test_an_average_of_two_seeds_says_so_on_its_needle_control_row(tmp_path, monkeypatch):
    avg = _average(tmp_path, seeds=(0, 1))
    ledger = _ledger(tmp_path, *(_avg_row(s) for s in (0, 1)))
    _, captured, _ = _run_avg_needle_control(
        monkeypatch, _avg_control_args(avg, ledger, seeds=(0, 1))
    )
    assert any("an average of 2 seeds" in r for r in captured["quick_reasons"])


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing_row", "0 rows match"),
        ("quick_row", "quick"),
        ("rows_out_of_order", "seed"),
        ("other_seeds", "--seeds"),
        ("altered_weights", "safetensors_sha256"),
    ],
)
def test_an_averages_needle_control_is_refused_when_it_cannot_be_paired(
    tmp_path, monkeypatch, case, match
):
    """The same refusals as the average's score row, before the tower is built."""
    rows = {s: _avg_row(s) for s in (0, 1, 2)}
    ft_ids, seeds = None, (0, 1, 2)
    if case == "missing_row":
        del rows[2]
    elif case == "quick_row":
        rows[1] = _avg_row(1, quick=True)
    elif case == "rows_out_of_order":
        ft_ids = [AVG_IDS[1], AVG_IDS[0], AVG_IDS[2]]
    elif case == "other_seeds":
        seeds = (0, 1)
    avg = _average(tmp_path, ft_ids=ft_ids)
    if case == "altered_weights":
        avg.write_bytes(avg.read_bytes() + b" ")
    ledger = _ledger(tmp_path, *rows.values())

    def never(**kwargs):
        raise AssertionError("the tower was built for an average that cannot be paired")

    with pytest.raises(SystemExit, match=match):
        _run_avg_needle_control(
            monkeypatch, _avg_control_args(avg, ledger, seeds=seeds), step_reached=never
        )


# --- end to end: three tiny master checkpoints, averaged, scored on CPU --------------------


def _defect_shards(out: Path):
    """Train and val code.defect_class shard sets over the byte tokenizer -- the
    shuffled-label fixture's corpus, without the training run it also makes."""
    from test_real_ft_shuffled_label import REV, _defect_rows, _offsets, _remap, _tokenize

    from qd_data.dedupe import dedupe
    from qd_data.defect_class import DEFECT_FAMILY_ID, DEFECT_SOURCE_ID
    from qd_data.manifest import build_manifests
    from qd_data.mixture import build_mixture
    from qd_data.split import HELD_OUT, split
    from qd_train.shards import write_shards

    config = DataConfig()
    mixture = build_mixture(
        {DEFECT_SOURCE_ID: _defect_rows(48)}, config=config, families=[DEFECT_FAMILY_ID]
    )
    report = dedupe(list(mixture.rows), config=config)
    split_report = split(report, config=config)
    manifests = build_manifests(
        config=config, mixture=mixture, dedupe_report=report, split_report=split_report
    )
    paths: dict[str, Path] = {}
    for name, manifest in manifests.items():
        path = out / "data" / (HELD_OUT if name == HELD_OUT else "pool") / f"{name}.json"
        manifest.write(path)
        paths[name] = path
    train = list(split_report.rows_by_split["train"])
    val = list(split_report.rows_by_split["val"])
    assert train and val
    for name, rows in (("train", train), ("val", val)):
        write_shards(
            paths[name], rows, out_dir=out / "shards" / name, remap=_remap(),
            tokenize=_tokenize, token_offsets=_offsets, config=config, repo_root=out,
            corpus_rev=REV,
        )
    return train, val


@dataclasses.dataclass(frozen=True)
class TinyCheckpoints:
    """Three tiny ``--optimizer master`` epoch checkpoints over the byte-tokenized defect
    shard sets, with the ft ledger that names them: the fixture every CPU end-to-end
    ``--score-checkpoint`` test (one seed, an average, an ensemble) scores."""

    out: Path
    train: list
    val: list
    snapshot: Path
    paths: list[Path]
    ids: list[str]
    ft_ledger: Path
    tokenizer: Path


def tiny_master_checkpoints(tmp_path: Path, monkeypatch) -> TinyCheckpoints:
    import test_backbone as tb
    from test_real_ft_family_metrics import _tokenizer_json
    from test_real_ft_shuffled_label import REV, _patch

    import qd_train.backbone as backbone
    from qd_train.memory import OptimizerSpec
    from qd_train.run_control import Checkpoint, LossLog, LRSchedule, Position
    from qd_train.shards import ShardReader

    out = tmp_path / "corpus"
    train, val = _defect_shards(out)
    vocab = 256  # the fixture's identity remap is 256 byte ids wide
    snapshot = tmp_path / "tiny-snapshot"
    tb._write_tiny_snapshot(snapshot, vocab=vocab)
    spec = tb._tiny_spec(vocab)
    real_load = backbone.load_text_tower
    # _real_step loads with the 2B ModelSpec, which refuses a tiny tower; the spec is the only
    # thing replaced -- _real_step, the remap, the step and load_weights all run as written.
    monkeypatch.setattr(
        backbone, "load_text_tower", lambda *a, **k: real_load(*a, **{**k, "spec": spec})
    )

    steps = 2
    master = OptimizerSpec(*tb.MASTER_SPEC_ARGS, keeps_fp32_master=True)
    paths = []
    n_params = 0
    for seed in (0, 1, 2):
        tower = backbone.load_text_tower(
            snapshot, gradient_checkpointing=True, optimizer=master,
            attn_implementation="sdpa", dtype="bf16", rows=1, width=64,
        )
        n_params = sum(p.numel() for p in tower.model.parameters())
        step = backbone.QwenDecisionStep(
            tower, seed=seed, lr=1e-2, total_steps=steps, max_width=64
        )
        for k in range(steps):
            batch = tb._seeded_batch(seed, k)
            step.accumulate_span(batch, tb.ft_supervision(batch))
            step.apply(lr=1e-2)
        path = tmp_path / "ckpt" / f"epoch-seed{seed}-cpu.json"
        Checkpoint(
            position=Position(epoch=0, index=steps), optimizer_step=steps, seed=seed,
            schedule=LRSchedule(peak_lr=1e-2, total_steps=steps, warmup_steps=1, min_lr=1e-3),
            loss_log=LossLog().snapshot(), consumed_digest=f"{seed + 1:064x}",
            model_state=step.state(),
        ).write(path)
        paths.append(path)

    reader = ShardReader(out / "shards" / "train", config=DataConfig(), repo_root=out,
                         expect_rev=REV)
    ids = [f"e2e0000{s}-0000-4000-8000-00000000000{s}" for s in (0, 1, 2)]
    ft_ledger = _ledger(tmp_path, *(
        {
            "row_id": rid, "run_kind": "ft", "status": "completed", "quick": False,
            "recipe": {
                "tool": "tools/real_ft_run.py", "tag": "epoch", "device": "cpu",
                "shard_hash": reader.header.shard_hash(), "backbone_snapshot": snapshot.name,
                "backbone_params": n_params, "backbone_vocab": vocab,
                "attn_implementation": "sdpa", "lr": 1e-2, "span_weight": 1.0,
                "optimizer_recipe": "master",
            },
            "protocol": {
                "seed": s, "recipe_hash": RECIPE_HASH,
                "data_snapshot_hash": reader.header.data_snapshot_hash,
                "tokenizer_hash": reader.header.tokenizer_hash, "backbone_commit": "tiny",
            },
            "metrics": {
                "train.optimizer_steps": {"value": steps},
                "train.termination": {"value": "steps_exhausted"},
            },
        }
        for s, rid in zip((0, 1, 2), ids, strict=True)
    ))
    _patch(monkeypatch, (train, val))
    return TinyCheckpoints(
        out=out, train=train, val=val, snapshot=snapshot, paths=paths, ids=ids,
        ft_ledger=ft_ledger, tokenizer=_tokenizer_json(tmp_path / "tokenizer.json"),
    )


def test_an_average_of_three_tiny_master_checkpoints_is_scored_end_to_end_on_cpu(
    tmp_path, monkeypatch
):
    """The J7 sequence on CPU: three ``--optimizer master`` checkpoints, averaged from their
    masters by ``tools/ckpt_average.py``, scored by ``real_ft_run.py --score-checkpoint``
    through the same val, permutation and gate code as a per-seed score. The eval row names
    its seeds, its ft rows, its manifest's sha256 and its source; it is quick only because a
    CPU is not a campaign device, never for a seed, schedule or subsample shortfall."""
    from test_real_ft_shuffled_label import REV

    from qd_train.ledger import Ledger

    tiny = tiny_master_checkpoints(tmp_path, monkeypatch)
    out, snapshot, paths, ids, ft_ledger = (
        tiny.out, tiny.snapshot, tiny.paths, tiny.ids, tiny.ft_ledger
    )
    avg = tmp_path / "avg" / "avg.safetensors"
    assert ckpt_average.main(
        [*map(str, paths), "--out", str(avg), "--from", "masters", "--ft-row-ids", *ids]
    ) == 0
    manifest_file = ckpt_average.manifest_path(avg)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    eval_ledger = tmp_path / "eval.jsonl"
    verdicts = tmp_path / "verdicts.jsonl"
    assert real_ft_run.main([
        "--out", str(out), "--rev", REV, "--devices", "cpu", "--seeds", "0", "1", "2",
        "--score-val", "--score-checkpoint", str(avg), "--real-backbone", str(snapshot),
        "--ft-ledger", str(ft_ledger), "--ledger", str(eval_ledger),
        "--tokenizer-json", str(tiny.tokenizer),
        "--verdicts-out", str(verdicts),
    ]) == 0

    rows = Ledger(eval_ledger).rows()
    assert [r.run_kind for r in rows] == ["eval"]
    row = rows[0]
    recipe = dict(row.recipe or {})
    assert recipe["tag"] == "avg-score-val"
    assert recipe["averaged"] == {
        "seeds": [0, 1, 2], "ft_row_ids": ids,
        "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "source": "masters", "n_inputs": 3, "protocol_seed": SUITE_SEED,
        "suite_seed": SUITE_SEED,
    }
    assert recipe["scored_checkpoint"] == f"avg.safetensors:{manifest['safetensors_sha256']}"
    assert recipe["score_dtype"] == "fp32"
    assert row.protocol.seed == SUITE_SEED
    assert row.protocol.recipe_hash != RECIPE_HASH
    assert row.metrics["ft_run_row_ids"].value == ",".join(ids)
    assert "ft_run_row_id" not in row.metrics
    only_the_device = real_ft_run.quick_reasons(
        tag="epoch", device="cpu", real_backbone=True,
        corpus=real_ft_run.CorpusFacts(snapshot_not_run=None, history_rows={}),
    )
    assert row.quick and row.quick_reason == "; ".join(only_the_device)
    assert "human's decision" in row.notes
    assert {"permutation_consistency", "ece", "needle_hunk_recall", "ood_abstain"} <= set(
        row.gates
    )
    lines = [json.loads(x) for x in verdicts.read_text(encoding="utf-8").splitlines()]
    assert lines and {(v["eval_row_id"], v["seed"]) for v in lines} == {
        (row.row_id, SUITE_SEED)
    }
