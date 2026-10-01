"""``tools/real_ft_run.py --score-checkpoint``: score a saved epoch checkpoint, train nothing.

The three phase-3 weight sets of 2026-09-30 were saved but never scored outside the process
that trained them. Scoring them later means pairing a file with the ft row that wrote it,
and every way that pairing could silently put the wrong weights under the wrong row is
refused before a tower loads.
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

import real_ft_run  # noqa: E402

SHARD_HASH = "a" * 64
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
            needle_suite=_NO_NEEDLE, ood_suite=_NO_OOD,
        )


def test_the_seed_in_the_filename_must_be_the_seed_asked_for(tmp_path):
    args = _args(tmp_path, _ledger(tmp_path, _ft_row()), seeds=(1,))
    with pytest.raises(SystemExit, match="is seed 0"):
        real_ft_run._score_checkpoint(
            args, reader=_reader(), val=None, device="mps", ledger=None,  # type: ignore[arg-type]
            reasons_for=lambda *a, **k: [], second_pass=None,  # type: ignore[arg-type]
            needle_suite=_NO_NEEDLE, ood_suite=_NO_OOD,
        )


def test_only_an_epoch_checkpoint_is_scored(tmp_path):
    args = _args(tmp_path, _ledger(tmp_path, _ft_row()), name="memorise-seed0-cuda.json")
    with pytest.raises(SystemExit, match="only an epoch checkpoint"):
        real_ft_run._score_checkpoint(
            args, reader=_reader(), val=None, device="mps", ledger=None,  # type: ignore[arg-type]
            reasons_for=lambda *a, **k: [], second_pass=None,  # type: ignore[arg-type]
            needle_suite=_NO_NEEDLE, ood_suite=_NO_OOD,
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
    assert set(real_ft_run.SCORED_CHECKPOINT_KEYS) == {"score_dtype", "scored_checkpoint"}


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
