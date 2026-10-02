"""``tools/real_ft_run.py --min-lr``: the cosine floor as a flag (v5 recipe.added[1]).

Until v5 the floor was hard-coded ``lr / 10`` (``_control``). The flag replaces it only when
given: a run without it builds the schedule every row so far trained on and records no
``min_lr`` key, so v4 recipe hashes do not move -- checked here against F seed 0's own ft
row (973cd4e3), not against a restatement of it.

Torch-gated, like ``test_real_ft_pieces.py``: the tool raises at import without torch.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402

from qd_train.optim import apply_lr, layerwise_param_groups  # noqa: E402
from qd_train.run_control import LRSchedule  # noqa: E402

#: F seed 0's ft row, the recipe v5's base is (campaign/v5-preregistered.DRAFT.json recipe.base).
F_LEDGER = REPO / "ledger" / "gh200-p4-v4-2026-10-01.jsonl"
F_SEED0_FT = "973cd4e3"

#: Every piece off, as _recipe_pieces' positional half.
OFF = {
    "lower_layers_n": 0, "lower_lr_scale": 1.0, "beta2": rft.DEFAULT_BETA2,
    "permutation": None, "replay": None,
}


def test_without_the_flag_the_floor_is_still_lr_over_ten():
    for lr, steps in ((1e-5, 200), (1e-5, 9683), (3e-5, 20)):
        schedule = rft._control(steps, device="cpu", lr=lr).schedule
        assert schedule.min_lr.hex() == (lr / 10).hex()
        assert schedule.warmup_steps == max(1, steps // 20)


def test_the_flag_sets_the_floor_and_nothing_else():
    default = rft._control(9683, device="cpu", lr=1e-5).schedule
    floored = rft._control(9683, device="cpu", lr=1e-5, min_lr=0.0).schedule
    assert floored.min_lr == 0.0
    assert (floored.peak_lr, floored.total_steps, floored.warmup_steps) == (
        default.peak_lr, default.total_steps, default.warmup_steps
    )
    # The warmup half does not read the floor, so it is bit-identical; the decay half is not.
    for step in range(floored.warmup_steps):
        assert floored.lr_at(step).hex() == default.lr_at(step).hex()
    last = floored.total_steps - 1
    assert 0.0 < floored.lr_at(last) < default.lr_at(last)


def test_min_lr_is_a_recipe_key_only_when_given():
    assert rft._recipe_pieces(**OFF) == {}
    assert rft._recipe_pieces(**OFF, min_lr=None) == {}
    assert rft._recipe_pieces(**OFF, min_lr=0.0) == {"min_lr": 0.0}
    # Given equal to the default it is still recorded: "only when given", and lr / 10 is
    # not even the literal 1e-6 bitwise (rung (d), schedule_oracle.rs).
    assert rft._recipe_pieces(**OFF, min_lr=1e-6) == {"min_lr": 1e-6}
    assert "min_lr" in rft.RECIPE_PIECE_KEYS


def _f_seed0_ft_row() -> dict:
    rows = [
        json.loads(line) for line in F_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    (row,) = [r for r in rows if r["row_id"].startswith(F_SEED0_FT)]
    return row


def test_f_seed0s_recipe_and_its_hash_are_what_the_pieces_rebuild_without_the_flag():
    """The v4 recipe hash does not move: F's flags through today's _recipe_pieces give
    exactly the piece keys F's row recorded, and that recipe hashes to F's recipe_hash."""
    row = _f_seed0_ft_row()
    recipe = row["recipe"]
    pieces = rft._recipe_pieces(
        lower_layers_n=8, lower_lr_scale=0.1, beta2=rft.DEFAULT_BETA2, permutation=None,
        replay=None, cap_s=32_400.0, no_memorise=True, batch_tokens=35_403,
        checkpoint_skip_layers=6,
    )
    recorded = {k: recipe[k] for k in rft.RECIPE_PIECE_KEYS if k in recipe}
    assert pieces == recorded
    assert "min_lr" not in recipe
    digest = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert digest == row["protocol"]["recipe_hash"]


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("-1e-7", "must be finite and non-negative"),
        ("nan", "must be finite and non-negative"),
        ("inf", "must be finite and non-negative"),
        ("1.0", "exceeds the peak"),
    ],
)
def test_a_floor_the_schedule_could_not_use_is_refused_at_argv_time(tmp_path, value, match):
    with pytest.raises(SystemExit, match=match):
        rft.main(["--out", str(tmp_path), "--min-lr", value])


def test_with_lower_layers_at_a_tenth_their_floor_is_zero_too():
    """--lower-layers-lr-scale 0.1 scales the schedule's rate, floor included, so a zero
    floor is zero for the lower layers as well: their rate is 0.1x the schedule's at every
    step, and at the last step it is positive (apply_lr refuses 0) and below the 0.1 *
    lr / 10 floor every v4 run's lower layers had."""
    layers = {
        "layers.0.w": torch.nn.Parameter(torch.zeros(3)),
        "layers.1.w": torch.nn.Parameter(torch.zeros(3)),
    }
    groups = layerwise_param_groups(layers.items(), lower_layers_n=1, lower_lr_scale=0.1)
    opt = torch.optim.AdamW(groups, lr=1e-5)
    by_name = {g["name"]: g for g in opt.param_groups}
    lr, steps = 1e-5, 9683
    floored = rft._control(steps, device="cpu", lr=lr, min_lr=0.0).schedule
    assert floored.min_lr * by_name["base_lower"]["lr_scale"] == 0.0
    for step in (0, floored.warmup_steps, steps // 2, steps - 1):
        rate = floored.lr_at(step)
        apply_lr(opt, rate)
        assert by_name["base"]["lr"] == rate
        assert by_name["base_lower"]["lr"] == rate * 0.1
    last = by_name["base_lower"]["lr"]
    assert 0.0 < last < 0.1 * (lr / 10)
    assert math.isfinite(last)


def test_a_resume_across_a_floor_change_is_refused():
    """train_ft compares the checkpoint's schedule with the run's before it loads anything;
    the floor is part of the schedule, so a v4 checkpoint (floor lr / 10) is not resumed
    under --min-lr 0, and a floor-0 checkpoint is not resumed without the flag."""
    from types import SimpleNamespace

    from qd_train.run_control import EMPTY_PREFIX_DIGEST, Checkpoint, LossLog, Position
    from qd_train.trainer import TrainerContractViolation, train_ft

    v4 = rft._control(100, device="cpu", lr=1e-5)
    v5 = rft._control(100, device="cpu", lr=1e-5, min_lr=0.0)
    assert LRSchedule.from_json(v5.schedule.to_json()) == v5.schedule
    recorder = SimpleNamespace(run_kind="ft", protocol=SimpleNamespace(seed=0))
    for taken, resumed in ((v4, v5), (v5, v4)):
        checkpoint = Checkpoint(
            position=Position(epoch=0, index=0), optimizer_step=0, seed=0,
            schedule=taken.schedule, loss_log=LossLog(), consumed_digest=EMPTY_PREFIX_DIGEST,
        )
        with pytest.raises(TrainerContractViolation, match="schedule differs"):
            train_ft(
                iter(()), epoch=0, step=object(), control=resumed, recorder=recorder,
                resume_from=checkpoint,
            )
