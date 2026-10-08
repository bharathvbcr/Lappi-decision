"""``tools/read_p1_parity.py``: P1's pre-registered reading, on synthetic inputs.

One fixture per outcome word, one per refusal. The inputs are the real formats: ledger rows
written by ``RunRecorder``, checkpoint bodies carrying a verified ``payload_digest`` and the
``channel_log`` ``QwenDecisionStep.state`` writes, ``--verdicts-out`` JSONL, and
``gh200_footprint.py --out`` JSON with its ``shape``. No torch.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import read_p1_parity as p1  # noqa: E402

from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder  # noqa: E402
from qd_train.run_control import _payload_digest  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

N_VAL = 400
BASE = [2.0 - 1.5 * s / p1.STEPS for s in range(1, p1.STEPS + 1)]
SHAPE = {"rows": 1, "widths": [2048, 4096], "loss": "proxy", "gradient_checkpointing": True,
         "checkpoint_skip_layers": [], "attn_implementation": "sdpa", "dtype": "bf16",
         "snapshot": "s", "peak": "max_memory_allocated"}


def _env() -> Environment:
    return Environment(torch="t", transformers_sha="none", device="cpu", host="test",
                       fla_present=NotRun(reason="no CUDA"),
                       causal_conv1d_present=NotRun(reason="no CUDA"))


def _recorder(ledger: Path, run_kind: str, recipe: dict[str, Any], seed: int) -> RunRecorder:
    return RunRecorder(
        Ledger(ledger),
        protocol=Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
                          backbone_commit="b" * 40, recipe_hash="r" * 64, seed=seed),
        run_kind=run_kind, repo=REPO, env=_env(), wall_clock_s=1.0, cost=None, recipe=recipe,
    )


def _timing(value: float | None) -> Any:
    if value is None:
        return NotRun(reason="12 of the 500 steps of window 101-600 timed")
    return Ran(passed=True, value=value, n=500, n_total=500,
               detail="median wall seconds over optimizer steps 101-600")


def _arm(tmp: Path, ledger: Path, name: str, *, letter: list[float], correct: list[bool],
         apply_s: float | None = 0.5, step_s: float = 2.0, optimizer: str | None = None,
         termination: str = "steps_exhausted", steps: int = p1.STEPS, seed: int = 0,
         digest: str = "c" * 64, recipe_extra: dict[str, Any] | None = None,
         tamper: bool = False) -> p1.ArmInputs:
    recipe = {"tool": "tools/real_ft_run.py", "tag": "epoch", "lr": 1e-5,
              "optimizer_recipe": optimizer or p1.ARM_RECIPE[name], **(recipe_extra or {})}
    with (rec := _recorder(ledger, "ft", recipe, seed)):
        rec.metric("train.termination", Ran(passed=True, value=termination))
        rec.metric(p1.APPLY_METRIC, _timing(apply_s))
        rec.metric(p1.STEP_METRIC, _timing(step_s))
    ft_id = rec.row.row_id
    n_ok = sum(correct)
    with (ev := _recorder(ledger, "eval", {**recipe, "tag": "epoch-score-val"}, seed)):
        ev.metric("ft_run_row_id", Ran(passed=True, value=ft_id))
        ev.metric("val_top1.choice", Ran(passed=True, value=n_ok / len(correct), n=n_ok,
                                         n_total=len(correct)))
    verdicts = tmp / f"{name}.verdicts.jsonl"
    lines = [{"eval_row_id": ev.row.row_id, "seed": seed, "row_id": f"v{i}", "kind": "choice",
              "correct": c, "top": 0, "gold_row": 0} for i, c in enumerate(correct)]
    lines.append({"eval_row_id": ev.row.row_id, "seed": seed, "row_id": "s0", "kind": "span",
                  "correct": False, "top": 0, "gold_row": 0})
    verdicts.write_text("".join(json.dumps(x) + "\n" for x in lines))
    body: dict[str, Any] = {
        "optimizer_step": steps, "seed": seed, "consumed_digest": digest,
        "model_state": {"micro_batches": len(letter),
                        "channel_log": {"letter": [float(x).hex() for x in letter],
                                        "span": [(0.0).hex()] * len(letter)}},
    }
    body["payload_digest"] = _payload_digest(body)
    if tamper:
        body["seed"] = 1
    ckpt = tmp / f"{name}.json"
    ckpt.write_text(json.dumps(body))
    return p1.ArmInputs(ckpt, ft_id, verdicts)


def _footprint(tmp: Path, recipe: str, peaks: list[int | None], shape=SHAPE) -> Path:
    path = tmp / f"fp_{recipe}.json"
    results = [{"width": w, "rows": 1, "measured_bytes": b} if b is not None
               else {"width": w, "rows": 1, "oom": "CUDA out of memory"}
               for w, b in zip(shape["widths"], peaks, strict=True)]
    path.write_text(json.dumps({"recipe": recipe, "shape": shape, "results": results}))
    return path


def _scaled(factor: float) -> list[float]:
    return [x * factor for x in BASE]


def _span_only(letter: list[float], steps: list[int]) -> list[float]:
    """``letter`` with the letter channel empty (0.0) at each step of ``steps``."""
    out = list(letter)
    for s in steps:
        out[s - 1] = 0.0
    return out


WINDOW = list(range(p1.LOSS_WINDOW[0], p1.LOSS_WINDOW[1] + 1))


def _correct(n_wrong: int) -> list[bool]:
    return [i >= n_wrong for i in range(N_VAL)]


def _setup(tmp_path: Path, *, kahan_letter=None, kahan_correct=None, kahan_apply=0.6,
           repeat_letter=None, kahan_peaks=(70, 70), master_peaks=(100, 100),
           kahan_shape=SHAPE, **kahan_kw) -> tuple[dict, dict, list[Path]]:
    ledger = tmp_path / "ledger.jsonl"
    arms = {
        "master": _arm(tmp_path, ledger, "master", letter=BASE, correct=_correct(100)),
        "master-repeat": _arm(tmp_path, ledger, "master-repeat",
                              letter=repeat_letter or _scaled(1.005), correct=_correct(100)),
        "kahan": _arm(tmp_path, ledger, "kahan", letter=kahan_letter or _scaled(1.01),
                      correct=kahan_correct or _correct(100), apply_s=kahan_apply, **kahan_kw),
    }
    prints = {"master": _footprint(tmp_path, "master", list(master_peaks)),
              "kahan": _footprint(tmp_path, "kahan", list(kahan_peaks), shape=kahan_shape)}
    return arms, prints, [ledger]


def _read(tmp_path: Path, **kw) -> p1.Reading:
    return p1.read(*_setup(tmp_path, **kw))


# --- each outcome word ---------------------------------------------------------------------------


def test_admissible(tmp_path):
    r = _read(tmp_path)
    assert r.word == "admissible", r.refusals
    assert r.numbers["loss_gap_kahan_master"] == pytest.approx(0.01)
    assert r.numbers["floor"] == pytest.approx(0.005)
    assert r.numbers["loss_gap_threshold"] == pytest.approx(0.02)
    assert r.numbers["val_gap_interval"] == [0.0, 0.0]
    assert r.numbers["peak_ratio"] == pytest.approx(0.7)
    assert r.numbers["step_ratio"] == pytest.approx(1.2)
    assert r.numbers["val_rows_paired"] == N_VAL
    assert r.numbers["loss_gap_steps_used"] == len(WINDOW) == 216
    assert r.numbers["loss_gap_steps_excluded"] == []


def test_step_ratio_reports_the_master_repeat_noise_floor_and_step_time(tmp_path):
    """Ruling 3: master-repeat's ratio and train.step_time_s are reported; neither decides."""
    r = _read(tmp_path)
    assert r.numbers["step_ratio_repeat_master"] == pytest.approx(1.0)
    assert r.numbers["step_time_ratio_kahan_master"] == pytest.approx(1.0)
    assert r.numbers["optimizer_step_s_median"]["master-repeat"] == pytest.approx(0.5)
    # A whole step 3x master's decides nothing: only apply (optimizer_step_s) does.
    sub = tmp_path / "slow-whole-step"
    sub.mkdir()
    assert _read(sub, step_s=6.0).word == "admissible"


def test_slow(tmp_path):
    assert _read(tmp_path, kahan_apply=0.8).word == "slow"


def test_step_time_at_exactly_one_and_a_half_is_not_slow(tmp_path):
    assert _read(tmp_path, kahan_apply=0.75).word == "admissible"


def test_no_saving(tmp_path):
    assert _read(tmp_path, kahan_peaks=(70, 81)).word == "no_saving"


def test_the_peak_condition_must_hold_at_every_width(tmp_path):
    r = _read(tmp_path, kahan_peaks=(60, 85))
    assert r.word == "no_saving" and r.numbers["peak_ratio"] == pytest.approx(0.85)
    assert r.numbers["peak_ratio_by_width"] == {2048: pytest.approx(0.6),
                                                4096: pytest.approx(0.85)}


def test_diverges_on_the_loss_gap(tmp_path):
    assert _read(tmp_path, kahan_letter=_scaled(1.05)).word == "diverges"


def test_diverges_on_the_val_interval(tmp_path):
    r = _read(tmp_path, kahan_correct=[*([False] * 140), *([True] * (N_VAL - 140))])
    assert r.word == "diverges"
    _, hi = r.numbers["val_gap_interval"]
    assert r.numbers["val_gap_kahan_master"] == pytest.approx(-0.1) and hi < -0.02


def test_diverges_wins_over_no_saving_and_slow(tmp_path):
    r = _read(tmp_path, kahan_letter=_scaled(1.05), kahan_peaks=(95, 95), kahan_apply=5.0)
    assert r.word == "diverges"


def test_a_wide_floor_widens_the_loss_threshold(tmp_path):
    """floor 0.015 -> threshold 0.03, so a 0.025 gap is admissible."""
    r = _read(tmp_path, repeat_letter=_scaled(1.015), kahan_letter=_scaled(1.025))
    assert r.word == "admissible" and r.numbers["loss_gap_threshold"] == pytest.approx(0.03)


def test_the_bootstrap_is_paired_seeded_and_reproducible():
    a = {f"v{i}": i % 3 != 0 for i in range(300)}
    b = {f"v{i}": i % 4 != 0 for i in range(300)}
    one, two = p1.val_gap(a, b), p1.val_gap(a, b)
    assert one == two
    gap, lo, hi, n = one
    assert n == 300 and lo <= gap <= hi
    # Identical arms: every paired draw is exactly 0.
    assert p1.val_gap(a, a)[1:3] == (0.0, 0.0)


def _sequential_interval(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ruling 2 verbatim: default_rng(20261006), 2,000 draws in sequence, each
    integers(0, n, size=n), then percentile(D, [2.5, 97.5]) with linear interpolation."""
    rng = np.random.default_rng(20261006)
    n = len(d)
    means = np.array([d[rng.integers(0, n, size=n)].mean() for _ in range(2000)])
    return means, np.percentile(means, [2.5, 97.5], method="linear")


@pytest.mark.parametrize("n", [1, 7, 300, 401])
@pytest.mark.parametrize("chunk", [1, 7, 100, 2000])
def test_the_chunked_bootstrap_is_the_sequential_form_bit_for_bit(monkeypatch, n, chunk):
    """The ruling's required test: the reader's chunked draws equal the sequential form."""
    monkeypatch.setattr(p1, "BOOTSTRAP_CHUNK", chunk)
    a = {f"v{i}": (i * 7) % 5 != 0 for i in range(n)}
    b = {f"v{i}": (i * 3) % 4 != 0 for i in range(n)}
    d = np.array([float(a[i]) - float(b[i]) for i in sorted(a)])
    means, (lo, hi) = _sequential_interval(d)
    assert np.array_equal(p1.bootstrap_means(d), means)
    gap, got_lo, got_hi, got_n = p1.val_gap(a, b)
    assert (got_lo, got_hi, got_n) == (float(lo), float(hi), n)
    assert gap == float(d.mean())


# --- each refusal --------------------------------------------------------------------------------


def _refused(r: p1.Reading, needle: str) -> None:
    assert r.word == "refused"
    assert any(needle in x for x in r.refusals), r.refusals


def test_a_missing_arm_is_refused(tmp_path):
    arms, prints, ledgers = _setup(tmp_path)
    del arms["master-repeat"]
    _refused(p1.read(arms, prints, ledgers), "arm master-repeat not given")


def test_a_missing_footprint_is_refused(tmp_path):
    arms, prints, ledgers = _setup(tmp_path)
    prints["kahan"].unlink()
    _refused(p1.read(arms, prints, ledgers), "does not exist")


def test_a_missing_checkpoint_is_refused(tmp_path):
    arms, prints, ledgers = _setup(tmp_path)
    arms["kahan"].checkpoint.unlink()
    _refused(p1.read(arms, prints, ledgers), "kahan: checkpoint")


def test_an_arm_that_hit_its_cap_is_refused(tmp_path):
    _refused(_read(tmp_path, termination="wall_clock_cap"), "did not run its schedule out")


def test_an_arm_short_of_600_steps_is_refused(tmp_path):
    _refused(_read(tmp_path, steps=300, kahan_letter=_scaled(1.01)[:300]), "not 600")


def test_an_uncovered_step_time_window_is_refused(tmp_path):
    _refused(_read(tmp_path, kahan_apply=None), "did not run")


def test_a_nan_letter_loss_is_refused(tmp_path):
    letter = _scaled(1.01)
    letter[450] = math.nan
    _refused(_read(tmp_path, kahan_letter=letter), "non-finite letter loss at step(s) [451]")


def _three(tmp_path: Path, *, master: list[float], repeat: list[float],
           kahan: list[float]) -> p1.Reading:
    arms, prints, ledgers = _setup(tmp_path, repeat_letter=repeat, kahan_letter=kahan)
    arms["master"] = _arm(tmp_path, ledgers[0], "master", letter=master, correct=_correct(100))
    return p1.read(arms, prints, ledgers)


def test_span_only_steps_shared_by_every_arm_are_excluded_as_a_pair(tmp_path):
    """Ruling 1: S = steps 385-600 with letter_master > 0, from master, applied to every arm."""
    gone = [385, 400, 401, 599, 600]
    # Outside S the arms are wildly apart only at steps that S excludes: they must not count.
    r = _three(tmp_path, master=_span_only(BASE, gone), repeat=_span_only(_scaled(1.005), gone),
               kahan=_span_only(_scaled(1.01), gone))
    assert r.word == "admissible", r.refusals
    assert r.numbers["loss_gap_steps_used"] == 216 - len(gone)
    assert r.numbers["loss_gap_steps_excluded"] == gone
    assert r.numbers["loss_gap_kahan_master"] == pytest.approx(0.01)
    assert r.numbers["floor"] == pytest.approx(0.005)


def test_span_only_steps_outside_the_window_do_not_count(tmp_path):
    gone = [1, 50, 384]
    r = _three(tmp_path, master=_span_only(BASE, gone), repeat=_span_only(_scaled(1.005), gone),
               kahan=_span_only(_scaled(1.01), gone))
    assert r.word == "admissible", r.refusals
    assert r.numbers["loss_gap_steps_excluded"] == []


@pytest.mark.parametrize("which", ["master-repeat", "kahan"])
def test_an_arm_whose_span_only_steps_differ_from_masters_is_refused(tmp_path, which):
    """Ruling 1: letter_a[s] > 0 iff letter_master[s] > 0, for every s and every arm."""
    shared = [400]
    logs = {"master": _span_only(BASE, shared),
            "master-repeat": _span_only(_scaled(1.005), shared),
            "kahan": _span_only(_scaled(1.01), shared)}
    logs[which] = _span_only(logs[which], [450])
    r = _three(tmp_path, master=logs["master"], repeat=logs["master-repeat"],
               kahan=logs["kahan"])
    _refused(r, f"{which}: the letter channel is empty at different steps than master's")
    assert r.numbers["loss_gap_steps_used"] is None


def test_a_master_only_span_only_step_is_refused(tmp_path):
    r = _three(tmp_path, master=_span_only(BASE, [450]), repeat=_scaled(1.005),
               kahan=_scaled(1.01))
    _refused(r, "empty at different steps than master's")


def test_fewer_than_108_letter_steps_is_refused_and_108_is_read(tmp_path):
    """Ruling 1: |S| < 108 refuses the reading; |S| = 108 is read."""
    for n_gone, word in ((len(WINDOW) - 108, "admissible"), (len(WINDOW) - 107, "refused")):
        gone = WINDOW[:n_gone]
        sub = tmp_path / str(n_gone)
        sub.mkdir()
        r = _three(sub, master=_span_only(BASE, gone), repeat=_span_only(_scaled(1.005), gone),
                   kahan=_span_only(_scaled(1.01), gone))
        assert r.word == word, (n_gone, r.refusals)
    _refused(r, "only 107 step(s) in 385-600 carry a letter loss")


def test_a_negative_letter_loss_is_refused(tmp_path):
    letter = _scaled(1.01)
    letter[449] = -0.5
    _refused(_read(tmp_path, kahan_letter=letter), "kahan: negative letter loss at step(s) [450]")


def test_the_zero_letter_choice_states_the_ruling():
    words = p1.CHOICES["zero_letter"]
    assert "letter_master > 0" in words and "every arm" in words and "excluded" in words
    assert "|S| < 108" in words and "refused" in words
    assert "refused, not skipped" not in words


def test_a_wrong_optimizer_is_refused(tmp_path):
    _refused(_read(tmp_path, optimizer="master"), "not 'kahan'")


def test_a_different_seed_is_refused(tmp_path):
    _refused(_read(tmp_path, seed=1), "not 0")


def test_different_data_order_is_refused(tmp_path):
    _refused(_read(tmp_path, digest="e" * 64), "consumed different data")


def test_a_recipe_differing_beyond_the_optimizer_is_refused(tmp_path):
    _refused(_read(tmp_path, recipe_extra={"lr": 2e-5}), "differ in more than the optimizer")


def test_a_tampered_checkpoint_is_refused(tmp_path):
    _refused(_read(tmp_path, tamper=True), "payload_digest")


def test_different_footprint_shapes_are_refused(tmp_path):
    _refused(_read(tmp_path, kahan_shape={**SHAPE, "gradient_checkpointing": False}),
             "different shapes: ['gradient_checkpointing']")


def test_a_footprint_oom_is_refused(tmp_path):
    _refused(_read(tmp_path, kahan_peaks=(70, None)), "went out of memory")


def test_a_footprint_without_a_shape_is_refused(tmp_path):
    arms, prints, ledgers = _setup(tmp_path)
    raw = json.loads(prints["master"].read_text())
    del raw["shape"]
    prints["master"].write_text(json.dumps(raw))
    _refused(p1.read(arms, prints, ledgers), "records no shape")


def test_verdicts_that_disagree_with_the_eval_row_are_refused(tmp_path):
    arms, prints, ledgers = _setup(tmp_path)
    path = arms["kahan"].verdicts
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["correct"] = not first["correct"]
    path.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    _refused(p1.read(arms, prints, ledgers), "val_top1.choice")


def test_verdicts_of_another_ft_row_are_refused(tmp_path):
    arms, prints, ledgers = _setup(tmp_path)
    arms["kahan"].verdicts = arms["master"].verdicts
    _refused(p1.read(arms, prints, ledgers), "is not of ft row")


def test_no_ledger_is_refused(tmp_path):
    arms, prints, _ = _setup(tmp_path)
    _refused(p1.read(arms, prints, []), "no --ledger")


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_decide_refuses_a_non_finite_value(bad):
    assert p1.decide(gap=bad, floor=0.0, val=(0.0, 0.0), peak_ratio=0.5,
                     step_ratio=1.0) == "refused"


# --- the command, and the registration it reads ----------------------------------------------


def test_main_prints_and_writes_the_word_and_refuses_to_overwrite(tmp_path, capsys):
    arms, prints, ledgers = _setup(tmp_path)
    out = tmp_path / "reading.json"
    argv = ["--ledger", str(ledgers[0]), "--out", str(out)]
    for name, a in arms.items():
        argv += ["--arm", name, str(a.checkpoint), a.ft_row_id, str(a.verdicts)]
    for name, path in prints.items():
        argv += ["--footprint", name, str(path)]
    assert p1.main(argv) == 0
    assert json.loads(out.read_text())["word"] == "admissible"
    assert "P1: admissible" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="already exists"):
        p1.main(argv)


def test_main_exits_non_zero_on_refused(tmp_path):
    assert p1.main(["--ledger", str(tmp_path / "none.jsonl")]) == 3


def test_the_constants_are_the_drafts():
    draft = json.loads((REPO / p1.DRAFT).read_text())["P1"]
    defs, outcomes = draft["definitions"], draft["outcomes"]
    assert "385..600" in defs["S"] and "letter_master[s] > 0" in defs["S"]
    assert f"|S| < {p1.LOSS_STEPS_MIN}" in defs["S"]
    assert "mean over s in S" in defs["loss_gap(a,b)"]
    assert "default_rng(20261006)" in defs["val_gap(a,b)"]
    assert "2,000 draws" in defs["val_gap(a,b)"]
    assert "integers(0, n, size=n)" in defs["val_gap(a,b)"]
    assert "percentile(D, [2.5, 97.5])" in defs["val_gap(a,b)"]
    assert (p1.BOOTSTRAP_DRAWS, p1.BOOTSTRAP_SEED) == (2000, 20261006)
    assert p1.APPLY_METRIC in defs["step_ratio"] and "master-repeat" in defs["step_ratio"]
    assert "max over widths" in defs["peak_ratio"]
    assert "max(2 x floor, 0.02)" in outcomes["admissible"]
    assert "[-0.02, +0.02]" in outcomes["admissible"]
    assert "peak_ratio <= 0.80" in outcomes["admissible"]
    assert "step_ratio <= 1.5" in outcomes["admissible"]
    assert "peak_ratio > 0.80" in outcomes["no_saving"]
    assert "step_ratio > 1.5" in outcomes["slow"]
    assert "steps 101-600" in draft["per_arm"]
    assert "model_state.channel_log.letter" in draft["per_arm"]
    assert "letter_log" not in draft["per_arm"]
    assert "seed (0)" in draft["what"] and "--max-steps 600" in draft["what"]
    assert "--verdicts-out" in draft["what"] and "--wall-clock-cap-s 3600" in draft["what"]
    assert "--checkpoint-every 100" in draft["what"]
    assert draft["data_fact"].startswith("NOT YET COMPUTED")
    assert tuple(draft["first_matching_word_wins_in_order"]) == p1.ORDER
