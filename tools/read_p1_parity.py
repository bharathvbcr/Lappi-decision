"""Read P1 of campaign/next-train-first-box-2026-10-06.DRAFT.json: master vs master-repeat vs
kahan on the 2B tower, seed 0, 600 steps. A pure reader: no torch, no training, no ledger row.

Inputs (all required; a missing one is ``refused``, never a smaller reading):

* per arm (``master``, ``master-repeat``, ``kahan``): the final checkpoint body that
  ``real_ft_run.py --checkpoint-dir`` wrote, the arm's ft row id, and the ``--verdicts-out``
  JSONL its ``--score-val`` wrote;
* ``--ledger``: the ledger(s) holding the three ft rows and their eval rows;
* ``--footprint master|kahan``: ``tools/gh200_footprint.py --out`` for each recipe.

The definitions, as Fable's ruling of 2026-10-08 fixed them
(``AUDIT/v6-rulings-2026-10-08/fable-p1-reading-rule-ruling.md``)::

    S             = { s in 385..600 : letter_master[s] > 0 }, computed once from master
    loss_gap(a,b) = mean over s in S of |letter_a[s] - letter_b[s]| / letter_b[s]
    floor         = loss_gap(master-repeat, master), over the same S
    val_gap(a,b)  = mean over paired val rows of correct_a - correct_b, with a paired
                    bootstrap 95% percentile interval (2,000 draws, seed 20261006)
    step_ratio    = kahan's train.optimizer_step_s / master's
    peak_ratio    = max over widths w of measured_kahan[w] / measured_master[w]

and the outcome words, first match wins in the order refused, diverges, no_saving, slow,
admissible. The output names each reading choice under ``"choices"``:

* ``letter`` is the checkpoint's ``model_state.channel_log.letter`` (float.hex; written by
  ``QwenDecisionStep.state``, ``python/qd_train/backbone.py``), one entry per micro-batch;
  real_ft_run runs one micro-batch per optimizer step, and the reader refuses a checkpoint
  where the two counts differ. Step ``s`` is entry ``s - 1``.
* A step whose letter loss is 0.0 is a span-only batch (the trainer logs 0.0 for an empty
  letter channel). The batch plan is a pure function of seed, epoch, batch_tokens and the
  shard set, so a span-only step is span-only in every arm: ``S`` is taken from master and
  applied to every arm, and an arm whose zero steps in the window differ from master's is
  ``refused``. So is ``|S| < 108`` and a negative letter loss. The reading reports
  ``loss_gap_steps_used`` (``|S|``) and ``loss_gap_steps_excluded`` (the steps).
* "val choice top-1" is the mean of ``correct`` over the verdict lines of kind ``choice``
  (what ``val_top1.choice`` on the eval row is; the reader checks the two agree). Rows are
  paired by ``row_id``; the two arms must have scored the same set.
* The bootstrap is ``numpy.random.default_rng(20261006)`` (PCG64), 2,000 draws in sequence,
  each ``rng.integers(0, n, size=n)``, and the interval is ``numpy.percentile(D, [2.5, 97.5])``
  (linear). The reader draws in chunks to bound memory; each draw is still its own
  ``integers`` call, so the chunks are the sequential form bit for bit (a test asserts it).
* ``step_ratio`` decides on ``train.optimizer_step_s``: the median over steps 101-600 of
  ``apply`` alone (clip, ``optimizer.step()``, ``zero_grad``), synced on the device.
  ``train.step_time_s`` and master-repeat's ratio (the noise floor) are reported and decide
  nothing.
* ``peak_ratio`` is over ``measured_bytes`` (``max_memory_allocated`` over the second step)
  at every width of the footprint tool's own shape; the maximum ratio decides. Shapes that
  differ, a width either side did not measure, or an OOM anywhere is ``refused``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np

from qd_train.ledger import Ledger, LedgerRow
from qd_train.replay import ReplayRefusal, write_text_atomic
from qd_train.run_control import Checkpoint
from qd_train.tristate import Ran

DRAFT: Final[str] = "campaign/next-train-first-box-2026-10-06.DRAFT.json"
ARMS: Final[tuple[str, ...]] = ("master", "master-repeat", "kahan")
#: Which ``--optimizer`` each arm was trained with (ft row ``recipe.optimizer_recipe``).
ARM_RECIPE: Final[Mapping[str, str]] = {"master": "master", "master-repeat": "master",
                                        "kahan": "kahan"}
FOOTPRINTS: Final[tuple[str, ...]] = ("master", "kahan")
SEED: Final[int] = 0
STEPS: Final[int] = 600
LOSS_WINDOW: Final[tuple[int, int]] = (385, 600)
#: Fewer letter-bearing steps than this in the window refuses the reading (the ruling's floor).
LOSS_STEPS_MIN: Final[int] = 108
STEP_TIME_WINDOW: Final[tuple[int, int]] = (101, 600)
LOSS_GAP_MIN: Final[float] = 0.02
VAL_BAND: Final[tuple[float, float]] = (-0.02, 0.02)
PEAK_RATIO_MAX: Final[float] = 0.80
STEP_RATIO_MAX: Final[float] = 1.5
BOOTSTRAP_DRAWS: Final[int] = 2000
BOOTSTRAP_SEED: Final[int] = 20261006
#: Draws per chunk, so the index matrix is at most this x n_rows int64s at a time.
BOOTSTRAP_CHUNK: Final[int] = 100
ORDER: Final[tuple[str, ...]] = ("refused", "diverges", "no_saving", "slow", "admissible")
APPLY_METRIC: Final[str] = "train.optimizer_step_s"
STEP_METRIC: Final[str] = "train.step_time_s"
#: Bounds on what is read: a P1 val set is thousands of rows, a footprint a few widths.
MAX_VERDICT_LINES: Final[int] = 1_000_000
MAX_INPUT_BYTES: Final[int] = 512 * 1024 * 1024


class Refused(Exception):
    """One input or check that makes the reading ``refused``."""


@dataclass
class ArmInputs:
    checkpoint: Path
    ft_row_id: str
    verdicts: Path


@dataclass
class Arm:
    name: str
    letter: list[float]
    consumed_digest: str
    seed: int
    ft: LedgerRow
    choice: dict[str, bool]
    apply_s: float
    step_s: float


@dataclass
class Reading:
    word: str
    refusals: list[str] = field(default_factory=list)
    numbers: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "rule": DRAFT, "probe": "P1", "word": self.word, "refusals": self.refusals,
            "numbers": self.numbers, "choices": CHOICES,
        }


CHOICES: Final[dict[str, str]] = {
    "letter": "checkpoint model_state.channel_log.letter (float.hex), entry s-1 is step s",
    "zero_letter": "loss_gap is over S = steps 385-600 with letter_master > 0, taken once from "
                   "master and applied to every arm; the span-only steps are excluded as a "
                   "pair and listed; an arm whose zero steps differ from master's, or "
                   f"|S| < {LOSS_STEPS_MIN}, is refused (Fable's ruling of 2026-10-08)",
    "val_top1": "mean correct over --verdicts-out lines of kind 'choice', paired by row_id",
    "bootstrap": "numpy.random.default_rng(20261006); 2,000 draws in sequence, each "
                 "integers(0, n, size=n); percentile [2.5, 97.5] linear",
    "step_time": f"step_ratio = kahan / master ft row {APPLY_METRIC} (apply alone, median of "
                 f"steps 101-600) decides; {STEP_METRIC} and master-repeat's ratio reported",
    "peak": "peak_ratio = max over the footprint shape's widths of kahan / master "
            "measured_bytes; equal shapes required, OOM refused",
}


# --- reading the inputs ------------------------------------------------------------------------


def _read_text(path: Path, what: str) -> str:
    if not path.is_file():
        raise Refused(f"{what} {path} does not exist")
    size = path.stat().st_size
    if size > MAX_INPUT_BYTES:
        raise Refused(f"{what} {path} is {size} bytes, above the {MAX_INPUT_BYTES} bound")
    return path.read_text(encoding="utf-8")


def _rows_by_id(ledgers: Sequence[Path]) -> dict[str, LedgerRow]:
    if not ledgers:
        raise Refused("no --ledger given")
    out: dict[str, LedgerRow] = {}
    for path in ledgers:
        if not path.is_file():
            raise Refused(f"ledger {path} does not exist")
        try:
            rows = Ledger(path).rows()
        except (ValueError, KeyError, TypeError) as exc:
            raise Refused(f"ledger {path} does not parse: {exc}") from exc
        for row in rows:
            if row.row_id in out:
                raise Refused(f"row {row.row_id} appears twice across the ledgers given")
            out[row.row_id] = row
    return out


def _median_metric(ft: LedgerRow, name: str, arm: str) -> float:
    state = ft.metrics.get(name)
    if state is None:
        raise Refused(f"{arm}: ft row {ft.row_id} carries no {name} (written before StepTimer)")
    if not isinstance(state, Ran):
        raise Refused(f"{arm}: ft row {ft.row_id} {name} did not run: {state.reason}")
    lo, hi = STEP_TIME_WINDOW
    want = hi - lo + 1
    if state.n != want or state.n_total != want:
        raise Refused(f"{arm}: {name} covers {state.n}/{state.n_total} steps, not {want}/{want}")
    if f"steps {lo}-{hi}" not in state.detail:
        raise Refused(f"{arm}: {name} is not over steps {lo}-{hi}: {state.detail!r}")
    value = state.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refused(f"{arm}: {name} value {value!r} is not a number")
    if not (math.isfinite(value) and value > 0.0):
        raise Refused(f"{arm}: {name} value {value!r} is not a finite positive time")
    return float(value)


def _letter_log(body: Mapping[str, Any], arm: str) -> list[float]:
    state = body.get("model_state")
    if not isinstance(state, Mapping):
        raise Refused(f"{arm}: the checkpoint has no model_state")
    log = state.get("channel_log")
    if not isinstance(log, Mapping) or not isinstance(log.get("letter"), list):
        raise Refused(f"{arm}: the checkpoint's model_state carries no channel_log.letter")
    try:
        letter = [float.fromhex(x) for x in log["letter"]]
    except (TypeError, ValueError) as exc:
        raise Refused(f"{arm}: channel_log.letter is not float.hex strings: {exc}") from exc
    steps = body.get("optimizer_step")
    if steps != STEPS:
        raise Refused(f"{arm}: the checkpoint is at optimizer step {steps}, not {STEPS}")
    if state.get("micro_batches") != len(letter) or len(letter) != steps:
        raise Refused(
            f"{arm}: {len(letter)} letter entries, {state.get('micro_batches')} micro-batches "
            f"and {steps} optimizer steps; the reader maps entry s-1 to step s only when all "
            "three agree (one micro-batch per step)"
        )
    bad = [i + 1 for i, x in enumerate(letter) if not math.isfinite(x)]
    if bad:
        raise Refused(f"{arm}: non-finite letter loss at step(s) {bad[:10]}")
    return letter


def _choice_verdicts(path: Path, arm: str, rows: Mapping[str, LedgerRow],
                     ft_row_id: str) -> dict[str, bool]:
    text = _read_text(path, f"{arm}: verdicts")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise Refused(f"{arm}: verdicts {path} is empty")
    if len(lines) > MAX_VERDICT_LINES:
        raise Refused(f"{arm}: verdicts {path} has {len(lines)} lines, above the bound")
    eval_ids: set[str] = set()
    choice: dict[str, bool] = {}
    for i, ln in enumerate(lines):
        try:
            v = json.loads(ln)
        except json.JSONDecodeError as exc:
            raise Refused(f"{arm}: verdicts line {i + 1} is not JSON: {exc}") from exc
        if not isinstance(v, dict) or not isinstance(v.get("correct"), bool):
            raise Refused(f"{arm}: verdicts line {i + 1} has no bool 'correct'")
        eval_ids.add(str(v.get("eval_row_id")))
        if v.get("kind") != "choice":
            continue
        rid = str(v.get("row_id"))
        if rid in choice:
            raise Refused(f"{arm}: val row {rid} appears twice in {path}")
        choice[rid] = v["correct"]
    if len(eval_ids) != 1:
        raise Refused(f"{arm}: verdicts {path} mixes eval rows {sorted(eval_ids)}")
    if not choice:
        raise Refused(f"{arm}: verdicts {path} hold no choice row")
    (eval_id,) = eval_ids
    ev = rows.get(eval_id)
    if ev is None:
        raise Refused(f"{arm}: the verdicts' eval row {eval_id} is in no ledger given")
    if ev.run_kind != "eval" or ev.status != "completed":
        raise Refused(f"{arm}: eval row {eval_id} is {ev.run_kind}/{ev.status}")
    of = ev.metrics.get("ft_run_row_id")
    if not isinstance(of, Ran) or of.value != ft_row_id:
        raise Refused(f"{arm}: eval row {eval_id} is not of ft row {ft_row_id}")
    top1 = ev.metrics.get("val_top1.choice")
    mine = sum(choice.values())
    if not isinstance(top1, Ran) or top1.n != mine or top1.n_total != len(choice):
        raise Refused(
            f"{arm}: eval row {eval_id} val_top1.choice is "
            f"{None if top1 is None else top1.to_json()}, the verdicts give {mine}/{len(choice)}"
        )
    return choice


def read_arm(name: str, inputs: ArmInputs, rows: Mapping[str, LedgerRow]) -> Arm:
    if not inputs.checkpoint.is_file():
        raise Refused(f"{name}: checkpoint {inputs.checkpoint} does not exist")
    try:
        body = Checkpoint.read_body(inputs.checkpoint)
    except (ValueError, OSError) as exc:
        raise Refused(f"{name}: checkpoint {inputs.checkpoint}: {exc}") from exc
    letter = _letter_log(body, name)
    ft = rows.get(inputs.ft_row_id)
    if ft is None:
        raise Refused(f"{name}: ft row {inputs.ft_row_id} is in no ledger given")
    if ft.run_kind != "ft" or ft.status != "completed":
        raise Refused(f"{name}: row {ft.row_id} is {ft.run_kind}/{ft.status}, not ft/completed")
    term = ft.metrics.get("train.termination")
    if not isinstance(term, Ran) or term.value != "steps_exhausted":
        raise Refused(f"{name}: ft row {ft.row_id} did not run its schedule out "
                      f"(train.termination {None if term is None else term.to_json()})")
    recipe = dict(ft.recipe or {})
    if recipe.get("optimizer_recipe") != ARM_RECIPE[name]:
        raise Refused(f"{name}: ft row {ft.row_id} trained with optimizer "
                      f"{recipe.get('optimizer_recipe')!r}, not {ARM_RECIPE[name]!r}")
    if ft.protocol.seed != SEED or body.get("seed") != SEED:
        raise Refused(f"{name}: seed {ft.protocol.seed} (row) / {body.get('seed')} "
                      f"(checkpoint), not {SEED}")
    return Arm(
        name=name, letter=letter, consumed_digest=str(body.get("consumed_digest")),
        seed=SEED, ft=ft,
        choice=_choice_verdicts(inputs.verdicts, name, rows, ft.row_id),
        apply_s=_median_metric(ft, APPLY_METRIC, name),
        step_s=_median_metric(ft, STEP_METRIC, name),
    )


def read_footprint(name: str, path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(_read_text(path, f"footprint {name}"))
    except json.JSONDecodeError as exc:
        raise Refused(f"footprint {name} {path} is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise Refused(f"footprint {name} {path} is not a JSON object")
    if raw.get("recipe") != name:
        raise Refused(f"footprint {path} is recipe {raw.get('recipe')!r}, not {name!r}")
    if not isinstance(raw.get("shape"), dict):
        raise Refused(f"footprint {name} {path} records no shape (written before --out)")
    peaks: dict[int, int] = {}
    for r in raw.get("results") or []:
        if "oom" in r:
            raise Refused(f"footprint {name}: width {r.get('width')} went out of memory; "
                          "no peak to compare there")
        b = r.get("measured_bytes")
        if isinstance(b, bool) or not isinstance(b, int) or b <= 0:
            raise Refused(f"footprint {name}: width {r.get('width')} measured {b!r}")
        peaks[int(r["width"])] = b
    if sorted(peaks) != sorted(raw["shape"].get("widths") or []):
        raise Refused(f"footprint {name}: measured widths {sorted(peaks)} are not the swept "
                      f"{raw['shape'].get('widths')}")
    return {"shape": raw["shape"], "peaks": peaks}


# --- the definitions ---------------------------------------------------------------------------


def loss_steps(master: Sequence[float],
               others: Mapping[str, Sequence[float]]) -> tuple[list[int], list[int]]:
    """``(S, excluded)``: the window's steps with ``letter_master > 0``, and the rest.

    Taken once from master. Every other arm must have its letter channel empty at exactly
    the same steps (the batch plan is shared), and ``S`` must hold ``LOSS_STEPS_MIN`` steps.
    """
    lo, hi = LOSS_WINDOW
    logs = {"master": master, **others}
    for name, log in logs.items():
        if len(log) < hi:
            raise Refused(f"{name}: the letter log is shorter than step {hi}")
        neg = [s for s in range(lo, hi + 1) if log[s - 1] < 0.0]
        if neg:
            raise Refused(f"{name}: negative letter loss at step(s) {neg[:10]} in {lo}-{hi}")
    used = [s for s in range(lo, hi + 1) if master[s - 1] > 0.0]
    excluded = [s for s in range(lo, hi + 1) if not master[s - 1] > 0.0]
    for name, log in others.items():
        zero = [s for s in range(lo, hi + 1) if not log[s - 1] > 0.0]
        if zero != excluded:
            differ = sorted(set(zero) ^ set(excluded))
            raise Refused(
                f"{name}: the letter channel is empty at different steps than master's in "
                f"{lo}-{hi} ({len(differ)} differ, first {differ[:10]}); the arms did not "
                "train on the same batch plan"
            )
    if len(used) < LOSS_STEPS_MIN:
        raise Refused(
            f"only {len(used)} step(s) in {lo}-{hi} carry a letter loss ({len(excluded)} are "
            f"span-only), below the {LOSS_STEPS_MIN} the reading needs"
        )
    return used, excluded


def loss_gap(a: Sequence[float], b: Sequence[float], steps: Sequence[int]) -> float:
    """Mean over ``steps`` (``S`` from ``loss_steps``) of ``|a - b| / b``."""
    if not steps:
        raise Refused("loss_gap over an empty step set")
    if any(not b[s - 1] > 0.0 for s in steps):
        raise Refused("loss_gap's step set holds a step where the reference letter loss is 0")
    return float(np.mean([abs(a[s - 1] - b[s - 1]) / b[s - 1] for s in steps]))


def bootstrap_means(d: np.ndarray) -> np.ndarray:
    """The ``BOOTSTRAP_DRAWS`` paired-bootstrap means of ``d``, in draw order.

    Each draw is its own ``rng.integers(0, n, size=n)`` call, as the ruling states it; only
    the means are batched, ``BOOTSTRAP_CHUNK`` draws at a time, to bound the index matrix.
    """
    n = len(d)
    if n == 0:
        raise Refused("no paired val rows to bootstrap")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_DRAWS)
    for start in range(0, BOOTSTRAP_DRAWS, BOOTSTRAP_CHUNK):
        k = min(BOOTSTRAP_CHUNK, BOOTSTRAP_DRAWS - start)
        idx = np.stack([rng.integers(0, n, size=n) for _ in range(k)])
        means[start:start + k] = d[idx].mean(axis=1)
    return means


def val_gap(a: Mapping[str, bool], b: Mapping[str, bool]) -> tuple[float, float, float, int]:
    """``(gap, lo, hi, n)``: top-1 of a minus b over the paired rows, and its interval."""
    if set(a) != set(b):
        raise Refused(f"the arms scored different val rows ({len(set(a) ^ set(b))} differ)")
    ids = sorted(a)
    d = np.array([float(a[i]) - float(b[i]) for i in ids])
    lo, hi = np.percentile(bootstrap_means(d), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), len(ids)


def decide(*, gap: float, floor: float, val: tuple[float, float], peak_ratio: float,
           step_ratio: float) -> str:
    """The first matching word after ``refused``, in the DRAFT's order."""
    values = (gap, floor, *val, peak_ratio, step_ratio)
    if not all(math.isfinite(x) for x in values):
        return "refused"
    threshold = max(2.0 * floor, LOSS_GAP_MIN)
    inside = VAL_BAND[0] <= val[0] and val[1] <= VAL_BAND[1]
    if gap > threshold or not inside:
        return "diverges"
    if peak_ratio > PEAK_RATIO_MAX:
        return "no_saving"
    if step_ratio > STEP_RATIO_MAX:
        return "slow"
    return "admissible"


def read(arms: Mapping[str, ArmInputs | None], footprints: Mapping[str, Path | None],
         ledgers: Sequence[Path]) -> Reading:
    reading = Reading(word="refused")
    n = reading.numbers

    def attempt(fn, *args):
        try:
            return fn(*args)
        except Refused as exc:
            reading.refusals.append(str(exc))
            return None

    rows = attempt(_rows_by_id, ledgers) or {}
    read_arms: dict[str, Arm] = {}
    for name in ARMS:
        given = arms.get(name)
        if given is None:
            reading.refusals.append(f"arm {name} not given")
            continue
        got = attempt(read_arm, name, given, rows)
        if got is not None:
            read_arms[name] = got
    prints: dict[str, dict[str, Any]] = {}
    for name in FOOTPRINTS:
        path = footprints.get(name)
        if path is None:
            reading.refusals.append(f"footprint {name} not given")
            continue
        got = attempt(read_footprint, name, path)
        if got is not None:
            prints[name] = got

    if len(read_arms) == len(ARMS):
        digests = {a.consumed_digest for a in read_arms.values()}
        if len(digests) != 1:
            reading.refusals.append(
                f"the arms consumed different data (consumed_digest {sorted(digests)})"
            )
        recipes = {
            name: {k: v for k, v in (a.ft.recipe or {}).items() if k != "optimizer_recipe"}
            for name, a in read_arms.items()
        }
        base = recipes["master"]
        differ = sorted({
            k for r in recipes.values() for k in set(r) | set(base) if r.get(k) != base.get(k)
        })
        if differ:
            reading.refusals.append(f"the ft recipes differ in more than the optimizer: {differ}")
        m, r, k = read_arms["master"], read_arms["master-repeat"], read_arms["kahan"]
        window = attempt(loss_steps, m.letter, {r.name: r.letter, k.name: k.letter})
        floor = gap = None
        if window is not None:
            floor = attempt(loss_gap, r.letter, m.letter, window[0])
            gap = attempt(loss_gap, k.letter, m.letter, window[0])
        vg = attempt(val_gap, k.choice, m.choice)
        n.update({
            "loss_gap_steps_used": None if window is None else len(window[0]),
            "loss_gap_steps_excluded": None if window is None else window[1],
            "loss_gap_kahan_master": gap, "floor": floor,
            "loss_gap_threshold": None if floor is None else max(2.0 * floor, LOSS_GAP_MIN),
            "val_top1_choice": {a.name: sum(a.choice.values()) / len(a.choice)
                                for a in read_arms.values()},
            "val_gap_kahan_master": None if vg is None else vg[0],
            "val_gap_interval": None if vg is None else [vg[1], vg[2]],
            "val_rows_paired": None if vg is None else vg[3],
            "optimizer_step_s_median": {a.name: a.apply_s for a in read_arms.values()},
            "step_time_s_median": {a.name: a.step_s for a in read_arms.values()},
            "step_ratio": k.apply_s / m.apply_s,
            "step_ratio_repeat_master": r.apply_s / m.apply_s,
            "step_time_ratio_kahan_master": k.step_s / m.step_s,
        })
    if len(prints) == len(FOOTPRINTS):
        a, b = prints["kahan"], prints["master"]
        if a["shape"] != b["shape"]:
            diff = sorted(x for x in set(a["shape"]) | set(b["shape"])
                          if a["shape"].get(x) != b["shape"].get(x))
            reading.refusals.append(f"the two footprints measured different shapes: {diff}")
        elif sorted(a["peaks"]) != sorted(b["peaks"]):
            reading.refusals.append("the two footprints measured different widths")
        else:
            ratios = {w: a["peaks"][w] / b["peaks"][w] for w in sorted(b["peaks"])}
            n["peak_bytes"] = {"master": b["peaks"], "kahan": a["peaks"]}
            n["peak_ratio_by_width"] = ratios
            n["peak_ratio"] = max(ratios.values())
        n["footprint_shape"] = b["shape"]

    if reading.refusals:
        return reading
    reading.word = decide(
        gap=n["loss_gap_kahan_master"], floor=n["floor"], val=tuple(n["val_gap_interval"]),
        peak_ratio=n["peak_ratio"], step_ratio=n["step_ratio"],
    )
    if reading.word == "refused":
        reading.refusals.append("a computed value is not finite")
    return reading


# --- the command -------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", type=Path, action="append", default=[])
    ap.add_argument("--arm", nargs=4, action="append", default=[],
                    metavar=("NAME", "CHECKPOINT", "FT_ROW_ID", "VERDICTS_JSONL"))
    ap.add_argument("--footprint", nargs=2, action="append", default=[],
                    metavar=("RECIPE", "FOOTPRINT_JSON"))
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the reading as JSON here, atomically; refused if it exists")
    args = ap.parse_args(argv)
    if args.out is not None and args.out.exists():
        raise SystemExit(f"--out {args.out} already exists; refusing to overwrite it")
    arms: dict[str, ArmInputs | None] = {}
    for name, ckpt, row_id, verdicts in args.arm:
        if name not in ARMS or name in arms:
            raise SystemExit(f"--arm {name!r}: one each of {list(ARMS)}")
        arms[name] = ArmInputs(Path(ckpt), row_id, Path(verdicts))
    footprints: dict[str, Path | None] = {}
    for name, path in args.footprint:
        if name not in FOOTPRINTS or name in footprints:
            raise SystemExit(f"--footprint {name!r}: one each of {list(FOOTPRINTS)}")
        footprints[name] = Path(path)
    reading = read(arms, footprints, args.ledger)
    body = json.dumps(reading.to_json(), indent=2, sort_keys=True)
    print(body)
    print(f"P1: {reading.word}")
    if args.out is not None:
        try:
            write_text_atomic(args.out, body + "\n")
        except ReplayRefusal as exc:
            raise SystemExit(f"--out: {exc}") from exc
    return 3 if reading.word == "refused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
