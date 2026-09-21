"""Summarise a seed sweep from its LEDGER ROWS, with the floor that applies to it.

Rule 5: *"A number in a report cites a ledger row or is not in the report."* Nothing in
``tools/`` read rows back until this file; every runner here writes them and the reading was
done by parsing each sweep's stdout. That is not a stylistic difference. On 2026-09-21 a
log-parsing script reported "capacity helps +2.59pp" from the 4096 e10 arms, because the
tool printed a collapse warning *beside* the accuracy lines and the parser read the
accuracies. The rows carry the collapse verdict as a field, so reading rows makes that
particular mistake unavailable rather than merely discouraged.

## Three things this gets right that a log parser did not

**Grouping is on ``(recipe_hash, backbone_commit)``.** A capacity sweep's arms share one
recipe -- same epochs, batch size, lr, span weight, subsample, rev -- and differ only in the
model, which lives in ``backbone_commit``. Grouping on ``recipe_hash`` alone merges three
8-seed arms into one 24-seed population whose spread is mostly the capacity difference. I
made exactly that mistake reading these rows before finding the second field, and for a
minute believed the sweep's own arms were indistinguishable in the ledger.

**Collapse comes from the gate, not from prose.** A seed has no fit to generalise when its
training accuracy is at or below the training-set majority share, and
``train_choice_top1_over_train_majority`` already records that verdict as ``passed``. Its
``detail`` says the same thing in English, and re-deriving the verdict by parsing that
sentence would make this file a second owner of the answer, free to disagree with the first
after a wording change. Checked against all 24 e10 rows: the flag and the sentence agree on
every one, so the flag is used and the sentence is only displayed.

**A collapsed arm's spread is reported separately from a fitted arm's.** A constant
predictor scores the majority-class baseline, so a collapsed arm's held-out mean sits *at*
the baseline and looks like a near-miss; and it has almost nothing to vary, so its sd is
tight and looks like precision. Measured on the e10 arms: 1.11pp and 0.67pp where the
fitted e30 arms gave 4.41-5.58pp. Priming a later sweep with a collapsed arm's sd as its
prior would claim roughly three times the sensitivity the design has. Both samples are
printed, neither stands in for the other, and an arm with fewer than two fitted seeds says
so rather than reporting a spread over one.

**Every floor is printed twice, known-sd and estimated-sd.** ``resolvable_difference``
assumes ``sd`` is known; here it is estimated from the same handful of runs, so the honest
quantile is Student's t and the bound is wider -- by 7.5% to 32.7% over the seed counts
these sweeps use. Both numbers appear, and an arm-vs-arm difference is judged against the
honest one, with ``(clears the known-sd bound only)`` said explicitly when it falls between
them. That band is exactly where a null gets read as a finding.

Printed as numbers rather than as a percentage for the reader to apply. On 2026-09-21 an
AUDIT file quoted ``power.py``'s own 14% figure against a comparison whose true penalty was
16.4%, because the 14% belongs to the two-sample n=5 case and the comparison was one-sample
at n=8 -- and it erred on the flattering side. One more step is one more place to pick the
wrong case.

## What this does not compute

Either floor. ``qd_train.power`` owns both: ``resolvable_difference`` for the known-sd
bound, deriving its constant from ``NormalDist`` rather than hard-coding ~2.80, and
``estimated_sd_penalty`` for the multiplier, which picks ``df`` itself -- ``n-1`` against a
fixed reference, ``2(n-1)`` against another arm. Both make ``against_known_reference`` a
required argument, because the two answers differ by 1.41x and the wrong one is always the
flattering one. A first draft of this file carried its own ``Z = 2.801585``; that is a
second owner of one number, and it was deleted rather than kept in agreement by hand.

Usage::

    python tools/ledger_arms.py ledger/gh200-rung0-capacity-4096-e30-2026-09-21.jsonl
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from qd_train.power import estimated_sd_penalty, resolvable_difference

#: The held-out result, and the gate that says whether the arm fit at all. Both are written
#: by ``tools/rung0_real_run.py``; an arm missing either is reported unmeasured, never
#: silently dropped, because a seed that recorded nothing and a seed that scored zero are
#: different facts.
VAL = "val_choice_top1_over_baseline"
TRAIN = "train_choice_top1_over_train_majority"


@dataclass(frozen=True, slots=True)
class Arm:
    """One (recipe, backbone) population, with its two samples kept apart."""

    recipe_hash: str
    backbone_commit: str
    #: Held-out accuracy for every seed that recorded both metrics.
    values: list[float] = field(default_factory=list)
    #: The subset of ``values`` whose seed cleared the training-set majority.
    fitted: list[float] = field(default_factory=list)
    #: Seeds that recorded both metrics and did not clear it.
    collapsed: int = 0
    #: Seeds whose row is missing one of the two metrics. Not a zero, not a pass.
    unmeasured: int = 0
    #: The majority-class baseline, or ``None`` when no row stated it readably.
    baseline: float | None = None

    @property
    def n_rows(self) -> int:
        return len(self.values) + self.unmeasured

    @property
    def label(self) -> str:
        """Both halves of the key, because either one alone names two different arms.

        A capacity sweep's arms share a recipe and differ in the backbone; a learning
        curve's points share a backbone and differ in the recipe, through
        ``train_subsample``. Labelling an arm by its backbone alone is fine for the first
        and prints "X minus X" for the second -- which is what the first version of this
        file did on the curve's own rows, found by running it on them before the numbers
        mattered.
        """
        return f"{self.backbone_commit} recipe {self.recipe_hash[:10]}"

    @property
    def key(self) -> tuple[str, str]:
        """Sort order. On the full key, so the report is deterministic when one half ties."""
        return (self.backbone_commit, self.recipe_hash)


def _metric(row: dict, key: str) -> dict | None:
    got = row.get("metrics", {}).get(key)
    return got if isinstance(got, dict) else None


def _percent_after(metric: dict | None, marker: str) -> float | None:
    """A percentage the metric states in its own detail line, or ``None``.

    Display only. Nothing this function returns decides whether a seed fit.
    """
    if metric is None or marker not in metric.get("detail", ""):
        return None
    try:
        return float(metric["detail"].split(marker, 1)[1].split("%", 1)[0]) / 100.0
    except ValueError:
        return None


def arms_of(rows: list[dict]) -> dict[tuple[str, str], Arm]:
    """Group rows into arms by ``(recipe_hash, backbone_commit)``."""
    out: dict[tuple[str, str], Arm] = {}
    for row in rows:
        protocol = row["protocol"]
        key = (protocol["recipe_hash"], protocol["backbone_commit"])
        arm = out.get(key)
        if arm is None:
            arm = Arm(recipe_hash=key[0], backbone_commit=key[1])
            out[key] = arm

        val, train = _metric(row, VAL), _metric(row, TRAIN)
        if val is None or train is None or "passed" not in train:
            out[key] = _bump(arm, unmeasured=1)
            continue
        base = arm.baseline
        if base is None:
            base = _percent_after(val, "majority-class baseline of ")
        arm.values.append(float(val["value"]))
        if train["passed"]:
            arm.fitted.append(float(val["value"]))
            out[key] = _bump(arm, baseline=base)
        else:
            out[key] = _bump(arm, collapsed=1, baseline=base)
    return out


def _bump(
    arm: Arm, *, collapsed: int = 0, unmeasured: int = 0, baseline: float | None = None
) -> Arm:
    """A new ``Arm`` with the counters advanced; the two lists are shared by reference."""
    return Arm(
        recipe_hash=arm.recipe_hash,
        backbone_commit=arm.backbone_commit,
        values=arm.values,
        fitted=arm.fitted,
        collapsed=arm.collapsed + collapsed,
        unmeasured=arm.unmeasured + unmeasured,
        baseline=arm.baseline if baseline is None else baseline,
    )


def read_rows(paths: list[Path]) -> list[dict]:
    """Every row in every named ledger. A missing file raises rather than reading fewer."""
    rows: list[dict] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is not a file. A report over the ledgers that happened to exist "
                "would be a coverage claim nobody could reconstruct."
            )
        text = path.read_text(encoding="utf-8")
        rows += [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    return rows


def _floors(sd: float, n: int, *, against_known_reference: bool) -> tuple[float, float]:
    """The known-sd bound and the honest one, as two numbers rather than one and a caveat.

    ``resolvable_difference`` assumes ``sd`` is known; here it is estimated from the same
    handful of runs, so the honest quantile is Student's t and the bound is wider.
    ``estimated_sd_penalty`` owns the multiplier and picks ``df`` for the one- and
    two-sample cases -- n-1 against a fixed reference, 2(n-1) against another arm.

    Printed as a number, not as a percentage for the reader to apply. On 2026-09-21 an audit
    quoted this module's own 14% figure against a comparison whose true penalty was 16.4%,
    because the 14% belongs to the two-sample n=5 case and the comparison was one-sample at
    n=8. One more step is one more place to pick the wrong case.
    """
    known = resolvable_difference(
        sd=sd, n_per_arm=n, against_known_reference=against_known_reference
    )
    penalty = estimated_sd_penalty(
        n_per_arm=n, against_known_reference=against_known_reference
    )
    return known, known * penalty


def _spread(label: str, sample: list[float], baseline: float | None) -> list[str]:
    if len(sample) < 2:
        return [f"    {label}: n={len(sample)} -- no spread to report"]
    mean, sd = statistics.mean(sample), statistics.stdev(sample)
    ref_known, ref_honest = _floors(sd, len(sample), against_known_reference=True)
    arm_known, arm_honest = _floors(sd, len(sample), against_known_reference=False)
    delta = "" if baseline is None else f"  ({(mean - baseline) * 100:+.2f}pp vs baseline)"
    return [
        f"    {label}: n={len(sample)} mean {mean * 100:.2f}% sd {sd * 100:.2f}pp{delta}",
        f"        smallest visible difference at this n, known-sd / estimated-sd: "
        f"{ref_known * 100:.2f} / {ref_honest * 100:.2f}pp against a fixed reference, "
        f"{arm_known * 100:.2f} / {arm_honest * 100:.2f}pp against another arm",
    ]


def render(arms: dict[tuple[str, str], Arm]) -> list[str]:
    lines: list[str] = []
    for arm in sorted(arms.values(), key=lambda a: a.key):
        lines.append(f"=== {arm.label} ===")
        lines.append(
            f"    rows {arm.n_rows}   measured {len(arm.values)}   "
            f"unmeasured {arm.unmeasured}"
        )
        if arm.baseline is None:
            lines.append("    majority-class baseline: NOT STATED by any row -- deltas omitted")
        else:
            lines.append(f"    majority-class baseline on val: {arm.baseline * 100:.1f}%")
        if arm.collapsed:
            lines.append(
                f"    COLLAPSED on {arm.collapsed} of {arm.n_rows} seed(s): training accuracy "
                "at or below the training-set majority. Those seeds' held-out numbers are "
                "not evidence about generalisation -- there was no fit to generalise."
            )
        lines += _spread("all seeds", arm.values, arm.baseline)
        lines += _spread("fitted seeds", arm.fitted, arm.baseline)
        lines.append("")
    return lines


def render_pairwise(arms: dict[tuple[str, str], Arm]) -> list[str]:
    """Arm-vs-arm differences, at the two-arm floor rather than the one-sample one."""
    lines = ["=== arm vs arm (fitted seeds; pooled sd; the sqrt(2) floor) ==="]
    pairs = 0
    ordered = sorted(arms.values(), key=lambda a: a.key)
    for left, right in itertools.combinations(ordered, 2):
        if len(left.fitted) < 2 or len(right.fitted) < 2:
            continue
        pairs += 1
        sl, sr = statistics.stdev(left.fitted), statistics.stdev(right.fitted)
        pooled = math.sqrt((sl**2 + sr**2) / 2.0)
        n = min(len(left.fitted), len(right.fitted))
        known, honest = _floors(pooled, n, against_known_reference=False)
        diff = statistics.mean(right.fitted) - statistics.mean(left.fitted)
        # Judged against the HONEST floor. A difference between the two bounds is one this
        # design has not been shown to see, and calling it visible is the flattering read.
        verdict = "VISIBLE" if abs(diff) > honest else "inside the floor -- NOT a difference"
        if known < abs(diff) <= honest:
            verdict += " (clears the known-sd bound only)"
        lines.append(
            f"    {right.label}\n        minus {left.label}\n        "
            f"{diff * 100:+.2f}pp   pooled sd {pooled * 100:.2f}pp   "
            f"floor at n={n} known-sd / estimated-sd {known * 100:.2f} / {honest * 100:.2f}pp"
            f"   {verdict}"
        )
    if not pairs:
        lines.append("    no pair of arms has two fitted seeds each; nothing to compare")
    return lines


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    rows = read_rows([Path(p) for p in argv])
    arms = arms_of(rows)
    print(f"{len(rows)} row(s) over {len(arms)} arm(s)\n")
    for line in render(arms) + render_pairwise(arms):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
