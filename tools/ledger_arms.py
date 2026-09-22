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

## One run per arm

An arm is a population of SEEDS, so no seed may be measured in one twice. The grouping above
cannot see a re-run -- it shares the recipe and the backbone of the run it repeats -- and
this file used to count every row as a new seed. On 2026-09-22 six committed ledgers
measured a seed twice inside one arm: three re-runs at later code, two reproducibility pairs
at one code, and one pair of runs over two data snapshots. The four-way ``after`` arm was
reported from one of them as n=13 over 8 seeds. Pooled, a repeated seed overstates n and
understates the spread, and both floors are computed from the two: the flattering direction.

So a seed measured twice in one arm is refused, naming the rows. ``--split-by-code`` is the
named way to read a re-run: one arm per code version, keyed on the ``code_that_ran`` digest
and falling back to the launch rev. Not on the rev alone -- all three re-runs share their
launch rev with the run they repeat, so grouping per rev would have pooled them as before. A
repeat that survives the split has nothing in its code to tell the runs apart and stays
refused: two runs of one seed at one code are a reproducibility check, and summarised as
seeds they would print run-to-run noise as a seed spread.

Not dedupe-to-one-run either, because that has to choose a run. "Keep the last" is re-running
until the number suits; "keep the first" would have built the four-way arm from two code
versions. Rows that measured nothing may repeat a seed -- the chain ledgers' build, smoke and
verdict rows are all seed 0, and a killed run leaves a row beside its retry. None of them
enters a sample, so they are counted and said rather than refused.

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
    python tools/ledger_arms.py --split-by-code ledger/gh200-fourway-2026-09-22.jsonl
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import statistics
import sys
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from qd_train.ledger import CODE_THAT_RAN
from qd_train.power import estimated_sd_penalty, resolvable_difference

#: The held-out result, and the gate that says whether the arm fit at all. Both are written
#: by ``tools/rung0_real_run.py``; an arm missing either is reported unmeasured, never
#: silently dropped, because a seed that recorded nothing and a seed that scored zero are
#: different facts.
VAL = "val_choice_top1_over_baseline"
TRAIN = "train_choice_top1_over_train_majority"

#: The named override. See "One run per arm" in the module docstring.
SPLIT_BY_CODE = "--split-by-code"

#: An arm's identity: recipe, backbone, and -- only under ``--split-by-code`` -- the code its
#: rows ran. ``None`` in the third place means arms were not split, not that no code ran.
ArmKey = tuple[str, str, str | None]


@dataclass(frozen=True, slots=True)
class Arm:
    """One (recipe, backbone) population, with its two samples kept apart."""

    recipe_hash: str
    backbone_commit: str
    #: The code every row of this arm ran, when arms are split by it; ``None`` when not.
    code: str | None = None
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
    #: Which code each measured row ran, counted. Two entries are two runs in one arm.
    codes: collections.Counter[str] = field(default_factory=collections.Counter)
    #: Rows beyond the first for their seed. None of them is a second measurement of it --
    #: that is refused -- so they make the row count larger than the seed count and move
    #: no sample.
    repeated_rows: int = 0
    #: Measured rows carrying no ``protocol.seed``, among which a repeat cannot be ruled out.
    unseeded: int = 0

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
        mattered. Split by code, a run and its re-run share both halves, so the code is the
        third part of the name.
        """
        name = f"{self.backbone_commit} recipe {self.recipe_hash[:10]}"
        return name if self.code is None else f"{name} code {self.code}"

    @property
    def key(self) -> tuple[str, str, str]:
        """Sort order. On the full key, so the report is deterministic when one half ties."""
        return (self.backbone_commit, self.recipe_hash, self.code or "")


class RepeatedSeed(ValueError):
    """A population of seeds would hold one seed twice; raised, naming the rows, instead."""


#: How many repeated seeds a refusal names before summarising the rest by count.
REPEATS_SHOWN = 5


class SeedClaims:
    """Which rows claimed each seed of each population, so a repeat is refused, not pooled.

    The one owner of "a population of seeds holds each seed once" for the two tools that read
    populations back from rows: this file and ``tools/operator_holdout_report.py``. Each
    decides which rows belong to which population and how a row is named; this decides what
    a repeat is and how a refusal reads -- counted in seeds rather than in extra rows, with
    every row of a repeated seed named, and capped with the count of what it did not show.
    """

    def __init__(self) -> None:
        self._claims: dict[tuple[Hashable, object], tuple[str, list[dict]]] = {}

    def claim(self, population: Hashable, seed: object, row: dict, *, label: str) -> None:
        """Record ``row`` as seed ``seed`` of ``population``, which a refusal calls ``label``."""
        self._claims.setdefault((population, seed), (label, []))[1].append(row)

    def repeated(self) -> list[tuple[str, object, list[dict]]]:
        """``(label, seed, rows)`` for every seed claimed more than once, first claim first."""
        return [
            (label, seed, rows)
            for (_, seed), (label, rows) in self._claims.items()
            if len(rows) > 1
        ]

    def refuse(self, *, within: str, name: Callable[[dict], str], why: str) -> None:
        """Raise :class:`RepeatedSeed` over every repeated seed, or return if there is none."""
        repeated = self.repeated()
        if not repeated:
            return
        lines = [
            f"{label} seed {seed}: " + _joined([name(row) for row in rows])
            for label, seed, rows in repeated
        ]
        shown = lines[:REPEATS_SHOWN]
        raise RepeatedSeed(
            f"refusing to pool: {len(lines)} seed(s) appear more than once in {within} -- "
            f"showing {len(shown)} of {len(lines)}:\n  " + "\n  ".join(shown) + "\n" + why
        )


def _joined(names: list[str]) -> str:
    """``a and b``, or ``a, b and c``."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def metric_of(row: dict, key: str) -> dict | None:
    """One metric off one row, or ``None`` when the row does not carry it.

    Public because ``tools/operator_holdout_report.py`` reads the same rows for a different
    question and a second copy of "how a metric is read off a row" is a second thing to
    keep in agreement with the ledger's shape. It returns ``None`` rather than ``{}`` for a
    missing metric so the caller must decide what an absent measurement means, instead of
    reading a default and calling it zero.
    """
    got = row.get("metrics", {}).get(key)
    return got if isinstance(got, dict) else None


def code_of(row: dict) -> str:
    """The code a row ran, as precisely as the row states it, and never an empty string.

    The digest over the code the run imported where the row has one -- comparable across
    hosts, and the one field that tells each committed re-run from the run it repeats --
    else the launch rev, else ``unrecorded``. Public, like :func:`metric_of`, because
    ``tools/operator_holdout_report.py`` asks the same question of the same rows. The
    metric's name is imported from its writer rather than spelled here.
    """
    ran = metric_of(row, CODE_THAT_RAN)
    if ran is not None and ran.get("state") == "ran" and ran.get("value"):
        return str(ran["value"])[:16]
    rev = (row.get("recipe") or {}).get("rev")
    return f"rev {str(rev)[:12]}" if rev else "unrecorded"


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


def arms_of(rows: list[dict], *, split_by_code: bool = False) -> dict[ArmKey, Arm]:
    """Group rows into arms by ``(recipe_hash, backbone_commit)`` -- and by :func:`code_of`
    as well when ``split_by_code`` -- refusing any arm that would measure one seed twice.

    Raises :class:`RepeatedSeed`, naming the rows. See "One run per arm" in the module
    docstring.
    """
    members: dict[ArmKey, list[dict]] = {}
    for row in rows:
        protocol = row["protocol"]
        key = (
            protocol["recipe_hash"],
            protocol["backbone_commit"],
            code_of(row) if split_by_code else None,
        )
        members.setdefault(key, []).append(row)

    claims = SeedClaims()
    arms = {key: _arm(key, group, claims) for key, group in members.items()}
    repeated = claims.repeated()
    if repeated:
        claims.refuse(
            within="the measured rows of one arm",
            name=_row_name,
            why=_why_refused(repeated, split_by_code=split_by_code),
        )
    return arms


def _arm(key: ArmKey, rows: list[dict], claims: SeedClaims) -> Arm:
    """One arm from its rows, with every measured seed claimed so a repeat can be refused."""
    values: list[float] = []
    fitted: list[float] = []
    codes: collections.Counter[str] = collections.Counter()
    seeds: collections.Counter[object] = collections.Counter()
    measured: list[tuple[object, dict]] = []
    collapsed = unmeasured = unseeded = 0
    baseline: float | None = None
    for row in rows:
        seed = row["protocol"].get("seed")
        if seed is not None:
            seeds[seed] += 1
        val, train = metric_of(row, VAL), metric_of(row, TRAIN)
        if val is None or train is None or "passed" not in train:
            unmeasured += 1
            continue
        if seed is None:
            unseeded += 1
        else:
            measured.append((seed, row))
        if baseline is None:
            baseline = _percent_after(val, "majority-class baseline of ")
        values.append(float(val["value"]))
        codes[code_of(row)] += 1
        if train["passed"]:
            fitted.append(float(val["value"]))
        else:
            collapsed += 1

    arm = Arm(
        recipe_hash=key[0],
        backbone_commit=key[1],
        code=key[2],
        values=values,
        fitted=fitted,
        collapsed=collapsed,
        unmeasured=unmeasured,
        baseline=baseline,
        codes=codes,
        repeated_rows=sum(n - 1 for n in seeds.values()),
        unseeded=unseeded,
    )
    for seed, row in measured:
        claims.claim(key, seed, row, label=arm.label)
    return arm


def _row_name(row: dict) -> str:
    """A row as a refusal names it: whatever tells a re-run from the run it repeats."""
    protocol = row.get("protocol") or {}
    return (
        f"row {row.get('row_id', '?')} (code {code_of(row)}, "
        f"data {str(protocol.get('data_snapshot_hash', '?'))[:10]}, "
        f"written {str(row.get('written_at', '?'))[:19]})"
    )


def _why_refused(
    repeated: list[tuple[str, object, list[dict]]], *, split_by_code: bool
) -> str:
    """What pooling these would have printed, and which of them the named flag separates."""
    across = sum(1 for _, _, rows in repeated if len({code_of(r) for r in rows}) > 1)
    within = len(repeated) - across
    lines = [
        "Pooled, each of these seeds would count as two: n overstated, the spread "
        "understated, and every floor computed from both -- the flattering direction. An "
        "arm is one run of its seeds."
    ]
    if across:
        lines.append(
            f"{across} of them repeat across code versions -- a re-run beside the run it "
            f"repeats: pass {SPLIT_BY_CODE} to read each code version as its own arm."
        )
    if within:
        verdict = "does not" if split_by_code else "would not"
        lines.append(
            f"{within} of them repeat under one code identity -- one digest, one launch rev, "
            f"or none recorded -- which {SPLIT_BY_CODE} {verdict} separate. "
            "Where the rows name one data snapshot this is a reproducibility check: compare "
            "the rows with each other, because they are not a population of seeds. Where the "
            "snapshots differ they are two arms this key cannot tell apart."
        )
    return "\n".join(lines)


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


def render(arms: dict[ArmKey, Arm]) -> list[str]:
    lines: list[str] = []
    for arm in sorted(arms.values(), key=lambda a: a.key):
        lines.append(f"=== {arm.label} ===")
        lines.append(
            f"    rows {arm.n_rows}   measured {len(arm.values)}   "
            f"unmeasured {arm.unmeasured}"
        )
        if arm.repeated_rows:
            lines.append(
                f"    {arm.repeated_rows} of these {arm.n_rows} rows repeat a seed already in "
                "this arm without measuring it again: the counts above are rows, not seeds, "
                "and no sample below counts a seed twice"
            )
        if arm.unseeded:
            lines.append(
                f"    {arm.unseeded} measured row(s) carry no protocol.seed, so a seed "
                "counted twice among them cannot be ruled out"
            )
        if len(arm.codes) > 1:
            ran = ", ".join(f"{code} ({n} measured)" for code, n in arm.codes.most_common())
            lines.append(
                f"    ARM RAN DIFFERENT CODE: {ran}. Its seeds are not one run, and the "
                f"spread below includes whatever the change did; {SPLIT_BY_CODE} reads each "
                "version as its own arm"
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


def render_pairwise(arms: dict[ArmKey, Arm]) -> list[str]:
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


def render_gates(rows: list[dict]) -> list[str]:
    """Which gates and controls actually reached these rows, and what they said.

    Across the 988 rows that preceded 2026-09-21 the answer was: none of them, for three
    different reasons -- built and unreachable, built and never called, named and never
    built. A summary that showed only accuracies made that invisible for months, because a
    gate nobody evaluates and a gate that passes look identical in a mean.

    So `not_run` is counted and printed beside `pass` and `fail` rather than omitted. A
    check that could not run must never report the same result as one that ran and passed.
    """
    lines = ["", "=== gates and controls, over every row read ==="]
    for block in ("gates", "controls"):
        names = sorted({name for row in rows for name in (row.get(block) or {})})
        if not names:
            continue
        lines.append(f"  {block}:")
        for name in names:
            states = collections.Counter()
            values: list[float] = []
            detail = ""
            for row in rows:
                entry = (row.get(block) or {}).get(name) or {}
                if entry.get("state") != "ran":
                    states["not_run"] += 1
                    continue
                states["pass" if entry.get("passed") else "FAIL"] += 1
                if isinstance(entry.get("value"), int | float):
                    values.append(float(entry["value"]))
                detail = detail or str(entry.get("detail") or "")
            summary = "  ".join(f"{k} {v}" for k, v in sorted(states.items()))
            line = f"    {name:28} {summary}"
            if values:
                line += f"   value mean {statistics.mean(values):+.4f}"
                if len(values) > 1:
                    line += f" sd {statistics.stdev(values):.4f}"
            lines.append(line)
            if detail:
                lines.append(f"        {detail[:150]}")
    return lines


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(
        prog="ledger_arms.py",
        description="Summarise seed sweeps from their ledger rows; see the module docstring.",
    )
    parser.add_argument("ledgers", nargs="+", type=Path, help="ledger files, every one read")
    parser.add_argument(
        SPLIT_BY_CODE,
        action="store_true",
        help=f"read each code version ({CODE_THAT_RAN}, else the launch rev) as its own arm "
        "instead of refusing an arm whose seeds were measured again at other code",
    )
    args = parser.parse_args(argv)
    rows = read_rows(args.ledgers)
    try:
        arms = arms_of(rows, split_by_code=args.split_by_code)
    except RepeatedSeed as refused:
        print(refused, file=sys.stderr)
        return 2
    split = ", one per code version" if args.split_by_code else ""
    print(f"{len(rows)} row(s) over {len(arms)} arm(s){split}\n")
    for line in render(arms) + render_pairwise(arms) + render_gates(rows):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
