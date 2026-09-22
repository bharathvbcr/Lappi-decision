"""The model half of the operator-holdout finding, read from ledger rows.

``tools/operator_holdout.py`` asked the char-n-gram control whether it had learned the
change or the generator, and answered *the generator*: stub.panic 99.10% -> 3.08%,
logic.change_constant 91.12% -> 19.63%, cosmetic.rename_local 58.71% -> 0.00%. Its own
docstring says what it cannot say -- "This measures the CONTROL, not the model ... the model
may be doing exactly the same thing, and this does not check." This renders the check.

## The three arms, and why three

Two arms cannot answer it. Holding an operator out and watching accuracy fall is explained
as well by *the training set shrank* as by *the fingerprint went*, and on this corpus
stub.panic alone is 43% of the rows. So each operator gets:

* **reference** -- everything trained on, the operator's rows scored separately;
* **holdout** -- the operator's rows removed from training only, after the split;
* **sizematch** -- the same NUMBER of rows dropped at random instead.

reference minus holdout is the total effect. reference minus sizematch is what removing
that much data costs on its own. The difference between those two is what the operator's
absence cost beyond its volume.

## The fourth number, which is the one that makes the rest readable

Every operator in this corpus produces exactly one class, so holding one out also moves the
class prior -- removing stub.panic takes `stub` from about 46% of training to about 7%. A
model that then under-predicts `stub` scores badly on the held-out rows for a reason with
nothing to do with recognising generators, and the size-matched arm cannot separate the two
because dropping uniformly preserves the prior it would need to disturb.

So every arm also reports its **siblings**: validation rows of the same class from a
different operator, which training did see and which sit under the identical shifted prior.

* siblings hold while the operator collapses -> the prior is intact, the generator was the
  signal, and the model shares the control's shortcut;
* both collapse -> the arm moved the prior, and the operator's own number says little by
  itself;
* neither collapses -> the model reads the change rather than the generator, which is the
  outcome that would make the margin against the control a real statement about it.

## One run at a time

A cell is a population of SEEDS, so no seed may be in one twice. Grouping is by operator and
condition and deliberately not by rev -- the minor operators ran at a later commit than the
dominant three, and one report reads both -- which means a re-run of the same arms, appended
beside the run it repeats or passed in as a second ledger, would be pooled with it: sixteen
"seeds" where eight exist, a spread understated by counting each twice, and one run's
`not_run` margins averaged in beside the other's measured ones. That is refused, naming the
rows, rather than rendered.

## What this does not compute

Any threshold. It prints the three arms and their differences; nothing here passes or fails
anything. `paired_margin_vs_linear` is `not_run` on any arm whose training set had no cached
control when the arm started -- every holdout and size-matched arm until
`tools/fit_operator_holdout_controls.sh` has fitted one, and the arm has been re-run after it,
because a cached control does not backfill a row already written. The report prints the
reason each run recorded rather than assuming one. A `not_run` margin is reported as not
measured and never as a loss.

Usage::

    python tools/operator_holdout_report.py ledger/gh200-operator-holdout-model-*.jsonl
"""

from __future__ import annotations

import collections
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ledger_arms import metric_of, read_rows

#: Accuracy on the named operator's own validation rows, on every arm that named one.
OPERATOR = "val_choice_top1_on_measured_operator"
#: Same class, other generators. What separates a collapse from a moved prior.
SIBLINGS = "val_choice_top1_on_measured_operator_siblings"
#: Overall held-out accuracy, so an arm that collapsed everywhere is visible as such.
OVERALL = "val_choice_top1_over_baseline"
#: Whether the seed fit at all. A seed at or below the training majority has no fit to
#: generalise, and its held-out number is the prior rather than a measurement.
FIT = "train_choice_top1_over_train_majority"
#: The model against the char-n-gram control, paired over the same rows in the same order.
#: ``not_run`` on any arm whose training set has no fitted control -- which is every holdout
#: and size-matched arm until ``tools/fit_operator_holdout_controls.sh`` has been run, and
#: which must be read as NOT MEASURED rather than as a loss.
MARGIN = "paired_margin_vs_linear"
#: The sha256 over the code the run imported (``what_ran_state``), comparable across hosts.
CODE = "code_that_ran"

#: The three conditions, in the order they are read against each other.
REFERENCE, HOLDOUT, SIZEMATCH = "reference", "holdout", "sizematch"
ORDER = (REFERENCE, HOLDOUT, SIZEMATCH)


def condition_of(recipe: dict) -> tuple[str, str] | None:
    """``(operator, condition)`` for a row, or ``None`` when the row named no operator.

    Read from the recipe rather than from a run label, because the recipe is what the hash
    covers and a label is a string somebody typed. A row that named no operator carries
    neither metric and belongs to a different experiment; it is skipped rather than
    assigned a condition it never ran under.
    """
    held = recipe.get("hold_out_operator") or ""
    measured = recipe.get("measure_operator") or held
    if not measured:
        return None
    if held:
        return measured, HOLDOUT
    if recipe.get("drop_random_train"):
        return measured, SIZEMATCH
    return measured, REFERENCE


@dataclass(slots=True)
class Cell:
    """One (operator, condition) population. Seeds that measured nothing stay countable."""

    operator: str
    condition: str
    operator_acc: list[float] = field(default_factory=list)
    sibling_acc: list[float] = field(default_factory=list)
    overall_acc: list[float] = field(default_factory=list)
    #: Seeds whose training accuracy did not clear the training-set majority.
    collapsed: int = 0
    #: Seeds whose row is missing the operator metric. Not a zero and not a pass.
    unmeasured: int = 0
    #: Seeds whose sibling metric was recorded as not_run, with the reason kept.
    siblings_not_run: int = 0
    not_run_reason: str = ""
    #: Rows the measured operator produced, as the run counted them.
    n_operator_rows: int = 0
    n_sibling_rows: int = 0
    #: Paired margin against the control, for the seeds that had an opponent at all.
    margins: list[float] = field(default_factory=list)
    #: Seeds with no margin, and the reason each run recorded for it, counted per reason.
    #: Read from the row rather than assumed: "no control was cached" is the usual cause,
    #: but an unconverged control and an over-budget projection are recorded the same way.
    margin_not_run: int = 0
    margin_not_run_reasons: collections.Counter[str] = field(
        default_factory=collections.Counter
    )
    #: Rows that carry no ``protocol.seed``, among which a repeated seed cannot be ruled out.
    unseeded: int = 0
    #: Which code each row ran, counted: ``code_that_ran`` where the row has it, the launch
    #: rev where it does not. The arms of one operator are only a controlled comparison if
    #: they ran the same code, and nothing about grouping by condition guarantees that.
    code: collections.Counter[str] = field(default_factory=collections.Counter)

    @property
    def n_seeds(self) -> int:
        return len(self.operator_acc) + self.unmeasured


def code_of(row: dict) -> str:
    """The code a row ran, as precisely as the row states it, and never an empty string."""
    ran = metric_of(row, CODE)
    if ran is not None and ran.get("state") == "ran" and ran.get("value"):
        return str(ran["value"])[:16]
    rev = (row.get("recipe") or {}).get("rev")
    return f"rev {str(rev)[:12]}" if rev else "unrecorded"


class RepeatedSeed(ValueError):
    """Two rows claim the same seed of the same (operator, condition) cell."""


#: How many colliding seeds a refusal names before summarising the rest by count.
_REPEATS_SHOWN = 5
#: A recorded not_run reason is printed up to this many characters, then marked as cut.
_REASON_CHARS = 240


def cells_of(rows: list[dict]) -> dict[tuple[str, str], Cell]:
    """Group rows by ``(operator, condition)``, counting what each seed did not measure.

    Raises :class:`RepeatedSeed` when any cell would hold one seed twice. See "One run at a
    time" in the module docstring.
    """
    out: dict[tuple[str, str], Cell] = {}
    first_claim: dict[tuple[str, str, object], str] = {}
    repeats: list[str] = []
    for row in rows:
        key = condition_of(row.get("recipe") or {})
        if key is None:
            continue
        cell = out.setdefault(key, Cell(operator=key[0], condition=key[1]))
        cell.code[code_of(row)] += 1
        seed = (row.get("protocol") or {}).get("seed")
        if seed is None:
            cell.unseeded += 1
        else:
            where = (
                f"row {row.get('row_id', '?')} "
                f"(rev {str((row.get('recipe') or {}).get('rev', '?'))[:12]})"
            )
            claim = (*key, seed)
            if claim in first_claim:
                repeats.append(
                    f"{key[0]} {key[1]} seed {seed}: {first_claim[claim]} and {where}"
                )
            else:
                first_claim[claim] = where
        op, sib = metric_of(row, OPERATOR), metric_of(row, SIBLINGS)
        fit, overall = metric_of(row, FIT), metric_of(row, OVERALL)
        # Read from gates or metrics: the runner records the margin as a GATE, and reading
        # only one of the two containers would report every arm as unmeasured on a ledger
        # whose shape had moved rather than on a run that had no control.
        margin = (row.get("gates") or {}).get(MARGIN) or metric_of(row, MARGIN)
        if isinstance(margin, dict) and margin.get("state") == "ran":
            cell.margins.append(float(margin["value"]))
        else:
            cell.margin_not_run += 1
            if not isinstance(margin, dict):
                why = f"the row carries no {MARGIN} at all"
            else:
                why = str(margin.get("reason") or "not_run, with no reason recorded")
            cell.margin_not_run_reasons[why] += 1
        if op is None or op.get("state") != "ran":
            cell.unmeasured += 1
            continue
        cell.operator_acc.append(float(op["value"]))
        cell.n_operator_rows = max(cell.n_operator_rows, int(op.get("n_total") or 0))
        if fit is not None and fit.get("state") == "ran" and not fit.get("passed"):
            cell.collapsed += 1
        if overall is not None and overall.get("state") == "ran":
            cell.overall_acc.append(float(overall["value"]))
        if sib is None or sib.get("state") != "ran":
            cell.siblings_not_run += 1
            if sib is not None and sib.get("reason"):
                cell.not_run_reason = str(sib["reason"])
        else:
            cell.sibling_acc.append(float(sib["value"]))
            cell.n_sibling_rows = max(cell.n_sibling_rows, int(sib.get("n_total") or 0))
    if repeats:
        shown = repeats[:_REPEATS_SHOWN]
        raise RepeatedSeed(
            f"refusing to pool: {len(repeats)} seed(s) appear more than once in the same "
            f"(operator, condition) cell -- showing {len(shown)} of {len(repeats)}:\n  "
            + "\n  ".join(shown)
            + "\nThese are two runs of the same arms, or one ledger read twice. Pooled, "
            "each seed would count twice, the spread would be understated, and one run's "
            "not_run margins would be averaged in beside the other's measured ones. Read "
            "one run at a time: pass each run's ledger to its own report."
        )
    return out


def _mean_sd(sample: list[float]) -> str:
    """Mean and spread, with a one-seed sample saying so instead of reporting sd 0."""
    if not sample:
        return "not measured"
    if len(sample) == 1:
        return f"{sample[0]:.2%} (n=1, no spread)"
    return f"{statistics.mean(sample):.2%} +/- {statistics.stdev(sample):.2%}"


def _delta(base: list[float], arm: list[float]) -> str:
    """The drop between two arms, or why it could not be taken.

    Never a number when either side is empty: an arm that measured nothing and an arm that
    measured zero are different facts, and subtracting from an absent mean would turn the
    first into the second.
    """
    if not base or not arm:
        return "not measured"
    return f"{(statistics.mean(base) - statistics.mean(arm)) * 100:+.2f}pp drop"


def render(cells: dict[tuple[str, str], Cell]) -> list[str]:
    """The three arms per operator, then what their differences do and do not establish."""
    out: list[str] = []
    operators = sorted({op for op, _ in cells})
    if not operators:
        return ["No row named a measured operator. Nothing in these ledgers is this report."]

    for op in operators:
        out.append("")
        out.append(f"======== {op} ========")
        present = [c for c in ORDER if (op, c) in cells]
        missing = [c for c in ORDER if (op, c) not in cells]
        for cond in present:
            cell = cells[(op, cond)]
            out.append(f"  {cond:<10} {cell.n_seeds} seed(s)")
            out.append(
                f"    on {op} ({cell.n_operator_rows} val row(s)): "
                f"{_mean_sd(cell.operator_acc)}"
            )
            out.append(
                f"    on its siblings ({cell.n_sibling_rows} val row(s), same class, "
                f"other generators): {_mean_sd(cell.sibling_acc)}"
            )
            out.append(f"    overall held-out: {_mean_sd(cell.overall_acc)}")
            if cell.margins:
                wins = sum(1 for m in cell.margins if m > 0)
                out.append(
                    f"    paired margin vs the control: {_mean_sd(cell.margins)} "
                    f"({wins} of {len(cell.margins)} seed(s) positive)"
                )
            if cell.margin_not_run:
                out.append(
                    f"    {cell.margin_not_run} seed(s) have NO margin: the model had no "
                    "opponent. Not measured -- never a loss. What the run recorded:"
                )
                for why, n in cell.margin_not_run_reasons.most_common():
                    shown = why if len(why) <= _REASON_CHARS else why[:_REASON_CHARS] + "..."
                    out.append(f"      [{n} seed(s)] {shown}")
                out.append(
                    "    If the arm's training set had no cached control, "
                    "tools/fit_operator_holdout_controls.sh fits one; the arm must then be "
                    "re-run, because a cached control does not backfill a written row."
                )
            if cell.unseeded:
                out.append(
                    f"    {cell.unseeded} row(s) carry no protocol.seed, so a seed counted "
                    "twice among them cannot be ruled out"
                )
            if cell.collapsed:
                out.append(
                    f"    {cell.collapsed} of {cell.n_seeds} seed(s) did not clear the "
                    "training-set majority: no fit to generalise, so their held-out "
                    "numbers are the prior"
                )
            if cell.unmeasured:
                out.append(
                    f"    {cell.unmeasured} seed(s) recorded no operator metric at all -- "
                    "counted, not dropped, and not read as zero"
                )
            if cell.siblings_not_run:
                out.append(
                    f"    {cell.siblings_not_run} seed(s) have no sibling measurement"
                    + (f": {cell.not_run_reason}" if cell.not_run_reason else "")
                )
        if missing:
            out.append(
                f"  MISSING ARM(S): {', '.join(missing)}. The comparison below is "
                "incomplete and the differences it can still take are printed as such."
            )
        codes = collections.Counter[str]()
        for cond in present:
            codes.update(cells[(op, cond)].code)
        if len(codes) > 1:
            where = "; ".join(
                f"{code} ({', '.join(c for c in present if code in cells[(op, c)].code)})"
                for code, _ in codes.most_common()
            )
            out.append(
                f"  ARMS RAN DIFFERENT CODE: {where}. The differences below cross a code "
                "change, so they are not a controlled comparison -- read each arm against "
                "arms from its own run."
            )

        ref = cells.get((op, REFERENCE))
        hold = cells.get((op, HOLDOUT))
        size = cells.get((op, SIZEMATCH))
        out.append("  --- what the differences say ---")
        if ref is None or hold is None:
            out.append(
                "    total effect: not measured -- it needs both the reference and the "
                "holdout arm, and this operator does not have both."
            )
            continue
        out.append(
            f"    total effect on {op}: {_delta(ref.operator_acc, hold.operator_acc)}"
        )
        out.append(
            f"    same rows' siblings:  {_delta(ref.sibling_acc, hold.sibling_acc)}"
        )
        if size is not None:
            out.append(
                f"    volume alone:         {_delta(ref.operator_acc, size.operator_acc)} "
                "(same number of training rows dropped at random)"
            )
        else:
            out.append(
                "    volume alone:         not measured -- without the size-matched arm "
                "the total effect cannot be separated from the smaller training set."
            )
        out += _verdict(op, ref, hold)
    return out


def _verdict(op: str, ref: Cell, hold: Cell) -> list[str]:
    """Which of the three readings the numbers support, or that they support none.

    States the reading, never a pass or a fail: the thresholds that would turn this into a
    gate are not this tool's to invent, and rule 2 makes them read-only besides.
    """
    if not ref.operator_acc or not hold.operator_acc:
        return ["    reading: not available -- one arm measured nothing."]
    if not ref.sibling_acc or not hold.sibling_acc:
        return [
            "    reading: NOT AVAILABLE. The operator's own drop is measured, but without "
            "sibling accuracy on both arms it cannot be separated from the class-prior "
            "shift that holding out a whole operator causes. The drop above is a fact "
            "about the arm, not yet evidence about generators."
        ]
    op_drop = statistics.mean(ref.operator_acc) - statistics.mean(hold.operator_acc)
    sib_drop = statistics.mean(ref.sibling_acc) - statistics.mean(hold.sibling_acc)
    return [
        f"    reading: {op} fell {op_drop * 100:+.2f}pp while its siblings -- same class, "
        f"same shifted prior, seen in training -- fell {sib_drop * 100:+.2f}pp.",
        "    " + _which_reading(op_drop, sib_drop),
    ]


def _which_reading(op_drop: float, sib_drop: float) -> str:
    """The three readings, named. The comparison is between the two drops, not against a bar."""
    if op_drop <= 0:
        return (
            "The operator did not fall. On this arm the model does not depend on having "
            "seen the generator, which is what reading the change rather than the "
            "generator would look like."
        )
    if sib_drop >= op_drop:
        return (
            "The siblings fell at least as far, so the arm moved the class prior and the "
            "operator's own drop is not evidence of generator recognition. What this arm "
            "measures is the cost of removing a class's dominant operator, not a shortcut."
        )
    return (
        f"The operator fell {(op_drop - sib_drop) * 100:.2f}pp further than its siblings "
        "under the identical prior, which is the part attributable to the generator's "
        "absence rather than to the prior or the volume."
    )


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    rows = read_rows([Path(p) for p in argv])
    try:
        cells = cells_of(rows)
    except RepeatedSeed as refused:
        print(refused, file=sys.stderr)
        return 2
    print(f"{len(rows)} ledger row(s) read from {len(argv)} file(s)")
    skipped = len(rows) - sum(c.n_seeds for c in cells.values())
    if skipped:
        print(f"{skipped} row(s) named no measured operator and are not this report")
    for line in render(cells):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
