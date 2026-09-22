"""``tools/ledger_arms.py``: the three things a log parser got wrong.

Each test here fails against a specific wrong implementation, and each of those wrong
implementations is one somebody wrote:

* grouping on ``recipe_hash`` alone merges a capacity sweep's arms into one population --
  the three 4096 arms share recipe ``aa8aeac9b4`` and differ only in ``backbone_commit``;
* reading the accuracy without the collapse gate reports a constant predictor's score as a
  result, which is how "capacity helps +2.59pp" was reported from the e10 arms on
  2026-09-21;
* using the one-sample floor on an arm-vs-arm difference claims 1.41x more sensitivity than
  the comparison has, which flipped a verdict the same day;
* counting every row of an arm as a new seed pools a re-run with the run it repeats -- the
  four-way ``after`` arm of 2026-09-22 was reported as n=13 over 8 seeds.

The tests on committed ledgers read the rows rather than a fixture, so the row shape this
module parses stays pinned to the row shape the runners actually write.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import ledger_arms  # noqa: E402

#: The sweep whose three arms share one recipe. Committed at 7f191bd, off the rented box.
CAPACITY_E30 = REPO / "ledger" / "gh200-rung0-capacity-4096-e30-2026-09-21.jsonl"
#: The four-way `after` arm twice: seeds 0-4 of a first run (5 killed), then all of 0-7 again.
FOURWAY = REPO / "ledger" / "gh200-fourway-2026-09-22.jsonl"
#: Seed 0 run twice at one code on purpose, to ask whether the two runs agree.
REPRO = REPO / "ledger" / "gh200-rung0-repro-2026-09-21.jsonl"
REPRODET = REPO / "ledger" / "gh200-rung0-reprodet-2026-09-21.jsonl"


def _row(
    *,
    backbone: str,
    val: float,
    fit: bool = True,
    recipe: str = "r" * 64,
    baseline: float = 0.484,
    metrics: bool = True,
    seed: int | None = None,
    code: str = "",
    row_id: str = "",
) -> dict:
    """One row carrying only the fields ``ledger_arms`` reads.

    ``seed=None`` leaves ``protocol.seed`` out, which every real row carries; the tests that
    are not about seeds use that, so no two of their rows can collide. ``code`` is the
    ``code_that_ran`` digest, spelled out rather than read from the tool so a name that
    drifted away from what the runner writes fails here.
    """
    protocol: dict = {"recipe_hash": recipe, "backbone_commit": backbone}
    if seed is not None:
        protocol["seed"] = seed
    row: dict = {"protocol": protocol, "metrics": {}}
    if row_id:
        row["row_id"] = row_id
    if code:
        row["metrics"]["code_that_ran"] = {"state": "ran", "passed": True, "value": code}
    if metrics:
        row["metrics"] |= {
            ledger_arms.VAL: {
                "state": "ran",
                "passed": val > baseline,
                "value": val,
                "detail": (
                    f"choice top-1 {val * 100:.1f}% of 288 held-out rows, against a "
                    f"majority-class baseline of {baseline * 100:.1f}%."
                ),
            },
            ledger_arms.TRAIN: {
                "state": "ran",
                "passed": fit,
                "value": 0.65 if fit else 0.40,
                "detail": "against a training-set majority share of 46.9%.",
            },
        }
    return row


def test_two_backbones_at_one_recipe_are_two_arms() -> None:
    """The grouping bug. A capacity sweep's arms differ in the model, not in the recipe.

    Against a version that keys on ``recipe_hash`` alone this returns one arm of four
    seeds whose spread is mostly the capacity difference it was supposed to measure.
    """
    rows = [
        _row(backbone="rung0-scratch:128x4:2layer:ctx4096", val=0.38),
        _row(backbone="rung0-scratch:128x4:2layer:ctx4096", val=0.39),
        _row(backbone="rung0-scratch:512x8:6layer:ctx4096", val=0.40),
        _row(backbone="rung0-scratch:512x8:6layer:ctx4096", val=0.41),
    ]
    arms = ledger_arms.arms_of(rows)
    assert len(arms) == 2, f"grouped into {len(arms)} arm(s); the backbone is half the key"
    assert {len(a.values) for a in arms.values()} == {2}


def test_one_backbone_at_two_recipes_is_also_two_arms() -> None:
    """The other half of the same key, which a learning curve depends on.

    Its points run one model over different training-set sizes, so they share a backbone
    and differ only in ``recipe_hash`` -- via ``train_subsample``, which ``0e897b3`` put
    into the recipe for exactly this reason.
    """
    rows = [
        _row(backbone="rung0-scratch:128x4:2layer:ctx8192", val=0.52, recipe="a" * 64),
        _row(backbone="rung0-scratch:128x4:2layer:ctx8192", val=0.54, recipe="b" * 64),
    ]
    assert len(ledger_arms.arms_of(rows)) == 2


def test_a_seed_that_did_not_clear_the_training_majority_is_not_in_the_fitted_sample() -> None:
    """Collapse comes from the gate the runner already recorded.

    A collapsed seed's held-out number is what a constant predictor scores, so including
    it in the sample being interpreted is how a majority-class predictor gets reported as
    a result.
    """
    rows = [
        _row(backbone="b", val=0.484, fit=False),
        _row(backbone="b", val=0.483, fit=False),
        _row(backbone="b", val=0.42, fit=True),
        _row(backbone="b", val=0.44, fit=True),
    ]
    arm = next(iter(ledger_arms.arms_of(rows).values()))
    assert arm.collapsed == 2
    assert len(arm.values) == 4, "every measured seed belongs to the all-seeds sample"
    assert sorted(arm.fitted) == [0.42, 0.44]


def test_a_collapsed_arms_tight_spread_is_reported_beside_the_fitted_one_not_instead() -> None:
    """Both numbers, never one standing in for the other.

    The e10 arms gave 1.11pp and 0.67pp against 4.41-5.58pp for the fitted e30 arms. That
    tightness is a constant predictor having almost nothing to vary; a report that showed
    only the all-seeds spread would hand the next sweep a prior three times too confident.
    """
    rows = [_row(backbone="b", val=v, fit=False) for v in (0.484, 0.485, 0.483, 0.484)]
    rows += [_row(backbone="b", val=v, fit=True) for v in (0.40, 0.46)]
    text = "\n".join(ledger_arms.render(ledger_arms.arms_of(rows)))
    assert "all seeds: n=6" in text
    assert "fitted seeds: n=2" in text
    assert "COLLAPSED on 4 of 6" in text


def test_a_row_missing_its_metrics_is_unmeasured_not_a_zero() -> None:
    """A seed that recorded nothing and a seed that scored nothing are different facts."""
    rows = [_row(backbone="b", val=0.5), _row(backbone="b", val=0.0, metrics=False)]
    arm = next(iter(ledger_arms.arms_of(rows).values()))
    assert arm.unmeasured == 1
    assert arm.values == [0.5], "an unmeasured row must not enter the sample as a number"
    assert arm.n_rows == 2, "and must not vanish from the row count either"


def test_an_arm_with_one_fitted_seed_reports_no_spread_rather_than_a_zero_one() -> None:
    """One run per arm has no resolution at all, which is not the same as perfect."""
    rows = [_row(backbone="b", val=0.45, fit=True)]
    rows += [_row(backbone="b", val=v, fit=False) for v in (0.484, 0.485)]
    text = "\n".join(ledger_arms.render(ledger_arms.arms_of(rows)))
    assert "fitted seeds: n=1 -- no spread to report" in text


def test_a_baseline_no_row_states_is_said_rather_than_silently_dropped() -> None:
    """A delta that cannot be computed is named; the arm is still reported."""
    row = _row(backbone="b", val=0.5)
    row["metrics"][ledger_arms.VAL]["detail"] = "no baseline in this sentence"
    text = "\n".join(ledger_arms.render(ledger_arms.arms_of([row, _row_no_base()])))
    assert "NOT STATED by any row" in text


def _row_no_base() -> dict:
    row = _row(backbone="b", val=0.51)
    row["metrics"][ledger_arms.VAL]["detail"] = "also no baseline here"
    return row


def test_a_pairwise_line_names_both_arms_distinguishably() -> None:
    """A learning curve's points share a backbone, so the backbone cannot be the label.

    Against the first version of this file the line reads
    ``rung0-scratch:128x4:2layer:ctx8192 minus rung0-scratch:128x4:2layer:ctx8192`` -- two
    different arms with the same name, on the report that compares them. Found by running
    the tool on the curve's own partial rows before its numbers mattered, which is the only
    reason it is fixed rather than shipped.
    """
    backbone = "rung0-scratch:128x4:2layer:ctx8192"
    quarter = [_row(backbone=backbone, val=v, recipe="1" * 64) for v in (0.49, 0.50, 0.48)]
    whole = [_row(backbone=backbone, val=v, recipe="2" * 64) for v in (0.52, 0.53, 0.51)]
    line = next(
        ln for ln in ledger_arms.render_pairwise(ledger_arms.arms_of(quarter + whole))
        if "minus" in ln
    )
    left, right = line.split("minus", 1)
    assert "1111111111" in right and "2222222222" in left, (
        f"the two sides of the comparison are not distinguishable:\n{line}"
    )


def test_the_arm_vs_arm_floor_is_the_wider_one() -> None:
    """The sqrt(2). Two noisy arms, not an arm against a fixed number.

    Against a version that passes ``against_known_reference=True`` for a difference
    between arms, the printed floor is 1.41x too small -- and 1.41x too small is always
    the direction that turns a null into a finding.
    """
    left = [_row(backbone="aaa", val=v) for v in (0.30, 0.34, 0.32, 0.36)]
    right = [_row(backbone="bbb", val=v) for v in (0.40, 0.44, 0.42, 0.46)]
    text = "\n".join(ledger_arms.render_pairwise(ledger_arms.arms_of(left + right)))
    # The known-sd column of both lines, so this compares like with like: the sqrt(2) is a
    # property of the comparison, and the t penalty is a separate factor that differs
    # between the two cases (df = n-1 against df = 2(n-1)) and would otherwise be folded in.
    pairwise = float(
        text.split("known-sd / estimated-sd ", 1)[1].split(" / ", 1)[0]
    )
    one_arm = "\n".join(ledger_arms.render(ledger_arms.arms_of(left)))
    fixed = float(
        one_arm.split("known-sd / estimated-sd: ", 1)[1].split(" / ", 1)[0]
    )
    # Both numbers are read back off a line rounded to two decimals, so the tolerance is
    # the printing precision and not the arithmetic's. It is still an order of magnitude
    # tighter than the gap being tested: against the one-sample floor this reads 3.62
    # against an expected 5.12, which no rounding accounts for.
    assert pairwise == pytest.approx(fixed * 2**0.5, abs=0.02)


def test_both_floors_are_printed_and_the_honest_one_is_the_wider() -> None:
    """The known-sd bound and the estimated-sd bound, as two numbers on the line.

    ``resolvable_difference`` assumes ``sd`` is known; it is estimated from these runs, so
    the honest quantile is Student's t. A reader given one number and a percentage has to
    pick the case themselves, and on 2026-09-21 an AUDIT file picked the wrong one -- it
    applied this module's two-sample n=5 figure of 14% to a one-sample n=8 comparison whose
    true penalty is 16.4%, erring on the flattering side.
    """
    rows = [_row(backbone="b", val=v) for v in (0.50, 0.52, 0.54, 0.51, 0.53)]
    line = next(
        ln for ln in ledger_arms.render(ledger_arms.arms_of(rows)) if "known-sd" in ln
    )
    fixed = line.split("known-sd / estimated-sd: ", 1)[1]
    known, honest = (float(x) for x in fixed.split("pp", 1)[0].split(" / "))
    assert honest > known, f"the estimated-sd bound must be the wider one:\n{line}"
    assert honest == pytest.approx(
        known * ledger_arms.estimated_sd_penalty(n_per_arm=5, against_known_reference=True),
        abs=0.01,
    ), "the penalty is not the one qd_train.power computes for this case"


def _arm_at(backbone: str, *, mean: float, sd: float, n: int = 5) -> list[dict]:
    """``n`` seeds with exactly this mean and sample sd, so a fixture can be aimed."""
    step = sd / statistics.stdev([i - (n - 1) / 2 for i in range(n)])
    return [_row(backbone=backbone, val=mean + step * (i - (n - 1) / 2)) for i in range(n)]


def test_a_difference_between_the_two_bounds_is_not_called_visible() -> None:
    """The band where a null gets read as a finding.

    A difference that clears the known-sd floor but not the honest one has not been shown
    to be visible, and calling it VISIBLE is the flattering read. It is the exact error the
    curve's audit made before the correction: +3.78pp reported as clearing 2.86pp, when the
    bound that applies is 3.33pp.

    The fixture is AIMED at the band rather than hoped into it. At n=5 per arm the floor is
    ``2.80159 * s * sqrt(2/5)`` and the honest one is 14.0% wider, so a pooled sd of 1.00pp
    puts the band at roughly [1.77, 2.02]pp and a 1.90pp difference lands inside it. If the
    arithmetic ever moves, this fails rather than skipping -- a test that skips is a test
    that did not run, which is the thing this repository refuses.
    """
    left = _arm_at("aaa", mean=0.3000, sd=0.0100)
    right = _arm_at("bbb", mean=0.3190, sd=0.0100)
    text = "\n".join(ledger_arms.render_pairwise(ledger_arms.arms_of(left + right)))

    known, honest = (
        float(x)
        for x in text.split("known-sd / estimated-sd ", 1)[1].split("pp", 1)[0].split(" / ")
    )
    diff = abs(statistics.mean([r["metrics"][ledger_arms.VAL]["value"] for r in right])
               - statistics.mean([r["metrics"][ledger_arms.VAL]["value"] for r in left])) * 100
    assert known < diff <= honest, (
        f"the fixture no longer lands in the band: {diff:.2f}pp against [{known}, {honest}]"
    )
    assert "clears the known-sd bound only" in text, text
    assert "VISIBLE" not in text.replace("clears the known-sd bound only", ""), text


def test_a_ledger_that_is_not_there_is_refused_rather_than_reported_over_the_rest() -> None:
    """Fail closed. A report over the files that happened to exist is a coverage claim
    nobody can reconstruct."""
    with pytest.raises(FileNotFoundError, match="coverage claim"):
        ledger_arms.read_rows([CAPACITY_E30, REPO / "ledger" / "does-not-exist.jsonl"])


def test_the_committed_capacity_ledger_reads_as_three_arms_of_eight() -> None:
    """Against the real rows, so the fixture above cannot drift from the written shape.

    These 24 rows are the 4096 e30 sweep: three capacities, 8 seeds each, no seed
    collapsed -- every one trained above its training-set majority, which is what makes
    all 24 held-out numbers statements about generalisation.
    """
    arms = ledger_arms.arms_of(ledger_arms.read_rows([CAPACITY_E30]))
    assert len(arms) == 3, f"expected three arms, got {sorted(k[1] for k in arms)}"
    assert {a.recipe_hash for a in arms.values()} == {
        next(iter(arms.values())).recipe_hash
    }, "the three arms share one recipe; they are separated by the backbone"
    for arm in arms.values():
        assert len(arm.values) == 8, f"{arm.backbone_commit} has {len(arm.values)} measured"
        assert arm.unmeasured == 0
        assert arm.collapsed == 0, f"{arm.backbone_commit} collapsed on {arm.collapsed}"
        assert arm.baseline == pytest.approx(0.484, abs=5e-4)


# -- one run per arm: a seed measured twice is two runs of one seed, not two seeds -------


def test_a_seed_measured_twice_in_one_arm_is_refused_naming_both_rows() -> None:
    """The defect. A re-run shares the recipe and the backbone of the run it repeats, so it
    lands in the same arm, and every row used to count as a new seed: n overstated, the
    spread understated, and both floors computed from the two -- the flattering direction.

    Against the version that pooled, this is an arm of n=4 where two seeds exist.
    """
    rows = [
        _row(backbone="b", val=0.50, seed=0, code="aaaa", row_id="first-0"),
        _row(backbone="b", val=0.52, seed=1, code="aaaa", row_id="first-1"),
        _row(backbone="b", val=0.49, seed=0, code="bbbb", row_id="rerun-0"),
        _row(backbone="b", val=0.53, seed=1, code="bbbb", row_id="rerun-1"),
    ]
    with pytest.raises(ledger_arms.RepeatedSeed) as refused:
        ledger_arms.arms_of(rows)
    msg = str(refused.value)
    assert "refusing to pool: 2 seed(s) appear more than once" in msg, msg
    assert "seed 0: row first-0 (code aaaa" in msg and "row rerun-0 (code bbbb" in msg, msg
    assert "--split-by-code" in msg, "a repeat across code versions has a named way to read it"


def test_the_same_seed_in_different_arms_is_not_a_repeat() -> None:
    """Every arm of a sweep runs seeds 0..n-1, so a seed repeats only inside ONE arm. A
    check keyed on the seed alone would refuse every sweep ever run."""
    rows = [
        _row(backbone="small", val=0.50, seed=0),
        _row(backbone="large", val=0.52, seed=0),
        _row(backbone="small", val=0.51, seed=0, recipe="q" * 64),
    ]
    assert len(ledger_arms.arms_of(rows)) == 3


def test_a_seed_claimed_three_times_is_one_repeated_seed_with_every_row_named() -> None:
    """The count in a refusal is of SEEDS. Counting each extra claim would say "2 seed(s)"
    for one seed run three times -- and a list that stopped at two rows would hide the third."""
    rows = [
        _row(backbone="b", val=v, seed=0, row_id=f"run{i}") for i, v in enumerate((0.5, 0.6, 0.7))
    ]
    with pytest.raises(ledger_arms.RepeatedSeed) as refused:
        ledger_arms.arms_of(rows)
    msg = str(refused.value)
    assert "1 seed(s) appear more than once" in msg, msg
    assert all(f"row run{i} " in msg for i in range(3)), msg


def test_split_by_code_reads_each_code_version_as_its_own_arm() -> None:
    """The named way out, keyed on the field that separates the committed re-runs: the
    ``code_that_ran`` digest. Not the rev -- see the four-way test below."""
    first = [_row(backbone="b", val=v, seed=s, code="aaaa") for s, v in enumerate((0.5, 0.52))]
    rerun = [_row(backbone="b", val=v, seed=s, code="bbbb") for s, v in enumerate((0.4, 0.42))]
    arms = ledger_arms.arms_of(first + rerun, split_by_code=True)
    assert sorted((a.code, len(a.values)) for a in arms.values()) == [("aaaa", 2), ("bbbb", 2)]
    text = "\n".join(ledger_arms.render(arms) + ledger_arms.render_pairwise(arms))
    assert "code aaaa" in text and "code bbbb" in text, "the two runs must be told apart by name"


@pytest.mark.parametrize("split_by_code", [False, True])
def test_a_repeat_nothing_in_the_row_separates_is_refused_even_split(split_by_code) -> None:
    """Two measurements of one seed at one code: a reproducibility check, which is what the
    two repro ledgers are. No field tells the runs apart, so no flag can make them seeds."""
    rows = [
        _row(backbone="b", val=0.503, seed=0, code="aaaa", row_id="x"),
        _row(backbone="b", val=0.531, seed=0, code="aaaa", row_id="y"),
    ]
    with pytest.raises(ledger_arms.RepeatedSeed) as refused:
        ledger_arms.arms_of(rows, split_by_code=split_by_code)
    assert "reproducibility check" in str(refused.value), str(refused.value)


def test_a_repeat_among_rows_that_measured_nothing_is_counted_rather_than_refused() -> None:
    """The chain ledgers repeat seeds on rows that measure nothing -- build, smoke and
    verdict rows, all at seed 0 -- and a killed run leaves one beside its retry. None of
    them enters a sample, so no floor moves and the tool still reads them; the report says
    the row count above is not a count of seeds."""
    rows = [
        _row(backbone="b", val=0.0, metrics=False, seed=0),
        _row(backbone="b", val=0.0, metrics=False, seed=0),
        _row(backbone="b", val=0.51, seed=0),
        _row(backbone="b", val=0.53, seed=1),
    ]
    arms = ledger_arms.arms_of(rows)
    (arm,) = arms.values()
    assert arm.values == [0.51, 0.53]
    assert "2 of these 4 rows repeat a seed" in "\n".join(ledger_arms.render(arms))


def test_an_arm_that_ran_two_code_versions_without_a_repeat_is_named() -> None:
    """Disjoint seeds across a code change pass the repeat check and are still not one run:
    the spread includes whatever the change did. Named, as the operator-holdout report names
    it -- not refused, because no seed counts twice and a wider spread flatters nothing."""
    rows = [
        _row(backbone="b", val=0.50, seed=0, code="aaaa"),
        _row(backbone="b", val=0.52, seed=1, code="aaaa"),
        _row(backbone="b", val=0.41, seed=2, code="bbbb"),
    ]
    text = "\n".join(ledger_arms.render(ledger_arms.arms_of(rows)))
    assert "ARM RAN DIFFERENT CODE: aaaa (2 measured), bbbb (1 measured)" in text, text


def test_measured_rows_without_a_seed_say_the_repeat_check_could_not_run() -> None:
    """A check that could not run must not read like one that ran and found nothing."""
    text = "\n".join(ledger_arms.render(ledger_arms.arms_of([_row(backbone="b", val=0.5)])))
    assert "1 measured row(s) carry no protocol.seed" in text, text


def test_the_committed_four_way_rerun_is_refused_and_reads_per_code_with_the_flag() -> None:
    """The ledger this was found on. It holds the four-way ``after`` arm twice -- a first run
    whose seeds 0-4 completed and seed 5 was killed, then all of 0-7 at later code -- and
    HANDOFF/context-source-2026-09-22.md reported the two pooled: n=13, top-1 50.20%.

    Both runs carry ONE launch rev, which is why the override splits by the code digest:
    grouping per rev would have pooled them exactly as before.
    """
    rows = ledger_arms.read_rows([FOURWAY])
    with pytest.raises(ledger_arms.RepeatedSeed, match=r"5 seed\(s\) appear more than once"):
        ledger_arms.arms_of(rows)
    after_rows = [r for r in rows if r["protocol"]["recipe_hash"].startswith("1e819e2cad")]
    assert len({r["recipe"]["rev"] for r in after_rows}) == 1

    arms = ledger_arms.arms_of(rows, split_by_code=True)
    after = {a.code: a for a in arms.values() if a.recipe_hash.startswith("1e819e2cad")}
    assert sorted((code, len(a.values), a.unmeasured) for code, a in after.items()) == [
        ("065920c4f8635051", 5, 1),
        ("d10f23542617e9f5", 8, 0),
    ]
    # The complete run alone: 50.10%, where pooling both runs printed 50.20%.
    assert statistics.mean(after["d10f23542617e9f5"].values) == pytest.approx(0.50098, abs=1e-5)


@pytest.mark.parametrize("ledger", [REPRO, REPRODET], ids=["repro", "reprodet"])
def test_the_committed_repro_pairs_are_refused_even_split(ledger) -> None:
    """Seed 0 twice at one code, on purpose: the question was whether the two runs agree
    (reprodet bit-identical, repro 2.8pp apart). Summarised as an arm of two seeds, their
    run-to-run difference would be printed as a seed spread, with floors computed from it."""
    with pytest.raises(ledger_arms.RepeatedSeed, match="reproducibility check"):
        ledger_arms.arms_of(ledger_arms.read_rows([ledger]), split_by_code=True)


def test_main_refuses_with_exit_2_and_names_the_way_to_read_it(capsys) -> None:
    assert ledger_arms.main([str(FOURWAY)]) == 2
    captured = capsys.readouterr()
    assert "refusing to pool" in captured.err and "--split-by-code" in captured.err
    assert captured.out == "", "a refused report prints no arm, not the arms before the refusal"
    assert ledger_arms.main([str(FOURWAY), "--split-by-code"]) == 0
    assert "code d10f23542617e9f5" in capsys.readouterr().out


def test_one_ledger_named_twice_is_refused(capsys) -> None:
    """The likeliest way to double every arm at once."""
    assert ledger_arms.main([str(CAPACITY_E30)]) == 0
    capsys.readouterr()
    assert ledger_arms.main([str(CAPACITY_E30), str(CAPACITY_E30)]) == 2
    assert "refusing to pool" in capsys.readouterr().err


#: Every ledger a HANDOFF or AUDIT file reads with this tool, plus the sweeps its docstring
#: and its floors were checked on. Each ran one code per arm and repeats no seed.
DOCUMENTED = [
    "gh200-rung0-capacity-4096-e10-2026-09-21.jsonl",
    "gh200-rung0-capacity-4096-e30-2026-09-21.jsonl",
    "gh200-rung0-capacity-sw005-nondet-2026-09-21.jsonl",
    "gh200-rung0-learning-curve-2026-09-21.jsonl",
    "gh200-rung0-curve-n24-2026-09-22.jsonl",
    "gh200-span-in-diff-2026-09-22.jsonl",
]


@pytest.mark.parametrize("name", DOCUMENTED)
def test_a_documented_ledger_reads_the_same_split_or_not(name) -> None:
    """The refusal must not break a command a handoff tells the next lane to run, and on a
    ledger with one code per arm the override must change nothing it prints about an arm."""
    rows = ledger_arms.read_rows([REPO / "ledger" / name])
    pooled = ledger_arms.arms_of(rows)
    split = ledger_arms.arms_of(rows, split_by_code=True)
    assert sorted((k[0], k[1], a.values) for k, a in pooled.items()) == sorted(
        (k[0], k[1], a.values) for k, a in split.items()
    )


# -- render_gates: a gate nobody evaluated and a gate that passed must not look alike ----


def _gate_row(gates: dict, controls: dict) -> dict:
    """A row carrying only the blocks `render_gates` reads."""
    return {
        "protocol": {"recipe_hash": "r" * 64, "backbone_commit": "b"},
        "metrics": {},
        "gates": gates,
        "controls": controls,
    }


def test_a_gate_that_never_ran_is_counted_not_omitted():
    """The 988 rows before 2026-09-21 carried nine of these, every one `not_run`.

    A summary that printed only the gates that ran would have shown an empty section and
    said nothing was wrong, which is how the condition survived for months.
    """
    rows = [_gate_row({"ece": {"state": "not_run", "reason": "never evaluated"}}, {})]
    out = "\n".join(ledger_arms.render_gates(rows))
    assert "ece" in out
    assert "not_run 1" in out


def test_pass_and_fail_and_not_run_are_three_distinct_counts():
    rows = [
        _gate_row({"ece": {"state": "ran", "passed": True, "value": 0.01}}, {}),
        _gate_row({"ece": {"state": "ran", "passed": False, "value": 0.19}}, {}),
        _gate_row({"ece": {"state": "not_run", "reason": "never evaluated"}}, {}),
    ]
    out = "\n".join(ledger_arms.render_gates(rows))
    assert "FAIL 1" in out
    assert "pass 1" in out
    assert "not_run 1" in out


def test_a_failing_gate_is_not_softened_into_a_pass():
    """`passed: False` on a row that ran is a FAIL, never folded in with not_run.

    paired_margin_vs_linear went negative on the first real row of 2026-09-21. A summary
    that reported it as anything other than a failure would be reporting a win.
    """
    rows = [
        _gate_row(
            {"paired_margin_vs_linear": {
                "state": "ran", "passed": False, "value": -0.0243,
                "detail": "paired margin -0.0243, CI includes zero, so this is not a win",
            }},
            {},
        )
    ]
    out = "\n".join(ledger_arms.render_gates(rows))
    assert "FAIL 1" in out
    assert "pass" not in out.split("paired_margin_vs_linear")[1].split("\n")[0]
    assert "not a win" in out


def test_values_are_averaged_only_over_the_rows_that_ran():
    """A not_run row contributes no value; counting it as zero would move the mean."""
    rows = [
        _gate_row({"g": {"state": "ran", "passed": True, "value": 1.0}}, {}),
        _gate_row({"g": {"state": "ran", "passed": True, "value": 3.0}}, {}),
        _gate_row({"g": {"state": "not_run", "reason": "no"}}, {}),
    ]
    out = "\n".join(ledger_arms.render_gates(rows))
    assert "value mean +2.0000" in out


def test_controls_are_reported_beside_gates_not_instead_of_them():
    rows = [
        _gate_row(
            {"ece": {"state": "ran", "passed": False, "value": 0.19}},
            {"degenerate_head": {"state": "ran", "passed": True, "value": 0.89}},
        )
    ]
    out = "\n".join(ledger_arms.render_gates(rows))
    assert "gates:" in out
    assert "controls:" in out
    assert "degenerate_head" in out
    assert "ece" in out


def test_the_committed_rows_still_report_every_gate_as_never_run():
    """Pins the finding to the rows themselves: the capacity sweep evaluated nothing.

    If a later change made these rows carry a verdict, this test fails and the claim in the
    handoff has to be rewritten rather than quietly becoming false.
    """
    path = (
        REPO / "ledger" / "gh200-rung0-capacity-4096-e30-2026-09-21.jsonl"
    )
    rows = ledger_arms.read_rows([path])
    assert len(rows) == 24
    out = "\n".join(ledger_arms.render_gates(rows))
    for name in (
        "ece", "needle_hunk_recall", "ood_abstain", "paired_margin_vs_linear",
        "permutation_consistency", "degenerate_head", "privileged_hunk",
        "shuffled_label", "transfer_gate",
    ):
        assert f"{name:28} not_run 24" in out, name
