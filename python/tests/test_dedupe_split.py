"""Dedupe and the repo-level split: ``docs/hardening.md`` section 2 end to end.

*"Leakage is the failure that makes every other number meaningless."* So the tests
here are built around corpora that are **designed to leak**, not around random text:
a vendored copy of one file in a second repository, a reformatted copy of one
function, and a family holdout that a repo hash could otherwise undo. A dedupe test
over unrelated strings proves only that unrelated strings are not duplicates.
"""

from __future__ import annotations

import pytest
from data_fixtures import (
    clinc_row,
    code_body,
    commitpackft_row,
    small_corpus,
    squad_row,
    vendored_pair,
)

from qd_data.config import DataConfig
from qd_data.dedupe import content_unit_key, dedupe
from qd_data.mixture import build_mixture
from qd_data.rows import DataRow, GoldAnswer
from qd_data.schema import ChoiceSlot, Request
from qd_data.split import HELD_OUT, TRAINING_SPLITS, assign_repo, split
from qd_train.tristate import NotRun, Ran


def _row(
    row_id: str,
    *,
    repo: str,
    text: str,
    family: str = "code.commit_intent",
    path: str = "a.py",
) -> DataRow:
    return DataRow(
        row_id=row_id,
        source_id="bigcode/commitpackft",
        host="huggingface",
        family_id=family,
        repo_key=repo,
        identity_key=f"{repo}::{path}",
        licence_id="mit",
        obligations=("attribution",),
        request=Request(
            task=family,
            context=text.encode("utf-8"),
            question="q?",
            slots=(ChoiceSlot(name="implements_claim", options=("yes", "no")),),
            example_id=row_id,
        ),
        gold=(GoldAnswer(slot_name="implements_claim", value="yes"),),
        dedupe_text=text,
    )


def _pipeline(corpus: dict[str, list[object]], config: DataConfig | None = None):
    config = config or DataConfig()
    mixture = build_mixture(corpus, config=config)
    report = dedupe(list(mixture.rows), config=config)
    return mixture, report, split(report, config=config)


# -- dedupe operates on content units, not on examples -----------------------


def test_the_three_families_of_one_source_row_are_one_content_unit() -> None:
    """The bug this catches deleted two thirds of the mixture while reporting clean.

    One commitpackft row becomes a commit-intent, a language-id and a change-scope
    example from the same file. Comparing *examples* makes those exact duplicates.
    """
    config = DataConfig()
    mixture = build_mixture(
        {"bigcode/commitpackft": [commitpackft_row(i) for i in range(6)]}, config=config
    )
    assert len(mixture.rows) == 18, "6 rows x 3 families"
    assert len({content_unit_key(r) for r in mixture.rows}) == 6

    report = dedupe(list(mixture.rows), config=config)
    assert report.n_input_rows == 18
    assert report.n_input_units == 6
    assert len(report.kept) == 18, "no family may be deduped away against its siblings"


def test_many_squad_questions_on_one_passage_all_survive() -> None:
    """Several SQuAD questions share a passage and sit near 0.85 Jaccard by
    construction. Comparing passages alone would keep one question in six."""
    config = DataConfig()
    rows = [squad_row(i, force_impossible=False) for i in range(6)]
    same_passage = [
        type(rows[0])(
            qid=f"shared{i}",
            title="Article0",
            context=rows[0].context,
            question=f"Which distinct question number {i} is asked here?",
            answers=rows[0].answers,
            answer_starts=rows[0].answer_starts,
            is_impossible=False,
        )
        for i in range(6)
    ]
    mixture = build_mixture({"rajpurkar/squad_v2": same_passage}, config=config)
    report = dedupe(list(mixture.rows), config=config)
    assert len(report.kept) == len(mixture.rows)


# -- the vendored-copy corpus ------------------------------------------------


def test_a_vendored_copy_in_a_second_repo_is_removed_before_splitting() -> None:
    """``docs/hardening.md``: *vendored trees, forks and copied files put the same
    code in two repos*. Repo-level splitting alone cannot separate them."""
    config = DataConfig()
    a, b = vendored_pair()
    mixture = build_mixture(
        {"bigcode/commitpackft": [a, b, commitpackft_row(50), commitpackft_row(51)]},
        config=config,
    )
    report = dedupe(list(mixture.rows), config=config)

    assert report.n_cross_repo_pairs >= 1, "the vendored pair must be confirmed"
    assert report.n_dropped_units == 1
    assert report.n_dropped_rows == 3, "one unit, three families"
    surviving_repos = {r.repo_key for r in report.kept if "parser.py" in r.identity_key}
    assert len(surviving_repos) == 1, f"both copies survived: {surviving_repos}"
    assert isinstance(report.status, Ran) and report.status.passed


def test_within_repo_near_duplicates_are_counted_and_kept() -> None:
    """Same repo means same side of the split, so dropping them removes data without
    removing a leak -- and for SQuAD it would remove almost everything."""
    config = DataConfig()
    shared = code_body("dup", lines=30)
    rows = [
        commitpackft_row(1, repo="org/one", body=shared, path="x.py"),
        commitpackft_row(2, repo="org/one", body=shared, path="y.py"),
    ]
    mixture = build_mixture({"bigcode/commitpackft": rows}, config=config)
    report = dedupe(list(mixture.rows), config=config)
    assert report.n_within_repo_pairs >= 1
    assert report.n_dropped_rows == 0
    assert "within one repo and were kept" in report.status.detail  # type: ignore[union-attr]


def test_dedupe_of_an_empty_corpus_is_not_run_not_a_clean_pass() -> None:
    report = dedupe([], config=DataConfig())
    assert isinstance(report.status, NotRun)
    assert "zero rows" in report.status.reason


def test_duplicate_row_ids_are_refused() -> None:
    row = _row("dup", repo="org/a", text=code_body("a"))
    with pytest.raises(ValueError, match="duplicate row_id"):
        dedupe([row, row], config=DataConfig())


def test_a_truncated_candidate_search_is_not_run_not_a_clean_dedupe() -> None:
    """*"No near-duplicates" and "we stopped looking" must not read the same.*"""
    text = code_body("identical", lines=40)
    rows = [
        _row(f"r{i:03d}", repo=f"org/repo{i}", text=text, path=f"f{i}.py") for i in range(30)
    ]
    report = dedupe(rows, config=DataConfig(), max_candidate_pairs=5)
    assert isinstance(report.status, NotRun)
    assert "hit its bound" in report.status.reason


def test_dedupe_keeps_the_lexicographically_smallest_unit_whatever_the_input_order() -> None:
    text = code_body("same", lines=40)
    rows = [
        _row("z", repo="org/zzz", text=text, path="p.py"),
        _row("a", repo="org/aaa", text=text, path="p.py"),
        _row("m", repo="org/mmm", text=text, path="p.py"),
    ]
    first = dedupe(rows, config=DataConfig())
    second = dedupe(list(reversed(rows)), config=DataConfig())
    assert {r.row_id for r in first.kept} == {r.row_id for r in second.kept}
    assert len(first.kept) == 1


# -- the split ---------------------------------------------------------------


def test_repo_assignment_is_a_pure_function_of_repo_and_seed() -> None:
    for repo in ("org/a", "org/b", "some/other"):
        first = assign_repo(repo, seed=1, train_fraction=0.9, val_fraction=0.05)
        second = assign_repo(repo, seed=1, train_fraction=0.9, val_fraction=0.05)
        assert first == second
        assert first in {"train", "val", HELD_OUT}


def test_repo_assignment_does_not_move_when_the_corpus_grows() -> None:
    """A shuffle-and-slice would reshuffle every repo when one is added, silently
    invalidating every earlier comparison."""
    before = {
        r: assign_repo(r, seed=9, train_fraction=0.9, val_fraction=0.05)
        for r in (f"org/repo{i}" for i in range(20))
    }
    after = {
        r: assign_repo(r, seed=9, train_fraction=0.9, val_fraction=0.05)
        for r in (f"org/repo{i}" for i in range(40))
    }
    assert all(after[r] == v for r, v in before.items())


def test_empty_repo_key_is_refused_not_defaulted() -> None:
    with pytest.raises(ValueError, match="empty repo_key"):
        assign_repo("   ", seed=1, train_fraction=0.9, val_fraction=0.05)


def test_no_repo_appears_in_two_repo_splits() -> None:
    _, _, report = _pipeline(small_corpus(24))
    assert isinstance(report.repo_disjoint, Ran) and report.repo_disjoint.passed
    where: dict[str, set[str]] = {}
    for a in report.assignments:
        where.setdefault(a.repo_key, set()).add(a.repo_split)
    assert all(len(v) == 1 for v in where.values())


def test_no_function_identity_appears_in_two_repo_splits() -> None:
    """``docs/hardening.md`` section 1: enforced by function-level identity, not by
    diff hash -- the same function reformatted has a different diff hash."""
    _, _, report = _pipeline(small_corpus(24))
    assert isinstance(report.identity_disjoint, Ran) and report.identity_disjoint.passed


def test_held_out_families_appear_in_no_training_shard() -> None:
    config = DataConfig()
    _, _, report = _pipeline(small_corpus(24), config)
    training = [a for a in report.assignments if a.split in TRAINING_SPLITS]
    assert training, "the fixture must produce training rows or the test is vacuous"
    assert not [a for a in training if config.is_held_out_family(a.family_id)]
    check = report.held_out_families_absent_from_training
    assert isinstance(check, Ran) and check.passed


def test_a_held_out_family_cannot_be_rescued_by_a_train_repo_hash() -> None:
    """The family holdout and the repo holdout are different mechanisms. A row whose
    repo hashes to ``train`` still goes to the held-out shard if its family is out."""
    config = DataConfig()
    _, _, report = _pipeline(small_corpus(24), config)
    overridden = [
        a
        for a in report.assignments
        if a.reason == "family_holdout" and a.repo_split in TRAINING_SPLITS
    ]
    assert overridden, "the fixture must contain a held-out family in a training repo"
    assert all(a.split == HELD_OUT for a in overridden)


def test_the_holdout_shard_reports_its_two_populations_apart() -> None:
    _, _, report = _pipeline(small_corpus(24))
    breakdown = report.holdout_breakdown()
    assert breakdown["family_holdout"] > 0
    assert sum(breakdown.values()) == report.counts()[HELD_OUT]


def test_no_near_duplicate_pair_spans_a_repo_split() -> None:
    _, _, report = _pipeline(small_corpus(24))
    check = report.near_duplicate_disjoint
    assert isinstance(check, Ran) and check.passed
    assert check.n == check.n_total, "coverage must be complete, not sampled"


def test_a_not_run_dedupe_makes_the_split_not_run() -> None:
    """``aggregate`` refuses to be more confident than its least-informed input."""
    config = DataConfig()
    text = code_body("same", lines=40)
    rows = [
        _row(f"r{i:03d}", repo=f"org/repo{i}", text=text, path=f"f{i}.py") for i in range(30)
    ]
    report = dedupe(rows, config=config, max_candidate_pairs=5)
    assert isinstance(report.status, NotRun)
    split_report = split(report, config=config)
    assert isinstance(split_report.status, NotRun)
    assert "dedupe" in split_report.status.reason


def test_a_single_row_corpus_reports_the_leak_check_as_not_run() -> None:
    """With one row there is no pair to cross a boundary; that is not evidence."""
    config = DataConfig()
    report = dedupe([_row("only", repo="org/a", text=code_body("a"))], config=config)
    split_report = split(report, config=config)
    assert isinstance(split_report.near_duplicate_disjoint, NotRun)
    assert isinstance(split_report.status, NotRun)


def test_a_cross_split_near_duplicate_is_caught_if_dedupe_is_bypassed() -> None:
    """The split's own leak check is an independent second derivation.

    Constructed by handing the splitter a report whose ``kept`` still contains a
    cross-repo duplicate -- i.e. simulating a dedupe that missed one. The check must
    fail, not pass, or a bug in dedupe would report itself as a clean split.
    """
    config = DataConfig()
    text = code_body("leaky", lines=40)
    # Find two repo keys that land in different repo splits.
    pairs = [
        (f"org/leak{i}", assign_repo(f"org/leak{i}", seed=config.seed,
                                     train_fraction=config.train_fraction,
                                     val_fraction=config.val_fraction))
        for i in range(400)
    ]
    train_repo = next(r for r, s in pairs if s == "train")
    other_repo = next(r for r, s in pairs if s != "train")

    rows = [
        _row("a", repo=train_repo, text=text, path="p.py"),
        _row("b", repo=other_repo, text=text, path="p.py"),
    ]
    clean = dedupe(rows, config=config)
    assert clean.n_dropped_rows == 1, "dedupe itself must catch this"

    # Now bypass dedupe's removal and hand the splitter both rows.
    bypassed = type(clean)(
        kept=tuple(rows), clusters=(), n_input_rows=2, n_input_units=2,
        n_dropped_rows=0, n_dropped_units=0, n_cross_repo_pairs=0,
        n_within_repo_pairs=0, n_truncated_for_shingling=0,
        band_config=clean.band_config, threshold=clean.threshold,
        n_candidate_pairs=0, status=Ran(passed=True, n=2, n_total=2),
    )
    report = split(bypassed, config=config)
    check = report.near_duplicate_disjoint
    assert isinstance(check, Ran) and not check.passed
    assert "span a repo split boundary" in check.detail
    assert isinstance(report.status, Ran) and not report.status.passed


def test_clinc_utterances_of_one_intent_never_straddle_a_split() -> None:
    """CLINC has no repository, so the intent label is the split unit -- otherwise
    ~100 near-paraphrases per intent would sit on both sides."""
    config = DataConfig()
    _, _, report = _pipeline(
        {"clinc/clinc_oos": [clinc_row(i) for i in range(40)]}, config
    )
    where: dict[str, set[str]] = {}
    for a in report.assignments:
        if a.repo_key.startswith("clinc-intent:"):
            where.setdefault(a.repo_key, set()).add(a.repo_split)
    assert where, "fixture produced no clinc rows"
    assert all(len(v) == 1 for v in where.values())


# -- the recall bound reaches the report: GAP-DATA-LSH-RECALL-BOUND ----------


def test_the_dedupe_report_states_its_recall_bound_rather_than_implying_completeness(
) -> None:
    """A clean dedupe must not read as "there are no near-duplicates".

    The report carried ``bands`` and ``rows_per_band`` -- the *cause* of the recall
    loss -- and then a ``Ran`` whose coverage pair was ``n units of n units``, which
    is complete coverage in the only dimension a machine reads. Every unit was indeed
    signed and banded; what was not exhaustive is the pair search, and that is the
    dimension the claim is about. Both numbers now live in the pair dimension, and
    the bound itself is in the JSON that reaches the manifest.
    """
    config = DataConfig()
    mixture = build_mixture(small_corpus(24), config=config)
    report = dedupe(list(mixture.rows), config=config)

    blob = report.to_json()
    assert blob["lsh_per_pair_recall_at_threshold"] == pytest.approx(0.947049, abs=1e-6)
    assert blob["lsh_per_pair_recall_at_threshold"] < 1.0
    assert blob["n_possible_unit_pairs"] == report.n_input_units * (
        report.n_input_units - 1
    ) // 2
    assert blob["n_candidate_pairs"] < blob["n_possible_unit_pairs"], (
        "the fixture must exercise a filtered pair search or the test is vacuous"
    )

    status = report.status
    assert isinstance(status, Ran) and status.passed
    assert not status.is_complete_coverage, (
        "a probabilistic pair search must never report complete coverage"
    )
    assert (status.n, status.n_total) == (
        report.n_candidate_pairs,
        report.n_possible_unit_pairs,
    )
    assert "probabilistic candidate generator" in status.detail
    assert "lower bound" in status.detail
    assert f"{report.lsh_recall_at_threshold:.4f}" in status.detail


def test_the_reported_bound_is_the_one_the_banding_actually_used() -> None:
    """Derived from ``band_config``, so it cannot drift from the configuration that
    produced the candidate list -- a stored copy could."""
    config = DataConfig()
    mixture = build_mixture(small_corpus(12), config=config)
    report = dedupe(list(mixture.rows), config=config)
    assert report.lsh_recall_at_threshold == report.band_config.recall_at_threshold
    assert report.band_config.threshold == config.dedupe_threshold


def test_a_truncated_pair_search_is_still_not_run_not_a_thinner_bound() -> None:
    """The bound describes a *completed* probabilistic search. A search that hit its
    bound did not complete, and those are different facts: one is ``Ran`` with a
    stated recall, the other is ``NotRun``."""
    text = code_body("identical", lines=40)
    rows = [
        _row(f"r{i:03d}", repo=f"org/repo{i}", text=text, path=f"f{i}.py") for i in range(30)
    ]
    report = dedupe(rows, config=DataConfig(), max_candidate_pairs=5)
    assert isinstance(report.status, NotRun)
    assert "hit its bound" in report.status.reason
    # The bound is still reportable -- it is a property of the banding, not of the
    # run -- but it must not be mistaken for a completed measurement.
    assert report.to_json()["lsh_per_pair_recall_at_threshold"] < 1.0
    assert report.to_json()["status"]["state"] == "not_run"
