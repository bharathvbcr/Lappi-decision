"""SQuAD is partitioned by article title between ``qa.answerability`` and ``qa.answer_span``.

GAP-DATA-SQUAD-SPAN-NOUL-LEAKS-HELD-OUT-ANSWERABILITY: both families were built from the same
SQuAD v2 rows with a byte-identical context, and a span gold is ``noul`` exactly when the
question is unanswerable -- so training span on a context taught the held-out family's answer
for it. User decision 2026-09-29, option (a): each title feeds exactly one family, decided
by a keyed hash of the title. Rule 2: this re-partitions a held-out set, which is permitted
here only because a human chose it; the fraction is pinned below so moving it is visible.
"""

from __future__ import annotations

import dataclasses

import pytest
from data_fixtures import commitpackft_row, squad_title_for

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.loaders import SquadRow
from qd_data.mixture import RowRefused, build_mixture, rewrite_squad
from qd_data.split import (
    CONTENT_DISJOINT_FAMILIES,
    SQUAD_ANSWERABILITY_TITLE_FRACTION,
    SQUAD_TITLE_FAMILIES,
    content_disjoint_families,
    split,
    squad_title_family,
)
from qd_train.tristate import NotRun, Ran

ANSWERABILITY, SPAN = SQUAD_TITLE_FAMILIES


def _squad_rows(n_titles: int = 60, per_title: int = 3) -> list[SquadRow]:
    """Several questions per article, as in SQuAD itself, over many articles."""
    out = []
    for t in range(n_titles):
        for q in range(per_title):
            passage = (
                f"Article {t} paragraph {q} opens here.\n"
                f"Its second line names Item{t}x{q} as the subject.\n"
                "A third line is filler."
            )
            impossible = (t + q) % 4 == 0
            needle = f"Item{t}x{q}"
            out.append(
                SquadRow(
                    qid=f"t{t:03d}q{q}", title=f"Title{t}", context=passage,
                    question=f"What does paragraph {q} of article {t} name?",
                    answers=() if impossible else (needle,),
                    answer_starts=() if impossible else (passage.index(needle),),
                    is_impossible=impossible,
                )
            )
    return out


def test_the_partition_parameters_are_pinned() -> None:
    """0.2: lead decision under user delegation, 2026-09-29. Moving it re-partitions the
    held-out answerability set, so it must show up as a changed test, not a quiet edit."""
    assert SQUAD_ANSWERABILITY_TITLE_FRACTION == 0.2
    assert SQUAD_TITLE_FAMILIES == ("qa.answerability", "qa.answer_span")
    assert frozenset(SQUAD_TITLE_FAMILIES) in CONTENT_DISJOINT_FAMILIES


def test_the_title_decides_the_family_and_is_stable() -> None:
    seed = DataConfig().seed
    assert squad_title_family("Normans", seed=seed) == squad_title_family("Normans", seed=seed)
    share = sum(
        squad_title_family(f"T{i}", seed=seed) == ANSWERABILITY for i in range(4000)
    ) / 4000
    assert share == pytest.approx(SQUAD_ANSWERABILITY_TITLE_FRACTION, abs=0.02)
    with pytest.raises(ValueError, match="empty SQuAD title"):
        squad_title_family("  ", seed=seed)


def test_no_title_context_or_identity_feeds_both_families() -> None:
    config = DataConfig()
    raws = _squad_rows()
    mixture = build_mixture({"rajpurkar/squad_v2": raws}, config=config)
    assert isinstance(mixture.status, Ran) and mixture.status.passed, mixture.status
    by_family: dict[str, list] = {f: [] for f in SQUAD_TITLE_FAMILIES}
    for r in mixture.rows:
        by_family[r.family_id].append(r)
    assert by_family[ANSWERABILITY] and by_family[SPAN], "both families must be built"
    for key in (
        lambda r: r.repo_key,
        lambda r: r.identity_key,
        lambda r: r.request.context,
    ):
        assert not {key(r) for r in by_family[ANSWERABILITY]} & {
            key(r) for r in by_family[SPAN]
        }
    # Every raw question lands in exactly one family: none dropped, none duplicated.
    assert len(mixture.rows) == len(raws)
    # Routed, not refused: the other family's rows are not data-quality losses.
    assert "squad_title_reserved_for_other_family" not in mixture.refusals.get(
        "rajpurkar/squad_v2", {}
    )
    for fam in SQUAD_TITLE_FAMILIES:
        cov = mixture.family_coverage[fam]
        assert isinstance(cov, Ran)
        assert cov.n == cov.n_total == len(by_family[fam])
        assert "routed to another family by design" in cov.detail

    report = split(dedupe(list(mixture.rows), config=config), config=config)
    assert isinstance(report.content_disjoint_families, Ran)
    assert report.content_disjoint_families.passed


def test_a_title_in_both_families_fails_the_split_check_and_its_status() -> None:
    """The split-level check is the independent second signal: rows that did not come
    through the mixture's routing are caught by what they share."""
    config = DataConfig()
    title = squad_title_for(SPAN, stem="Leak")
    raw = SquadRow(
        qid="leak0", title=title, context="One.\nTwo names Leak here.\nThree.",
        question="Which line names it?", answers=("Leak",),
        answer_starts=("One.\nTwo names Leak here.\nThree.".index("Leak"),),
        is_impossible=False,
    )
    span_row = rewrite_squad(raw, family_id=SPAN, index=0, config=config)
    leaked = dataclasses.replace(
        span_row, family_id=ANSWERABILITY, row_id="squad:qa.answerability:leak0"
    )
    state = content_disjoint_families([span_row, leaked])
    assert isinstance(state, Ran) and not state.passed
    for label in ("repo_key", "identity_key", "context"):
        assert label in state.detail

    report = split(dedupe([span_row, leaked], config=config), config=config)
    assert isinstance(report.content_disjoint_families, Ran)
    assert not report.content_disjoint_families.passed
    assert not isinstance(report.status, Ran) or not report.status.passed


def test_a_corpus_without_squad_leaves_the_check_not_run_and_the_status_untouched() -> None:
    config = DataConfig()
    mixture = build_mixture(
        {"bigcode/commitpackft": [commitpackft_row(i) for i in range(12)]}, config=config
    )
    report = split(dedupe(list(mixture.rows), config=config), config=config)
    assert isinstance(report.content_disjoint_families, NotRun)
    assert "content_disjoint_families" not in str(report.status.to_json())


def test_rewrite_squad_refuses_a_title_owned_by_the_other_family() -> None:
    """Fail closed for a caller that reaches the rewriter without the mixture's routing."""
    config = DataConfig()
    title = squad_title_for(ANSWERABILITY, stem="Owned")
    raw = SquadRow(
        qid="own0", title=title, context="A line.\nAnother line.", question="Is it here?",
        answers=(), answer_starts=(), is_impossible=True,
    )
    with pytest.raises(RowRefused) as exc:
        rewrite_squad(raw, family_id=SPAN, index=0, config=config)
    assert exc.value.reason_code == "squad_title_reserved_for_other_family"
    assert rewrite_squad(raw, family_id=ANSWERABILITY, index=0, config=config).family_id == (
        ANSWERABILITY
    )
