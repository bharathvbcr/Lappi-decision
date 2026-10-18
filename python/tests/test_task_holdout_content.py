"""The task holdout holds out tasks, and now content too (M12).

GAP-PORTED-HELDOUT-FAMILY-CONTENT-OVERLAPS-TRAIN: a commitpackft row feeds trained families
and the held-out ``code.language_id`` at once, and replay drawn from training shards hit
held-out content on 222 of 234 rows. ``task_holdout_content_disjoint`` reports it by exact key.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_fixtures import clinc_row, commitpackft_row, small_corpus, squad_row

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.mixture import build_mixture
from qd_data.split import split, task_holdout_overlap
from qd_train.tristate import NotRun, Ran


def _split(corpus: dict[str, list[object]], config: DataConfig):
    mixture = build_mixture(corpus, config=config)
    return split(dedupe(list(mixture.rows), config=config), config=config), mixture


def test_commitpackft_rows_feeding_a_trained_and_the_held_out_family_are_reported() -> None:
    config = DataConfig.legacy_v5()
    report, _ = _split(small_corpus(24), config)
    check = report.task_holdout_content_disjoint
    assert isinstance(check, Ran)
    assert not check.passed and check.value and check.value > 0
    # the same raw row's trained families are the ones that overlap
    split_of = {a.row_id: a.split for a in report.assignments}
    kept = [r for rows in report.rows_by_split.values() for r in rows]
    overlap = task_holdout_overlap(kept, split_of=split_of, config=config)
    assert overlap and all(rid.startswith("cpft:") for rid in overlap)
    assert all("identity_key" in kinds for kinds in overlap.values())


def test_the_status_refuses_only_when_the_config_says_so() -> None:
    config = DataConfig.legacy_v5()
    corpus = small_corpus(24)
    report, _ = _split(corpus, config)
    assert not isinstance(report.status, Ran) or report.status.passed
    gated, _ = _split(corpus, replace(config, refuse_task_holdout_content_overlap=True))
    assert isinstance(gated.status, Ran) and not gated.status.passed
    assert "task_holdout_content_disjoint" in gated.status.detail


def test_a_corpus_with_no_shared_content_passes() -> None:
    config = replace(DataConfig.legacy_v5(), refuse_task_holdout_content_overlap=True)
    corpus = {
        "clinc/clinc_oos": [clinc_row(i) for i in range(24)],
        "rajpurkar/squad_v2": [
            squad_row(i, family="qa.answerability" if i % 4 == 0 else "qa.answer_span")
            for i in range(24)
        ],
    }
    report, _ = _split(corpus, config)
    check = report.task_holdout_content_disjoint
    assert isinstance(check, Ran) and check.passed, check
    assert isinstance(report.status, Ran) and report.status.passed


def test_no_held_out_row_is_not_run_rather_than_passed() -> None:
    config = replace(DataConfig.legacy_v5(), refuse_task_holdout_content_overlap=True)
    report, _ = _split({"clinc/clinc_oos": [clinc_row(i) for i in range(12)]}, config)
    assert isinstance(report.task_holdout_content_disjoint, NotRun)


def test_a_copy_under_another_path_is_caught_by_the_content_digest() -> None:
    config = DataConfig.legacy_v5()
    mixture = build_mixture({"bigcode/commitpackft": [commitpackft_row(i) for i in range(8)]},
                            config=config)
    rows = list(mixture.rows)
    held = next(r for r in rows if config.is_held_out_family(r.family_id))
    trained = next(r for r in rows if not config.is_held_out_family(r.family_id))
    moved = replace(trained, identity_key="other/repo::elsewhere.py", dedupe_text=held.dedupe_text)
    others = [r for r in rows if r.row_id != trained.row_id]
    split_of = {r.row_id: "train" for r in rows}
    overlap = task_holdout_overlap([*others, moved], split_of=split_of, config=config)
    assert overlap[moved.row_id] == ("content_digest",)
