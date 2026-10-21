"""Adversarial (ft-57fc7484 L2 b1, 154dfaa): the CommonsenseQA fixture change.

``test_replay_rows_are_marked_and_never_in_the_gold_train_split`` asserts only a negative
(no REPLAY_ONLY row in train). It passes when the replay draw is empty, which is exactly the
state the MMLU fixture was in under v6 (it passed on d09222d). These tests pin that the
scenario that test describes actually has replay rows to be absent from train, and that a
draw of zero cannot satisfy it."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_real_ft_general_record import (
    EMPTY_DRAW_MESSAGE,
    MMLU_ROWS,
    PINNED,
    REPLAY_ONLY,
    REPLAY_ROLE_KEY,
    REPO,
    _entry,
    _file,
    _record,
    assert_replay_rows_exist_are_marked_and_stay_out_of_train,
    resolve_rev,
    rft,
)

from qd_data.config import DataConfig


def _mmlu_only_record(root: Path) -> Path:
    import json

    mmlu = _file(root, "cais__mmlu/rev/test.jsonl", MMLU_ROWS)
    path = root / "fetch-record-mmlu.json"
    path.write_text(json.dumps([_entry("cais/mmlu", mmlu, len(MMLU_ROWS))]), encoding="utf-8")
    return path


@pytest.mark.usefixtures("qd_prep")
def test_the_marked_and_never_in_train_scenario_has_a_nonempty_replay_slice(
    tmp_path: Path,
) -> None:
    rev = resolve_rev(REPO, PINNED)
    record = _record(tmp_path)
    kw = dict(commitpackft=None, max_pairs=3, rev=rev, config=DataConfig(), general_record=record)
    part = rft.ft_splits(replay_partition=True, **kw)
    unpart = rft.ft_splits(**kw)
    replay = {r.row_id for r in unpart["train"]} - {r.row_id for r in part["train"]}
    assert replay, "the replay draw is empty: the 'never in train' assertion is vacuous"
    assert not any(r.metadata.get(REPLAY_ROLE_KEY) == REPLAY_ONLY for r in part["train"])


@pytest.mark.usefixtures("qd_prep")
def test_a_zero_draw_fails_the_marked_and_never_in_train_check(tmp_path: Path) -> None:
    """With only MMLU rows (refused under v6) the draw is zero. The check the scenario's own
    test makes must not be satisfiable by that: it raises on the empty slice, and on nothing
    else (the repository history keeps train non-empty, so the old negative-only assertions
    pass here and only the nonempty-draw assertion can tell)."""
    record = _mmlu_only_record(tmp_path)
    with pytest.raises(AssertionError, match="the replay draw is empty") as raised:
        assert_replay_rows_exist_are_marked_and_stay_out_of_train(record, repo_history=True)
    assert EMPTY_DRAW_MESSAGE in str(raised.value)
