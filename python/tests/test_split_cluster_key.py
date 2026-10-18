"""The split seam for qd-mutate's ``cluster_key`` (survey M4, ``docs/hardening.md`` section 1).

Every row sharing a ``cluster_key`` must be in one split. The repo hash makes that structural
(the key includes the repo), so these tests force the case the hash cannot: two rows of one
function pinned to different splits.
"""

from __future__ import annotations

import dataclasses

import pytest
from data_fixtures import code_body

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.rows import DataRow, GoldAnswer
from qd_data.schema import ChoiceSlot, Request
from qd_data.sources import PINNED_SPLIT_KEY
from qd_data.split import CLUSTER_KEY_METADATA, split
from qd_train.tristate import Ran


def _row(row_id: str, *, repo: str, pinned: str, cluster: str | None, seed: int) -> DataRow:
    text = code_body(seed)
    metadata = {PINNED_SPLIT_KEY: pinned}
    if cluster is not None:
        metadata[CLUSTER_KEY_METADATA] = cluster
    return DataRow(
        row_id=row_id,
        source_id="qd-mutate",
        host="local",
        family_id="code.defect_class",
        repo_key=repo,
        identity_key=f"{repo}::{row_id}",
        licence_id="mit",
        obligations=("attribution",),
        request=Request(
            task="code.defect_class",
            context=text.encode("utf-8"),
            question="q?",
            slots=(ChoiceSlot(name="implements_claim", options=("yes", "no")),),
            example_id=row_id,
        ),
        gold=(GoldAnswer(slot_name="implements_claim", value="yes"),),
        dedupe_text=text,
        metadata=metadata,
    )


def _split(rows: list[DataRow]):
    config = DataConfig.legacy_v5()
    return split(dedupe(rows, config=config), config=config)


def test_two_rows_of_one_function_in_two_splits_fail_the_cluster_check() -> None:
    rows = [
        _row("a", repo="o/r", pinned="train", cluster="fk1:aa", seed=1),
        _row("b", repo="o/r2", pinned="val", cluster="fk1:aa", seed=2),
    ]
    report = _split(rows)
    # Their identity keys differ, so only the cluster key can see the straddle.
    assert isinstance(report.identity_disjoint, Ran) and report.identity_disjoint.passed
    check = report.cluster_key_disjoint
    assert isinstance(check, Ran) and not check.passed
    assert "cluster_key" in check.detail
    assert not isinstance(report.status, Ran) or not report.status.passed


def test_rows_of_one_function_in_one_split_pass() -> None:
    rows = [
        _row("a", repo="o/r", pinned="train", cluster="fk1:aa", seed=1),
        _row("b", repo="o/r", pinned="train", cluster="fk1:aa", seed=2),
        _row("c", repo="o/r", pinned="val", cluster="fk1:bb", seed=3),
    ]
    report = _split(rows)
    check = report.cluster_key_disjoint
    assert isinstance(check, Ran) and check.passed
    assert report.cluster_key_rows == 3 and report.identity_fallback_rows == 0


def test_unkeyed_rows_fall_back_to_identity_and_are_counted() -> None:
    rows = [
        _row("a", repo="o/r", pinned="train", cluster="fk1:aa", seed=1),
        _row("b", repo="o/r", pinned="train", cluster=None, seed=2),
    ]
    report = _split(rows)
    assert report.cluster_key_rows == 1
    assert report.identity_fallback_rows == 1
    body = report.to_json()
    assert body["identity_fallback_rows"] == 1


def test_a_corpus_with_no_keys_reports_as_it_did() -> None:
    report = _split([_row("a", repo="o/r", pinned="train", cluster=None, seed=1)])
    assert report.cluster_key_disjoint is None
    assert "cluster_key_disjoint" not in report.to_json()


def test_a_blank_key_is_refused() -> None:
    row = _row("a", repo="o/r", pinned="train", cluster=" ", seed=1)
    with pytest.raises(ValueError, match="blank"):
        _split([row])


def test_replace_keeps_the_metadata() -> None:
    row = _row("a", repo="o/r", pinned="train", cluster="fk1:aa", seed=1)
    assert dataclasses.replace(row, row_id="z").metadata[CLUSTER_KEY_METADATA] == "fk1:aa"
