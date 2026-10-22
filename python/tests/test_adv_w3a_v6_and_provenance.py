"""Adversarial tests against lane W3a (lane/ft-cc9fac73), written by the adversarial reviewer.

Two groups:

* the ledger-provenance exemption (`1191554`) over inputs W3a's two tests do not cover; and
* a v6-valid rebuild of the label-pairing test W3a moved to the legacy v5 rules (`a528a37`),
  to show whether that move was needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

from test_ledger_provenance import _row, _training_rows_of  # noqa: E402

# --- ledger provenance: what the "ended" exemption lets through ---------------------------


def _ids(rows: list[dict]) -> list[str]:
    return [r["row_id"] for r in _training_rows_of(rows)]


def test_a_started_row_that_names_itself_is_not_ended_by_a_terminal_row() -> None:
    """The docstring says a started row is excluded 'once a terminal row names it'. A started
    row carrying its own id in ``lifecycle.started_row_id`` has no terminal row at all, and
    must stay in (an orphan: a run whose end nobody saw)."""
    rows = [_row("s1", status="started", ended="s1")]
    assert _ids(rows) == ["s1"]


def test_a_started_row_named_only_by_another_started_row_is_not_ended() -> None:
    """A terminal row is a row that is not itself ``started``. Two started markers, one naming
    the other, end nothing."""
    rows = [_row("s1", status="started"), _row("s2", status="started", ended="s1")]
    assert _ids(rows) == ["s1", "s2"]


def test_a_started_row_is_ended_only_by_a_terminal_row_of_its_own_ledger_file() -> None:
    """``ledger/`` holds duplicate row_ids across files (forks of one ledger copied rows), so
    a terminal row in file B naming ``s1`` must not end the started row ``s1`` of file A."""
    started = _row("s1", status="started")
    started["_file"] = "a.jsonl"
    terminal = _row("t1", status="completed", digest=True, ended="s1")
    terminal["_file"] = "b.jsonl"
    assert _ids([started, terminal]) == ["s1", "t1"]


def test_a_terminal_row_naming_no_started_row_changes_nothing() -> None:
    rows = [_row("s1", status="started"), _row("t9", status="completed", digest=True, ended="zz")]
    assert _ids(rows) == ["s1", "t9"]


def test_two_terminal_rows_naming_one_started_row_both_stay_and_are_checked() -> None:
    rows = [
        _row("s1", status="started"),
        _row("t1", status="completed", digest=True, ended="s1"),
        _row("t2", status="failed", ended="s1"),
    ]
    assert _ids(rows) == ["t1", "t2"]


def test_a_terminal_row_of_another_run_kind_does_not_end_a_started_row() -> None:
    """A smoke started row ended by a *build* terminal row (a different run) is not ended."""
    started = _row("s1", status="started")
    terminal = _row("t1", status="completed", digest=True, ended="s1")
    terminal["run_kind"] = "build"
    assert _ids([started, terminal]) == ["s1"]


# --- label pairing under the v6 rules ------------------------------------------------------

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
import real_ft_run as rft  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402

DEFECT_CORPUS = REPO / "data" / "pool" / "commitpackft-corpus-v2"
CPFT = REPO / "data" / "pool" / "commitpackft"


@pytest.mark.skipif(
    not (DEFECT_CORPUS / "examples.jsonl").is_file() or not CPFT.is_dir(),
    reason="the qd-mutate corpus or the commitpackft download is not on this host",
)
@pytest.mark.usefixtures("qd_prep")
def test_the_defect_class_relabel_runs_under_the_v6_rules_with_fewer_pairs(tmp_path: Path) -> None:
    """``test_a_defect_class_shard_set_is_relabelled_by_id...`` was moved to ``legacy_v5`` because
    R1 refused its 20-pair / 40-defect-row build (defect share 38.7% < 55.8%). Fewer pairs keeps
    the same 40 defect rows and raises the share, so the same pairing check runs under v6."""
    pytest.importorskip("transformers")
    import real_tokenizer_pipeline as pipeline
    from repo_git import resolve_rev

    from qd_train.shards import ShardReader

    rev = resolve_rev(REPO, "7f233553a2766fd2ad956ec018c359eef056d4d2")
    out = tmp_path / "out"
    try:
        pipeline.run(out=out, max_pairs=4, blank_line_runs=False, rev=rev, commitpackft=CPFT,
                     defect_class=DEFECT_CORPUS, defect_max_rows=40)
    except OSError as exc:
        pytest.skip(f"the pipeline could not load its tokenizer: {exc}")
    config = pipeline.data_rules(DataConfig(), v6=True) if hasattr(pipeline, "data_rules") \
        else DataConfig()
    manifest = json.loads((out / rft.TRAIN_MANIFEST).read_text())
    reader = ShardReader(out / "shards" / "train", config=config, repo_root=out, expect_rev=rev)
    train, _ = rft.ft_split_rows(commitpackft=CPFT, max_pairs=4, rev=rev, config=config,
                                 defect_class=DEFECT_CORPUS, defect_max_rows=40)
    assert len(train) == manifest["n_rows"]
    labels, excluded = rft._labels(train, config=config)
    paired = rft.pair_labels(reader, labels, require_index=True)
    assert [(lb.row_id, lb.slot_name) for lb in paired] == list(reader.sequence_index.sequences)
    rft._inventory(reader, paired, excluded)
    rft._letter_ids(reader, paired)


@pytest.mark.skipif(
    not (DEFECT_CORPUS / "examples.jsonl").is_file() or not CPFT.is_dir(),
    reason="the qd-mutate corpus or the commitpackft download is not on this host",
)
@pytest.mark.usefixtures("qd_prep")
def test_the_val_remap_characterization_runs_under_the_v6_rules(tmp_path: Path) -> None:
    """``test_pipeline_val_shards`` was moved to ``legacy_v5`` because its plain-commitpackft build
    has no defect class and R1 refuses it. The same val-remap assertions with a defect corpus
    and few pairs run under the v6 default rules."""
    pytest.importorskip("transformers")
    import real_tokenizer_pipeline as pipeline

    from qd_train.shards import ShardReader
    from qd_train.tristate import NotRun, Ran

    try:
        measured = pipeline.run(
            out=tmp_path, max_pairs=4, blank_line_runs=False, rev="HEAD", commitpackft=CPFT,
            defect_class=DEFECT_CORPUS, defect_max_rows=40, val_shards=True,
            vocab=pipeline.VOCAB_CORPUS,
        )
    except OSError as exc:
        pytest.skip(f"the pipeline could not load its tokenizer: {exc}")
    coverage = measured.metrics["val_shard_coverage"]
    assert isinstance(coverage, Ran) and coverage.passed and coverage.n == coverage.n_total > 0
    assert isinstance(measured.metrics["remap_covers_val_rows"], NotRun)
    config = DataConfig()
    train = ShardReader(tmp_path / "shards" / "train", config=config, repo_root=tmp_path)
    val = ShardReader(tmp_path / "shards" / "val", config=config, repo_root=tmp_path)
    assert val.header.split == "val"
    assert val.header.remap_hash == train.header.remap_hash
