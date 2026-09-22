"""``--val-shards``: a val split written under the remap a trained model is scored with.

Nothing scored a fine-tuned 2B on rows it did not train on
(``GAP-RUNG3-NOTHING-SCORES-A-TRAINED-MODEL-ON-ROWS-IT-DID-NOT-TRAIN-ON``), and the reason
was not only the missing scorer: the pipeline wrote train shards alone, under a remap built
from the train rows alone, which encodes 5 of 184 val rows (row d9b461ca). ``--val-shards``
builds the remap over train AND val -- the written corpus, which is the remap policy's own
rule -- and writes the val split through the same door and writer as train.

End to end, because the claim is about the artifacts: a val shard set exists, holds every
val row, and shares the train set's remap. Needs the live tokenizer and the local
commitpackft download, and skips where either is absent. 60 pairs take seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_tokenizer_pipeline as pipeline  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_train.shards import ShardReader  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

DOWNLOAD = REPO / "data" / "pool" / "commitpackft"


def test_the_val_split_is_written_under_the_remap_the_train_split_uses(tmp_path) -> None:
    pytest.importorskip("transformers")
    if not pipeline.MODEL_REF.exists():
        pytest.skip(f"{pipeline.MODEL} is not in this host's HF cache")
    if not any((DOWNLOAD / f"{lang}.jsonl").exists() for lang in ("go", "python")):
        pytest.skip("the commitpackft download is not on this host; only its manifest is")

    measured = pipeline.run(
        out=tmp_path, max_pairs=60, blank_line_runs=False, rev="HEAD",
        commitpackft=DOWNLOAD, val_shards=True,
    )

    coverage = measured.metrics["val_shard_coverage"]
    assert isinstance(coverage, Ran), coverage
    assert coverage.passed and coverage.n == coverage.n_total and coverage.n_total > 0
    # Built over val, so a count of val's coverage would measure nothing -- and says so.
    assert isinstance(measured.metrics["remap_covers_val_rows"], NotRun)
    held_out = measured.metrics["remap_covers_heldout_rows"]
    assert isinstance(held_out, Ran) and "train and val remap kept" in held_out.detail

    config = DataConfig()
    train = ShardReader(tmp_path / "shards" / "train", config=config, repo_root=tmp_path)
    val = ShardReader(tmp_path / "shards" / "val", config=config, repo_root=tmp_path)
    assert val.header.split == "val"
    assert val.header.vocab_size == train.header.vocab_size
    assert val.header.remap_hash == train.header.remap_hash, "scored under another remap"


def test_without_the_flag_no_val_set_is_claimed() -> None:
    """The default path's new metric states that nothing was written, rather than being
    absent -- a check that did not run must not read like one that passed."""
    source = (REPO / "tools" / "real_tokenizer_pipeline.py").read_text(encoding="utf-8")
    assert "--val-shards was not passed, so no val shard set was written" in source
