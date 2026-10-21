"""A shard set that is not there is refused naming ``header.json``, the file the format requires.

Before the fix ``assert_shard_files_trainable`` let the directory listing's own error escape
(``[Errno 2] No such file or directory: '.../shards/train'``), so an operator who pointed a
scoring run at a set that had not been built was told about a directory, not about the file
the reader needs. Fails closed either way; this pins that the message names the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_data.config import DataConfig  # noqa: E402
from qd_train.shards import HEADER_NAME, assert_shard_files_trainable  # noqa: E402


def test_a_missing_shard_directory_is_refused_naming_header_json(tmp_path: Path) -> None:
    missing = tmp_path / "shards" / "train"
    with pytest.raises(FileNotFoundError, match=r"shards/train/header\.json") as raised:
        assert_shard_files_trainable(missing, config=DataConfig(), repo_root=REPO)
    assert raised.value.filename == str(missing / HEADER_NAME)
    assert "no shard set at" in str(raised.value)


def test_an_existing_directory_is_still_listed_not_refused_as_missing(tmp_path: Path) -> None:
    (tmp_path / HEADER_NAME).write_text("{}", encoding="utf-8")
    identities = assert_shard_files_trainable(tmp_path, config=DataConfig(), repo_root=REPO)
    assert set(identities) == {HEADER_NAME}


def test_the_reader_refuses_a_directory_with_no_header_naming_it_not_as_a_held_out_violation(
    tmp_path: Path,
) -> None:
    """Through the real caller, ``ShardReader``. Before the fix the sweep returned {} for an
    existing directory with no header.json, and the reader's later open of it raised a
    ``HeldOutViolation`` ("not in the directory when it was swept"): a refusal, but one that
    accuses a set of rule 3 for the crime of being unbuilt. It is a missing file."""
    from qd_data.errors import HeldOutViolation
    from qd_train.shards import ShardReader

    empty = tmp_path / "train"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=r"train/header\.json") as raised:
        ShardReader(empty, config=DataConfig(), repo_root=REPO)
    assert not isinstance(raised.value, HeldOutViolation)


def test_a_directory_of_other_files_but_no_header_is_still_a_missing_header(tmp_path: Path) -> None:
    (tmp_path / "tokens.u32").write_bytes(b"\x00")
    with pytest.raises(FileNotFoundError, match=r"header\.json"):
        assert_shard_files_trainable(tmp_path, config=DataConfig(), repo_root=REPO)
