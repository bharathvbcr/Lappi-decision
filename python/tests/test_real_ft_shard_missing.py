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
