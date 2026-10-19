"""Rule 3 at the shard reader: a shard FILE or DIRECTORY reached through a symlink.

``ShardReader`` ran the held-out door on the shard directory only. The files it then opens
(header, offsets, tokens, supervision, ...) were never passed through it, so a file symlinked
into a held-out root or marker directory was read as training data. Found by W2-2's adversarial
run; the reader is the canonical owner, so every caller (real_ft_run included) is covered here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from test_shards import snapshot  # noqa: F401 - fixture
from test_shards import train_shards as _built_shards  # noqa: F401 - fixture

from qd_data.errors import HeldOutViolation
from qd_train.artifacts import ShardContractViolation
from qd_train.shards import (
    MAX_SHARD_DIR_ENTRIES,
    OFFSETS_NAME,
    TOKENS_NAME,
    ShardReader,
    assert_shard_files_trainable,
)


@pytest.fixture
def train_shards(_built_shards):  # noqa: F811 - the aliased fixture is requested by name
    return _built_shards


def _read(snap, out: Path) -> None:
    ShardReader(out, config=snap.config, repo_root=snap.root)


def _swap_for_link(shard: Path, name: str, target: Path) -> None:
    """Move shard/name to target and leave a symlink in its place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(shard / name), target)
    (shard / name).symlink_to(target)


def test_control_untouched_shard_set_is_admitted(train_shards) -> None:
    snap, out, _ = train_shards
    _read(snap, out)


@pytest.mark.parametrize("name", [TOKENS_NAME, OFFSETS_NAME])
def test_file_symlink_into_configured_held_out_root_is_refused(train_shards, name) -> None:
    snap, out, _ = train_shards
    _swap_for_link(out, name, snap.root / "data" / "heldout" / "stash" / name)
    with pytest.raises(HeldOutViolation):
        _read(snap, out)


def test_file_symlink_into_marker_directory_is_refused(train_shards, tmp_path) -> None:
    snap, out, _ = train_shards
    _swap_for_link(out, TOKENS_NAME, tmp_path / "HeldOut" / TOKENS_NAME)
    with pytest.raises(HeldOutViolation):
        _read(snap, out)


def test_relative_file_symlink_with_dotdot_is_refused(train_shards, tmp_path) -> None:
    snap, out, _ = train_shards
    held = tmp_path / "held_out"
    held.mkdir()
    shutil.move(str(out / TOKENS_NAME), held / TOKENS_NAME)
    depth = len(out.relative_to(tmp_path).parts)
    (out / TOKENS_NAME).symlink_to(Path(*([".."] * depth)) / "held_out" / TOKENS_NAME)
    with pytest.raises(HeldOutViolation):
        _read(snap, out)


def test_chained_file_symlink_is_refused(train_shards, tmp_path) -> None:
    snap, out, _ = train_shards
    real = tmp_path / "held-out" / TOKENS_NAME
    real.parent.mkdir()
    shutil.move(str(out / TOKENS_NAME), real)
    hop2 = tmp_path / "hop2"
    hop2.symlink_to(real)
    hop1 = tmp_path / "hop1"
    hop1.symlink_to(hop2)
    (out / TOKENS_NAME).symlink_to(hop1)
    with pytest.raises(HeldOutViolation):
        _read(snap, out)


def test_directory_symlink_into_held_out_is_refused(train_shards, tmp_path) -> None:
    snap, out, _ = train_shards
    real = snap.root / "data" / "heldout" / "shards"
    real.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(out, real)
    clean = tmp_path / "clean"
    clean.symlink_to(real)
    with pytest.raises(HeldOutViolation):
        _read(snap, clean)


def test_dotdot_through_a_directory_symlink_is_refused(train_shards, tmp_path) -> None:
    snap, out, _ = train_shards
    held = tmp_path / "heldout"
    shutil.copytree(out, held / "set")
    link = tmp_path / "link"
    link.symlink_to(held / "set")
    # link/.. is physically tmp/heldout, so link/../set is a held-out path.
    with pytest.raises(HeldOutViolation):
        _read(snap, link / ".." / "set")


def test_directory_over_the_entry_cap_is_refused(train_shards) -> None:
    snap, out, _ = train_shards
    for i in range(MAX_SHARD_DIR_ENTRIES):
        (out / f"pad-{i}").touch()
    with pytest.raises(ShardContractViolation, match="cannot all be checked"):
        _read(snap, out)


def test_public_check_admits_a_clean_set_and_opens_no_file(train_shards) -> None:
    snap, out, _ = train_shards
    # Unreadable members: any open would raise PermissionError, so a pass proves none was opened.
    unreadable = [out / name for name in (TOKENS_NAME, OFFSETS_NAME, "header.json")]
    for path in unreadable:
        path.chmod(0)
    try:
        assert_shard_files_trainable(out, config=snap.config, repo_root=snap.root)
    finally:
        for path in unreadable:
            path.chmod(0o644)


def test_public_check_refuses_a_symlinked_member_without_a_reader(train_shards) -> None:
    snap, out, _ = train_shards
    _swap_for_link(out, TOKENS_NAME, snap.root / "data" / "heldout" / "stash" / TOKENS_NAME)
    with pytest.raises(HeldOutViolation):
        assert_shard_files_trainable(out, config=snap.config, repo_root=snap.root)


def test_public_check_refuses_a_dangling_symlink_into_a_marker_directory(
    train_shards, tmp_path
) -> None:
    snap, out, _ = train_shards
    (out / "extra.bin").symlink_to(tmp_path / "heldout" / "nowhere.bin")
    with pytest.raises(HeldOutViolation):
        assert_shard_files_trainable(out, config=snap.config, repo_root=snap.root)


def test_public_check_refuses_a_directory_over_the_entry_cap(train_shards) -> None:
    snap, out, _ = train_shards
    for i in range(MAX_SHARD_DIR_ENTRIES):
        (out / f"pad-{i}").touch()
    with pytest.raises(ShardContractViolation, match="cannot all be checked"):
        assert_shard_files_trainable(out, config=snap.config, repo_root=snap.root)


def test_marker_in_the_given_path_refuses_even_when_it_resolves_clean(
    train_shards, tmp_path
) -> None:
    snap, out, _ = train_shards
    (tmp_path / "heldout").symlink_to(out, target_is_directory=True)
    with pytest.raises(HeldOutViolation):
        _read(snap, tmp_path / "heldout")
