"""Adversarial (ft-57fc7484 L2 b1, 154dfaa): every bad shard root must fail loud, name the
problem, and never be returned as trainable. Rule 3 (held-out) must run before any listing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_data.config import DataConfig  # noqa: E402
from qd_data.errors import HeldOutViolation  # noqa: E402
from qd_train import shards as shards_module  # noqa: E402
from qd_train.shards import (  # noqa: E402
    HEADER_NAME,
    ShardContractViolation,
    assert_shard_files_trainable,
)


def _call(root: Path):
    return assert_shard_files_trainable(root, config=DataConfig(), repo_root=REPO)


def test_a_regular_file_root_is_refused_naming_the_path(tmp_path: Path) -> None:
    f = tmp_path / "train"
    f.write_text("not a directory", encoding="utf-8")
    with pytest.raises(OSError) as raised:
        _call(f)
    assert str(f) in str(raised.value), "the refusal must name the root"


def test_a_dangling_symlink_root_is_refused_naming_header_json(tmp_path: Path) -> None:
    link = tmp_path / "train"
    link.symlink_to(tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError, match=r"header\.json"):
        _call(link)


def test_a_symlink_root_into_a_held_out_path_is_a_held_out_violation_not_missing(
    tmp_path: Path,
) -> None:
    held = tmp_path / "heldout" / "shards"
    held.mkdir(parents=True)
    (held / HEADER_NAME).write_text("{}", encoding="utf-8")
    link = tmp_path / "train"
    link.symlink_to(held)
    with pytest.raises(HeldOutViolation):
        _call(link)


def test_a_missing_root_under_a_held_out_path_is_a_held_out_violation_first(
    tmp_path: Path,
) -> None:
    with pytest.raises(HeldOutViolation):
        _call(tmp_path / "heldout" / "shards" / "train")


def test_a_dangling_symlink_into_a_held_out_path_is_a_held_out_violation_first(
    tmp_path: Path,
) -> None:
    link = tmp_path / "train"
    link.symlink_to(tmp_path / "heldout" / "missing")
    with pytest.raises(HeldOutViolation):
        _call(link)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory modes")
def test_an_unreadable_root_fails_loud_and_names_the_root(tmp_path: Path) -> None:
    d = tmp_path / "train"
    d.mkdir()
    (d / HEADER_NAME).write_text("{}", encoding="utf-8")
    d.chmod(0)
    try:
        with pytest.raises(PermissionError) as raised:
            _call(d)
        assert str(d) in str(raised.value)
    finally:
        d.chmod(0o700)


def test_unicode_and_spaces_in_the_root_are_named_verbatim(tmp_path: Path) -> None:
    root = tmp_path / "shärds wï th spaces" / "train ☃"
    with pytest.raises(FileNotFoundError) as raised:
        _call(root)
    assert str(root / HEADER_NAME) in str(raised.value)
    assert raised.value.filename == str(root / HEADER_NAME)


def test_an_existing_empty_directory_is_not_reported_as_a_trainable_shard_set(
    tmp_path: Path,
) -> None:
    """The listing succeeds on an empty directory, so the function returns {} as 'trainable'
    though the format's required header.json is absent. The docstring promises only the rule-3
    sweep, but the message the fix added says header.json is what is required; an existing
    directory without it should be refused the same way (or the caller must check)."""
    d = tmp_path / "train"
    d.mkdir()
    with pytest.raises((FileNotFoundError, ShardContractViolation)):
        _call(d)


def test_the_directory_deleted_between_the_held_out_check_and_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = tmp_path / "train"
    d.mkdir()
    (d / HEADER_NAME).write_text("{}", encoding="utf-8")
    real = shards_module.assert_path_not_held_out
    calls = {"n": 0}

    def check_then_delete(path: Path, **kw: object) -> None:
        real(path, **kw)
        if calls["n"] == 0:
            calls["n"] += 1
            (d / HEADER_NAME).unlink()
            d.rmdir()

    monkeypatch.setattr(shards_module, "assert_path_not_held_out", check_then_delete)
    with pytest.raises(FileNotFoundError, match=r"header\.json"):
        _call(d)


def test_the_directory_replaced_by_a_file_between_the_check_and_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = tmp_path / "train"
    d.mkdir()
    real = shards_module.assert_path_not_held_out
    calls = {"n": 0}

    def check_then_swap(path: Path, **kw: object) -> None:
        real(path, **kw)
        if calls["n"] == 0:
            calls["n"] += 1
            d.rmdir()
            d.write_text("x", encoding="utf-8")

    monkeypatch.setattr(shards_module, "assert_path_not_held_out", check_then_swap)
    with pytest.raises(OSError) as raised:
        _call(d)
    assert str(d) in str(raised.value)


def test_a_member_deleted_during_the_sweep_is_a_contract_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = tmp_path / "train"
    d.mkdir()
    (d / HEADER_NAME).write_text("{}", encoding="utf-8")
    (d / "zz-shard-0.bin").write_bytes(b"x")
    real = shards_module.assert_path_not_held_out

    def check_then_delete(path: Path, **kw: object) -> None:
        real(path, **kw)
        if Path(path).name == HEADER_NAME and (d / "zz-shard-0.bin").exists():
            (d / "zz-shard-0.bin").unlink()

    monkeypatch.setattr(shards_module, "assert_path_not_held_out", check_then_delete)
    with pytest.raises(ShardContractViolation, match="vanished"):
        _call(d)
