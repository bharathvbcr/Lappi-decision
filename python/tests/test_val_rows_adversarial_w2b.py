"""Adversarial tests against lane W2b's VAL_ROWS writer (reviewer: adv/ft-c5ea3e71-w2b).

Each test states the invariant it expects; a failing one is a reported break."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from test_val_rows_writer import CONFIG, _shard_set

from qd_data import val_requests
from qd_data.errors import HeldOutViolation
from qd_data.manifest import Manifest
from qd_data.rows import row_content_hash
from qd_data.val_requests import ValRowsNotRun, binding_path, check_binding, write_val_rows

REPO = Path(__file__).resolve().parents[2]


def _kw(tmp: Path) -> dict[str, object]:
    train, val, rows = _shard_set(tmp / "set")
    return {"rows": rows, "val_manifest": val, "train_manifest": train,
            "config": CONFIG, "repo_root": tmp / "set"}


def test_out_under_a_held_out_root_is_refused(tmp_path: Path) -> None:
    kw = _kw(tmp_path)
    held = tmp_path / "set" / "data" / "heldout"
    held.mkdir(parents=True)
    out = held / "val_rows.jsonl"
    with pytest.raises(HeldOutViolation):
        write_val_rows(out=out, **kw)  # type: ignore[arg-type]
    assert not out.exists() and not binding_path(out).exists()


def test_out_through_a_symlinked_dir_into_a_held_out_root_is_refused(tmp_path: Path) -> None:
    kw = _kw(tmp_path)
    held = tmp_path / "set" / "data" / "heldout"
    held.mkdir(parents=True)
    link = tmp_path / "innocent"
    link.symlink_to(held, target_is_directory=True)
    with pytest.raises(HeldOutViolation):
        write_val_rows(out=link / "val_rows.jsonl", **kw)  # type: ignore[arg-type]
    assert not (held / "val_rows.jsonl").exists()


def test_out_a_directory_fails_closed_and_leaves_nothing(tmp_path: Path) -> None:
    kw = _kw(tmp_path)
    out = tmp_path / "isdir"
    out.mkdir()
    with pytest.raises((ValueError, OSError)):
        write_val_rows(out=out, **kw)  # type: ignore[arg-type]
    assert not binding_path(out).exists()


def test_dangling_symlink_out_does_not_write_through_to_the_target(tmp_path: Path) -> None:
    kw = _kw(tmp_path)
    target = tmp_path / "elsewhere" / "x.jsonl"
    out = tmp_path / "val_rows.jsonl"
    out.symlink_to(target)
    try:
        write_val_rows(out=out, **kw)  # type: ignore[arg-type]
    except (ValueError, OSError):
        pass
    assert not target.exists()


def test_unwritable_dir_fails_closed_without_a_bound_pair(tmp_path: Path) -> None:
    kw = _kw(tmp_path)
    d = tmp_path / "ro"
    d.mkdir()
    d.chmod(0o500)
    try:
        with pytest.raises(OSError):
            write_val_rows(out=d / "v.jsonl", **kw)  # type: ignore[arg-type]
    finally:
        d.chmod(0o700)
    assert not (d / "v.jsonl").exists()


def test_identical_rows_with_a_sidecar_for_other_rows_is_refused(tmp_path: Path) -> None:
    kw = _kw(tmp_path)
    out = tmp_path / "v.jsonl"
    write_val_rows(out=out, **kw)  # type: ignore[arg-type]
    side = binding_path(out)
    stale = json.loads(side.read_text())
    stale["train_manifest_sha256"] = "ab" * 32
    side.write_text(json.dumps(stale, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="differs"):
        write_val_rows(out=out, **kw)  # type: ignore[arg-type]


def test_identical_rows_with_a_missing_sidecar_is_completed_and_bound(tmp_path: Path) -> None:
    kw = _kw(tmp_path)
    out = tmp_path / "v.jsonl"
    write_val_rows(out=out, **kw)  # type: ignore[arg-type]
    binding_path(out).unlink()
    with pytest.raises(ValRowsNotRun):
        check_binding(out, train_manifest_sha256="ab" * 32)
    write_val_rows(out=out, **kw)  # type: ignore[arg-type]
    check_binding(out, train_manifest_sha256=json.loads(
        binding_path(out).read_text())["train_manifest_sha256"])


def test_a_crash_between_rows_and_sidecar_is_never_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kw = _kw(tmp_path)
    out = tmp_path / "v.jsonl"
    real = val_requests._write_atomic
    calls: list[Path] = []

    def boom(path: Path, text: str) -> None:
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("killed")
        real(path, text)

    monkeypatch.setattr(val_requests, "_write_atomic", boom)
    with pytest.raises(RuntimeError):
        write_val_rows(out=out, **kw)  # type: ignore[arg-type]
    assert out.exists() and not binding_path(out).exists()
    with pytest.raises(ValRowsNotRun):
        check_binding(out, train_manifest_sha256="ab" * 32)


def test_a_held_out_family_row_in_the_val_manifest_is_refused(tmp_path: Path) -> None:
    """VAL_ROWS must never contain a held-out row: a val manifest whose entry (and row) carry a
    held-out family id is the held-out data reaching a replay file."""
    train, val, rows = _shard_set(tmp_path / "set")
    held = CONFIG.held_out_families[0]
    victim = rows[0]
    bad = replace(victim, family_id=held)
    manifest = Manifest.read(val)
    entries = tuple(
        replace(e, family_id=held, content_hash=row_content_hash(bad))
        if e.row_id == victim.row_id else e
        for e in manifest.entries
    )
    forged = replace(manifest, entries=entries)
    forged.write(val, allow_not_run=True)
    out = tmp_path / "v.jsonl"
    with pytest.raises((HeldOutViolation, ValueError)):
        write_val_rows([bad, *rows[1:]], val_manifest=val, train_manifest=train, out=out,
                       config=CONFIG, repo_root=tmp_path / "set")
    assert not out.exists()


def test_makefile_default_survives_spaces_in_ft_out() -> None:
    proc = subprocess.run(
        ["make", "-n", "-f", str(REPO / "Makefile"), "release-replay", "RELEASE_DIR=/rel",
         "MIN_ANSWERED=7", "REPLAY_RESULTS=/r.jsonl", "SOCKET=/s.sock", "FT_OUT=/my ft/run 1"],
        cwd=REPO, capture_output=True, text=True, check=False, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'VAL_ROWS="/my ft/run 1/data/pool/val_rows.jsonl"' in proc.stdout
