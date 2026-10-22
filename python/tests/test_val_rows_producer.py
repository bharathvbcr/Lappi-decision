"""The VAL_ROWS producer (GAP-W11B-VAL-ROWS-PRODUCER-NEEDS-REAL-FT-RUN-2026-10-19,
GAP-W9B-NO-TRACKED-VAL-ROWS-FILE-WRITER-2026-10-18): ``real_ft_run.py --write-val-rows`` writes the
file ``make release-replay`` defaults VAL_ROWS to, through ``write_val_rows`` and nothing else.

The pure-writer tests need no torch; the ones that import ``tools/real_ft_run.py`` are torch-gated
like the rest of that module's tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_val_rows_writer import CONFIG, _shard_set

from qd_data import val_requests
from qd_data.val_requests import ValRowsNotRun, binding_path, check_binding, write_val_rows

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REL = "data/pool/val_rows.jsonl"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rft() -> Any:
    pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
    pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")
    sys.path.insert(0, str(REPO / "tools"))
    sys.path.insert(0, str(REPO / "python"))
    import real_ft_run

    return real_ft_run


def _replay_check(rows: Path, out: Path, train_sha: str) -> subprocess.CompletedProcess[str]:
    """The binding check ``tools/release_replay.sh`` runs as its step 1."""
    return subprocess.run(
        [sys.executable, "-m", "qd_data.val_requests", "--val-rows", str(rows),
         "--out", str(out), "--expect-train-manifest-sha256", train_sha],
        env={**os.environ, "PYTHONPATH": str(REPO / "python")},
        capture_output=True, text=True, check=False, timeout=120,
    )


def _make_n(*extra: str) -> str:
    proc = subprocess.run(
        ["make", "-n", "-f", str(REPO / "Makefile"), "release-replay", "RELEASE_DIR=/rel",
         "MIN_ANSWERED=7", "REPLAY_RESULTS=/r.jsonl", "SOCKET=/s.sock", *extra],
        cwd=REPO, capture_output=True, text=True, check=False, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


# --- the Makefile default -------------------------------------------------------------------


def test_the_make_target_defaults_val_rows_to_the_producers_file() -> None:
    assert f'VAL_ROWS="/ft/{DEFAULT_REL}"' in _make_n("FT_OUT=/ft")


def test_an_explicit_val_rows_beats_the_default_and_no_ft_out_leaves_it_empty() -> None:
    assert 'VAL_ROWS="/v.jsonl"' in _make_n("FT_OUT=/ft", "VAL_ROWS=/v.jsonl")
    assert 'VAL_ROWS=""' in _make_n()


# --- the flag and its place in main ----------------------------------------------------------


def test_the_flag_is_on_the_parser_and_runs_after_the_split_before_training(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rft = _rft()
    with pytest.raises(SystemExit) as stop:
        rft.main(["--help"])
    assert stop.value.code == 0
    assert "--write-val-rows" in capsys.readouterr().out
    import inspect

    src = inspect.getsource(rft.main)
    call = src.index("write_run_val_rows(")
    assert src.index("train_rows, val_rows = rebuild()") < call < src.index("relabel_train(")


# --- the producer, then release_replay's check ------------------------------------------------


def test_the_produced_file_passes_release_replays_binding_check(tmp_path: Path) -> None:
    rft = _rft()
    train, _, rows = _shard_set(tmp_path / "set")
    out = tmp_path / "set" / DEFAULT_REL
    n = rft.write_run_val_rows(out, out=tmp_path / "set", val_rows=rows, config=CONFIG)
    assert n == len(rows) > 0
    proc = _replay_check(out, tmp_path / "requests.jsonl", _sha(train))
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "requests.jsonl").read_text().count("\n") == n


def test_a_hand_made_file_at_the_default_path_is_still_refused(tmp_path: Path) -> None:
    train, _, _ = _shard_set(tmp_path / "set")
    hand = tmp_path / "set" / DEFAULT_REL
    hand.write_text("", encoding="utf-8")
    proc = _replay_check(hand, tmp_path / "requests.jsonl", _sha(train))
    assert proc.returncode == 3 and "binding" in proc.stderr


def test_a_held_out_row_is_never_written(tmp_path: Path) -> None:
    rft = _rft()
    _, _, rows = _shard_set(tmp_path / "set")
    from data_fixtures import small_corpus

    from qd_data.dedupe import dedupe
    from qd_data.mixture import build_mixture
    from qd_data.split import HELD_OUT, split

    mixture = build_mixture(small_corpus(24), config=CONFIG)
    report = split(dedupe(list(mixture.rows), config=CONFIG), config=CONFIG)
    held = list(report.rows_by_split.get(HELD_OUT, ()))
    assert held, "the fixture has no held-out rows to test with"
    out = tmp_path / "set" / DEFAULT_REL
    with pytest.raises(ValueError, match="ids differ"):
        rft.write_run_val_rows(out, out=tmp_path / "set", val_rows=[*rows, held[0]], config=CONFIG)
    assert not out.exists() and not binding_path(out).exists()
    ok = _write_ok(rft, tmp_path / "set", rows, tmp_path / "ok.jsonl")
    written = {json.loads(line)["row_id"] for line in ok.read_text().splitlines()}
    assert held[0].row_id not in written


def _write_ok(rft: Any, root: Path, rows: list[Any], dest: Path) -> Path:
    rft.write_run_val_rows(dest, out=root, val_rows=rows, config=CONFIG)
    return dest


def test_another_shard_set_has_another_binding_the_old_sidecar_refuses(tmp_path: Path) -> None:
    rft = _rft()
    train_a, _, rows_a = _shard_set(tmp_path / "a")
    out_a = tmp_path / "a" / DEFAULT_REL
    rft.write_run_val_rows(out_a, out=tmp_path / "a", val_rows=rows_a, config=CONFIG)

    # Another seed: a different split, hence different manifests.
    from data_fixtures import small_corpus

    from qd_data.dedupe import dedupe
    from qd_data.manifest import build_manifests
    from qd_data.mixture import build_mixture
    from qd_data.split import split

    cfg_b = replace(CONFIG, seed=CONFIG.seed + 1)
    mixture = build_mixture(small_corpus(30), config=cfg_b)
    dd = dedupe(list(mixture.rows), config=cfg_b)
    report = split(dd, config=cfg_b)
    manifests = build_manifests(
        config=cfg_b, mixture=mixture, dedupe_report=dd, split_report=report,
        provenance={"built_at_utc": "2026-01-02T00:00:00+00:00"},
    )
    pool = tmp_path / "b" / "data" / "pool"
    pool.mkdir(parents=True)
    for name in ("train", "val"):
        manifests[name].write(pool / f"{name}.json", allow_not_run=True)
    out_b = tmp_path / "b" / DEFAULT_REL
    rft.write_run_val_rows(
        out_b, out=tmp_path / "b", val_rows=list(report.rows_by_split["val"]), config=CONFIG,
    )
    sha_a, sha_b = _sha(train_a), _sha(pool / "train.json")
    assert sha_a != sha_b
    assert json.loads(binding_path(out_a).read_text())["train_manifest_sha256"] == sha_a
    assert json.loads(binding_path(out_b).read_text())["train_manifest_sha256"] == sha_b
    # The release built from set B refuses set A's file, and the reverse.
    assert _replay_check(out_a, tmp_path / "r.jsonl", sha_b).returncode == 1
    assert _replay_check(out_b, tmp_path / "r.jsonl", sha_a).returncode == 1
    assert _replay_check(out_b, tmp_path / "r.jsonl", sha_b).returncode == 0


# --- the writer's overwrite and interruption rules (no torch) ---------------------------------


def _kw(root: Path, train: Path, val: Path) -> dict[str, Any]:
    return {"val_manifest": val, "train_manifest": train, "config": CONFIG, "repo_root": root}


def test_rewriting_identical_bytes_is_a_no_op_and_different_bytes_are_refused(
    tmp_path: Path,
) -> None:
    train, val, rows = _shard_set(tmp_path / "set")
    out = tmp_path / "v.jsonl"
    write_val_rows(rows, out=out, **_kw(tmp_path / "set", train, val))
    before = (out.read_bytes(), binding_path(out).read_bytes())
    write_val_rows(rows, out=out, **_kw(tmp_path / "set", train, val))
    assert (out.read_bytes(), binding_path(out).read_bytes()) == before

    # The train manifest changes (another seed's set): the binding would differ, so refuse.
    train.write_text(train.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not overwritten"):
        write_val_rows(rows, out=out, **_kw(tmp_path / "set", train, val))
    assert (out.read_bytes(), binding_path(out).read_bytes()) == before

    # A tampered rows file is refused too, never silently replaced.
    out.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="not overwritten"):
        write_val_rows(rows, out=out, **_kw(tmp_path / "set", train, val))
    assert out.read_bytes() == b"{}\n"


def test_concurrent_writers_never_share_a_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    real = Path.replace

    def replace(self: Path, target: Any) -> Path:
        seen.append(self.name)
        return real(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    val_requests._write_atomic(tmp_path / "v.jsonl", "a\n")
    val_requests._write_atomic(tmp_path / "v.jsonl", "a\n")
    monkeypatch.undo()
    assert len(set(seen)) == 2, seen
    assert all(n.startswith(".v.jsonl.") and str(os.getpid()) in n for n in seen), seen

    # A second writer that drew the same name does not interleave: O_EXCL refuses it, and the
    # first writer's temp file is left as it was.
    monkeypatch.setattr(val_requests.secrets, "token_hex", lambda _n: "0" * 16)
    taken = tmp_path / f".w.jsonl.{os.getpid()}.{'0' * 16}.partial"
    taken.write_text("the other writer's bytes", encoding="utf-8")
    with pytest.raises(FileExistsError):
        val_requests._write_atomic(tmp_path / "w.jsonl", "mine\n")
    assert taken.read_text(encoding="utf-8") == "the other writer's bytes"
    assert not (tmp_path / "w.jsonl").exists()


@pytest.mark.parametrize("fail_on", [1, 2])
def test_an_interrupted_write_leaves_no_file_that_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_on: int,
) -> None:
    train, val, rows = _shard_set(tmp_path / "set")
    out = tmp_path / "v.jsonl"
    real = Path.replace
    calls = {"n": 0}

    def replace(self: Path, target: Any) -> Path:
        calls["n"] += 1
        if calls["n"] == fail_on:
            raise OSError("interrupted")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match="interrupted"):
        write_val_rows(rows, out=out, **_kw(tmp_path / "set", train, val))
    monkeypatch.undo()
    with pytest.raises((ValRowsNotRun, ValueError, OSError)):
        check_binding(out, train_manifest_sha256=_sha(train))
    assert not binding_path(out).exists()
    # A rerun completes the pair, and the result passes.
    write_val_rows(rows, out=out, **_kw(tmp_path / "set", train, val))
    check_binding(out, train_manifest_sha256=_sha(train))
    assert val_requests.read_val_rows(out)
