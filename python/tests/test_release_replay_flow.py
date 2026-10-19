"""`make release-replay` / `tools/release_replay.sh`: build the requests, replay them through the
socket path, check, in one tracked command
(GAP-QD-RELEASE-CHECK-NO-TRACKED-REQUEST-BUILDER-2026-10-17's residual).

Driven end to end over a tiny fixture release with a stand-in `qd`, the way
crates/qd-preflight/tests/release_check.rs drives `replay`. The stand-in answers every request, so
this proves the flow's wiring and its verdicts, not that a model serves anything: the fixture can
never PASS overall, because the headline figures need verdicts from a real eval.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from qd_data.schema import ChoiceSlot, Request, SpanSlot

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "release_replay.sh"
WEIGHT = "6f7b9ba73bca3a0b2e4ec9b2c0fc2aa6e7459f7e3286b26386633a222667dfcb"
CALIB = "7e56b34e378907adccc7cfc0a086507d81c762ed6627694a0a59c3c977aeb677"
OK_REPLY = json.dumps({
    "status": "ok", "schema_version": 1, "backend": "qd-metal/test", "degraded": False,
    "slots": {"x": {"value": "a", "conformal_set": ["a"], "score": 0.5, "noul": False, "degraded": False}},
})
REQUIRED = ("RELEASE_DIR", "VAL_ROWS", "MIN_ANSWERED", "REPLAY_RESULTS", "SOCKET")


def _choice(n: int) -> ChoiceSlot:
    return ChoiceSlot(name="x", options=tuple(f"o{i}" for i in range(n)))


def _row(task: str, i: int, slots: tuple, *, split: str = "val") -> dict:
    request = Request(task=task, context=b"x", question="Q?", slots=slots, example_id=f"{task}:{i}")
    return {"row_id": f"{task}:{i}:{split}", "split": split, "request": request.to_wire()}


def _write_lines(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _release(d: Path) -> Path:
    rel = d / "rel"
    rel.mkdir(exist_ok=True)
    (rel / "release_manifest.json").write_text(json.dumps({
        "format": "qd-release.v2",
        "trained_families": ["code.defect_class", "fam.b"],
        "excluded_families": [],
        "train_manifest": {"ledger_binding": {"verified": True, "train_manifest_sha256": "ab" * 32}},
        "expected_identity": {"weight_hash": WEIGHT, "calibration_hash": CALIB},
        "trained_slot_shapes": {"code.defect_class": [["choice:5", "span"]], "fam.b": [["choice:4"]]},
    }), encoding="utf-8")
    return rel


def _fake_qd(d: Path) -> Path:
    p = d / "fake-qd.sh"
    p.write_text(
        "#!/bin/sh\nIN=$(cat)\ncase \"$IN\" in\n"
        f"  *'\"op\"'*) echo '{{\"status\":\"status\",\"backend\":{{\"weight_hash\":\"{WEIGHT}\","
        f"\"calibration_hash\":\"{CALIB}\"}}}}' ;;\n"
        f"  *) echo '{OK_REPLY}' ;;\nesac\necho 'qd oneshot: served by socket' 1>&2\n",
        encoding="utf-8",
    )
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def _val_rows(n_b: int, *, extra: list[dict] | None = None) -> list[dict]:
    rows = [_row("code.defect_class", i, (_choice(4), SpanSlot(name="s"))) for i in range(3)]
    rows += [_row("fam.b", i, (_choice(3),)) for i in range(n_b)]
    rows += [_row("not.admitted", i, (_choice(3),)) for i in range(4)]
    return rows + (extra or [])


def _run(
    d: Path, release_check_bin: Path, rows: list[dict], min_answered: str | None, *, drop: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "RELEASE_DIR": str(_release(d)),
        "VAL_ROWS": str(_write_lines(d / "val_rows.jsonl", rows)),
        "REPLAY_RESULTS": str(d / "out" / "results.jsonl"),
        "SOCKET": str(d / "q.sock"),
        "QD": str(_fake_qd(d)),
        "QD_PYTHON": sys.executable,
        "QD_RELEASE_CHECK_BIN": str(release_check_bin),
        "PYTHONPATH": str(REPO / "python"),
    }
    if min_answered is not None:
        env["MIN_ANSWERED"] = min_answered
    if drop is not None:
        env.pop(drop, None)
    return subprocess.run(
        ["bash", str(SCRIPT)], cwd=d, env=env, capture_output=True, text=True, timeout=300, check=False,
    )


def _last(proc: subprocess.CompletedProcess[str]) -> str:
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""


@pytest.mark.parametrize("missing", REQUIRED)
def test_a_missing_required_argument_is_notrun_and_runs_nothing(
    tmp_path: Path, release_check_bin: Path, missing: str,
) -> None:
    proc = _run(tmp_path, release_check_bin, _val_rows(4), "1", drop=missing)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert f"{missing} is not set" in proc.stdout
    assert _last(proc) == "RESULT: NotRun"
    assert not (tmp_path / "out" / "results.jsonl").exists()
    assert not (tmp_path / "out" / "replay_requests.jsonl").exists()


def test_min_answered_has_no_default_and_zero_is_refused(tmp_path: Path, release_check_bin: Path) -> None:
    unset = _run(tmp_path, release_check_bin, _val_rows(4), None)
    assert unset.returncode == 3 and "MIN_ANSWERED is not set" in unset.stdout
    zero = _run(tmp_path, release_check_bin, _val_rows(4), "0")
    assert zero.returncode == 3 and "not a positive integer" in zero.stdout


def test_a_short_family_shape_is_failed_and_nothing_is_replayed(tmp_path: Path, release_check_bin: Path) -> None:
    # fam.b at choice:4 has 2 val rows; 3 are asked for.
    proc = _run(tmp_path, release_check_bin, _val_rows(2), "3")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "fam.b" in proc.stdout and "(short)" in proc.stdout
    assert "FAILED: release-replay" in proc.stdout
    assert _last(proc) == "RESULT: FAILED"
    assert not (tmp_path / "out" / "results.jsonl").exists()


def test_a_train_row_in_the_val_rows_is_failed(tmp_path: Path, release_check_bin: Path) -> None:
    rows = _val_rows(4, extra=[_row("fam.b", 99, (_choice(3),), split="train")])
    proc = _run(tmp_path, release_check_bin, rows, "1")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "not 'val'" in proc.stderr
    assert _last(proc) == "RESULT: FAILED"
    assert not (tmp_path / "out" / "results.jsonl").exists()


def test_a_full_fixture_passes_every_check_line_but_is_never_pass_overall(
    tmp_path: Path, release_check_bin: Path,
) -> None:
    proc = _run(tmp_path, release_check_bin, _val_rows(4), "3")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert _last(proc) == "RESULT: NotRun"
    assert "RESULT: PASS" not in [line.strip() for line in proc.stdout.splitlines()]
    assert "check step RESULT: PASS" in proc.stdout
    assert "headline figures were not computed" in proc.stdout
    results = tmp_path / "out" / "results.jsonl"
    assert results.is_file(), "preflight-release reads this file"
    assert (tmp_path / "out" / "replay_requests.jsonl").is_file()
    assert json.loads(results.read_text(encoding="utf-8").splitlines()[0])["kind"] == "header"
    check_lines = [
        line for line in proc.stdout.splitlines()
        if line.split(" ", 1)[0] in {"Pass", "Fail", "NotRun"} and not line.startswith("NotRun headline")
    ]
    assert check_lines, proc.stdout
    assert all(line.startswith("Pass ") for line in check_lines), check_lines
    # The check's own PASS is what the flow downgrades; the check itself said so.
    assert "RESULT: PASS" in (tmp_path / "out" / "check_output.txt").read_text(encoding="utf-8")


def test_the_make_target_is_notrun_without_its_arguments() -> None:
    proc = subprocess.run(
        ["make", "-f", str(REPO / "Makefile"), "release-replay"],
        cwd=REPO, capture_output=True, text=True, check=False, timeout=60,
    )
    assert proc.returncode != 0
    assert "RELEASE_DIR is not set" in proc.stdout
    assert "RESULT: NotRun" in proc.stdout


def test_the_make_target_passes_its_arguments_to_the_script() -> None:
    proc = subprocess.run(
        ["make", "-n", "-f", str(REPO / "Makefile"), "release-replay", "RELEASE_DIR=/rel", "VAL_ROWS=/v.jsonl",
         "MIN_ANSWERED=7", "REPLAY_RESULTS=/r.jsonl", "SOCKET=/s.sock"],
        cwd=REPO, capture_output=True, text=True, check=False, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    for part in ('RELEASE_DIR="/rel"', 'VAL_ROWS="/v.jsonl"', 'MIN_ANSWERED="7"',
                 'REPLAY_RESULTS="/r.jsonl"', 'SOCKET="/s.sock"', "tools/release_replay.sh"):
        assert part in proc.stdout, part
