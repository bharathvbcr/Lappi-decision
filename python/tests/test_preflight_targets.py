"""`make preflight-train`, `make preflight-release`, `make gpu-pins` (docs/hardening.md s8).

The targets tally subshell checks with their true 0/1/3 status. Asserted here, with the heavy
checks (`gates`, cargo-run binaries) replaced through make's command-line overrides so the test
exercises the tally and the argument handling rather than a full workspace build:

1. A target missing a required argument is NotRun, never PASS, and says which argument.
2. One FAILED makes the target FAILED even when another check is NotRun.
3. A pass needs every check to have run and passed.
4. `gpu-pins` is NotRun without a declared GPU window, without Apple-silicon Metal and without the
   snapshot; it names the `--ignored` digest-pin test; `gates` does not run it.

GNU make collapses a failed recipe to exit 2, so the verdict is read from the `RESULT:` line.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAKEFILE = REPO / "Makefile"
TIMEOUT_S = 600
OK = "exit 0"
GATES_OK = "printf 'RESULT: PASS\\n'; exit 0"


def _make(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "-f", str(MAKEFILE), "--no-print-directory", *args],
        capture_output=True, text=True, timeout=TIMEOUT_S, cwd=str(REPO),
    )


def _light(**overrides: str) -> list[str]:
    """Every heavy check replaced with `exit 0`, then the given ones overridden."""
    cmds = {
        "GATES_VIA_MAKE_CMD": GATES_OK, "BASE_FRESH_CMD": OK, "TRAIN_DATA_CMD": OK,
        "PREREG_BIND_CMD": OK, "RELEASE_CHECK_CMD": OK, "CITE_CMD": OK, "GPU_PINS_CMD": OK,
    }
    cmds.update(overrides)
    return [f"{k}={v}" for k, v in cmds.items()]


def _result(proc: subprocess.CompletedProcess[str]) -> str:
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")]
    assert lines, proc.stdout
    return lines[-1]


def test_every_check_passing_is_the_only_way_to_pass():
    for target in ("preflight-train", "preflight-release"):
        proc = _make(target, *_light())
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert _result(proc).startswith("RESULT: PASS"), proc.stdout


def test_preflight_train_without_arguments_is_notrun_naming_each_missing_one():
    proc = _make("preflight-train", "GATES_VIA_MAKE_CMD=" + GATES_OK, "BASE_FRESH_CMD=" + OK)
    assert proc.returncode != 0, "preflight-train with no arguments exited 0"
    assert _result(proc).startswith("RESULT: NotRun"), proc.stdout
    assert "not a pass" in proc.stdout, proc.stdout
    for name in ("TRAIN_SHARDS", "PREREG"):
        assert f"{name} is not set" in proc.stdout, (name, proc.stdout)
    assert "train-data         NotRun" in proc.stdout, proc.stdout
    assert "prereg-bind        NotRun" in proc.stdout, proc.stdout


def test_preflight_release_without_arguments_is_notrun_and_names_the_missing_argument():
    proc = _make(
        "preflight-release", "GATES_VIA_MAKE_CMD=" + GATES_OK, "BASE_FRESH_CMD=" + OK,
        "CITE_CMD=" + OK,
        "RELEASE_DIR=/x", "REPLAY_RESULTS=/y",
    )
    assert proc.returncode != 0
    assert _result(proc).startswith("RESULT: NotRun"), proc.stdout
    assert "MIN_ANSWERED is not set" in proc.stdout, proc.stdout
    assert "release-check      NotRun" in proc.stdout, proc.stdout
    # No GPU window is open here, so the digest pins did not run either.
    assert "gpu-pins           NotRun" in proc.stdout, proc.stdout


def test_a_failed_check_outranks_a_notrun_one():
    proc = _make("preflight-release", *_light(BASE_FRESH_CMD="exit 1", CITE_CMD="exit 3"))
    assert proc.returncode != 0
    assert _result(proc).startswith("RESULT: FAILED"), proc.stdout
    assert "base-fresh         FAILED (exit 1)" in proc.stdout, proc.stdout
    assert "cite               NotRun" in proc.stdout, proc.stdout


def test_a_gates_run_that_prints_no_pass_is_not_a_pass():
    # The nested `make gates` result is read from its RESULT: line; exit 0 alone is not enough.
    proc = _make("preflight-train", *_light(GATES_VIA_MAKE_CMD="exit 1"))
    assert _result(proc).startswith("RESULT: FAILED"), proc.stdout


def test_gpu_pins_is_notrun_without_a_window_metal_or_the_snapshot():
    cases = [
        ([], "no GPU window declared"),
        (["GPU_WINDOW=1", "GPU_UNAME_S=Linux"], "not Apple-silicon macOS"),
        (["GPU_WINDOW=1", "GPU_UNAME_S=Darwin", "GPU_UNAME_M=arm64",
          "GPU_METAL_PROBE=false"], "no Metal GPU"),
        (["GPU_WINDOW=1", "GPU_UNAME_S=Darwin", "GPU_UNAME_M=arm64", "GPU_METAL_PROBE=true",
          "GPU_SNAPSHOTS=/nonexistent/snapshots"], "no Qwen3.5-2B-Base snapshot"),
    ]
    for args, reason in cases:
        proc = _make("gpu-pins", *args)
        assert proc.returncode != 0, f"{args}: gpu-pins exited 0 without running a GPU test"
        assert f"NotRun: gpu-pins - {reason}" in proc.stdout, (args, proc.stdout)


def test_gpu_pins_runs_the_ignored_digest_pin_test_and_gates_does_not():
    recipe = _make("-n", "gpu-pins", "GPU_WINDOW=1").stdout
    assert "gpu_prefix_state_digest_v1_is_pinned" in recipe and "--ignored" in recipe, recipe
    assert "-p qd-metal --test gpu" in recipe, recipe
    assert "gpu_prefix_state_digest_v1_is_pinned" not in _make("-n", "gates").stdout
    release = _make("-n", "preflight-release", "GATES_VIA_MAKE_CMD=" + GATES_OK).stdout
    assert "gpu_prefix_state_digest_v1_is_pinned" in release, release


def test_help_documents_both_targets():
    out = _make("help").stdout
    names = ("preflight-train", "preflight-release", "gpu-pins", "TRAIN_SHARDS", "RELEASE_DIR",
             "MIN_ANSWERED")
    for name in names:
        assert name in out, (name, out)
