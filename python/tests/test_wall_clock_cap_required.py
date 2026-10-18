"""A registered run names its wall-clock cap; only a run that says quick gets the quick one.

GAP-LORA2B-REDUCED-RUN-DEFAULT-CAP-1800S-2026-10-17 and
GAP-RUN-CONTROL-DEFAULT-CAP-CALLERS-OUTSIDE-L-OPS-2026-10-17: ``real_ft_run.py`` filled in a
1800 s cap when ``--wall-clock-cap-s`` was left out, and a registered, non-quick run silently
stopped after about 13 steps. ``WallClockCap.resolve(cap, quick)`` is the Python twin of
``crates/qd-train/src/run_control.rs`` ``WallClockCap::resolve``; the tests pin the two to one
value and one refusal.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

from qd_train.run_control import MAX_CAP_S, QUICK_CAP_S, WallClockCap  # noqa: E402

RUST = (REPO / "crates" / "qd-train" / "src" / "run_control.rs").read_text(encoding="utf-8")


def _rust_quick_cap_s() -> float:
    found = re.search(r"pub const QUICK_CAP_S: f64 = ([0-9_.]+);", RUST)
    assert found, "run_control.rs no longer defines QUICK_CAP_S"
    return float(found.group(1).replace("_", ""))


def _rust_refusal() -> str:
    block = RUST[RUST.index("pub fn resolve("):]
    literal = re.search(r'\(None, false\) => Err\(RunControlError::Invalid\(\s*"(.*?)"\s*\.into',
                        block, re.S)
    assert literal, "run_control.rs resolve() no longer refuses (None, false)"
    # A Rust string continuation (backslash, newline, indentation) joins with nothing.
    return re.sub(r"\\\n\s*", "", literal.group(1))


def test_the_quick_cap_and_the_refusal_are_the_rust_ones() -> None:
    assert QUICK_CAP_S == _rust_quick_cap_s() == 1_800.0
    with pytest.raises(ValueError) as refused:
        WallClockCap.resolve(None, False)
    assert str(refused.value) == _rust_refusal()


def test_resolve_follows_the_rust_truth_table() -> None:
    assert WallClockCap.resolve(None, True).cap_s == QUICK_CAP_S
    assert WallClockCap.resolve(32_400.0, False).cap_s == 32_400.0
    assert WallClockCap.resolve(32_400.0, True).cap_s == 32_400.0
    for bad in (0.0, -1.0, float("nan"), MAX_CAP_S + 1.0):
        with pytest.raises(ValueError):
            WallClockCap.resolve(bad, False)


def test_there_is_one_python_owner_of_the_quick_cap() -> None:
    import real_ft_run

    assert real_ft_run.QUICK_CAP_S is QUICK_CAP_S
    assert real_ft_run.WALL_CLOCK_CAP_S == QUICK_CAP_S
    assert not re.search(r"^QUICK_CAP_S\s*[:=]", (REPO / "tools" / "real_ft_run.py").read_text(),
                         re.M)


def test_a_training_run_without_a_cap_or_quick_is_refused_at_argv(tmp_path: Path) -> None:
    import real_ft_run

    with pytest.raises(SystemExit) as refused:
        real_ft_run.main(["--out", str(tmp_path), "--epoch"])
    assert _rust_refusal() in str(refused.value)
    assert "--wall-clock-cap-s" in str(refused.value)


def test_quick_gets_the_quick_cap_and_a_named_cap_needs_no_quick(tmp_path: Path) -> None:
    import real_ft_run

    for argv in (["--quick"], ["--wall-clock-cap-s", "32400"]):
        # Past every argv check, the first thing the run reads is the (absent) shard set.
        with pytest.raises(FileNotFoundError, match="header.json"):
            real_ft_run.main(["--out", str(tmp_path), "--epoch", *argv])


def test_a_declared_quick_run_is_marked_quick() -> None:
    import real_ft_run

    base = dict(
        tag="epoch", device="cuda", real_backbone=True,
        corpus=real_ft_run.CorpusFacts(snapshot_not_run=None, history_rows=0),
        termination="steps_exhausted",
    )
    assert real_ft_run.quick_reasons(**base) == []
    reasons = real_ft_run.quick_reasons(**base, declared_quick=True)
    assert len(reasons) == 1 and "--quick" in reasons[0]
