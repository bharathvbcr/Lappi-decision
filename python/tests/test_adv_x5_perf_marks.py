"""Adversarial tests for X5's perf stale-code marks. BREAK tests fail on lane/x5-v5r-retire."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import perf_parity  # noqa: E402


def _row(tag: str, **extra: object) -> dict[str, object]:
    return {
        "tag": tag, "steps": 3, "loss_log_digest": "d", "consumed_digest": "c",
        "letter": [(1.0).hex()] * 3, "span": [(2.0).hex()] * 3, "pos_per_s": 1.0,
        "peak_alloc_gib": 1.0, **extra,
    }


def test_break_compare_of_stale_arms_drops_the_stale_mark(tmp_path: Path) -> None:
    """Two stale arms (stale_code, quick true) compared: the report says identical: true with
    no stale_code / quick, so the verdict loses the mark the rows carried."""
    f = tmp_path / "r.jsonl"
    f.write_text("\n".join(json.dumps(_row(t, stale_code=True, quick=True)) for t in "ab") + "\n")
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert perf_parity.compare([f"{f}:a", f"{f}:b"]) == 0
    report = json.loads(buf.getvalue())
    assert report["identical"] is True
    assert report.get("stale_code") is True and report.get("quick") is True


def test_break_p2_screen_of_stale_arms_drops_the_stale_mark() -> None:
    """The P2 screen's verdict row (appended to --p2-out and read by --p2-gate to cancel or
    allow the outcome run) is computed from arm rows and never carries stale_code."""
    import inspect

    src = inspect.getsource(perf_parity.p2_screen) + inspect.getsource(perf_parity.p2_gate)
    assert "stale_code" in src
