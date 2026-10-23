"""A stale arm can never yield a promotable pass (Fable 2026-10-22, items 6 and 7).

``compare``, ``p2_screen`` and ``p2_gate`` carry ``stale_code`` and ``quick`` from any arm
read with ``--allow-stale-shards`` into their report and verdict.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import perf_parity as pp  # noqa: E402

N = 10


def _arm(tag: str, scale: float, *, mask: str, stale: bool = False, jitter: float = 0.0) -> dict:
    letter = [(2.0 - 0.1 * t) * scale * (1 + (jitter if t else 0.0)) for t in range(N)]
    span = [(1.0 - 0.05 * t) * scale * (1 + (jitter if t else 0.0)) for t in range(N)]
    row = {
        "tag": tag, "steps": N, "consumed_digest": "c" * 64, "termination": "steps_exhausted",
        "train_attention_mask": mask, "train_path": {"tag": tag},
        "letter": [x.hex() for x in letter], "span": [x.hex() for x in span],
    }
    if stale:
        row.update(stale_code=True, quick=True)
    return row


def _bases(*, stale_first: bool = False) -> list[dict]:
    return [
        _arm(f"A-mask-{k + 1}", 1.0, mask="padding", stale=stale_first and k == 0,
             jitter=s * 1e-4)
        for k, s in enumerate((-1.0, 0.0, 1.0))
    ]


def _cands() -> list[dict]:
    return [_arm(f"A-none-{i + 1}", 1.0, mask="none") for i in range(3)]


def test_p2_screen_without_a_stale_arm_is_unchanged_and_unmarked() -> None:
    out = pp.p2_screen(_bases(), _cands())
    assert out["verdict"] == "pass"
    assert "stale_code" not in out and "quick" not in out


def test_one_stale_arm_makes_the_p2_verdict_a_marked_not_run() -> None:
    out = pp.p2_screen(_bases(stale_first=True), _cands())
    assert out["screen_verdict"] == "pass"  # what the statistic said, kept beside
    assert out["verdict"] == "not_run"  # what the gate reads: never a pass
    assert out["stale_code"] is True and out["quick"] is True
    assert out["stale_arms"] == ["A-mask-1"]


def test_the_gate_cancels_on_a_stale_verdict_row_even_if_it_says_pass() -> None:
    stale = pp.p2_screen(_bases(stale_first=True), _cands())
    clean = pp.p2_screen(_bases(), _cands())
    b_clean = {**clean, "base": ["B-mask-1"]}
    ok, _ = pp.p2_gate([clean, b_clean])
    assert ok  # the control: two clean passes let the outcome run go
    forged = {**stale, "verdict": "pass"}  # a row someone edited to read pass
    ok, detail = pp.p2_gate([forged, b_clean])
    assert not ok and "stale_code" in detail
    ok, _ = pp.p2_gate([stale, b_clean])
    assert not ok


def test_compare_carries_the_stale_mark_to_the_report(tmp_path: Path) -> None:
    rows = [_arm("a", 1.0, mask="padding", stale=True), _arm("b", 1.0, mask="padding")]
    for r in rows:
        r.update(loss_log_digest="d", pos_per_s=1.0, peak_alloc_gib=1.0)
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert pp.compare([f"{path}:a", f"{path}:b"]) == 0
    report = json.loads(buf.getvalue())
    assert report["identical"] is True
    assert report["stale_code"] is True and report["quick"] is True
    assert report["stale_arms"] == ["a"]
