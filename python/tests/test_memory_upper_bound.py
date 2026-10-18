"""The memory predictor's admission figure is an upper bound on every recorded peak.

GAP-MEMORY-CAPACITY-PREDICTOR-ADMITS-MORE-ROWS-THAN-FIT and
GAP-MEMORY-PREDICTS-084-TO-097X-OF-MEASURED-BACKBONE-FOOTPRINT-ON-GH200: ``estimate_step``
errs low, and a run killed for memory (memguard SIGTERM, or the CUDA allocator) writes no row.

The recorded measurements are read from the committed files rather than retyped:

* ``AUDIT/gh200_footprint-2026-09-20.json`` (ADAMW_BF16) and
  ``AUDIT/gh200_footprint_master-2026-09-20.json`` (ADAMW_MASTER): ``tools/gh200_footprint.py``,
  ``torch.cuda.max_memory_allocated`` over the second optimizer step, rows=1, a proxy loss, so
  the fused-CE head is not in the measurement and not in the predicted side either.
* ``tools/gh200_rows.py`` walks, recorded in ``gaps.jsonl``
  GAP-MEMORY-CAPACITY-PREDICTOR-ADMITS-MORE-ROWS-THAN-FIT (and repeated in
  ``test_memory.py``): the real step, ADAMW_BF16, full vocabulary. Peaks are quoted there to
  two decimals of GiB, so each carries up to 5 MiB of rounding.

No constant is fitted here: a bound has to hold on every row, and a row it fails is reported.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qd_train.memory import (
    ADAMW_BF16,
    ADAMW_MASTER,
    QWEN3_5_2B_TEXT,
    estimate_step,
    max_positions_that_fit,
)

REPO = Path(__file__).resolve().parents[2]
GIB = 1024**3

#: (width, rows the device fitted, peak GiB at that shape, rows the predictor admitted at the
#: 91.42 GiB `mem_get_info` reported free before the fix). Source: gaps.jsonl
#: GAP-MEMORY-CAPACITY-PREDICTOR-ADMITS-MORE-ROWS-THAN-FIT.
FULL_STEP_ROWS = ((34_522, 8, 83.05, 9), (8_192, 36, 87.95, 38))
FREE_GIB = 91.42


def _footprint_rows() -> list[tuple[str, object, int, int, int]]:
    """(label, optimizer, width, rows, measured peak bytes), the larger of the two steps."""
    out = []
    for name, spec in (("gh200_footprint-2026-09-20.json", ADAMW_BF16),
                       ("gh200_footprint_master-2026-09-20.json", ADAMW_MASTER)):
        record = json.loads((REPO / "AUDIT" / name).read_text(encoding="utf-8"))
        for r in record["results"]:
            peak = max(int(r["measured_bytes"]), int(r["first_step_bytes"]))
            out.append((f"{name}@{r['width']}", spec, int(r["width"]), int(r["rows"]), peak))
    return out


def test_the_recorded_table_is_read_in_full():
    rows = _footprint_rows()
    assert len(rows) == 10, f"expected 5 widths x 2 recipes, read {len(rows)}"


@pytest.mark.parametrize("label,spec,width,rows,measured", _footprint_rows())
def test_the_prediction_is_an_upper_bound_on_every_footprint_measurement(
    label, spec, width, rows, measured
):
    fp = estimate_step(QWEN3_5_2B_TEXT, rows=rows, width=width, optimizer=spec)
    # The probe's proxy loss runs no fused-CE head, so that term is not in what was measured.
    predicted = fp.total_bytes - fp.loss_head_bytes
    assert predicted >= measured, (
        f"{label}: the predictor says {predicted / GIB:.2f} GiB and the device peaked at "
        f"{measured / GIB:.2f} GiB ({measured / predicted:.3f}x). An admission figure below a "
        "recorded peak admits a run that is then killed and leaves no row."
    )


@pytest.mark.parametrize("width,fitted,peak_gib,_admitted", FULL_STEP_ROWS)
def test_the_prediction_is_an_upper_bound_on_every_full_step_measurement(
    width, fitted, peak_gib, _admitted
):
    fp = estimate_step(QWEN3_5_2B_TEXT, rows=fitted, width=width, optimizer=ADAMW_BF16)
    measured = round(peak_gib * GIB) + GIB // 200  # the quote's 2-decimal rounding, upward
    assert fp.total_bytes >= measured, (
        f"{width} x {fitted}: predicted {fp.total_bytes / GIB:.2f} GiB, measured "
        f"{peak_gib:.2f} GiB"
    )


@pytest.mark.parametrize("width,fitted,_peak,admitted_before", FULL_STEP_ROWS)
def test_the_capacity_answer_no_longer_admits_more_rows_than_the_device_fitted(
    width, fitted, _peak, admitted_before
):
    positions = max_positions_that_fit(
        QWEN3_5_2B_TEXT, device_bytes=int(FREE_GIB * GIB), width=width, optimizer=ADAMW_BF16
    )
    rows = positions // width
    assert rows <= fitted, (
        f"width {width}: the predictor admits {rows} rows against the {fitted} the device "
        f"fitted (it admitted {admitted_before} before the fix)"
    )
    assert rows >= 1, "an answer of zero would be safe and useless; the bound must not be that"


# --- memguard's ceiling: read from mac_heavy.sh, never copied ---------------------------------


def _plist(tmp_path: Path, gib: str) -> Path:
    plist = tmp_path / "memguard.plist"
    plist.write_text(
        "<plist><dict><array><string>-cap-gib</string>"
        f"<string>{gib}</string></array></dict></plist>\n"
    )
    return plist


def test_the_ceiling_is_memguards_threshold_less_mac_heavys_margin(tmp_path, monkeypatch):
    from qd_train.memory import host_footprint_ceiling_bytes

    monkeypatch.setenv("MAC_HEAVY_MEMGUARD_PLIST", str(_plist(tmp_path, "32")))
    assert host_footprint_ceiling_bytes() == (32 * 1024 - 4096) * 1024 * 1024
    monkeypatch.setenv("MAC_HEAVY_MEMGUARD_PLIST", str(_plist(tmp_path, "16")))
    assert host_footprint_ceiling_bytes() == (16 * 1024 - 4096) * 1024 * 1024


def test_a_prediction_at_or_above_the_ceiling_is_refused(tmp_path, monkeypatch):
    import dataclasses

    from qd_train.memory import (
        MemoryRefused,
        host_footprint_ceiling_bytes,
        refuse_if_over_host_ceiling,
    )

    monkeypatch.setenv("MAC_HEAVY_MEMGUARD_PLIST", str(_plist(tmp_path, "32")))
    ceiling = host_footprint_ceiling_bytes()
    fp = estimate_step(QWEN3_5_2B_TEXT, rows=1, width=2048, optimizer=ADAMW_BF16)
    base = fp.total_bytes - fp.safety_bytes
    below = dataclasses.replace(fp, safety_bytes=ceiling - 1 - base)
    at = dataclasses.replace(fp, safety_bytes=ceiling - base)
    assert below.total_bytes == ceiling - 1 and at.total_bytes == ceiling
    refuse_if_over_host_ceiling(below, where="below")
    with pytest.raises(MemoryRefused, match="at or above"):
        refuse_if_over_host_ceiling(at, where="at")
    # The master recipe is refused on a 32 GiB-threshold Mac: 2B + master AdamW is ~45 GB.
    big = estimate_step(QWEN3_5_2B_TEXT, rows=1, width=2048, optimizer=ADAMW_MASTER)
    with pytest.raises(MemoryRefused):
        refuse_if_over_host_ceiling(big, where="2B master")


def test_an_unreadable_threshold_refuses_rather_than_admitting(tmp_path, monkeypatch):
    from qd_train.memory import MemoryRefused, refuse_if_over_host_ceiling

    bad = tmp_path / "bad.plist"
    bad.write_text("<plist><dict></dict></plist>\n")
    monkeypatch.setenv("MAC_HEAVY_MEMGUARD_PLIST", str(bad))
    monkeypatch.setenv("MAC_HEAVY_MEMGUARD_MAIN", str(tmp_path / "nope.go"))
    fp = estimate_step(QWEN3_5_2B_TEXT, rows=1, width=128, optimizer=ADAMW_BF16)
    with pytest.raises(MemoryRefused, match="cannot read memguard's threshold"):
        refuse_if_over_host_ceiling(fp, where="unreadable")


def test_print_limits_runs_nothing_and_takes_no_lock(tmp_path):
    import os
    import subprocess

    from qd_train.memory import MAC_HEAVY_SCRIPT

    lock = tmp_path / "lock.d"
    done = subprocess.run(
        ["bash", str(MAC_HEAVY_SCRIPT), "--print-limits"],
        env={**os.environ, "MAC_HEAVY_MEMGUARD_PLIST": str(_plist(tmp_path, "32")),
             "MAC_HEAVY_LOCK": str(lock)},
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == [
        "memguard_threshold_mib=32768", "default_footprint_cap_mib=28672"
    ]
    assert not lock.exists()
