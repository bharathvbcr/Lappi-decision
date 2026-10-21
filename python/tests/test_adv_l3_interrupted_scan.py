"""Adversarial extension of L3's ``test_containment_scan_interrupted.py`` (adv/ft-57fc7484-l3).

The lane cuts a request at 0, 1, 8, 25 %, 50 %, 90 % and size-1. Here: a cut at, one byte before
and one byte after the end of every *field* the writer emits (header, corpus, export, checks,
set names, scan specs, the row count and the first and last rows' five fields), plus trailing
bytes after a whole request. Every cut must make ``qd-prep containment`` exit non-zero with no
attestation; trailing bytes are expected to be refused too (a request is read whole).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import BinaryIO

import pytest
from test_containment_exclusions import CORPUS, _report

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from containment_scan import (  # noqa: E402
    run_containment,
    runnable_scans,
    scan_sets,
    splitter_checks,
    write_request,
)

from qd_data.config import DataConfig  # noqa: E402
from qd_train.exclusions import ATTESTATION_NAME  # noqa: E402


class _Spy:
    """Records the offset after every ``write`` call: the writer's field boundaries."""

    def __init__(self, fh: BinaryIO) -> None:
        self.fh, self.ends = fh, []

    def write(self, data: bytes) -> int:
        n = self.fh.write(data)
        self.ends.append(self.fh.tell())
        return n


@pytest.fixture(scope="module")
def request_and_bounds(tmp_path_factory: pytest.TempPathFactory) -> tuple[bytes, list[int]]:
    report = _report()
    sets, strip = scan_sets(report, config=DataConfig.legacy_v5())
    path = tmp_path_factory.mktemp("advreq") / "scan.request.bin"
    with path.open("xb") as fh:
        spy = _Spy(fh)
        write_request(spy, sets, runnable_scans(sets), corpus=CORPUS,  # type: ignore[arg-type]
                      export={"template_strip": strip}, checks=splitter_checks(report))
    return path.read_bytes(), spy.ends


def _cuts(size: int, ends: list[int]) -> list[int]:
    """Every field boundary +-1 for the header part and the first and last 12 writes."""
    chosen = set(ends[:80]) | set(ends[-60:])
    cuts = {c + d for c in chosen for d in (-1, 0, 1)}
    return sorted(c for c in cuts if 0 <= c < size)


def test_every_field_boundary_cut_is_refused_and_leaves_no_attestation(
    qd_prep_bin: Path, tmp_path: Path, request_and_bounds: tuple[bytes, list[int]],
) -> None:
    body, ends = request_and_bounds
    accepted: list[int] = []
    for keep in _cuts(len(body), ends):
        request = tmp_path / f"cut{keep}.request.bin"
        request.write_bytes(body[:keep])
        out = tmp_path / f"cut{keep}"
        try:
            run_containment(qd_prep_bin, request, out, threads=2, timeout_s=120.0)
        except SystemExit:
            assert not (out / ATTESTATION_NAME).exists(), keep
            continue
        accepted.append(keep)
    assert not accepted, f"cuts accepted as whole requests: {accepted[:10]} of {len(body)} bytes"


@pytest.mark.parametrize("tail", [b"\x00", b"junk", b"\x00" * 64])
def test_trailing_bytes_after_a_whole_request_are_refused(
    qd_prep_bin: Path, tmp_path: Path, request_and_bounds: tuple[bytes, list[int]], tail: bytes,
) -> None:
    body, _ends = request_and_bounds
    request = tmp_path / "tail.request.bin"
    request.write_bytes(body + tail)
    with pytest.raises(SystemExit, match="containment exited"):
        run_containment(qd_prep_bin, request, tmp_path / "tail", threads=2, timeout_s=120.0)
    assert not (tmp_path / "tail" / ATTESTATION_NAME).exists()
