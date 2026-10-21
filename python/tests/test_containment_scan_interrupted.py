"""An interrupted scan leaves nothing that passes as a finished one.

``tools/containment_scan.py`` writes the request, then ``qd-prep containment`` writes the
attestation, the pair list and the exclusion list. A scan killed in either step (memguard's
SIGTERM, mac_heavy's cap, a full disk) must not leave a directory a build would read:
``qd_train.exclusions.read_exclusions`` needs the attestation beside the list, and the binary
writes the attestation only for a request it read whole. Pinned here: a request cut short at any
point, or empty, is refused by the binary with a non-zero exit (``run_containment`` raises) and no
``attestation.json``; a rerun into the same names is refused by ``main`` before it overwrites
anything; and the same request whole scans clean (the control).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from test_containment_exclusions import CORPUS, _report

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import containment_scan  # noqa: E402
from containment_scan import (  # noqa: E402
    run_containment,
    runnable_scans,
    scan_sets,
    splitter_checks,
    write_request,
)

from qd_data.config import DataConfig  # noqa: E402
from qd_train.exclusions import ATTESTATION_NAME  # noqa: E402


@pytest.fixture(scope="module")
def request_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    report = _report()
    sets, strip = scan_sets(report, config=DataConfig.legacy_v5())
    path = tmp_path_factory.mktemp("req") / "scan.request.bin"
    with path.open("xb") as fh:
        write_request(fh, sets, runnable_scans(sets), corpus=CORPUS,
                      export={"template_strip": strip}, checks=splitter_checks(report))
    return path.read_bytes()


def _run(binary: Path, tmp_path: Path, body: bytes, name: str) -> Path:
    request = tmp_path / f"{name}.request.bin"
    request.write_bytes(body)
    out = tmp_path / name
    run_containment(binary, request, out, threads=2, timeout_s=120.0)
    return out


def test_the_whole_request_scans_and_writes_an_attestation(
    qd_prep_bin: Path, tmp_path: Path, request_bytes: bytes,
) -> None:
    out = _run(qd_prep_bin, tmp_path, request_bytes, "whole")
    assert (out / ATTESTATION_NAME).is_file()


@pytest.mark.parametrize("cut", [0, 1, 8, 0.25, 0.5, 0.9, -1])
def test_a_request_cut_short_is_refused_and_leaves_no_attestation(
    qd_prep_bin: Path, tmp_path: Path, request_bytes: bytes, cut: float,
) -> None:
    size = len(request_bytes)
    keep = int(size + cut) if cut == -1 else (int(cut) if cut >= 1 else int(size * cut))
    assert 0 <= keep < size
    with pytest.raises(SystemExit, match="containment exited"):
        _run(qd_prep_bin, tmp_path, request_bytes[:keep], f"cut{keep}")
    assert not (tmp_path / f"cut{keep}" / ATTESTATION_NAME).exists()


def test_a_rerun_into_an_existing_scan_or_request_is_refused_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "scan"
    out.mkdir()
    (out / "marker").write_text("a scan someone may need", encoding="utf-8")
    argv = ["--out-dir", str(out), "--rev", "HEAD", "--general-record", str(tmp_path / "none")]
    monkeypatch.setattr(containment_scan, "prep_binary", lambda: tmp_path / "qd-prep")
    with pytest.raises(SystemExit, match="exists; refusing to overwrite"):
        containment_scan.main(argv)
    assert [p.name for p in out.iterdir()] == ["marker"]
    fresh = tmp_path / "fresh"
    (tmp_path / "fresh.request.bin").write_bytes(b"half a request")
    with pytest.raises(SystemExit, match="exists; refusing to overwrite"):
        containment_scan.main(["--out-dir", str(fresh), "--rev", "HEAD",
                               "--general-record", str(tmp_path / "none")])
    assert (tmp_path / "fresh.request.bin").read_bytes() == b"half a request"
    assert not fresh.exists()
