"""``qd-prep containment`` against its parity oracle, ``qd_train.replay.decontaminate``.

Fable's ruling of 2026-10-02 (Q2, item 3): the v5 decontamination scan is Rust, and its
oracle is ``decontaminate`` on a committed fixture, with a bit-identical COMPLETE pair list
-- every (source row, target row) at or over the threshold, not the best target per row.
The fixture (``crates/qd-prep/tests/fixtures/containment-parity.jsonl``) is synthetic text
with a Unicode torture vocabulary (Turkish dotted capitals, Greek final sigma, combining
marks, non-Latin digits, fullwidth and circled letters), rows too short for an 8-gram on
every side, containment exactly at 0.5, and a source row that hits several target sets.

The oracle's word definition is Python's (``re`` ``\\w`` over ``str.lower``); the binary
carries tables generated from CPython (``tools/qd_prep_unicode_tables.py``), pinned to one
Unicode version. ``test_the_unicode_tables_are_this_pythons`` regenerates them and compares,
on a Python whose ``unicodedata`` is that version.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from containment_scan import (  # noqa: E402
    ScanSet,
    ScanSpec,
    run_containment,
    write_request,
)

from qd_train.exclusions import ATTESTATION_NAME, EXCLUSIONS_NAME  # noqa: E402
from qd_train.replay import decontaminate, target_digest  # noqa: E402

FIXTURE = REPO / "crates" / "qd-prep" / "tests" / "fixtures" / "containment-parity.jsonl"
PAIRS_NAME = "pairs.tsv"
PAIRS_HEADER = (
    "source_set\tsource_key\tsource_identity_key\tsource_family\t"
    "target_set\ttarget_key\ttarget_family\tshared\ttarget_ngrams\n"
)
#: The Unicode version ``crates/qd-prep/src/pyunicode_tables.rs`` was generated under.
TABLES_UNICODE = "16.0.0"
CHECKS = [("identity_disjoint", 2, "fixture"), ("near_duplicate_disjoint", 2, "fixture"),
          ("repo_disjoint", 2, "fixture")]


def _fixture() -> tuple[list[ScanSet], list[ScanSpec]]:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    rows = [json.loads(line) for line in lines[1:]]
    assert header["rows"] == len(rows)
    sets = [
        ScanSet(name=name, unrenderable={}, rows=tuple(
            (r["key"], r["identity_key"], r["family"], r["text"])
            for r in rows if r["set"] == name
        ))
        for name in header["sets"]
    ]
    scans = [ScanSpec(source, target, enforced) for source, target, enforced in header["scans"]]
    return sets, scans


def _scan(binary: Path, tmp: Path, sets, scans, *, checks=CHECKS, threads: int | None = 3,
          name: str = "out") -> Path:
    request = tmp / f"{name}.request.bin"
    with request.open("xb") as fh:
        write_request(fh, sets, scans, corpus={"fixture": "containment-parity"}, export={},
                      checks=checks)
    out = tmp / name
    run_containment(binary, request, out, threads=threads, timeout_s=120.0)
    return out


def _oracle(sets: list[ScanSet], scans: list[ScanSpec]) -> tuple[str, dict[str, dict]]:
    """The oracle's complete pair list as ``pairs.tsv`` would hold it, and its target counts:
    one ``decontaminate`` call per source set, in the order the scans first name them, with
    that source's targets in scan order."""
    by_name = {s.name: s for s in sets}
    rows = {s.name: {key: (ident, fam, text) for key, ident, fam, text in s.rows} for s in sets}
    sources: list[str] = []
    for scan in scans:
        if scan.source not in sources:
            sources.append(scan.source)
    lines = [PAIRS_HEADER]
    counted: dict[str, dict] = {}
    for source in sources:
        targets = {
            scan.target: {k: v[2] for k, v in rows[scan.target].items()}
            for scan in scans if scan.source == source
        }
        report = decontaminate({k: v[2] for k, v in rows[source].items()}, targets)
        counted.update(report.targets)
        for p in report.pairs:
            ident, fam, _ = rows[source][p.replay_row]
            target_fam = rows[p.target][p.target_row][1]
            lines.append(
                f"{source}\t{p.replay_row}\t{ident}\t{fam}\t{p.target}\t{p.target_row}\t"
                f"{target_fam}\t{p.shared}\t{p.target_ngrams}\n"
            )
    assert set(counted) <= set(by_name)
    return "".join(lines), counted


def test_the_complete_pair_list_is_the_oracles_bit_for_bit(qd_prep_bin: Path,
                                                            tmp_path: Path) -> None:
    sets, scans = _fixture()
    want, _ = _oracle(sets, scans)
    out = _scan(qd_prep_bin, tmp_path, sets, scans)
    got = (out / PAIRS_NAME).read_text(encoding="utf-8")
    assert got == want
    # Not vacuous: pairs in both source sets, at exactly the threshold, over several targets.
    body = [line.split("\t") for line in want.splitlines()[1:]]
    assert {b[0] for b in body} == {"train", "val"}
    assert {b[4] for b in body} == {"val", "heldout", "heldout-family:qa.answerability"}
    assert any(int(b[7]) * 2 == int(b[8]) for b in body), "no pair at exactly 0.5"
    assert len(body) >= 30


def test_the_attestation_counts_what_the_oracle_counts(qd_prep_bin: Path,
                                                       tmp_path: Path) -> None:
    sets, scans = _fixture()
    pairs, counted = _oracle(sets, scans)
    out = _scan(qd_prep_bin, tmp_path, sets, scans)
    att = json.loads((out / ATTESTATION_NAME).read_text(encoding="utf-8"))
    assert att["version"] == 2 and att["tool"] == "qd-prep containment"
    assert (att["n"], att["threshold"]) == (8, 0.5)
    assert att["unicode_version"] == TABLES_UNICODE
    for name, oracle in counted.items():
        mine = att["sets"][name]
        assert (mine["rows"], mine["indexed"], mine["too_short"], mine["digest"]) == (
            oracle["rows"], oracle["rows_indexed"], oracle["rows_too_short"], oracle["digest"]
        ), name
    by_name = {s.name: s for s in sets}
    train = {key: text for key, _i, _f, text in by_name["train"].rows}
    assert att["sets"]["train"]["digest"] == target_digest(train)
    # exclusions.txt: the enforced scans' source identity keys, byte-sorted, unique, LF.
    enforced = {(s.source, s.target) for s in scans if s.enforced}
    keys = sorted(
        {line.split("\t")[2] for line in pairs.splitlines()[1:]
         if tuple(line.split("\t")[i] for i in (0, 4)) in enforced},
        key=lambda k: k.encode(),
    )
    raw = (out / EXCLUSIONS_NAME).read_bytes()
    assert raw == "".join(f"{k}\n" for k in keys).encode()
    assert att["exclusions_sha256"] == hashlib.sha256(raw).hexdigest()
    assert att["n_exclusions"] == len(keys) > 0
    assert att["pairs_sha256"] == hashlib.sha256(pairs.encode()).hexdigest()
    assert att["n_pairs"] == len(pairs.splitlines()) - 1
    assert att["excluded_from"] == "train"
    assert att["enforced_targets"] == ["val", "heldout"]
    assert att["unenforced_targets"] == ["heldout-family:qa.answerability"]
    assert att["report_only_scans"] == [
        {"source": "val", "target": "heldout"},
        {"source": "val", "target": "heldout-family:qa.answerability"},
    ]
    assert att["remaining_hits"] == {"val": 0, "heldout": 0}
    assert att["clean"] is True and att["not_clean_because"] == []
    hits = {(h["target"], h["source_family"]): h["source_rows_hit"]
            for h in att["hits_by_source_family"] if h["source"] == "train"}
    assert sum(v for (t, _f), v in hits.items() if t == "val") == len(
        {line.split("\t")[1] for line in pairs.splitlines()[1:]
         if line.startswith("train\t") and line.split("\t")[4] == "val"}
    )


def test_the_output_does_not_depend_on_the_thread_count(qd_prep_bin: Path,
                                                        tmp_path: Path) -> None:
    sets, scans = _fixture()
    one = _scan(qd_prep_bin, tmp_path, sets, scans, threads=1, name="one")
    many = _scan(qd_prep_bin, tmp_path, sets, scans, threads=7, name="seven")
    for name in (PAIRS_NAME, EXCLUSIONS_NAME, ATTESTATION_NAME):
        assert (one / name).read_bytes() == (many / name).read_bytes(), name


@pytest.mark.parametrize(("checks", "because"), [
    ([("identity_disjoint", 1, "two splits share a key")], "failed"),
    ([("near_duplicate_disjoint", 0, "candidate bound hit")], "did not run"),
    ([], "no splitter check"),
])
def test_a_splitter_check_that_failed_or_did_not_run_is_not_clean(
    qd_prep_bin: Path, tmp_path: Path, checks, because: str,
) -> None:
    sets, scans = _fixture()
    out = _scan(qd_prep_bin, tmp_path, sets, scans, checks=checks)
    att = json.loads((out / ATTESTATION_NAME).read_text(encoding="utf-8"))
    assert att["clean"] is False
    assert any(because in reason for reason in att["not_clean_because"]), att


def test_an_existing_out_dir_and_a_text_rust_cannot_hold_are_refused(
    qd_prep_bin: Path, tmp_path: Path,
) -> None:
    sets, scans = _fixture()
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        _scan(qd_prep_bin, tmp_path, sets, scans)
    bad = [ScanSet("train", (("k", "i", "f", "lone \ud800 surrogate"),), {}), *sets[1:]]
    with pytest.raises(SystemExit, match="not encodable as UTF-8"), \
            (tmp_path / "bad.bin").open("xb") as fh:
        write_request(fh, bad, scans, corpus={}, export={}, checks=CHECKS)


# --- benchmark --------------------------------------------------------------------------

#: Rounds of each A/B; each round runs the oracle, then the binary.
BENCH_ROUNDS = int(os.environ.get("QD_PREP_BENCH_ROUNDS", "5"))


def _tiled(sets: list[ScanSet], copies: int) -> list[ScanSet]:
    """Every set ``copies`` times over, each copy's keys suffixed and every word of its text
    suffixed too, so copy ``c`` shares no n-gram with copy ``d``: the work and the pair list
    grow linearly with ``copies``. A suffix can change where a torture word splits, so a copy
    is a variant of the fixture's pattern, not its exact repeat; both arms see the same one."""
    def text_of(text: str, c: int) -> str:
        return text if c == 0 else re.sub(r"(\w+)", lambda m: f"{m.group(1)}x{c}", text)

    return [
        ScanSet(name=s.name, unrenderable={}, rows=tuple(
            (f"{key}~{c:04d}", f"{ident}~{c:04d}", fam, text_of(text, c))
            for c in range(copies) for key, ident, fam, text in s.rows
        ))
        for s in sets
    ]


@pytest.mark.skipif(
    os.environ.get("QD_PREP_BENCH") != "1",
    reason="the containment benchmark runs only with QD_PREP_BENCH=1",
)
@pytest.mark.parametrize("copies", [1, 20, 100, 500])
def test_benchmark_containment_against_the_oracle_interleaved_min_of_n(
    qd_prep_bin: Path, tmp_path: Path, copies: int,
) -> None:
    """The committed A/B behind the speed claim: the oracle (one ``decontaminate`` per source
    set, the pair list formatted as ``pairs.tsv``) against the binary (request written,
    ``qd-prep containment`` run, ``pairs.tsv`` read), alternating, min of N, the outputs
    compared byte for byte after every pair, outside the timers.

        QD_PREP_BENCH=1 pytest -s python/tests/test_qd_prep_containment_parity.py -k benchmark
    """
    sets, scans = _fixture()
    sets = _tiled(sets, copies)
    oracle_s: list[float] = []
    binary_s: list[float] = []
    for i in range(BENCH_ROUNDS):
        started = time.perf_counter()
        want, _ = _oracle(sets, scans)
        oracle_s.append(time.perf_counter() - started)
        started = time.perf_counter()
        out = _scan(qd_prep_bin, tmp_path, sets, scans, threads=None, name=f"round{i}")
        got = (out / PAIRS_NAME).read_text(encoding="utf-8")
        binary_s.append(time.perf_counter() - started)
        assert got == want, f"round {i}: the arms disagree"
    rows = sum(len(s.rows) for s in sets)
    print(json.dumps({
        "benchmark": "containment", "fixture": FIXTURE.name, "copies": copies, "rows": rows,
        "text_bytes": sum(len(t.encode()) for s in sets for *_, t in s.rows),
        "pairs": len(want.splitlines()) - 1, "rounds": BENCH_ROUNDS,
        "host_cpus": os.cpu_count(),
        "oracle_s": [round(x, 4) for x in oracle_s], "qd_prep_s": [round(x, 4) for x in binary_s],
        "oracle_min_s": round(min(oracle_s), 4), "qd_prep_min_s": round(min(binary_s), 4),
        "speedup_min_over_min": round(min(oracle_s) / min(binary_s), 1),
        "checked": "pairs.tsv byte-identical every round",
    }))


def test_the_unicode_tables_are_this_pythons() -> None:
    """The generator's ``--check``: the committed tables are what this CPython's ``re`` and
    ``str.lower`` say, code point by code point. Only meaningful on the version they pin."""
    if unicodedata.unidata_version != TABLES_UNICODE:
        pytest.skip(
            f"this Python's unicodedata is {unicodedata.unidata_version}; the tables pin "
            f"{TABLES_UNICODE}, so a comparison here would test the version, not the tables"
        )
    done = subprocess.run(
        [sys.executable, str(REPO / "tools" / "qd_prep_unicode_tables.py"), "--check",
         "--out", str(REPO / "crates" / "qd-prep" / "src" / "pyunicode_tables.rs")],
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
