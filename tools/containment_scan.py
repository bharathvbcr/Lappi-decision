"""Export Fable's v5 decontamination request and run ``qd-prep containment`` on it.

Fable's ruling of 2026-10-02 (``AUDIT/fable-optimize-2026-10-02/fable-optimize-ruling.md``,
Q2 "The rule, end to end"). The scan is Rust (``crates/qd-prep/src/containment.rs``); this
tool only rebuilds the corpus the way ``real_ft_run.ft_splits`` does -- through
``real_ft_run.ft_split_report``, the split before any exclusion or replay draw -- renders
every row as ``tools/replay_decontam.py`` renders its targets (``row_texts``: ``render`` at
``seed=None``, then ``qd_train.replay.prompt_content``), and hands the binary one request:

* sources: every ``train`` row of every family, gold or replay-drawn (the replay draw has
  not happened yet), one text per slot, keyed ``row_id#slot``;
* enforced targets: every ``val`` row, and every ``heldout`` row of a trainable family (the
  repo-disjoint held-out split) -- a train row containing half of one's word 8-grams is
  excluded by identity key;
* unenforced targets: the two task-holdout families (``heldout-family:<family>``), which
  overlap train content by construction (GAP-PORTED-HELDOUT-FAMILY-CONTENT-OVERLAPS-TRAIN):
  reported per (source family, target), never excluded -- that is the human's call;
* report-only: ``val`` against every held-out target;
* keys (i) and (iii) are the splitter's own tri-states (``identity_disjoint``,
  ``near_duplicate_disjoint``, ``repo_disjoint``), carried into the attestation, which is
  clean only when every one ran and passed.

The binary writes ``OUT_DIR/{pairs.tsv, exclusions.txt, attestation.json}``; the pipeline's
and the trainer's ``--exclude-identity-keys OUT_DIR/exclusions.txt`` read them through
``qd_train.exclusions``. Rows ``render`` refuses are in no shard set, so they are neither
compared nor excluded; the attestation's ``export`` block counts them.

Usage (the corpus flags exactly as the pipeline was given them)::

    QD_PREP_BIN=/abs/qd-prep python tools/containment_scan.py --out-dir DIR \\
        --rev REV [--max-pairs N] [--no-repo-history] [--commitpackft DIR] \\
        [--defect-class DIR --defect-download DIR --defect-max-rows N --defect-noul DIR] \\
        [--general-record FILE --general-max-rows N] [--request-out FILE] [--threads N]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import struct
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linear_control_native import engine_sha256, prep_binary
from replay_decontam import row_texts
from repo_git import resolve_rev

from qd_data.config import DataConfig
from qd_data.rows import DataRow
from qd_data.split import HELD_OUT, SplitReport
from qd_train.exclusions import ATTESTATION_NAME, EXCLUSIONS_NAME, containment_corpus
from qd_train.replay import DEFAULT_N, DEFAULT_THRESHOLD
from qd_train.tristate import NotRun, Ran, TriState

REPO = Path(__file__).resolve().parents[1]
REQUEST_MAGIC: Final[bytes] = b"QDPCTIN1"
TOOL: Final[str] = "tools/containment_scan.py"
#: The sets, by name. The task-holdout families get one set each, so the attestation counts
#: their overlap per family and enforces none of it.
TRAIN: Final[str] = "train"
VAL: Final[str] = "val"
HELDOUT_FAMILY_PREFIX: Final[str] = "heldout-family:"
#: The splitter's post-conditions that are keys (i) and (iii) of the rule, plus the repo
#: disjointness both rest on.
SPLITTER_CHECKS: Final[tuple[str, ...]] = (
    "identity_disjoint", "near_duplicate_disjoint", "repo_disjoint",
)
#: Rows ``render`` refused that the export block names, per set; the count is always whole.
MAX_UNRENDERABLE_NAMED: Final[int] = 50
#: Seconds one ``qd-prep containment`` call may take. The scan is linear in the corpus's
#: n-grams; HANDOFF/prep-containment-2026-10-02.md estimates the full v4 corpus in minutes.
DEFAULT_TIMEOUT_S: Final[float] = 4 * 3600.0
#: ``crates/qd-prep/src/containment.rs``'s bounds, refused here before a byte is written.
MAX_STR_BYTES: Final[int] = 64 << 20
MAX_ROWS_PER_SET: Final[int] = 2_000_000


@dataclass(frozen=True)
class ScanSet:
    """One named set: ``(key, identity_key, family_id, text)`` per rendered slot."""

    name: str
    rows: tuple[tuple[str, str, str, str], ...]
    unrenderable: Mapping[str, str]


@dataclass(frozen=True)
class ScanSpec:
    source: str
    target: str
    enforced: bool


def _rendered(name: str, rows: Iterable[DataRow]) -> ScanSet:
    """``rows`` rendered one row at a time through ``replay_decontam.row_texts``, so every
    slot text is attributed to the row it came from without parsing ``row_id#slot`` back."""
    out: list[tuple[str, str, str, str]] = []
    refused: dict[str, str] = {}
    for row in rows:
        texts, unrenderable = row_texts([row])
        refused.update(unrenderable)
        for key, text in texts.items():
            out.append((key, row.identity_key, row.family_id, text))
    return ScanSet(name=name, rows=tuple(out), unrenderable=refused)


def scan_sets(report: SplitReport, *, config: DataConfig) -> list[ScanSet]:
    """train, val, the repo-disjoint held-out split and one set per task-holdout family."""
    by_split = report.rows_by_split
    heldout = by_split.get(HELD_OUT, ())
    sets = [
        _rendered(TRAIN, by_split.get(TRAIN, ())),
        _rendered(VAL, by_split.get(VAL, ())),
        _rendered(HELD_OUT, (r for r in heldout if not config.is_held_out_family(r.family_id))),
    ]
    for family in config.held_out_families:
        sets.append(_rendered(
            f"{HELDOUT_FAMILY_PREFIX}{family}", (r for r in heldout if r.family_id == family)
        ))
    return sets


def scan_specs(sets: Sequence[ScanSet]) -> list[ScanSpec]:
    """The rule's scans over ``sets`` (as :func:`scan_sets` names them)."""
    families = [s.name for s in sets if s.name.startswith(HELDOUT_FAMILY_PREFIX)]
    return [
        ScanSpec(TRAIN, VAL, True),
        ScanSpec(TRAIN, HELD_OUT, True),
        *(ScanSpec(TRAIN, f, False) for f in families),
        ScanSpec(VAL, HELD_OUT, False),
        *(ScanSpec(VAL, f, False) for f in families),
    ]


def splitter_checks(report: SplitReport) -> list[tuple[str, int, str]]:
    """``(name, state, detail)`` per :data:`SPLITTER_CHECKS`: 0 not run, 1 failed, 2 passed."""
    states: dict[str, TriState] = {
        "identity_disjoint": report.identity_disjoint,
        "near_duplicate_disjoint": report.near_duplicate_disjoint,
        "repo_disjoint": report.repo_disjoint,
    }
    out: list[tuple[str, int, str]] = []
    for name in SPLITTER_CHECKS:
        state = states[name]
        if isinstance(state, Ran):
            out.append((name, 2 if state.passed else 1, state.detail))
        elif isinstance(state, NotRun):
            out.append((name, 0, state.reason))
        else:
            raise SystemExit(f"split report's {name} is {state!r}, not a tri-state")
    return out


def _utf8(value: str, what: str) -> bytes:
    try:
        raw = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        # A lone surrogate: Rust's str cannot hold it, and dropping it would compare a text
        # the shard never held.
        raise SystemExit(f"{what} is not encodable as UTF-8 ({exc}); refusing to export it") \
            from exc
    if len(raw) > MAX_STR_BYTES:
        raise SystemExit(f"{what}: {len(raw)} bytes; qd-prep's bound is {MAX_STR_BYTES}")
    return raw


def _str(fh: BinaryIO, value: str, what: str) -> int:
    raw = _utf8(value, what)
    fh.write(struct.pack("<I", len(raw)))
    fh.write(raw)
    return 4 + len(raw)


def export_block(sets: Sequence[ScanSet], *, engine: Path) -> dict[str, object]:
    """What the exporter left out, and which engine scanned: the attestation's ``export``."""
    return {
        "tool": TOOL,
        "engine_sha256": engine_sha256(engine),
        "render": "qd_data.render.render(request, seed=None) -> qd_train.replay.prompt_content",
        "unrenderable": {
            s.name: {
                "n": len(s.unrenderable),
                "rows": dict(sorted(s.unrenderable.items())[:MAX_UNRENDERABLE_NAMED]),
            }
            for s in sets
        },
    }


def write_request(
    fh: BinaryIO, sets: Sequence[ScanSet], scans: Sequence[ScanSpec], *,
    corpus: Mapping[str, object], export: Mapping[str, object],
    checks: Sequence[tuple[str, int, str]], n: int = DEFAULT_N,
    threshold: float = DEFAULT_THRESHOLD,
) -> int:
    """The ``QDPCTIN1`` request, streamed to ``fh``; the bytes written."""
    names = [s.name for s in sets]
    if len(set(names)) != len(names):
        raise SystemExit(f"set names repeat: {names}")
    index = {name: i for i, name in enumerate(names)}
    for s in sets:
        if len(s.rows) > MAX_ROWS_PER_SET:
            raise SystemExit(f"set {s.name!r}: {len(s.rows)} rows; the bound is "
                             f"{MAX_ROWS_PER_SET}")
    fh.write(REQUEST_MAGIC)
    fh.write(struct.pack("<I", n))
    fh.write(struct.pack("<d", threshold))
    size = len(REQUEST_MAGIC) + 12
    size += _str(fh, json.dumps(dict(corpus), sort_keys=True), "corpus")
    size += _str(fh, json.dumps(dict(export), sort_keys=True), "export")
    fh.write(struct.pack("<I", len(checks)))
    size += 4
    for name, state, detail in checks:
        size += _str(fh, name, "check name")
        fh.write(bytes([state]))
        size += 1 + _str(fh, detail, f"check {name} detail")
    fh.write(struct.pack("<I", len(sets)))
    size += 4
    for name in names:
        size += _str(fh, name, "set name")
    fh.write(struct.pack("<I", len(scans)))
    size += 4
    for scan in scans:
        fh.write(struct.pack("<IIB", index[scan.source], index[scan.target], scan.enforced))
        size += 9
    fh.write(struct.pack("<Q", sum(len(s.rows) for s in sets)))
    size += 8
    for i, s in enumerate(sets):
        for key, identity, family, text in s.rows:
            fh.write(struct.pack("<I", i))
            size += 4
            size += _str(fh, key, "row key")
            size += _str(fh, identity, f"row {key} identity_key")
            size += _str(fh, family, f"row {key} family")
            size += _str(fh, text, f"row {key} text")
    return size


def run_containment(
    binary: Path, request: Path, out_dir: Path, *, threads: int | None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> str:
    """``qd-prep containment --input REQUEST --out-dir OUT_DIR``; its summary line."""
    if not timeout_s > 0:
        raise SystemExit(f"a qd-prep timeout of {timeout_s} s would never let it run")
    cmd = [str(binary), "containment", "--input", str(request), "--out-dir", str(out_dir)]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                              check=False)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"{binary} containment ran past {timeout_s:.0f} s and was killed") \
            from exc
    except OSError as exc:
        raise SystemExit(f"{binary} could not be run ({exc})") from exc
    if done.returncode != 0:
        raise SystemExit(
            f"{binary} containment exited {done.returncode}: {done.stderr.strip()[-2000:]}"
        )
    return done.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="the directory qd-prep creates; refused if it exists")
    parser.add_argument(
        "--request-out", type=Path, default=None,
        help="keep the request here (default: OUT_DIR.request.bin, removed after the scan)",
    )
    parser.add_argument("--rev", required=True)
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="as the pipeline's --max-pairs, with its default; refused under "
             "--no-repo-history without --commitpackft, where it bounds nothing",
    )
    parser.add_argument("--commitpackft", type=Path, default=None)
    parser.add_argument("--defect-class", type=Path, default=None)
    parser.add_argument("--defect-download", type=Path, default=None)
    parser.add_argument("--defect-max-rows", type=int, default=None)
    parser.add_argument("--defect-noul", type=Path, default=None)
    parser.add_argument("--general-record", type=Path, default=None)
    parser.add_argument("--general-max-rows", type=int, default=None)
    parser.add_argument("--no-repo-history", dest="repo_history", action="store_false")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)
    from real_tokenizer_pipeline import DEFAULT_MAX_PAIRS

    if not args.repo_history and args.commitpackft is None and args.max_pairs is not None:
        raise SystemExit(
            "--max-pairs bounds the repository-history rows and the --commitpackft sample; "
            "under --no-repo-history without --commitpackft it bounds nothing. Drop it."
        )
    if args.max_pairs is None:
        # The pipeline's default, which its corpus identity records either way.
        args.max_pairs = DEFAULT_MAX_PAIRS
    if args.out_dir.exists():
        raise SystemExit(f"{args.out_dir} exists; refusing to overwrite it")
    request = args.request_out or args.out_dir.with_name(args.out_dir.name + ".request.bin")
    if request.exists():
        raise SystemExit(f"{request} exists; refusing to overwrite it")
    engine = prep_binary()  # before the rebuild: a missing engine costs nothing to find

    from real_ft_run import ft_split_report, replay_corpus_identity

    config = DataConfig()
    rev = resolve_rev(REPO, args.rev)
    started = time.monotonic()
    report = ft_split_report(
        commitpackft=args.commitpackft, max_pairs=args.max_pairs, rev=rev, config=config,
        defect_class=args.defect_class, defect_download=args.defect_download,
        defect_max_rows=args.defect_max_rows, repo_history=args.repo_history,
        general_record=args.general_record, general_max_rows=args.general_max_rows,
        defect_noul=args.defect_noul,
    )
    built = time.monotonic()
    sets = scan_sets(report, config=config)
    rendered = time.monotonic()
    scans = scan_specs(sets)
    # The name ft_splits and the pipeline check an exclusion list against.
    corpus = containment_corpus(replay_corpus_identity(
        rev=rev, max_pairs=args.max_pairs, commitpackft=args.commitpackft,
        defect_class=args.defect_class, defect_max_rows=args.defect_max_rows,
        repo_history=args.repo_history, general_record=args.general_record,
        general_max_rows=args.general_max_rows, defect_noul=args.defect_noul,
    ))
    with request.open("xb") as fh:
        size = write_request(
            fh, sets, scans, corpus=corpus, export=export_block(sets, engine=engine),
            checks=splitter_checks(report),
        )
    exported = time.monotonic()
    print(f"split {report.counts()} in {built - started:.1f} s; rendered "
          f"{ {s.name: len(s.rows) for s in sets} } slot text(s) in {rendered - built:.1f} s; "
          f"request {size} bytes -> {request} in {exported - rendered:.1f} s")
    # The rows and their texts are the request's now: released before the binary loads its
    # own copy, so the two peaks do not stack.
    del report, sets
    gc.collect()
    summary = run_containment(engine, request, args.out_dir, threads=args.threads,
                              timeout_s=args.timeout_s)
    scanned = time.monotonic()
    if args.request_out is None:
        request.unlink()
    attestation = json.loads((args.out_dir / ATTESTATION_NAME).read_text(encoding="utf-8"))
    print(f"{summary} in {scanned - exported:.1f} s")
    print(f"exclusions: {args.out_dir / EXCLUSIONS_NAME} sha256 "
          f"{hashlib.sha256((args.out_dir / EXCLUSIONS_NAME).read_bytes()).hexdigest()}; "
          f"clean={attestation['clean']} remaining_hits={attestation['remaining_hits']}")
    return 0 if attestation["clean"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
