"""Export Fable's v5 decontamination request and run ``qd-prep containment`` on it.

Fable's ruling of 2026-10-02 (``AUDIT/fable-optimize-2026-10-02/fable-optimize-ruling.md``,
Q2 "The rule, end to end"). The scan is Rust (``crates/qd-prep/src/containment.rs``); this
tool only rebuilds the corpus the way ``real_ft_run.ft_splits`` does -- through
``real_ft_run.ft_split_report``, the split before any exclusion or replay draw -- renders
every row as ``tools/replay_decontam.py`` renders its targets (``render`` at ``seed=None``,
cut where ``qd_train.replay.prompt_content`` cuts), strips the constant template text
(``qd_train.containment_strip.strip_template``, v5's rule at ``STRIP_VERSION`` 2: the question
line, family-constant options, and every option value of the four intent.* families; the strip,
what it removed and the per-set ``key_ii_blind`` keys are in the attestation's
``export.template_strip``), and hands the binary one request:

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
  clean only when every one ran and passed;
* external target sets (v6, data-clean plan section 3 item 6), each ``target:<name>``:
  ``--target NAME=FILE`` takes a ``{"id", "text"}`` JSONL set in the format ``qd-prep decisions
  --target`` reads (the jevjudge pairs -- RM-Bench, RewardBench 2, JudgeBench -- are one such
  set, so the code families are scanned against them too, not only the decision pool), and
  ``--v6-benchmark-targets`` builds under ``DataConfig.with_v6_benchmark_targets()`` and adds
  MMLU test and dev and CLINC test (``qd_data.sources.BENCHMARK_TARGET_SPLITS``), read from the
  sha-checked ``--general-record`` caches. ``train`` against each is enforced; ``val`` against
  each is report-only. Target texts are not template-stripped: they are external text, not
  rendered rows. Without any target the request is the bytes it was.

The binary writes ``OUT_DIR/{pairs.tsv, exclusions.txt, attestation.json}``; the pipeline's
and the trainer's ``--exclude-identity-keys OUT_DIR/exclusions.txt`` read them through
``qd_train.exclusions``. Rows ``render`` refuses are in no shard set, so they are neither
compared nor excluded; the attestation's ``export`` block counts them.

Usage (the corpus flags exactly as the pipeline was given them)::

    QD_PREP_BIN=/abs/qd-prep python tools/containment_scan.py --out-dir DIR \\
        --rev REV [--max-pairs N] [--no-repo-history] [--commitpackft DIR] \\
        [--defect-class DIR --defect-download DIR --defect-max-rows N --defect-noul DIR] \\
        [--general-record FILE --general-max-rows N] [--decisions-pool DIR] \\
        [--request-out FILE] [--threads N] \\
        [--target NAME=FILE ...] [--v6-benchmark-targets] \\
        [--no-template-strip]

``--no-template-strip`` is for measurement only (the pre-strip definition, to report rates
beside the stripped ones); ``qd_train.exclusions`` refuses a list made with it.
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
from repo_git import resolve_rev

from qd_data.config import DataConfig
from qd_data.errors import QdRefusal
from qd_data.loaders import MmluRow
from qd_data.render import render
from qd_data.rows import DataRow
from qd_data.split import HELD_OUT, SplitReport
from qd_train.containment_strip import PartsRow, slot_parts, strip_template
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
#: An external target set's name in the request: ``target:<name>``. Its rows' family is the set
#: name too, so the same-family scope (``decisions_pool_same_family_not_enforced``) never
#: matches a target pair: every pair against a target set is enforced.
TARGET_PREFIX: Final[str] = "target:"
#: A target name: what ``qd-prep decisions --target`` accepts as a config key, kept to a
#: conservative alphabet because it is a set name, a key prefix and an attestation field.
_TARGET_NAME_CHARS: Final[frozenset[str]] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
)
#: One target line; the longest jevjudge pair is well under this.
MAX_TARGET_LINE_BYTES: Final[int] = 8 << 20


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


@dataclass(frozen=True)
class TargetSet:
    """An external decontamination target: ``(id, text)`` rows and where they came from."""

    name: str
    rows: tuple[tuple[str, str], ...]
    #: sha256 of the bytes read (a file) or of the record entry's cache (a benchmark split).
    sha256: str
    source: str
    #: Items whose text could not be built through the row funnel and were taken as their raw
    #: question and options instead (:func:`mmlu_target_text`); 0 for a target file.
    raw_fallback: int = 0


def mmlu_target_text(row: MmluRow) -> tuple[str, bool]:
    """``(text, fell back)``: what this MMLU item scans as when it is a train row -- built
    through the row funnel (``rewrite_mmlu`` under v5's config, so a test-split item is not
    refused; ``render`` at ``seed=None``; ``slot_parts``), less the question line the template
    strip removes: the question, then each option value, one per line. An exact copy of the
    item in train therefore contains every one of its 8-grams. An item the funnel refuses (it
    was never a row, e.g. two equal options) is taken as its stripped question and options,
    and counted."""
    from qd_data.general import MMLU_FAMILY, rewrite_mmlu

    try:
        built = rewrite_mmlu(row, family_id=MMLU_FAMILY, index=0, config=DataConfig())
        rendered = render(built.request, seed=None)
    except QdRefusal:
        return "\n".join((row.question.strip(), *(c.strip() for c in row.choices))), True
    (slot,) = rendered.slots
    return slot_parts(rendered.prompt_for(slot.name)).joined(question=False), False


def _target_name(name: str) -> str:
    if not name or len(name) > 64 or not set(name) <= _TARGET_NAME_CHARS:
        raise SystemExit(
            f"target name {name!r}: 1-64 characters of [A-Za-z0-9._-]; it names a set and "
            "prefixes every key in it"
        )
    return name


def _target_rows(
    name: str, lines: Iterable[tuple[int, bytes]], where: str,
) -> list[tuple[str, str]]:
    """``(id, text)`` per non-blank JSONL line, each an object with string ``id`` and ``text``;
    ids distinct. Refused, never skipped: a target line that is not one would be a target
    nobody scanned against."""
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for lineno, line in lines:
        if not line.strip():
            continue
        if len(line) > MAX_TARGET_LINE_BYTES:
            raise SystemExit(f"{where}:{lineno}: a line over {MAX_TARGET_LINE_BYTES} bytes")
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SystemExit(f"{where}:{lineno}: not JSON ({exc})") from exc
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), str) \
                or not isinstance(obj.get("text"), str):
            raise SystemExit(
                f"{where}:{lineno}: a target line is an object with string id and text"
            )
        if obj["id"] in seen:
            raise SystemExit(f"{where}:{lineno}: target id {obj['id']!r} repeats")
        seen.add(obj["id"])
        rows.append((obj["id"], obj["text"]))
        if len(rows) > MAX_ROWS_PER_SET:
            raise SystemExit(f"{where}: more than {MAX_ROWS_PER_SET} target rows")
    if not rows:
        raise SystemExit(f"{where}: target set {name!r} has no rows; a scan against nothing is "
                         "not a check")
    return rows


def read_target_file(name: str, path: Path) -> TargetSet:
    """A ``{"id", "text"}`` JSONL target set, as ``qd-prep decisions --target`` reads one, with
    the sha256 of the bytes it was read from (recorded in the attestation's ``export``)."""
    name = _target_name(name)
    raw = path.read_bytes()
    lines = enumerate(raw.splitlines(keepends=True), start=1)
    rows = _target_rows(name, lines, str(path))
    return TargetSet(name=name, rows=tuple(rows), sha256=hashlib.sha256(raw).hexdigest(),
                     source=str(path))


def benchmark_target_sets(record: Path) -> list[TargetSet]:
    """The v6 benchmark targets (``qd_data.sources.BENCHMARK_TARGET_SPLITS``) from a general
    fetch record: one set per target split, named ``<prefix>-<split>`` (``mmlu-test``,
    ``mmlu-dev``, ``clinc-test``). Each cache must sit under the record's directory and hash to
    the record's ``jsonl_sha256``, as ``real_tokenizer_pipeline.general_rows`` requires of the
    rows it reads. The text is what the item scans as when it is a row: MMLU's through
    :func:`mmlu_target_text`; CLINC's utterance, which is all a stripped intent row keeps."""
    from real_tokenizer_pipeline import fetch_record_entries

    from qd_data.defect_class import sha256_file
    from qd_data.loaders import parse_mmlu
    from qd_data.sources import BENCHMARK_TARGET_PREFIX, BENCHMARK_TARGET_SPLITS

    _raw, entries = fetch_record_entries(record)
    root = record.parent.resolve()
    out: list[TargetSet] = []
    for entry in entries:
        dataset = str(entry["dataset"])
        path = Path(str(entry["jsonl"])).resolve()
        split_name = path.stem
        if split_name not in BENCHMARK_TARGET_SPLITS.get(dataset, ()):
            continue
        if not path.is_relative_to(root):
            raise SystemExit(f"{record}: {path} is outside the cache root {root}")
        found = sha256_file(path)
        if found != entry["jsonl_sha256"]:
            raise SystemExit(f"{path}: sha256 {found} but the fetch record says "
                             f"{entry['jsonl_sha256']}; the cache is not the approved download")
        name = f"{BENCHMARK_TARGET_PREFIX[dataset]}-{split_name}"
        texts: list[str] = []
        fallback = 0
        with path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if dataset == "cais/mmlu":
                    text, fell_back = mmlu_target_text(
                        parse_mmlu(obj, index=i, split_name=split_name)
                    )
                    texts.append(text)
                    fallback += fell_back
                else:
                    text = obj.get("text") if isinstance(obj, dict) else None
                    if not isinstance(text, str) or not text.strip():
                        raise SystemExit(f"{path}:{i + 1}: a CLINC row with no text")
                    texts.append(text)
        if len(texts) != int(entry["rows"]):
            raise SystemExit(f"{path}: {len(texts)} rows but the fetch record says {entry['rows']}")
        if any(t.name == name for t in out):
            raise SystemExit(f"{record}: target {name} named twice")
        out.append(TargetSet(
            name=name, rows=tuple((f"{split_name}-{i}", t) for i, t in enumerate(texts)),
            sha256=found, source=f"{dataset} {split_name} ({path})", raw_fallback=fallback,
        ))
    want = sorted(f"{BENCHMARK_TARGET_PREFIX[d]}-{s}" for d, ss in BENCHMARK_TARGET_SPLITS.items()
                  for s in ss)
    if sorted(t.name for t in out) != want:
        raise SystemExit(f"{record} supplies targets {sorted(t.name for t in out)}, not {want}")
    return out


def _rendered(rows: Iterable[DataRow]) -> tuple[list[PartsRow], dict[str, str]]:
    """``(parts rows, {row_id: refusal})``: every slot of every row rendered at ``seed=None``
    and cut by ``containment_strip.slot_parts`` (which checks its cut against
    ``prompt_content``), keyed ``row_id#slot`` as ``replay_decontam.row_texts`` keys it. A
    row ``render`` refuses is counted and skipped, as ``row_texts`` does: ``write_shards``
    refuses it too, so it is in no shard set."""
    out: list[PartsRow] = []
    refused: dict[str, str] = {}
    for row in rows:
        try:
            rendered = render(row.request, seed=None)
        except QdRefusal as exc:
            refused[row.row_id] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        for slot in rendered.slots:
            out.append((f"{row.row_id}#{slot.name}", row.identity_key, row.family_id,
                        slot.name, slot_parts(rendered.prompt_for(slot.name))))
    return out, refused


def check_benchmark_sources_built(report: SplitReport, record: Path) -> None:
    """Under ``--v6-benchmark-targets``, refuse a corpus in which a ``BENCHMARK_TARGET_SPLITS``
    source the record supplies built no row at all. v6 refuses a CLINC row whose upstream split
    is unstated, and ``real_tokenizer_pipeline.general_rows`` does not yet pass ``split_name`` to
    ``parse_clinc``, so until it does every CLINC row is refused and the scan would attest a
    corpus with no CLINC val or held-out set -- clean only because nothing was there."""
    from real_tokenizer_pipeline import fetch_record_entries

    from qd_data.sources import BENCHMARK_TARGET_SPLITS

    _raw, entries = fetch_record_entries(record)
    supplied = {str(e["dataset"]) for e in entries} & set(BENCHMARK_TARGET_SPLITS)
    built = {r.source_id for rows in report.rows_by_split.values() for r in rows}
    missing = sorted(supplied - built)
    if missing:
        raise SystemExit(
            f"--v6-benchmark-targets: {missing} supplied by {record} built no row. For "
            "clinc/clinc_oos this is every row refused upstream_split_unstated: "
            "real_tokenizer_pipeline.general_rows must call parse_clinc(..., "
            "split_name=split_name) before a v6 build can read CLINC"
        )


def scan_sets(
    report: SplitReport, *, config: DataConfig, template_strip: bool = True,
    targets: Sequence[TargetSet] = (),
) -> tuple[list[ScanSet], dict[str, object]]:
    """``(sets, strip record)``: train, val, the repo-disjoint held-out split and one set per
    task-holdout family, their texts stripped of constant template text by
    ``strip_template`` over the union of all of them (``template_strip=False`` only to measure
    the pre-strip definition), then one ``target:<name>`` set per external target, last and
    unstripped. The record is the attestation's ``export.template_strip``."""
    by_split = report.rows_by_split
    heldout = by_split.get(HELD_OUT, ())
    # Each set's rows are materialised in its own iteration: a generator closing over the
    # loop's ``family`` and consumed after the loop would read the last family for all.
    chosen: list[tuple[str, tuple[DataRow, ...]]] = [
        (TRAIN, tuple(by_split.get(TRAIN, ()))),
        (VAL, tuple(by_split.get(VAL, ()))),
        (HELD_OUT, tuple(r for r in heldout if not config.is_held_out_family(r.family_id))),
    ]
    for family in config.held_out_families:
        chosen.append((f"{HELDOUT_FAMILY_PREFIX}{family}",
                       tuple(r for r in heldout if r.family_id == family)))
    parts: dict[str, list[PartsRow]] = {}
    refused: dict[str, dict[str, str]] = {}
    for name, rows in chosen:
        if name in parts:
            raise SystemExit(f"set {name!r} named twice")
        parts[name], refused[name] = _rendered(rows)
    texts, record = strip_template(parts, apply=template_strip)
    del parts
    sets = [
        ScanSet(name=name, rows=tuple(texts.pop(name)), unrenderable=refused[name])
        for name, _rows in chosen
    ]
    for t in targets:
        name = f"{TARGET_PREFIX}{_target_name(t.name)}"
        if any(s.name == name for s in sets):
            raise SystemExit(f"target {t.name!r} named twice")
        sets.append(ScanSet(
            name=name,
            rows=tuple((f"{t.name}:{i}", f"{t.name}:{i}", name, text) for i, text in t.rows),
            unrenderable={},
        ))
    return sets, record


def scan_specs(sets: Sequence[ScanSet]) -> list[ScanSpec]:
    """The rule's scans over ``sets`` (as :func:`scan_sets` names them). External targets come
    after the rule's own scans, so a request without one is the bytes it was."""
    families = [s.name for s in sets if s.name.startswith(HELDOUT_FAMILY_PREFIX)]
    targets = [s.name for s in sets if s.name.startswith(TARGET_PREFIX)]
    return [
        ScanSpec(TRAIN, VAL, True),
        ScanSpec(TRAIN, HELD_OUT, True),
        *(ScanSpec(TRAIN, f, False) for f in families),
        ScanSpec(VAL, HELD_OUT, False),
        *(ScanSpec(VAL, f, False) for f in families),
        *(ScanSpec(TRAIN, t, True) for t in targets),
        *(ScanSpec(VAL, t, False) for t in targets),
    ]


def targets_block(targets: Sequence[TargetSet]) -> dict[str, object]:
    """The attestation's ``export.targets``: per set, its row count, sha256 and source."""
    return {
        f"{TARGET_PREFIX}{t.name}": {"rows": len(t.rows), "sha256": t.sha256, "source": t.source,
                                     "raw_fallback": t.raw_fallback}
        for t in targets
    }


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


def export_block(
    sets: Sequence[ScanSet], *, engine: Path, strip: Mapping[str, object],
    targets: Sequence[TargetSet] = (),
) -> dict[str, object]:
    """What the exporter left out, which engine scanned, what the template strip removed
    (``strip``, :func:`scan_sets`'s record) and, when there are any, the external targets
    (:func:`targets_block`): the attestation's ``export``."""
    extra: dict[str, object] = {"targets": targets_block(targets)} if targets else {}
    return {
        **extra,
        "tool": TOOL,
        "engine_sha256": engine_sha256(engine),
        "render": (
            "qd_data.render.render(request, seed=None) -> qd_train.containment_strip.slot_parts "
            "(== qd_train.replay.prompt_content) -> strip_template"
        ),
        "template_strip": dict(strip),
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
    parser.add_argument(
        "--decisions-pool", type=Path, default=None,
        help="as the pipeline's --decisions-pool: the pool's rows are scanned with every "
             "other row, and the corpus identity names the pool",
    )
    parser.add_argument(
        "--drop-before-dedupe", type=Path, default=None, dest="pre_dedupe_drops",
        help="as the pipeline's --drop-before-dedupe: the listed train rows leave the corpus "
             "before dedupe here too, and the corpus identity names the list by its sha256",
    )
    parser.add_argument(
        "--target", action="append", default=[], metavar="NAME=FILE",
        help="an external {\"id\", \"text\"} JSONL target set (repeat per set), scanned as "
             "target:NAME: train against it enforced, val against it report-only. The jevjudge "
             "pairs that qd-prep decisions --target jevjudge=FILE reads are one",
    )
    parser.add_argument(
        "--v6-benchmark-targets", action="store_true",
        help="build under DataConfig.with_v6_benchmark_targets() (MMLU test/dev and CLINC test "
             "are not rows) and scan against those splits, read from --general-record, as "
             "targets mmlu-test, mmlu-dev and clinc-test; the corpus names the opt-in",
    )
    parser.add_argument("--no-repo-history", dest="repo_history", action="store_false")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--no-template-strip", dest="template_strip", action="store_false",
        help="measurement only: scan the unstripped prompt_content (the pre-v5 definition); "
             "the attestation records applied=false and no build accepts its list",
    )
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
    # Every target is read and checked before the rebuild, so a bad one costs nothing.
    targets: list[TargetSet] = []
    for spec in args.target:
        name, sep, file = spec.partition("=")
        if not sep or not file:
            raise SystemExit(f"--target {spec!r}: expected NAME=FILE")
        targets.append(read_target_file(name, Path(file)))
    if args.v6_benchmark_targets:
        if args.general_record is None:
            raise SystemExit("--v6-benchmark-targets reads the benchmark splits from "
                             "--general-record; without it there is nothing to scan against")
        targets.extend(benchmark_target_sets(args.general_record))

    from real_ft_run import ft_split_report, replay_corpus_identity

    config = DataConfig()
    if args.v6_benchmark_targets:
        config = config.with_v6_benchmark_targets()
    rev = resolve_rev(REPO, args.rev)
    started = time.monotonic()
    report = ft_split_report(
        commitpackft=args.commitpackft, max_pairs=args.max_pairs, rev=rev, config=config,
        defect_class=args.defect_class, defect_download=args.defect_download,
        defect_max_rows=args.defect_max_rows, repo_history=args.repo_history,
        general_record=args.general_record, general_max_rows=args.general_max_rows,
        defect_noul=args.defect_noul, decisions_pool=args.decisions_pool,
        pre_dedupe_drops=args.pre_dedupe_drops,
    )
    built = time.monotonic()
    if args.v6_benchmark_targets:
        check_benchmark_sources_built(report, args.general_record)
    sets, strip = scan_sets(report, config=config, template_strip=args.template_strip,
                            targets=targets)
    rendered = time.monotonic()
    scans = scan_specs(sets)
    # The name ft_splits and the pipeline check an exclusion list against.
    corpus = containment_corpus(replay_corpus_identity(
        rev=rev, max_pairs=args.max_pairs, commitpackft=args.commitpackft,
        defect_class=args.defect_class, defect_max_rows=args.defect_max_rows,
        repo_history=args.repo_history, general_record=args.general_record,
        general_max_rows=args.general_max_rows, defect_noul=args.defect_noul,
        decisions_pool=args.decisions_pool, drop_before_dedupe=args.pre_dedupe_drops,
    ))
    if config.benchmark_eval_splits_are_targets:
        # A v6 corpus holds other rows than the v5 corpus of the same inputs, so it is named
        # apart: a build that does not name the opt-in refuses this list (read_exclusions).
        corpus = {**corpus, "benchmark_eval_splits_are_targets": True}
    with request.open("xb") as fh:
        size = write_request(
            fh, sets, scans, corpus=corpus,
            export=export_block(sets, engine=engine, strip=strip, targets=targets),
            checks=splitter_checks(report),
        )
    exported = time.monotonic()
    print(f"split {report.counts()} in {built - started:.1f} s; rendered "
          f"{ {s.name: len(s.rows) for s in sets} } slot text(s) in {rendered - built:.1f} s "
          f"(template strip version {strip['version']} "
          f"{'applied' if args.template_strip else 'NOT applied'}); "
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
