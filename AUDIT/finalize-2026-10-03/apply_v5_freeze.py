#!/usr/bin/env python3
"""Freeze the v5 pre-registration: fill amendments_pending in place and write the renamed file.

Lane L-v5-freeze, 2026-10-03. A one-shot AUDIT script (the language policy's role 3), standard
library only. It reads; the only file it writes is --out, and only without --dry-run.

What it does, under Fable's freeze rulings (2026-10-03 ~20:30Z, relayed by the lead):

1. Every item of the DRAFT's ``amendments_pending`` is filled IN PLACE: its text gains
   " Filled <UTC date>: <values, each with its source path or sha256>.", or "Pending after
   launch: <what fills it>.", or "Not filled: <reason>." No item is removed or reordered: tools
   read the list by index (tools/v5_a7_check.py and python/tests/test_v5_a7_check.py name
   amendments_pending[11]). Each item is found by a prefix of its text, which must match exactly
   one item at its expected index; any item no prefix claims refuses. The top-level ``draft``
   key is dropped. ``amendments_applied`` gains one provenance sentence, as every earlier
   amendment script's did. Everything else is byte-for-byte the DRAFT's (textual edits).
2. Item 24 can fail and is computed FIRST: code.defect_class's effective token share of the v5
   train shard set must be at least two-thirds of v4's. Both are computed by v4's own method
   (AUDIT/v5-plan-2026-10-02/v4_token_accounting.py: each sequence's real tokens from
   shards/train/offsets.npy, its category from its sequence_index.json row id), and the method is
   first re-run on v4's shard set and must reproduce v4_token_accounting.json exactly. Below the
   bound the script refuses, naming the number.
3. Cross-input checks bind the fill to one build: the train manifest's data_snapshot_hash, the
   build ledger row's and the train shard header's must agree; the scan's exclusions_sha256 must
   be the one the build applied (the scan directory is replaced by rescans, so this is what
   proves the attestation read is the build's); the decision pool's examples sha256 must agree
   across its manifest, its examples file, its report, the scan and the build row; A7 must have
   passed on this build.

Never read, here or in any input this script opens: a held-out entry (data/heldout/heldout.json is
read up to its "entries" key only, for its data_snapshot_hash), a containment request
(*.request.bin), pairs.tsv, or the identity-key and utterance lists inside a7.json and the
zero-checks output. Only counts, booleans and sha256s are copied out of those.

Usage at the freeze (after the v5 build, its A7 check and its attestation):

    python3 AUDIT/finalize-2026-10-03/apply_v5_freeze.py \\
        --draft campaign/v5-preregistered.DRAFT.json --repo . \\
        --build-out /Users/bharath/qd-campaign/phase4-v5-2026-10-03 \\
        --build-ledger ledger/mac-v5-shards-2026-10-03.jsonl \\
        --scan /Users/bharath/qd-campaign/v5-containment-v2-2026-10-03 \\
        --pool /Users/bharath/qd-campaign/v5-decisions-data-2026-10-03/pool-v5-decisions-v4 \\
        --pool-report /Users/bharath/qd-campaign/v5-decisions-data-2026-10-03/patches/POOL-RESULT-v4.md \\
        --fetch-record /Users/bharath/qd-campaign/v5-decisions-data-2026-10-03/fetch-record-decisions-v4-2026-10-03.json \\
        [--family-rates FAMILY_RATES.json] [--zero-checks ZERO_CHECKS.json] \\
        [--build-log BUILD.log] [--scan-log SCAN.log] [--gh200-state 'TEXT'] \\
        --out campaign/v5-preregistered.json [--dry-run]

--format-test (with --dry-run only) runs every parser against a build that is not v5 (v4's,
whose formats are the same): cross-input checks that cannot hold there are reported as "not
applicable (format test)" instead of refusing, and nothing is written.

Exit: 0 written (or printed, under --dry-run); 2 refused, nothing written.
"""

from __future__ import annotations

import argparse
import array
import ast
import bisect
import datetime
import hashlib
import json
import re
import struct
import sys
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

CHUNK = 1 << 20
HEAD_LIMIT = 512 * CHUNK
ANSWERS = "AUDIT/finalize-2026-10-03/human-answers-2026-10-03-v5-launch.md"
VERBATIM = "Yes to all, waive R9, approve ~$400"
ACCOUNTING = "AUDIT/v5-plan-2026-10-02/v4_token_accounting.json"
ACCOUNTING_SCRIPT = "AUDIT/v5-plan-2026-10-02/v4_token_accounting.py"
PASS_CHECK = "AUDIT/prep3-2026-10-02/pass-check-v2.json"
V5_COMMON = "campaign/post-f-queue/v5_common.sh"
V5_PRELUDE = "campaign/post-f-queue/v5_prelude_mac.py"
F_LEDGER = "ledger/gh200-p4-v4-2026-10-01.jsonl"
SCRIPT = "AUDIT/finalize-2026-10-03/apply_v5_freeze.py"
PRELUDE_GAP = "GAP-V5-FREEZE-PRELUDE-DIGESTS-CANNOT-LIVE-IN-THE-LANE-COMMIT-2026-10-03"
DEFECT = "code.defect_class"
CLINC_OOS_PREFIX = "clinc-oos:"
CLINC_FAMILIES = (
    "intent.classification",
    "intent.domain",
    "intent.in_scope",
    "intent.within_domain",
)
EXPECTED_OOS = {"train": 1216, "val": 72, "heldout": 62}
EXPECTED_VAL_CHOICE_SLOTS = 2304
POOL_BOUND = 12_500_000
POOL_TARGETS = ("arc-test", "boolq-val", "vitaminc-test")
QDM_DEFECT = f"qdm:{DEFECT}:"

# Each item's prefix and the index it must sit at. The DRAFT's own words, cut short enough that
# a later amendment appending to an item still matches, long enough that no two items share one.
ANCHORS: tuple[str, ...] = (
    "v5 build commit sha; train-manifest hash (data_snapshot_hash)",
    "the CLINC oos key spelling and its per-split counts",
    "attestation v2 sha256, exclusions_sha256",
    "defect-noul-v3c and own-prose-v1 examples sha256",
    "defect_class val slots written",
    "the rebuilt needle suite's digest",
    "R4 as decided, with its deciding row ids.",
    "v5's recipe hash and the Mac prelude's printed batches and width",
    "C2a's and C2b's `qd-post-f-rules tierb` words",
    "the census: 9-10k supply and the build's peak RSS",
    "containment runtime and exclusion count per source family",
    "(A7) per-family val and held-out counts",
    "own-prose-v1/files.jsonl sha256 and the human's ticked repo list",
    "the first seed's measured trajectory-scoring time",
    "the pin names V5_CONTINUE, V5NW_HUMAN_YES, v5nw.room",
    "the template strip's subsample re-run under STRIP_VERSION 2",
    "the G6 source: per-language rows",
    "C3's fsucc reading",
    "per-seed corpus.plan_order_digest as the Mac prelude printed them",
    "the batch_order key present on every v5",
    "the human's docs/promotion-decisions.json record",
    "contingency, pre-registered 2026-10-03",
    "the epoch arm (v5, J5' and the noul-weight arm) no longer runs",
    "C2a: off.",
    "data.sources: decision data.",
    "readings (report-only, rule 5 needs a row)",
    "calibration: the fit is reopened as a CPU lane",
    "the box's measured per-run hours",
    "the final v5 build's dedupe and split under POOL_MAX_CANDIDATE_PAIRS",
)


class Refusal(Exception):
    """Nothing is written."""


# --- small readers -------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(CHUNK), b""):
                h.update(block)
    except OSError as exc:
        raise Refusal(f"{path}: unreadable ({exc})") from exc
    return h.hexdigest()


def read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refusal(f"{what} {path}: unreadable ({exc})") from exc


def n(v: int) -> str:
    return f"{v:,}"


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.4f}%"


_SEP = re.compile(r"[\s,]*")


def _array_start(fh: Any, key: str, path: Path) -> tuple[str, str]:
    """Read ``fh`` up to the top-level ``"key": [``: (the text through ``"key":``, the text after
    its ``[``). The first part plus ``[]`` plus the rest of the file is the object with the
    array read as empty."""
    pat = re.compile(r'"%s"\s*:\s*\[' % re.escape(key))
    buf = ""
    searched = 0
    while True:
        chunk = fh.read(CHUNK)
        if not chunk:
            raise Refusal(f'{path}: no "{key}" array')
        buf += chunk
        m = pat.search(buf, max(0, searched - 64))
        if m:
            return buf[: m.end() - 1], buf[m.end() :]
        searched = len(buf)
        if len(buf) > HEAD_LIMIT:
            raise Refusal(f'{path}: no "{key}" array in its first {HEAD_LIMIT} characters')


def json_head(path: Path, key: str) -> dict[str, Any]:
    """The object at ``path`` with its ``key`` array (its last member) read as empty. The array
    itself is never read: this is how heldout.json is read, for its header fields only."""
    try:
        with path.open(encoding="utf-8") as fh:
            head, _ = _array_start(fh, key, path)
    except OSError as exc:
        raise Refusal(f"{path}: unreadable ({exc})") from exc
    try:
        obj = json.loads(head + "[]}")
    except ValueError as exc:
        raise Refusal(f'{path}: its text before "{key}" is not an object\'s head ({exc})') from exc
    if not isinstance(obj, dict) or obj.get(key) != []:
        raise Refusal(f'{path}: "{key}" is not a top-level member')
    return obj


def stream_array(path: Path, key: str, on_item: Callable[[Any], None]) -> tuple[dict, int]:
    """Call ``on_item`` on each element of the top-level ``key`` array without holding the array;
    return (the object with ``key`` read as empty, the element count)."""
    dec = json.JSONDecoder()
    count = 0
    try:
        fh = path.open(encoding="utf-8")
    except OSError as exc:
        raise Refusal(f"{path}: unreadable ({exc})") from exc
    with fh:
        head, buf = _array_start(fh, key, path)
        pos = 0
        eof = False
        while True:
            pos = _SEP.match(buf, pos).end()
            if pos >= len(buf):
                if eof:
                    raise Refusal(f'{path}: "{key}" is not closed')
                more = fh.read(CHUNK)
                eof = not more
                buf, pos = buf[pos:] + more, 0
                continue
            if buf[pos] == "]":
                tail = buf[pos + 1 :] + fh.read()
                break
            try:
                item, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError as exc:
                if eof:
                    raise Refusal(
                        f'{path}: element {count} of "{key}" is malformed ({exc})'
                    ) from exc
                more = fh.read(CHUNK)
                eof = not more
                buf, pos = buf[pos:] + more, 0
                continue
            on_item(item)
            count += 1
            pos = end
            if pos > CHUNK:
                buf, pos = buf[pos:], 0
    try:
        obj = json.loads(head + "[]" + tail)
    except ValueError as exc:
        raise Refusal(f'{path}: not one JSON object around "{key}" ({exc})') from exc
    if not isinstance(obj, dict) or obj.get(key) != []:
        raise Refusal(f'{path}: "{key}" is not a top-level member')
    return obj, count


def read_npy_ints(path: Path) -> array.array:
    """A 1-D little-endian integer .npy, without numpy."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise Refusal(f"{path}: unreadable ({exc})") from exc
    if raw[:6] != b"\x93NUMPY":
        raise Refusal(f"{path}: not a .npy file")
    if raw[6] == 1:
        (hlen,), start = struct.unpack("<H", raw[8:10]), 10
    elif raw[6] in (2, 3):
        (hlen,), start = struct.unpack("<I", raw[8:12]), 12
    else:
        raise Refusal(f"{path}: .npy version {raw[6]} is not one this reader knows")
    header = ast.literal_eval(raw[start : start + hlen].decode("latin1"))
    codes = {"<i8": "q", "<u8": "Q", "<i4": "i", "<u4": "I"}
    descr, shape = header.get("descr"), header.get("shape")
    if (
        header.get("fortran_order")
        or descr not in codes
        or not isinstance(shape, tuple)
        or len(shape) != 1
    ):
        raise Refusal(f"{path}: {header} is not a 1-D little-endian integer array")
    out = array.array(codes[descr])
    if out.itemsize != int(descr[2]):
        raise Refusal(f"{path}: this platform's array('{codes[descr]}') is {out.itemsize} bytes")
    out.frombytes(raw[start + hlen :])
    if sys.byteorder != "little":
        out.byteswap()
    if len(out) != shape[0]:
        raise Refusal(f"{path}: {len(out)} values, its header says {shape[0]}")
    return out


# --- the token accounting, v4's method -----------------------------------------------------------


def v4_category(row_id: str) -> str:
    """Verbatim AUDIT/v5-plan-2026-10-02/v4_token_accounting.py category()."""
    if ":compose:" in row_id:
        return "defect.composed"
    head = row_id.split(":", 2)
    prefix, fam = head[0], (head[1] if len(head) > 1 else "")
    if fam == "code.defect_class":
        return "defect.noul" if "noul" in row_id else f"defect.{prefix}"
    return f"{fam}"


class ShardAccount:
    """One train shard set, measured as v4_token_accounting.py measures it."""

    def __init__(self, shard_dir: Path, *, tokens_by_row: bool) -> None:
        self.dir = shard_dir
        self.header = read_json(shard_dir / "header.json", "shard header")
        buckets = list(self.header["buckets"])
        offsets = read_npy_ints(shard_dir / "offsets.npy")
        if not offsets or offsets[0] != 0:
            raise Refusal(f"{shard_dir}/offsets.npy does not start at 0")
        by: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        rows: dict[str, set[str]] = defaultdict(set)
        hist: Counter[str] = Counter()
        self.row_tokens: dict[str, int] | None = {} if tokens_by_row else None
        i = 0

        def on_seq(s: dict) -> None:
            nonlocal i
            if i + 1 >= len(offsets):
                raise Refusal(f"{shard_dir}: more sequences than offsets.npy has lengths")
            length = offsets[i + 1] - offsets[i]
            i += 1
            row_id = s["row_id"]
            at = bisect.bisect_left(buckets, length)
            if at >= len(buckets):
                raise Refusal(f"{shard_dir}: a {length}-token sequence is wider than every bucket")
            cat = v4_category(row_id)
            rec = by[cat]
            rec[0] += 1
            rec[1] += length
            rec[2] += buckets[at]
            rows[cat].add(row_id)
            if cat == "defect.composed":
                hist[f"{1000 * (length // 1000):05d}"] += 1
            if self.row_tokens is not None:
                self.row_tokens[row_id] = self.row_tokens.get(row_id, 0) + length

        index, count = stream_array(shard_dir / "sequence_index.json", "sequences", on_seq)
        if count != len(offsets) - 1 or count != self.header.get("n_sequences"):
            raise Refusal(
                f"{shard_dir}: {count} indexed sequences, {len(offsets) - 1} offsets, header "
                f"n_sequences {self.header.get('n_sequences')}"
            )
        self.rows_in = index.get("rows_in")
        self.by_category = {
            k: {"sequences": v[0], "tokens": v[1], "positions": v[2], "rows": len(rows[k])}
            for k, v in sorted(by.items())
        }
        self.total = {
            k: sum(v[j] for v in by.values())
            for j, k in enumerate(("sequences", "tokens", "positions"))
        }
        if self.total["tokens"] != self.header.get("total_tokens"):
            raise Refusal(
                f"{shard_dir}: the sequences sum to {self.total['tokens']} tokens, the header says "
                f"{self.header.get('total_tokens')}"
            )
        self.composed_hist = dict(sorted(hist.items()))
        self.defect_tokens = sum(
            v["tokens"] for k, v in self.by_category.items() if k.startswith("defect.")
        )
        self.defect_positions = sum(
            v["positions"] for k, v in self.by_category.items() if k.startswith("defect.")
        )

    @property
    def share(self) -> float:
        return self.defect_tokens / self.total["tokens"]


def v4_reference(repo: Path, v4_shards: Path | None) -> tuple[dict, int, int, str]:
    """v4's measured defect share from v4_token_accounting.json, and that this script's reading
    of v4's own shard set reproduces it exactly (the method check)."""
    path = repo / ACCOUNTING
    acc = read_json(path, "v4 token accounting")
    measured = acc["measured_v4"]
    cats = measured["by_category"]
    d_tokens = sum(v["tokens"] for k, v in cats.items() if k.startswith("defect."))
    total = measured["total"]["tokens"]
    shard_dir = Path(measured["shard_dir"]) if v4_shards is None else v4_shards
    if not (shard_dir / "offsets.npy").is_file():
        raise Refusal(
            f"{shard_dir}: v4's shard set is not here, so this script cannot show its method "
            f"reproduces {path} (pass --v4-shards)"
        )
    mine = ShardAccount(shard_dir, tokens_by_row=False)
    if mine.by_category != cats or mine.total != measured["total"]:
        raise Refusal(
            f"this script's reading of {shard_dir} does not reproduce {path}: "
            f"{mine.total} vs {measured['total']}; the method is not v4's"
        )
    if mine.composed_hist != measured["composed_sequence_length_hist_1k"]:
        raise Refusal(f"{shard_dir}: the composed length histogram differs from {path}")
    return measured, d_tokens, total, sha256_file(path)


# --- the build -----------------------------------------------------------------------------------


class Manifest:
    """A data/pool manifest streamed: its header, and per row what the fills need."""

    def __init__(self, path: Path, *, keep_rows: bool) -> None:
        self.path = path
        self.family: Counter[str] = Counter()
        self.source: Counter[str] = Counter()
        self.licences: dict[str, Counter[str]] = defaultdict(Counter)
        self.oos_rows: Counter[str] = Counter()
        self.oos_keys: set[str] = set()
        self.oos_bad: list[str] = []
        self.prose_or_contrast = 0
        self.rows: dict[str, tuple[str, str]] = {}
        intern = sys.intern

        def on_entry(e: dict) -> None:
            fam, src = intern(e["family_id"]), intern(e["source_id"])
            self.family[fam] += 1
            self.source[src] += 1
            self.licences[src][intern(e["licence_id"])] += 1
            repo_key = e["repo_key"]
            if repo_key.startswith(CLINC_OOS_PREFIX):
                self.oos_rows[fam] += 1
                self.oos_keys.add(repo_key)
                if not re.fullmatch(r"clinc-oos:[0-9a-f]{16}", repo_key) and len(self.oos_bad) < 3:
                    self.oos_bad.append(repo_key)
            rid = e["row_id"]
            if (
                rid.startswith(f"{QDM_DEFECT}own-prose:")
                or rid.startswith(f"{QDM_DEFECT}contrast:")
                or repo_key.startswith("own-prose:")
            ):
                self.prose_or_contrast += 1
            if keep_rows:
                self.rows[rid] = (fam, src)

        self.header, self.n = stream_array(path, "entries", on_entry)
        if self.n != self.header.get("n_rows"):
            raise Refusal(f"{path}: {self.n} entries, n_rows {self.header.get('n_rows')}")

    @property
    def hash(self) -> str:
        return str(self.header.get("data_snapshot_hash"))


def ledger_row(path: Path, snapshot: str) -> dict:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise Refusal(f"build ledger {path}: unreadable ({exc})") from exc
    hits = []
    for line in lines:
        if line.strip():
            r = json.loads(line)
            if r.get("protocol", {}).get("data_snapshot_hash") == snapshot:
                hits.append(r)
    if len(hits) != 1:
        raise Refusal(
            f"{path}: {len(hits)} rows carry protocol.data_snapshot_hash {snapshot}; the build's "
            "row must be exactly one"
        )
    if hits[0].get("status") != "completed":
        raise Refusal(f"{path} row {hits[0].get('row_id')}: status {hits[0].get('status')!r}")
    return hits[0]


def metric(row: dict, name: str) -> dict | None:
    m = row.get("metrics", {}).get(name)
    return m if isinstance(m, dict) else None


def time_block(path: Path) -> tuple[float | None, int | None, str]:
    """(wall seconds, peak RSS bytes, what was read) from one /usr/bin/time block in a log."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, None, f"{path} unreadable ({exc})"
    real = re.findall(r"^\s*([\d.]+)\s+real\s+[\d.]+\s+user\s+[\d.]+\s+sys\s*$", text, re.M)
    rss = re.findall(r"^\s*(\d+)\s+maximum resident set size\s*$", text, re.M)
    gnu_rss = re.findall(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    gnu_wall = re.findall(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([\d:.]+)", text)
    if len(real) + len(gnu_wall) > 1 or len(rss) + len(gnu_rss) > 1:
        return None, None, f"{path} holds more than one time block"
    wall = float(real[0]) if real else None
    if gnu_wall:
        secs = 0.0
        for part in gnu_wall[0].split(":"):
            secs = secs * 60 + float(part)
        wall = secs
    peak = int(rss[0]) if rss else (int(gnu_rss[0]) * 1024 if gnu_rss else None)
    return wall, peak, f"{path} (sha256 {sha256_file(path)})"


def by_id(path: Path, fields: tuple[str, ...]) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    o = json.loads(line)
                    out[str(o["id"])] = tuple(o.get(f) for f in fields)
    except (OSError, ValueError, KeyError) as exc:
        raise Refusal(f"{path}: unreadable ({exc})") from exc
    return out


# --- the fills -----------------------------------------------------------------------------------


class Item:
    def __init__(self, idx: int) -> None:
        self.idx = idx
        self.filled: list[str] = []
        self.not_filled: list[str] = []
        self.pending: str | None = None
        self.mismatch: list[str] = []

    def text(self, date: str) -> str:
        out = []
        if self.filled:
            out.append(f"Filled {date}: " + "; ".join(self.filled) + ".")
        if self.not_filled:
            lead = (
                "Not filled: "
                if self.filled or self.pending
                else f"Not filled at the freeze ({date}): "
            )
            out.append(lead + "; ".join(self.not_filled) + ".")
        if self.pending:
            out.append(f"Pending after launch (freeze {date}): {self.pending}.")
        if not out:
            raise Refusal(f"item {self.idx} has no fill text")
        return " " + " ".join(out)

    def state(self) -> str:
        if self.filled and not self.not_filled:
            return "filled" + (" + pending after launch" if self.pending else "")
        if self.filled:
            return "partly filled" + (" + pending after launch" if self.pending else "")
        if self.pending:
            return "pending after launch" + (" (part not filled)" if self.not_filled else "")
        return "NOT FILLED"


class Freeze:
    def __init__(self, args: argparse.Namespace) -> None:
        self.a = args
        self.repo: Path = args.repo.resolve()
        self.items = [Item(i) for i in range(len(ANCHORS))]
        self.na: list[str] = []
        self.stamp = datetime.datetime.now(datetime.UTC)
        self.date = self.stamp.strftime("%Y-%m-%d")

    # Refuse, or under --format-test record that the check does not apply to a non-v5 build.
    def bind(self, ok: bool, message: str) -> bool:
        if ok:
            return True
        if self.a.format_test:
            self.na.append(message)
            return False
        raise Refusal(message)

    def rel(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.repo))
        except ValueError:
            return str(path)

    # --- step 1: item 24's gate, first ------------------------------------------------------
    def share_gate(self) -> None:
        a = self.a
        measured, v4_d, v4_t, acc_sha = v4_reference(self.repo, a.v4_shards)
        self.v4_share = v4_d / v4_t
        self.bound = 2 * v4_d / (3 * v4_t)
        self.acc = ShardAccount(a.build_out / "shards" / "train", tokens_by_row=True)
        share = self.acc.share
        lo = 2 * 0.837 / 3
        print(
            f"item 24 gate: code.defect_class effective token share {100 * share:.4f}% "
            f"({n(self.acc.defect_tokens)} of {n(self.acc.total['tokens'])} train tokens, "
            f"{a.build_out}/shards/train); bound two-thirds of v4's {100 * self.v4_share:.4f}% "
            f"({n(v4_d)} of {n(v4_t)}, {ACCOUNTING}) = {100 * self.bound:.4f}% "
            f"(two-thirds of the rounded 83.7% is {100 * lo:.4f}%)"
        )
        if share < self.bound:
            band = " It lies in the rounding band between the two." if share >= lo else ""
            raise Refusal(
                f"item 24: code.defect_class's effective token share is {100 * share:.4f}% "
                f"({n(self.acc.defect_tokens)} of {n(self.acc.total['tokens'])} tokens), below "
                f"two-thirds of v4's {100 * self.v4_share:.4f}% = {100 * self.bound:.4f}%.{band} "
                "The bound is the pre-registration's (rule 2); this is for the human"
            )
        self.acc_sha = acc_sha
        self.v4_measured = measured
        self.share_text = (
            f"code.defect_class's effective token share {100 * share:.4f}% "
            f"({n(self.acc.defect_tokens)} of {n(self.acc.total['tokens'])} real train tokens; "
            f"{100 * self.acc.defect_positions / self.acc.total['positions']:.4f}% of "
            f"{n(self.acc.total['positions'])} positions at bucket width), from the build's "
            f"shards/train (offsets.npy lengths by sequence_index.json row id, {ACCOUNTING_SCRIPT}'s "
            f"method, re-run on v4's shard set by this script and equal to {ACCOUNTING}, sha256 "
            f"{acc_sha}); the bound is two-thirds of v4's {100 * self.v4_share:.4f}% ({n(v4_d)} of "
            f"{n(v4_t)}) = {100 * self.bound:.4f}%: it holds"
        )

    # --- step 2: the build, bound to one row and one scan ----------------------------------
    def load_build(self) -> None:
        a, out = self.a, self.a.build_out
        self.train = Manifest(out / "data" / "pool" / "train.json", keep_rows=True)
        self.val = Manifest(out / "data" / "pool" / "val.json", keep_rows=False)
        held = out / "data" / "heldout" / "heldout.json"
        self.held_head = json_head(held, "entries")
        self.held_sha = sha256_file(held)
        self.train_sha = sha256_file(self.train.path)
        self.val_sha = sha256_file(self.val.path)
        for m, split in (
            (self.train.header, "train"),
            (self.val.header, "val"),
            (self.held_head, "heldout"),
        ):
            if m.get("split") != split:
                raise Refusal(f"{out}: the {split} manifest says split {m.get('split')!r}")
        self.th = self.acc.header
        self.vh = read_json(out / "shards" / "val" / "header.json", "val shard header")
        self.remap = read_json(out / "shards" / "train" / "remap.json", "remap")
        self.row = ledger_row(a.build_ledger, self.train.hash)
        if self.th.get("data_snapshot_hash") != self.train.hash:
            raise Refusal(
                f"shards/train/header.json data_snapshot_hash {self.th.get('data_snapshot_hash')} "
                f"is not data/pool/train.json's {self.train.hash}"
            )
        if self.th.get("split") != "train" or self.vh.get("split") != "val":
            raise Refusal("the shard headers' splits are not train and val")
        if not (
            self.th.get("remap_hash") == self.vh.get("remap_hash") == self.remap.get("remap_hash")
        ):
            raise Refusal("the train header, val header and remap.json name different remap hashes")
        if self.th.get("tokenizer_hash") != self.row["protocol"].get("tokenizer_hash"):
            raise Refusal("the train header's tokenizer_hash is not the build row's")
        # The manifest's families against the accounting's: the gate counted the same rows.
        man_defect = 0
        for rid, tok in (self.acc.row_tokens or {}).items():
            got = self.train.rows.get(rid)
            if got is None:
                raise Refusal(f"shards/train sequence row {rid} is not in data/pool/train.json")
            if got[0] == DEFECT:
                man_defect += tok
        if man_defect != self.acc.defect_tokens:
            raise Refusal(
                f"the manifest's family_id gives code.defect_class {n(man_defect)} tokens, v4's "
                f"row-id method {n(self.acc.defect_tokens)}: the two readings disagree"
            )
        recipe = self.row.get("recipe", {})
        # The scan the build applied.
        att_path = a.scan / "attestation.json"
        self.att = read_json(att_path, "attestation")
        self.att_sha = sha256_file(att_path)
        excl_sha = sha256_file(a.scan / "exclusions.txt")
        # The attestation version the build's own reader accepts, not a copy of it.
        excl_py = self.repo / "python" / "qd_train" / "exclusions.py"
        found = re.findall(
            r"^ATTESTATION_VERSION(?::\s*Final\[int\])?\s*=\s*(\d+)\s*$",
            self._read_text(excl_py) or "",
            re.M,
        )
        if len(found) != 1:
            raise Refusal(f"{excl_py}: ATTESTATION_VERSION is defined {len(found)} times")
        version = int(found[0])
        if (
            self.att.get("version") != version
            or self.att.get("clean") is not True
            or any(v != 0 for v in (self.att.get("remaining_hits") or {"none": 1}).values())
        ):
            raise Refusal(
                f"{att_path} is not a CLEAN version {version} attestation "
                f"({self.rel(excl_py)} ATTESTATION_VERSION)"
            )
        self.att_version = version
        if self.att.get("exclusions_sha256") != excl_sha:
            raise Refusal(
                f"{att_path} vouches for exclusions {self.att.get('exclusions_sha256')}, the file is {excl_sha}"
            )
        self.scan_bound = self.bind(
            recipe.get("exclusions_sha256") == excl_sha == self.th.get("exclusions_sha256"),
            f"the scan at {a.scan} is not the one this build applied: exclusions sha256 "
            f"{excl_sha} (scan), {recipe.get('exclusions_sha256')} (build row recipe), "
            f"{self.th.get('exclusions_sha256')} (train header)",
        )
        # The decision pool.
        self.pool = read_json(a.pool / "manifest.json", "decision pool manifest")
        pool_sha = str(self.pool.get("examples_sha256"))
        file_sha = sha256_file(a.pool / "examples.jsonl")
        try:
            report = a.pool_report.read_text(encoding="utf-8")
        except OSError as exc:
            raise Refusal(f"pool report {a.pool_report}: unreadable ({exc})") from exc
        stated = re.findall(r"examples\.jsonl sha256:\*\*\s*`([0-9a-f]{64})`", report)
        if file_sha != pool_sha or stated != [pool_sha]:
            raise Refusal(
                f"the decision pool's examples sha256 disagree: examples.jsonl {file_sha}, "
                f"manifest {pool_sha}, {a.pool_report} states {stated}"
            )
        self.pool_sha = pool_sha
        self.pool_bound = self.bind(
            recipe.get("decisions_pool_examples_sha256")
            == pool_sha
            == (self.att.get("corpus") or {}).get("decisions_pool_examples_sha256"),
            f"the decision pool {pool_sha} is not the one the build read "
            f"({recipe.get('decisions_pool_examples_sha256')}) and the scan scanned "
            f"({(self.att.get('corpus') or {}).get('decisions_pool_examples_sha256')})",
        )
        # A7.
        a7_path = out / "a7.json"
        self.a7 = read_json(a7_path, "A7 report") if a7_path.is_file() else None
        self.a7_sha = sha256_file(a7_path) if self.a7 is not None else None
        if self.a7 is None:
            self.bind(False, f"{a7_path} does not exist: A7 has not run on this build")
        else:
            snaps = self.a7.get("data_snapshot_hash", {})
            if self.a7.get("passed") is not True:
                raise Refusal(f"{a7_path}: A7 REFUSED ({self.a7.get('failed')}); rule 2")
            if snaps.get("val", {}).get("v5") != self.val.hash or snaps.get("heldout", {}).get(
                "v5"
            ) != self.held_head.get("data_snapshot_hash"):
                raise Refusal(f"{a7_path} was run on another build's val or held-out manifest")
            if self.a7.get("decisions_pool_examples_sha256") not in (None, pool_sha):
                raise Refusal(f"{a7_path} read another decision pool")
        self.val_seqs = self._val_slots()

    def _val_slots(self) -> Counter:
        c: Counter = Counter()

        def on_seq(s: dict) -> None:
            if s["row_id"].startswith(QDM_DEFECT):
                c[s["slot_name"]] += 1

        stream_array(
            self.a.build_out / "shards" / "val" / "sequence_index.json", "sequences", on_seq
        )
        return c

    # --- step 3: the noul parts' ids -------------------------------------------------------
    def load_parts(self) -> None:
        root = self.a.noul_parts
        self.parts_ok = False
        self.parts_why = ""
        v3c_path = root / "defect-noul-v3c" / "manifest.json"
        if not v3c_path.is_file():
            self.parts_why = f"{v3c_path} is not there (--noul-parts)"
            return
        v3c = read_json(v3c_path, "defect-noul-v3c manifest")
        pins = {p["name"]: p for p in v3c["parts"]}
        files = {
            "defect-noul-v3b": root / "defect-noul-v3b" / "examples.jsonl",
            "own-prose-v1": root / "own-prose-v1" / "units.jsonl",
            "commitpackft-g6-v1": root / "commitpackft-g6-v1" / "examples.jsonl",
        }
        for name, path in files.items():
            if name not in pins:
                raise Refusal(f"{v3c_path} pins no part {name}")
            if not path.is_file():
                self.parts_why = f"{path} is not there (gitignored; --noul-parts must be the build checkout's data/pool)"
                return
            got = sha256_file(path)
            if got != pins[name]["data_sha256"]:
                raise Refusal(
                    f"{path} hashes to {got}; {v3c_path} pins {pins[name]['data_sha256']}"
                )
        self.v3c = v3c
        self.v3c_path = v3c_path
        self.v3b = by_id(files["defect-noul-v3b"], ("noul_source", "noul_form"))
        self.prose = by_id(files["own-prose-v1"], ("form", "repo"))
        self.g6 = by_id(files["commitpackft-g6-v1"], ("language",))
        self.g6_manifest = read_json(root / "commitpackft-g6-v1" / "manifest.json", "G6 manifest")
        self.parts_ok = True

    def routes(self) -> tuple[Counter, Counter, int]:
        """Train rows per (route, form) and G6 train rows per language, after exclusions."""
        routes: Counter = Counter()
        g6: Counter = Counter()
        unknown = 0
        for rid, (fam, _src) in self.train.rows.items():
            if fam != DEFECT or not rid.startswith(QDM_DEFECT):
                continue
            ex = rid[len(QDM_DEFECT) :]
            if ex.startswith("own-prose:"):
                form = self.prose.get(ex[len("own-prose:") :], (None,))[0]
                routes[("own-prose", form or "?")] += 1
                unknown += form is None
            elif ex.startswith("contrast:"):
                twin = ex[len("contrast:") :].split(":", 2)
                routes[("contrast", twin[1] if len(twin) > 1 else "?")] += 1
            elif ex in self.g6:
                routes[("commitpackft-g6", "commit")] += 1
                g6[self.g6[ex][0]] += 1
            elif ex in self.v3b:
                src, form = self.v3b[ex]
                routes[(f"v3b {src}", form)] += 1
            elif ex.startswith("noul:"):
                routes[("noul (no part names it)", "?")] += 1
                unknown += 1
        return routes, g6, unknown

    # --- the items -------------------------------------------------------------------------
    def fill(self) -> None:
        a, it = self.a, self.items
        row, recipe = self.row, self.row.get("recipe", {})
        ledger_ref = (
            f"{self.rel(a.build_ledger)} row {row['row_id']} (written_at {row['written_at']})"
        )

        # 0
        x = it[0]
        commit = str(row.get("code_commit"))
        if commit.endswith("-dirty") or not re.fullmatch(r"[0-9a-f]{40}", commit):
            x.not_filled.append(
                f"the build commit: the build row's code_commit is {commit!r}, not a clean 40-hex commit ({ledger_ref})"
            )
        else:
            x.filled.append(
                f"build commit {commit} (code_commit of {ledger_ref}; recipe.rev {recipe.get('rev')})"
            )
        x.filled.append(
            f"train-manifest hash (data_snapshot_hash) {self.train.hash} ({a.build_out}/data/pool/train.json, "
            f"{n(self.train.n)} rows, file sha256 {self.train_sha}; equal to the row's "
            "protocol.data_snapshot_hash and the train shard header's)"
        )
        x.filled.append(
            f"val manifest hash {self.val.hash} (data/pool/val.json, {n(self.val.n)} rows, file sha256 {self.val_sha})"
        )
        x.filled.append(
            f"held-out manifest hash {self.held_head.get('data_snapshot_hash')} "
            f"(data/heldout/heldout.json, {n(self.held_head.get('n_rows', 0))} rows, file sha256 "
            f"{self.held_sha}; read up to its entries only)"
        )
        x.filled.append(
            f"train shard hash {self.th.get('shard_hash')} (shards/train/header.json: "
            f"{n(self.th['n_sequences'])} sequences, {n(self.th['total_tokens'])} tokens, max width "
            f"{n(self.th['max_seq_len'])}), val shard hash {self.vh.get('shard_hash')} (shards/val/header.json)"
        )
        x.filled.append(
            f"remap hash {self.remap.get('remap_hash')} (shards/train/remap.json, equal to both headers')"
        )

        # 1
        x = it[1]
        if self.train.oos_bad:
            x.not_filled.append(
                f"the key spelling: repo keys {self.train.oos_bad} are not clinc-oos:<16 hex>"
            )
        train_oos = {f: self.train.oos_rows.get(f, 0) for f in CLINC_FAMILIES}
        x.filled.append(
            f"the key is clinc-oos:<16 lower-hex> (repo_key of each CLINC oos utterance, the "
            f"blake2b-8 digest of candidate_key; {n(len(self.train.oos_keys))} distinct such keys "
            f"in data/pool/train.json, every one of that form)"
        )
        x.filled.append(
            "train rows per CLINC family " + ", ".join(f"{f} {n(v)}" for f, v in train_oos.items())
        )
        if any(v != EXPECTED_OOS["train"] for v in train_oos.values()):
            x.mismatch.append(
                f"train oos rows {train_oos}, expected {EXPECTED_OOS['train']} per family"
            )
        if self.a7 is not None:
            got = {
                (r["split"], r["family"]): r.get("oos")
                for r in self.a7["families"]
                if r.get("kind") == "clinc"
            }
            for split in ("val", "heldout"):
                vals = {f: got.get((split, f)) for f in CLINC_FAMILIES}
                x.filled.append(
                    f"{split} oos rows per CLINC family "
                    + ", ".join(f"{f} {v}" for f, v in vals.items())
                    + " (a7.json)"
                )
                if any(v != EXPECTED_OOS[split] for v in vals.values()):
                    x.mismatch.append(f"{split} oos rows {vals}, expected {EXPECTED_OOS[split]}")
            x.filled.append(
                "against the expected 1,216 / 72 / 62: "
                + (
                    "equal"
                    if not x.mismatch
                    else "DIFFERENT ("
                    + "; ".join(x.mismatch)
                    + "); the build's own counts are the record (data.clinc_oos_rekey.not_modelled)"
                )
            )
        else:
            x.not_filled.append("val and held-out oos counts: no a7.json in the build output")

        # 2, 10: the attestation
        att = self.att
        x = it[2]
        hits = defaultdict(list)
        for h in att.get("hits_by_source_family", []):
            hits[(h["source"], h["target"])].append(
                f"{h['source_family']} {n(h['source_rows_hit'])}"
            )
        enforced = set(att.get("enforced_targets", []))
        hit_text = "; ".join(
            f"{src}->{tgt}{'' if tgt in enforced and src == 'train' else ' (report-only)'}: "
            + ", ".join(v)
            for (src, tgt), v in sorted(
                hits.items(), key=lambda kv: (kv[0][0] != "train", kv[0][1] not in enforced, kv[0])
            )
        )
        bound_txt = (
            "equal to the build row's recipe.exclusions_sha256 and the train header's"
            if self.scan_bound
            else "NOT the exclusions this build applied (format test)"
        )
        dec = metric(row, "decontam_exclusion")
        x.filled.append(
            f"attestation sha256 {self.att_sha} ({a.scan}/attestation.json: version 2, CLEAN, "
            f"remaining hits {att.get('remaining_hits')})"
        )
        x.filled.append(
            f"exclusions_sha256 {att.get('exclusions_sha256')} ({bound_txt}): {n(att.get('n_exclusions', 0))} "
            "identity keys"
            + (
                f", {n(dec['value'])} train rows left the split (the row's decontam_exclusion)"
                if dec and dec.get("state") == "ran"
                else ""
            )
        )
        x.filled.append(
            f"train rows hit per source family and target set (attestation hits_by_source_family): {hit_text}"
        )
        fam_rates = self._family_rates()
        if isinstance(fam_rates, str):
            x.not_filled.append(f"excluded identity keys per source family: {fam_rates}")
            it[10].not_filled.append(
                f"exclusion count per source family: {fam_rates}; the per-family hits are item 2's"
            )
        else:
            txt = ", ".join(
                f"{f} {n(v['excluded'])} of {n(v['train_keys'])}"
                for f, v in sorted(fam_rates[0].items())
                if v["excluded"]
            )
            x.filled.append(f"excluded identity keys per source family ({fam_rates[1]}): {txt}")
            it[10].filled.append(
                f"exclusion count per source family on the final set: {txt} ({fam_rates[1]})"
            )

        # 3
        x = it[3]
        x.filled.append(
            f"defect-noul-v3c examples sha256 {recipe.get('defect_noul_examples_sha256')} (the build row's recipe.defect_noul_examples_sha256)"
        )
        if self.parts_ok:
            self.bind(
                recipe.get("defect_noul_examples_sha256") == self.v3c["examples_sha256"],
                f"the build row read noul examples {recipe.get('defect_noul_examples_sha256')}, "
                f"{self.rel(self.v3c_path)} is {self.v3c['examples_sha256']}",
            )
            pins = {p["name"]: p for p in self.v3c["parts"]}
            op = pins["own-prose-v1"]
            x.filled.append(
                f"own-prose-v1 examples (units.jsonl) sha256 {op['data_sha256']}, manifest sha256 {op['manifest_sha256']} ({self.rel(self.v3c_path)} parts, file verified)"
            )
            routes, self.g6_train, unknown = self.routes()
            x.filled.append(
                "train rows per route and form after exclusions (train manifest row ids joined to the parts' ids): "
                + ", ".join(f"{r} {f} {n(v)}" for (r, f), v in sorted(routes.items()))
            )
            if unknown:
                x.mismatch.append(f"{unknown} noul train rows name no part's id")
        else:
            x.not_filled.append(
                f"own-prose-v1 examples sha256 and rows per route and form: {self.parts_why}"
            )
        x.filled.append(
            f"composed-v2 examples sha256 {recipe.get('defect_class_examples_sha256')} (the build row's recipe.defect_class_examples_sha256)"
        )
        x.filled.append(
            "composed train sequences per 1k real-token bin (shards/train lengths, v4's accounting method): "
            + ", ".join(f"{k} {n(v)}" for k, v in self.acc.composed_hist.items())
        )
        surv = metric(row, "composed_span_survival_by_length")
        if surv and surv.get("state") == "ran":
            bins = json.loads(surv["detail"])["bins"]
            x.filled.append(
                "composed rows per 1k real-token bin (the row's composed_span_survival_by_length): "
                + ", ".join(f"{k} {n(v['rows'])}" for k, v in bins.items())
            )
            self.bins = bins
        else:
            self.bins = None
            x.not_filled.append(
                "composed rows per bin: the row's composed_span_survival_by_length did not run"
            )
        over = metric(row, "over_max_seq_len_composed_train")
        if over and over.get("state") == "ran":
            x.filled.append(
                f"over_max_seq_len_composed_train {n(over['value'])} of {n(over['n_total'])} (target 0: {'met' if over['value'] == 0 else 'NOT MET'})"
            )
            if over["value"] != 0:
                x.mismatch.append(f"over_max_seq_len_composed_train {over['value']}, target 0")
        else:
            x.not_filled.append("over_max_seq_len_composed_train: not on the build row")

        # 4
        x = it[4]
        choice, span = self.val_seqs.get("defect_class", 0), self.val_seqs.get("defect_span", 0)
        if choice != EXPECTED_VAL_CHOICE_SLOTS:
            self.bind(
                False,
                f"item 4: {choice} code.defect_class val choice slots written, the pre-registration expects {EXPECTED_VAL_CHOICE_SLOTS} (rule 2: the val population moved)",
            )
        x.filled.append(
            f"code.defect_class val choice slots written {n(choice)} (shards/val/sequence_index.json slot "
            f"defect_class; expected 2,304: {'equal' if choice == EXPECTED_VAL_CHOICE_SLOTS else 'DIFFERENT'}), span slots {n(span)}"
        )
        if self.val.prose_or_contrast:
            raise Refusal(
                f"item 4: {self.val.prose_or_contrast} own-prose or contrast rows are in data/pool/val.json"
            )
        x.filled.append(
            f"own-prose or contrast rows in val: 0 of {n(self.val.family.get(DEFECT, 0))} code.defect_class "
            "val rows (no val.json row id under qdm:code.defect_class:own-prose: or :contrast:, no repo_key own-prose:)"
        )
        if self.a7 is not None:
            held = [
                r for r in self.a7["families"] if r["split"] == "heldout" and r["family"] == DEFECT
            ]
            if (
                len(held) == 1
                and held[0]["passed"]
                and held[0].get("missing") == 0
                and held[0].get("extra") == 0
            ):
                x.filled.append(
                    f"in held-out: none, because a7.json (sha256 {self.a7_sha}) finds held-out code.defect_class's identity keys equal to v4's ({n(held[0]['v5'])} rows, 0 missing, 0 extra); held-out entries were not read"
                )
            else:
                raise Refusal(
                    "item 4: a7.json does not show held-out code.defect_class equal to v4's"
                )
        else:
            x.not_filled.append(
                "own-prose or contrast rows in held-out: no a7.json, and held-out entries are not read here"
            )

        # 5, 7, 18: the prelude
        prelude_where = (
            "recorded in the Mac prelude record (sha256 pinned at deploy as V5_PRELUDE_SHA256, checked "
            "by v5_common.sh v5_prelude_check against V5_LANE_AT); copied into this file by the first "
            f"post-launch amendment, before seed 0's ft row ({PRELUDE_GAP}: the prelude runs at the lane "
            "commit, which carries this file)"
        )
        prelude_text = self._read_text(self.repo / V5_PRELUDE)
        no_needle = prelude_text is not None and "needle" not in prelude_text.lower()
        it[5].pending = (
            (
                "not in the Mac prelude record (" + V5_PRELUDE + " has no needle field)"
                if no_needle
                else "the Mac prelude record was not checked for a needle field"
            )
            + ". The rebuilt suite is built when a seed is scored: min, median and max "
            "needle_suite_tokens come from v5 seed 0's epoch-score-val row (metrics.needle_suite_tokens, "
            "the median as its value, the rest in its detail); the suite's digest is on no row "
            "(GAP-V5-RULES-REBUILT-SUITE-NOT-ON-THE-ROW-2026-10-02) and is filled from the scoring log "
            "or stays not filled"
        )
        it[7].pending = (
            f"the batches and width are {prelude_where}. v5's recipe hash is not in the prelude record: "
            "it is v5 seed 0's ft row's protocol.recipe_hash, filled after that row"
        )
        it[18].pending = f"per plan seed 0-4 plan_order_digest is {prelude_where}"

        # 6, 8, 17: by the human's answer
        state = (
            f"GH200 marker state given at the freeze: {a.gh200_state}" if a.gh200_state else None
        )
        if state is None:
            for i in (6, 8, 17):
                it[i].not_filled.append(
                    "the GH200 marker state and its time (not given to the script: --gh200-state)"
                )
        human = f"the human's answer ('{VERBATIM}', ~18:42Z 2026-10-03, {ANSWERS})"
        r4 = self._r4()
        it[6].filled.append(
            "R4 is kept, under its own 'otherwise' clause, consistent with V5_LOWER=keep: its first "
            "condition (J6(f) holds prose) has no deciding row at launch"
            + (f" ({state})" if state else "")
            + (f"; its second reads {r4}" if r4 else "")
        )
        if r4.startswith("not filled: "):
            it[6].not_filled.append(r4[len("not filled: ") :])
            r4 = ""
        it[6].filled.append(
            f"C1, C2b and C3 are off by {human}: V5_C1=off, V5_C2B=off, V5_LOWER=keep, V5_LRSET=f; "
            "C2a is off by Fable's ruling (item 23). Their GH200 rows are read afterwards, report-only"
        )
        it[8].filled.append(
            f"C2a is off by Fable's ruling (item 23) and C2b by {human}. Neither tierb word, decision "
            "JSON, candidate ledger nor row id exists at launch"
            + (f" ({state})" if state else "")
            + "; they are read afterwards, report-only"
        )
        it[17].filled.append(
            f"C3 is off by {human}: V5_LOWER=keep, V5_LRSET=f. The fsucc word, its decision JSON, "
            "detail.arms.j6dv4.wins and the advance-answer branch do not exist at launch"
            + (f" ({state})" if state else "")
            + "; they are read afterwards, report-only"
        )

        # 9
        x = it[9]
        nine = self.acc.composed_hist.get("09000", 0)
        x.filled.append(
            f"9-10k supply: {n(nine)} composed train sequences of 9,000-9,999 real tokens (shards/train)"
            + (
                f", {n(self.bins['09000-09999']['rows'])} composed rows in the row's 09000-09999 bin"
                if self.bins and "09000-09999" in self.bins
                else ""
            )
            + "; compose --census counts file supply, not row lengths (HANDOFF/v5-data-2026-10-02.md section 6)"
        )
        if a.build_log:
            wall, peak, src = time_block(a.build_log)
            if peak is None:
                x.not_filled.append(
                    f"the build's peak RSS: {src} carries no single /usr/bin/time block"
                )
            else:
                x.filled.append(
                    f"the build's peak RSS {peak / 1e9:.2f} GB ({n(peak)} bytes, {src}"
                    + (f"; wall {wall:,.0f} s" if wall else "")
                    + ")"
                )
        else:
            x.not_filled.append("the build's peak RSS: no --build-log (the build row records none)")

        # 10
        x = it[10]
        if a.scan_log:
            wall, peak, src = time_block(a.scan_log)
            if wall is None:
                x.not_filled.append(
                    f"containment runtime: {src} carries no single /usr/bin/time block"
                )
            else:
                x.filled.append(
                    f"containment runtime {wall:,.0f} s"
                    + (f", peak RSS {peak / 1e9:.2f} GB" if peak else "")
                    + f" ({src})"
                )
        else:
            x.not_filled.append("containment runtime: no --scan-log (the attestation records none)")
        sets = att.get("sets", {})
        x.filled.append(
            "scan sets (attestation): "
            + ", ".join(
                f"{k} {n(v['rows'])} rows ({n(v['too_short'])} too short)" for k, v in sets.items()
            )
        )

        # 11
        x = it[11]
        if self.a7 is not None:
            fams = self.a7["families"]
            parts = []
            for split in ("val", "heldout"):
                rows_ = [r for r in fams if r["split"] == split]
                parts.append(
                    f"{split}: "
                    + ", ".join(
                        f"{r['family']} {n(r['v5'])}"
                        + (f" (v4 {n(r['v4'])})" if r.get("kind") != "decision-pool" else " (pool)")
                        for r in rows_
                    )
                )
            snaps = self.a7["data_snapshot_hash"]
            x.filled.append(
                f"a7.json (sha256 {self.a7_sha}) PASS, {self.a7['families_checked']} families, 0 failed; "
                + "; ".join(parts)
                + f"; data_snapshot_hash val v4 {snaps['val']['v4']} / v5 {snaps['val']['v5']}, held-out v4 {snaps['heldout']['v4']} / v5 {snaps['heldout']['v5']}"
            )
        else:
            x.not_filled.append(f"no a7.json in {a.build_out}")

        # 12
        x = it[12]
        files = self.repo / "data" / "pool" / "own-prose-v1" / "files.jsonl"
        man = self.repo / "data" / "pool" / "own-prose-v1" / "manifest.json"
        if files.is_file() and man.is_file():
            op = read_json(man, "own-prose-v1 manifest")
            got = sha256_file(files)
            if got != op["files"]["sha256"]:
                raise Refusal(f"{files} hashes to {got}; {man} pins {op['files']['sha256']}")
            repos = [r["repo"] for r in op["admitted_repos"]]
            strike = op.get("strike", {})
            with_units = sorted(op.get("totals", {}).get("by_repo", {}))
            x.filled.append(
                f"data/pool/own-prose-v1/files.jsonl sha256 {got} ({n(op['files']['count'])} files; equal to "
                f"own-prose-v1/manifest.json files.sha256, manifest sha256 {sha256_file(man)})"
            )
            x.filled.append(
                "the human's tick, 'All but Lappi-decision (Recommended)' at d24c865 (data.sources[3].provenance): "
                f"the {len(repos)} repos it admits (own-prose-v1/manifest.json admitted_repos) "
                + ", ".join(repos)
                + "; then the human's strike, 'Strike personal + business (Recommended)' (data.sources[3].human_strike): "
                "repos "
                + ", ".join(strike.get("repos", []))
                + " and paths "
                + ", ".join(f"{p['repo']}/{p['prefix']}" for p in strike.get("paths", []))
                + f" (manifest strike); {len(with_units)} repos hold units after both (totals.by_repo): "
                + ", ".join(with_units)
            )
        else:
            x.not_filled.append(f"{self.rel(files)} or its manifest is not in --repo")

        # 13, 19, 20, 22, 25, 26, 27: after launch
        it[
            13
        ].pending = "v5 seed 0's trajectory waiter (v5traj-s0): its trajectory-ood rows' wall clock from the first to the last, in v5's ledger"
        it[
            19
        ].pending = "recipe.batch_order read on each v5 (seeds 0-4), noul-weight arm and J5' ft row as it is written, in v5's and the arm's ledgers"
        it[
            20
        ].pending = "docs/promotion-decisions.json as committed at v5 seed 0's ft row's written_at, with that commit's sha"
        it[
            22
        ].pending = "v5 seed 0's ft row's wall_clock_s against F's 16,024 s (973cd4e3), and the box log showing no _evaluate pass"
        it[
            25
        ].pending = "each v5 seed's epoch-score-val row (ood_abstain suite-half per category, in-distribution abstention per family, calibration_states' selective-risk grid), and qd-gate-report's recomputation"
        it[
            26
        ].pending = "the CPU calibration lane's fit on val, after its three prerequisites; the margin's distance from the c = 2 policy and in-distribution abstention against the 5% Wilson cap on every seed's val"
        it[
            27
        ].pending = "the probe's row (the probe ledger) and v5 seed 0's ft row's wall_clock_s on the box, replacing launch.projected_cost_usd's GH200-cadence hours"

        # 14
        x = it[14]
        common = self.repo / V5_COMMON
        text = self._read_text(common)
        if text is None:
            x.not_filled.append(f"{V5_COMMON} is not in --repo")
        else:
            lines = text.splitlines()
            found = []
            for name, pat in (
                ("V5_CONTINUE", r"^V5_CONTINUE=\$Q/V5_CONTINUE$"),
                ("V5NW_HUMAN_YES", r"^V5NW_HUMAN_YES=\$Q/V5NW_HUMAN_YES$"),
                ("v5nw.room", r"^V5NW_ROOM=\$Q/v5nw\.room$"),
            ):
                hits_ = [i + 1 for i, l in enumerate(lines) if re.match(pat, l)]
                if len(hits_) != 1:
                    raise Refusal(f"{common}: {name} is defined {len(hits_)} times, not once")
                found.append(f"{lines[hits_[0] - 1]} (:{hits_[0]})")
            x.filled.append(
                f"{V5_COMMON} (sha256 {sha256_file(common)}): "
                + ", ".join(found)
                + "; markers the lead writes with the human's words, an empty file not a yes"
            )

        # 15
        x = it[15]
        pc = self.repo / PASS_CHECK
        if pc.is_file():
            d = read_json(pc, "pass check")
            if d.get("pass") is not True or d.get("all_three_sizes") is not True:
                raise Refusal(f"{pc}: the subsample re-run did not pass at all three sizes")
            per = []
            for r in d["results"]:
                c1, c2, c3, c4 = r["1"], r["2"], r["3"], r["4"]
                per.append(
                    f"{r['size']}: (1) CLINC train keys excluded {c1['clinc_train_keys_excluded']} of {c1['clinc_train_keys']}, "
                    f"enforced pairs with an intent row {c1['enforced_pairs_with_an_intent_row']}; (2) non-intent exclusions "
                    f"{c2['non_intent_keys']} keys, sha256 {'equal to' if c2['sha256'] == c2['expected_sha256'] else 'NOT'} version 1's; "
                    f"(3) key_ii_blind {c3['key_ii_blind_by_set']}, too_short_after_strip {c3['too_short_after_strip_by_set_slot_texts_all_families']}; "
                    f"(4) CLEAN {c4['clean']}, splitter checks "
                    + ", ".join(
                        f"{k} {v['state']}/{v['passed']}" for k, v in c4["splitter_checks"].items()
                    )
                )
            x.filled.append(
                f"subsample re-run under version 2 ({PASS_CHECK}, sha256 {sha256_file(pc)}): pass at 1/2/5% — "
                + " | ".join(per)
            )
        else:
            x.not_filled.append(f"{PASS_CHECK} is not in --repo")
        ts = att["export"]["template_strip"]
        x.filled.append(
            f"full scan ({a.scan}/attestation.json): strip version {ts.get('version')}, key_ii_blind "
            + ", ".join(f"{k} {n(v['n'])}" for k, v in ts.get("key_ii_blind", {}).items())
            + f"; too_short_after_strip {ts.get('too_short_after_strip_by_set')}"
        )
        if not isinstance(fam_rates, str):
            g = fam_rates[2]
            x.filled.append(
                f"CLINC rate on the full scan: {n(g['excluded'])} of {n(g['train_keys'])} utterance keys excluded ({fam_rates[1]})"
            )
        else:
            x.not_filled.append(f"the CLINC rate on the full scan: {fam_rates}")
        zc = self._zero_checks()
        if isinstance(zc, str):
            x.not_filled.append(f"the two zero-checks on the full scan: {zc}")
        else:
            vals, src = zc
            x.filled.append(
                "the two zero-checks on the full scan ("
                + src
                + "): "
                + "; ".join(
                    f"{s} key_ii_blind {v['key_ii_blind']} (matches the attestation: {v['key_ii_blind_matches_attestation']}), exact {v['exact_match_in_train']}, contiguous word-subsequence {v['contiguous_word_subsequence_of_a_train_utterance']}"
                    for s, v in vals.items()
                )
            )
            for s, v in vals.items():
                if (
                    v["exact_match_in_train"]
                    or v["contiguous_word_subsequence_of_a_train_utterance"]
                ):
                    x.mismatch.append(
                        f"{s} zero-check nonzero: a GAP for the human (build_order step 2)"
                    )

        # 16
        x = it[16]
        if self.parts_ok:
            gm = self.g6_manifest
            langs = sorted(gm["totals"]["by_language"])
            x.filled.append(
                f"commitpackft-g6-v1 examples sha256 {gm['examples_sha256']} (file verified), per-language cap "
                f"{gm['per_language_cap']}; per language rows after the licence filter / selected / train rows after "
                "exclusions: "
                + ", ".join(
                    f"{l} {n(gm['allowlist_counts'][l]['after_licence'])}/{n(gm['totals']['by_language'][l])}/{n(self.g6_train.get(l, 0))}"
                    for l in langs
                )
            )
        else:
            x.not_filled.append(f"G6 per-language rows: {self.parts_why}")

        # 21
        x = it[21]
        x.filled.append(
            "nothing is filled at the freeze: this is a contingency read only after v5's rows exist"
        )
        if self.parts_ok:
            fun = self.g6_manifest["funnel"]
            band = sum(v["rendered_in_band"] for v in fun.values())
            sel = sum(v["selected"] for v in fun.values())
            x.filled.append(
                f"supply as built (report-only): the G6 funnel renders {n(band)} rows in band against {n(sel)} selected "
                f"(2x is {n(2 * sel)}; commitpackft-g6-v1 manifest funnel, before its max_per_repo {self.g6_manifest['max_per_repo']}); "
                "v3b's SQuAD paragraph and question forms are at their stated ceiling, 3 per title over 335 train "
                "titles (defect-noul-v3b manifest preregistered.form_mix.basis), so 2x needs a new draw rule"
            )

        # 23
        it[23].filled.append(
            "settled before the freeze by Fable's ruling as written above; no build value to fill. V5_C2A=off "
            "is a deploy pin (v5_common.sh, committed UNSET), not this file's"
        )

        # 24
        x = it[24]
        x.filled.append(self.share_text)
        x.filled.append(self._sources())
        x.filled.append(
            f"the decision pool's examples sha256 {self.pool_sha} ({a.pool}/manifest.json, examples.jsonl and {self.rel(a.pool_report)} agree)"
        )
        pool_val = self.pool.get("val_by_family", {})
        x.filled.append(
            "val rows per new family (data/pool/val.json): "
            + ", ".join(f"{f} {n(self.val.family.get(f, 0))}" for f in sorted(pool_val))
            + f", {n(sum(self.val.family.get(f, 0) for f in pool_val))} in total"
        )
        if any(self.val.family.get(f, 0) > 1000 for f in pool_val):
            x.mismatch.append("a new family has more than 1,000 val rows")
        x.pending = "per-stratum val accuracy for procedural.decisions and per-family val results, report-only, from each seed's eval row"

        # 28
        x = it[28]
        dd = self.train.header.get("dedupe", {})
        sp = self.train.header.get("split_report", {})
        x.filled.append(
            f"dedupe: {n(dd.get('n_candidate_pairs', 0))} candidate pairs under max_candidate_pairs "
            f"{n(dd.get('max_candidate_pairs', 5_000_000))}"
            + (
                ""
                if dd.get("max_candidate_pairs") == POOL_BOUND
                else " (NOT the pool bound 12,500,000)"
            )
            + f", status {dd.get('status', {}).get('state')}/{dd.get('status', {}).get('passed')}"
        )
        if dd.get("max_candidate_pairs") != POOL_BOUND:
            self.bind(
                False,
                f"item 28: the train manifest's dedupe ran under {dd.get('max_candidate_pairs', 'the default 5,000,000')}, not POOL_MAX_CANDIDATE_PAIRS {POOL_BOUND}",
            )
        ndd = sp.get("near_duplicate_disjoint", {})
        m_ = re.search(r"([\d]+) candidate pairs", ndd.get("detail", ""))
        x.filled.append(
            f"split near_duplicate_disjoint: {m_.group(1) if m_ else '?'} candidate pairs, {ndd.get('state')}/{ndd.get('passed')}, n {n(ndd.get('n', 0))} of {n(ndd.get('n_total', 0))}"
        )
        scope = dd.get("near_duplicate_scope")
        if scope:
            x.filled.append(
                f"scoped rows by family (exact content, {scope.get('ruling')}): "
                + ", ".join(f"{k} {n(v)}" for k, v in scope["exact_content_rows_by_family"].items())
                + f"; MinHash rows {n(scope['minhash_rows'])}"
            )
            clusters = dd.get("exact_content_clusters", [])
            x.filled.append(
                f"exact-content drops {n(sum(c['n_rows_dropped'] for c in clusters))} rows in {n(len(clusters))} clusters"
            )
        else:
            x.not_filled.append(
                "scoped rows and exact-content drops: the train manifest's dedupe has no near_duplicate_scope"
            )
        ecd = sp.get("exact_content_disjoint")
        if ecd:
            x.filled.append(
                f"exact_content_disjoint {ecd.get('state')}/{ecd.get('passed')}, n {n(ecd.get('n', 0))} of {n(ecd.get('n_total', 0))}"
            )
        else:
            x.not_filled.append("exact_content_disjoint: not in the train manifest's split_report")
        short = self.pool.get("decontamination", {}).get("rows_too_short_by_set", {})
        x.filled.append(
            f"rebuilt decision pool examples sha256 {self.pool_sha}"
            + (
                " (= the build row's recipe and the scan's corpus)"
                if self.pool_bound
                else " (format test: NOT the build's)"
            )
            + "; too short to scan: "
            + ", ".join(
                f"{t} {n(short[t]['too_short'])} of {n(short[t]['rows'])}"
                for t in POOL_TARGETS
                if t in short
            )
        )

    # --- helpers for the items ------------------------------------------------------------
    def _read_text(self, path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def _family_rates(self) -> tuple[dict, str, dict] | str:
        if self.a.family_rates is None:
            return "no --family-rates (AUDIT/prep2-2026-10-02/family_rates.py's output over this scan; it reads the request, which this script does not)"
        d = read_json(self.a.family_rates, "family rates")
        mine = [r for r in d if Path(r.get("scan", "")).resolve() == self.a.scan.resolve()]
        if len(mine) != 1:
            return f"{self.a.family_rates} has {len(mine)} entries for {self.a.scan}"
        r = mine[0]
        if r.get("n_exclusions") != self.att.get("n_exclusions"):
            return f"{self.a.family_rates} counts {r.get('n_exclusions')} exclusions, the attestation {self.att.get('n_exclusions')}: another scan"
        return (
            r["families"],
            f"{self.a.family_rates}, sha256 {sha256_file(self.a.family_rates)}",
            r["groups"]["CLINC (intent.*)"],
        )

    def _zero_checks(self) -> tuple[dict, str] | str:
        if self.a.zero_checks is None:
            return "no --zero-checks (AUDIT/prep3-2026-10-02/key_ii_blind_zero_checks.py's output over this scan)"
        d = read_json(self.a.zero_checks, "zero checks")
        mine = [r for r in d if Path(r.get("scan", "")).resolve() == self.a.scan.resolve()]
        if len(mine) != 1:
            return f"{self.a.zero_checks} has {len(mine)} entries for {self.a.scan}"
        r = mine[0]
        if r.get("computed_and_agrees_with_attestation") is not True:
            return f"{self.a.zero_checks}: not computed, or its blind sets disagree with the attestation"
        keep = (
            "key_ii_blind",
            "key_ii_blind_matches_attestation",
            "exact_match_in_train",
            "contiguous_word_subsequence_of_a_train_utterance",
        )
        return {
            s: {k: r[s][k] for k in keep} for s in ("val", "heldout")
        }, f"{self.a.zero_checks}, sha256 {sha256_file(self.a.zero_checks)}"

    def _r4(self) -> str:
        """R4's second condition over F seeds 0-4 (readings.R4: 'read x/5 over F seeds 0-4')."""
        path = self.a.f_ledger if self.a.f_ledger is not None else self.repo / F_LEDGER
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return f"not filled: F seeds 0-4's prose counts: {path} is unreadable ({exc})"
        rows = {}
        for line in lines:
            if not line.strip():
                continue
            r = json.loads(line)
            if (
                r.get("run_kind") == "eval"
                and r.get("status") == "completed"
                and r.get("recipe", {}).get("tag") == "epoch-score-val"
            ):
                m = r["metrics"].get("ood_abstain.prose", {})
                seed = r["protocol"]["seed"]
                if seed in rows:
                    raise Refusal(f"{path}: two epoch-score-val rows for F seed {seed}")
                if not isinstance(m.get("n"), int) or not isinstance(m.get("n_total"), int):
                    return f"not filled: F seed {seed}'s ood_abstain.prose is not a count in {path}"
                rows[seed] = (r["row_id"][:8], m["n"], m["n_total"])
        if sorted(rows) != [0, 1, 2, 3, 4]:
            return (
                f"not filled: R4 reads F seeds 0-4, and {path} holds epoch-score-val rows for seeds "
                f"{sorted(rows)} (pass --f-ledger with main's copy)"
            )
        holds = [s for s, (_, k, t) in rows.items() if 2 * k >= t]
        listed = ", ".join(f"seed {s} {rid} {k}/{t}" for s, (rid, k, t) in sorted(rows.items()))
        fewer = 2 * len(holds) < 5
        return (
            f"{len(holds)} of 5 with-flag F seeds hold prose (2n >= n_total, i.e. >= 30 of 60): {listed} "
            f"({F_LEDGER}, epoch-score-val rows, sha256 {sha256_file(path)}), "
            + (
                "fewer than half, so the drop turns on J6(f) alone, which has no row"
                if fewer
                else "not fewer than half, so the drop's second condition fails as well"
            )
        )

    def _sources(self) -> str:
        fetch = None
        if self.a.fetch_record is not None:
            got = sha256_file(self.a.fetch_record)
            want = self.pool.get("inputs", {}).get("fetch_record")
            if got != want:
                raise Refusal(
                    f"{self.a.fetch_record} hashes to {got}; the pool manifest's inputs.fetch_record is {want}"
                )
            fetch = read_json(self.a.fetch_record, "fetch record")
        tokens: Counter = Counter()
        for rid, tok in (self.acc.row_tokens or {}).items():
            tokens[self.train.rows[rid][1]] += tok
        inputs = self.pool.get("inputs", {})
        parts = []
        for src in sorted(self.train.source):
            lic = ", ".join(f"{k} {n(v)}" for k, v in self.train.licences[src].most_common())
            files = [k for k in inputs if k.startswith(src + "/")]
            prov = []
            for f in files:
                rel_ = f[len(src) + 1 :]
                rec = [
                    e for e in (fetch or []) if e.get("dataset") == src and e.get("file") == rel_
                ]
                if len(rec) == 1:
                    e = rec[0]
                    which = (
                        "sha256"
                        if e.get("sha256") == inputs[f]
                        else ("jsonl_sha256" if e.get("jsonl_sha256") == inputs[f] else None)
                    )
                    prov.append(
                        f"{rel_} @{e['revision'][:8]} {inputs[f][:16]}..."
                        + ("" if which else " (NOT the fetch record's)")
                    )
                else:
                    prov.append(
                        f"{rel_} {inputs[f][:16]}... (revision: {'no --fetch-record' if fetch is None else 'not in the fetch record'})"
                    )
            parts.append(
                f"{src}: {n(self.train.source[src])} train rows, {n(tokens[src])} tokens, licence {lic}"
                + (f", inputs {'; '.join(prov)}" if prov else "")
            )
        return "per source (train manifest joined to the shard set): " + " | ".join(parts)

    # --- step 4: the text ------------------------------------------------------------------
    def apply(self, text: str, draft: dict) -> str:
        pending = draft["amendments_pending"]
        for i, item in enumerate(self.items):
            add = item.text(self.date)
            old = pending[i]
            text = replace_once(text, old, old + add, what=f"amendments_pending[{i}]")
        text = replace_once_raw(
            text,
            f'\n  "draft": {json.dumps(draft["draft"], ensure_ascii=True)},',
            "",
            what="the draft key",
        )
        applied = draft["amendments_applied"]
        sentence = (
            f" On {self.date} ({self.stamp.strftime('%H:%MZ')}) the build-time values were filled in place and "
            f"the file renamed from campaign/v5-preregistered.DRAFT.json, by {SCRIPT} under Fable's freeze "
            "rulings (2026-10-03 ~20:30Z): each of the 29 amendments_pending items kept in its place, with its "
            "'Filled', 'Pending after launch' or 'Not filled' text appended; the draft key dropped; "
            + self.share_text.split(", from the build's")[0]
            + f" against the bound {100 * self.bound:.4f}%. No gate, threshold or population moves."
        )
        return replace_once(text, applied, applied + sentence, what="amendments_applied")


def replace_once_raw(text: str, old: str, new: str, *, what: str) -> str:
    if text.count(old) != 1:
        raise Refusal(f"{what}: its text is in the file {text.count(old)} times, not once")
    return text.replace(old, new)


def replace_once(text: str, old: str, new: str, *, what: str) -> str:
    for ascii_ in (True, False):
        o = json.dumps(old, ensure_ascii=ascii_)
        if text.count(o) == 1:
            return text.replace(o, json.dumps(new, ensure_ascii=ascii_))
    raise Refusal(f"{what}: its JSON string is not in the file exactly once")


def anchor(pending: list) -> None:
    if len(pending) != len(ANCHORS) or not all(isinstance(p, str) for p in pending):
        raise Refusal(
            f"amendments_pending holds {len(pending)} items; this freeze anchors {len(ANCHORS)} strings"
        )
    for i, prefix in enumerate(ANCHORS):
        hits = [j for j, p in enumerate(pending) if p.startswith(prefix)]
        if hits != [i]:
            raise Refusal(f"anchor {i} {prefix!r} matches items {hits}, not exactly item {i}")
        if " Filled " in pending[i] or "Pending after launch" in pending[i]:
            raise Refusal(f"amendments_pending[{i}] is already filled")


FORBIDDEN = re.compile(
    r"first_missing|first_extra|subsequence_utterances|exact_keys|subsequence_keys|request\.bin|pairs\.tsv"
)


def validate(before: dict, after_text: str, items: list[Item]) -> dict:
    after = json.loads(after_text)
    if "draft" in after:
        raise Refusal("the output still carries the draft key")
    old, new = before["amendments_pending"], after["amendments_pending"]
    if len(new) != len(old):
        raise Refusal("the output's amendments_pending changed length")
    for i, (o, w) in enumerate(zip(old, new)):
        if not w.startswith(o) or w == o:
            raise Refusal(f"amendments_pending[{i}] is not its old text plus a fill")
        if FORBIDDEN.search(w[len(o) :]):
            raise Refusal(f"amendments_pending[{i}]'s fill names a list this script must not copy")
    for k in before:
        if k in ("draft", "amendments_pending", "amendments_applied"):
            continue
        if after.get(k) != before[k]:
            raise Refusal(f"top-level {k!r} changed")
    if set(after) != set(before) - {"draft"}:
        raise Refusal("the output's top-level keys are not the DRAFT's less draft")
    if not after["amendments_applied"].startswith(before["amendments_applied"]):
        raise Refusal("amendments_applied lost text")
    return after


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--draft", type=Path, required=True)
    p.add_argument(
        "--repo", type=Path, required=True, help="the checkout the freeze commit is made in"
    )
    p.add_argument("--build-out", type=Path, required=True)
    p.add_argument("--build-ledger", type=Path, required=True)
    p.add_argument("--scan", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--pool-report", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--noul-parts", type=Path, help="data/pool of the build checkout (default: --repo's)"
    )
    p.add_argument("--fetch-record", type=Path)
    p.add_argument("--family-rates", type=Path)
    p.add_argument("--zero-checks", type=Path)
    p.add_argument("--build-log", type=Path)
    p.add_argument("--scan-log", type=Path)
    p.add_argument(
        "--gh200-state", help="the GH200 queue's marker state and its UTC time, as the lead read it"
    )
    p.add_argument(
        "--v4-shards", type=Path, help="v4's shards/train (default: the accounting's shard_dir)"
    )
    p.add_argument(
        "--f-ledger",
        type=Path,
        help=f"F's GH200 ledger with seeds 0-4 (default: --repo's {F_LEDGER})",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--format-test", action="store_true")
    args = p.parse_args(argv)
    if args.noul_parts is None:
        args.noul_parts = args.repo / "data" / "pool"
    try:
        if args.format_test and not args.dry_run:
            raise Refusal("--format-test runs only with --dry-run")
        if not args.dry_run and args.out.exists():
            raise Refusal(f"{args.out} exists; this freeze writes a new file")
        if args.out.resolve() == args.draft.resolve():
            raise Refusal("--out is the DRAFT")
        for path in (args.build_out, args.scan, args.pool):
            if not path.is_dir():
                raise Refusal(f"{path} is not a directory")
        text = args.draft.read_text(encoding="utf-8")
        draft = json.loads(text)
        if "draft" not in draft:
            raise Refusal(f"{args.draft} has no draft key: it is not the DRAFT")
        anchor(draft["amendments_pending"])
        f = Freeze(args)
        f.share_gate()
        f.load_build()
        f.load_parts()
        f.fill()
        out_text = f.apply(text, draft)
        after = validate(draft, out_text, f.items)
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OSError) as exc:
        # An input whose shape this script does not know: refused, loudly, with where.
        traceback.print_exc()
        print(f"REFUSED: an input is not the shape this script reads ({exc!r})", file=sys.stderr)
        return 2
    label = "FORMAT TEST (the numbers are not v5's) " if args.format_test else ""
    print(f"\n== {label}amendments_pending, filled ==")
    for item in f.items:
        print(
            f"\n[{item.idx}] {item.state()}\n  {after['amendments_pending'][item.idx][len(draft['amendments_pending'][item.idx]) :].strip()}"
        )
    print(
        f"\n[amendments_applied] +\n  {after['amendments_applied'][len(draft['amendments_applied']) :].strip()}"
    )
    print("\n== summary ==")
    for item in f.items:
        print(
            f"  item {item.idx:2d}: {item.state()}"
            + (f"; not filled: {' | '.join(item.not_filled)}" if item.not_filled else "")
        )
    mism = [(i.idx, m) for i in f.items for m in i.mismatch]
    print("  differences from the DRAFT's stated expectations: " + ("none" if not mism else ""))
    for i, m in mism:
        print(f"    item {i}: {m}")
    if f.na:
        print("  NOT APPLICABLE under --format-test (each refuses on a real freeze):")
        for m in f.na:
            print(f"    {m}")
    print(f"  output parses; 29 items in order; no draft key; {len(out_text)} characters")
    if args.dry_run:
        print("  --dry-run: nothing written")
        return 0
    args.out.write_text(out_text, encoding="utf-8")
    print(f"  wrote {args.out} (sha256 {sha256_file(args.out)})")
    print(
        "  next, in the same commit: git rm the DRAFT, git add this file and the reader patch's files"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
