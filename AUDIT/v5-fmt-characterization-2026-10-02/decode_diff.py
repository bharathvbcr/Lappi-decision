"""Throwaway analysis (never shipped): explain every byte by which the 60-pair build moved from
format 1 (old, 92e1c74) to format 2 (new, v5-build), per
GAP-L-V5FMT-CHARACTERIZATION-CRITERION-STALE-2026-10-02.

Each shard set's sequences are decoded through its own remap and the pinned tokenizer, and each
new text must equal the old text with exactly two edits: the `<|qd_prompt_format|>2` line
inserted after `<|qd_begin|>`, and the question line moved from before `<|qd_context_begin|>`
to after `<|qd_context_end|>`. Supervision arrays and JSON files are diffed field by field, and
every file the two builds wrote is listed as identical or differing.

Usage: python decode_diff.py <old out> <new out>   (run from the v5-build tree's python env)
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

WT = Path("/Users/bharath/Code/research/Lappi-decision/build/v5-build-wt")
sys.path.insert(0, str(WT / "tools"))
sys.path.insert(0, str(WT / "python"))
sys.path.insert(0, str(WT / "python" / "tests"))

from real_tokenizer_pipeline import RealTokenizer  # noqa: E402
from test_containment_exclusions import _normalised, build_digest  # noqa: E402

from qd_train.artifacts import RemapTable  # noqa: E402

BEGIN = "<|qd_begin|>\n"
FORMAT_LINE = "<|qd_prompt_format|>2\n"
QUESTION = "<|qd_question|>"
CTX_BEGIN = "<|qd_context_begin|>\n"
CTX_END = "<|qd_context_end|>\n"


def expected_format2(old: str) -> str:
    """The format-2 text that `old` (format 1) becomes under the two intended edits."""
    if not old.startswith(BEGIN):
        raise ValueError("old text does not start with the begin line")
    q_at = old.index("\n" + QUESTION) + 1
    q_end = old.index("\n", q_at) + 1
    question_line = old[q_at:q_end]
    if old[q_end:q_end + len(CTX_BEGIN)] != CTX_BEGIN:
        raise ValueError("format-1 question line is not followed by the context block")
    without_q = old[:q_at] + old[q_end:]
    ce = without_q.index("\n" + CTX_END) + 1 + len(CTX_END)
    moved = without_q[:ce] + question_line + without_q[ce:]
    return BEGIN + FORMAT_LINE + moved[len(BEGIN):]


def json_diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else str(k)
            if k not in a:
                out.append(f"  + {p} = {json.dumps(b[k])[:160]}")
            elif k not in b:
                out.append(f"  - {p} (was {json.dumps(a[k])[:160]})")
            else:
                out.extend(json_diff(a[k], b[k], p))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(json_diff(x, y, f"{path}[{i}]"))
        return out
    if a != b:
        return [f"  ~ {path}: {json.dumps(a)[:120]} -> {json.dumps(b)[:120]}"]
    return []


def sequences(shard: Path, tok: RealTokenizer) -> tuple[list[str], np.ndarray]:
    offsets = np.load(shard / "offsets.npy")
    ids = np.fromfile(shard / "tokens.u32", dtype="<u4")
    remap = RemapTable.read(shard / "remap")
    texts = [
        tok.decode(remap.new_to_old[ids[offsets[i]:offsets[i + 1]].astype(np.int64)])
        for i in range(len(offsets) - 1)
    ]
    return texts, np.diff(offsets)


def main() -> int:
    old_out, new_out = Path(sys.argv[1]), Path(sys.argv[2])
    print(f"old digest {build_digest(old_out)}")
    print(f"new digest {build_digest(new_out)}")
    old_files = {p.relative_to(old_out) for p in old_out.rglob("*") if p.is_file()}
    new_files = {p.relative_to(new_out) for p in new_out.rglob("*") if p.is_file()}
    for rel in sorted(old_files - new_files):
        print(f"ONLY OLD {rel}")
    for rel in sorted(new_files - old_files):
        print(f"ONLY NEW {rel}")
    differing = []
    for rel in sorted(old_files & new_files):
        same = (hashlib.sha256(_normalised(old_out / rel)).digest()
                == hashlib.sha256(_normalised(new_out / rel)).digest())
        print(f"{'same' if same else 'DIFF'} {rel}")
        if not same:
            differing.append(rel)

    unexplained = 0
    tok = RealTokenizer.load(memo_limit=0)
    for rel in differing:
        if rel.suffix == ".json":
            a = json.loads((old_out / rel).read_text(encoding="utf-8"))
            b = json.loads((new_out / rel).read_text(encoding="utf-8"))
            print(f"--- json diff {rel}")
            for line in json_diff(a, b):
                print(line)
    for shard in sorted({(old_out / r).parent for r in old_files if r.name == "tokens.u32"}):
        rel = shard.relative_to(old_out)
        old_t, old_len = sequences(shard, tok)
        new_t, new_len = sequences(new_out / rel, tok)
        print(f"--- shard {rel}: {len(old_t)} old / {len(new_t)} new sequences")
        if len(old_t) != len(new_t):
            unexplained += 1
            print("  UNEXPLAINED: sequence counts differ")
            continue
        bad = [i for i, (o, n) in enumerate(zip(old_t, new_t)) if expected_format2(o) != n]
        print(f"  decode check: {len(old_t) - len(bad)}/{len(old_t)} sequences equal the "
              "old text with the format line inserted and the question line moved")
        for i in bad[:3]:
            unexplained += 1
            print(f"  UNEXPLAINED seq {i}:\n    expected {expected_format2(old_t[i])!r}\n"
                  f"    got      {new_t[i]!r}")
        unexplained += max(0, len(bad) - 3)
        delta = new_len - old_len
        print(f"  token-count delta per sequence: {dict(Counter(delta.tolist()))}")
        with np.load(shard / "supervision.npz") as zo, \
                np.load(new_out / rel / "supervision.npz") as zn:
            for key in sorted(set(zo.files) | set(zn.files)):
                if key not in zo.files or key not in zn.files:
                    unexplained += 1
                    print(f"  UNEXPLAINED supervision key {key} on one side only")
                    continue
                a, b = zo[key], zn[key]
                if a.shape != b.shape:
                    unexplained += 1
                    print(f"  UNEXPLAINED supervision {key}: shape {a.shape} -> {b.shape}")
                elif np.array_equal(a, b):
                    print(f"  supervision {key}: identical")
                elif key == "target_index":
                    ok = np.array_equal(b - a, delta.astype(b.dtype))
                    unexplained += 0 if ok else 1
                    print(f"  supervision target_index: shifted by each sequence's token delta: "
                          f"{ok}")
                else:
                    d = (b.astype(np.int64) - a.astype(np.int64)).ravel()
                    print(f"  supervision {key}: differs; deltas {dict(Counter(d.tolist()).most_common(8))}")
    print(f"UNEXPLAINED total: {unexplained}")
    return 1 if unexplained else 0


if __name__ == "__main__":
    raise SystemExit(main())
