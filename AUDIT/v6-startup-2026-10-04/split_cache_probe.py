"""Can the split rebuild's RESULT be cached? Size, round-trip time, determinism, verifiability.

Analysis only (stdlib + the repo's own modules). Runs ``tools/real_ft_run.py``'s ``main`` with
the given argv; when ``ft_split_rows`` returns ``(train_rows, val_rows)`` it measures:

- the rebuild's own wall seconds;
- ``pickle`` (protocol 5): bytes, dumps seconds, file write seconds, file read + loads seconds,
  and whether the loaded rows equal the originals;
- a canonical digest independent of pickle's framing: sha256 over each split's rows in order,
  ``row_id`` and ``qd_data.rows.row_content_hash(row)`` per row, and its seconds -- the cost
  of verifying a cached result row by row;
- the val rows against the build's own ``data/pool/val.json`` manifest: every row id present
  and every content hash equal, and its seconds;

then stops (``SystemExit``), before the val set or any model. Run it in two processes and
compare the digests: Python's string hashing is randomised per process, so equal digests
across two runs mean the rebuild's output (order included) does not depend on it.

Usage: python split_cache_probe.py <PICKLE_OUT> <real_ft_run.py argv...>
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import resource
import sys
import time
from pathlib import Path
from typing import Any

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / "tools"))
sys.path.insert(0, str(WT / "python"))

import real_ft_run as rft  # noqa: E402

from qd_data.rows import row_content_hash  # noqa: E402


def _say(text: str) -> None:
    print(f"[probe] {text}", file=sys.stderr, flush=True)


def _rss_gib() -> float:
    # ru_maxrss is bytes on macOS, KiB on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 << 30) if sys.platform == "darwin" else peak / (1 << 20)


def _canonical(splits: dict[str, list[Any]]) -> str:
    h = hashlib.sha256()
    for name in sorted(splits):
        h.update(f"split {name} {len(splits[name])}\n".encode())
        for row in splits[name]:
            h.update(f"{row.row_id} {row_content_hash(row)}\n".encode())
    return h.hexdigest()


def main() -> None:
    pickle_out = Path(sys.argv[1])
    argv = sys.argv[2:]
    out_dir = Path(argv[argv.index("--out") + 1])
    inner = rft.ft_split_rows

    def probe(**kw: Any) -> Any:
        began = time.perf_counter()
        train, val = inner(**kw)
        _say(f"ft_split_rows: {time.perf_counter() - began:.1f} s; train {len(train)} rows, "
             f"val {len(val)} rows; peak RSS {_rss_gib():.1f} GiB")
        splits = {"train": train, "val": val}

        t = time.perf_counter()
        digest = _canonical(splits)
        _say(f"canonical digest (row_id + row_content_hash, in order) {digest}: "
             f"{time.perf_counter() - t:.1f} s")

        t = time.perf_counter()
        blob = pickle.dumps((train, val), protocol=5)
        _say(f"pickle.dumps: {len(blob)} bytes ({len(blob) / (1 << 30):.2f} GiB) in "
             f"{time.perf_counter() - t:.1f} s; sha256 {hashlib.sha256(blob).hexdigest()}")
        t = time.perf_counter()
        with pickle_out.open("wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        _say(f"write + fsync: {time.perf_counter() - t:.1f} s")
        del blob
        t = time.perf_counter()
        raw = pickle_out.read_bytes()
        read_s = time.perf_counter() - t
        t = time.perf_counter()
        loaded = pickle.loads(raw)
        _say(f"read {read_s:.1f} s + pickle.loads {time.perf_counter() - t:.1f} s; peak RSS "
             f"{_rss_gib():.1f} GiB")
        del raw
        t = time.perf_counter()
        same = loaded[0] == train and loaded[1] == val
        _say(f"loaded rows == rebuilt rows: {same} ({time.perf_counter() - t:.1f} s)")

        t = time.perf_counter()
        manifest = json.loads((out_dir / "data" / "pool" / "val.json").read_text(encoding="utf-8"))
        entries = {e["row_id"]: e["content_hash"] for e in manifest["entries"]}
        mism = sum(1 for r in val if entries.get(r.row_id) != row_content_hash(r))
        _say(f"val rows vs data/pool/val.json: {len(val)} rebuilt, {len(entries)} in the "
             f"manifest, {mism} rebuilt rows missing or with another content hash "
             f"({time.perf_counter() - t:.1f} s)")
        raise SystemExit("split_cache_probe: measured; stopping before the val set")

    rft.ft_split_rows = probe
    rft.main(argv)


if __name__ == "__main__":
    main()
