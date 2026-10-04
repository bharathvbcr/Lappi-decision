"""The split cache A/B, read against its pre-registered bars (HANDOFF/v6-startup-2026-10-04.md).

Analysis only (stdlib). Reads ``build/v6-profile/ab-*.log``, the logs of
``build/v6-profile/ab_cache.sh``: one MISS run, then OFF and HIT interleaved three times. Each
log is the run's output with a ``HH:MM:SS `` UTC prefix, and ends with ``/usr/bin/time -l``'s
report. From each log it takes:

- the wall seconds (``real``) and the peak RSS (``maximum resident set size``, in bytes);
- the ``split cache:`` lines;
- the *downstream window*: every line from the first ``val set:`` line to the end. The time
  report, the ``end`` marker and timestamps are removed, and every ``<number> s`` duration is
  replaced by ``<t> s``, because the OOD suite's own MinHash/LSH lines print their seconds.

It then states each pre-registered bar with its numbers:

1. min(OFF) - min(HIT) >= 60 s;
2. HIT's peak RSS <= OFF's peak RSS (max over each arm's runs);
3. MISS wall <= min(OFF) + 30 s, and MISS peak RSS <= OFF peak RSS + 2 GiB;
4. every HIT run's downstream window equals every OFF run's.

It exits 0 for GO and 1 for NO-GO.

Usage: python ab_compare.py <DIR holding ab-*.log>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

GIB = 1 << 30
_STAMP = re.compile(r"^\d\d:\d\d:\d\d ")
_REAL = re.compile(r"^\s*([0-9.]+) real\s")
_RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size")
_DURATION = re.compile(r"\b\d+(\.\d+)? s\b")
_TIME_REPORT = re.compile(
    r"^\s*\d+(\.\d+)?\s+(real|user|sys)\b|^\s*-?\d+\s+[a-z].*(size|reclaims|faults|swaps|operations"
    r"|sent|received|signals|switches|retired|elapsed|footprint)"
)


def read(path: Path) -> dict[str, object]:
    wall = rss = None
    cache_lines: list[str] = []
    window: list[str] = []
    in_window = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = _STAMP.sub("", raw, count=1)
        if (m := _REAL.match(line)) is not None:
            wall = float(m.group(1))
            continue
        if (m := _RSS.match(line)) is not None:
            rss = int(m.group(1))
            continue
        if _TIME_REPORT.match(line) or line.startswith(("start ab-", "end ab-")):
            continue
        if line.startswith("split cache:"):
            cache_lines.append(line)
            continue
        if line.startswith("val set:"):
            in_window = True
        if in_window:
            window.append(_DURATION.sub("<t> s", line))
    if wall is None or rss is None:
        raise SystemExit(f"{path}: no /usr/bin/time -l report")
    if not window:
        raise SystemExit(f"{path}: no 'val set:' line, so no downstream window")
    return {"wall": wall, "rss": rss, "cache": cache_lines, "window": window}


def main(directory: Path) -> int:
    runs = {p.stem: read(p) for p in sorted(directory.glob("ab-*.log"))}
    off = [v for k, v in runs.items() if k.startswith("ab-off")]
    hit = [v for k, v in runs.items() if k.startswith("ab-hit")]
    miss = runs.get("ab-miss")
    if miss is None or len(off) != 3 or len(hit) != 3:
        raise SystemExit(f"want ab-miss and three each of ab-off/ab-hit, found {sorted(runs)}")
    for name, r in runs.items():
        print(f"{name:8} wall {r['wall']:7.2f} s  peak RSS {r['rss'] / GIB:6.2f} GiB  "
              f"window {len(r['window'])} lines")
        for line in r["cache"]:
            print(f"         {line[:230]}")
    min_off = min(float(r["wall"]) for r in off)
    min_hit = min(float(r["wall"]) for r in hit)
    rss_off = max(int(r["rss"]) for r in off)
    rss_hit = max(int(r["rss"]) for r in hit)
    saving = min_off - min_hit
    bars = [
        (f"1. saving min(OFF) - min(HIT) = {min_off:.2f} - {min_hit:.2f} = {saving:.2f} s >= 60 s",
         saving >= 60.0),
        (f"2. HIT peak RSS {rss_hit / GIB:.2f} GiB <= OFF peak RSS {rss_off / GIB:.2f} GiB",
         rss_hit <= rss_off),
        (f"3. MISS wall {float(miss['wall']):.2f} s <= min(OFF) + 30 = {min_off + 30:.2f} s, and "
         f"MISS peak RSS {int(miss['rss']) / GIB:.2f} GiB <= OFF peak + 2 = "
         f"{(rss_off + 2 * GIB) / GIB:.2f} GiB",
         float(miss["wall"]) <= min_off + 30 and int(miss["rss"]) <= rss_off + 2 * GIB),
    ]
    reference = off[0]["window"]
    differing = [k for k, r in runs.items() if k != "ab-miss" and r["window"] != reference]
    bars.append((f"4. downstream windows equal across OFF and HIT ({len(reference)} lines): "
                 f"{'all equal' if not differing else 'differ: ' + ', '.join(differing)}",
                 not differing))
    for text, ok in bars:
        print(f"{'PASS' if ok else 'FAIL'}  {text}")
    print(f"(not a bar) MISS's downstream window equals OFF's: {miss['window'] == reference}")
    go = all(ok for _, ok in bars)
    print("GO" if go else "NO-GO")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
