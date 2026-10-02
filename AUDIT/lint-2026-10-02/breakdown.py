"""L-lint throwaway: per-file and per-rule counts from a concise ruff report.

    python AUDIT/lint-2026-10-02/breakdown.py <concise-report> <label> <gate command>

Prints the gate command, the label, the total, then counts by file and by rule.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROW = re.compile(r"^(\S+):\d+:\d+: (\S+) ")


def main() -> int:
    report, label, command = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = [m for line in Path(report).read_text().splitlines() if (m := ROW.match(line))]
    by_file = Counter(m.group(1) for m in rows)
    by_rule = Counter(m.group(2) for m in rows)
    print(f"label: {label}")
    print(f"gate command: {command}")
    print(f"source: {report}")
    print(f"total findings: {len(rows)} in {len(by_file)} files")
    print()
    print("by file:")
    for path, n in sorted(by_file.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:4d}  {path}")
    print()
    print("by rule:")
    for rule, n in sorted(by_rule.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:4d}  {rule}")
    print()
    print("by file and rule:")
    pairs = Counter((m.group(1), m.group(2)) for m in rows)
    for (path, rule), n in sorted(pairs.items()):
        print(f"  {n:4d}  {rule:7s} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
