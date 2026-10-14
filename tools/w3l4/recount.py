"""Recount gaps.jsonl: open ids by latest record, and dump open records for triage. Read-only."""

import json
import sys
from pathlib import Path

from qd_train.gaps import CLOSED_STATUSES, current_records, read_gaps

path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
cur = current_records(path)
opn = [i for i, r in cur.items() if r["status"] not in CLOSED_STATUSES]
print(json.dumps({"lines": len(read_gaps(path)), "ids": len(cur), "open": len(opn)}))
if len(sys.argv) > 2:
    out = Path(sys.argv[2])
    out.write_text(
        "\n".join(json.dumps(cur[i], ensure_ascii=False) for i in opn) + "\n", encoding="utf-8"
    )
