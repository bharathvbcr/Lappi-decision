"""Print the latest record for each id given on argv, in full. Read-only."""

import json
import sys

from qd_train.gaps import current_records

cur = current_records()
for gid in sys.argv[1:]:
    rec = cur.get(gid)
    print(f"=== {gid}")
    print(json.dumps(rec, ensure_ascii=False, indent=1) if rec else "MISSING")
