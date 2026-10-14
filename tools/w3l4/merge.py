"""Wave-3 L4 repair: an open record's re-check must not hide the record's own earlier fields.

evidence.py appended re-check records for open ids that carried only question/status/answer, so the
latest line per id lost the earlier finding, impact and workaround text. Append-only means the
repair is a further line: the record as it stood before this lane, plus the re-check under
`recheck_wave3_l4`. Dry run unless --apply.
"""

import sys

from qd_train.gaps import append_gap, read_gaps

APPLY = "--apply" in sys.argv
LANE = "wave3-l4-gaps-remainder (ft-5c92df4cc91)"

lines = read_gaps()
last: dict[str, dict] = {}
before: dict[str, dict] = {}
for rec in lines:
    gid = rec["id"]
    if rec.get("lane") != LANE:
        before[gid] = rec
    last[gid] = rec

fixed = 0
for gid, rec in last.items():
    if rec.get("lane") != LANE or rec["status"] != "open" or "duplicate_of" in rec:
        continue
    if "answer" not in rec or "checked" not in rec:
        continue
    base = dict(before[gid])
    recheck = {"checked": rec["checked"], "evidence": rec["answer"]}
    for key in ("board_card", "owner", "evidence_commit", "residual"):
        if key in rec:
            recheck[key] = rec[key]
    merged = {**base, "recheck_wave3_l4": recheck}
    if "owner" in rec:
        merged["owner"] = rec["owner"]
    if "board_card" in rec:
        merged["board_card"] = rec["board_card"]
    if not merged.get("question"):
        merged["question"] = rec["question"]
    fixed += 1
    if APPLY:
        append_gap(merged)
    else:
        print(gid)
print("merged", fixed)
