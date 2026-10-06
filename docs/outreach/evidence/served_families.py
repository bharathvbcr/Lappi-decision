"""Read-only, throwaway analysis (never ships): what the v0.1 preview's headline numbers cover.

The release notes report choice top-1 0.857 and two-fold coverage 91.9% at 87.0% precision over
all 17,254 val choice rows of score row 6af73bef. Those rows include code.defect_class, which the
release does not serve (GAP-PREVIEW-DOES-NOT-SERVE-CODE-DEFECT-CLASS-2026-10-06). This splits both
numbers by family so public text can state what a caller of the release actually gets.

    python3 docs/outreach/evidence/served_families.py \
        <campaign>/preview-2026-10-05/v5-avg-verdicts/verdicts-avg.jsonl \
        <campaign>/preview-2026-10-05/calib-two-fold/rows.jsonl

Definitions follow build/v6x/coverage.py: a row is answered when it is not
abstained_noul_row_or_margin, precision is correct among answered, and unscored rows are skipped
(the qd-calib-fit report skips them too). The script reproduces the release notes' all-rows
figures as a check before it prints the split.
"""

import collections
import json
import sys

UNSERVED = "code.defect_class"

verdicts_path, rows_path = sys.argv[1], sys.argv[2]

n = collections.Counter()
top1 = collections.Counter()
family_of = {}
for line in open(verdicts_path, encoding="utf-8"):
    v = json.loads(line)
    if v.get("kind") != "choice":
        continue
    f = v["family_id"]
    family_of[v["row_id"]] = f
    n[f] += 1
    top1[f] += bool(v.get("correct"))

scored = collections.Counter()
answered = collections.Counter()
answered_ok = collections.Counter()
unscored = 0
for line in open(rows_path, encoding="utf-8"):
    r = json.loads(line)
    if not r.get("scored"):
        unscored += 1
        continue
    f = family_of[r["row_id"]]
    scored[f] += 1
    if not r["abstained_noul_row_or_margin"]:
        answered[f] += 1
        answered_ok[f] += bool(r["correct"])


def top1_line(label, keep):
    rows = sum(n[f] for f in n if keep(f))
    ok = sum(top1[f] for f in n if keep(f))
    print(f"{label}: choice top-1 {ok}/{rows} = {ok / rows:.4f}")


def coverage_line(label, keep):
    rows = sum(scored[f] for f in scored if keep(f))
    a = sum(answered[f] for f in scored if keep(f))
    c = sum(answered_ok[f] for f in scored if keep(f))
    print(
        f"{label}: two-fold at the 0.80 target, answered {a}/{rows} ({a / rows:.1%}), "
        f"precision on answered {c}/{a} ({c / a:.1%})"
    )


def every(_f):
    return True


def served(f):
    return f != UNSERVED


print(f"families with choice rows: {len(n)}; unscored two-fold rows skipped: {unscored}")
top1_line("all choice rows (the release notes' figure)", every)
top1_line(f"served families only (excluding {UNSERVED})", served)
coverage_line("all choice rows (the release notes' figure)", every)
coverage_line(f"served families only (excluding {UNSERVED})", served)
print()
print(f"{'family':32s} {'rows':>6s} {'top-1':>6s}")
for f in sorted(n):
    mark = "  (not served)" if f == UNSERVED else ""
    print(f"{f:32s} {n[f]:6d} {top1[f] / n[f]:6.3f}{mark}")
