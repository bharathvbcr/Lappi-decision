"""Throwaway (never ships): in-distribution selective risk on F's defect family. For each p_top
threshold, how many defect val choice rows are answered, how many of those are wrong, and how
many abstain. Uncalibrated (temperature 1). Reported, nothing chosen from it (rule 2)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

BOX = Path("/Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box")


def p_top(logits: list[float], top: int) -> float:
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    return e[top] / sum(e)


out = {}
for n in (0, 1, 2):
    rows = []
    for line in (BOX / f"verdicts-s{n}.jsonl").open():
        v = json.loads(line)
        if (v["kind"] != "choice" or v.get("expected_abstain")
                or v["family_id"] != "code.defect_class"):
            continue
        noul = v["noul_row"]
        abst = v["top"] == noul or v["top_permuted"] == noul or not v["permutation_agreed"]
        rows.append((abst, p_top(v["row_logits"], v["top"]), bool(v["correct"])))
    table = []
    for tau in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        answered = [(p, ok) for a, p, ok in rows if not a and p >= tau]
        wrong = sum(1 for _, ok in answered if not ok)
        table.append({"tau": tau, "rows": len(rows), "answered": len(answered),
                      "abstained": len(rows) - len(answered), "answered_wrong": wrong,
                      "risk": round(wrong / len(answered), 4) if answered else None})
    out[f"seed{n}"] = table
    print(f"seed{n}")
    for r in table:
        print("  ", r)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
