"""Throwaway analysis (never ships): could a confidence threshold at inference move F's OOD suite
from 132/61/62-of-180 to the gate's floor while keeping in-distribution abstention under its cap?

Reads F's val and suite verdicts (seeds 0-2, pulled 2026-10-01/02). Confidence is UNCALIBRATED
(temperature 1): a plausibility read of the mechanism, not a fitted threshold. It must not pick
c or a threshold (rule 2): the curve is reported, nothing is chosen from it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/Users/bharath/Code/research/Lappi-decision/build/v5-build-wt/python")
from qd_train.ood import (
    OOD_MAX_IN_DISTRIBUTION_ABSTAIN,
    OOD_MIN_ABSTAIN,
    wilson_interval,
)

BOX = Path("/Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box")
DEFECT = "code.defect_class"
CATS = ("prose", "unseen-language", "scrambled")


def softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def conf(logits: list[float], top: int) -> tuple[float, float]:
    p = softmax(logits)
    second = max(q for i, q in enumerate(p) if i != top)
    return p[top], p[top] - second


def quantiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    pick = lambda q: s[min(len(s) - 1, int(q * (len(s) - 1)))]  # noqa: E731
    return {"n": len(s), "p10": pick(0.1), "p50": pick(0.5), "p90": pick(0.9), "min": s[0]}


def seed(n: int) -> dict[str, object]:
    indist_defect: list[tuple[bool, float, float, bool]] = []  # abstained, p_top, margin, correct
    indist_all: list[tuple[bool, float, float]] = []
    eval_row = None
    for line in (BOX / f"verdicts-s{n}.jsonl").open():
        v = json.loads(line)
        eval_row = v["eval_row_id"]
        if v["kind"] != "choice" or v.get("expected_abstain"):
            continue
        top, noul = v["top"], v["noul_row"]
        abstained = top == noul
        if v.get("perm") is not None and v.get("top_permuted") is not None:
            abstained = abstained or v["top_permuted"] == noul or not v["permutation_agreed"]
        p_top, margin = conf(v["row_logits"], top)
        indist_all.append((abstained, p_top, margin))
        if v["family_id"] == DEFECT:
            indist_defect.append((abstained, p_top, margin, bool(v["correct"])))
    ood: dict[str, list[tuple[bool, float, float]]] = {c: [] for c in CATS}
    for line in (BOX / f"suite-verdicts-s{n}.jsonl").open():
        v = json.loads(line)
        if v.get("suite") != "ood":
            continue
        p1, m1 = conf(v["row_logits_1"], v["top1"])
        p2, m2 = conf(v["row_logits_2"], v["top2"])
        ood[v["category"]].append((bool(v["abstained"]), min(p1, p2), min(m1, m2)))

    base = {
        "eval_row_id": eval_row,
        "ood_abstained": {c: sum(a for a, _, _ in ood[c]) for c in CATS},
        "defect_indist_abstained": [sum(a for a, *_ in indist_defect), len(indist_defect)],
        "pooled_indist_abstained": [sum(a for a, *_ in indist_all), len(indist_all)],
        "p_top_of_answered_ood": {c: quantiles([p for a, p, _ in ood[c] if not a]) for c in CATS},
        "p_top_of_answered_defect_correct": quantiles(
            [p for a, p, _, ok in indist_defect if not a and ok]),
        "p_top_of_answered_defect_wrong": quantiles(
            [p for a, p, _, ok in indist_defect if not a and not ok]),
    }
    curves: dict[str, list[dict[str, object]]] = {}
    for stat, idx in (("p_top", 1), ("margin", 2)):
        grid = [i / 100 for i in range(0, 100)] + [0.995, 0.999]
        rows = []
        for tau in grid:
            in_k = sum(1 for r in indist_defect if r[0] or r[idx] < tau)
            in_pooled = sum(1 for r in indist_all if r[0] or r[idx] < tau)
            per = {c: sum(1 for r in ood[c] if r[0] or r[idx] < tau) for c in CATS}
            ood_k = sum(per.values())
            ood_lo, _ = wilson_interval(ood_k, 180)
            _, in_hi = wilson_interval(in_k, len(indist_defect))
            rows.append({
                "tau": tau, "ood": ood_k, "per_category": per, "ood_wilson_lo": round(ood_lo, 4),
                "defect_indist": in_k, "defect_indist_wilson_hi": round(in_hi, 4),
                "pooled_indist": in_pooled,
                "both_bounds_hold": ood_lo >= OOD_MIN_ABSTAIN
                and in_hi <= OOD_MAX_IN_DISTRIBUTION_ABSTAIN,
            })
        curves[stat] = rows
    best = {}
    for stat, rows in curves.items():
        ok = [r for r in rows if r["defect_indist_wilson_hi"] <= OOD_MAX_IN_DISTRIBUTION_ABSTAIN]
        top = max(ok, key=lambda r: r["ood"]) if ok else None
        best[stat] = {"most_ood_abstained_under_the_in_dist_cap": top,
                      "any_threshold_passes_both": any(r["both_bounds_hold"] for r in rows)}
    return {**base, "best_under_cap": best, "curves": curves}


def main() -> None:
    out = {f"seed{n}": seed(n) for n in (0, 1, 2)}
    path = Path(sys.argv[1])
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, s in out.items():
        print(k, s["eval_row_id"], "ood", s["ood_abstained"], "defect in-dist",
              s["defect_indist_abstained"])
        for c in CATS:
            print("   answered", c, s["p_top_of_answered_ood"][c])
        print("   defect answered correct", s["p_top_of_answered_defect_correct"])
        print("   defect answered wrong  ", s["p_top_of_answered_defect_wrong"])
        for stat, b in s["best_under_cap"].items():
            t = b["most_ood_abstained_under_the_in_dist_cap"]
            print(f"   {stat}: any tau passes both bounds: {b['any_threshold_passes_both']}; "
                  f"most OOD under the cap: tau={t['tau'] if t else None} "
                  f"ood={t['ood'] if t else None} per={t['per_category'] if t else None} "
                  f"defect_in={t['defect_indist'] if t else None}")


if __name__ == "__main__":
    main()
