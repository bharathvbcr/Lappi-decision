"""Mutations of a valid span-refit row line; Python's verdict on each."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
from qd_train.ledger import Ledger  # noqa: E402

FIX = REPO / "crates/qd-train/tests/fixtures/span-refit-gate-cases.json"
BASE = json.loads(next(c for c in json.loads(FIX.read_text()) if c["name"] == "new_sha_valid")["line"])
NEW = BASE["recipe"]["script_sha256"]["span_refit_score.py"]
OLD = "b88715afc031a2ef38aead1b0f0320d7d0763d4a990532ac8015c0652d7e07d5"
S, E, N = ("val_span_start_top1_over_chance", "val_span_end_top1_over_chance",
           "val_span_pointing_added_line_chance")


def mut(f):
    d = copy.deepcopy(BASE)
    f(d)
    return d


def setm(name, **kw):
    def f(d):
        d["metrics"][name].update(kw)
    return f


def delm(name, key):
    def f(d):
        del d["metrics"][name][key]
    return f


cases: dict[str, dict] = {}
cases["tag_case"] = mut(lambda d: (d["recipe"].update(tag="Span-Refit-a"), d["metrics"].clear()))
cases["tag_leading_space"] = mut(lambda d: (d["recipe"].update(tag=" span-refit-a"), d["metrics"].clear()))
cases["tag_trailing_space"] = mut(lambda d: (d["recipe"].update(tag="span-refit-a "), d["metrics"].clear()))
cases["tag_exact_prefix_only"] = mut(lambda d: (d["recipe"].update(tag="span-refit-"), d["metrics"].clear()))
cases["tag_no_dash"] = mut(lambda d: (d["recipe"].update(tag="span-refit"), d["metrics"].clear()))
cases["tag_list"] = mut(lambda d: (d["recipe"].update(tag=["span-refit-a"]), d["metrics"].clear()))
cases["sha_upper"] = mut(lambda d: (d["recipe"]["script_sha256"].update({"span_refit_score.py": NEW.upper()}), d["metrics"].clear()))
cases["sha_space"] = mut(lambda d: (d["recipe"]["script_sha256"].update({"span_refit_score.py": NEW + " "}), d["metrics"].clear()))
cases["sha_key_case"] = mut(lambda d: d["recipe"]["script_sha256"].update({"Span_refit_score.py": NEW, "span_refit_score.py": OLD}) )
cases["sha_under_other_key"] = mut(lambda d: d["recipe"].update(script_sha256={"tools/span_refit_score.py": NEW}) or d["metrics"].clear())
cases["old_sha_forged_top_new_field"] = mut(lambda d: (d["recipe"]["script_sha256"].update({"span_refit_score.py": OLD}), d["recipe"].update(scorer_sha256=NEW), d["metrics"].clear()))
cases["old_sha_plus_forged_nested"] = mut(lambda d: (d["recipe"]["script_sha256"].update({"span_refit_score.py": OLD, "span_refit_score.py.new": NEW}), d["metrics"].clear()))
cases["sha_is_list"] = mut(lambda d: (d["recipe"].update(script_sha256=[NEW]), d["metrics"].clear()))
cases["sha_not_string"] = mut(lambda d: (d["recipe"]["script_sha256"].update({"span_refit_score.py": 5}), d["metrics"].clear()))
cases["no_script_sha256"] = mut(lambda d: (d["recipe"].pop("script_sha256"), d["metrics"].clear()))
cases["no_recipe"] = mut(lambda d: (d.pop("recipe"), d["metrics"].clear()))
cases["recipe_null"] = mut(lambda d: (d.update(recipe=None), d["metrics"].clear()))
# gate claims
gp = {"state": "ran", "passed": True, "value": 5596.0}
gf = {"state": "ran", "passed": False, "value": 5000.0}
cases["gate_passed_no_metrics"] = mut(lambda d: (d["gates"].update({"span_refit.validity_gate": gp}), d["metrics"].clear()))
cases["gate_failed_no_metrics"] = mut(lambda d: (d["gates"].update({"span_refit.validity_gate": gf}), d["metrics"].clear()))
cases["gate_failed_other_arm_b"] = mut(lambda d: (d["gates"].update({"span_refit.validity_gate": gf}), d["recipe"].update(tag="span-refit-b"), d["metrics"].clear()))
cases["gate_failed_but_partial_metrics"] = mut(lambda d: (d["gates"].update({"span_refit.validity_gate": gf}), d["metrics"].pop(S)))
cases["gate_not_run_no_metrics"] = mut(lambda d: (d["gates"].update({"span_refit.validity_gate": {"state": "not_run", "reason": "x"}}), d["metrics"].clear()))
cases["gate_passed_str_false"] = mut(lambda d: (d["gates"].update({"span_refit.validity_gate": {"state": "ran", "passed": "false", "value": 1.0}}), d["metrics"].clear()))
cases["gate_wrongly_named"] = mut(lambda d: (d["gates"].update({"span_refit.validity_gate ": gf}), d["metrics"].clear()))
cases["gate_in_metrics_not_gates"] = mut(lambda d: (d["metrics"].clear(), d["metrics"].update({"span_refit.validity_gate": gf})))
cases["gate_list"] = mut(lambda d: (d.update(gates=[]), d["metrics"].clear()))
cases["gate_null"] = mut(lambda d: (d.update(gates=None), d["metrics"].clear()))
cases["gates_missing"] = mut(lambda d: (d.pop("gates"), d["metrics"].clear()))
# count fields
for nm, val in [("str", "50"), ("neg", -1), ("float", 50.0), ("float_frac", 50.5), ("bool", True), ("zero", 0), ("null", None), ("huge", 10**30), ("neg_float", -0.0)]:
    cases[f"ntotal_{nm}_null_metric"] = mut(setm(N, n_total=val))
    cases[f"ntotal_{nm}_all"] = mut(lambda d, val=val: [d["metrics"][k].update(n_total=val) for k in (S, E, N)])
    cases[f"n_{nm}_start"] = mut(setm(S, n=val))
cases["n_gt_ntotal"] = mut(setm(S, n=51))
cases["n_missing_only"] = mut(delm(S, "n"))
cases["ntotal_missing_only"] = mut(delm(S, "n_total"))
cases["ntotal_50_vs_50.0"] = mut(setm(E, n_total=50.0))
cases["passed_false_metric"] = mut(setm(S, passed=False))
cases["value_nan_str"] = mut(setm(S, value="NaN"))
cases["value_missing"] = mut(delm(S, "value"))
cases["state_case"] = mut(setm(S, state="Ran"))
cases["status_case"] = mut(lambda d: d.update(status="Completed"))
cases["status_started"] = mut(lambda d: (d.update(status="started"), d["metrics"].clear()))
cases["status_missing"] = mut(lambda d: d.pop("status"))
cases["metric_name_case"] = mut(lambda d: d["metrics"].update({S.upper(): d["metrics"].pop(S)}))
cases["extra_metrics_only"] = mut(lambda d: (d["metrics"].clear(), d["metrics"].update({"val_top1.span": {"state": "ran", "passed": True, "value": 1.0, "n": 1, "n_total": 2}})))


def verdict(raw: dict) -> str:
    line = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "l.jsonl"
        p.write_bytes(line.encode() + b"\n")
        try:
            Ledger(p).verify_chain()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "span-null scorer" in msg or "pointing-row count" in msg:
                return "refuse"
            return "other:" + type(e).__name__
    return "accept"


if __name__ == "__main__":
    out = []
    for name, raw in cases.items():
        line = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        out.append({"name": name, "python": verdict(raw), "line": line})
        print(f"{name:40s} {out[-1]['python']}")
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
