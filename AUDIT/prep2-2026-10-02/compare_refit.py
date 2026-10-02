"""Report-only refit rows beside F's linear-control rows of record (lane L-prep2, 2026-10-02).

For each refit row in ledger/mac-linear-control-refit-2026-10-02.jsonl, the F row of record
for the same eval row in ledger/gh200-p4-v4-2026-10-01.jsonl (c89b89a1, c0e438e6, 3f72112a):
the gate's state, every task's linear-control convergence, and every paired margin. Nothing
here is a gate reading: F's rows of record stay what they are (the human's answer 4).

    python AUDIT/prep2-2026-10-02/compare_refit.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REFIT = REPO / "ledger" / "mac-linear-control-refit-2026-10-02.jsonl"
F_LEDGER = REPO / "ledger" / "gh200-p4-v4-2026-10-01.jsonl"
TOOL = "tools/ft_linear_control.py"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _control_row(rows: list[dict], eval_row: str) -> dict:
    """The letter-control row (not --option-control, whose recipe names its ``control``) for
    ``eval_row``."""
    found = [r for r in rows if r.get("recipe", {}).get("tool") == TOOL
             and r["recipe"].get("eval_row_id") == eval_row
             and "control" not in r["recipe"]]
    if len(found) != 1:
        raise SystemExit(f"{len(found)} letter-control rows for eval row {eval_row}")
    return found[0]


def _summary(row: dict) -> dict:
    # Metric names are flat: "<metric>.<task or family>".
    metrics = row["metrics"]
    conv_prefix, margin_prefix = "linear_control_convergence.", "paired_margin_vs_linear."
    conv = {name[len(conv_prefix):]: m.get("detail") or f"not_run: {m.get('reason')}"
            for name, m in metrics.items() if name.startswith(conv_prefix)}
    margins = {name[len(margin_prefix):]: (m.get("value"), m.get("detail") or m.get("reason"))
               if m["state"] == "ran" else ("not_run", m.get("reason"))
               for name, m in metrics.items() if name.startswith(margin_prefix)}
    gate = row["gates"]["paired_margin_vs_linear"]
    return {
        "row_id": row["row_id"],
        "gate": {k: gate.get(k) for k in ("state", "passed", "value", "n", "n_total", "detail",
                                          "reason") if k in gate},
        "max_iter": row["recipe"].get("max_iter"),
        "control_engine_sha256": row["recipe"].get("control_engine_sha256"),
        "linear_control_convergence": conv,
        "paired_margin_vs_linear": margins,
        "wall_clock_s": row.get("wall_clock_s"),
        "notes": row.get("notes"),
    }


def main() -> int:
    f_rows = _rows(F_LEDGER)
    out = []
    for refit in _rows(REFIT):
        eval_row = refit["recipe"]["eval_row_id"]
        out.append({
            "eval_row_id": eval_row,
            "refit_report_only": _summary(refit),
            "f_row_of_record": _summary(_control_row(f_rows, eval_row)),
        })
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
