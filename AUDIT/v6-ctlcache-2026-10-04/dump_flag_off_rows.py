"""Write the flag-off letter and option control rows of the split-cache test fixture, normalized,
so the rows written before and after the change can be compared byte for byte.

    QD_PREP_BIN=<abs> python AUDIT/v6-ctlcache-2026-10-04/dump_flag_off_rows.py OUT.json

The fixture is ``python/tests/test_ft_linear_control_split_cache.py``'s: one eval row, its
verdicts and a fake ``real_ft_run`` whose rebuild returns a fixed split. Run once against the tool
before the change and once after, on one machine, with one ``QD_PREP_BIN``; ``cmp`` the outputs.

The comparison drops only what must differ between two runs of one row, and says so:

* the row's identity and timing: ``row_id``, ``prev_row_hash``, ``written_at``, ``wall_clock_s``;
* code provenance: ``code_commit`` and ``metrics.code_that_ran``. Both name the code that ran,
  and the tool's own bytes change in this lane by construction;
* the fixture's eval row id, which is a fresh uuid each time the fixture writes its eval row. It
  appears in ``recipe.eval_row_id``, ``metrics.scored_eval_row_id`` and ``notes`` and is
  replaced by ``<EVAL_ROW_ID>`` everywhere.

Everything else stays in: the protocol, recipe, metrics, gates, controls, env, cost, notes, and
the exit codes.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / "python" / "tests"))
sys.path.insert(0, str(WT / "tools"))
sys.path.insert(0, str(WT / "python"))

import ft_linear_control as ftc  # noqa: E402
from test_ft_linear_control_split_cache import IDENTITY, build_fixture  # noqa: E402

from qd_train.ledger import CODE_THAT_RAN, Ledger  # noqa: E402

DROPPED = (*IDENTITY, "code_commit")


def normalized(row: dict, eval_id: str) -> dict:
    out = {k: v for k, v in row.items() if k not in DROPPED}
    out["metrics"] = {k: v for k, v in row["metrics"].items() if k != CODE_THAT_RAN}
    return json.loads(json.dumps(out).replace(eval_id, "<EVAL_ROW_ID>"))


def main(argv: list[str]) -> int:
    (out_path,) = argv
    with tempfile.TemporaryDirectory(prefix="ctlcache-dump-") as tmp_dir:
        tmp = Path(tmp_dir)
        fixture = build_fixture(tmp)
        (eval_row,) = Ledger(fixture.ledger).rows()
        mp = pytest.MonkeyPatch()
        try:
            fixture.runner.install(mp)
            codes = {
                "letter": ftc.main(fixture.argv(write="letter")),
                "option": ftc.main(fixture.argv("--option-control", write="option")),
            }
        finally:
            mp.undo()
        rows = {name: normalized(fixture.row(name), eval_row.row_id) for name in codes}
        dump = {"exit_codes": codes, "rows": rows,
                "rebuild_calls": len(fixture.runner.rebuilds),
                "split_rebuild_inputs_calls": len(fixture.runner.inputs)}
    Path(out_path).write_text(json.dumps(dump, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_path}: exit codes {codes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
