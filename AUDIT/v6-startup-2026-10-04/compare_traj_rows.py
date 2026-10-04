"""Compare v5's per-snapshot trajectory rows with a v6 --score-plan run's, on the same snapshots.

Analysis only (stdlib; never imported by a tool). It is the check
``python/tests/test_score_plan_trajectory.py`` makes on the toy corpus, made on real rows:

    python3 compare_traj_rows.py --old-ledger L5 --old-dir D5 --new-ledger L6 --new-dir D6 \
        --seed 2 --steps 12176 1000

For each step it finds the one ``trajectory-ood`` eval row whose ``scored_checkpoint`` names
``epoch-seed<seed>-cuda-step<step>.json`` in each ledger, and the per-snapshot suite-verdict file
``traj-s<seed>-step<step>.jsonl`` in each directory. Every row field must be equal except the row
identity and timing (row_id, prev_row_hash, written_at, wall_clock_s, and cost_usd, which the
ledger prices from wall_clock_s; --wall-clock-cap-s is not recorded in the row) and ``notes`` (the
new notes must be the old notes plus one space and the plan's sentence). Every suite line must be
byte-equal once the row id is replaced and the plan's ``score_kind`` key removed. Exit 0 when
everything else is equal, 1 with every difference listed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IDENTITY = ("row_id", "prev_row_hash", "written_at", "wall_clock_s", "cost_usd")


def _rows(ledger: Path) -> list[dict]:
    rows = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if r.get("run_kind") == "eval"
            and (r.get("recipe") or {}).get("tag") == "trajectory-ood"]


def _row_for(rows: list[dict], seed: int, step: int, where: Path) -> dict:
    name = f"epoch-seed{seed}-cuda-step{step}.json"
    found = [r for r in rows
             if str(r["recipe"].get("scored_checkpoint", "")).split(":")[0].endswith(name)]
    if len(found) != 1:
        raise SystemExit(f"{where}: {len(found)} trajectory-ood rows score {name}, not one")
    return found[0]


def _lines(path: Path, row_id: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    out = []
    for raw in text.splitlines():
        line = json.loads(raw)
        if line["eval_row_id"] != row_id:
            raise SystemExit(f"{path}: a line names row {line['eval_row_id']}, not {row_id}")
        line["eval_row_id"] = "<row>"
        line.pop("score_kind", None)
        out.append(json.dumps(line, sort_keys=True))
    return out


def _diff(old: object, new: object, at: str, out: list[str]) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                out.append(f"{at}.{key}: only in {'new' if key in new else 'old'}")
            else:
                _diff(old[key], new[key], f"{at}.{key}", out)
    elif old != new:
        out.append(f"{at}: old {json.dumps(old)[:200]} new {json.dumps(new)[:200]}")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--old-ledger", type=Path, required=True)
    p.add_argument("--old-dir", type=Path, required=True)
    p.add_argument("--new-ledger", type=Path, required=True)
    p.add_argument("--new-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, nargs="+", required=True)
    a = p.parse_args(argv)
    if len(a.steps) < 2:
        raise SystemExit("the parity bar is two snapshots or more")
    old_rows, new_rows = _rows(a.old_ledger), _rows(a.new_ledger)
    bad = 0
    for step in a.steps:
        old = _row_for(old_rows, a.seed, step, a.old_ledger)
        new = _row_for(new_rows, a.seed, step, a.new_ledger)
        diffs: list[str] = []
        if not str(new["notes"]).startswith(f"{old['notes']} Scored as kind 'step{step}'"):
            diffs.append("notes: the new notes are not the old notes plus the plan's sentence")
        _diff({k: v for k, v in old.items() if k not in (*IDENTITY, "notes")},
              {k: v for k, v in new.items() if k not in (*IDENTITY, "notes")}, "row", diffs)
        name = f"traj-s{a.seed}-step{step}.jsonl"
        old_lines = _lines(a.old_dir / name, old["row_id"])
        new_lines = _lines(a.new_dir / name, new["row_id"])
        if old_lines != new_lines:
            differing = sum(x != y for x, y in zip(old_lines, new_lines, strict=False))
            diffs.append(f"suite lines: {len(old_lines)} old, {len(new_lines)} new, "
                         f"{differing} of the paired lines differ")
        verdict = (
            f"{len(diffs)} differences" if diffs else "EQUAL but for the named differences"
        )
        print(f"step {step}: old row {old['row_id']} new row {new['row_id']}: {verdict}")
        for d in diffs:
            print(f"  {d}")
        bad += bool(diffs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
