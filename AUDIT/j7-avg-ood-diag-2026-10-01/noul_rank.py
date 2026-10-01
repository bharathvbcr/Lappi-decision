"""Where the noul row stands on each OOD case: its rank and its gap to the top row.

Throwaway analysis (never shipped, never imported): reads ``real_ft_run.py`` suite-verdict
JSONL -- the ``ood_abstain`` gate's lines and the ``--score-plan`` OOD diagnostic's
(``ood_abstain.diagnostic``) -- and reports, per scoring kind, per category and per pass:

* the noul row's RANK among the case's rows (0 = it won, so the runtime abstains);
* its GAP to the top row: ``logit[top] - logit[noul]`` (0 when noul won). For one model the
  lines carry raw logits and for a logit ensemble mean log-probabilities; either way the
  difference is a log-probability ratio in nats, because a log-softmax only shifts a row by
  a constant -- so the gaps of a seed, an average and an ensemble are comparable;
* the row that beat it, and the signed ``gap_to_rival``: the best OTHER row's logit minus
  noul's -- positive by how much noul lost, negative by how much it won. The summaries
  average this one.

Every line is checked before it is counted: ``top1``/``top2`` must be the argmax of the
logits the line carries, the noul row the last of ``rows`` (``noul_first`` is never used on
the OOD suite), and both passes present. A line that fails is a refusal, not a skip.

Usage::

    python3 noul_rank.py --kind seed0=2f5fe57a --kind avg=1d93b3ee FILE.jsonl [...]
        [--out SUMMARY.json] [--cases CASES.jsonl]

``--kind NAME=EVAL_ROW_ID_PREFIX`` names the kinds; a line that carries ``score_kind`` (a
``--score-plan`` line) is named by it and needs no ``--kind``. A line whose row is named by
neither is a refusal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GATES = ("ood_abstain", "ood_abstain.diagnostic")
CATEGORIES = ("prose", "unseen-language", "scrambled")
MAX_BYTES = 256 * 1024 * 1024


def _argmax(xs: list[float]) -> int:
    best = 0
    for i, x in enumerate(xs):
        if x > xs[best]:
            best = i
    return best


def _pass(line: dict, k: int, where: str) -> dict:
    logits = line.get(f"row_logits_{k}")
    top = line.get(f"top{k}")
    rows, noul = line.get("rows"), line.get("noul_row")
    if not (isinstance(logits, list) and logits
            and all(isinstance(x, (int, float)) for x in logits)):
        raise SystemExit(f"{where}: pass {k} carries no row_logits_{k}")
    if rows != len(logits) or noul != rows - 1:
        raise SystemExit(f"{where}: pass {k} has {len(logits)} logits, rows {rows}, noul {noul}")
    if top != _argmax(logits):
        raise SystemExit(
            f"{where}: top{k} = {top} is not the argmax {_argmax(logits)} of its logits"
        )
    best = max(logits)
    rank = sum(1 for x in logits if x > logits[noul])
    others = [i for i in range(len(logits)) if i != noul]
    rival = max(others, key=lambda i: (logits[i], -i))
    return {
        "noul_rank": rank, "gap": best - logits[noul], "top": top,
        "noul_won": top == noul, "rival": rival,
        # The margin by which noul would have to rise to win: its gap to the best OTHER row.
        "gap_to_rival": logits[rival] - logits[noul],
    }


def _summary(values: list[dict]) -> dict:
    gaps = [v["gap_to_rival"] for v in values]
    ranks = defaultdict(int)
    for v in values:
        ranks[v["noul_rank"]] += 1
    return {
        "n": len(values),
        "noul_won": sum(v["noul_won"] for v in values),
        "rank_histogram": {str(r): ranks[r] for r in sorted(ranks)},
        "gap_to_rival": {
            "mean": statistics.fmean(gaps), "median": statistics.median(gaps),
            "min": min(gaps), "max": max(gaps),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--kind", action="append", default=[], metavar="NAME=ROW_PREFIX")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cases", type=Path, default=None)
    args = parser.parse_args(argv)
    named: dict[str, str] = {}
    for spec in args.kind:
        name, _, prefix = spec.partition("=")
        if not name or len(prefix) < 8:
            raise SystemExit(f"--kind {spec!r}: NAME=EVAL_ROW_ID_PREFIX, the prefix 8+ chars")
        named[prefix] = name

    cases: list[dict] = []
    seen: set[tuple[str, str]] = set()
    rows_by_kind: dict[str, set[str]] = defaultdict(set)
    inputs: list[dict] = []
    for path in args.files:
        if path.stat().st_size > MAX_BYTES:
            raise SystemExit(f"{path}: larger than {MAX_BYTES} bytes")
        data = path.read_bytes()
        inputs.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
        for n, raw in enumerate(data.decode("utf-8").splitlines(), 1):
            line = json.loads(raw)
            if line.get("gate") not in GATES:
                continue
            where = f"{path}:{n}"
            row = str(line["eval_row_id"])
            kind = line.get("score_kind") or next(
                (name for prefix, name in named.items() if row.startswith(prefix)), None
            )
            if kind is None:
                raise SystemExit(f"{where}: eval row {row[:8]} is named by no --kind")
            if line.get("category") not in CATEGORIES:
                raise SystemExit(f"{where}: category {line.get('category')!r}")
            key = (kind, str(line["case_id"]))
            if key in seen:
                raise SystemExit(f"{where}: case {key} seen twice")
            seen.add(key)
            rows_by_kind[kind].add(row)
            for k in (1, 2):
                if k == 2 and line.get("top2") is None:
                    continue  # no derangement: the case has one pass
                cases.append({
                    "kind": kind, "eval_row_id": row, "case_id": line["case_id"],
                    "category": line["category"], "pass": k,
                    "abstained": bool(line["abstained"]), **_pass(line, k, where),
                })
    multi = {k: sorted(v) for k, v in rows_by_kind.items() if len(v) > 1}
    if multi:
        raise SystemExit(f"a kind spans more than one eval row: {multi}")

    by: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for c in cases:
        by[(c["kind"], c["category"], c["pass"])].append(c)
        by[(c["kind"], "all", c["pass"])].append(c)
    summary = {
        kind: {
            "eval_row_id": next(iter(rows_by_kind[kind])),
            "abstained": {
                cat: sum(1 for c in cases if c["kind"] == kind and c["pass"] == 1
                         and c["abstained"] and cat in (c["category"], "all"))
                for cat in (*CATEGORIES, "all")
            },
            "passes": {
                str(p): {cat: _summary(by[(kind, cat, p)])
                         for cat in (*CATEGORIES, "all") if by[(kind, cat, p)]}
                for p in (1, 2)
            },
        }
        for kind in sorted(rows_by_kind)
    }
    body = json.dumps({"inputs": inputs, "kinds": summary}, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(body + "\n", encoding="utf-8")
    if args.cases is not None:
        args.cases.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in cases),
                              encoding="utf-8")
    print(f"{'kind':8s} {'category':16s} pass  n   noul-won  rank0..4            "
          "gap-to-rival mean / median (nats)")
    for kind in sorted(summary):
        for p in ("1", "2"):
            for cat, s in summary[kind]["passes"][p].items():
                hist = [s["rank_histogram"].get(str(r), 0) for r in range(5)]
                g = s["gap_to_rival"]
                print(f"{kind:8s} {cat:16s} {p:>4s} {s['n']:3d} {s['noul_won']:5d}     "
                      f"{hist!s:20s} {g['mean']:8.3f} / {g['median']:8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
