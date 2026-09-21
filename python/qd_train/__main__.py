"""`qd-ledger`: read the decision record.

docs/ledger-schema.md names this command -- *"`qd-ledger verify` recomputes the chain
and reports the first break"* -- but nothing implemented it, so `Ledger.verify_chain`,
`Ledger.promotion_verdict`, `PromotionVerdict.__str__` and `Ran.coverage_str` were
reachable only from tests. The decision reads the ledger, and until now there was no
way to read the ledger except from a Python REPL.

**This tool renders; it never judges.** It prints the tri-state each row already
records, in the schema's own vocabulary (`ran passed=true`, `not_run`), rather than a
word like PASS of its own invention. The one verdict it shows -- promotion -- is
computed by `Ledger.promotion_verdict` from the recorded rows; this module adds no
condition of its own and cannot discharge a gate.

Exit codes: 0 the answer is yes / intact, 1 the answer is no / broken, 2 the question
could not be asked (missing or unreadable ledger). A 1 and a 2 are deliberately
different: "this ledger refuses to promote" and "I could not read this ledger" must
never share an exit code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ledger import Ledger, LedgerChainError, LedgerRow
from .tristate import NotRun, TriState

__all__ = ["build_parser", "main"]


def _render(t: TriState) -> str:
    """One tri-state, in the schema's vocabulary, always carrying its coverage.

    The schema's Coverage rule says a report renders `n/n_total`; `Ran.coverage_str`
    is that renderer, and this is the report.
    """
    if isinstance(t, NotRun):
        return f"not_run          reason={t.reason}"
    bits = [f"ran passed={str(t.passed).lower():<5}", f"[{t.coverage_str()}]"]
    if t.value is not None:
        bits.append(f"value={t.value}")
    if t.detail:
        bits.append(t.detail)
    return "  ".join(bits)


def _render_row(row: LedgerRow) -> str:
    lines = [
        f"row {row.row_id}",
        f"  written_at   {row.written_at}",
        f"  run_kind     {row.run_kind}    status {row.status}",
        f"  protocol     {row.protocol_hash[:16]}...  seed {row.protocol.seed}",
        f"  seed_family  {row.protocol.hash_without_seed()[:16]}...",
        f"  code_commit  {row.code_commit}",
        f"  quick        {row.quick}" + (f" ({row.quick_reason})" if row.quick else ""),
        f"  wall_clock_s {row.wall_clock_s:.1f}    cost_usd {row.cost_usd:.2f}",
        f"  noul_rate    {_render(row.noul_rate)}",
    ]
    for label, group in (("gate", row.gates), ("control", row.controls), ("metric", row.metrics)):
        for name, t in sorted(group.items()):
            lines.append(f"  {label:<8} {name:<28} {_render(t)}")
    return "\n".join(lines)


def _open(args: argparse.Namespace) -> Ledger:
    return Ledger(args.ledger)


def cmd_verify(args: argparse.Namespace) -> int:
    led = _open(args)
    try:
        led.verify_chain()
    except LedgerChainError as exc:
        print(f"CHAIN BROKEN: {exc}", file=sys.stderr)
        return 1
    n = len(led.raw_lines())
    print(f"chain intact over {n} row(s)")
    return 0


def cmd_promotion(args: argparse.Namespace) -> int:
    led = _open(args)
    verdict = led.promotion_verdict(args.seed_family)
    print(verdict)
    return 0 if verdict.promoted else 1


def cmd_families(args: argparse.Namespace) -> int:
    """Seed families present, so `promotion` has a hash to be given."""
    rows = _open(args).rows()
    if not rows:
        print("no rows", file=sys.stderr)
        return 1
    families: dict[str, list[LedgerRow]] = {}
    for r in rows:
        families.setdefault(r.protocol.hash_without_seed(), []).append(r)
    for fam, members in sorted(families.items()):
        seeds = sorted(r.protocol.seed for r in members)
        print(f"{fam}  {len(members)} row(s)  seeds {seeds}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    rows = _open(args).rows()
    if not rows:
        print("no rows", file=sys.stderr)
        return 1
    for r in rows:
        if args.row_id and r.row_id != args.row_id:
            continue
        print(_render_row(r))
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qd-ledger", description=__doc__)
    p.add_argument("--ledger", required=True, help="path to runs.jsonl")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify", help="recompute the chain and report the first break").set_defaults(
        func=cmd_verify
    )

    prom = sub.add_parser("promotion", help="may this seed family promote a decision?")
    prom.add_argument("--seed-family", required=True, dest="seed_family")
    prom.set_defaults(func=cmd_promotion)

    sub.add_parser("families", help="seed families present, with their seeds").set_defaults(
        func=cmd_families
    )

    show = sub.add_parser("show", help="render rows, with every tri-state's coverage")
    show.add_argument("--row-id", default=None, dest="row_id")
    show.set_defaults(func=cmd_show)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.ledger).exists():
        print(f"ledger not found: {args.ledger}", file=sys.stderr)
        return 2
    try:
        return int(args.func(args))
    except LedgerChainError as exc:
        # A row that will not parse is not an empty ledger and must not read as one.
        print(f"ledger unreadable: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError) as exc:
        print(f"ledger unreadable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
