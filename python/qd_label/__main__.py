"""`python -m qd_label` — label diffs by hand, resumably.

Four subcommands:

    label    present unlabelled diffs and record a decision for each
    stats    progress, class balance, pace
    ceiling  intra-rater kappa: the bound the teacher's kappa is measured against
    export   write the held-out manifest

The labelling loop is deliberately keyboard-only and one key per decision. 300 diffs
at 20 seconds each is under two hours; at 60 seconds each it is five, and the
difference is entirely in how much the tool makes you type.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from .session import VALID_LABELS, LabelItem, LabelSession

KEYS: dict[str, str] = {
    "s": "stub",
    "l": "logic",
    "c": "cosmetic",
    "k": "clean",
    "u": "unsure",
}

# ANSI, disabled when not a tty so a redirected log stays readable.
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def load_pool(path: Path) -> list[LabelItem]:
    """Accepts a JSON object with an `items` array, or a JSONL file of items."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        doc = json.loads(raw)
        rows = doc["items"] if isinstance(doc, dict) else doc
    return [
        LabelItem(
            item_id=str(r["item_id"]),
            repo=str(r["repo"]),
            path=str(r["path"]),
            language=str(r["language"]),
            diff=str(r["diff"]),
        )
        for r in rows
    ]


def load_disjoint_ids(paths: Sequence[str]) -> set[str]:
    """Item ids this session's pool must not contain.

    Accepts either a **pool** file or a **store** file — the append-only decisions JSONL —
    because at the moment the second set is opened the first may exist as either: a pool
    that has been drawn but not yet labelled, or a store part-way through. Both carry
    `item_id`, and reading whichever exists is what makes the check usable in practice
    rather than only in the order the plan imagined.
    """
    ids: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            doc = json.loads(text)
            rows = doc["items"] if isinstance(doc, dict) else doc
        for r in rows:
            if "item_id" in r:
                ids.add(str(r["item_id"]))
    return ids


def render_diff(diff: str, *, max_lines: int = 120) -> str:
    out: list[str] = []
    lines = diff.splitlines()
    shown = lines[:max_lines]
    for line in shown:
        if line.startswith("+") and not line.startswith("+++"):
            out.append(_c("32", line))
        elif line.startswith("-") and not line.startswith("---"):
            out.append(_c("31", line))
        elif line.startswith("@@"):
            out.append(_c("36", line))
        else:
            out.append(line)
    if len(lines) > max_lines:
        # Say how much was hidden. A silently truncated diff is one the labeller
        # judges without knowing they only saw part of it.
        out.append(_c("33", f"... {len(lines) - max_lines} more lines hidden of {len(lines)} total"))
    return "\n".join(out)


def cmd_label(args: argparse.Namespace) -> int:
    pool = load_pool(Path(args.pool))
    session = LabelSession(
        pool, args.store, purpose=args.purpose, repeat_fraction=args.repeat_fraction, seed=args.seed,
        disjoint_from=load_disjoint_ids(args.disjoint_from)
    )
    pending = len(session.pending())
    if pending == 0:
        print(f"All {len(pool)} items already labelled. Nothing to do.")
        return 0

    print(f"{pending} of {len(pool)} items left. Keys: "
          f"[s]tub  [l]ogic  [c]osmetic  [k]=clean  [u]nsure  [q]uit\n")

    for item, presentation in session.next_items(limit=args.limit):
        tag = _c("33", "  [repeat — consistency check]") if presentation == 2 else ""
        print("=" * 78)
        print(f"{_c('1', item.item_id)}  {item.repo}  {item.path}  ({item.language}){tag}")
        print("=" * 78)
        print(render_diff(item.diff, max_lines=args.max_lines))
        print()

        started = time.monotonic()
        while True:
            try:
                key = input("label> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nstopped; everything so far is saved")
                return 0
            if key == "q":
                print("stopped; everything so far is saved")
                return 0
            if key in KEYS:
                break
            print(f"  unrecognised {key!r}. One of: {', '.join(KEYS)} or q to stop.")

        label = KEYS[key]
        note = ""
        if label == "unsure":
            while not note.strip():
                note = input("  why unsure (required)> ").strip()
        elif args.always_note:
            note = input("  note (optional)> ").strip()

        session.record(
            item.item_id, label, note=note,
            seconds=time.monotonic() - started, presentation=presentation,
        )
        print(f"  -> {_c('1', label)}\n")

    print("\n" + str(session.stats()))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    session = LabelSession(
        load_pool(Path(args.pool)), args.store, purpose=args.purpose,
        disjoint_from=load_disjoint_ids(args.disjoint_from),
    )
    print(session.stats())
    return 0


def cmd_ceiling(args: argparse.Namespace) -> int:
    """Intra-rater kappa — the bound the teacher's agreement is measured against.

    Deliberately does NOT reuse `kappa_gate`: that function's wording is written for
    the teacher gate ("rewrite the rubric"), which is nonsense advice about your own
    self-consistency. Same statistic, different question, so different prose.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from qd_train.agreement import kappa_with_ci

    session = LabelSession(
        load_pool(Path(args.pool)), args.store, purpose=args.purpose,
        disjoint_from=load_disjoint_ids(args.disjoint_from),
    )
    first, second = session.intra_rater_pairs()
    if len(first) < 2:
        print(
            f"Only {len(first)} repeat pair(s) on file — not enough to estimate a ceiling.\n"
            "Keep labelling; repeats are offered automatically once more than ten items are done."
        )
        return 1

    exact = sum(a == b for a, b in zip(first, second, strict=True))
    result = kappa_with_ci(first, second, seed=args.seed)

    print(f"self-agreement on {len(first)} re-presented item(s): "
          f"{exact}/{len(first)} identical ({exact / len(first):.0%})")
    print(f"  {result}")
    print()
    print("This is the CEILING on any teacher's kappa against you: a labeller cannot")
    print("agree with you more than you agree with yourself.")
    print()

    if len(set(first)) < 2:
        # p_e -> 1 makes kappa 0/0, which cohens_kappa reports as 0.0. Saying
        # "your self-agreement is zero" here would be actively wrong.
        print("CAUTION: every re-presented item got the same label, so kappa is undefined")
        print("  (chance agreement is already 100%) and reads as 0.00. That is a property of")
        print("  the sample, not of you. Label a more varied set before reading a ceiling.")
    elif len(first) < 20:
        print(f"CAUTION: {len(first)} pairs is too few to estimate this precisely. The interval")
        print("  above is the honest width. Treat it as a smoke test, not a measurement.")
    elif result.ci_high < 0.7:
        print("Your own labelling is inconsistent enough that a teacher scoring ~0.6 against")
        print("  you would be near the ceiling. Tighten the rubric's boundary cases first —")
        print("  the ambiguity is in the task definition, not in the model.")
    else:
        print(f"A teacher kappa should be read against this: {result.kappa:.2f} is the most any")
        print("  labeller could score against you on this sample.")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    session = LabelSession(
        load_pool(Path(args.pool)), args.store, purpose=args.purpose,
        disjoint_from=load_disjoint_ids(args.disjoint_from),
    )
    manifest = session.export_manifest(args.out)
    print(
        f"wrote {args.out}: {manifest['n_labelled']} labelled, "
        f"{manifest['n_usable']} usable, {manifest['n_unsure']} unsure, held_out=True"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qd_label", description=__doc__)
    p.add_argument("--pool", required=True, help="JSON/JSONL of items to label")
    p.add_argument("--store", required=True, help="append-only JSONL of decisions")
    p.add_argument("--purpose", choices=("heldout", "agreement"), default="heldout")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--disjoint-from",
        dest="disjoint_from",
        action="append",
        default=[],
        metavar="PATH",
        help="a pool or store this pool must not overlap. Repeatable. The held-out set and "
        "the agreement set measure different things and cannot share items; without this "
        "the constraint is maintained by memory across two invocations and 600 items, and "
        "an overlap is silent until after the labelling time is spent.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    lab = sub.add_parser("label", help="label unlabelled items")
    lab.add_argument("--limit", type=int, default=None, help="stop after this many presentations")
    lab.add_argument("--max-lines", type=int, default=120, dest="max_lines")
    lab.add_argument("--repeat-fraction", type=float, default=0.1, dest="repeat_fraction")
    lab.add_argument("--always-note", action="store_true", dest="always_note")
    lab.set_defaults(func=cmd_label)

    sub.add_parser("stats", help="progress and class balance").set_defaults(func=cmd_stats)
    sub.add_parser("ceiling", help="intra-rater kappa").set_defaults(func=cmd_ceiling)

    exp = sub.add_parser("export", help="write the held-out manifest")
    exp.add_argument("--out", required=True)
    exp.set_defaults(func=cmd_export)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.pool).exists():
        print(f"pool not found: {args.pool}", file=sys.stderr)
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
