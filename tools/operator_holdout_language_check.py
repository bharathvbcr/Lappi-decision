"""Does holding out an operator remove a LANGUAGE rather than a generator?

The operator-holdout arms show rung 0 collapsing on a held-out operator's rows while its
same-class siblings hold, which reads as the model classifying by recognising the generator.
The obvious confound is language. ``stub.panic`` suggests Rust; if the operator were
language-bound, holding it out would remove that language's training data and the collapse
would say nothing about generators.

Two things have to be true for the confound to be dead, and only the second is interesting:

1. the operator is not language-bound -- measured, ``stub.panic`` is 89% Python;
2. the holdout arm and its size-matched control remove comparable amounts of each language.

The second is what makes the pair a controlled comparison rather than two arms that happen
to share a row count. Both arms are the same size by construction; if the holdout arm also
removed most of the Python and the size-matched one did not, "the Python went" would explain
the gap as well as "the fingerprint went".

Replicates the run's own split and both of its filters -- ``split_by_file`` and
``filter_train_rows``, the same functions, the same ``--drop-random-seed`` -- so the counts
are the training sets the arms actually trained on and not corpus-level estimates.

Usage::

    python tools/operator_holdout_language_check.py \\
        --examples /home/ubuntu/commitpackft-corpus-v2/examples.jsonl \\
        --operator stub.panic
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from rung0_real_run import (  # noqa: E402
    DEFAULT_VAL_SHARE,
    filter_train_rows,
    split_by_file,
)


def _langs(rows: list[dict]) -> collections.Counter[str]:
    return collections.Counter(str(r.get("language") or "?") for r in rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--val-share", type=float, default=DEFAULT_VAL_SHARE)
    parser.add_argument("--drop-random-seed", type=int, default=0)
    args = parser.parse_args(argv)

    examples = [
        json.loads(line)
        for line in args.examples.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not examples:
        raise SystemExit(f"{args.examples} holds no examples")

    train_raw, val_raw = split_by_file(examples, val_share=args.val_share)
    print(f"corpus {len(examples)}: train {len(train_raw)}, val {len(val_raw)}")

    held, held_note = filter_train_rows(train_raw, hold_out_operator=args.operator)
    n_dropped = len(train_raw) - len(held)
    print(f"  {held_note}")
    # The size-matched arm's OWN draw, through the same filter with the same seed, so this
    # is the set it trained on rather than a fresh sample that would differ from it.
    sized, sized_note = filter_train_rows(
        train_raw, drop_random_train=n_dropped, drop_random_seed=args.drop_random_seed
    )
    print(f"  {sized_note}")

    full, hl, sl = _langs(train_raw), _langs(held), _langs(sized)
    print(
        f"\n{'language':<12}{'in train':>10}{'holdout kept':>14}{'sizematch kept':>16}"
        f"{'holdout cut':>13}{'sizematch cut':>15}"
    )
    for lang in sorted(full):
        print(
            f"{lang:<12}{full[lang]:>10}{hl[lang]:>14}{sl[lang]:>16}"
            f"{full[lang] - hl[lang]:>13}{full[lang] - sl[lang]:>15}"
        )
    print(
        f"{'TOTAL':<12}{len(train_raw):>10}{len(held):>14}{len(sized):>16}"
        f"{n_dropped:>13}{n_dropped:>15}"
    )
    print(
        "\nBoth arms are the same size by construction. Read the last two columns: where "
        "they are close, language volume cannot explain a difference in the arms' scores, "
        "and what is left is which rows went."
    )
    # Validation is never filtered, so both arms are scored on identical rows -- stated
    # rather than assumed, because the whole comparison rests on it.
    print(
        f"Validation is untouched by both filters: {len(val_raw)} row(s), "
        f"{dict(_langs(val_raw))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
