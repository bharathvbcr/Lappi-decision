"""Has the control learned the change, or learned the generator?

## The suspicion

A char-n-gram logistic regression reaches 88.5% on the four-way diff task
(`AUDIT/after-vs-diff-leak.md`), and its per-class ordering tracks how distinctive the
mutation operator's textual fingerprint is: stub 98.1% (an inserted `panic!()`), logic
89.5% (`<` becomes `<=`), cosmetic 69.3% (a rename, no fixed signature), clean 68.5% (a
real commit, no signature at all).

``qd-mutate`` applies exactly one operator per example, so `logic.widen_comparison` leaves
`-  x > 0` / `+  x <= 0` in the diff. Hashed char n-grams read that directly. If that is
what the control is doing, then "the model loses to the control by 12 points" is partly a
statement that this task is easy for n-grams, and not only that the model is weak.

## The test

Hold out whole **operators** rather than whole repos, and score the same rows twice:

* **seen** — the operator is in training (repo-disjoint split only, as every arm so far);
* **unseen** — every example produced by that operator is removed from training.

Both score the identical validation rows, so the difference is attributable to the
operator's absence and nothing else. Other operators of the same class stay in training, so
the label is still reachable: this asks whether the control can recognise `logic` from an
operator it has never seen, which is what "has learned the change" would mean.

A control that has memorised generator fingerprints collapses on **unseen**. One that has
learned what the classes mean does not.

## What it cannot say

This measures the CONTROL, not the model. If the control collapses, the 88.5% is inflated
by generator recognition and the margin against it is not a clean statement about the
architecture -- but the model may be doing exactly the same thing, and this does not check.
The same experiment against the trained model needs a GPU and is not this tool.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from after_vs_diff import window  # noqa: E402
from rung0_real_run import DEFAULT_CONTEXT_BYTES, split_by_file  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_train.baseline import LinearBaseline  # noqa: E402
from qd_train.mutate_adapter import read_example_objects  # noqa: E402
from qd_train.tristate import Ran  # noqa: E402

DEFAULT_EXAMPLES: Final[Path] = (
    REPO / "data" / "pool" / "commitpackft-corpus-v2" / "examples.jsonl"
)
SEED: Final[int] = 0
MAX_ITER: Final[int] = 6_000

#: One operator per class that has a fingerprint, plus one that should not. `cosmetic.
#: rename_local` is the control-within-the-control: it leaves no fixed lexical signature,
#: so if the collapse were an artifact of removing training data rather than of removing a
#: fingerprint, it would show up here too.
DEFAULT_OPERATORS: Final[tuple[str, ...]] = (
    "stub.panic",
    "logic.change_constant",
    "cosmetic.rename_local",
)


def fit_and_score(
    train: list[dict], val: list[dict], *, tag: str
) -> tuple[float, int] | None:
    """Top-1 over `val`, or `None` if the control refused to converge."""
    train_docs = [window(r["diff"], max_bytes=DEFAULT_CONTEXT_BYTES) for r in train]
    train_lab = [r["class"] for r in train]
    val_docs = [window(r["diff"], max_bytes=DEFAULT_CONTEXT_BYTES) for r in val]
    val_lab = [r["class"] for r in val]

    model = LinearBaseline(seed=SEED, max_iter=MAX_ITER, dense_budget_bytes=8 * 1024**3)
    started = time.monotonic()
    model.fit(train_docs, train_lab)
    took = time.monotonic() - started
    conv = model.convergence()
    if not (isinstance(conv, Ran) and conv.passed):
        print(f"    {tag}: NOT RUN -- the control did not converge in {took:.0f}s")
        return None

    predicted = model.predict(val_docs)
    hits = sum(p == g for p, g in zip(predicted, val_lab, strict=True))
    print(
        f"    {tag:>8}: {hits / len(val_lab):7.2%} of {len(val_lab)} row(s)  "
        f"({len(train_docs)} train doc(s), {took:.0f}s)"
    )
    return hits / len(val_lab), len(val_lab)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--sample", type=int, default=12_000)
    parser.add_argument("--operator", action="append", default=[])
    args = parser.parse_args(argv)

    # Through rule 3's door rather than around it -- see read_example_objects. This tool
    # fits the control that sets the bar the model is measured against, so reading held-out
    # data here is the same violation as training on it, wearing a different hat.
    rows = read_example_objects(args.examples, config=DataConfig(), repo_root=REPO)
    print(f"corpus: {len(rows)} example(s) from {args.examples}")
    operators = tuple(args.operator) or DEFAULT_OPERATORS

    present = Counter(r.get("operator") for r in rows if r.get("operator"))
    missing = [op for op in operators if op not in present]
    if missing:
        # Fail closed: a typo would otherwise remove nothing from training and report the
        # seen and unseen conditions as identical, which reads as "no collapse".
        raise SystemExit(
            f"these operators produced no example: {missing}. Present: "
            f"{sorted(present)[:12]}"
        )

    sample = random.Random(SEED).sample(rows, min(args.sample, len(rows)))
    train_raw, val_raw = split_by_file(sample, val_share=0.25)
    print(f"sample {len(sample)}, repo-disjoint: {len(train_raw)} train / {len(val_raw)} val\n")

    for operator in operators:
        target = [r for r in val_raw if r.get("operator") == operator]
        if len(target) < 30:
            print(f"{operator}: only {len(target)} validation row(s); skipped as too few\n")
            continue
        without = [r for r in train_raw if r.get("operator") != operator]
        removed = len(train_raw) - len(without)
        # Removing an operator removes training rows, and `stub.panic` alone is 43% of this
        # corpus. Without a size-matched control the comparison confounds "the fingerprint
        # is gone" with "half the data is gone", and the second explains a collapse just as
        # well as the first. So a third condition drops the same NUMBER of rows at random,
        # keeping the operator. `unseen` against `matched` is the measurement; `seen` is
        # printed for reference only.
        matched = random.Random(SEED + 1).sample(train_raw, len(without))
        print(
            f"{operator}: {len(target)} validation row(s), "
            f"{removed} of {len(train_raw)} training row(s) removed "
            f"({removed / len(train_raw):.0%})"
        )
        seen = fit_and_score(train_raw, target, tag="seen")
        matched_score = fit_and_score(matched, target, tag="matched")
        unseen = fit_and_score(without, target, tag="unseen")
        if matched_score and unseen:
            drop = matched_score[0] - unseen[0]
            verdict = (
                "the control leaned on this operator's fingerprint"
                if drop > 0.20
                else "no collapse: the class is reachable without this operator"
            )
            print(f"    {'drop':>8}: {drop:+.2%} against the size-matched control -- {verdict}")
            if seen:
                print(
                    f"    {'(volume)':>8}: {seen[0] - matched_score[0]:+.2%} of the raw "
                    "seen-vs-unseen gap is attributable to training size alone"
                )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
