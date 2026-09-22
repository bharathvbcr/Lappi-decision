"""Does the unified diff carry the change, or does it carry the label?

## Why this exists

``GAP-RUNG0-CHOICE-SEES-THE-POST-IMAGE-NOT-THE-DIFF`` recorded a 38-point gap: the same
control, on the same rows, scored 55.8% reading the post-image and 93.8% reading the
unified diff. That number was measured over all four classes, and over all four classes it
cannot mean what it appears to mean.

``qd-mutate`` emits ``clean`` rows with ``before == after`` and ``diff: String::new()``
(``crates/qd-mutate/src/generate.rs``). So on this corpus ``diff == ""`` holds for exactly
the 8,450 ``clean`` rows and for no other row -- an exact biconditional, measured, not
assumed. A model told to read the diff can answer ``clean`` by measuring the length of its
input, without reading a byte of it.

That is not a corpus defect. A clean row genuinely has no change, so its unified diff is
genuinely empty. It is a defect in what the four-way comparison can be read to prove, and
the fix is to measure the part the leak cannot reach.

## What is measured

* **all four classes** -- reproduces the contaminated comparison, so the recorded number
  can be checked rather than merely doubted;
* **the three mutated classes only** -- every row has a non-empty diff here, so the leak
  cannot operate. If the diff still wins, the diff carries signal and repairing ``clean``'s
  representation is worth doing. If it does not, the 38 points were the leak.

Both halves are reported whatever they say. A measurement that is only run when it is
expected to confirm something is not a measurement.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from rung0_real_run import DEFAULT_CONTEXT_BYTES, split_by_file  # noqa: E402

from qd_train.baseline import LinearBaseline  # noqa: E402
from qd_train.tristate import Ran  # noqa: E402

EXAMPLES: Final[Path] = REPO / "data" / "pool" / "commitpackft-corpus" / "examples.jsonl"
N_SAMPLE: Final[int] = 9_000
SEED: Final[int] = 0
MAX_ITER: Final[int] = 6_000
CLEAN: Final[str] = "clean"


def window(text: str, *, max_bytes: int) -> str:
    """The same tail-keeping truncation ``encode_context`` applies, on the same bytes.

    Comparing against a control that saw more of the file than the model does would be
    comparing two different tasks.
    """
    raw = text.encode("utf-8")
    if len(raw) > max_bytes:
        raw = raw[len(raw) - max_bytes :]
    return raw.decode("utf-8", errors="replace")


def score(train: list[dict], val: list[dict], *, field: str, tag: str) -> float | None:
    """Fit the control on one field and report top-1, per class. ``None`` if it refused."""
    train_docs = [window(r[field], max_bytes=DEFAULT_CONTEXT_BYTES) for r in train]
    train_lab = [r["class"] for r in train]
    val_docs = [window(r[field], max_bytes=DEFAULT_CONTEXT_BYTES) for r in val]
    val_lab = [r["class"] for r in val]

    model = LinearBaseline(seed=SEED, max_iter=MAX_ITER, dense_budget_bytes=8 * 1024**3)
    started = time.monotonic()
    model.fit(train_docs, train_lab)
    took = time.monotonic() - started
    conv = model.convergence()
    if not (isinstance(conv, Ran) and conv.passed):
        # Not scored rather than scored badly: an unconverged control is a weak opponent,
        # and a weak opponent flatters whatever it is compared against.
        print(f"  {tag:>22}: NOT RUN -- the control did not converge in {took:.0f}s")
        return None

    predicted = model.predict(val_docs)
    ok = [p == g for p, g in zip(predicted, val_lab, strict=True)]
    accuracy = sum(ok) / len(ok)

    per: dict[str, list[int]] = {}
    for good, gold in zip(ok, val_lab, strict=True):
        row = per.setdefault(gold, [0, 0])
        row[0] += int(good)
        row[1] += 1

    assert model.fit_ is not None  # convergence() above already refused an unfitted model
    print(
        f"  {tag:>22}: top-1 {accuracy:.2%} of {len(ok)} rows "
        f"({took:.0f}s, {model.fit_.iterations} iterations)"
    )
    for cls in sorted(per):
        good, total = per[cls]
        print(f"      {cls:>9} {good / total:7.2%}  ({good}/{total})")
    empty = sum(1 for d in val_docs if not d.strip())
    if empty:
        print(f"      [{empty} validation row(s) carry an EMPTY context]")
    return accuracy


def main() -> int:
    if not EXAMPLES.is_file():
        raise SystemExit(f"no corpus at {EXAMPLES}")
    rows = [
        json.loads(line)
        for line in EXAMPLES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"corpus: {len(rows)} example(s) from {EXAMPLES}")

    empty_all = Counter(r["class"] for r in rows if r["diff"] == "")
    nonempty_clean = sum(1 for r in rows if r["class"] == CLEAN and r["diff"] != "")
    print(f"empty diff, whole corpus, by class: {dict(empty_all)}")
    print(f"clean rows WITH a non-empty diff:   {nonempty_clean}")

    sample = random.Random(SEED).sample(rows, N_SAMPLE)
    print(f"\nsample: {len(sample)}  {dict(Counter(r['class'] for r in sample))}")
    train, val = split_by_file(sample, val_share=0.25)
    print(f"repo-disjoint split: {len(train)} train / {len(val)} val\n")

    print("== all four classes: the comparison as GAP-...-NOT-THE-DIFF recorded it ==")
    after_all = score(train, val, field="after", tag="after / post-image")
    diff_all = score(train, val, field="diff", tag="diff / unified")

    mut_train = [r for r in train if r["class"] != CLEAN]
    mut_val = [r for r in val if r["class"] != CLEAN]
    print(
        f"\n== the three mutated classes: every row has a non-empty diff, so the "
        f"length leak cannot operate ({len(mut_train)} train / {len(mut_val)} val) =="
    )
    after_mut = score(mut_train, mut_val, field="after", tag="after / post-image")
    diff_mut = score(mut_train, mut_val, field="diff", tag="diff / unified")

    majority = Counter(r["class"] for r in mut_train).most_common(1)[0][0]
    prior = sum(r["class"] == majority for r in mut_val) / len(mut_val)
    print(f"\n  majority prior on the mutated-only validation set: {prior:.2%} ({majority})")

    if None not in (after_all, diff_all, after_mut, diff_mut):
        print(
            f"\n  four-class gap   {diff_all - after_all:+.2%}"
            f"   (this is the number the gap record carries)"
            f"\n  mutated-only gap {diff_mut - after_mut:+.2%}"
            f"   (this is the number the leak cannot reach)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
