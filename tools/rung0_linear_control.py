"""The control arm for rung 0: can the weakest reasonable model find the class signal?

Rung 0's choice head converges to the entropy of the label distribution -- it has learned
the class prior and nothing conditional on the context
(``GAP-RUNG0-CHOICE-HEAD-CONVERGES-TO-THE-CLASS-PRIOR``). Two explanations remained and
they call for opposite work:

* the signal IS in the context and ``ByteDecider`` cannot extract it, which is an
  architecture or optimisation problem;
* the signal is not there at this corpus size, which is a task or corpus problem and no
  amount of model tuning addresses.

A held-out number from the model under test cannot separate those, because both produce the
same number. A control can. :class:`qd_train.baseline.LinearBaseline` -- multinomial
logistic regression on hashed char n-grams -- sees exactly the bytes the model sees and is
the weakest thing that could possibly work. If it beats the majority class on the same
held-out **files**, the signal is there.

## The control has to have trained

``LinearBaseline.convergence()`` returns ``NotRun`` when the fit did not converge, and this
tool stops rather than scoring it. That is not caution, it is the control's own rule:
"A model cannot beat a baseline that never finished training, and scoring it as though it
had is how a weak control manufactures a win." At the library default of 500 iterations
this fit stops at a gradient norm of 4.883e-04 against a 1e-04 tolerance and is correctly
refused, so the default here is higher. The tolerance is not loosened -- the tolerance is
what makes the control worth beating.

## Reading the result

The comparison is against ``tools/rung0_real_run.py`` on the *same* split, and the split is
by file so neither model has seen a validation file. A control that wins says the corpus is
not the problem. A control that ties the prior says the task as encoded is not learnable at
this size, and that rung 0's failure is not evidence about ``ByteDecider`` at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from rung0_real_run import (  # noqa: E402
    DEFAULT_CONTEXT_BYTES,
    decisions_of,
    majority_baseline,
    split_by_file,
)

from qd_train.baseline import LinearBaseline  # noqa: E402
from qd_train.byte_decider import ByteDeciderConfig  # noqa: E402
from qd_train.ledger import (  # noqa: E402
    DEFAULT_LEDGER_PATH,
    Environment,
    Ledger,
    Protocol,
    RunRecorder,
)
from qd_train.mutate_adapter import MUTATION_CLASSES  # noqa: E402
from qd_train.tristate import NotRun, Ran, TriState  # noqa: E402

#: Above the library default of 500, which does not converge on this corpus. See the
#: module docstring: the ITERATION BUDGET moves, never the tolerance.
DEFAULT_MAX_ITER: Final[int] = 6_000

#: A bound, because an unconverged control is refused and an unbounded one never returns.
MAX_ITER_CEILING: Final[int] = 200_000


def context_texts(decisions) -> tuple[list[str], list[str]]:
    """Exactly the bytes the model sees, as text, with the gold class name.

    ``EncodedContext.ids`` are byte values and ``ids[i] == raw[i]`` for every kept byte, so
    this is the model's own input rather than a re-read of the source file -- the same
    truncation, the same window. Comparing against a control that saw *more* of the file
    would be comparing two different tasks.
    """
    docs: list[str] = []
    labels: list[str] = []
    for d in decisions:
        # errors="replace" only where a multi-byte character was split by the window. The
        # model sees those same split bytes; the replacement affects the control's
        # tokenisation of one character at the boundary, not what either model was given.
        docs.append(bytes(d.context.ids).decode("utf-8", errors="replace"))
        labels.append(MUTATION_CLASSES[d.gold_option])
    return docs, labels


def _control_gate(measured: float, baseline: float, *, n: int) -> TriState:
    """Whether the control beat the prior, carrying both numbers."""
    if n == 0:
        return NotRun(reason="no held-out rows, so the control was not scored")
    return Ran(
        passed=measured > baseline,
        value=measured,
        n=round(measured * n),
        n_total=n,
        detail=(
            f"linear control top-1 {measured:.1%} of {n} held-out rows against a "
            f"majority-class baseline of {baseline:.1%}, a gap of {measured - baseline:+.1%}. "
            "A control ABOVE the prior means the context carries class signal, so a model "
            "that sits at the prior is failing to extract signal that is there."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True, help="qd-mutate examples.jsonl")
    parser.add_argument("--manifest-in", type=Path, required=True, help="the manifest beside it")
    parser.add_argument("--context-bytes", type=int, default=DEFAULT_CONTEXT_BYTES)
    parser.add_argument("--val-share", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    args = parser.parse_args(argv)

    if not 1 <= args.max_iter <= MAX_ITER_CEILING:
        raise SystemExit(f"--max-iter must be in [1, {MAX_ITER_CEILING}], got {args.max_iter}")

    examples = [
        json.loads(line)
        for line in args.examples.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not examples:
        raise SystemExit(f"{args.examples} holds no examples")
    manifest = json.loads(args.manifest_in.read_text(encoding="utf-8"))

    config = ByteDeciderConfig(max_context_bytes=args.context_bytes)
    train_raw, val_raw = split_by_file(examples, val_share=args.val_share)
    train_d, _, _ = decisions_of(train_raw, config=config)
    val_d, _, _ = decisions_of(val_raw, config=config)
    if not train_d or not val_d:
        raise SystemExit("one side of the split is empty; nothing can be measured")

    train_docs, train_labels = context_texts(train_d)
    val_docs, val_labels = context_texts(val_d)
    baseline, majority = majority_baseline(val_d)
    print(f"train {len(train_docs)} docs, val {len(val_docs)} docs, file-disjoint")
    print(f"val majority-class baseline: {baseline:.1%} ({majority})")
    print(f"train labels: {Counter(train_labels).most_common()}")

    model = LinearBaseline(seed=args.seed, max_iter=args.max_iter)
    model.fit(train_docs, train_labels)
    convergence = model.convergence()
    if not isinstance(convergence, Ran):
        print(f"\ncontrol convergence: NOT RUN -- {convergence.reason}")
        print("An unconverged control cannot be beaten. Raise --max-iter; do not lower the")
        print("tolerance, which is what makes the control worth beating.")
        return 1
    print(f"\ncontrol convergence: {convergence.detail}")

    def accuracy(docs: list[str], labels: list[str]) -> float:
        predicted = model.predict(docs)
        return sum(p == g for p, g in zip(predicted, labels, strict=True)) / len(labels)

    train_acc = accuracy(train_docs, train_labels)
    val_acc = accuracy(val_docs, val_labels)
    print(f"\nlinear control TRAIN    : {train_acc:.1%}")
    print(f"linear control HELD-OUT : {val_acc:.1%}  (baseline {baseline:.1%}, "
          f"gap {val_acc - baseline:+.1%})")
    print(f"predicted classes on val: {Counter(model.predict(val_docs)).most_common()}")

    ledger = Ledger(args.ledger)
    protocol = Protocol(
        data_snapshot_hash=hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        tokenizer_hash="chargram-hashed",
        backbone_commit=f"linear-control:ctx{args.context_bytes}",
        recipe_hash=hashlib.sha256(
            json.dumps(
                {"max_iter": args.max_iter, "val_share": args.val_share}, sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
        seed=args.seed,
    )
    with RunRecorder(
        ledger,
        protocol=protocol,
        run_kind="smoke",
        repo=REPO,
        env=Environment.detect(device="cpu"),
        quick=True,
        quick_reason=(
            "a control arm on a corpus drawn from this repository rather than the pool the "
            "plan names; it promotes nothing and exists to interpret another run"
        ),
        notes=(
            "tools/rung0_linear_control.py -- can a linear bag-of-n-grams find the class "
            "signal ByteDecider does not"
        ),
    ) as recorder:
        recorder.metric("linear_control_beats_the_prior", _control_gate(
            val_acc, baseline, n=len(val_labels)
        ))
        recorder.metric("linear_control_convergence", convergence)
        recorder.metric(
            "linear_control_train_accuracy",
            Ran(
                passed=True,
                value=train_acc,
                n=round(train_acc * len(train_labels)),
                n_total=len(train_labels),
                detail="accuracy on the rows it was fitted on, for the over/underfit gap",
            ),
        )

    print()
    if val_acc > baseline:
        print("SIGNAL IS PRESENT: the context carries class information above the prior, so a")
        print("model sitting at the prior is failing to extract signal that is there. That")
        print("makes rung 0's result an architecture/optimisation finding, not a corpus one.")
    else:
        print("NO SIGNAL AT THIS SIZE: the weakest reasonable model does not beat the prior")
        print("either, so rung 0's result is not evidence about ByteDecider.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
