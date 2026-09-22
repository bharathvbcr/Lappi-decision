"""Fit the linear control off the GPU, and write the verdict `rung0_real_run.py` reads.

## What this is for

`paired_margin_vs_linear` is the gate the program rests on. Scoring it means fitting a
char-n-gram logistic regression, which is single-threaded CPU work. On the rung-0 corpus
that is 38 seconds. On the commitpackft corpus the plan names -- 39,946 training documents
-- it projects to hours, and `rung0_real_run.py` now REFUSES to start a fit that exceeds
its time budget rather than disappearing into one with the GPU at 0%.

This is the other half of that refusal. Run it on any machine with CPU and memory -- it
never touches a GPU -- and the next training run finds the answer waiting.

## It must reproduce the split exactly

The cache key covers the documents, so every step that decides which documents exist, and
in what order, is replicated here by *calling the same functions* rather than by
reimplementing them: `split_by_file`, `decisions_of`, `bucketed_chunks`. In particular the
validation order depends on `--batch-size`, because the run scores the rows in bucketed
order; a tool that skipped that would compute a different key, miss its own cache, and be
silently useless.

If the arguments do not match the run's, the key differs and the run simply misses. That is
the safe failure: a mismatched cache is never served.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from rung0_real_run import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_BYTES,
    DEFAULT_VAL_SHARE,
    LINEAR_CONTROL_MAX_ITER,
    bucketed_chunks,
    decisions_of,
    filter_train_rows,
    split_by_file,
)

from qd_data.config import DataConfig  # noqa: E402
from qd_train.baseline import LinearBaseline, context_texts  # noqa: E402
from qd_train.byte_decider import ByteDeciderConfig  # noqa: E402
from qd_train.control_cache import control_key, load_control, store_control  # noqa: E402
from qd_train.mutate_adapter import (  # noqa: E402
    CONTEXT_AFTER,
    CONTEXT_SOURCES,
    read_example_objects,
    refuse_leaky_diff_corpus,
)
from qd_train.tristate import NotRun, Ran  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--control-cache", type=Path, required=True)
    # Every default below is IMPORTED, never restated. The cache key covers the documents,
    # and the documents are decided by the split and the context width -- so a default that
    # disagrees with the run's produces a different key, a guaranteed miss, and an hour of
    # CPU spent on a verdict nothing can read. Restating them diverged them within a day:
    # val-share was 0.20 here against the runner's 0.25.
    parser.add_argument("--val-share", type=float, default=DEFAULT_VAL_SHARE)
    parser.add_argument("--context-bytes", type=int, default=DEFAULT_CONTEXT_BYTES)
    parser.add_argument("--width", type=int, default=ByteDeciderConfig().width)
    parser.add_argument("--layers", type=int, default=ByteDeciderConfig().n_layers)
    parser.add_argument("--heads", type=int, default=ByteDeciderConfig().n_heads)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    # The control must read the bytes the arm reads. A control fitted on the post-image
    # and paired against a model fitted on the diff is not a control at all -- it is two
    # models on two tasks with one margin between them.
    parser.add_argument(
        "--context-source", choices=CONTEXT_SOURCES, default=CONTEXT_AFTER
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=LINEAR_CONTROL_MAX_ITER)
    parser.add_argument(
        "--dense-budget-gb", type=float, default=24.0,
        help=(
            "how much memory the fit may use to hold the design matrix densely. This job "
            "owns the machine, so the default is far above the in-process one: the "
            "commitpackft corpus is 20.9 GB dense, and dense is ~86x faster than the "
            "sparse gather it replaces. Below the budget the corpus stays sparse and the "
            "fit takes days rather than hours"
        ),
    )
    parser.add_argument(
        "--hold-out-operator",
        default="",
        help=(
            "fit on the training set an operator-holdout arm trains on, so the control is "
            "that arm's opponent rather than a stronger model fitted on data the arm never "
            "saw. Must match the arm's flag exactly: the cache key covers the training "
            "documents, so any difference is a miss and the arm's margin stays not_run"
        ),
    )
    parser.add_argument("--drop-random-train", type=int, default=0)
    parser.add_argument("--drop-random-seed", type=int, default=0)
    parser.add_argument(
        "--force", action="store_true",
        help="refit even on a cache hit. The verdict is deterministic given the key, so "
             "this is for re-timing, not for changing an answer",
    )
    args = parser.parse_args(argv)

    # Through rule 3's door rather than around it -- see read_example_objects. A control
    # fitted on held-out data would not train a model, but it would set the bar the model
    # is measured against using data the model may never see, which is the same violation
    # wearing a different hat.
    examples = read_example_objects(args.examples, config=DataConfig(), repo_root=REPO)
    if not examples:
        raise SystemExit(f"{args.examples} holds no examples")
    print(f"corpus: {len(examples)} example(s) from {args.examples}")

    config = ByteDeciderConfig(
        max_context_bytes=args.context_bytes,
        width=args.width,
        n_layers=args.layers,
        n_heads=args.heads,
    )
    refuse_leaky_diff_corpus(examples, context_source=args.context_source)
    train_raw, val_raw = split_by_file(examples, val_share=args.val_share)
    # The SAME filter the run applies, through the same function. An operator-holdout arm
    # trains on a reduced set, so a control fitted on the full one is not its opponent --
    # and because the cache key covers the training documents, fitting the full set simply
    # produces a key the arm never asks for, leaving paired_margin_vs_linear at not_run.
    # Validation is untouched on both sides, so the two are scored on identical rows.
    train_raw, note = filter_train_rows(
        train_raw,
        hold_out_operator=args.hold_out_operator,
        drop_random_train=args.drop_random_train,
        drop_random_seed=args.drop_random_seed,
    )
    if note:
        print(f"  {note}")
    train_d, train_refused, _, _ = decisions_of(
        train_raw, config=config, context_source=args.context_source
    )
    val_d, val_refused, _, _ = decisions_of(
        val_raw, config=config, context_source=args.context_source
    )
    if not train_d or not val_d:
        raise SystemExit("one side of the split is empty; nothing can be fitted")
    print(f"  decisions: {len(train_d)} train (refused {train_refused or 'none'}), "
          f"{len(val_d)} val (refused {val_refused or 'none'})")

    # Bucketed exactly as the run scores them: the key covers this order.
    scored_val = [d for chunk in bucketed_chunks(val_d, batch_size=args.batch_size)
                  for d in chunk]
    train_docs, train_labels = context_texts(train_d)
    val_docs, val_labels = context_texts(scored_val)

    model = LinearBaseline(
        seed=args.seed,
        max_iter=args.max_iter,
        dense_budget_bytes=int(args.dense_budget_gb * 1024**3),
    )
    key = control_key(
        train_docs=train_docs,
        train_labels=train_labels,
        val_docs=val_docs,
        seed=args.seed,
        max_iter=args.max_iter,
        hasher_params=(model.hasher.n_min, model.hasher.n_max, model.hasher.dim),
        l2_grid=model.l2_grid,
        tol=model.tol,
        lr=model.lr,
    )
    print(f"  key {key[:32]}  ({len(train_docs)} train doc(s), {len(val_docs)} scored)")

    existing = load_control(args.control_cache, key, expected_n=len(val_docs))
    if existing is not None and not args.force:
        print(f"  CACHE HIT already: fitted in {existing.fitted_s:.1f}s at "
              f"{existing.fitted_at}. Nothing to do; pass --force to refit.")
        return 0

    projected = model.projected_fit_seconds(train_docs, n_classes=len(set(train_labels)))
    print(f"  projected fit: {projected / 60:.1f} minute(s). Starting.", flush=True)

    started = time.monotonic()
    model.fit(train_docs, train_labels)
    fitted_s = time.monotonic() - started

    convergence = model.convergence()
    if not (isinstance(convergence, Ran) and convergence.passed):
        detail = (
            convergence.reason if isinstance(convergence, NotRun) else convergence.detail
        )
        # Nothing is written. A stored verdict from an unconverged fit is a weak opponent
        # that would flatter the model on every run that read it, forever.
        raise SystemExit(
            f"the control did not converge after {fitted_s:.1f}s: {detail}\n"
            "Nothing was cached: a model cannot beat a baseline that never finished "
            "training, and storing it as though it had is how a weak control "
            "manufactures a win."
        )

    predicted = model.predict(val_docs)
    correct = [p == gold for p, gold in zip(predicted, val_labels, strict=True)]
    fit = model.fit_
    assert fit is not None  # convergence() above already refused an unfitted model
    path = store_control(
        args.control_cache,
        key,
        correct,
        fitted_s=fitted_s,
        n_train=len(train_docs),
        l2=fit.l2,
        iterations=fit.iterations,
        final_grad_norm=fit.final_grad_norm,
    )
    accuracy = sum(correct) / len(correct)
    print(f"  fitted in {fitted_s:.1f}s: L2 {fit.l2:g}, {fit.iterations} iteration(s), "
          f"final grad norm {fit.final_grad_norm:.3e}")
    print(f"  control accuracy on the {len(correct)} scored row(s): {accuracy:.1%}")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
