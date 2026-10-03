"""v5's Mac prelude (campaign/v5-preregistered.json build_order step 5; Fable's seed-order ruling,
AUDIT/post-f-2026-10-02/fable-seed-order-ruling.md section 1). The lead runs it once on the Mac,
on v5's real shard set, before the v5 waiters are staged. It writes a small JSON record whose
sha256 the waiters pin (V5_PRELUDE_SHA256 in v5_common.sh); v5_prelude_check reads its fields.

It is j6g_prelude_mac.py's form: it runs ``real_ft_run.main`` from a clean checkout on v5 seed 0's
training argv and stops at ``Ledger(args.ledger)``, before the ledger opens and before any tower
loads, taking main's locals from its frame there. Everything it prints is computed by
``tools/real_ft_run.py``'s own code, never a copy of it:

  - per plan seed 0-4 (``plan_seed_for("seed", ...)``): ``plan_order_digest`` of the epoch plan
    and of arm 2's cut (``max_width=--max-width``), the order digests the ft rows record as
    ``corpus.plan_order_digest`` and amendments_pending asks for;
  - the seed-invariant shape (``SeedArms.shape()``: batch counts, widest batch, rows and padded
    positions of both plans) of every seed's plan, built by ``seed_arms`` one at a time and
    released, against main's seed-0 plan; any difference, or two seeds with one digest, fails;
  - with ``--option-permutation-seed`` (C1): the validation loop of ``_train`` over each seed's
    plan (``permutation.apply`` per batch with ``_alphabets``) and the "Option permutation seed
    ..." sentence; a ``PermutationRefusal`` fails;
  - the noul-weight position count: ``noul_weight_plan`` over main's seed-0 epoch plan at the
    weight given (the arm's ``arm_noul_weight.w.value``), with the letter-mass ratio.

Run it from anywhere. It changes into --root (a clean checkout with v5's corpora linked into
data/pool) before main runs, because main reads data/pool/... relative to it:

    PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 QD_PREP_BIN=<qd-prep at the checkout> \
      bash tools/mac_heavy.sh "v5 prelude" <python> -u campaign/post-f-queue/v5_prelude_mac.py \
      --root <checkout> --record <new path> --noul-weight 4 -- <v5 seed 0's training argv>

Exit codes: 0 with "PRELUDE OK" and the record written; 2 a usage error (bad options, the record
exists, the checkout is not clean, the argv is not v5 seed 0's); 3 main refused, raised or
returned without reaching Ledger(args.ledger); 4 a PermutationRefusal; 5 a seed-invariant shape
differs, two plan seeds share a digest, or main's seed-0 plan is not plan seed 0's. The record
is written only on exit 0. It writes nothing else.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

#: The plan seeds whose orders v5 trains on: seeds 0-2 (v5, the arm, J5') and 3-4 (seeds.seeds_3_4).
PLAN_SEEDS = (0, 1, 2, 3, 4)
#: v5 seed 0's training argv must carry these: recipe.base as F's ft row 973cd4e3 records it and
#: recipe.added's --min-lr 0 and --batch-order seed (v5_common.sh v5_recipe). The lr, beta2,
#: lower-layer flags and the conditionals vary with the pinned decisions; the waiter compares
#: the whole recipe with its own (v5_prelude_check).
REQUIRED_FLAGS = (
    ("--seeds", "0"),
    ("--batch-order", "seed"),
    ("--min-lr", "0"),
    ("--batch-tokens", "35403"),
    ("--checkpoint-skip-layers", "6"),
    ("--optimizer", "master"),
    ("--wall-clock-cap-s", "32400"),
)
REQUIRED_SWITCHES = ("--epoch", "--no-memorise", "--score-val")
#: Not v5 seed 0's: the arm's weight is given to the prelude, not to main, and the rest are
#: other runs' modes.
REFUSED_FLAGS = (
    "--noul-weight",
    "--replay-shards",
    "--shuffled-label",
    "--score-plan",
    "--score-checkpoint",
    "--resume-from",
)
#: main's locals the prelude reads at Ledger(args.ledger).
FRAME_NAMES = (
    "args",
    "reader",
    "labels",
    "config",
    "batch_tokens",
    "arms",
    "permutation",
    "epoch_alphabets",
    "letter_id",
    "replay_plan",
)


class PreludeDone(Exception):
    """Raised in place of Ledger(args.ledger), carrying main's locals at that line."""

    def __init__(self, captured: dict[str, object]) -> None:
        super().__init__("prelude complete; stopped at Ledger(args.ledger) in main")
        self.captured = captured


class PreludeFailed(Exception):
    """A prelude failure with its exit code; nothing is written."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def usage(msg: str) -> PreludeFailed:
    return PreludeFailed(2, f"PRELUDE USAGE: {msg}")


def parse(argv: list[str]) -> tuple[Path, Path, float, list[str]]:
    if "--" not in argv:
        raise usage("expected --root DIR --record PATH --noul-weight W -- <v5 seed 0's argv>")
    cut = argv.index("--")
    ours, training = argv[:cut], argv[cut + 1 :]
    opts: dict[str, str] = {}
    it = iter(ours)
    for flag in it:
        if flag not in ("--root", "--record", "--noul-weight") or flag in opts:
            raise usage(f"unknown or repeated option {flag!r}")
        value = next(it, None)
        if value is None:
            raise usage(f"{flag} needs a value")
        opts[flag] = value
    if set(opts) != {"--root", "--record", "--noul-weight"}:
        raise usage("--root, --record and --noul-weight are all required")
    try:
        weight = float(opts["--noul-weight"])
    except ValueError as exc:
        raise usage(f"--noul-weight {opts['--noul-weight']!r} is not a number") from exc
    if not weight > 0 or weight != weight or weight == float("inf"):
        raise usage(f"--noul-weight {weight} is not finite and > 0")
    # Both made absolute here, before the prelude changes into --root.
    return Path(opts["--root"]).resolve(), Path(opts["--record"]).resolve(), weight, training


def check_checkout(root: Path) -> str:
    """The checkout's HEAD, refusing a checkout with tracked changes."""
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if head.returncode != 0 or len(head.stdout.strip()) != 40:
        raise usage(f"{root}: git rev-parse HEAD failed: {head.stderr.strip()}")
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise usage(f"{root} has tracked changes or git failed: {dirty.stdout or dirty.stderr}")
    return head.stdout.strip()


def check_argv(training: list[str]) -> None:
    """v5 seed 0's training argv, in the form the waiter runs it (v5_train_seed)."""
    for flag, value in REQUIRED_FLAGS:
        if training.count(flag) != 1:
            raise usage(f"the argv has {flag} {training.count(flag)} time(s), not once")
        got = training[training.index(flag) + 1 : training.index(flag) + 2]
        if got != [value]:
            raise usage(f"the argv has {flag} {got}, not {value}")
    for switch in REQUIRED_SWITCHES:
        if training.count(switch) != 1:
            raise usage(f"the argv has {switch} {training.count(switch)} time(s), not once")
    for refused in REFUSED_FLAGS:
        if refused in training:
            raise usage(f"the argv carries {refused}; the prelude runs v5 seed 0's own argv")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_main(ft: Any, training: list[str]) -> dict[str, object]:
    """Run ``ft.main(training)`` until ``Ledger(args.ledger)`` and return main's locals there.
    ``ft.Ledger`` is put back whatever happens."""
    real_ledger = ft.Ledger

    def _stop(path: object, *args: object, **kwargs: object) -> None:
        frame = sys._getframe(1)
        if frame.f_code.co_name != "main" or frame.f_globals is not vars(ft):
            raise PreludeFailed(
                3,
                f"PRELUDE FAILED: Ledger({path}) was called from {frame.f_code.co_name} in "
                f"{frame.f_code.co_filename}, not from real_ft_run.main",
            )
        missing = [n for n in FRAME_NAMES if n not in frame.f_locals]
        if missing:
            raise PreludeFailed(3, f"PRELUDE FAILED: main has no {missing} at Ledger(...)")
        raise PreludeDone({n: frame.f_locals[n] for n in FRAME_NAMES})

    ft.Ledger = _stop
    try:
        rc = ft.main(training)
    except PreludeDone as done:
        return done.captured
    except PreludeFailed:
        raise
    except SystemExit as refused:
        raise PreludeFailed(3, f"PRELUDE FAILED: main refused: {refused.code}") from refused
    except Exception as exc:
        raise PreludeFailed(3, f"PRELUDE FAILED: main raised {exc!r}") from exc
    finally:
        ft.Ledger = real_ledger
    raise PreludeFailed(3, f"PRELUDE FAILED: main returned {rc!r} without reaching Ledger")


def permuted_per_pass(
    ft: Any, plan: list[Any], labels_for: Any, permutation: Any, seed: int
) -> int:
    """``_train``'s validation loop over one seed's plan: every choice row's option block,
    re-permuted. A refusal fails the prelude (exit 4)."""
    from qd_train.trainer import PermutationRefusal

    alphabets = ft._alphabets(plan, labels_for)
    n = 0
    for b, batch in enumerate(plan):
        try:
            n += permutation.apply(batch, alphabets[b])[1]
        except PermutationRefusal as refusal:
            raise PreludeFailed(
                4,
                f"PermutationRefusal: {refusal}\nPRELUDE FAILED: option permutation refused a "
                f"row of plan seed {seed}'s batch {b} of {len(plan)}",
            ) from refusal
    return n


def run_prelude(ft: Any, training: list[str], *, commit: str, noul_weight: float) -> dict:
    """The record, from ``ft`` (tools/real_ft_run.py, already imported from the checkout)."""
    t0 = time.monotonic()
    captured = capture_main(ft, training)
    print(f"main reached Ledger(args.ledger) in {time.monotonic() - t0:.0f} s", flush=True)
    args, reader, arms = captured["args"], captured["reader"], captured["arms"]
    config, batch_tokens = captured["config"], int(captured["batch_tokens"])
    permutation = captured["permutation"]
    if args.batch_order != ft.BATCH_ORDER_SEED:
        raise PreludeFailed(3, f"PRELUDE FAILED: main ran with --batch-order {args.batch_order!r}")
    if captured["replay_plan"] is not None:
        raise PreludeFailed(3, "PRELUDE FAILED: main built a replay plan; v5 has no replay")
    want_seed0 = ft.plan_seed_for(args.batch_order, seed=PLAN_SEEDS[0], config=config)
    if arms.plan_seed != want_seed0:
        raise PreludeFailed(
            5, f"PRELUDE FAILED: main's plan is plan seed {arms.plan_seed}, not {want_seed0}"
        )

    # The noul-weight count over main's seed-0 epoch plan (the arm's weight, not main's: v5
    # trains without it). The counts are per pass; the rows are the same for every plan seed.
    nw = ft.noul_weight_plan(
        arms.plan_all, arms.labels_by_batch_all, weight=noul_weight, letter_id=captured["letter_id"]
    )
    print(
        f"noul weight {nw.weight:g} ({nw.scope}) on plan seed {arms.plan_seed}'s epoch plan: "
        f"{nw.weighted_positions} of {nw.supervised_positions} supervised letter positions per "
        f"pass are weighted; letter weight mass x{nw.weight_mass_ratio}",
        flush=True,
    )

    shape0 = arms.shape()
    digests: dict[str, str] = {}
    digests_small: dict[str, str] = {}
    permuted: dict[str, int] = {}
    for seed in PLAN_SEEDS:
        plan_seed = ft.plan_seed_for(args.batch_order, seed=seed, config=config)
        digest = ft.plan_order_digest(reader, batch_tokens=batch_tokens, plan_seed=plan_seed)
        small = ft.plan_order_digest(
            reader, batch_tokens=batch_tokens, plan_seed=plan_seed, max_width=args.max_width
        )
        if seed == PLAN_SEEDS[0]:
            one = arms
        else:
            # One plan held at a time beside main's, as arms_for holds one.
            one = ft.seed_arms(
                reader,
                captured["labels"],
                config=config,
                batch_tokens=batch_tokens,
                plan_seed=plan_seed,
                max_width=args.max_width,
            )
        try:
            shape = one.shape()
            if (one.order_digest, one.order_digest_small) != (digest, small):
                raise PreludeFailed(
                    5,
                    f"PRELUDE FAILED: plan seed {plan_seed}'s SeedArms digests "
                    "are not plan_order_digest's",
                )
            if permutation is not None:
                permuted[str(seed)] = permuted_per_pass(
                    ft, one.plan_all, one.labels_by_batch_all, permutation, plan_seed
                )
        finally:
            if one is not arms:
                one.release()
        digests[str(seed)], digests_small[str(seed)] = digest, small
        print(
            f"plan seed {plan_seed}: corpus.plan_order_digest {digest} (arm 2 {small}); "
            f"shape {list(shape)}",
            flush=True,
        )
        if shape != shape0:
            raise PreludeFailed(
                5,
                f"PRELUDE FAILED: plan seed {plan_seed}'s shape {list(shape)} "
                f"is not plan seed 0's {list(shape0)}",
            )
    if len(set(digests.values())) != len(PLAN_SEEDS):
        raise PreludeFailed(5, f"PRELUDE FAILED: two plan seeds share an order digest: {digests}")
    sentence = None
    if permutation is not None:
        counts = set(permuted.values())
        sentence = (
            f"Option permutation seed {permutation.seed}: "
            f"{permuted[str(PLAN_SEEDS[0])]} choice rows re-permuted per pass on top of the "
            "shard set's own shuffle."
        )
        print(sentence + f" (per plan seed: {permuted})", flush=True)
        if len(counts) != 1 or not min(counts):
            raise PreludeFailed(
                5,
                f"PRELUDE FAILED: choice rows re-permuted per plan seed "
                f"{permuted}, not one count >= 1",
            )

    header = json.loads((reader.root / ft.HEADER_NAME).read_text(encoding="utf-8"))
    tokenizer = None if args.tokenizer_json is None else Path(args.tokenizer_json)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    names = ("batches", "max_width", "rows", "padded_positions")
    return {
        "what": "v5 Mac prelude: campaign/v5-preregistered.json build_order step 5",
        "tool": "campaign/post-f-queue/v5_prelude_mac.py",
        "code_commit": commit,
        "argv": training,
        "batch_order": args.batch_order,
        "plan_seeds": list(PLAN_SEEDS),
        "plan_order_digest": digests,
        "plan_order_digest_small": digests_small,
        "shape": list(shape0),
        "shape_equal_across_seeds": True,
        **{f"plan_{n}": shape0[i] for i, n in enumerate(names)},
        **{f"plan_small_{n}": shape0[4 + i] for i, n in enumerate(names)},
        "shard_hash": reader.header.shard_hash(),
        "data_snapshot_hash": header.get("data_snapshot_hash"),
        "tokenizer_json": None if tokenizer is None else str(tokenizer),
        "tokenizer_json_sha256": None if tokenizer is None else sha256_file(tokenizer),
        "option_permutation_seed": None if permutation is None else permutation.seed,
        "choice_rows_permuted_per_pass": permuted or None,
        "sentence": sentence,
        "noul_weight": nw.weight,
        "noul_weight_scope": nw.scope,
        "noul_weighted_positions": nw.weighted_positions,
        "noul_supervised_positions": nw.supervised_positions,
        "noul_weight_mass_ratio": nw.weight_mass_ratio,
        "wall_s": round(time.monotonic() - t0, 1),
        "max_rss": ru,
        "max_rss_units": "bytes" if sys.platform == "darwin" else "KiB",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": ft.torch.__version__,
        "ok": True,
    }


def write_record(path: Path, record: dict) -> str:
    """Write the record once (O_EXCL, fsync) and return its sha256."""
    body = (json.dumps(record, sort_keys=True, indent=1) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(body).hexdigest()


def main(argv: list[str]) -> int:
    t0 = time.monotonic()
    try:
        root, record_path, weight, training = parse(argv)
        if record_path.exists():
            raise usage(f"--record {record_path} already exists")
        if not record_path.parent.is_dir():
            raise usage(f"--record's directory {record_path.parent} does not exist")
        check_argv(training)
        commit = check_checkout(root)
        # main reads data/pool/... relative to the checkout.
        os.chdir(root)
        sys.path.insert(0, str(root / "python"))
        sys.path.insert(0, str(root / "tools"))
        import real_ft_run as ft

        if not Path(ft.__file__).resolve().is_relative_to(root):
            raise usage(f"real_ft_run imported from {ft.__file__}, not from {root}")
        record = run_prelude(ft, training, commit=commit, noul_weight=weight)
    except PreludeFailed as failed:
        print(str(failed), flush=True)
        return failed.code
    digest = write_record(record_path, record)
    print(f"RECORD {record_path} sha256 {digest}")
    print(f"PRELUDE OK in {time.monotonic() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
