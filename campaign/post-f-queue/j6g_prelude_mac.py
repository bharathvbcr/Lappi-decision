"""J6(g)'s Mac prelude: the STOP gate of campaign/j6g-preregistered.json
(arm.option_permutation.stop_gate). The lead runs it once on the Mac before box_q_j6g.sh is
staged. It writes a small JSON record whose sha256 the waiter pins (J6G_PRELUDE_SHA256), and the
waiter checks the record's fields as well as its pin.

It is the form of /home/ubuntu/scratch/f_prelude_box.py (sha256
9f86f797e607b2b9a48714d182a6ef15108bee059a03694b8a8494696037a3dc) and of
campaign/post-f-queue/j6a_prelude_box.py: it runs real_ft_run.main from an a502670 checkout on
J6(g)'s training argv and stops at `ledger = Ledger(args.ledger)` (tools/real_ft_run.py:8512 at
a502670). That is the first statement after the epoch arm's ported pieces are built (:8488-8495).
So it stops before the ledger is opened and before any tower loads.

What it adds, and why. At a502670 the option blocks of the whole plan are validated inside
`_train` (:2220-2231). That is after `_real_step` has loaded the real tower (:2150-2156) and
before `train_ft`. So no prelude that stops before the tower ever reaches that loop. A refusal
there costs a tower load on the box, not a rented epoch. The sentence the pre-registration asks
for, "Option permutation seed ... choice rows re-permuted ...", is appended to `what_ran`
(:2228-2231). That text goes into the ft row's notes: real_ft_run never prints it to stdout.
So this prelude does three things:
  - it takes, from main's own frame at the Ledger call, the objects main hands to `_train`
    (`plan_all`, `epoch_alphabets` and `permutation`; :8706-8729);
  - it runs the validation loop of :2224-2226 over them, verbatim;
  - it prints the sentence of :2228-2231, verbatim.
Any PermutationRefusal fails the prelude.

It also records which recipe keys the flag adds, by calling a502670's own `_recipe_pieces` on
main's values with and without the permutation, and the plan's batch count and width.

Run it from anywhere. It changes into --root (an a502670 checkout with the v4 corpora linked into
data/pool) before main runs, because main reads data/pool/... relative to it:

    PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 QD_PREP_BIN=<a502670 qd-prep> \
      <python> -u <this file> --root <the checkout> --record <new path> -- <J6(g)'s argv>

HANDOFF/j6g-rule-2026-10-02.md has the full command.

Exit codes:
  - 0 with "PRELUDE OK", the sentence, and the record written;
  - 2 for a usage error (bad options, the record exists, the checkout is not a clean a502670,
    or the argv is not J6(g)'s);
  - 3 when main refused, raised or returned without reaching Ledger(args.ledger);
  - 4 on a PermutationRefusal;
  - 5 when a recorded value is not the pre-registration's (batches, width, shard hash, data
    snapshot, tokenizer, the recipe delta). The record is written only on exit 0.

It writes nothing else. main opens nothing for writing before Ledger (by read of :7368-8512):
no ledger row, no checkpoint and no verdicts.
"""

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

#: The pre-registration's values (campaign/j6g-preregistered.json, dec48d8).
SEED = 20260919
A502670 = "a5026707b6e3e57c253be32003ba32420f2d78e2"
#: F seed 0's ft row 973cd4e3: recipe.batches, recipe.width, recipe.shard_hash, and
#: protocol.data_snapshot_hash.
EXPECTED = {
    "plan_batches": 9683,
    "plan_max_width": 7936,
    "shard_hash": "8bcf56ad89bbf059f9926035b7b798796a0a41dd4960c2939b66671f24d2fb80",
    "data_snapshot_hash": "ea3215c4f36d57f74d291fb94c3fa8724fa5a14a303ea7572aa0dafb2a0933a1",
    # post_f_common.sh TOKENIZER_SHA256: the snapshot's tokenizer.json on the box.
    "tokenizer_json_sha256": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    # arm.identity.recipe: exactly one added key.
    "recipe_piece_delta": {"option_permutation_seed": SEED},
    "recipe_piece_removed": [],
}
#: J6(g)'s argv must carry these (box_q_j6g.sh J6G_ARGV; F's recipe flags plus the one flag).
REQUIRED_FLAGS = (
    ("--option-permutation-seed", str(SEED)),
    ("--seeds", "0"),
    ("--batch-tokens", "35403"),
    ("--lower-layers-n", "8"),
    ("--lower-layers-lr-scale", "0.1"),
    ("--checkpoint-skip-layers", "6"),
    ("--optimizer", "master"),
    ("--lr", "1e-5"),
    ("--wall-clock-cap-s", "32400"),
)
REQUIRED_SWITCHES = ("--epoch", "--no-memorise", "--score-val")


class PreludeDone(Exception):
    """Raised in place of Ledger(args.ledger), carrying main's locals at that line."""

    def __init__(self, captured: dict[str, object]) -> None:
        super().__init__("prelude complete; stopped at Ledger(args.ledger) in main")
        self.captured = captured


def usage(msg: str) -> "SystemExit":
    print(f"PRELUDE USAGE: {msg}", flush=True)
    return SystemExit(2)


def parse(argv: list[str]) -> tuple[Path, Path, list[str]]:
    if "--" not in argv:
        raise usage("expected --root DIR --record PATH -- <J6(g)'s training argv>")
    cut = argv.index("--")
    ours, training = argv[:cut], argv[cut + 1 :]
    opts: dict[str, str] = {}
    it = iter(ours)
    for flag in it:
        if flag not in ("--root", "--record") or flag in opts:
            raise usage(f"unknown or repeated option {flag!r}")
        value = next(it, None)
        if value is None:
            raise usage(f"{flag} needs a value")
        opts[flag] = value
    if set(opts) != {"--root", "--record"}:
        raise usage("both --root and --record are required")
    # Both made absolute here, before main() changes into --root.
    return Path(opts["--root"]).resolve(), Path(opts["--record"]).resolve(), training


def check_checkout(root: Path) -> str:
    """The checkout is clean in its tracked files and at a502670."""
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if head.returncode != 0 or head.stdout.strip() != A502670:
        raise usage(f"{root} is at {head.stdout.strip() or head.stderr.strip()!r}, not {A502670}")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise usage(f"{root} has tracked changes or git failed: {dirty.stdout or dirty.stderr}")
    return head.stdout.strip()


def check_argv(training: list[str]) -> None:
    for flag, value in REQUIRED_FLAGS:
        if training.count(flag) != 1:
            raise usage(f"J6(g)'s argv has {flag} {training.count(flag)} time(s), not once")
        got = training[training.index(flag) + 1 : training.index(flag) + 2]
        if got != [value]:
            raise usage(f"J6(g)'s argv has {flag} {got}, not {value}")
    for switch in REQUIRED_SWITCHES:
        if training.count(switch) != 1:
            raise usage(f"J6(g)'s argv has {switch} {training.count(switch)} time(s), not once")
    for refused in ("--replay-shards", "--shuffled-label", "--score-plan", "--score-checkpoint"):
        if refused in training:
            raise usage(f"J6(g)'s argv carries {refused}; J6(g) is F's recipe plus one flag")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    t0 = time.monotonic()
    root, record_path, training = parse(sys.argv[1:])
    if record_path.exists():
        raise usage(f"--record {record_path} already exists")
    if not record_path.parent.is_dir():
        raise usage(f"--record's directory {record_path.parent} does not exist")
    commit = check_checkout(root)
    check_argv(training)
    # main reads data/pool/... relative to the checkout, as F's lane runs from qd-lane8.
    os.chdir(root)

    sys.path.insert(0, str(root / "python"))
    sys.path.insert(0, str(root / "tools"))
    import real_ft_run as ft

    from qd_train.trainer import PermutationRefusal

    if not Path(ft.__file__).resolve().is_relative_to(root):
        raise usage(f"real_ft_run imported from {ft.__file__}, not from {root}")

    def _stop(path: object, *args: object, **kwargs: object) -> None:
        frame = sys._getframe(1)
        if frame.f_code.co_name != "main" or frame.f_globals is not vars(ft):
            raise SystemExit(
                f"PRELUDE FAILED: Ledger({path}) was called from "
                f"{frame.f_code.co_name} in {frame.f_code.co_filename}, not from real_ft_run.main"
            )
        names = (
            "args",
            "reader",
            "plan_all",
            "epoch_alphabets",
            "permutation",
            "replay_plan",
            "recipe_batch_tokens",
        )
        missing = [n for n in names if n not in frame.f_locals]
        if missing:
            raise SystemExit(f"PRELUDE FAILED: main has no {missing} at Ledger(...)")
        raise PreludeDone({n: frame.f_locals[n] for n in names})

    ft.Ledger = _stop  # type: ignore[assignment,misc]

    try:
        rc = ft.main(training)
    except PreludeDone as done:
        captured = done.captured
    else:
        print(f"PRELUDE FAILED: main returned {rc!r} without reaching Ledger(args.ledger)")
        return 3
    print(f"main reached Ledger(args.ledger) in {time.monotonic() - t0:.0f} s", flush=True)

    args = captured["args"]
    reader = captured["reader"]
    plan = captured["plan_all"]
    alphabets = captured["epoch_alphabets"]
    permutation = captured["permutation"]
    if permutation is None or alphabets is None:
        print("PRELUDE FAILED: main built no option permutation (permutation or alphabets is None)")
        return 3
    if permutation.seed != SEED:
        print(f"PRELUDE FAILED: main's permutation seed is {permutation.seed}, not {SEED}")
        return 3
    if captured["replay_plan"] is not None:
        print("PRELUDE FAILED: main built a replay plan; J6(g) has no replay")
        return 3

    # tools/real_ft_run.py:2222-2231 at a502670, verbatim but for the try.
    t1 = time.monotonic()
    permuted_per_pass = 0
    try:
        for b, batch in enumerate(plan):
            permuted_per_pass += permutation.apply(batch, alphabets[b])[1]
    except PermutationRefusal as refusal:
        print(f"PermutationRefusal: {refusal}")
        print(f"PRELUDE FAILED: option permutation refused a row of plan batch {b} of {len(plan)}")
        return 4
    print(
        f"Option permutation seed {permutation.seed}: {permuted_per_pass} choice rows "
        "re-permuted per pass on top of the shard set's own shuffle.",
        flush=True,
    )
    validate_s = time.monotonic() - t1

    # Which recipe keys the flag adds: a502670's own _recipe_pieces on main's values (:8706-8731).
    pieces = dict(
        lower_layers_n=args.lower_layers_n,
        lower_lr_scale=args.lower_layers_lr_scale,
        beta2=args.beta2,
        replay=captured["replay_plan"],
        cap_s=args.wall_clock_cap_s,
        no_memorise=args.no_memorise,
        batch_tokens=captured["recipe_batch_tokens"],
        shuffled_label=None,
        checkpoint_skip_layers=args.checkpoint_skip_layers,
        fused_adamw=args.fused_adamw,
    )
    with_flag = ft._recipe_pieces(permutation=permutation, **pieces)
    without = ft._recipe_pieces(permutation=None, **pieces)
    delta = {k: v for k, v in with_flag.items() if k not in without or without[k] != v}
    removed = sorted(k for k in without if k not in with_flag)

    header = json.loads((reader.root / ft.HEADER_NAME).read_text(encoding="utf-8"))
    choice_rows = sum(1 for rows in alphabets.values() for a in rows if a is not None)
    tokenizer = Path(args.tokenizer_json)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {
        "what": (
            "J6(g) Mac prelude: campaign/j6g-preregistered.json arm.option_permutation.stop_gate"
        ),
        "tool": "campaign/post-f-queue/j6g_prelude_mac.py",
        "code_commit": commit,
        "argv": training,
        "option_permutation_seed": permutation.seed,
        "choice_rows_permuted_per_pass": permuted_per_pass,
        "choice_rows_in_plan": choice_rows,
        "permutation_refusals": 0,
        "sentence": (
            f"Option permutation seed {permutation.seed}: {permuted_per_pass} choice rows "
            "re-permuted per pass on top of the shard set's own shuffle."
        ),
        "plan_batches": len(plan),
        "plan_max_width": max(int(b.tokens.shape[1]) for b in plan),
        "plan_rows": sum(int(b.tokens.shape[0]) for b in plan),
        "plan_batches_whose_index_is_not_their_position": sum(
            1 for i, b in enumerate(plan) if int(b.index) != i
        ),
        "shard_hash": reader.header.shard_hash(),
        "data_snapshot_hash": header.get("data_snapshot_hash"),
        "tokenizer_json": str(tokenizer),
        "tokenizer_json_sha256": sha256_file(tokenizer),
        "recipe_piece_delta": delta,
        "recipe_piece_removed": removed,
        "wall_s": round(time.monotonic() - t0, 1),
        "validate_s": round(validate_s, 1),
        "max_rss": ru,
        "max_rss_units": "bytes" if sys.platform == "darwin" else "KiB",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": ft.torch.__version__,
    }
    wrong = {
        k: {"recorded": record[k], "expected": v} for k, v in EXPECTED.items() if record[k] != v
    }
    if choice_rows != permuted_per_pass:
        wrong["choice_rows_permuted_per_pass"] = {
            "recorded": permuted_per_pass,
            "expected": choice_rows,
        }
    if not permuted_per_pass:
        wrong["choice_rows_permuted_per_pass"] = {"recorded": 0, "expected": ">= 1"}
    if wrong:
        print(f"PRELUDE FAILED: not the pre-registration's values: {json.dumps(wrong)}")
        return 5
    record["ok"] = True
    body = (json.dumps(record, sort_keys=True, indent=1) + "\n").encode("utf-8")
    fd = os.open(record_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    print(f"RECORD {record_path} sha256 {hashlib.sha256(body).hexdigest()}")
    print(f"PRELUDE OK in {time.monotonic() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
