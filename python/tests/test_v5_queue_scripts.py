"""The v5 block's box waiters, driven on the Mac (no box, no GPU).

Files under test: ``campaign/post-f-queue/{box_q_v5.sh, box_q_v5traj.sh, v5_common.sh}`` (lanes
L-v5-queue, then L-v5-2gpu). The block runs on one Lambda 2x H100 box as two GPU lanes,
``box_q_v5.sh 0`` and ``box_q_v5.sh 1``; each lane takes the earliest job in the human's order
whose inputs are ready (Fable's rule; build/v5-h100/queue-lane-spec.md, its REVISION ~17:45Z
governing). The order is 11 jobs: v5 seeds 0-4 (unconditional since the amendment a29bca1
applied), the noul-weight arm x3 (seeds 0-2) iff room, J5' x3 (seeds 0-2). The GH200's three
sequential waiters (box_q_v5s34.sh, box_q_v5nw.sh, box_q_v5j5.sh) are retired: their decisions
and runs are functions of ``v5_common.sh`` that either lane calls, and seeds34 is not read.
Four kinds of test:

* static: ``bash -n`` and shellcheck on every file; the caps and constants are the
  pre-registration's; held-out data and the containment request are named only by the guard
  that refuses them; the rules binary is reached only through the two functions that verify its
  pin first; the committed pins are the literal UNSET (fail closed); nothing names the GH200's
  chain, lock, rate or ledger names.
* the decision, cost and lock functions in ``v5_common.sh``, sourced under ``/bin/bash`` with a
  stub rules binary in a temp dir.
* the rule: ``v5_next_job`` on hand-made queue states, and a discrete-event simulation with equal
  run lengths that drives the real ``v5_next_job`` through every branch (R9 waived / kept and
  continue / kept and pause x room / no_room) and compares each lane's sequence with an
  independent Python oracle of the greedy rule, anchored on the lead's own derivation.
* dry runs of the two lanes: a copy of the queue directory with ``/home/ubuntu`` rewritten to a
  temp root, a fake ``python`` that records every ``tools/real_ft_run.py`` /
  ``ft_linear_control.py`` argv with its ``CUDA_VISIBLE_DEVICES``, the file its fd 9 names and
  whether that file is flock-held (and execs the real interpreter for the waiters' own ``-c`` /
  ``-`` checks), a fake rules binary driven by a scenario file, a real ``flock`` (fcntl on the
  inherited fd), and fakes for ``timeout``, ``sleep``, ``setsid``, ``nohup`` and ``git``.

``QD_V5_QUEUE_DIR`` points the tests at another copy of the queue directory (the fail-first run
uses 1ab477f's).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
QUEUE = Path(os.environ.get("QD_V5_QUEUE_DIR", str(REPO / "campaign" / "post-f-queue")))
DRAFT = REPO / "campaign" / "v5-preregistered.DRAFT.json"
NOUL = REPO / "campaign" / "v4-noul-v3b-preregistered.json"
BASH = "/bin/bash"
WAITERS = ("box_q_v5.sh", "box_q_v5traj.sh")
V5_FILES = (*WAITERS, "v5_common.sh")
#: The GH200's sequential waiters, replaced by the lanes (their bodies are v5_common.sh's
#: v5_decide_room / v5_decide_arm / v5_job_seed / v5_job_j5; seeds34 is retired with them).
RETIRED = ("box_q_v5s34.sh", "box_q_v5nw.sh", "box_q_v5j5.sh")
PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
LANE_AT = "0123456789abcdef0123456789abcdef01234567"
BUILD_REV = "89abcdef0123456789abcdef0123456789abcdef"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
EVAL_ROW = "eeeeeeee-1111-2222-3333-444444444444"
#: The box the dry runs fill the pins for: Lambda 2x H100 80 GB SXM5 at $8.38/h, $4.19/GPU-h
#: (AUDIT/finalize-2026-10-03/report-to-human-2026-10-03-pipeline.md item 1). Its ledger name,
#: h100x2, and the probe's 12 GiB margin are the pre-registration's (seeds.ledger,
#: hardware.probe.margin_gib).
BOX = "h100x2"
INSTANCE = "lambda-2xH100-80GB-SXM5"
USD_INSTANCE = "8.38"
USD_GPU = "4.19"
PROBE_MARGIN = "12"
EXPANDABLE = "expandable_segments:True"
#: The human's words on the box, as the report records them (item 1, ~16:26Z), with the report's
#: multiplication sign, kept because the words are quoted verbatim.
BOX_WORDS = "Go with 2\N{MULTIPLICATION SIGN} H100 (Lambda)"
#: A stand-in for the human's yes on the 11-run plan, which the lead fills into the pin
#: V5_HUMAN_YES at deploy, verbatim (the d24c865 yes covered ~$137 / ~60 GPU-h, not this plan).
HUMAN_YES = "Bharath (human, stand-in for the dry runs): yes, the 11 runs on the 2x H100 box"
#: The recipe for the default decision pins (C1/C2a/C2b off, lower layers kept, F's lr):
#: recipe.base's flags as ft row 973cd4e3 records them, then recipe.added's --min-lr 0 and
#: --batch-order seed.
BASE_RECIPE = [
    "--optimizer",
    "master",
    "--lr",
    "1e-5",
    "--epoch",
    "--no-memorise",
    "--batch-tokens",
    "35403",
    "--lower-layers-n",
    "8",
    "--lower-layers-lr-scale",
    "0.1",
    "--checkpoint-skip-layers",
    "6",
    "--min-lr",
    "0",
    "--batch-order",
    "seed",
]
VALUED = {
    "--optimizer",
    "--lr",
    "--beta2",
    "--batch-tokens",
    "--lower-layers-n",
    "--lower-layers-lr-scale",
    "--checkpoint-skip-layers",
    "--min-lr",
    "--batch-order",
    "--option-permutation-seed",
    "--train-attention-mask",
    "--noul-weight",
    "--span-weight",
    "--train-dtype",
    "--attn-implementation",
}
SWITCHES = {"--epoch", "--no-memorise", "--fused-adamw", "--deterministic"}


def recipe_pairs(argv: list[str]) -> list[tuple[str, str | None]]:
    """The recipe flags of an argv as (flag, value) pairs, sorted (the test's own oracle)."""
    out: list[tuple[str, str | None]] = []
    i = 0
    while i < len(argv):
        if argv[i] in VALUED:
            out.append((argv[i], argv[i + 1]))
            i += 2
            continue
        if argv[i] in SWITCHES:
            out.append((argv[i], None))
        i += 1
    return sorted(out, key=repr)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def renamed_prereg() -> dict:
    """The DRAFT as the rename will bind it: the same text without its top-level draft key."""
    d = json.loads(DRAFT.read_text(encoding="utf-8"))
    d.pop("draft")
    return d


def write_exec(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def bash(script: str, *, env_path: str = PATH, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BASH, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={"PATH": env_path, "HOME": os.environ.get("HOME", "/tmp"), "LC_ALL": "C"},
        check=False,
    )


def code_lines(text: str) -> list[str]:
    """A shell file's lines with comments dropped (whole-line comments, and ` # ...` tails)."""
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        out.append(re.split(r"\s#\s", line, maxsplit=1)[0])
    return out


# --- static ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", V5_FILES)
def test_bash_n_parses_every_file(name: str) -> None:
    path = QUEUE / name
    assert path.is_file(), f"{path} does not exist"
    r = subprocess.run(
        [BASH, "-n", str(path)], capture_output=True, text=True, check=False, timeout=30
    )
    assert r.returncode == 0, r.stderr


def test_shellcheck_is_clean() -> None:
    sc = shutil.which("shellcheck", path=f"/opt/homebrew/bin:{PATH}")
    if sc is None:
        pytest.skip("shellcheck is not installed here: NOT RUN (not a pass)")
    for name in V5_FILES:
        assert (QUEUE / name).is_file(), f"{name} does not exist"
    r = subprocess.run(
        [sc, "-x", "-P", "SCRIPTDIR", *(str(QUEUE / n) for n in V5_FILES)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_gh200s_sequential_waiters_are_retired() -> None:
    """Replace, don't accumulate: the lanes run every job the three chained waiters ran, so the
    waiters are gone rather than left beside the lanes as dead code a deploy could launch."""
    present = [n for n in RETIRED if (QUEUE / n).exists()]
    assert present == [], f"still present: {present}"


def common_text() -> str:
    return (QUEUE / "v5_common.sh").read_text(encoding="utf-8")


PINS = (
    "RULES_V5_SHA256",
    "V5_PREP_BIN_SHA256",
    "V5_LANE_AT",
    "V5_PREREG_SHA256",
    "V5_DATE",
    "V5_PRELUDE_SHA256",
    "V5_BOX",
    "V5_INSTANCE",
    "V5_USD_PER_INSTANCE_HOUR",
    "V5_USD_PER_GPU_HOUR",
    "V5_R9",
    "V5_HUMAN_YES",
    "V5_C1",
    "V5_C2A",
    "V5_C2B",
    "V5_LOWER",
    "V5_LRSET",
)


def test_committed_pins_are_unset_so_every_waiter_fails_closed() -> None:
    text = common_text()
    for pin in PINS:
        assert re.search(rf"^{pin}=UNSET$", text, re.M), f"{pin} is not the literal UNSET"
    assert re.search(r"^V5_SPLIT=\(UNSET\)$", text, re.M)
    # the old binary stays where it is; the v5 build is a new path
    assert re.search(r"^RULES_V5=/home/ubuntu/bin/qd-post-f-rules-v5$", text, re.M)


def test_caps_and_constants_are_the_pre_registrations() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    text = common_text()
    want = {
        "V5_TRAIN_CAP_S": ("32400", "--wall-clock-cap-s 32400"),
        "V5_NEEDLE_CAP_S": ("5400", "cap 5,400 s"),
        "V5_TRAJ_CAP_S": ("3600", "cap 3,600 s"),
        "V5_CKPT_EVERY": ("100000", "--checkpoint-every 100000"),
        "V5_RETAIN_EVERY": ("1000", "--retain-tower-every 1000"),
        "V5_C1_SEED": ("20260919", "--option-permutation-seed 20260919"),
    }
    for name, (value, phrase) in want.items():
        assert re.search(rf"^{name}={value}$", text, re.M), name
        assert phrase in draft, f"the DRAFT does not say {phrase!r}"
    assert "--batch-order seed" in draft and "--min-lr 0" in draft


def test_the_probe_margin_and_cap_are_the_pre_registrations() -> None:
    """hardware.probe (the amendment a29bca1 applied): margin_gib 12 and 'cap 1,800 s'. The
    constants agree with it here, and v5_read_prereg refuses a pre-registration that says
    another margin (test_v5_read_prereg_refuses_a_probe_margin_it_does_not_carry)."""
    probe = renamed_prereg()["hardware"]["probe"]
    text = common_text()
    assert probe["margin_gib"] == int(PROBE_MARGIN)
    assert re.search(rf"^V5_PROBE_MARGIN_GIB={PROBE_MARGIN}$", text, re.M)
    assert "cap 1,800 s" in probe["job"]
    assert re.search(r"^V5_PROBE_CAP_S=1800$", text, re.M)
    assert EXPANDABLE in probe["fallback"] and f"PYTORCH_CUDA_ALLOC_CONF={EXPANDABLE}" in text


def test_nothing_names_the_gh200s_chain_lock_rate_or_ledgers() -> None:
    """v5 runs on the 2x H100 box: no wait on the GH200's post-F chain (j5pp, j6a, j6g), no
    single gpu.lock, no post_f_common.sh COST ($2.29/h on lambda-1xgh200), no gh200-v5 ledger,
    and no seeds34 reading (seeds 3-4 are unconditional since the amendment a29bca1)."""
    for name in V5_FILES:
        code = "\n".join(code_lines((QUEUE / name).read_text(encoding="utf-8")))
        for word in ("j5pp", "j6a", "j6g", "seeds34"):
            assert not re.search(rf"\b{word}\b", code), f"{name} still names {word}"
        assert "gpu.lock" not in code, f"{name} still holds the GH200's one gpu.lock"
        # post_f_common.sh's COST array, not a name ending in it (v5_lane_set's V5_COST)
        assert not re.search(r"(?<!\w)COST\[", code), f"{name} prices with COST"
        # (the V5_BOX guard names gh200 to refuse it: v5_pins_check's gh200* case)
        assert "2.29" not in code and "lambda-1xgh200" not in code, name
        assert "gh200-v5" not in code.lower(), f"{name} names a GH200 ledger"


def test_the_ledgers_are_named_by_the_box_pin() -> None:
    text = common_text()
    assert re.search(r"^V5_LEDGER=/home/ubuntu/ledger/\$V5_BOX-v5-\$V5_DATE\.jsonl$", text, re.M)
    assert re.search(
        r"^V5NW_LEDGER=/home/ubuntu/ledger/\$V5_BOX-v5-noulw-\$V5_DATE\.jsonl$", text, re.M
    )
    assert re.search(
        r"^V5_PROBE_LEDGER=/home/ubuntu/ledger/\$V5_BOX-v5-probe-\$V5_DATE\.jsonl$", text, re.M
    )


def header(name: str) -> str:
    lines = (QUEUE / name).read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines[1:]:
        if not line.startswith("#"):
            break
        out.append(line.lstrip("# "))
    return " ".join(out)


def test_the_lane_header_names_the_rule_the_order_and_the_amendment() -> None:
    h = header("box_q_v5.sh")
    assert "earliest job in the human's order whose inputs are ready" in h
    for job in ("v5 seeds 0-4", "noul-weight arm", "J5'"):
        assert job in h, job
    assert "Fable" in h and "amendment" in h and "interleav" in h
    assert "gpu0.lock" in h and "gpu1.lock" in h and "CUDA_VISIBLE_DEVICES" in h
    assert "--probe-shapes" in h and "V5_PROBE_YES" in h and "expandable_segments" in h


def test_the_cpu_controls_run_outside_the_lock_and_say_two_lanes_may_overlap() -> None:
    text = (QUEUE / "box_q_v5traj.sh").read_text(encoding="utf-8")
    release = text.index("exec 9>&-")
    controls = text.index("\nv5_controls ")
    assert release < controls, "the controls must start after the lane's lock is released"
    comment = text[release:controls]
    assert "concurrently" in comment and "no lock" in comment


def function_body(text: str, name: str) -> str:
    start = text.index(f"\n{name}() {{")
    end = text.index("\n}\n", start)
    return text[start:end]


def test_held_out_and_containment_are_named_only_by_the_guard_that_refuses_them() -> None:
    pattern = re.compile(r"heldout|held-out|held_out|containment", re.I)
    common = common_text()
    allowed = function_body(common, "v5_split_check") + function_body(common, "v5_data_check")
    for name in V5_FILES:
        text = (QUEUE / name).read_text(encoding="utf-8")
        for line in text.splitlines():
            code = line.split("#", 1)[0] if not line.lstrip().startswith("#") else ""
            if pattern.search(code):
                assert name == "v5_common.sh" and line in allowed, f"{name}: {line.strip()}"


def test_the_rules_binary_is_reached_only_through_its_pin_check() -> None:
    common = common_text()
    for name in V5_FILES:
        text = (QUEUE / name).read_text(encoding="utf-8")
        code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
        # never the old binary or post_f_common.sh's rule(), which maps continue/pause/room/
        # no_room/wins to refused
        assert '"$RULES"' not in code, name
        assert not re.search(r"(^\s*|;\s*|&&\s*|\|\|\s*|\$\()rule\s+[a-z\"$]", code, re.M), name
        for m in re.finditer(r'"\$RULES_V5"', code):
            line = code[code.rfind("\n", 0, m.start()) + 1 : code.find("\n", m.end())]
            if line.lstrip().startswith("pin ") or line.lstrip().startswith("RULES_V5="):
                continue
            assert name == "v5_common.sh", f"{name} runs the binary directly: {line.strip()}"
    # one function runs the binary, and it checks the pin before every attempt, inside its loop
    callers = [
        m[1]
        for m in re.finditer(r"^(\w+)\(\) \{$", common, re.M)
        if '"$RULES_V5" "$@"' in function_body(common, m[1])
    ]
    assert callers == ["v5_rules_run"], callers
    body = function_body(common, "v5_rules_run")
    loop = body.index("while :; do")
    assert loop < body.index('pin "$RULES_V5" "$RULES_V5_SHA256"') < body.index('"$RULES_V5" "$@"')
    assert body.count('"$RULES_V5" "$@"') == 1
    for fn in ("v5_rule", "v5_rules_call"):
        fb = function_body(common, fn)
        assert "v5_rules_run " in fb and '"$RULES_V5"' not in fb, fn


# --- the decision functions -------------------------------------------------------------------


def lib(tmp: Path, extra: str = "") -> str:
    q = tmp / "q"
    q.mkdir(parents=True, exist_ok=True)
    return f"""
source "{QUEUE}/post_f_common.sh" || exit 90
Q="{q}"; PF="{tmp}"
source "{QUEUE}/idle_common.sh" || exit 91
source "{QUEUE}/v5_common.sh" || exit 92
DEC_V5="{tmp}/dec"
{extra}
"""


def stub_rules(tmp: Path, word: str, rc: int, *, json_out: bool = True) -> Path:
    d = tmp / "stub"
    d.mkdir(exist_ok=True)
    (d / "word").write_text(word)
    (d / "rc").write_text(str(rc))
    if not json_out:
        (d / "nojson").write_text("")
    return write_exec(
        d / "rules",
        f"""#!/bin/bash
echo "$*" >> "{d}/calls.log"
out=""; prev=""
for a in "$@"; do [ "$prev" = --out ] && out=$a; prev=$a; done
if [ ! -e "{d}/nojson" ] && [ -n "$out" ]; then echo '{{"stub": true}}' > "$out"; fi
printf '%s\\n' "$(cat "{d}/word")"
exit "$(cat "{d}/rc")"
""",
    )


def run_rule(
    tmp: Path, words: str, word: str, rc: int, *, json_out: bool = True, pin_ok: bool = True
) -> tuple[str, bool, str]:
    stub = stub_rules(tmp, word, rc, json_out=json_out)
    pin = sha256(stub) if pin_ok else "0" * 64
    r = bash(
        lib(tmp, f'RULES_V5="{stub}"; RULES_V5_SHA256={pin}')
        + f"""
v5_rule t "{words}" some-subcommand --a b
echo "WORD=$V5_WORD"
echo "SAID=$V5_SAID"
"""
    )
    assert r.returncode == 0, r.stdout + r.stderr
    got = re.findall(r"^WORD=(.*)$", r.stdout, re.M)
    return got[-1], (tmp / "stub" / "calls.log").exists(), r.stdout


# (subcommand's words, stdout word, exit code, JSON written) -> the word v5_rule reports
RULE_MATRIX = [
    ("continue pause", "continue", 0, True, "continue"),
    ("continue pause", "pause", 0, True, "pause"),
    ("continue pause", "refused", 3, True, "refused"),
    ("continue pause", "continue", 3, True, "refused"),
    ("continue pause", "pause", 3, True, "refused"),
    ("continue pause", "refused", 0, True, "refused"),
    ("continue pause", "", 0, True, "refused"),
    ("continue pause", "", 2, False, "refused"),
    ("continue pause", "Continue", 0, True, "refused"),
    ("continue pause", "continue", 0, False, "refused"),
    ("continue pause", "room", 0, True, "refused"),
    ("room no_room", "room", 0, True, "room"),
    ("room no_room", "no_room", 0, True, "no_room"),
    ("room no_room", "refused", 3, True, "refused"),
    ("room no_room", "room", 1, True, "refused"),
    ("room no_room", "continue", 0, True, "refused"),
    ("fires quiet", "fires", 0, True, "fires"),
    ("fires quiet", "quiet", 0, True, "quiet"),
    ("fires quiet", "refused", 3, True, "refused"),
    ("fires quiet", "fires", 3, True, "refused"),
    ("wins quiet", "wins", 0, True, "wins"),
    ("wins quiet", "quiet", 0, True, "quiet"),
    ("wins quiet", "refused", 3, True, "refused"),
    ("wins quiet", "wins quiet", 0, True, "refused"),
    ("room no_room", " room", 0, True, "refused"),
    ("continue pause", "continue\npause", 0, True, "refused"),
]


@pytest.mark.parametrize(("words", "word", "rc", "json_out", "want"), RULE_MATRIX)
def test_v5_rule_reads_only_a_consistent_word(
    tmp_path: Path, words: str, word: str, rc: int, json_out: bool, want: str
) -> None:
    got, called, _ = run_rule(tmp_path, words, word, rc, json_out=json_out)
    assert called
    assert got == want


@pytest.mark.parametrize("word", ["continue", "room", "fires", "wins"])
def test_v5_rule_never_runs_a_binary_that_is_not_the_pinned_one(tmp_path: Path, word: str) -> None:
    got, called, out = run_rule(tmp_path, f"{word} other", word, 0, pin_ok=False)
    assert got == "refused"
    assert not called, "the binary ran although its sha256 is not the pin"
    assert "not the pinned one; NOT RUN" in out


def test_v5_rule_refuses_when_the_binary_is_absent(tmp_path: Path) -> None:
    r = bash(
        lib(tmp_path, f'RULES_V5="{tmp_path}/nope"; RULES_V5_SHA256={"a" * 64}')
        + """
v5_rule t "continue pause" v5-pause
echo "WORD=$V5_WORD"
"""
    )
    assert re.findall(r"^WORD=(.*)$", r.stdout, re.M)[-1] == "refused"


def test_v5_rules_call_checks_the_pin_before_running(tmp_path: Path) -> None:
    stub = stub_rules(tmp_path, EVAL_ROW, 0)
    good = bash(
        lib(tmp_path, f'RULES_V5="{stub}"; RULES_V5_SHA256={sha256(stub)}')
        + 'X=$(v5_rules_call eval-row --x y) && echo "GOT=$X"'
    )
    assert f"GOT={EVAL_ROW}" in good.stdout
    (tmp_path / "stub" / "calls.log").unlink()
    bad = bash(
        lib(tmp_path, f'RULES_V5="{stub}"; RULES_V5_SHA256={"0" * 64}')
        + 'X=$(v5_rules_call eval-row --x y); echo "RC=$? GOT=$X"'
    )
    assert "RC=3 GOT=" in bad.stdout and EVAL_ROW not in bad.stdout
    assert not (tmp_path / "stub" / "calls.log").exists()


# --- what a hold marker says (Fable's ruling A, 2026-10-02) ---------------------------------
# V5_SAID is what $Q/v5.paused carries when R9 holds: the word the binary said when it is one of
# the subcommand's words; unknown:<word> when it exited 0 with one word-shaped token it may not
# say; refused for everything else (a refusal, an inconsistent exit, garbage, a listed word with
# no JSON, a binary that is not the pinned one). (stdout, exit code, JSON written) -> V5_SAID.
SAID_MATRIX = [
    ("continue", 0, True, "continue"),
    ("pause", 0, True, "pause"),
    ("refused", 3, True, "refused"),
    ("hold", 0, True, "unknown:hold"),
    ("hold", 0, False, "unknown:hold"),
    ("Continue", 0, True, "unknown:Continue"),
    ("fires:j6f", 0, True, "unknown:fires:j6f"),
    ("continue", 3, True, "refused"),
    ("hold", 3, True, "refused"),
    ("hold", 2, True, "refused"),
    ("refused", 0, True, "refused"),
    ("continue", 0, False, "refused"),
    ("", 0, True, "refused"),
    ("wins quiet", 0, True, "refused"),
    ("continue\npause", 0, True, "refused"),
    ("h" * 65, 0, True, "refused"),
    ("hold!", 0, True, "refused"),
]


@pytest.mark.parametrize(("word", "rc", "json_out", "want"), SAID_MATRIX)
def test_v5_rule_says_what_the_binary_said_as_a_marker_can_carry_it(
    tmp_path: Path, word: str, rc: int, json_out: bool, want: str
) -> None:
    got, _, out = run_rule(tmp_path, "continue pause", word, rc, json_out=json_out)
    assert re.findall(r"^SAID=(.*)$", out, re.M)[-1] == want
    # the action still reads V5_WORD: only a listed word with its JSON ever runs anything
    assert got == (want if want in ("continue", "pause") else "refused")


def test_v5_rule_says_refused_for_a_binary_that_is_not_the_pinned_one(tmp_path: Path) -> None:
    _, called, out = run_rule(tmp_path, "continue pause", "hold", 0, pin_ok=False)
    assert not called
    assert re.findall(r"^SAID=(.*)$", out, re.M)[-1] == "refused"


# --- the ledger read race (Fable's ruling B, 2026-10-02) ---------------------------------------
# A post-seed waiter appends its trajectory and control rows after it releases its lane's lock,
# and the other lane appends its own rows to the same ledger, so a reading can meet a row
# half-written as its ledger's last line. Only that refusal is retried. The binary's own words
# for it, from the v5 build run on a half-written ledger
# (AUDIT/v5-queue-2026-10-02/read-race-probe.txt; parse_rows and main in qd_post_f_rules.rs):
# exit 3, stdout `refused`, last stderr line
# `qd-post-f-rules: refused: <ledger> line <N>: malformed ledger line (<serde error>)`.
ROW = '{"row_id": "aaaaaaaa-1111-2222-3333-444444444444", "completed": true}'
HALF = '{"row_id": "half'
RETRIES_K = 5
RETRY_S = "10"
REFUSED_PREFIX = "qd-post-f-rules: refused: "


def malformed(ledger: Path, line: int) -> str:
    return (
        f"{REFUSED_PREFIX}{ledger} line {line}: malformed ledger line "
        "(EOF while parsing a string at line 1 column 16)"
    )


def half_written(d: Path, name: str = "ledger.jsonl") -> Path:
    """Two rows and half a third: line 3 is the last line, and it is malformed."""
    d.mkdir(parents=True, exist_ok=True)
    ledger = d / name
    ledger.write_text(f"{ROW}\n{ROW}\n{HALF}")
    return ledger


def completed(d: Path) -> Path:
    """Three whole rows: the writer finished line 3 after the binary read half of it."""
    d.mkdir(parents=True, exist_ok=True)
    ledger = d / "ledger.jsonl"
    ledger.write_text(f"{ROW}\n{ROW}\n{ROW}\n")
    return ledger


def inner_malformed(d: Path) -> Path:
    """Line 2 of 3 is malformed and line 3 is whole: not a row being written."""
    d.mkdir(parents=True, exist_ok=True)
    ledger = d / "ledger.jsonl"
    ledger.write_text(f"{ROW}\n{HALF}\n{ROW}\n")
    return ledger


def script_rules(tmp: Path, steps: list[tuple[int, str, str]]) -> Path:
    """A rules binary that answers its i-th call with steps[i] (the last step repeats): exit
    code, stdout, and the last stderr line. Like the real one it writes a JSON to a new --out
    and refuses an --out that already exists."""
    d = tmp / "stub"
    d.mkdir(parents=True, exist_ok=True)
    for i, (rc, out, err) in enumerate(steps):
        (d / f"{i}.rc").write_text(str(rc))
        (d / f"{i}.out").write_text(out)
        (d / f"{i}.err").write_text(err)
    (d / "n").write_text(str(len(steps)))
    return write_exec(
        d / "rules",
        f"""#!/bin/bash
echo "$*" >> "{d}/calls.log"
i=$(( $(wc -l < "{d}/calls.log") - 1 ))
n=$(cat "{d}/n")
if [ "$i" -ge "$n" ]; then i=$((n - 1)); fi
out=""; prev=""
for a in "$@"; do [ "$prev" = --out ] && out=$a; prev=$a; done
if [ -n "$out" ] && [ -e "$out" ]; then
  echo "qd-post-f-rules: refused: --out $out already exists" >&2
  echo refused
  exit 3
fi
if [ -n "$out" ]; then echo '{{"stub": true}}' > "$out"; fi
echo '{{"tool": "qd-post-f-rules", "stub": true}}' >&2
if [ -s "{d}/$i.err" ]; then cat "{d}/$i.err" >&2; echo >&2; fi
printf '%s\\n' "$(cat "{d}/$i.out")"
exit "$(cat "{d}/$i.rc")"
""",
    )


READING = """v5_rule t "fires quiet" seeds34 --preregistration "{prereg}" \\
  --f-ledger "{ledger}" --ft-row 0=x
echo "WORD=$V5_WORD SAID=$V5_SAID JSON=$V5_RULE_JSON"
"""
PLAIN = """X=$(v5_rules_call ft-rows --ledger "{ledger}" --ft-row 0=x)
echo "RC=$? GOT=[$X]"
"""


def race_run(
    tmp: Path,
    steps: list[tuple[int, str, str]],
    call: str,
    ledger: Path,
    prereg: Path | None = None,
) -> tuple[subprocess.CompletedProcess, int, list[str]]:
    """Run one reading against script_rules; `sleep` is recorded, not slept."""
    tmp.mkdir(parents=True, exist_ok=True)
    stub = script_rules(tmp, steps)
    extra = (
        f'RULES_V5="{stub}"; RULES_V5_SHA256={sha256(stub)}\n'
        f'sleep() {{ echo "$*" >> "{tmp}/sleeps"; }}'
    )
    r = bash(lib(tmp, extra) + call.format(ledger=ledger, prereg=prereg or tmp / "prereg.json"))
    assert r.returncode == 0, r.stdout + r.stderr
    log = tmp / "stub" / "calls.log"
    calls = len(log.read_text().splitlines()) if log.exists() else 0
    sleeps = (tmp / "sleeps").read_text().split() if (tmp / "sleeps").exists() else []
    return r, calls, sleeps


def word_of(r: subprocess.CompletedProcess) -> str:
    return re.findall(r"^WORD=(\S*)", r.stdout, re.M)[-1]


@pytest.mark.parametrize("via", ["reading", "plain"])
@pytest.mark.parametrize("ledger_at_check", ["still half-written", "completed since"])
def test_a_refusal_on_a_half_written_last_line_is_retried_and_the_answer_accepted(
    tmp_path: Path, via: str, ledger_at_check: str
) -> None:
    ledger = (half_written if ledger_at_check == "still half-written" else completed)(tmp_path)
    race = (3, "refused", malformed(ledger, 3))
    answer = (0, "fires", "") if via == "reading" else (0, EVAL_ROW, "")
    r, calls, sleeps = race_run(
        tmp_path, [race, race, answer], READING if via == "reading" else PLAIN, ledger
    )
    assert calls == 3, r.stdout + r.stderr
    assert sleeps == [RETRY_S, RETRY_S]
    assert "half-written" in r.stdout + r.stderr
    if via == "reading":
        assert word_of(r) == "fires"
        jsons = sorted((tmp_path / "dec").glob("t-*.json"))
        assert len(jsons) == 3, "each attempt writes its own new --out"
        assert re.findall(r"JSON=(\S*)", r.stdout)[-1] == str(jsons[-1])
        assert len(sorted((tmp_path / "dec").glob("t-*.stderr"))) == 3
    else:
        # only the answer reaches the caller's $(...): no `refused` from the earlier attempts
        assert f"RC=0 GOT=[{EVAL_ROW}]" in r.stdout


@pytest.mark.parametrize("via", ["reading", "plain"])
def test_a_last_line_refusal_that_persists_is_held_after_k_attempts(
    tmp_path: Path, via: str
) -> None:
    ledger = half_written(tmp_path)
    r, calls, sleeps = race_run(
        tmp_path,
        [(3, "refused", malformed(ledger, 3))],
        READING if via == "reading" else PLAIN,
        ledger,
    )
    assert calls == RETRIES_K, r.stdout + r.stderr
    assert sleeps == [RETRY_S] * (RETRIES_K - 1)
    if via == "reading":
        assert word_of(r) == "refused"
        assert "SAID=refused" in r.stdout
    else:
        assert "RC=3 GOT=[refused]" in r.stdout


def other_reasons(ledger: Path) -> dict[str, tuple[int, str, str]]:
    last = malformed(ledger, 3)
    return {
        "another reason": (
            3,
            "refused",
            "qd-post-f-rules: refused: the envelope is v5 seeds [0, 1, 2] only; got [0, 1]",
        ),
        "the race and another reason": (3, "refused", f"{last}; no target has room"),
        "another reason and the race": (
            3,
            "refused",
            f"qd-post-f-rules: refused: no target has room; {last.removeprefix(REFUSED_PREFIX)}",
        ),
        "exit 3 without refused on stdout": (3, "pause", last),
        "exit 2": (2, "refused", last),
        "exit 0": (0, "refused", last),
        "an --out that already exists": (
            3,
            "refused",
            f"qd-post-f-rules: refused: --out {ledger} already exists",
        ),
    }


@pytest.mark.parametrize(
    "case",
    [
        "another reason",
        "the race and another reason",
        "another reason and the race",
        "exit 3 without refused on stdout",
        "exit 2",
        "exit 0",
        "an --out that already exists",
    ],
)
def test_a_refusal_for_any_other_reason_is_held_at_once(tmp_path: Path, case: str) -> None:
    # the twin: the same stub, its first answer the race itself, is retried (so this test is
    # not satisfied by a waiter that never retries)
    twin_ledger = half_written(tmp_path / "twin")
    _, twin_calls, _ = race_run(
        tmp_path / "twin",
        [(3, "refused", malformed(twin_ledger, 3)), (0, "fires", "")],
        READING,
        twin_ledger,
    )
    assert twin_calls == 2
    ledger = half_written(tmp_path / "case")
    r, calls, sleeps = race_run(
        tmp_path / "case", [other_reasons(ledger)[case], (0, "fires", "")], READING, ledger
    )
    assert calls == 1, r.stdout + r.stderr
    assert sleeps == []
    assert word_of(r) == "refused"


@pytest.mark.parametrize(
    "case",
    [
        "an inner line",
        "an earlier line of a half-written ledger",
        "a line past the end",
        "a file the call names by a flag that is not a ledger",
        "a ledger the call does not name",
    ],
)
def test_a_malformed_line_that_is_not_the_last_line_is_not_retried(
    tmp_path: Path, case: str
) -> None:
    twin_ledger = half_written(tmp_path / "twin")
    _, twin_calls, _ = race_run(
        tmp_path / "twin",
        [(3, "refused", malformed(twin_ledger, 3)), (0, "fires", "")],
        READING,
        twin_ledger,
    )
    assert twin_calls == 2
    d = tmp_path / "case"
    prereg = None
    if case == "an inner line":
        ledger = inner_malformed(d)
        err = malformed(ledger, 2)
    elif case == "an earlier line of a half-written ledger":
        ledger = half_written(d)
        err = malformed(ledger, 2)
    elif case == "a line past the end":
        ledger = half_written(d)
        err = malformed(ledger, 4)
    elif case == "a file the call names by a flag that is not a ledger":
        ledger = half_written(d)
        prereg = half_written(d, "prereg.jsonl")
        err = malformed(prereg, 3)
    else:
        ledger = half_written(d)
        err = malformed(half_written(d, "other.jsonl"), 3)
    r, calls, sleeps = race_run(d, [(3, "refused", err), (0, "fires", "")], READING, ledger, prereg)
    assert calls == 1, r.stdout + r.stderr
    assert sleeps == []
    assert word_of(r) == "refused"


def test_the_read_race_retry_is_bounded_as_ruled() -> None:
    text = common_text()
    assert re.search(rf"^V5_READ_TRIES={RETRIES_K}$", text, re.M)
    assert re.search(rf"^V5_READ_RETRY_S={RETRY_S}$", text, re.M)
    # never gated on a trajectory waiter's .done: its CPU controls run after it releases the lock
    assert "traj-s2.done" not in "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )


ACTIONS = [
    # R9: continue -> run seeds 1-2; pause or anything else -> hold until V5_CONTINUE
    ("v5_pause_action continue", "run"),
    ("v5_pause_action pause", "hold"),
    ("v5_pause_action refused", "hold"),
    ("v5_pause_action ''", "hold"),
    ("v5_pause_action 'continue '", "hold"),
    ("v5_pause_action room", "hold"),
    ("v5_pause_action unknown:continue", "hold"),
    ("v5_pause_action no-seed-0-row", "hold"),
    # the arm's launch: room; no_room only with V5NW_HUMAN_YES; refused never
    ("v5_room_action room no", "run"),
    ("v5_room_action room yes", "run"),
    ("v5_room_action no_room no", "skip"),
    ("v5_room_action no_room yes", "run"),
    ("v5_room_action refused yes", "skip"),
    ("v5_room_action refused no", "skip"),
    ("v5_room_action '' yes", "skip"),
    ("v5_room_action quiet yes", "skip"),
    # (seeds 3-4's v5_s34_action rows went with seeds34: the amendment a29bca1 makes seeds 3-4
    # unconditional, Fable's ruling of ~17:25Z; test_seeds_3_4_run_unconditionally_and_
    # seeds34_is_never_read covers the new path)
    # the arm's reading: recorded, never acted on
    ("v5_noulw_action wins", "record"),
    ("v5_noulw_action quiet", "record"),
    ("v5_noulw_action refused", "record-refused"),
    ("v5_noulw_action ''", "record-refused"),
]


@pytest.mark.parametrize(("call", "want"), ACTIONS)
def test_each_word_maps_to_the_action_the_draft_names(tmp_path: Path, call: str, want: str) -> None:
    r = bash(lib(tmp_path) + f'echo "ACTION=$({call})"')
    assert re.findall(r"^ACTION=(.*)$", r.stdout, re.M) == [want], r.stdout + r.stderr


def test_the_draft_names_these_words() -> None:
    d = json.loads(DRAFT.read_text(encoding="utf-8"))
    r9 = d["readings"]["R9_pause_after_seed_0"]
    assert "prints continue iff" in r9 and "otherwise pause" in r9 and "V5_CONTINUE" in r9
    launch = d["arm_noul_weight"]["launch_condition"]
    assert "room or no_room" in launch and "V5NW_HUMAN_YES" in launch and "SKIPPED (R7)" in launch
    assert d["arm_noul_weight"]["outcomes"]["words"] == ["wins", "quiet", "refused"]


GOOD_PINS = f"""
RULES_V5_SHA256={"1" * 64}; V5_PREP_BIN_SHA256={"2" * 64}; V5_LANE_AT={LANE_AT}
V5_PREREG_SHA256={"3" * 64}; V5_DATE=2026-10-04; V5_PRELUDE_SHA256={"4" * 64}
V5_BOX={BOX}; V5_INSTANCE={INSTANCE}; V5_USD_PER_INSTANCE_HOUR={USD_INSTANCE}
V5_USD_PER_GPU_HOUR={USD_GPU}; V5_R9=keep; V5_HUMAN_YES='{HUMAN_YES}'
V5_C1=off; V5_C2A=off; V5_C2B=off; V5_LOWER=keep; V5_LRSET=f
V5_SPLIT=(--out /data/v5 --rev {BUILD_REV} --real-backbone "$BACKBONE")
"""


def test_v5_pins_unset_lists_every_unset_pin(tmp_path: Path) -> None:
    r = bash(lib(tmp_path) + 'echo "UNSET=$(v5_pins_unset)"')
    got = re.findall(r"^UNSET=(.*)$", r.stdout, re.M)[0].split()
    assert sorted(got) == sorted([*PINS, "V5_SPLIT"])
    r = bash(lib(tmp_path, GOOD_PINS) + 'echo "UNSET=$(v5_pins_unset)|CHECK=$(v5_pins_check)|"')
    assert "UNSET=|CHECK=|" in r.stdout, r.stdout


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("RULES_V5_SHA256=abc", "RULES_V5_SHA256 is 'abc'"),
        ("V5_LANE_AT=0123456", "V5_LANE_AT is '0123456'"),
        ("V5_DATE=04-10-2026", "V5_DATE is '04-10-2026'"),
        ("V5_C1=yes", "V5_C1 is 'yes'"),
        ("V5_C2B=ON", "V5_C2B is 'ON'"),
        ("V5_LOWER=dropped", "V5_LOWER is 'dropped'"),
        ("V5_LRSET=3e-5", "V5_LRSET is '3e-5'"),
        # the 2x H100 box's pins
        ("V5_BOX=gh200", "V5_BOX is 'gh200': v5 runs on the 2x H100 box, never the GH200"),
        ("V5_BOX=gh200-b", "V5_BOX is 'gh200-b': v5 runs on the 2x H100 box, never the GH200"),
        ("V5_BOX=H100x2", "V5_BOX is 'H100x2', not"),
        ("V5_BOX=h100/x2", "V5_BOX is 'h100/x2', not"),
        ("V5_INSTANCE=-x", "V5_INSTANCE is '-x', not"),
        ("V5_USD_PER_GPU_HOUR=abc", "V5_USD_PER_GPU_HOUR is 'abc', not a positive rate"),
        ("V5_USD_PER_GPU_HOUR=0", "V5_USD_PER_GPU_HOUR is '0', not a positive rate"),
        (
            "V5_USD_PER_INSTANCE_HOUR=-8.38",
            "V5_USD_PER_INSTANCE_HOUR is '-8.38', not a positive rate",
        ),
        # a swapped column: the instance rate equal to the per-GPU one (DESIGN-4's error)
        (
            f"V5_USD_PER_INSTANCE_HOUR={USD_GPU}",
            f"V5_USD_PER_INSTANCE_HOUR {USD_GPU} is not V5_USD_PER_GPU_HOUR {USD_GPU} x 2 GPUs",
        ),
        ("V5_R9=yes", "V5_R9 is 'yes', not keep or waive"),
        ("V5_HUMAN_YES=", "V5_HUMAN_YES is empty"),
        ("V5_HUMAN_YES='  '", "V5_HUMAN_YES is empty"),
    ],
)
def test_v5_pins_check_refuses_a_malformed_pin(tmp_path: Path, override: str, message: str) -> None:
    r = bash(lib(tmp_path, GOOD_PINS + override) + 'echo "CHECK=$(v5_pins_check | tr "\\n" ";")"')
    assert message in r.stdout, r.stdout + r.stderr


SPLIT_OK = (
    f"--out /home/ubuntu/phase5 --no-repo-history --rev {BUILD_REV} "
    "--defect-class data/pool/commitpackft-composed-v2 --defect-noul data/pool/defect-noul-v3c "
    '--general-record "$REC" --general-max-rows 200000 --exclude-identity-keys /home/ubuntu/x.txt '
    '--real-backbone "$BACKBONE"'
)


@pytest.mark.parametrize(
    ("split", "ok", "why"),
    [
        (SPLIT_OK, True, ""),
        (SPLIT_OK + " --defect-max-rows 5", True, ""),
        (
            SPLIT_OK.replace("/home/ubuntu/x.txt", "/home/ubuntu/phase5/data/heldout/heldout.json"),
            False,
            "rule 3",
        ),
        (SPLIT_OK + " --defect-noul /x/containment-request.jsonl", False, "rule 3"),
        (SPLIT_OK + " --lr 1e-5", False, "--lr is not a data flag"),
        (SPLIT_OK + " --batch-order seed", False, "--batch-order is not a data flag"),
        (SPLIT_OK + " --seeds 0", False, "--seeds is not a data flag"),
        (SPLIT_OK + " --out /home/ubuntu/other", False, "2 --out"),
        (SPLIT_OK.replace(BUILD_REV, "89abcde"), False, "not a full 40-hex commit"),
        (SPLIT_OK.replace('"$BACKBONE"', "/elsewhere"), False, "not the box's snapshot"),
        (SPLIT_OK.replace('--real-backbone "$BACKBONE"', ""), False, "0 --real-backbone"),
        (SPLIT_OK.replace("/home/ubuntu/phase5", "phase5"), False, "not an absolute path"),
    ],
)
def test_v5_split_check(tmp_path: Path, split: str, ok: bool, why: str) -> None:
    r = bash(lib(tmp_path, f"V5_SPLIT=({split})") + 'v5_split_check; echo "RC=$? DATA=$V5_DATA"')
    if ok:
        assert "RC=0 DATA=/home/ubuntu/phase5" in r.stdout, r.stdout
    else:
        assert "RC=3" in r.stdout and why in r.stdout, r.stdout


def test_v5_ctl_split_drops_out_and_backbone(tmp_path: Path) -> None:
    r = bash(
        lib(tmp_path, f"V5_SPLIT=({SPLIT_OK})") + 'v5_ctl_split; printf "<%s>" "${V5_CTL_SPLIT[@]}"'
    )
    got = re.findall(r"<([^>]*)>", r.stdout)
    assert "--out" not in got and "--real-backbone" not in got
    assert got[:3] == ["--no-repo-history", "--rev", BUILD_REV]
    assert "--exclude-identity-keys" in got


@pytest.mark.parametrize(
    ("pins", "want"),
    [
        ("V5_C1=off V5_C2A=off V5_C2B=off V5_LOWER=keep V5_LRSET=f", BASE_RECIPE),
        (
            "V5_C1=on V5_C2A=off V5_C2B=off V5_LOWER=keep V5_LRSET=f",
            [*BASE_RECIPE, "--option-permutation-seed", "20260919"],
        ),
        (
            "V5_C1=off V5_C2A=on V5_C2B=on V5_LOWER=keep V5_LRSET=f",
            [*BASE_RECIPE, "--train-attention-mask", "none", "--fused-adamw"],
        ),
        (
            "V5_C1=off V5_C2A=off V5_C2B=off V5_LOWER=drop V5_LRSET=f",
            [
                x
                for x in BASE_RECIPE
                if x not in ("--lower-layers-n", "8", "--lower-layers-lr-scale", "0.1")
            ],
        ),
        (
            "V5_C1=off V5_C2A=off V5_C2B=off V5_LOWER=keep V5_LRSET=j6dv4",
            ["--optimizer", "master", "--lr", "3e-5", "--beta2", "0.95", *BASE_RECIPE[4:]],
        ),
    ],
)
def test_v5_recipe_is_f_plus_the_added_flags_and_the_pinned_conditionals(
    tmp_path: Path, pins: str, want: list[str]
) -> None:
    r = bash(
        lib(tmp_path, pins.replace(" ", "; "))
        + 'v5_recipe; echo "RC=$?"; printf "<%s>" "${V5_RECIPE[@]}"'
    )
    assert "RC=0" in r.stdout
    got = re.findall(r"<([^>]*)>", r.stdout)
    assert got == want
    assert ("--batch-order", "seed") in recipe_pairs(got) and ("--min-lr", "0") in recipe_pairs(got)


@pytest.mark.parametrize("pins", ["V5_C1=UNSET", "V5_LRSET=UNSET", "V5_LOWER=both", "V5_C2B=maybe"])
def test_v5_recipe_refuses_an_unset_or_unknown_decision(tmp_path: Path, pins: str) -> None:
    base = "V5_C1=off; V5_C2A=off; V5_C2B=off; V5_LOWER=keep; V5_LRSET=f; "
    r = bash(lib(tmp_path, base + pins) + 'v5_recipe; echo "RC=$?"')
    assert "RC=3" in r.stdout


def read_prereg(
    tmp: Path, prereg: dict | str, rate: str = USD_GPU, box: str = BOX
) -> subprocess.CompletedProcess:
    p = tmp / "prereg.json"
    p.write_text(prereg if isinstance(prereg, str) else json.dumps(prereg), encoding="utf-8")
    return bash(
        lib(
            tmp, f'V5_PREREG="{p}"; PY="{sys.executable}"; V5_USD_PER_GPU_HOUR={rate}; V5_BOX={box}'
        )
        + """
v5_read_prereg; echo "RC=$?"
echo "NUMS=$V5_EST_SEED_USD $V5_EST_SEED_H $V5_EST_J5_USD $V5_EST_J5_H" \
  "$V5_APPROVED_USD $V5_APPROVED_H $V5NW_W"
"""
    )


def test_v5_read_prereg_reads_the_drafts_numbers(tmp_path: Path) -> None:
    """Changed for the 2x H100 box: the DRAFT is the amended one (a29bca1), priced at the pinned
    per-GPU rate ($4.19, no longer COST's GH200 $2.29), 11 runs: $309.7 / 74.0 GPU-h."""
    r = read_prereg(tmp_path, renamed_prereg())
    assert "RC=0" in r.stdout, r.stdout + r.stderr
    assert "NUMS=29.3 7.0 25.1 6.0 309.7 74.0 4" in r.stdout


def test_v5_read_prereg_prices_with_the_per_gpu_rate(tmp_path: Path) -> None:
    """The per-seed $ is checked against the per-GPU rate: the amended DRAFT reads at $4.19 and
    refuses at the GH200's $2.29 on exactly the two rate checks."""
    r = read_prereg(tmp_path, renamed_prereg(), "2.29")
    assert "RC=3" in r.stdout and "REFUSED" in r.stdout, r.stdout + r.stderr
    assert (
        "v5 seed h x rate = v5 seed $" in r.stdout and "J5' seed h x rate = J5' seed $" in r.stdout
    )
    assert "the four blocks = total" not in r.stdout


def test_v5_read_prereg_refuses_the_pre_amendment_form(tmp_path: Path) -> None:
    """1ab477f's DRAFT (if_seeds_3_4, 60 GPU-h, priced at $2.29) is not the form this queue
    reads: seeds 3-4 are a block of the total now, and there is no hours if_seeds_3_4."""
    old = subprocess.run(
        ["git", "-C", str(REPO), "show", "1ab477f:campaign/v5-preregistered.DRAFT.json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if old.returncode != 0:
        pytest.skip(f"1ab477f is not in this checkout: NOT RUN ({old.stderr.strip()})")
    d = json.loads(old.stdout)
    d.pop("draft")
    for rate in ("2.29", USD_GPU):
        r = read_prereg(tmp_path, d, rate)
        assert "RC=3" in r.stdout and "REFUSED" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("rate", ["UNSET", "", "abc", "0"])
def test_v5_read_prereg_refuses_without_a_positive_per_gpu_rate(tmp_path: Path, rate: str) -> None:
    r = read_prereg(tmp_path, renamed_prereg(), f"'{rate}'")
    assert "RC=3" in r.stdout and "V5_USD_PER_GPU_HOUR" in r.stdout, r.stdout + r.stderr


def test_v5_read_prereg_refuses_the_draft_itself(tmp_path: Path) -> None:
    r = read_prereg(tmp_path, DRAFT.read_text(encoding="utf-8"))
    assert "RC=3" in r.stdout and "'draft' key" in r.stdout


def test_v5_read_prereg_refuses_a_probe_margin_it_does_not_carry(tmp_path: Path) -> None:
    d = renamed_prereg()
    d["hardware"]["probe"]["margin_gib"] = 10
    r = read_prereg(tmp_path, d)
    assert "RC=3" in r.stdout and "margin_gib" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("box", ["h100x3", "gh200b"])
def test_v5_read_prereg_refuses_a_box_the_ledger_names_do_not_carry(
    tmp_path: Path, box: str
) -> None:
    """V5_BOX is a pin whose value the pre-registration fixes (seeds.ledger and hardware.ledger
    name h100x2-v5-, h100x2-v5-noulw- and h100x2-v5-probe-): another value refuses."""
    r = read_prereg(tmp_path, renamed_prereg(), box=box)
    assert "RC=3" in r.stdout and f"{box}-v5-" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["launch"]["projected_cost_usd"].__setitem__(
            "check",
            d["launch"]["projected_cost_usd"]["check"].replace("~ 7.0 h, $29.3", "~ 7.0 h, $31.0"),
        ),
        lambda d: d["launch"]["projected_cost_usd"].__setitem__("total", 251.4),
        lambda d: d["launch"]["projected_cost_usd"].__setitem__("v5_seeds_3_4", 60.0),
        lambda d: d["launch"]["projected_cost_usd"].pop("v5_seeds_3_4"),
        lambda d: d["launch"]["projected_cost_usd"].__setitem__("if_seeds_3_4", 58.6),
        lambda d: d["launch"]["projected_gpu_hours"].__setitem__("total", 60.0),
        lambda d: d["launch"]["projected_gpu_hours"].__setitem__("if_seeds_3_4", 14.0),
        lambda d: d["launch"]["projected_cost_usd"].__setitem__(
            "check", "no per-seed figures here"
        ),
        lambda d: d["arm_noul_weight"]["w"].__setitem__("value", 0),
        lambda d: d["arm_noul_weight"]["w"].__setitem__("value", "4"),
        lambda d: d["launch"].pop("projected_gpu_hours"),
        lambda d: d.pop("hardware"),
    ],
)
def test_v5_read_prereg_refuses_figures_that_disagree(tmp_path: Path, mutate) -> None:
    d = renamed_prereg()
    mutate(d)
    r = read_prereg(tmp_path, d)
    assert "RC=3" in r.stdout and "REFUSED" in r.stdout, r.stdout


#: v5_budget_ok's inputs as v5_read_prereg sets them from the amended DRAFT (a29bca1): $4.19 per
#: GPU-hour, a seed ~ 7.0 h / $29.3, a J5' seed ~ 6.0 h / $25.1, 11 runs ~ $309.7 / 74.0 GPU-h.
DRAFT_MONEY = (
    f"V5_USD_PER_GPU_HOUR={USD_GPU}; V5_APPROVED_USD=309.7; V5_APPROVED_H=74.0; "
    "V5_EST_SEED_USD=29.3; V5_EST_SEED_H=7.0; V5_EST_J5_USD=25.1; V5_EST_J5_H=6.0"
)


def budget(
    tmp: Path,
    spent_s: int | None,
    *,
    yes: str | None = None,
    est: str = "29.3 7.0",
    files: dict[str, str] | None = None,
) -> str:
    """Changed for the 2x H100 box: the rate variable is the pinned V5_USD_PER_GPU_HOUR (was
    V5_USD_PER_HOUR, read from COST's GH200 instance rate), and the figures are the amended
    DRAFT's (11 runs); the approved total no longer grows on a seeds34 word."""
    q = tmp / "q"
    q.mkdir(parents=True, exist_ok=True)
    if spent_s is not None:
        (q / "v5.spend").write_text(f"2026-10-04T00:00:00Z\tprior\t{spent_s}\t0\n")
    if yes is not None:
        (q / "V5_OVER_BUDGET_YES").write_text(yes)
    for name, text in (files or {}).items():
        (q / name).write_text(text)
    r = bash(lib(tmp, DRAFT_MONEY) + f'v5_budget_ok {est} "v5 seed 2"; echo "RC=$?"')
    return r.stdout


def test_v5_budget_ok_passes_inside_the_approved_total(tmp_path: Path) -> None:
    assert "RC=0" in budget(tmp_path / "a", None)
    # 66 h = $276.54 spent; + ~$29.3 / 7.0 h = $305.84 / 73 h, inside $309.7 / 74.0 h
    assert "RC=0" in budget(tmp_path / "b", 66 * 3600)
    # 67 h = $280.73; + $29.3 = $310.03 crosses $309.7 although 67 + 7.0 = 74 h does not
    assert "RC=3" in budget(tmp_path / "c", 67 * 3600)


def test_v5_budget_ok_refuses_a_run_that_would_cross_it(tmp_path: Path) -> None:
    out = budget(tmp_path, 68 * 3600)  # 68 + 7.0 > 74 GPU-h
    assert "RC=3" in out and "REFUSED" in out and "NOT RUN" in out


def test_v5_budget_ok_runs_past_it_only_on_the_humans_words(tmp_path: Path) -> None:
    assert "RC=3" in budget(tmp_path / "a", 68 * 3600, yes="")
    out = budget(tmp_path / "b", 68 * 3600, yes="Bharath: yes, finish the block")
    assert "RC=0" in out and "Bharath: yes, finish the block" in out


def test_v5_budget_ok_does_not_grow_on_a_seeds34_word(tmp_path: Path) -> None:
    """Replaces test_v5_budget_ok_adds_seeds_3_4_once_seeds34_fired: seeds 3-4 are a block of
    the approved 11-run total since the amendment a29bca1, so no word adds to it."""
    out = budget(tmp_path, 68 * 3600, files={"v5s34.word": "fires\n"})
    assert "RC=3" in out and "~$309.7 / 74.0 GPU-h" in out, out


def test_v5_budget_ok_counts_what_the_other_lanes_running_jobs_may_still_spend(
    tmp_path: Path,
) -> None:
    """Two lanes: a run that fits the approved total alone but not beside the jobs still running
    waits (exit 4) instead of starting, and starts once they end; one over the total even alone
    is refused (exit 3). A running job counts its estimate less what v5.spend already holds for
    it (its own .spend), and a seed job runs until its trajectory waiter is done."""
    running = {"v5job-v5-s0.claimed": "lane 0\n"}
    # 61 h spent + 7.0 h for v5-s0 still running + 7.0 h = 75 h > 74 h; alone 68 h fits
    out = budget(tmp_path / "a", 61 * 3600, files=running)
    assert "RC=4" in out and "waits" in out and "REFUSED" not in out, out
    # its job is done and no trajectory waiter was started: nothing is running
    out = budget(tmp_path / "b", 61 * 3600, files={**running, "v5job-v5-s0.done": "ft x\n"})
    assert "RC=0" in out, out
    # done, but its trajectory waiter is still running: it still counts
    out = budget(
        tmp_path / "c",
        61 * 3600,
        files={**running, "v5job-v5-s0.done": "ft x\n", "v5traj-s0.queued": ""},
    )
    assert "RC=4" in out, out
    # 6 of v5-s0's 7.0 h are already in v5.spend (62 h in all): 62 + 1 + 7 = 70 h fits;
    # counting the whole estimate again (62 + 7 + 7 = 76 h) would not
    out = budget(tmp_path / "d", 62 * 3600, files={**running, "v5job-v5-s0.spend": "21600\n"})
    assert "RC=0" in out, out
    # over the total even alone: refused, whatever runs
    out = budget(tmp_path / "e", 68 * 3600, files=running)
    assert "RC=3" in out and "REFUSED" in out, out


def test_v5_spend_add_and_running_total(tmp_path: Path) -> None:
    """Changed for the 2x H100 box: the rate is the pinned per-GPU rate ($4.19, was COST's
    GH200 $2.29) and the approved total the amended DRAFT's."""
    r = bash(
        lib(tmp_path, f"V5_USD_PER_GPU_HOUR={USD_GPU}; V5_APPROVED_USD=309.7; V5_APPROVED_H=74.0")
        + 'v5_spend_add "v5 seed 0 train+score" 3600; v5_spend_add "x" 7200'
    )
    assert "v5 running total: 1.0000 GPU-h, $4.19 of the approved ~$309.7 / 74.0 GPU-h" in r.stdout
    assert "v5 running total: 3.0000 GPU-h, $12.57 of the approved" in r.stdout
    lines = (tmp_path / "q" / "v5.spend").read_text().splitlines()
    assert [ln.split("\t")[1:] for ln in lines] == [
        ["v5 seed 0 train+score", "3600", "4.19"],
        ["x", "7200", "8.38"],
    ]


def test_v5_spend_add_also_charges_the_running_job(tmp_path: Path) -> None:
    r = bash(
        lib(tmp_path, f"V5_USD_PER_GPU_HOUR={USD_GPU}; V5_APPROVED_USD=1; V5_APPROVED_H=1")
        + 'V5_JOB=v5-s1; v5_spend_add "v5 seed 1 train+score" 100; v5_spend_add "x" 20; '
        + 'V5_JOB=""; v5_spend_add "probe" 7'
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "q" / "v5job-v5-s1.spend").read_text().split() == ["100", "20"]
    assert len((tmp_path / "q" / "v5.spend").read_text().splitlines()) == 3


# --- the box's cost, as the tool prices it ---------------------------------------------------


def test_the_tool_refuses_the_whole_instance_rate_on_one_visible_gpu() -> None:
    """Why a lane's COST is the per-GPU rate in both flags (verified against the tool's own
    CostEstimate, python/qd_train/run_control.py): under CUDA_VISIBLE_DEVICES=N a run counts one
    GPU (tools/run_cost.py n_gpus_for_device is torch.cuda.device_count(), which honours the
    variable: inferred, not run on a CUDA box). At one GPU the tool refuses the instance rate
    beside the per-GPU one ($8.38 with $4.19), and $8.38 alone would price each lane's run as
    the whole box. $4.19 in both flags passes at one GPU and is refused at two, so a lane that
    somehow saw both GPUs fails at argv time, before the tower loads."""
    sys.path.insert(0, str(REPO / "python"))
    from qd_train.run_control import CostEstimate, WallClockCap

    cap = WallClockCap(cap_s=32400)

    def est(n: int, per_instance: float, per_gpu: float | None) -> CostEstimate:
        return CostEstimate.for_device(
            cap=cap,
            device="cuda",
            n_gpus=n,
            usd_per_hour=per_instance,
            usd_per_gpu_hour=per_gpu,
            instance=INSTANCE,
        )

    with pytest.raises(ValueError, match="disagree at n_gpus=1"):
        est(1, float(USD_INSTANCE), float(USD_GPU))
    one = est(1, float(USD_GPU), float(USD_GPU))
    assert round(one.projected_usd, 2) == 37.71 and one.requires_human_approval
    with pytest.raises(ValueError, match="x 2 GPUs"):
        est(2, float(USD_GPU), float(USD_GPU))
    assert round(est(1, float(USD_INSTANCE), None).projected_usd, 2) == 75.42  # the whole box


def lane_lib(tmp: Path, extra: str = "") -> str:
    return lib(tmp, GOOD_PINS + extra)


def test_v5_lane_set_pins_the_lane_gpu_and_the_per_gpu_cost(tmp_path: Path) -> None:
    r = bash(
        lane_lib(tmp_path)
        + 'v5_lane_set 1; echo "RC=$? GPU=$V5_GPU CVD=$(printenv CUDA_VISIBLE_DEVICES)"; '
        + 'printf "<%s>" "${V5_COST[@]}"; echo'
    )
    assert "RC=0 GPU=1 CVD=1" in r.stdout, r.stdout + r.stderr
    assert re.findall(r"<([^>]*)>", r.stdout) == [
        "--instance",
        f"{INSTANCE}:gpu1",
        "--usd-per-hour",
        USD_GPU,
        "--usd-per-gpu-hour",
        USD_GPU,
    ]
    for bad in ("2", "", "x", "01"):
        r = bash(lane_lib(tmp_path) + f'v5_lane_set "{bad}"; echo "RC=$?"')
        assert "RC=3" in r.stdout, (bad, r.stdout)


def test_v5_lock_refuses_without_a_lane(tmp_path: Path) -> None:
    r = bash(lane_lib(tmp_path) + 'v5_lock; echo "RC=$?"')
    assert "RC=3" in r.stdout and not list((tmp_path / "q").glob("gpu*.lock")), r.stdout


def test_v5_approved_states_the_box_rate_and_the_humans_words(tmp_path: Path) -> None:
    r = bash(
        lane_lib(tmp_path)
        + 'echo "YES=$(v5_approved "v5 seed 0")"; '
        + 'echo "PROBE=$(v5_approved "v5 memory probe" "$V5_PROBE_CAP_S")"'
    )
    said = re.findall(r"^YES=(.*)$", r.stdout, re.M)[0]
    assert said.startswith(HUMAN_YES) and BOX_WORDS in said
    assert f"v5 seed 0 capped 32400 s at ${USD_GPU}/GPU-h = $37.71" in said
    # the GH200's yes and rate are not cited: d24c865 was a yes on ~$137 / ~60 GPU-h
    assert "d24c865" not in said and "2.29" not in said and "20.61" not in said
    probe = re.findall(r"^PROBE=(.*)$", r.stdout, re.M)[0]
    assert probe.startswith(HUMAN_YES)
    assert f"v5 memory probe capped 1800 s at ${USD_GPU}/GPU-h = $2.10" in probe


def real_flock(fakebin: Path) -> None:
    """A flock that locks: fcntl.flock on the fd the shell hands it. The lock belongs to the open
    file description, which the calling shell keeps open after this process exits."""
    write_exec(
        fakebin / "flock",
        f"#!{sys.executable}\nimport fcntl, sys\nfcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX)\n",
    )


def hold_lane_lock(tmp: Path, lane: str, tag: str, fakebin: Path) -> subprocess.Popen:
    script = lane_lib(tmp) + (
        f'v5_lane_set {lane} || exit 9\nv5_lock || exit 8\ntouch "{tmp}/{tag}.in"\n'
        f'/bin/sleep 1\ntouch "{tmp}/{tag}.out"\n'
    )
    return subprocess.Popen(
        [BASH, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={"PATH": f"{fakebin}:{PATH}", "HOME": str(tmp), "LC_ALL": "C"},
    )


@pytest.mark.parametrize(("lanes", "together"), [(("0", "1"), True), (("1", "1"), False)])
def test_two_waiters_on_two_locks_proceed_concurrently_and_on_one_lock_serialize(
    tmp_path: Path, lanes: tuple[str, str], together: bool
) -> None:
    fakebin = tmp_path / "fakebin"
    real_flock(fakebin)
    ps = [hold_lane_lock(tmp_path, lane, f"w{i}", fakebin) for i, lane in enumerate(lanes)]
    outs = [p.communicate(timeout=30)[0] for p in ps]
    assert [p.returncode for p in ps] == [0, 0], outs
    t = {
        f"w{i}.{e}": (tmp_path / f"w{i}.{e}").stat().st_mtime_ns
        for i in (0, 1)
        for e in ("in", "out")
    }
    overlap = max(t["w0.in"], t["w1.in"]) < min(t["w0.out"], t["w1.out"])
    assert overlap is together, t
    assert sorted(p.name for p in (tmp_path / "q").glob("gpu*.lock")) == sorted(
        {f"gpu{lane}.lock" for lane in lanes}
    )


# --- the rule ---------------------------------------------------------------------------------
# v5_next_job reads only marker files in $Q and prints what a lane does next: stop, hold,
# "decide s34|room|arm", "job <id>", wait, or none.

JOBS = (
    "v5-s0",
    "v5-s1",
    "v5-s2",
    "v5-s3",
    "v5-s4",
    "v5nw-s0",
    "v5nw-s1",
    "v5nw-s2",
    "j5-s0",
    "j5-s1",
    "j5-s2",
)
FT = {j: f"{i:08x}-aaaa-bbbb-cccc-dddddddddddd" for i, j in enumerate(JOBS)}


def next_job(tmp: Path, files: dict[str, str], *, r9: str = "keep") -> str:
    q = tmp / "q"
    q.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (q / name).write_text(text)
    r = bash(lib(tmp, f"V5_R9={r9}") + 'echo "NEXT=$(v5_next_job)"')
    got = re.findall(r"^NEXT=(.*)$", r.stdout, re.M)
    assert got, r.stdout + r.stderr
    return got[-1]


def ran(*jobs: str, ft: bool = True) -> dict[str, str]:
    out = {}
    for j in jobs:
        out[f"v5job-{j}.claimed"] = "lane 0\n"
        out[f"v5job-{j}.done"] = f"ft {FT[j]}\n" if ft else "no-ft\n"
    return out


def running(*jobs: str) -> dict[str, str]:
    return {f"v5job-{j}.claimed": "lane 0\n" for j in jobs}


NO_ROOM = {"v5nw.room": "no_room\n", "v5nw.launch": "skip\n"}
ROOM = {"v5nw.room": "room\n", "v5nw.launch": "run\n"}
MAIN = ("v5-s0", "v5-s1", "v5-s2", "v5-s3", "v5-s4")


@pytest.mark.parametrize(
    ("r9", "files", "want"),
    [
        # the spec's own case: lane B takes J5' s0 once s0 is done and s2 is not (here every
        # main seed is taken, so J5' s0 is the earliest job whose inputs are ready)
        ("waive", {**ran("v5-s0", "v5-s1"), **running("v5-s2", "v5-s3", "v5-s4")}, "job j5-s0"),
        (
            "keep",
            {
                **ran("v5-s0", "v5-s1"),
                **running("v5-s2", "v5-s3", "v5-s4"),
                "v5r9.word": "continue\n",
            },
            "job j5-s0",
        ),
        # nothing has run: seed 0 first
        ("keep", {}, "job v5-s0"),
        ("waive", {}, "job v5-s0"),
        # R9 kept: seeds 1-4 wait for its word; waived: they need none (seeds 3-4 included,
        # unconditional since the amendment a29bca1: seeds34 is not read)
        ("keep", running("v5-s0"), "wait"),
        ("waive", running("v5-s0"), "job v5-s1"),
        ("waive", running("v5-s0", "v5-s1"), "job v5-s2"),
        ("waive", running("v5-s0", "v5-s1", "v5-s2"), "job v5-s3"),
        ("keep", {**running("v5-s0"), "v5r9.word": "continue\n"}, "job v5-s1"),
        ("keep", {**running("v5-s0", "v5-s1", "v5-s2"), "v5r9.word": "continue\n"}, "job v5-s3"),
        # a seeds34 word from anywhere changes nothing
        ("waive", {**running("v5-s0", "v5-s1", "v5-s2"), "v5s34.word": "quiet\n"}, "job v5-s3"),
        # R9 held: every lane holds, J5' s0 included, until the human's words
        ("keep", {**ran("v5-s0"), "v5r9.word": "pause\n", "v5.paused": "pause\n"}, "hold"),
        ("keep", {**ran("v5-s0"), "v5r9.word": "refused\n", "V5_CONTINUE": ""}, "hold"),
        (
            "keep",
            {**ran("v5-s0"), "v5r9.word": "pause\n", "V5_CONTINUE": "Bharath: go on\n"},
            "job v5-s1",
        ),
        # V5_STOP: no lane starts anything
        ("waive", {"V5_STOP": "Bharath: stop\n"}, "stop"),
        # the room is decided before any pick, once seeds 0-2 are done (never 3-4), once
        ("waive", ran("v5-s0", "v5-s1", "v5-s2"), "decide room"),
        ("waive", {**ran("v5-s0", "v5-s1", "v5-s2"), **ROOM}, "job v5-s3"),
        ("waive", {**ran(*MAIN), **ROOM}, "job v5nw-s0"),
        # no room: J5' takes the arm's slots
        (
            "waive",
            {**ran("v5-s0", "v5-s1", "v5-s2"), **running("v5-s3", "v5-s4"), **NO_ROOM},
            "job j5-s0",
        ),
        # refused is never read as room
        (
            "waive",
            {**ran(*MAIN), "v5nw.room": "refused\n", "v5nw.launch": "skip\n"},
            "job j5-s0",
        ),
        # the arm on its launch word only; its reading once its three seeds are done
        ("waive", {**ran(*MAIN, "v5nw-s0", "v5nw-s1", "v5nw-s2"), **ROOM}, "decide arm"),
        (
            "waive",
            {**ran(*MAIN, "v5nw-s0", "v5nw-s1", "v5nw-s2"), **ROOM, "v5nw.word": "quiet\n"},
            "job j5-s0",
        ),
        # J5' s needs v5 seed s's ft row: a seed with none leaves its J5' out
        (
            "waive",
            {
                **ran("v5-s0", "v5-s2", "v5-s3", "v5-s4"),
                **ran("v5-s1", ft=False),
                **NO_ROOM,
                **ran("j5-s0"),
            },
            "job j5-s2",
        ),
        (
            "waive",
            {
                **ran("v5-s0", "v5-s2", "v5-s3", "v5-s4", "j5-s0", "j5-s2"),
                **ran("v5-s1", ft=False),
                **NO_ROOM,
            },
            "none",
        ),
        # a seed waiting for its rows keeps the lane waiting; nothing left is none
        (
            "waive",
            {**ran("v5-s0", "v5-s1", "v5-s3", "v5-s4"), **running("v5-s2", "j5-s0", "j5-s1")},
            "wait",
        ),
        ("waive", {**ran(*MAIN, "j5-s0", "j5-s1", "j5-s2"), **NO_ROOM}, "none"),
    ],
)
def test_v5_next_job(tmp_path: Path, r9: str, files: dict[str, str], want: str) -> None:
    assert next_job(tmp_path, files, r9=r9) == want


#: The tick the human writes V5_CONTINUE in the R9-pause branches.
CONTINUE_AT = 5
#: Every job holds its lane for two ticks (equal run lengths); a seed's train+score is its first.
JOB_TICKS = 2


class Sim:
    """A discrete-event model of the two lanes with equal run lengths.

    The model: every job holds its lane for JOB_TICKS ticks; v5 seed 0's R9 word is written
    after its first tick (train+score), as the job writes it before its needle control; a job's
    rows are read once it is done. At each tick: jobs end, R9 speaks, the human's continue
    arrives, then the free lanes pick, lane 0 first (the tie-break); a lane making a pick first
    makes every decision whose inputs are done. Every decision word is the branch's. ``pick`` is
    either the scripts' v5_next_job (through bash, on the marker files this class mirrors) or
    the Python oracle below."""

    def __init__(self, tmp: Path, r9: str, room: str) -> None:
        self.tmp = tmp
        self.q = tmp / "q"
        self.q.mkdir(parents=True, exist_ok=True)
        self.r9, self.room = r9, room
        self.claimed: dict[str, int] = {}
        self.claim_tick: dict[str, int] = {}
        self.done: set[str] = set()
        self.words: dict[str, str] = {}
        self.cont = False
        self.lanes: dict[int, tuple[str, int] | None] = {0: None, 1: None}
        self.finished = {0: False, 1: False}
        self.seq: dict[int, list[str]] = {0: [], 1: []}
        self.decided: list[tuple[int, str]] = []
        self.version = 0

    @property
    def r9_pin(self) -> str:
        return "waive" if self.r9 == "waive" else "keep"

    def put(self, name: str, text: str) -> None:
        (self.q / name).write_text(text + "\n")
        self.version += 1

    def word(self, name: str, text: str) -> None:
        self.words[name] = text
        self.put(name, text)

    def decide(self, d: str, t: int) -> None:
        assert d not in [x for _, x in self.decided], f"{d} decided twice"
        self.decided.append((t, d))
        if d == "room":
            self.word("v5nw.room", self.room)
            self.word("v5nw.launch", "run" if self.room == "room" else "skip")
        elif d == "arm":
            self.word("v5nw.word", "quiet")
        else:
            raise AssertionError(f"unknown decision {d!r}")

    def run(self, pick) -> Sim:
        seen = {0: -1, 1: -1}
        for t in range(60):
            for lane, r in self.lanes.items():
                if r and r[1] + JOB_TICKS == t:
                    self.done.add(r[0])
                    self.put(f"v5job-{r[0]}.done", f"ft {FT[r[0]]}")
                    self.lanes[lane] = None
            for r in self.lanes.values():
                if r and r[0] == "v5-s0" and r[1] + 1 == t and self.r9 != "waive":
                    w = "continue" if self.r9 == "continue" else "pause"
                    if w != "continue":
                        self.put("v5.paused", w)
                    self.word("v5r9.word", w)
            if self.r9 == "pause" and t == CONTINUE_AT:
                self.cont = True
                self.put("V5_CONTINUE", "Bharath: continue v5")
            for lane in (0, 1):
                if self.lanes[lane] or self.finished[lane] or seen[lane] == self.version:
                    continue
                while True:
                    act = pick(self, lane)
                    if act.startswith("decide "):
                        self.decide(act.split()[1], t)
                        continue
                    if act.startswith("job "):
                        job = act.split()[1]
                        assert job not in self.claimed, f"{job} picked twice"
                        self.claimed[job] = lane
                        self.claim_tick[job] = t
                        self.put(f"v5job-{job}.claimed", f"lane {lane}")
                        self.lanes[lane] = (job, t)
                        self.seq[lane].append(job)
                    elif act == "none":
                        self.finished[lane] = True
                    else:
                        assert act in ("wait", "hold"), act
                        seen[lane] = self.version
                    break
            if all(self.finished.values()) and not any(self.lanes.values()):
                return self
        raise AssertionError(f"the lanes did not finish: {self.seq}")


def bash_pick(sim: Sim, lane: int) -> str:
    r = bash(lib(sim.tmp, f"V5_R9={sim.r9_pin}") + 'echo "NEXT=$(v5_next_job)"')
    got = re.findall(r"^NEXT=(.*)$", r.stdout, re.M)
    assert got and got[-1], r.stdout + r.stderr
    return got[-1]


def oracle_input(sim: Sim, job: str) -> str:
    """The spec's inputs (section 5), read from the model's state: ready, possible, impossible."""
    if job == "v5-s0":
        return "ready"
    if job in ("v5-s1", "v5-s2", "v5-s3", "v5-s4"):
        if sim.r9 == "waive":
            return "ready"
        w = sim.words.get("v5r9.word")
        if w is None:
            return "possible"
        return "ready" if w == "continue" or sim.cont else "possible"
    if job.startswith("v5nw-"):
        w = sim.words.get("v5nw.launch")
        return "possible" if w is None else ("ready" if w == "run" else "impossible")
    return "ready" if f"v5-s{job[-1]}" in sim.done else "possible"


def oracle_pick(sim: Sim, lane: int) -> str:
    """The greedy rule: decisions first, then the earliest job in the human's order whose
    inputs are ready; an R9 hold holds every lane."""
    if sim.r9 != "waive" and sim.words.get("v5r9.word", "continue") != "continue" and not sim.cont:
        return "hold"
    if {"v5-s0", "v5-s1", "v5-s2"} <= sim.done and "v5nw.room" not in sim.words:
        return "decide room"
    arm = {"v5nw-s0", "v5nw-s1", "v5nw-s2"}
    if sim.words.get("v5nw.launch") == "run" and arm <= sim.done and "v5nw.word" not in sim.words:
        return "decide arm"
    waiting = False
    for job in JOBS:
        if job in sim.claimed:
            continue
        state = oracle_input(sim, job)
        if state == "ready":
            return f"job {job}"
        waiting = waiting or state == "possible"
    return "wait" if waiting else "none"


def test_the_oracle_is_the_leads_derivation_for_r9_waived_and_room(tmp_path: Path) -> None:
    """The lead's derivation (build/v5-h100/queue-lane-spec.md REVISION item 2; the amended
    DRAFT's projected_gpu_hours.slot), R9 waived, room: [s0|s1] [s2|s3] [s4|nw0] [nw1|nw2]
    [J5'0|J5'1] [J5'2|idle], six rounds."""
    sim = Sim(tmp_path, "waive", "room").run(oracle_pick)
    assert sim.seq == {
        0: ["v5-s0", "v5-s2", "v5-s4", "v5nw-s1", "j5-s0", "j5-s2"],
        1: ["v5-s1", "v5-s3", "v5nw-s0", "v5nw-s2", "j5-s1"],
    }
    assert "slot" in renamed_prereg()["launch"]["projected_gpu_hours"]
    assert (
        "[s0|s1] [s2|s3] [s4|nw0] [nw1|nw2] [J5'0|J5'1] [J5'2|idle]"
        in renamed_prereg()["launch"]["projected_gpu_hours"]["slot"]
    )


def test_with_no_room_j5_takes_the_arms_slots(tmp_path: Path) -> None:
    sim = Sim(tmp_path, "waive", "no_room").run(oracle_pick)
    assert sim.seq == {
        0: ["v5-s0", "v5-s2", "v5-s4", "j5-s1"],
        1: ["v5-s1", "v5-s3", "j5-s0", "j5-s2"],
    }


BRANCHES = [(r9, room) for r9 in ("waive", "continue", "pause") for room in ("room", "no_room")]


@pytest.mark.parametrize(("r9", "room"), BRANCHES)
def test_every_branch_yields_the_greedy_sequence_per_lane(
    tmp_path: Path, r9: str, room: str
) -> None:
    """R9 waived / kept (continue, or pause until the human's continue) x room / no_room: the
    scripts' v5_next_job, driven through the model, gives each lane the same sequence and
    makes the same decisions at the same ticks as the greedy oracle."""
    want = Sim(tmp_path / "oracle", r9, room).run(oracle_pick)
    got = Sim(tmp_path / "bash", r9, room).run(bash_pick)
    assert got.seq == want.seq
    assert got.decided == want.decided
    jobs = sorted(got.seq[0] + got.seq[1])
    expected = ["v5-s0", "v5-s1", "v5-s2", "v5-s3", "v5-s4", "j5-s0", "j5-s1", "j5-s2"]
    expected += ["v5nw-s0", "v5nw-s1", "v5nw-s2"] if room == "room" else []
    assert jobs == sorted(expected)
    assert [d for _, d in got.decided] == ["room", *(["arm"] if room == "room" else [])]
    if r9 != "waive":
        # R9 kept: seed 0 runs alone until R9 speaks
        assert got.seq[0][0] == "v5-s0" and got.claim_tick[got.seq[1][0]] >= 1


def test_the_hold_holds_j5_s0_until_the_humans_continue(tmp_path: Path) -> None:
    """Under R9's hold nothing starts, J5' s0 included, although its seed is done at tick 2:
    the block waits for the human, as the GH200's chain did (every later waiter waited on
    v5.done, which the hold delayed)."""
    sim = Sim(tmp_path, "pause", "no_room").run(bash_pick)
    assert sim.claim_tick["v5-s0"] == 0
    later = {j: t for j, t in sim.claim_tick.items() if j != "v5-s0"}
    assert "j5-s0" in later and min(later.values()) == CONTINUE_AT, sim.claim_tick


# --- dry runs of the lanes --------------------------------------------------------------------


FAKE_PYTHON = r"""#!{real}
import fcntl, json, os, sys, time, uuid
REAL = {real!r}
ROOT = {root!r}
args = sys.argv[1:]
if args[:1] in (["-c"], ["-"]):
    os.execv(REAL, [REAL] + args)
T0 = time.time()
scen = json.load(open(os.path.join(ROOT, "scenario.json")))


def opt(name):
    return args[args.index(name) + 1] if name in args else None


def fd9():
    # the file this process's fd 9 names, and whether that file is flock-held right now
    try:
        if hasattr(fcntl, "F_GETPATH"):
            path = fcntl.fcntl(9, fcntl.F_GETPATH, bytes(1024)).split(b"\0", 1)[0].decode()
        else:
            path = os.readlink("/proc/self/fd/9")
    except OSError:
        return None, None
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return os.path.basename(path), True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return os.path.basename(path), False
    finally:
        os.close(fd)


NAME, LOCKED = fd9()


def record():
    rec = {{"argv": args, "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"), "fd9": NAME,
            "locked": LOCKED, "alloc": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "t0": T0, "t1": time.time()}}
    with open(os.path.join(ROOT, "argv.log"), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")


def append_row(path, row):
    # as qd_train/ledger.py appends: under flock, never into another writer's unfinished line
    end = time.monotonic() + 10
    while True:
        with open(path, "ab+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0, 2)
            size = f.tell()
            last = b""
            if size:
                f.seek(size - 1)
                last = f.read(1)
            if size == 0 or last == b"\n":
                f.seek(0, 2)
                f.write((json.dumps(row) + "\n").encode())
                return
        if time.monotonic() > end:
            raise SystemExit(f"{{path}} stays half-written")
        time.sleep(0.05)


rc = 0
if "tools/real_ft_run.py" in args and "--probe-shapes" in args:
    # probe_rc: the exit code of each attempt in turn (the last repeats)
    count = os.path.join(ROOT, "probe.count")
    n = int(open(count).read()) if os.path.exists(count) else 0
    open(count, "w").write(str(n + 1))
    codes = scen.get("probe_rc", [0])
    rc = codes[min(n, len(codes) - 1)]
    row = str(uuid.uuid4())
    append_row(opt("--ledger"), {{"row_id": row, "quick": True,
                                  "metrics": {{"memory_probe": {{"state": "ran"}}}}}})
    print(f"ft row {{row}}")
    if rc:
        print("memory probe: max_memory_reserved 70.20 GiB is over 67.15 GiB "
              "(79.15 GiB less the 12 GiB margin): FAILED")
elif "tools/real_ft_run.py" in args and "--score-checkpoint" not in args:
    seed = opt("--seeds")
    if scen.get("tamper_rules_on_train"):
        with open(os.path.join(ROOT, "bin", "qd-post-f-rules-v5"), "a") as f:
            f.write("\n# swapped after the waiter checked it\n")
    time.sleep(scen.get("train_sleep", 0))
    if scen.get("train_fails"):
        rc = 1
    else:
        if opt("--verdicts-out"):
            open(opt("--verdicts-out"), "w").write('{{"verdict": "stub"}}\n')
        row = str(uuid.uuid4())
        digest = scen.get("digests", {{}}).get(seed)
        append_row(opt("--ledger"), {{"row_id": row, "metrics": {{"corpus.plan_order_digest":
                   {{"state": "ran", "value": digest}}}}}})
        ck = opt("--checkpoint-dir")
        if ck:
            open(os.path.join(ck, f"epoch-seed{{seed}}-cuda.json"), "w").write("{{}}")
            if opt("--retain-tower-every"):
                for s in scen.get("steps", [1000, 2000, 2470]):
                    snap = os.path.join(ck, f"epoch-seed{{seed}}-cuda-step{{s}}.json")
                    open(snap, "w").write("{{}}")
        print(f"ft row {{row}}")
record()
sys.exit(rc)
"""

FAKE_RULES = r"""#!{real}
import fcntl, json, os, sys
ROOT = {root!r}
args = sys.argv[1:]
with open(os.path.join(ROOT, "rules.log"), "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(json.dumps(args) + "\n")
scen = json.load(open(os.path.join(ROOT, "scenario.json")))
sub = args[0]
key = sub + (" --room" if "--room" in args else "")
out = args[args.index("--out") + 1] if "--out" in args else None
race = scen.get("race", {{}}).get(key)
if race:
    # a writer mid-append: the first n calls meet half a row as the ledger's last line and
    # refuse as the real binary does; the next call finds the row completed. Two lanes may make
    # a look-up at once, so the count is read and bumped under a lock: n refusals in all.
    race_lock = open(os.path.join(ROOT, "race.lock"), "a")
    fcntl.flock(race_lock, fcntl.LOCK_EX)
    flag, n = race
    led = args[args.index(flag) + 1]
    count = os.path.join(ROOT, "race-" + key.replace(" ", "") + ".count")
    seen = int(open(count).read()) if os.path.exists(count) else 0
    # the ledger is read and written under its flock, as every fake writer appends, so the
    # line this refusal names is the ledger's last line when the refusal is made
    with open(led, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        text = f.read()
        if seen < n:
            if not text or text.endswith("\n"):
                f.write('{{"row_id": "half')
                text += '{{"row_id": "half'
        elif text and not text.endswith("\n"):
            f.write('-row"}}\n')
    if seen < n:
        open(count, "w").write(str(seen + 1))
        reason = (f"{{led}} line {{text.count(chr(10)) + 1}}: malformed ledger line "
                  "(EOF while parsing a string at line 1 column 16)")
        if out:
            open(out, "w").write(json.dumps({{"decision": "refused", "refused": reason}}))
        sys.stderr.write(json.dumps({{"refused": reason}}) + "\n")
        sys.stderr.write("qd-post-f-rules: refused: " + reason + "\n")
        print("refused")
        sys.exit(3)
if sub == "ft-rows":
    sys.exit(scen.get("ft-rows", 0))
if sub == "eval-row":
    print(scen.get("eval-row", "{eval_row}"))
    sys.exit(0)
word, rc = scen["rules"][key]
open(out, "w").write(json.dumps({{"stub": key}}))
print(word)
sys.exit(rc)
"""

#: Every reading's answer when a test does not set its own: R9 continues, no room for the arm;
#: J5''s look-ups pass. A full block is then v5 seeds 0-4 and J5' x3. seeds34 answers refused,
#: should anything ask it (nothing may: test_seeds_3_4_run_unconditionally_and_seeds34_is_
#: never_read).
DEFAULT_RULES = {
    "v5-pause": ["continue", 0],
    "seeds34": ["refused", 3],
    "v5-noulw --room": ["no_room", 0],
    "v5-noulw": ["quiet", 0],
}


class Box:
    """A /home/ubuntu stand-in with the v5 lanes' pins filled for the 2x H100 box."""

    def __init__(
        self,
        tmp: Path,
        rules: dict | None = None,
        *,
        scenario: dict | None = None,
        fill_pins: bool = True,
        keep_unset: tuple[str, ...] = (),
        pins: dict[str, str] | None = None,
        consts: dict[str, str] | None = None,
    ) -> None:
        self.root = root = tmp / "box"
        self.q = root / "queue"
        for d in ("queue", "ledger", "logs", "bin", "post-f", "v5data/shards/train", "fakebin"):
            (root / d).mkdir(parents=True, exist_ok=True)
        self.scenario = {"rules": {**DEFAULT_RULES, **(rules or {})}, "ft-rows": 0}
        self.scenario.update(scenario or {})
        real = sys.executable
        write_exec(
            root / "qd-venv" / "bin" / "python", FAKE_PYTHON.format(real=real, root=str(root))
        )
        rules_bin = write_exec(
            root / "bin" / "qd-post-f-rules-v5",
            FAKE_RULES.format(real=real, root=str(root), eval_row=EVAL_ROW),
        )
        prep = write_exec(root / "bin" / "qd-prep-v5", "#!/bin/sh\nexit 0\n")
        for name, body in {
            "timeout": '#!/bin/sh\nshift\nexec "$@"\n',
            "sleep": "#!/bin/sh\nexec /bin/sleep 0.01\n",
            "setsid": '#!/bin/sh\nexec "$@"\n',
            "nohup": '#!/bin/sh\nexec "$@"\n',
            "git": (
                '#!/bin/sh\ncase "$*" in\n'
                f'  "rev-parse HEAD") echo {LANE_AT} ;;\n'
                f'  "rev-parse --short HEAD") echo {LANE_AT[:7]} ;;\n'
                "esac\nexit 0\n"
            ),
        }.items():
            write_exec(root / "fakebin" / name, body)
        real_flock(root / "fakebin")
        lane = root / "qd-lane-v5"
        (lane / "tools").mkdir(parents=True)
        (lane / "campaign").mkdir()
        (lane / "tools" / "real_ft_run.py").write_text(
            "raise SystemExit('the fake python runs instead')\n"
        )
        (lane / "tools" / "ft_linear_control.py").write_text(
            "raise SystemExit('the fake python runs instead')\n"
        )
        (lane / "tools" / "ckpt_average.py").write_text(
            'TRAJECTORY_NOT_SCORABLE = "stub: --score-checkpoint refuses a trajectory average"\n'
        )
        prereg = lane / "campaign" / "v5-preregistered.json"
        prereg.write_text(json.dumps(renamed_prereg()), encoding="utf-8")
        shutil.copy(NOUL, lane / "campaign" / NOUL.name)
        # the scripts, with /home/ubuntu moved to the temp root
        for src in QUEUE.glob("*.sh"):
            text = src.read_text(encoding="utf-8").replace("/home/ubuntu", str(root))
            (root / "post-f" / src.name).write_text(text, encoding="utf-8")
        common = (root / "post-f" / "post_f_common.sh").read_text(encoding="utf-8")
        backbone = re.search(r"^BACKBONE=(.*)$", common, re.M)[1]
        tok = Path(backbone) / "tokenizer.json"
        tok.parent.mkdir(parents=True)
        tok.write_text('{"model": "stub"}')
        old_tok = re.search(r"^TOKENIZER_SHA256=(.*)$", common, re.M)[1]
        (root / "post-f" / "post_f_common.sh").write_text(common.replace(old_tok, sha256(tok)))
        header = {"shard_hash": "a" * 64, "data_snapshot_hash": "b" * 64}
        (root / "v5data" / "shards" / "train" / "header.json").write_text(json.dumps(header))
        (root / "v5data" / "exclusions.txt").write_text("k1\n")
        self.split = (
            f"--out {root}/v5data --no-repo-history --rev {BUILD_REV} "
            "--defect-class data/pool/commitpackft-composed-v2 "
            '--defect-noul data/pool/defect-noul-v3c --general-record "$REC" '
            f"--general-max-rows 200000 --exclude-identity-keys {root}/v5data/exclusions.txt "
            '--real-backbone "$BACKBONE"'
        )
        decisions = {
            "V5_C1": "off",
            "V5_C2A": "off",
            "V5_C2B": "off",
            "V5_LOWER": "keep",
            "V5_LRSET": "f",
            "V5_R9": "keep",
        }
        decisions.update(pins or {})
        recipe = list(BASE_RECIPE)
        rec = prelude_record(
            sha256(tok), header, recipe, c1=20260919 if decisions["V5_C1"] == "on" else None
        )
        if decisions["V5_C1"] == "on":
            rec["argv"] += ["--option-permutation-seed", "20260919"]
        record = root / "post-f" / "v5-prelude-record.json"
        record.write_text(json.dumps(rec), encoding="utf-8")
        self.digests = rec["plan_order_digest"]
        self.scenario["digests"] = self.digests
        self.write_scenario()
        if fill_pins:
            fills = {
                "RULES_V5_SHA256": sha256(rules_bin),
                "V5_PREP_BIN_SHA256": sha256(prep),
                "V5_LANE_AT": LANE_AT,
                "V5_PREREG_SHA256": sha256(prereg),
                "V5_DATE": "2026-10-04",
                "V5_PRELUDE_SHA256": sha256(record),
                "V5_BOX": BOX,
                "V5_INSTANCE": INSTANCE,
                "V5_USD_PER_INSTANCE_HOUR": USD_INSTANCE,
                "V5_USD_PER_GPU_HOUR": USD_GPU,
                "V5_HUMAN_YES": shlex.quote(HUMAN_YES),
                **decisions,
            }
            path = root / "post-f" / "v5_common.sh"
            text = path.read_text(encoding="utf-8")
            for name, value in fills.items():
                if name in keep_unset:
                    continue
                text, n = re.subn(rf"^{name}=UNSET$", f"{name}={value}", text, flags=re.M)
                assert n == 1, name
            text, n = re.subn(r"^V5_SPLIT=\(UNSET\)$", f"V5_SPLIT=({self.split})", text, flags=re.M)
            assert n == 1
            for name, value in (consts or {}).items():
                text, n = re.subn(rf"^{name}=[0-9]+$", f"{name}={value}", text, flags=re.M)
                assert n == 1, name
            path.write_text(text, encoding="utf-8")
        self.ledger = root / "ledger" / f"{BOX}-v5-2026-10-04.jsonl"
        self.arm_ledger = root / "ledger" / f"{BOX}-v5-noulw-2026-10-04.jsonl"
        self.probe_ledger = root / "ledger" / f"{BOX}-v5-probe-2026-10-04.jsonl"

    def write_scenario(self) -> None:
        (self.root / "scenario.json").write_text(json.dumps(self.scenario))

    @property
    def env(self) -> dict[str, str]:
        return {"PATH": f"{self.root}/fakebin:{PATH}", "HOME": str(self.root), "LC_ALL": "C"}

    def popen(self, waiter: str, *args: str) -> subprocess.Popen:
        return subprocess.Popen(
            [BASH, str(self.root / "post-f" / waiter), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=self.env,
            stdin=subprocess.DEVNULL,
        )

    def start_lanes(self, *, launch: bool = True) -> list[subprocess.Popen]:
        """Both lanes, then the human's launch words once each lane is queued (or ended): the
        deploy order, so no lane's stale-marker check can race the other's first write."""
        ps = [self.popen("box_q_v5.sh", str(n)) for n in (0, 1)]
        end = time.monotonic() + 60
        for n in (0, 1):
            while not (
                (self.q / f"v5lane{n}.queued").exists() or (self.q / f"v5lane{n}.done").exists()
            ):
                if time.monotonic() > end or ps[n].poll() is not None:
                    break
                time.sleep(0.02)
        if launch:
            (self.q / "V5_LAUNCH_YES").write_text("Bharath: yes, launch v5 (answer 1 at d24c865)\n")
        return ps

    @staticmethod
    def finish(ps: list[subprocess.Popen], timeout: float = 180) -> list[tuple[int, str]]:
        try:
            outs = [p.communicate(timeout=timeout)[0] for p in ps]
        finally:
            for p in ps:
                if p.poll() is None:
                    p.kill()
        return [(p.returncode, o) for p, o in zip(ps, outs, strict=True)]

    def run_lanes(self, *, launch: bool = True, timeout: float = 180) -> list[tuple[int, str]]:
        return self.finish(self.start_lanes(launch=launch), timeout)

    def records(self, tool: str = "tools/real_ft_run.py") -> list[dict]:
        log = self.root / "argv.log"
        if not log.exists():
            return []
        return [r for r in map(json.loads, log.read_text().splitlines()) if tool in r["argv"]]

    def calls(self, tool: str = "tools/real_ft_run.py") -> list[list[str]]:
        return [r["argv"] for r in self.records(tool)]

    def training(self) -> list[list[str]]:
        return [
            a for a in self.calls() if "--score-checkpoint" not in a and "--probe-shapes" not in a
        ]

    def v5_training(self) -> list[list[str]]:
        return [
            a
            for a in self.training()
            if opt(a, "--ledger") == str(self.ledger) and "--shuffled-label" not in a
        ]

    def j5_training(self) -> list[list[str]]:
        return [a for a in self.training() if "--shuffled-label" in a]

    def arm_training(self) -> list[list[str]]:
        return [a for a in self.training() if opt(a, "--ledger") == str(self.arm_ledger)]

    def rules(self) -> list[list[str]]:
        log = self.root / "rules.log"
        return [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []

    def wait_for(self, *names: str, timeout: float = 60) -> None:
        end = time.monotonic() + timeout
        for n in names:
            while not (self.q / n).exists():
                assert time.monotonic() < end, f"{n} never appeared"
                time.sleep(0.05)


def opt(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def seeds_of(calls: list[list[str]]) -> list[str]:
    return sorted(opt(a, "--seeds") for a in calls)


def both(results: list[tuple[int, str]]) -> str:
    return "\n".join(o for _, o in results)


def assert_cost(argv: list[str]) -> None:
    """COST is the pinned box's, priced per GPU as the tool prices one visible GPU."""
    assert re.fullmatch(rf"{re.escape(INSTANCE)}:gpu[01]", opt(argv, "--instance") or ""), argv
    assert opt(argv, "--usd-per-hour") == USD_GPU and opt(argv, "--usd-per-gpu-hour") == USD_GPU


def assert_training_form(argv: list[str], seed: int, *, extra: list[tuple[str, str]] = ()) -> None:
    """Changed for the 2x H100 box: the rate is $4.19/GPU-h in both cost flags (was --usd-per-
    hour 2.29, COST's GH200 rate), and --approved-by carries the filled V5_HUMAN_YES pin and the
    human's box words (was the d24c865 yes, on ~$137 / ~60 GPU-h, which does not cover 11 runs)."""
    assert opt(argv, "--seeds") == str(seed)
    assert recipe_pairs(argv) == sorted([*recipe_pairs(BASE_RECIPE), *extra], key=repr)
    assert opt(argv, "--batch-order") == "seed"
    assert opt(argv, "--wall-clock-cap-s") == "32400"
    said = opt(argv, "--approved-by") or ""
    assert said.startswith(HUMAN_YES) and BOX_WORDS in said and "d24c865" not in said
    assert f"${USD_GPU}/GPU-h = $37.71" in said
    assert_cost(argv)
    assert not any(re.search(r"heldout|held-out|containment", a, re.I) for a in argv), argv


def assert_on_its_lane(box: Box) -> None:
    """Every GPU step (probe, train, needle control, trajectory scoring) ran with
    CUDA_VISIBLE_DEVICES its lane, holding that lane's lock on fd 9, and a job ran on the lane
    that claimed it; every CPU control ran with fd 9 closed (outside every lock)."""
    for r in box.records():
        lane = r["cvd"]
        assert lane in ("0", "1"), r
        assert r["fd9"] == f"gpu{lane}.lock" and r["locked"] is True, r
        assert opt(r["argv"], "--instance") == f"{INSTANCE}:gpu{lane}", r
    for r in box.records("tools/ft_linear_control.py"):
        assert r["fd9"] is None, r
    for argv, r in ((r["argv"], r) for r in box.records()):
        if "--score-checkpoint" in argv or "--probe-shapes" in argv:
            continue
        seed = opt(argv, "--seeds")
        job = (
            f"j5-s{seed}"
            if "--shuffled-label" in argv
            else f"v5nw-s{seed}"
            if opt(argv, "--ledger") == str(box.arm_ledger)
            else f"v5-s{seed}"
        )
        claimed = (box.q / f"v5job-{job}.claimed").read_text()
        assert claimed.startswith(f"lane {r['cvd']} "), (job, claimed, r["cvd"])


def prelude_record(tok: str, header: dict, recipe: list[str], *, c1: int | None = None) -> dict:
    return {
        "ok": True,
        "tool": "campaign/post-f-queue/v5_prelude_mac.py",
        "code_commit": LANE_AT,
        "batch_order": "seed",
        "tokenizer_json_sha256": tok,
        "shape_equal_across_seeds": True,
        "option_permutation_seed": c1,
        "shard_hash": header["shard_hash"],
        "data_snapshot_hash": header["data_snapshot_hash"],
        "plan_order_digest": {str(s): f"{s + 1:x}" * 64 for s in range(5)},
        "argv": ["--out", "/Users/x/v5", *recipe, "--seeds", "0"],
    }


HEADER = {"shard_hash": "a" * 64, "data_snapshot_hash": "b" * 64}


@pytest.mark.parametrize(
    ("mutate", "ok"),
    [
        (lambda r: None, True),
        (lambda r: r.__setitem__("ok", False), False),
        (lambda r: r.__setitem__("code_commit", "f" * 40), False),
        (lambda r: r.__setitem__("batch_order", None), False),
        (lambda r: r.__setitem__("tokenizer_json_sha256", "0" * 64), False),
        (lambda r: r.__setitem__("shape_equal_across_seeds", False), False),
        (lambda r: r.__setitem__("shard_hash", "c" * 64), False),
        (lambda r: r.__setitem__("option_permutation_seed", 20260919), False),
        (lambda r: r["plan_order_digest"].pop("4"), False),
        (lambda r: r["plan_order_digest"].__setitem__("1", r["plan_order_digest"]["0"]), False),
        (
            lambda r: r.__setitem__(
                "argv", [a for a in r["argv"] if a not in ("--batch-order", "seed")]
            ),
            False,
        ),
        (lambda r: r["argv"].extend(["--noul-weight", "4"]), False),
    ],
)
def test_v5_prelude_check(tmp_path: Path, mutate, ok: bool) -> None:
    rec = prelude_record("t" * 64, HEADER, BASE_RECIPE)
    mutate(rec)
    data = tmp_path / "data"
    (data / "shards" / "train").mkdir(parents=True)
    (data / "shards" / "train" / "header.json").write_text(json.dumps(HEADER))
    (tmp_path / "rec.json").write_text(json.dumps(rec))
    r = bash(
        lib(
            tmp_path,
            f'PY="{sys.executable}"; V5_PRELUDE="{tmp_path}/rec.json"; V5_DATA="{data}"; '
            f"V5_LANE_AT={LANE_AT}; TOKENIZER_SHA256={'t' * 64}; V5_C1=off; "
            "V5_C2A=off; V5_C2B=off; V5_LOWER=keep; V5_LRSET=f",
        )
        + 'v5_recipe; v5_prelude_check; echo "RC=$?"'
    )
    # the check must have run (a missing function prints no RC at all) and decided
    assert re.findall(r"^RC=(\d+)$", r.stdout, re.M) == ["0" if ok else "1"], r.stdout + r.stderr
    assert "prelude record: digests" in r.stdout


def test_v5_waits_for_the_launch_words(tmp_path: Path) -> None:
    """Changed for the 2x H100 box: the GH200's chain gates (j5pp.done, j6a, j6g) are gone; the
    human's launch words are the one gate before verification, as the spec rules."""
    box = Box(tmp_path)
    ps = box.start_lanes(launch=False)
    try:
        (box.q / "V5_LAUNCH_YES").write_text("")  # empty is not a yes
        time.sleep(1.5)
        assert box.calls() == [] and not list(box.q.glob("v5lane*.started"))
        (box.q / "V5_LAUNCH_YES").write_text("Bharath: launch v5\n")
        results = box.finish(ps)
    finally:
        for p in ps:
            if p.poll() is None:
                p.kill()
    assert [rc for rc, _ in results] == [0, 0], both(results)
    out = both(results)
    assert "the human's v5 launch yes: " in out and "Bharath: launch v5" in out
    assert seeds_of(box.v5_training()) == ["0", "1", "2", "3", "4"]
    box.wait_for("v5traj-s4.done")


def test_nothing_waits_on_j5pp_j6a_or_j6g(tmp_path: Path) -> None:
    """The GH200's post-F chain is not on this box: with j6a and j6g queued and never done, and
    no j5pp.done, the lanes run the whole block."""
    box = Box(tmp_path)
    for n in ("j6a.queued", "j6g.queued", "j5pp.queued"):
        (box.q / n).write_text("")
    results = box.run_lanes()
    assert [rc for rc, _ in results] == [0, 0], both(results)
    assert seeds_of(box.v5_training()) == ["0", "1", "2", "3", "4"]
    assert seeds_of(box.j5_training()) == ["0", "1", "2"]


def test_v5_stop_ends_the_wait_for_the_launch_words(tmp_path: Path) -> None:
    box = Box(tmp_path)
    ps = box.start_lanes(launch=False)
    try:
        time.sleep(1.0)
        (box.q / "V5_STOP").write_text("Bharath: not now\n")
        results = box.finish(ps, 60)
    finally:
        for p in ps:
            if p.poll() is None:
                p.kill()
    assert [rc for rc, _ in results] == [3, 3]
    assert both(results).count("v5 NOT RUN") >= 2 and box.calls() == []


def test_the_probe_runs_first_on_gpu0_with_probe_shapes_into_its_own_ledger(tmp_path: Path) -> None:
    box = Box(tmp_path)
    results = box.run_lanes()
    assert [rc for rc, _ in results] == [0, 0], both(results)
    recs = box.records()
    probes = [r for r in recs if "--probe-shapes" in r["argv"]]
    assert len(probes) == 1, "the probe runs once"
    probe = probes[0]
    argv = probe["argv"]
    assert probe["cvd"] == "0" and probe["fd9"] == "gpu0.lock" and probe["locked"] is True
    assert probe["alloc"] is None, "the allocator setting is the retry's only"
    assert opt(argv, "--probe-shapes") == PROBE_MARGIN and opt(argv, "--seeds") == "0"
    assert opt(argv, "--ledger") == str(box.probe_ledger)
    assert opt(argv, "--wall-clock-cap-s") == "1800"
    said = opt(argv, "--approved-by") or ""
    assert said.startswith(HUMAN_YES) and f"1800 s at ${USD_GPU}/GPU-h = $2.10" in said
    assert recipe_pairs(argv) == recipe_pairs(BASE_RECIPE)  # v5's recipe: --epoch --no-memorise
    assert {"--epoch", "--no-memorise"} <= set(argv)
    for refused in (
        "--checkpoint-dir",
        "--retain-tower-every",
        "--score-val",
        "--needle",
        "--needle-control",
        "--ood",
        "--noul-weight",
        "--max-steps",
    ):
        assert refused not in argv, refused
    assert_cost(argv)
    others = [r for r in recs if r is not probe]
    assert others and all(r["t0"] > probe["t1"] for r in others), (
        "a GPU step started before the probe ended"
    )
    # its row is in its own ledger; v5's ledger never sees it
    assert box.probe_ledger.read_text().count("memory_probe") == 1
    assert "memory_probe" not in box.ledger.read_text()
    assert (box.q / "v5.probe-passed").read_text().startswith("attempt 1")
    assert not (box.q / "v5.probe-failed").exists()
    assert_on_its_lane(box)


def test_a_failing_probe_is_retried_once_with_expandable_segments(tmp_path: Path) -> None:
    """hardware.probe.fallback: on a fail, the probe again with PYTORCH_CUDA_ALLOC_CONF=
    expandable_segments:True at the same 12 GiB, for that call only; a pass there lets v5 go."""
    box = Box(tmp_path, scenario={"probe_rc": [1, 0]})
    results = box.run_lanes()
    assert [rc for rc, _ in results] == [0, 0], both(results)
    probes = [r for r in box.records() if "--probe-shapes" in r["argv"]]
    assert [p["alloc"] for p in probes] == [None, EXPANDABLE]
    assert probes[0]["argv"] == probes[1]["argv"], "the retry is the same call"
    others = [r for r in box.records() if "--probe-shapes" not in r["argv"]]
    assert others and all(r["alloc"] is None for r in others), "the setting leaked past the retry"
    assert all(r["t0"] > probes[1]["t1"] for r in others)
    assert (box.q / "v5.probe-passed").read_text().startswith("attempt 2")
    assert box.probe_ledger.read_text().count("memory_probe") == 2
    assert seeds_of(box.v5_training()) == ["0", "1", "2", "3", "4"]


def test_a_probe_that_fails_twice_stops_everything_until_the_humans_marker(tmp_path: Path) -> None:
    box = Box(tmp_path, scenario={"probe_rc": [1, 1]})
    ps = box.start_lanes()
    try:
        box.wait_for("v5.probe-failed")
        why = (box.q / "v5.probe-failed").read_text()
        assert "exit 1" in why and "FAILED" in why and "expandable_segments" in why, why
        time.sleep(1.5)
        assert box.training() == [], "a seed started after a failed probe"
        (box.q / "V5_PROBE_YES").write_text("")  # empty is not a yes
        time.sleep(0.5)
        assert box.training() == []
        (box.q / "V5_PROBE_YES").write_text("Bharath: the probe's margin is fine; go\n")
        results = box.finish(ps)
    finally:
        for p in ps:
            if p.poll() is None:
                p.kill()
    assert [rc for rc, _ in results] == [0, 0], both(results)
    out = both(results)
    assert "Bharath: the probe's margin is fine; go" in out
    assert len([a for a in box.calls() if "--probe-shapes" in a]) == 2
    assert seeds_of(box.v5_training()) == ["0", "1", "2", "3", "4"]


def test_a_failing_probe_and_v5_stop_end_both_lanes(tmp_path: Path) -> None:
    box = Box(tmp_path, scenario={"probe_rc": [2]})
    ps = box.start_lanes()
    try:
        box.wait_for("v5.probe-failed")
        time.sleep(0.5)
        (box.q / "V5_STOP").write_text("Bharath: stop\n")
        results = box.finish(ps, 60)
    finally:
        for p in ps:
            if p.poll() is None:
                p.kill()
    assert [rc for rc, _ in results] == [3, 3], both(results)
    assert box.training() == []
    assert "exit 2" in (box.q / "v5.probe-failed").read_text()


def test_every_gpu_step_runs_on_its_lanes_gpu_under_its_lanes_lock(tmp_path: Path) -> None:
    box = Box(tmp_path, {"v5-noulw --room": ["room", 0]})
    results = box.run_lanes()
    assert [rc for rc, _ in results] == [0, 0], both(results)
    for n in ("v5traj-s0", "v5traj-s4", "v5nwtraj-s2"):
        box.wait_for(f"{n}.done")
    assert seeds_of(box.v5_training()) == ["0", "1", "2", "3", "4"]
    assert seeds_of(box.arm_training()) == ["0", "1", "2"]
    assert seeds_of(box.j5_training()) == ["0", "1", "2"]
    assert {r["cvd"] for r in box.records()} == {"0", "1"}, "both GPUs worked"
    assert_on_its_lane(box)
    for job in JOBS:
        assert (box.q / f"v5job-{job}.done").exists(), job


def test_two_lanes_never_train_two_seeds_together_past_the_budget(tmp_path: Path) -> None:
    """Two lanes, room for one more seed only: seed 1 waits while seed 0 runs (exit 4 of
    v5_budget_ok, logged), and starts once seed 0 ended under its estimate; at no time do two
    seeds train together. 64 h spent: 64 + 7 fits the 74.0 GPU-h, 64 + 7 + 7 does not."""
    box = Box(
        tmp_path,
        pins={"V5_R9": "waive"},
        scenario={"train_sleep": 0.6, "ft-rows": 1},
    )
    (box.q / "v5.spend").write_text(f"2026-10-04T00:00:00Z\tearlier\t{64 * 3600}\t268.16\n")
    results = box.run_lanes(timeout=300)
    out = both(results)
    assert [rc for rc, _ in results] == [0, 0], out
    runs = sorted(
        (r["t0"], r["t1"])
        for r in box.records()
        if "--score-checkpoint" not in r["argv"]
        and "--probe-shapes" not in r["argv"]
        and opt(r["argv"], "--ledger") == str(box.ledger)
    )
    assert len(runs) == 5, runs
    assert all(a[1] <= b[0] for a, b in pairwise(runs)), f"two seeds trained together: {runs}"
    assert "waits" in out and "REFUSED" not in out


def test_the_trajectory_cap_leaves_steps_not_run_and_the_controls_still_run(tmp_path: Path) -> None:
    """Changed for the 2x H100 box: the waiter takes its parent's lane as a fourth argument."""
    box = Box(tmp_path, consts={"V5_TRAJ_CAP_S": "100"})
    ck = box.root / "ckpt" / "v5"
    ck.mkdir(parents=True)
    for s in (1000, 2000):
        (ck / f"epoch-seed0-cuda-step{s}.json").write_text("{}")
    (box.root / "v5").mkdir()
    (box.root / "v5" / "verdicts-s0.jsonl").write_text('{"verdict": "stub"}\n')
    p = subprocess.run(
        [BASH, str(box.root / "post-f" / "box_q_v5traj.sh"), "v5", "0", EVAL_ROW, "1"],
        capture_output=True,
        text=True,
        env=box.env,
        timeout=60,
        check=False,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert [a for a in box.calls() if "--score-checkpoint" in a] == []
    assert "the 100s cap left steps 2000 1000 NOT RUN" in p.stdout
    assert len(box.calls("tools/ft_linear_control.py")) == 2
    assert (box.q / "v5traj-s0.started").exists() and (box.q / "v5traj-s0.done").exists()
    assert (box.q / "gpu1.lock").exists() and not (box.q / "gpu0.lock").exists()


def test_the_trajectory_waiter_scores_on_its_lanes_gpu(tmp_path: Path) -> None:
    box = Box(tmp_path)
    ck = box.root / "ckpt" / "v5"
    ck.mkdir(parents=True)
    (ck / "epoch-seed2-cuda-step1000.json").write_text("{}")
    p = subprocess.run(
        [BASH, str(box.root / "post-f" / "box_q_v5traj.sh"), "v5", "2", EVAL_ROW, "1"],
        capture_output=True,
        text=True,
        env=box.env,
        timeout=60,
        check=False,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    (scored,) = box.records()
    assert scored["cvd"] == "1" and scored["fd9"] == "gpu1.lock" and scored["locked"] is True
    assert opt(scored["argv"], "--ledger") == str(box.ledger)
    assert_cost(scored["argv"])
    spent = (box.q / "v5job-v5-s2.spend").read_text().split()
    assert len(spent) == 1, "the trajectory's seconds are charged to its seed's job"


def test_v5_with_its_pins_unset_trains_nothing_and_touches_done(tmp_path: Path) -> None:
    box = Box(tmp_path, fill_pins=False)
    results = box.run_lanes()
    assert [rc for rc, _ in results] == [3, 3]
    assert all("deferred: pins UNSET:" in o for _, o in results)
    assert box.calls() == [] and box.rules() == []
    for n in (0, 1):
        assert (box.q / f"v5lane{n}.done").exists() and not (box.q / f"v5lane{n}.started").exists()


@pytest.mark.parametrize(
    "unset",
    [
        "V5_BOX",
        "V5_INSTANCE",
        "V5_USD_PER_INSTANCE_HOUR",
        "V5_USD_PER_GPU_HOUR",
        "V5_R9",
        "V5_HUMAN_YES",
    ],
)
def test_a_box_pin_left_unset_refuses_everything(tmp_path: Path, unset: str) -> None:
    """The GH200-safe guard: every other pin filled, one box pin the literal UNSET, and neither
    lane runs, probes, reads or locks anything. V5_HUMAN_YES among them: no row may cite a yes
    the human has not given for this plan."""
    box = Box(tmp_path, keep_unset=(unset,))
    results = box.run_lanes()
    assert [rc for rc, _ in results] == [3, 3]
    for _, out in results:
        assert "deferred: pins UNSET:" in out and unset in out.split("deferred: pins UNSET:")[1]
    assert box.calls() == [] and box.rules() == [] and not list(box.q.glob("gpu*.lock"))


def test_v5_continue_runs_its_seeds_in_v5s_form(tmp_path: Path) -> None:
    """Changed for the 2x H100 box: two lanes run the whole block (here v5 seeds 0-4, then J5'
    x3 with no room for the arm), so the seeds are a set, not one waiter's order; seeds 3-4
    are unconditional (the amendment a29bca1); the ledger is named by V5_BOX; the cost line is
    the per-GPU rate's and the amended DRAFT's."""
    box = Box(tmp_path)
    results = box.run_lanes()
    out = both(results)
    assert [rc for rc, _ in results] == [0, 0], out
    assert out.count("all done") >= 2
    box.wait_for(*(f"v5traj-s{s}.done" for s in range(5)))
    train = box.v5_training()
    assert seeds_of(train) == ["0", "1", "2", "3", "4"]
    for argv in train:
        assert_training_form(argv, int(opt(argv, "--seeds")))
        assert (
            opt(argv, "--retain-tower-every") == "1000"
            and opt(argv, "--checkpoint-every") == "100000"
        )
        assert {"--score-val", "--needle", "--ood"} <= set(argv)
        assert opt(argv, "--ledger") == str(box.ledger)
    assert box.ledger.name == f"{BOX}-v5-2026-10-04.jsonl"
    scored = [a for a in box.calls() if "--score-checkpoint" in a]
    needle = [a for a in scored if "--needle-control" in a]
    assert seeds_of(needle) == ["0", "1", "2", "3", "4"]
    for a in needle:
        assert (
            opt(a, "--needle-control") == "1024,2048,4096"
            and opt(a, "--wall-clock-cap-s") == "5400"
        )
        assert opt(a, "--score-dtype") == "fp32" and "--batch-order" not in a
        assert_cost(a)
    traj = [a for a in scored if "--needle-control" not in a]
    for s in range(5):
        steps = [
            opt(a, "--score-checkpoint").rsplit("-step", 1)[1]
            for a in traj
            if opt(a, "--seeds") == str(s)
        ]
        assert steps == ["2470.json", "1000.json", "2000.json"], (
            "the final step first, then ascending"
        )
    for a in traj:
        assert "--ood" in a and "--score-val" not in a and "--batch-order" not in a
    controls = box.calls("tools/ft_linear_control.py")
    assert len(controls) == 10 and sum("--option-control" in a for a in controls) == 5
    for a in controls:
        assert "--out" not in a and "--real-backbone" not in a and opt(a, "--rev") == BUILD_REV
    subs = [r[0] for r in box.rules()]
    assert subs[0] == "v5-pause" and subs.count("v5-pause") == 1 and "seeds34" not in subs
    assert box.rules()[0][box.rules()[0].index("--ft-row") + 1].startswith("0=")
    assert (
        f"cost: v5 seed 0 train+score: cap 32400 s = $37.71 at ${USD_GPU}/GPU-h; "
        "pre-registration estimate ~ 7.0 h, $29.3" in out
    )
    assert "v5 running total:" in out and "of the approved ~$309.7 / 74.0 GPU-h" in out
    assert out.count("MATCHES the prelude's") == 5
    spend = (box.q / "v5.spend").read_text().splitlines()
    assert any("v5 seed 2 needle control" in ln for ln in spend)
    assert any("v5 memory probe" in ln for ln in spend)
    traj_log = (box.root / "logs" / "q-v5traj-s0.log").read_text()
    assert (
        "last-3 average row NOT RUN (GAP-V5TRAIN-TRAJECTORY-AVERAGE-NOT-SCORABLE-2026-10-02)"
        in traj_log
    )
    assert "stub: --score-checkpoint refuses a trajectory average" in traj_log
    assert not (box.q / "v5.paused").exists()
    assert (box.q / "v5r9.word").read_text() == "continue\n"
    assert_on_its_lane(box)


def test_r9_waived_reads_no_pause_and_runs_seed_1_beside_seed_0(tmp_path: Path) -> None:
    box = Box(tmp_path, pins={"V5_R9": "waive"}, scenario={"train_sleep": 0.8})
    results = box.run_lanes()
    out = both(results)
    assert [rc for rc, _ in results] == [0, 0], out
    assert "v5-pause" not in [r[0] for r in box.rules()]
    assert not (box.q / "v5r9.word").exists() and "R9 waived" in out
    runs = {opt(r["argv"], "--seeds"): r for r in box.records() if r["argv"] in box.v5_training()}
    assert runs["0"]["cvd"] != runs["1"]["cvd"]
    assert runs["1"]["t0"] < runs["0"]["t1"], "seed 1 did not start beside seed 0"


@pytest.mark.parametrize(("word", "rc"), [("pause", 0), ("refused", 3), ("continue", 3), ("", 0)])
def test_r9_holds_seeds_1_2_on_anything_but_continue(tmp_path: Path, word: str, rc: int) -> None:
    box = Box(tmp_path, {"v5-pause": [word, rc]})
    ps = box.start_lanes()
    try:
        box.wait_for("v5.paused")
        assert (
            (box.q / "v5.paused")
            .read_text()
            .startswith("pause" if (word, rc) == ("pause", 0) else "refused")
        )
        time.sleep(0.8)
        assert seeds_of(box.training()) == ["0"]
        (box.q / "V5_STOP").write_text("Bharath: stop v5 here\n")
        results = box.finish(ps, 60)
    finally:
        for p in ps:
            if p.poll() is None:
                p.kill()
    assert [r for r, _ in results] == [3, 3] and "seeds 1-4 NOT RUN" in both(results)
    assert seeds_of(box.training()) == ["0"]
    box.wait_for("v5traj-s0.done")


def test_r9_hold_resumes_on_the_humans_continue(tmp_path: Path) -> None:
    box = Box(tmp_path, {"v5-pause": ["pause", 0]})
    ps = box.start_lanes()
    try:
        box.wait_for("v5.paused")
        (box.q / "V5_CONTINUE").write_text("")  # empty is not a yes
        time.sleep(0.8)
        assert seeds_of(box.training()) == ["0"]
        (box.q / "V5_CONTINUE").write_text("Bharath: continue v5 seeds 1-2\n")
        results = box.finish(ps)
    finally:
        for p in ps:
            if p.poll() is None:
                p.kill()
    out = both(results)
    assert [r for r, _ in results] == [0, 0], out
    assert "R9's hold: " in out and "Bharath: continue v5 seeds 1-2" in out
    assert seeds_of(box.v5_training()) == ["0", "1", "2", "3", "4"]
    box.wait_for("v5traj-s4.done")


def test_r9_never_runs_a_rules_binary_that_is_not_the_pinned_one(tmp_path: Path) -> None:
    box = Box(tmp_path, scenario={"tamper_rules_on_train": True})
    ps = box.start_lanes()
    try:
        box.wait_for("v5.paused")
        (box.q / "V5_STOP").write_text("stop\n")
        results = box.finish(ps, 60)
    finally:
        for p in ps:
            if p.poll() is None:
                p.kill()
    assert box.rules() == [], "the swapped binary ran"
    assert "not the pinned one; NOT RUN" in both(results)
    assert (box.q / "v5.paused").read_text() == "refused\n"
    box.wait_for("v5traj-s0.done")


@pytest.mark.parametrize(
    ("said", "train_fails", "reason"),
    [
        (["pause", 0], False, "pause"),
        (["refused", 3], False, "refused"),
        (["continue", 3], False, "refused"),
        (["hold", 0], False, "unknown:hold"),
        (["continue", 0], True, "no-seed-0-row"),
    ],
)
def test_r9_hold_marker_says_why(
    tmp_path: Path, said: list, train_fails: bool, reason: str
) -> None:
    """$Q/v5.paused carries exactly one of pause, refused, unknown:<word> or no-seed-0-row, so
    the human tells a reading from a tool failure without reading logs. (Changed for the 2x
    H100 box: the hold holds seeds 1-4, seeds 3-4 being unconditional since a29bca1.)"""
    box = Box(tmp_path, {"v5-pause": said}, scenario={"train_fails": train_fails})
    ps = box.start_lanes()
    try:
        box.wait_for("v5.paused")
        assert (box.q / "v5.paused").read_text() == f"{reason}\n"
        (box.q / "V5_STOP").write_text("Bharath: stop v5 here\n")
        results = box.finish(ps, 60)
    finally:
        for p in ps:
            if p.poll() is None:
                p.kill()
    assert [r for r, _ in results] == [3, 3] and "seeds 1-4 NOT RUN" in both(results), both(results)
    assert seeds_of(box.training()) == ["0"]
    assert [r[0] for r in box.rules()] == ([] if train_fails else ["v5-pause"])
    if not train_fails:
        box.wait_for("v5traj-s0.done")


def test_v5_refuses_a_run_past_the_approved_total_without_the_humans_words(tmp_path: Path) -> None:
    """Changed for the 2x H100 box: the approved total is the amended DRAFT's 11 runs, $309.7 /
    74.0 GPU-h, so 68 h already spent ($284.92 at $4.19/GPU-h) leaves no room for a seed; a
    seed 0 that never ran holds R9 as no-seed-0-row."""
    box = Box(tmp_path)
    (box.q / "v5.spend").write_text(f"2026-10-04T00:00:00Z\tearlier\t{68 * 3600}\t284.92\n")
    ps = box.start_lanes()
    try:
        box.wait_for("v5.paused")
        (box.q / "V5_STOP").write_text("stop\n")
        results = box.finish(ps, 60)
    finally:
        for p in ps:
            if p.poll() is None:
                p.kill()
    out = both(results)
    assert box.training() == []
    assert "v5 seed 0 REFUSED" in out and "would cross the approved ~$309.7 / 74.0 GPU-h" in out
    assert (box.q / "v5.paused").read_text() == "no-seed-0-row\n"


def test_v5_with_the_over_budget_words_runs(tmp_path: Path) -> None:
    box = Box(tmp_path)
    (box.q / "v5.spend").write_text(f"2026-10-04T00:00:00Z\tearlier\t{68 * 3600}\t284.92\n")
    (box.q / "V5_OVER_BUDGET_YES").write_text("Bharath: finish v5 past the estimate\n")
    results = box.run_lanes()
    assert [rc for rc, _ in results] == [0, 0], both(results)
    assert seeds_of(box.v5_training()) == ["0", "1", "2", "3", "4"]
    box.wait_for("v5traj-s4.done")


@pytest.mark.parametrize(
    "stale",
    [
        "V5_CONTINUE",
        "V5_STOP",
        "v5.paused",
        "V5_PROBE_YES",
        "v5.probe-passed",
        "v5.probe-failed",
        "v5r9.word",
        "v5nw.room",
        "v5nw.launch",
        "v5nw.word",
        "v5job-v5-s0.claimed",
        "v5traj-s1.queued",
    ],
)
def test_v5_refuses_stale_markers(tmp_path: Path, stale: str) -> None:
    """Changed for the 2x H100 box (split from test_v5_refuses_stale_markers_and_an_existing_
    ledger, whose markers were the GH200 waiter's three): both lanes refuse over every marker an
    earlier launch writes, the probe's, the lanes' decision words and the per-job ones included."""
    box = Box(tmp_path)
    (box.q / stale).write_text("old\n")
    results = box.run_lanes()
    assert [rc for rc, _ in results] == [3, 3], both(results)
    assert "stale" in both(results) and box.calls() == []


@pytest.mark.parametrize("ledger", ["ledger", "arm_ledger", "probe_ledger"])
def test_v5_refuses_an_existing_ledger(tmp_path: Path, ledger: str) -> None:
    """Changed for the 2x H100 box (split from test_v5_refuses_stale_markers_and_an_existing_
    ledger, which knew one gh200-v5 ledger): lane 0 refuses when any of the three ledgers named by
    V5_BOX exists, and lane 1, waiting for the probe, refuses once lane 0 has ended."""
    box = Box(tmp_path)
    getattr(box, ledger).write_text("{}\n")
    results = box.run_lanes()
    assert [rc for rc, _ in results] == [3, 3], both(results)
    assert "new ledger" in results[0][1] and box.calls() == []
    assert "lane 0" in results[1][1]


def test_v5_refuses_a_second_copy_of_a_lane(tmp_path: Path) -> None:
    box = Box(tmp_path)
    (box.q / "v5lane1.queued").write_text("")
    p = box.popen("box_q_v5.sh", "1")
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 3 and "stale" in out and box.calls() == []
    p = box.popen("box_q_v5.sh", "2")
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 2 and "usage" in out


def test_v5_refuses_a_rule3_split_in_its_data_dir(tmp_path: Path) -> None:
    # (not named after the split: pytest puts the test's name in tmp_path, and the split guard
    # refuses any argv word naming held-out data, which would stop this run one check earlier)
    box = Box(tmp_path)
    (box.root / "v5data" / "data" / "heldout").mkdir(parents=True)
    results = box.run_lanes()
    assert [rc for rc, _ in results] == [3, 3]
    assert "(rule 3); refusing" in both(results) and box.calls() == []


@pytest.mark.parametrize("word", [["quiet", 0], ["refused", 3], ["fires", 0]])
def test_seeds_3_4_run_unconditionally_and_seeds34_is_never_read(
    tmp_path: Path, word: list
) -> None:
    """Replaces test_v5s34, whose expectation was the conditional seeds 3-4 path that the
    amendment a29bca1 retired (Fable, ~17:25Z): whatever seeds34 would say, it is never asked,
    no v5s34.word is written, and seeds 3 and 4 run in v5's form on plan seeds 3 and 4, each
    with its post-seed waiter. A second launch on the same queue refuses: decided once."""
    box = Box(tmp_path, {"seeds34": word})
    results = box.run_lanes()
    out = both(results)
    assert [r for r, _ in results] == [0, 0], out
    assert [c for c in box.rules() if c[0] == "seeds34"] == []
    assert not (box.q / "v5s34.word").exists()
    assert seeds_of(box.v5_training()) == ["0", "1", "2", "3", "4"]
    for argv in box.v5_training():
        assert_training_form(argv, int(opt(argv, "--seeds")))
        assert opt(argv, "--ledger") == str(box.ledger)
    box.wait_for("v5traj-s3.done", "v5traj-s4.done")
    assert "of the approved ~$309.7 / 74.0 GPU-h" in out
    results2 = box.run_lanes()
    assert [r for r, _ in results2] == [3, 3] and "decided once" in both(results2)


YES_WORDS = "Bharath: run the arm anyway\n"


@pytest.mark.parametrize(
    ("word", "rc", "yes", "arm"),
    [
        ("room", 0, None, True),
        ("no_room", 0, None, False),
        ("no_room", 0, "", False),  # an empty V5NW_HUMAN_YES is not a yes
        ("no_room", 0, YES_WORDS, True),
        ("refused", 3, YES_WORDS, False),
        ("room", 3, YES_WORDS, False),
        ("", 0, YES_WORDS, False),
    ],
)
def test_v5nw(tmp_path: Path, word: str, rc: int, yes: str | None, arm: bool) -> None:
    """Changed for the 2x H100 box: the room, the arm and its reading are lane decisions and
    jobs (v5_decide_room, v5_job_seed v5nw, v5_decide_arm), not box_q_v5nw.sh's."""
    box = Box(tmp_path, {"v5-noulw --room": [word, rc], "v5-noulw": ["quiet", 0]})
    if yes is not None:
        (box.q / "V5NW_HUMAN_YES").write_text(yes)
    results = box.run_lanes()
    out = both(results)
    assert [r for r, _ in results] == [0, 0], out
    room_calls = [c for c in box.rules() if c[:2] == ["v5-noulw", "--room"]]
    assert len(room_calls) == 1
    room_call = room_calls[0]
    assert opt(room_call, "--noul-preregistration").endswith("v4-noul-v3b-preregistered.json")
    assert [room_call[i + 1][:2] for i, a in enumerate(room_call) if a == "--ft-row"] == [
        "0=",
        "1=",
        "2=",
    ]
    assert (box.q / "v5nw.room").read_text().strip() == (word if rc == 0 and word else "refused")
    assert (box.q / "v5nw.launch").read_text().strip() == ("run" if arm else "skip")
    readings = [c for c in box.rules() if c[0] == "v5-noulw" and "--room" not in c]
    train = box.arm_training()
    if not arm:
        assert train == [] and readings == []
        assert ("SKIPPED (R7)" in out) is (word == "no_room" and rc == 0)
        return
    assert seeds_of(train) == ["0", "1", "2"]
    for argv in train:
        assert_training_form(argv, int(opt(argv, "--seeds")), extra=[("--noul-weight", "4")])
        assert opt(argv, "--ledger") == str(box.arm_ledger)
        if yes and word == "no_room":
            assert "run the arm anyway" in opt(argv, "--approved-by")
    assert len(readings) == 1
    reading = readings[0]
    assert opt(reading, "--arm-ledger") == str(box.arm_ledger)
    assert [reading[i + 1][:2] for i, a in enumerate(reading) if a == "--arm-ft-row"] == [
        "0=",
        "1=",
        "2=",
    ]
    assert (box.q / "v5nw.word").read_text().strip() == "quiet"
    box.wait_for("v5nwtraj-s0.done", "v5nwtraj-s1.done", "v5nwtraj-s2.done")


def test_v5j5_runs_three_shuffled_label_seeds_on_v5s_recipe(tmp_path: Path) -> None:
    """Changed for the 2x H100 box: J5' seed s is a lane job once v5 seed s is done, checked
    on that seed's own ft row (ft-rows with one --ft-row), not box_q_v5j5.sh's after v5.done."""
    box = Box(tmp_path, scenario={"eval-row": EVAL_ROW})
    results = box.run_lanes()
    out = both(results)
    assert [r for r, _ in results] == [0, 0], out
    train = box.j5_training()
    assert seeds_of(train) == ["0", "1", "2"]
    for argv in train:
        assert_training_form(argv, int(opt(argv, "--seeds")))
        assert opt(argv, "--shuffled-label") == EVAL_ROW and "--score-val" in argv
        assert opt(argv, "--ledger") == str(box.ledger)
        for absent in (
            "--checkpoint-dir",
            "--retain-tower-every",
            "--needle",
            "--ood",
            "--noul-weight",
        ):
            assert absent not in argv
    ft_rows = [c for c in box.rules() if c[0] == "ft-rows"]
    assert len(ft_rows) == 3
    assert sorted(c[c.index("--ft-row") + 1][:2] for c in ft_rows) == ["0=", "1=", "2="]
    assert all(c.count("--ft-row") == 1 for c in ft_rows)
    assert len([c for c in box.rules() if c[0] == "eval-row"]) == 3
    assert "pre-registration estimate ~ 6.0 h, $25.1" in out


def test_v5j5_is_skipped_without_a_completed_v5_ft_row(tmp_path: Path) -> None:
    """Replaces test_v5j5_is_skipped_without_three_completed_v5_ft_rows, whose box_q_v5j5.sh
    checked all three v5 ft rows at once after v5.done: J5' seed s is now a lane job checked on v5
    seed s's own ft row (hardware.lanes; GAP-V5-2GPU-J5-PER-SEED-VS-RUNS-IFF-2026-10-03), so each
    seed is skipped on its own."""
    box = Box(tmp_path, scenario={"ft-rows": 1})
    results = box.run_lanes()
    out = both(results)
    assert [r for r, _ in results] == [0, 0], out
    assert out.count("v5j5 SKIPPED") == 3 and box.j5_training() == []
    for s in range(3):
        assert (box.q / f"v5job-j5-s{s}.done").read_text().startswith("skipped")


# Every reading a post-seed waiter or the other lane may race (Fable's ruling B): --room on
# v5's ledger, the arm's reading on the arm's, and J5''s ft-rows and eval-row on v5's. (seeds34
# was a site; the amendment a29bca1 retired it.)
RACE_SITES = {
    "room": (
        {"v5-noulw --room": ["room", 0], "v5-noulw": ["quiet", 0]},
        {},
        ("v5-noulw --room", "--v5-ledger", 2),
    ),
    "arm reading": (
        {"v5-noulw --room": ["room", 0], "v5-noulw": ["quiet", 0]},
        {},
        ("v5-noulw", "--arm-ledger", 2),
    ),
    "ft-rows": ({}, {"ft-rows": 0}, ("ft-rows", "--ledger", 2)),
    "eval-row": ({}, {"ft-rows": 0}, ("eval-row", "--ledger", 1)),
}


def race_box(tmp: Path, site: str, n: int | None = None) -> tuple[Box, int]:
    rules, extra, (key, flag, races) = RACE_SITES[site]
    races = races if n is None else n
    box = Box(tmp, rules, scenario={**extra, "race": {key: [flag, races]}})
    return box, races


def calls_of(box: Box, key: str) -> list[list[str]]:
    sub, _, room = key.partition(" ")
    return [c for c in box.rules() if c[0] == sub and (("--room" in c) == bool(room))]


@pytest.mark.parametrize("site", list(RACE_SITES))
def test_each_reading_a_post_seed_waiter_may_race_is_retried_then_read(
    tmp_path: Path, site: str
) -> None:
    """Changed for the 2x H100 box: the readings are made by the lanes; J5''s look-ups are per
    seed, so the race hits whichever seed's look-up comes first."""
    box, races = race_box(tmp_path, site)
    key = RACE_SITES[site][2][0]
    results = box.run_lanes()
    out = both(results)
    assert [r for r, _ in results] == [0, 0], out
    # one reading each, plus one call per refused attempt; J5''s look-ups are once per seed
    reads = 3 if key in ("eval-row", "ft-rows") else 1
    assert len(calls_of(box, key)) == races + reads, out
    if key in ("eval-row", "ft-rows"):
        seeds = sorted({c[c.index("--ft-row") + 1][:1] for c in calls_of(box, key)})
        assert seeds == ["0", "1", "2"]
    assert "half-written" in out
    assert seeds_of(box.v5_training()) == ["0", "1", "2", "3", "4"]
    if site in ("room", "arm reading"):
        assert (box.q / "v5nw.room").read_text() == "room\n"
        assert (box.q / "v5nw.word").read_text() == "quiet\n"
        assert seeds_of(box.arm_training()) == ["0", "1", "2"]
        box.wait_for("v5nwtraj-s0.done", "v5nwtraj-s1.done", "v5nwtraj-s2.done")
    else:
        assert seeds_of(box.j5_training()) == ["0", "1", "2"]
        assert all(opt(a, "--shuffled-label") == EVAL_ROW for a in box.j5_training())


def test_a_reading_whose_ledger_stays_half_written_is_refused_after_k_attempts(
    tmp_path: Path,
) -> None:
    """Changed for the 2x H100 box: the site is the arm's reading (seeds34's is retired); the arm
    ledger has no writer after it, so its half row can stay without stalling another run."""
    box, _ = race_box(tmp_path, "arm reading", n=99)
    results = box.run_lanes()
    assert [r for r, _ in results] == [0, 0], both(results)
    assert len(calls_of(box, "v5-noulw")) == RETRIES_K
    assert (box.q / "v5nw.word").read_text() == "refused\n"
    assert "never read as quiet" in both(results)
    assert seeds_of(box.arm_training()) == ["0", "1", "2"]
    assert seeds_of(box.j5_training()) == ["0", "1", "2"]


def test_the_post_seed_waiter_refuses_a_bad_ft_row_and_still_touches_done(tmp_path: Path) -> None:
    """Changed for the 2x H100 box: the waiter takes its parent's lane as a fourth argument."""
    box = Box(tmp_path)
    p = subprocess.run(
        [BASH, str(box.root / "post-f" / "box_q_v5traj.sh"), "v5", "0", "not-a-row", "0"],
        capture_output=True,
        text=True,
        env=box.env,
        timeout=60,
        check=False,
    )
    assert p.returncode == 3 and "is not a row id" in p.stdout
    assert (box.q / "v5traj-s0.done").exists() and box.calls() == []
    for argv in (["v5nw", "3", EVAL_ROW, "0"], ["v5", "0", EVAL_ROW, "2"], ["v5", "0", EVAL_ROW]):
        p = subprocess.run(
            [BASH, str(box.root / "post-f" / "box_q_v5traj.sh"), *argv],
            capture_output=True,
            text=True,
            env=box.env,
            timeout=60,
            check=False,
        )
        assert p.returncode == 2 and "usage" in p.stdout, argv
