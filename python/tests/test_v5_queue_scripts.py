"""The v5 block's box waiters, driven on the Mac (no box, no GPU).

Files under test: ``campaign/post-f-queue/{box_q_v5.sh, box_q_v5s34.sh, box_q_v5nw.sh,
box_q_v5j5.sh, box_q_v5traj.sh, v5_common.sh}`` (lane L-v5-queue). Three kinds of test:

* static: ``bash -n`` and shellcheck on every file; the caps and constants are the
  pre-registration's; held-out data and the containment request are named only by the guard
  that refuses them; the rules binary is reached only through the two functions that verify its
  pin first; the committed pins are the literal UNSET (fail closed).
* the decision functions in ``v5_common.sh``, sourced under ``/bin/bash`` with a stub rules
  binary in a temp dir: every word each subcommand can print, every exit code, a missing JSON,
  a binary that is not the pinned one (then it is never run), and unknown or empty output, mapped
  to the action the DRAFT names (an unknown or empty word holds or skips).
* dry runs of each waiter: a copy of the queue directory with ``/home/ubuntu`` rewritten to a temp
  root, a fake ``python`` that records every ``tools/real_ft_run.py`` / ``ft_linear_control.py``
  argv (and execs the real interpreter for the waiters' own ``-c`` / ``-`` checks), a fake rules
  binary driven by a scenario file, and fakes for ``flock``, ``timeout``, ``sleep``, ``setsid``,
  ``nohup`` and ``git`` (the box has the real ones; the Mac has no flock or timeout). The
  assertions are on the recorded argv: ``--batch-order seed`` on every training call, the caps,
  ``--approved-by``, the arm's and J5''s recipe equal to v5's, and no argv naming held-out data.

``QD_V5_QUEUE_DIR`` points the tests at another copy of the queue directory (the fail-first run
uses a tree without the v5 files).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
QUEUE = Path(os.environ.get("QD_V5_QUEUE_DIR", str(REPO / "campaign" / "post-f-queue")))
DRAFT = REPO / "campaign" / "v5-preregistered.DRAFT.json"
NOUL = REPO / "campaign" / "v4-noul-v3b-preregistered.json"
BASH = "/bin/bash"
WAITERS = ("box_q_v5.sh", "box_q_v5s34.sh", "box_q_v5nw.sh", "box_q_v5j5.sh", "box_q_v5traj.sh")
V5_FILES = (*WAITERS, "v5_common.sh")
PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
LANE_AT = "0123456789abcdef0123456789abcdef01234567"
BUILD_REV = "89abcdef0123456789abcdef0123456789abcdef"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
EVAL_ROW = "eeeeeeee-1111-2222-3333-444444444444"
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


def common_text() -> str:
    return (QUEUE / "v5_common.sh").read_text(encoding="utf-8")


def test_committed_pins_are_unset_so_every_waiter_fails_closed() -> None:
    text = common_text()
    for pin in (
        "RULES_V5_SHA256",
        "V5_PREP_BIN_SHA256",
        "V5_LANE_AT",
        "V5_PREREG_SHA256",
        "V5_DATE",
        "V5_PRELUDE_SHA256",
        "V5_C1",
        "V5_C2A",
        "V5_C2B",
        "V5_LOWER",
        "V5_LRSET",
    ):
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
    for fn in ("v5_rule", "v5_rules_call"):
        body = function_body(common, fn)
        assert body.index('pin "$RULES_V5" "$RULES_V5_SHA256"') < body.index('"$RULES_V5" "$@"'), fn


# --- the decision functions -------------------------------------------------------------------


def lib(tmp: Path, extra: str = "") -> str:
    q = tmp / "q"
    q.mkdir(exist_ok=True)
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


ACTIONS = [
    # R9: continue -> run seeds 1-2; pause or anything else -> hold until V5_CONTINUE
    ("v5_pause_action continue", "run"),
    ("v5_pause_action pause", "hold"),
    ("v5_pause_action refused", "hold"),
    ("v5_pause_action ''", "hold"),
    ("v5_pause_action 'continue '", "hold"),
    ("v5_pause_action room", "hold"),
    # the arm's launch: room; no_room only with V5NW_HUMAN_YES; refused never
    ("v5_room_action room no", "run"),
    ("v5_room_action room yes", "run"),
    ("v5_room_action no_room no", "skip"),
    ("v5_room_action no_room yes", "run"),
    ("v5_room_action refused yes", "skip"),
    ("v5_room_action refused no", "skip"),
    ("v5_room_action '' yes", "skip"),
    ("v5_room_action quiet yes", "skip"),
    # seeds 3-4
    ("v5_s34_action fires", "run"),
    ("v5_s34_action quiet", "skip"),
    ("v5_s34_action refused", "notrun"),
    ("v5_s34_action ''", "notrun"),
    ("v5_s34_action fires:j6f", "notrun"),
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
V5_C1=off; V5_C2A=off; V5_C2B=off; V5_LOWER=keep; V5_LRSET=f
V5_SPLIT=(--out /data/v5 --rev {BUILD_REV} --real-backbone "$BACKBONE")
"""


def test_v5_pins_unset_lists_every_unset_pin(tmp_path: Path) -> None:
    r = bash(lib(tmp_path) + 'echo "UNSET=$(v5_pins_unset)"')
    got = re.findall(r"^UNSET=(.*)$", r.stdout, re.M)[0].split()
    assert sorted(got) == sorted(
        [
            "RULES_V5_SHA256",
            "V5_PREP_BIN_SHA256",
            "V5_LANE_AT",
            "V5_PREREG_SHA256",
            "V5_DATE",
            "V5_PRELUDE_SHA256",
            "V5_C1",
            "V5_C2A",
            "V5_C2B",
            "V5_LOWER",
            "V5_LRSET",
            "V5_SPLIT",
        ]
    )
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
    ],
)
def test_v5_pins_check_refuses_a_malformed_pin(tmp_path: Path, override: str, message: str) -> None:
    r = bash(lib(tmp_path, GOOD_PINS + override) + 'echo "CHECK=$(v5_pins_check | tr "\\n" ";")"')
    assert message in r.stdout


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


def read_prereg(tmp: Path, prereg: dict | str) -> subprocess.CompletedProcess:
    p = tmp / "prereg.json"
    p.write_text(prereg if isinstance(prereg, str) else json.dumps(prereg), encoding="utf-8")
    return bash(
        lib(tmp, f'V5_PREREG="{p}"; PY="{sys.executable}"')
        + """
v5_read_prereg; echo "RC=$?"
echo "NUMS=$V5_EST_SEED_USD $V5_EST_SEED_H $V5_EST_J5_USD $V5_EST_J5_H" \
  "$V5_APPROVED_USD $V5_APPROVED_H $V5_S34_USD $V5_S34_H $V5NW_W"
"""
    )


def test_v5_read_prereg_reads_the_drafts_numbers(tmp_path: Path) -> None:
    r = read_prereg(tmp_path, renamed_prereg())
    assert "RC=0" in r.stdout, r.stdout + r.stderr
    assert "NUMS=16.0 7.0 13.7 6.0 137.2 60.0 32.0 14.0 4" in r.stdout


def test_v5_read_prereg_refuses_the_draft_itself(tmp_path: Path) -> None:
    r = read_prereg(tmp_path, DRAFT.read_text(encoding="utf-8"))
    assert "RC=3" in r.stdout and "'draft' key" in r.stdout


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["launch"]["projected_cost_usd"].__setitem__(
            "check",
            d["launch"]["projected_cost_usd"]["check"].replace("~ 7.0 h, $16.0", "~ 7.0 h, $18.0"),
        ),
        lambda d: d["launch"]["projected_cost_usd"].__setitem__("total", 150.0),
        lambda d: d["launch"]["projected_gpu_hours"].__setitem__("if_seeds_3_4", 20.0),
        lambda d: d["launch"]["projected_cost_usd"].__setitem__(
            "check", "no per-seed figures here"
        ),
        lambda d: d["arm_noul_weight"]["w"].__setitem__("value", 0),
        lambda d: d["arm_noul_weight"]["w"].__setitem__("value", "4"),
        lambda d: d["launch"].pop("projected_gpu_hours"),
    ],
)
def test_v5_read_prereg_refuses_figures_that_disagree(tmp_path: Path, mutate) -> None:
    d = renamed_prereg()
    mutate(d)
    r = read_prereg(tmp_path, d)
    assert "RC=3" in r.stdout and "REFUSED" in r.stdout, r.stdout


def budget(
    tmp: Path,
    spent_s: int | None,
    *,
    yes: str | None = None,
    s34: str | None = None,
    est: str = "16.0 7.0",
) -> str:
    q = tmp / "q"
    q.mkdir(exist_ok=True)
    if spent_s is not None:
        (q / "v5.spend").write_text(f"2026-10-04T00:00:00Z\tprior\t{spent_s}\t0\n")
    if yes is not None:
        (q / "V5_OVER_BUDGET_YES").write_text(yes)
    if s34 is not None:
        (q / "v5s34.word").write_text(s34 + "\n")
    r = bash(
        lib(
            tmp,
            "V5_USD_PER_HOUR=2.29; V5_APPROVED_USD=137.2; V5_APPROVED_H=60.0; "
            "V5_S34_USD=32.0; V5_S34_H=14.0",
        )
        + f'v5_budget_ok {est} "v5 seed 2"; echo "RC=$?"'
    )
    return r.stdout


def test_v5_budget_ok_passes_inside_the_approved_total(tmp_path: Path) -> None:
    assert "RC=0" in budget(tmp_path, None)
    # 52 h = $119.08 spent; + ~$16.0 / 7.0 h = $135.08 / 59 h, inside $137.2 / 60 h
    assert "RC=0" in budget(tmp_path, 52 * 3600)
    # 53 h = $121.37; + $16.0 = $137.37 crosses $137.2 although 53 + 7.0 = 60 h does not
    assert "RC=3" in budget(tmp_path, 53 * 3600)


def test_v5_budget_ok_refuses_a_run_that_would_cross_it(tmp_path: Path) -> None:
    out = budget(tmp_path, 54 * 3600)  # 54 + 7.0 > 60 GPU-h
    assert "RC=3" in out and "REFUSED" in out and "NOT RUN" in out


def test_v5_budget_ok_runs_past_it_only_on_the_humans_words(tmp_path: Path) -> None:
    assert "RC=3" in budget(tmp_path, 54 * 3600, yes="")
    out = budget(tmp_path, 54 * 3600, yes="Bharath: yes, finish the block")
    assert "RC=0" in out and "Bharath: yes, finish the block" in out


def test_v5_budget_ok_adds_seeds_3_4_once_seeds34_fired(tmp_path: Path) -> None:
    assert "RC=0" in budget(tmp_path, 54 * 3600, s34="fires")
    assert "RC=3" in budget(tmp_path, 68 * 3600, s34="fires")
    assert "RC=3" in budget(tmp_path, 54 * 3600, s34="quiet")


def test_v5_spend_add_and_running_total(tmp_path: Path) -> None:
    r = bash(
        lib(tmp_path, "V5_USD_PER_HOUR=2.29; V5_APPROVED_USD=137.2; V5_APPROVED_H=60.0")
        + 'v5_spend_add "v5 seed 0 train+score" 3600; v5_spend_add "x" 7200'
    )
    assert "v5 running total: 1.0000 GPU-h, $2.29 of the approved ~$137.2 / 60.0 GPU-h" in r.stdout
    assert "v5 running total: 3.0000 GPU-h, $6.87 of the approved" in r.stdout
    lines = (tmp_path / "q" / "v5.spend").read_text().splitlines()
    assert [ln.split("\t")[1:] for ln in lines] == [
        ["v5 seed 0 train+score", "3600", "2.29"],
        ["x", "7200", "4.58"],
    ]


# --- the prelude record, the waiter's side of the contract ----------------------------------


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


# --- dry runs of the waiters ----------------------------------------------------------------


FAKE_PYTHON = r"""#!{real}
import json, os, sys, uuid
REAL = {real!r}
ROOT = {root!r}
args = sys.argv[1:]
if args[:1] in (["-c"], ["-"]):
    os.execv(REAL, [REAL] + args)
with open(os.path.join(ROOT, "argv.log"), "a") as f:
    f.write(json.dumps(args) + "\n")
scen = json.load(open(os.path.join(ROOT, "scenario.json")))


def opt(name):
    return args[args.index(name) + 1] if name in args else None


if "tools/real_ft_run.py" in args and "--score-checkpoint" not in args:
    seed = opt("--seeds")
    if scen.get("tamper_rules_on_train"):
        with open(os.path.join(ROOT, "bin", "qd-post-f-rules-v5"), "a") as f:
            f.write("\n# swapped after the waiter checked it\n")
    if scen.get("train_fails"):
        sys.exit(1)
    if opt("--verdicts-out"):
        open(opt("--verdicts-out"), "w").write('{{"verdict": "stub"}}\n')
    row = str(uuid.uuid4())
    with open(opt("--ledger"), "a") as f:
        digest = scen.get("digests", {{}}).get(seed)
        f.write(json.dumps({{"row_id": row, "metrics": {{"corpus.plan_order_digest":
                {{"state": "ran", "value": digest}}}}}}) + "\n")
    ck = opt("--checkpoint-dir")
    if ck:
        open(os.path.join(ck, f"epoch-seed{{seed}}-cuda.json"), "w").write("{{}}")
        if opt("--retain-tower-every"):
            for s in scen.get("steps", [1000, 2000, 2470]):
                open(os.path.join(ck, f"epoch-seed{{seed}}-cuda-step{{s}}.json"), "w").write("{{}}")
    print(f"ft row {{row}}")
sys.exit(0)
"""

FAKE_RULES = r"""#!{real}
import json, os, sys
ROOT = {root!r}
args = sys.argv[1:]
with open(os.path.join(ROOT, "rules.log"), "a") as f:
    f.write(json.dumps(args) + "\n")
scen = json.load(open(os.path.join(ROOT, "scenario.json")))
sub = args[0]
if sub == "ft-rows":
    sys.exit(scen.get("ft-rows", 0))
if sub == "eval-row":
    print(scen.get("eval-row", "{eval_row}"))
    sys.exit(0)
key = sub + (" --room" if "--room" in args else "")
word, rc = scen["rules"][key]
out = args[args.index("--out") + 1]
open(out, "w").write(json.dumps({{"stub": key}}))
print(word)
sys.exit(rc)
"""


class Box:
    """A /home/ubuntu stand-in with the v5 waiters' pins filled for it."""

    def __init__(
        self,
        tmp: Path,
        scenario: dict,
        *,
        fill_pins: bool = True,
        pins: dict[str, str] | None = None,
        consts: dict[str, str] | None = None,
    ) -> None:
        self.root = root = tmp / "box"
        self.q = root / "queue"
        for d in ("queue", "ledger", "logs", "bin", "post-f", "v5data/shards/train", "fakebin"):
            (root / d).mkdir(parents=True, exist_ok=True)
        self.scenario = scenario
        self.write_scenario()
        real = sys.executable
        write_exec(
            root / "qd-venv" / "bin" / "python", FAKE_PYTHON.format(real=real, root=str(root))
        )
        rules = write_exec(
            root / "bin" / "qd-post-f-rules-v5",
            FAKE_RULES.format(real=real, root=str(root), eval_row=EVAL_ROW),
        )
        prep = write_exec(root / "bin" / "qd-prep-v5", "#!/bin/sh\nexit 0\n")
        for name, body in {
            "flock": "#!/bin/sh\nexit 0\n",
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
        if fill_pins:
            fills = {
                "RULES_V5_SHA256": sha256(rules),
                "V5_PREP_BIN_SHA256": sha256(prep),
                "V5_LANE_AT": LANE_AT,
                "V5_PREREG_SHA256": sha256(prereg),
                "V5_DATE": "2026-10-04",
                "V5_PRELUDE_SHA256": sha256(record),
                **decisions,
            }
            path = root / "post-f" / "v5_common.sh"
            text = path.read_text(encoding="utf-8")
            for name, value in fills.items():
                text, n = re.subn(rf"^{name}=UNSET$", f"{name}={value}", text, flags=re.M)
                assert n == 1, name
            text, n = re.subn(r"^V5_SPLIT=\(UNSET\)$", f"V5_SPLIT=({self.split})", text, flags=re.M)
            assert n == 1
            for name, value in (consts or {}).items():
                text, n = re.subn(rf"^{name}=[0-9]+$", f"{name}={value}", text, flags=re.M)
                assert n == 1, name
            path.write_text(text, encoding="utf-8")
        self.ledger = root / "ledger" / "gh200-v5-2026-10-04.jsonl"
        self.arm_ledger = root / "ledger" / "gh200-v5-noulw-2026-10-04.jsonl"

    def write_scenario(self) -> None:
        (self.root / "scenario.json").write_text(json.dumps(self.scenario))

    @property
    def env(self) -> dict[str, str]:
        return {"PATH": f"{self.root}/fakebin:{PATH}", "HOME": str(self.root), "LC_ALL": "C"}

    def popen(self, waiter: str) -> subprocess.Popen:
        return subprocess.Popen(
            [BASH, str(self.root / "post-f" / waiter)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=self.env,
            stdin=subprocess.DEVNULL,
        )

    def run(self, waiter: str, timeout: float = 120) -> tuple[int, str]:
        p = self.popen(waiter)
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out

    def calls(self, tool: str = "tools/real_ft_run.py") -> list[list[str]]:
        log = self.root / "argv.log"
        if not log.exists():
            return []
        return [a for a in map(json.loads, log.read_text().splitlines()) if tool in a]

    def training(self) -> list[list[str]]:
        return [a for a in self.calls() if "--score-checkpoint" not in a]

    def rules(self) -> list[list[str]]:
        log = self.root / "rules.log"
        return [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []

    def wait_for(self, *names: str, timeout: float = 60) -> None:
        end = time.monotonic() + timeout
        for n in names:
            while not (self.q / n).exists():
                assert time.monotonic() < end, f"{n} never appeared"
                time.sleep(0.05)

    def fake_v5_logs(self, seeds: tuple[int, ...] = (0, 1, 2), out: str = "v5") -> list[str]:
        ids = []
        (self.root / out).mkdir(exist_ok=True)
        for s in seeds:
            row = f"{s:08x}-aaaa-bbbb-cccc-dddddddddddd"
            (self.root / out / f"train-s{s}.log").write_text(f"... ft row {row}\n")
            ids.append(row)
        return ids


def opt(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def assert_training_form(argv: list[str], seed: int, *, extra: list[tuple[str, str]] = ()) -> None:
    assert opt(argv, "--seeds") == str(seed)
    assert recipe_pairs(argv) == sorted([*recipe_pairs(BASE_RECIPE), *extra], key=repr)
    assert opt(argv, "--batch-order") == "seed"
    assert opt(argv, "--wall-clock-cap-s") == "32400"
    assert opt(argv, "--approved-by") and "d24c865" in opt(argv, "--approved-by")
    assert opt(argv, "--usd-per-hour") == "2.29"
    assert not any(re.search(r"heldout|held-out|containment", a, re.I) for a in argv), argv


def started_box(tmp: Path, rules: dict, **kw) -> Box:
    box = Box(tmp, {"rules": rules, "digests": {}}, **kw)
    box.scenario["digests"] = box.digests
    box.write_scenario()
    (box.q / "j5pp.done").write_text("")
    (box.q / "V5_LAUNCH_YES").write_text("Bharath: yes, launch v5 (answer 1 at d24c865)\n")
    return box


def test_v5_waits_for_the_chain_then_for_the_humans_launch_words(tmp_path: Path) -> None:
    box = started_box(tmp_path, {"v5-pause": ["continue", 0]})
    (box.q / "j5pp.done").unlink()
    (box.q / "j6g.queued").write_text("")
    (box.q / "V5_LAUNCH_YES").write_text("")  # empty is not a yes
    p = box.popen("box_q_v5.sh")
    try:
        time.sleep(1.0)
        assert box.calls() == [] and not (box.q / "v5.started").exists()  # j5pp not done
        (box.q / "j5pp.done").write_text("")
        time.sleep(1.0)
        assert box.calls() == [] and not (box.q / "v5.started").exists()  # j6g queued, not done
        (box.q / "j6g.done").write_text("")
        time.sleep(1.0)
        assert box.calls() == [] and not (box.q / "v5.started").exists()  # no launch words
        (box.q / "V5_LAUNCH_YES").write_text("Bharath: launch v5\n")
        out, _ = p.communicate(timeout=120)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 0, out
    assert "j6g is queued, so v5 waits for j6g.done" in out and "j6a is NOT queued" in out
    assert "the human's v5 launch yes: " in out and "Bharath: launch v5" in out
    assert [opt(a, "--seeds") for a in box.training()] == ["0", "1", "2"]
    box.wait_for("v5traj-s2.done")


def test_v5_stop_ends_the_wait_for_the_launch_words(tmp_path: Path) -> None:
    box = started_box(tmp_path, {"v5-pause": ["continue", 0]})
    (box.q / "V5_LAUNCH_YES").unlink()
    p = box.popen("box_q_v5.sh")
    try:
        time.sleep(1.0)
        (box.q / "V5_STOP").write_text("Bharath: not now\n")
        out, _ = p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 3 and "v5 NOT RUN" in out and box.calls() == []


def test_the_trajectory_cap_leaves_steps_not_run_and_the_controls_still_run(tmp_path: Path) -> None:
    box = Box(tmp_path, {"rules": {}}, consts={"V5_TRAJ_CAP_S": "100"})
    ck = box.root / "ckpt" / "v5"
    ck.mkdir(parents=True)
    for s in (1000, 2000):
        (ck / f"epoch-seed0-cuda-step{s}.json").write_text("{}")
    (box.root / "v5").mkdir()
    (box.root / "v5" / "verdicts-s0.jsonl").write_text('{"verdict": "stub"}\n')
    p = subprocess.run(
        [BASH, str(box.root / "post-f" / "box_q_v5traj.sh"), "v5", "0", EVAL_ROW],
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


def test_v5_with_its_pins_unset_trains_nothing_and_touches_done(tmp_path: Path) -> None:
    box = started_box(tmp_path, {}, fill_pins=False)
    rc, out = box.run("box_q_v5.sh")
    assert rc == 3 and "v5 deferred: pins UNSET:" in out
    assert box.calls() == [] and box.rules() == []
    assert (box.q / "v5.done").exists() and not (box.q / "v5.started").exists()


def test_v5_continue_runs_three_seeds_in_v5s_form(tmp_path: Path) -> None:
    box = started_box(tmp_path, {"v5-pause": ["continue", 0]})
    rc, out = box.run("box_q_v5.sh")
    assert rc == 0, out
    assert "v5 all done" in out
    box.wait_for("v5traj-s0.done", "v5traj-s1.done", "v5traj-s2.done")
    train = box.training()
    assert [opt(a, "--seeds") for a in train] == ["0", "1", "2"]
    for s, argv in enumerate(train):
        assert_training_form(argv, s)
        assert (
            opt(argv, "--retain-tower-every") == "1000"
            and opt(argv, "--checkpoint-every") == "100000"
        )
        assert {"--score-val", "--needle", "--ood"} <= set(argv)
        assert opt(argv, "--ledger") == str(box.ledger)
    scored = [a for a in box.calls() if "--score-checkpoint" in a]
    needle = [a for a in scored if "--needle-control" in a]
    assert [opt(a, "--seeds") for a in needle] == ["0", "1", "2"]
    for a in needle:
        assert (
            opt(a, "--needle-control") == "1024,2048,4096"
            and opt(a, "--wall-clock-cap-s") == "5400"
        )
        assert opt(a, "--score-dtype") == "fp32" and "--batch-order" not in a
    traj = [a for a in scored if "--needle-control" not in a]
    for s in range(3):
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
    assert len(controls) == 6 and sum("--option-control" in a for a in controls) == 3
    for a in controls:
        assert "--out" not in a and "--real-backbone" not in a and opt(a, "--rev") == BUILD_REV
    assert [r[0] for r in box.rules()] == ["v5-pause"]
    assert box.rules()[0][box.rules()[0].index("--ft-row") + 1].startswith("0=")
    assert (
        "cost: v5 seed 0 train+score: cap 32400 s = $20.61 at $2.29/h; pre-registration estimate ~ "
        "7.0 h, $16.0" in out
    )
    assert "v5 running total:" in out and "of the approved ~$137.2 / 60.0 GPU-h" in out
    assert out.count("MATCHES the prelude's") == 3
    spend = (box.q / "v5.spend").read_text().splitlines()
    assert any("v5 seed 2 needle control" in ln for ln in spend)
    traj_log = (box.root / "logs" / "q-v5traj-s0.log").read_text()
    assert (
        "last-3 average row NOT RUN (GAP-V5TRAIN-TRAJECTORY-AVERAGE-NOT-SCORABLE-2026-10-02)"
        in traj_log
    )
    assert "stub: --score-checkpoint refuses a trajectory average" in traj_log
    assert not (box.q / "v5.paused").exists()


@pytest.mark.parametrize(("word", "rc"), [("pause", 0), ("refused", 3), ("continue", 3), ("", 0)])
def test_r9_holds_seeds_1_2_on_anything_but_continue(tmp_path: Path, word: str, rc: int) -> None:
    box = started_box(tmp_path, {"v5-pause": [word, rc]})
    p = box.popen("box_q_v5.sh")
    try:
        box.wait_for("v5.paused")
        assert (
            (box.q / "v5.paused")
            .read_text()
            .startswith("pause" if (word, rc) == ("pause", 0) else "refused")
        )
        time.sleep(0.5)
        assert [opt(a, "--seeds") for a in box.training()] == ["0"]
        (box.q / "V5_STOP").write_text("Bharath: stop v5 here\n")
        out, _ = p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 3 and "seeds 1-2 NOT RUN" in out
    assert [opt(a, "--seeds") for a in box.training()] == ["0"]
    box.wait_for("v5traj-s0.done")


def test_r9_hold_resumes_on_the_humans_continue(tmp_path: Path) -> None:
    box = started_box(tmp_path, {"v5-pause": ["pause", 0]})
    p = box.popen("box_q_v5.sh")
    try:
        box.wait_for("v5.paused")
        (box.q / "V5_CONTINUE").write_text("")  # empty is not a yes
        time.sleep(0.5)
        assert [opt(a, "--seeds") for a in box.training()] == ["0"]
        (box.q / "V5_CONTINUE").write_text("Bharath: continue v5 seeds 1-2\n")
        out, _ = p.communicate(timeout=120)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 0, out
    assert "R9's hold: " in out and "Bharath: continue v5 seeds 1-2" in out
    assert [opt(a, "--seeds") for a in box.training()] == ["0", "1", "2"]
    box.wait_for("v5traj-s2.done")


def test_r9_never_runs_a_rules_binary_that_is_not_the_pinned_one(tmp_path: Path) -> None:
    box = started_box(tmp_path, {"v5-pause": ["continue", 0]})
    # seed 0's training swaps the binary after v5_verify checked it and before R9 calls it
    box.scenario["tamper_rules_on_train"] = True
    box.write_scenario()
    p = box.popen("box_q_v5.sh")
    try:
        box.wait_for("v5.paused")
        (box.q / "V5_STOP").write_text("stop\n")
        out, _ = p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
    assert box.rules() == [], "the swapped binary ran"
    assert "not the pinned one; NOT RUN" in out
    assert (box.q / "v5.paused").read_text().startswith("refused")
    box.wait_for("v5traj-s0.done")


def test_v5_refuses_a_run_past_the_approved_total_without_the_humans_words(tmp_path: Path) -> None:
    box = started_box(tmp_path, {"v5-pause": ["continue", 0]})
    (box.q / "v5.spend").write_text("2026-10-04T00:00:00Z\tearlier\t198000\t125.95\n")  # 55 h
    p = box.popen("box_q_v5.sh")
    try:
        box.wait_for("v5.paused")
        (box.q / "V5_STOP").write_text("stop\n")
        out, _ = p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
    assert box.training() == []
    assert "v5 seed 0 REFUSED" in out and "would cross the approved ~$137.2 / 60.0 GPU-h" in out


def test_v5_with_the_over_budget_words_runs(tmp_path: Path) -> None:
    box = started_box(tmp_path, {"v5-pause": ["continue", 0]})
    (box.q / "v5.spend").write_text("2026-10-04T00:00:00Z\tearlier\t198000\t125.95\n")
    (box.q / "V5_OVER_BUDGET_YES").write_text("Bharath: finish v5 past the estimate\n")
    rc, out = box.run("box_q_v5.sh")
    assert rc == 0, out
    assert [opt(a, "--seeds") for a in box.training()] == ["0", "1", "2"]
    box.wait_for("v5traj-s2.done")


def test_v5_refuses_stale_markers_and_an_existing_ledger(tmp_path: Path) -> None:
    box = started_box(tmp_path, {"v5-pause": ["continue", 0]})
    (box.q / "V5_CONTINUE").write_text("old\n")
    rc, out = box.run("box_q_v5.sh")
    assert rc == 3 and "stale" in out and box.calls() == []
    box2 = started_box(tmp_path / "b", {"v5-pause": ["continue", 0]})
    box2.ledger.write_text("{}\n")
    rc, out = box2.run("box_q_v5.sh")
    assert rc == 3 and "new ledger" in out and box2.calls() == []


def test_v5_refuses_a_rule3_split_in_its_data_dir(tmp_path: Path) -> None:
    # (not named after the split: pytest puts the test's name in tmp_path, and the split guard
    # refuses any argv word naming held-out data, which would stop this run one check earlier)
    box = started_box(tmp_path, {"v5-pause": ["continue", 0]})
    (box.root / "v5data" / "data" / "heldout").mkdir(parents=True)
    rc, out = box.run("box_q_v5.sh")
    assert rc == 3 and "(rule 3); refusing" in out and box.calls() == []


def done_v5(tmp: Path, rules: dict, **kw) -> Box:
    box = Box(tmp, {"rules": rules}, **kw)
    (box.q / "v5.done").write_text("")
    box.fake_v5_logs()
    return box


@pytest.mark.parametrize(
    ("word", "rc", "seeds", "code"),
    [
        ("fires", 0, ["3", "4"], 0),
        ("quiet", 0, [], 0),
        ("refused", 3, [], 3),
        ("fires", 3, [], 3),
    ],
)
def test_v5s34(tmp_path: Path, word: str, rc: int, seeds: list[str], code: int) -> None:
    box = done_v5(tmp_path, {"seeds34": [word, rc]})
    got, out = box.run("box_q_v5s34.sh")
    assert got == code, out
    train = box.training()
    assert [opt(a, "--seeds") for a in train] == seeds
    for argv in train:
        assert_training_form(argv, int(opt(argv, "--seeds")))
        assert opt(argv, "--ledger") == str(box.ledger)
    call = box.rules()[0]
    assert call[0] == "seeds34" and opt(call, "--f-ledger") == str(box.ledger)
    assert opt(call, "--preregistration").endswith("campaign/v5-preregistered.json")
    assert [call[i + 1][:2] for i, a in enumerate(call) if a == "--ft-row"] == ["0=", "1=", "2="]
    written = (box.q / "v5s34.word").read_text().strip()
    assert written == (
        "fires" if (word, rc) == ("fires", 0) else "quiet" if word == "quiet" else "refused"
    )
    if seeds:
        box.wait_for("v5traj-s3.done", "v5traj-s4.done")
        assert "of the approved ~$169.20 / 74.00 GPU-h" in out
    rc2, out2 = box.run("box_q_v5s34.sh")
    assert rc2 == 3 and "decided once" in out2


YES_WORDS = "Bharath: run the arm anyway\n"


@pytest.mark.parametrize(
    ("word", "rc", "yes", "arm", "code"),
    [
        ("room", 0, None, True, 0),
        ("no_room", 0, None, False, 0),
        ("no_room", 0, "", False, 0),  # an empty V5NW_HUMAN_YES is not a yes
        ("no_room", 0, YES_WORDS, True, 0),
        ("refused", 3, YES_WORDS, False, 3),
        ("room", 3, YES_WORDS, False, 3),
        ("", 0, YES_WORDS, False, 3),
    ],
)
def test_v5nw(tmp_path: Path, word: str, rc: int, yes: str | None, arm: bool, code: int) -> None:
    box = done_v5(tmp_path, {"v5-noulw --room": [word, rc], "v5-noulw": ["quiet", 0]})
    if yes is not None:
        (box.q / "V5NW_HUMAN_YES").write_text(yes)
    got, out = box.run("box_q_v5nw.sh")
    assert got == code, out
    room_call = box.rules()[0]
    assert room_call[:2] == ["v5-noulw", "--room"]
    assert opt(room_call, "--noul-preregistration").endswith("v4-noul-v3b-preregistered.json")
    assert [room_call[i + 1][:2] for i, a in enumerate(room_call) if a == "--ft-row"] == [
        "0=",
        "1=",
        "2=",
    ]
    assert (box.q / "v5nw.room").read_text().strip() == (word if rc == 0 and word else "refused")
    train = box.training()
    if not arm:
        assert train == [] and len(box.rules()) == 1
        assert ("SKIPPED (R7)" in out) is (word == "no_room")
        return
    assert [opt(a, "--seeds") for a in train] == ["0", "1", "2"]
    for s, argv in enumerate(train):
        assert_training_form(argv, s, extra=[("--noul-weight", "4")])
        assert opt(argv, "--ledger") == str(box.arm_ledger)
        if yes and word == "no_room":
            assert "run the arm anyway" in opt(argv, "--approved-by")
    reading = box.rules()[1]
    assert reading[0] == "v5-noulw" and "--room" not in reading
    assert opt(reading, "--arm-ledger") == str(box.arm_ledger)
    assert [reading[i + 1][:2] for i, a in enumerate(reading) if a == "--arm-ft-row"] == [
        "0=",
        "1=",
        "2=",
    ]
    assert (box.q / "v5nw.word").read_text().strip() == "quiet"
    box.wait_for("v5nwtraj-s0.done", "v5nwtraj-s1.done", "v5nwtraj-s2.done")


def test_v5j5_runs_three_shuffled_label_seeds_on_v5s_recipe(tmp_path: Path) -> None:
    box = done_v5(tmp_path, {})
    box.scenario.update({"ft-rows": 0, "eval-row": EVAL_ROW})
    box.write_scenario()
    rc, out = box.run("box_q_v5j5.sh")
    assert rc == 0, out
    train = box.training()
    assert [opt(a, "--seeds") for a in train] == ["0", "1", "2"]
    for s, argv in enumerate(train):
        assert_training_form(argv, s)
        assert opt(argv, "--shuffled-label") == EVAL_ROW and "--score-val" in argv
        for absent in (
            "--checkpoint-dir",
            "--retain-tower-every",
            "--needle",
            "--ood",
            "--noul-weight",
        ):
            assert absent not in argv
    assert [r[0] for r in box.rules()] == ["ft-rows", "eval-row", "eval-row", "eval-row"]
    assert "pre-registration estimate ~ 6.0 h, $13.7" in out


def test_v5j5_is_skipped_without_three_completed_v5_ft_rows(tmp_path: Path) -> None:
    box = done_v5(tmp_path, {})
    box.scenario["ft-rows"] = 1
    box.write_scenario()
    rc, out = box.run("box_q_v5j5.sh")
    assert rc == 0 and "v5j5 SKIPPED" in out and box.training() == []


def test_the_post_seed_waiter_refuses_a_bad_ft_row_and_still_touches_done(tmp_path: Path) -> None:
    box = done_v5(tmp_path, {})
    p = subprocess.run(
        [BASH, str(box.root / "post-f" / "box_q_v5traj.sh"), "v5", "0", "not-a-row"],
        capture_output=True,
        text=True,
        env=box.env,
        timeout=60,
        check=False,
    )
    assert p.returncode == 3 and "is not a row id" in p.stdout
    assert (box.q / "v5traj-s0.done").exists() and box.calls() == []
    p = subprocess.run(
        [BASH, str(box.root / "post-f" / "box_q_v5traj.sh"), "v5nw", "3", EVAL_ROW],
        capture_output=True,
        text=True,
        env=box.env,
        timeout=60,
        check=False,
    )
    assert p.returncode == 2 and "usage" in p.stdout
