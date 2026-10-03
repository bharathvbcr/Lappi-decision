"""Mutation pass over the v5 waiters (throwaway analysis; ships nothing).

Each mutant is a copy of campaign/post-f-queue under build/l-v5-queue/mut/<id>/ with ONE exact
replacement (it must occur exactly once). python/tests/test_v5_queue_scripts.py runs against the
copy through QD_V5_QUEUE_DIR; a mutant is caught when at least one test fails. The source tree is
never edited.

    uv run --no-project --python /Users/bharath/.venvs/ml/bin/python --with pytest \
      python AUDIT/v5-queue-2026-10-02/mutations.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "campaign" / "post-f-queue"
WORK = REPO / "build" / "l-v5-queue" / "mut"

MUTANTS = [
    (
        "M1",
        "v5_common.sh",
        "--min-lr 0 --batch-order seed)",
        "--min-lr 0)",
        "the recipe loses --batch-order seed",
    ),
    (
        "M2",
        "v5_common.sh",
        'v5_pause_action() { case "$1" in continue) echo run',
        'v5_pause_action() { case "$1" in continue|pause) echo run',
        "R9 pause runs seeds 1-2",
    ),
    (
        "M3",
        "v5_common.sh",
        "no_room:yes) echo run",
        "*:yes) echo run",
        "V5NW_HUMAN_YES overrides a refused room",
    ),
    (
        "M4",
        "v5_common.sh",
        "quiet) echo skip ;; *) echo notrun",
        "quiet) echo skip ;; *) echo skip",
        "a refused seeds34 reads as quiet",
    ),
    (
        "M5",
        "v5_common.sh",
        'if ! pin "$RULES_V5" "$RULES_V5_SHA256"; then',
        "if false; then",
        "v5_rule runs an unpinned binary",
    ),
    (
        "M6",
        "v5_common.sh",
        'if [ "$rc" = 0 ] && [ "$raw" = "$w" ]',
        'if [ "$raw" = "$w" ]',
        "v5_rule ignores the exit code",
    ),
    (
        "M7",
        "v5_common.sh",
        'if [ "$V5_WORD" != refused ] && [ ! -s "$V5_RULE_JSON" ]; then',
        "if false; then",
        "v5_rule accepts a word without its JSON",
    ),
    ("M8", "v5_common.sh", 'if [ "$over" = 1 ]; then', "if false; then", "no budget stop"),
    (
        "M9",
        "v5_common.sh",
        'if [ -s "$V5_OVER_BUDGET_YES" ]; then',
        'if [ -e "$V5_OVER_BUDGET_YES" ]; then',
        "an empty over-budget marker is a yes",
    ),
    (
        "M10",
        "v5_common.sh",
        "*heldout*|*held-out*|*held_out*|*containment*)",
        "*NEVER-MATCHES*)",
        "the split may name held-out data",
    ),
    (
        "M11",
        "v5_common.sh",
        'if [ -e "$V5_DATA/data/heldout" ]; then',
        "if false; then",
        "a held-out split on the box is not refused",
    ),
    (
        "M12",
        "v5_common.sh",
        'until [ -s "$path" ]; do',
        'until [ -e "$path" ]; do',
        "an empty human marker is a yes",
    ),
    (
        "M13",
        "v5_common.sh",
        'if [ "$(head -c 16 "$V5S34_WORD" 2>/dev/null | tr -d \'\\n\')" = fires ]; then',
        "if false; then",
        "seeds 3-4 never raise the approved total",
    ),
    (
        "M14",
        "v5_common.sh",
        'if not isinstance(d, dict) or "draft" in d:',
        "if not isinstance(d, dict):",
        "the DRAFT itself is read as binding",
    ),
    (
        "M15",
        "v5_common.sh",
        "  V5_RECIPE+=(--epoch --no-memorise --batch-tokens 35403)\n",
        "  V5_RECIPE+=(--epoch --no-memorise --batch-tokens 35403 --deterministic)\n",
        "the recipe gains a flag the DRAFT does not name",
    ),
    (
        "M16",
        "box_q_v5.sh",
        'if [ "$(v5_pause_action "$R9_WORD")" = run ]; then',
        "if true; then",
        "the waiter ignores R9",
    ),
    (
        "M17",
        "box_q_v5.sh",
        'v5_wait_marker "$V5_LAUNCH_YES" "the human\'s v5 launch yes" || { say "v5 NOT RUN"; exit '
        "3; }",
        "true",
        "v5 launches without the human's marker",
    ),
    (
        "M18",
        "box_q_v5.sh",
        "until [ -f $Q/j5pp.done ]; do sleep 60; done",
        "true",
        "v5 does not wait for the chain",
    ),
    ("M19", "box_q_v5.sh", "wait_queued j6a j6g", "wait_queued j6a", "v5 does not wait for j6g"),
    ("M27", "box_q_v5.sh", "wait_queued j6a j6g", "wait_queued j6g", "v5 does not wait for j6a"),
    (
        "M20",
        "box_q_v5nw.sh",
        'v5_train_seed v5nw "$SEED"',
        'v5_train_seed v5 "$SEED"',
        "the arm trains without its weight, into v5's ledger",
    ),
    (
        "M21",
        "box_q_v5j5.sh",
        'tools/real_ft_run.py "${V5_SPLIT[@]}" "${V5_RECIPE[@]}" \\\n',
        'tools/real_ft_run.py "${V5_SPLIT[@]}" \\\n',
        "J5' drops v5's recipe",
    ),
    (
        "M22",
        "box_q_v5s34.sh",
        '--f-ledger "$V5_LEDGER"',
        '--f-ledger "$F_LEDGER"',
        "seeds34 reads F's ledger",
    ),
    (
        "M23",
        "box_q_v5traj.sh",
        'ORDER="$FINAL $(echo "$STEPS" | grep -vx "$FINAL" | tr \'\\n\' \' \')"',
        "ORDER=$STEPS",
        "the final step is not scored first",
    ),
    (
        "M24",
        "box_q_v5traj.sh",
        'if [ "$LEFT" -lt "$V5_TRAJ_MIN_LEFT_S" ]; then',
        "if false; then",
        "the trajectory ignores its cap",
    ),
    (
        "M25",
        "box_q_v5nw.sh",
        'if [ -s "$V5NW_HUMAN_YES" ]; then YES=yes; fi',
        'if [ -e "$V5NW_HUMAN_YES" ]; then YES=yes; fi',
        "an empty V5NW_HUMAN_YES counts as the yes",
    ),
    (
        "M26",
        "box_q_v5j5.sh",
        "if ! v5_rules_call ft-rows",
        "if false && v5_rules_call ft-rows",
        "J5' runs without three completed v5 ft rows",
    ),
]


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    caught = 0
    lines = []
    for mid, name, old, new, what in MUTANTS:
        dst = WORK / mid / "campaign" / "post-f-queue"
        shutil.copytree(SRC, dst)
        path = dst / name
        text = path.read_text(encoding="utf-8")
        n = text.count(old)
        if n != 1:
            lines.append(f"{mid} INVALID: {old!r} occurs {n} times in {name}")
            continue
        path.write_text(text.replace(old, new), encoding="utf-8")
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "python/tests/test_v5_queue_scripts.py",
                "-q",
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
                "--tb=no",
                "-x",
            ],
            cwd=REPO,
            env={**os.environ, "QD_V5_QUEUE_DIR": str(dst)},
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        tail = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        word = "CAUGHT" if r.returncode != 0 else "SURVIVED"
        caught += r.returncode != 0
        lines.append(f"{mid} {word}: {what} ({name}) -- {tail}")
        print(lines[-1], flush=True)
    print(f"{caught} of {len(MUTANTS)} caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
