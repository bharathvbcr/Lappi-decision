"""The J6(a) launcher hands BOTH its processes ``--split-cache`` (ft-505105265aa).

The train process and the separate ``--needle-control`` process share ``J6A_SPLIT``, so the
flag in that array reaches both; the control rebuilt the whole split (~200 s, GPU idle) because
it was not there. A dry check, not a run: the array is evaluated by bash with stand-in values
and the control invocation is read from the script.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "campaign/post-f-queue/box_q_j6a.sh"


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_the_data_argv_carries_the_split_cache_and_the_dir_is_made_owner_only_first():
    text = _text()
    start = text.index("J6A_SPLIT=(")
    block = text[start : text.index(")\n", start) + 1]
    probe = (
        f'J6A_DATA=/d BACKBONE=/b REC=/r J6A_SPLIT_CACHE=/cache\n{block}\n'
        'printf "%s\\n" "${J6A_SPLIT[@]}"'
    )
    argv = subprocess.run(
        ["bash", "-c", probe], capture_output=True, text=True, check=True
    ).stdout.split("\n")
    assert argv[argv.index("--split-cache") + 1] == "/cache"
    assert text.index('mkdir -p -m 700 "$J6A_SPLIT_CACHE"') < start


def test_the_needle_control_invocation_is_built_from_that_same_array():
    text = _text()
    call = text[text.index("--needle-control 1024,2048,4096") - 200 :]
    call = call[: call.index("needle-control.log")]
    assert '"${J6A_SPLIT[@]}"' in call
    # The training invocation takes it the same way.
    assert 'J6A_ARGV=("${J6A_SPLIT[@]}"' in text
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
