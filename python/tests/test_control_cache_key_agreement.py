"""The fit tool and the run must split the corpus identically, or the cache is useless.

`tools/fit_linear_control.py` exists so the linear control is fitted off the GPU and read
back in one second. That only works if the key it computes is the key the run looks up, and
the key covers the DOCUMENTS -- which are decided by the split share and the context width.

It diverged within a day of being written. `--val-share` defaulted to 0.20 in the fit tool
and 0.25 in the runner, which would have produced two different splits, a guaranteed cache
miss, and roughly an hour of CPU spent on a verdict nothing could read -- while the gate
reported `not_run` and the arm looked like it had simply not been given a control.

Nothing about that failure is loud. There is no exception, no warning, and both tools
report success. So the invariant is asserted here instead: the shared defaults have one
owner and the other tool imports it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

pytest.importorskip("torch")

import fit_linear_control  # noqa: E402
import rung0_real_run  # noqa: E402

#: Flags that both tools accept and that change which documents exist, or in what order.
#: `--seed` and `--max-iter` are covered by the key's scalar block rather than the split.
SHARED_SPLIT_FLAGS = (
    "--val-share",
    "--context-bytes",
    "--width",
    "--layers",
    "--heads",
    "--batch-size",
)


@pytest.mark.parametrize(
    "name", ["DEFAULT_VAL_SHARE", "DEFAULT_CONTEXT_BYTES", "DEFAULT_BATCH_SIZE"]
)
def test_the_fit_tool_imports_the_runner_s_constant_rather_than_its_own(name):
    assert getattr(fit_linear_control, name) == getattr(rung0_real_run, name)


def test_the_val_share_default_is_the_one_that_actually_diverged():
    """Pins the specific value, so a change to one side fails here rather than silently
    in a cache miss nobody reads."""
    assert rung0_real_run.DEFAULT_VAL_SHARE == 0.25
    assert fit_linear_control.DEFAULT_VAL_SHARE == 0.25


def _defaults_in(path: Path) -> dict[str, str]:
    """Map each shared flag to the text of its `default=` in that file's parser."""
    source = path.read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for flag in SHARED_SPLIT_FLAGS:
        match = re.search(
            rf'add_argument\(\s*"{re.escape(flag)}".*?default=([^,)\n]+)',
            source,
            re.DOTALL,
        )
        if match:
            found[flag] = match.group(1).strip()
    return found


@pytest.mark.parametrize("flag", SHARED_SPLIT_FLAGS)
def test_neither_tool_writes_a_bare_literal_default_for_a_shared_split_flag(flag):
    """A literal is how they diverged. A name cannot diverge without the import breaking.

    `ByteDeciderConfig().width` counts as a name: it is the one owner of the model shape,
    and both tools read it rather than restating 128.
    """
    for path in (
        REPO / "tools" / "fit_linear_control.py",
        REPO / "tools" / "rung0_real_run.py",
    ):
        defaults = _defaults_in(path)
        if flag not in defaults:
            continue
        value = defaults[flag]
        assert not re.fullmatch(r"-?\d+(\.\d+)?", value), (
            f"{path.name} gives {flag} the literal default {value}. Shared split defaults "
            "have one owner and the other tool imports it -- a literal on both sides is "
            "exactly how val-share became 0.20 here and 0.25 there."
        )


def test_the_shared_flags_really_are_accepted_by_both_tools():
    """Guards the guard: if a flag were renamed on one side, the checks above would find
    nothing to compare and would pass by vacuum."""
    for path in (
        REPO / "tools" / "fit_linear_control.py",
        REPO / "tools" / "rung0_real_run.py",
    ):
        source = path.read_text(encoding="utf-8")
        for flag in SHARED_SPLIT_FLAGS:
            assert f'"{flag}"' in source, f"{path.name} no longer accepts {flag}"
