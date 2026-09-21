"""``tools/rung0_linear_control.py``: the control arm that tells two failures apart.

Rung 0's choice head converges to the class prior. That single fact has two causes with
opposite fixes -- signal present and not extracted (architecture) versus signal absent
(corpus) -- and the model under test produces the same number either way. This control is
what separates them, so the properties that make it a valid control are what is tested:

* it reads the model's OWN input, not a re-read of the source file;
* it is refused when it did not converge, because an untrained control cannot be beaten;
* it passes only ABOVE the prior.

Torch-free apart from the tool's import chain, which pulls torch through
``rung0_real_run``; gated accordingly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import rung0_linear_control as control  # noqa: E402

from qd_train.mutate_adapter import MUTATION_CLASSES  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402


class _Context:
    def __init__(self, ids: tuple[int, ...]) -> None:
        self.ids = ids


class _Decision:
    def __init__(self, text: bytes, gold: int) -> None:
        self.context = _Context(tuple(text))
        self.gold_option = gold


def test_the_control_reads_the_models_own_bytes() -> None:
    """Not a re-read of the source file. ``EncodedContext.ids`` are byte values and
    ``ids[i] == raw[i]`` for every kept byte, so this is the same window, the same
    truncation. A control that saw more of the file would be answering a different task and
    any gap between it and the model would be unreadable."""
    docs, labels = control.context_texts([_Decision(b"def f():\n    pass\n", 0)])
    assert docs == ["def f():\n    pass\n"]
    assert labels == [MUTATION_CLASSES[0]]


def test_a_window_split_multibyte_character_does_not_crash_the_control() -> None:
    """The context is truncated at a byte boundary, so the first character can be a
    fragment. The model sees those same bytes; the control must not refuse them."""
    # The first two bytes of a 3-byte UTF-8 character, then valid ASCII.
    docs, _ = control.context_texts([_Decision(b"\xe2\x82" + b"ok", 1)])
    assert docs[0].endswith("ok")


def test_the_control_passes_only_above_the_prior() -> None:
    at_prior = control._control_gate(0.521, 0.521, n=303)
    assert isinstance(at_prior, Ran)
    assert not at_prior.passed

    above = control._control_gate(0.574, 0.521, n=303)
    assert isinstance(above, Ran)
    assert above.passed
    assert "+5.3%" in above.detail


def test_no_held_out_rows_reads_as_not_run() -> None:
    state = control._control_gate(0.0, 0.5, n=0)
    assert isinstance(state, NotRun)
    assert not hasattr(state, "passed")


def test_the_iteration_default_is_above_the_library_default() -> None:
    """The library's 500 iterations leave this fit at a gradient norm of 4.883e-04 against
    a 1e-04 tolerance, and ``convergence()`` correctly refuses to be scored. The budget is
    what moves; the tolerance is what makes the control worth beating."""
    from qd_train.baseline import LinearBaseline

    assert LinearBaseline().max_iter < control.DEFAULT_MAX_ITER
    assert control.DEFAULT_MAX_ITER <= control.MAX_ITER_CEILING
    # The tolerance is untouched, which is the half that must not drift.
    assert LinearBaseline().tol == LinearBaseline(max_iter=control.DEFAULT_MAX_ITER).tol


def test_an_out_of_range_iteration_budget_is_refused() -> None:
    """Unbounded would never return on a control that cannot converge, and the refusal is
    the only thing standing between that and a hung run."""
    with pytest.raises(SystemExit, match="max-iter"):
        control.main([
            "--examples", str(REPO / "README.md"),
            "--manifest-in", str(REPO / "README.md"),
            "--max-iter", "0",
        ])
