"""Does the portability verifier tell an architecture difference from a calendar one?

``tools/verify_lock_portability.py`` answers one question: *will the pinned stack install
on the machine we are about to rent?* On 2026-09-20 a GH200 was rented, making that
question live for aarch64 for the first time.

Its first version resolved ``stack/*.in`` **once**, for the target, and called any
difference from the committed lock a portability failure. Run for aarch64 it reported::

    teacher.lock: 196 committed, 196 resolved for aarch64
        version moved  : tokenspeed-triton: 3.8.10.post20260906 -> 3.8.10.post20260920
    RESULT: NOT established — see above

That is wrong, and the control proves it: re-resolving for **x86_64** — the platform the
lock was committed at, compared against itself — moves ``tokenspeed-triton`` identically.
Upstream cut a new post-release on 2026-09-20. Nothing about that package differs by
architecture.

One comparison was carrying two quantities under one name: *upstream moved since the lock
was committed*, and *this package differs between machines*. Only the second is a
portability question, and a tool that conflates them fails a portable lock every time any
of 196 pinned packages cuts a release.

So the verifier now resolves twice — once at the committed platform, once at the target,
both **now** — and reports:

* ``committed -> control``: time drift. Real, worth seeing, not an architecture answer.
* ``control -> target``: architecture drift, with time held fixed. This alone decides.

Both directions are pinned below, because a "fix" that classified everything as time drift
would make the tool pass always and measure nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from verify_lock_portability import Drift, classify, diff

#: The real 2026-09-20 measurement, as three pin sets over the packages that moved.
#: ``teacher.lock`` pins 196 packages; the three below are the ones that carry the point.
COMMITTED = {"torch": "2.10.0+cu128", "tokenspeed-triton": "3.8.10.post20260906"}
CONTROL_X86 = {"torch": "2.10.0+cu128", "tokenspeed-triton": "3.8.10.post20260920"}
TARGET_ARM = {"torch": "2.10.0+cu128", "tokenspeed-triton": "3.8.10.post20260920"}


def test_a_package_that_moves_on_both_platforms_is_time_drift_not_an_arch_failure() -> None:
    """The measured case. Pre-fix this was reported as "NOT established" for aarch64."""
    time_drift, arch_drift = classify(COMMITTED, CONTROL_X86, TARGET_ARM)

    assert arch_drift == [], (
        "tokenspeed-triton moves identically at the committed platform, so it cannot be "
        f"evidence about architecture; got {[d.render() for d in arch_drift]}"
    )
    assert [d.package for d in time_drift] == ["tokenspeed-triton"]
    assert time_drift[0].before == "3.8.10.post20260906"
    assert time_drift[0].after == "3.8.10.post20260920"

    # What the pre-fix tool did: one comparison, committed vs target. It sees the move and
    # has no way to know the control saw it too. This is the failing-first property stated
    # as an executable fact rather than a claim in a commit message.
    pre_fix_single_comparison = diff(COMMITTED, TARGET_ARM)
    assert [d.package for d in pre_fix_single_comparison] == ["tokenspeed-triton"], (
        "the pre-fix comparison must still see the move -- if it did not, this test would "
        "be passing for the wrong reason and would not pin the fix at all"
    )


def test_a_package_that_differs_only_on_the_target_is_a_real_portability_failure() -> None:
    """The fix must not make everything pass. A genuine arch difference still fails."""
    committed = {"torch": "2.10.0+cu128", "causal-conv1d": "1.7.0"}
    control = {"torch": "2.10.0+cu128", "causal-conv1d": "1.7.0"}
    target = {"torch": "2.10.0+cu128"}  # no aarch64 resolution for causal-conv1d

    time_drift, arch_drift = classify(committed, control, target)

    assert time_drift == [], "nothing moved at the committed platform"
    assert [d.package for d in arch_drift] == ["causal-conv1d"]
    assert arch_drift[0].before == "1.7.0"
    assert arch_drift[0].after is None
    assert arch_drift[0].render() == "causal-conv1d: 1.7.0 -> ABSENT"


def test_both_drifts_are_reported_when_both_are_present() -> None:
    """They are independent axes: one does not mask or subsume the other."""
    committed = {"a": "1", "b": "1"}
    control = {"a": "2", "b": "1"}  # `a` moved with time
    target = {"a": "2"}  # `b` is absent on the target machine

    time_drift, arch_drift = classify(committed, control, target)

    assert [d.package for d in time_drift] == ["a"]
    assert [d.package for d in arch_drift] == ["b"]


def test_an_identical_resolution_reports_no_drift_of_either_kind() -> None:
    pins = {"torch": "2.10.0+cu128", "triton": "3.5.0"}
    assert classify(pins, dict(pins), dict(pins)) == ([], [])


def test_a_package_appearing_only_on_the_target_is_named_rather_than_ignored() -> None:
    """An addition is a difference. Reporting only removals would under-report by half."""
    time_drift, arch_drift = classify({"a": "1"}, {"a": "1"}, {"a": "1", "extra": "9"})
    assert time_drift == []
    assert arch_drift == [Drift("extra", None, "9")]
    assert arch_drift[0].render() == "extra: ABSENT -> 9"
