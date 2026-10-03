"""The needle suite sized in real tokens (v5 readings R1, R2, R8).

``build_suite`` used to grow each haystack until a 3-chars/token guess reached the target; F's
seed-0 suite measured a median of 8,473 real tokens against an 8,192 target. With a caller's
``measure`` -- ``tools/real_ft_run.py``'s render-and-encode of the span sequence -- every case
measures at most the target, one more filler hunk would not fit, and the needle is in place
when it is measured. What is pinned here:

* without ``measure`` the suite is the same bytes as before (digests from the pre-change
  ``needle.py``, the gate's 300 cases included);
* with one, the R1 rule on every case, every depth bucket still populated, the needle inside
  every measured candidate, and refusals for a measure or target that cannot work;
* ``prepare_needle`` -- the gate suite's and every ``--needle-control`` length's one builder --
  hands ``build_suite`` the measure that is its own encode, and refuses a case that encodes
  past the target.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_data.config import DataConfig  # noqa: E402
from qd_train.needle import (  # noqa: E402
    _FILLER,
    _FNS,
    _NAMES,
    DEPTH_BUCKETS,
    NeedleCase,
    build_suite,
    hunk_of_context_line,
)

#: The heuristic suite's digest under the pre-change needle.py (v5-build 01ab623), computed by
#: target/scratch/needle/digest.py's rule below: unchanged means unchanged bytes.
SMALL_DIGEST = "b26f36cb359ad5e314808788587b96aa175b24d89a954f672bf69137ea1a6f0c"
GATE_DIGEST = "e58d0271e06166d910d4c2c1e9ca357e832de6f82116ab6229b8f586701a4630"


def _digest(cases: list[NeedleCase]) -> str:
    body = "\n".join(
        f"{c.case_id}:{c.needle_index}:{c.n_hunks}:{c.needle_start_line}:{c.needle_end_line}:"
        f"{hashlib.sha256(c.context.encode()).hexdigest()}"
        for c in cases
    )
    return hashlib.sha256(body.encode()).hexdigest()


def test_without_a_measure_the_suite_is_the_same_bytes():
    """Characterization: the heuristic path is untouched, so no main-based scoring of a v4
    model sees a different suite (R8)."""
    assert _digest(build_suite(target_tokens=1024, cases_per_depth=2, seed=3)) == SMALL_DIGEST
    assert _digest(build_suite(
        target_tokens=8192, cases_per_depth=60, seed=DataConfig().seed
    )) == GATE_DIGEST


# --- the rule, under a stand-in measure -----------------------------------------------------

#: A stand-in for the rendered prompt around the context: a fixed overhead, then ~4 chars a
#: token -- not the Qwen tokenizer, which this repository's CPU tests do not load.
OVERHEAD = 61


def _measure(case: NeedleCase) -> int:
    return OVERHEAD + len(case.context) // 4


class _Recording:
    """``_measure``, keeping every candidate it was shown."""

    def __init__(self) -> None:
        self.seen: dict[int, dict[int, int]] = {}
        self.cases: list[NeedleCase] = []

    def __call__(self, case: NeedleCase) -> int:
        n = int(re.fullmatch(r"needle-[a-z]+-(\d{4})-[0-9a-f]{8}", case.case_id).group(1))  # type: ignore[union-attr]
        tokens = _measure(case)
        self.seen.setdefault(n, {})[case.n_hunks - 1] = tokens
        self.cases.append(case)
        return tokens


def _largest_filler() -> int:
    """An upper bound on what one filler hunk adds under ``_measure``: the longest template
    filled with the longest value of every placeholder."""
    longest = max(
        len(t.format(a=900, name=max(_NAMES, key=len), fn=max(_FNS, key=len), r="s"))
        for pool in _FILLER.values() for t in pool
    )
    return longest // 4 + 1


@pytest.mark.parametrize("target", [1024, 2048, 4096, 8192])
def test_every_case_is_at_most_the_target_and_the_next_hunk_would_not_fit(target):
    rec = _Recording()
    cases = build_suite(target_tokens=target, cases_per_depth=2, seed=7, measure=rec)
    assert len(cases) == 10
    for n, case in enumerate(cases):
        k = case.n_hunks - 1  # fillers; the needle is the other hunk
        tokens = rec.seen[n][k]
        assert tokens == _measure(case) <= target
        assert rec.seen[n][k + 1] > target, "the search never measured one hunk more"
        # Reading R1: each case lands in (target - largest filler hunk, target].
        assert tokens > target - _largest_filler()


def test_every_measured_candidate_already_holds_its_needle():
    rec = _Recording()
    build_suite(target_tokens=2048, cases_per_depth=1, seed=2, measure=rec)
    assert rec.cases
    for case in rec.cases:
        lines = case.context.split("\n")
        assert lines[case.needle_start_line - 1].startswith("@@")
        assert 0 <= case.needle_index < case.n_hunks
        # The rendered line of the needle's header is in the needle hunk (header lines first).
        from qd_data.defect_class import CONTEXT_HEADER_LINES

        line = CONTEXT_HEADER_LINES + case.needle_start_line - 1
        assert hunk_of_context_line(case, line) == case.needle_index


def test_depths_are_still_swept_and_the_suite_is_deterministic():
    a = build_suite(target_tokens=4096, cases_per_depth=3, seed=11, measure=_measure)
    b = build_suite(target_tokens=4096, cases_per_depth=3, seed=11, measure=_measure)
    assert _digest(a) == _digest(b)
    counts = {label: 0 for label in DEPTH_BUCKETS}
    for c in a:
        counts[c.depth_bucket] += 1
    assert all(v >= 2 for v in counts.values()), counts
    assert _digest(build_suite(target_tokens=4096, cases_per_depth=3, seed=12,
                               measure=_measure)) != _digest(a)


@pytest.mark.parametrize("bad", [0, -3, 1.5, True, None])
def test_a_measure_that_is_not_a_token_count_is_refused(bad):
    with pytest.raises(ValueError, match="not a positive token count"):
        build_suite(target_tokens=1024, cases_per_depth=1, measure=lambda case: bad)


def test_a_target_the_needle_alone_overruns_is_refused():
    with pytest.raises(ValueError, match="already measure"):
        build_suite(target_tokens=256, cases_per_depth=1,
                    measure=lambda case: 10_000 + len(case.context))


def test_a_measure_that_never_reaches_the_target_is_bounded():
    from qd_train.needle import MAX_FILLER_HUNKS

    with pytest.raises(ValueError, match=f"bounded to 1..{MAX_FILLER_HUNKS}"):
        build_suite(target_tokens=1024, cases_per_depth=1, measure=lambda case: 1)


# --- the call site ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="real_ft_run imports torch at module scope")
sys.path.insert(0, str(REPO / "tools"))
import real_ft_run as rft  # noqa: E402


class _Stop(Exception):
    pass


def test_prepare_needle_sizes_the_suite_with_its_own_encode(monkeypatch):
    """The gate suite and every --needle-control length come through prepare_needle, and it
    hands build_suite needle_measure: the same render and encode it then scores."""
    reader, tok, config = object(), object(), DataConfig()
    monkeypatch.setattr(rft, "_matching_tokenizer", lambda r, what: tok)
    handed: dict[str, object] = {}

    def spy(**kw):
        handed.update(kw)
        raise _Stop

    monkeypatch.setattr(rft, "build_suite", spy)
    with pytest.raises(_Stop):
        rft.prepare_needle(reader, config=config, enabled=True, target_tokens=2048)  # type: ignore[arg-type]
    assert handed["target_tokens"] == 2048 and callable(handed["measure"])

    case = build_suite(target_tokens=1024, cases_per_depth=1, seed=0)[0]
    calls: list[dict[str, object]] = []

    def encode(row, **kw):
        calls.append({"row": row, **kw})
        return None, None, SimpleNamespace(ids=np.zeros(1234, dtype=np.int32))

    monkeypatch.setattr(rft, "encode_slot_batch", encode)
    measure = handed["measure"]
    assert measure(case) == 1234  # type: ignore[operator]
    (call,) = calls
    assert call["slot_kind"] == rft.SLOT_SPAN and call["tok"] is tok
    assert call["reader"] is reader and call["config"] is config
    assert call["row"] == rft.needle_defect_row(case, config=config)


def test_a_case_that_encodes_past_its_target_is_refused(monkeypatch):
    case = build_suite(target_tokens=1024, cases_per_depth=1, seed=0)[0]
    monkeypatch.setattr(rft, "_matching_tokenizer", lambda r, what: object())
    monkeypatch.setattr(rft, "build_suite", lambda **kw: [case])
    monkeypatch.setattr(
        rft, "encode_slot_batch",
        lambda row, **kw: (None, None, SimpleNamespace(ids=np.zeros(1025, dtype=np.int32))),
    )
    with pytest.raises(SystemExit, match="encodes to 1025 tokens, over the 1024 target"):
        rft.prepare_needle(object(), config=DataConfig(), enabled=True, target_tokens=1024)  # type: ignore[arg-type]
