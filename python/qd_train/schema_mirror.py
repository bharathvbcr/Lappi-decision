"""The Python mirror of ``crates/qd-runtime/src/schema.rs``. One owner, no dependencies.

The Rust crate is not importable from Python, so these constants are duplicated. What
makes duplication survivable is that there is exactly **one** Python copy and one test
pinning it to the Rust source: ``python/tests/test_schema_mirror.py`` reads ``schema.rs``
and fails if any of them drift.

## Why this module exists at all

The constants used to live in :mod:`qd_train.heads`, which imports torch at module scope.
That made a pure integer unreachable from any module that must run without torch, and two
modules worked around it in two different ways:

* :mod:`qd_train.calibration_fit` restated all six with its own Rust-pin test — a second
  copy, correct but doubled.
* :mod:`qd_train.byte_batch` imported ``RESERVED_NOUL_ROWS`` from ``heads`` and therefore
  failed to import at all wherever torch is absent, which is exactly where its index
  arithmetic is supposed to run.

``docs/training-contract.md`` draws that line — *format, ordering, bucketing, coverage
policy ... are torch-free and run in CI* — so the contract constants belong below it, not
above it. :mod:`qd_train.heads` and :mod:`qd_train.calibration_fit` now re-export from
here, which keeps every existing import site working.

This module imports nothing but :mod:`typing`. That is the point: anything may depend on
it, so it may depend on nothing.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "CHOICE",
    "MAX_OPTIONS",
    "MIN_OPTIONS",
    "RESERVED_NOUL_ROWS",
    "SCORE",
    "SPAN",
]

#: ``pub const MAX_OPTIONS: usize = 16;`` — schema.rs:19.
MAX_OPTIONS: Final[int] = 16

#: ``pub const MIN_OPTIONS: usize = 2;`` — schema.rs:21.
MIN_OPTIONS: Final[int] = 2

#: ``pub const RESERVED_NOUL_ROWS: usize = 1;`` — schema.rs:110.
#:
#: Every decode head reserves this many rows for the abstention, **last**:
#: ``schema.rs`` sizes a choice slot at ``options.len() + RESERVED_NOUL_ROWS`` and
#: ``answer.rs`` reads the abstention back as ``plan.rows - RESERVED_NOUL_ROWS``. Writing
#: the arithmetic out rather than hard-coding the last index is what makes both sides move
#: together if this ever becomes 2.
RESERVED_NOUL_ROWS: Final[int] = 1

#: ``SlotKind::as_str`` — schema.rs:147-149. These strings are wire-visible: they key the
#: calibration table and appear in every serialised schema.
CHOICE: Final[str] = "choice"
SCORE: Final[str] = "score"
SPAN: Final[str] = "span"
