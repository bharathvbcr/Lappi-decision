"""Byte-level context encoding: the torch-free half of rung 0.

## Why a byte-level model exists at all

``AUDIT/prior-art-jev-nimble-2026-09-19.md`` records four public System-1 decision models.
One of them, ``cua-ai/cua-s1-forms``, is **706,048 parameters trained from scratch** on a
*byte-level* embedding and beats a hosted generalist on its own narrow task. That is three
orders of magnitude below this repo's 2B plan, and it arrives without a base model, without
a vocabulary remap, and without Hugging Face terms.

The reason it matters here is not size. It is that **a byte-level model dissolves this
repo's worst open gap family**:

``GAP-SPAN-HEAD-LINE-MAPPING-BPE-UNVERIFIED`` and ``GAP-S4-LINE-STARTS-COLLAPSE-UNDER-BPE``
both ask the same question — does "the token that starts line N" survive a BPE tokenizer
that merges across whitespace? S4 predicted its refusal would fire widely the first time a
real tokenizer ran, and the check that would settle it is against *decoded text*, which
nothing here can do yet.

In byte space the question does not arise. A line start **is** a byte offset. There is no
merge, no decode, and no mapping several conversions deep to get wrong. :func:`line_starts`
is exact and total, and its tests are exhaustive rather than probabilistic.

## What this module refuses to do

**Truncation is never silent.** :class:`EncodedContext` carries ``n_bytes_kept`` and
``n_bytes_total``, and a caller that reports the first without the second is presenting a
capped sample as complete coverage. A gold span that falls outside the kept window is a
:class:`SpanOutsideWindow` refusal, not a clamp to the last line — clamping would train the
head to point at whatever happened to survive truncation.

## Line termination

``\\n`` is the only terminator. ``\\r\\n`` therefore yields one line break, which is
correct; a lone ``\\r`` (classic Mac) yields none, which is a deliberate choice rather than
an oversight. Code contexts in this corpus are LF or CRLF, and treating a bare ``\\r`` as a
terminator would split ``\\r\\n`` into two lines and put a line start on the ``\\n`` byte.
:func:`line_starts` is tested on all four cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "BYTE_VOCAB_SIZE",
    "ID_NOUL",
    "ID_OPTION",
    "ID_PAD",
    "ID_QUESTION",
    "N_BYTE_VALUES",
    "ContextTooLarge",
    "EncodedContext",
    "SpanOutsideWindow",
    "encode_context",
    "line_of_offset",
    "line_starts",
]

#: A byte is a byte. No merges, no vocabulary, nothing to remap.
N_BYTE_VALUES: Final[int] = 256

# Reserved ids above the byte range. Kept contiguous and above 255 so that a raw byte
# value can never collide with a control id -- the failure that a "reserved token inside
# the vocabulary" scheme produces, where a context containing byte 0x02 becomes a
# structural separator and silently reframes the request.
ID_PAD: Final[int] = 256
ID_QUESTION: Final[int] = 257
ID_OPTION: Final[int] = 258
ID_NOUL: Final[int] = 259

BYTE_VOCAB_SIZE: Final[int] = 260


class ContextTooLarge(ValueError):
    """The context exceeds the window and the caller asked for no truncation."""


class SpanOutsideWindow(ValueError):
    """A gold span lies outside the kept window.

    Raised rather than clamped. Clamping to the last surviving line would teach the span
    head to point at an artefact of truncation, and the loss would fall while doing it.
    """


def line_starts(raw: bytes) -> tuple[int, ...]:
    """Byte offsets at which each line begins.

    Exact by construction: a line starts at offset 0 when the input is non-empty, and after
    every ``\\n`` that is not the final byte. No line is reported after a trailing newline,
    because there is no line there.

    >>> line_starts(b"")
    ()
    >>> line_starts(b"a")
    (0,)
    >>> line_starts(b"a\\n")
    (0,)
    >>> line_starts(b"a\\nb")
    (0, 2)
    >>> line_starts(b"a\\n\\nb")
    (0, 2, 3)
    >>> line_starts(b"a\\r\\nb")
    (0, 3)
    """
    if not raw:
        return ()
    out = [0]
    last = len(raw) - 1
    for i, byte in enumerate(raw):
        if byte == 0x0A and i != last:
            out.append(i + 1)
    return tuple(out)


def line_of_offset(starts: tuple[int, ...], offset: int) -> int:
    """Index of the line containing ``offset``.

    Linear scan from the end rather than a bisect: the caller is usually asking about a
    position near the end of the context, and a 40-line helper with an off-by-one in its
    bisect is a worse trade than a scan over a few hundred entries.
    """
    if not starts:
        raise ValueError("no lines: an offset cannot belong to a line that does not exist")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    for i in range(len(starts) - 1, -1, -1):
        if offset >= starts[i]:
            return i
    raise ValueError(f"offset {offset} precedes the first line start {starts[0]}")


@dataclass(frozen=True, slots=True)
class EncodedContext:
    """A context in byte space, with its line grid and what truncation cost.

    ``ids`` are byte values, so ``ids[i] == raw[i]`` for every kept byte: the identity that
    makes ``line_starts`` usable as model positions without a second mapping. The model
    prepends nothing; separators belong to the *request* encoder, not here, precisely so
    that this identity holds.
    """

    ids: tuple[int, ...]
    #: Offsets into ``ids`` where lines begin. Also offsets into the original bytes.
    starts: tuple[int, ...]
    n_bytes_kept: int
    n_bytes_total: int

    def __post_init__(self) -> None:
        if len(self.ids) != self.n_bytes_kept:
            raise ValueError(
                f"ids length {len(self.ids)} disagrees with n_bytes_kept {self.n_bytes_kept}"
            )
        if self.n_bytes_kept > self.n_bytes_total:
            raise ValueError(
                f"kept {self.n_bytes_kept} bytes of {self.n_bytes_total}, which is not possible"
            )
        if any(i >= N_BYTE_VALUES for i in self.ids):
            raise ValueError(
                "a context id is outside the byte range; control ids do not belong here"
            )
        for s in self.starts:
            if not 0 <= s < max(1, len(self.ids)):
                raise ValueError(f"line start {s} is outside the kept window of {len(self.ids)}")

    @property
    def truncated(self) -> bool:
        return self.n_bytes_kept < self.n_bytes_total

    @property
    def n_lines(self) -> int:
        return len(self.starts)

    def coverage(self) -> str:
        """``kept/total``. The pair, never the first number alone."""
        return f"{self.n_bytes_kept}/{self.n_bytes_total}"

    def check_span(self, gold_offset: int) -> int:
        """The line index a gold byte offset points at, or a refusal if it is out of window.

        **This does not validate that the offset is the right one.** Every offset in
        ``[0, n_bytes_kept)`` belongs to some line, so an offset that is semantically wrong
        still resolves to a line and this method reports it without complaint. Claiming
        otherwise would be the same error as
        ``GAP-S4-GOLD-ON-CANDIDATE-CANNOT-CATCH-LINE-SHIFT``, where a membership check was
        mistaken for a correctness check.

        What byte space changes is upstream of any check: in token space the gold arrived
        through ``gold line -> line start in the escaped region -> character offset -> token
        index``, and each hop could drift. Here the gold is a byte offset into the same bytes
        the grid is computed from, so **there is no mapping to drift**. The bug class is
        removed at the source rather than detected afterwards -- which is the stronger
        outcome, but a different claim.
        """
        if gold_offset < 0 or gold_offset >= self.n_bytes_kept:
            raise SpanOutsideWindow(
                f"gold offset {gold_offset} is outside the kept window "
                f"{self.coverage()}; clamping it would point the head at a truncation artefact"
            )
        return line_of_offset(self.starts, gold_offset)


def encode_context(
    raw: bytes, *, max_bytes: int, allow_truncation: bool = True
) -> EncodedContext:
    """Encode raw context bytes, keeping the **last** ``max_bytes`` when truncating.

    The tail is kept rather than the head because a decision about a change is about its
    end state, and a head-truncated diff loses the hunk while keeping the file preamble.
    Truncation happens on a line boundary where one exists inside the window, so the first
    surviving line is a whole line rather than a fragment starting mid-token.
    """
    if max_bytes < 1:
        raise ValueError(f"max_bytes must be at least 1, got {max_bytes}")
    total = len(raw)

    if total <= max_bytes:
        kept = raw
    elif not allow_truncation:
        raise ContextTooLarge(
            f"context is {total} bytes against a {max_bytes}-byte window and truncation "
            "was refused"
        )
    else:
        window_start = total - max_bytes
        window = raw[window_start:]
        if raw[window_start - 1] == 0x0A:
            # The window already opens a line. Advancing to the next newline here would throw
            # away a whole line that fits, which is up to `max_bytes` of budget spent on
            # nothing -- and for a context window that is the one resource there is.
            kept = window
        else:
            # Advance to the first line boundary inside the window so the leading line is not
            # a fragment. If the window holds no newline the whole of it is one long line and
            # there is nothing to align to.
            nl = window.find(b"\n")
            kept = window[nl + 1 :] if 0 <= nl < len(window) - 1 else window

    return EncodedContext(
        ids=tuple(kept),
        starts=line_starts(kept),
        n_bytes_kept=len(kept),
        n_bytes_total=total,
    )
