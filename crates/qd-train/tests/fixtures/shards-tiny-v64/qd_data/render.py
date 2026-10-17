"""The prompt renderer. One implementation, used by training and by serving.

``docs/hardening.md`` section 3: *the context is agent-authored diffs; treat every
byte of it as hostile input, because the model will be served diffs written by
other models.*

The whole defence rests on two invariants, and it is worth stating them plainly
because every attack in the hardening table reduces to one or the other:

    **1. No untrusted byte can produce the two-character sequence ``<|`` in the
    rendered prompt.**

    **2. No untrusted byte can produce an apparent line break that ``\\n`` counting
    does not see.**

Every structural marker in this format has the shape ``<|qd_...|>``, and every
special token in the Qwen tokenizer has the shape ``<|...|>``. So a single escape
rule -- break ``<|`` into ``<\\|`` -- simultaneously prevents:

* forging a question delimiter and terminating the context early,
* forging an option line and creating a 17th apparent option,
* smuggling ``<|im_start|>`` and friends in as control tokens rather than text.

It is one rule rather than a blocklist of known token names on purpose: a
blocklist goes stale the moment the tokenizer gains a token, and the failure is
silent. :func:`escape_block` and :func:`escape_inline` are property-tested over
arbitrary text for exactly this invariant.

The second invariant is the one the first version of this module got wrong, and the
failure was quiet in exactly the way the first was loud. ``str.splitlines()`` breaks
on eight characters besides ``\\n`` -- U+000B, U+000C, U+001C, U+001D, U+001E,
U+0085, U+2028, U+2029 -- and a bare ``\\r`` is a ninth. Passing them through meant:

* an **option** containing one rendered as two apparent option lines, creating the
  "17th apparent option" ``docs/hardening.md`` section 3 forbids, while
  ``prompt.count("\\n")`` reported the format intact; and
* a **context** containing one had two disagreeing line counts, so a span label
  derived one way pointed at a different line than the same label derived the other
  -- section 1's systematic off-by-one, arriving through the data rather than
  through the mutator.

Both are closed by escaping that set as ``\\uXXXX`` in *both* modes. That does not
change the ``\\n`` count, so the block-mode line invariant below is untouched.

Two further invariants, both load-bearing and both property-tested:

* ``escape_block`` preserves the newline count exactly. Span slots answer with
  line numbers into the context, so an escape that added or removed a line would
  systematically move every span label -- the same defect ``docs/hardening.md``
  section 1 calls out for the mutation engine.
* ``escape_inline`` emits no newline at all. That is what makes "one option, one
  line" true no matter what the option text contains.

**A third attack is handled here but not by escaping.** Unicode bidi overrides and
zero-width characters make text *render* differently from how it tokenises -- the
Trojan Source class -- without forging a line break or a ``<|``. Both invariants
above hold on such a row and every structural test passes, because the control it
defeats is a human reading the rendered prompt. This module owns the alphabet
(:data:`INVISIBLE_FORMAT_CHARS`, :func:`first_invisible_format_char`) and
``qd_data.mixture`` owns the policy: the row is refused at the corpus boundary and
counted. It is deliberately **not** added to :data:`HEX_ESCAPED`, which is a
hand-transcribed contract with ``crates/qd-runtime/src/render.rs`` -- see the note
beside :data:`INVISIBLE_FORMAT_CHARS` for why a one-sided widening would be a worse
bug than the one it fixes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Final

from .errors import (
    NOUL,
    ContextTooLargeRefusal,
    EmptyContextRefusal,
    OptionTooLongRefusal,
    RenderedPromptTooLargeRefusal,
)
from .schema import (
    NOUL_LETTER,
    OPTION_LETTERS,
    ChoiceSlot,
    Request,
    ScoreSlot,
    Slot,
    SpanSlot,
)

__all__ = [
    "DEFAULT_CAPS",
    "ESCAPE_WORST_CASE_GROWTH",
    "HEX_ESCAPED",
    "INVISIBLE_FORMAT_CHARS",
    "INVISIBLE_FORMAT_RANGES",
    "MARKERS",
    "PROMPT_FORMAT",
    "DeterministicRng",
    "RenderCaps",
    "RenderedPrompt",
    "SlotRender",
    "escape_block",
    "escape_inline",
    "first_invisible_format_char",
    "render",
    "render_for_serving",
    "second_pass_permutation",
    "shuffle_options",
    "unescape",
]

# -- structural markers ------------------------------------------------------
# Every one is of the form <|qd_...|>. See the module docstring: this shape is
# what lets a single escape rule protect the format and the tokenizer at once.

M_BEGIN: Final[str] = "<|qd_begin|>"
M_FORMAT: Final[str] = "<|qd_prompt_format|>"
M_VERSION: Final[str] = "<|qd_schema_version|>"
M_TASK: Final[str] = "<|qd_task|>"
M_ROUTE: Final[str] = "<|qd_route|>"
M_QUESTION: Final[str] = "<|qd_question|>"
M_CTX_BEGIN: Final[str] = "<|qd_context_begin|>"
M_CTX_END: Final[str] = "<|qd_context_end|>"
M_SLOT: Final[str] = "<|qd_slot|>"
M_TYPE: Final[str] = "<|qd_type|>"
M_OPT_BEGIN: Final[str] = "<|qd_options_begin|>"
M_OPT_END: Final[str] = "<|qd_options_end|>"
M_ANSWER: Final[str] = "<|qd_answer|>"

MARKERS: Final[tuple[str, ...]] = (
    M_BEGIN, M_FORMAT, M_VERSION, M_TASK, M_ROUTE, M_QUESTION, M_CTX_BEGIN, M_CTX_END,
    M_SLOT, M_TYPE, M_OPT_BEGIN, M_OPT_END, M_ANSWER,
)

#: The prompt layout this renderer writes, stated in every prompt on the ``M_FORMAT`` line.
#:
#: Format 2 (v5, ``campaign/v5-preregistered.DRAFT.json`` ``format.*``) puts the question
#: line **after** the context: begin, format, schema version, task, route, the context block,
#: then the question, then the slot suffix. Format 1 (v4) had no format line and rendered the
#: question between the route and the context. Only the question moved, so N questions over
#: one context share every byte through ``<|qd_context_end|>``.
#:
#: It is not the wire ``schema_version``, which stays 1: callers send the same requests. It
#: is what a model was trained on, so everything that pairs a model with prompts refuses a
#: mismatch: ``qd_train.artifacts.ShardHeader.prompt_format`` (the shard reader), the
#: release manifest's ``expected_identity.prompt_format`` (``crates/qd-runtime/src/release.rs``)
#: and the trainer's recipe. ``crates/qd-runtime/src/render.rs`` carries the same number,
#: pinned by ``python/tests/test_prompt_format_v5.py``.
PROMPT_FORMAT: Final[int] = 2

#: The escape sentinel. Chosen so that escaping is a pure ASCII transform that
#: never touches a multi-byte sequence, and so ``unescape`` is a single scan.
_ESC: Final[str] = "\\"

#: Characters that can produce an apparent line break, or terminate a C string,
#: without being ``\n``.
#:
#: This set is the second rule, alongside ``<|``, and it was added because the
#: renderer failed a test written from the hardening table. ``str.splitlines()``
#: breaks on U+000B, U+000C, U+001C, U+001D, U+001E, U+0085, U+2028 and U+2029 as
#: well as on ``\n`` and ``\r``, and so do several tokenizer preprocessors. An option
#: containing any of them rendered as **two** apparent option lines -- exactly the
#: "17th apparent option" ``docs/hardening.md`` section 3 forbids -- while
#: ``prompt.count("\n")`` said the format was intact.
#:
#: The same characters in the *context* are the span-label version of the same bug:
#: ``splitlines()`` and ``split("\n")`` disagree on how many lines the context has,
#: so two derivations of a line number disagree, which is ``docs/hardening.md``
#: section 1's off-by-one arriving through the data instead of through the mutator.
#: Escaping them in block mode does not change the ``\n`` count, so the newline
#: invariant is untouched and line counting becomes unambiguous.
#:
#: U+0000 is here for a different reason: it is not a line break, but the context
#: crosses the FFI boundary and a NUL truncates any C string that later carries it.
#:
#: **This set is a two-language contract and must not be widened from one side.**
#: ``crates/qd-runtime/src/render.rs`` carries a hand-transcribed copy, and the
#: serving prompt is produced by that copy. Escaping a codepoint here that Rust does
#: not escape means the model is served a prompt shape it never saw in training,
#: which is the drift this whole module exists to prevent -- and nothing in the Rust
#: golden-bytes test would catch it, because that golden's sample context contains
#: none of the candidate characters.
#: ``tests/test_render.py::test_the_escape_alphabet_is_identical_to_the_rust_transcription``
#: reads ``render.rs`` and pins the two sets together, so the next widening has to
#: land in both lanes or fail loudly in this one.
HEX_ESCAPED: Final[frozenset[str]] = frozenset(
    chr(c) for c in (0x00, 0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029)
)

# The escape emits exactly four hex digits (``\uXXXX``) and :func:`unescape` reads
# exactly four back, so a codepoint above the BMP would escape to five digits and the
# unescaper would silently leave the fifth as literal text. Nothing above U+FFFF may
# enter this set without the escape format changing first.
assert all(ord(c) < 0x10000 for c in HEX_ESCAPED), (
    "HEX_ESCAPED must stay inside the BMP: the \\uXXXX escape carries four hex digits "
    "and unescape reads exactly four back"
)

#: Worst-case bytes-out per byte-in for :func:`escape_block` / :func:`escape_inline`.
#: The maximum is six, reached by a one-byte character that takes a six-byte ``\\uXXXX``
#: escape -- U+0000, U+000B, U+000C, U+001C, U+001D, U+001E. Backslashes and newlines
#: are only 2x; the multi-byte separators U+0085 and U+2028 are 3x and 2x. This is a
#: bound on the *bytes*, not on the characters, and ``RenderCaps`` uses it so a legal
#: context can never be refused by arithmetic alone.
ESCAPE_WORST_CASE_GROWTH: Final[int] = 6


# -- invisible format characters: the Trojan Source class ---------------------
#
# ``GAP-DATA-RENDER-BIDI-UNESCAPED`` asked whether the renderer should *escape* the
# bidi overrides. The answer is that escaping is the right protection and the wrong
# seam, for two reasons, and the second one is the decisive one:
#
# 1. The previous record rejected escaping because it "would corrupt legitimate
#    right-to-left content". That reason is wrong: escaping is lossless --
#    :func:`unescape` is its exact inverse -- so RTL content survives as ``‮``
#    and is recovered byte-for-byte. The real cost is that a reviewer sees the
#    escape rather than the glyph, which is the entire point.
# 2. :data:`HEX_ESCAPED` is transcribed into ``crates/qd-runtime/src/render.rs`` by
#    hand and pinned by no test until now. Widening it here alone would desynchronise
#    the training renderer from the serving renderer **silently**. A one-sided fix for
#    a display-integrity bug that introduces a train/serve prompt drift is a bad trade.
#
# So the protection lands at the corpus boundary instead, where this lane owns both
# sides of the decision: ``qd_data.mixture`` refuses a row whose untrusted text
# carries one of these, with a counted ``reason_code``, and the escape alphabet is
# left byte-identical to Rust's. No row in the corpus can then display to a reviewer
# as something other than what it tokenises to.
#
# **The class, not the eight named characters.** The gap named U+202A-U+202E and
# U+2066-U+2069. Those eight are not a class: the same reordering is reachable with
# the implicit marks U+200E/U+200F/U+061C, and the same "displays identically,
# tokenises differently" trick is reachable with the zero-width characters
# (U+200B-U+200D, U+2060, U+FEFF) and with U+00AD. The closed class that contains all
# of them is Unicode general category **Cf (Format)** -- characters that contribute no
# glyph and exist only to change how their neighbours are displayed or joined.
#
# Enumerated and frozen rather than computed from ``unicodedata`` at import, because
# ``unicodedata`` tracks the interpreter's Unicode version: deriving the set live
# would make *which rows are refused* -- and therefore ``data_snapshot_hash`` -- a
# property of the Python build. ``test_render.py`` asserts the frozen table still
# equals this interpreter's answer, so a Unicode upgrade that adds a Cf character
# fails loudly and is decided by a human rather than absorbed.
#
# Restricted to the BMP for the same reason :data:`HEX_ESCAPED` is: if these are ever
# promoted into the escape alphabet, the ``\uXXXX`` form has four hex digits.
INVISIBLE_FORMAT_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (0x00AD, 0x00AD),  # SOFT HYPHEN
    (0x0600, 0x0605),  # ARABIC NUMBER SIGN .. ARABIC NUMBER MARK ABOVE
    (0x061C, 0x061C),  # ARABIC LETTER MARK -- the third implicit bidi mark
    (0x06DD, 0x06DD),  # ARABIC END OF AYAH
    (0x070F, 0x070F),  # SYRIAC ABBREVIATION MARK
    (0x0890, 0x0891),  # ARABIC POUND MARK ABOVE .. ARABIC PIASTRE MARK ABOVE
    (0x08E2, 0x08E2),  # ARABIC DISPUTED END OF AYAH
    (0x180E, 0x180E),  # MONGOLIAN VOWEL SEPARATOR
    (0x200B, 0x200F),  # ZWSP, ZWNJ, ZWJ, LRM, RLM
    (0x202A, 0x202E),  # LRE, RLE, PDF, LRO, RLO -- the Trojan Source overrides
    (0x2060, 0x2064),  # WORD JOINER .. INVISIBLE PLUS
    (0x2066, 0x206F),  # LRI, RLI, FSI, PDI .. NOMINAL DIGIT SHAPES
    (0xFEFF, 0xFEFF),  # ZERO WIDTH NO-BREAK SPACE (BOM)
    (0xFFF9, 0xFFFB),  # INTERLINEAR ANNOTATION ANCHOR .. TERMINATOR
)

INVISIBLE_FORMAT_CHARS: Final[frozenset[str]] = frozenset(
    chr(c) for lo, hi in INVISIBLE_FORMAT_RANGES for c in range(lo, hi + 1)
)

assert all(ord(c) < 0x10000 for c in INVISIBLE_FORMAT_CHARS), "BMP only; see above"
assert not (INVISIBLE_FORMAT_CHARS & HEX_ESCAPED), (
    "the two alphabets must stay disjoint: a character in both would be escaped by "
    "the renderer and refused by the corpus builder, which is two policies for one "
    "codepoint"
)


def first_invisible_format_char(text: str) -> str | None:
    """The first Unicode format character in ``text``, or ``None``.

    Returns the character rather than a bool so the caller's refusal can name the
    codepoint. A refusal that says only "invisible character present" sends a human
    hunting through a diff for something that is, by construction, invisible.
    """
    for ch in text:
        if ch in INVISIBLE_FORMAT_CHARS:
            return ch
    return None


# -- escaping ----------------------------------------------------------------


def _escape(text: str, *, inline: bool) -> str:
    """Single left-to-right pass. Sequential ``str.replace`` calls would be
    ambiguous to invert; this is written as a scanner so :func:`unescape` is its
    exact inverse."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == _ESC:
            out.append("\\\\")
        elif c == "<" and i + 1 < n and text[i + 1] == "|":
            # The one rule the whole defence rests on.
            out.append("<\\|")
            i += 2
            continue
        elif c == "\n":
            # Block mode keeps real newlines: span labels are line numbers into
            # this text, so the line count must survive escaping untouched.
            out.append("\\n" if inline else "\n")
        elif c == "\r":
            # Escaped in *both* modes. A bare CR is a line break to splitlines() but
            # not to split("\n"), so leaving it raw in block mode would leave the two
            # line counts disagreeing on any CR-only or mixed-ending file -- the very
            # corpus docs/hardening.md section 1 lists as an attack.
            out.append("\\r")
        elif c == "\t":
            out.append("\\t" if inline else "\t")
        elif c in HEX_ESCAPED:
            out.append(f"\\u{ord(c):04x}")
        else:
            out.append(c)
        i += 1
    return "".join(out)


def escape_block(text: str) -> str:
    """Escape untrusted multi-line text (the context).

    Guarantees: the result contains no ``<|``; it has exactly as many ``\\n`` as the
    input; and ``split("\\n")`` and ``splitlines()`` agree on it, so a line number
    derived either way names the same line.
    """
    return _escape(text, inline=False)


def escape_inline(text: str) -> str:
    """Escape untrusted text that must occupy exactly one line (options, question).

    Guarantees: the result contains no ``<|`` and no character that any line split
    treats as a break -- not ``\\n``, and not the eight others ``splitlines()``
    honours. So it cannot become a second option line under any reading.
    """
    return _escape(text, inline=True)


def unescape(text: str) -> str:
    """Exact inverse of :func:`escape_block` and :func:`escape_inline`.

    Exists so a test can assert the context region round-trips: if
    ``unescape(region) == original`` for adversarial inputs, then nothing in the
    context was able to terminate it early or change its meaning, which is a
    stronger statement than checking for any particular attack string.

    An unknown escape raises rather than being passed through: a renderer and an
    unescaper that disagree about the alphabet is exactly the drift this module
    exists to prevent.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == _ESC:
            if i + 1 >= n:
                raise ValueError("dangling escape at end of input")
            nxt = text[i + 1]
            if nxt == "u":
                digits = text[i + 2 : i + 6]
                if len(digits) != 4 or any(d not in "0123456789abcdef" for d in digits):
                    raise ValueError(
                        f"malformed \\u escape {text[i : i + 6]!r} at offset {i}: "
                        "exactly four lowercase hex digits are required"
                    )
                ch = chr(int(digits, 16))
                if ch not in HEX_ESCAPED:
                    # The renderer only ever emits \\u for the characters in that set.
                    # Accepting any other codepoint would make the unescaper a
                    # superset of the escaper's image, and a renderer and an
                    # unescaper that disagree about the alphabet is the exact drift
                    # this module exists to prevent.
                    raise ValueError(
                        f"unknown \\u escape for U+{ord(ch):04X} at offset {i}: the "
                        "renderer never emits this codepoint escaped"
                    )
                out.append(ch)
                i += 6
                continue
            mapped = {"\\": "\\", "|": "|", "n": "\n", "r": "\r", "t": "\t"}.get(nxt)
            if mapped is None:
                raise ValueError(f"unknown escape {_ESC + nxt!r} at offset {i}")
            out.append(mapped)
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


# -- deterministic RNG -------------------------------------------------------


class DeterministicRng:
    """Counter-based PRNG over blake2b.

    Python's ``random.Random`` would do, but its stream is an implementation
    detail of CPython's Mersenne Twister; ``hash()`` is salted per process and is
    not usable at all. This is a few lines and is reproducible across processes,
    interpreter versions and platforms, which is what "seeded and reproducible"
    has to mean for a training shuffle that a ledger row claims to identify.
    """

    __slots__ = ("_counter", "_key")

    def __init__(self, *parts: object) -> None:
        material = "\x1f".join(repr(p) for p in parts).encode("utf-8")
        self._key = hashlib.blake2b(material, digest_size=32).digest()
        self._counter = 0

    def _next_u64(self) -> int:
        block = hashlib.blake2b(
            self._counter.to_bytes(8, "big"), key=self._key, digest_size=8
        ).digest()
        self._counter += 1
        return int.from_bytes(block, "big")

    def below(self, n: int) -> int:
        """Uniform in ``[0, n)`` by rejection sampling -- no modulo bias."""
        if n <= 0:
            raise ValueError(f"below(n) needs n >= 1, got {n}")
        if n == 1:
            return 0
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            v = self._next_u64()
            if v < limit:
                return v % n

    def permutation(self, n: int) -> tuple[int, ...]:
        """Fisher-Yates. Implemented here so the stream is ours, not CPython's."""
        idx = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.below(i + 1)
            idx[i], idx[j] = idx[j], idx[i]
        return tuple(idx)

    def cyclic_permutation(self, n: int) -> tuple[int, ...]:
        """Sattolo: uniform over the ``(n-1)!`` cyclic permutations, so a derangement.

        One character from :meth:`permutation` -- ``below(i)``, not ``below(i + 1)`` -- and
        that character is the difference between a shuffle with fixed points and one where
        every option moves. The same algorithm as qd-runtime's
        ``CounterRng::cyclic_permutation``; the *stream* is not the same (blake2b here,
        sha256 there), so equal inputs do not give equal outputs across the two.
        """
        if n < 2:
            raise ValueError(f"a derangement needs at least 2 items, got {n}")
        idx = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.below(i)
            idx[i], idx[j] = idx[j], idx[i]
        return tuple(idx)


def second_pass_permutation(
    n: int, *, seed: int, example_id: str, slot_name: str
) -> tuple[int, ...]:
    """The permutation a choice slot's second pass presents its options in: a derangement.

    ``permutation_consistency`` asks whether the head gives the same *answer* when the
    options move; ``docs/schema-api.md`` requires every option to move, which a uniform
    shuffle does not guarantee. This module owns option order (:func:`shuffle_options` is the
    first pass), so it owns the second pass too.

    The domain string is the one ``qd_data.defect_class`` introduced it under, kept so the
    stream -- and every permutation drawn before it moved here -- is unchanged.

    **Not** ``render::second_pass_permutation`` in qd-runtime: that seeds a sha256
    ``CounterRng`` from ``request.digest()`` and the slot name, and this seeds blake2b from
    ``(seed, example_id, slot_name)``. Same algorithm, different stream, so the training gate
    and the served check draw different derangements for one request. Recorded as
    ``GAP-A3-SECOND-PASS-PERMUTATION-NOT-WIRED-INTO-RENDER``; parity is not claimed.
    """
    return DeterministicRng(
        "qd_data.defect_class.second_pass.v1", seed, example_id, slot_name
    ).cyclic_permutation(n)


def shuffle_options(
    options: tuple[str, ...] | list[str],
    *,
    seed: int,
    example_id: str,
    slot_name: str,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Permute a choice slot's options for one training example.

    ``docs/hardening.md`` section 3: *option order is shuffled per training
    example*, and permutation consistency >= 95% is a gate. Returns the permuted
    options and the permutation, so a caller can map the original answer index
    onto its new letter.

    Ordinal ``score`` bins are deliberately **not** shuffled anywhere in this
    module: the bins are ordered and the loss is cumulative (CORAL-style), so
    permuting them would destroy the ordering the loss depends on.
    """
    opts = tuple(options)
    perm = DeterministicRng("qd_data.shuffle_options.v1", seed, example_id, slot_name).permutation(
        len(opts)
    )
    return tuple(opts[i] for i in perm), perm


# -- caps --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RenderCaps:
    """Every payload is bounded, and the bounds are checked before the model sees
    anything. ``docs/schema-api.md``: over the cap is a refusal, never a truncation
    -- truncating moves the answer out of the window without saying so."""

    max_context_bytes: int = 131_072
    max_question_bytes: int = 4_096
    max_option_bytes: int = 512
    max_task_bytes: int = 256
    #: Six times the context cap plus slack: see ``ESCAPE_WORST_CASE_GROWTH``.
    max_rendered_bytes: int = 802_816

    def __post_init__(self) -> None:
        for name in (
            "max_context_bytes", "max_question_bytes", "max_option_bytes",
            "max_task_bytes", "max_rendered_bytes",
        ):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"RenderCaps.{name} must be a positive int, got {v!r}")
        if self.max_option_bytes > self.max_context_bytes:
            # docs/hardening.md: "Option text longer than the context cap ->
            # refusal". Keeping the option cap at or below the context cap is what
            # makes that rule automatically true.
            raise ValueError(
                f"max_option_bytes ({self.max_option_bytes}) must not exceed "
                f"max_context_bytes ({self.max_context_bytes})"
            )
        # If the rendered cap were tighter than the worst-case escaped size, a
        # context inside the context cap could still be refused purely by arithmetic
        # -- a spurious refusal, which is its own failure. Refuse the *configuration*
        # instead.
        floor = ESCAPE_WORST_CASE_GROWTH * self.max_context_bytes + 8192
        if self.max_rendered_bytes < floor:
            raise ValueError(
                f"max_rendered_bytes ({self.max_rendered_bytes}) is below the worst-case "
                f"escaped size of a legal context ({floor} = "
                f"{ESCAPE_WORST_CASE_GROWTH}x{self.max_context_bytes} + 8192); a legal "
                "request would be refused by arithmetic alone"
            )


DEFAULT_CAPS: Final[RenderCaps] = RenderCaps()


# -- rendered output ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SlotRender:
    """One slot's decode query, and the letter alphabet it decodes over."""

    name: str
    type_name: str
    suffix: str
    #: Letter -> the value that letter denotes. Always contains the noul letter.
    letter_to_value: dict[str, str]
    #: For choice slots: the permutation applied to the request's option order,
    #: or ``()`` when the options were rendered in the order given.
    permutation: tuple[int, ...] = ()

    @property
    def noul_letter(self) -> str:
        return NOUL_LETTER


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """The prompt, split the way the runtime consumes it.

    ``docs/schema-api.md`` answering procedure: prefill the context **once**,
    snapshot, then answer each slot as a 1-token query **from the snapshot**. So
    the prefix is rendered once and each slot contributes a short suffix.
    ``prompt_for`` returns exactly the bytes the model sees for one slot.
    """

    prefix: str
    slots: tuple[SlotRender, ...]

    def prompt_for(self, slot_name: str) -> str:
        for s in self.slots:
            if s.name == slot_name:
                return self.prefix + s.suffix
        raise KeyError(f"no slot named {slot_name!r}; have {[s.name for s in self.slots]}")

    def prompts(self) -> tuple[str, ...]:
        return tuple(self.prefix + s.suffix for s in self.slots)

    def slot(self, slot_name: str) -> SlotRender:
        for s in self.slots:
            if s.name == slot_name:
                return s
        raise KeyError(f"no slot named {slot_name!r}; have {[s.name for s in self.slots]}")

    def context_region(self) -> str:
        """The escaped context exactly as it appears between the delimiters.

        Used by the injection tests to assert the region round-trips, which is a
        stronger claim than checking for any one attack string.
        """
        start = self.prefix.index(M_CTX_BEGIN + "\n") + len(M_CTX_BEGIN) + 1
        end = self.prefix.rindex("\n" + M_CTX_END)
        return self.prefix[start:end]


# -- the renderer ------------------------------------------------------------


def _render_option_lines(pairs: list[tuple[str, str]], caps: RenderCaps, slot: Slot) -> list[str]:
    lines: list[str] = []
    for letter, value in pairs:
        raw = value.encode("utf-8")
        if len(raw) > caps.max_option_bytes:
            raise OptionTooLongRefusal(
                expected=f"at most {caps.max_option_bytes} bytes", actual=len(raw),
                detail=(
                    f"slot {slot.name!r} option {letter}: refused rather than truncated, "
                    "because a truncated option is a different option"
                ),
            )
        lines.append(f"{letter}. {escape_inline(value)}")
    return lines


def _slot_suffix(
    slot: Slot,
    caps: RenderCaps,
    *,
    seed: int | None,
    example_id: str,
) -> SlotRender:
    head = f"{M_SLOT}{escape_inline(slot.name)}\n{M_TYPE}{slot.type_name}\n"

    if isinstance(slot, ChoiceSlot):
        options = slot.options
        perm: tuple[int, ...] = ()
        if seed is not None:
            options, perm = shuffle_options(
                options, seed=seed, example_id=example_id, slot_name=slot.name
            )
        pairs = [(OPTION_LETTERS[i], v) for i, v in enumerate(options)]
    elif isinstance(slot, ScoreSlot):
        # Ordinal: letters map to bins 1..n in order, never shuffled.
        pairs = [(OPTION_LETTERS[i], str(i + 1)) for i in range(slot.bins)]
        perm = ()
    elif isinstance(slot, SpanSlot):
        # A span is decoded by a pointer head over line-start tokens, so it has no
        # letter alphabet -- but noul is present in *every* option set, so the
        # abstain letter is still rendered.
        pairs = []
        perm = ()
    else:  # pragma: no cover - Request.__post_init__ already refuses this
        raise TypeError(f"unrenderable slot type {type(slot).__name__}")

    pairs_with_noul = [*pairs, (NOUL_LETTER, NOUL)]
    lines = _render_option_lines(pairs_with_noul, caps, slot)
    body = f"{M_OPT_BEGIN}\n" + "".join(f"{ln}\n" for ln in lines) + f"{M_OPT_END}\n"
    return SlotRender(
        name=slot.name,
        type_name=slot.type_name,
        suffix=head + body + M_ANSWER,
        letter_to_value=dict(pairs_with_noul),
        permutation=perm,
    )


def render(
    request: Request,
    *,
    caps: RenderCaps = DEFAULT_CAPS,
    seed: int | None = None,
) -> RenderedPrompt:
    """Render a request to the exact string the model sees.

    This is the single renderer. ``seed`` is supplied on the training path to
    shuffle choice options per example, and left ``None`` on the serving path so
    the caller's option order is preserved. Everything else is identical, which is
    what makes the byte-identity test between the two paths meaningful.
    """
    ctx_bytes = bytes(request.context)
    if len(ctx_bytes) > caps.max_context_bytes:
        raise ContextTooLargeRefusal(
            expected=f"at most {caps.max_context_bytes} bytes", actual=len(ctx_bytes),
            detail=(
                "refused rather than truncated: truncation moves the answer out of the "
                "window silently, and line spans would then point at the wrong lines"
            ),
        )
    try:
        ctx_text = ctx_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContextTooLargeRefusal(
            expected="valid UTF-8 context bytes", actual=f"decode error at byte {exc.start}",
            detail=str(exc),
        ) from exc
    if not ctx_text.strip():
        raise EmptyContextRefusal(
            expected="context with at least one non-whitespace character",
            actual=f"{len(ctx_bytes)} bytes, all whitespace",
            detail="an empty context yields a confident letter from nothing; refuse instead",
        )

    q_bytes = request.question.encode("utf-8")
    if len(q_bytes) > caps.max_question_bytes:
        raise ContextTooLargeRefusal(
            expected=f"question at most {caps.max_question_bytes} bytes", actual=len(q_bytes),
            detail="refused rather than truncated",
        )
    t_bytes = request.task.encode("utf-8")
    if len(t_bytes) > caps.max_task_bytes:
        raise ContextTooLargeRefusal(
            expected=f"task at most {caps.max_task_bytes} bytes", actual=len(t_bytes),
            detail="refused rather than truncated",
        )

    # Prompt format 2 (see `PROMPT_FORMAT`): the question line follows the context.
    prefix = (
        f"{M_BEGIN}\n"
        f"{M_FORMAT}{PROMPT_FORMAT}\n"
        f"{M_VERSION}{request.schema_version}\n"
        f"{M_TASK}{escape_inline(request.task)}\n"
        f"{M_ROUTE}{request.route}\n"
        f"{M_CTX_BEGIN}\n"
        f"{escape_block(ctx_text)}\n"
        f"{M_CTX_END}\n"
        f"{M_QUESTION}{escape_inline(request.question)}\n"
    )

    slots = tuple(
        _slot_suffix(s, caps, seed=seed, example_id=request.example_id) for s in request.slots
    )

    rendered = RenderedPrompt(prefix=prefix, slots=slots)
    for text in rendered.prompts():
        size = len(text.encode("utf-8"))
        if size > caps.max_rendered_bytes:
            raise RenderedPromptTooLargeRefusal(
                expected=f"at most {caps.max_rendered_bytes} rendered bytes", actual=size,
                detail="the bound is on the bytes the model sees, not only on those that arrived",
            )
    return rendered


def render_for_serving(
    wire: dict[str, Any],
    *,
    caps: RenderCaps = DEFAULT_CAPS,
) -> RenderedPrompt:
    """The serving entry point: parse a wire request, then render it.

    Deliberately a thin wrapper over the same :func:`render`. If this function
    ever grows rendering logic of its own, the training and serving prompts can
    drift, and ``test_render.py::test_training_and_serving_render_byte_identically``
    is the test that would fail.
    """
    return render(Request.from_wire(wire), caps=caps, seed=None)
