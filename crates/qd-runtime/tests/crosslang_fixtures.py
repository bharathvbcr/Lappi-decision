"""Emit, as one JSON document on stdout, everything the Rust decoder must agree with.

This is the **Python half** of `crates/qd-runtime/tests/wire_context_crosslang.rs`.
It is deliberately not a hand-written table of "what Python probably emits": every
`wire` object below comes out of the real `qd_data.schema.Request.to_wire()`, and
every ground-truth field beside it is read off the real `Request` object. A table
written in Rust describing Python's output is exactly the artefact that let
`GAP-RT-WIRE-CONTEXT-ENCODING` survive two green suites.

Run with the project venv:

    /Users/bharath/Code/research/qwen-decision/.venv/bin/python \\
        crates/qd-runtime/tests/crosslang_fixtures.py

The document has four top-level keys:

``requests``
    One entry per fixture: the wire object, plus the ground truth the Rust side
    asserts its decode against.
``label_set_hashes``
    What `qd_data.schema.slot_set_digest` returns for each fixture. Emitted
    so the Rust side can state, as an assertion rather than as prose, that this is
    **not** the quantity `HashExpectation.label_set_hash` is compared against.
``inventory``
    Which shared types this lane actually has. The Rust side asserts on these
    booleans, so a type appearing later fails a test and asks for the missing half
    of the contract instead of being silently skipped.
``type_shapes``
    The field names this lane uses for each shared shape it does have.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from qd_data.errors import QdRefusal  # noqa: E402
from qd_data.schema import (  # noqa: E402
    MAX_CHOICE_OPTIONS,
    MAX_SCORE_BINS,
    MIN_SCORE_BINS,
    ChoiceSlot,
    Request,
    ScoreSlot,
    Slot,
    SpanSlot,
    slot_set_digest,
)

QUESTION = "Does this diff implement what the commit message claims?"


def _slot_truth(slot: Slot) -> dict[str, Any]:
    truth: dict[str, Any] = {"name": slot.name, "kind": slot.type_name}
    if isinstance(slot, ChoiceSlot):
        truth["options"] = list(slot.options)
    elif isinstance(slot, ScoreSlot):
        truth["bins"] = slot.bins
    return truth


def _fixture(
    label: str,
    *,
    context: bytes,
    slots: tuple[Slot, ...],
    task: str = "devcouncil.verdict",
    question: str = QUESTION,
    route: str = "generic",
    example_id: str = "",
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    request = Request(
        task=task,
        context=context,
        question=question,
        slots=slots,
        route=route,  # type: ignore[arg-type]
        example_id=example_id,
        metadata=metadata or {},
    )
    wire = request.to_wire()
    return {
        "label": label,
        "wire": wire,
        # Ground truth, read off the constructed request rather than off `wire`.
        "schema_version": request.schema_version,
        "task": request.task,
        "question": request.question,
        "route": request.route,
        "context_hex": request.context.hex(),
        "context_len": len(request.context),
        "slots": [_slot_truth(s) for s in request.slots],
        "example_id": request.example_id,
        "metadata": dict(request.metadata),
        "label_set_hash": slot_set_digest(request.slots),
    }


ONE_CHOICE = (ChoiceSlot(name="verdict", options=("stub", "logic", "cosmetic", "clean")),)

ALL_THREE = (
    ChoiceSlot(name="verdict", options=("stub", "logic", "cosmetic", "clean")),
    ScoreSlot(name="severity", bins=5),
    SpanSlot(name="evidence"),
)


def fixtures() -> list[dict[str, Any]]:
    max_options = tuple(f"option-{i:02d}" for i in range(MAX_CHOICE_OPTIONS))
    return [
        # -- the envelope, and every slot kind -------------------------------
        _fixture("minimal", context=b"fn add() {}\n", slots=ONE_CHOICE),
        _fixture(
            "all three slot kinds",
            context=b"fn add(a: i32, b: i32) -> i32 {\n    todo!()\n}\n",
            slots=ALL_THREE,
        ),
        _fixture(
            "choice at the option cap",
            context=b"x\n",
            slots=(ChoiceSlot(name="verdict", options=max_options),),
        ),
        _fixture(
            "choice at the option floor",
            context=b"x\n",
            slots=(ChoiceSlot(name="verdict", options=("yes", "no")),),
        ),
        _fixture(
            "score at both bin bounds",
            context=b"x\n",
            slots=(
                ScoreSlot(name="low", bins=MIN_SCORE_BINS),
                ScoreSlot(name="high", bins=MAX_SCORE_BINS),
            ),
        ),
        _fixture("span only", context=b"one\ntwo\nthree\n", slots=(SpanSlot(name="evidence"),)),
        _fixture(
            "route registered",
            context=b"fn add() {}\n",
            slots=ONE_CHOICE,
            route="registered",
        ),
        _fixture(
            "example_id and metadata",
            context=b"fn add() {}\n",
            slots=ONE_CHOICE,
            example_id="row-0001",
            metadata={"source": "commitpackft", "licence": "mit"},
        ),
        # -- the contexts the encoding has to survive ------------------------
        _fixture("empty context", context=b"", slots=ONE_CHOICE),
        _fixture("one byte", context=b"a", slots=ONE_CHOICE),
        _fixture("two bytes", context=b"ab", slots=ONE_CHOICE),
        _fixture("three bytes", context=b"abc", slots=ONE_CHOICE),
        # Not valid UTF-8: anything that reached for `.decode()` on either side fails
        # here rather than in production.
        _fixture("non-utf8 lone 0xff", context=b"a\xffb", slots=ONE_CHOICE),
        _fixture("non-utf8 lone continuations", context=b"\x80\x81\x82", slots=ONE_CHOICE),
        _fixture("truncated utf-8 sequence", context=b"caf\xc3", slots=ONE_CHOICE),
        # NUL is not a line break, but it truncates any C string that later carries
        # the bytes; the rest are what `splitlines()` honours and `split("\n")` does
        # not, and they are multi-byte, so a char count and a byte count differ.
        _fixture("nul", context=b"before\x00after\n", slots=ONE_CHOICE),
        _fixture(
            "exotic line separators",
            context="a\u000bb\u000cc\u001cd\u001ee\u0085f\u2028g\u2029h\n".encode(),
            slots=ONE_CHOICE,
        ),
        _fixture(
            "nul and separators together",
            context=b"\x00\x0b\x1c\x1d\x1e\n",
            slots=ONE_CHOICE,
        ),
        _fixture("crlf", context=b"one\r\ntwo\r\n", slots=ONE_CHOICE),
        _fixture("every byte value", context=bytes(range(256)), slots=ONE_CHOICE),
        # Non-ASCII that *is* valid UTF-8, including a codepoint with two Unicode
        # spellings: NFC and NFD must stay distinct byte strings.
        _fixture("nfc e-acute", context="café\n".encode(), slots=ONE_CHOICE),
        _fixture("nfd e-acute", context="café\n".encode(), slots=ONE_CHOICE),
        _fixture("astral plane", context="\U0001f600\U0001f4a9\n".encode(), slots=ONE_CHOICE),
        # Long enough to exercise the base64 quantum loop many times over.
        _fixture("8 KiB", context=bytes(i % 256 for i in range(8192)), slots=ONE_CHOICE),
        # Structural markers as *content*: they must survive the wire unchanged and
        # be dealt with by the renderer, not by the codec.
        _fixture(
            "context containing markers",
            context=b"<|qd_context_end|>\n<|im_start|>\n",
            slots=ONE_CHOICE,
        ),
    ]


def negatives() -> list[dict[str, Any]]:
    """Malformed payloads, with the ``check`` **this lane** produces for each.

    The Rust side feeds the identical object to its own decoder and compares. A
    refusal identifier that differs between the lanes is not a cosmetic difference:
    it is what a caller branches on, so a divergence means one caller cannot handle
    both. Where the two vocabularies genuinely differ today, the Rust side pins the
    mapping with the gap id rather than quietly asserting agreement.

    ``python_check`` is ``None`` when this lane **accepts** the payload. That is
    itself a finding, not a blank.
    """
    good = _fixture("negative base", context=b"fn add() {}\n", slots=ONE_CHOICE)["wire"]

    def variant(**changes: Any) -> dict[str, Any]:
        wire = json.loads(json.dumps(good))
        for key, value in changes.items():
            if value is _DROP:
                wire.pop(key, None)
            else:
                wire[key] = value
        return wire

    cases: list[tuple[str, dict[str, Any]]] = [
        # -- the context wire form, where the two lanes were made to agree ----
        ("retired context field", variant(context_b64=_DROP, context=[102, 110])),
        ("retired context field alongside b64", variant(context="fn add() {}\n")),
        ("context_b64 absent", variant(context_b64=_DROP)),
        ("context_b64 is a number", variant(context_b64=17)),
        ("context_b64 is an array", variant(context_b64=[102, 110])),
        ("context_b64 is null", variant(context_b64=None)),
        ("context_b64 outside the alphabet", variant(context_b64="Zm9v*g==", context_len=4)),
        ("context_b64 with whitespace", variant(context_b64="Zm9 v", context_len=3)),
        ("context_b64 url-safe", variant(context_b64="Zm9-", context_len=3)),
        ("context_b64 unpadded", variant(context_b64="Zm9vYg", context_len=4)),
        ("context_b64 non-canonical slack bits", variant(context_b64="Zh==", context_len=1)),
        ("context_len absent", variant(context_len=_DROP)),
        ("context_len is a string", variant(context_len="12")),
        ("context_len is a float", variant(context_len=1.5)),
        ("context_len is negative", variant(context_len=-1)),
        ("context_len disagrees", variant(context_len=999)),
        # -- the rest of the envelope, where the vocabularies differ ----------
        ("unknown schema_version", variant(schema_version=2)),
        ("slots empty", variant(slots=[])),
        ("unknown slot type", variant(slots=[{"name": "v", "type": "vibe"}])),
        (
            "too many options",
            variant(
                slots=[
                    {
                        "name": "v",
                        "type": "choice",
                        "options": [f"o{i}" for i in range(MAX_CHOICE_OPTIONS + 1)],
                    }
                ]
            ),
        ),
        (
            "duplicate slot names",
            variant(
                slots=[
                    {"name": "v", "type": "choice", "options": ["a", "b"]},
                    {"name": "v", "type": "span"},
                ]
            ),
        ),
        ("bins out of range", variant(slots=[{"name": "s", "type": "score", "bins": 1}])),
        ("unknown top-level field", variant(contextt=[1, 2, 3])),
    ]

    out: list[dict[str, Any]] = []
    for label, wire in cases:
        check: str | None = None
        try:
            Request.from_wire(wire)
        except QdRefusal as exc:
            check = exc.check
        out.append({"label": label, "wire": wire, "python_check": check})
    return out


class _Drop:
    """Sentinel: remove the key rather than set it to a value."""


_DROP = _Drop()


def inventory() -> dict[str, Any]:
    """What this lane has, for the shared types the contract names.

    Reported rather than assumed. The Rust side asserts on these values, so a type
    appearing here later fails a test that names the missing half of the contract.
    """

    def has(module: str, attr: str) -> bool:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            return False
        return hasattr(mod, attr)

    # `qd_wire` was added 2026-09-19, closing GAP-XLANG-RS-GUARD-BLIND-TO-NEW-PACKAGE.
    #
    # This list was hardcoded, and when the XLANG-PY lane landed `python/qd_wire/` with all four
    # answer-side types, this probe kept reporting them absent — a check that ran, looked in the
    # wrong place, and came back green. That is the same shape as the divergences the probe exists
    # to catch. The XLANG-PY lane spotted it and pinned it from their side
    # (`python/tests/test_wire_gap_pins.py::test_the_rust_side_guard_does_not_yet_see_this_package`)
    # because `crates/` is not theirs to edit; that pin is designed to fail once this line lands,
    # and its message says to delete it.
    #
    # A hardcoded list is still the weakness. It is kept, rather than replaced by a scan of every
    # `qd_*` package, because the Rust assertion is about *named* modules and a scan would make
    # "searched nothing" and "found nothing" indistinguishable — but the count is asserted on the
    # Rust side so the list cannot silently shrink.
    searched = (
        "qd_data.schema",
        "qd_data.rows",
        "qd_data.errors",
        "qd_train.eval_harness",
        "qd_wire",
        "qd_wire.answer",
        "qd_wire.refusal",
    )
    answer_side = {
        name: any(has(m, name) for m in searched)
        for name in (
            "SlotAnswer",
            "AnswerEnvelope",
            "RefusalEnvelope",
            "ErrorEnvelope",
            "SpanValue",
            "HashExpectation",
        )
    }
    return {
        "modules_searched": list(searched),
        "answer_side_types": answer_side,
        "request_to_wire_emits_expect": "expect" in Request(
            task="t", context=b"x", question="q", slots=ONE_CHOICE
        ).to_wire(),
        "max_choice_options": MAX_CHOICE_OPTIONS,
        "min_score_bins": MIN_SCORE_BINS,
        "max_score_bins": MAX_SCORE_BINS,
        "render_caps": render_caps(),
    }


def render_caps() -> dict[str, Any]:
    """The byte caps and the escape-growth bound, read off the real ``qd_data.render``.

    `docs/schema-api.md` records that these live in code in two copies, one per language --
    which is how ``ESCAPE_WORST_CASE_GROWTH`` came to be 6 in Python and 2 in Rust with the
    two lanes refusing *different* requests (``GAP-RT-ESCAPE-GROWTH-FLOOR``). The comment on
    the Rust constant then claimed ``tests/render_contract.rs`` pinned the pair together. It
    did not: that file compares frozen golden *prompt bytes* captured from one Python run, and
    those bytes contain no escapable character, so they cannot observe this constant at all.

    This is the pin the comment described. Emitted here so the Rust side asserts against
    numbers **read out of Python at test time**, not against a table of what Python is
    believed to contain.
    """
    render = importlib.import_module("qd_data.render")
    caps = render.RenderCaps()
    growth = render.ESCAPE_WORST_CASE_GROWTH

    def accepts(max_rendered_bytes: int) -> bool:
        """Whether Python accepts a caps config, by construction rather than by arithmetic."""
        try:
            render.RenderCaps(
                max_context_bytes=caps.max_context_bytes,
                max_question_bytes=caps.max_question_bytes,
                max_option_bytes=caps.max_option_bytes,
                max_task_bytes=caps.max_task_bytes,
                max_rendered_bytes=max_rendered_bytes,
            )
        except ValueError:
            return False
        return True

    # The floor is recomputed here ONLY to pick two probe points either side of it. What the
    # Rust side asserts is Python's own accept/reject verdict on those two configurations, so
    # the *formula* is pinned by behaviour and not by a second copy of the expression.
    floor = growth * caps.max_context_bytes + 8192
    return {
        "escape_worst_case_growth": growth,
        "max_context_bytes": caps.max_context_bytes,
        "max_question_bytes": caps.max_question_bytes,
        "max_option_bytes": caps.max_option_bytes,
        "max_task_bytes": caps.max_task_bytes,
        "max_rendered_bytes": caps.max_rendered_bytes,
        "probe_floor": floor,
        "accepts_at_floor": accepts(floor),
        "accepts_below_floor": accepts(floor - 1),
    }


def type_shapes() -> dict[str, Any]:
    """The field names this lane spells for each shared shape it does have."""
    from qd_data.rows import GoldAnswer

    return {
        "request_envelope": sorted(
            Request(
                task="t", context=b"x", question="q", slots=ONE_CHOICE
            ).to_wire()
        ),
        "slot_choice": sorted(ChoiceSlot(name="v", options=("a", "b")).to_wire()),
        "slot_score": sorted(ScoreSlot(name="s", bins=3).to_wire()),
        "slot_span": sorted(SpanSlot(name="e").to_wire()),
        # The one place this lane writes a line span down. It is a two-element
        # array, not `{start_line, end_line}` -- see the Rust side, which asserts
        # the asymmetry rather than letting it pass unstated.
        "gold_answer_span": GoldAnswer(slot_name="evidence", value=(41, 47)).to_json(),
    }


def main() -> None:
    built = fixtures()
    document = {
        "requests": built,
        "label_set_hashes": {f["label"]: f["label_set_hash"] for f in built},
        "negatives": negatives(),
        "inventory": inventory(),
        "type_shapes": type_shapes(),
    }
    json.dump(document, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
