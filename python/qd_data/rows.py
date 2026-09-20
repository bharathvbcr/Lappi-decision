"""The one row type the whole data lane passes around.

Every stage -- rewriting, dedupe, splitting, manifesting -- reads the same object,
so there is a single place where "what a training example is" is defined. The
alternative (a tuple here, a dict there, a dataclass in the splitter) is how a
field like ``repo`` quietly stops being carried, and ``docs/hardening.md`` section 2
opens with *"Keep the repo name on every row from the pull onward."*

Two fields are the load-bearing ones and neither may be empty:

``repo_key``
    What the split is taken over. Repo-level, never row-level.
``identity_key``
    ``repo + path + symbol``. ``docs/hardening.md`` section 1: *the same function
    never appears mutated and clean in different splits ... enforced at split time
    by function-level identity, not by diff hash -- the same function reformatted
    has a different diff hash and would leak.*

``licence_id`` and ``host`` are carried per row rather than per dataset because
``bigcode/commitpackft`` is ``mit`` as a dataset and its **rows** are not: three of
its thirteen declared values are copyleft or absent. The model card is generated
from these fields, so dropping them is not a cosmetic loss.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Final

from .schema import Request, canonical_json

__all__ = [
    "DataRow",
    "GoldAnswer",
    "gold_span_to_wire",
    "row_content_hash",
    "wire_span_to_gold",
]

#: Bounded: a row's dedupe text is hashed and shingled, and both are linear in its
#: length. The mixture refuses a row over this before it is ever constructed.
MAX_DEDUPE_TEXT_BYTES: Final[int] = 262_144


# -- the two spellings of a line span, and the one function between them ------
#
# ``GAP-XLANG-SPAN-THREE-SPELLINGS``. A line span is written two ways in this repo and
# they stay two, deliberately: ``GoldAnswer.value`` is a two-element array because it is
# a *training label* in a JSONL row, and ``SlotValue::Span`` is ``{start_line, end_line}``
# because it is a *runtime answer* on the wire. Forcing one shape on both would be
# over-unification -- they answer different questions and never travel together.
#
# What was wrong was not the two shapes; it was that only one of them had a rule.
# ``GAP-XLANG-SPAN-BOUNDS-UNPINNED`` made ``1 <= start_line <= end_line`` a term of the
# wire format, enforced in ``crates/qd-runtime/src/schema.rs::SpanValue``'s
# ``TryFrom<SpanValueWire>`` and in ``qd_wire.answer.parse_slot_value``. The gold
# spelling enforced nothing, so ``GoldAnswer(value=(47, 41))`` was constructible and
# failed only at whatever later tried to use it -- and the gap's own action_required
# says the conversion "must therefore carry the bounds, not only reshape the fields".
#
# So: one rule, checked in both spellings, and one named pair of functions between them
# rather than an inline ``{"start_line": v[0], ...}`` at each future call site. The
# agreement is asserted by execution, not by a second copy of the expression --
# ``test_qd_data_wire_agreement.py`` drives this module's output into
# ``qd_wire.answer.parse_slot_value`` and checks that what one refuses the other does.


def _checked_span(value: object, *, what: str) -> tuple[int, int]:
    """The one implementation of the span rule on this side.

    ``bool`` is excluded explicitly: it is an ``int`` subclass in Python, so
    ``(True, 5)`` would otherwise pass as the span ``(1, 5)``.
    """
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(
            f"{what}: a line span is exactly two values, got {value!r}. "
            "The runtime spelling is {start_line, end_line} and has no other arity."
        )
    start, end = value
    for label, v in (("start_line", start), ("end_line", end)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{what}: {label} must be an int, got {v!r}")
    if start < 1:
        raise ValueError(
            f"{what}: span starts at line {start}, but a span is 1-based and inclusive, "
            "so line 0 does not exist"
        )
    if start > end:
        raise ValueError(
            f"{what}: span runs backwards: start_line {start} is after end_line {end}. "
            "A backwards span is not a low-confidence span, it is not a span; it is "
            "refused rather than silently reordered into a plausible-looking answer"
        )
    return int(start), int(end)


def gold_span_to_wire(value: tuple[int, int]) -> dict[str, int]:
    """A gold span as the runtime spells one: ``[41, 47]`` -> ``{start_line, end_line}``.

    The single named conversion ``GAP-XLANG-SPAN-THREE-SPELLINGS`` asks for. An eval
    that scores model spans against gold answers converts here and nowhere else; an
    inline reshape at the call site would be the third spelling the gap warns about,
    and would carry no bounds.
    """
    start, end = _checked_span(value, what="gold span")
    return {"start_line": start, "end_line": end}


def wire_span_to_gold(value: dict[str, int]) -> tuple[int, int]:
    """The inverse: ``{start_line, end_line}`` -> ``(41, 47)``.

    Refuses an unknown key for the same reason ``SpanValue`` is ``deny_unknown_fields``:
    a field nobody reads is a field the producer believes was carried.
    """
    if not isinstance(value, dict):
        raise ValueError(
            f"a wire span is an object, got {type(value).__name__}. The gold spelling is "
            "the two-element array; use it directly rather than converting from it."
        )
    if set(value) != {"start_line", "end_line"}:
        raise ValueError(
            f"a wire span is exactly {{start_line, end_line}}, got {sorted(value)}"
        )
    return _checked_span((value["start_line"], value["end_line"]), what="wire span")


@dataclass(frozen=True, slots=True)
class GoldAnswer:
    """The supervision for one slot.

    ``value`` is the *semantic* answer (an option string, an ordinal bin, a line
    range), never a letter. Letters are assigned by the renderer after the per-example
    option shuffle, so storing a letter would bake in one permutation and silently
    mislabel every reshuffled epoch.
    """

    slot_name: str
    #: A span is the two-element array ``(start_line, end_line)``, 1-based and
    #: inclusive -- the same numeric convention as the runtime spelling, and since
    #: 2026-09-19 the same *rule*: it is checked in ``__post_init__`` by the same
    #: predicate :func:`gold_span_to_wire` uses, so a label that could not become a
    #: ``SlotValue::Span`` cannot be written to a shard either.
    value: str | int | tuple[int, int] | None
    #: True when the gold answer is *abstain*. This is a value the model may produce,
    #: never an error -- see ``qd_data.errors``. CLINC150's out-of-scope class and
    #: SQuAD 2.0's unanswerable questions are where this comes from for free.
    is_noul: bool = False

    def __post_init__(self) -> None:
        if not self.slot_name.strip():
            raise ValueError("GoldAnswer.slot_name must be non-empty")
        if isinstance(self.value, (tuple, list)):
            # A span label. Checked here rather than at whatever later reads it: an
            # unchecked backwards span used to survive construction, dedupe, splitting
            # and the shard writer, and surfaced only where something tried to make a
            # runtime span of it. `GAP-XLANG-SPAN-THREE-SPELLINGS`.
            _checked_span(self.value, what=f"slot {self.slot_name!r}")
        if self.is_noul and self.value is not None:
            raise ValueError(
                f"slot {self.slot_name!r}: an abstention carries no value, but got "
                f"{self.value!r}. A gold answer that is both noul and a value is two labels."
            )
        if not self.is_noul and self.value is None:
            raise ValueError(
                f"slot {self.slot_name!r}: a non-abstaining gold answer needs a value. "
                "None would be indistinguishable from an unlabelled row."
            )

    def to_json(self) -> dict[str, Any]:
        value: Any = list(self.value) if isinstance(self.value, tuple) else self.value
        return {"slot_name": self.slot_name, "value": value, "is_noul": self.is_noul}


@dataclass(frozen=True, slots=True)
class DataRow:
    """One example, with its provenance attached rather than remembered."""

    row_id: str
    source_id: str
    host: str
    family_id: str
    #: The split unit. For code rows this is the upstream repository; for rows with
    #: no repository it is a synthetic, stable key (see ``qd_data.mixture``), because
    #: a row with no split unit cannot be split safely and must not default to
    #: "its own group".
    repo_key: str
    #: ``repo + path + symbol``; equal for the same function seen twice.
    identity_key: str
    licence_id: str
    obligations: tuple[str, ...]
    request: Request
    gold: tuple[GoldAnswer, ...]
    #: The text dedupe compares. Deliberately separate from ``request.context``: for
    #: a code row it is the *changed code*, not the rendered prompt, because two
    #: vendored copies of one file differ in their commit message and would not be
    #: caught by comparing prompts.
    dedupe_text: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("row_id", "source_id", "host", "family_id", "repo_key",
                     "identity_key", "licence_id"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(
                    f"DataRow.{name} must be a non-empty string, got {v!r}. "
                    "A row that cannot say where it came from cannot be split safely."
                )
        if not self.gold:
            raise ValueError(f"row {self.row_id!r} carries no gold answer")
        slot_names = {s.name for s in self.request.slots}
        for g in self.gold:
            if g.slot_name not in slot_names:
                raise ValueError(
                    f"row {self.row_id!r}: gold answer names slot {g.slot_name!r}, which "
                    f"is not in the request. Request slots: {sorted(slot_names)}"
                )
        if len({g.slot_name for g in self.gold}) != len(self.gold):
            raise ValueError(f"row {self.row_id!r}: two gold answers for one slot")
        size = len(self.dedupe_text.encode("utf-8"))
        if size > MAX_DEDUPE_TEXT_BYTES:
            raise ValueError(
                f"row {self.row_id!r}: dedupe_text is {size} bytes, over the "
                f"{MAX_DEDUPE_TEXT_BYTES} bound"
            )
        if not self.dedupe_text.strip():
            raise ValueError(
                f"row {self.row_id!r}: dedupe_text is empty, so this row would be "
                "compared against nothing and could never be found as a duplicate"
            )

    def provenance(self) -> dict[str, Any]:
        """What the manifest and the model card record for this row."""
        return {
            "row_id": self.row_id,
            "source_id": self.source_id,
            "host": self.host,
            "family_id": self.family_id,
            "repo_key": self.repo_key,
            "identity_key": self.identity_key,
            "licence_id": self.licence_id,
            "obligations": list(self.obligations),
            "content_hash": row_content_hash(self),
        }


def row_content_hash(row: DataRow) -> str:
    """A content hash over everything that changes what the model is shown or told.

    Not over ``row_id``: two pulls that assign different ids to the same example must
    hash the same, or ``data_snapshot_hash`` would change for a corpus that did not.
    """
    body = {
        "source_id": row.source_id,
        "family_id": row.family_id,
        "identity_key": row.identity_key,
        "licence_id": row.licence_id,
        "request": row.request.to_wire(),
        "gold": [g.to_json() for g in row.gold],
        "dedupe_text": row.dedupe_text,
    }
    # `request.to_wire()` carries example_id, which is derived from row_id. Drop it
    # for the same reason row_id itself is dropped.
    body["request"].pop("example_id", None)  # type: ignore[union-attr]
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
