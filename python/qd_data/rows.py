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

__all__ = ["DataRow", "GoldAnswer", "row_content_hash"]

#: Bounded: a row's dedupe text is hashed and shingled, and both are linear in its
#: length. The mixture refuses a row over this before it is ever constructed.
MAX_DEDUPE_TEXT_BYTES: Final[int] = 262_144


@dataclass(frozen=True, slots=True)
class GoldAnswer:
    """The supervision for one slot.

    ``value`` is the *semantic* answer (an option string, an ordinal bin, a line
    range), never a letter. Letters are assigned by the renderer after the per-example
    option shuffle, so storing a letter would bake in one permutation and silently
    mislabel every reshuffled epoch.
    """

    slot_name: str
    value: str | int | tuple[int, int] | None
    #: True when the gold answer is *abstain*. This is a value the model may produce,
    #: never an error -- see ``qd_data.errors``. CLINC150's out-of-scope class and
    #: SQuAD 2.0's unanswerable questions are where this comes from for free.
    is_noul: bool = False

    def __post_init__(self) -> None:
        if not self.slot_name.strip():
            raise ValueError("GoldAnswer.slot_name must be non-empty")
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
