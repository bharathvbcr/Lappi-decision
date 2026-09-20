"""The two non-answer halves of the wire: ``RefusalEnvelope`` and ``ErrorEnvelope``.

``docs/schema-api.md``: a refusal "is never an empty answer, and never ``noul``".
This module keeps the three apart the way the Rust does — by type, not by care:

* a :class:`Refusal` and a :class:`BackendError` have no ``value``, no ``score``,
  no ``conformal_set`` and no ``noul`` attribute, at any depth;
* :func:`parse_refusal_envelope` refuses a payload carrying ``slots``, because the
  key sets are checked exactly and ``slots`` is not one of the three keys;
* :func:`qd_wire.response.caller_reading` maps neither onto ``MODEL_ABSTAINED``.

Both payloads are internally tagged on ``kind``, and every ``kind`` this build can
emit — 36 refusals and 12 backend errors, not the four ``docs/schema-api.md``
tabulates — is validated field for field against the generated tables in
:mod:`qd_wire.contract`.

One note on identifiers. These ``kind`` strings are *the runtime's* vocabulary.
``python/qd_data/errors.py`` names the same conditions after the property rather
than the failure (``options_len`` against ``too_many_options``, six pairs in all).
:data:`KNOWN_VOCABULARY_DIFFERENCES` pins those pairs on this side too, so the
divergence is visible from Python and cannot widen unnoticed.
``GAP-XLANG-REFUSAL-VOCABULARIES``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from qd_wire.contract import BACKEND_ERROR_KINDS, REFUSAL_KINDS, STRUCT_FIELDS
from qd_wire.errors import WireParseError, check_keys
from qd_wire.values import as_str, as_uint, check_rust_typed

__all__ = [
    "CONTEXT_CHECKS_NAMED_IDENTICALLY",
    "KNOWN_VOCABULARY_DIFFERENCES",
    "BackendError",
    "ErrorEnvelope",
    "Refusal",
    "RefusalEnvelope",
    "parse_backend_error",
    "parse_error_envelope",
    "parse_refusal",
    "parse_refusal_envelope",
]

#: The four context-wire checks both lanes name with the *same* string, per
#: ``docs/schema-api.md``, *"The four refusals, named identically in both lanes"*.
CONTEXT_CHECKS_NAMED_IDENTICALLY: Final[tuple[str, ...]] = (
    "context_len_missing",
    "context_length_mismatch",
    "context_not_base64",
    "context_not_bytes",
)

#: ``(python qd_data.errors check, rust Refusal::kind())`` where the two lanes name
#: one condition differently. Mirrors ``KNOWN_VOCABULARY_DIFFERENCES`` in
#: ``crates/qd-runtime/tests/wire_context_crosslang.rs``, pinned here so the Python
#: suite fails too if a seventh pair appears. ``GAP-XLANG-REFUSAL-VOCABULARIES``.
KNOWN_VOCABULARY_DIFFERENCES: Final[tuple[tuple[str, str], ...]] = (
    ("schema_version", "unknown_schema_version"),
    ("slots_non_empty", "empty_slots"),
    ("slot_type_known", "unknown_slot_type"),
    ("options_len", "too_many_options"),
    ("slot_names_unique", "duplicate_slot_name"),
    ("bins_range", "bins_out_of_range"),
)

_REFUSAL_ENVELOPE_KEYS: Final[frozenset[str]] = frozenset(STRUCT_FIELDS["RefusalEnvelope"])
_ERROR_ENVELOPE_KEYS: Final[frozenset[str]] = frozenset(STRUCT_FIELDS["ErrorEnvelope"])


@dataclass(frozen=True, slots=True)
class Refusal:
    """A request that was not answerable as posed.

    Deliberately carries no ``value``, ``score``, ``conformal_set`` or ``noul``: a
    reporter that renders a refusal as an abstention has to fabricate the attribute,
    which raises.
    """

    kind: str
    fields: Mapping[str, object]

    def __getitem__(self, name: str) -> object:
        return self.fields[name]


@dataclass(frozen=True, slots=True)
class BackendError:
    """The backend could not serve. Never an answer, and never a ``noul``."""

    kind: str
    fields: Mapping[str, object]

    def __getitem__(self, name: str) -> object:
        return self.fields[name]


@dataclass(frozen=True, slots=True)
class RefusalEnvelope:
    """``{"status": "refused", schema_version, refusal, message}``."""

    schema_version: int
    refusal: Refusal
    message: str

    status: Literal["refused"] = "refused"


@dataclass(frozen=True, slots=True)
class ErrorEnvelope:
    """``{"status": "error", schema_version, error, message}``."""

    schema_version: int
    error: BackendError
    message: str

    status: Literal["error"] = "error"


def _parse_tagged(
    raw: object,
    *,
    table: Mapping[str, Mapping[str, str]],
    what: str,
    path: str,
) -> tuple[str, Mapping[str, object]]:
    """Read an internally tagged ``{"kind": ..., <variant fields>}`` payload."""
    if not isinstance(raw, dict):
        raise WireParseError(path, f"expected a `{what}` object, got {type(raw).__name__}")
    if "kind" not in raw:
        raise WireParseError(
            path,
            f"a `{what}` is internally tagged on `kind` and this payload has none. "
            "Every refusal names the check that failed; an untagged one names nothing.",
        )
    kind = as_str(raw["kind"], path=f"{path}.kind")
    spec = table.get(kind)
    if spec is None:
        raise WireParseError(
            f"{path}.kind",
            f"unknown `{what}` kind {kind!r}. This build's `{what}` has "
            f"{len(table)} kinds and this is not one of them, so either the runtime "
            "grew a condition Python has never seen or the payload is not from this "
            "build. Refused: a caller branching on an unknown identifier would fall "
            "through to whatever its default branch does.",
        )
    check_keys(raw, required=frozenset(spec) | {"kind"}, path=path)
    fields = {
        name: check_rust_typed(raw[name], rust_type, path=f"{path}.{name}")
        for name, rust_type in spec.items()
    }
    return kind, MappingProxyType(fields)


def parse_refusal(raw: object, *, path: str) -> Refusal:
    kind, fields = _parse_tagged(raw, table=REFUSAL_KINDS, what="Refusal", path=path)
    return Refusal(kind=kind, fields=fields)


def parse_backend_error(raw: object, *, path: str) -> BackendError:
    kind, fields = _parse_tagged(raw, table=BACKEND_ERROR_KINDS, what="BackendError", path=path)
    return BackendError(kind=kind, fields=fields)


def parse_refusal_envelope(raw: dict[str, object], *, path: str = "$") -> RefusalEnvelope:
    check_keys(raw, required=_REFUSAL_ENVELOPE_KEYS, path=path)
    return RefusalEnvelope(
        schema_version=as_uint(
            raw["schema_version"], path=f"{path}.schema_version", rust_type="u32"
        ),
        refusal=parse_refusal(raw["refusal"], path=f"{path}.refusal"),
        message=as_str(raw["message"], path=f"{path}.message"),
    )


def parse_error_envelope(raw: dict[str, object], *, path: str = "$") -> ErrorEnvelope:
    check_keys(raw, required=_ERROR_ENVELOPE_KEYS, path=path)
    return ErrorEnvelope(
        schema_version=as_uint(
            raw["schema_version"], path=f"{path}.schema_version", rust_type="u32"
        ),
        error=parse_backend_error(raw["error"], path=f"{path}.error"),
        message=as_str(raw["message"], path=f"{path}.message"),
    )
