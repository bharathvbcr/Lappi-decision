"""Primitive readers that refuse the shapes JSON makes easy to confuse.

Three of these matter more than they look:

* ``True`` is an ``int`` in Python, so a naive ``isinstance(v, int)`` reads a JSON
  ``true`` as the score bin ``1``. Every integer read here rejects ``bool`` first.
* ``3.0`` is not a ``u32``. ``serde_json`` refuses a float where an integer type is
  declared, so accepting it here would make Python read a payload the runtime would
  not, in the lenient direction.
* A ``score`` of ``null`` is the signature of a non-finite ``f64``: ``serde_json``
  serializes ``NaN`` and infinity as ``null`` rather than failing. Reading it as
  "no score" would turn a broken calibration into a quiet zero.
"""

from __future__ import annotations

import math

from qd_wire.errors import WireParseError

__all__ = ["as_bool", "as_finite_float", "as_str", "as_uint", "check_rust_typed"]


def as_str(value: object, *, path: str) -> str:
    if not isinstance(value, str):
        raise WireParseError(path, f"expected a JSON string, got {_name(value)}")
    return value


def as_bool(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise WireParseError(path, f"expected a JSON boolean, got {_name(value)}")
    return value


def as_uint(value: object, *, path: str, rust_type: str = "usize") -> int:
    """A Rust unsigned integer: an exact JSON integer, never a bool and never a float."""
    if isinstance(value, bool):
        raise WireParseError(
            path,
            f"expected a `{rust_type}`, got a JSON boolean. Python's bool is a subclass "
            "of int, so this is refused explicitly rather than read as 0 or 1.",
        )
    if not isinstance(value, int):
        raise WireParseError(
            path,
            f"expected a `{rust_type}` as a JSON integer, got {_name(value)}. "
            "serde_json refuses a float for an integer field, so accepting one here "
            "would make Python the lenient side of the same wire.",
        )
    if value < 0:
        raise WireParseError(path, f"expected a `{rust_type}` (unsigned), got {value}")
    return value


def as_finite_float(value: object, *, path: str) -> float:
    if isinstance(value, bool):
        raise WireParseError(path, "expected an `f64`, got a JSON boolean")
    if value is None:
        raise WireParseError(
            path,
            "expected an `f64`, got null. serde_json writes a non-finite f64 as null, "
            "so a null here is a NaN or an infinity that reached the wire, not an "
            "absent score.",
        )
    if not isinstance(value, (int, float)):
        raise WireParseError(path, f"expected an `f64`, got {_name(value)}")
    out = float(value)
    if not math.isfinite(out):
        raise WireParseError(path, f"expected a finite `f64`, got {out}")
    return out


def check_rust_typed(value: object, rust_type: str, *, path: str) -> object:
    """Validate one payload field against the Rust type named in :mod:`qd_wire.contract`.

    Driven by the generated table, so a field whose Rust type changes is validated
    differently on the next regeneration without anybody editing a validator.

    A Rust type this function does not know is a hard failure, not a skip: a field
    the XLANG-RS lane adds with a new type must stop Python, because silently
    waving it through is how the unchecked field becomes the divergence.
    """
    if rust_type in ("usize", "u32", "u64"):
        return as_uint(value, path=path, rust_type=rust_type)
    if rust_type == "String":
        return as_str(value, path=path)
    if rust_type == "bool":
        return as_bool(value, path=path)
    if rust_type == "f64":
        return as_finite_float(value, path=path)
    if rust_type == "Vec<String>":
        return tuple(_each(value, rust_type, path, lambda v, p: as_str(v, path=p)))
    if rust_type == "Vec<u32>":
        return tuple(
            _each(value, rust_type, path, lambda v, p: as_uint(v, path=p, rust_type="u32"))
        )
    if rust_type == "HashKind":
        from qd_wire.contract import HASH_KINDS

        got = as_str(value, path=path)
        if got not in HASH_KINDS:
            raise WireParseError(
                path,
                f"unknown HashKind {got!r}; this build's `HashKind` is {list(HASH_KINDS)}",
            )
        return got
    raise WireParseError(
        path,
        f"no reader for the Rust type `{rust_type}`. The contract table names a type "
        "this parser cannot validate, which means the Rust side grew a shape Python "
        "has never checked. Refused rather than passed through unvalidated.",
    )


def _each(value: object, rust_type: str, path: str, read):
    if not isinstance(value, list):
        raise WireParseError(path, f"expected a `{rust_type}` as a JSON array, got {_name(value)}")
    return [read(v, f"{path}[{i}]") for i, v in enumerate(value)]


def _name(value: object) -> str:
    if value is None:
        return "null"
    return {
        bool: "a JSON boolean",
        int: "a JSON integer",
        float: "a JSON number (float)",
        str: "a JSON string",
        list: "a JSON array",
        dict: "a JSON object",
    }.get(type(value), f"a {type(value).__name__}")
