"""The one exception this package raises, and the rule it enforces.

A parser that *tolerates* a field it does not know is how the Rust -> Python half
of ``docs/schema-api.md`` would rot without anybody noticing: the runtime adds a
field, Python silently drops it, both suites stay green, and the divergence is
found by a caller. ``GAP-RT-WIRE-CONTEXT-ENCODING`` and
``GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS`` are the two instances this repo has
already paid for.

So every shape in :mod:`qd_wire` is closed in both directions:

* a key the parser does not know is a :class:`WireParseError`;
* a key the parser requires and does not find is a :class:`WireParseError`.

There is no lenient mode and no ``strict=False`` argument. A caller who wants the
lenient read can write it and own it.
"""

from __future__ import annotations

__all__ = ["WireParseError", "check_keys"]


class WireParseError(ValueError):
    """A wire payload did not match the contract.

    ``path`` is the dotted location inside the payload so a failure over a corpus
    of fixtures names the file *and* the field, not just "bad JSON".
    """

    def __init__(self, path: str, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"{path}: {detail}")


def check_keys(obj: dict[str, object], *, required: frozenset[str], path: str) -> None:
    """Refuse an object whose key set is not exactly ``required``.

    Both directions are refused in one place so no shape can be closed on one side
    only. The message names the offending keys and the full expected set, because
    the reader of this failure is usually the person who just changed the Rust.
    """
    present = set(obj)
    unknown = sorted(present - required)
    missing = sorted(required - present)
    if unknown:
        raise WireParseError(
            path,
            f"unknown field(s) {unknown}; this parser reads exactly {sorted(required)}. "
            "A field the Rust side emits and Python ignores is how the two halves of "
            "the contract drift apart, so this is a refusal and not a warning.",
        )
    if missing:
        raise WireParseError(
            path,
            f"missing required field(s) {missing}; this parser reads exactly {sorted(required)}. "
            "An absent field is not a default.",
        )
