"""Re-derive the wire contract from ``crates/qd-runtime/src`` mechanically.

This module exists so :mod:`qd_wire.contract` is never a **hand copy** of the Rust.
A hand copy is the artefact this repo has already been bitten by three times: one
name, two quantities, both suites green. Here the pinned tables in
:mod:`qd_wire.contract` are *generated* by this module, and
``python/tests/test_wire_contract_matches_rust.py`` re-derives them and asserts
equality. When the XLANG-RS lane renames a field, the Python test fails naming it.

It is a test-support module, not a runtime dependency: nothing in the parser calls
it. The Rust tree is absent from an installed wheel, and :func:`rust_src_dir`
reports that as *not found* so the caller can record a ``NotRun`` rather than a pass.

The parse is deliberately narrow: ``pub struct`` / ``pub enum`` bodies, field names
and their Rust types, plus the ``kind()`` match arms. It refuses to return a thin
answer — every extractor raises when it finds less than the shape it was asked for,
because an extractor that returns ``{}`` on a parse failure would make the
comparison test pass against nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "RustParseError",
    "RustSourceMissing",
    "derive_contract",
    "extract_const_u32_slice",
    "extract_enum_variants",
    "extract_kind_arms",
    "extract_struct_fields",
    "rust_src_dir",
    "snake_case",
]


class RustSourceMissing(FileNotFoundError):
    """The Rust tree is not present. A check, not a pass and not a failure."""


class RustParseError(RuntimeError):
    """The Rust source was present but did not parse into the shape expected.

    Raised rather than returning a partial table: a comparison against a table
    this module failed to build would compare two kinds of nothing and pass.
    """


def rust_src_dir(start: Path | None = None) -> Path:
    """Locate ``crates/qd-runtime/src``, or raise :class:`RustSourceMissing`."""
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        candidate = parent / "crates" / "qd-runtime" / "src"
        if candidate.is_dir():
            return candidate
    raise RustSourceMissing(
        "crates/qd-runtime/src was not found above "
        f"{here}. The Rust tree is absent (an installed wheel, or a partial "
        "checkout), so the contract could not be re-derived from it."
    )


def snake_case(camel: str) -> str:
    """serde's ``rename_all = "snake_case"`` for a CamelCase variant name.

    Matches serde_derive: an underscore before every uppercase character except the
    first, then lowercase throughout. Digits are left where they are, which is what
    makes ``ContextNotBase64`` -> ``context_not_base64`` rather than
    ``context_not_base_64``.
    """
    out: list[str] = []
    for i, ch in enumerate(camel):
        if ch.isupper():
            if i != 0:
                out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def _strip_noise(body: str) -> str:
    """Remove doc comments, line comments and attributes from a declaration body.

    Attributes carry braces inside string literals (``#[error("slot `{slot}`: ...")]``),
    so they are removed with the string-aware scanner rather than a regex.
    """
    lines: list[str] = []
    depth = 0
    for raw in body.splitlines():
        line = raw
        stripped = line.strip()
        if stripped.startswith("///") or stripped.startswith("//!"):
            continue
        if depth == 0:
            comment = _find_line_comment(line)
            if comment is not None:
                line = line[:comment]
                stripped = line.strip()
            if not stripped:
                continue
        if depth > 0 or stripped.startswith("#["):
            depth += _bracket_delta(line)
            if depth <= 0:
                depth = 0
            continue
        lines.append(line)
    return "\n".join(lines)


def _find_line_comment(line: str) -> int | None:
    """Index of a ``//`` that is not inside a string literal, or None."""
    in_str = False
    escaped = False
    for i, ch in enumerate(line):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
            return i
    return None


def _bracket_delta(line: str) -> int:
    """Net ``[``/``]`` depth of a line, ignoring bracket characters inside strings."""
    delta = 0
    in_str = False
    escaped = False
    for ch in line:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            delta += 1
        elif ch == "]":
            delta -= 1
    return delta


def _balanced_body(src: str, open_index: int, *, what: str) -> str:
    """Text between the ``{`` at ``open_index`` and its match, strings respected."""
    if src[open_index] != "{":
        raise RustParseError(f"{what}: expected a brace at offset {open_index}")
    depth = 0
    in_str = False
    escaped = False
    for i in range(open_index, len(src)):
        ch = src[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[open_index + 1 : i]
    raise RustParseError(f"{what}: unbalanced braces from offset {open_index}")


def _declaration_body(src: str, pattern: str, *, what: str) -> str:
    match = re.search(pattern, src)
    if match is None:
        raise RustParseError(f"{what}: declaration not found (pattern {pattern!r})")
    brace = src.find("{", match.end() - 1)
    if brace < 0:
        raise RustParseError(f"{what}: no body after the declaration")
    return _strip_noise(_balanced_body(src, brace, what=what))


_FIELD = re.compile(r"^(?:pub\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)$", re.DOTALL)


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` at nesting depth zero, respecting brackets and strings.

    Rust field lists nest (``BTreeMap<String, SlotAnswer>``) and the trailing comma
    is optional, so splitting by line or by a bare ``,`` both mis-read real
    declarations.
    """
    parts: list[str] = []
    depth = 0
    in_str = False
    escaped = False
    current: list[str] = []
    for ch in text:
        if in_str:
            current.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _parse_field_list(text: str, what: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in _split_top_level(text):
        m = _FIELD.match(part)
        if m is None:
            raise RustParseError(f"{what}: could not read a field from {part!r}")
        fields[m.group(1)] = _normalise_type(m.group(2))
    return fields


def extract_struct_fields(src: str, name: str) -> dict[str, str]:
    """``pub struct NAME { pub a: T, ... }`` -> ``{"a": "T"}``."""
    what = f"struct {name}"
    body = _declaration_body(src, rf"\bpub\s+struct\s+{re.escape(name)}\b", what=what)
    fields = _parse_field_list(body, what)
    if not fields:
        raise RustParseError(f"{what}: parsed zero fields, which cannot be right")
    return fields


def extract_enum_variants(src: str, name: str) -> dict[str, dict[str, str]]:
    """``pub enum NAME { A, B { x: T } }`` -> ``{"A": {}, "B": {"x": "T"}}``.

    Tuple variants (``Ok(AnswerEnvelope)``) are reported with the single key
    ``"0"`` mapped to the inner type, because an internally tagged newtype variant
    flattens its inner struct's fields onto the wire and the caller needs to know
    which struct that is.
    """
    what = f"enum {name}"
    body = _declaration_body(src, rf"\bpub\s+enum\s+{re.escape(name)}\b", what=what)
    variants: dict[str, dict[str, str]] = {}
    i = 0
    text = body
    while i < len(text):
        m = re.compile(r"\s*([A-Z][A-Za-z0-9_]*)\s*").match(text, i)
        if m is None:
            rest = text[i:].strip()
            if not rest:
                break
            raise RustParseError(f"{what}: could not read a variant at {rest[:80]!r}")
        variant = m.group(1)
        j = m.end()
        if j < len(text) and text[j] == "{":
            inner = _strip_noise(_balanced_body(text, j, what=f"{what}::{variant}"))
            variants[variant] = _parse_field_list(inner, f"{what}::{variant}")
            j = _skip_balanced(text, j)
        elif j < len(text) and text[j] == "(":
            close = _skip_balanced(text, j, opener="(", closer=")")
            variants[variant] = {"0": _normalise_type(text[j + 1 : close - 1])}
            j = close
        else:
            variants[variant] = {}
        comma = text.find(",", j)
        i = (comma + 1) if comma >= 0 else len(text)
    if not variants:
        raise RustParseError(f"{what}: parsed zero variants, which cannot be right")
    return variants


def _skip_balanced(text: str, open_index: int, *, opener: str = "{", closer: str = "}") -> int:
    depth = 0
    in_str = False
    escaped = False
    for i in range(open_index, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i + 1
    raise RustParseError(f"unbalanced {opener}{closer} from offset {open_index}")


def _normalise_type(ty: str) -> str:
    return " ".join(ty.split())


_KIND_ARM = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)::([A-Z][A-Za-z0-9_]*)\s*(?:\{\s*\.\.\s*\}\s*)?=>\s*\"([a-z0-9_]+)\""
)


def extract_kind_arms(src: str, enum_name: str) -> dict[str, str]:
    """The ``kind()`` match arms: ``{"HashMismatch": "hash_mismatch", ...}``.

    The serialized tag and ``kind()`` are two separate declarations in the Rust, so
    this is extracted and cross-checked against :func:`snake_case` rather than
    assumed equal.
    """
    arms = {
        variant: kind
        for owner, variant, kind in _KIND_ARM.findall(src)
        if owner == enum_name
    }
    if not arms:
        raise RustParseError(f"{enum_name}::kind(): parsed zero match arms")
    return arms


def extract_const_u32_slice(src: str, name: str) -> list[int]:
    """``pub const NAME: &[u32] = &[1];`` -> ``[1]``."""
    m = re.search(
        rf"\bpub\s+const\s+{re.escape(name)}\s*:\s*&\[u32\]\s*=\s*&\[([^\]]*)\]", src
    )
    if m is None:
        raise RustParseError(f"const {name}: not found")
    items = [p.strip() for p in m.group(1).split(",") if p.strip()]
    if not items:
        raise RustParseError(f"const {name}: parsed as empty")
    return [int(p) for p in items]


#: Structs whose wire shape this package parses, and the file that declares each.
RESPONSE_STRUCTS: tuple[str, ...] = (
    "AnswerEnvelope",
    "SlotAnswer",
    "SpanValue",
    "RefusalEnvelope",
    "ErrorEnvelope",
    "HashExpectation",
)


def derive_contract(src_dir: Path | None = None) -> dict[str, object]:
    """The whole answer-side contract, read out of the Rust source.

    Returns the same shape :mod:`qd_wire.contract` pins as literals, so the two can
    be compared field for field.
    """
    src_dir = src_dir or rust_src_dir()
    schema = (src_dir / "schema.rs").read_text(encoding="utf-8")
    refusal = (src_dir / "refusal.rs").read_text(encoding="utf-8")

    structs = {name: extract_struct_fields(schema, name) for name in RESPONSE_STRUCTS}

    response = extract_enum_variants(schema, "Response")
    status_tags = {snake_case(v): inner.get("0", "") for v, inner in response.items()}

    refusal_variants = extract_enum_variants(refusal, "Refusal")
    refusal_arms = extract_kind_arms(refusal, "Refusal")
    backend_variants = extract_enum_variants(refusal, "BackendError")
    backend_arms = extract_kind_arms(refusal, "BackendError")

    _cross_check_kinds("Refusal", refusal_variants, refusal_arms)
    _cross_check_kinds("BackendError", backend_variants, backend_arms)

    hash_kinds = extract_enum_variants(refusal, "HashKind")

    return {
        "structs": structs,
        "status_tags": status_tags,
        "refusal_kinds": {refusal_arms[v]: f for v, f in refusal_variants.items()},
        "backend_error_kinds": {backend_arms[v]: f for v, f in backend_variants.items()},
        "hash_kinds": sorted(snake_case(v) for v in hash_kinds),
        "supported_schema_versions": extract_const_u32_slice(
            schema, "SUPPORTED_SCHEMA_VERSIONS"
        ),
        "slot_kinds": sorted(snake_case(v) for v in extract_enum_variants(schema, "SlotKind")),
        "routes": sorted(snake_case(v) for v in extract_enum_variants(schema, "Route")),
    }


def _cross_check_kinds(
    enum_name: str, variants: dict[str, dict[str, str]], arms: dict[str, str]
) -> None:
    """``kind()`` and the serde tag are two declarations; assert they agree.

    A variant whose ``kind()`` string differs from its serde ``rename_all`` tag would
    put one name on the wire and a different one in every log line and test — the
    exact shape of ``GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS``, inside one file.
    """
    missing = sorted(set(variants) - set(arms))
    extra = sorted(set(arms) - set(variants))
    if missing or extra:
        raise RustParseError(
            f"{enum_name}: kind() covers {sorted(arms)} but the enum declares "
            f"{sorted(variants)} (missing from kind(): {missing}; unknown to the enum: {extra})"
        )
    disagree = {v: (arms[v], snake_case(v)) for v in variants if arms[v] != snake_case(v)}
    if disagree:
        raise RustParseError(
            f"{enum_name}: kind() disagrees with the serde snake_case tag for {disagree}. "
            "One variant would be named two things on one wire."
        )


if __name__ == "__main__":  # pragma: no cover - a generator, run by hand
    import json

    print(json.dumps(derive_contract(), indent=4, sort_keys=True))
