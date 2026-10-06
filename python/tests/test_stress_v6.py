"""The pre-registered data-side stress suite's Python half (AUDIT/data-stress-suite-2026-10-06.md).

- S7 (static): no test shares a temp path keyed only on the pid (the lane-4 race shape).
- S10: the loader door, ``qd_data.decisions.load_decision_pool``, and the row constructor it
  feeds (``rewrite_typed_decision``, where the licence is checked).
- S11: ``qd_train.data_access.assert_path_not_held_out`` refuses the held-out sets the
  pre-registration names (natural bugs, TSSB-3M, caller records).
- S12: licence default-deny in ``qd_data.licences``.

Every test is named ``test_s<item>_...`` so a runner selects one item with ``-k s10``. The
Rust half is ``crates/qd-prep/tests/stress_v6.rs``. Nothing here imports torch.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qd_data.config import DataConfig
from qd_data.decisions import DecisionPoolError, load_decision_pool, rewrite_typed_decision
from qd_data.errors import HeldOutViolation, LicenceRefused
from qd_data.licences import _ALIASES, LICENCE_POLICY, LicenceTier, admit_licence, classify
from qd_data.loaders import MalformedRowRefusal
from qd_train.data_access import assert_path_not_held_out

REPO = Path(__file__).resolve().parents[2]
OPEN_JEV = "ZefanCai/Open-Jev-v1.1"
APPLIED = {"allocation": {"state": "applied", "detail": "stress suite"}}


def _line(**over: object) -> dict[str, object]:
    line: dict[str, object] = {
        "id": "openjev:g/0/choice", "source_id": OPEN_JEV, "family_id": "openjev.policy",
        "stratum": "openjev.policy/choice", "split": "train", "group_key": "g",
        "licence": "cc0-1.0",
        "context": "Which candidate satisfies every requirement?\n\n{\"candidates\": []}",
        "slot_name": "answer", "options": ["K11", "P72", "K64"], "gold_option": "P72",
        "gold_noul": False, "label_basis": "hard",
    }
    line.update(over)
    return line


def _write_pool(root: Path, lines: list[dict[str, object]], **manifest: object) -> Path:
    root.mkdir(parents=True)
    body = "".join(json.dumps(x) + "\n" for x in lines).encode("utf-8")
    (root / "examples.jsonl").write_bytes(body)
    doc: dict[str, object] = {
        "schema": "qd-decisions/v1", "mode": "build", "examples": len(lines),
        "examples_sha256": hashlib.sha256(body).hexdigest(),
    }
    doc.update(manifest)
    (root / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------------------------
# S7, static: the lane-4 race shape. A temp path whose only varying part is the pid is shared by
# every test in one process; it is a race when two tests reach it (the same literal in two
# places, or a helper with two or more callers that adds no unique part of its own). A helper
# that draws a counter (``fetch_add``) is the lane-4 fix and is not flagged.

_RUST_PID = re.compile(r"process::id\(\)")
_RUST_FORMAT = re.compile(r'format!\(\s*"((?:[^"\\]|\\.)*)"')
_RUST_FN = re.compile(r"\bfn\s+(\w+)")
_PY_PID_FSTRING = re.compile(r"""f(["'])((?:(?!\1).)*\{os\.getpid\(\)\}(?:(?!\1).)*)\1""")


def _placeholders(literal: str) -> list[str]:
    return re.findall(r"\{[^{}]*\}", literal.replace("{{", "").replace("}}", ""))


def _rust_pid_sites(path: Path) -> list[tuple[str, str, int]]:
    """``(literal, enclosing fn, offset)`` of every pid-only ``format!`` in ``path``."""
    text = path.read_text(encoding="utf-8")
    sites = []
    for m in _RUST_PID.finditer(text):
        window = text[max(0, m.start() - 400): m.start()]
        formats = list(_RUST_FORMAT.finditer(window))
        if not formats:
            continue
        literal = formats[-1].group(1)
        if _placeholders(literal) != ["{}"]:
            continue
        fns = list(_RUST_FN.finditer(text[: m.start()]))
        sites.append((literal, fns[-1].group(1) if fns else "", m.start()))
    return sites


def _is_test_fn(text: str, name: str) -> bool:
    m = re.search(rf"#\[test\][^\n]*\n(?:\s*#\[[^\n]*\n)*\s*fn\s+{re.escape(name)}\b", text)
    return m is not None


def _fn_body(text: str, name: str) -> str:
    start = re.search(rf"\bfn\s+{re.escape(name)}\b", text)
    if start is None:
        return ""
    end = re.search(r"\n\s*(?:pub(?:\([^)]*\))?\s+)?fn\s", text[start.end():])
    return text[start.start(): start.end() + (end.start() if end else len(text))]


def test_s7_no_test_shares_a_temp_path_keyed_only_on_the_pid() -> None:
    rust_files = sorted((REPO / "crates/qd-prep/src").glob("*.rs")) + sorted(
        (REPO / "crates/qd-prep/tests").glob("*.rs")
    )
    py_files = sorted((REPO / "python/tests").glob("*.py"))
    assert len(rust_files) > 10 and len(py_files) > 10, "the scan found no files to scan"

    sites: list[tuple[Path, str, str]] = []
    for path in rust_files:
        sites.extend((path, lit, fn) for lit, fn, _ in _rust_pid_sites(path))
    # Non-vacuity: the scan must see the one pid-only site known when this test was written
    # (`decisions.rs`'s line-reader test), or its pattern is broken, not the tree clean.
    assert any(lit == "qd-prep-decisions-{}" for _, lit, _ in sites), sites

    shared: list[str] = []
    by_literal: dict[str, list[tuple[Path, str]]] = {}
    for path, lit, fn in sites:
        by_literal.setdefault(lit, []).append((path, fn))
    for lit, where in by_literal.items():
        if len(where) > 1:
            shared.append(f"{lit!r} in {len(where)} places: {where}")
    for path, lit, fn in sites:
        text = path.read_text(encoding="utf-8")
        if not fn or _is_test_fn(text, fn):
            continue
        callers = len(re.findall(rf"\b{re.escape(fn)}\(", text)) - 1
        if callers >= 2 and "fetch_add" not in _fn_body(text, fn):
            shared.append(f"{path.name}: helper {fn} ({lit!r}) has {callers} callers")

    py_literals: dict[str, list[str]] = {}
    for path in py_files:
        text = path.read_text(encoding="utf-8")
        for m in _PY_PID_FSTRING.finditer(text):
            if len(_placeholders(m.group(2))) == 1:
                py_literals.setdefault(m.group(2), []).append(path.name)
    shared.extend(f"{lit!r} in {where}" for lit, where in py_literals.items() if len(where) > 1)

    assert not shared, "temp paths keyed only on the pid, reached by more than one test:\n" + (
        "\n".join(shared)
    )


# ---------------------------------------------------------------------------------------------
# S10: the loader door. Reused (not repeated here): test_decisions_pool.py's
# test_the_loader_refuses_a_pool_its_manifest_does_not_describe (the examples sha256 mismatch,
# and a synth or convert pool whose allocation is not applied) and
# test_the_loader_refuses_a_family_of_another_source.


@pytest.mark.parametrize(
    "over",
    [
        # A synth pool's row, as `qd-prep synth` writes it: its source is not registered.
        {"id": "synth-email:email.category:0", "source_id": "lappi/synth-email",
         "family_id": "email.category", "stratum": "email.category/category",
         "licence": "synthetic-by-rule", "slot_name": "category"},
        # A registered source naming a family it does not own.
        {"family_id": "email.category", "stratum": "email.category/category"},
        # A converted pool's row: When2Call's family, not registered as a decision family.
        {"id": "convert:when2call:0", "source_id": "nvidia/When2Call",
         "family_id": "when2call.tool_select", "stratum": "when2call.tool_select/x"},
    ],
)
def test_s10_a_pool_row_of_an_unregistered_family_is_refused(
    tmp_path: Path, over: dict[str, object]
) -> None:
    with pytest.raises(MalformedRowRefusal):
        load_decision_pool(_write_pool(tmp_path / "pool", [_line(**over)], **APPLIED))


@pytest.mark.parametrize(
    "licence", ["synthetic-by-rule", "odc-by-1.0", "oanc", "NOASSERTION", "unknown", ""]
)
def test_s10_an_unregistered_licence_id_is_refused_before_the_row_is_built(
    tmp_path: Path, licence: str
) -> None:
    """``load_decision_pool`` does not check licences; ``rewrite_typed_decision`` (the one
    funnel every pool row goes through) does. Either layer refusing satisfies S10; a row that
    reaches the mixture with an unregistered id fails it."""
    try:
        pool = load_decision_pool(
            _write_pool(tmp_path / "pool", [_line(licence=licence)], **APPLIED)
        )
    except (DecisionPoolError, MalformedRowRefusal, LicenceRefused):
        return
    (raw,) = pool.raw[OPEN_JEV]
    with pytest.raises(LicenceRefused):
        rewrite_typed_decision(raw, family_id=raw.family_id, index=0, config=DataConfig())


@pytest.mark.parametrize(
    "held", ["heldout", "HeldOut", "held_out", "held-out", "heldout/natural-bugs"]
)
def test_s10_a_pool_under_a_held_out_path_is_refused_by_the_marker(
    tmp_path: Path, held: str
) -> None:
    pool = _write_pool(tmp_path / held / "pool", [_line()], **APPLIED)
    with pytest.raises((HeldOutViolation, DecisionPoolError), match=r"(?i)held"):
        load_decision_pool(pool)


def test_s10_the_same_pool_outside_a_held_out_path_loads(tmp_path: Path) -> None:
    """The control for the held-out case: the refusal above is the marker's, not the pool's."""
    pool = load_decision_pool(_write_pool(tmp_path / "pool", [_line()], **APPLIED))
    assert [r.example_id for r in pool.raw[OPEN_JEV]] == ["openjev:g/0/choice"]


# ---------------------------------------------------------------------------------------------
# S11: qd-train's path check (the Python door) refuses the held-out sets the pre-registration
# names. The Rust door is crates/qd-train/tests/{door,caller_records_are_held_out}.rs.

CALLER_RECORDS = Path.home() / "Library/Application Support/Lappi/heldout/caller-records"


@pytest.mark.parametrize(
    "rel",
    [
        "heldout/natural-bugs/natural-bugs.jsonl",
        "heldout/tssb-3m/tssb.jsonl",
        "v6/HeldOut/tssb-3m/rows.jsonl",
        "data/held_out/natural-bugs/report.json",
    ],
)
def test_s11_qd_train_refuses_the_held_out_sets(tmp_path: Path, rel: str) -> None:
    with pytest.raises(HeldOutViolation):
        assert_path_not_held_out(tmp_path / rel, config=DataConfig(), repo_root=REPO)


def test_s11_qd_train_refuses_a_caller_record_file() -> None:
    with pytest.raises(HeldOutViolation):
        assert_path_not_held_out(
            CALLER_RECORDS / "devcouncil" / "2026-10-06.jsonl", config=DataConfig(), repo_root=REPO
        )


def test_s11_qd_train_admits_the_same_file_outside_a_held_out_path(tmp_path: Path) -> None:
    """The control: the refusals above come from the marker, not from a check that refuses
    everything."""
    assert_path_not_held_out(tmp_path / "train" / "rows.jsonl", config=DataConfig(), repo_root=REPO)


# ---------------------------------------------------------------------------------------------
# S12: licence default-deny. Reused: test_decisions_pool.py's
# test_a_non_commercial_row_licence_is_refused, and crates/qd-prep/tests/convert_licence_parity.rs
# (Rust against this module on 74 strings).


@pytest.mark.parametrize("raw", ["unknown", "Unknown", "NOASSERTION", "noassertion", "", "   ",
                                 "Made-Up-Licence-9", "other"])
def test_s12_unknown_noassertion_and_empty_licences_are_denied(raw: str) -> None:
    with pytest.raises(LicenceRefused):
        admit_licence(raw, config=DataConfig().licence, source="stress")


def _variants(raw: str) -> list[str]:
    return [
        raw.upper(),
        f"  {raw}\t",
        raw.replace(" ", " \x1f  "),
        "".join(c.upper() if i % 2 == 0 else c for i, c in enumerate(raw)),
    ]


@pytest.mark.parametrize("raw", sorted(set(LICENCE_POLICY) | set(_ALIASES)))
def test_s12_case_and_whitespace_variants_resolve_to_the_same_id(raw: str) -> None:
    want = classify(raw)
    assert want.licence_id in LICENCE_POLICY, raw
    for v in _variants(raw):
        got = classify(v)
        assert (got.licence_id, got.tier) == (want.licence_id, want.tier), v


@settings(max_examples=500, deadline=None)
@given(st.text(max_size=40))
def test_s12_no_string_passes_unclassified(raw: str) -> None:
    """Whatever the string, ``admit_licence`` returns a table entry on the allowlist or raises
    ``LicenceRefused``: nothing is admitted that the table does not classify as ``allow``."""
    try:
        policy = admit_licence(raw, config=DataConfig().licence, source="stress")
    except LicenceRefused:
        return
    assert policy.tier is LicenceTier.ALLOW, (raw, policy)
    assert LICENCE_POLICY.get(policy.licence_id) == policy, (raw, policy)
