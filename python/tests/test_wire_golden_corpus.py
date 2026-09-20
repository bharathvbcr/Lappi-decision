"""The golden corpus, and the live runtime, read from the Python side.

Two independent cross-language checks live here, each with its own tri-state.

**The corpus** (``fixtures/wire/``) is the standing seam. The XLANG-RS lane owns a
Rust generator that serialises every envelope variant into it; this file parses
them back field for field. When the directory is absent the result is
``NotRun`` **with a reason** — never a pass, and never a bare ``skip`` that reads
as one in a summary line.

**The live binary** (``target/debug/qd``) is the check that is available *today*,
before the corpus exists. It drives the real Rust encoder into the real Python
parser in one process tree, which is the same shape as
``crates/qd-runtime/tests/wire_context_crosslang.rs`` in the other direction. It is
``NotRun`` when the binary is missing, and — importantly — also when the binary is
**older than the Rust sources**, because a stale binary answers questions about
code that no longer exists and would report agreement with a contract nobody is
building against.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from qd_data.schema import encode_context
from qd_train.tristate import NotRun, Ran
from qd_wire.answer import AnswerEnvelope, ChoiceValue, ScoreValue, SpanValue
from qd_wire.contract import BACKEND_ERROR_KINDS, REFUSAL_KINDS, SUPPORTED_SCHEMA_VERSIONS
from qd_wire.corpus import MANIFEST_NAME, fixture_dir, load_corpus
from qd_wire.errors import WireParseError
from qd_wire.refusal import ErrorEnvelope, RefusalEnvelope
from qd_wire.response import CallerReading, caller_reading, parse_response_line

REPO = Path(__file__).resolve().parents[2]
QD_BINARY = REPO / "target" / "debug" / "qd"
RUST_SRC = REPO / "crates" / "qd-runtime" / "src"
#: Every subprocess is bounded. An unbounded probe in a test suite is a hang.
QD_TIMEOUT_S = 60


# ==========================================================================================
# 1. the golden corpus
# ==========================================================================================


#: Point the corpus tests at a directory other than ``fixtures/wire/``.
#:
#: The generator is the XLANG-RS lane's (``qd_runtime::fixtures::write_corpus``), and
#: while that lane is mid-change its own gate is "the corpus is absent". Writing into
#: the repo's ``fixtures/wire/`` from here would destroy that gate's evidence, so this
#: override exists to run the same assertions against a corpus generated elsewhere —
#: for instance into a scratch directory by a throwaway consumer of the same public
#: function. The directory actually read is carried on :class:`CorpusLoad` and named in
#: every skip message, so an override can never be mistaken for the real corpus.
CORPUS_DIR_ENV = "QD_WIRE_CORPUS_DIR"


@pytest.fixture(scope="module")
def corpus():
    override = os.environ.get(CORPUS_DIR_ENV)
    return load_corpus(Path(override) if override else None)


def test_without_an_override_the_corpus_read_is_the_repository_one():
    """An override must never be able to masquerade as ``fixtures/wire/``."""
    if os.environ.get(CORPUS_DIR_ENV):
        pytest.skip(f"NOT RUN — {CORPUS_DIR_ENV} is set to {os.environ[CORPUS_DIR_ENV]!r}")
    assert load_corpus().directory == REPO / "fixtures" / "wire"


def test_the_corpus_is_either_parsed_or_recorded_as_not_run(corpus):
    """The tri-state itself. An absent corpus must not render like a passing one."""
    if isinstance(corpus.state, NotRun):
        assert not hasattr(corpus.state, "passed"), (
            "NotRun grew a `passed` attribute, so an unread corpus now renders "
            "identically to one that was read and matched"
        )
        assert corpus.state.reason.strip()
        pytest.skip(f"NOT RUN — {corpus.state.reason}")
    assert isinstance(corpus.state, Ran)
    assert corpus.state.passed is True
    assert corpus.state.is_complete_coverage, (
        "the corpus was read but coverage is not stated as complete; a sample "
        "presented as the whole is the failure this field exists to prevent"
    )


def test_no_file_in_the_corpus_was_silently_skipped(corpus):
    """A fixture the loader ignored is a fixture nobody asserted.

    Both directions: a file ``index.json`` names and the directory lacks, and a file
    the directory holds that nothing names. The generator removes strays itself, so a
    stray here means the two halves are looking at different trees.
    """
    if isinstance(corpus.state, NotRun):
        pytest.skip(f"NOT RUN — {corpus.state.reason}")
    assert corpus.missing == (), (
        f"{list(corpus.missing)} are named by {corpus.directory}/index.json and are not "
        "on disk. The corpus is incomplete, so a green parse of what remains would be a "
        "capped sample reported as full coverage."
    )
    assert corpus.strays == (), (
        f"{list(corpus.strays)} sit in {corpus.directory} and the manifest names none of "
        "them. A leftover from a renamed variant teaches this lane a shape the runtime "
        "no longer emits."
    )


def test_the_manifest_and_the_parsed_corpus_agree_entry_for_entry(corpus):
    """``index.json`` is a claim about the files; this is the check of it."""
    if isinstance(corpus.state, NotRun):
        pytest.skip(f"NOT RUN — {corpus.state.reason}")
    assert corpus.manifest is not None
    assert len(corpus.envelopes) == corpus.manifest.count

    counted: dict[str, int] = {}
    for env in corpus.envelopes:
        status = env.raw["status"]
        assert status == env.entry.status, (
            f"{env.label}: the manifest says status {env.entry.status!r} and the file "
            f"carries {status!r}"
        )
        counted[env.entry.status] = counted.get(env.entry.status, 0) + 1

        declared = env.entry.kind
        if isinstance(env.response, RefusalEnvelope):
            assert declared == env.response.refusal.kind, env.label
        elif isinstance(env.response, ErrorEnvelope):
            assert declared == env.response.error.kind, env.label
        else:
            assert declared is None, f"{env.label}: an answer has no refusal or error kind"

        assert env.entry.note.strip(), f"{env.label}: the manifest row says nothing about it"

    assert counted == corpus.manifest.by_status, (
        f"the manifest's by_status is {corpus.manifest.by_status} and the files parse to "
        f"{counted}"
    )


def test_every_schema_version_in_the_corpus_is_one_this_build_supports(corpus):
    if isinstance(corpus.state, NotRun):
        pytest.skip(f"NOT RUN — {corpus.state.reason}")
    assert corpus.manifest.schema_version in SUPPORTED_SCHEMA_VERSIONS
    for env in corpus.envelopes:
        assert env.response.schema_version in SUPPORTED_SCHEMA_VERSIONS, env.label


def test_every_corpus_envelope_parses_strictly_and_reads_as_exactly_one_thing(corpus):
    """The assertion the Rust -> Python direction never had.

    Strict in both directions: a fixture carrying a field this parser does not know
    fails here, and so does one missing a field it requires. That is the point — when
    the XLANG-RS lane changes the envelope, Python breaks rather than quietly
    accepting the new shape.
    """
    if isinstance(corpus.state, NotRun):
        pytest.skip(f"NOT RUN — {corpus.state.reason}")

    readings = {
        AnswerEnvelope: {CallerReading.MODEL_ANSWERED, CallerReading.MODEL_ABSTAINED},
        RefusalEnvelope: {CallerReading.REQUEST_REFUSED},
        ErrorEnvelope: {CallerReading.BACKEND_FAILED},
    }
    for env in corpus.envelopes:
        expected = readings[type(env.response)]
        if isinstance(env.response, AnswerEnvelope) and not env.response.slots:
            continue  # asserted separately: a slotless ok envelope has no reading
        assert caller_reading(env.response) in expected, env.label


def test_the_corpus_covers_all_three_statuses(corpus):
    """A corpus of twelve answers and no refusals asserts half the contract."""
    if isinstance(corpus.state, NotRun):
        pytest.skip(f"NOT RUN — {corpus.state.reason}")
    statuses = {e.raw.get("status") for e in corpus.envelopes}
    assert {"ok", "refused", "error"} <= statuses, (
        f"the corpus carries {sorted(statuses)}; an envelope variant with no fixture "
        "is a variant no Python code has ever parsed"
    )


def test_the_corpus_exercises_every_refusal_kind_and_every_backend_error_kind(corpus):
    """Complete coverage, carried as two numbers rather than asserted as a feeling.

    ``crates/qd-runtime/src/fixtures.rs`` says the corpus holds "every one of the
    ``Refusal`` variants, by ``kind()``" and the same for ``BackendError``, and its
    own suite checks exhaustiveness against the enums. This is the *other* half of
    that claim: that Python actually parsed one of each, out of real generated bytes.
    A kind with no fixture is a kind no Python code has ever read.
    """
    if isinstance(corpus.state, NotRun):
        pytest.skip(f"NOT RUN — {corpus.state.reason}")

    seen_refusals = {e.response.refusal.kind for e in corpus.by_status("refused")}
    seen_errors = {e.response.error.kind for e in corpus.by_status("error")}
    for kind in seen_refusals:
        assert kind in REFUSAL_KINDS, f"the corpus holds an unknown refusal kind {kind!r}"
    for kind in seen_errors:
        assert kind in BACKEND_ERROR_KINDS, f"the corpus holds an unknown error kind {kind!r}"

    assert seen_refusals == set(REFUSAL_KINDS), (
        f"parsed {len(seen_refusals)} of {len(REFUSAL_KINDS)} refusal kinds; no fixture "
        f"for {sorted(set(REFUSAL_KINDS) - seen_refusals)}"
    )
    assert seen_errors == set(BACKEND_ERROR_KINDS), (
        f"parsed {len(seen_errors)} of {len(BACKEND_ERROR_KINDS)} backend-error kinds; no "
        f"fixture for {sorted(set(BACKEND_ERROR_KINDS) - seen_errors)}"
    )


def test_every_answer_in_the_corpus_holds_the_value_noul_biconditional(corpus):
    """The rule ``docs/schema-api.md`` says it does not specify, checked over real output."""
    if isinstance(corpus.state, NotRun):
        pytest.skip(f"NOT RUN — {corpus.state.reason}")
    answers = corpus.by_status("ok")
    assert answers, "the corpus holds no answers, so the answer half is still unparsed"
    for env in answers:
        for name, slot in env.response.slots.items():
            assert slot.contradiction() is None, f"{env.label}:{name}"
            assert (slot.value is None) is slot.noul, f"{env.label}:{name}"
            if slot.noul:
                assert slot.conformal_set is None, (
                    f"{env.label}:{name} abstained but carries a conformal set; "
                    "`SlotAnswer::abstained` sets it to None on the Rust side"
                )


def test_the_envelope_degraded_flag_agrees_with_the_slots_in_every_answer(corpus):
    """``AnswerEnvelope.degraded`` is documented as "true if any slot is degraded".

    Asserted over the corpus rather than enforced in the parser: it is a property of
    how ``crate::answer`` fills the envelope, not a term the wire format pins, and a
    parser that refused a disagreeing envelope would be inventing one.
    """
    if isinstance(corpus.state, NotRun):
        pytest.skip(f"NOT RUN — {corpus.state.reason}")
    for env in corpus.by_status("ok"):
        slots = env.response.slots.values()
        if not slots:
            continue
        assert env.response.degraded == any(s.degraded for s in slots), env.label


# -- the tri-state and the coverage checks, on corpora built for the purpose ------------------
#
# These do not depend on `fixtures/wire/` being populated *now*: they build the damaged
# cases in a tmp directory. The absent-corpus branch in particular must be testable
# whether or not the generator has run, because "the corpus happens to exist today" is
# not a reason to stop checking that its absence is reported honestly.


def test_an_absent_corpus_is_not_run_and_carries_no_passed_field(tmp_path):
    load = load_corpus(tmp_path / "nothing-here")
    assert isinstance(load.state, NotRun), load.state
    assert not hasattr(load.state, "passed"), (
        "NotRun grew a `passed` attribute; an absent corpus now renders identically to "
        "one that was read and matched, which is the exact failure Rule 5 names"
    )
    assert load.ran is False
    assert "does not exist" in load.state.reason
    assert load.envelopes == ()


def test_an_empty_corpus_directory_is_not_run(tmp_path):
    empty = tmp_path / "wire"
    empty.mkdir()
    load = load_corpus(empty)
    assert isinstance(load.state, NotRun)
    assert not hasattr(load.state, "passed")


def test_a_corpus_directory_without_a_manifest_is_not_run(tmp_path):
    """Files with no ``index.json`` cannot be proved complete, so they are not a pass."""
    loose = tmp_path / "wire"
    loose.mkdir()
    (loose / "answer-something.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "schema_version": 1,
                "backend": "b",
                "degraded": False,
                "slots": {},
            }
        )
    )
    load = load_corpus(loose)
    assert isinstance(load.state, NotRun), load.state
    assert not hasattr(load.state, "passed")
    assert MANIFEST_NAME in load.state.reason


def _copy_real_corpus(tmp_path):
    source = load_corpus().directory
    if not source.is_dir() or not (source / MANIFEST_NAME).is_file():
        pytest.skip(
            f"NOT RUN — no generated corpus at {source} to damage; run "
            "`QD_WRITE_FIXTURES=1 cargo test -p qd-runtime --test wire_fixtures`"
        )
    dst = tmp_path / "wire"
    shutil.copytree(source, dst)
    return dst


def test_a_corpus_missing_a_file_its_manifest_names_is_caught(tmp_path):
    """The failure a glob-driven loader would report as a smaller corpus that passed."""
    dst = _copy_real_corpus(tmp_path)
    manifest = json.loads((dst / MANIFEST_NAME).read_text())
    victim = manifest["entries"][0]["file"]
    (dst / victim).unlink()

    load = load_corpus(dst)
    assert load.missing == (victim,), load.missing
    assert isinstance(load.state, Ran)
    assert load.state.passed is False
    assert load.state.n < load.state.n_total, (
        "coverage must carry both numbers, so a partial read cannot read as the whole"
    )


def test_a_stray_file_in_the_corpus_is_caught(tmp_path):
    """A leftover from a renamed variant teaches this lane a shape the runtime stopped emitting."""
    dst = _copy_real_corpus(tmp_path)
    (dst / "answer-from-a-previous-build.json").write_text('{"status": "ok"}\n')

    load = load_corpus(dst)
    assert load.strays == ("answer-from-a-previous-build.json",), load.strays
    assert isinstance(load.state, Ran)
    assert load.state.passed is False


def test_a_manifest_that_disagrees_with_itself_is_refused(tmp_path):
    """``count`` and ``entries`` are two claims; neither is a coverage total if they differ."""
    dst = _copy_real_corpus(tmp_path)
    manifest = json.loads((dst / MANIFEST_NAME).read_text())
    manifest["count"] += 1
    (dst / MANIFEST_NAME).write_text(json.dumps(manifest))

    with pytest.raises(WireParseError, match="disagrees with itself"):
        load_corpus(dst)


def test_a_fixture_with_a_field_this_parser_does_not_know_is_refused(tmp_path):
    """The whole point of the corpus: Python breaks when the envelope grows a field."""
    dst = _copy_real_corpus(tmp_path)
    manifest = json.loads((dst / MANIFEST_NAME).read_text())
    victim = next(e["file"] for e in manifest["entries"] if e["status"] == "ok")
    doc = json.loads((dst / victim).read_text())
    doc["latency_ms"] = 12
    (dst / victim).write_text(json.dumps(doc))

    with pytest.raises(WireParseError, match="unknown field"):
        load_corpus(dst)


def test_a_fixture_missing_a_field_this_parser_requires_is_refused(tmp_path):
    dst = _copy_real_corpus(tmp_path)
    manifest = json.loads((dst / MANIFEST_NAME).read_text())
    victim = next(e["file"] for e in manifest["entries"] if e["status"] == "ok")
    doc = json.loads((dst / victim).read_text())
    del doc["backend"]
    (dst / victim).write_text(json.dumps(doc))

    with pytest.raises(WireParseError, match="missing required field"):
        load_corpus(dst)


def test_the_corpus_directory_and_manifest_are_where_both_lanes_agree_they_are():
    """One path and one manifest name, so the two halves cannot read different trees.

    ``fixtures.rs`` declares ``CORPUS_DIR = "fixtures/wire"`` and
    ``MANIFEST = "index.json"``; this asserts the Python constants still match the
    Rust literals rather than a memory of them.
    """
    assert fixture_dir() == REPO / "fixtures" / "wire"
    fixtures_rs = (RUST_SRC / "fixtures.rs").read_text(encoding="utf-8")
    assert 'CORPUS_DIR: &str = "fixtures/wire"' in fixtures_rs, (
        "crates/qd-runtime/src/fixtures.rs no longer writes to fixtures/wire; "
        "qd_wire.corpus.CORPUS_DIR is reading somewhere the generator does not write"
    )
    assert f'MANIFEST: &str = "{MANIFEST_NAME}"' in fixtures_rs, (
        f"the generator's manifest is no longer called {MANIFEST_NAME}"
    )


# ==========================================================================================
# 2. the live runtime — available before the corpus exists
# ==========================================================================================

SRC = b"fn add(a: u32, b: u32) -> u32 {\n    a - b\n}\n"


def _request(**over):
    body = {
        "schema_version": 1,
        "task": "devcouncil.verdict",
        "context_b64": encode_context(SRC),
        "context_len": len(SRC),
        "question": "Does this diff implement what the commit message claims?",
        "slots": [
            {"name": "verdict", "type": "choice", "options": ["stub", "logic", "clean"]},
            {"name": "severity", "type": "score", "bins": 5},
            {"name": "evidence", "type": "span"},
        ],
        "route": "generic",
    }
    for key, value in over.items():
        if value is None:
            body.pop(key, None)
        else:
            body[key] = value
    return body


def _live_state():
    """Whether the live probe may run, as a tri-state rather than a boolean."""
    if not QD_BINARY.exists():
        return NotRun(
            reason=(
                f"{QD_BINARY} does not exist; build it with `cargo build -p qd-runtime "
                "--bin qd` to assert the Rust encoder against this parser"
            )
        )
    if not RUST_SRC.is_dir():
        return NotRun(reason=f"{RUST_SRC} is absent, so the binary's freshness cannot be judged")
    built = QD_BINARY.stat().st_mtime
    newer = sorted(p.name for p in RUST_SRC.rglob("*.rs") if p.stat().st_mtime > built)
    if newer:
        return NotRun(
            reason=(
                f"{QD_BINARY.name} was built before {newer} changed. A stale binary "
                "answers for code that no longer exists, and agreement with it would be "
                "agreement with a contract nobody is building against. Rebuild with "
                "`cargo build -p qd-runtime --bin qd`."
            )
        )
    return Ran(passed=True, detail=f"{QD_BINARY.name} is newer than every .rs under {RUST_SRC}")


def _ask(payload, *, reference=True):
    cmd = [str(QD_BINARY), "oneshot"] + (["--reference-backend"] if reference else [])
    proc = subprocess.run(
        cmd,
        input=json.dumps(payload).encode(),
        capture_output=True,
        timeout=QD_TIMEOUT_S,
        check=False,
    )
    assert proc.stdout.strip(), (
        f"`{' '.join(cmd)}` wrote nothing to stdout (exit {proc.returncode}): "
        f"{proc.stderr.decode('utf-8', 'replace')[:400]}"
    )
    return parse_response_line(proc.stdout, path="qd oneshot")


def test_the_live_probe_is_either_run_or_recorded_as_not_run():
    state = _live_state()
    if isinstance(state, NotRun):
        assert not hasattr(state, "passed")
        pytest.skip(f"NOT RUN — {state.reason}")
    assert state.passed is True


@pytest.fixture(scope="module")
def live():
    state = _live_state()
    if isinstance(state, NotRun):
        pytest.skip(f"NOT RUN — {state.reason}")
    return state


def test_a_real_answer_from_the_real_encoder_parses_field_for_field(live):
    """The Rust encoder driven into the Python parser, in one process tree.

    This is the assertion ``GAP-XLANG-NO-PY-ANSWER-PARSER`` says could not exist.
    Nothing here is a literal copied from the Rust source: the bytes come from the
    binary, and a field it emits that this parser does not know fails the parse.
    """
    response = _ask(_request())
    assert isinstance(response, AnswerEnvelope), response
    assert response.schema_version == 1
    assert response.backend, "an answer must name the backend that produced it"
    assert set(response.slots) == {"verdict", "severity", "evidence"}
    for name, answer in response.slots.items():
        assert answer.contradiction() is None, name
    assert caller_reading(response) in {
        CallerReading.MODEL_ANSWERED,
        CallerReading.MODEL_ABSTAINED,
    }


def test_the_reference_backend_flags_every_answer_degraded(live):
    """It is not a model. ``degraded`` is how an answer says so, and a caller reads it."""
    response = _ask(_request())
    assert isinstance(response, AnswerEnvelope)
    assert response.degraded is True
    assert all(s.degraded for s in response.slots.values()), (
        "the envelope is degraded but a slot is not; `AnswerEnvelope.degraded` is "
        "documented as true when any slot is, and qd_wire reads both"
    )


def test_each_slot_kind_answers_with_the_value_shape_its_kind_implies(live):
    """choice -> string, score -> integer, span -> ``{start_line, end_line}``.

    Asserted against the live encoder rather than against this parser's own opinion:
    a ``span`` answered as ``[41, 47]`` would be ``GAP-XLANG-SPAN-THREE-SPELLINGS``
    becoming live, and this is what would say so.
    """
    expected = {"verdict": ChoiceValue, "severity": ScoreValue, "evidence": SpanValue}
    seen: dict[str, type] = {}
    for i in range(40):
        response = _ask(_request(question=f"probe {i}"))
        assert isinstance(response, AnswerEnvelope)
        for name, answer in response.slots.items():
            if answer.value is not None:
                seen.setdefault(name, type(answer.value))
        if set(seen) == set(expected):
            break
    assert set(seen) == set(expected), (
        f"only {sorted(seen)} of {sorted(expected)} produced a non-abstaining answer in "
        "40 probes; the remaining slot kinds' value shapes were not exercised"
    )
    assert seen == expected


@pytest.mark.parametrize(
    ("label", "override", "expected_kind"),
    [
        ("empty slots", {"slots": []}, "empty_slots"),
        ("unknown schema_version", {"schema_version": 99}, "unknown_schema_version"),
        ("bins below the floor", {"slots": [{"name": "s", "type": "score", "bins": 1}]},
         "bins_out_of_range"),
        ("context_b64 as an integer array", {"context_b64": [102, 110]}, "context_not_bytes"),
        ("context_len absent", {"context_len": None}, "context_len_missing"),
        ("context_len disagrees", {"context_len": 999}, "context_length_mismatch"),
        ("unknown route", {"route": "turbo"}, "unknown_route"),
        ("empty task", {"task": "   "}, "empty_task"),
    ],
)
def test_real_refusals_parse_and_name_the_check_the_contract_names(
    live, label, override, expected_kind
):
    response = _ask(_request(**override))
    assert isinstance(response, RefusalEnvelope), f"{label}: {response}"
    assert response.refusal.kind == expected_kind, label
    assert response.message, "every refusal renders both compared values for a human"
    assert caller_reading(response) is CallerReading.REQUEST_REFUSED


def test_a_real_backend_error_parses_and_is_not_an_abstention(live):
    """No ``--reference-backend``: the honest reply is a typed error, never a ``noul``."""
    response = _ask(_request(), reference=False)
    assert isinstance(response, ErrorEnvelope), response
    assert response.error.kind == "unavailable"
    assert caller_reading(response) is CallerReading.BACKEND_FAILED
    assert caller_reading(response) is not CallerReading.MODEL_ABSTAINED
