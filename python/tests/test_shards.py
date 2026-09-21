"""S4: the writer, the reader, the ordering guarantee and the padding-waste gate.

The tests that matter most here are not the round-trip ones. They are:

* :func:`test_rows_from_another_split_are_refused_even_with_a_training_manifest` --
  ``write_shards`` takes rows *and* a manifest path, and rule 3 only guards the manifest.
  Without the content-hash match, held-out rows tokenized beside a training manifest would
  pass every check in the repo.
* :func:`test_a_dropped_token_propagates_and_is_not_substituted` -- the S2<->S4 cross-lane
  check. It must escape ``write_shards``, not be absorbed by it.
* :func:`test_batch_order_is_a_pure_function_of_seed_epoch_and_batch_tokens` -- S5's
  bit-exact resume reconstructs a position from ``(epoch, index)``, which is only sound
  while asking twice gives the same answer twice.
* :func:`test_every_batch_row_holds_exactly_one_sequence` -- SAFETY-2's read-side half.
  ``ShardHeader`` refusing ``packed=True`` stops a header from *claiming* packing; this is
  what stops the reader from doing it anyway.
* :func:`test_the_gate_describes_the_padding_the_reader_actually_pays` -- a gate measured
  against a padding scheme the reader does not use is a number about nothing.

The tokenizer is injected as a plain ``Callable[[str], list[int]]`` and is byte-level, so
none of this needs torch or transformers -- neither is in the repo venv.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from data_fixtures import small_corpus

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.errors import ContextTooLargeRefusal, EmptyContextRefusal, HeldOutViolation
from qd_data.fingerprint import code_fingerprint
from qd_data.manifest import Manifest, build_manifests
from qd_data.mixture import _line_span as mixture_line_span
from qd_data.mixture import build_mixture
from qd_data.render import DEFAULT_CAPS, ESCAPE_WORST_CASE_GROWTH, RenderCaps, render
from qd_data.rows import DataRow
from qd_data.schema import Request, SpanSlot
from qd_data.split import HELD_OUT, split
from qd_train import artifacts
from qd_train import shards as shards_module
from qd_train.artifacts import (
    MAX_PADDING_WASTE,
    NO_SPAN,
    SLOT_CHOICE,
    SLOT_SCORE,
    SLOT_SPAN,
    SPAN_ABSTAIN,
    Batch,
    RemapTable,
    ShardContractViolation,
    ShardHeader,
    TokenNotInRemap,
    assign_buckets,
    bucket_for,
    line_start_indices,
    padding_waste,
)
from qd_train.byte_context import line_starts as byte_line_starts
from qd_train.shards import (
    CONTRADICTION_NAME,
    COVERAGE_NAME,
    HEADER_NAME,
    MAX_POSITIONS_PER_BATCH,
    MAX_ROWS_PER_BATCH,
    PAD_ID,
    REMAP_NAME,
    SPAN_CHECK_NAME,
    SUPERVISION_NAME,
    TOKENS_NAME,
    SequenceSpec,
    ShardReader,
    UnencodableGold,
    choose_buckets,
    corpus_contradictions,
    line_starts,
    training_texts,
    write_shards,
)
from qd_train.tristate import NotRun, Ran

SOURCE_VOCAB = 512
TOKENIZER_HASH = "tokhash-0123456789abcdef"
#: `python/`, so a second process imports the same `qd_train` this one does.
REPO_PYTHON = Path(__file__).resolve().parents[1]
#: The repository, for the AUDIT artifacts a test measures against.
REPO_ROOT = Path(__file__).resolve().parents[2]


# -- the injected tokenizer and a remap over it ----------------------------------------


def byte_tokenize(text: str) -> list[int]:
    """A real tokenizer's shape without a real tokenizer: UTF-8 bytes as ids."""
    return list(text.encode("utf-8"))


def byte_offsets(text: str) -> list[tuple[int, int]]:
    """Character offsets for ``byte_tokenize``, one entry per byte token.

    A multi-byte character yields several tokens that all report that character's span,
    which is the honest answer: every one of those bytes is inside that character. Exact
    for this tokenizer, which is what makes the line-to-token tests sharp -- a real BPE
    tokenizer's offsets are its own business, and that is a gap, not something this fixture
    can stand in for.
    """
    out: list[tuple[int, int]] = []
    for ci, ch in enumerate(text):
        out.extend((ci, ci + 1) for _ in ch.encode("utf-8"))
    return out


def byte_decode(ids: Sequence[int]) -> str:
    """``tokenizer.decode`` for ``byte_tokenize``: ids back to the text they stand for.

    Exact, and *therefore* unable to fail for the reason the decode check exists -- in byte
    space a line start is a byte offset and no token straddles a line boundary. It is here
    so the check has a happy path to run on. The check's teeth are shown by the three
    tests that wrap it in a *dishonest* decode:
    :func:`test_a_token_that_does_not_decode_to_its_claimed_characters_is_refused`,
    :func:`test_a_recorded_line_start_that_is_not_one_in_the_decoded_text_is_refused` and
    :func:`test_ids_that_do_not_round_trip_to_their_text_are_refused`.
    """
    return bytes(int(i) for i in ids).decode("utf-8", errors="strict")


def _collapsing_tokenize(text: str) -> list[int]:
    """Two tokens, the first covering nearly the whole text.

    Stands in for the BPE case this writer refuses: a tokenization in which several context
    line starts fall inside one token, so the pointer head cannot tell those lines apart.
    """
    return [65, 66]


def _collapsing_offsets(text: str) -> list[tuple[int, int]]:
    split = max(1, len(text) - 1)
    return [(0, split), (split, len(text))]


def byte_remap(*, drop: frozenset[int] = frozenset()) -> RemapTable:
    """A remap over the byte vocabulary, optionally dropping some ids.

    ``drop`` is what makes the S2<->S4 cross-lane failure reproducible: a remap built over
    a corpus that did not contain some byte, applied to one that does.
    """
    old_to_new = np.full(SOURCE_VOCAB, -1, dtype=np.int32)
    kept = [i for i in range(256) if i not in drop]
    for new, old in enumerate(kept):
        old_to_new[old] = new
    return RemapTable(
        old_to_new=old_to_new,
        new_to_old=np.asarray(kept, dtype=np.int32),
        tokenizer_hash=TOKENIZER_HASH,
        special_ids=(),
    )


@dataclass(frozen=True)
class Snapshot:
    root: Path
    config: DataConfig
    paths: dict[str, Path]
    rows: dict[str, tuple[DataRow, ...]]


@pytest.fixture
def snapshot(tmp_path: Path) -> Snapshot:
    """A written snapshot plus the rows behind it, the way the pipeline hands them over."""
    config = DataConfig()
    mixture = build_mixture(small_corpus(24), config=config)
    report = dedupe(list(mixture.rows), config=config)
    split_report = split(report, config=config)
    manifests = build_manifests(
        config=config, mixture=mixture, dedupe_report=report, split_report=split_report
    )
    paths: dict[str, Path] = {}
    for name, manifest in manifests.items():
        directory = tmp_path / "data" / (HELD_OUT if name == HELD_OUT else "pool")
        path = directory / f"{name}.json"
        manifest.write(path)
        paths[name] = path
    return Snapshot(
        root=tmp_path,
        config=config,
        paths=paths,
        rows={k: tuple(v) for k, v in split_report.rows_by_split.items()},
    )


def _write(
    snap: Snapshot,
    split_name: str,
    out: Path,
    *,
    remap: RemapTable | None = None,
    rows: list[DataRow] | None = None,
    **kwargs: object,
) -> ShardHeader:
    return write_shards(
        snap.paths[split_name],
        snap.rows[split_name] if rows is None else rows,
        out_dir=out,
        remap=remap if remap is not None else byte_remap(),
        config=snap.config,
        repo_root=snap.root,
        **{  # type: ignore[arg-type]
            "tokenize": byte_tokenize,
            "token_offsets": byte_offsets,
            **kwargs,
        },
    )


@pytest.fixture
def train_shards(snapshot: Snapshot, tmp_path: Path) -> tuple[Snapshot, Path, ShardHeader]:
    """The train split, written on the default path.

    Deliberately no ``allow_unencodable``: since ``SPAN_ABSTAIN`` landed every row of this
    corpus encodes, so the escape hatch should not be needed and a fixture that kept
    passing it would hide the day it starts being needed again.
    """
    out = tmp_path / "shards" / "train"
    header = _write(snapshot, "train", out)
    return snapshot, out, header


@pytest.fixture
def reader(train_shards: tuple[Snapshot, Path, ShardHeader]) -> ShardReader:
    snap, out, _ = train_shards
    return ShardReader(out, config=snap.config, repo_root=snap.root)


# -- rule 3, at both doors -------------------------------------------------------------


def test_writing_shards_from_a_held_out_manifest_raises(snapshot: Snapshot, tmp_path: Path) -> None:
    """The manifest door. ``write_shards`` goes through ``open_training_data``."""
    with pytest.raises(HeldOutViolation) as excinfo:
        _write(snapshot, HELD_OUT, tmp_path / "nope")
    assert "rule 3" in str(excinfo.value)
    assert not (tmp_path / "nope").exists(), "refused, and nothing was written"


def test_rows_from_another_split_are_refused_even_with_a_training_manifest(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """The hole the content-hash match exists to close.

    ``open_training_data`` cleared the *manifest*. Nothing about that constrains the rows
    handed in beside it, so a held-out row smuggled into the list would be tokenized into
    a training shard with every check in the repo still green.
    """
    smuggled = [*snapshot.rows["train"], snapshot.rows[HELD_OUT][0]]
    with pytest.raises(ShardContractViolation) as excinfo:
        _write(snapshot, "train", tmp_path / "nope", rows=smuggled, allow_unencodable=True)
    assert "do not appear in this manifest" in str(excinfo.value)


def test_a_manifest_entry_with_no_row_is_refused(snapshot: Snapshot, tmp_path: Path) -> None:
    """The other direction: a header would pin a hash for a corpus bigger than the shard."""
    with pytest.raises(ShardContractViolation) as excinfo:
        _write(
            snapshot,
            "train",
            tmp_path / "nope",
            rows=list(snapshot.rows["train"][:-1]),
            allow_unencodable=True,
        )
    assert "no supplied row" in str(excinfo.value)


def test_the_reader_refuses_a_held_out_shard_set(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """Rule 3 at the shard boundary -- ``assert_shard_trainable`` in the constructor.

    Built by relabelling a real shard set, because that is the shape of the bypass: the
    files are a valid shard set and only the split says what they are.
    """
    snap, out, header = train_shards
    relabelled = ShardHeader(
        split=HELD_OUT,
        data_snapshot_hash=header.data_snapshot_hash,
        tokenizer_hash=header.tokenizer_hash,
        remap_hash=header.remap_hash,
        vocab_size=header.vocab_size,
        n_sequences=header.n_sequences,
        total_tokens=header.total_tokens,
        max_seq_len=header.max_seq_len,
        buckets=header.buckets,
    )
    (out / HEADER_NAME).write_text(json.dumps(relabelled.to_json()), encoding="utf-8")
    with pytest.raises(ShardContractViolation) as excinfo:
        ShardReader(out, config=snap.config, repo_root=snap.root)
    assert "rule 3" in str(excinfo.value).lower()


# -- the S2 <-> S4 cross-lane check ----------------------------------------------------


def test_a_dropped_token_propagates_and_is_not_substituted(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """``RemapTable.encode`` raises; ``write_shards`` must not absorb it.

    Byte 90 is ``Z``, the noul letter, so it is in every rendered option block. A writer
    that caught this and substituted a token would produce a shard set that trains --
    slightly worse loss, no other symptom.
    """
    with pytest.raises(TokenNotInRemap) as excinfo:
        _write(
            snapshot,
            "val",
            tmp_path / "nope",
            remap=byte_remap(drop=frozenset({ord("Z")})),
        )
    assert "dropped by the remap" in str(excinfo.value)


def test_the_header_takes_its_tokenizer_hash_from_the_remap(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """One source of truth. A second argument for it is a second way to disagree."""
    _, _, header = train_shards
    remap = byte_remap()
    assert header.tokenizer_hash == remap.tokenizer_hash == TOKENIZER_HASH
    assert header.remap_hash == remap.remap_hash()
    assert header.vocab_size == remap.vocab_size == 256


def test_the_header_pins_the_manifests_data_snapshot_hash(
    train_shards: tuple[Snapshot, Path, ShardHeader], snapshot: Snapshot
) -> None:
    _, _, header = train_shards
    assert header.data_snapshot_hash == Manifest.read(snapshot.paths["train"]).snapshot_hash()
    assert header.split == "train"


# -- what a sequence is ----------------------------------------------------------------


def test_a_sequence_ends_with_the_gold_answer_letter(snapshot: Snapshot) -> None:
    """The answer is the final token, so ``target_index`` is ``lengths - 2`` uniformly."""
    row = next(r for r in snapshot.rows["train"] if r.family_id == "intent.in_scope")
    specs = training_texts(row, seed=snapshot.config.seed)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.slot_name == "in_scope"
    assert spec.slot_kind == SLOT_CHOICE
    assert spec.span_char_starts is None
    assert spec.text.endswith(("A", "B", "Z"))
    assert "<|qd_answer|>" in spec.text
    # The letter is the one this rendering assigned, not the request's option order.
    rendered = render(row.request, seed=snapshot.config.seed)
    gold = next(g for g in row.gold if g.slot_name == spec.slot_name)
    assert rendered.slots[0].letter_to_value[spec.text[-1]] == str(gold.value)


def test_each_slot_type_gets_its_artifacts_slot_kind(snapshot: Snapshot) -> None:
    """A row the trainer cannot classify is a row it would supervise by guesswork."""
    seen: dict[int, str] = {}
    for row in snapshot.rows["train"]:
        try:
            specs = training_texts(row, seed=snapshot.config.seed)
        except UnencodableGold:
            continue
        for spec in specs:
            seen.setdefault(spec.slot_kind, row.family_id)
    assert set(seen) == {SLOT_CHOICE, SLOT_SCORE, SLOT_SPAN}


def test_an_abstaining_span_is_encoded_not_refused(snapshot: Snapshot) -> None:
    """SQuAD's unanswerable questions are the abstention supervision, not a write failure.

    ``qa.answerability`` is one of the two held-out families, so on this corpus the span
    slot is the only place the model would learn to decline to cite evidence. Before
    ``SPAN_ABSTAIN`` these 5 rows could not be written at all.
    """
    row = next(
        r
        for r in snapshot.rows["train"]
        if r.family_id == "qa.answer_span" and r.gold[0].is_noul
    )
    spec = training_texts(row, seed=snapshot.config.seed)[0]
    assert spec.slot_kind == SLOT_SPAN
    assert spec.span_abstains is True
    assert spec.span_char_starts is None, "an abstention points at no evidence"
    assert spec.line_char_starts, "but it still needs the pointer head's candidate set"


def test_the_whole_corpus_encodes_with_no_escape_hatch(
    reader: ShardReader, snapshot: Snapshot
) -> None:
    """106/106. If this ever drops, that is a finding, not a reason to pass a flag."""
    coverage = reader.coverage
    assert isinstance(coverage, Ran)
    assert coverage.n == coverage.n_total == len(snapshot.rows["train"]) == 106
    assert coverage.passed and coverage.is_complete_coverage


def test_an_unencodable_row_is_still_counted_in_coverage_never_dropped_quietly(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """The exclusion path still carries both numbers, even though nothing uses it today.

    Driven by a tokenizer that collapses the context onto too few tokens to tell its lines
    apart, which is the realistic remaining way a span row becomes unwritable.
    """
    out = tmp_path / "shards" / "collapsed"
    header = _write(
        snapshot,
        "train",
        out,
        tokenize=_collapsing_tokenize,
        token_offsets=_collapsing_offsets,
        buckets=[2],
        allow_unencodable=True,
    )
    assert header.n_sequences < 106, "the span rows could not be written"
    reader = ShardReader(out, config=snapshot.config, repo_root=snapshot.root)
    coverage = reader.coverage
    assert isinstance(coverage, Ran)
    assert coverage.n is not None and coverage.n < coverage.n_total == 106
    assert not coverage.passed and not coverage.is_complete_coverage


def _caps_refusing_the_longest_contexts(rows: Sequence[DataRow]) -> tuple[RenderCaps, int]:
    """Caps tight enough to refuse some of these rows and loose enough to admit others.

    Derived from the corpus rather than written as a constant: a constant that drifts above
    every context refuses nothing and the test passes while measuring nothing, which is the
    vacuous-pass shape this repo keeps finding in its own gates. The count of rows over the
    cap is returned so the caller can assert the case is actually exercised.
    """
    sizes = sorted(len(r.request.context) for r in rows)
    cap = max(sizes[len(sizes) // 2], 512)
    caps = RenderCaps(
        max_context_bytes=cap,
        max_option_bytes=min(512, cap),
        max_rendered_bytes=ESCAPE_WORST_CASE_GROWTH * cap + 8192,
    )
    return caps, sum(1 for n in sizes if n > cap)


def test_a_row_the_renderer_refuses_is_excluded_and_counted_not_fatal(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """A row over the render caps must be counted like any other unwritable row.

    Measured against a real corpus on 2026-09-20: four of this repository's own
    ``(commit, path)`` pairs carry a context above ``RenderCaps.max_context_bytes``, and
    ``ContextTooLargeRefusal`` is a ``QdRefusal``, not an ``UnencodableGold``. The writer's
    ``except`` named only the latter, so ``allow_unencodable=True`` -- documented as
    "writes the rest and records the exclusion ... never silent" -- did not cover it: the
    whole write died at that row, after 316 rows of tokenization, with no ``coverage.json``
    written and no count of what was refused.

    The refusal itself is right; a context over the cap must never be truncated. What is
    wrong is that it was outside the one mechanism that carries both numbers.
    """
    rows = snapshot.rows["train"]
    caps, n_over = _caps_refusing_the_longest_contexts(rows)
    assert 0 < n_over < len(rows), (
        f"the cap must refuse some rows and admit others; it refuses {n_over} of {len(rows)}"
    )

    out = tmp_path / "shards" / "over-cap"
    header = _write(snapshot, "train", out, caps=caps, allow_unencodable=True)
    assert header.n_sequences == len(rows) - n_over

    reader = ShardReader(out, config=snapshot.config, repo_root=snapshot.root)
    coverage = reader.coverage
    assert isinstance(coverage, Ran)
    assert coverage.n == len(rows) - n_over and coverage.n_total == len(rows)
    assert not coverage.passed and not coverage.is_complete_coverage
    assert "at most" in (coverage.detail or ""), (
        "the exclusion must name why the row was refused, not only that it was: "
        f"{coverage.detail!r}"
    )


def test_a_renderer_refusal_without_the_flag_still_refuses_the_whole_write(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """The default is unchanged: no flag, no partial corpus, and the original exception."""
    caps, n_over = _caps_refusing_the_longest_contexts(snapshot.rows["train"])
    assert n_over > 0
    out = tmp_path / "shards" / "nope"
    with pytest.raises(ContextTooLargeRefusal):
        _write(snapshot, "train", out, caps=caps)
    assert not out.exists(), "refused, and nothing was written"


def test_a_shard_set_records_whether_its_span_mapping_was_decode_verified(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """``decode=`` is optional, and a shard set written without it looked identical.

    ``write_shards``' own docstring: *"a real-tokenizer run passes it, and a run that omits
    it has not had its span mapping verified."* Measured on 2026-09-20 with the real Qwen
    tokenizer: writing the same corpus with ``decode=`` and without produced a
    byte-identical ``header.json`` (modulo ``created_at``), a byte-identical
    ``coverage.json``, and an identical ``ShardReader.to_json()`` -- which is what a ledger
    row records. So the one check that does not consult the offsets it is checking
    (``GAP-SPAN-HEAD-LINE-MAPPING-BPE-UNVERIFIED``, and
    ``GAP-S4-GOLD-ON-CANDIDATE-CANNOT-CATCH-LINE-SHIFT`` for why nothing else can) left no
    trace of having run, and a shard set whose spans were never verified reads downstream
    exactly like one whose spans were.
    """
    checked = tmp_path / "shards" / "decoded"
    unchecked = tmp_path / "shards" / "undecoded"
    _write(snapshot, "train", checked, decode=byte_decode)
    _write(snapshot, "train", unchecked)

    a = ShardReader(checked, config=snapshot.config, repo_root=snapshot.root).to_json()
    b = ShardReader(unchecked, config=snapshot.config, repo_root=snapshot.root).to_json()
    assert "span_check" in a, (
        "a shard set must state whether its span mapping was verified against decoded "
        f"text; to_json() carries only {sorted(a)}"
    )
    assert a["span_check"]["state"] == "ran" and a["span_check"]["passed"] is True
    assert b["span_check"]["state"] == "not_run", (
        "a write with no decode= never ran that check, and not_run is the only honest "
        f"answer; got {b['span_check']}"
    )
    assert "passed" not in b["span_check"], "NotRun has no passed to misread"
    assert a["span_check"]["n"] == a["span_check"]["n_total"] > 0, (
        "the check must carry how many span rows it actually verified"
    )


def test_a_shard_set_carries_the_remap_its_header_pins(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """A file of renumbered ids is unreadable without the table that renumbered them.

    ``ShardHeader.remap_hash`` pins the vocabulary, and ``assert_shard_trainable`` reports
    it as ``shard_provenance_pinned`` -- a hash of an artifact that, until this test,
    **nothing in the repository kept**. ``RemapTable.write`` had exactly one caller and it
    was a round-trip test in ``test_artifacts.py``; the only real remap ever built (by
    ``tools/real_tokenizer_pipeline.py``, 248,077 -> 13,787 tokens over the Qwen
    vocabulary) lived in one process and died with it. The shard set it wrote is on disk
    and not one of its 2,485,641 ids can be turned back into text.

    So the writer stores it beside the ids, and the reader re-reads and re-hashes it rather
    than trusting the header. ``RemapTable.read`` recomputes the hash from the tables, so a
    sidecar edited after the fact raises instead of returning a pass.
    """
    snap, out, header = train_shards
    stem = out / REMAP_NAME
    assert stem.with_suffix(".npz").exists() and stem.with_suffix(".json").exists(), (
        "a shard set must carry the remap its header pins; the directory holds only "
        f"{sorted(p.name for p in out.iterdir())}, and header.remap_hash="
        f"{header.remap_hash[:16]}… names nothing on disk"
    )
    back = RemapTable.read(stem)
    assert back.remap_hash() == header.remap_hash
    assert back.vocab_size == header.vocab_size

    reader = ShardReader(out, config=snap.config, repo_root=snap.root)
    check = reader.checks["shard_remap_matches_header"]
    assert isinstance(check, Ran) and check.passed, check
    assert reader.remap is not None
    assert reader.to_json()["checks"]["shard_remap_matches_header"]["state"] == "ran"


def test_a_missing_remap_reads_as_not_run_not_as_matching(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """Absent is unknown provenance, not checked provenance.

    Every shard set written before ``REMAP_NAME`` existed is in this state, so an absent
    table is not a contract violation. It must not read like a checked one: the header's
    ``remap_hash`` is reported by ``shard_provenance_pinned`` either way, which is exactly
    the trap. ``reader.remap`` stays ``None`` so no consumer can decode against a table it
    never got.
    """
    snap, out, _ = train_shards
    stem = out / REMAP_NAME
    stem.with_suffix(".npz").unlink()
    stem.with_suffix(".json").unlink()
    fresh = ShardReader(out, config=snap.config, repo_root=snap.root)
    check = fresh.checks["shard_remap_matches_header"]
    assert isinstance(check, NotRun)
    assert not hasattr(check, "passed"), "NotRun has no passed to misread"
    assert fresh.remap is None
    assert isinstance(fresh.checks["shard_provenance_pinned"], Ran), (
        "the header's own claim still reports passed=True with the table gone -- which is "
        "why the absent case has to be a separate, non-passing answer"
    )


def test_a_remap_that_is_not_the_headers_is_refused_not_recorded(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """A set whose ids mean something other than the header says is not a soft finding.

    ``coverage`` and ``span_check`` are recorded because they describe how much of a corpus
    got in and whether one check ran. This is different in kind: every id in the file means
    a different token. ``ShardContractViolation`` is the class ``write_shards`` already
    reserves for "the artifacts disagree with each other".
    """
    out = tmp_path / "shards" / "train"
    _write(snapshot, "train", out)
    other = byte_remap(drop=frozenset({7}))
    assert other.remap_hash() != json.loads(
        (out / HEADER_NAME).read_text(encoding="utf-8")
    )["remap_hash"]
    other.write(out / REMAP_NAME)
    with pytest.raises(ShardContractViolation, match="would mean a different token"):
        ShardReader(out, config=snapshot.config, repo_root=snapshot.root)


def test_a_missing_span_check_reads_as_not_run_not_as_verified(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """Every shard set written before this file existed has no such file. Silence is
    ``NotRun``, the same rule ``coverage.json``'s absence follows."""
    snap, out, _ = train_shards
    (out / SPAN_CHECK_NAME).unlink()
    fresh = ShardReader(out, config=snap.config, repo_root=snap.root)
    assert isinstance(fresh.span_check, NotRun)
    assert not hasattr(fresh.span_check, "passed")


def test_a_split_with_no_span_rows_reports_the_span_check_as_not_run(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """0 of 0 verified is not a pass.

    ``artifacts.padding_waste`` refuses to score an empty shard set for the same reason,
    and ``all([])`` being ``True`` is that trap one level down. A span check reported as
    passed over a set with no span in it would make the strongest statement this writer can
    make about spans available to every set that has none.
    """
    rows = snapshot.rows["val"]
    assert rows and not any(
        isinstance(slot, SpanSlot) for r in rows for slot in r.request.slots
    ), "this case needs a non-empty split that holds no span row"

    out = tmp_path / "shards" / "val"
    _write(snapshot, "val", out, decode=byte_decode)
    reader = ShardReader(out, config=snapshot.config, repo_root=snapshot.root)
    assert isinstance(reader.span_check, NotRun)
    assert "0 of 0" in reader.span_check.reason


def test_a_context_whose_lines_share_one_token_is_refused(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """Two lines the pointer head cannot tell apart is a train/serve mismatch.

    ``qd-runtime/src/answer.rs:64`` sizes the head at ``line_count + RESERVED_NOUL_ROWS``,
    so training over a smaller candidate set trains a differently shaped head. Refused
    rather than deduplicated -- deduplicating would make the mismatch invisible.
    """
    with pytest.raises(UnencodableGold) as excinfo:
        _write(
            snapshot,
            "train",
            tmp_path / "nope",
            tokenize=_collapsing_tokenize,
            token_offsets=_collapsing_offsets,
            buckets=[2],
        )
    assert "distinct tokens" in str(excinfo.value)
    assert "GAP-S4-LINE-STARTS-COLLAPSE-UNDER-BPE" in str(excinfo.value)


def test_offsets_describing_a_different_string_are_refused(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """The right number of offsets for the wrong text.

    A tokenizer handed a normalised copy -- newlines stripped here -- returns one offset
    per token and every one of them is wrong by a shifting amount. The count check cannot
    see it, because the ids come from the same normalised copy; only the reach can.
    """
    with pytest.raises(ShardContractViolation) as excinfo:
        _write(
            snapshot,
            "train",
            tmp_path / "nope",
            tokenize=lambda text: byte_tokenize(text.replace("\n", "")),
            token_offsets=lambda text: byte_offsets(text.replace("\n", "")),
        )
    assert "describe a different string" in str(excinfo.value)


def test_a_split_with_no_span_rows_needs_no_exclusion(snapshot: Snapshot, tmp_path: Path) -> None:
    """The default path: no flag, full coverage, and coverage says so."""
    out = tmp_path / "shards" / "val"
    header = _write(snapshot, "val", out)
    assert header.n_sequences == len(snapshot.rows["val"])
    reader = ShardReader(out, config=snapshot.config, repo_root=snapshot.root)
    assert isinstance(reader.coverage, Ran)
    assert reader.coverage.passed and reader.coverage.is_complete_coverage


def test_missing_coverage_reads_as_not_run_not_as_full(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """Absent coverage is unknown coverage. ``NotRun`` has no ``passed`` to misread."""
    snap, out, _ = train_shards
    (out / COVERAGE_NAME).unlink()
    fresh = ShardReader(out, config=snap.config, repo_root=snap.root)
    assert isinstance(fresh.coverage, NotRun)
    assert not hasattr(fresh.coverage, "passed")


# -- the round trip and the on-disk invariants -----------------------------------------


def test_sequences_round_trip_through_the_memmap(
    reader: ShardReader, train_shards: tuple[Snapshot, Path, ShardHeader]
) -> None:
    snap, _out, header = train_shards
    assert len(reader) == header.n_sequences
    assert sum(reader.lengths()) == header.total_tokens
    assert max(reader.lengths()) == header.max_seq_len
    assert reader.header.shard_hash() == header.shard_hash()

    row = min(snap.rows["train"], key=lambda r: r.row_id)
    text = training_texts(row, seed=snap.config.seed)[0].text
    expected = byte_remap().encode(np.asarray(byte_tokenize(text)))
    assert np.array_equal(reader.sequence(0), expected), "written in sorted row_id order"
    assert reader.sequence(0).dtype == np.int32


def test_a_tampered_header_is_refused(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    snap, out, _ = train_shards
    raw = json.loads((out / HEADER_NAME).read_text(encoding="utf-8"))
    raw["n_sequences"] = int(raw["n_sequences"]) - 1
    (out / HEADER_NAME).write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ShardContractViolation) as excinfo:
        ShardReader(out, config=snap.config, repo_root=snap.root)
    assert "modified after it was written" in str(excinfo.value)


def test_a_truncated_token_file_is_refused(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """A short read must be loud. A memmap over a truncated file reads zeros."""
    snap, out, _ = train_shards
    path = out / TOKENS_NAME
    with path.open("r+b") as fh:
        fh.truncate(path.stat().st_size - 4)
    with pytest.raises(ShardContractViolation) as excinfo:
        ShardReader(out, config=snap.config, repo_root=snap.root)
    assert "bytes on disk" in str(excinfo.value)


def test_the_header_never_claims_packing(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    _, _, header = train_shards
    assert header.packed is False
    assert header.dtype == "uint32"


# -- the tokenizer boundary ------------------------------------------------------------


@pytest.mark.parametrize("returned", [[], [65]])
def test_a_sequence_shorter_than_prompt_plus_answer_is_refused(
    snapshot: Snapshot, tmp_path: Path, returned: list[int]
) -> None:
    """``trainer.ft_supervision`` supervises ``lengths - 2``; one token has no such position.

    The trainer calls a one-token row "a writer-side error". This is the writer refusing
    it, while the row it came from can still be named.
    """
    with pytest.raises(ShardContractViolation) as excinfo:
        write_shards(
            snapshot.paths["val"],
            snapshot.rows["val"],
            out_dir=tmp_path / "nope",
            remap=byte_remap(),
            tokenize=lambda _text: list(returned),
            config=snapshot.config,
            repo_root=snapshot.root,
        )
    assert "at least two" in str(excinfo.value)
    assert "row " in str(excinfo.value)


def test_a_tokenizer_returning_floats_is_refused(snapshot: Snapshot, tmp_path: Path) -> None:
    """A float id would index the remap by silent truncation."""
    with pytest.raises(ShardContractViolation) as excinfo:
        write_shards(
            snapshot.paths["val"],
            snapshot.rows["val"],
            out_dir=tmp_path / "nope",
            remap=byte_remap(),
            tokenize=lambda text: [float(b) for b in text.encode()],  # type: ignore[misc]
            config=snapshot.config,
            repo_root=snapshot.root,
        )
    assert "must be" in str(excinfo.value) and "integer" in str(excinfo.value)


# -- bucketing -------------------------------------------------------------------------


def test_choose_buckets_covers_the_longest_sequence(reader: ShardReader) -> None:
    lengths = reader.lengths()
    buckets = choose_buckets(lengths)
    assert buckets[-1] == max(lengths), "a sequence that fits no bucket must be impossible"
    assert list(buckets) == sorted(set(buckets)), "strictly increasing and unique"
    assert all(n <= buckets[-1] for n in lengths)


def test_choose_buckets_refuses_an_empty_distribution() -> None:
    with pytest.raises(ValueError, match="zero lengths"):
        choose_buckets([])


def test_a_sequence_longer_than_the_largest_bucket_is_refused(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """Refused at write time, naming the row -- never truncated at read time."""
    with pytest.raises(ShardContractViolation) as excinfo:
        _write(snapshot, "val", tmp_path / "nope", buckets=[8])
    assert "fits no bucket" in str(excinfo.value)
    assert "row " in str(excinfo.value), "the refusal names the row, not a memmap index"


# -- ordering: the property S5's resume depends on -------------------------------------


def _order(reader: ShardReader, *, seed: int, epoch: int, batch_tokens: int) -> list[bytes]:
    return [
        b.tokens.tobytes()
        for b in reader.batches(batch_tokens=batch_tokens, seed=seed, epoch=epoch)
    ]


def test_batch_order_is_a_pure_function_of_seed_epoch_and_batch_tokens(
    reader: ShardReader,
) -> None:
    """Asked twice, answered identically. No hidden iterator RNG, nothing consumed."""
    width = reader.header.buckets[-1]
    first = _order(reader, seed=7, epoch=0, batch_tokens=width * 4)
    second = _order(reader, seed=7, epoch=0, batch_tokens=width * 4)
    assert first == second
    # And interleaving two live iterators does not disturb either.
    a = reader.batches(batch_tokens=width * 4, seed=7, epoch=0)
    b = reader.batches(batch_tokens=width * 4, seed=7, epoch=0)
    assert [next(a).tokens.tobytes(), next(b).tokens.tobytes()] == [first[0], first[0]]


def test_a_different_seed_gives_a_different_order(reader: ShardReader) -> None:
    width = reader.header.buckets[-1]
    assert _order(reader, seed=7, epoch=0, batch_tokens=width * 4) != _order(
        reader, seed=8, epoch=0, batch_tokens=width * 4
    )


def test_a_different_batch_budget_gives_a_different_order_at_the_same_indices(
    reader: ShardReader,
) -> None:
    """The third argument, which this section named and did not test.

    It is the one that matters for a resume, because unlike the seed and the epoch it can
    change the *contents* of an epoch without changing its *shape*: one extra token in the
    budget here leaves the batch count and every index alone and replaces every batch. A
    resume that pins the seed and not the budget lands on the right index in the wrong
    epoch, which is what `run_control.Checkpoint.consumed_digest` exists to catch.
    """
    width = reader.header.buckets[-1]
    same_shape = _order(reader, seed=7, epoch=0, batch_tokens=width * 4)
    one_more_token = _order(reader, seed=7, epoch=0, batch_tokens=width * 4 + 1)
    assert len(one_more_token) == len(same_shape), "the shape is unchanged..."
    assert one_more_token != same_shape, "...and the contents are not"
    indices = lambda bt: [  # noqa: E731 - a two-use local, not an API
        b.index for b in reader.batches(batch_tokens=bt, seed=7, epoch=0)
    ]
    assert indices(width * 4 + 1) == indices(width * 4), (
        "the indices agree, which is exactly why an index cannot identify an order"
    )


def test_the_order_is_the_same_in_a_second_process(
    train_shards: tuple[Snapshot, Path, ShardHeader], reader: ShardReader, tmp_path: Path
) -> None:
    """"A pure function" is a claim about processes, not about one interpreter.

    A resume happens in a new process by definition, so a property measured only inside one
    is not the property S5 needs. Same shard set on disk, a fresh interpreter, and a
    deliberately different `PYTHONHASHSEED`.
    """
    snap, out, _ = train_shards
    width = int(reader.header.buckets[-1])
    script = tmp_path / "second_process.py"
    script.write_text(
        "import hashlib, sys\n"
        f"sys.path.insert(0, {str(REPO_PYTHON)!r})\n"
        "from pathlib import Path\n"
        "from qd_data.config import DataConfig\n"
        "from qd_train.shards import ShardReader\n"
        f"r = ShardReader(Path({str(out)!r}), config=DataConfig(), "
        f"repo_root=Path({str(snap.root)!r}))\n"
        "h = hashlib.sha256()\n"
        f"for b in r.batches(batch_tokens={width * 4}, seed=7, epoch=0):\n"
        "    h.update(str(b.index).encode()); h.update(b.tokens.tobytes())\n"
        "print(h.hexdigest())\n",
        encoding="utf-8",
    )
    here = hashlib.sha256()
    for b in reader.batches(batch_tokens=width * 4, seed=7, epoch=0):
        here.update(str(b.index).encode())
        here.update(b.tokens.tobytes())
    # This interpreter, on a file this test just wrote.
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONHASHSEED": "1234", "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == here.hexdigest()


def test_a_different_epoch_gives_a_different_order(reader: ShardReader) -> None:
    width = reader.header.buckets[-1]
    assert _order(reader, seed=7, epoch=0, batch_tokens=width * 4) != _order(
        reader, seed=7, epoch=1, batch_tokens=width * 4
    )


def test_batch_index_is_its_position_in_the_epoch(reader: ShardReader) -> None:
    """``(epoch, index)`` is the whole of a resume position, so index must be the position."""
    width = reader.header.buckets[-1]
    batches = list(reader.batches(batch_tokens=width * 4, seed=7, epoch=0))
    assert [b.index for b in batches] == list(range(len(batches)))


def test_one_epoch_yields_every_sequence_exactly_once(reader: ShardReader) -> None:
    width = reader.header.buckets[-1]
    seen: list[bytes] = []
    for batch in reader.batches(batch_tokens=width * 4, seed=7, epoch=0):
        for r in range(batch.tokens.shape[0]):
            seen.append(batch.tokens[r, : int(batch.lengths[r])].tobytes())
    stored = [reader.sequence(i).tobytes() for i in range(len(reader))]
    assert sorted(seen) == sorted(stored)
    assert len(seen) == len(reader)


# -- SAFETY-2, on the read side --------------------------------------------------------


def test_every_batch_row_holds_exactly_one_sequence(reader: ShardReader) -> None:
    """No concatenation. A GDN layer carries state along a row, and no mask resets it."""
    width = reader.header.buckets[-1]
    stored = {reader.sequence(i).tobytes() for i in range(len(reader))}
    for batch in reader.batches(batch_tokens=width * 4, seed=3, epoch=0):
        assert isinstance(batch, Batch)
        for r in range(batch.tokens.shape[0]):
            n = int(batch.lengths[r])
            assert batch.tokens[r, :n].tobytes() in stored, "a row is one stored sequence"
            tail = batch.tokens[r, n:]
            assert np.all(tail == PAD_ID), "everything past the length is padding"


def test_every_row_in_a_batch_shares_the_buckets_width(reader: ShardReader) -> None:
    width = reader.header.buckets[-1]
    for batch in reader.batches(batch_tokens=width * 4, seed=3, epoch=0):
        assert batch.tokens.shape[1] == reader.header.buckets[batch.bucket]
        assert int(batch.lengths.max()) <= batch.tokens.shape[1]
        expected = assign_buckets(
            [int(n) for n in batch.lengths], reader.header.buckets
        )
        assert set(expected) == {batch.bucket}, "a batch does not mix buckets"


# -- bounds ----------------------------------------------------------------------------


def test_a_batch_budget_below_a_used_bucket_width_is_refused(reader: ShardReader) -> None:
    """Fail closed: an over-budget batch emitted quietly becomes an OOM blamed on the model."""
    with pytest.raises(ValueError, match="below bucket width"):
        list(reader.batches(batch_tokens=reader.header.buckets[-1] - 1, seed=1, epoch=0))


def test_rows_per_batch_are_bounded_as_well_as_tokens(
    reader: ShardReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A narrow bucket plus a generous token budget must not make an unbounded batch.

    The budget used to be ``width * 1000``, an arbitrary stand-in for "generous". Since
    2026-09-20 "generous" has a ceiling -- :data:`MAX_POSITIONS_PER_BATCH` -- so this asks
    the stronger question instead: *the most generous budget the planner will accept* still
    yields no more than the row cap. The assertion is unchanged.
    """
    monkeypatch.setattr(shards_module, "MAX_ROWS_PER_BATCH", 2)
    batches = list(
        reader.batches(batch_tokens=MAX_POSITIONS_PER_BATCH, seed=1, epoch=0)
    )
    assert batches, "the bound must not empty the epoch"
    assert max(b.tokens.shape[0] for b in batches) <= 2
    assert MAX_ROWS_PER_BATCH == 4096, "the shipped default is still a real bound"


def test_a_batch_budget_above_the_position_ceiling_is_refused(reader: ShardReader) -> None:
    """The hole `MAX_ROWS_PER_BATCH` did not close: rows were bounded, positions were not.

    Measured against the pre-fix code on the real 321-sequence shard set,
    ``batch_tokens=10_000_000_000`` was **accepted** and planned a batch of 1,380,880
    positions -- about 260 GB of activations for this model. ``_plan`` refused a budget
    below the widest bucket and accepted any budget above it, so the only ceiling on
    ``rows x width`` was the number the caller typed.
    """
    with pytest.raises(ValueError, match="exceeds MAX_POSITIONS_PER_BATCH"):
        list(
            reader.batches(
                batch_tokens=MAX_POSITIONS_PER_BATCH + 1, seed=0, epoch=0
            )
        )


def test_the_position_ceiling_admits_the_widest_bucket_it_could_ever_face(
    reader: ShardReader,
) -> None:
    """A ceiling below a legal sequence would make the two refusals contradict each other.

    ``_plan`` requires ``batch_tokens >= buckets[-1]`` and now also
    ``batch_tokens <= MAX_POSITIONS_PER_BATCH``. If the ceiling were ever below the widest
    bucket a shard set could hold, no budget would satisfy both and the corpus would be
    untrainable for a reason nobody stated. 262,144 is this model's
    ``max_position_embeddings``, the longest sequence that can exist.
    """
    assert MAX_POSITIONS_PER_BATCH >= 262_144
    assert reader.header.buckets[-1] <= MAX_POSITIONS_PER_BATCH
    batches = list(
        reader.batches(batch_tokens=MAX_POSITIONS_PER_BATCH, seed=0, epoch=0)
    )
    assert batches, "the widest legal budget must still produce an epoch"


def test_a_negative_seed_or_epoch_is_refused(reader: ShardReader) -> None:
    width = reader.header.buckets[-1]
    with pytest.raises(ValueError, match="non-negative"):
        list(reader.batches(batch_tokens=width, seed=-1, epoch=0))
    with pytest.raises(ValueError, match="non-negative"):
        list(reader.batches(batch_tokens=width, seed=0, epoch=-1))


def test_a_non_positive_batch_budget_is_refused(reader: ShardReader) -> None:
    with pytest.raises(ValueError, match="batch_tokens must be positive"):
        list(reader.batches(batch_tokens=0, seed=0, epoch=0))


def test_the_sequence_bound_is_enforced(snapshot: Snapshot, tmp_path: Path) -> None:
    with pytest.raises(ShardContractViolation, match="sequence bound"):
        _write(snapshot, "val", tmp_path / "nope", max_sequences=2)


def test_the_token_bound_is_enforced(snapshot: Snapshot, tmp_path: Path) -> None:
    with pytest.raises(ShardContractViolation, match="token bound"):
        _write(snapshot, "val", tmp_path / "nope", max_total_tokens=100)


# -- the gate --------------------------------------------------------------------------


def test_padding_waste_delegates_to_the_contracts_function(reader: ShardReader) -> None:
    """One implementation of the gate, imported -- not a second one that agrees today."""
    mine = reader.padding_waste()
    theirs = padding_waste(reader.lengths(), reader.header.buckets)
    assert isinstance(mine, Ran) and isinstance(theirs, Ran)
    assert (mine.value, mine.passed, mine.n, mine.n_total) == (
        theirs.value,
        theirs.passed,
        theirs.n,
        theirs.n_total,
    )


def test_the_gate_describes_the_padding_the_reader_actually_pays(reader: ShardReader) -> None:
    """Measured against the batches, not against an idea of them.

    If ``batches`` padded to the longest sequence *in the batch* rather than to the
    bucket's width, this number would be lower than the gate's and the gate would be a
    figure about a padding scheme nobody trains with.
    """
    width = reader.header.buckets[-1]
    padded = 0
    real = 0
    for batch in reader.batches(batch_tokens=width * 4, seed=11, epoch=0):
        padded += int(batch.tokens.size)
        real += int(batch.lengths.sum())
    measured = (padded - real) / padded
    gate = reader.padding_waste()
    assert isinstance(gate, Ran) and gate.value is not None
    assert measured == pytest.approx(float(gate.value), abs=1e-6)


def test_the_padding_waste_gate_is_reported_with_its_threshold(reader: ShardReader) -> None:
    """Report the tri-state; never move the threshold (rule 2)."""
    gate = reader.padding_waste()
    assert isinstance(gate, Ran)
    assert gate.n == gate.n_total == len(reader)
    assert f"{MAX_PADDING_WASTE:.0%}" in gate.detail
    assert gate.passed == (float(gate.value) <= MAX_PADDING_WASTE)  # type: ignore[arg-type]


def test_an_empty_length_list_is_not_a_free_pass() -> None:
    """``padding_waste([])`` is ``0/0``. ``all([]) is True`` one level down."""
    assert isinstance(padding_waste([], (16,)), NotRun)


# -- lines, and the mapping onto tokens ------------------------------------------------


@pytest.mark.parametrize(
    ("text", "lines"),
    [("a\n", 1), ("a\nb\n", 2), ("a\nb\nc", 3), ("a\nb\nc\n", 3), ("no newline at all", 1)],
)
def test_line_starts_matches_the_rust_definition(text: str, lines: int) -> None:
    """The exact vectors from the Rust suite, which owns the definition of "line".

    ``crates/qd-runtime/tests/answering_procedure.rs``
    ``::the_pointer_head_ranges_over_the_contexts_line_starts_plus_the_abstain_row``.
    This Python copy exists only because the writer is in Python; pinning it against the
    Rust suite's own vectors is what keeps the two from drifting silently, which is the
    shape of every ``GAP-XLANG-*`` record in this repo.
    """
    assert len(line_starts(text)) == lines


def test_a_trailing_newline_does_not_open_an_empty_final_line() -> None:
    assert line_starts("a\nb\n") == [0, 2]
    assert line_starts("a\nb") == [0, 2]


def test_carriage_return_is_content_not_a_line_terminator() -> None:
    """``docs/hardening.md`` section 1: CRLF is where offsets and line numbers diverge."""
    assert len(line_starts("a\rb")) == 1
    assert len(line_starts("a\r\nb")) == 2


def test_an_empty_context_has_no_lines_not_one_empty_line() -> None:
    """The vector the Rust suite's own table does not carry, and the one that diverged.

    ``crates/qd-runtime/src/context.rs:167`` ``Context::line_count`` special-cases the
    empty context to **0**, and ``::an_empty_context_is_a_legal_value_not_an_error``
    (context.rs:322) asserts it. The pointer head is sized at
    ``line_count + RESERVED_NOUL_ROWS``, so an empty context serves the abstain row
    *alone*. Reporting one line here hands the writer a candidate the runtime will never
    offer -- a train/serve mismatch of exactly the shape ``docs/hardening.md`` section 1
    calls invisible in loss.

    The five vectors in ``test_line_starts_matches_the_rust_definition`` are copied from
    the Rust suite and not one of them is empty, which is precisely why the two Python
    implementations could disagree here with both suites green.
    """
    assert line_starts("") == []


def test_the_two_python_line_rules_agree_vector_for_vector() -> None:
    """``qd_train.byte_context.line_starts`` is the *other* Python implementation.

    Same name, same package, same question -- "which positions start a line" -- and rung
    0's byte path reaches it through :mod:`qd_train.mutate_adapter` while S4 reaches its
    own through :func:`qd_train.shards.line_starts`. Two functions of one name in one
    package is the shape that has bitten this repo five times (``GAP-RT-WIRE-CONTEXT-
    ENCODING``, ``GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS``, the ``value``/``noul`` pair,
    the hand-transcribed ``HEX_ESCAPED``, the ``label_set_hash`` rename), every time with
    both sides' suites green.

    They cannot be merged into one function: S4 needs **character** offsets, because a
    tokenizer's offset mapping is expressed in characters, and rung 0 needs **byte**
    offsets, because its model's positions *are* bytes. So they are pinned instead, and
    the pin is over the rule rather than over the unit: on ASCII the two agree offset for
    offset, and on non-ASCII they must still report the same *number* of lines, since a
    ``\\n`` is one character and one byte.

    ``GAP-S4-LINE-STARTS-SECOND-IMPLEMENTATION``.
    """
    ascii_vectors = [
        "",
        "a",
        "a\n",
        "a\nb",
        "a\nb\n",
        "a\n\nb",
        "a\r\nb",
        "a\rb",
        "\n",
        "\n\n",
        "no newline at all",
        "a\nb\nc",
        "a\nb\nc\n",
    ]
    for text in ascii_vectors:
        assert line_starts(text) == list(byte_line_starts(text.encode("utf-8"))), text

    # Non-ASCII: the units legitimately differ, the line grid may not.
    for text in ["é\nb", "你好\n世界\n", "\U0001f600\n\U0001f600"]:
        assert len(line_starts(text)) == len(byte_line_starts(text.encode("utf-8"))), text


def test_a_span_over_a_context_with_no_lines_is_refused_not_pointed_at_offset_zero() -> None:
    """What the converged line rule exposes, one layer down, as a typed refusal.

    With a line rule that reports one line for an empty context, this row is written with
    one candidate -- in range, plausible, and a position ``qd-runtime`` will never offer,
    because ``Context::line_count`` is 0 there and the pointer head is
    ``line_count + RESERVED_NOUL_ROWS``. With the converged rule the candidate set is
    empty, and an empty candidate set must be a named refusal rather than the
    ``ValueError`` that ``max(())`` would raise several lines later.

    ``render`` refuses an empty context before ``training_texts`` can reach here
    (``EmptyContextRefusal``, asserted by the test below), so this is the second of two
    guards rather than the live path. It is the one that stands if the first is ever
    relaxed, and it is checked at its own boundary rather than assumed.
    """
    text = "prompt with no context lines"
    spec = SequenceSpec(
        slot_name="evidence",
        text=text,
        slot_kind=SLOT_SPAN,
        span_char_starts=None,
        span_abstains=True,
        line_char_starts=(),
    )
    # A *consistent* offset mapping: four tokens, four offsets, reaching the end of the
    # text -- so every other check in `_span_token_positions` is satisfied and the empty
    # candidate set is the only thing left to object to. Without the guard this reaches
    # `max(candidates)` on an empty tuple and dies as a bare `ValueError`, which names
    # neither the row nor the reason.
    offsets = [(0, 7), (7, 12), (12, 20), (20, len(text))]
    with pytest.raises(UnencodableGold) as excinfo:
        shards_module._span_token_positions(
            spec,
            [1, 2, 3, 4],
            token_offsets=lambda _t: offsets,
            where="row r1 slot evidence",
        )
    assert "no lines" in str(excinfo.value)
    assert "answer.rs:64" in str(excinfo.value)


def test_render_refuses_an_empty_context_before_the_writer_ever_sees_one() -> None:
    """The upstream half of the pair above, asserted rather than assumed.

    A guard whose only justification is "something upstream prevents this" is worth
    exactly as much as that claim is checked.
    """
    def request_over(context: bytes) -> Request:
        return Request(
            task="qa",
            context=context,
            question="Which line is the evidence on?",
            slots=(SpanSlot(name="evidence"),),
            example_id="empty-ctx",
        )

    with pytest.raises(EmptyContextRefusal):
        render(request_over(b""), seed=7)
    # Whitespace only is the same refusal: a context of newlines has a line grid and
    # nothing on it, which would let the head point confidently at a blank.
    with pytest.raises(EmptyContextRefusal):
        render(request_over(b"\n\n"), seed=7)
    # The control. Without it this test would still pass if `render` refused everything.
    assert render(request_over(b"alpha\nbeta\n"), seed=7).context_region()


def test_a_span_row_carries_gold_token_positions(snapshot: Snapshot) -> None:
    """The line -> token mapping, end to end, against an exactly-known tokenization.

    Over **every** pointing span row in the split, not the first one. This is the only
    check in the suite that can catch a line->token mapping which is consistently wrong:
    ``GAP-S4-GOLD-ON-CANDIDATE-CANNOT-CATCH-LINE-SHIFT`` measured that the
    gold-lands-on-a-candidate invariant cannot, because the gold and the candidates are
    drawn from one list of offsets and a whole-line shift moves the gold onto a different
    but equally valid candidate. The baseline here is independent -- the region's own line
    split -- so it is the one that bites, and running it on one row of nineteen was a
    sample reported as coverage.
    """
    pointing = [
        r
        for r in snapshot.rows["train"]
        if r.family_id == "qa.answer_span" and not r.gold[0].is_noul
    ]
    assert len(pointing) == 19, "the fixture's pointing span rows"

    for row in pointing:
        spec = training_texts(row, seed=snapshot.config.seed)[0]
        assert spec.slot_kind == SLOT_SPAN
        assert spec.span_char_starts is not None

        start_char, end_char = spec.span_char_starts
        # Each gold offset really is the first character of a line of the rendered prompt.
        assert start_char == 0 or spec.text[start_char - 1] == "\n", row.row_id
        assert end_char == 0 or spec.text[end_char - 1] == "\n", row.row_id

        # The expectation is derived from the context independently -- split the region
        # into lines and take the gold's 1-based inclusive slice -- rather than from the
        # offsets under test. An earlier version of this test computed its baseline *from*
        # start_char, which made it self-consistent: shifting every gold line by one still
        # passed it, and that shift is docs/hardening.md section 1's systematic
        # off-by-one, the one that is invisible in accuracy. A test that cannot fail that
        # way is not testing the mapping.
        region = render(row.request, seed=snapshot.config.seed).context_region()
        gold_start, gold_end = (int(v) for v in row.gold[0].value)  # type: ignore[union-attr]
        region_lines = region.split("\n")
        evidence = "\n".join(region_lines[gold_start - 1 : gold_end])
        assert spec.text[start_char : start_char + len(evidence)] == evidence, row.row_id
        last_line = region_lines[gold_end - 1]
        assert spec.text[end_char : end_char + len(last_line)] == last_line, row.row_id
        # And the evidence really is where the answer lives, so a shift cannot pass by luck.
        assert evidence.strip(), f"{row.row_id}: the gold lines must not be blank"
        assert evidence in region, row.row_id


def test_the_gold_producers_line_numbers_index_this_line_grid(snapshot: Snapshot) -> None:
    """The third statement of the line rule, pinned to the one that owns it.

    ``qd_data.mixture._line_span`` is where a span gold's line *numbers* come from, and it
    states the rule a third way -- ``text.count("\\n", 0, offset) + 1`` -- rather than by
    asking for a line grid. It lives in another lane's module, so it cannot be converged
    into :func:`qd_train.artifacts.line_start_indices`; it is read here and pinned
    instead, which is the standing check ``GAP-S4-LINE-STARTS-SECOND-IMPLEMENTATION``
    asks for when two implementations must stay apart.

    The pin is the only thing that matters between them: **a line number it returns must
    name the line of the canonical grid that the offset actually falls on.** A passage
    with a trailing newline is the vector where a counting rule and a grid rule come
    apart, so one is included deliberately.
    """
    vectors = [
        ("alpha\nbeta\ngamma", 0, 5),
        ("alpha\nbeta\ngamma", 6, 4),
        ("alpha\nbeta\ngamma", 11, 5),
        ("alpha\nbeta\ngamma\n", 11, 5),
        ("alpha\nbeta\ngamma\n", 0, 17),
        ("one line only", 4, 4),
        ("a\n\nb", 3, 1),
        ("crlf\r\nis\r\none\r\nbreak", 6, 2),
    ]
    def line_of(grid: tuple[int, ...], pos: int) -> int:
        """1-based index of the canonical grid line containing ``pos``."""
        return max(i for i, s in enumerate(grid, 1) if s <= pos)

    def agrees(span: tuple[int, int], text: str, offset: int, length: int) -> bool:
        start_line, end_line = span
        grid = line_start_indices(text)
        return (
            1 <= start_line <= end_line <= len(grid)
            and start_line == line_of(grid, offset)
            and end_line == line_of(grid, offset + length - 1)
        )

    for text, offset, length in vectors:
        assert agrees(mixture_line_span(text, offset, length), text, offset, length), (
            text,
            offset,
            length,
        )

    # The negative control. `agrees` is the whole content of this pin, so a version of it
    # that cannot fail would make the loop above decorative -- and a systematic one-line
    # shift is exactly the drift the pin exists for, the same class
    # GAP-S4-GOLD-ON-CANDIDATE-CANNOT-CATCH-LINE-SHIFT records the candidate invariant
    # being blind to. Every vector must reject it.
    for text, offset, length in vectors:
        shifted = mixture_line_span(text, offset, length)
        assert not agrees((shifted[0] + 1, shifted[1] + 1), text, offset, length), (
            f"a whole-line shift passed on {text!r} at {offset}: the pin is vacuous"
        )

    # And the same relation on the fixture's real rows, over the passage the gold was
    # measured against, so the pin is not only over hand-written vectors.
    checked = 0
    for row in snapshot.rows["train"]:
        if row.family_id != "qa.answer_span" or row.gold[0].is_noul:
            continue
        context = row.request.context.decode("utf-8")
        gold_start, gold_end = (int(v) for v in row.gold[0].value)  # type: ignore[union-attr]
        grid = line_start_indices(context)
        assert 1 <= gold_start <= gold_end <= len(grid), row.row_id
        checked += 1
    assert checked == 19


# -- the decoded-text check: the one that does not consult the offsets it checks --------


def test_the_whole_train_split_passes_the_decode_check(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """The happy path, and no more than that.

    This says the check is wired into ``write_shards`` and does not fire on a correct
    mapping. It does **not** verify the mapping: with a byte-level stand-in a line start
    *is* a byte offset, so there is nothing here for the check to catch.
    ``GAP-S4-SPAN-OFFSETS-UNVERIFIED-ON-REAL-TOKENIZER`` stays open until a real tokenizer
    runs it; what the next test establishes is that it would bite when one does.
    """
    header = _write(snapshot, "train", tmp_path / "decoded", decode=byte_decode)
    assert header.n_sequences == 106


def test_a_token_that_does_not_decode_to_its_claimed_characters_is_refused(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """Statement (1): the offsets describe a different string than the ids do.

    A tokenizer fed a normalised copy of the text returns offsets of the right length that
    reach the right end and describe the wrong content. ``write_shards`` already refuses
    the crude version by checking reach; only a decode catches the version that keeps the
    reach and moves the characters.
    """

    def swapped_decode(ids: Sequence[int]) -> str:
        return byte_decode(ids).swapcase()

    with pytest.raises(ShardContractViolation) as excinfo:
        _write(snapshot, "train", tmp_path / "normalised", decode=swapped_decode)
    assert "decodes to" in str(excinfo.value)
    assert "offsets claim" in str(excinfo.value)


def test_a_recorded_line_start_that_is_not_one_in_the_decoded_text_is_refused(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """Statement (3), and the one ``GAP-SPAN-HEAD-LINE-MAPPING-BPE-UNVERIFIED`` asks for.

    The stand-in models a real HuggingFace behaviour rather than a contrived one:
    ``decode`` applies ``clean_up_tokenization_spaces`` over a **sequence**, so a tokenizer
    can be perfectly honest token by token -- statement (1) passes on every one -- and
    still not reproduce the whitespace of the text the offsets were measured against. When
    that whitespace is a newline, every recorded line start after it is no longer the start
    of a line in what the model will actually see, and nothing that consults the candidate
    set can tell, because the candidate set was projected from the same offsets.

    The control below is the point: the identical corpus, written without ``decode``,
    succeeds. This check is the only thing in the path that sees it.
    """

    def cleaning_decode(ids: Sequence[int]) -> str:
        text = byte_decode(ids)
        # Honest for a single token; "cleans up" newlines only over a sequence.
        return text if len(list(ids)) == 1 else text.replace("\n", " ")

    with pytest.raises(ShardContractViolation) as excinfo:
        _write(snapshot, "train", tmp_path / "cleaned", decode=cleaning_decode)
    message = str(excinfo.value)
    assert "not line starts of the text these ids decode to" in message
    assert "GAP-SPAN-HEAD-LINE-MAPPING-BPE-UNVERIFIED" in message

    # The control. Same corpus, same tokenizer, no decode -- and it writes cleanly.
    assert _write(snapshot, "train", tmp_path / "unchecked").n_sequences == 106


def test_ids_that_do_not_round_trip_to_their_text_are_refused(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """Statement (2), reached only when (1) and (3) are both satisfied.

    Appending to the decoded sequence leaves every token honest and every recorded line
    start a genuine line start, and still means the ids are not the text the spans were
    measured against.
    """

    def trailing_decode(ids: Sequence[int]) -> str:
        text = byte_decode(ids)
        return text if len(list(ids)) == 1 else text + " tail"

    with pytest.raises(ShardContractViolation) as excinfo:
        _write(snapshot, "train", tmp_path / "trailing", decode=trailing_decode)
    assert "do not decode to the text the spans were measured against" in str(excinfo.value)


def test_a_span_row_without_token_offsets_is_refused_not_guessed(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """Token ids cannot say which characters a token covers, so the mapping has no input."""
    with pytest.raises(UnencodableGold) as excinfo:
        write_shards(
            snapshot.paths["train"],
            snapshot.rows["train"],
            out_dir=tmp_path / "nope",
            remap=byte_remap(),
            tokenize=byte_tokenize,
            config=snapshot.config,
            repo_root=snapshot.root,
            allow_unencodable=False,
        )
    text = str(excinfo.value)
    assert "token_offsets" in text or "abstain" in text


def test_offsets_that_disagree_with_the_ids_are_refused(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """Two callables describing different tokenizations would index each other wrongly."""
    with pytest.raises(ShardContractViolation, match="different tokenizations"):
        _write(
            snapshot,
            "train",
            tmp_path / "nope",
            token_offsets=lambda text: byte_offsets(text)[:-1],
            allow_unencodable=True,
        )


# -- the supervision channel -----------------------------------------------------------


def test_every_batch_carries_slot_kind_and_target_index(reader: ShardReader) -> None:
    width = reader.header.buckets[-1]
    for batch in reader.batches(batch_tokens=width * 4, seed=5, epoch=0):
        assert batch.slot_kind is not None and batch.target_index is not None
        assert batch.slot_kind.shape == (batch.tokens.shape[0],)
        assert batch.slot_kind.dtype == np.uint8
        assert batch.target_index.dtype == np.int32
        # The gold is the token after target_index, and it is the last real token.
        assert np.array_equal(batch.target_index, batch.lengths.astype(np.int32) - 2)


def test_span_rows_carry_positions_or_abstain_and_others_carry_no_span(
    reader: ShardReader,
) -> None:
    width = reader.header.buckets[-1]
    pointing = abstaining = 0
    for batch in reader.batches(batch_tokens=width * 4, seed=5, epoch=0):
        assert batch.slot_kind is not None
        is_span = batch.slot_kind == SLOT_SPAN
        if not is_span.any():
            assert batch.span_target is None, "no span row, so no span positions"
            assert batch.line_starts is None, "and no candidate set either"
            continue
        assert batch.span_target is not None
        assert np.all(batch.span_target[~is_span] == NO_SPAN)
        starts, ends = batch.span_target[is_span, 0], batch.span_target[is_span, 1]
        # Both positions abstain or neither: half of each is a gold nobody can read.
        assert np.all((starts == SPAN_ABSTAIN) == (ends == SPAN_ABSTAIN))
        abstaining += int((starts == SPAN_ABSTAIN).sum())
        real = starts != SPAN_ABSTAIN
        pointing += int(real.sum())
        assert np.all(starts[real] >= 0) and np.all(starts[real] <= ends[real])
        assert np.all(ends[real] < batch.lengths[is_span][real])
    assert (pointing, abstaining) == (19, 5), "all 24 span rows, none dropped"


def test_every_span_batch_carries_the_pointer_heads_candidate_set(
    reader: ShardReader,
) -> None:
    """``line_starts`` is required with a span row and refused without one."""
    width = reader.header.buckets[-1]
    for batch in reader.batches(batch_tokens=width * 4, seed=5, epoch=0):
        assert batch.slot_kind is not None
        is_span = batch.slot_kind == SLOT_SPAN
        if not is_span.any():
            assert batch.line_starts is None
            continue
        assert batch.line_starts is not None
        assert batch.line_starts.dtype == np.bool_
        assert batch.line_starts.shape == batch.tokens.shape
        for r in np.flatnonzero(is_span):
            n = int(batch.lengths[r])
            assert batch.line_starts[r, n:].sum() == 0, "no candidate may sit in padding"
            assert batch.line_starts[r, :n].any(), "a sequence has at least one line"


def test_a_span_gold_always_lands_on_a_candidate(reader: ShardReader) -> None:
    """The invariant that turns the line->token chain from plausible into checked.

    A mapping off by one line still produces an in-range, perfectly plausible token; what
    it does not do is land on a line start. Asserted here over every span row of a real
    shard set rather than only inside ``Batch``.
    """
    width = reader.header.buckets[-1]
    checked = 0
    for batch in reader.batches(batch_tokens=width * 4, seed=5, epoch=0):
        assert batch.slot_kind is not None
        if batch.span_target is None or batch.line_starts is None:
            continue
        is_span = batch.slot_kind == SLOT_SPAN
        for r in np.flatnonzero(is_span):
            start, end = int(batch.span_target[r, 0]), int(batch.span_target[r, 1])
            if start == SPAN_ABSTAIN:
                continue
            assert batch.line_starts[r, start], "gold start is not a line-start token"
            assert batch.line_starts[r, end], "gold end is not a line-start token"
            checked += 1
    assert checked == 19


def test_the_candidate_set_is_the_contexts_line_starts(
    reader: ShardReader, snapshot: Snapshot
) -> None:
    """One candidate per context line -- the shape qd-runtime's pointer head serves.

    ``answer.rs:64`` sizes the head at ``line_count + RESERVED_NOUL_ROWS``, so a candidate
    set with a different number of lines is a train/serve mismatch no loss curve shows.
    """
    by_id = {r.row_id: r for r in snapshot.rows["train"]}
    ordered = sorted(by_id)
    span_ids = [
        rid for rid in ordered if by_id[rid].family_id == "qa.answer_span"
    ]
    assert span_ids, "the fixture must contain span rows"
    for i in range(len(reader)):
        if int(reader._slot_kind[i]) != SLOT_SPAN:
            assert reader.candidates(i).size == 0
            continue
        positions = reader.candidates(i)
        assert positions.size > 0
        assert len(set(positions.tolist())) == positions.size, "candidates are distinct"
        assert np.all(np.diff(positions) > 0), "and in reading order"
    # And the count matches the context's own line count, for one row checked end to end.
    row = by_id[span_ids[0]]
    n_lines = len(line_starts(row.request.context.decode("utf-8")))
    idx = next(
        i
        for i in range(len(reader))
        if int(reader._slot_kind[i]) == SLOT_SPAN
        and reader.candidates(i).size == n_lines
    )
    assert reader.candidates(idx).size == n_lines


def test_the_supervision_channel_survives_the_round_trip(reader: ShardReader) -> None:
    """Every sequence's kind is read back, and all three kinds are present."""
    width = reader.header.buckets[-1]
    kinds: list[int] = []
    for batch in reader.batches(batch_tokens=width * 4, seed=5, epoch=0):
        assert batch.slot_kind is not None
        kinds.extend(int(k) for k in batch.slot_kind)
    assert len(kinds) == len(reader)
    assert set(kinds) == {SLOT_CHOICE, SLOT_SCORE, SLOT_SPAN}


def test_a_shard_set_without_a_supervision_file_is_refused(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """Absent supervision must not default to CPT: that is a different objective."""
    snap, out, _ = train_shards
    (out / SUPERVISION_NAME).unlink()
    with pytest.raises(ShardContractViolation) as excinfo:
        ShardReader(out, config=snap.config, repo_root=snap.root)
    assert "different objective" in str(excinfo.value)


def test_a_supervision_file_pointing_past_a_sequence_is_refused(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """Refused at open, not at some batch in the middle of an epoch."""
    snap, out, _ = train_shards
    with np.load(out / SUPERVISION_NAME) as z:
        fields = {k: z[k].copy() for k in z.files}
    pointing = np.flatnonzero(
        (fields["slot_kind"] == SLOT_SPAN) & (fields["span_target"][:, 0] != SPAN_ABSTAIN)
    )
    fields["span_target"][int(pointing[0]), 1] = 10**6
    np.savez(out / SUPERVISION_NAME, **fields)
    with pytest.raises(ShardContractViolation, match="points past it"):
        ShardReader(out, config=snap.config, repo_root=snap.root)


def test_a_gold_moved_off_its_line_start_is_refused_at_open(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """A gold that is in range but not a candidate: the off-by-one-line failure exactly."""
    snap, out, _ = train_shards
    with np.load(out / SUPERVISION_NAME) as z:
        fields = {k: z[k].copy() for k in z.files}
    pointing = np.flatnonzero(
        (fields["slot_kind"] == SLOT_SPAN) & (fields["span_target"][:, 0] != SPAN_ABSTAIN)
    )
    i = int(pointing[0])
    fields["span_target"][i, 0] = int(fields["span_target"][i, 0]) + 1
    np.savez(out / SUPERVISION_NAME, **fields)
    with pytest.raises(ShardContractViolation, match="not one of that sequence"):
        ShardReader(out, config=snap.config, repo_root=snap.root)


def test_a_candidate_in_padding_is_refused_at_open(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """A candidate past the real tokens lets the pointer head answer with padding."""
    snap, out, _ = train_shards
    with np.load(out / SUPERVISION_NAME) as z:
        fields = {k: z[k].copy() for k in z.files}
    i = int(np.flatnonzero(fields["slot_kind"] == SLOT_SPAN)[0])
    lo = int(fields["candidate_offsets"][i])
    fields["candidate_positions"][lo] = 10**6
    np.savez(out / SUPERVISION_NAME, **fields)
    with pytest.raises(ShardContractViolation, match="outside its"):
        ShardReader(out, config=snap.config, repo_root=snap.root)


def test_a_span_row_with_no_candidates_is_refused_at_open(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """An empty candidate set is a mapping failure, not a sequence with no lines."""
    snap, out, _ = train_shards
    with np.load(out / SUPERVISION_NAME) as z:
        fields = {k: z[k].copy() for k in z.files}
    i = int(np.flatnonzero(fields["slot_kind"] == SLOT_SPAN)[0])
    offs = fields["candidate_offsets"]
    keep = np.concatenate(
        [fields["candidate_positions"][: offs[i]], fields["candidate_positions"][offs[i + 1] :]]
    )
    removed = int(offs[i + 1] - offs[i])
    offs[i + 1 :] -= removed
    np.savez(
        out / SUPERVISION_NAME,
        slot_kind=fields["slot_kind"],
        target_index=fields["target_index"],
        span_target=fields["span_target"],
        candidate_offsets=offs,
        candidate_positions=keep,
    )
    with pytest.raises(ShardContractViolation, match="no line-start candidates"):
        ShardReader(out, config=snap.config, repo_root=snap.root)


def test_a_half_abstaining_span_is_refused_at_open(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """Abstain in one position only is a gold nobody can read."""
    snap, out, _ = train_shards
    with np.load(out / SUPERVISION_NAME) as z:
        fields = {k: z[k].copy() for k in z.files}
    abst = np.flatnonzero(
        (fields["slot_kind"] == SLOT_SPAN) & (fields["span_target"][:, 0] == SPAN_ABSTAIN)
    )
    fields["span_target"][int(abst[0]), 1] = 3
    np.savez(out / SUPERVISION_NAME, **fields)
    with pytest.raises(ShardContractViolation, match="abstains in one position only"):
        ShardReader(out, config=snap.config, repo_root=snap.root)


def test_the_trainer_accepts_these_batches_and_routes_spans_to_the_pointer_head(
    reader: ShardReader,
) -> None:
    """The cross-lane check, run rather than read.

    ``qd_train.trainer.ft_supervision`` is pure numpy, so the seam between S4's reader and
    the trainer can actually be exercised here instead of the two lanes agreeing on paper
    -- which is how ``GAP-RT-WIRE-CONTEXT-ENCODING`` and
    ``GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS`` both passed their own suites while the
    system did not work.

    The property that matters: a ``SLOT_SPAN`` row must be **out** of the letter mask. Its
    trailing token is the noul letter, so supervising it as a letter is exactly the
    abstain-always bug the span channel was added to prevent.
    """
    trainer = pytest.importorskip("qd_train.trainer")
    width = reader.header.buckets[-1]
    span_rows_seen = 0
    for batch in reader.batches(batch_tokens=width * 4, seed=13, epoch=0):
        supervision = trainer.ft_supervision(batch)
        assert batch.slot_kind is not None
        is_span = batch.slot_kind == SLOT_SPAN
        # No span row contributes a supervised letter position.
        assert not supervision.mask[is_span].any(), "a span row was supervised as a letter"
        # Every non-span row contributes exactly one.
        assert np.all(supervision.mask[~is_span].sum(axis=1) == 1)
        if is_span.any():
            span_rows_seen += int(is_span.sum())
            assert supervision.span is not None
            assert supervision.span.rows.size == int(is_span.sum())
            # The pointer head's query is the position S4 wrote, not a re-derivation.
            assert np.array_equal(
                supervision.span.query_index,
                batch.target_index[is_span].astype(np.int64),  # type: ignore[index]
            )
            assert np.all(supervision.span.start <= supervision.span.end)
    assert span_rows_seen == 24, "19 pointing + 5 abstaining"


def test_to_json_carries_the_checks_the_coverage_and_the_gate(reader: ShardReader) -> None:
    payload = reader.to_json()
    assert payload["header"]["split"] == "train"
    assert payload["header"]["packed"] is False
    assert payload["checks"]["shard_split_trainable"]["passed"] is True
    assert payload["checks"]["shard_not_packed"]["passed"] is True
    assert payload["padding_waste"]["state"] == "ran"
    assert payload["coverage"]["n_total"] == payload["coverage"]["n"] == 106
    # The `reader` fixture writes without decode=, so this set's span mapping was never
    # checked against decoded text -- and a ledger row built from this payload has to say
    # so rather than being silent about it.
    assert payload["span_check"]["state"] == "not_run"
    assert "passed" not in payload["span_check"]


# -- self-consistency, at the write boundary -------------------------------------------
#
# GAP-REALFT-CONTRADICTORY-SPAN-SUPERVISION was found by a training tool and fixed in the
# corpus generator, and the residual it left was this: nothing refused to *write* a
# self-contradictory corpus. A causal model conditions on `tokens[:target_index + 1]`, so
# two sequences that agree there and disagree about the gold cannot both be fitted -- the
# loss floors at the group's label entropy and an eval reports that floor as accuracy.
# Before these tests the only thing that would notice was a GPU-hours-long run.


def _prompt_collapsing_tokenize(text: str) -> list[int]:
    """Erase the prompt, keep the answer: every byte but the last becomes a constant.

    The defect at its sharpest, and a tokenizer failure rather than an invented corpus:
    whatever made two rows different is gone by the time the model sees it, and only the
    golds still disagree. One token per byte, so ``byte_offsets`` still describes it exactly
    and the span projection is not what is under test here.

    Note what this does *not* do: it does not collapse rows of different lengths together.
    Two rows contradict under it only if they were the same length and had different golds,
    which is why the test asserts it actually produced some.
    """
    raw = text.encode("utf-8")
    return [65] * (len(raw) - 1) + [raw[-1]]


def _collapsed_contradiction_count(snap: Snapshot, split_name: str) -> int:
    """How many rows the collapsing tokenizer actually makes contradict, counted directly.

    The precondition, measured rather than assumed: a fixture corpus that stopped producing
    same-length rows with different golds would make the refusal tests pass vacuously.
    """
    texts = _render_last(snap, split_name)
    seqs = [np.asarray(_prompt_collapsing_tokenize(t), dtype=np.int32) for t in texts]
    return int(
        corpus_contradictions(
            seqs,
            target_index=[len(s) - 2 for s in seqs],
            slot_kinds=[SLOT_CHOICE] * len(seqs),
            span_targets=[(NO_SPAN, NO_SPAN)] * len(seqs),
            labels=[str(i) for i in range(len(seqs))],
        )["contradicting_rows"]
    )


def test_write_shards_refuses_a_corpus_whose_rows_contradict_each_other(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """The write boundary refuses, where previously only the trainer reported."""
    n = _collapsed_contradiction_count(snapshot, "train")
    assert n > 0, (
        "the collapsing tokenizer produced no same-length pair with differing golds, so "
        "there is no contradiction for the writer to catch and this test proves nothing"
    )

    with pytest.raises(ShardContractViolation, match="disagree about the gold"):
        _write(
            snapshot,
            "train",
            tmp_path / "shards" / "contradictory",
            tokenize=_prompt_collapsing_tokenize,
        )


def test_a_refused_corpus_writes_nothing_at_all(snapshot: Snapshot, tmp_path: Path) -> None:
    """Refused *before* the first byte, not after.

    A writer that lays down tokens.u32 and then raises leaves a directory that looks like a
    shard set to anything that opens it by path, and the next run's failure is about a
    truncated memmap rather than about the corpus.
    """
    out = tmp_path / "shards" / "contradictory"
    with pytest.raises(ShardContractViolation):
        _write(snapshot, "train", out, tokenize=_prompt_collapsing_tokenize)
    assert not out.exists() or not any(out.iterdir()), (
        f"{out} holds {[p.name for p in out.iterdir()]} after a refused write"
    )


def test_a_contradictory_corpus_can_be_written_deliberately_and_says_so(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """The escape hatch carries both numbers, the way ``allow_unencodable`` does.

    Writing one deliberately is a decision about the corpus; the artifact recording that it
    was taken is not optional, because a set whose check ran and found nothing and a set
    written over the refusal must not read the same.
    """
    out = tmp_path / "shards" / "deliberate"
    _write(
        snapshot,
        "train",
        out,
        tokenize=_prompt_collapsing_tokenize,
        allow_contradictions=True,
    )
    report = json.loads((out / CONTRADICTION_NAME).read_text(encoding="utf-8"))
    assert report["allowed"] is True
    assert report["contradicting_rows"] > 0
    assert report["contradicting_groups"] > 0


def test_a_clean_corpus_records_a_clean_report_rather_than_no_report(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """The other half: the real corpus passes, and the passing is written down.

    This is the regression guard for over-refusal. ``train`` is the full corpus with span
    rows, choice rows and duplicated *prompts that do not contradict*; if the check ever
    starts counting those, this fails.
    """
    _snap, out, header = train_shards
    report = json.loads((out / CONTRADICTION_NAME).read_text(encoding="utf-8"))
    assert report["contradicting_rows"] == 0
    assert report["contradicting_groups"] == 0
    assert report["allowed"] is False
    assert report["sequences"] == header.n_sequences


def _render_last(snap: Snapshot, split_name: str) -> list[str]:
    """The rendered text of each row in a split, which is what the tokenizer is handed."""
    return [
        spec.text
        for row in sorted(snap.rows[split_name], key=lambda r: r.row_id)
        for spec in training_texts(row, seed=snap.config.seed, caps=DEFAULT_CAPS)
    ]


# -- the canonical checker itself ------------------------------------------------------


def _arrays(pairs: Sequence[tuple[list[int], int]]) -> dict[str, object]:
    """``(sequence, slot_kind)`` pairs as the five parallel arrays the checker takes."""
    seqs = [np.asarray(s, dtype=np.int32) for s, _ in pairs]
    return {
        "sequences": seqs,
        "target_index": [len(s) - 2 for s, _ in pairs],
        "slot_kinds": [k for _, k in pairs],
        "span_targets": [(NO_SPAN, NO_SPAN)] * len(pairs),
        "labels": [f"row{i}" for i in range(len(pairs))],
    }


def test_two_rows_with_one_prefix_and_two_golds_are_contradicting() -> None:
    args = _arrays([([7, 7, 65], SLOT_CHOICE), ([7, 7, 66], SLOT_CHOICE)])
    report = corpus_contradictions(args.pop("sequences"), **args)  # type: ignore[arg-type]
    assert report["contradicting_rows"] == 2
    assert report["contradicting_groups"] == 1
    assert report["rows_sharing_a_prefix"] == 2
    assert report["examples"][0]["prefix_tokens"] == 2  # type: ignore[index]


def test_two_rows_with_one_prefix_and_the_same_gold_are_not_contradicting() -> None:
    """A duplicate is wasted supervision, not an unfittable pair, and only one of those
    is a reason to refuse a corpus. ``rows_sharing_a_prefix`` still counts it, because
    the two facts are different and collapsing them loses the one that is merely wasteful.
    """
    args = _arrays([([7, 7, 65], SLOT_CHOICE), ([7, 7, 65], SLOT_CHOICE)])
    report = corpus_contradictions(args.pop("sequences"), **args)  # type: ignore[arg-type]
    assert report["contradicting_rows"] == 0
    assert report["rows_sharing_a_prefix"] == 2


def test_rows_that_differ_before_the_supervised_position_are_independent() -> None:
    args = _arrays([([7, 8, 65], SLOT_CHOICE), ([7, 9, 66], SLOT_CHOICE)])
    report = corpus_contradictions(args.pop("sequences"), **args)  # type: ignore[arg-type]
    assert report["contradicting_rows"] == 0
    assert report["rows_sharing_a_prefix"] == 0


def test_span_rows_are_compared_by_their_span_gold_not_their_last_token() -> None:
    """A span row's answer is in ``span_target``; its trailing token is structural. Comparing
    the wrong one would call every span pair identical and find nothing."""
    seqs = [np.asarray([7, 7, 65], dtype=np.int32)] * 2
    report = corpus_contradictions(
        seqs,
        target_index=[1, 1],
        slot_kinds=[SLOT_SPAN, SLOT_SPAN],
        span_targets=[(3, 4), (9, 10)],
        labels=["a", "b"],
    )
    assert report["contradicting_rows"] == 2
    assert report["per_kind"] == {str(SLOT_SPAN): {"groups": 1, "rows": 2}}


def test_the_same_ids_in_two_dtypes_are_one_prefix_not_two() -> None:
    """``write_shards`` holds freshly encoded arrays and ``ShardReader`` holds a memmap.
    Hashing raw bytes without normalising dtype would make the two callers disagree about
    whether a corpus contradicts itself -- the writer clean, the trainer not."""
    report = corpus_contradictions(
        [np.asarray([7, 7, 65], dtype=np.int32), np.asarray([7, 7, 66], dtype=np.int64)],
        target_index=[1, 1],
        slot_kinds=[SLOT_CHOICE, SLOT_CHOICE],
        span_targets=[(NO_SPAN, NO_SPAN)] * 2,
        labels=["a", "b"],
    )
    assert report["contradicting_rows"] == 2


# -- the padding gate, on the distribution that failed it -------------------------------


def _measured_lengths() -> list[int]:
    """The real train-set length distribution the GH200 runs read.

    An AUDIT artifact rather than a literal here, and rather than a synthetic stand-in:
    ``choose_buckets`` is fitted to a distribution, so a regression test for a bucketing
    failure has to use the distribution that failed. A generated one that happens to fail
    today would drift away from the defect it was written for.
    """
    payload = json.loads(
        (REPO_ROOT / "AUDIT" / "shard-lengths-2026-09-20.json").read_text(encoding="utf-8")
    )
    lengths = [int(n) for n in payload["lengths"]]
    assert len(lengths) == payload["n_sequences"] == 341
    return lengths


def test_the_default_bucketing_clears_the_padding_gate_on_the_set_that_failed_it() -> None:
    """The gate is read-only (rule 2); the bucketing is not, and it is what was wrong.

    At the old default of 8 this distribution wastes 25.66% against a 15% bar. That is the
    number in HANDOFF/gh200-2026-09-20.md, and it is what this asserts is gone.
    """
    lengths = _measured_lengths()
    state = padding_waste(lengths, choose_buckets(lengths))
    assert isinstance(state, Ran)
    assert state.passed, (
        f"the default bucketing wastes {state.value:.2%} of positions on the measured "
        f"train distribution, against a gate of {MAX_PADDING_WASTE:.0%}"
    )


def test_the_old_default_of_eight_buckets_is_what_failed_and_still_would() -> None:
    """The contrast, pinned. Without this the test above could be passing because the
    distribution is easy rather than because the default changed."""
    lengths = _measured_lengths()
    old = padding_waste(lengths, choose_buckets(lengths, n_buckets=8))
    assert isinstance(old, Ran)
    assert not old.passed
    assert old.value == pytest.approx(0.2566, abs=5e-5), (
        f"n_buckets=8 now wastes {old.value:.4%}; the failing GH200 run measured 25.66%, so "
        "either the distribution artifact or padding_waste has changed underneath this test"
    )


def test_more_buckets_never_orphans_a_sequence() -> None:
    """The thing a rebucketing could break: ``bucket_for`` refuses a length that fits no
    bucket, and a boundary set that orphans a row turns a padding problem into a refused
    write. Measured across the sweep rather than asserted for the chosen value alone."""
    lengths = _measured_lengths()
    for n_buckets in (8, 16, 32, 64):
        buckets = choose_buckets(lengths, n_buckets=n_buckets)
        for length in lengths:
            bucket_for(length, buckets)  # raises ValueError if it fits nowhere


def test_arrays_of_different_lengths_are_refused_rather_than_zipped_short() -> None:
    """``zip`` would silently truncate to the shortest, checking a prefix of the corpus and
    reporting it as the whole -- a capped sample presented as complete coverage."""
    with pytest.raises(ValueError, match="gold came from a different index"):
        corpus_contradictions(
            [np.asarray([7, 65], dtype=np.int32)] * 3,
            target_index=[0, 0, 0],
            slot_kinds=[SLOT_CHOICE, SLOT_CHOICE],
            span_targets=[(NO_SPAN, NO_SPAN)] * 3,
            labels=["a", "b", "c"],
        )


# -- the code that made the rows, pinned beside the corpus they came from ---------------
#
# GAP-SHARD-SET-GOES-STALE-AGAINST-THE-CORPUS-CODE-THAT-REPRODUCES-ITS-LABELS. The header
# pinned the corpus, the tokenizer and the remap; between the corpus and the rows sits
# `qd_data`, and nothing pinned that. A set with all three hashes matching was found to
# reproduce 321 rows where it stored 341.


def test_a_written_shard_set_records_the_code_that_produced_its_rows(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """On disk, not merely on the object: the reader gets the file, not the return value."""
    _, out, header = train_shards
    raw = json.loads((out / HEADER_NAME).read_text(encoding="utf-8"))

    assert raw["code_fingerprint"], "header.json carries no code_fingerprint"
    assert raw["code_fingerprint"] == header.code_fingerprint
    assert raw["code_fingerprint"] == code_fingerprint()
    # The two modules whose drift caused the recorded incident are both covered.
    assert "mixture.py" in raw["code_fingerprint"]
    assert "render.py" in raw["code_fingerprint"]


def test_opening_a_set_whose_generating_code_moved_is_refused(
    train_shards: tuple[Snapshot, Path, ShardHeader], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case no count check can see.

    Every hash in this header still matches and every byte of the shard set is the one that
    was written. Only `qd_data` moved -- which is the incident exactly, minus the accident
    that made that one visible: there the drift ALSO changed the row count, and
    `tools/real_ft_run.py` compares the reconstructed order against supervision.npz entry
    for entry. A change that moves which label attaches to which row without moving how
    many rows there are produces no count mismatch anywhere.
    """
    snap, out, _ = train_shards
    # Fine before the drift, so the refusal below is the drift and not the fixture.
    ShardReader(out, config=snap.config, repo_root=snap.root)

    drifted = dict(code_fingerprint())
    drifted["render.py"] = "0" * 64
    monkeypatch.setattr(artifacts, "code_fingerprint", lambda: drifted)

    with pytest.raises(ShardContractViolation, match="qd_data has changed"):
        ShardReader(out, config=snap.config, repo_root=snap.root)
    # Raw, because the dot is a metacharacter and the point of this assertion is that the
    # refusal names the MODULE -- a pattern that would also match "renderXpy" is not that.
    with pytest.raises(ShardContractViolation, match=r"render\.py"):
        ShardReader(out, config=snap.config, repo_root=snap.root)


# -- the revision the corpus was read at, pinned beside the code that read it -----------
#
# GAP-SHARD-HEADER-DOES-NOT-RECORD-THE-REV-ITS-CORPUS-WAS-READ-AT. The third pin, and the
# one neither of the other two can stand in for: `data_snapshot_hash` hashes the rows that
# came out and `code_fingerprint` hashes the code that turned them into rows, so a set
# built at the WRONG REVISION is self-consistent in both and passes everything. Recovering
# the rev from a ledger row's free-text notes, which is what 2026-09-21 required, is not a
# pin. These are the reader-side half; `test_artifacts.py` holds the header-side half.


def test_a_written_shard_set_records_the_revision_its_corpus_was_read_at(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """On disk, not merely on the object -- the reader gets the file."""
    out = tmp_path / "shards" / "at-a-rev"
    header = _write(snapshot, "train", out, corpus_rev="0632f69")
    raw = json.loads((out / HEADER_NAME).read_text(encoding="utf-8"))

    assert raw["corpus_rev"] == "0632f69"
    assert raw["corpus_rev"] == header.corpus_rev


def test_a_set_written_without_a_revision_stays_readable_and_reads_not_run(
    train_shards: tuple[Snapshot, Path, ShardHeader],
) -> None:
    """Every shard set that exists today predates the field, ``~/shardset-v2`` included.

    Adding a pin must not brick them: they open, their ledger rows keep their evidence, and
    the check reports ``NotRun`` -- which is the honest answer and is emphatically not a
    pass. A field that refused old sets would be reverted on the first full train, and a
    field that passed them would be a check reporting on a comparison it could not make.
    """
    snap, out, _ = train_shards
    reader = ShardReader(
        out, config=snap.config, repo_root=snap.root, expect_rev="0632f69"
    )
    state = reader.checks["shard_rev_matches"]
    assert isinstance(state, NotRun)
    assert "carries no corpus_rev" in state.reason


def test_opening_a_set_built_at_another_revision_is_refused(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """The case no hash in this header can see.

    Every byte of the set is the one that was written and every other check still passes.
    It matters because ``tools/real_ft_run.py`` does not read labels out of the shard set:
    it RECONSTRUCTS them from the repository at ``--rev`` and pairs them with these
    sequences. The 2026-09-21 mismatch was caught only because it moved the row COUNT, 321
    against 341; a revision that changes which rows exist without changing how many lands
    every label on the wrong sequence with no other symptom.
    """
    out = tmp_path / "shards" / "built-at-0632f69"
    _write(snapshot, "train", out, corpus_rev="0632f69")
    # Fine at its own revision, so the refusal below is the mismatch and not the fixture.
    ShardReader(out, config=snapshot.config, repo_root=snapshot.root, expect_rev="0632f69")

    with pytest.raises(ShardContractViolation) as excinfo:
        ShardReader(
            out, config=snapshot.config, repo_root=snapshot.root, expect_rev="deadbee"
        )
    message = str(excinfo.value)
    assert "0632f69" in message and "deadbee" in message, (
        "a refusal that names neither revision leaves the reader to guess which of the two "
        "is the wrong one"
    )


def test_a_reader_that_names_no_revision_has_not_checked_anything(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """The other half of the middle answer, and the one that is easy to get wrong.

    A header that pins its rev, opened by a caller that never said what it expected, is a
    set describing itself -- not a set that was verified. ``passed=True`` there would be a
    check reporting on a comparison it skipped, which is exactly how "approved" comes to
    mean "unexamined".
    """
    out = tmp_path / "shards" / "pinned"
    _write(snapshot, "train", out, corpus_rev="0632f69")

    silent = ShardReader(out, config=snapshot.config, repo_root=snapshot.root)
    state = silent.checks["shard_rev_matches"]
    assert isinstance(state, NotRun)
    assert "named no revision" in state.reason

    asked = ShardReader(
        out, config=snapshot.config, repo_root=snapshot.root, expect_rev="0632f69"
    )
    matched = asked.checks["shard_rev_matches"]
    assert isinstance(matched, Ran) and matched.passed
    assert matched.value == "0632f69"


def test_a_deliberate_cross_revision_read_is_recorded_as_a_failure_not_a_pass(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    """``allow_rev_mismatch`` buys admission, not a clean record.

    Same contract as ``allow_stale_code`` and ``write_shards(allow_contradictions=...)``:
    the escape hatch stops the raise and leaves ``Ran(passed=False)`` in the checks, so a
    ledger row written from a deliberate cross-revision read still says so. An escape hatch
    that also launders the check is how a knowingly-wrong run comes to look like a clean
    one three weeks later.
    """
    out = tmp_path / "shards" / "read-across"
    _write(snapshot, "train", out, corpus_rev="0632f69")

    reader = ShardReader(
        out,
        config=snapshot.config,
        repo_root=snapshot.root,
        expect_rev="deadbee",
        allow_rev_mismatch=True,
    )
    state = reader.checks["shard_rev_matches"]
    assert isinstance(state, Ran) and not state.passed
    assert state.value == "0632f69"
    assert len(reader) > 0, "admitted, and the set is actually usable"
