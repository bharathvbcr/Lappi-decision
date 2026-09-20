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

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from data_fixtures import small_corpus

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.errors import EmptyContextRefusal, HeldOutViolation
from qd_data.manifest import Manifest, build_manifests
from qd_data.mixture import _line_span as mixture_line_span
from qd_data.mixture import build_mixture
from qd_data.render import render
from qd_data.rows import DataRow
from qd_data.schema import Request, SpanSlot
from qd_data.split import HELD_OUT, split
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
    line_start_indices,
    padding_waste,
)
from qd_train.byte_context import line_starts as byte_line_starts
from qd_train.shards import (
    COVERAGE_NAME,
    HEADER_NAME,
    MAX_ROWS_PER_BATCH,
    PAD_ID,
    SUPERVISION_NAME,
    TOKENS_NAME,
    SequenceSpec,
    ShardReader,
    UnencodableGold,
    choose_buckets,
    line_starts,
    training_texts,
    write_shards,
)
from qd_train.tristate import NotRun, Ran

SOURCE_VOCAB = 512
TOKENIZER_HASH = "tokhash-0123456789abcdef"


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
    """A narrow bucket plus a generous token budget must not make an unbounded batch."""
    monkeypatch.setattr(shards_module, "MAX_ROWS_PER_BATCH", 2)
    width = reader.header.buckets[-1]
    batches = list(reader.batches(batch_tokens=width * 1000, seed=1, epoch=0))
    assert batches, "the bound must not empty the epoch"
    assert max(b.tokens.shape[0] for b in batches) <= 2
    assert MAX_ROWS_PER_BATCH == 4096, "the shipped default is still a real bound"


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
