"""Statement (1) of the span decode check over characters a byte-level BPE splits.

The 2026-09-29 shard rebuild with real SQuAD v2 aborted at stage 6 on
``squad:qa.answer_span:56df6a8d5ca0a614008f99da``: a context line begins with ``Ṣ``
(three UTF-8 bytes), Qwen3.5's tokenizer emits it as three byte tokens that all claim the
character's span, and the first of them -- correctly chosen as the line-start token --
decoded alone to U+FFFD. The tokenizer was honest; the check compared a part of a
character with the whole of it. Observed offsets for ``"x\\nṢab"``:
``(0,1) (1,2) (2,3) (2,3) (2,3) (3,5)``, and with a merge into the preceding space,
``(19,21) (20,21) (20,21)``.

The fix compares the *run* of tokens that overlap the checked token's span, decoded
together, against the characters that run covers. A tokenizer whose offsets describe a
different string must still be refused -- the last two tests hold that line.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from qd_train.artifacts import SLOT_SPAN, ShardContractViolation
from qd_train.shards import SequenceSpec, _assert_spans_decode_to_their_text

S_DOT = "Ṣ"  # LATIN CAPITAL LETTER S WITH DOT BELOW, 3 bytes in UTF-8


def _bytes_of(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """UTF-8 byte ids with HuggingFace byte-level offsets: each byte claims its character."""
    ids: list[int] = []
    offsets: list[tuple[int, int]] = []
    for ci, ch in enumerate(text):
        for b in ch.encode("utf-8"):
            ids.append(b)
            offsets.append((ci, ci + 1))
    return ids, offsets


def replacing_decode(ids: Sequence[int]) -> str:
    """What a byte-level tokenizer's ``decode`` does with a partial character."""
    return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


def _spec(text: str) -> SequenceSpec:
    starts = (0, *(i + 1 for i, c in enumerate(text) if c == "\n"))
    return SequenceSpec(
        slot_name="evidence", text=text, slot_kind=SLOT_SPAN, span_char_starts=None,
        span_abstains=True, line_char_starts=starts,
    )


def test_a_line_that_starts_with_a_byte_split_character_passes() -> None:
    text = f"x\n{S_DOT}ab\nc"
    ids, offsets = _bytes_of(text)
    first_piece = 2  # 'x', '\n', then the first of the character's three bytes
    assert offsets[first_piece : first_piece + 3] == [(2, 3)] * 3
    _assert_spans_decode_to_their_text(
        _spec(text), ids, offsets, (0, first_piece, 7), decode=replacing_decode, where="t"
    )


def test_a_split_character_merged_into_the_preceding_token_passes() -> None:
    """Qwen's shape: ``' Ṣ'`` is one token over (19, 21), then two bytes over (20, 21)."""
    text = f"is {S_DOT}al"
    raw = text.encode("utf-8")
    # ids: 'i', 's', then ' '+first byte merged, the two continuation bytes, 'a', 'l'.
    pieces = [raw[0:1], raw[1:2], raw[2:4], raw[4:5], raw[5:6], raw[6:7], raw[7:8]]
    offsets = [(0, 1), (1, 2), (2, 4), (3, 4), (3, 4), (4, 5), (5, 6)]
    vocab = {i: p for i, p in enumerate(pieces)}

    def decode(ids: Sequence[int]) -> str:
        return b"".join(vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    spec = SequenceSpec(
        slot_name="evidence", text=text, slot_kind=SLOT_SPAN, span_char_starts=None,
        span_abstains=True, line_char_starts=(0,),
    )
    _assert_spans_decode_to_their_text(
        spec, list(vocab), offsets, (0, 2), decode=decode, where="t"
    )


def test_a_middle_piece_of_a_split_character_is_still_refused() -> None:
    """A candidate that lands on a continuation byte is a wrong candidate."""
    text = f"x\n{S_DOT}ab"
    ids, offsets = _bytes_of(text)
    with pytest.raises(ShardContractViolation, match="decodes to"):
        _assert_spans_decode_to_their_text(
            _spec(text), ids, offsets, (0, 3), decode=replacing_decode, where="t"
        )


def test_offsets_over_a_normalised_copy_are_still_refused() -> None:
    """The check's reason to exist: offsets measured on ``e + combining acute``, ids that
    decode to the precomposed ``é``. Same reach, different characters."""
    text = "ab\néx"
    ids = [ord("a"), ord("b"), ord("\n"), 0xE9, ord("x")]
    offsets = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 6)]

    def decode(seq: Sequence[int]) -> str:
        return "".join(chr(int(i)) for i in seq)

    with pytest.raises(ShardContractViolation, match="decodes to"):
        _assert_spans_decode_to_their_text(
            _spec(text), ids, offsets, (0, 3), decode=decode, where="t"
        )


QWEN_TOKENIZER = (
    "/Users/bharath/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/"
    "b1485b2fa6dfa1287294f269f5fb618e03d52d7c/tokenizer.json"
)


def test_text_the_real_tokenizer_nfc_normalises_refuses_its_slot_not_the_write() -> None:
    """The 2026-10-01 corpus-v3 build aborted at stage 6 on
    ``qdm:code.defect_class:7d1d1a07...:gem/migrations/0014_add_default_tags.py#0``: a diff
    line holds Bengali U+09DF, a composition exclusion that Qwen3.5's NFC normalizer
    decomposes, so the ids decode one character longer than the text and every later line
    start is off by one. The offsets were honest (aligned to the original); the text is one
    the tokenizer rewrites, so its line mapping cannot be verified against decoded text.
    That is this row's property -- refused for its slot, counted -- not a wiring fault."""
    tokenizers = pytest.importorskip("tokenizers")
    from pathlib import Path

    from qd_train.remap import full_vocab_remap
    from qd_train.shards import UnencodableGold, encode_slot

    if not Path(QWEN_TOKENIZER).is_file():
        pytest.skip("the Qwen3.5 tokenizer.json is not in this machine's HF cache")
    tok = tokenizers.Tokenizer.from_file(QWEN_TOKENIZER)
    # Spelled as escapes: an editor that normalises its buffer would silently remove U+09DF.
    text = "@@ -1,1 +1,4 @@\n+t = 'পিরিয়ড'\n+x = 1\n+y = 2"
    assert "য়" in text
    remap = full_vocab_remap(
        source_vocab_size=tok.get_vocab_size(), tokenizer_hash="qwen", special_ids=()
    )
    with pytest.raises(UnencodableGold, match="NFC"):
        encode_slot(
            _spec(text), remap=remap, where="t",
            tokenize=lambda t: tok.encode(t, add_special_tokens=False).ids,
            token_offsets=lambda t: list(tok.encode(t, add_special_tokens=False).offsets),
            decode=lambda ids: tok.decode([int(i) for i in ids], skip_special_tokens=False),
        )


def test_a_byte_split_run_that_decodes_to_other_characters_is_refused() -> None:
    """Overlapping offsets widen what is compared; they do not excuse a wrong decode."""
    text = f"x\n{S_DOT}ab"
    ids, offsets = _bytes_of(text)

    def lying(seq: Sequence[int]) -> str:
        out = replacing_decode(seq)
        return out.replace(S_DOT, "S")

    with pytest.raises(ShardContractViolation, match="decodes to"):
        _assert_spans_decode_to_their_text(
            _spec(text), ids, offsets, (0, 2), decode=lying, where="t"
        )
