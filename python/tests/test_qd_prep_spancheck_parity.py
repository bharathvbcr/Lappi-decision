"""``qd-prep spancheck`` answers what ``qd_train.shards._span_token_positions`` answers.

Stage 6 of ``tools/real_tokenizer_pipeline.py`` is ``write_shards``, and its per-slot Python is
the span projection: the offsets, their reach, the token holding each line start, the collapse
rule, the gold's checks, and the decode check's runs and line grid
(``AUDIT/perf-pipeline-shards-2026-09-30.md``). Under ``--native-spancheck`` that work runs in
``crates/qd-prep/src/spancheck.rs`` (``build_native_spancheck``); decoding stays in Python, and
``qd_train.shards`` is the unedited reference every unsettled lookup goes back to. These tests
hold the port to it:

* the reply's status class, value, line-grid and NFC flags and runs against the reference with
  ``decode=None``, on adversarial offsets (unsorted, overlapping, empty, byte pieces, specials)
  over multibyte, astral, combining and NFC-unstable text with CRLF and lone-CR lines, and the
  status's detail against the reference's own message;
* named cases for every refusal the reference can reach, spans at the start and the end of a
  context, an empty one, an abstaining one, under both collapse policies;
* ``write_shards`` through the installed table: every artifact identical to the reference's
  (``header.json`` without its ``created_at``; each ``.npz`` by its arrays, because the zip
  members carry their write time; every other file byte for byte), with ``decode`` and without,
  under both policies, with real collapses; and the three dishonest decodes of ``test_shards``
  raising the reference's own exception, text and all;
* every refusal of the installer: a lookup the table does not hold, a table never read, another
  offsets callable, other ids, a canary that disagrees, a reply of the wrong shape, a request
  past the binary's bound, a sequence the wire cannot carry, a second install;
* with the Qwen3.5 tokenizer when this machine has it, real BPE offsets.

The binary is built by ``conftest.qd_prep_bin``; without cargo these are SKIPPED, never passed.
The benchmark runs only with ``QD_PREP_BENCH=1``.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
import os
import struct
import subprocess
import sys
import time
import unicodedata
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_shards import (
    _BLANK_PAIR_LINES,
    _BLANK_PAIR_TEXT,
    _blank_pair_tokens,
    _snapshot,
    _write,
    byte_decode,
    byte_offsets,
    byte_tokenize,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import real_tokenizer_pipeline as pipeline  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.errors import QdRefusal  # noqa: E402
from qd_train import shards as shards_module  # noqa: E402
from qd_train.artifacts import (  # noqa: E402
    SLOT_SPAN,
    SPAN_ABSTAIN,
    ShardContractViolation,
    line_start_indices,
)
from qd_train.shards import (  # noqa: E402
    HEADER_NAME,
    SEQUENCE_INDEX_NAME,
    SPAN_COLLAPSE_REFUSE_ANY,
    SPAN_COLLAPSE_REFUSE_GOLD,
    SequenceSpec,
    UnencodableGold,
    training_texts,
)

CONFIG = DataConfig()
ANY, GOLD = SPAN_COLLAPSE_REFUSE_ANY, SPAN_COLLAPSE_REFUSE_GOLD
#: The checkout whose ``data/pool`` the end-to-end build and the benchmark read (only read).
#: A worktree holds the pool's tracked manifests and none of its downloads, so point this at
#: a checkout that has them; without them those tests are SKIPPED, never passed.
DATA_ROOT = Path(os.environ.get("QD_PREP_DATA_ROOT", str(REPO)))
REFERENCE = pipeline._REFERENCE_SPAN_TOKEN_POSITIONS
#: The ``qd-prep`` path of the tests that replace ``_run_prep``: nothing runs, nothing is built.
NOT_RUN = Path("/nonexistent/qd-prep")
#: One drawn sequence: its wire item, the offsets it was drawn with, and its ids.
Drawn = tuple[pipeline.SpancheckItem, list[tuple[int, int]], list[int]]
#: Characters chosen for what they do to offsets and lines: ASCII, a space, LF, CR, a two-byte
#: precomposed letter, a combining acute (so NFD sequences occur), a three-byte letter a
#: byte-level BPE splits into pieces, an astral emoji (four bytes, one code point), and two
#: characters NFC rewrites (U+09DF, a composition exclusion, and U+212B, the Angstrom sign).
#: Spelled by name: an editor that normalises its buffer would silently rewrite the last three.
ALPHABET = [
    "a", "b", " ", "\n", "\r", "\N{LATIN SMALL LETTER E WITH ACUTE}",
    "\N{COMBINING ACUTE ACCENT}", "\N{LATIN CAPITAL LETTER S WITH DOT BELOW}", "\U0001f9ea",
    "\N{BENGALI LETTER YYA}", "\N{ANGSTROM SIGN}",
]
#: The reference's message names what refused; the reply's detail must be that same thing.
DETAIL_IN_MESSAGE = {
    "offset_count": "token_offsets returned {detail} offsets",
    "reach": "token_offsets reach character {detail} of",
    "line_in_no_token": "character {detail} lies in no token",
    "lines_collapse": "map to only {detail} distinct tokens",
    "gold_in_no_token": "character {detail} lies in no token",
    "gold_reversed": "maps to token {start} and its last to {end}",
    "gold_in_one_token": "both fall inside token {detail}",
    "gold_not_candidate": "gold position(s) [{detail}",
    "gold_shares_token": "shares token(s) [{detail}",
}


def byte_encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """``test_shards``' byte stand-in, ids and offsets from one call."""
    return byte_tokenize(text), byte_offsets(text)


def _reference(
    spec: SequenceSpec,
    ids: Sequence[int],
    offsets: list[tuple[int, int]],
    policy: str,
    decode: Callable[[Sequence[int]], str] | None = None,
) -> tuple[Any, ...]:
    return pipeline._outcome(functools.partial(
        REFERENCE, spec, np.asarray(ids), token_offsets=lambda _text: offsets, decode=decode,
        where="w", span_collapse_policy=policy,
    ))


def _reference_run(offsets: list[tuple[int, int]], pos: int, n: int) -> tuple[int, int, int]:
    """The run ``_assert_spans_decode_to_their_text`` walks from ``pos`` -- its loop under "A
    byte-level BPE splits a multi-byte character", transcribed -- as ``(run_end, first, last)``."""
    first, last = offsets[pos]
    run_end = pos + 1
    while (
        run_end < n
        and first <= offsets[run_end][0] < last
        and offsets[run_end][1] > offsets[run_end][0]
    ):
        last = max(last, offsets[run_end][1])
        run_end += 1
    return run_end, first, last


def _spec(text: str, lines: Sequence[int], gold: tuple[int, int] | None, *,
          flagged_abstain: bool = False) -> SequenceSpec:
    return SequenceSpec(
        slot_name="evidence", text=text, slot_kind=SLOT_SPAN, span_char_starts=gold,
        span_abstains=flagged_abstain, line_char_starts=tuple(lines),
    )


def _assert_agrees(
    item: pipeline.SpancheckItem, reply: pipeline.SpanReply, offsets: list[tuple[int, int]],
    ids: Sequence[int],
) -> tuple[Any, ...]:
    """The reply against the reference with ``decode=None``: the same value, or a refusal of
    the same exception class whose detail the reference's message names. Returns the outcome."""
    want = _reference(item.spec, ids, offsets, item.policy)
    if want[0] == "ok":
        assert reply.status_name == "ok", (reply, want)
        assert pipeline._reply_value(reply, abstains=item.abstains) == want[1]
        text, lines = item.spec.text, item.spec.line_char_starts or ()
        off_grid = not set(lines) <= set(line_start_indices(text))
        assert bool(reply.flags & pipeline._SC_OFF_GRID) == off_grid
        assert bool(reply.flags & pipeline._SC_NFC_UNSTABLE) == (
            not unicodedata.is_normalized("NFC", text)
        )
        gold = [] if item.abstains else [reply.start, reply.end]
        positions = gold + reply.candidates.tolist()
        assert reply.runs.tolist() == [
            list(_reference_run(offsets, p, len(ids))) for p in positions
        ]
        return want
    kind, cls, message = want
    assert kind == "raise"
    assert reply.status_name != "ok", (reply, message)
    codes = pipeline._SC_UNENCODABLE if cls is UnencodableGold else pipeline._SC_CONTRACT
    assert cls in (UnencodableGold, ShardContractViolation), message
    assert reply.status in codes, (reply.status_name, message)
    assert reply.status_name in DETAIL_IN_MESSAGE, reply.status_name
    named = DETAIL_IN_MESSAGE[reply.status_name].format(
        detail=reply.detail, start=reply.start, end=reply.end
    )
    assert named in message, (named, message)
    return want


@st.composite
def _sequences(draw: st.DrawFn) -> Drawn:
    text = draw(st.text(alphabet=st.sampled_from(ALPHABET), max_size=24))
    n = len(text)
    shape = draw(st.sampled_from(["bytes", "chars", "merged", "specials", "random"]))
    cuts = sorted(draw(st.sets(st.integers(1, n - 1), max_size=n))) if n > 1 else []
    bounds = [0, *cuts, n] if n else []
    merged = list(pairwise(bounds))
    if shape == "bytes":
        offsets = byte_offsets(text)
    elif shape == "chars":
        offsets = [(i, i + 1) for i in range(n)]
    elif shape == "merged":
        # A monotonic tokenization whose tokens may hold several lines: collapses.
        offsets = merged
    elif shape == "specials":
        offsets = list(merged)
        for _ in range(draw(st.integers(0, 3))):
            offsets.insert(draw(st.integers(0, len(offsets))), (0, 0))
        if offsets and draw(st.booleans()):
            # A character split into byte pieces that each claim it.
            j = draw(st.integers(0, len(offsets) - 1))
            offsets[j + 1 : j + 1] = [offsets[j]] * draw(st.integers(1, 3))
    else:
        offsets = draw(st.lists(
            st.tuples(st.integers(0, n + 2), st.integers(0, n + 2)), max_size=n + 3
        ))
        if offsets and draw(st.booleans()):
            offsets[-1] = (offsets[-1][0], n)
    lines = list(line_start_indices(text))
    if not lines or draw(st.booleans()):
        lines = draw(st.lists(st.integers(0, n + 2), min_size=1, max_size=6))
    gold_kind = draw(st.sampled_from(["abstain", "flagged", "lines", "random"]))
    if gold_kind == "lines":
        i = draw(st.integers(0, len(lines) - 1))
        gold: tuple[int, int] | None = (lines[i], lines[draw(st.integers(i, len(lines) - 1))])
    elif gold_kind in ("random", "flagged"):
        # "flagged" also sets span_abstains, which the reference reads before the positions.
        gold = (draw(st.integers(0, n + 2)), draw(st.integers(0, n + 2)))
    else:
        gold = None
    spec = _spec(text, lines, gold, flagged_abstain=gold_kind == "flagged")
    n_ids = max(0, len(offsets) + draw(st.sampled_from([0, 0, 0, -1, 1])))
    policy = draw(st.sampled_from([ANY, GOLD]))
    item = pipeline.spancheck_item(spec, policy=policy, n_ids=n_ids, offsets=offsets)
    return item, offsets, list(range(n_ids))


@settings(max_examples=200, deadline=None)
@given(batch=st.lists(_sequences(), min_size=1, max_size=8))
def test_every_reply_is_the_references_answer_on_adversarial_offsets(
    qd_prep_bin: Path, batch: list[Drawn]
) -> None:
    """Several sequences per request, so the columnar reply's slicing is exercised too."""
    replies = pipeline._prep_spancheck(qd_prep_bin, [item for item, _, _ in batch])
    for (item, offsets, ids), reply in zip(batch, replies, strict=True):
        _assert_agrees(item, reply, offsets, ids)


_PER_CHAR = [(i, i + 1) for i in range(5)]
_BLANK_PAIR_OFFSETS = _blank_pair_tokens(_BLANK_PAIR_TEXT)[1]
_LINES3 = "one\ntwo\nthree"
#: "e" and a combining acute, then a line: NFD, which NFC would compose into one code point.
_NFD = "e\N{COMBINING ACUTE ACCENT}\nx"

#: name -> (text, offsets, line starts, gold or None, policy, status, detail or None, n_ids or None)
CASES: dict[str, tuple[Any, ...]] = {
    "collapse_refuse_any": (
        _BLANK_PAIR_TEXT, _BLANK_PAIR_OFFSETS, _BLANK_PAIR_LINES, (11, 11), ANY,
        "lines_collapse", 5, None,
    ),
    "collapse_missing_the_gold": (
        _BLANK_PAIR_TEXT, _BLANK_PAIR_OFFSETS, _BLANK_PAIR_LINES, (11, 11), GOLD, "ok", None,
        None,
    ),
    "collapse_on_the_gold": (
        _BLANK_PAIR_TEXT, _BLANK_PAIR_OFFSETS, _BLANK_PAIR_LINES, (9, 9), GOLD,
        "gold_shares_token", 6, None,
    ),
    "collapse_abstaining": (
        _BLANK_PAIR_TEXT, _BLANK_PAIR_OFFSETS, _BLANK_PAIR_LINES, None, GOLD, "ok", None, None,
    ),
    "offset_count": ("ab\ncd", _PER_CHAR, (0, 3), None, ANY, "offset_count", 5, 4),
    "reach": ("ab\ncd", [(0, 2), (2, 4)], (0, 3), None, ANY, "reach", 4, None),
    "line_in_no_token": ("ab\ncd", [(0, 1), (2, 5)], (0, 1, 3), None, ANY,
                         "line_in_no_token", 1, None),
    "line_past_the_text": ("ab\ncd", _PER_CHAR, (0, 9), None, ANY, "line_in_no_token", 9, None),
    "gold_in_no_token": ("ab\ncd", [(0, 1), (2, 5)], (0, 3), (0, 1), ANY,
                         "gold_in_no_token", 1, None),
    "gold_reversed": ("ab\ncd", [(3, 5), (0, 3)], (0, 3), (0, 3), ANY, "gold_reversed", 1,
                      None),
    "gold_in_one_token": ("a\nb\nc", [(0, 1), (1, 2), (2, 5)], (0, 2, 4), (2, 4), GOLD,
                          "gold_in_one_token", 2, None),
    "gold_not_candidate": ("ab\ncd", _PER_CHAR, (0, 3), (1, 1), ANY, "gold_not_candidate", 1,
                           None),
    "span_at_the_start": (_LINES3, byte_offsets(_LINES3), (0, 4, 8), (0, 0), ANY, "ok", None,
                          None),
    "span_at_the_end": (_LINES3, byte_offsets(_LINES3), (0, 4, 8), (8, 8), ANY, "ok", None,
                        None),
    "span_over_the_whole_context": (_LINES3, byte_offsets(_LINES3), (0, 4, 8), (0, 8), ANY,
                                    "ok", None, None),
    "crlf_lines": ("a\r\nb\r\nc", byte_offsets("a\r\nb\r\nc"), (0, 3, 6), (3, 6), ANY, "ok",
                   None, None),
    "lone_cr_is_content": ("a\rb\rc\nd", byte_offsets("a\rb\rc\nd"), (0, 6), (6, 6), ANY, "ok",
                           None, None),
    "line_start_off_the_grid": ("a\r\nb", byte_offsets("a\r\nb"), (0, 2), None, ANY, "ok",
                                None, None),
    "nfd_text": (_NFD, byte_offsets(_NFD), (0, 3), (3, 3), ANY, "ok", None, None),
    "astral_and_trailing_newline": (
        "\U0001f9ea\nz\n", byte_offsets("\U0001f9ea\nz\n"), (0, 2), (2, 2), ANY, "ok", None,
        None,
    ),
    "empty_text": ("", [], (0,), None, ANY, "line_in_no_token", 0, None),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_named_case_is_answered_as_the_reference_answers_it(
    qd_prep_bin: Path, name: str
) -> None:
    text, offsets, lines, gold, policy, status, detail, n_ids = CASES[name]
    spec = _spec(text, lines, gold)
    ids = list(range(len(offsets) if n_ids is None else n_ids))
    item = pipeline.spancheck_item(spec, policy=policy, n_ids=len(ids), offsets=offsets)
    (reply,) = pipeline._prep_spancheck(qd_prep_bin, [item])
    assert reply.status_name == status
    if detail is not None:
        assert reply.detail == detail
    _assert_agrees(item, reply, list(offsets), ids)


def test_the_blank_pair_keeps_its_collapse_under_refuse_gold(qd_prep_bin: Path) -> None:
    """``test_shards``' pinned answer, through the binary."""
    spec = _spec(_BLANK_PAIR_TEXT, _BLANK_PAIR_LINES, (11, 11))
    item = pipeline.spancheck_item(
        spec, policy=GOLD, n_ids=len(_BLANK_PAIR_OFFSETS), offsets=_BLANK_PAIR_OFFSETS
    )
    (reply,) = pipeline._prep_spancheck(qd_prep_bin, [item])
    assert pipeline._reply_value(reply, abstains=False) == ((8, 8), (0, 4, 6, 6, 8, 11))


# -- write_shards through the installed table ------------------------------------------------


def _artifacts(out: Path) -> dict[str, Any]:
    """Every file of a shard set, comparably: ``header.json`` without ``created_at``, each
    ``.npz`` by its arrays (a zip member carries its write time), everything else byte for
    byte."""
    got: dict[str, Any] = {}
    for path in sorted(out.iterdir()):
        if path.name == HEADER_NAME:
            header = json.loads(path.read_text(encoding="utf-8"))
            header.pop("created_at")
            got[path.name] = header
        elif path.suffix == ".npz":
            with np.load(path) as npz:
                got[path.name] = {
                    k: (npz[k].dtype.str, npz[k].shape, npz[k].tobytes())
                    for k in sorted(npz.files)
                }
        else:
            got[path.name] = path.read_bytes()
    return got


@pytest.mark.parametrize("policy", [ANY, GOLD])
@pytest.mark.parametrize("decode", [byte_decode, None], ids=["decode", "no-decode"])
def test_write_shards_through_the_table_is_the_reference_byte_for_byte(
    qd_prep: Path, tmp_path: Path, policy: str, decode: Callable[[Sequence[int]], str] | None
) -> None:
    snap = _snapshot(tmp_path)
    _write(snap, "train", tmp_path / "reference", decode=decode, span_collapse_policy=policy)
    with pipeline.native_spancheck(
        [(snap.rows["train"], policy)], config=snap.config, encode=byte_encode,
        token_offsets=byte_offsets, decode=decode,
    ) as table:
        assert shards_module._span_token_positions == table.lookup, "installed for the block"
        _write(snap, "train", tmp_path / "native", decode=decode, span_collapse_policy=policy)
    assert shards_module._span_token_positions is REFERENCE
    # An honest byte tokenizer: every span slot answered from the reply, none by the reference.
    assert table.reads > 0
    assert table.native_answers == table.reads, dict(table.to_reference)
    assert table.canaries > 0 and table.passthrough > 0
    assert _artifacts(tmp_path / "native") == _artifacts(tmp_path / "reference")


def _chunks(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Fixed-width tokens whose width is a hash of the text: one character on about half the
    rows, where no two line starts can share a token, and 64-127 characters on the rest, where
    the fixture's lines collapse, on the gold and off it. A width of 64 or more alone collapses
    every row of the fixture, which leaves refuse-any nothing to answer from the reply. Ids are
    one byte of a hash of each chunk, inside ``test_shards``' byte remap; not decodable, so
    decode=None."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=1).digest()[0]
    width = 1 if digest < 128 else 64 + digest % 64
    bounds = [*range(0, len(text), width), len(text)]
    offsets = list(pairwise(bounds))
    ids = [hashlib.blake2b(text[a:b].encode("utf-8"), digest_size=1).digest()[0]
           for a, b in offsets]
    return ids, offsets


def _chunk_tokenize(text: str) -> list[int]:
    return _chunks(text)[0]


def _chunk_offsets(text: str) -> list[tuple[int, int]]:
    return _chunks(text)[1]


def _slot_refusals(out: Path) -> list[str]:
    index = json.loads((out / SEQUENCE_INDEX_NAME).read_text(encoding="utf-8"))
    return [e["detail"] for e in index["excluded"] if e["scope"] == "slot"]


def test_collapsed_lines_are_refused_or_shared_as_the_reference_does(
    qd_prep: Path, tmp_path: Path
) -> None:
    """Hashed one-byte ids can make two sequences' prefixes equal by accident, which the
    writer's contradiction check would refuse in both writes alike; allowed here, because what
    is compared is the projection, and ``contradictions.json`` is compared with the rest."""
    snap = _snapshot(tmp_path)
    kw: dict[str, Any] = {
        "tokenize": _chunk_tokenize, "token_offsets": _chunk_offsets,
        "allow_unencodable": True, "allow_contradictions": True,
    }
    answered: dict[str, int] = {}
    for policy in (ANY, GOLD):
        _write(snap, "train", tmp_path / f"reference-{policy}", span_collapse_policy=policy, **kw)
        with pipeline.native_spancheck(
            [(snap.rows["train"], policy)], config=snap.config, encode=_chunks,
            token_offsets=_chunk_offsets, decode=None,
        ) as table:
            _write(snap, "train", tmp_path / f"native-{policy}", span_collapse_policy=policy, **kw)
        assert _artifacts(tmp_path / f"native-{policy}") == _artifacts(
            tmp_path / f"reference-{policy}"
        )
        assert table.native_answers > 0, ("some span slot was answered from the reply", policy)
        assert any(why.startswith("refused:") for why in table.to_reference), (
            policy, table.to_reference
        )
        answered[policy] = table.native_answers
    # Every row refuse-any answers, refuse-gold answers too; a row only refuse-gold answers is
    # one whose collapsed lines are off the gold, so its shared candidates came from the reply.
    assert answered[GOLD] > answered[ANY], answered
    # Not vacuous: lines collapse under this tokenizer, refused whole under refuse-any and on
    # the gold under refuse-gold.
    refused_any = _slot_refusals(tmp_path / f"reference-{ANY}")
    refused_gold = _slot_refusals(tmp_path / f"reference-{GOLD}")
    assert any("share one candidate" in d for d in refused_any), refused_any
    assert any("gold's line start shares token" in d for d in refused_gold), refused_gold


def _swapped(ids: Sequence[int]) -> str:
    return byte_decode(ids).swapcase()


def _cleaning(ids: Sequence[int]) -> str:
    text = byte_decode(ids)
    return text if len(list(ids)) == 1 else text.replace("\n", " ")


def _trailing(ids: Sequence[int]) -> str:
    text = byte_decode(ids)
    return text if len(list(ids)) == 1 else text + " tail"


@pytest.mark.parametrize(
    ("decode", "said"),
    [
        (_swapped, "decodes to"),
        (_cleaning, "not line starts of the text these ids decode to"),
        (_trailing, "do not decode to the text the spans were measured against"),
    ],
    ids=["piece", "line-grid", "round-trip"],
)
def test_a_dishonest_decode_raises_the_references_exception_text_and_all(
    qd_prep: Path, tmp_path: Path, decode: Callable[[Sequence[int]], str], said: str
) -> None:
    """``test_shards``' three statements of the decode check, each through the table: the
    reply cannot settle them, so the reference runs and raises its own words."""
    snap = _snapshot(tmp_path)
    with pytest.raises(ShardContractViolation) as want:
        _write(snap, "train", tmp_path / "reference", decode=decode)
    assert said in str(want.value)
    with (
        pytest.raises(ShardContractViolation) as got,
        pipeline.native_spancheck(
            [(snap.rows["train"], ANY)], config=snap.config, encode=byte_encode,
            token_offsets=byte_offsets, decode=decode,
        ),
    ):
        _write(snap, "train", tmp_path / "native", decode=decode)
    assert str(got.value) == str(want.value)
    assert shards_module._span_token_positions is REFERENCE


# -- the installer's refusals ------------------------------------------------------------------


def _block(snap: Any, *, policy: str = ANY, **kw: Any) -> Any:
    return pipeline.native_spancheck(
        [(snap.rows["train"], policy)], config=snap.config, encode=byte_encode,
        token_offsets=byte_offsets, decode=kw.pop("decode", None), **kw,
    )


def test_a_lookup_the_table_does_not_hold_is_refused(qd_prep: Path, tmp_path: Path) -> None:
    snap = _snapshot(tmp_path)
    with pytest.raises(SystemExit, match="is not among the"), _block(snap, policy=ANY):
        _write(snap, "train", tmp_path / "other-policy", span_collapse_policy=GOLD)
    assert shards_module._span_token_positions is REFERENCE


def test_a_table_never_read_is_refused(qd_prep: Path, tmp_path: Path) -> None:
    snap = _snapshot(tmp_path)
    with pytest.raises(SystemExit, match="never read"), _block(snap):
        pass


def test_another_offsets_callable_is_refused(qd_prep: Path, tmp_path: Path) -> None:
    snap = _snapshot(tmp_path)
    with pytest.raises(SystemExit, match="spancheck table was built from"), _block(snap):
        _write(snap, "train", tmp_path / "other", token_offsets=lambda t: byte_offsets(t))


def _a_span_spec(snap: Any) -> SequenceSpec:
    for row in snap.rows["train"]:
        for spec in training_texts(row, seed=snap.config.seed):
            if spec.line_char_starts:
                return spec
    raise AssertionError("the train split has no span slot; the test would be vacuous")


def test_other_ids_for_a_held_text_are_refused(qd_prep: Path, tmp_path: Path) -> None:
    snap = _snapshot(tmp_path)
    spec = _a_span_spec(snap)
    ids = np.asarray(byte_tokenize(spec.text))
    other = ids.copy()
    other[0] ^= 1
    with _block(snap):
        served = shards_module._span_token_positions(
            spec, ids, token_offsets=byte_offsets, where="w"
        )
        assert served == REFERENCE(spec, ids, token_offsets=byte_offsets, where="w")
        with pytest.raises(SystemExit, match="ids at write time"):
            shards_module._span_token_positions(
                spec, ids[:-1], token_offsets=byte_offsets, where="w"
            )
        with pytest.raises(SystemExit, match="other ids"):
            shards_module._span_token_positions(spec, other, token_offsets=byte_offsets, where="w")


@pytest.mark.parametrize("drift", ["moves a candidate", "refuses an answered sequence"])
def test_a_reply_that_drifts_is_refused_by_the_canaries(
    qd_prep: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    """The first sequence of every call is a canary; here its reply is altered after the binary
    answered, as a binary whose arithmetic drifted would answer."""
    real = pipeline._prep_spancheck

    def drifted(binary: Path, items: Sequence[pipeline.SpancheckItem]) -> list[pipeline.SpanReply]:
        replies = real(binary, items)
        first = replies[0]
        assert first.status_name == "ok", "the byte tokenizer answers every sequence"
        if drift == "moves a candidate":
            moved = first.candidates.astype(np.uint32)
            moved[-1] += 1
            replies[0] = dataclasses.replace(first, candidates=moved)
        else:
            empty = np.zeros(0, dtype="<u4")
            replies[0] = dataclasses.replace(
                first, status=3, candidates=empty, runs=empty.reshape(0, 3)
            )
        return replies

    monkeypatch.setattr(pipeline, "_prep_spancheck", drifted)
    snap = _snapshot(tmp_path)
    with pytest.raises(SystemExit, match="its projection is not the reference's"), _block(snap):
        raise AssertionError("the canaries let a drifted reply through")
    assert shards_module._span_token_positions is REFERENCE


def _abstaining_item() -> pipeline.SpancheckItem:
    return pipeline.spancheck_item(
        _spec("ab\ncd", (0, 3), None), policy=ANY, n_ids=5, offsets=_PER_CHAR
    )


def _reply(
    *, status: int = 0, flags: int = 1, detail: int = 0, start: int = 0, end: int = 0,
    cands: Sequence[int] = (0, 3), runs: Sequence[tuple[int, int, int]] = ((1, 0, 1), (4, 3, 4)),
    reserved: int = 0, n: int = 1, magic: bytes = b"QDPSCOK1", n_cands: int | None = None,
) -> bytes:
    head = struct.pack(
        "<BBHIIIII", status, flags, reserved, detail, start, end,
        len(cands) if n_cands is None else n_cands, len(runs),
    )
    body = b"".join(struct.pack("<I", c) for c in cands)
    body += b"".join(struct.pack("<III", *r) for r in runs)
    return magic + struct.pack("<Q", n) + head + body


def test_the_well_formed_reply_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control for the next test: the hand-built reply is accepted as it stands."""
    monkeypatch.setattr(pipeline, "_run_prep", lambda *a, **k: _reply())
    (reply,) = pipeline._prep_spancheck(NOT_RUN, [_abstaining_item()])
    assert pipeline._reply_value(reply, abstains=True) == ((SPAN_ABSTAIN, SPAN_ABSTAIN), (0, 3))


@pytest.mark.parametrize(
    ("reply", "match"),
    [
        (_reply(magic=b"QDPMHOK1"), "not a b'QDPSCOK1' file"),
        (_reply(n=2), "2 sequences for the 1 asked"),
        (b"QDPSCOK1" + struct.pack("<Q", 1) + bytes(10), "cannot hold 1 sequence headers"),
        (_reply(n_cands=3), "where its headers describe"),
        (_reply(status=99), "status 99"),
        (_reply(reserved=1), "reserved 1"),
        (_reply(flags=0x11), "flags 0x11"),
        (_reply(flags=0), "do not echo"),
        (_reply(status=3, cands=(), runs=((1, 0, 1),)), "carries an answer"),
        (_reply(cands=(0,), runs=((1, 0, 1),)), "1 candidates and 1 runs for 2 line starts"),
        (_reply(cands=(0, 5)), "a candidate past the 5 ids"),
        (_reply(start=1), r"gold \(1, 0\) over 5 ids"),
        (_reply(runs=((0, 0, 1), (4, 3, 4))), "a run that does not end past its position"),
        (_reply(runs=((1, 0, 1), (6, 3, 4))), "a run that does not end past its position"),
        (_reply(flags=1 | 8, runs=((1, 0, 1), (4, 3, 4))), "0 checked positions"),
    ],
)
def test_a_reply_of_the_wrong_shape_is_refused(
    monkeypatch: pytest.MonkeyPatch, reply: bytes, match: str
) -> None:
    monkeypatch.setattr(pipeline, "_run_prep", lambda *a, **k: reply)
    with pytest.raises(SystemExit, match=match):
        pipeline._prep_spancheck(NOT_RUN, [_abstaining_item()])


def test_a_request_past_the_binarys_bound_is_refused_before_it_is_read(
    qd_prep_bin: Path, tmp_path: Path
) -> None:
    """Sparse, so the file costs no disk; refused on its size, and no reply is written."""
    bound = 1 << 30
    request, reply = tmp_path / "request.bin", tmp_path / "reply.bin"
    with request.open("wb") as fh:
        fh.truncate(bound + 1)
    done = subprocess.run(
        [str(qd_prep_bin), "spancheck", "--input", str(request), "--output", str(reply)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert done.returncode != 0 and not reply.exists(), done.stderr
    assert f"{bound + 1} bytes; the bound is {bound}" in done.stderr


def test_a_sequence_past_the_request_bound_is_refused_and_small_requests_change_nothing(
    qd_prep: Path, tmp_path: Path
) -> None:
    snap = _snapshot(tmp_path)
    groups = [(snap.rows["train"], ANY)]
    kw: dict[str, Any] = {
        "config": snap.config, "encode": byte_encode, "token_offsets": byte_offsets,
        "decode": byte_decode,
    }
    with pytest.raises(SystemExit, match="request bound"):
        pipeline.build_native_spancheck(groups, request_bytes=64, **kw)
    one = pipeline.build_native_spancheck(groups, **kw)
    largest = max(
        pipeline.spancheck_item(
            spec, policy=ANY, n_ids=len(byte_tokenize(spec.text)),
            offsets=byte_offsets(spec.text),
        ).wire_bytes
        for row in snap.rows["train"]
        for spec in training_texts(row, seed=snap.config.seed)
        if spec.line_char_starts
    )
    many = pipeline.build_native_spancheck(groups, request_bytes=16 + largest, **kw)
    assert one.calls == 1 and many.calls > 1, (one.calls, many.calls)

    def answers(table: pipeline.NativeSpancheck) -> dict[bytes, tuple[Any, ...]]:
        return {
            key: (e.reply.status, e.reply.flags, e.reply.detail, e.reply.start, e.reply.end,
                  e.reply.candidates.tolist(), e.reply.runs.tolist(), e.ids_digest)
            for key, e in table.table.items()
        }

    assert answers(many) == answers(one)


@pytest.mark.parametrize(
    ("text", "lines", "offsets", "policy", "match"),
    [
        ("a\ud800b\nc", (0, 4), [(i, i + 1) for i in range(5)], ANY, "not UTF-8"),
        ("ab\ncd", (0, 3), [(-1, 1), (1, 5)], ANY, "an offset lies outside"),
        ("ab\ncd", (0, 3), [(0.0, 2.0), (2.0, 5.0)], ANY, "not integer"),
        ("ab\ncd", (0, 3), [(0, 1, 2)], ANY, "not integer"),
        ("ab\ncd", (), _PER_CHAR, ANY, "no line starts"),
        ("ab\ncd", (0, 3), _PER_CHAR, "share-everything", "is not one of"),
    ],
)
def test_a_sequence_the_wire_cannot_carry_is_refused_loudly(
    text: str, lines: tuple[int, ...], offsets: Any, policy: str, match: str
) -> None:
    with pytest.raises(SystemExit, match=match):
        pipeline.spancheck_item(_spec(text, lines, None), policy=policy, n_ids=5, offsets=offsets)


def test_a_second_install_and_a_missing_binary_are_refused(
    qd_prep: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap = _snapshot(tmp_path)
    table = pipeline.build_native_spancheck(
        [(snap.rows["train"], ANY)], config=snap.config, encode=byte_encode,
        token_offsets=byte_offsets, decode=None,
    )
    with table.installed():
        with (
            pytest.raises(SystemExit, match="is not the reference this tool imported"),
            table.installed(),
        ):
            raise AssertionError("installed twice")
        assert shards_module._span_token_positions == table.lookup, "the outer install stands"
    assert shards_module._span_token_positions is REFERENCE
    with pytest.raises(SystemExit, match="never read"):
        table.finish()
    monkeypatch.delenv(pipeline.PREP_BIN_ENV)
    with pytest.raises(SystemExit, match="is unset"):
        pipeline.build_native_spancheck(
            [(snap.rows["train"], ANY)], config=snap.config, encode=byte_encode,
            token_offsets=byte_offsets, decode=None,
        )


def test_the_flag_is_off_unless_asked_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """v5 runs without the flag, so without it nothing about the build may change."""
    assert inspect.signature(pipeline.run).parameters["spancheck_in_qd_prep"].default is False

    class Reached(Exception):
        pass

    seen: list[bool] = []

    def run(**kw: Any) -> None:
        seen.append(kw["spancheck_in_qd_prep"])
        raise Reached

    monkeypatch.setattr(pipeline, "run", run)
    for argv in (["--out", str(tmp_path)], ["--out", str(tmp_path), "--native-spancheck"]):
        with pytest.raises(Reached):
            pipeline.main(argv)
    assert seen == [False, True]


def _measured(measured: pipeline.Measured, out: Path) -> dict[str, Any]:
    """What ``run`` returned, comparably: every TriState by its JSON, the run's own directory
    spelled ``<out>`` (the two builds write to two directories and nothing else differs)."""
    text = json.dumps({
        "metrics": {k: v.to_json() for k, v in measured.metrics.items()},
        "gates": {k: v.to_json() for k, v in measured.gates.items()},
        "data_snapshot_hash": measured.data_snapshot_hash,
        "tokenizer_hash": measured.tokenizer_hash,
        "notes": measured.notes,
        "quick_reason": measured.quick_reason,
    }, sort_keys=True)
    parsed: dict[str, Any] = json.loads(text.replace(str(out), "<out>"))
    return parsed


@pytest.mark.parametrize("blank_line_runs", [False, True], ids=["single-blank", "blank-runs"])
@pytest.mark.parametrize("policy", [ANY, GOLD])
def test_the_pipeline_writes_every_shard_set_the_reference_writes(
    qd_prep: Path, tmp_path: Path, policy: str, blank_line_runs: bool
) -> None:
    """The bar for ``--native-spancheck``, end to end: ``run`` with and without it, on the
    real tokenizer, with the val set. Every shard set it writes -- train, val, and the train
    set written without decode -- is the reference's in every file, and so is everything
    ``run`` measured but the one metric the flag adds. Needs ``transformers``, the model's
    tokenizer in the HF cache and the commitpackft download under :data:`DATA_ROOT`; SKIPPED
    without any of them.

    Blank-line runs are passages joined by two blank lines, which Qwen3.5's BPE merges into one
    token holding several line starts. Every passage then collapses: refuse-any refuses each
    such slot whole (at 2026-10-04T00:24Z, 88 of 88 lookups, each through the reference).
    Under refuse-gold the train slots were answered from the reply (84 of 88), and the 4
    refused were the val set's, which ``run`` writes under refuse-any whatever the train
    policy. So refuse-any with blank-line runs is the case that carries the refusal path end
    to end, and each of the other three must answer some slot from the reply."""
    pytest.importorskip("transformers")
    if not pipeline.MODEL_REF.exists():
        pytest.skip(f"{pipeline.MODEL} is not in this host's HF cache")
    download = DATA_ROOT / "data" / "pool" / "commitpackft"
    if not any((download / f"{lang}.jsonl").exists() for lang in ("go", "python")):
        pytest.skip(f"the commitpackft download is not under {download} (set QD_PREP_DATA_ROOT)")

    built: dict[str, tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}
    for arm, flag in (("reference", False), ("native", True)):
        out = tmp_path / arm
        measured = pipeline.run(
            out=out, max_pairs=60, blank_line_runs=blank_line_runs, rev="HEAD",
            commitpackft=download, val_shards=True, span_collapse_policy=policy,
            spancheck_in_qd_prep=flag,
        )
        sets = {d.name: _artifacts(d) for d in sorted((out / "shards").iterdir())}
        built[arm] = (_measured(measured, out), sets)
    (ref, ref_sets), (nat, nat_sets) = built["reference"], built["native"]

    assert sorted(ref_sets) == ["train", "train-no-decode", "val"], sorted(ref_sets)
    assert sorted(nat_sets) == sorted(ref_sets)
    for name in ref_sets:
        assert sorted(nat_sets[name]) == sorted(ref_sets[name]), name
        for file in ref_sets[name]:
            assert nat_sets[name][file] == ref_sets[name][file], (name, file)

    assert "native_spancheck" not in ref["metrics"], "without the flag, no metric"
    native = nat["metrics"].pop("native_spancheck")
    assert nat == ref
    assert native["state"] == "ran" and native["passed"] is True, native
    assert native["value"] == hashlib.sha256(qd_prep.read_bytes()).hexdigest()
    detail = json.loads(native["detail"])
    # Every call the writes made is counted: looked up (answered from the reply or handed to
    # the reference, by reason) or passed straight to the reference.
    assert native["n"] == detail["native_answers"], detail
    assert native["n_total"] == detail["reads"] + detail["passthrough"], detail
    assert detail["reads"] == detail["native_answers"] + sum(detail["to_reference"].values())
    # Not vacuous: span slots were looked up, and answered from the reply -- or, where the
    # docstring says every slot collapses under refuse-any, refused through the reference.
    assert detail["reads"] > 0, detail
    if policy == ANY and blank_line_runs:
        assert detail["to_reference"].get("refused:lines_collapse", 0) > 0, detail
    else:
        assert detail["native_answers"] > 0, detail


# -- the real tokenizer ------------------------------------------------------------------------

QWEN_TOKENIZER = (
    "/Users/bharath/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/"
    "b1485b2fa6dfa1287294f269f5fb618e03d52d7c/tokenizer.json"
)
#: Texts whose real BPE offsets have shapes the byte stand-in cannot: a character split into
#: pieces at a line start and merged into a preceding space, CRLF and lone CR, NFD, the U+09DF
#: line the 2026-10-01 build refused, blank lines a merge can collapse, an astral character.
QWEN_TEXTS = (
    "x\n\N{LATIN CAPITAL LETTER S WITH DOT BELOW}ab\nc",
    "is \N{LATIN CAPITAL LETTER S WITH DOT BELOW}al\nnext line",
    "a\r\nb\r\nc",
    "a\rb\nc",
    "e\N{COMBINING ACUTE ACCENT}\nx\n",
    # test_span_decode_multibyte's line: U+09DF, which Qwen3.5's NFC normalizer decomposes.
    "@@ -1,1 +1,4 @@\n+t = '\N{BENGALI LETTER PA}\N{BENGALI VOWEL SIGN I}"
    "\N{BENGALI LETTER RA}\N{BENGALI VOWEL SIGN I}\N{BENGALI LETTER YYA}"
    "\N{BENGALI LETTER DDA}'\n+x = 1\n+y = 2",
    "\U0001f9ea\n\n  z\n",
    "def f():\n\n\n    return 1\n\n\n",
)


def _qwen() -> tuple[Callable[[str], tuple[list[int], list[tuple[int, int]]]],
                     Callable[[Sequence[int]], str]]:
    tokenizers = pytest.importorskip("tokenizers")
    if not Path(QWEN_TOKENIZER).is_file():
        pytest.skip("the Qwen3.5 tokenizer.json is not in this machine's HF cache")
    tok = tokenizers.Tokenizer.from_file(QWEN_TOKENIZER)

    def encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = tok.encode(text, add_special_tokens=False)
        return list(enc.ids), [tuple(o) for o in enc.offsets]

    def decode(ids: Sequence[int]) -> str:
        return str(tok.decode([int(i) for i in ids], skip_special_tokens=False))

    return encode, decode


def test_real_bpe_offsets_are_answered_as_the_reference_answers_them(qd_prep_bin: Path) -> None:
    encode, decode = _qwen()
    items, offsets_of, ids_of = [], [], []
    for text in QWEN_TEXTS:
        lines = tuple(line_start_indices(text))
        for gold in (None, (lines[0], lines[-1]), (lines[-1], lines[-1])):
            for policy in (ANY, GOLD):
                ids, offsets = encode(text)
                items.append(pipeline.spancheck_item(
                    _spec(text, lines, gold), policy=policy, n_ids=len(ids), offsets=offsets
                ))
                offsets_of.append(offsets)
                ids_of.append(np.asarray(ids))
    replies = pipeline._prep_spancheck(qd_prep_bin, items)
    for item, reply, offsets, ids in zip(items, replies, offsets_of, ids_of, strict=True):
        _assert_agrees(item, reply, offsets, ids.tolist())
        # And with the real decode: the native answer, decode-checked here, is the reference's.
        pipeline._canary(
            qd_prep_bin, item, ids, reply, token_offsets=lambda t: encode(t)[1], decode=decode
        )
    assert any(r.status_name == "ok" for r in replies)


# -- the benchmark -----------------------------------------------------------------------------

#: A/B rounds; each runs both arms once, alternating which goes first.
BENCH_ROUNDS = 5
#: Defect rows in the fixed slice: a sha256-ordered sample (``load_defect_rows(max_rows=)``).
BENCH_ROWS = 600


def _bench_tokenizer(name: str) -> tuple[Any, Any, Any, Any]:
    """``(tokenize, token_offsets, encode, decode)`` for one benchmark arm's tokenizer."""
    if name == "bytes":
        return byte_tokenize, byte_offsets, byte_encode, byte_decode
    encode, decode = _qwen()

    def tokenize(text: str) -> list[int]:
        return encode(text)[0]

    def token_offsets(text: str) -> list[tuple[int, int]]:
        return encode(text)[1]

    return tokenize, token_offsets, encode, decode


@pytest.mark.skipif(
    os.environ.get("QD_PREP_BENCH") != "1",
    reason="the reference-vs-qd-prep spancheck benchmark runs only with QD_PREP_BENCH=1",
)
@pytest.mark.parametrize("tokenizer", ["bytes", "qwen"])
def test_benchmark_span_projection_reference_against_qd_prep_interleaved_min_of_n(
    qd_prep: Path, tokenizer: str
) -> None:
    """The committed A/B behind ``--native-spancheck``: ``write_shards``' per-slot span work
    (render, ``_tokenize_checked``, ``_span_token_positions`` with the decode check) over a
    fixed slice of the real defect corpus, by the reference and through the installed table,
    in alternating rounds, min of :data:`BENCH_ROUNDS`. The native arm's time is everything
    the port does: the pre-pass (render and encode once more per span slot), the request,
    the process, the reply's checks, the canaries and every lookup. Every round's answers --
    values, and refusals by class and text -- are equal. Nothing here asserts a speed-up; the
    numbers are the finding.

        QD_PREP_BENCH=1 QD_PREP_DATA_ROOT=<checkout with data/pool> \\
            pytest -s python/tests/test_qd_prep_spancheck_parity.py -k benchmark
    """
    from qd_data.defect_class import DEFECT_SOURCE_ID, load_defect_rows
    from qd_data.mixture import build_mixture

    root = DATA_ROOT
    corpus = root / "data" / "pool" / "commitpackft-corpus-v2"
    if not (corpus / "examples.jsonl").is_file():
        pytest.skip(f"{corpus}/examples.jsonl is not on disk (set QD_PREP_DATA_ROOT)")
    tokenize, token_offsets, encode, decode = _bench_tokenizer(tokenizer)
    load = load_defect_rows(
        corpus, download_root=root / "data" / "pool" / "commitpackft", config=CONFIG,
        repo_root=root, max_rows=BENCH_ROWS,
    )
    rows = sorted(
        build_mixture({DEFECT_SOURCE_ID: list(load.rows)}, config=CONFIG).rows,
        key=lambda r: r.row_id,
    )

    def project(fn: Callable[..., Any]) -> list[tuple[Any, ...]]:
        out: list[tuple[Any, ...]] = []
        for row in rows:
            try:
                specs = training_texts(row, seed=CONFIG.seed)
            except (UnencodableGold, QdRefusal):
                continue
            for spec in specs:
                if not spec.line_char_starts:
                    continue
                ids = shards_module._tokenize_checked(tokenize, spec.text, where="bench")
                out.append(pipeline._outcome(functools.partial(
                    fn, spec, ids, token_offsets=token_offsets, decode=decode, where="bench",
                    span_collapse_policy=ANY,
                )))
        return out

    times: dict[str, list[float]] = {"reference": [], "qd_prep": []}
    answers: dict[str, list[tuple[Any, ...]]] = {}
    stats: dict[str, Any] = {}
    for r in range(BENCH_ROUNDS):
        for arm in ("reference", "qd_prep") if r % 2 == 0 else ("qd_prep", "reference"):
            started = time.perf_counter()
            if arm == "reference":
                answers[arm] = project(REFERENCE)
            else:
                with pipeline.native_spancheck(
                    [(rows, ANY)], config=CONFIG, encode=encode, token_offsets=token_offsets,
                    decode=decode,
                ) as table:
                    answers[arm] = project(shards_module._span_token_positions)
                stats = {
                    "calls": table.calls, "native_answers": table.native_answers,
                    "reads": table.reads, "to_reference": dict(table.to_reference),
                    "prepass_s": round(table.prepass_s, 3),
                }
            times[arm].append(time.perf_counter() - started)
        assert answers["qd_prep"] == answers["reference"]
    assert answers["reference"], "the slice must hold span slots or the comparison is vacuous"
    print(json.dumps({
        "benchmark": "span_projection", "tokenizer": tokenizer, "rows": len(rows),
        "span_slots": len(answers["reference"]), "rounds": BENCH_ROUNDS, "native": stats,
        "reference_s": [round(x, 3) for x in times["reference"]],
        "qd_prep_s": [round(x, 3) for x in times["qd_prep"]],
        "reference_min_s": round(min(times["reference"]), 3),
        "qd_prep_min_s": round(min(times["qd_prep"]), 3),
        "speedup_min_over_min": round(min(times["reference"]) / min(times["qd_prep"]), 2),
    }, sort_keys=True))
