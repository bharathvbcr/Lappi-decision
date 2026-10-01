"""``tools/real_tokenizer_pipeline.py`` with ``--defect-class``: the parts that run without a model.

Three things, each of which the first real build over the 50,177-row corpus hit:

* ``shard_sequences`` compared sequences to *rows*. Every family had one slot, so the two
  numbers coincided; the first two-slot family wrote 78,643 sequences from 46,167 rows and
  ``Ran`` refused ``n > n_total`` after the whole shard set had been written.
* A census refusal drops a whole row, so the class balance that reaches the shards is not the
  split's; ``defect_balance`` reports both, with the refusals broken down by class.
* The tokenizer memo's bound is fixed at 8,192 and the corpus renders ~100k sequences, so the
  bound is now a flag, and ``0`` encodes every call instead of evicting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_tokenizer_pipeline as pipeline  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.defect_class import DEFECT_FAMILY_ID, DefectRow  # noqa: E402
from qd_data.mixture import rewrite_defect_class  # noqa: E402
from qd_data.rows import DataRow  # noqa: E402

DIFF = "@@ -1,2 +1,2 @@\n def f(x):\n-    return x\n+    return 0\n"


def _row(n: int, cls: str, *, repo: str = "o/r") -> DataRow:
    raw = DefectRow(
        example_id=f"c{n}:x.py#0", pool_id=f"c{n}:x.py", repo=repo, path="x.py",
        symbol="f", arity=1, language="python", mutation_class=cls, operator=f"{cls}.op",
        diff=DIFF.replace("0", str(n)), diff_span=None if cls == "clean" else (3, 3),
        span_refusal=None, licence="mit",
    )
    return rewrite_defect_class(raw, family_id=DEFECT_FAMILY_ID, index=n, config=DataConfig())


def test_sequences_are_counted_against_slots_not_rows() -> None:
    rows = [_row(i, "stub") for i in range(3)]
    # Each defect row renders two sequences. The old denominator (rows) made this 6 of 3.
    metric = pipeline.shard_sequences_metric(n_sequences=6, rows=rows, total_tokens=100)
    assert (metric.n, metric.n_total) == (6, 6)
    partial = pipeline.shard_sequences_metric(n_sequences=4, rows=rows, total_tokens=100)
    assert (partial.n, partial.n_total) == (4, 6)


def test_the_balance_that_reaches_the_shards_excludes_refused_rows_by_class() -> None:
    rows = [_row(0, "stub"), _row(1, "stub"), _row(2, "clean"), _row(3, "logic")]
    before = pipeline.defect_balance(rows)
    assert before["classes"] == {"clean": 1, "logic": 1, "stub": 2}
    assert before["majority"] == {"class": "stub", "n": 2, "rate": 0.5}
    after = pipeline.defect_balance(
        rows, refused={rows[0].row_id: "span:line_starts_collapse_under_bpe"}
    )
    assert after["rows"] == 3 and after["of"] == 4
    assert after["classes"] == {"clean": 1, "logic": 1, "stub": 1}
    assert after["refused_by_class_and_reason"] == {
        "stub:span:line_starts_collapse_under_bpe": 1
    }


def test_a_span_slot_refusal_keeps_the_class_in_the_balance_and_is_reported_beside_it() -> None:
    """GAP-A3-BPE-SPAN-COLLAPSE-DROPS-DEFECT-CLASS-LABELS: the writer now refuses only the
    span slot, so the class label still reaches the shards and the majority rate the choice
    head is judged against is the split's. Only a refused CHOICE slot removes a class."""
    rows = [_row(0, "stub"), _row(1, "stub"), _row(2, "clean"), _row(3, "logic")]
    collapse = "span:line_starts_collapse_under_bpe"
    after = pipeline.defect_balance(
        rows,
        refused={},
        refused_slots={
            (rows[0].row_id, "defect_span"): collapse,
            (rows[1].row_id, "defect_span"): collapse,
            (rows[3].row_id, "defect_class"): "gold:not_a_rendered_option",
        },
    )
    assert after["rows"] == 3 and after["classes"] == {"clean": 1, "stub": 2}
    assert after["majority"] == {"class": "stub", "n": 2, "rate": round(2 / 3, 4)}
    assert after["slot_refused_by_class_and_reason"] == {
        "logic:defect_class:gold:not_a_rendered_option": 1,
        f"stub:defect_span:{collapse}": 2,
    }


class _FakeTok:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str, **_: Any) -> dict[str, list[Any]]:
        self.calls += 1
        return {"input_ids": [ord(c) for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_memo_limit_zero_encodes_every_call_and_stores_nothing() -> None:
    fake = _FakeTok()
    tok = pipeline.RealTokenizer(tok=fake, _memo={}, memo_limit=0)
    assert tok.tokenize("ab") == [97, 98]
    assert tok.offsets("ab") == [(0, 1), (1, 2)]
    assert fake.calls == 2 and tok._memo == {}


def test_a_positive_memo_limit_still_refuses_rather_than_evicting() -> None:
    tok = pipeline.RealTokenizer(tok=_FakeTok(), _memo={}, memo_limit=1)
    tok.tokenize("a")
    tok.tokenize("a")  # a hit, not a second entry
    with pytest.raises(RuntimeError, match="1-entry bound"):
        tok.tokenize("b")


class _CharTok(_FakeTok):
    """One token per character, and a decode that inverts it."""

    def decode(self, ids: list[int], **_: Any) -> str:
        return "".join(chr(i) for i in ids)


def test_the_census_refuses_a_row_over_max_seq_len_whole_as_the_writer_does() -> None:
    """``--max-seq-len`` is mirrored in the census, so the row's count matches the shards'.

    A long row is refused whole (both of its slots), named by id, and never truncated; the
    rows under the width are unaffected.
    """
    rows = [_row(i, "stub") for i in range(4)]
    raw_long = DefectRow(
        example_id="long:x.py#0", pool_id="long:x.py", repo="o/r", path="x.py", symbol="f",
        arity=1, language="python", mutation_class="stub", operator="stub.op",
        diff="@@ -1,2 +1,2 @@\n def f(x):\n-    return x\n+    return " + "9" * 400 + "\n",
        diff_span=(3, 3), span_refusal=None, licence="mit",
    )
    long_row = rewrite_defect_class(raw_long, family_id=DEFECT_FAMILY_ID, index=9,
                                    config=DataConfig())
    rows.append(long_row)
    tok = pipeline.RealTokenizer(tok=_CharTok(), _memo={}, memo_limit=0)
    free = pipeline.census(rows, tok=tok, config=DataConfig())
    assert free.rows_out == 5 and not free.refused
    short_max = max(free.lengths[:8])
    cap = short_max + 10
    assert max(free.lengths) > cap, "the long row must be over the cap the short ones fit"
    capped = pipeline.census(rows, tok=tok, config=DataConfig(), max_seq_len=cap)
    assert capped.refused == {pipeline.OVER_MAX_SEQ_LEN: 1}
    assert capped.refused_rows == {long_row.row_id: pipeline.OVER_MAX_SEQ_LEN}
    assert capped.rows_out == 4 and capped.sequences_out == 8
    assert max(capped.lengths) <= cap
    # Its ids were still recorded: the writer encodes before it can measure, so the remap
    # must cover them.
    assert long_row.row_id in capped.id_rows
