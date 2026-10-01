"""``qd_train.composed_slice``: the report-only composed slice's cases, hunks, hit rule and tables.

The fixture rows (``data/composed_slice_rows.jsonl``) are three real rows of the compose lane's
v4 train corpus (``data/pool/commitpackft-composed-v1``, one clean, a stub and a cosmetic),
renamed into the slice's id space -- one as a diag row -- with ``after`` dropped. The slice's
own rows have the same shape (compose lane, 2026-10-01).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_train import composed_slice as cs  # noqa: E402
from qd_train.needle import depth_bucket_label  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "composed_slice_rows.jsonl"


def _rows() -> list[dict]:
    return [json.loads(x) for x in FIXTURE.read_text(encoding="utf-8").splitlines()]


def _lines(row: dict) -> list[str]:
    diff = row["diff"]
    return (diff[:-1] if diff.endswith("\n") else diff).split("\n")


# --- the cases --------------------------------------------------------------------------------


def test_the_fixture_rows_parse_and_their_hunks_are_the_diffs_own():
    cases = cs.load_cases([FIXTURE])
    assert sorted(cases) == sorted(cs.SHARD_ROW_PREFIX + r["id"] for r in _rows())
    for row in _rows():
        case = cases[cs.SHARD_ROW_PREFIX + row["id"]]
        lines = _lines(row)
        assert case.rendered_lines == 2 + len(lines)
        headers = {b["first_line"] + j for b in row["constituents"] for j in range(3)}
        # An independent count: hunk headers seen so far, outside file headers.
        seen = -1
        for i, text in enumerate(lines, 1):
            if i in headers:
                assert case.hunk_of_context_line(i + 1) is None, (row["id"], i)
                continue
            seen += text.startswith("@@ ")
            assert case.hunk_of_context_line(i + 1) == seen, (row["id"], i)
        assert case.hunk_of_context_line(0) is None and case.hunk_of_context_line(1) is None
        assert case.files_bin == cs._bin(row["n_files"], cs.FILES_BINS, cs.FILES_OVER)
        if row["class"] == "clean":
            assert case.is_clean and case.gold_hunk is None and case.depth_bucket == "clean"
            continue
        start = row["diff_span"]["start_line"]
        # The gold hunk is the last "@@ " at or before the span's first line (compose lane).
        last_at = max(i for i, t in enumerate(lines[:start], 1) if t.startswith("@@ "))
        assert case.gold_hunk == case.hunk_of_context_line(last_at + 1)
        assert case.gold_line == start + 1, "rendered = diff + 2 lines, 0-based"
        assert case.depth_bucket == depth_bucket_label(row["needle_index"] / (row["n_files"] - 1))
    diag = [c for c in cases.values() if c.diag_half is not None]
    assert [c.slice_set for c in diag] == ["diag.seen_filler"]
    assert diag[0].needle_train_appearances == 2


def _parse(row: dict):
    return cs.parse_case(row, where="test")


def _needle_first(row: dict) -> int:
    """The needle block's first line: its `diff --git` header, not its body."""
    return row["constituents"][row["needle_index"]]["first_line"]


def _inside(case) -> list[int]:
    """Every rendered line in the gold hunk."""
    return [c for c in range(case.rendered_lines) if case.hunk_of_context_line(c) == case.gold_hunk]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r.update(id="compose:train:000001"), "not a compose:val/diag row"),
        (lambda r: r.update(composed=False), "composed is False"),
        (lambda r: r.update(**{"class": "typo"}), "class 'typo'"),
        (lambda r: r.update(n_files=r["n_files"] + 1), "constituents is not a list"),
        (lambda r: r["constituents"][1].update(first_line=r["constituents"][1]["first_line"] + 1),
         "must tile the diff"),
        (lambda r: r["constituents"][-1].update(last_line=r["constituents"][-1]["last_line"] - 1),
         "the blocks end at line"),
        (lambda r: r.update(needle_index=(r["needle_index"] + 1) % r["n_files"]),
         "is not the one needle"),
        (lambda r: r.update(diff_span={"start": 1, "end": 1}), "is not \\{start_line, end_line\\}"),
        (lambda r: r["diff_span"].update(start_line=_needle_first(r)),
         "not inside the needle block's body"),
        (lambda r: r.update(diag_half="seen_filler"), "carries diag fields"),
    ],
)
def test_a_needle_row_that_does_not_say_what_the_tables_read_is_refused(mutate, match):
    row = copy.deepcopy(next(r for r in _rows() if r["class"] == "stub"))
    _parse(row)
    mutate(row)
    with pytest.raises(cs.SliceRefusal, match=match):
        _parse(row)


def test_a_header_that_is_not_a_file_header_and_a_hunk_is_refused():
    row = copy.deepcopy(next(r for r in _rows() if r["class"] == "stub"))
    lines = _lines(row)
    first = row["constituents"][2]["first_line"]
    lines[first + 2] = " " + lines[first + 2]  # the block's first hunk header, displaced
    row["diff"] = "\n".join(lines) + "\n"
    with pytest.raises(cs.SliceRefusal, match="not a file header"):
        _parse(row)


def test_clean_and_diag_rows_are_refused_when_they_contradict_themselves():
    clean = copy.deepcopy(next(r for r in _rows() if r["class"] == "clean"))
    _parse(clean)
    with pytest.raises(cs.SliceRefusal, match="clean row with a needle"):
        _parse({**clean, "needle_index": 0})
    diag = copy.deepcopy(next(r for r in _rows() if r["id"].startswith(cs.DIAG_PREFIX)))
    for change, match in (
        ({"diag_half": "seen"}, "diag_half 'seen'"),
        ({"needle_train_appearances": 0}, "for the seen_filler half"),
        ({"diag_half": "unseen"}, "for the unseen half"),
        ({"needle_train_appearances": None}, "needle_train_appearances is None"),
    ):
        with pytest.raises(cs.SliceRefusal, match=match):
            _parse({**diag, **change})
    assert _parse({**diag, "diag_half": "unseen", "needle_train_appearances": 0}).slice_set == (
        "diag.unseen"
    )


def test_a_file_that_repeats_a_row_or_is_not_the_slice_is_refused(tmp_path):
    rows = FIXTURE.read_text(encoding="utf-8").splitlines()
    twice = tmp_path / "twice.jsonl"
    twice.write_text("\n".join([rows[0], rows[0]]) + "\n", encoding="utf-8")
    with pytest.raises(cs.SliceRefusal, match="appears twice"):
        cs.load_cases([twice])
    with pytest.raises(cs.SliceRefusal, match="not it"):
        cs.load_cases([FIXTURE], max_rows=2)


# --- the head's rows, the alignment check and the hit rule -----------------------------------


def _stub():
    row = next(r for r in _rows() if r["class"] == "stub")
    return row, cs.load_cases([FIXTURE])[cs.SHARD_ROW_PREFIX + row["id"]]


def _candidates(case, shared: tuple[int, ...] = ()) -> list[int]:
    """One token per rendered line, ascending; each line in ``shared`` starts on the token of
    the line before it (two consecutive lines on one token, as refuse-gold collapses them)."""
    out: list[int] = []
    pos = 0
    for c in range(case.rendered_lines):
        if c in shared:
            out.append(out[-1])
            continue
        pos += 7
        out.append(pos)
    return out


def test_a_head_row_is_a_token_and_every_line_on_it():
    assert cs.lines_on_token([3, 9, 9, 15], 0) == (0,)
    assert cs.lines_on_token([3, 9, 9, 15], 1) == (1, 2)
    assert cs.lines_on_token([3, 9, 9, 15], 2) == (3,)
    assert cs.lines_on_token([3, 9, 9, 15], 3) == (), "the abstention: unique tokens, then it"
    with pytest.raises(cs.SliceRefusal, match="head row 4"):
        cs.lines_on_token([3, 9, 9, 15], 4)


def test_a_sequence_aligned_with_its_row_passes_and_one_line_off_is_refused():
    _, case = _stub()
    gold = case.gold_line
    shared = (gold + 3,)
    candidates = _candidates(case, shared)
    unique = sorted(set(candidates))
    cs.check_alignment(case, candidates, unique.index(candidates[gold]))
    with pytest.raises(cs.SliceRefusal, match="the corpus's gold start"):
        cs.check_alignment(case, candidates, unique.index(candidates[gold]) + 1)
    with pytest.raises(cs.SliceRefusal, match="is not a line number"):
        cs.check_alignment(case, candidates[:-1], unique.index(candidates[gold]))
    on_gold = _candidates(case, (gold,))  # gold shares a token: refuse-gold never writes it
    with pytest.raises(cs.SliceRefusal, match="gold token starts lines"):
        cs.check_alignment(case, on_gold, sorted(set(on_gold)).index(on_gold[gold]))


def test_a_clean_rows_gold_is_the_abstention():
    clean = next(c for c in cs.load_cases([FIXTURE]).values() if c.is_clean)
    candidates = _candidates(clean)
    cs.check_alignment(clean, candidates, len(candidates))
    with pytest.raises(cs.SliceRefusal, match="not the abstention"):
        cs.check_alignment(clean, candidates, 0)


def test_a_shared_token_is_a_hit_only_if_every_line_on_it_is_in_the_gold_hunk():
    """Condition 8."""
    _, case = _stub()
    gold = case.gold_line
    inside = _inside(case)
    outside = next(c for c in range(case.rendered_lines)
                   if case.hunk_of_context_line(c) not in (None, case.gold_hunk))
    header = next(c for c in range(2, case.rendered_lines) if case.hunk_of_context_line(c) is None)
    assert cs.needle_hunk_hit(case, (gold,))
    assert cs.needle_hunk_hit(case, tuple(inside[:2])), "two lines, both in the gold hunk"
    assert not cs.needle_hunk_hit(case, (inside[-1], outside)), "one line outside: a miss"
    assert not cs.needle_hunk_hit(case, (header,)), "a file header is in no hunk"
    assert not cs.needle_hunk_hit(case, ()), "the abstention"
    clean = next(c for c in cs.load_cases([FIXTURE]).values() if c.is_clean)
    with pytest.raises(cs.SliceRefusal, match="no needle hunk"):
        cs.needle_hunk_hit(clean, (2,))


# --- the tables -------------------------------------------------------------------------------


def _as(case, k: int):
    """``case`` under row id ``...-k``: one fixture row standing for several slice rows."""
    return dataclasses.replace(case, row_id=f"{case.row_id}-{k}")


def _verdict(case, *, tokens=3000, shared=False, span=True, lines=None, choice=True):
    return cs.SliceVerdict(
        case=case, length_tokens=tokens, shared_candidates=shared, span_correct=span,
        predicted_lines=(case.gold_line,) if lines is None and not case.is_clean else (
            lines if lines is not None else ()
        ),
        choice_correct=choice,
    )


def test_span_cells_carry_both_populations_and_choice_cells_one():
    """Condition 6, as the compose lane read Fable's ruling (2026-10-01). The two populations
    are span populations. A row whose gold line's start shares a token has no span sequence
    under either policy, so it is excluded from both and counted, never a miss. The choice
    slot is one population: a span refusal drops only the span sequence, so both policies
    write the same choice sequences, and a refuse-any choice subset is a selection no
    refuse-any build makes."""
    cases = cs.load_cases([FIXTURE])
    _, stub = _stub()
    clean = next(c for c in cases.values() if c.is_clean)
    diag = next(c for c in cases.values() if c.diag_half is not None)
    off = next(c for c in range(stub.rendered_lines)
               if stub.hunk_of_context_line(c) not in (None, stub.gold_hunk))
    verdicts = [
        _verdict(_as(stub, 1)),                                         # hit, refuse-any
        _verdict(_as(stub, 2), shared=True, span=False, lines=(off,)),  # miss, shared cands
        cs.SliceVerdict(_as(stub, 3), 3000, None, None, None, False,
                        span_excluded="gold_shares_token"),             # no span sequence
        _verdict(clean, tokens=900),                                    # clean: abstained
        _verdict(diag, tokens=5000, span=False, lines=()),              # diag: a miss
    ]
    m = cs.slice_metrics(verdicts, row_exclusions=())

    def at(name):
        return m[f"composed.val.{name}"]

    excluded = at("span_excluded.gold_shares_token.all.all")
    assert (excluded.value, excluded.n_total) == (1, 4)
    assert at("span_excluded.gold_shares_token.depth.clean").value == 0
    assert at("refuse_gold.span_sequences").value == 3
    assert at("refuse_any.span_sequences").value == 2
    hit_gold, hit_any = at("refuse_gold.all.all.hunk_hit"), at("refuse_any.all.all.hunk_hit")
    assert (hit_gold.n, hit_gold.n_total) == (1, 2), "the excluded row is not a miss"
    assert (hit_any.n, hit_any.n_total) == (1, 1)
    assert "Wilson 95%" in hit_gold.detail and "refuse_gold" in hit_gold.detail
    assert at("delta.all.all.hunk_hit").value == pytest.approx(0.5)
    choice = at("both_policies.all.all.choice_top1")
    assert (choice.n, choice.n_total) == (3, 4), "every choice sequence, whatever its span"
    assert "both policies write the same" in choice.detail
    assert not any(k.endswith(".choice_top1") and (".refuse_gold." in k or ".refuse_any." in k)
                   for k in m), "the choice slot has no span population"
    assert not any(k.startswith("composed.val.delta.") and k.endswith(".choice_top1") for k in m)
    assert at("refuse_gold.depth.clean.span_top1").value == 1.0
    assert isinstance(at("refuse_gold.depth.clean.hunk_hit"), NotRun), "clean rows have no needle"
    assert at("refuse_gold.length.le1k.span_top1").n_total == 1
    lxd = f"refuse_gold.length_x_depth.2k-4k.{stub.depth_bucket}.hunk_hit"
    assert at(lxd).n_total == 2
    shared = at("refuse_gold.all.all.shared_token_predictions")
    assert isinstance(shared, Ran) and shared.value == 0, "shared candidates, unshared prediction"
    half = m["composed.diag.seen_filler.refuse_gold.all.all.hunk_hit"]
    assert (half.n, half.n_total) == (0, 1)
    assert not any(k.startswith("composed.diag.") and ".length_x_depth." in k for k in m)
    assert all(not k.startswith("composed.val.") or "diag" not in k for k in m)


def test_a_prediction_on_a_shared_token_is_counted_and_scored_by_condition_8():
    _, stub = _stub()
    inside = _inside(stub)
    outside = next(c for c in range(stub.rendered_lines)
                   if stub.hunk_of_context_line(c) not in (None, stub.gold_hunk))
    m = cs.slice_metrics([
        _verdict(_as(stub, 1), shared=True, span=False, lines=tuple(inside[:2])),
        _verdict(_as(stub, 2), shared=True, span=False, lines=(inside[-1], outside)),
    ], row_exclusions=())
    shared = m["composed.val.refuse_gold.all.all.shared_token_predictions"]
    assert (shared.value, shared.n_total) == (2, 2)
    hit = m["composed.val.refuse_gold.all.all.hunk_hit"]
    assert (hit.n, hit.n_total) == (1, 2)
    assert isinstance(m["composed.val.refuse_any.all.all.hunk_hit"], NotRun)
    assert isinstance(m["composed.val.delta.all.all.hunk_hit"], NotRun)


# --- exclusions: counted under their own bucket, never misses ----------------------------------

#: Details as qd_train.shards writes them (compose lane's writer): "<where>: <reason>".
_COLLISION = (
    "qdm:code.defect_class:compose:val:000007 slot defect_span: the gold's line start shares "
    "token(s) [41] with line(s) [12]"
)
_NFC = (
    "qdm:code.defect_class:compose:val:000008 slot defect_span: the context is not NFC-stable "
    "and the tokenizer's NFC normalizer rewrites it"
)


def test_every_exclusion_falls_in_one_known_bucket_or_the_pass_refuses():
    """The compose lane's v4 census (2026-10-01): gold collisions, NFC-unstable contexts and
    rows over 8,192. Anything else is unknown until the slice's own list pins it."""
    assert cs.exclusion_bucket("slot", "UnencodableGold", _COLLISION) == "gold_shares_token"
    assert cs.exclusion_bucket("slot", "UnencodableGold", _NFC) == "nfc_unstable"
    assert cs.exclusion_bucket("row", "OverMaxSeqLen", "8,342 tokens > 8,192") == (
        "over_max_seq_len"
    )
    for scope, refusal, detail in (
        ("slot", "UnencodableGold", "the span is empty"),        # a refusal never measured
        ("row", "UnencodableGold", _COLLISION),                   # a known detail, wrong scope
        ("slot", "OverMaxSeqLen", "8,342 tokens > 8,192"),        # a known refusal, wrong scope
        ("slot", "TokenNotInRemap", _NFC),                        # a known detail, wrong refusal
    ):
        with pytest.raises(cs.SliceRefusal, match="known bucket"):
            cs.exclusion_bucket(scope, refusal, detail)
    assert cs.SLOT_EXCLUSIONS == ("gold_shares_token", "nfc_unstable")
    assert cs.ROW_EXCLUSIONS == ("over_max_seq_len",)


def test_a_slot_is_decoded_or_excluded_for_a_known_reason_never_neither_or_both():
    _, stub = _stub()
    for bad in (
        dict(shared_candidates=None, span_correct=None, predicted_lines=None),   # neither
        dict(span_excluded="nfc_unstable"),                                       # both
        dict(shared_candidates=None, span_correct=None, predicted_lines=None,
             span_excluded="over_max_seq_len"),                                   # a row bucket
        dict(choice_correct=None),                                                # choice: neither
    ):
        fields = dict(case=stub, length_tokens=3000, shared_candidates=False, span_correct=True,
                      predicted_lines=(stub.gold_line,), choice_correct=True)
        with pytest.raises(cs.SliceRefusal):
            cs.SliceVerdict(**{**fields, **bad})
    with pytest.raises(cs.SliceRefusal, match="row exclusion"):
        cs.SliceVerdict(stub, 3000, None, None, None, None, span_excluded="nfc_unstable",
                        choice_excluded="nfc_unstable")


def test_exclusions_are_counted_per_bucket_and_never_scored():
    """nfc_unstable is not a collision: its own bucket, in neither span population, and not
    a miss. A row over 8,192 has no sequence, so it is counted per set and cut by nothing."""
    _, stub = _stub()
    verdicts = [
        _verdict(_as(stub, 1)),
        cs.SliceVerdict(_as(stub, 2), 3000, None, None, None, True,
                        span_excluded="gold_shares_token"),
        cs.SliceVerdict(_as(stub, 3), 3000, None, None, None, True, span_excluded="nfc_unstable"),
        cs.SliceVerdict(_as(stub, 4), 3000, False, False, (), None,
                        choice_excluded="nfc_unstable"),                 # abstained: a miss
    ]
    m = cs.slice_metrics(verdicts, row_exclusions=[(_as(stub, 5), "over_max_seq_len")])

    def at(name):
        return m[f"composed.val.{name}"]

    assert at("span_excluded.gold_shares_token").value == 1
    assert at("span_excluded.nfc_unstable").value == 1
    assert at("span_excluded.nfc_unstable.all.all").n_total == 4
    assert at("choice_excluded.nfc_unstable").value == 1
    assert at("choice_excluded.gold_shares_token").value == 0
    assert "choice_excluded.gold_shares_token.all.all" not in {
        k.removeprefix("composed.val.") for k in m
    }, "per-cell counts only for the buckets the set has"
    rows = at("rows_excluded.over_max_seq_len")
    assert (rows.value, rows.n_total) == (1, 5)
    assert at("refuse_gold.span_sequences").value == 2, "two span sequences decoded, two excluded"
    hit = at("refuse_gold.all.all.hunk_hit")
    assert (hit.n, hit.n_total) == (1, 2), "the exclusions are not misses"
    choice = at("both_policies.all.all.choice_top1")
    assert (choice.n, choice.n_total) == (3, 3)
    with pytest.raises(cs.SliceRefusal, match="counted twice"):
        cs.slice_metrics(verdicts, row_exclusions=[(_as(stub, 1), "over_max_seq_len")])
    with pytest.raises(cs.SliceRefusal, match="not a row exclusion"):
        cs.slice_metrics(verdicts, row_exclusions=[(_as(stub, 6), "nfc_unstable")])
