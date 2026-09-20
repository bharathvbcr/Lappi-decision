"""The qd-mutate -> byte-space seam. Torch-free, runs in the repo venv.

The load-bearing test is `test_the_phantom_final_line_is_refused_not_clamped`: qd-mutate
counts a line that byte space cannot represent, and mapping it to the last real line would
mislabel every edit appended at end-of-file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from data_fixtures import mutate_row as row

from qd_data.config import DataConfig
from qd_data.errors import HeldOutViolation
from qd_train.byte_context import SpanOutsideWindow, line_starts
from qd_train.mutate_adapter import (
    CLEAN,
    MUTATION_CLASSES,
    LineSpan,
    MalformedExample,
    PhantomFinalLine,
    contrastive_pairs,
    mutate_total_lines,
    pair_key,
    parse_example,
    parse_line_span,
    read_examples,
    span_end_offset,
    span_start_offset,
    to_decision,
)

OPS_RS = Path(__file__).resolve().parents[2] / "crates" / "qd-mutate" / "src" / "ops.rs"


# ---------------------------------------------------------------------------
# The phantom final line
# ---------------------------------------------------------------------------


def test_the_two_line_counting_conventions_differ_and_both_are_named():
    """The gap this module exists to bridge, asserted rather than described.

    qd-mutate's `total_lines` is newline count + 1. `line_starts` reports only lines that
    hold bytes. They agree unless the text ends in a newline.
    """
    assert mutate_total_lines(b"a") == 1 == len(line_starts(b"a"))
    assert mutate_total_lines(b"a\nb") == 2 == len(line_starts(b"a\nb"))

    # Ends in a newline: qd-mutate counts a phantom, byte space does not.
    assert mutate_total_lines(b"a\n") == 2
    assert len(line_starts(b"a\n")) == 1
    assert mutate_total_lines(b"a\nb\n") == 3
    assert len(line_starts(b"a\nb\n")) == 2


def test_the_phantom_final_line_is_refused_not_clamped():
    """An edit appended at EOF of a newline-terminated file lands on the phantom.

    Clamping it to the last real line would put the span label one line early on every such
    example, and nothing in the loss would say so.
    """
    after = b"fn f() {}\n"  # qd-mutate: 2 lines. byte space: 1.
    assert mutate_total_lines(after) == 2
    assert len(line_starts(after)) == 1

    with pytest.raises(PhantomFinalLine, match="phantom final line"):
        span_start_offset(after, LineSpan(start_line=2, end_line=2))

    # The real line is still perfectly addressable.
    assert span_start_offset(after, LineSpan(start_line=1, end_line=1)) == 0


def test_a_span_past_even_the_phantom_is_a_different_error():
    """Out-of-file and on-the-phantom are distinct faults and must not share a message."""
    after = b"a\n"
    with pytest.raises(MalformedExample, match="past the end"):
        span_start_offset(after, LineSpan(start_line=3, end_line=3))


def test_a_file_not_ending_in_a_newline_has_no_phantom():
    after = b"one\ntwo\nthree"
    assert mutate_total_lines(after) == 3 == len(line_starts(after))
    assert span_start_offset(after, LineSpan(3, 3)) == 8


# ---------------------------------------------------------------------------
# 1-based inclusive -> byte offset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [(1, 0), (2, 5), (3, 9), (4, 13)],
)
def test_line_numbers_convert_to_the_offsets_line_starts_reports(line: int, expected: int):
    after = b"zero\none\ntwo\nthree"
    assert line_starts(after) == (0, 5, 9, 13)
    assert span_start_offset(after, LineSpan(line, line)) == expected


def test_a_zero_or_negative_line_is_refused_because_spans_are_one_based():
    with pytest.raises(MalformedExample, match="1-based"):
        LineSpan(start_line=0, end_line=1)


def test_an_inverted_span_is_refused():
    with pytest.raises(MalformedExample, match="precedes start_line"):
        LineSpan(start_line=5, end_line=2)


def test_a_single_line_span_is_valid_because_both_ends_are_inclusive():
    assert LineSpan(3, 3).line_count == 1
    assert LineSpan(3, 5).line_count == 3


# ---------------------------------------------------------------------------
# Wire field names
# ---------------------------------------------------------------------------


def test_the_wire_spelling_is_start_line_and_end_line():
    assert parse_line_span({"start_line": 1, "end_line": 4}, where="x") == LineSpan(1, 4)


def test_the_short_spelling_is_refused_and_says_why():
    """`{start, end}` is the spelling qd-runtime refuses under deny_unknown_fields."""
    with pytest.raises(MalformedExample, match="deny_unknown_fields"):
        parse_line_span({"start": 1, "end": 4}, where="x")


def test_a_non_integer_line_is_refused():
    with pytest.raises(MalformedExample, match="must be an int"):
        parse_line_span({"start_line": 1.5, "end_line": 4}, where="x")
    # bool is an int in Python; it is not a line number.
    with pytest.raises(MalformedExample, match="must be an int"):
        parse_line_span({"start_line": True, "end_line": 4}, where="x")


# ---------------------------------------------------------------------------
# clean vs abstention
# ---------------------------------------------------------------------------


def test_clean_is_a_real_class_with_a_real_label_and_an_abstaining_span():
    ex = parse_example(row(id="c", **{"class": CLEAN}, span=None))
    assert ex.mutation_class == CLEAN
    assert ex.span is None

    d = to_decision(ex, max_context_bytes=512)
    assert d.options[d.gold_option] == b"clean", "clean supervises the choice slot normally"
    assert d.span_is_noul, "and abstains on the span slot, because it points at nothing"


def test_a_clean_example_carrying_a_span_is_refused():
    with pytest.raises(MalformedExample, match="points at nothing"):
        parse_example(row(**{"class": CLEAN}, span={"start_line": 1, "end_line": 1}))


def test_a_mutated_example_without_a_span_is_refused_not_demoted_to_noul():
    """Demoting it would teach the span head to abstain on real mutations."""
    with pytest.raises(MalformedExample, match="must not be silently demoted"):
        parse_example(row(span=None))


def test_the_span_abstain_row_is_one_past_the_last_line():
    """Same layout as answer.rs: `span_rows = line_count + RESERVED_NOUL_ROWS`, abstain last."""
    ex = parse_example(row(id="c", **{"class": CLEAN}, span=None, after="a\nb\nc"))
    d = to_decision(ex, max_context_bytes=512)
    assert len(d.context.starts) == 3
    assert d.span_abstain_row == 3


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_a_span_in_the_dropped_head_is_refused_not_repointed():
    after = "\n".join(f"line {i}" for i in range(200))
    ex = parse_example(row(after=after, span={"start_line": 1, "end_line": 1}))
    with pytest.raises(SpanOutsideWindow, match="truncation dropped"):
        to_decision(ex, max_context_bytes=64)


def test_a_span_that_survives_truncation_is_rebased_onto_the_window():
    lines = [f"line {i}" for i in range(40)]
    after = "\n".join(lines)
    # Target a late line so it survives a tail-keeping truncation.
    target = 39
    ex = parse_example(row(after=after, span={"start_line": target, "end_line": target}))

    full = to_decision(ex, max_context_bytes=4096)
    cut = to_decision(ex, max_context_bytes=64)
    assert cut.context.truncated and not full.context.truncated

    # Same line of text, different index, because the window starts later.
    full_line = bytes(full.context.ids)[full.context.starts[full.gold_span_line] :].split(b"\n")[0]
    cut_line = bytes(cut.context.ids)[cut.context.starts[cut.gold_span_line] :].split(b"\n")[0]
    assert full_line == cut_line == b"line 38"
    assert cut.gold_span_line < full.gold_span_line


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_a_missing_required_field_is_an_error_not_a_default():
    bad = row()
    del bad["silent"]
    with pytest.raises(MalformedExample, match="missing required field 'silent'"):
        parse_example(bad)


def test_an_unknown_class_is_refused():
    with pytest.raises(MalformedExample, match="is not one of"):
        parse_example(row(**{"class": "refactor"}))


def test_a_gold_outside_the_option_set_is_unanswerable():
    ex = parse_example(row())
    with pytest.raises(MalformedExample, match="unanswerable"):
        to_decision(ex, max_context_bytes=512, options=("stub", "clean"))


def _corpus(tmp_path: Path, *lines: str, name: str = "ex.jsonl") -> Path:
    """Write a JSONL corpus and return its path. ``read`` opens it under a clean config."""
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(lines), encoding="utf-8")
    return p


def read(path: Path) -> list:
    """``read_examples`` with the default holdout, rooted at the corpus's own directory.

    Named once so the rule-3 arguments are not restated in every call site, and so the
    held-out cases below differ from the ordinary ones in the *path* and nothing else.
    """
    return list(read_examples(path, config=DataConfig(), repo_root=path.parent))


def test_read_examples_names_the_line_it_could_not_read(tmp_path: Path):
    p = _corpus(tmp_path, json.dumps(row()) + "\n", "{not json\n")
    with pytest.raises(MalformedExample, match=r"ex\.jsonl:2: not JSON"):
        read(p)


def test_read_examples_stops_rather_than_skipping_a_bad_row(tmp_path: Path):
    """A silently skipped row makes the corpus size unreconstructable."""
    bad = row(id="ex-2")
    del bad["seed"]
    p = _corpus(tmp_path, json.dumps(row()) + "\n", json.dumps(bad) + "\n")
    with pytest.raises(MalformedExample, match=r"ex\.jsonl:2:.*'seed'"):
        read(p)


def test_blank_lines_are_skipped(tmp_path: Path):
    p = _corpus(tmp_path, "\n", json.dumps(row()) + "\n", "\n")
    assert len(read(p)) == 1


# ---------------------------------------------------------------------------
# Rule 3: rung 0's door
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marker", ["heldout", "held_out", "held-out"])
def test_read_examples_refuses_a_held_out_corpus(tmp_path: Path, marker: str):
    """Rule 3 at rung 0's door, which is the door rung 0 actually uses.

    Measured against this module before the check existed: ``assert_path_not_held_out``
    refused ``data/heldout/mutate.jsonl`` while ``read_examples`` parsed it and
    ``to_decision`` turned it into a trainable ``ByteDecision``. The S4 path carries the
    same check twice -- at the manifest and again at the shard, the second one added
    because the first could be bypassed by indirection -- and rung 0 carried it nowhere.

    Parametrised over every marker ``DataConfig`` declares, because a check that fires on
    one spelling of the directory and not the others is the check working by luck.
    """
    p = _corpus(tmp_path, json.dumps(row()) + "\n", name=f"data/{marker}/mutate.jsonl")
    with pytest.raises(HeldOutViolation, match="rule 3"):
        list(read_examples(p, config=DataConfig(), repo_root=tmp_path))


def test_read_examples_refuses_before_it_opens_the_file(tmp_path: Path):
    """The refusal is about the intent to read, not about a successful read.

    A check that only fired on an existing file would pass silently for a path typo and
    would let a corpus be admitted the moment someone created it.
    """
    missing = tmp_path / "data" / "heldout" / "does-not-exist.jsonl"
    assert not missing.exists()
    with pytest.raises(HeldOutViolation, match="rule 3"):
        list(read_examples(missing, config=DataConfig(), repo_root=tmp_path))


def test_read_examples_refuses_a_path_that_climbs_into_the_held_out_tree(tmp_path: Path):
    """Layer 1 is compared on resolved paths, so ``train/../heldout`` does not launder one."""
    _corpus(tmp_path, json.dumps(row()) + "\n", name="data/heldout/mutate.jsonl")
    (tmp_path / "data" / "train").mkdir(parents=True, exist_ok=True)
    sneaky = tmp_path / "data" / "train" / ".." / "heldout" / "mutate.jsonl"
    with pytest.raises(HeldOutViolation, match="rule 3"):
        list(read_examples(sneaky, config=DataConfig(), repo_root=tmp_path))


def test_read_examples_admits_an_ordinary_corpus_under_the_same_config(tmp_path: Path):
    """The control. Without it the four refusals above are also satisfied by a door that
    refuses everything, which would be a rule 3 that passes by training on nothing."""
    p = _corpus(tmp_path, json.dumps(row()) + "\n", name="data/train/mutate.jsonl")
    got = list(read_examples(p, config=DataConfig(), repo_root=tmp_path))
    assert len(got) == 1


# ---------------------------------------------------------------------------
# Contrastive pairs
# ---------------------------------------------------------------------------


def test_a_mutation_pairs_with_a_clean_example_of_the_same_function():
    fn = {"repo": "r", "path": "p", "symbol": "f", "arity": 1}
    mutated = parse_example(row(id="m", function=fn))
    clean = parse_example(row(id="c", function=fn, **{"class": CLEAN}, span=None))
    pairs = contrastive_pairs([mutated, clean])
    assert pairs == [(mutated, clean)]


def test_functions_with_no_clean_counterpart_yield_no_pair():
    fn = {"repo": "r", "path": "p", "symbol": "f", "arity": 1}
    only_mutated = [parse_example(row(id=f"m{i}", function=fn)) for i in range(3)]
    assert contrastive_pairs(only_mutated) == []


def test_pairing_does_not_cross_functions():
    a = {"repo": "r", "path": "p", "symbol": "f", "arity": 1}
    b = {"repo": "r", "path": "p", "symbol": "g", "arity": 1}
    m = parse_example(row(id="m", function=a))
    c = parse_example(row(id="c", function=b, **{"class": CLEAN}, span=None))
    assert contrastive_pairs([m, c]) == []


def test_arity_is_part_of_the_identity_so_overloads_do_not_pair():
    one = {"repo": "r", "path": "p", "symbol": "f", "arity": 1}
    two = {"repo": "r", "path": "p", "symbol": "f", "arity": 2}
    m = parse_example(row(id="m", function=one))
    c = parse_example(row(id="c", function=two, **{"class": CLEAN}, span=None))
    assert contrastive_pairs([m, c]) == []
    assert pair_key(m) != pair_key(c)


def test_one_clean_anchors_a_group_so_pair_counts_are_not_inflated():
    fn = {"repo": "r", "path": "p", "symbol": "f", "arity": 1}
    mutated = [parse_example(row(id=f"m{i}", function=fn)) for i in range(2)]
    cleans = [
        parse_example(row(id=f"c{i}", function=fn, **{"class": CLEAN}, span=None))
        for i in range(3)
    ]
    pairs = contrastive_pairs(mutated + cleans)
    assert len(pairs) == 2, "2 mutations x 3 cleans must not become 6 pairs"
    assert {p[1].example_id for p in pairs} == {"c0"}


def test_pair_key_matches_the_rust_display_format():
    ex = parse_example(row())
    assert pair_key(ex) == "acme/widget:src/lib.rs:f/1"


# ---------------------------------------------------------------------------
# The Rust seam
# ---------------------------------------------------------------------------


def test_mutation_classes_still_match_ops_rs():
    """The Python mirror is duplication; this is what stops it rotting."""
    src = OPS_RS.read_text(encoding="utf-8")
    body = re.search(r"pub enum MutationClass \{(.*?)\}", src, re.S)
    assert body, f"MutationClass not found in {OPS_RS}"
    variants = tuple(v.strip().rstrip(",").lower() for v in body.group(1).split() if v.strip())
    assert variants == MUTATION_CLASSES, (
        f"ops.rs declares {variants}, this module mirrors {MUTATION_CLASSES}"
    )
    assert 'rename_all = "lowercase"' in src, "the lowercase serde spelling is what we mirror"


# ---------------------------------------------------------------------------
# The end pointer resolves through the same checks as the start
# ---------------------------------------------------------------------------


def test_span_end_offset_resolves_the_last_line_not_the_first():
    """`SpanPointerHead` has two pointers; the end one needs its own offset."""
    after = b"aa\nbb\ncc\n"
    span = LineSpan(start_line=1, end_line=3)
    assert span_start_offset(after, span) == 0
    assert span_end_offset(after, span) == 6


def test_a_single_line_span_puts_both_pointers_on_one_offset():
    after = b"aa\nbb\ncc\n"
    span = LineSpan(start_line=2, end_line=2)
    assert span_start_offset(after, span) == span_end_offset(after, span) == 3


def test_an_end_line_on_the_phantom_is_refused_even_when_the_start_is_real():
    """The case the shared `_span_offset` exists for.

    A check that only ever guarded `start_line` would let the end pointer address the
    phantom final line -- a position byte space cannot represent -- while the start
    resolved cleanly, so the refusal would never fire.
    """
    after = b"aa\nbb\n"  # qd-mutate counts 3 lines; only 2 hold bytes
    assert mutate_total_lines(after) == 3
    assert len(line_starts(after)) == 2

    span = LineSpan(start_line=2, end_line=3)
    assert span_start_offset(after, span) == 3, "the start is a real line"
    with pytest.raises(PhantomFinalLine, match="end_line 3 is qd-mutate's phantom"):
        span_end_offset(after, span)


def test_the_refusal_names_which_pointer_hit_the_phantom():
    """`start_line 3` and `end_line 3` are different faults; one message for both hides it."""
    after = b"aa\nbb\n"
    with pytest.raises(PhantomFinalLine, match="start_line 3"):
        span_start_offset(after, LineSpan(start_line=3, end_line=3))
    with pytest.raises(PhantomFinalLine, match="end_line 3"):
        span_end_offset(after, LineSpan(start_line=2, end_line=3))


def test_to_decision_refuses_an_example_whose_end_lands_on_the_phantom():
    """End to end: the decision never carries a fabricated end line."""
    with pytest.raises(PhantomFinalLine):
        to_decision(
            parse_example(row(after="aa\nbb\n", span={"start_line": 2, "end_line": 3})),
            max_context_bytes=4096,
        )
