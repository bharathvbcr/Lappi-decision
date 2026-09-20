"""The corpus generator: no two rows may share a prompt and disagree about the answer.

``tools/real_tokenizer_pipeline.py::span_rows`` emits two rows per Markdown file, one
answerable and one not. Until 2026-09-20 both asked ``f"Which line of {name} states the
rule?"`` over the same passage, so the prompt was byte-identical and only the gold differed
-- one a real line span, the other ``noul``.

A causal model conditions on ``tokens[:target_index + 1]`` and nothing else, so it cannot
fit both: the optimum over such a pair is its label distribution, and the measured
consequence was a span-loss floor of **0.693147 = ln 2**, the entropy of a fair coin, with
span accuracy capped at 39 of 78 permanently. On a rented GPU the only symptom is a loss
that plateaus and an eval that reports 50%.

The fix was one line -- give the unanswerable row a different question, as SQuAD 2.0 does
-- and nothing guarded it. This is that guard. Measured after the fix on a rebuilt corpus:
``prompt-identical rows: 0 of 341`` and a plan-level span floor of 0.0, against 78 of 321
in 39 groups and 0.693147 before.

Torch-free, so it runs in the repo venv and is counted by ``make gates``: the failure it
catches costs a GPU-hours-long run, and the check for it must not need a GPU.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_tokenizer_pipeline as pipeline  # noqa: E402

#: Enough files to exercise many pairs without making the suite depend on repository size.
MAX_ROWS = 60


@pytest.fixture(scope="module")
def rows():
    generated, _capped = pipeline.span_rows(
        max_rows=MAX_ROWS, blank_line_runs=False, rev="HEAD"
    )
    if len(generated) < 4:  # pragma: no cover - depends on the tree having Markdown
        pytest.skip("HEAD carries too few Markdown files to exercise the generator")
    return generated


def test_no_two_span_rows_share_a_prompt(rows) -> None:
    """The defect, stated directly over the generator's output.

    The prompt is ``(question, context)``: the two things the model sees. Two rows agreeing
    on both and disagreeing on the gold is the unfittable case, and this refuses the
    weaker version too -- any duplicate prompt at all, whether or not the golds differ,
    because a duplicated prompt with the SAME gold is merely wasted supervision and the
    check that distinguishes them is the one that just failed to exist.
    """
    prompts = Counter((r.question, r.context) for r in rows)
    repeated = {p: n for p, n in prompts.items() if n > 1}
    assert not repeated, (
        f"{len(repeated)} prompt(s) appear more than once across {len(rows)} rows; "
        f"first repeated question: {next(iter(repeated))[0]!r}"
    )


def test_the_answerable_and_unanswerable_rows_of_one_file_ask_different_questions(rows):
    """The specific pairing that produced the ln 2 floor.

    Grouped by ``context`` rather than by title, because the passage is what makes two
    questions comparable -- the same file could legitimately yield two passages.
    """
    by_context: dict[str, list] = defaultdict(list)
    for r in rows:
        by_context[r.context].append(r)

    pairs_seen = 0
    for members in by_context.values():
        answerable = [r for r in members if not r.is_impossible]
        impossible = [r for r in members if r.is_impossible]
        if not (answerable and impossible):
            continue
        pairs_seen += 1
        questions = {r.question for r in members}
        assert len(questions) == len(members), (
            f"a passage from {members[0].title!r} yields {len(members)} rows but only "
            f"{len(questions)} distinct question(s): {sorted(questions)!r}. One of them is "
            "unanswerable, so identical questions make the pair unfittable by construction."
        )
    assert pairs_seen > 0, "no answerable/unanswerable pair was generated, so nothing was checked"


def test_an_unanswerable_row_carries_no_gold_and_an_answerable_one_does(rows) -> None:
    """The other half of the contract: the rows must actually differ in what they assert,
    or distinct questions would be hiding two rows that say the same thing."""
    for r in rows:
        if r.is_impossible:
            assert r.answers == () and r.answer_starts == (), (
                f"{r.qid} is marked impossible but carries {r.answers!r}"
            )
        else:
            assert len(r.answers) == 1 and len(r.answer_starts) == 1, (
                f"{r.qid} is answerable but carries {len(r.answers)} answer(s)"
            )
            answer, start = r.answers[0], r.answer_starts[0]
            assert r.context[start : start + len(answer)] == answer, (
                f"{r.qid}'s answer is not at the offset it claims -- the gold line span "
                "derived from it would point at different text"
            )


def test_an_unanswerable_question_does_not_contain_its_own_answer(rows) -> None:
    """Pinned from an inline ``assert`` in the generator, which runs only when the
    generator does. A question containing the gold makes an 'unanswerable' row answerable
    from the prompt alone."""
    by_context: dict[str, list] = defaultdict(list)
    for r in rows:
        by_context[r.context].append(r)
    for members in by_context.values():
        golds = [r.answers[0] for r in members if not r.is_impossible]
        for r in members:
            if not r.is_impossible:
                continue
            for gold in golds:
                assert gold not in r.question, (
                    f"{r.qid} is unanswerable but its question contains the answerable "
                    f"row's gold: {gold!r}"
                )
