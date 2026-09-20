"""Synthetic corpora for the data-lane tests. Not a test module.

Everything here is generated, small and deterministic. The lane brief forbids
downloading the 62 GB pool, and a fixture that needed the network would make the
whole suite report "not run" on a plane -- which is correct behaviour for a network
test and useless as the foundation for a hundred offline ones.

The generators are deliberately *shaped*: :func:`vendored_pair` builds the exact
corpus ``docs/hardening.md`` section 2 warns about -- the same file in two
repositories -- because a dedupe test over random text proves only that random text
is not duplicated.
"""

from __future__ import annotations

from qd_data.loaders import ClincRow, CommitPackFtRow, SquadRow
from qd_data.mixture import LANGUAGE_OPTIONS

INTENT_VOCABULARY: tuple[str, ...] = tuple(f"intent_{i:03d}" for i in range(40))


def code_body(tag: str, lines: int = 12) -> str:
    return "\n".join(f"    value_{tag}_{i} = compute(step={i}, tag='{tag}')" for i in range(lines))


def commitpackft_row(
    i: int,
    *,
    licence: str = "mit",
    repo: str | None = None,
    lang: str | None = None,
    body: str | None = None,
    path: str | None = None,
) -> CommitPackFtRow:
    """One synthetic commitpackft row with every documented field populated."""
    old = body if body is not None else code_body(f"r{i}")
    return CommitPackFtRow(
        commit=f"{i:040x}",
        repos=repo if repo is not None else f"org/repo{i % 7}",
        old_file=path or f"src/module_{i}.py",
        new_file=path or f"src/module_{i}.py",
        old_contents=f"def handler_{i}():\n{old}\n",
        new_contents=f"def handler_{i}():\n{old}\n    return True\n",
        subject=f"add return to handler {i}",
        message=f"add an explicit return to handler {i} so callers can branch on it",
        lang=lang or LANGUAGE_OPTIONS[i % len(LANGUAGE_OPTIONS)],
        licence=licence,
    )


def vendored_pair(*, repo_a: str = "org/app", repo_b: str = "org/vendored") -> tuple[
    CommitPackFtRow, CommitPackFtRow
]:
    """The same file in two repositories.

    ``docs/hardening.md`` section 2: *vendored trees, forks and copied files put the
    same code in two repos*, which is exactly what a repo-level split cannot separate.
    The two rows differ only in the repo and the commit id, as a real vendoring does.
    """
    shared = code_body("shared", lines=30)
    a = commitpackft_row(1, repo=repo_a, body=shared, path="lib/parser.py")
    b = CommitPackFtRow(
        commit=f"{999:040x}",
        repos=repo_b,
        old_file="third_party/lib/parser.py",
        new_file="third_party/lib/parser.py",
        old_contents=a.old_contents,
        new_contents=a.new_contents,
        subject=a.subject,
        message=a.message,
        lang=a.lang,
        licence=a.licence,
    )
    return a, b


def clinc_row(i: int, *, force_oos: bool | None = None) -> ClincRow:
    oos = (i % 9 == 0) if force_oos is None else force_oos
    return ClincRow(
        utterance=f"can you please handle request number {i} for my account today",
        intent="oos" if oos else INTENT_VOCABULARY[i % len(INTENT_VOCABULARY)],
        is_oos=oos,
    )


def squad_row(i: int, *, force_impossible: bool | None = None) -> SquadRow:
    passage = (
        f"Paragraph {i} opens with background material.\n"
        f"The second line states that Subject{i} was founded in {1800 + i}.\n"
        f"A third line adds unrelated detail about the region."
    )
    impossible = (i % 5 == 0) if force_impossible is None else force_impossible
    needle = f"Subject{i}"
    start = passage.index(needle)
    return SquadRow(
        qid=f"q{i:05d}",
        title=f"Article{i % 6}",
        context=passage,
        question=f"What was founded in paragraph {i}?",
        answers=() if impossible else (needle,),
        answer_starts=() if impossible else (start,),
        is_impossible=impossible,
    )


def small_corpus(n: int = 24) -> dict[str, list[object]]:
    """A mixture-ready corpus across all three admitted sources."""
    return {
        "bigcode/commitpackft": [commitpackft_row(i) for i in range(n)],
        "clinc/clinc_oos": [clinc_row(i) for i in range(n)],
        "rajpurkar/squad_v2": [squad_row(i) for i in range(n)],
    }
