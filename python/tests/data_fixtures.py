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

import functools
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from qd_data.config import DataConfig
from qd_data.loaders import ClincRow, CommitPackFtRow, SquadRow
from qd_data.mixture import LANGUAGE_OPTIONS
from qd_data.split import squad_title_family

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


def squad_title_for(family_id: str, *, stem: str = "Article", seed: int | None = None) -> str:
    """The first ``{stem}{k}`` title that SQuAD's title partition gives to ``family_id``.

    SQuAD is partitioned by article title between qa.answerability and qa.answer_span
    (``qd_data.split.squad_title_family``, user decision 2026-09-29). Tests that need a row
    in a particular family ask for its title here rather than hard-coding one, so no test
    pins where a given string happens to hash, and moving the fraction moves no test.
    """
    seed = DataConfig().seed if seed is None else seed
    for k in range(100_000):
        title = f"{stem}{k}"
        if squad_title_family(title, seed=seed) == family_id:
            return title
    raise AssertionError(f"no {stem}<k> title routes to {family_id} in 100,000 tries")


def squad_row(
    i: int, *, force_impossible: bool | None = None, family: str | None = None
) -> SquadRow:
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
        # One article per question, or one chosen to land in `family`: SQuAD is partitioned
        # by title between its two families (split.squad_title_family).
        title=squad_title_for(family, stem=f"Article{i}-") if family else f"Article{i}",
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
        # A quarter to the held-out answerability family, by construction rather than by
        # where the titles hash, so the corpus has both SQuAD families at any fraction.
        "rajpurkar/squad_v2": [
            squad_row(i, family="qa.answerability" if i % 4 == 0 else "qa.answer_span")
            for i in range(n)
        ],
    }


def mutate_row(**over) -> dict:
    """A valid qd-mutate example row, overridable per test.

    Shaped after ``crates/qd-mutate/src/generate.rs``' output: a Rust `negate_condition`
    on a two-line span, hunk-constrained, with every documented field populated. Tests
    override single keys to build the malformed cases rather than restating the row.
    """
    base = {
        "id": "ex-1",
        "pool_id": "pool-1",
        "repo": "acme/widget",
        "path": "src/lib.rs",
        "language": "Rust",
        "class": "logic",
        "operator": "negate_condition",
        "silent": False,
        "span": {"start_line": 2, "end_line": 2},
        "function": {"repo": "acme/widget", "path": "src/lib.rs", "symbol": "f", "arity": 1},
        "node_kind": "function_item",
        "is_nested": False,
        "before": "fn f(x: i32) -> bool {\n    x > 0\n}\n",
        "after": "fn f(x: i32) -> bool {\n    x < 0\n}\n",
        "diff": "-    x > 0\n+    x < 0\n",
        "normalization": {"bom": False, "crlf": False, "lone_cr": False, "mixed_endings": False},
        "hunk_constrained": True,
        "detail": "",
        "seed": 7,
        "tool_version": "0.1.0",
    }
    base.update(over)
    return base


#: ``tools/real_tokenizer_pipeline.PREP_BIN_ENV``, restated so this module imports no tool;
#: ``test_qd_prep_minhash_parity.py`` asserts the two agree.
QD_PREP_BIN_ENV = "QD_PREP_BIN"
_REPO = Path(__file__).resolve().parents[2]


@functools.cache
def qd_prep_bin() -> Path:
    """The ``qd-prep`` binary MinHash signing runs in, built once per test process.

    The one the environment names when it names one; otherwise ``cargo build --release -p
    qd-prep`` in this checkout. Without either the calling test is SKIPPED with that reason:
    a dedupe that could not run is not a dedupe that passed.
    """
    named = os.environ.get(QD_PREP_BIN_ENV, "")
    if named:
        return Path(named)
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip(f"cargo is not on PATH and {QD_PREP_BIN_ENV} is unset: qd-prep NOT built")
    done = subprocess.run(
        [cargo, "build", "--release", "-p", "qd-prep", "--bin", "qd-prep"],
        cwd=_REPO, capture_output=True, text=True, timeout=900, check=False,
    )
    if done.returncode != 0:
        raise AssertionError(f"cargo build -p qd-prep failed: {done.stderr[-3000:]}")
    return _REPO / "target" / "release" / "qd-prep"


def use_qd_prep(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``QD_PREP_BIN`` at :func:`qd_prep_bin` for one test; dedupe and split sign
    there (``real_tokenizer_pipeline.native_minhash``), with no Python fallback."""
    binary = qd_prep_bin()
    monkeypatch.setenv(QD_PREP_BIN_ENV, str(binary))
    return binary
