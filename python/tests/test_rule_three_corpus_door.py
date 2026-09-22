"""Rule 3 for a pre-generated corpus: does the door the tools claim to use actually run?

``CLAUDE.md`` rule 3: *"Held-out data and the two task-holdout families are never read by a
training process. A path check in qd-train refuses them; removing that check is a refused
change."*

``tools/rung0_real_run.py`` states in its own module docstring that
``qd_train.mutate_adapter.read_examples`` is rung 0's held-out door. It is a door, and
nothing walked through it. All three rung-0 tools take ``--examples`` and read that JSONL
with a bare ``json.loads`` comprehension, so the path check never ran on the path every real
run uses. Measured 2026-09-22 against the unmodified tool: a corpus placed under
``data/heldout/`` was read, split file-disjointly and trained on, printing a baseline and a
seed line, with no refusal anywhere.

That is the precise failure ``mutate_adapter``'s own docstring warns about for the S4 path
-- *"without this the held-out check is satisfied on paper and bypassed by indirection"* --
and it was live on rung 0's only real entry point.

These tests are about the DOOR, not about any one caller: each tool is checked separately,
because a fix applied to one and forgotten in another is exactly how this arose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from qd_data.config import DataConfig  # noqa: E402
from qd_data.errors import HeldOutViolation  # noqa: E402
from qd_train.mutate_adapter import read_example_objects  # noqa: E402

#: One well-formed row, so a refusal cannot be confused with a parse failure.
ROW = {
    "id": "a",
    "function": {"repo": "r", "path": "a.py", "symbol": "f", "arity": 1},
    "language": "python",
    "class": "logic",
    "operator": "logic.off_by_one",
    "after": "line 0\nline 1\nline 2\n",
    "span": {"start_line": 2, "end_line": 2},
    "silent": False,
    "hunk_constrained": False,
    "seed": 1,
}


def _corpus(tmp_path: Path, *, under: str) -> Path:
    target = tmp_path / under
    target.mkdir(parents=True, exist_ok=True)
    path = target / "examples.jsonl"
    path.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    return path


def test_a_corpus_under_a_held_out_root_is_refused(tmp_path: Path) -> None:
    """The refusal names rule 3 rather than failing obscurely, because the next person to
    meet it is deciding whether to route around it."""
    path = _corpus(tmp_path, under="data/heldout")
    with pytest.raises(HeldOutViolation) as caught:
        read_example_objects(path, config=DataConfig(), repo_root=tmp_path)
    assert "rule 3" in str(caught.value).lower()


def test_a_corpus_under_an_ordinary_root_is_read(tmp_path: Path) -> None:
    """The check must refuse held-out paths and nothing else; a door that refused
    everything would be as useless as one that refused nothing."""
    path = _corpus(tmp_path, under="data/pool")
    rows = read_example_objects(path, config=DataConfig(), repo_root=tmp_path)
    assert [r["id"] for r in rows] == ["a"]


def test_a_corpus_outside_the_repo_root_still_reads(tmp_path: Path) -> None:
    """The production shape on the GH200: the corpus sits at
    ``/home/ubuntu/commitpackft-corpus-v2`` while the repo is ``/home/ubuntu/qwen-decision``.
    ``held_out_roots`` are relative and resolve against the repo root, so a door that
    refused everything outside the repo would have stopped every real arm the moment it
    landed -- which is the way a new guard usually breaks production.
    """
    repo = tmp_path / "qwen-decision"
    (repo / "data").mkdir(parents=True)
    corpus = tmp_path / "commitpackft-corpus-v2"
    corpus.mkdir()
    path = corpus / "examples.jsonl"
    path.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    assert read_example_objects(path, config=DataConfig(), repo_root=repo)[0]["id"] == "a"


def test_the_check_runs_before_the_file_is_opened(tmp_path: Path) -> None:
    """The INTENT to read is what is checked. A door that only refused paths that exist
    would pass on a typo and fail on the real thing, which is backwards."""
    missing = tmp_path / "data" / "heldout" / "not-there.jsonl"
    with pytest.raises(HeldOutViolation):
        read_example_objects(missing, config=DataConfig(), repo_root=tmp_path)


def test_a_held_out_marker_anywhere_in_the_path_is_refused(tmp_path: Path) -> None:
    """``held_out_path_markers`` covers a directory named for the split even outside the
    configured roots -- a corpus copied to ``scratch/held_out/`` is still held out."""
    path = _corpus(tmp_path, under="scratch/held_out")
    with pytest.raises(HeldOutViolation):
        read_example_objects(path, config=DataConfig(), repo_root=tmp_path)


def test_a_malformed_line_names_its_line_number_and_stops(tmp_path: Path) -> None:
    """Stops rather than skips, for the reason ``read_examples`` gives: a corpus with
    silently dropped rows has a coverage number nobody can reconstruct."""
    from qd_train.mutate_adapter import MalformedExample

    path = tmp_path / "data" / "pool" / "examples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ROW) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(MalformedExample, match=":2:"):
        read_example_objects(path, config=DataConfig(), repo_root=tmp_path)


def test_a_row_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    from qd_train.mutate_adapter import MalformedExample

    path = tmp_path / "data" / "pool" / "examples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(MalformedExample, match="is an object"):
        read_example_objects(path, config=DataConfig(), repo_root=tmp_path)


#: Every tool that takes a ``--examples`` corpus path. Kept as a list rather than
#: discovered, so adding a tool that reads a corpus is a decision someone makes here rather
#: than an omission nobody notices.
#:
#: Two of these train nothing. They are on the list anyway: ``filter_corpus`` reads a corpus
#: and writes another one somewhere else, so without the door it is the one tool that could
#: launder held-out rows into a path the training door would then admit; and the language
#: check reads a corpus for the same reason every other tool does.
CORPUS_READING_TOOLS = [
    "rung0_real_run",
    "rung0_linear_control",
    "fit_linear_control",
    "after_vs_diff",
    "operator_holdout",
    "filter_corpus",
    "operator_holdout_language_check",
]


@pytest.mark.parametrize("module_name", CORPUS_READING_TOOLS)
def test_every_corpus_reading_tool_goes_through_the_door(module_name: str) -> None:
    """Each tool separately. All of them took ``--examples`` and read it with a bare
    ``json.loads`` comprehension; a fix applied to one and forgotten in another is how this
    arose, so the property is asserted per tool rather than once.

    Checked on the imported module's own reference, which is what its ``main`` calls -- not
    on the source text, which would pass for a tool that imported the name and then did not
    use it.
    """
    pytest.importorskip("torch")
    module = __import__(module_name)
    assert getattr(module, "read_example_objects", None) is read_example_objects, (
        f"{module_name} must read its corpus through the rule-3 door, not around it"
    )


def test_no_corpus_reading_tool_is_missing_from_the_list() -> None:
    """The list above is the enforcement, so a tool that takes ``--examples`` and is not on
    it is the next instance of this bug. Found by reading the sources, which is the right
    direction here: the risk is a tool that never imports the door at all, and a check over
    imported modules cannot see one it was never told about.
    """
    tools = REPO / "tools"
    takes_a_corpus = {
        path.stem
        for path in sorted(tools.glob("*.py"))
        if '"--examples"' in path.read_text(encoding="utf-8")
    }
    missing = sorted(takes_a_corpus - set(CORPUS_READING_TOOLS))
    assert not missing, (
        f"these tools take --examples and are not on the rule-3 list: {missing}. Add them "
        "to CORPUS_READING_TOOLS and route them through read_example_objects, or say in "
        "the list's comment why the door does not apply."
    )
