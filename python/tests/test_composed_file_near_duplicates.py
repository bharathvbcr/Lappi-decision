"""A composed row's files are checked against the other splits' files, one file at a time
(mistake M9, ``GAP-COMPOSE-NEAR-DUP-FILES-ACROSS-SPLITS-UNCHECKED``; data-clean-v6 plan item 5:
*"a composed train row holding a val file's near-duplicate is excluded"*).

A composed row holds 3 to 48 files of one split, and dedupe and the split's cross-check compare
whole rows: a 40-file row is never a near-duplicate of a one-file val row as a whole, so a fork's
copy of one of its files crossed the boundary unseen. ``split`` now signs each file block.

Every test fails on the pre-change code: ``SplitReport`` has no
``composed_file_near_duplicate_disjoint`` and ``composed_file_blocks`` does not exist.

Torch-free.
"""

from __future__ import annotations

from data_fixtures import code_body

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.rows import DataRow, GoldAnswer
from qd_data.schema import ChoiceSlot, Request
from qd_data.split import assign_repo, composed_file_blocks, split
from qd_train.tristate import Ran

CONFIG = DataConfig()


def _block(path: str, body: str) -> str:
    plus = "\n".join("+" + line for line in body.splitlines())
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -0,0 +1,40 @@\n{plus}\n"


def _row(row_id: str, *, repo: str, text: str) -> DataRow:
    family = "code.defect_class"
    return DataRow(
        row_id=row_id, source_id="qd-mutate/commitpackft", host="local", family_id=family,
        repo_key=repo, identity_key=f"{repo}::{row_id}", licence_id="mit",
        obligations=("attribution",),
        request=Request(
            task=family, context=text.encode("utf-8"), question="q?",
            slots=(ChoiceSlot(name="implements_claim", options=("yes", "no")),),
            example_id=row_id,
        ),
        gold=(GoldAnswer(slot_name="implements_claim", value="yes"),),
        dedupe_text=text,
    )


def _repo_in(split_name: str, prefix: str) -> str:
    for i in range(2000):
        repo = f"{prefix}{i:04d}"
        got = assign_repo(
            repo, seed=CONFIG.seed, train_fraction=CONFIG.train_fraction,
            val_fraction=CONFIG.val_fraction,
        )
        if got == split_name:
            return repo
    raise AssertionError(f"no repo reached {split_name}")


def _near_copy(body: str) -> str:
    lines = body.splitlines()
    lines[7] = lines[7] + "  # touched"
    return "\n".join(lines)


def _corpus(*, fork: bool) -> list[DataRow]:
    train_repo, val_repo = _repo_in("train", "org/t"), _repo_in("val", "org/v")
    files = {f"src/f{i}.py": code_body(f"unit{i}", lines=40) for i in range(4)}
    composed = "".join(_block(p, b) for p, b in files.items())
    val_body = _near_copy(files["src/f2.py"]) if fork else code_body("unrelated", lines=40)
    return [
        _row("composed-train", repo=train_repo, text=composed),
        _row("single-val", repo=val_repo, text=_block("lib/other.py", val_body)),
    ]


def test_blocks_are_the_files_of_a_multi_file_diff_without_their_headers() -> None:
    body = code_body("one", lines=40)
    text = _block("a.py", body) + _block("b.py", body)
    blocks = composed_file_blocks(text)
    assert len(blocks) == 2
    assert all("diff --git" not in b and "+++ " not in b and "@@" not in b for b in blocks)
    assert composed_file_blocks(_block("a.py", body)) == [], "one file is not composed"


def test_the_header_width_is_the_generators() -> None:
    from qd_data import split as split_module
    from qd_data.defect_class import COMPOSED_FILE_HEADER_LINES

    assert split_module._FILE_HEADER_LINES == COMPOSED_FILE_HEADER_LINES


def test_a_forked_copy_of_one_composed_file_in_a_val_repo_is_found() -> None:
    rows = _corpus(fork=True)
    report = split(dedupe(rows, config=CONFIG), config=CONFIG)
    check = report.composed_file_near_duplicate_disjoint
    assert isinstance(check, Ran) and check.passed is False, check
    assert "composed-train" in check.detail and "single-val" in check.detail
    assert report.status.to_json()["passed"] is False, "and the split's own status says so"


def test_whole_row_dedupe_and_the_row_level_check_do_not_see_it() -> None:
    """The reason the file-level check exists."""
    report = split(dedupe(_corpus(fork=True), config=CONFIG), config=CONFIG)
    assert report.near_duplicate_disjoint.to_json().get("passed") is not False


def test_unrelated_files_pass_and_say_how_many_blocks_were_signed() -> None:
    report = split(dedupe(_corpus(fork=False), config=CONFIG), config=CONFIG)
    check = report.composed_file_near_duplicate_disjoint
    assert isinstance(check, Ran) and check.passed is True, check
    assert check.n == 5 and check.n_total == 5, "four composed blocks and one single-file row"


def test_without_a_composed_row_the_check_is_absent_from_every_report() -> None:
    train_repo, val_repo = _repo_in("train", "org/t"), _repo_in("val", "org/v")
    rows = [
        _row("a", repo=train_repo, text=_block("a.py", code_body("x", lines=40))),
        _row("b", repo=val_repo, text=_block("b.py", code_body("y", lines=40))),
    ]
    report = split(dedupe(rows, config=CONFIG), config=CONFIG)
    assert report.composed_file_near_duplicate_disjoint is None
    assert "composed_file_near_duplicate_disjoint" not in report.to_json()
