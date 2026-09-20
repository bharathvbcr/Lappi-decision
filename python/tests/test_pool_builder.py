"""Building a qd-mutate pool from local source trees."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from qd_data.licences import LicenceConfig, LicenceRefused
from qd_data.pool_builder import (
    POOL_EXTENSIONS,
    SKIP_DIRS,
    build_pool,
    write_pool,
)
from qd_train.tristate import NotRun, Ran

POOL_RS = Path(__file__).resolve().parents[2] / "crates" / "qd-mutate" / "src" / "pool.rs"
MIT = "MIT"


def tree(root: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Licence comes first
# ---------------------------------------------------------------------------


def test_the_licence_is_checked_before_any_file_is_read(tmp_path: Path):
    """A refused pool costs nothing and the refusal is not buried under a partial result."""
    tree(tmp_path, {"a.py": "def f():\n    return 1\n"})
    # `normalise_licence` lowercases, and the refusal names the normalised id.
    with pytest.raises(LicenceRefused, match=r"gpl\-3\.0"):
        build_pool(tmp_path, repo="r", licence="GPL-3.0")


def test_a_licence_needing_a_human_call_can_be_admitted_only_with_a_reason(tmp_path: Path):
    tree(tmp_path, {"a.py": "def f():\n    return 1\n"})
    with pytest.raises(ValueError, match="no justification"):
        LicenceConfig(admitted_by_human={"MPL-2.0": "  "})


def test_the_admitted_licence_and_its_obligations_are_carried(tmp_path: Path):
    tree(tmp_path, {"a.py": "def f():\n    return 1\n"})
    report = build_pool(tmp_path, repo="r", licence=MIT)
    # Normalised, not echoed: the model card must carry one spelling per licence.
    assert report.licence_id == "mit"
    assert isinstance(report.obligations, tuple)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_only_languages_qd_mutate_parses_are_selected(tmp_path: Path):
    tree(
        tmp_path,
        {
            "keep.rs": "fn f() {}\n",
            "keep.py": "def f():\n    return 1\n",
            "keep.swift": "func f() {}\n",
            "skip.md": "# docs\n",
            "skip.json": "{}\n",
            "skip.txt": "hello\n",
        },
    )
    report = build_pool(tmp_path, repo="r", licence=MIT)
    assert {Path(r.path).name for r in report.records} == {"keep.rs", "keep.py", "keep.swift"}
    assert report.skipped["not a supported language"] == 3


def test_declaration_files_are_skipped_because_they_have_no_function_bodies(tmp_path: Path):
    tree(
        tmp_path,
        {
            "types.d.ts": "export declare function f(): void;\n",
            "impl.ts": "function f() {}\n",
        },
    )
    report = build_pool(tmp_path, repo="r", licence=MIT)
    assert [Path(r.path).name for r in report.records] == ["impl.ts"]


@pytest.mark.parametrize("skip_dir", ["node_modules", "target", ".git", "vendor", "__pycache__"])
def test_generated_and_vendored_trees_are_pruned(tmp_path: Path, skip_dir: str):
    tree(tmp_path, {"src/a.py": "def f():\n    return 1\n", f"{skip_dir}/b.py": "def g(): pass\n"})
    report = build_pool(tmp_path, repo="r", licence=MIT)
    assert [r.path for r in report.records] == ["src/a.py"]


def test_symlinks_are_not_followed_out_of_the_tree(tmp_path: Path):
    """Following one would attach this repo's licence to code from somewhere else."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "other.py").write_text("def other(): pass\n", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (root / "link.py").symlink_to(outside / "other.py")

    report = build_pool(root, repo="r", licence=MIT)
    assert [r.path for r in report.records] == ["a.py"]


def test_a_non_utf8_file_is_skipped_and_counted(tmp_path: Path):
    tree(tmp_path, {"good.py": "def f():\n    return 1\n"})
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe\x00def f(): pass\n")
    report = build_pool(tmp_path, repo="r", licence=MIT)
    assert [Path(r.path).name for r in report.records] == ["good.py"]
    assert report.skipped["not valid utf-8"] == 1


def test_an_empty_file_is_skipped(tmp_path: Path):
    tree(tmp_path, {"a.py": "def f():\n    return 1\n", "empty.py": "   \n\n"})
    report = build_pool(tmp_path, repo="r", licence=MIT)
    assert [Path(r.path).name for r in report.records] == ["a.py"]
    assert report.skipped["empty"] == 1


def test_the_record_carries_a_posix_relative_path_and_a_stable_id(tmp_path: Path):
    tree(tmp_path, {"pkg/mod/a.py": "def f():\n    return 1\n"})
    report = build_pool(tmp_path, repo="acme/widget", licence=MIT)
    record = report.records[0]
    assert record.path == "pkg/mod/a.py"
    assert record.id == "acme/widget:pkg/mod/a.py"
    assert record.repo == "acme/widget"


def test_hunks_are_absent_not_empty(tmp_path: Path):
    """`pool.rs` distinguishes "no diff attached" from "a diff touched nothing"."""
    tree(tmp_path, {"a.py": "def f():\n    return 1\n"})
    report = build_pool(tmp_path, repo="r", licence=MIT)
    assert "hunks" not in report.records[0].to_json()


# ---------------------------------------------------------------------------
# Coverage: the two numbers, never one
# ---------------------------------------------------------------------------


def test_selected_and_seen_are_both_carried(tmp_path: Path):
    tree(tmp_path, {"a.py": "def f():\n    return 1\n", "b.md": "x\n", "c.txt": "y\n"})
    report = build_pool(tmp_path, repo="r", licence=MIT)
    assert (report.n_selected, report.n_seen) == (1, 3)
    assert report.coverage() == "1/3"


def test_a_capped_walk_is_not_run_not_a_clean_pass(tmp_path: Path):
    tree(tmp_path, {f"f{i}.py": "def f():\n    return 1\n" for i in range(10)})
    report = build_pool(tmp_path, repo="r", licence=MIT, max_files=3)
    assert report.n_selected == 3
    assert isinstance(report.status, NotRun)
    assert not hasattr(report.status, "passed")
    assert "capped sample" in report.status.reason


def test_an_empty_pool_is_not_run_not_a_clean_pass(tmp_path: Path):
    tree(tmp_path, {"readme.md": "# nothing to mutate\n"})
    report = build_pool(tmp_path, repo="r", licence=MIT)
    assert report.n_selected == 0
    assert isinstance(report.status, NotRun)
    assert "empty pool" in report.status.reason


def test_a_complete_build_reports_ran_with_both_counts(tmp_path: Path):
    tree(tmp_path, {"a.py": "def f():\n    return 1\n", "b.rs": "fn g() {}\n", "c.md": "x\n"})
    report = build_pool(tmp_path, repo="r", licence=MIT)
    status = report.status
    assert isinstance(status, Ran)
    assert status.passed
    assert (status.n, status.n_total) == (2, 3)
    assert "unconstrained" in status.detail, "the hunk-free build must say what it is"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_writing_produces_jsonl_qd_mutate_can_read(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    tree(src, {"a.py": "def f():\n    return 1\n", "b.rs": "fn g() {}\n"})
    report = build_pool(src, repo="r", licence=MIT)
    out = tmp_path / "pool.jsonl"
    assert write_pool(report, out) == 2

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for r in rows:
        assert set(r) == {"id", "repo", "path", "source"}
        assert isinstance(r["source"], str) and r["source"]


def test_a_capped_build_refuses_to_be_written(tmp_path: Path):
    """On disk a capped pool is indistinguishable from a complete one."""
    src = tmp_path / "src"
    src.mkdir()
    tree(src, {f"f{i}.py": "def f():\n    return 1\n" for i in range(5)})
    report = build_pool(src, repo="r", licence=MIT, max_files=2)
    with pytest.raises(ValueError, match="did not complete"):
        write_pool(report, tmp_path / "pool.jsonl")


def test_a_missing_root_is_refused(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        build_pool(tmp_path / "nope", repo="r", licence=MIT)


# ---------------------------------------------------------------------------
# The Rust seam
# ---------------------------------------------------------------------------


def test_the_extension_map_still_matches_pool_rs():
    """The Python mirror is duplication; this is what stops it rotting."""
    src = POOL_RS.read_text(encoding="utf-8")
    body = re.search(r"pub fn language_from_path.*?\n\}", src, re.S)
    assert body, f"language_from_path not found in {POOL_RS}"
    # Arms are `"py" | "pyi" => Some(LangId::Python)`. A regex that matches only the last
    # alternative would silently compare a 5-entry map against a 9-entry one and pass on the
    # overlap, so the alternation is parsed explicitly.
    arms = re.findall(
        r'((?:"[a-z]+"\s*\|\s*)*"[a-z]+")\s*=>\s*Some\(LangId::(\w+)\)', body.group(0)
    )
    assert arms, "no extension arms parsed; the mirror check would pass vacuously"

    rust_map = {
        f".{ext}": lang
        for alternatives, lang in arms
        for ext in re.findall(r'"([a-z]+)"', alternatives)
    }
    assert len(rust_map) > len(arms), "the alternation was not expanded; this check is vacuous"
    assert rust_map == POOL_EXTENSIONS, (
        f"pool.rs maps {rust_map}, this module mirrors {POOL_EXTENSIONS}"
    )
    assert '.d.ts' in body.group(0), "the declaration-file exclusion is part of the contract"


def test_skip_dirs_holds_the_obvious_generated_trees():
    for name in ("node_modules", "target", ".git", "__pycache__", ".venv"):
        assert name in SKIP_DIRS
