"""``code.defect_class``: the licence join, the span rebase, the split, the permutation hook.

Built over a miniature corpus written to ``tmp_path`` in the exact on-disk shape qd-mutate,
the pool builder and the commitpackft download produce, so the loader is exercised through
its real sha256 checks rather than around them.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.defect_class import (
    CHOICE_SLOT,
    CONTEXT_HEADER_LINES,
    DEFECT_CLASSES,
    DEFECT_FAMILY_ID,
    DEFECT_SOURCE_ID,
    SPAN_SLOT,
    DefectCorpusError,
    DefectLoad,
    DefectRow,
    diff_line_span,
    load_defect_rows,
    second_pass_permutation,
    with_permuted_options,
)
from qd_data.errors import HeldOutViolation
from qd_data.mixture import ABSTAINING_FAMILIES, build_mixture, rewrite_defect_class
from qd_data.render import render
from qd_data.split import split
from qd_train.mutate_adapter import MUTATION_CLASSES
from qd_train.shards import training_texts
from qd_train.tristate import Ran

BEFORE = "def f(x):\n    a = 1\n    return x + a\n\n\ndef g():\n    return 0\n"
AFTER = "def f(x):\n    a = 1\n    return x - a\n\n\ndef g():\n    return 0\n"
#: after line 3 is diff line 5: the header, two context lines, the removed line, then it.
DIFF = (
    "@@ -1,3 +1,3 @@\n"
    " def f(x):\n"
    "     a = 1\n"
    "-    return x + a\n"
    "+    return x - a\n"
)
CLEAN_DIFF = "@@ -5,2 +5,2 @@\n def g():\n-    return 1\n+    return 0\n"


def _example(
    n: int,
    *,
    repo: str,
    cls: str = "logic",
    span: dict[str, int] | None = None,
    diff: str = DIFF,
) -> dict[str, Any]:
    commit = hashlib.sha1(f"{repo}-{n}".encode()).hexdigest()
    path = f"src/m{n}.py"
    pool_id = f"{commit}:{path}"
    if cls != "clean" and span is None:
        span = {"start_line": 3, "end_line": 3}
    return {
        "id": f"{pool_id}#0",
        "pool_id": pool_id,
        "repo": repo,
        "path": path,
        "language": "python",
        "class": cls,
        "operator": "clean" if cls == "clean" else f"{cls}.op",
        "silent": False,
        "span": None if cls == "clean" else span,
        "function": {"repo": repo, "path": path, "symbol": "f", "arity": 1},
        "before": BEFORE,
        "after": AFTER,
        "diff": CLEAN_DIFF if cls == "clean" else diff,
        "hunk_constrained": True,
        "seed": 0,
    }


def _write_corpus(
    root: Path,
    examples: list[dict[str, Any]],
    *,
    licences: dict[str, str] | None = None,
    drop_from_pool: frozenset[str] | set[str] = frozenset(),
    drop_from_download: frozenset[str] | set[str] = frozenset(),
    duplicate_download: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Corpus dir, pool file under ``root/data/pool`` and a download, all sha-pinned."""
    pool_dir = root / "data" / "pool"
    corpus = pool_dir / "corpus"
    download = pool_dir / "commitpackft"
    corpus.mkdir(parents=True)
    download.mkdir(parents=True)

    pool_rows = [
        {"id": e["pool_id"], "repo": e["repo"], "path": e["path"], "hunks": [],
         "source": AFTER, "prior_source": BEFORE}
        for e in examples if e["pool_id"] not in drop_from_pool
    ]
    seen: set[str] = set()
    pool_rows = [r for r in pool_rows if not (r["id"] in seen or seen.add(r["id"]))]
    pool_path = pool_dir / "pool.jsonl"
    pool_path.write_text("".join(json.dumps(r) + "\n" for r in pool_rows), encoding="utf-8")

    dl_rows = []
    for r in pool_rows:
        if r["id"] in drop_from_download:
            continue
        commit, new_file = r["id"].split(":", 1)
        lic = (licences or {}).get(r["id"], "mit")
        dl_rows.append({"commit": commit, "new_file": new_file, "license": lic})
        if duplicate_download and r["id"] in duplicate_download:
            dl_rows.append(
                {"commit": commit, "new_file": new_file,
                 "license": duplicate_download[r["id"]]}
            )
    dl_path = download / "python.jsonl"
    dl_path.write_text("".join(json.dumps(r) + "\n" for r in dl_rows), encoding="utf-8")
    (download / "manifest.json").write_text(json.dumps({
        "source_id": "bigcode/commitpackft",
        "languages": {"python": {"sha256": hashlib.sha256(dl_path.read_bytes()).hexdigest()}},
    }), encoding="utf-8")

    ex_path = corpus / "examples.jsonl"
    ex_path.write_text("".join(json.dumps(e) + "\n" for e in examples), encoding="utf-8")
    (corpus / "manifest.json").write_text(json.dumps({
        "examples_sha256": hashlib.sha256(ex_path.read_bytes()).hexdigest(),
        "pool": {
            # A foreign absolute path, as in the real manifest: re-anchored by file name.
            "path": "/elsewhere/data/pool/pool.jsonl",
            "sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
            "records": len(pool_rows),
        },
        "totals": {"examples": len(examples)},
    }), encoding="utf-8")
    return corpus, download


def _load(tmp_path: Path, corpus: Path, download: Path, **kw: Any) -> DefectLoad:
    return load_defect_rows(
        corpus, download_root=download, config=DataConfig(), repo_root=tmp_path, **kw
    )


def _mixed(n_repos: int = 12) -> list[dict[str, Any]]:
    out = []
    for i in range(n_repos):
        repo = f"org/repo{i}"
        for j, cls in enumerate(DEFECT_CLASSES):
            ex = _example(i * 10 + j, repo=repo, cls=cls)
            # Distinct diffs, so no two rows render one prompt.
            ex["diff"] = ex["diff"].replace("return", f"return {i}{j} or")
            out.append(ex)
    return out


# -- the option set --------------------------------------------------------------


def test_the_option_set_is_qd_mutates_own_label_set() -> None:
    """Restated for the import cycle, so pinned to the source it restates."""
    assert DEFECT_CLASSES == MUTATION_CLASSES


# -- the licence join --------------------------------------------------------------


def test_every_row_takes_its_licence_from_the_download_through_the_pool(tmp_path: Path) -> None:
    ex = [_example(0, repo="a/x"), _example(1, repo="b/y", cls="clean")]
    corpus, download = _write_corpus(
        tmp_path, ex, licences={ex[0]["pool_id"]: "apache-2.0", ex[1]["pool_id"]: "bsd-3-clause"}
    )
    load = _load(tmp_path, corpus, download)
    assert {r.example_id: r.licence for r in load.rows} == {
        ex[0]["id"]: "apache-2.0", ex[1]["id"]: "bsd-3-clause"
    }
    assert load.by_class == {"clean": 1, "logic": 1}
    assert load.n_corpus == 2 and not load.capped


def test_a_pool_id_the_pool_does_not_hold_refuses_the_whole_load(tmp_path: Path) -> None:
    ex = [_example(0, repo="a/x"), _example(1, repo="b/y")]
    corpus, download = _write_corpus(tmp_path, ex, drop_from_pool={ex[1]["pool_id"]})
    with pytest.raises(DefectCorpusError, match="pool_id that is not a pool id"):
        _load(tmp_path, corpus, download)


def test_a_pool_id_with_no_download_row_has_no_licence_and_refuses(tmp_path: Path) -> None:
    ex = [_example(0, repo="a/x"), _example(1, repo="b/y")]
    corpus, download = _write_corpus(tmp_path, ex, drop_from_download={ex[0]["pool_id"]})
    with pytest.raises(DefectCorpusError, match="no licence"):
        _load(tmp_path, corpus, download)


def test_a_download_key_with_two_licences_is_ambiguous_and_refused(tmp_path: Path) -> None:
    ex = [_example(0, repo="a/x")]
    corpus, download = _write_corpus(
        tmp_path, ex, duplicate_download={ex[0]["pool_id"]: "agpl-3.0"}
    )
    with pytest.raises(DefectCorpusError, match="cannot be decided"):
        _load(tmp_path, corpus, download)


def test_a_corpus_that_changed_after_its_manifest_is_refused(tmp_path: Path) -> None:
    corpus, download = _write_corpus(tmp_path, [_example(0, repo="a/x")])
    with (corpus / "examples.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_example(1, repo="a/x")) + "\n")
    with pytest.raises(DefectCorpusError, match="changed after it was recorded"):
        _load(tmp_path, corpus, download)


def test_a_repo_that_disagrees_with_its_pool_row_is_refused(tmp_path: Path) -> None:
    ex = _example(0, repo="a/x")
    corpus, download = _write_corpus(tmp_path, [ex])
    lines = (corpus / "examples.jsonl").read_text(encoding="utf-8")
    bad = lines.replace('"repo": "a/x", "path"', '"repo": "evil/fork", "path"', 1)
    (corpus / "examples.jsonl").write_text(bad, encoding="utf-8")
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    manifest["examples_sha256"] = hashlib.sha256(bad.encode()).hexdigest()
    (corpus / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DefectCorpusError, match="disagrees with its pool row"):
        _load(tmp_path, corpus, download)


def test_a_held_out_corpus_is_refused_before_it_is_read(tmp_path: Path) -> None:
    """Rule 3's door: the intent to read is checked, whether or not the file exists."""
    with pytest.raises(HeldOutViolation):
        load_defect_rows(
            tmp_path / "data" / "heldout" / "corpus",
            download_root=tmp_path / "data" / "pool" / "commitpackft",
            config=DataConfig(), repo_root=tmp_path,
        )


def test_a_cap_is_a_sha256_ordered_sample_and_says_so(tmp_path: Path) -> None:
    corpus, download = _write_corpus(tmp_path, _mixed(3))
    load = _load(tmp_path, corpus, download, max_rows=5)
    assert load.capped and load.n_corpus == 12 and len(load.rows) == 5
    order = sorted(e["id"] for e in _mixed(3))
    expected = sorted(order, key=lambda i: hashlib.sha256(i.encode()).hexdigest())[:5]
    assert [r.example_id for r in load.rows] == expected


def test_a_non_admitted_licence_is_counted_per_row_not_dropped(tmp_path: Path) -> None:
    ex = [_example(0, repo="a/x"), _example(1, repo="b/y")]
    corpus, download = _write_corpus(tmp_path, ex, licences={ex[1]["pool_id"]: "agpl-3.0"})
    load = _load(tmp_path, corpus, download)
    mix = build_mixture({DEFECT_SOURCE_ID: load.rows}, config=DataConfig(),
                        families=[DEFECT_FAMILY_ID])
    assert len(mix.rows) == 1
    assert mix.refusals[DEFECT_SOURCE_ID] == {"licence:agpl-3.0": 1}


# -- the span rebase -----------------------------------------------------------------


def test_the_span_is_rebased_from_after_lines_to_diff_lines() -> None:
    assert diff_line_span(DIFF.encode(), start_line=3, end_line=3) == (5, 5)
    # A two-line span resolves both ends independently.
    assert diff_line_span(DIFF.encode(), start_line=2, end_line=3) == (3, 5)


def test_the_span_gold_names_the_mutated_line_of_the_rendered_context(tmp_path: Path) -> None:
    corpus, download = _write_corpus(tmp_path, [_example(0, repo="a/x")])
    (row,) = _load(tmp_path, corpus, download).rows
    data_row = rewrite_defect_class(row, family_id=DEFECT_FAMILY_ID, index=0, config=DataConfig())
    span = next(g for g in data_row.gold if g.slot_name == SPAN_SLOT)
    assert span.value == (5 + CONTEXT_HEADER_LINES, 5 + CONTEXT_HEADER_LINES)
    lines = data_row.request.context.decode().split("\n")
    assert lines[span.value[0] - 1] == "+    return x - a"
    # And the shard path accepts it: the gold lands on a candidate line start.
    specs = training_texts(data_row, seed=DataConfig().seed)
    assert [s.slot_name for s in specs] == [CHOICE_SLOT, SPAN_SLOT]


def test_clean_is_a_class_on_the_choice_slot_and_an_abstention_on_the_span(
    tmp_path: Path,
) -> None:
    corpus, download = _write_corpus(tmp_path, [_example(0, repo="a/x", cls="clean")])
    (row,) = _load(tmp_path, corpus, download).rows
    data_row = rewrite_defect_class(row, family_id=DEFECT_FAMILY_ID, index=0, config=DataConfig())
    gold = {g.slot_name: g for g in data_row.gold}
    assert gold[CHOICE_SLOT].value == "clean" and not gold[CHOICE_SLOT].is_noul
    assert gold[SPAN_SLOT].is_noul and gold[SPAN_SLOT].value is None


def test_a_span_the_diff_does_not_represent_is_refused_and_counted(tmp_path: Path) -> None:
    far = _example(0, repo="a/x", span={"start_line": 7, "end_line": 7})
    corpus, download = _write_corpus(tmp_path, [far, _example(1, repo="b/y")])
    load = _load(tmp_path, corpus, download)
    assert load.span_refusals == {"defect_span_outside_diff": 1}
    mix = build_mixture({DEFECT_SOURCE_ID: load.rows}, config=DataConfig(),
                        families=[DEFECT_FAMILY_ID])
    assert len(mix.rows) == 1
    assert mix.refusals[DEFECT_SOURCE_ID] == {"defect_span_outside_diff": 1}


def test_an_unlocated_mutation_cannot_become_a_silent_noul() -> None:
    with pytest.raises(DefectCorpusError, match="silent noul"):
        DefectRow(
            example_id="e", pool_id="p", repo="r", path="x.py", symbol="f", arity=0,
            language="python", mutation_class="stub", operator="stub.x", diff=DIFF,
            diff_span=None, span_refusal=None, licence="mit",
        )


# -- the mixture, the split, abstention -------------------------------------------------


def test_the_family_builds_splits_by_repo_and_teaches_span_abstention(tmp_path: Path) -> None:
    corpus, download = _write_corpus(tmp_path, _mixed(40))
    load = _load(tmp_path, corpus, download)
    config = DataConfig()
    mix = build_mixture({DEFECT_SOURCE_ID: load.rows}, config=config,
                        families=[DEFECT_FAMILY_ID])
    assert isinstance(mix.status, Ran) and mix.status.passed, mix.status
    assert all(r.metadata["language"] == "python" for r in mix.rows)

    span = mix.abstention["span"]
    assert isinstance(span, Ran) and span.passed and span.value == 40
    choice = mix.abstention["choice"]
    # Per channel: the family abstains on its span, never on its class.
    assert isinstance(choice, Ran) and not choice.passed
    assert "able to abstain at all: none" in choice.detail
    assert ABSTAINING_FAMILIES[DEFECT_FAMILY_ID] == frozenset({"span"})

    report = split(dedupe(list(mix.rows), config=config), config=config)
    assert report.repo_disjoint.passed and report.identity_disjoint.passed  # type: ignore[union-attr]
    where: dict[str, set[str]] = {}
    for a in report.assignments:
        where.setdefault(a.repo_key, set()).add(a.split)
    assert all(len(s) == 1 for s in where.values())
    assert not config.is_held_out_family(DEFECT_FAMILY_ID)
    assert Counter(a.reason for a in report.assignments) == {"repo_hash": len(mix.rows)}


# -- the second-pass permutation ---------------------------------------------------------


def test_the_second_pass_is_a_derangement_and_reaches_every_cycle() -> None:
    seen = set()
    for i in range(600):
        perm = second_pass_permutation(4, seed=1, example_id=f"e{i}", slot_name=CHOICE_SLOT)
        assert sorted(perm) == [0, 1, 2, 3]
        assert all(perm[k] != k for k in range(4)), perm
        seen.add(perm)
    assert len(seen) == 6  # (4 - 1)! cyclic permutations; Fisher-Yates would give 24
    assert second_pass_permutation(4, seed=1, example_id="e", slot_name="s") == (
        second_pass_permutation(4, seed=1, example_id="e", slot_name="s")
    )
    with pytest.raises(ValueError, match="at least 2"):
        second_pass_permutation(1, seed=1, example_id="e", slot_name="s")


def test_permuted_options_move_every_option_and_keep_the_gold(tmp_path: Path) -> None:
    corpus, download = _write_corpus(tmp_path, [_example(0, repo="a/x")])
    (row,) = _load(tmp_path, corpus, download).rows
    data_row = rewrite_defect_class(row, family_id=DEFECT_FAMILY_ID, index=0, config=DataConfig())
    perm = second_pass_permutation(
        4, seed=7, example_id=data_row.request.example_id, slot_name=CHOICE_SLOT
    )
    moved = with_permuted_options(data_row.request, slot_name=CHOICE_SLOT, permutation=perm)
    before = data_row.request.slots[0].options  # type: ignore[union-attr]
    after = moved.slots[0].options  # type: ignore[union-attr]
    assert sorted(after) == sorted(before)
    assert all(a != b for a, b in zip(after, before, strict=True))
    assert moved.slots[1] == data_row.request.slots[1]
    # The gold is by value, so it is still an option of the permuted prompt.
    assert "logic" in after
    assert render(moved, seed=None).prefix == render(data_row.request, seed=None).prefix
    with pytest.raises(ValueError, match="not a permutation"):
        with_permuted_options(data_row.request, slot_name=CHOICE_SLOT, permutation=(5, 6, 7, 8))
    with pytest.raises(ValueError, match="not a choice slot"):
        with_permuted_options(data_row.request, slot_name=SPAN_SLOT, permutation=(0,))
