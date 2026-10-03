"""v5's noul routes in ``qd_data``: the one prose builder, the own-prose and G6 parts, and the
composite corpus (defect-noul-v3c) that unions them with v3b.

* ``prose_defect_row`` is THE "prose text -> ``code.defect_class`` gold-noul row" builder: the
  own-prose loader and ``qd_train.contrast`` both call it, and ``rewrite_defect_class`` renders
  it. Its rows are the text under ``file: <path>``, gold noul on both slots, pinned to train.
* A routed row is pinned to train whatever its unit hashes to; a row v4 read is rendered and
  hashed exactly as before.
* The own-prose part (``qd-noul-rows own-prose``) and the G6 part (``qd-noul-rows g6``) are
  re-checked row by row; one bad row refuses the part.
* G6's languages are disjoint from the pool, the OOD suite, the needle suite and v3b's
  templates, read off each one's own constants.

Every test fails on the pre-change code: none of these names exist there.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.defect_class import (
    CHOICE_SLOT,
    DEFECT_FAMILY_ID,
    G6_FORM,
    G6_LANGUAGES,
    G6_SCHEMA,
    NOUL_COMPOSITE_SCHEMA,
    NOUL_FORM_KEY,
    NOUL_ROUTE_G6,
    NOUL_ROUTE_KEY,
    NOUL_ROUTE_OWN_PROSE,
    OWN_PROSE_MIN_ASCII_RATIO,
    OWN_PROSE_MIN_PARAGRAPH_WORDS,
    OWN_PROSE_QUESTION_MARKUP,
    OWN_PROSE_QUESTION_WORDS,
    OWN_PROSE_SCHEMA,
    SPAN_SLOT,
    DefectCorpusError,
    DefectRow,
    _load_noul_corpus,
    composite_digest,
    prose_defect_row,
)
from qd_data.licences import LicenceTier, classify
from qd_data.mixture import rewrite_defect_class
from qd_data.render import INVISIBLE_FORMAT_RANGES
from qd_data.sources import OWN_REPOS_SOURCE_ID, PINNED_SPLIT_KEY, source_by_id
from qd_data.split import assign_repo, split
from qd_train import ood
from qd_train.needle import _EXTENSIONS as NEEDLE_EXTENSIONS

CONFIG = DataConfig()
REPO = Path(__file__).resolve().parents[2]

PARA = (
    "This repository keeps a small ledger of every experiment the author ran, with the reason "
    "each was started and what it measured, so that a later reader can tell which numbers are "
    "comparable and which were taken under different settings."
)
QUESTION = "Why does the scheduler retry a failed upload before it reports the error?"


def _split_of(key: str) -> str:
    return assign_repo(
        key,
        seed=CONFIG.seed,
        train_fraction=CONFIG.train_fraction,
        val_fraction=CONFIG.val_fraction,
    )


def _key_in(split_name: str, prefix: str) -> str:
    return next(f"{prefix}{k}" for k in range(10_000) if _split_of(f"{prefix}{k}") == split_name)


def _render(raw: DefectRow):
    return rewrite_defect_class(raw, family_id=DEFECT_FAMILY_ID, index=0, config=CONFIG)


# -- the one prose builder -------------------------------------------------------------------


def test_a_prose_row_is_the_text_under_its_path_with_gold_noul_pinned_to_train() -> None:
    row = _render(
        prose_defect_row(
            PARA,
            route=NOUL_ROUTE_OWN_PROSE,
            form="paragraph",
            example_id="own-prose:u1",
            repo="own-prose:apps/Example",
            path="docs/design.md",
            licence="owner-granted",
        )
    )
    assert row.request.context.decode("utf-8") == f"file: docs/design.md\n\n{PARA}"
    gold = {g.slot_name: g for g in row.gold}
    assert gold[CHOICE_SLOT].is_noul and gold[SPAN_SLOT].is_noul
    assert row.metadata[NOUL_ROUTE_KEY] == NOUL_ROUTE_OWN_PROSE
    assert row.metadata[NOUL_FORM_KEY] == "paragraph"
    assert row.metadata[PINNED_SPLIT_KEY] == "train"
    assert row.licence_id == "owner-granted"
    assert row.identity_key == "own-prose:apps/Example::docs/design.md::own-prose:u1/0"


def test_a_routed_row_lands_in_train_even_when_its_unit_hashes_elsewhere() -> None:
    texts = {"val": PARA, "heldout": QUESTION}  # distinct, so dedupe keeps both
    rows = [
        _render(
            prose_defect_row(
                texts[name],
                route=NOUL_ROUTE_OWN_PROSE,
                form="paragraph",
                example_id=f"own-prose:{name}",
                repo=_key_in(name, "own-prose:repo-"),
                path="README.md",
                licence="owner-granted",
            )
        )
        for name in ("val", "heldout")
    ]
    assert {_split_of(r.repo_key) for r in rows} == {"val", "heldout"}
    report = split(dedupe(rows, config=CONFIG), config=CONFIG)
    assert {r.row_id for r in report.rows_by_split["train"]} == {r.row_id for r in rows}
    assert not report.rows_by_split["val"] and not report.rows_by_split["heldout"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"path": "notes.txt"}, "OOD suite's prose path"),
        ({"path": "docs/notes.txt"}, "OOD suite's prose path"),
        ({"path": "a\nb.md"}, "not one line"),
        ({"route": NOUL_ROUTE_G6}, "not a prose route"),
        ({"form": "commit"}, "prose form"),
        ({"text": "   "}, "empty text"),
    ],
)
def test_the_prose_builder_refuses_what_is_not_a_prose_row(kwargs: dict, match: str) -> None:
    base = {
        "text": PARA,
        "route": NOUL_ROUTE_OWN_PROSE,
        "form": "paragraph",
        "example_id": "own-prose:x",
        "repo": "own-prose:r",
        "path": "README.md",
        "licence": "owner-granted",
    }
    base.update(kwargs)
    text = base.pop("text")
    with pytest.raises(DefectCorpusError, match=match):
        prose_defect_row(text, **base)


def test_a_row_v4_read_renders_and_hashes_exactly_as_before() -> None:
    """Characterization at 541d423: a v3b template row's DataRow, field by field."""
    raw = DefectRow(
        example_id="noul:unseen-language:0123456789abcdef",
        pool_id="noul-template/kotlin/x",
        repo="noul-template/kotlin/x",
        path="app/X.kt",
        symbol="noul:unseen-language:0123",
        arity=0,
        language="kotlin",
        mutation_class="noul",
        operator="noul.unseen-language",
        diff="@@ -1,2 +1,2 @@\n-a\n+b\n",
        diff_span=None,
        span_refusal=None,
        licence="apache-2.0",
        noul_source="unseen-language",
    )
    row = _render(raw)
    assert row.identity_key == "noul-template/kotlin/x::app/X.kt::noul:unseen-language:0123/0"
    assert row.metadata == {
        "language": "kotlin",
        "operator": "noul.unseen-language",
        "pool_id": "noul-template/kotlin/x",
        "noul_source": "unseen-language",
    }


def test_a_route_field_without_a_route_or_on_a_class_row_is_refused() -> None:
    base = {
        "example_id": "e",
        "pool_id": "p",
        "repo": "r",
        "path": "a.go",
        "symbol": "s",
        "arity": 0,
        "language": "go",
        "diff": "@@ -1 +1 @@\n",
        "diff_span": None,
        "span_refusal": None,
        "licence": "mit",
    }
    with pytest.raises(DefectCorpusError, match="without a noul_route"):
        DefectRow(
            **base,
            mutation_class="noul",
            operator="o",
            noul_source="prose",
            identity_key="contrast:x",
        )
    with pytest.raises(DefectCorpusError, match="only a 'noul' row has one"):
        DefectRow(**base, mutation_class="clean", operator="o", noul_route=NOUL_ROUTE_G6)


def test_owner_granted_is_admitted_and_the_own_repos_source_is_registered() -> None:
    assert classify("owner-granted").tier is LicenceTier.ALLOW
    source = source_by_id(OWN_REPOS_SOURCE_ID)
    assert source.declared_licence == "owner-granted"
    assert source.admission_refusals(CONFIG.licence) == ()
    assert source.benchmark_reportable is False


# -- the parts and the composite ------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _own_prose_part(root: Path, units: list[dict], **manifest_over: object) -> Path:
    d = root / "own-prose-v1"
    d.mkdir(parents=True)
    files = sorted({(u["repo"], u["path"], u["file_sha256"]) for u in units})
    _write_jsonl(
        d / "files.jsonl", [{"repo": r, "path": p, "sha256": s, "words": 100} for r, p, s in files]
    )
    _write_jsonl(d / "units.jsonl", units)
    manifest = {
        "schema": OWN_PROSE_SCHEMA,
        "source_id": OWN_REPOS_SOURCE_ID,
        "licence": "owner-granted",
        "invisible_format_ranges": [list(r) for r in INVISIBLE_FORMAT_RANGES],
        "admitted_repos": [{"repo": "apps/Example"}, {"repo": "devtools/Other"}],
        "files": {"sha256": _sha(d / "files.jsonl"), "count": len(files)},
        "units": {"sha256": _sha(d / "units.jsonl"), "count": len(units)},
    }
    manifest.update(manifest_over)
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def _unit(i: int, **over: object) -> dict:
    u = {
        "id": f"u{i:04d}",
        "repo": "apps/Example",
        "path": "README.md",
        "file_sha256": "f" * 64,
        "form": "paragraph",
        "text": f"{PARA} Paragraph {i}.",
    }
    u.update(over)
    return u


def _g6_row(i: int, **over: object) -> dict:
    diff = "@@ -1,3 +1,3 @@\n (defn f [x]\n-  (inc x))\n+  (dec x))\n" + ";" * (i % 7) + "\n"
    r = {
        "id": f"noul:unseen-language:g6{i:012d}",
        "class": "noul",
        "noul_source": "unseen-language",
        "noul_route": NOUL_ROUTE_G6,
        "noul_form": G6_FORM,
        "repo": _key_in("train", f"someone/clj-{i}-"),
        "path": f"src/core_{i}.clj",
        "language": "clojure",
        "diff": diff,
        "licence": "mit",
        "commit": f"{i:040x}",
    }
    r.update(over)
    return r


def _g6_part(root: Path, rows: list[dict], cap: int = 250) -> Path:
    d = root / "commitpackft-g6-v1"
    d.mkdir(parents=True)
    _write_jsonl(d / "examples.jsonl", rows)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "schema": G6_SCHEMA,
                "examples_sha256": _sha(d / "examples.jsonl"),
                "split": {
                    "seed": CONFIG.seed,
                    "train_fraction": CONFIG.train_fraction,
                    "val_fraction": CONFIG.val_fraction,
                },
                "per_language_cap": cap,
                "diff_band": {"min_chars": 10, "max_chars": 4000, "max_hunks": 5},
                "totals": {"examples": len(rows)},
            }
        ),
        encoding="utf-8",
    )
    return d


def _composite(root: Path, parts: list[tuple[str, str, str, int]], **over: object) -> Path:
    d = root / "defect-noul-v3c"
    d.mkdir()
    entries = [
        {
            "name": name,
            "kind": kind,
            "manifest_sha256": _sha(root / name / "manifest.json"),
            "data_sha256": _sha(root / name / data),
            "rows": n,
        }
        for name, kind, data, n in parts
    ]
    manifest = {
        "schema": NOUL_COMPOSITE_SCHEMA,
        "parts": entries,
        "examples_sha256": composite_digest(entries),
        "totals": {"examples": sum(p[3] for p in parts)},
    }
    manifest.update(over)
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def _load(noul_dir: Path):
    return _load_noul_corpus(noul_dir, config=CONFIG, repo_root=REPO, licences={}, pools={})


def test_a_composite_loads_its_parts_in_order_with_their_routes(tmp_path: Path) -> None:
    units = [_unit(0), _unit(1, form="question", text=QUESTION, path="docs/faq.md")]
    _own_prose_part(tmp_path, units)
    _g6_part(tmp_path, [_g6_row(0), _g6_row(1)])
    noul = _composite(
        tmp_path,
        [
            ("own-prose-v1", "own-prose", "units.jsonl", 2),
            ("commitpackft-g6-v1", "g6", "examples.jsonl", 2),
        ],
    )
    load = _load(noul)
    assert [r.noul_route for r in load.rows] == [NOUL_ROUTE_OWN_PROSE] * 2 + [NOUL_ROUTE_G6] * 2
    assert load.by_route == {NOUL_ROUTE_OWN_PROSE: 2, NOUL_ROUTE_G6: 2}
    assert load.by_source == {"prose": 2, "unseen-language": 2}
    own = load.rows[0]
    assert own.repo == "own-prose:apps/Example" and own.licence == "owner-granted"
    g6 = _render(load.rows[2])
    assert g6.metadata[PINNED_SPLIT_KEY] == "train" and g6.metadata[NOUL_FORM_KEY] == G6_FORM


def test_a_composite_whose_part_bytes_moved_is_refused(tmp_path: Path) -> None:
    _own_prose_part(tmp_path, [_unit(0)])
    noul = _composite(tmp_path, [("own-prose-v1", "own-prose", "units.jsonl", 1)])
    with (tmp_path / "own-prose-v1" / "units.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(DefectCorpusError, match="changed after it was recorded"):
        _load(noul)


@pytest.mark.parametrize(
    ("unit", "reason"),
    [
        (_unit(9, text="Too short to be a paragraph here."), "paragraph_too_short"),
        (_unit(9, form="question", text="Is this a question with words?"), "question_out_of_band"),
        (_unit(9, form="question", text=PARA), "question_out_of_band"),
        (
            _unit(9, form="question", text="trained accuracy under a hard winner evaluation?"),
            "question_not_a_sentence_start",
        ),
        (
            _unit(9, form="question", text="It asks: **how much did the attention model lose?"),
            "question_has_markup",
        ),
        (
            _unit(9, form="question", text="Which gate moved the `POST /synthesis` route to Go?"),
            "question_has_markup",
        ),
        (
            _unit(
                9, form="question", text='Users ask questions like "What should I work on today?'
            ),
            "question_unbalanced",
        ),
        (_unit(9, path="notes.txt"), "path_refused"),
        (_unit(9, repo="research/Lappi-decision"), "repo_not_admitted"),
        (_unit(9, text=PARA + " " + "é" * 400), "ascii_ratio_below_min"),
        (_unit(9, text=PARA + "\nsecond line"), "text_is_not_one_line"),
    ],
)
def test_one_own_prose_unit_that_breaks_a_rule_refuses_the_part(
    tmp_path: Path, unit: dict, reason: str
) -> None:
    _own_prose_part(tmp_path, [_unit(0), unit])
    noul = _composite(tmp_path, [("own-prose-v1", "own-prose", "units.jsonl", 2)])
    with pytest.raises(DefectCorpusError, match=reason):
        _load(noul)


STRIKE = {
    "basis": "the human's answer: Strike personal + business (Recommended)",
    "repos": ["devtools/Other"],
    "paths": [{"repo": "apps/Example", "prefix": "docs/business/"}],
    "files_struck": {"repo:devtools/Other": 2, "path:apps/Example:docs/business/": 3},
}


def test_a_strike_that_covers_nothing_loaded_loads(tmp_path: Path) -> None:
    units = [_unit(0), _unit(1, path="docs/business-plan.md"), _unit(2, path="docs/other/x.md")]
    _own_prose_part(tmp_path, units, strike=STRIKE)
    noul = _composite(tmp_path, [("own-prose-v1", "own-prose", "units.jsonl", 3)])
    assert len(_load(noul).rows) == 3


@pytest.mark.parametrize(
    "unit",
    [
        _unit(9, repo="devtools/Other"),
        _unit(9, path="docs/business/yc.md"),
        _unit(9, path="docs/business/archive/pitch.md"),
    ],
)
def test_a_unit_or_file_the_strike_covers_refuses_the_part(tmp_path: Path, unit: dict) -> None:
    _own_prose_part(tmp_path, [_unit(0), unit], strike=STRIKE)
    noul = _composite(tmp_path, [("own-prose-v1", "own-prose", "units.jsonl", 2)])
    # files.jsonl carries the unit's file, so the file check fires before the unit check.
    with pytest.raises(DefectCorpusError, match="the manifest's strike covers"):
        _load(noul)


def test_a_struck_unit_is_refused_even_when_files_jsonl_omits_its_file(tmp_path: Path) -> None:
    d = _own_prose_part(tmp_path, [_unit(0)], strike=STRIKE)
    struck = _unit(9, path="docs/business/yc.md")
    with (d / "units.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(struck) + "\n")
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    manifest["units"] = {"sha256": _sha(d / "units.jsonl"), "count": 2}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    noul = _composite(tmp_path, [("own-prose-v1", "own-prose", "units.jsonl", 2)])
    with pytest.raises(DefectCorpusError, match="struck_by_the_human"):
        _load(noul)


@pytest.mark.parametrize(
    ("strike", "match"),
    [
        ({**STRIKE, "basis": " "}, "without its basis"),
        ({**STRIKE, "repos": ["web/Unknown"]}, "not admitted"),
        ({**STRIKE, "repos": [], "paths": []}, "names no rule"),
        ({**STRIKE, "paths": [{"repo": "apps/Example", "prefix": "docs/business"}]}, "directory"),
        ({**STRIKE, "paths": [{"repo": "apps/Example", "prefix": "../x/"}]}, "directory"),
        ({"repos": ["devtools/Other"], "paths": []}, "must carry basis"),
    ],
)
def test_a_malformed_strike_refuses_the_part(tmp_path: Path, strike: dict, match: str) -> None:
    _own_prose_part(tmp_path, [_unit(0)], strike=strike)
    noul = _composite(tmp_path, [("own-prose-v1", "own-prose", "units.jsonl", 1)])
    with pytest.raises(DefectCorpusError, match=match):
        _load(noul)


def test_own_prose_is_capped_per_repository(tmp_path: Path) -> None:
    _own_prose_part(tmp_path, [_unit(i) for i in range(151)])
    noul = _composite(tmp_path, [("own-prose-v1", "own-prose", "units.jsonl", 151)])
    with pytest.raises(DefectCorpusError, match="over_per_repo_cap"):
        _load(noul)


def test_own_prose_walked_under_other_invisible_ranges_is_refused(tmp_path: Path) -> None:
    _own_prose_part(tmp_path, [_unit(0)], invisible_format_ranges=[[0x200B, 0x200B]])
    noul = _composite(tmp_path, [("own-prose-v1", "own-prose", "units.jsonl", 1)])
    with pytest.raises(DefectCorpusError, match="invisible-format ranges"):
        _load(noul)


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"repo": _key_in("val", "someone/clj-val-")}, "split_unit_in_val"),
        ({"repo": _key_in("heldout", "someone/clj-ho-")}, "split_unit_in_heldout"),
        ({"licence": "agpl-3.0"}, "licence_not_admitted"),
        ({"licence": "epl-1.0"}, "licence_not_admitted"),
        ({"language": "python"}, "language_not_g6"),
        ({"repo": "a/b,c/d"}, "repo_is_not_a_primary_repo"),
        ({"diff": "not a hunk"}, "diff_is_not_a_hunk"),
        ({"noul_route": None}, "not_a_g6_row"),
    ],
)
def test_one_g6_row_that_breaks_a_rule_refuses_the_part(
    tmp_path: Path, over: dict, reason: str
) -> None:
    _g6_part(tmp_path, [_g6_row(0), _g6_row(1, **over)])
    noul = _composite(tmp_path, [("commitpackft-g6-v1", "g6", "examples.jsonl", 2)])
    with pytest.raises(DefectCorpusError, match=reason):
        _load(noul)


def test_g6_is_capped_per_language(tmp_path: Path) -> None:
    _g6_part(tmp_path, [_g6_row(i) for i in range(3)], cap=2)
    noul = _composite(tmp_path, [("commitpackft-g6-v1", "g6", "examples.jsonl", 3)])
    with pytest.raises(DefectCorpusError, match="over_per_language_cap"):
        _load(noul)


def test_g6_languages_are_in_no_suite_and_no_other_training_source() -> None:
    v3b = json.loads((REPO / "data/pool/defect-noul-v3b/manifest.json").read_text())
    templates = set(v3b["totals"]["by_language"]) - {"prose", "go", "python", "rust", "typescript"}
    assert templates == {"csharp", "elixir", "kotlin", "php", "scala", "shell"}
    pool = set(ood._EXT)
    suites = set(ood._UNSEEN) | set(NEEDLE_EXTENSIONS)
    assert len(G6_LANGUAGES) == 8
    assert not set(G6_LANGUAGES) & (pool | suites | templates)


def _rust_const(source: str, name: str) -> str:
    m = re.search(rf"pub const {name}: [^=]+= (.*?);\n", source, re.DOTALL)
    assert m is not None, f"no pub const {name}"
    return m.group(1)


def _rust_strs(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r'"((?:[^"\\]|\\.)*)"', value))


def test_the_walkers_rules_are_the_loaders_rules() -> None:
    """The Rust generators write rows by these rules and the loader re-checks them; a rule
    changed on one side only would refuse every real corpus or admit what the walker drops."""
    own = (REPO / "crates/qd-mutate/src/noul_rows/own_prose.rs").read_text(encoding="utf-8")
    g6 = (REPO / "crates/qd-mutate/src/noul_rows/g6.rs").read_text(encoding="utf-8")
    assert _rust_strs(_rust_const(own, "SCHEMA")) == (OWN_PROSE_SCHEMA,)
    assert _rust_strs(_rust_const(own, "SOURCE_ID")) == (OWN_REPOS_SOURCE_ID,)
    assert int(_rust_const(own, "MIN_PARAGRAPH_WORDS")) == OWN_PROSE_MIN_PARAGRAPH_WORDS
    lo, hi = (int(x) for x in _rust_const(own, "QUESTION_WORDS").strip("()").split(","))
    assert (lo, hi) == OWN_PROSE_QUESTION_WORDS
    assert int(_rust_const(own, "MIN_ASCII_PERMILLE")) == round(OWN_PROSE_MIN_ASCII_RATIO * 1000)
    assert _rust_strs(_rust_const(own, "QUESTION_MARKUP")) == OWN_PROSE_QUESTION_MARKUP
    assert _rust_strs(_rust_const(g6, "SCHEMA")) == (G6_SCHEMA,)
    assert _rust_strs(_rust_const(g6, "LANGUAGES")) == G6_LANGUAGES
    assert _rust_strs(_rust_const(g6, "FORM")) == (G6_FORM,)
    assert _rust_strs(_rust_const(g6, "ROUTE")) == (NOUL_ROUTE_G6,)
