"""``code.defect_class`` rows whose gold is ``noul``: the R2 corpus, end to end.

The OOD gate's probe (2026-09-30) abstained on 31/60 prose, 1/60 scrambled and 0/60
unseen-language cases: CLINC's out-of-scope rows did not teach abstention on code-shaped
inputs. ``crates/qd-mutate``'s ``qd-noul-rows`` writes defect-class rows whose contexts are
not the model's kind, and ``qd_data.defect_class.load_noul_rows`` admits them. These tests
build a miniature world in ``tmp_path`` -- a SQuAD file, a commitpackft pool and download --
whose titles and repos are chosen BY THE CANONICAL SPLIT to fall on every side of it, run
the real binary over it, and check:

* rule 3: no held-out SQuAD title, no val or held-out repo or template unit reaches a row --
  not in the allowlist, not in the generated rows, and not past the loader if forced in;
* the loader fails closed on a bad licence or an unresolvable pool id, and the main
  corpus's commitpackft join still does;
* the render: the choice gold is the noul letter, the span slot abstains;
* disjointness from the OOD suite (``qd_train.ood``): no row shares >= 0.5 of its word
  8-grams with any suite case in either direction, by ``qd_train.replay.decontaminate`` --
  the repository's own containment check -- and no template language or identifier is the
  suite's.

The real corpus (``data/pool/defect-noul-v1``) is checked the same way where it exists.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

from qd_data.config import DataConfig  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.defect_class import (  # noqa: E402
    CHOICE_SLOT,
    DEFECT_FAMILY_ID,
    DEFECT_SOURCE_ID,
    NOUL_CLASS,
    NOUL_PROSE,
    NOUL_SCRAMBLED,
    NOUL_SOURCES,
    NOUL_TEMPLATE_UNIT_PREFIX,
    NOUL_UNSEEN_LANGUAGE,
    SPAN_SLOT,
    DefectCorpusError,
    DefectRow,
    load_defect_rows,
    load_noul_rows,
    noul_allowlist,
)
from qd_data.mixture import build_mixture, rewrite_defect_class  # noqa: E402
from qd_data.schema import NOUL_LETTER  # noqa: E402
from qd_data.split import (  # noqa: E402
    assign_repo,
    split,
    squad_title_family,
    squad_title_repo_key,
)
from qd_train import ood  # noqa: E402
from qd_train.replay import decontaminate, word_ngrams  # noqa: E402
from qd_train.shards import training_texts  # noqa: E402
from qd_train.tristate import Ran  # noqa: E402

CONFIG = DataConfig()
PER_SOURCE = 6
SUITE_SEEDS = (CONFIG.seed, 0, 1, 2, 3)

REAL_NOUL = REPO / "data" / "pool" / "defect-noul-v1"
#: v3.1's re-form (``qd-noul-rows generate --forms v2``): same share and thirds, wider forms.
REAL_NOUL_V2 = REPO / "data" / "pool" / "defect-noul-v2"
#: v2's forms and their exact counts at 834 rows per source.
V2_FORMS = {
    (NOUL_PROSE, "paragraph"): 417, (NOUL_PROSE, "question"): 417,
    (NOUL_SCRAMBLED, "lines"): 417, (NOUL_SCRAMBLED, "tokens"): 417,
    (NOUL_UNSEEN_LANGUAGE, "template"): 834,
}
REAL_DEFECT = REPO / "data" / "pool" / "commitpackft-corpus-v2"
#: The corpus v2's scrambled rows are made from (and the v3.1 shards' defect corpus).
REAL_DEFECT_V3 = REPO / "data" / "pool" / "commitpackft-corpus-v3"
REAL_DOWNLOAD = REPO / "data" / "pool" / "commitpackft"
GENERAL_RECORD = Path.home() / ".cache/qd-decision/general/fetch-record-2026-09-29.json"


def _split(unit: str) -> str:
    return assign_repo(unit, seed=CONFIG.seed, train_fraction=CONFIG.train_fraction,
                       val_fraction=CONFIG.val_fraction)


def _find(candidates: Any, want: Any, n: int) -> list[str]:
    out = [c for c in candidates if want(c)][:n]
    assert len(out) == n, f"found {len(out)} of {n}"
    return out


def _title_kind(title: str) -> str:
    """``family:split`` under the canonical functions, e.g. ``qa.answer_span:train``."""
    family = squad_title_family(title, seed=CONFIG.seed)
    return f"{family}:{_split(squad_title_repo_key(title))}"


_ALL_TITLES = [f"Fixture Article {i}" for i in range(5000)]
TRAIN_TITLES = _find(_ALL_TITLES, lambda t: _title_kind(t) == "qa.answer_span:train", 4)
#: A held-out-FAMILY title whose unit would otherwise hash to train: only the family check
#: can keep it out, so it tests that check and not the split's.
ANSWERABILITY_TITLE = _find(
    _ALL_TITLES, lambda t: _title_kind(t) == "qa.answerability:train", 1
)[0]
VAL_TITLE = _find(_ALL_TITLES, lambda t: _title_kind(t) == "qa.answer_span:val", 1)[0]
HELDOUT_TITLE = _find(_ALL_TITLES, lambda t: _title_kind(t) == "qa.answer_span:heldout", 1)[0]

_ALL_REPOS = [f"fixture-org/repo-{i}" for i in range(5000)]
TRAIN_REPOS = _find(_ALL_REPOS, lambda r: _split(r) == "train", 12)
VAL_REPO = _find(_ALL_REPOS, lambda r: _split(r) == "val", 1)[0]
HELDOUT_REPO = _find(_ALL_REPOS, lambda r: _split(r) == "heldout", 1)[0]

#: Words that may appear only in content the rows must never draw from.
MARK_ANSWERABILITY = "zqanswerabilitymark"
MARK_VAL = "zqvaltitlemark"
MARK_HELDOUT = "zqheldouttitlemark"
MARK_VAL_REPO = "zqvalrepomark"
MARK_HELDOUT_REPO = "zqheldoutrepomark"
MARKS = (MARK_ANSWERABILITY, MARK_VAL, MARK_HELDOUT, MARK_VAL_REPO, MARK_HELDOUT_REPO)

SENTENCES = (
    "The northern valley survey recorded that the river moved two kilometres east over a "
    "century, cutting channels through farmland and leaving lakes behind.",
    "Archive letters from the period describe floods that reshaped the banks and forced "
    "several villages to rebuild further up the slope.",
    "Later researchers compared those accounts with aerial photographs and found the "
    "descriptions accurate to within a few hundred metres.",
    "The regional council funded a museum exhibit about the floods, and the exhibit drew "
    "visitors from across the province for a decade.",
)

CODE = {
    "python": ("py", "import os\nimport sys\n\n\ndef {name}(path, defaults=None):\n"
               "    defaults = defaults or {{}}\n    with open(path) as handle:\n"
               "        raw = handle.read()\n    values = dict(defaults)\n"
               "    for line in raw.splitlines():\n        key, _, value = line.partition('=')\n"
               "        values[key.strip()] = value.strip()\n    return values\n"),
    "go": ("go", "package conf\n\nimport (\n\t\"os\"\n\t\"strings\"\n)\n\n"
           "func {name}(path string) (map[string]string, error) {{\n"
           "\traw, err := os.ReadFile(path)\n\tif err != nil {{\n\t\treturn nil, err\n\t}}\n"
           "\tout := map[string]string{{}}\n"
           "\tfor _, line := range strings.Split(string(raw), \"\\n\") {{\n"
           "\t\tkey, value, _ := strings.Cut(line, \"=\")\n\t\tout[key] = value\n\t}}\n"
           "\treturn out, nil\n}}\n"),
    "rust": ("rs", "use std::collections::HashMap;\nuse std::fs;\n\n"
             "pub fn {name}(path: &str) -> std::io::Result<HashMap<String, String>> {{\n"
             "    let raw = fs::read_to_string(path)?;\n    let mut out = HashMap::new();\n"
             "    for line in raw.lines() {{\n"
             "        if let Some((k, v)) = line.split_once('=') {{\n"
             "            out.insert(k.trim().to_string(), v.trim().to_string());\n"
             "        }}\n    }}\n    Ok(out)\n}}\n"),
    "typescript": ("ts", "import {{ readFileSync }} from 'fs';\n\n"
                   "export function {name}(path: string): Record<string, string> {{\n"
                   "  const raw = readFileSync(path, 'utf8');\n"
                   "  const out: Record<string, string> = {{}};\n"
                   "  for (const line of raw.split('\\n')) {{\n"
                   "    const [key, value] = line.split('=');\n"
                   "    if (key) out[key.trim()] = (value ?? '').trim();\n  }}\n  return out;\n}}\n"
                   "\nexport default {name};\n"),
}


def _para(title: str, k: int, mark: str = "") -> str:
    return f"{title} part {k} {mark}. " + " ".join(SENTENCES[k % 4:] + SENTENCES[: k % 4])


def _write_world(root: Path) -> dict[str, Path]:
    """A SQuAD file, a pool and a download whose units sit on every side of the split."""
    pool_dir = root / "data" / "pool"
    download = pool_dir / "commitpackft"
    download.mkdir(parents=True)
    cache = root / "cache"
    cache.mkdir()

    squad_rows = []
    for title, mark in [*((t, "") for t in TRAIN_TITLES), (ANSWERABILITY_TITLE, MARK_ANSWERABILITY),
                        (VAL_TITLE, MARK_VAL), (HELDOUT_TITLE, MARK_HELDOUT)]:
        for k in range(3):
            for q in range(2):  # two questions per paragraph: one candidate, counted once
                squad_rows.append({"id": f"{title}-{k}-{q}", "title": title,
                                   "context": _para(title, k, mark), "question": "?",
                                   "answers": {"text": [], "answer_start": []}})
    squad = cache / "train.jsonl"
    squad.write_text("".join(json.dumps(r) + "\n" for r in squad_rows), encoding="utf-8")

    pool_rows: list[dict[str, Any]] = []
    by_lang: dict[str, list[dict[str, Any]]] = {lang: [] for lang in CODE}
    repos = [(r, "") for r in TRAIN_REPOS] + [(VAL_REPO, MARK_VAL_REPO),
                                               (HELDOUT_REPO, MARK_HELDOUT_REPO)]
    for i, (repo, mark) in enumerate(repos):
        lang = list(CODE)[i % len(CODE)]
        ext, body = CODE[lang]
        name = f"{mark or 'load'}_settings_{i}"
        path = f"pkg/settings_{i}.{ext}"
        commit = hashlib.sha1(f"{repo}-{i}".encode()).hexdigest()
        pid = f"{commit}:{path}"
        source = body.format(name=name)
        pool_rows.append({"id": pid, "repo": repo, "path": path, "source": source,
                          "prior_source": source, "hunks": [{"start_line": 6, "end_line": 8}]})
        by_lang[lang].append({"commit": commit, "new_file": path, "license": "mit"})
    pool = pool_dir / "pool.jsonl"
    pool.write_text("".join(json.dumps(r) + "\n" for r in pool_rows), encoding="utf-8")
    langs = {}
    for lang, rows in by_lang.items():
        p = download / f"{lang}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        langs[lang] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (download / "manifest.json").write_text(
        json.dumps({"source_id": "bigcode/commitpackft", "languages": langs}), encoding="utf-8"
    )
    return {"squad": squad, "pool": pool, "download": download}


def _units(binary: Path) -> dict[str, Any]:
    out = subprocess.run([str(binary), "units"], capture_output=True, check=True, timeout=60)
    return json.loads(out.stdout)


def _generate(binary: Path, root: Path, world: dict[str, Path], out: Path, *,
              per_source: int = PER_SOURCE, seed: int = 0) -> subprocess.CompletedProcess[bytes]:
    allow_path = root / f"allowlist-{out.name}.json"
    if not allow_path.exists():
        allow = noul_allowlist(squad=world["squad"], pool=world["pool"],
                               download_root=world["download"], template_units=_units(binary),
                               config=CONFIG, repo_root=root)
        allow_path.write_text(json.dumps(allow, sort_keys=True), encoding="utf-8")
    return subprocess.run(
        [str(binary), "generate", "--allowlist", str(allow_path), "--squad", str(world["squad"]),
         "--pool", str(world["pool"]), "--out", str(out), "--per-source", str(per_source),
         "--seed", str(seed)],
        capture_output=True, timeout=300,
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    """One object per ``\\n``-terminated line. Not ``str.splitlines``, which also breaks on
    U+2028 / U+0085 -- characters a JSON string may hold raw, and SQuAD's do."""
    return [json.loads(x) for x in path.read_text("utf-8").split("\n") if x.strip()]


def _rows(corpus: Path) -> list[dict[str, Any]]:
    return _jsonl(corpus / "examples.jsonl")


def _rewrite(corpus: Path, rows: list[dict[str, Any]]) -> None:
    """Replace the corpus's rows and re-pin its manifest, so only the row checks can refuse."""
    body = "".join(json.dumps(r) + "\n" for r in rows)
    (corpus / "examples.jsonl").write_text(body, encoding="utf-8")
    manifest = json.loads((corpus / "manifest.json").read_text("utf-8"))
    manifest["examples_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    manifest["totals"]["examples"] = len(rows)
    (corpus / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    return _write_world(tmp_path)


@pytest.fixture
def corpus(noul_rows_bin: Path, tmp_path: Path, world: dict[str, Path]) -> Path:
    out = tmp_path / "noul"
    done = _generate(noul_rows_bin, tmp_path, world, out)
    assert done.returncode == 0, done.stderr.decode()
    return out


def _load(tmp_path: Path, corpus: Path, world: dict[str, Path], **kw: Any) -> Any:
    return load_noul_rows(corpus, download_root=world["download"],
                          config=kw.pop("config", CONFIG), repo_root=tmp_path)


# -- rule 3: the split is the canonical one, at every step ---------------------------------


def test_the_allowlist_keeps_only_train_units_by_the_canonical_split(
    noul_rows_bin: Path, tmp_path: Path, world: dict[str, Path]
) -> None:
    units = _units(noul_rows_bin)
    allow = noul_allowlist(squad=world["squad"], pool=world["pool"],
                           download_root=world["download"], template_units=units,
                           config=CONFIG, repo_root=tmp_path)
    assert set(allow["squad"]["titles"]) == set(TRAIN_TITLES)
    assert allow["squad"]["titles"] == {t: squad_title_repo_key(t) for t in TRAIN_TITLES}
    assert allow["squad"]["excluded"] == {
        "family:qa.answerability": 1, "split:heldout": 1, "split:val": 1,
    }
    pool_repos = {r["id"]: r["repo"] for r in _jsonl(world["pool"])}
    assert {pool_repos[p] for p in allow["pool"]["files"]} == set(TRAIN_REPOS)
    assert allow["pool"]["excluded"] == {"split:heldout": 1, "split:val": 1}
    assert set(allow["pool"]["files"].values()) == {"mit"}
    assert allow["templates"]["units"] == sorted(
        u for u in units["units"] if _split(u) == "train"
    )
    assert sum(allow["templates"]["excluded"].values()) == len(units["units"]) - len(
        allow["templates"]["units"])
    assert allow["templates"]["licence"] == "apache-2.0"
    assert allow["split"] == {"seed": CONFIG.seed, "train_fraction": CONFIG.train_fraction,
                              "val_fraction": CONFIG.val_fraction}


def test_generated_rows_come_only_from_train_units_in_equal_shares(
    corpus: Path, tmp_path: Path, world: dict[str, Path]
) -> None:
    rows = _rows(corpus)
    assert Counter(r["noul_source"] for r in rows) == {s: PER_SOURCE for s in NOUL_SOURCES}
    assert all(r["class"] == NOUL_CLASS for r in rows)
    text = (corpus / "examples.jsonl").read_text("utf-8")
    for mark in MARKS:
        assert mark not in text, f"{mark}: content of a non-train unit reached a row"
    for r in rows:
        assert _split(r["repo"]) == "train", r
        if r["noul_source"] == NOUL_PROSE:
            assert r["squad_title"] in TRAIN_TITLES and r["licence"] == "cc-by-sa-4.0"
        elif r["noul_source"] == NOUL_SCRAMBLED:
            assert r["repo"] in TRAIN_REPOS and r["licence"] == "mit"
        else:
            assert r["repo"].startswith(NOUL_TEMPLATE_UNIT_PREFIX)
            assert r["licence"] == "apache-2.0"
        # One hunk in the corpus's exact header shape, never a section heading.
        header = r["diff"].split("\n", 1)[0]
        assert re.fullmatch(r"@@ -(\d+),\d+ \+\1,\d+ @@", header), header
        assert r["diff"].count("\n@@") == 0
    manifest = json.loads((corpus / "manifest.json").read_text("utf-8"))
    assert manifest["totals"]["by_source"] == {s: PER_SOURCE for s in NOUL_SOURCES}
    assert manifest["pool"]["path"] == "pool.jsonl"
    assert manifest["excluded_by_allowlist"]["prose"] == {
        "family:qa.answerability": 1, "split:heldout": 1, "split:val": 1,
    }

    load = _load(tmp_path, corpus, world)
    assert load.by_source == {s: PER_SOURCE for s in NOUL_SOURCES}
    assert all(r.mutation_class == NOUL_CLASS and r.diff_span is None for r in load.rows)


def test_the_generator_is_deterministic_and_refuses_bytes_it_was_not_allowed(
    noul_rows_bin: Path, tmp_path: Path, world: dict[str, Path], corpus: Path
) -> None:
    again = tmp_path / "again"
    (tmp_path / "allowlist-again.json").write_bytes(
        (tmp_path / "allowlist-noul.json").read_bytes())
    assert _generate(noul_rows_bin, tmp_path, world, again).returncode == 0
    for name in ("examples.jsonl", "manifest.json"):
        assert (again / name).read_bytes() == (corpus / name).read_bytes(), name
    # A different seed draws a different corpus from the same allowed units.
    (tmp_path / "allowlist-seed1.json").write_bytes(
        (tmp_path / "allowlist-noul.json").read_bytes())
    seed1 = tmp_path / "seed1"
    assert _generate(noul_rows_bin, tmp_path, world, seed1, seed=1).returncode == 0
    assert (seed1 / "examples.jsonl").read_bytes() != (corpus / "examples.jsonl").read_bytes()
    # The corpus is written once.
    clobber = _generate(noul_rows_bin, tmp_path, world, corpus)
    assert clobber.returncode == 2 and b"already holds a corpus" in clobber.stderr
    # SQuAD bytes the allowlist was not computed from are refused.
    with world["squad"].open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"title": TRAIN_TITLES[0], "context": _para("x", 9)}) + "\n")
    (tmp_path / "allowlist-changed.json").write_bytes(
        (tmp_path / "allowlist-noul.json").read_bytes())
    changed = _generate(noul_rows_bin, tmp_path, world, tmp_path / "changed")
    assert changed.returncode == 2 and b"re-run the allowlist step" in changed.stderr


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        ("answerability_title", "title_of_the_held_out_family"),
        ("val_title", "split_unit_in_val"),
        ("heldout_title", "split_unit_in_heldout"),
        ("heldout_repo", "split_unit_in_heldout"),
        ("val_template_unit", "split_unit_in_val"),
        ("prose_licence", "licence_is_not_squads"),
        ("scrambled_licence", "licence_disagrees_with_download"),
        ("unknown_pool_id", "pool_id_not_in_pool"),
        ("template_licence", "licence_is_not_the_template_licence"),
    ],
)
def test_the_loader_refuses_the_whole_corpus_for_one_row_it_cannot_admit(
    corpus: Path, tmp_path: Path, world: dict[str, Path], edit: str, reason: str
) -> None:
    rows = _rows(corpus)
    prose = next(r for r in rows if r["noul_source"] == NOUL_PROSE)
    scrambled = next(r for r in rows if r["noul_source"] == NOUL_SCRAMBLED)
    template = next(r for r in rows if r["noul_source"] == NOUL_UNSEEN_LANGUAGE)
    pool = {r["repo"]: r["id"] for r in _jsonl(world["pool"])}
    if edit.endswith("_title"):
        title = {"answerability_title": ANSWERABILITY_TITLE, "val_title": VAL_TITLE,
                 "heldout_title": HELDOUT_TITLE}[edit]
        prose.update(squad_title=title, repo=squad_title_repo_key(title))
    elif edit == "heldout_repo":
        scrambled.update(repo=HELDOUT_REPO, pool_id=pool[HELDOUT_REPO])
    elif edit == "val_template_unit":
        unit = next(u for u in (f"{NOUL_TEMPLATE_UNIT_PREFIX}kotlin/x{i}" for i in range(999))
                    if _split(u) == "val")
        template.update(repo=unit)
    elif edit == "prose_licence":
        prose["licence"] = "mit"
    elif edit == "scrambled_licence":
        scrambled["licence"] = "apache-2.0"
    elif edit == "unknown_pool_id":
        scrambled["pool_id"] = "0" * 40 + ":nowhere.py"
    elif edit == "template_licence":
        template["licence"] = "mit"
    _rewrite(corpus, rows)
    with pytest.raises(DefectCorpusError, match=reason):
        _load(tmp_path, corpus, world)


def test_a_corpus_generated_under_another_split_is_refused(
    corpus: Path, tmp_path: Path, world: dict[str, Path]
) -> None:
    with pytest.raises(DefectCorpusError, match="generated under split"):
        _load(tmp_path, corpus, world, config=DataConfig(seed=CONFIG.seed + 1))


def test_a_held_out_corpus_path_is_refused_before_it_is_read(
    tmp_path: Path, world: dict[str, Path]
) -> None:
    from qd_data.errors import HeldOutViolation

    with pytest.raises(HeldOutViolation):
        load_noul_rows(tmp_path / "data" / "heldout" / "noul", download_root=world["download"],
                       config=CONFIG, repo_root=tmp_path)


# -- the main corpus beside it ---------------------------------------------------------------


def _main_corpus(root: Path, world: dict[str, Path], *, extra: list[dict[str, Any]]) -> Path:
    """A qd-mutate corpus over the same pool, in the shape qd-mutate writes."""
    pool_row = _jsonl(world["pool"])[0]
    after = pool_row["source"]
    lines = after.split("\n")
    before = "\n".join([*lines[:6], lines[6] + "  # was", *lines[7:]])
    example = {
        "id": f"{pool_row['id']}#0", "pool_id": pool_row["id"], "repo": pool_row["repo"],
        "path": pool_row["path"], "language": "python", "class": "clean", "operator": "clean",
        "silent": False, "span": None,
        "function": {"repo": pool_row["repo"], "path": pool_row["path"], "symbol": "f",
                     "arity": 1},
        "before": before, "after": after,
        "diff": f"@@ -5,3 +5,3 @@\n {lines[4]}\n {lines[5]}\n-{lines[6]}  # was\n+{lines[6]}\n",
        "hunk_constrained": True, "seed": 0,
    }
    corpus = root / "main"
    corpus.mkdir()
    rows = [example, *extra]
    body = "".join(json.dumps(r) + "\n" for r in rows)
    (corpus / "examples.jsonl").write_text(body, encoding="utf-8")
    (corpus / "manifest.json").write_text(json.dumps({
        "examples_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "pool": {"path": "/elsewhere/pool.jsonl",
                 "sha256": hashlib.sha256(world["pool"].read_bytes()).hexdigest(),
                 "records": len(_jsonl(world["pool"]))},
        "totals": {"examples": len(rows)},
    }), encoding="utf-8")
    return corpus


def test_the_noul_corpus_is_appended_and_the_main_join_still_fails_closed(
    corpus: Path, tmp_path: Path, world: dict[str, Path]
) -> None:
    main = _main_corpus(tmp_path, world, extra=[])
    load = load_defect_rows(main, download_root=world["download"], config=CONFIG,
                            repo_root=tmp_path, noul_dir=corpus)
    assert load.n_corpus == 1 and load.n_noul == 3 * PER_SOURCE
    assert load.rows[0].mutation_class == "clean"
    assert load.by_class == {"clean": 1, NOUL_CLASS: 3 * PER_SOURCE}
    assert load.noul_by_source == {s: PER_SOURCE for s in NOUL_SOURCES}

    orphan = json.loads((main / "examples.jsonl").read_text("utf-8"))
    orphan.update(id="deadbeef:x.py#0", pool_id="deadbeef:x.py")
    (tmp_path / "u").mkdir()
    unresolved = _main_corpus(tmp_path / "u", world, extra=[orphan])
    with pytest.raises(DefectCorpusError, match="pool_id that is not a pool id"):
        load_defect_rows(unresolved, download_root=world["download"], config=CONFIG,
                         repo_root=tmp_path, noul_dir=corpus)


def test_a_qd_mutate_example_cannot_claim_the_noul_class(
    tmp_path: Path, world: dict[str, Path]
) -> None:
    """The noul corpus is the only door: the main corpus's parser refuses the class."""
    main = _main_corpus(tmp_path, world, extra=[])
    row = json.loads((main / "examples.jsonl").read_text("utf-8"))
    row["class"] = NOUL_CLASS
    body = json.dumps(row) + "\n"
    (main / "examples.jsonl").write_text(body, encoding="utf-8")
    manifest = json.loads((main / "manifest.json").read_text("utf-8"))
    manifest["examples_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    (main / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DefectCorpusError, match="is not one of"):
        load_defect_rows(main, download_root=world["download"], config=CONFIG,
                         repo_root=tmp_path)


def _noul_row(**kw: Any) -> DefectRow:
    base: dict[str, Any] = dict(
        example_id="noul:prose:0", pool_id="squad-title:T", repo="squad-title:T",
        path="docs/t.md", symbol="noul:prose:0", arity=0, language="prose",
        mutation_class=NOUL_CLASS, operator="noul.prose",
        diff="@@ -1,2 +1,2 @@\n Some prose here.\n-An old line.\n+A new line.\n",
        diff_span=None, span_refusal=None, licence="cc-by-sa-4.0", noul_source=NOUL_PROSE,
    )
    base.update(kw)
    return DefectRow(**base)


def test_a_noul_row_names_its_source_and_points_at_nothing() -> None:
    _noul_row()
    with pytest.raises(DefectCorpusError, match="noul_source"):
        _noul_row(noul_source=None)
    with pytest.raises(DefectCorpusError, match="noul_source"):
        _noul_row(noul_source="clinc")
    with pytest.raises(DefectCorpusError, match="only a 'noul' row has one"):
        _noul_row(mutation_class="clean")
    with pytest.raises(DefectCorpusError, match="points at nothing"):
        _noul_row(diff_span=(2, 2))


# -- the render ------------------------------------------------------------------------------


def test_a_noul_row_renders_the_noul_letter_as_its_choice_gold() -> None:
    row = rewrite_defect_class(_noul_row(), family_id=DEFECT_FAMILY_ID, index=0, config=CONFIG)
    gold = {g.slot_name: g for g in row.gold}
    assert gold[CHOICE_SLOT].is_noul and gold[CHOICE_SLOT].value is None
    assert gold[SPAN_SLOT].is_noul
    assert row.metadata["noul_source"] == NOUL_PROSE
    assert row.licence_id == "cc-by-sa-4.0" and "share-alike" in row.obligations
    # The options are still the four classes; the abstention is the row every set ends in.
    assert row.request.slots[0].options == ("stub", "logic", "cosmetic", "clean")  # type: ignore[union-attr]
    choice, span = training_texts(row, seed=CONFIG.seed)
    assert choice.slot_name == CHOICE_SLOT and choice.text.endswith(NOUL_LETTER)
    assert span.span_abstains
    # A main-corpus row still carries no noul_source key at all: its content hash is unchanged.
    clean = rewrite_defect_class(
        _noul_row(mutation_class="clean", noul_source=None, operator="clean"),
        family_id=DEFECT_FAMILY_ID, index=0, config=CONFIG,
    )
    assert "noul_source" not in clean.metadata
    assert {g.slot_name: g.value for g in clean.gold}[CHOICE_SLOT] == "clean"


def test_noul_rows_teach_the_letter_channel_and_stay_whole_through_dedupe_and_split(
    corpus: Path, tmp_path: Path, world: dict[str, Path]
) -> None:
    main = _main_corpus(tmp_path, world, extra=[])
    load = load_defect_rows(main, download_root=world["download"], config=CONFIG,
                            repo_root=tmp_path, noul_dir=corpus)
    mix = build_mixture({DEFECT_SOURCE_ID: load.rows}, config=CONFIG,
                        families=[DEFECT_FAMILY_ID])
    assert isinstance(mix.status, Ran) and mix.status.passed, mix.status
    assert mix.refusals[DEFECT_SOURCE_ID] == {}
    choice = mix.abstention["choice"]
    assert isinstance(choice, Ran) and choice.passed and choice.n == 3 * PER_SOURCE
    assert "code.defect_class" in choice.detail
    report = dedupe(list(mix.rows), config=CONFIG)
    assert len(report.kept) == len(mix.rows), "dedupe dropped a noul row across units"
    splits = split(report, config=CONFIG)
    noul_ids = {r.row_id for r in mix.rows if r.metadata.get("noul_source")}
    assert noul_ids <= {r.row_id for r in splits.rows_by_split["train"]}


# -- disjointness from the OOD suite -----------------------------------------------------------


def _prose_pool() -> list[str]:
    """Every MMLU and CommonsenseQA question the general record holds -- a superset of the
    suite's val draw, so a pass here holds for any suite drawn from it. Falls back to the
    fixture sentences where the record is absent, and the caller says so."""
    if not GENERAL_RECORD.is_file():
        # Question-shaped and sharing no sentence with the fixture SQuAD paragraphs.
        return [f"Which statement about quantity {i} holds when the reading doubles?"
                for i in range(80)]
    out: list[str] = []
    for entry in json.loads(GENERAL_RECORD.read_text("utf-8")):
        if entry["dataset"] not in ("cais/mmlu", "tau/commonsense_qa"):
            continue
        for row in _jsonl(Path(entry["jsonl"])):
            q = row.get("question")
            if isinstance(q, str) and q.strip():
                out.append(q)
    return out


def _suite_texts(prose: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for seed in SUITE_SEEDS:
        for case in ood.build_ood_suite(prose, seed=seed):
            out[f"{seed}:{case.case_id}"] = case.context
    return out


def _checked_clean(rows: dict[str, str], suite: dict[str, str]) -> None:
    """Zero hits both ways -- and not vacuously: every row and every suite code case was
    long enough to have an 8-gram, so each was compared rather than skipped."""
    there = decontaminate(rows, {"suite": suite})
    assert sum(there.hits.values()) == 0, there.hit_examples
    assert there.replay_rows_too_short == 0 and there.replay_rows_checked == len(rows)
    back = decontaminate(suite, {"rows": rows})
    assert sum(back.hits.values()) == 0, back.hit_examples
    code_cases = [k for k in suite if "-prose-" not in k]
    assert all(word_ngrams(suite[k]) for k in code_cases), "a code case has no 8-gram"
    some = next(iter(rows.values()))
    assert _max_containment({"a": some}, {"b": some}) == 1.0, "the measure cannot see a copy"


def _max_containment(rows: dict[str, str], targets: dict[str, str]) -> float:
    """The largest share of either side's word 8-grams found in the other, over every pair.
    ``word_ngrams`` is the same function ``decontaminate`` compares with."""
    grams_r = [g for g in (word_ngrams(t) for t in rows.values()) if g]
    grams_t = [g for g in (word_ngrams(t) for t in targets.values()) if g]
    best = 0.0
    for a in grams_r:
        for b in grams_t:
            shared = len(a & b)
            if shared:
                best = max(best, shared / len(a), shared / len(b))
    return best


def _assert_disjoint(rows: list[dict[str, Any]], *, require_real_prose: bool) -> float:
    if require_real_prose and not GENERAL_RECORD.is_file():
        pytest.skip(f"{GENERAL_RECORD} is absent: the suite's prose cannot be drawn")
    prose = _prose_pool()
    suite = _suite_texts(prose)
    r2 = {r["id"]: r["diff"] for r in rows}
    _checked_clean(r2, suite)
    # Every row against every candidate the suite's prose could be drawn from.
    pool_report = decontaminate(r2, {"mmlu_csqa": {str(i): q for i, q in enumerate(prose)}})
    assert sum(pool_report.hits.values()) == 0, pool_report.hit_examples

    templates = [r for r in rows if r["noul_source"] == NOUL_UNSEEN_LANGUAGE]
    assert templates
    suite_languages = set(ood._UNSEEN) | set(ood._EXT) | {"swift"}
    assert not {r["language"] for r in templates} & suite_languages
    names = set(ood._NAMES) | set(ood._FNS)
    for r in templates:
        words = set(re.findall(r"\w+", r["diff"] + " " + r["path"]))
        assert not words & names, (r["id"], words & names)
    # No path is one of the suite's: notes.txt, src/scrambled_<i>.<ext>, src/<lang>_<i>.<ext>.
    suite_path = re.compile(
        r"notes\.txt|src/scrambled_\d+\.\w+|src/(" + "|".join(ood._UNSEEN) + r")_\d+\.\w+"
    )
    for r in rows:
        assert not suite_path.fullmatch(r["path"]) and "scrambled" not in r["path"], r["path"]
    return _max_containment(r2, suite)


def test_the_rows_share_no_8_gram_mass_with_the_ood_suite(corpus: Path) -> None:
    worst = _assert_disjoint(_rows(corpus), require_real_prose=False)
    assert worst < 0.5


# -- the pipeline and the FT rebuild ------------------------------------------------------------


def test_the_class_balance_names_the_abstention() -> None:
    import real_tokenizer_pipeline as pipeline

    noul = rewrite_defect_class(_noul_row(), family_id=DEFECT_FAMILY_ID, index=0, config=CONFIG)
    clean = rewrite_defect_class(
        _noul_row(example_id="c", symbol="c", mutation_class="clean", noul_source=None,
                  operator="clean"),
        family_id=DEFECT_FAMILY_ID, index=1, config=CONFIG,
    )
    balance = pipeline.defect_balance([noul, clean], refused={noul.row_id: "x"})
    assert balance["classes"] == {"clean": 1}
    assert balance["refused_by_class_and_reason"] == {"noul:x": 1}
    assert pipeline.defect_balance([noul, clean])["classes"] == {"clean": 1, "noul": 1}


def test_the_pipeline_refuses_noul_rows_without_the_defect_family(tmp_path: Path) -> None:
    import real_tokenizer_pipeline as pipeline

    with pytest.raises(SystemExit, match="--defect-noul needs --defect-class"):
        pipeline.run(out=tmp_path, max_pairs=1, blank_line_runs=False, rev="HEAD",
                     defect_noul=tmp_path)


def test_the_flag_reaches_run_and_the_recipe_names_the_corpus_by_its_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import real_tokenizer_pipeline as pipeline

    class Reached(Exception):
        pass

    got: dict[str, Any] = {}

    def run(**kw: Any) -> None:
        got.update(kw)
        raise Reached

    monkeypatch.setattr(pipeline, "run", run)
    with pytest.raises(Reached):
        pipeline.main(["--out", str(tmp_path), "--defect-class", str(tmp_path / "d"),
                       "--defect-noul", str(tmp_path / "n")])
    assert got["defect_noul"] == tmp_path / "n"

    def corpus_dir(name: str, sha: str) -> Path:
        d = tmp_path / name
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"examples_sha256": sha}), encoding="utf-8")
        return d

    defect = corpus_dir("defect", "a" * 64)
    download = tmp_path / "download"
    download.mkdir()
    (download / "manifest.json").write_text(json.dumps(
        {"source_id": "bigcode/commitpackft", "languages": {"python": {"sha256": "b" * 64}}}
    ), encoding="utf-8")
    noul_a, noul_b = corpus_dir("noul_a", "c" * 64), corpus_dir("noul_b", "d" * 64)
    ref = tmp_path / "refs-main"
    ref.write_text("e" * 40, encoding="utf-8")
    monkeypatch.setattr(pipeline, "MODEL_REF", ref)
    seen: list[str] = []

    def protocol(**kw: Any) -> None:
        seen.append(kw["recipe_hash"])
        raise Reached

    monkeypatch.setattr(pipeline, "Protocol", protocol)
    base = ["--out", str(tmp_path), "--rev", "0632f693d3b765b726499e7b4bf19c67959b75cb",
            "--ledger", str(tmp_path / "l.jsonl"), "--defect-class", str(defect),
            "--defect-download", str(download)]
    for extra in ([], ["--defect-noul", str(noul_a)], ["--defect-noul", str(noul_b)]):
        with pytest.raises(Reached):
            pipeline.main([*base, *extra])
    assert len(set(seen)) == 3, "with and without the noul corpus, or two of them, share a hash"


def test_the_ft_rebuild_refuses_noul_rows_without_the_defect_family() -> None:
    pytest.importorskip("torch", reason="tools/real_ft_run.py raises at import without torch")
    import real_ft_run as rft

    with pytest.raises(ValueError, match="read nothing"):
        rft.ft_splits(commitpackft=None, max_pairs=1, rev="x", config=CONFIG,
                      defect_noul=Path("noul"))
    kw = {"rev": "r", "max_pairs": 4, "commitpackft": None, "defect_class": None,
          "defect_max_rows": None}
    assert rft.replay_corpus_identity(**kw) == kw


# -- the real corpus ----------------------------------------------------------------------------


def _real_present(noul: Path = REAL_NOUL) -> bool:
    return (noul / "examples.jsonl").is_file() and (
        REPO / "data" / "pool" / "commitpackft-pool-v2.jsonl"
    ).is_file() and (REAL_DOWNLOAD / "manifest.json").is_file()


#: Both real corpora, each skipped where its rows are not on this host.
REAL_CORPORA = [
    pytest.param(d, id=d.name, marks=pytest.mark.skipif(
        not _real_present(d), reason=f"data/pool/{d.name} is not on this host"))
    for d in (REAL_NOUL, REAL_NOUL_V2)
]


@pytest.mark.parametrize("noul", REAL_CORPORA)
def test_the_real_corpus_loads_in_equal_thirds_of_train_units_and_is_disjoint_from_the_suite(
    capsys: pytest.CaptureFixture[str], noul: Path,
) -> None:
    load = load_noul_rows(noul, download_root=REAL_DOWNLOAD, config=CONFIG,
                          repo_root=REPO)
    manifest = json.loads((noul / "manifest.json").read_text("utf-8"))
    per = int(manifest["per_source"])
    assert load.by_source == {s: per for s in NOUL_SOURCES}
    rows = _rows(noul)
    mix = build_mixture({DEFECT_SOURCE_ID: load.rows}, config=CONFIG,
                        families=[DEFECT_FAMILY_ID])
    assert mix.refusals[DEFECT_SOURCE_ID] == {}, mix.refusals
    report = dedupe(list(mix.rows), config=CONFIG)
    assert len(report.kept) == len(mix.rows)
    splits = split(report, config=CONFIG)
    assert {s: len(v) for s, v in splits.rows_by_split.items()} == {
        "train": 3 * per, "val": 0, "heldout": 0,
    }
    worst = _assert_disjoint(rows, require_real_prose=True)
    with capsys.disabled():
        print(f"\n{noul.name}: max 8-gram containment against the suite = {worst:.4f}")
    assert worst < 0.5


@pytest.mark.parametrize("noul", REAL_CORPORA)
def test_the_real_corpus_is_disjoint_from_the_exact_suite_the_gate_builds(
    capsys: pytest.CaptureFixture[str], qd_prep: Path, noul: Path,
) -> None:
    """The suite ``real_ft_run.prepare_ood`` builds, rebuilt the way it builds it: prose from
    the general record's val split through ``ft_splits``, at the config seed. The superset
    check above varies the seed; this one is the gate's own draw, whose code cases depend on
    the prose pool through the shared RNG."""
    if not GENERAL_RECORD.is_file():
        pytest.skip(f"{GENERAL_RECORD} is absent: the gate's suite cannot be drawn")
    pytest.importorskip("torch", reason="tools/real_ft_run.py raises at import without torch")
    import real_ft_run as rft
    from repo_git import resolve_rev

    splits = rft.ft_splits(
        commitpackft=None, max_pairs=0, rev=resolve_rev(REPO, "HEAD"), config=CONFIG,
        repo_history=False, general_record=GENERAL_RECORD,
        general_max_rows=rft.OOD_GENERAL_MAX_ROWS,
    )
    prose = [row.request.context.decode("utf-8") for row in splits["val"]
             if row.family_id in rft.OOD_PROSE_FAMILIES]
    suite = {c.case_id: c.context for c in ood.build_ood_suite(prose, seed=CONFIG.seed)}
    assert Counter(c.split("-")[1] for c in suite) == {
        "prose": ood.OOD_CASES_PER_CATEGORY, "unseen": ood.OOD_CASES_PER_CATEGORY,
        "scrambled": ood.OOD_CASES_PER_CATEGORY,
    }
    r2 = {r["id"]: r["diff"] for r in _rows(noul)}
    _checked_clean(r2, suite)
    worst = _max_containment(r2, suite)
    with capsys.disabled():
        print(f"\n{noul.name} vs the gate's own suite (seed {CONFIG.seed}): "
              f"{len(r2)} rows x {len(suite)} cases, max 8-gram containment {worst:.4f}")
    assert worst < 0.5


# -- defect-noul-v2: the v3.1 re-form ----------------------------------------------------------


@pytest.mark.skipif(not _real_present(REAL_NOUL_V2),
                    reason="data/pool/defect-noul-v2 is not on this host")
def test_v2_fills_each_form_exactly_and_records_the_preregistered_bars() -> None:
    """Fable F2/F3: the same 5% share (2,502 rows, equal thirds), each third's forms in exact
    halves, the scrambled rows drawn from the defect corpus's own train-split diffs, and the
    bars recorded in the manifest before any run reads the corpus."""
    rows = _rows(REAL_NOUL_V2)
    manifest = json.loads((REAL_NOUL_V2 / "manifest.json").read_text("utf-8"))
    v1 = json.loads((REAL_NOUL / "manifest.json").read_text("utf-8"))
    assert len(rows) == v1["totals"]["examples"] == 2502
    assert Counter((r["noul_source"], r["noul_form"]) for r in rows) == V2_FORMS
    assert manifest["forms"] == "v2"
    assert manifest["corpus"]["dir"] == REAL_DEFECT_V3.name
    v3 = json.loads((REAL_DEFECT_V3 / "manifest.json").read_text("utf-8"))
    assert manifest["corpus"]["examples_sha256"] == v3["examples_sha256"]
    bars = manifest["preregistered"]
    assert "30/60" in bars["ood_abstain_by_category"]["bar"]
    assert "2%" in bars["defect_class_in_distribution_abstention"]["bound"]
    # Every scrambled row's diff is a permutation of one corpus row's diff from its own pool
    # file: same headers, same body lines (lines form) or same per-line tokens (tokens form).
    by_pool: dict[str, list[str]] = {}
    for ex in _jsonl(REAL_DEFECT_V3 / "examples.jsonl") if (
            REAL_DEFECT_V3 / "examples.jsonl").is_file() else []:
        by_pool.setdefault(ex["pool_id"], []).append(ex["diff"])
    if not by_pool:
        pytest.skip("data/pool/commitpackft-corpus-v3/examples.jsonl is not on this host")

    def lines_key(diff: str) -> list[str]:
        return sorted(diff.split("\n"))

    def tokens_key(diff: str) -> list[str]:
        return sorted(line[:1] + " ".join(sorted(line[1:].split()))
                      if not line.startswith("@@ -") else line for line in diff.split("\n"))

    for r in rows:
        if r["noul_source"] != NOUL_SCRAMBLED:
            continue
        key = lines_key if r["noul_form"] == "lines" else tokens_key
        originals = by_pool[r["pool_id"]]
        assert r["diff"] not in originals, r["id"]
        assert any(key(r["diff"]) == key(o) for o in originals), r["id"]
        assert r["diff"].startswith("@@ -")


@pytest.mark.skipif(not _real_present(REAL_NOUL_V2),
                    reason="data/pool/defect-noul-v2 is not on this host")
def test_v2_question_rows_are_disjoint_from_the_general_val_split(qd_prep: Path) -> None:
    """The question rows are SQuAD questions of train-split titles, so by the title split no
    one is a val row; this checks the text too, against the general val split the v3.1 shards
    are built with (the build's own --general-max-rows), in both directions."""
    if not GENERAL_RECORD.is_file():
        pytest.skip(f"{GENERAL_RECORD} is absent: the general val split cannot be built")
    pytest.importorskip("torch", reason="tools/real_ft_run.py raises at import without torch")
    import real_ft_run as rft
    from repo_git import resolve_rev

    splits = rft.ft_splits(
        commitpackft=None, max_pairs=0, rev=resolve_rev(REPO, "HEAD"), config=CONFIG,
        repo_history=False, general_record=GENERAL_RECORD, general_max_rows=200_000,
    )
    val = {row.row_id: row.request.context.decode("utf-8") for row in splits["val"]}
    assert val, "the general val split is empty: nothing was compared"
    questions = {r["id"]: r["diff"] for r in _rows(REAL_NOUL_V2)
                 if r["noul_form"] == "question"}
    assert len(questions) == V2_FORMS[(NOUL_PROSE, "question")]
    there = decontaminate(questions, {"general_val": val})
    assert sum(there.hits.values()) == 0, there.hit_examples
    assert there.replay_rows_too_short == 0
    back = decontaminate(val, {"questions": questions})
    assert sum(back.hits.values()) == 0, back.hit_examples
    # And no val text contains a question row's questions verbatim.
    texts = [" ".join(line[1:] for line in d.split("\n")[1:] if line) for d in questions.values()]
    joined_val = "\n".join(val.values())
    assert not [t for t in texts if t in joined_val]


@pytest.mark.skipif(
    not (_real_present() and (REAL_NOUL / "allowlist.json").is_file()
         and GENERAL_RECORD.is_file()),
    reason="defect-noul-v1's rows, allowlist or the general record is not on this host",
)
def test_the_real_v1_regenerates_byte_for_byte_at_the_default_forms(
    noul_rows_bin: Path, tmp_path: Path,
) -> None:
    """``--forms v1`` (the default) still writes defect-noul-v1 exactly: the generator's v2
    work left v1 reproducible."""
    squad = next(Path(e["jsonl"]) for e in json.loads(GENERAL_RECORD.read_text("utf-8"))
                 if e["dataset"] == "rajpurkar/squad_v2" and Path(e["jsonl"]).name == "train.jsonl")
    out = tmp_path / "v1"
    done = subprocess.run(
        [str(noul_rows_bin), "generate", "--allowlist", str(REAL_NOUL / "allowlist.json"),
         "--squad", str(squad),
         "--pool", str(REPO / "data" / "pool" / "commitpackft-pool-v2.jsonl"),
         "--out", str(out)],
        capture_output=True, timeout=600,
    )
    assert done.returncode == 0, done.stderr.decode()
    assert (out / "manifest.json").read_bytes() == (REAL_NOUL / "manifest.json").read_bytes()
    assert (out / "examples.jsonl").read_bytes() == (REAL_NOUL / "examples.jsonl").read_bytes()


@pytest.mark.skipif(
    not (_real_present() and (REAL_DEFECT / "examples.jsonl").is_file()),
    reason="the noul corpus, the defect corpus or the download is not on this host",
)
def test_a_shard_set_built_with_noul_rows_is_relabelled_only_with_them(
    tmp_path: Path, qd_prep: Path
) -> None:
    """Against the pipeline's own output, as the defect-class test in test_real_ft_pieces
    does: the noul rows reach the shards with the abstain letter as their gold, the FT
    rebuild with --defect-noul pairs every sequence, and the rebuild without it is refused."""
    pytest.importorskip("torch", reason="tools/real_ft_run.py raises at import without torch")
    pytest.importorskip("transformers")
    import real_ft_run as rft
    import real_tokenizer_pipeline as pipeline
    from repo_git import resolve_rev

    from qd_train.shards import ShardReader

    rev = resolve_rev(REPO, "HEAD")
    out = tmp_path / "out"
    try:
        pipeline.run(out=out, max_pairs=0, blank_line_runs=False, rev=rev, repo_history=False,
                     defect_class=REAL_DEFECT, defect_max_rows=40, defect_noul=REAL_NOUL,
                     memo_limit=0)
    except OSError as exc:  # pragma: no cover - no cached tokenizer on this host
        pytest.skip(f"the pipeline could not load its tokenizer: {exc}")
    reader = ShardReader(out / "shards" / "train", config=CONFIG, repo_root=out,
                         expect_rev=rev)
    train, _ = rft.ft_split_rows(commitpackft=None, max_pairs=0, rev=rev, config=CONFIG,
                                 repo_history=False, defect_class=REAL_DEFECT,
                                 defect_max_rows=40, defect_noul=REAL_NOUL)
    labels, _excluded = rft._labels(train, config=CONFIG)
    paired = rft.pair_labels(reader, labels, require_index=True)
    noul_choice = [lb for lb in paired
                   if lb.row_id.startswith(f"qdm:{DEFECT_FAMILY_ID}:noul:")
                   and lb.slot_name == CHOICE_SLOT]
    assert noul_choice and all(lb.gold_letter == NOUL_LETTER for lb in noul_choice)
    without, _ = rft.ft_split_rows(commitpackft=None, max_pairs=0, rev=rev, config=CONFIG,
                                   repo_history=False, defect_class=REAL_DEFECT,
                                   defect_max_rows=40)
    with pytest.raises(SystemExit, match="was not built from these rows"):
        rft.pair_labels(reader, rft._labels(without, config=CONFIG)[0], require_index=True)
