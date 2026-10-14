"""``load_defect_rows`` for a corpus whose pool is the human's own repositories, and the proof
that the commitpackft (v5) path did not move when that was added.

The own-repos path takes its licence from the source's declaration
(``source_by_id(OWN_REPOS_SOURCE_ID).declared_licence``) through the same three checks
``_load_own_prose`` makes. A self-declared manifest is not provenance, so admission only
narrows: the pool's repos must be train repositories of the pinned own-repos manifest, the pool
must equal its files list row for row by content, and the commitpackft download must be read to
prove no pool id is also in it (a check that cannot run refuses). One corpus is one source.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from test_defect_class import _load, _mixed, _write_corpus

from qd_data import defect_class
from qd_data.config import DataConfig
from qd_data.defect_class import DefectCorpusError, DefectLoad, load_defect_rows

ROOT = Path(__file__).resolve().parents[2]


def load_digest(load: DefectLoad) -> str:
    """sha256 over every loaded row, in order, and the load's aggregates."""
    h = hashlib.sha256()
    for row in load.rows:
        h.update(json.dumps(dataclasses.asdict(row), sort_keys=True, default=str).encode())
        h.update(b"\n")
    h.update(
        json.dumps(
            [load.n_corpus, load.capped, load.by_class, load.span_refusals, load.n_noul,
             load.n_composed],
            sort_keys=True,
        ).encode()
    )
    return h.hexdigest()


#: The digest of the rows ``load_defect_rows`` returns for ``_mixed()`` written by
#: ``_write_corpus``, from the loader as it stood at 7cc6042, before the own-repos path existed.
COMMITPACKFT_SYNTHETIC_LOAD_SHA256 = (
    "8fcb502238e3956c3064a2764a2745a124cf3837cc6315b2a4b6b498786b022b"
)

#: The same, for the real v5 defect corpus (data/pool/commitpackft-corpus-v3, 49,953 rows),
#: measured at 7cc6042 with data/pool/.build-b/v5digest.py.
V5_REAL_LOAD_SHA256 = "931c0f63b4b333e8aa575d7457f3b5be386da84019cd232e39df8a3f84935fa9"
V5_REAL_LOAD_ROWS = 49953


def test_the_commitpackft_load_is_byte_identical_to_the_loader_before_the_own_repos_path(
    tmp_path: Path,
) -> None:
    corpus, download = _write_corpus(tmp_path, _mixed())
    load = _load(tmp_path, corpus, download)
    assert load_digest(load) == COMMITPACKFT_SYNTHETIC_LOAD_SHA256


def test_the_real_v5_defect_corpus_loads_byte_identically() -> None:
    corpus = ROOT / "data" / "pool" / "commitpackft-corpus-v3"
    download = ROOT / "data" / "pool" / "commitpackft"
    if not (corpus / "examples.jsonl").is_file() or not (download / "python.jsonl").is_file():
        pytest.skip("the v5 defect corpus and its download are git-ignored, not in this tree")
    load = load_defect_rows(
        corpus, download_root=download, config=DataConfig(), repo_root=ROOT
    )
    assert len(load.rows) == V5_REAL_LOAD_ROWS
    assert load_digest(load) == V5_REAL_LOAD_SHA256


OWN_REPO = "bharathvbcr/chronosflow"
OWN_LICENCE = "owner-granted"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_download(
    download: Path, *, extra_contents: tuple[str, ...] = (), extra_old: tuple[str, ...] = (),
    languages: bool = True,
) -> None:
    """A commitpackft download holding one row whose content is not an own-repos pool file,
    plus one row per ``extra_contents`` text (as its ``new_contents``) and per ``extra_old`` text
    (as its ``old_contents``); ``languages=False`` pins no file at all."""
    download.mkdir(parents=True, exist_ok=True)
    dl = download / "python.jsonl"
    rows = [{"commit": "c" * 40, "new_file": "x.py", "license": "mit",
             "new_contents": "print(1)\n", "old_contents": ""}]
    for i, text in enumerate(extra_contents):
        rows.append({"commit": f"{i:040d}", "new_file": f"y{i}.py", "license": "mit",
                     "new_contents": text, "old_contents": ""})
    for i, text in enumerate(extra_old):
        rows.append({"commit": f"{i + 500:040d}", "new_file": f"z{i}.py", "license": "mit",
                     "new_contents": "print(2)\n", "old_contents": text})
    dl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (download / "manifest.json").write_text(
        json.dumps({"source_id": "bigcode/commitpackft",
                    "languages": {"python": {"sha256": _sha(dl.read_bytes())}} if languages
                    else {}})
    )


def _own_world(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    licence: str = OWN_LICENCE,
    pool_repo: str = OWN_REPO,
    train_slugs: tuple[str, ...] = (OWN_REPO,),
    heldout_slugs: tuple[str, ...] = ("bharathvbcr/gitpulse",),
    pin_manifest: bool = True,
    write_manifest: bool = True,
    pin_pool_manifest: bool = True,
    n: int = 6,
) -> tuple[Path, Path]:
    """An own-repos corpus: a flat pool, its manifest and files list, the pinned own-repos
    manifest, a corpus dir, and an unrelated commitpackft download. Returns (corpus dir,
    download root)."""
    from test_defect_class import BEFORE, _example

    pool_dir = root / "data" / "pool"
    corpus = pool_dir / "own-corpus"
    corpus.mkdir(parents=True)
    examples, pool_rows, files_rows = [], [], []
    for i in range(n):
        ex = _example(i, repo=pool_repo, cls="logic" if i % 2 else "stub")
        pid = f"own-swift:{pool_repo}:{ex['path']}"
        ex["pool_id"], ex["id"] = pid, f"{pid}#0"
        ex["diff"] = ex["diff"].replace("return", f"return {i} or")
        examples.append(ex)
        # The pool holds the pristine file; qd-mutate's `before` is that file, `after` the mutant.
        pool_rows.append({"id": pid, "repo": pool_repo, "path": ex["path"], "source": BEFORE})
        files_rows.append({"repo": pool_repo, "path": ex["path"], "sha256": _sha(BEFORE.encode())})
    _finish_own_world(
        pool_dir, corpus, examples, pool_rows, files_rows, monkeypatch, licence=licence,
        train_slugs=train_slugs, heldout_slugs=heldout_slugs, pin_manifest=pin_manifest,
        write_manifest=write_manifest, pin_pool_manifest=pin_pool_manifest,
    )
    download = pool_dir / "commitpackft"
    _write_download(download)
    return corpus, download


def _finish_own_world(
    pool_dir: Path,
    corpus: Path,
    examples: list[dict],
    pool_rows: list[dict],
    files_rows: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    *,
    licence: str = OWN_LICENCE,
    train_slugs: tuple[str, ...] = (OWN_REPO,),
    heldout_slugs: tuple[str, ...] = (),
    pin_manifest: bool = True,
    write_manifest: bool = True,
    pin_pool_manifest: bool = True,
    pin_corpus: bool = True,
) -> None:
    """Write the pool, files list, pool manifest, own-repos manifest and corpus from rows."""
    pool_path = pool_dir / "own-pool.jsonl"
    pool_path.write_text("".join(json.dumps(r) + "\n" for r in pool_rows), encoding="utf-8")
    files_path = pool_dir / "own-pool.files.jsonl"
    files_path.write_text("".join(json.dumps(r) + "\n" for r in files_rows), encoding="utf-8")
    own_repos = {
        "admitted": [{"repo": s.split("/")[1], "slug": s, "split": "train"} for s in train_slugs]
        + [{"repo": s.split("/")[1], "slug": s, "split": "heldout"} for s in heldout_slugs]
    }
    own_path = pool_dir / defect_class.OWN_REPOS_MANIFEST_FILE
    if write_manifest:
        own_path.write_text(json.dumps(own_repos), encoding="utf-8")
    own_sha = _sha(json.dumps(own_repos).encode())
    if pin_manifest:
        monkeypatch.setattr(defect_class, "OWN_REPOS_MANIFEST_SHA256", own_sha)
    pool_manifest = pool_dir / "own-pool.manifest.json"
    pool_manifest.write_text(
        json.dumps({
            "schema": "qd-own-swift-pool/v1",
            "source_id": "bharathvbcr/own-repositories",
            "licence": licence,
            "own_repos_manifest": {"path": "elsewhere", "sha256": own_sha},
            "files": {"file": "files.jsonl", "sha256": _sha(files_path.read_bytes()),
                      "count": len(files_rows)},
            "pool": {"file": "pool.jsonl", "records": len(pool_rows),
                     "sha256": _sha(pool_path.read_bytes())},
        }),
        encoding="utf-8",
    )
    if pin_pool_manifest:
        _pin_pool_manifest(monkeypatch, pool_manifest)
    ex_path = corpus / "examples.jsonl"
    ex_path.write_text("".join(json.dumps(e) + "\n" for e in examples), encoding="utf-8")
    (corpus / "manifest.json").write_text(
        json.dumps({
            "examples_sha256": _sha(ex_path.read_bytes()),
            "pool": {"path": "/elsewhere/data/pool/own-pool.jsonl",
                     "sha256": _sha(pool_path.read_bytes()), "records": len(pool_rows)},
            "totals": {"examples": len(examples)},
        }),
        encoding="utf-8",
    )
    if pin_corpus:
        _pin_corpora(monkeypatch, corpus)


def _pin_pool_manifest(monkeypatch: pytest.MonkeyPatch, manifest: Path) -> None:
    """Register ``manifest``'s current bytes as the pre-registered pool manifest. ``raising=False``
    so the same helper runs against a loader that has no such pin (to show a test red)."""
    monkeypatch.setattr(
        defect_class, "OWN_POOL_MANIFEST_SHA256S", frozenset({_sha(manifest.read_bytes())}),
        raising=False,
    )


def _pin_corpora(monkeypatch: pytest.MonkeyPatch, *corpora: Path) -> None:
    """Register each corpus's current ``examples.jsonl`` bytes in ``OWN_CORPUS_PINS``, as the
    pre-registration would. ``raising=False`` so the helper runs against a loader without it."""
    monkeypatch.setattr(
        defect_class, "OWN_CORPUS_PINS",
        {c.name: _sha((c / "examples.jsonl").read_bytes()) for c in corpora}, raising=False,
    )


def _rewrite(
    root: Path, corpus: Path, fn, monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
    *, repin_corpus: bool,
) -> None:
    """Apply ``fn(pool_rows, files_rows, examples)`` and re-pin every sha, as an attacker would,
    including the pool manifest's own pin (so the checks after it are the ones under test).

    An attacker cannot edit code, so the registered corpus bytes (``OWN_CORPUS_PINS``) move only
    when ``repin_corpus``: tests of checks that run after the corpus pin ask what happens to
    registered bytes and pass True; tests of the pin itself pass False."""
    pool_dir = root / "data" / "pool"
    read = lambda p: [json.loads(x) for x in p.read_text().splitlines()]  # noqa: E731
    pool_rows = read(pool_dir / "own-pool.jsonl")
    files_rows = read(pool_dir / "own-pool.files.jsonl")
    examples = read(corpus / "examples.jsonl")
    fn(pool_rows, files_rows, examples)
    pool_path = pool_dir / "own-pool.jsonl"
    pool_path.write_text("".join(json.dumps(r) + "\n" for r in pool_rows))
    files_path = pool_dir / "own-pool.files.jsonl"
    files_path.write_text("".join(json.dumps(r) + "\n" for r in files_rows))
    mp = pool_dir / "own-pool.manifest.json"
    m = json.loads(mp.read_text())
    m["files"].update(sha256=_sha(files_path.read_bytes()), count=len(files_rows))
    m["pool"].update(sha256=_sha(pool_path.read_bytes()), records=len(pool_rows))
    mp.write_text(json.dumps(m))
    _pin_pool_manifest(monkeypatch, mp)
    ex_path = corpus / "examples.jsonl"
    ex_path.write_text("".join(json.dumps(e) + "\n" for e in examples))
    cm = json.loads((corpus / "manifest.json").read_text())
    cm["examples_sha256"] = _sha(ex_path.read_bytes())
    cm["pool"].update(sha256=_sha(pool_path.read_bytes()), records=len(pool_rows))
    cm["totals"]["examples"] = len(examples)
    (corpus / "manifest.json").write_text(json.dumps(cm))
    if repin_corpus:
        _pin_corpora(monkeypatch, corpus)


def test_an_own_repos_corpus_takes_the_sources_declared_licence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, download = _own_world(tmp_path, monkeypatch)
    load = _load(tmp_path, corpus, download)
    assert len(load.rows) == 6
    assert {r.licence for r in load.rows} == {OWN_LICENCE}


def test_third_party_rows_relabelled_and_re_pinned_are_not_admitted_as_owner_granted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adversarial bypass: ids and repo renamed to torvalds/linux, every sha re-pinned. The
    manifests are self-declared, so only the pinned own-repos manifest can say whose repo it is."""
    corpus, download = _own_world(tmp_path, monkeypatch)

    def relabel(pool_rows: list, files_rows: list, examples: list) -> None:
        for r, f, e in zip(pool_rows, files_rows, examples, strict=True):
            r["id"] = e["pool_id"] = f"abc{r['path']}:x"
            e["id"] = e["pool_id"] + "#0"
            r["repo"] = f["repo"] = e["repo"] = "torvalds/linux"

    _rewrite(tmp_path, corpus, relabel, monkeypatch, repin_corpus=True)
    with pytest.raises(DefectCorpusError, match="not a train repository"):
        _load(tmp_path, corpus, download)


def test_a_pool_of_a_held_out_own_repository_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, download = _own_world(
        tmp_path, monkeypatch, pool_repo="bharathvbcr/gitpulse", train_slugs=(OWN_REPO,),
        heldout_slugs=("bharathvbcr/gitpulse",),
    )
    with pytest.raises(DefectCorpusError, match="not a train repository"):
        _load(tmp_path, corpus, download)


def test_the_pinned_own_repos_manifest_must_exist_and_hash_to_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, download = _own_world(tmp_path, monkeypatch, write_manifest=False)
    with pytest.raises(DefectCorpusError, match="own-repos manifest"):
        _load(tmp_path, corpus, download)


def test_an_own_repos_manifest_that_is_not_the_pinned_one_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, download = _own_world(tmp_path, monkeypatch, pin_manifest=False)
    with pytest.raises(DefectCorpusError, match="own-repos manifest"):
        _load(tmp_path, corpus, download)


def test_a_pool_manifest_naming_another_own_repos_manifest_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, download = _own_world(tmp_path, monkeypatch)
    mp = tmp_path / "data" / "pool" / "own-pool.manifest.json"
    m = json.loads(mp.read_text())
    m["own_repos_manifest"]["sha256"] = "0" * 64
    mp.write_text(json.dumps(m))
    _pin_pool_manifest(monkeypatch, mp)
    with pytest.raises(DefectCorpusError, match="own-repos manifest"):
        _load(tmp_path, corpus, download)


@pytest.mark.parametrize("flaw", ["extra_file", "missing_file", "duplicate_pool_row",
                                  "wrong_sha", "swapped_path", "duplicate_file_row"])
def test_the_pool_must_equal_its_files_list_row_for_row_by_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flaw: str
) -> None:
    corpus, download = _own_world(tmp_path, monkeypatch)

    def damage(pool_rows: list, files_rows: list, examples: list) -> None:
        if flaw == "extra_file":
            files_rows.append({**files_rows[0], "path": "src/never_in_the_pool.py"})
        elif flaw == "missing_file":
            files_rows.pop()
        elif flaw == "duplicate_pool_row":
            pool_rows.append({**pool_rows[0], "id": "own-swift:dup"})
            files_rows.append(dict(files_rows[0]))
        elif flaw == "wrong_sha":
            files_rows[0]["sha256"] = "1" * 64
        elif flaw == "swapped_path":
            a, b = files_rows[0], files_rows[1]
            a["path"], b["path"] = b["path"], a["path"]
            files_rows[0]["sha256"] = "2" * 64
        elif flaw == "duplicate_file_row":
            files_rows[1] = dict(files_rows[0])

    _rewrite(tmp_path, corpus, damage, monkeypatch, repin_corpus=True)
    with pytest.raises(DefectCorpusError, match="files list"):
        _load(tmp_path, corpus, download)


def test_an_own_repos_pool_whose_manifest_licence_is_not_the_sources_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, download = _own_world(tmp_path, monkeypatch, licence="mit")
    with pytest.raises(DefectCorpusError, match="licence"):
        _load(tmp_path, corpus, download)


def test_an_own_repos_pool_whose_files_list_changed_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, download = _own_world(tmp_path, monkeypatch)
    files = tmp_path / "data" / "pool" / "own-pool.files.jsonl"
    files.write_text(files.read_text() + "\n")
    with pytest.raises(DefectCorpusError, match="changed after it was recorded"):
        _load(tmp_path, corpus, download)


def test_an_own_repos_pool_with_a_pool_id_in_the_commitpackft_download_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One corpus is one source: a pool file whose bytes the download also holds would have two
    licences. Compared by content, since pool ids and download keys are different key spaces."""
    from test_defect_class import BEFORE

    corpus, download = _own_world(tmp_path, monkeypatch)
    _write_download(download, extra_contents=(BEFORE,))
    with pytest.raises(DefectCorpusError, match="one corpus is one source"):
        _load(tmp_path, corpus, download)


_VARIANTS = {
    "crlf": lambda t: t.replace("\n", "\r\n"),
    "lone_cr": lambda t: t.replace("\n", "\r"),
    "bom": lambda t: "﻿" + t,
    "no_final_newline": lambda t: t.rstrip("\n"),
    "extra_final_newlines": lambda t: t + "\n\n\n",
    "tabs_for_indent": lambda t: t.replace("    ", "\t"),
    "trailing_blanks": lambda t: t.replace("\n", "  \t\n"),
}


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
@pytest.mark.parametrize("side", ["new_contents", "old_contents"])
def test_a_download_copy_that_differs_only_in_whitespace_is_still_a_clash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str, side: str
) -> None:
    """The comparison is by normalised content (BOM, line endings, final newlines and the
    whitespace on a line), on both sides of a download commit."""
    from test_defect_class import BEFORE

    corpus, download = _own_world(tmp_path, monkeypatch)
    text = _VARIANTS[variant](BEFORE)
    if side == "new_contents":
        _write_download(download, extra_contents=(text,))
    else:
        _write_download(download, extra_old=(text,))
    with pytest.raises(DefectCorpusError, match="one corpus is one source"):
        _load(tmp_path, corpus, download)


def test_a_download_file_with_different_tokens_is_not_a_clash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_defect_class import BEFORE

    corpus, download = _own_world(tmp_path, monkeypatch)
    _write_download(download, extra_contents=(BEFORE.replace("x + a", "x * a"),))
    assert len(_load(tmp_path, corpus, download).rows) == 6


def test_an_example_that_does_not_carry_its_pool_records_bytes_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The card's goal: the admitted text must be the owner's. Third-party text in the examples,
    with every manifest re-pinned, was admitted as owner-granted before this check."""
    corpus, download = _own_world(tmp_path, monkeypatch)
    evil = "// (c) Third Party GPL\nfunc evil() {}\n"

    def swap(pool_rows: list, files_rows: list, examples: list) -> None:
        for e in examples:
            e["before"] = e["after"] = evil

    _rewrite(tmp_path, corpus, swap, monkeypatch, repin_corpus=True)
    with pytest.raises(DefectCorpusError, match="pool record's source"):
        _load(tmp_path, corpus, download)


@pytest.mark.parametrize("side", ["before", "after"])
def test_a_clean_example_must_carry_the_pool_bytes_on_both_sides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, side: str
) -> None:
    from test_defect_class import BEFORE

    corpus, download = _own_world(tmp_path, monkeypatch)

    def clean_with_foreign_side(pool_rows: list, files_rows: list, examples: list) -> None:
        e = examples[0]
        e.update({"class": "clean", "operator": "clean", "span": None, "before": BEFORE,
                  "after": BEFORE})
        e[side] = "// (c) Third Party GPL\n"

    _rewrite(tmp_path, corpus, clean_with_foreign_side, monkeypatch, repin_corpus=True)
    with pytest.raises(DefectCorpusError, match="pool record's source"):
        _load(tmp_path, corpus, download)


def test_an_example_is_tied_to_the_pool_source_as_qd_mutate_normalises_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``normalize.rs``: one leading BOM and CRLF/CR become LF, nothing else. A pool file with
    CRLF and a BOM is carried by an example holding the normalised text, and only that."""
    corpus, download = _own_world(tmp_path, monkeypatch)
    raw = "﻿def f():\r\n    return 1\r\rdef g():\r\n    return 0\r\n"
    normalised = "def f():\n    return 1\n\ndef g():\n    return 0\n"

    def crlf_pool(pool_rows: list, files_rows: list, examples: list) -> None:
        pool_rows[0]["source"] = raw
        files_rows[0]["sha256"] = _sha(raw.encode())
        examples[0]["before"] = normalised

    _rewrite(tmp_path, corpus, crlf_pool, monkeypatch, repin_corpus=True)
    assert len(_load(tmp_path, corpus, download).rows) == 6

    def raw_before(pool_rows: list, files_rows: list, examples: list) -> None:
        examples[0]["before"] = raw

    _rewrite(tmp_path, corpus, raw_before, monkeypatch, repin_corpus=True)
    with pytest.raises(DefectCorpusError, match="pool record's source"):
        _load(tmp_path, corpus, download)


def test_the_pool_manifest_is_read_once_and_hashed_from_the_bytes_that_were_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest swapped between a parse and a re-read must not pass on the strength of the
    bytes that were hashed: one read feeds both."""
    _own_world(tmp_path, monkeypatch)
    pool_path = tmp_path / "data" / "pool" / "own-pool.jsonl"
    manifest = pool_path.with_name("own-pool.manifest.json")
    reads: list[str] = []
    real_bytes, real_text = Path.read_bytes, Path.read_text

    def count_bytes(self: Path) -> bytes:
        reads.append(self.name)
        return real_bytes(self)

    def count_text(self: Path, *a: object, **k: object) -> str:
        reads.append(self.name)
        return real_text(self, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_bytes", count_bytes)
    monkeypatch.setattr(Path, "read_text", count_text)
    found = defect_class._own_repos_pool_manifest(pool_path)
    assert found is not None
    parsed, digest = found
    assert digest == _sha(real_bytes(manifest))
    assert parsed["source_id"] == "bharathvbcr/own-repositories"
    assert reads.count(manifest.name) == 1


def test_a_download_row_keyed_like_a_pool_id_but_with_other_content_is_not_a_clash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old check compared ids across key spaces that cannot meet; content is what counts."""
    corpus, download = _own_world(tmp_path, monkeypatch)
    _write_download(download, extra_contents=("something else entirely\n",))
    assert len(_load(tmp_path, corpus, download).rows) == 6


def test_an_empty_languages_download_manifest_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest pinning no file makes any join over it vacuously clean: not a pass."""
    corpus, download = _own_world(tmp_path, monkeypatch)
    _write_download(download, languages=False)
    with pytest.raises(DefectCorpusError, match="pins no 'languages'"):
        _load(tmp_path, corpus, download)


def test_a_commitpackft_load_over_an_empty_languages_download_manifest_is_refused(
    tmp_path: Path,
) -> None:
    corpus, download = _write_corpus(tmp_path, _mixed())
    (download / "manifest.json").write_text(
        json.dumps({"source_id": "bigcode/commitpackft", "languages": {}})
    )
    with pytest.raises(DefectCorpusError, match="pins no 'languages'"):
        _load(tmp_path, corpus, download)


UNSAFE_PATHS = [
    "/etc/passwd", "", "src/../../x.swift", "..", "a\x00b.swift", "src\\a.swift",
    "a//b.swift", "a/b/", "./a.swift", "a/./b.swift", "a/ ../b.swift", "..\t", "a/ /b.swift",
    "a/b/..\n/c.swift",
]


@pytest.mark.parametrize("bad", UNSAFE_PATHS, ids=repr)
@pytest.mark.parametrize("where", ["both", "pool_only", "files_only"])
def test_an_unsafe_pool_path_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str, where: str
) -> None:
    corpus, download = _own_world(tmp_path, monkeypatch)

    def poison(pool_rows: list, files_rows: list, examples: list) -> None:
        if where != "files_only":
            pool_rows[0]["path"] = bad
        if where != "pool_only":
            files_rows[0]["path"] = bad

    _rewrite(tmp_path, corpus, poison, monkeypatch, repin_corpus=True)
    with pytest.raises(DefectCorpusError, match="unsafe path"):
        _load(tmp_path, corpus, download)


def test_a_dotted_but_plain_path_is_still_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Narrow, not blanket: ``..`` is refused as a segment, not as a substring."""
    corpus, download = _own_world(tmp_path, monkeypatch)

    def rename(pool_rows: list, files_rows: list, examples: list) -> None:
        pool_rows[0]["path"] = files_rows[0]["path"] = "src/v1..2/a.swift"

    _rewrite(tmp_path, corpus, rename, monkeypatch, repin_corpus=True)
    assert len(_load(tmp_path, corpus, download).rows) == 6


def test_a_lookalike_unicode_segment_is_an_opaque_name_not_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Documented residual: U+FF0E FULLWIDTH FULL STOP is not '.', so it is not refused."""
    corpus, download = _own_world(tmp_path, monkeypatch)

    def rename(pool_rows: list, files_rows: list, examples: list) -> None:
        pool_rows[0]["path"] = files_rows[0]["path"] = f"src/{chr(0xFF0E) * 2}/a.swift"

    _rewrite(tmp_path, corpus, rename, monkeypatch, repin_corpus=True)
    assert len(_load(tmp_path, corpus, download).rows) == 6


def test_the_loader_itself_refuses_a_pool_manifest_the_registration_does_not_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard lives in the loader, not in the Arm B driver: a re-pinned, self-consistent pool
    whose manifest bytes are not the pre-registered ones is refused."""
    corpus, download = _own_world(tmp_path, monkeypatch, pin_pool_manifest=False)
    with pytest.raises(DefectCorpusError, match="pool manifest"):
        _load(tmp_path, corpus, download)


def test_a_pool_file_that_is_not_the_one_its_manifest_records_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, download = _own_world(tmp_path, monkeypatch)
    mp = tmp_path / "data" / "pool" / "own-pool.manifest.json"
    m = json.loads(mp.read_text())
    m["pool"]["sha256"] = "3" * 64
    mp.write_text(json.dumps(m))
    _pin_pool_manifest(monkeypatch, mp)
    with pytest.raises(DefectCorpusError, match="changed after it was recorded"):
        _load(tmp_path, corpus, download)


def test_the_pinned_pool_manifest_is_the_one_the_preregistration_pins() -> None:
    prereg = json.loads(
        (ROOT / "campaign" / "v6-arm-b-mining-preregistered.json").read_text(encoding="utf-8")
    )
    pins = {
        c["pool_manifest"]["sha256"]
        for c in _corpora(prereg)
        if c.get("own_repos_train")
    }
    assert pins and pins == set(defect_class.OWN_POOL_MANIFEST_SHA256S)


def _corpora(node: object) -> list[dict]:
    """Every dict in ``node`` that names a ``pool_manifest``."""
    found: list[dict] = []
    if isinstance(node, dict):
        if "pool_manifest" in node:
            found.append(node)
        for v in node.values():
            found.extend(_corpora(v))
    elif isinstance(node, list):
        for v in node:
            found.extend(_corpora(v))
    return found


def test_a_clash_check_that_cannot_run_refuses_rather_than_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent download proved nothing, so the check did not run: that is not a pass."""
    corpus, download = _own_world(tmp_path, monkeypatch)
    (download / "manifest.json").unlink()
    with pytest.raises(DefectCorpusError, match="could not run"):
        _load(tmp_path, corpus, download)
    (download / "manifest.json").write_text("{not json")
    with pytest.raises(DefectCorpusError, match="could not run"):
        _load(tmp_path, corpus, download)


# -- the text a row carries is tied to the registered corpus, not to the pool alone ----------
# Fable's ruling (AUDIT/board-sweep-2026-10-13/fable-own-repos-diff-tie-ruling.md): an own-repos
# corpus is admitted only if its bytes are ones the pre-registration pins (OWN_CORPUS_PINS).

EVIL = "// (c) Third Party GPL\nfunc evil() {}\n"
EVIL_BODY = "@@ -5,2 +5,2 @@\n // (c) Third Party GPL\n-func evil() {}\n+func evil() {}\n"


def _own_composed_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, evil_fillers: bool = False,
    n_rows: int = 1, pin: tuple[str, ...] = ("base", "composed"),
) -> tuple[Path, Path, Path]:
    """An own base corpus (``own-corpus``) and a ``qd-compose/v1`` corpus over it
    (``own-composed``), both under ``tmp_path/data/pool``. Pool repo is a train repo, since a
    composed row re-derives its split. ``pin`` names which are registered in OWN_CORPUS_PINS.
    Returns (base, composed, download)."""
    from test_defect_class import CLEAN_DIFF, _composed_row, _repos_in

    repo = _repos_in("train", 1)[0]
    base, download = _own_world(
        tmp_path, monkeypatch, pool_repo=repo, train_slugs=(repo,), heldout_slugs=()
    )
    files = [
        {"pool_id": f"own-swift:{repo}:src/m{i}.py", "repo": repo, "path": f"src/m{i}.py"}
        for i in range(3)
    ]
    rows = []
    for index in range(n_rows):
        row = _composed_row(files, needle=1, index=index)
        if evil_fillers:
            row["diff"] = row["diff"].replace(CLEAN_DIFF, EVIL_BODY)
            assert EVIL_BODY in row["diff"]
        rows.append(row)
    composed = base.parent / "own-composed"
    composed.mkdir()
    ex = composed / "examples.jsonl"
    ex.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    cfg = DataConfig()
    (composed / "manifest.json").write_text(json.dumps({
        "schema": "qd-compose/v1",
        "base_corpus": {"name": base.name, "examples_sha256": base_manifest["examples_sha256"]},
        "split_params": {"seed": cfg.seed, "train_fraction": cfg.train_fraction,
                         "val_fraction": cfg.val_fraction},
        "pool": base_manifest["pool"],
        "examples_sha256": _sha(ex.read_bytes()),
        "totals": {"examples": len(rows)},
    }), encoding="utf-8")
    _pin_corpora(
        monkeypatch, *[{"base": base, "composed": composed}[k] for k in pin]
    )
    return base, composed, download


def test_a_plain_row_whose_diff_is_not_registered_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repro (a): ``before`` is the pool file, so the row-level tie holds, but the hunk the model
    reads is third-party text. The corpus bytes are not the registered ones."""
    corpus, download = _own_world(tmp_path, monkeypatch)

    def poison(pool_rows: list, files_rows: list, examples: list) -> None:
        examples[0]["diff"] = (
            "@@ -1,3 +1,3 @@\n // (c) Third Party GPL\n def f(x):\n-    return x + a\n"
            "+    return evil(x)\n"
        )

    _rewrite(tmp_path, corpus, poison, monkeypatch, repin_corpus=False)
    with pytest.raises(DefectCorpusError, match="OWN_CORPUS_PINS"):
        _load(tmp_path, corpus, download)


def test_a_composed_corpus_as_pinned_is_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Characterisation (not red-first): the fixture world loads when both corpora are
    registered, so the refusals below are the pin's doing."""
    _, composed, download = _own_composed_world(tmp_path, monkeypatch)
    load = _load(tmp_path, composed, download)
    assert load.n_composed == 1 and len(load.rows) == 7


def test_a_composed_row_with_unregistered_block_bodies_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repro (b): composed rows have no ``before``; their block bodies were free text."""
    _, composed, download = _own_composed_world(
        tmp_path, monkeypatch, evil_fillers=True, pin=("base",)
    )
    with pytest.raises(DefectCorpusError, match="OWN_CORPUS_PINS"):
        _load(tmp_path, composed, download)


def _own_noul_dir(tmp_path: Path, base: Path) -> Path:
    """A ``qd-noul-rows`` corpus over the own pool: one scrambled row, free-text diff."""
    repo = json.loads((base / "examples.jsonl").read_text().splitlines()[0])["repo"]
    noul = base.parent / "own-noul"
    noul.mkdir()
    row = {
        "id": "noul:own:0", "repo": repo, "path": "src/m0.py", "language": "swift",
        "class": "noul", "noul_source": "scrambled", "licence": OWN_LICENCE,
        "pool_id": f"own-swift:{repo}:src/m0.py", "diff": "@@ -1,1 +1,1 @@\n-" + EVIL + "+" + EVIL,
    }
    ex = noul / "examples.jsonl"
    ex.write_text(json.dumps(row) + "\n", encoding="utf-8")
    base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    cfg = DataConfig()
    (noul / "manifest.json").write_text(json.dumps({
        "examples_sha256": _sha(ex.read_bytes()), "pool": base_manifest["pool"],
        "split": {"seed": cfg.seed, "train_fraction": cfg.train_fraction,
                  "val_fraction": cfg.val_fraction},
        "totals": {"examples": 1},
    }), encoding="utf-8")
    return noul


def test_a_noul_scrambled_row_over_an_own_pool_is_refused_unless_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repro (c): a noul row is licensed from the pool and its diff was free text."""
    from test_defect_class import _repos_in

    repo = _repos_in("train", 1)[0]
    base, download = _own_world(
        tmp_path, monkeypatch, pool_repo=repo, train_slugs=(repo,), heldout_slugs=()
    )
    noul = _own_noul_dir(tmp_path, base)
    with pytest.raises(DefectCorpusError, match="OWN_CORPUS_PINS"):
        _load(tmp_path, base, download, noul_dir=noul)
    _pin_corpora(monkeypatch, base, noul)
    assert _load(tmp_path, base, download, noul_dir=noul).n_noul == 1


def _materialise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, keep: tuple[int, ...] = (0, 2)
) -> tuple[Path, Path, Path]:
    """A pinned own composed corpus of 3 rows (``data/pool/own-composed``) and the form
    ``tools/arm_b_mine.py materialise`` writes for it: ``out/own-composed`` holding the ``keep``
    raw lines byte for byte, over an empty ``own-composed-planbase``. Returns
    (materialised dir, pinned source dir, download)."""
    base, composed, download = _own_composed_world(tmp_path, monkeypatch, n_rows=3)
    lines = (composed / "examples.jsonl").read_bytes().split(b"\n")[:3]
    planbase = base.parent / "own-composed-planbase"
    planbase.mkdir()
    empty = _sha(b"")
    base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    (planbase / "examples.jsonl").write_bytes(b"")
    (planbase / "manifest.json").write_text(json.dumps(
        {**base_manifest, "examples_sha256": empty, "totals": {"examples": 0}}
    ))
    # `materialise` names its output after the corpus (`own-swift-composed-v2`), which is not the
    # pinned directory's name (`own-swift-composed-v2-prefixed`): the proof must not depend on it.
    out = tmp_path / "out" / "own-composed-materialised"
    out.mkdir(parents=True)
    body = b"".join(lines[i] + b"\n" for i in keep)
    (out / "examples.jsonl").write_bytes(body)
    manifest = json.loads((composed / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        examples_sha256=_sha(body), totals={"examples": len(keep)},
        base_corpus={"name": planbase.name, "examples_sha256": empty},
        selected_by_plan={
            "source_examples_sha256": _sha((composed / "examples.jsonl").read_bytes()),
            "kept": len(keep),
        },
    )
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out, composed, download


def test_a_materialised_subset_of_a_pinned_corpus_is_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Characterisation, not red-first (admitted before and after): the materialise path
    survives the pin, 2 rows over an empty planbase."""
    out, _, download = _materialise(tmp_path, monkeypatch)
    load = _load(tmp_path, out, download)
    assert len(load.rows) == 2 and load.n_composed == 2


def test_a_materialised_row_that_is_not_a_line_of_the_pinned_corpus_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _, download = _materialise(tmp_path, monkeypatch)
    ex = out / "examples.jsonl"
    body = ex.read_bytes().replace(b'"est_tokens": 30', b'"est_tokens": 31', 1)
    assert body != ex.read_bytes()
    ex.write_bytes(body)
    m = json.loads((out / "manifest.json").read_text())
    m["examples_sha256"] = _sha(body)
    (out / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(DefectCorpusError, match="not a line of the pinned corpus"):
        _load(tmp_path, out, download)


def test_a_materialised_corpus_whose_kept_count_disagrees_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every line is a pinned line, but the manifest says 3 were kept and 2 are present. (Three
    present would be the pinned file itself, admitted by its sha before any subset proof.)"""
    out, _, download = _materialise(tmp_path, monkeypatch)
    m = json.loads((out / "manifest.json").read_text())
    m["selected_by_plan"]["kept"] = 3
    (out / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(DefectCorpusError, match="must agree"):
        _load(tmp_path, out, download)


def test_a_materialised_corpus_whose_source_cannot_be_read_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, composed, download = _materialise(tmp_path, monkeypatch)
    src = composed / "examples.jsonl"
    original = src.read_bytes()
    src.write_bytes(original + b"\n")  # exists, hashes to another value
    with pytest.raises(DefectCorpusError, match="could not be proved against the pinned corpus"):
        _load(tmp_path, out, download)
    src.unlink()
    with pytest.raises(DefectCorpusError, match="could not be proved against the pinned corpus"):
        _load(tmp_path, out, download)


def test_an_empty_own_corpus_is_admitted_without_a_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero rows is zero text (a planbase, or a materialised corpus that kept nothing).
    Characterisation: admitted before and after."""
    corpus, download = _own_world(tmp_path, monkeypatch)
    _rewrite(tmp_path, corpus, lambda p, f, e: e.clear(), monkeypatch, repin_corpus=False)
    assert (corpus / "examples.jsonl").stat().st_size == 0
    assert _load(tmp_path, corpus, download).rows == ()


def test_the_pinned_own_corpora_are_the_ones_the_preregistration_pins() -> None:
    """The tripwire: every ``own_repos_train`` corpus and its structured base corpus, by
    directory name, equal ``OWN_CORPUS_PINS``; ``superseded_pin`` is never in it. A corpus
    that names a base only in prose carries no pin the loader could mirror."""
    prereg = json.loads(
        (ROOT / "campaign" / "v6-arm-b-mining-preregistered.json").read_text(encoding="utf-8")
    )
    want: dict[str, str] = {}
    missing_base: list[str] = []
    for c in _corpora(prereg):
        if not c.get("own_repos_train"):
            continue
        want[Path(c["path"]).parent.name] = c["sha256"]
        base = c.get("base_corpus")
        if isinstance(base, dict) and base.get("path") and base.get("sha256"):
            want[Path(base["path"]).parent.name] = base["sha256"]
        else:
            missing_base.append(str(c.get("name")))
    assert want, "the pre-registration pins no own-repos corpus"
    assert not missing_base, (
        f"corpora {missing_base} carry no structured base_corpus {{path, sha256}} pin"
    )
    assert want == dict(defect_class.OWN_CORPUS_PINS)
