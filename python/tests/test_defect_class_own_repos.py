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
    download: Path, *, extra_contents: tuple[str, ...] = (), languages: bool = True
) -> None:
    """A commitpackft download holding one row whose content is not an own-repos pool file,
    plus one row per ``extra_contents`` text; ``languages=False`` pins no file at all."""
    download.mkdir(parents=True, exist_ok=True)
    dl = download / "python.jsonl"
    rows = [{"commit": "c" * 40, "new_file": "x.py", "license": "mit",
             "new_contents": "print(1)\n"}]
    for i, text in enumerate(extra_contents):
        rows.append({"commit": f"{i:040d}", "new_file": f"y{i}.py", "license": "mit",
                     "new_contents": text})
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
    from test_defect_class import AFTER, _example

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
        pool_rows.append({"id": pid, "repo": pool_repo, "path": ex["path"], "source": AFTER})
        files_rows.append({"repo": pool_repo, "path": ex["path"], "sha256": _sha(AFTER.encode())})
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


def _pin_pool_manifest(monkeypatch: pytest.MonkeyPatch, manifest: Path) -> None:
    """Register ``manifest``'s current bytes as the pre-registered pool manifest. ``raising=False``
    so the same helper runs against a loader that has no such pin (to show a test red)."""
    monkeypatch.setattr(
        defect_class, "OWN_POOL_MANIFEST_SHA256S", frozenset({_sha(manifest.read_bytes())}),
        raising=False,
    )


def _rewrite(
    root: Path, corpus: Path, fn, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
) -> None:
    """Apply ``fn(pool_rows, files_rows, examples)`` and re-pin every sha, as an attacker would,
    including the pool manifest's own pin (so the checks after it are the ones under test)."""
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

    _rewrite(tmp_path, corpus, relabel, monkeypatch)
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

    _rewrite(tmp_path, corpus, damage, monkeypatch)
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
    from test_defect_class import AFTER

    corpus, download = _own_world(tmp_path, monkeypatch)
    _write_download(download, extra_contents=(AFTER,))
    with pytest.raises(DefectCorpusError, match="one corpus is one source"):
        _load(tmp_path, corpus, download)


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


UNSAFE_PATHS = ["/etc/passwd", "", "src/../../x.swift", "..", "a\x00b.swift", "src\\a.swift"]


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

    _rewrite(tmp_path, corpus, poison, monkeypatch)
    with pytest.raises(DefectCorpusError, match="unsafe path"):
        _load(tmp_path, corpus, download)


def test_a_dotted_but_plain_path_is_still_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Narrow, not blanket: ``..`` is refused as a segment, not as a substring."""
    corpus, download = _own_world(tmp_path, monkeypatch)

    def rename(pool_rows: list, files_rows: list, examples: list) -> None:
        pool_rows[0]["path"] = files_rows[0]["path"] = "src/v1..2/a.swift"

    _rewrite(tmp_path, corpus, rename, monkeypatch)
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
