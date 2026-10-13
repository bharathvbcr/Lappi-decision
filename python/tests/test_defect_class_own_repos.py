"""``load_defect_rows`` for a corpus whose pool is the human's own repositories, and the proof
that the commitpackft (v5) path did not move when that was added.

The own-repos path takes its licence from the source's declaration
(``source_by_id(OWN_REPOS_SOURCE_ID).declared_licence``) through the same three checks
``_load_own_prose`` makes, and never reads the commitpackft download. One corpus is one source.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from test_defect_class import _load, _mixed, _write_corpus

from qd_data.config import DataConfig
from qd_data.defect_class import DefectLoad, load_defect_rows

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


def _own_world(
    root: Path,
    *,
    licence: str = OWN_LICENCE,
    files_sha_override: str | None = None,
    n: int = 6,
) -> tuple[Path, Path]:
    """An own-repos corpus: a flat pool, its flat manifest and files list, and the corpus dir.
    Returns (corpus dir, a download root that does not exist: the own path must not need it)."""
    from test_defect_class import AFTER, _example

    pool_dir = root / "data" / "pool"
    corpus = pool_dir / "own-corpus"
    corpus.mkdir(parents=True)
    examples = []
    pool_rows = []
    files_rows = []
    for i in range(n):
        ex = _example(i, repo=OWN_REPO, cls="logic" if i % 2 else "stub")
        pid = f"own-swift:{OWN_REPO}:{ex['path']}"
        ex["pool_id"], ex["id"] = pid, f"{pid}#0"
        ex["diff"] = ex["diff"].replace("return", f"return {i} or")
        examples.append(ex)
        pool_rows.append({"id": pid, "repo": OWN_REPO, "path": ex["path"], "source": AFTER})
        files_rows.append(
            {"repo": OWN_REPO, "path": ex["path"],
             "sha256": hashlib.sha256(AFTER.encode()).hexdigest()}
        )
    pool_path = pool_dir / "own-pool.jsonl"
    pool_path.write_text("".join(json.dumps(r) + "\n" for r in pool_rows), encoding="utf-8")
    files_path = pool_dir / "own-pool.files.jsonl"
    files_path.write_text("".join(json.dumps(r) + "\n" for r in files_rows), encoding="utf-8")
    (pool_dir / "own-pool.manifest.json").write_text(
        json.dumps({
            "schema": "qd-own-swift-pool/v1",
            "source_id": "bharathvbcr/own-repositories",
            "licence": licence,
            "files": {
                "file": "files.jsonl",
                "sha256": files_sha_override or hashlib.sha256(files_path.read_bytes()).hexdigest(),
                "count": len(files_rows),
            },
            "pool": {"file": "pool.jsonl", "records": len(pool_rows)},
        }),
        encoding="utf-8",
    )
    ex_path = corpus / "examples.jsonl"
    ex_path.write_text("".join(json.dumps(e) + "\n" for e in examples), encoding="utf-8")
    (corpus / "manifest.json").write_text(
        json.dumps({
            "examples_sha256": hashlib.sha256(ex_path.read_bytes()).hexdigest(),
            "pool": {
                "path": "/elsewhere/data/pool/own-pool.jsonl",
                "sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
                "records": len(pool_rows),
            },
            "totals": {"examples": len(examples)},
        }),
        encoding="utf-8",
    )
    return corpus, pool_dir / "commitpackft-absent"


def test_an_own_repos_corpus_takes_the_sources_declared_licence_and_never_needs_the_download(
    tmp_path: Path,
) -> None:
    corpus, no_download = _own_world(tmp_path)
    assert not no_download.exists()
    load = _load(tmp_path, corpus, no_download)
    assert len(load.rows) == 6
    assert {r.licence for r in load.rows} == {OWN_LICENCE}


def test_an_own_repos_pool_whose_manifest_licence_is_not_the_sources_is_refused(
    tmp_path: Path,
) -> None:
    from qd_data.defect_class import DefectCorpusError

    corpus, no_download = _own_world(tmp_path, licence="mit")
    with pytest.raises(DefectCorpusError, match="licence"):
        _load(tmp_path, corpus, no_download)


def test_an_own_repos_pool_whose_files_list_changed_is_refused(tmp_path: Path) -> None:
    from qd_data.defect_class import DefectCorpusError

    corpus, no_download = _own_world(tmp_path, files_sha_override="0" * 64)
    with pytest.raises(DefectCorpusError, match="changed after it was recorded"):
        _load(tmp_path, corpus, no_download)


def test_an_own_repos_pool_with_a_pool_id_in_the_commitpackft_download_is_refused(
    tmp_path: Path,
) -> None:
    """One corpus is one source: a pool id the download also holds would have two licences."""
    from qd_data.defect_class import DefectCorpusError

    corpus, download = _own_world(tmp_path)
    download.mkdir()
    pid = f"own-swift:{OWN_REPO}:src/m0.py"
    commit, new_file = pid.split(":", 1)
    dl = download / "python.jsonl"
    dl.write_text(json.dumps({"commit": commit, "new_file": new_file, "license": "mit"}) + "\n")
    (download / "manifest.json").write_text(
        json.dumps({
            "source_id": "bigcode/commitpackft",
            "languages": {"python": {"sha256": hashlib.sha256(dl.read_bytes()).hexdigest()}},
        })
    )
    with pytest.raises(DefectCorpusError, match="one corpus is one source"):
        _load(tmp_path, corpus, download)


def test_an_own_repos_pool_beside_an_unrelated_download_still_loads(tmp_path: Path) -> None:
    corpus, download = _own_world(tmp_path)
    download.mkdir()
    dl = download / "python.jsonl"
    dl.write_text(json.dumps({"commit": "c" * 40, "new_file": "x.py", "license": "mit"}) + "\n")
    (download / "manifest.json").write_text(
        json.dumps({
            "source_id": "bigcode/commitpackft",
            "languages": {"python": {"sha256": hashlib.sha256(dl.read_bytes()).hexdigest()}},
        })
    )
    assert {r.licence for r in _load(tmp_path, corpus, download).rows} == {OWN_LICENCE}
