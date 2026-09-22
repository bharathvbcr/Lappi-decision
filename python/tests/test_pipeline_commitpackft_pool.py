"""The shard pipeline's second code source: the plan's own commitpackft download.

``tools/real_tokenizer_pipeline.py`` built every FT shard set from this repository's history
rendered AS commitpackft rows. That measures the tokenizer path on real code, but trains on
one repository. ``--commitpackft`` reads the download the plan names instead, and these pin
the three ways that read could quietly be wrong:

* a file that changed after download holds rows nobody can name -- refused, not read;
* a manifest for some other source is not commitpackft -- refused;
* a cap taken as a PREFIX of files read in language order is a one-language sample -- so
  the cap is a sha256-ordered sample, reproducible, and the total is returned beside it.

Torch-free, like the module's other corpus tests.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_tokenizer_pipeline as pipeline  # noqa: E402


def _row(commit: str, lang: str, n: int) -> dict[str, str]:
    """One raw commitpackft row, in the upstream spelling (``license``, not ``licence``)."""
    return {
        "commit": commit,
        "repos": f"example/{lang}-{n}",
        "old_file": f"src/f{n}.{lang[:2]}",
        "new_file": f"src/f{n}.{lang[:2]}",
        "old_contents": f"x = {n}\n",
        "new_contents": f"x = {n + 1}\n",
        "subject": f"bump {n}",
        "message": f"bump {n}\n",
        "lang": lang.capitalize(),
        "license": "mit",
    }


def _download(tmp_path: Path, per_lang: dict[str, int]) -> Path:
    """A directory shaped like data/pool/commitpackft: <lang>.jsonl plus manifest.json."""
    root = tmp_path / "commitpackft"
    root.mkdir()
    languages = {}
    for lang, n in per_lang.items():
        body = "".join(
            json.dumps(_row(f"{lang}{i:04d}", lang, i)) + "\n" for i in range(n)
        ).encode("utf-8")
        (root / f"{lang}.jsonl").write_bytes(body)
        languages[lang] = {"sha256": hashlib.sha256(body).hexdigest(), "rows": n}
    (root / "manifest.json").write_text(
        json.dumps({"source_id": "bigcode/commitpackft", "languages": languages}),
        encoding="utf-8",
    )
    return root


def test_every_row_is_read_when_the_cap_does_not_bind(tmp_path) -> None:
    root = _download(tmp_path, {"go": 3, "python": 5})
    rows, capped, total = pipeline.commitpackft_pool_rows(root, max_pairs=100)
    assert (len(rows), capped, total) == (8, False, 8)
    assert {r.licence for r in rows} == {"mit"}, "upstream `license` read as `licence`"


def test_a_cap_is_a_sample_across_languages_not_the_first_file(tmp_path) -> None:
    """Files are read in language order, so a prefix of 20 would be all Go. The sample is
    ordered by a hash of each row's identity instead."""
    root = _download(tmp_path, {"go": 40, "python": 40})
    rows, capped, total = pipeline.commitpackft_pool_rows(root, max_pairs=20)
    assert capped and total == 80 and len(rows) == 20
    langs = {r.lang for r in rows}
    assert langs == {"Go", "Python"}, f"a capped read took one language only: {langs}"


def test_the_sample_is_reproducible(tmp_path) -> None:
    root = _download(tmp_path, {"go": 30, "rust": 30})
    first, _, _ = pipeline.commitpackft_pool_rows(root, max_pairs=10)
    second, _, _ = pipeline.commitpackft_pool_rows(root, max_pairs=10)
    assert [r.commit for r in first] == [r.commit for r in second]


def test_a_file_that_changed_after_download_is_refused(tmp_path) -> None:
    root = _download(tmp_path, {"go": 3, "python": 3})
    with (root / "python.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_row("late", "python", 99)) + "\n")
    with pytest.raises(SystemExit, match="changed after it was downloaded"):
        pipeline.commitpackft_pool_rows(root, max_pairs=100)


def test_a_manifest_for_another_source_is_refused(tmp_path) -> None:
    root = _download(tmp_path, {"go": 1})
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["source_id"] = "somebody/else"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="not bigcode/commitpackft"):
        pipeline.commitpackft_pool_rows(root, max_pairs=10)


def test_the_row_carries_the_reason_the_run_measured(tmp_path, monkeypatch) -> None:
    """`main` wrote one literal whatever `run` had read, so row 0c3fd775 -- the first shard
    set built from the download -- says its corpus was "drawn from this repository alone"
    beside a recipe pinning four commitpackft sha256s. The row now carries the reason `run`
    returns. `run` is replaced because it loads the real tokenizer; what is under test is
    that nothing between it and the ledger substitutes a sentence of its own."""
    root = _download(tmp_path, {"go": 3, "python": 5})
    ref = tmp_path / "refs-main"
    ref.write_text("b" * 40, encoding="utf-8")
    monkeypatch.setattr(pipeline, "MODEL_REF", ref)
    read, _, held = pipeline.commitpackft_pool_rows(root, max_pairs=4)
    reason = pipeline.quick_reason_for(commitpackft_rows=(len(read), held))

    def measured(**_: object) -> pipeline.Measured:
        return pipeline.Measured(
            metrics={}, gates={}, data_snapshot_hash="d" * 64, tokenizer_hash="e" * 64,
            notes="run replaced by the test", quick_reason=reason,
        )

    monkeypatch.setattr(pipeline, "run", measured)
    ledger = tmp_path / "ledger.jsonl"
    assert pipeline.main([
        "--out", str(tmp_path / "out"), "--max-pairs", "4",
        "--commitpackft", str(root), "--ledger", str(ledger),
    ]) == 0
    row = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert row["quick"] is True
    assert row["quick_reason"] == reason
    assert "4-of-8 sha256-ordered sample" in row["quick_reason"]


def test_the_real_download_matches_its_manifest() -> None:
    """The directory the plan's FT shards would be built from, checked as it is on disk.
    Skipped where it was never downloaded; never skipped where it was and has drifted.

    "Never downloaded" is read off the data files, not the manifest: `.gitignore` tracks
    `data/pool/**/manifest.json` and ignores everything beside it, so every clone has the
    manifest and none has the files. Keyed on the manifest, this failed on a fresh checkout
    with FileNotFoundError instead of skipping. A download holding some files but not all is
    partial, and fails."""
    root = REPO / "data" / "pool" / "commitpackft"
    expected = pipeline.pool_manifest_shas(root)
    present = sorted(lang for lang in expected if (root / f"{lang}.jsonl").exists())
    if not present:
        pytest.skip("the commitpackft download is not on this host; only its manifest is")
    for lang, sha in expected.items():
        path = root / f"{lang}.jsonl"
        assert path.exists(), f"{lang}.jsonl is missing from a download holding {present}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == sha, f"{lang}.jsonl no longer matches its manifest"
