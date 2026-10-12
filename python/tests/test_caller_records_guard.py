"""The caller-record content guard: one oracle for three implementations, and every Python path
that opens pool or source rows refuses a copied record file.

The oracle is ``fixtures/caller/guard-cases.json``. qd-train's
``tests/caller_records_are_held_out.rs`` and ``crates/qd-prep/src/heldout.rs`` run the same cases.
Each entry-point test copies ``fixtures/caller/valid-records.jsonl`` into a training input
location (never under a ``heldout``
segment) and shows it refused.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_data.caller_records import CallerRecordRefusal, assert_no_caller_records, scan_stream

REPO = Path(__file__).resolve().parents[2]
ORACLE = json.loads((REPO / "fixtures/caller/guard-cases.json").read_text(encoding="utf-8"))
RECORDS = REPO / "fixtures/caller/valid-records.jsonl"
DEFAULT_LINE = 1 << 20
DEFAULT_TOTAL = 8 << 30


def case_bytes(case: dict) -> bytes:
    if "text" in case:
        return case["text"].encode("utf-8")
    if "hex" in case:
        return bytes.fromhex(case["hex"])
    g = case["gen"]
    eol = g.get("eol", "\n")
    if "record_padded_to" in g:
        head = '{"record":"lappi.caller_record","pad":"'
        core = head + "x" * (g["record_padded_to"] - len(head) - 2) + '"}' + eol
    elif "filler_line" in g:
        core = '{"pad":"' + "x" * (g["filler_line"] - 10) + '"}' + eol
    else:
        core = g["repeat"] * g["times"]
    return (g.get("first", "") + core + g.get("then", "")).encode("utf-8")


@pytest.mark.parametrize("case", ORACLE["cases"], ids=[c["name"] for c in ORACLE["cases"]])
def test_every_oracle_case_has_its_verdict(case):
    data = case_bytes(case)
    kw = {
        "label": case["name"],
        "line_bytes": case.get("line_bytes", DEFAULT_LINE),
        "max_bytes": case.get("total_bytes", DEFAULT_TOTAL),
    }
    if case["expect"] == "clean":
        scan_stream(io.BytesIO(data), **kw)
        return
    with pytest.raises(CallerRecordRefusal) as err:
        scan_stream(io.BytesIO(data), **kw)
    assert err.value.verdict == case["expect"], str(err.value)


def test_the_oracle_pins_the_three_verdicts_and_the_boundary_cases():
    verdicts = {c["expect"] for c in ORACLE["cases"]}
    assert verdicts == {"record", "clean", "unscannable"}
    names = " | ".join(c["name"] for c in ORACLE["cases"])
    for needle in ("garbage first line", "comment first line", "json array", "gzip", "zstd",
                   "utf-16le", "exactly the bound", "one byte over"):
        assert needle in names


def test_a_missing_path_and_a_directory_have_no_bytes_and_are_left_to_the_read(tmp_path):
    assert_no_caller_records(tmp_path / "missing.jsonl")
    assert_no_caller_records(tmp_path)


# -- every Python path that opens rows ---------------------------------------------------------


def _copied(tmp_path: Path, rel: str) -> Path:
    dest = tmp_path / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(RECORDS, dest)
    return dest


def test_read_jsonl_refuses_a_copied_record_file(tmp_path):
    from qd_data.loaders import read_jsonl

    f = _copied(tmp_path, "data/pool/anything/rows.jsonl")
    with pytest.raises(CallerRecordRefusal):
        read_jsonl(f, limit=10, max_row_bytes=1 << 20)


@pytest.mark.xfail(
    strict=True,
    reason="python/qd_data/decisions.py is lane 3's file; the one-line call site is in "
    "lane7-fixround-handoff.md. strict: once applied this must flip to a pass and lose the mark.",
)
def test_load_decision_pool_refuses_a_copied_record_file(tmp_path):
    from qd_data.decisions import load_decision_pool

    _copied(tmp_path, "data/pool/decisions-copy/examples.jsonl")
    (tmp_path / "data/pool/decisions-copy/manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CallerRecordRefusal):
        load_decision_pool(tmp_path / "data/pool/decisions-copy")


def _config():
    from qd_data.config import DataConfig

    return DataConfig()


def test_load_defect_rows_refuses_a_copied_record_file(tmp_path):
    from qd_data.defect_class import load_defect_rows

    _copied(tmp_path, "data/pool/corpus-copy/examples.jsonl")
    (tmp_path / "data/pool/corpus-copy/manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CallerRecordRefusal):
        load_defect_rows(
            tmp_path / "data/pool/corpus-copy", download_root=tmp_path / "dl",
            config=_config(), repo_root=tmp_path,
        )


def test_the_noul_g6_and_json_lines_readers_refuse_a_copied_record_file(tmp_path):
    from qd_data import defect_class as dc

    d = tmp_path / "data/pool/part"
    _copied(tmp_path, "data/pool/part/examples.jsonl")
    (d / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CallerRecordRefusal):
        dc._json_lines(d / "examples.jsonl", config=_config(), limit=10)
    with pytest.raises(CallerRecordRefusal):
        dc._load_noul(d, config=_config(), repo_root=tmp_path, licences={}, pools={})
    with pytest.raises(CallerRecordRefusal):
        dc._load_g6(d, config=_config(), repo_root=tmp_path)


def test_the_licence_pool_id_and_allowlist_readers_refuse_a_copied_record_file(tmp_path):
    from qd_data import defect_class as dc

    pool = _copied(tmp_path, "data/pool/pool.jsonl")
    with pytest.raises(CallerRecordRefusal):
        dc._load_pool_ids(pool)
    dl = tmp_path / "dl"
    _copied(tmp_path, "dl/python.jsonl")
    (dl / "manifest.json").write_text(
        json.dumps({"source_id": "bigcode/commitpackft", "languages": {"python": {
            "sha256": "0" * 64}}}), encoding="utf-8")
    with pytest.raises(CallerRecordRefusal):
        dc._load_licences(dl)


def test_the_mutate_adapter_readers_refuse_a_copied_record_file(tmp_path):
    from qd_train import mutate_adapter as ma

    f = _copied(tmp_path, "data/pool/mutated/examples.jsonl")
    with pytest.raises(CallerRecordRefusal):
        ma.read_example_objects(f, config=_config(), repo_root=tmp_path)
    with pytest.raises(CallerRecordRefusal):
        list(ma.read_examples(f, config=_config(), repo_root=tmp_path))


def test_the_tokenizer_pipeline_row_readers_refuse_a_copied_record_file(tmp_path):
    sys.path.insert(0, str(REPO / "tools"))
    import real_tokenizer_pipeline as rtp

    root = tmp_path / "data/pool/commitpackft"
    _copied(tmp_path, "data/pool/commitpackft/python.jsonl")
    (root / "manifest.json").write_text(
        json.dumps({"source_id": "bigcode/commitpackft",
                    "languages": {"python": {"sha256": "0" * 64}}}), encoding="utf-8")
    with pytest.raises(CallerRecordRefusal):
        rtp.commitpackft_pool_rows(root, max_pairs=1)
    # the report-only slice reader: the guard runs before its manifest is read
    _copied(tmp_path, "data/pool/slice/examples.jsonl")
    (tmp_path / "data/pool/slice/manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CallerRecordRefusal):
        rtp.load_report_only_slice(
            tmp_path / "data/pool/slice", download_root=root, config=_config(),
            repo_root=tmp_path, base_max_rows=1,
        )
