"""Every JSONL reader on the way to training splits lines exactly as the caller-record guard does
(``qd_data.caller_records``): on a newline, a preceding CR stripped, and on nothing else.

Defect D-A1 (final verify of the guard): the guard split on a newline while some readers also
split on ``str.splitlines()``'s set (VT, FF, FS, GS, RS, U+0085, U+2028, U+2029) or, in text
mode, on a bare CR. A record file re-separated with one of those passed the guard as one
over-long line and was read as many rows. Two halves are held here:

* a row whose string value holds U+0085, U+2028 or U+2029 (legal JSON, legitimate data) is ONE
  row in every reader (a ``splitlines()`` reader cut it in two, and dropped or refused both);
* a record file re-separated with any of the nine separators is refused at the guard, and so at
  every reader that calls it.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from qd_data.caller_records import (
    CallerRecordRefusal,
    assert_no_caller_records,
    jsonl_lines,
    scan_stream,
    split_jsonl_text,
)
from qd_data.config import DataConfig

REPO = Path(__file__).resolve().parents[2]
RECORD = (REPO / "fixtures/caller/valid-records.jsonl").read_text(encoding="utf-8").splitlines()[0]

#: What ``str.splitlines()`` splits on beyond ``\n`` (and ``\r``, which text mode splits on).
ALL_SEPARATORS = ["\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]
#: The ones that are legal, raw, inside a JSON string value.
STRING_SAFE = ["\x85", "\u2028", "\u2029"]


def _sep_id(sep: str) -> str:
    return f"U+{ord(sep):04X}"


def _write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    """One row per ``\\n``, written the way a JSON writer that does not escape non-ASCII does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()
    )
    return path


# -- the canonical iterator ---------------------------------------------------------------------


@pytest.mark.parametrize("sep", STRING_SAFE, ids=_sep_id)
def test_the_iterators_do_not_split_on_a_separator_inside_a_row(tmp_path, sep):
    text = f'{{"a":"x{sep}y"}}\n{{"b":2}}\n'
    assert split_jsonl_text(text) == [f'{{"a":"x{sep}y"}}', '{"b":2}']
    path = tmp_path / "rows.jsonl"
    path.write_bytes(text.encode())
    assert list(jsonl_lines(path)) == [f'{{"a":"x{sep}y"}}\n', '{"b":2}\n']


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("", []),
        ("\n", [""]),
        ("a", ["a"]),
        ("a\n", ["a"]),
        ("a\n\nb", ["a", "", "b"]),
        ("a\r\nb\r\n", ["a", "b"]),
        ("a\rb\n", ["a\rb"]),
        ("a\r", ["a"]),
        ("\r\n", [""]),
        ("a\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029b\n", ["a\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029b"]),
    ],
)
def test_split_jsonl_text_is_newline_with_an_optional_cr(text, want):
    assert split_jsonl_text(text) == want


@pytest.mark.parametrize("sep", ALL_SEPARATORS, ids=_sep_id)
def test_jsonl_lines_agrees_with_split_jsonl_text_on_every_separator(tmp_path, sep):
    text = f"a{sep}b\nc\r\nd{sep}\n\ne"
    path = tmp_path / "rows.jsonl"
    path.write_bytes(text.encode())
    got = [line.removesuffix("\n") for line in jsonl_lines(path)]
    assert got == split_jsonl_text(text)


def test_a_crlf_row_reads_as_lf_so_a_row_byte_bound_sees_the_same_length(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"a":1}\r\n{"b":2}\n')
    assert list(jsonl_lines(path)) == ['{"a":1}\n', '{"b":2}\n']


# -- characterisation: a row with a separator in a string value is one row, in every reader -----


@pytest.mark.parametrize("sep", STRING_SAFE, ids=_sep_id)
def test_loaders_read_jsonl_keeps_a_row_whole(tmp_path, sep):
    from qd_data.loaders import read_jsonl

    path = _write_rows(tmp_path / "rows.jsonl", [{"i": i, "t": f"a{sep}b"} for i in range(3)])
    rows, capped = read_jsonl(path, limit=10, max_row_bytes=1 << 20)
    assert [r["i"] for r in rows] == [0, 1, 2]
    assert not capped


@pytest.mark.parametrize("sep", STRING_SAFE, ids=_sep_id)
def test_mutate_adapter_read_example_objects_keeps_a_row_whole(tmp_path, sep):
    from qd_train import mutate_adapter as ma

    path = _write_rows(
        tmp_path / "data/pool/x/examples.jsonl", [{"i": i, "t": f"a{sep}b"} for i in range(3)]
    )
    rows = ma.read_example_objects(path, config=DataConfig(), repo_root=tmp_path)
    assert [r["i"] for r in rows] == [0, 1, 2]


@pytest.mark.parametrize("sep", STRING_SAFE, ids=_sep_id)
def test_defect_class_json_lines_and_pool_ids_keep_a_row_whole(tmp_path, sep):
    from qd_data import defect_class as dc

    rows = [{"id": f"p{i}", "repo": f"r{sep}{i}"} for i in range(3)]
    path = _write_rows(tmp_path / "data/pool/pool.jsonl", rows)
    assert len(dc._json_lines(path, config=DataConfig(), limit=10)) == 3
    assert dc._load_pool_ids(path) == {f"p{i}": f"r{sep}{i}" for i in range(3)}


@pytest.mark.parametrize("sep", STRING_SAFE, ids=_sep_id)
def test_defect_class_licence_reader_keeps_a_row_whole(tmp_path, sep):
    from qd_data import defect_class as dc

    dl = tmp_path / "dl"
    rows = [{"commit": f"c{sep}{i}", "new_file": "f.py", "license": "mit"} for i in range(3)]
    path = _write_rows(dl / "python.jsonl", rows)
    (dl / "manifest.json").write_text(
        json.dumps({
            "source_id": "bigcode/commitpackft",
            "languages": {"python": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
        }),
        encoding="utf-8",
    )
    assert len(dc._load_licences(dl)) == 3


def _commitpackft_row(i: int, sep: str) -> dict[str, object]:
    return {
        "commit": f"c{i}", "repos": "o/r", "old_file": "f.py", "new_file": "f.py",
        "old_contents": "a = 1\n", "new_contents": "a = 2\n", "subject": f"s{sep}{i}",
        "message": f"m{sep}{i}", "lang": "python", "license": "mit",
    }


@pytest.mark.parametrize("sep", STRING_SAFE, ids=_sep_id)
def test_tokenizer_pipeline_commitpackft_rows_keep_a_row_whole(tmp_path, sep):
    import real_tokenizer_pipeline as rtp

    root = tmp_path / "data/pool/commitpackft"
    path = _write_rows(root / "python.jsonl", [_commitpackft_row(i, sep) for i in range(3)])
    (root / "manifest.json").write_text(
        json.dumps({
            "source_id": "bigcode/commitpackft",
            "languages": {"python": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
        }),
        encoding="utf-8",
    )
    rows, _capped, total = rtp.commitpackft_pool_rows(root, max_pairs=10)
    assert total == 3
    assert sorted(r.commit for r in rows) == ["c0", "c1", "c2"]


@pytest.mark.parametrize("sep", STRING_SAFE, ids=_sep_id)
def test_build_commitpackft_pool_keeps_a_row_whole(tmp_path, sep):
    import build_commitpackft_pool as bcp

    _write_rows(tmp_path / "src/python.jsonl", [_commitpackft_row(i, sep) for i in range(3)])
    report = bcp.build(tmp_path / "src", tmp_path / "out/pool.jsonl", max_records=None)
    assert report["records"] == 3, report["dropped_by_reason"]


@pytest.mark.parametrize("sep", STRING_SAFE, ids=_sep_id)
def test_compose_split_map_keeps_a_row_whole(tmp_path, sep):
    import compose_split_map as csm

    pool = _write_rows(
        tmp_path / "data/pool/pool.jsonl", [{"id": f"p{i}", "repo": f"r{sep}{i}"} for i in range(3)]
    )
    body, repos = csm.split_map(pool, config=DataConfig(), repo_root=tmp_path)
    assert len(repos) == 3
    assert body["pool"]["records"] == 3


# -- a record file re-separated with any separator is refused ------------------------------------


def _resplit(sep: str) -> bytes:
    """600 padded rows and a caller record, joined by ``sep`` and ended by ``\\n``: over the 1 MiB
    line budget as one line, so a guard that does not split on ``sep`` skips it unparsed."""
    filler = json.dumps({"pad": "x" * 2000})
    return sep.join([filler] * 600 + [RECORD]).encode("utf-8") + b"\n"


@pytest.mark.parametrize("sep", ALL_SEPARATORS, ids=_sep_id)
def test_the_guard_refuses_a_record_file_re_separated_with_any_splitlines_separator(tmp_path, sep):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(_resplit(sep))
    with pytest.raises(CallerRecordRefusal) as err:
        assert_no_caller_records(path)
    assert err.value.verdict == "unscannable"
    with pytest.raises(CallerRecordRefusal):
        scan_stream(io.BytesIO(_resplit(sep)), label="bytes")


@pytest.mark.parametrize("sep", ALL_SEPARATORS, ids=_sep_id)
def test_every_reader_refuses_a_re_separated_record_file(tmp_path, sep):
    import compose_split_map as csm

    from qd_data import defect_class as dc
    from qd_data.loaders import read_jsonl
    from qd_train import mutate_adapter as ma

    path = tmp_path / "data/pool/rows.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(_resplit(sep))
    with pytest.raises(CallerRecordRefusal):
        read_jsonl(path, limit=100_000, max_row_bytes=1 << 20)
    with pytest.raises(CallerRecordRefusal):
        ma.read_example_objects(path, config=DataConfig(), repo_root=tmp_path)
    with pytest.raises(CallerRecordRefusal):
        dc._json_lines(path, config=DataConfig(), limit=100_000)
    with pytest.raises(CallerRecordRefusal):
        dc._load_pool_ids(path)
    with pytest.raises(CallerRecordRefusal):
        csm.split_map(path, config=DataConfig(), repo_root=tmp_path)


# -- the readers do not open the file in a splitting mode -----------------------------------------

#: file -> the functions that read pool or example rows on the way to training. None may call
#: ``.splitlines()`` or iterate a text-mode handle: both split on more than the guard does.
_READERS: dict[str, list[str]] = {
    "python/qd_data/loaders.py": ["read_jsonl"],
    "python/qd_data/defect_class.py": [
        "_load_licences", "_load_pool_ids", "load_defect_rows", "_load_noul", "_load_g6",
        "_json_lines", "noul_allowlist", "noul_v5_allowlist",
    ],
    "python/qd_train/mutate_adapter.py": ["read_example_objects", "read_examples"],
    "python/qd_train/composed_slice.py": ["load_cases"],
    "tools/real_tokenizer_pipeline.py": ["commitpackft_pool_rows", "load_report_only_slice"],
    "tools/build_commitpackft_pool.py": ["build"],
    "tools/compose_split_map.py": ["split_map"],
    "tools/rung0_real_run.py": ["generate_examples"],
}


def _function_nodes(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _splitting_calls(fn: ast.FunctionDef) -> list[str]:
    bad: list[str] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "splitlines":
            bad.append(f"line {node.lineno}: .splitlines()")
        if node.func.attr == "open" and not any(kw.arg == "newline" for kw in node.keywords):
            # A text-mode read open translates a bare \r into a line break; binary mode, a write
            # and newline="\n" (what jsonl_lines does) do not. Manifests are parsed whole.
            mode = next((a.value for a in node.args if isinstance(a, ast.Constant)), "r")
            if isinstance(mode, str) and not set(mode) & set("bwax"):
                bad.append(f"line {node.lineno}: text-mode .open()")
    return bad


@pytest.mark.parametrize("rel", sorted(_READERS))
def test_no_row_reader_uses_splitlines_or_a_text_mode_open(rel):
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    fns = _function_nodes(tree)
    missing = [name for name in _READERS[rel] if name not in fns]
    # a name that moved is a reader the check no longer covers: fail loud rather than pass
    assert not missing, f"{rel}: reader(s) {missing} not found; update _READERS"
    for name in _READERS[rel]:
        assert _splitting_calls(fns[name]) == [], f"{rel}:{name}"


# -- the separator scan is chunking-independent ----------------------------------------------------


def _reference_refuses(line: bytes) -> bool:
    """The rule, one byte at a time over the whole line: a backslash escapes the next byte, a
    quote toggles the string, control separators refuse anywhere, U+0085/2028/2029 outside one."""
    in_string = escaped = False
    for i, b in enumerate(line):
        if b in (0x0B, 0x0C, 0x1C, 0x1D, 0x1E):
            return True
        if not in_string:
            if b == 0x85 and line[i - 1 : i] == b"\xc2":
                return True
            if b in (0xA8, 0xA9) and line[i - 2 : i] == b"\xe2\x80" and i >= 2:
                return True
        if escaped:
            escaped = False
        elif b == 0x5C:
            escaped = True
        elif b == 0x22:
            in_string = not in_string
    return False


def test_the_separator_scan_gives_the_same_answer_however_the_line_is_chunked():
    import random

    from qd_data.caller_records import _BreakScan

    rng = random.Random(20261012)
    atoms = [b'"', b"\\", b"x", b" ", b"{", b"\xc2\x85", b"\xe2\x80\xa8", b"\xe2\x80\xa9",
             b"\xe2\x80\xa6", b"\xc3\x85", b"\x1e", b"\\\\", b'\\"']
    for _ in range(3000):
        line = b"".join(rng.choice(atoms) for _ in range(rng.randint(0, 40)))
        want = _reference_refuses(line)
        cuts = sorted(rng.sample(range(len(line) + 1), min(len(line) + 1, rng.randint(0, 4))))
        scan, found, prev = _BreakScan(), False, 0
        for cut in [*cuts, len(line)]:
            found = found or scan.feed(line[prev:cut]) is not None
            prev = cut
        assert found == want, (line, cuts)
