"""The shard header pins the modules that determine shard content, not the whole package.

Fable Q7.4 / GAP-LDATA-ALL-89-SHARD-HEADERS-DRIFT-FROM-QD-DATA-HEAD-2026-10-17:
fingerprinting every qd_data/*.py made any edit -- a comment, or a module no shard builder
imports -- turn every pinned shard set unreadable, so the guard was always red.
"""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import qd_data.fingerprint as fp  # noqa: E402
from qd_train import artifacts  # noqa: E402
from qd_train.artifacts import ShardContractViolation, ShardHeader  # noqa: E402
from qd_train.tristate import Ran  # noqa: E402

QD_DATA = REPO / "python" / "qd_data"

#: The files whose imports define "the code that builds shard bytes". qd_train/shards.py
#: writes the sequences; tools/real_ft_run.py builds the rows (dedupe, split, mixture,
#: decisions) it feeds them.
ENTRY_POINTS = (REPO / "python" / "qd_train" / "shards.py", REPO / "tools" / "real_ft_run.py")

#: In the closure and still not shard content: it describes the code, it does not run in the
#: row pipeline. This is the one named exemption.
NOT_CONTENT = frozenset({"fingerprint.py"})


def _qd_data_imports(path: Path) -> set[str]:
    """Module files of qd_data that one source file imports, absolute or relative, at top
    level or inside a function."""
    found: set[str] = set()
    in_pkg = path.parent == QD_DATA
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and in_pkg:
                if node.module:
                    found.add(node.module.split(".")[0])
                else:
                    found.update(a.name for a in node.names)
            elif node.level == 0 and node.module:
                parts = node.module.split(".")
                if parts[0] == "qd_data":
                    if len(parts) > 1:
                        found.add(parts[1])
                    else:
                        found.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "qd_data" and len(parts) > 1:
                    found.add(parts[1])
    return {f"{name}.py" for name in found if (QD_DATA / f"{name}.py").is_file()}


def _import_closure() -> set[str]:
    seen: set[str] = set()
    frontier: set[str] = set()
    for entry in ENTRY_POINTS:
        frontier |= _qd_data_imports(entry)
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        frontier |= _qd_data_imports(QD_DATA / name) - seen
    return seen


def _copy_package(dest: Path) -> None:
    for path in QD_DATA.glob("*.py"):
        shutil.copy(path, dest / path.name)


def test_every_module_the_shard_builders_import_is_in_the_list() -> None:
    """(a) The list is checked against the import closure of the shard-building entry
    points, so a module added to the pipeline cannot be left out unnoticed."""
    closure = _import_closure() - NOT_CONTENT
    assert closure, "the closure found nothing; the walker is broken"
    missing = closure - set(fp.SHARD_CONTENT_MODULES)
    assert not missing, f"{sorted(missing)} shape shard bytes but are not fingerprinted"


def test_the_list_names_nothing_outside_the_closure() -> None:
    """The converse, so the list cannot grow into the whole package again."""
    closure = _import_closure() - NOT_CONTENT
    assert set(fp.SHARD_CONTENT_MODULES) <= closure
    assert all((QD_DATA / name).is_file() for name in fp.SHARD_CONTENT_MODULES)
    assert list(fp.SHARD_CONTENT_MODULES) == sorted(fp.SHARD_CONTENT_MODULES)


def test_a_module_no_shard_builder_imports_does_not_move_the_fingerprint(tmp_path: Path) -> None:
    _copy_package(tmp_path)
    before = fp.shard_code_fingerprint(tmp_path)
    assert "pool_builder.py" not in before
    (tmp_path / "pool_builder.py").write_text("# edited\n", encoding="utf-8")
    assert fp.shard_code_fingerprint(tmp_path) == before


def test_a_comment_edit_to_a_listed_module_still_moves_the_fingerprint(tmp_path: Path) -> None:
    """(d) Honest on purpose: no comment normalisation. A tokenize-based digest would make
    every pinned fingerprint a function of the running Python's tokenizer."""
    _copy_package(tmp_path)
    before = fp.shard_code_fingerprint(tmp_path)
    target = tmp_path / "render.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# a comment\n", encoding="utf-8")
    assert fp.shard_code_fingerprint(tmp_path)["render.py"] != before["render.py"]


def test_a_missing_listed_module_is_refused_not_skipped(tmp_path: Path) -> None:
    _copy_package(tmp_path)
    (tmp_path / "render.py").unlink()
    with pytest.raises(RuntimeError, match="render.py"):
        fp.shard_code_fingerprint(tmp_path)


def _header(**over: object) -> ShardHeader:
    fields: dict[str, object] = {
        "split": "train", "data_snapshot_hash": "a" * 64, "tokenizer_hash": "b" * 64,
        "remap_hash": "c" * 64, "vocab_size": 128, "n_sequences": 4, "total_tokens": 40,
        "max_seq_len": 16, "buckets": (16,),
    }
    fields.update(over)
    return ShardHeader(**fields)  # type: ignore[arg-type]


def _check(header: ShardHeader):
    return artifacts._shard_code_check(header, path=Path("h.json"))


def test_a_v2_header_passes_when_only_an_unlisted_module_changed(tmp_path: Path, monkeypatch) -> None:
    recorded = fp.shard_code_fingerprint()
    header = _header(code_fingerprint=recorded, code_fingerprint_version=fp.FINGERPRINT_VERSION)
    _copy_package(tmp_path)
    (tmp_path / "pool_builder.py").write_text("# edited\n", encoding="utf-8")
    monkeypatch.setattr(fp, "_package_dir", lambda: tmp_path)
    state = _check(header)
    assert isinstance(state, Ran) and state.passed, state


def test_a_stale_v2_header_is_still_refused() -> None:
    stale = dict(fp.shard_code_fingerprint())
    stale["render.py"] = "0" * 64
    header = _header(code_fingerprint=stale, code_fingerprint_version=fp.FINGERPRINT_VERSION)
    state = _check(header)
    assert isinstance(state, Ran) and not state.passed and "render.py" in state.detail


def test_a_v1_header_is_checked_by_the_v1_rule_with_a_counted_note() -> None:
    full = fp.code_fingerprint()
    ok = _check(_header(code_fingerprint=full))
    assert isinstance(ok, Ran) and ok.passed and "v1" in ok.detail
    assert f"{len(full)} qd_data module" in ok.detail
    stale = dict(full)
    stale["pool_builder.py"] = "0" * 64  # v1 covers every module, so this is still drift
    bad = _check(_header(code_fingerprint=stale))
    assert isinstance(bad, Ran) and not bad.passed and "pool_builder.py" in bad.detail


def test_the_version_is_covered_by_the_shard_hash() -> None:
    """The version cannot be edited to move a header between rules; v1 hashes as before."""
    fpr = fp.shard_code_fingerprint()
    v2 = _header(code_fingerprint=fpr, code_fingerprint_version=2)
    as_v1 = _header(code_fingerprint=fpr, code_fingerprint_version=1)
    assert v2.shard_hash() != as_v1.shard_hash()
    assert "code_fingerprint_version" not in as_v1.to_json()
    assert ShardHeader.from_json(v2.to_json()).code_fingerprint_version == 2
    tampered = v2.to_json()
    tampered["code_fingerprint_version"] = 1
    with pytest.raises(ShardContractViolation):
        ShardHeader.from_json(tampered)


def test_an_unknown_future_version_is_refused() -> None:
    with pytest.raises(ShardContractViolation):
        _header(code_fingerprint_version=99)


def test_the_writer_stamps_the_current_version() -> None:
    src = (REPO / "python" / "qd_train" / "shards.py").read_text(encoding="utf-8")
    assert "code_fingerprint=shard_code_fingerprint()" in src
    assert "code_fingerprint_version=FINGERPRINT_VERSION" in src
