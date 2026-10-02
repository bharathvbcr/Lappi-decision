"""The A7 checker: v5's val and held-out rows are v4's plus CLINC's oos rows, and nothing else.

The gate's numbers are re-read from the DRAFT, so the checker's constants cannot drift from the
pre-registration; the rest builds two tiny builds and changes one thing at a time.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from v5_a7_check import GATE, SPLIT_PATHS, A7Gate, check, main

from qd_data.manifest import Manifest, ManifestEntry
from qd_train.tristate import Ran

REPO = Path(__file__).resolve().parents[2]
DRAFT = REPO / "campaign" / "v5-preregistered.DRAFT.json"


def _gate_text() -> str:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    hits = [s for s in draft["amendments_pending"] if isinstance(s, str) and s.startswith("(A7)")]
    assert len(hits) == 1, f"{DRAFT} has {len(hits)} (A7) amendments"
    return hits[0]


def _n(text: str) -> int:
    return int(text.replace(",", ""))


def test_the_constants_are_the_drafts_numbers() -> None:
    text = _gate_text()
    val = re.search(r"\(val: ([^;]+); held-out: ([^)]+)\)", text)
    assert val is not None
    short = {
        "code.defect_class": "code.defect_class",
        "commonsense": "commonsense.multiple_choice",
        "knowledge": "knowledge.multiple_choice",
        "qa.answer_span": "qa.answer_span",
        "qa.answerability": "qa.answerability",
    }

    def table(chunk: str) -> dict[str, int]:
        out = {}
        for name, n in re.findall(r"([a-z._]+) ([\d,]+\d)", chunk):
            out[short[name]] = _n(n)
        return out

    assert table(val.group(1)) == GATE.non_clinc["val"]
    assert table(val.group(2)) == GATE.non_clinc["heldout"]
    clinc = re.search(
        r"CLINC val ([\d,]+) \(within_domain ([\d,]+)\) and held-out ([\d,]+) per family", text
    )
    assert clinc is not None
    every, within, held = (_n(g) for g in clinc.groups())
    assert GATE.clinc["val"] == {
        "intent.classification": every,
        "intent.domain": every,
        "intent.in_scope": every,
        "intent.within_domain": within,
    }
    assert GATE.clinc["heldout"] == dict.fromkeys(GATE.clinc["val"], held)
    effect = json.loads(DRAFT.read_text(encoding="utf-8"))
    blob = json.dumps(effect)
    oos = re.search(r"oos utterances: train [\d,]+, val ([\d,]+), held-out ([\d,]+)", blob)
    assert oos is not None
    assert GATE.oos == {"val": _n(oos.group(1)), "heldout": _n(oos.group(2))}


# -- a tiny pair of builds ----------------------------------------------------------------

SMALL = A7Gate(
    non_clinc={"val": {"qa.answer_span": 2}, "heldout": {"qa.answerability": 1}},
    clinc={"val": {"intent.domain": 3}, "heldout": {"intent.domain": 2}},
    oos={"val": 1, "heldout": 1},
)


def _entry(split: str, family: str, identity: str, repo: str, i: int) -> ManifestEntry:
    return ManifestEntry(
        row_id=f"{split}:{family}:{i}:{identity}",
        content_hash=f"{i:064x}",
        split=split,
        source_id="clinc/clinc_oos" if family.startswith("intent.") else "rajpurkar/squad_v2",
        host="huggingface",
        family_id=family,
        repo_key=repo,
        identity_key=identity,
        licence_id="cc-by-3.0",
        obligations=("attribution",),
    )


def _rows(split: str, *, oos: bool) -> list[tuple[str, str, str]]:
    """(family, identity, repo) for one split of the tiny build."""
    if split == "val":
        rows = [
            ("qa.answer_span", "squad:a", "squad:t1"),
            ("qa.answer_span", "squad:b", "squad:t2"),
            ("intent.domain", "clinc-intent:alarm::01", "clinc-intent:alarm"),
            ("intent.domain", "clinc-intent:timer::02", "clinc-intent:timer"),
        ]
        if oos:
            rows.append(("intent.domain", "clinc-intent:oos::03", "clinc-oos:03"))
        return rows
    rows = [
        ("qa.answerability", "squad:c", "squad:t3"),
        ("intent.domain", "clinc-intent:bill::04", "clinc-intent:bill"),
    ]
    if oos:
        rows.append(("intent.domain", "clinc-intent:oos::05", "clinc-oos:05"))
    return rows


def _write(root: Path, rows_by_split: dict[str, list[tuple[str, str, str]]]) -> Path:
    for split, rows in rows_by_split.items():
        Manifest(
            split=split,
            entries=tuple(
                _entry(split, f, ident, repo, i) for i, (f, ident, repo) in enumerate(rows)
            ),
            config_fingerprint={},
            admitted_source_ids=(),
            refused_sources={},
            held_out_families=(),
            licence_histogram={},
            obligations={},
            host_histogram={},
            mixture_json={},
            dedupe_json={},
            split_json={},
            status=Ran(passed=True),
            provenance={},
        ).write(root / SPLIT_PATHS[split])
    return root


@pytest.fixture
def v4(tmp_path: Path) -> Path:
    return _write(tmp_path / "v4", {s: _rows(s, oos=False) for s in SPLIT_PATHS})


def _v5(tmp_path: Path, **change: list[tuple[str, str, str]]) -> Path:
    rows = {s: _rows(s, oos=True) for s in SPLIT_PATHS}
    rows.update(change)
    return _write(tmp_path / "v5", rows)


def _failed(report: dict) -> dict[str, list[str]]:
    return {
        f"{r['split']}/{r['family']}": r["reasons"] for r in report["families"] if not r["passed"]
    }


def test_v4_plus_the_oos_rows_passes(tmp_path: Path, v4: Path) -> None:
    report = check(v4, _v5(tmp_path), gate=SMALL)
    assert report["passed"], _failed(report)
    assert report["families_checked"] == 4


def test_a_val_row_knocked_out_in_dedupe_is_named(tmp_path: Path, v4: Path) -> None:
    rows = [r for r in _rows("val", oos=True) if r[1] != "squad:b"]
    report = check(v4, _v5(tmp_path, val=rows), gate=SMALL)
    assert not report["passed"]
    [row] = [r for r in report["families"] if not r["passed"]]
    assert (row["family"], row["first_missing"], row["missing"]) == (
        "qa.answer_span",
        ["squad:b"],
        1,
    )


def test_a_swapped_identity_with_the_same_count_refuses(tmp_path: Path, v4: Path) -> None:
    rows = [(f, "squad:z" if i == "squad:b" else i, r) for f, i, r in _rows("val", oos=True)]
    report = check(v4, _v5(tmp_path, val=rows), gate=SMALL)
    assert _failed(report) == {
        "val/qa.answer_span": ["identity keys differ from v4's: 1 missing, 1 extra"]
    }


def test_a_new_val_family_refuses(tmp_path: Path, v4: Path) -> None:
    rows = [*_rows("val", oos=True), ("code.defect_class", "own-prose:x", "own-prose:r")]
    report = check(v4, _v5(tmp_path, val=rows), gate=SMALL)
    assert list(_failed(report)) == ["val/code.defect_class"]


def test_a_clinc_in_scope_row_lost_refuses_even_with_an_extra_oos_row(
    tmp_path: Path, v4: Path
) -> None:
    rows = [r for r in _rows("heldout", oos=True) if r[1] != "clinc-intent:bill::04"]
    rows.append(("intent.domain", "clinc-intent:oos::06", "clinc-oos:06"))
    report = check(v4, _v5(tmp_path, heldout=rows), gate=SMALL)
    reasons = _failed(report)["heldout/intent.domain"]
    assert "2 oos rows, the gate's re-key moves 1" in reasons
    assert "in-scope identity keys differ from v4's: 1 missing, 0 extra" in reasons


def test_an_oos_split_unit_carrying_an_in_scope_identity_refuses(tmp_path: Path, v4: Path) -> None:
    rows = [
        (f, i, "clinc-oos:04" if i == "clinc-intent:bill::04" else r)
        for f, i, r in _rows("heldout", oos=False)
    ]
    report = check(v4, _v5(tmp_path, heldout=rows), gate=SMALL)
    reasons = _failed(report)["heldout/intent.domain"]
    assert any("are not oos identities" in r for r in reasons)


def test_the_wrong_baseline_refuses(tmp_path: Path) -> None:
    rows = {s: _rows(s, oos=False) for s in SPLIT_PATHS}
    rows["val"] = rows["val"][1:]
    wrong_v4 = _write(tmp_path / "v4", rows)
    report = check(wrong_v4, _v5(tmp_path), gate=SMALL)
    assert any(r.startswith("baseline:") for r in _failed(report)["val/qa.answer_span"])


def test_an_edited_manifest_refuses(tmp_path: Path, v4: Path) -> None:
    v5 = _v5(tmp_path)
    path = v5 / SPLIT_PATHS["val"]
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["entries"][0]["family_id"] = "qa.answerability"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="data_snapshot_hash"):
        check(v4, v5, gate=SMALL)


def test_the_cli_exits_nonzero_on_the_real_gate(tmp_path: Path, v4: Path) -> None:
    # The tiny build does not meet the real gate's numbers, so the CLI must refuse.
    assert main(["--v4", str(v4), "--v5", str(_v5(tmp_path))]) == 1
