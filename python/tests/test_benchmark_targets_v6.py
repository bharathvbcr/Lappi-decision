"""v6 benchmark re-pin (data-clean plan section 3 item 6).

GAP-DECISION-INDEX-PANEL-CLINC-AND-MMLU-TEST-ITEMS-ARE-LAPPI-TRAINING-DATA-2026-10-03: v5 pins
MMLU's test and dev splits to train (``qd_data.sources`` ``cais/mmlu``) and reads CLINC's test
split as rows, split by intent, so the Decision Index's MMLU and CLINC150 panels score Lappi on
items it trained on. v6 reads neither as rows: under
``DataConfig.with_v6_benchmark_targets()`` those upstream splits are refused, counted, and become
decontamination targets of the in-corpus scan (``tools/containment_scan.py``), which also takes
any ``{"id", "text"}`` target set -- the jevjudge pairs among them -- with ``--target``.

v5 stays the default: ``test_general_families.py`` pins MMLU test and dev to train, and a v5
rebuild (``real_ft_run.ft_splits``, the containment scan) must re-derive v5's rows.

The first test is a characterization of the unmodified code: under the default config, train
identity keys derive from MMLU test/dev and CLINC test rows. It passes before and after.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qd_data.config import DataConfig  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.loaders import ClincRow, MmluRow  # noqa: E402
from qd_data.mixture import build_mixture, clinc_keys  # noqa: E402
from qd_data.rows import DataRow  # noqa: E402
from qd_data.split import split  # noqa: E402

WORDS = [
    "anchor", "basin", "cobalt", "delta", "ember", "fjord", "glacier", "harbor", "island",
    "jungle", "kettle", "lantern", "meadow", "nectar", "orchard", "pepper", "quarry", "raven",
    "saddle", "timber", "umber", "velvet", "willow", "xenon", "yonder", "zephyr", "amber",
    "bramble", "cinder", "dapple", "ferric", "gossamer", "hollow", "indigo", "jasper", "kelp",
]
INTENTS = tuple(f"intent_{i:03d}" for i in range(24))
#: The upstream splits v6 refuses as rows, by source.
TARGET_SPLITS = {"cais/mmlu": {"test", "dev"}, "clinc/clinc_oos": {"test"}}


def _text(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


def _mmlu(rng: random.Random, upstream: str, n: int) -> list[MmluRow]:
    return [
        MmluRow(subject=f"subject_{j % 3}", question=f"{upstream} {j} {_text(rng, 14)}?",
                choices=(f"{upstream}a{j}", f"{upstream}b{j}", f"{upstream}c{j}",
                         f"{upstream}d{j}"),
                answer_index=j % 4, upstream_split=upstream)
        for j in range(n)
    ]


def _clinc_raw(rng: random.Random, upstream: str, n: int) -> list[dict[str, object]]:
    """CLINC rows as the hub's JSONL holds them (``text``, ``intent`` by name)."""
    return [
        {"text": f"{upstream} {j} {_text(rng, 10)}",
         "intent": "oos" if j % 7 == 0 else INTENTS[j % len(INTENTS)]}
        for j in range(n)
    ]


def _clinc_v5(raw: dict[str, object]) -> ClincRow:
    """A CLINC row as v5 holds it: no upstream split."""
    intent = str(raw["intent"])
    return ClincRow(utterance=str(raw["text"]), intent=intent, is_oos=intent == "oos")


def _files() -> dict[str, dict[str, list[object]]]:
    rng = random.Random(20261006)
    return {
        "cais/mmlu": {s: list(_mmlu(rng, s, n))
                      for s, n in (("test", 10), ("dev", 4), ("validation", 8))},
        "clinc/clinc_oos": {s: list(_clinc_raw(rng, s, n))
                            for s, n in (("train", 120), ("validation", 30), ("test", 40))},
    }


def _target_identities(files: dict[str, dict[str, list[object]]]) -> set[str]:
    """The identity key every row of a v6 target file has when it is built as a row (v5)."""
    out: set[str] = set()
    v5 = build_mixture(
        {"cais/mmlu": [r for s in TARGET_SPLITS["cais/mmlu"]
                       for r in files["cais/mmlu"][s]]},
        config=DataConfig(),
    )
    out |= {r.identity_key for r in v5.rows}
    for raw in files["clinc/clinc_oos"]["test"]:
        row = _clinc_v5(raw)  # type: ignore[arg-type]
        out.add(clinc_keys(row, row.utterance.strip()).identity_key)
    return out


def _rows_by_split(raw: dict[str, list[object]], config: DataConfig):
    mixture = build_mixture(raw, config=config)  # type: ignore[arg-type]
    report = split(dedupe(list(mixture.rows), config=config), config=config)
    return mixture, report.rows_by_split


def test_v5_trains_on_rows_read_from_the_benchmark_test_files() -> None:
    """Characterization of the unmodified code (the gap, as built)."""
    files = _files()
    raw = {
        "cais/mmlu": [r for rows in files["cais/mmlu"].values() for r in rows],
        "clinc/clinc_oos": [_clinc_v5(r) for rows in files["clinc/clinc_oos"].values()
                            for r in rows],  # type: ignore[arg-type]
    }
    _mixture, by_split = _rows_by_split(raw, DataConfig())
    train = {r.identity_key for r in by_split["train"]}
    targets = _target_identities(files)
    from_targets = train & targets
    assert any(k.startswith("mmlu-train:") for k in from_targets), "MMLU test/dev train in v5"
    assert any(k.startswith("clinc-intent:") for k in from_targets), "CLINC test trains in v5"


def _v6_raw(files: dict[str, dict[str, list[object]]]) -> dict[str, list[object]]:
    from qd_data.loaders import parse_clinc

    return {
        "cais/mmlu": [r for rows in files["cais/mmlu"].values() for r in rows],
        "clinc/clinc_oos": [
            parse_clinc(r, index=i, split_name=s)  # type: ignore[arg-type]
            for s, rows in files["clinc/clinc_oos"].items() for i, r in enumerate(rows)
        ],
    }


def test_v6_no_row_identity_key_derives_from_a_benchmark_test_file() -> None:
    files = _files()
    config = DataConfig().with_v6_benchmark_targets()
    mixture, by_split = _rows_by_split(_v6_raw(files), config)
    targets = _target_identities(files)
    every = {r.identity_key for rows in by_split.values() for r in rows}
    assert by_split["train"], "the fixture trains on something"
    assert not every & targets, sorted(every & targets)[:5]
    for r in by_split["train"]:
        assert r.metadata.get("upstream_split") not in TARGET_SPLITS.get(r.source_id, set()), r
    # Refused and counted, never silently absent: one per row per family that asked.
    mmlu_families, clinc_families = 1, 2
    assert mixture.refusals["cais/mmlu"]["benchmark_eval_split_is_a_target"] == (
        (10 + 4) * mmlu_families
    )
    assert mixture.refusals["clinc/clinc_oos"]["benchmark_eval_split_is_a_target"] == (
        40 * clinc_families
    )
    # MMLU's validation split is still the family's val; CLINC's train and validation still
    # split by intent.
    assert {r.metadata["upstream_split"] for r in by_split["val"]
            if r.source_id == "cais/mmlu"} == {"validation"}
    assert any(r.source_id == "clinc/clinc_oos" for r in by_split["train"])


def test_v6_refuses_a_clinc_row_whose_upstream_split_is_unstated() -> None:
    """Fail closed: a row that cannot say which file it came from may be a test row."""
    files = _files()
    raw = {"clinc/clinc_oos": [_clinc_v5(r) for r in files["clinc/clinc_oos"]["train"]]}
    v5 = build_mixture(raw, config=DataConfig())  # type: ignore[arg-type]
    assert "upstream_split_unstated" not in v5.refusals["clinc/clinc_oos"]
    v6 = build_mixture(raw, config=DataConfig().with_v6_benchmark_targets())  # type: ignore[arg-type]
    assert not v6.rows
    assert v6.refusals["clinc/clinc_oos"]["upstream_split_unstated"] == 120 * 2


def test_the_v6_opt_in_is_in_the_fingerprint_only_when_set() -> None:
    v5 = DataConfig()
    assert "benchmark_eval_splits_are_targets" not in v5.fingerprint()
    v6 = v5.with_v6_benchmark_targets()
    assert v6.fingerprint()["benchmark_eval_splits_are_targets"] is True
    assert {k: v for k, v in v6.fingerprint().items()
            if k != "benchmark_eval_splits_are_targets"} == v5.fingerprint()
    with pytest.raises(TypeError):
        DataConfig(benchmark_eval_splits_are_targets=1)  # type: ignore[arg-type]


# -- the in-corpus containment scan takes target sets ----------------------------------------


def _request_bytes(sets, scans, report) -> bytes:
    from containment_scan import splitter_checks, write_request

    buf = io.BytesIO()
    write_request(buf, sets, scans, corpus={"fixture": "v6 targets"},
                  export={"fixture": True}, checks=splitter_checks(report))
    return buf.getvalue()


def _report_and_rows():
    from qd_data.split import SplitReport

    rng = random.Random(7)
    rows = _mmlu(rng, "validation", 6) + _mmlu(rng, "test", 30)
    config = DataConfig()
    mixture = build_mixture({"cais/mmlu": rows}, config=config)
    report: SplitReport = split(dedupe(list(mixture.rows), config=config), config=config)
    return report, rows


#: sha256 of the request ``scan_sets`` / ``scan_specs`` / ``write_request`` wrote for
#: :func:`_report_and_rows` on the unmodified code (main 2cee2a8), before targets existed.
PINNED_REQUEST_SHA256 = "40c5b61096259d1eedbdba4fc822cbf8764ad69f273f0c867756f887f8d145a5"


def test_without_targets_the_scan_request_is_the_bytes_it_was() -> None:
    from containment_scan import scan_sets, scan_specs

    report, _rows = _report_and_rows()
    sets, _strip = scan_sets(report, config=DataConfig())
    got = hashlib.sha256(_request_bytes(sets, scan_specs(sets), report)).hexdigest()
    assert got == PINNED_REQUEST_SHA256


def test_a_target_set_excludes_the_train_row_that_copies_a_benchmark_item(
    qd_prep_bin: Path, tmp_path: Path,
) -> None:
    from containment_scan import (
        TARGET_PREFIX,
        read_target_file,
        run_containment,
        scan_sets,
        scan_specs,
    )

    report, _rows = _report_and_rows()
    train = [r for r in report.rows_by_split["train"] if r.source_id == "cais/mmlu"]
    victim: DataRow = train[3]
    # A jevjudge-shaped target file: one item copies the victim's question, one is unrelated.
    target = tmp_path / "jevjudge.jsonl"
    context = victim.request.context
    copied = context.decode("utf-8") if isinstance(context, bytes) else str(context)
    target.write_text(
        json.dumps({"id": "pair-1", "text": copied}) + "\n"
        + json.dumps({"id": "pair-2", "text": _text(random.Random(1), 30)}) + "\n",
        encoding="utf-8",
    )
    tset = read_target_file("jevjudge", target)
    assert tset.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    sets, _strip = scan_sets(report, config=DataConfig(), targets=(tset,))
    names = [s.name for s in sets]
    assert names[-1] == f"{TARGET_PREFIX}jevjudge"
    specs = scan_specs(sets)
    assert any(s.source == "train" and s.target == names[-1] and s.enforced for s in specs)
    request = tmp_path / "req.bin"
    request.write_bytes(_request_bytes(sets, specs, report))
    out = tmp_path / "out"
    run_containment(qd_prep_bin, request, out, threads=2, timeout_s=120.0)
    att = json.loads((out / "attestation.json").read_text(encoding="utf-8"))
    keys = (out / "exclusions.txt").read_text(encoding="utf-8").splitlines()
    assert victim.identity_key in keys
    assert names[-1] in att["enforced_targets"], att["enforced_targets"]
    # Excluding the victim leaves no enforced pair against the target. (The attestation as a
    # whole is not clean here only because this MMLU-only fixture has no held-out rows.)
    assert att["remaining_hits"][names[-1]] == 0, att["remaining_hits"]
    scan = next(s for s in att["scans"] if s["target"] == names[-1] and s["source"] == "train")
    assert scan["enforced"] is True and scan["source_rows_hit"] == 1, scan
    assert not any(names[-1] in r for r in att["not_clean_because"]), att["not_clean_because"]


def test_a_target_file_that_is_not_id_text_jsonl_is_refused(tmp_path: Path) -> None:
    from containment_scan import read_target_file

    bad = tmp_path / "bad.jsonl"
    for body, match in (
        ("", "no rows"),
        ('{"id": "a"}\n', "text"),
        ('{"id": "a", "text": "x"}\n{"id": "a", "text": "y"}\n', "repeats"),
        ("not json\n", "JSON"),
    ):
        bad.write_text(body, encoding="utf-8")
        with pytest.raises(SystemExit, match=match):
            read_target_file("bad", bad)
    with pytest.raises(SystemExit, match="name"):
        read_target_file("has space", bad)


def test_benchmark_targets_come_from_the_fetch_record_and_are_sha_checked(
    tmp_path: Path,
) -> None:
    from containment_scan import benchmark_target_sets

    files = _files()
    root = tmp_path / "cache"
    entries = []

    def write(dataset: str, split_name: str, rows: list[dict[str, object]]) -> None:
        d = root / dataset.replace("/", "__") / "rev"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{split_name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        entries.append({"dataset": dataset, "jsonl": str(path), "rows": len(rows),
                        "jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "n_intent_names": None})

    for s, mrows in files["cais/mmlu"].items():
        write("cais/mmlu", s, [{"question": r.question, "subject": r.subject,  # type: ignore[union-attr]
                                "choices": list(r.choices), "answer": r.answer_index}  # type: ignore[union-attr]
                               for r in mrows])
    for s, crows in files["clinc/clinc_oos"].items():
        write("clinc/clinc_oos", s, crows)  # type: ignore[arg-type]
    record = root / "fetch-record.json"
    record.write_text(json.dumps(entries), encoding="utf-8")
    got = {t.name: t for t in benchmark_target_sets(record)}
    assert sorted(got) == ["clinc-test", "mmlu-dev", "mmlu-test"]
    assert len(got["mmlu-test"].rows) == 10 and len(got["mmlu-dev"].rows) == 4
    assert len(got["clinc-test"].rows) == 40
    q = files["cais/mmlu"]["test"][0]
    assert got["mmlu-test"].rows[0][1] == "\n".join((q.question, *q.choices))  # type: ignore[union-attr]
    assert got["clinc-test"].rows[0][1] == files["clinc/clinc_oos"]["test"][0]["text"]
    # A cache that is not the approved download is refused.
    path = Path(entries[0]["jsonl"])
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="sha256"):
        benchmark_target_sets(record)


def test_a_short_mmlu_item_copied_into_train_is_found_by_its_target_text(
    qd_prep_bin: Path, tmp_path: Path,
) -> None:
    """Adversarial: a short stem and four short options, so most of a hand-assembled target's
    8-grams would straddle a boundary the rendered row does not have. The target text is built
    through the row funnel (``mmlu_target_text``), so an exact copy contains all of it."""
    from containment_scan import (
        TargetSet,
        mmlu_target_text,
        run_containment,
        scan_sets,
        scan_specs,
    )

    item = MmluRow(subject="biology", question="Which gas do green plants absorb in light?",
                   choices=("carbon dioxide from open air", "oxygen released by their leaves",
                            "nitrogen fixed by root bacteria", "water vapour from wet soil"),
                   answer_index=0, upstream_split="test")
    text, fell_back = mmlu_target_text(item)
    assert not fell_back
    assert text.splitlines() == [item.question, *item.choices]
    rng = random.Random(11)
    rows = [item, *_mmlu(rng, "validation", 6), *_mmlu(rng, "test", 20)]
    config = DataConfig()
    report = split(dedupe(list(build_mixture({"cais/mmlu": rows}, config=config).rows),
                          config=config), config=config)
    victim = next(r for r in report.rows_by_split["train"]
                  if r.metadata.get("subject") == "biology")
    tset = TargetSet(name="mmlu-test", rows=(("test-0", text),), sha256="0" * 64, source="t")
    sets, _strip = scan_sets(report, config=config, targets=(tset,))
    request = tmp_path / "req.bin"
    request.write_bytes(_request_bytes(sets, scan_specs(sets), report))
    run_containment(qd_prep_bin, request, tmp_path / "out", threads=2, timeout_s=120.0)
    keys = (tmp_path / "out" / "exclusions.txt").read_text(encoding="utf-8").splitlines()
    assert keys == [victim.identity_key]


def test_a_v6_scan_refuses_a_corpus_where_a_benchmark_source_built_nothing(
    tmp_path: Path,
) -> None:
    """Until ``general_rows`` passes ``split_name`` to ``parse_clinc``, a v6 build refuses every
    CLINC row; the scan must say so rather than attest a corpus with no CLINC rows."""
    from containment_scan import check_benchmark_sources_built

    files = _files()
    record = tmp_path / "fetch-record.json"
    record.write_text(json.dumps([
        {"dataset": "cais/mmlu", "jsonl": str(tmp_path / "test.jsonl")},
        {"dataset": "clinc/clinc_oos", "jsonl": str(tmp_path / "train.jsonl")},
    ]), encoding="utf-8")
    config = DataConfig().with_v6_benchmark_targets()
    unstated = {
        "cais/mmlu": [r for rows in files["cais/mmlu"].values() for r in rows],
        "clinc/clinc_oos": [_clinc_v5(r) for r in files["clinc/clinc_oos"]["train"]],
    }
    report = split(dedupe(list(build_mixture(unstated, config=config).rows),  # type: ignore[arg-type]
                          config=config), config=config)
    with pytest.raises(SystemExit, match="clinc/clinc_oos"):
        check_benchmark_sources_built(report, record)
    stated = {**unstated, "clinc/clinc_oos": _v6_raw(files)["clinc/clinc_oos"]}
    report = split(dedupe(list(build_mixture(stated, config=config).rows),  # type: ignore[arg-type]
                          config=config), config=config)
    check_benchmark_sources_built(report, record)
