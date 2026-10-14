"""``tools/real_ft_run.py --score-checkpoint --arm-b-candidates``: the Arm B throughput probe.

Fable's 2026-10-14 ruling
(``AUDIT/board-sweep-2026-10-13/fable-arm-b-probe-construction-ruling.md``, ruling 2 (c)):
the 200 candidate rows ``tools/arm_b_mine.py probe`` wrote are tokenized by the
pipeline into a train-split, refuse-gold shard set and scored by a mode of ``--score-checkpoint``
modelled on the OOD-only trajectory row and the composed slice: no ``--score-val``, the val set
never opened, never a gate population, a quick row tagged ``arm-b-probe`` that records
``rows_per_s`` and the abstention count and no gate, and verdict lines for
``arm_b_mine.abstentions``. Everything here is CPU-only and over tiny synthetic shard sets.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import arm_b_mine as mine  # noqa: E402
import real_ft_run as rft  # noqa: E402
from test_real_ft_composed_slice import _run_args  # noqa: E402
from test_real_ft_score_checkpoint import _CapturedRecorder, _defect_shards  # noqa: E402
from test_real_ft_shuffled_label import REV, _defect_rows, _offsets, _remap, _tokenize  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.defect_class import DEFECT_FAMILY_ID  # noqa: E402
from qd_train.artifacts import ShardContractViolation  # noqa: E402
from qd_train.shards import ShardReader, write_shards  # noqa: E402
from qd_train.tristate import NotRun  # noqa: E402

BUILD_ROW = "165ba699-0000-4000-8000-000000000001"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --- the argv -----------------------------------------------------------------------------------


def _argv(tmp_path: Path, *, drop: tuple[str, ...] = (), extra: tuple[str, ...] = (),
          set_out: Path | None = None) -> list[str]:
    """A complete candidate-set argv (v5 seed 1's shape), less the flags in ``drop`` (each a
    flag and its one value), plus ``extra``."""
    probe = set_out if set_out is not None else tmp_path / "w" / "probe"
    pairs = [
        ("--out", str(tmp_path)), ("--rev", "0" * 40), ("--real-backbone", "/x"),
        ("--ft-ledger", "/l"), ("--devices", "mps"), ("--seeds", "1"),
        ("--score-checkpoint", "/c/epoch-seed1-cuda-step12176.json"),
        ("--ft-row-id", "b0d9454e"),
        ("--arm-b-candidates", str(probe)),
        ("--arm-b-candidates-corpus", str(probe / "corpus" / "c")),
        ("--arm-b-candidates-build-row", BUILD_ROW[:8]),
    ]
    out = [x for flag, value in pairs if flag not in drop for x in (flag, value)]
    return [*out, *extra]


def test_a_complete_candidate_argv_passes_every_argv_check(tmp_path):
    # Every argv check passed; the first disk read is resolving --rev, which no repository here has.
    with pytest.raises(subprocess.CalledProcessError, match="rev-parse"):
        rft.main(_argv(tmp_path))


@pytest.mark.parametrize(
    ("drop", "extra", "match"),
    [
        (("--score-checkpoint", "--ft-row-id"), (), "--arm-b-candidates needs --score-checkpoint"),
        ((), ("--score-val",), "refuses --score-val"),
        ((), ("--ood", "--ood-general-record", "/g.json"), "refuses --ood"),
        ((), ("--needle",), "refuses --needle"),
        ((), ("--score-plan", "/p.json"), "refuses --score-plan"),
        ((), ("--composed-slice", "/s"), "refuses --composed-slice"),
        (("--arm-b-candidates-corpus",), (), "needs --arm-b-candidates-corpus"),
        (("--arm-b-candidates-build-row",), (), "needs --arm-b-candidates-build-row"),
    ],
)
def test_candidate_argv_is_refused_before_anything_loads(tmp_path, drop, extra, match):
    with pytest.raises(SystemExit, match=match):
        rft.main(_argv(tmp_path, drop=drop, extra=extra))


def test_the_companion_flags_without_the_set_are_refused(tmp_path):
    argv = _argv(tmp_path, drop=("--arm-b-candidates",))
    with pytest.raises(SystemExit, match="read only with --arm-b-candidates"):
        rft.main(argv)


@pytest.mark.parametrize("marker", ["heldout", "held_out", "held-out"])
def test_a_held_out_marked_candidate_path_is_refused_before_anything_loads(tmp_path, marker):
    """Rule 3: every input and output path of the mode passes the path check."""
    held = tmp_path / marker / "probe"
    with pytest.raises(SystemExit, match="held-out"):
        rft.main(_argv(tmp_path, set_out=held))
    with pytest.raises(SystemExit, match="held-out"):
        rft.main(_argv(tmp_path, extra=("--verdicts-out", str(tmp_path / marker / "v.jsonl"))))
    with pytest.raises(SystemExit, match="held-out"):
        rft.main(_argv(tmp_path, extra=("--arm-b-candidates-ledger",
                                        str(tmp_path / marker / "l.jsonl"))))
    assert not held.exists()


# --- the set, opened ----------------------------------------------------------------------------


@dataclasses.dataclass
class CandidateWorld:
    tmp: Path
    train: ShardReader
    set_out: Path
    corpus: Path
    ledger: Path
    defect: list
    config: DataConfig


def _candidate_world(tmp_path, monkeypatch, *, policy="refuse-gold", remap=None,
                     build_over=None, probe_over=None, drop_ids=0) -> CandidateWorld:
    """v5's train set, and beside it a candidate set: the same rows tokenized again under their
    own out dir (so a shard set of its own), its corpus, ``probe.json`` and the pipeline's build
    row. ``load_defect_rows`` is stood in for by the rows the shards were written from."""
    out = tmp_path / "v5"
    train_rows, _ = _defect_shards(out)
    config = DataConfig()
    ids = {r.row_id for r in train_rows}
    defect = [r for r in _defect_rows(48)
              if f"qdm:{DEFECT_FAMILY_ID}:{r.example_id}" in ids]
    assert len(defect) == len(train_rows) > 4
    set_out = tmp_path / "w" / "probe"
    write_shards(
        out / "data" / "pool" / "train.json", train_rows, out_dir=set_out / "shards" / "train",
        remap=remap if remap is not None else _remap(), tokenize=_tokenize,
        token_offsets=_offsets, config=config, repo_root=out, corpus_rev=REV,
        span_collapse_policy=policy,
    )
    corpus = set_out / "corpus" / "c"
    corpus.mkdir(parents=True)
    examples = b"".join(json.dumps({"id": r.example_id}).encode() + b"\n" for r in defect)
    (corpus / "examples.jsonl").write_bytes(examples)
    (corpus / "manifest.json").write_bytes(b'{"examples_sha256": "x"}')
    probe = {
        "schema": "qd-arm-b-probe/v1", "rows": len(defect),
        "row_ids": [r.example_id for r in defect],
        "corpus": {"name": "c", "examples_sha256": _sha(examples),
                   "manifest_sha256": _sha((corpus / "manifest.json").read_bytes())},
        **(probe_over or {}),
    }
    (set_out / "probe.json").write_text(json.dumps(probe))
    built = ShardReader(set_out / "shards" / "train", config=config, repo_root=set_out,
                        expect_rev=REV)
    row = {
        "row_id": BUILD_ROW, "run_kind": "smoke", "status": "completed",
        "recipe": {
            "tool": "tools/real_tokenizer_pipeline.py", "rev": REV,
            "defect_class_examples_sha256": _sha(examples), "span_collapse_policy": "refuse-gold",
        },
        "metrics": {"shard_sequences": {
            "state": "ran", "passed": True, "value": len(built), "n": len(built),
            "n_total": len(built), "detail": "x"}},
        **(build_over or {}),
    }
    ledger = tmp_path / "arm-b-shards.jsonl"
    ledger.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(
        "qd_data.defect_class.load_defect_rows",
        lambda *a, **k: SimpleNamespace(rows=defect[: len(defect) - drop_ids]),
    )
    train = ShardReader(out / "shards" / "train", config=config, repo_root=out, expect_rev=REV)
    return CandidateWorld(tmp_path, train, set_out, corpus, ledger, defect, config)


def _open(w: CandidateWorld, **over):
    kwargs = dict(
        corpus_dir=w.corpus, download_root=w.tmp / "dl", repo_root=w.tmp, config=w.config,
        train=w.train, letter_id={}, build_ledger=w.ledger, build_row=BUILD_ROW[:8],
    )
    kwargs.update(over)
    return rft.open_candidate_set(w.set_out, **kwargs)


def test_the_candidate_set_is_never_a_gate_population_and_never_opened_as_val(
    tmp_path, monkeypatch
):
    def boom(*a, **k):
        raise AssertionError("the candidate set was opened through open_val_reader")

    monkeypatch.setattr(rft, "open_val_reader", boom)
    w = _candidate_world(tmp_path, monkeypatch)
    cand = _open(w)
    header = cand.val.reader.header
    assert header.split == "train" and header.span_collapse_policy == "refuse-gold"
    assert not header.report_only
    # The gate readers' own check refuses it: this set can never be a gate's population.
    with pytest.raises(ShardContractViolation, match="not a gate population"):
        header.require_gate_population(where="the candidate set")
    assert header.remap_hash == w.train.header.remap_hash
    assert len(cand.val.labels) == len(cand.val.reader) > 0
    assert sum(len(g) for g in cand.val.labels_for.values()) == len(cand.val.reader)
    assert cand.val.plan and cand.recipe["build_row"] == BUILD_ROW
    assert cand.recipe["shard_hash"] == header.shard_hash()
    assert cand.recipe["rows_not_in_shard"] == 0 and cand.recipe["rows"] == len(w.defect)


def test_the_candidate_set_is_refused_whenever_it_could_be_scored_as_something_else(
    tmp_path, monkeypatch
):
    # A refuse-any set is a gate population: the gate readers accept it, so this mode must not.
    w = _candidate_world(tmp_path / "a", monkeypatch, policy="refuse-any")
    with pytest.raises(SystemExit, match="refuse-gold"):
        _open(w)
    # Another remap: the same id would name a different embedding row in the scored model.
    w = _candidate_world(
        tmp_path / "b", monkeypatch,
        remap=dataclasses.replace(_remap(), tokenizer_hash="another-tokenizer"),
    )
    with pytest.raises(SystemExit, match="remap"):
        _open(w)
    # A build row that built another corpus, another revision, or another number of sequences.
    for over, match in (
        ({"recipe": {"tool": "tools/real_tokenizer_pipeline.py", "rev": REV,
                     "defect_class_examples_sha256": "0" * 64,
                     "span_collapse_policy": "refuse-gold"}}, "examples"),
        ({"recipe": {"tool": "tools/real_tokenizer_pipeline.py", "rev": "b" * 40,
                     "span_collapse_policy": "refuse-gold"}}, "rev|examples"),
        ({"status": "failed"}, "completed"),
    ):
        w = _candidate_world(tmp_path / f"c{match[:3]}", monkeypatch, build_over=over)
        with pytest.raises(SystemExit, match=match):
            _open(w)
    w = _candidate_world(tmp_path / "d", monkeypatch)
    with pytest.raises(SystemExit, match="0 rows match"):
        _open(w, build_row="deadbeef")
    with pytest.raises(SystemExit, match="at least 8"):
        _open(w, build_row="165b")
    # A corpus that is not the one probe.json pinned, and rows the probe never named.
    (w.corpus / "examples.jsonl").write_bytes(b"{}\n")
    with pytest.raises(SystemExit, match=r"examples\.jsonl hashes"):
        _open(w)
    w = _candidate_world(tmp_path / "e", monkeypatch, probe_over={"row_ids": ["only-one"]})
    with pytest.raises(SystemExit, match=r"not in probe\.json"):
        _open(w)


def test_a_loader_row_the_shards_lack_is_counted_and_a_shard_row_the_loader_lacks_is_refused(
    tmp_path, monkeypatch
):
    """The pipeline may drop a row (dedupe) the loader still returns: counted, never silent. A
    shard row with no loader row cannot be labelled: refused."""
    w = _candidate_world(tmp_path / "a", monkeypatch)
    extra = dataclasses.replace(w.defect[0], example_id="not-in-the-shards:x.py#0",
                                pool_id="not-in-the-shards:x.py")
    monkeypatch.setattr("qd_data.defect_class.load_defect_rows",
                        lambda *a, **k: SimpleNamespace(rows=[*w.defect, extra]))
    probe = json.loads((w.set_out / "probe.json").read_text())
    probe["row_ids"].append(extra.example_id)
    (w.set_out / "probe.json").write_text(json.dumps(probe))
    assert _open(w).recipe["rows_not_in_shard"] == 1
    w = _candidate_world(tmp_path / "b", monkeypatch, drop_ids=2)
    with pytest.raises(SystemExit, match="sequence"):
        _open(w)


# --- the row -------------------------------------------------------------------------------------


def _first_pass() -> dict:
    def choice(rid, top, expected_abstain=False):
        return {"row_id": rid, "slot_name": "defect_class", "kind": "choice", "correct": True,
                "top": top, "gold_row": 0, "noul_row": 4, "expected_abstain": expected_abstain,
                "rows": 5, "family_id": "code.defect_class"}

    return {
        "verdicts": [
            choice("r0", 4),                      # the first pass abstains
            choice("r1", 1),                      # agrees under the permutation
            choice("r2", 2),                      # the permuted winner maps to another option
            choice("r3", 4, expected_abstain=True),  # gold is noul: never counted
            {"row_id": "r0", "slot_name": "defect_span", "kind": "span", "correct": True,
             "top": [1, 1], "gold_row": 1, "noul_row": 9},
        ],
        "rows_decoded": 5, "rows_not_decoded": 0, "letters_without_id": [],
    }


def _second_pass_verdicts() -> dict:
    def choice(rid, top):
        return {"row_id": rid, "slot_name": "defect_class", "kind": "choice", "top": top,
                "noul_row": 4, "correct": False, "gold_row": 0}

    return {"verdicts": [choice("r0", 0), choice("r1", 0), choice("r2", 0), choice("r3", 0)]}


PERM = (1, 2, 3, 0)
SECOND = rft.SecondPass(
    ["second"], {}, {("r0", "defect_class"): PERM, ("r1", "defect_class"): PERM,
                     ("r2", "defect_class"): PERM, ("r3", "defect_class"): PERM},
)


class _Clock:
    def __init__(self, *ticks: float):
        self.ticks = list(ticks)

    def __call__(self) -> float:
        return self.ticks.pop(0)


def _run(monkeypatch, *, first=None, second_pass=SECOND, clock=None, captured=None):
    captured = {} if captured is None else captured
    asked: list[object] = []

    def decode(step, batches, labels_for, letter_id):
        asked.append(batches)
        if batches is second_pass.batches:
            return _second_pass_verdicts()
        return first or _first_pass()

    def recorder(ledger, **kw):
        captured.update(kw)
        captured["recorder"] = _CapturedRecorder(captured)
        return captured["recorder"]

    monkeypatch.setattr(rft, "_decode", decode)
    monkeypatch.setattr(rft, "_recorder", recorder)
    monkeypatch.setattr(rft, "_cost", lambda **k: None)
    reader = SimpleNamespace(header=SimpleNamespace(shard_hash=lambda: "t" * 64))
    candidates = rft.CandidateSet(
        val=SimpleNamespace(plan=["first"], labels_for={}, letter_id={}),  # type: ignore[arg-type]
        recipe={"build_row": BUILD_ROW, "shard_hash": "c" * 64, "rows": 4},
    )
    ft = {"row_id": "ft0", "metrics": {"train.termination": {"value": "steps_exhausted"}}}
    loaded = ("step", ft, {"lr": 1e-5}, 1, {"sidecar": {"digest": "d" * 64}})
    result = rft.run_arm_b_probe(
        _run_args(), loaded=loaded, load_s=7.5, reader=reader, candidates=candidates,
        second_pass=second_pass, device="mps", ledger=None,  # type: ignore[arg-type]
        reasons_for=lambda tag, device, termination=None: ["device mps is not the campaign's"],
        clock=clock or _Clock(100.0, 150.0),
    )
    return result, captured, asked


def test_the_probe_row_is_quick_records_rows_per_s_and_abstained_and_no_gate(monkeypatch):
    (row_id, lines, seed), captured, asked = _run(monkeypatch)
    assert (row_id, seed, len(lines)) == ("control-row", 1, 5)
    assert asked == [["first"], ["second"]], "both passes, over the candidate set's own plan"
    metrics = captured["metrics"]
    assert metrics["ft_run_row_id"].value == "ft0"
    assert set(metrics) == {
        "ft_run_row_id", "arm_b_probe.rows_decoded", "arm_b_probe.choice_rows",
        "arm_b_probe.decode_s", "arm_b_probe.load_s", "arm_b_probe.rows_per_s",
        "arm_b_probe.abstained",
    }, "no val, needle, OOD or gate metric: nothing here is a gate"
    assert metrics["arm_b_probe.rows_decoded"].value == 5
    assert metrics["arm_b_probe.choice_rows"].value == 4
    assert metrics["arm_b_probe.decode_s"].value == pytest.approx(50.0)
    assert metrics["arm_b_probe.load_s"].value == pytest.approx(7.5), "load is recorded apart"
    assert metrics["arm_b_probe.rows_per_s"].value == pytest.approx(4 / 50.0)
    # r0 abstains on the first pass, r2 on the permuted mapping, r3's gold is noul: 2 of 3.
    assert metrics["arm_b_probe.abstained"].value == 2
    assert "gates nothing" in metrics["arm_b_probe.abstained"].detail
    assert captured["recipe"]["tag"] == rft.ARM_B_PROBE_TAG == "arm-b-probe"
    assert captured["recipe"]["arm_b_candidates"]["build_row"] == BUILD_ROW
    assert captured["recipe"]["shard_hash"] == "t" * 64, "protocol pinned to the train set's"
    assert rft.ARM_B_PROBE_QUICK_REASON in captured["quick_reasons"]
    assert "device mps is not the campaign's" in captured["quick_reasons"]
    assert "no yield estimate" in rft.ARM_B_PROBE_QUICK_REASON
    assert captured["run_kind"] == "eval" and captured["wall_clock_s"] == pytest.approx(50.0)
    assert isinstance(captured["recorder"].noul_rate, NotRun)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"clock": _Clock(5.0, 5.0)}, "did not advance"),
        ({"second_pass": rft.SecondPass([], {}, {}, not_run="no tokenizer.json")},
         "second pass did not run"),
        ({"first": {**_first_pass(), "rows_not_decoded": 2, "letters_without_id": ["E"]}},
         "2 candidate sequences were not decoded"),
    ],
)
def test_a_probe_that_did_not_measure_writes_no_row(monkeypatch, kwargs, match):
    captured: dict[str, object] = {}
    with pytest.raises(SystemExit, match=match):
        _run(monkeypatch, captured=captured, **kwargs)
    assert "recorder" not in captured, "a refused probe recorded a row"


def test_probe_lines_carry_every_key_arm_b_mine_abstentions_reads(monkeypatch):
    (_, lines, _), _, _ = _run(monkeypatch)
    choice = [x for x in lines if x["kind"] == "choice"]
    assert len(choice) == 4
    for line in lines:
        assert line["gate"] == rft.ARM_B_PROBE_GATE == "arm_b.probe.report_only"
        assert {"eval_row_id", "seed", "row_id", "kind", "correct"} <= set(line)
    for line in choice:
        assert {"top", "noul_row", "expected_abstain", "perm", "top_permuted"} <= set(line)
    # arm_b_mine's reader on the scorer's own lines gives the scorer's count.
    assert mine.abstentions(lines) == {"r0": True, "r1": False, "r2": True}
    json.dumps(lines)  # one JSON object per line, as --verdicts-out writes them
