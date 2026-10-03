"""Per-family, report-only metrics on the FT eval row: the numbers a gate's population is
ruled on from.

J1 (GH200, full mixture, seed 0, eval row 09ff303f) failed ``permutation_consistency`` at
91.6% and put ``ood_abstain``'s in-distribution half at 14.2%, while ``code.defect_class``
choice accuracy was 99.74% and MMLU 52.7%. The gates were defined when val was
``code.defect_class`` alone; whether to scope them per family is the human's decision, and
until now the per-family numbers existed only by inference.

What is pinned here: every per-family number sums to the pooled gate or metric it breaks down
(the same rows, the same rule, one owner of it); ``--verdicts-out`` carries the second pass's
top so all of it can be recomputed offline; and rule 2 -- no gate's state moves, checked
against the pre-change functions restated as oracles.
"""

from __future__ import annotations

import dataclasses
import json
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
from test_real_ft_shuffled_label import (  # noqa: E402
    REV,
    _defect_rows,
    _offsets,
    _patch,
    _remap,
    _tokenize,
)

from qd_data.config import DataConfig  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.defect_class import DEFECT_FAMILY_ID, DEFECT_SOURCE_ID  # noqa: E402
from qd_data.loaders import ClincRow  # noqa: E402
from qd_data.manifest import build_manifests  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.rows import DataRow  # noqa: E402
from qd_data.schema import NOUL_LETTER, OPTION_LETTERS  # noqa: E402
from qd_data.split import HELD_OUT, assign_repo, split  # noqa: E402
from qd_train.calibration_fit import ece_gate  # noqa: E402
from qd_train.eval_harness import (  # noqa: E402
    degenerate_head_check,
    permutation_consistency_state,
)
from qd_train.ledger import Ledger, LedgerRow  # noqa: E402
from qd_train.shards import write_shards  # noqa: E402
from qd_train.tristate import NotRun, Ran, TriState, aggregate  # noqa: E402

CLINC_SOURCE_ID = "clinc/clinc_oos"
CLINC_FAMILIES = ("intent.classification", "intent.in_scope")

# --- the end-to-end fixture: code.defect_class + two CLINC families on the CPU stand-in -------

def _utterance(rng: random.Random) -> str:
    """Nine lowercase pseudo-words: no capital to read as an option letter, and too unlike
    each other for dedupe to fold two utterances into one."""
    return " ".join(
        "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6)) for _ in range(9)
    )


def _intents(config: DataConfig) -> tuple[list[str], list[str], str]:
    """14 in-scope intents the split puts in train, 6 it puts in val, and an out-of-scope
    label it puts in val: chosen by the split's own hash, so the val set holds CLINC rows,
    some with a ``noul`` gold, whatever the fractions are."""
    def where(name: str) -> str:
        return assign_repo(
            f"clinc-intent:{name}", seed=config.seed, train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
        )

    train: list[str] = []
    val: list[str] = []
    for i in range(400):
        name = f"intent{i:03d}"
        side = where(name)
        if side == "train" and len(train) < 14:
            train.append(name)
        elif side == "val" and len(val) < 6:
            val.append(name)
        if len(train) == 14 and len(val) == 6:
            break
    oos = next(f"oos{i}" for i in range(400) if where(f"oos{i}") == "val")
    assert len(train) == 14 and len(val) == 6
    return train, val, oos


def _clinc_rows(config: DataConfig) -> list[ClincRow]:
    train, val, oos = _intents(config)
    rng = random.Random(5)
    out: list[ClincRow] = []
    for intent in (*train, *val):
        for _ in range(5):
            out.append(ClincRow(_utterance(rng), intent, False))
    for _ in range(8):
        out.append(ClincRow(_utterance(rng), oos, True))
    return out


def _tokenizer_json(path: Path) -> Path:
    """A ``tokenizer.json`` for the byte tokenizer: the letters at their bytes and the
    byte-level newline -- all the second pass reads off a vocabulary."""
    vocab = {letter: ord(letter) for letter in (*OPTION_LETTERS, NOUL_LETTER)}
    vocab["Ċ"] = ord("\n")  # "Ċ", byte-level BPE's newline
    path.write_text(json.dumps({"model": {"vocab": vocab}}), encoding="utf-8")
    return path


@dataclasses.dataclass(frozen=True)
class FamilyCorpus:
    out: Path
    train: list[DataRow]
    val: list[DataRow]
    ledger: Path
    verdicts: Path
    tokenizer: Path


def build_family_corpus(out: Path) -> FamilyCorpus:
    """Train and val shard sets over three families, and one epoch arm scored on val with
    the permuted second pass (``--tokenizer-json``) and ``--verdicts-out``."""
    config = DataConfig()
    mixture = build_mixture(
        {DEFECT_SOURCE_ID: _defect_rows(48), CLINC_SOURCE_ID: _clinc_rows(config)},
        config=config, families=[DEFECT_FAMILY_ID, *CLINC_FAMILIES],
    )
    report = dedupe(list(mixture.rows), config=config)
    split_report = split(report, config=config)
    manifests = build_manifests(
        config=config, mixture=mixture, dedupe_report=report, split_report=split_report
    )
    paths: dict[str, Path] = {}
    for name, manifest in manifests.items():
        path = out / "data" / (HELD_OUT if name == HELD_OUT else "pool") / f"{name}.json"
        manifest.write(path)
        paths[name] = path
    train = list(split_report.rows_by_split["train"])
    val = list(split_report.rows_by_split["val"])
    for name, rows in (("train", train), ("val", val)):
        write_shards(
            paths[name], rows, out_dir=out / "shards" / name, remap=_remap(),
            tokenize=_tokenize, token_offsets=_offsets, config=config, repo_root=out,
            corpus_rev=REV,
        )
    corpus = FamilyCorpus(
        out=out, train=train, val=val, ledger=out / "ledger.jsonl",
        verdicts=out / "verdicts.jsonl", tokenizer=_tokenizer_json(out / "tokenizer.json"),
    )
    with pytest.MonkeyPatch.context() as mp:
        _patch(mp, (train, val))
        rft.main([
            "--out", str(out), "--rev", REV, "--devices", "cpu", "--seeds", "0",
            "--epoch", "--no-memorise", "--score-val", "--ledger", str(corpus.ledger),
            "--tokenizer-json", str(corpus.tokenizer), "--verdicts-out", str(corpus.verdicts),
        ])
    return corpus


@pytest.fixture(scope="module")
def family_corpus(tmp_path_factory: pytest.TempPathFactory) -> FamilyCorpus:
    return build_family_corpus(tmp_path_factory.mktemp("family-metrics"))


def _eval_row(corpus: FamilyCorpus) -> LedgerRow:
    evals = [r for r in Ledger(corpus.ledger).rows() if r.run_kind == "eval"]
    assert len(evals) == 1, [r.run_kind for r in Ledger(corpus.ledger).rows()]
    return evals[0]


def _lines(corpus: FamilyCorpus) -> list[dict[str, Any]]:
    return [json.loads(x) for x in corpus.verdicts.read_text(encoding="utf-8").splitlines()]


def _by_family(states: Mapping[str, TriState], prefix: str) -> dict[str, TriState]:
    head = f"{prefix}.family."
    return {k[len(head):]: v for k, v in states.items() if k.startswith(head)}


def _counts(state: TriState) -> tuple[int, int]:
    assert isinstance(state, Ran) and state.n is not None and state.n_total is not None, state
    return state.n, state.n_total


# --- 1. end to end on the CPU stand-in: the eval row and --verdicts-out ------------------------


def test_permutation_consistency_is_broken_down_by_family_and_sums_to_the_gate(
    family_corpus: FamilyCorpus,
) -> None:
    row = _eval_row(family_corpus)
    gate = row.gates["permutation_consistency"]
    assert isinstance(gate, Ran), "the fixture gives the second pass its tokenizer.json"
    families = _by_family(row.metrics, "permutation_consistency")
    choice = [v for v in _lines(family_corpus) if v["kind"] == "choice"]
    assert set(families) == {v["family_id"] for v in choice} == {
        DEFECT_FAMILY_ID, *CLINC_FAMILIES,
    }
    assert all(isinstance(s, Ran) and s.passed for s in families.values()), "report-only"
    assert sum(_counts(s)[0] for s in families.values()) == gate.n
    assert sum(_counts(s)[1] for s in families.values()) == gate.n_total == len(choice)
    for family, state in families.items():
        assert _counts(state)[1] == sum(1 for v in choice if v["family_id"] == family)
    assert not [k for k in row.gates if "family" in k], "a per-family number is never a gate"


def test_verdict_lines_carry_the_second_pass_and_recompute_the_family_numbers_offline(
    family_corpus: FamilyCorpus,
) -> None:
    row = _eval_row(family_corpus)
    lines = _lines(family_corpus)
    assert all(isinstance(v.get("family_id"), str) for v in lines)
    agreed_by_family: dict[str, list[bool]] = {}
    for v in lines:
        if v["kind"] != "choice":
            assert not {"top_permuted", "permutation_agreed", "perm"} & set(v), (
                "a span row is never permuted"
            )
            continue
        perm, top1, top2, noul = v["perm"], v["top"], v["top_permuted"], v["noul_row"]
        assert sorted(perm) == list(range(noul)) and all(perm[j] != j for j in range(noul))
        abstained = (top1 == noul, top2 == noul)
        assert v["permutation_agreed"] is (
            all(abstained) if any(abstained) else perm[top2] == top1
        )
        agreed_by_family.setdefault(v["family_id"], []).append(v["permutation_agreed"])
    for family, agreed in agreed_by_family.items():
        state = row.metrics[f"permutation_consistency.family.{family}"]
        assert _counts(state) == (sum(agreed), len(agreed))


def test_the_rows_gates_are_what_the_pre_change_functions_compute_from_its_verdicts(
    family_corpus: FamilyCorpus,
) -> None:
    """Rule 2 on a real run: the verdict lines rebuild both passes, and the pre-change
    functions (restated below as oracles) compute exactly the gate states on the row."""
    row = _eval_row(family_corpus)
    lines = _lines(family_corpus)
    scored = {"verdicts": lines}
    second = {"verdicts": [{**v, "top": v["top_permuted"]} for v in lines if "perm" in v]}
    perms = {(v["row_id"], v["slot_name"]): tuple(v["perm"]) for v in lines if "perm" in v}
    assert row.gates["permutation_consistency"].to_json() == (
        _pre_p2_permutation_agreement(scored, second, perms).to_json()
    )
    _, ece, degenerate = _pre_p2_calibration_states(scored)
    assert row.gates["ece"].to_json() == ece.to_json()
    assert row.controls["degenerate_head"].to_json() == degenerate.to_json()


def test_ece_is_reported_per_family_and_shape_and_never_reaches_the_gate(
    family_corpus: FamilyCorpus,
) -> None:
    row = _eval_row(family_corpus)
    shapes = {
        f"ece.family.{v['family_id']}.{v['kind']}.k{v['rows'] - rft.RESERVED_NOUL_ROWS}"
        for v in _lines(family_corpus) if v["kind"] != "span"
    }
    assert {k for k in row.metrics if k.startswith("ece.family")} == shapes
    for name in shapes:
        state = row.metrics[name]
        assert isinstance(state, NotRun) and "at least 100 examples" in state.reason, (
            "the stand-in's val families are under ece_gate's sample floor, and say so"
        )
    gate = row.gates["ece"]
    assert isinstance(gate, NotRun) and "ece.family" not in gate.reason


# --- 2. the pieces, on synthetic decodes ------------------------------------------------------

PERM4 = (1, 2, 3, 0)  # position j shows the option first shown at PERM4[j]


def _v(row_id: str, top: int, *, family: str | None, k: int = 4,
       expected: bool = False) -> dict[str, object]:
    v: dict[str, object] = {
        "kind": "choice", "row_id": row_id, "slot_name": "s", "top": top, "noul_row": k,
        "expected_abstain": expected, "correct": False,
    }
    if family is not None:
        v["family_id"] = family
    return v


Decodes = tuple[dict[str, Any], dict[str, Any], dict[tuple[str, str], tuple[int, ...]]]


def _two_families() -> Decodes:
    """fa: agree, disagree, both abstain (agree), one row with no derangement; fb: agree,
    one pass abstaining (disagree), agree. The gate is 4 of 6 asked, of 7."""
    first = [
        _v("a0", 1, family="fa"), _v("a1", 1, family="fa"), _v("a2", 4, family="fa"),
        _v("a3", 0, family="fa", k=1),
        _v("b0", 2, family="fb"), _v("b1", 4, family="fb"), _v("b2", 3, family="fb"),
    ]
    tops2 = {"a0": PERM4.index(1), "a1": PERM4.index(2), "a2": 4,
             "b0": PERM4.index(2), "b1": 0, "b2": PERM4.index(3)}
    second = [{**v, "top": tops2[str(v["row_id"])]} for v in first if v["row_id"] in tops2]
    perms = {(r, "s"): PERM4 for r in tops2}
    return {"verdicts": first}, {"verdicts": second}, perms


def _val() -> rft.ValSet:
    return rft.ValSet(reader=None, labels=[], plan=[], labels_for={}, letter_id={})  # type: ignore[arg-type]


def test_the_second_pass_lands_on_the_first_pass_verdicts_and_the_families_sum_to_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scored, second, perms = _two_families()
    monkeypatch.setattr(rft, "_decode", lambda *a, **k: second)
    gate, decoded = rft.score_permutation_consistency(
        None, _val(), rft.SecondPass([], {}, perms), scored,  # type: ignore[arg-type]
    )
    assert decoded is second
    assert _counts(gate) == (4, 6)
    by_id = {v["row_id"]: v for v in scored["verdicts"]}
    assert (by_id["a1"]["top_permuted"], by_id["a1"]["permutation_agreed"]) == (1, False)
    assert by_id["a1"]["perm"] == list(PERM4)
    assert by_id["a2"]["permutation_agreed"] is True, "both passes abstaining is agreement"
    assert not {"top_permuted", "permutation_agreed", "perm"} & set(by_id["a3"])

    families = _by_family(rft.permutation_family_metrics(scored, gate), "permutation_consistency")
    assert {f: _counts(s) for f, s in families.items()} == {"fa": (2, 3), "fb": (2, 3)}
    assert all(isinstance(s, Ran) and s.passed for s in families.values())
    assert "1 had fewer than two live options" in families["fa"].detail  # type: ignore[union-attr]

    lines = rft._verdict_lines(scored, eval_row_id="e", seed=0)
    assert [(x["row_id"], x.get("top_permuted"), x.get("permutation_agreed")) for x in lines] == [
        ("a0", 0, True), ("a1", 1, False), ("a2", 4, True), ("a3", None, None),
        ("b0", 1, True), ("b1", 0, False), ("b2", 2, True),
    ]
    assert all(x["family_id"] in ("fa", "fb") for x in lines)


def test_per_family_permutation_numbers_fail_closed_and_never_raise() -> None:
    scored, second, perms = _two_families()
    not_run = rft.permutation_family_metrics(scored, NotRun(reason="no tokenizer.json"))
    assert set(not_run) == {"permutation_consistency.family.fa",
                            "permutation_consistency.family.fb"}
    assert all(isinstance(s, NotRun) and "no tokenizer.json" in s.reason
               for s in not_run.values())

    gate = rft.permutation_agreement(scored, second, perms)
    unannotated = rft.permutation_family_metrics(scored, gate)
    assert list(unannotated) == ["permutation_consistency.family"], (
        "families that do not sum to the gate are not reported as numbers"
    )
    assert "do not carry the second pass" in unannotated[  # type: ignore[union-attr]
        "permutation_consistency.family"].reason

    anonymous = {"verdicts": [_v("x", 1, family=None)]}
    got = rft.permutation_family_metrics(anonymous, gate)
    assert list(got) == ["permutation_consistency.family"]
    assert "carry no family_id" in got["permutation_consistency.family"].reason  # type: ignore[union-attr]


PROSE = [f"Which of these statements about subject {i} holds?" for i in range(150)]


def test_score_ood_breaks_the_in_distribution_bound_down_by_family_and_counts_noul_golds_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qd_train.ood import build_ood_suite, ood_gate

    cases = build_ood_suite(PROSE, per_category=2, seed=0)
    passes = iter([
        {"verdicts": [_v(c.case_id, 4, family=DEFECT_FAMILY_ID) for c in cases]},
        {"verdicts": [_v(c.case_id, 0, family=DEFECT_FAMILY_ID) for c in cases]},
    ])
    monkeypatch.setattr(rft, "_decode", lambda *a, **k: next(passes))
    suite = rft.OodSuite(cases, _val(), rft.SecondPass([], {}, {(c.case_id, "s"): PERM4
                                                                 for c in cases}))
    first = [
        _v("fa0", 1, family="fa"), _v("fa1", 4, family="fa"), _v("fa2", 1, family="fa"),
        _v("fag0", 4, family="fa", expected=True), _v("fag1", 2, family="fa", expected=True),
        _v("fb0", 1, family="fb"), _v("fb1", 3, family="fb"),
    ]
    tops2 = {"fa0": PERM4.index(1), "fa1": 0, "fa2": PERM4.index(1), "fag0": 4,
             "fag1": PERM4.index(2), "fb0": PERM4.index(2), "fb1": PERM4.index(3)}
    scored = {"verdicts": first}
    in_second = {"verdicts": [{**v, "top": tops2[str(v["row_id"])]} for v in first]}
    in_perms = {(r, "s"): PERM4 for r in tops2}
    gate, metrics, _ = rft.score_ood(
        None, suite, scored=scored, val_second=in_second,  # type: ignore[arg-type]
        val_second_pass=rft.SecondPass([], {}, in_perms),
    )
    pooled = metrics["ood_abstain.in_distribution"]
    assert _counts(pooled) == (2, 5), "fa1 answered noul; fb0 moved under the derangement"
    families = _by_family(metrics, "ood_abstain.in_distribution")
    assert {f: _counts(s) for f, s in families.items()} == {"fa": (1, 3), "fb": (1, 2)}
    assert all(isinstance(s, Ran) and s.passed for s in families.values())

    gold_noul = _by_family(metrics, "ood_abstain.in_distribution.gold_noul")
    assert _counts(gold_noul["fa"]) == (1, 2), "fag0 abstained; fag1 answered an option"
    assert isinstance(gold_noul["fb"], NotRun) and "no fb val choice row" in gold_noul["fb"].reason

    want = ood_gate(
        ood_abstained=len(cases), ood_total=len(cases),
        **dict(zip(("in_abstained", "in_total"), _counts(pooled), strict=True)),
    )
    assert gate.to_json() == want.to_json()
    indist = _pre_p2_choice_rule_abstentions(scored, in_second, in_perms)
    assert (sum(indist.values()), len(indist)) == (2, 5)


def _calibrated(family: str | None, n: int, start: int = 0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(start, start + n):
        v: dict[str, object] = {
            "kind": "choice", "row_id": f"r{i}", "rows": 3, "gold_row": 0 if i % 2 else 1,
            "row_logits": [40.0, 0.0, 0.0] if i % 2 else [0.0, 40.0, 0.0],
            "language": "python",
        }
        if family is not None:
            v["family_id"] = family
        rows.append(v)
    return rows


def test_ece_per_family_runs_above_the_floor_says_why_below_it_and_leaves_the_gate_alone() -> None:
    verdicts = _calibrated("fa", 120) + _calibrated("fb", 30, start=120)
    metrics, ece, degenerate = rft.calibration_states({"verdicts": verdicts})
    fa = metrics["ece.family.fa.choice.k2"]
    assert isinstance(fa, Ran) and fa.n == 120 and fa.passed
    fb = metrics["ece.family.fb.choice.k2"]
    assert isinstance(fb, NotRun) and "at least 100 examples" in fb.reason
    old_metrics, old_ece, old_degenerate = _pre_p2_calibration_states({"verdicts": verdicts})
    assert (ece.to_json(), degenerate.to_json()) == (old_ece.to_json(), old_degenerate.to_json())
    assert {k: v.to_json() for k, v in metrics.items() if k in old_metrics} == {
        k: v.to_json() for k, v in old_metrics.items()
    }
    # The metric-only additions, and nothing else: P2's per-family ECEs, then Fable G's
    # per-family degenerate_head numbers and pooled / per-language ECE (family_heads,
    # report_eces), then the structured class share per slot shape (2026-10-03), which the
    # verdict reads only under a share-only degenerate_head_floor. Each is a metric only; the
    # two states above are unchanged.
    assert set(metrics) - set(old_metrics) == {
        "ece.family.fa.choice.k2", "ece.family.fb.choice.k2",
        "degenerate_head.family.fa.choice.k2", "degenerate_head.family.fb.choice.k2",
        "ece.report.pooled", "ece.report.lang.python",
        "degenerate_head.choice.k2.top_class_share",
    }

    anonymous, _, _ = rft.calibration_states({"verdicts": _calibrated(None, 5)})
    assert "carry no family_id" in anonymous["ece.family"].reason  # type: ignore[union-attr]


# --- 3. rule 2: no gate moves, against the pre-change functions -------------------------------
#
# Restated verbatim from tools/real_ft_run.py at f622660, before the per-family metrics: the
# gate's agreement count, the in-distribution bound's rule, and the ece/degenerate_head states.
# Helpers they call (letter_distributions, ece_gate, language_eces, aggregate) are unchanged.


def _pre_p2_permutation_agreement(
    scored: Mapping[str, Any], second: Mapping[str, Any],
    perms: Mapping[tuple[str, str], tuple[int, ...]],
) -> TriState:
    first = {
        (str(v["row_id"]), str(v["slot_name"])): v
        for v in scored["verdicts"]
        if v["kind"] == "choice"
    }
    again = {
        (str(v["row_id"]), str(v["slot_name"])): v
        for v in second["verdicts"]
        if v["kind"] == "choice"
    }
    agree = asked = 0
    for key, v in first.items():
        perm = perms.get(key)
        if perm is None:
            continue
        w = again.get(key)
        if w is None:
            raise SystemExit(f"row {key} was decoded in the first pass and not the second")
        asked += 1
        top1, top2 = int(v["top"]), int(w["top"])
        abstained1 = top1 == int(v["noul_row"])
        abstained2 = top2 == int(w["noul_row"])
        if abstained1 or abstained2:
            agree += int(abstained1 and abstained2)
        elif perm[top2] == top1:
            agree += 1
    return permutation_consistency_state(agree=agree, asked=asked, total=len(first))


def _pre_p2_choice_rule_abstentions(
    first: Mapping[str, Any], second: Mapping[str, Any],
    perms: Mapping[tuple[str, str], tuple[int, ...]],
) -> dict[str, bool]:
    again = {
        (str(v["row_id"]), str(v["slot_name"])): v
        for v in second["verdicts"]
        if v["kind"] == "choice"
    }
    out: dict[str, bool] = {}
    for v in first["verdicts"]:
        if v["kind"] != "choice" or bool(v["expected_abstain"]):
            continue
        key = (str(v["row_id"]), str(v["slot_name"]))
        top1 = int(v["top"])
        abstained = top1 == int(v["noul_row"])
        perm = perms.get(key)
        if perm is not None:
            w = again.get(key)
            if w is None:
                raise SystemExit(f"row {key} was decoded in the first pass and not the second")
            top2 = int(w["top"])
            abstained = abstained or top2 == int(w["noul_row"]) or perm[top2] != top1
        out[key[0]] = abstained
    return out


def _pre_p2_calibration_states(
    scored: Mapping[str, Any],
) -> tuple[dict[str, TriState], TriState, TriState]:
    verdicts = scored["verdicts"]
    metrics: dict[str, TriState] = {}
    eces: dict[str, TriState] = {}
    degenerate: dict[str, TriState] = {}
    for key, (probs, gold) in rft.letter_distributions(verdicts).items():
        eces[f"ece.{key}"] = ece_gate(probs, gold)
        degenerate[f"degenerate_head.{key}"] = degenerate_head_check(probs)
    eces.update(rft.language_eces(verdicts))
    metrics.update(eces)
    metrics.update(degenerate)
    if not eces:
        eces["ece.lang"] = NotRun(reason="no letter rows were decoded")
    return metrics, aggregate(eces, name="ece"), aggregate(degenerate, name="degenerate_head")


@st.composite
def _decodes(draw: st.DrawFn) -> Decodes:
    """A first pass over up to 40 letter rows of 2-4 options in three families, its second
    pass, and a derangement for most of the choice rows -- the rest were never asked."""
    first: list[dict[str, object]] = []
    second: list[dict[str, object]] = []
    perms: dict[tuple[str, str], tuple[int, ...]] = {}
    for i in range(draw(st.integers(0, 40))):
        k = draw(st.integers(2, 4))  # letters_key refuses a letter row of fewer options
        kind = "score" if draw(st.booleans()) else "choice"
        v: dict[str, object] = {
            "kind": kind, "row_id": f"r{i}", "slot_name": "s",
            "family_id": draw(st.sampled_from(["fa", "fb", "fc"])),
            "top": draw(st.integers(0, k)), "noul_row": k, "rows": k + 1,
            "expected_abstain": draw(st.booleans()), "gold_row": draw(st.integers(0, k)),
            "language": draw(st.sampled_from([None, "python", "go"])),
            "row_logits": draw(st.lists(st.floats(-8.0, 8.0), min_size=k + 1, max_size=k + 1)),
        }
        first.append(v)
        # Some choice rows were never asked: they have no derangement in the map.
        if kind == "choice" and draw(st.integers(0, 4)):
            perms[(f"r{i}", "s")] = (*range(1, k), 0)
            second.append({**v, "top": draw(st.integers(0, k))})
    return {"verdicts": first}, {"verdicts": second}, perms


@settings(max_examples=200, deadline=None)
@given(_decodes())
def test_rule_2_no_gate_state_moves(decodes: Decodes) -> None:
    """The gates' states are byte-identical to the pre-change functions' on the same inputs
    -- before and after the second pass is written onto the verdicts -- and the per-family
    numbers that ran sum to the gate. A characterization: it holds of the pre-change code by
    construction; what it guards is the change."""
    scored, second, perms = decodes
    old_perm = _pre_p2_permutation_agreement(scored, second, perms).to_json()
    old_indist = _pre_p2_choice_rule_abstentions(scored, second, perms)
    old_metrics, old_ece, old_degenerate = _pre_p2_calibration_states(scored)

    gate = rft.permutation_agreement(scored, second, perms)
    assert gate.to_json() == old_perm
    assert rft.choice_rule_abstentions(scored, second, perms) == old_indist
    rft.annotate_second_pass(scored, second, perms)
    assert rft.permutation_agreement(scored, second, perms).to_json() == old_perm
    assert rft.choice_rule_abstentions(scored, second, perms) == old_indist
    metrics, ece, degenerate = rft.calibration_states(scored)
    assert (ece.to_json(), degenerate.to_json()) == (old_ece.to_json(), old_degenerate.to_json())
    assert {k: v.to_json() for k, v in metrics.items() if k in old_metrics} == {
        k: v.to_json() for k, v in old_metrics.items()
    }
    # Every addition is a named metric (P2's ece.family; Fable G's degenerate_head.family and
    # ece.report; the per-shape class share of 2026-10-03), never a renamed or moved gate input.
    assert all(k.startswith(("ece.family", "degenerate_head.family", "ece.report."))
               or (k.startswith("degenerate_head.") and k.endswith(".top_class_share"))
               for k in set(metrics) - set(old_metrics))

    families = rft.permutation_family_metrics(scored, gate)
    ran = [s for s in families.values() if isinstance(s, Ran)]
    if isinstance(gate, Ran):
        assert sum(_counts(s)[0] for s in ran) == gate.n
        assert sum(_counts(s)[1] for s in ran) == gate.n_total
    gold_noul = rft.choice_rule_abstentions(scored, second, perms, gold_noul=True)
    assert not set(gold_noul) & set(old_indist), "the two populations are disjoint"
    split = rft.in_distribution_family_metrics(scored, old_indist, gold_noul)
    in_ran = [s for k, s in _by_family(split, "ood_abstain.in_distribution").items()
              if isinstance(s, Ran)]
    assert sum(_counts(s)[0] for s in in_ran) == sum(old_indist.values())
    assert sum(_counts(s)[1] for s in in_ran) == len(old_indist)


