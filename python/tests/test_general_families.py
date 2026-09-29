"""General decision families: MMLU, CommonsenseQA, and the two-stage CLINC design.

Offline only. No row of any of these datasets is on this host and the lane may not
download one, so the fixtures below are synthetic rows in the **upstream** shape
(datasets-server ``/info`` for ``cais/mmlu`` config ``all``, ``tau/commonsense_qa``
``default`` and ``clinc/clinc_oos`` ``plus``, read 2026-09-29). They prove the
parsers and rewriters against that schema; they do not prove the schema still
holds, which is what a network test over real rows would add once a download is
approved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.general import (
    CLINC_DOMAIN_FAMILY,
    CLINC_DOMAINS_SHA256,
    CLINC_TWO_STAGE_FAMILIES,
    CLINC_WITHIN_DOMAIN_FAMILY,
    CSQA_FAMILY,
    MAX_DOMAIN_MAP_BYTES,
    MMLU_FAMILY,
    PINNED_SPLIT_KEY,
    REPLAY_ONLY,
    REPLAY_ROLE_KEY,
    ClincDomainMap,
    ReplayPartition,
    load_clinc_domains,
    partition_replay,
    rewrite_clinc_two_stage,
    rewrite_csqa,
    rewrite_mmlu,
)
from qd_data.loaders import (
    ClincRow,
    CsqaRow,
    MalformedRowRefusal,
    MmluRow,
    SourceUnavailableRefusal,
    check_read_admitted,
    fetch_rows,
    parse_clinc,
    parse_csqa,
    parse_mmlu,
)
from qd_data.mixture import RowRefused, check_prompt_consistency
from qd_data.render import render
from qd_data.schema import NOUL, NOUL_LETTER
from qd_data.sources import (
    TASK_FAMILIES,
    admitted_task_families,
    source_by_id,
    task_family_by_id,
)
from qd_data.split import split

CONFIG = DataConfig()


def mmlu_raw(i: int, **over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "question": f"Which number is the successor of {i}?",
        "subject": f"subject_{i % 7}",
        "choices": [str(i + 1), str(i + 2), str(i - 1), str(i * 3 + 5)],
        "answer": 0,
    }
    base.update(over)
    return base


def csqa_raw(i: int, **over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": f"csqa{i:05d}",
        "question": f"Where would you most likely keep item number {i}?",
        "question_concept": f"concept_{i % 5}",
        "choices": {
            "label": ["A", "B", "C", "D", "E"],
            "text": [f"drawer {i}", f"shelf {i}", f"garage {i}", f"attic {i}", f"car {i}"],
        },
        "answerKey": "B",
    }
    base.update(over)
    return base


def _noul_is_the_last_option_line(request) -> None:
    """``RESERVED_NOUL_ROWS``: the abstain row is last, after every named option."""
    prompt = render(request, seed=CONFIG.seed).prompt_for(request.slots[0].name)
    block = prompt.split("<|qd_options_begin|>\n", 1)[1].split("\n<|qd_options_end|>", 1)[0]
    lines = block.split("\n")
    assert lines[-1] == f"{NOUL_LETTER}. {NOUL}"
    assert sum(ln.startswith(f"{NOUL_LETTER}. ") for ln in lines) == 1
    assert len(lines) == len(request.slots[0].options) + 1


# -- registry ----------------------------------------------------------------


def test_the_general_families_are_registered_and_admitted() -> None:
    admitted = {f.family_id for f in admitted_task_families()}
    assert {MMLU_FAMILY, CSQA_FAMILY} <= admitted
    assert task_family_by_id(MMLU_FAMILY).source_id == "cais/mmlu"
    assert task_family_by_id(CSQA_FAMILY).source_id == "tau/commonsense_qa"
    assert MMLU_FAMILY in CONFIG.training_families
    assert CSQA_FAMILY in CONFIG.training_families


# -- loaders -----------------------------------------------------------------


def test_clinc_classlabel_ints_are_resolved_through_their_names() -> None:
    """The hub types ``intent`` as ``ClassLabel``, so real rows carry an int. The
    parser used to demand a str and would have refused every real row."""
    names = ["restaurant_reviews", "oos", "balance"]
    row = parse_clinc({"text": "what is my balance", "intent": 2}, index=0, label_names=names)
    assert row.intent == "balance" and row.is_oos is False
    oos = parse_clinc({"text": "tell me a riddle", "intent": 1}, index=1, label_names=names)
    assert oos.is_oos is True


def test_a_clinc_index_without_names_or_out_of_range_is_refused() -> None:
    with pytest.raises(MalformedRowRefusal, match="label_names"):
        parse_clinc({"text": "x", "intent": 3}, index=0)
    with pytest.raises(MalformedRowRefusal, match="intent index"):
        parse_clinc({"text": "x", "intent": 3}, index=0, label_names=["a", "b"])
    with pytest.raises(MalformedRowRefusal):
        parse_clinc({"text": "x", "intent": True}, index=0, label_names=["a", "b"])


def test_mmlu_parses_the_upstream_shape() -> None:
    row = parse_mmlu(mmlu_raw(3), index=0, split_name="test")
    assert row.choices == ("4", "5", "2", "14") and row.answer_index == 0


@pytest.mark.parametrize(
    "over",
    [{"answer": 4}, {"answer": -1}, {"answer": True}, {"answer": "A"},
     {"choices": "A,B"}, {"choices": ["a", 2]}],
)
def test_a_malformed_mmlu_row_is_refused(over: dict[str, object]) -> None:
    with pytest.raises(MalformedRowRefusal):
        parse_mmlu(mmlu_raw(1, **over), index=0, split_name="test")


def test_csqa_parses_the_upstream_shape_including_the_unlabelled_test_split() -> None:
    row = parse_csqa(csqa_raw(2), index=0, split_name="train")
    assert row.labels == ("A", "B", "C", "D", "E") and row.answer_key == "B"
    unlabelled = parse_csqa(csqa_raw(2, answerKey=""), index=0, split_name="train")
    assert unlabelled.answer_key == ""


@pytest.mark.parametrize(
    "over",
    [
        {"answerKey": "F"},
        {"choices": {"label": ["A", "A"], "text": ["x", "y"]}},
        {"choices": {"label": ["A", "B"], "text": ["x"]}},
        {"choices": ["x", "y"]},
    ],
)
def test_a_malformed_csqa_row_is_refused(over: dict[str, object]) -> None:
    with pytest.raises(MalformedRowRefusal):
        parse_csqa(csqa_raw(1, **over), index=0, split_name="train")


@pytest.mark.parametrize(
    ("config_name", "split_name"),
    [("all", "auxiliary_train"), ("auxiliary_train", "train")],
)
def test_mmlu_auxiliary_train_is_refused_before_any_fetch(
    config_name: str, split_name: str
) -> None:
    with pytest.raises(SourceUnavailableRefusal, match="ARC, OpenBookQA"):
        check_read_admitted("cais/mmlu", config_name=config_name, split_name=split_name)
    # fetch_rows refuses on the same check, before opening a connection.
    with pytest.raises(SourceUnavailableRefusal, match="ARC, OpenBookQA"):
        fetch_rows("cais/mmlu", config_name=config_name, split_name=split_name, limit=1)
    check_read_admitted("cais/mmlu", config_name="all", split_name="validation")


# -- MMLU / CSQA rewriters -----------------------------------------------------


def test_mmlu_renders_a_choice_whose_gold_is_the_option_text() -> None:
    row = rewrite_mmlu(
        parse_mmlu(mmlu_raw(5), index=0, split_name="test"), family_id=MMLU_FAMILY,
        index=0, config=CONFIG)
    assert row.gold[0].value == "6" and row.gold[0].is_noul is False
    assert row.request.slots[0].options == ("6", "7", "4", "20")
    assert row.repo_key == "mmlu-train:subject_5"
    assert row.licence_id == "mit" and "attribution" in row.obligations
    _noul_is_the_last_option_line(row.request)


def test_csqa_renders_a_choice_whose_gold_is_the_answer_key_text() -> None:
    row = rewrite_csqa(parse_csqa(csqa_raw(4), index=0, split_name="train"), family_id=CSQA_FAMILY,
                       index=0, config=CONFIG)
    assert row.gold[0].value == "shelf 4"
    assert row.repo_key == "csqa-train:concept_4"
    _noul_is_the_last_option_line(row.request)


def test_an_unlabelled_csqa_row_is_refused_under_its_own_code() -> None:
    with pytest.raises(RowRefused) as excinfo:
        rewrite_csqa(parse_csqa(csqa_raw(4, answerKey=""), index=0, split_name="train"),
                     family_id=CSQA_FAMILY, index=0, config=CONFIG)
    assert excinfo.value.reason_code == "unlabelled"


@pytest.mark.parametrize(
    ("choices", "code"),
    [
        (["Paris", "paris ", "Rome", "Oslo"], "duplicate_options"),
        (["Paris", "Paris", "Rome", "Oslo"], "duplicate_options"),
        (["Paris", "  ", "Rome", "Oslo"], "empty_option"),
        (["Paris", "NOUL", "Rome", "Oslo"], "option_named_noul"),
        (["Paris"], "too_few_options"),
        ([str(k) for k in range(17)], "too_many_options"),
        (["x" * 600, "b", "c", "d"], "option_too_long"),
        (["Pa‮ris", "Rome", "Oslo", "Bern"], "invisible_format_characters"),
    ],
)
def test_a_bad_option_set_is_a_counted_row_refusal_not_a_crash(
    choices: list[str], code: str
) -> None:
    """``ChoiceSlot``'s own refusals are not ``RowRefused``, so one bad upstream row
    would abort a whole build. Each is checked first and counted instead."""
    raw = MmluRow(subject="geo", question="Capital?", choices=tuple(choices), answer_index=0,
                  upstream_split="test")
    with pytest.raises(RowRefused) as excinfo:
        rewrite_mmlu(raw, family_id=MMLU_FAMILY, index=0, config=CONFIG)
    assert excinfo.value.reason_code == code


def test_empty_split_units_are_refused() -> None:
    raw = MmluRow(subject="  ", question="Q?", choices=("a", "b"), answer_index=0,
                  upstream_split="test")
    with pytest.raises(RowRefused) as excinfo:
        rewrite_mmlu(raw, family_id=MMLU_FAMILY, index=0, config=CONFIG)
    assert excinfo.value.reason_code == "missing_subject"
    c = CsqaRow(qid="q", question="Q?", concept="", labels=("A", "B"),
                texts=("x", "y"), answer_key="A", upstream_split="train")
    with pytest.raises(RowRefused) as excinfo:
        rewrite_csqa(c, family_id=CSQA_FAMILY, index=0, config=CONFIG)
    assert excinfo.value.reason_code == "missing_concept"


def test_the_wrong_family_for_the_source_is_refused() -> None:
    with pytest.raises(RowRefused) as excinfo:
        rewrite_mmlu(
            parse_mmlu(mmlu_raw(1), index=0, split_name="test"), family_id=CSQA_FAMILY,
            index=0, config=CONFIG)
    assert excinfo.value.reason_code == "unknown_family_for_source"


def test_mmlu_row_ids_do_not_depend_on_read_order_beyond_the_index() -> None:
    a = rewrite_mmlu(
        parse_mmlu(mmlu_raw(8), index=0, split_name="test"), family_id=MMLU_FAMILY,
        index=3, config=CONFIG)
    b = rewrite_mmlu(
        parse_mmlu(mmlu_raw(8), index=0, split_name="test"), family_id=MMLU_FAMILY,
        index=3, config=CONFIG)
    assert a.row_id == b.row_id and a.identity_key == b.identity_key


def test_general_rows_survive_dedupe_and_split_with_disjoint_subjects() -> None:
    rows = [
        rewrite_mmlu(
            parse_mmlu(mmlu_raw(i), index=i, split_name="test"), family_id=MMLU_FAMILY,
            index=i, config=CONFIG)
        for i in range(40)
    ] + [
        rewrite_csqa(parse_csqa(csqa_raw(i), index=i, split_name="train"), family_id=CSQA_FAMILY,
                     index=i, config=CONFIG)
        for i in range(40)
    ]
    report = dedupe(rows, config=CONFIG)
    result = split(report, config=CONFIG)
    assert result.repo_disjoint.passed is True  # type: ignore[union-attr]
    assert result.identity_disjoint.passed is True  # type: ignore[union-attr]
    status, contradictions = check_prompt_consistency(tuple(rows))
    assert status.passed is True and not contradictions  # type: ignore[union-attr]


# -- CLINC two-stage ---------------------------------------------------------

DOMAINS: dict[str, tuple[str, ...]] = {
    f"domain_{d}": tuple(f"intent_{d}_{k}" for k in range(15)) for d in range(10)
}


def test_the_clinc_shape_fits_both_stages() -> None:
    m = ClincDomainMap(domains=DOMAINS)
    assert m.domain_of("intent_3_7") == "domain_3"
    assert m.unmapped(["intent_3_7", "oos", "stray"]) == ("stray",)


@pytest.mark.parametrize(
    ("domains", "match"),
    [
        ({"a": ("x", "y")}, "2..16"),
        ({f"d{i}": ("x", "y") for i in range(17)}, "2..16"),
        ({"a": ("x", "y"), "b": ("y", "z")}, "both"),
        ({"a": ("x",), "b": ("y", "z")}, "needs 2..16"),
        ({"a": tuple(f"i{k}" for k in range(17)), "b": ("y", "z")}, "needs 2..16"),
        ({"a": ("x", "noul"), "b": ("y", "z")}, "reserved"),
        ({"a": ("x", "oos"), "b": ("y", "z")}, "reserved"),
        ({"a": ("x", " "), "b": ("y", "z")}, "empty"),
    ],
)
def test_a_domain_map_that_does_not_fit_the_slot_is_refused(
    domains: dict[str, tuple[str, ...]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ClincDomainMap(domains=domains)


def test_load_clinc_domains_reads_the_expected_shape_and_refuses_others(
    tmp_path: Path,
) -> None:
    good = tmp_path / "domains.json"
    good.write_text(json.dumps({k: list(v) for k, v in DOMAINS.items()}), encoding="utf-8")
    assert load_clinc_domains(good, expect_sha256=None).domains == DOMAINS
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(["banking", "travel"]), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_clinc_domains(bad, expect_sha256=None)
    bad.write_text(json.dumps({"banking": "balance"}), encoding="utf-8")
    with pytest.raises(ValueError, match="list of intent names"):
        load_clinc_domains(bad, expect_sha256=None)
    big = tmp_path / "big.json"
    big.write_text(" " * (MAX_DOMAIN_MAP_BYTES + 1), encoding="utf-8")
    with pytest.raises(ValueError, match="refused rather than read"):
        load_clinc_domains(big, expect_sha256=None)


def test_stage_one_asks_over_domains_and_oos_abstains() -> None:
    m = ClincDomainMap(domains=DOMAINS)
    ins = ClincRow(utterance="please move money", intent="intent_2_4", is_oos=False)
    row = rewrite_clinc_two_stage(ins, family_id=CLINC_DOMAIN_FAMILY, index=0,
                                  config=CONFIG, domain_map=m)
    assert row.request.slots[0].options == tuple(sorted(DOMAINS))
    assert row.gold[0].value == "domain_2"
    _noul_is_the_last_option_line(row.request)
    oos = ClincRow(utterance="sing me a sea shanty", intent="oos", is_oos=True)
    row = rewrite_clinc_two_stage(oos, family_id=CLINC_DOMAIN_FAMILY, index=1,
                                  config=CONFIG, domain_map=m)
    assert row.gold[0].is_noul is True and row.gold[0].value is None


def test_stage_two_asks_within_the_gold_domain_and_oos_abstains_inside_one() -> None:
    m = ClincDomainMap(domains=DOMAINS)
    ins = ClincRow(utterance="please move money", intent="intent_2_4", is_oos=False)
    row = rewrite_clinc_two_stage(ins, family_id=CLINC_WITHIN_DOMAIN_FAMILY, index=0,
                                  config=CONFIG, domain_map=m)
    assert set(row.request.slots[0].options) == set(DOMAINS["domain_2"])
    assert len(row.request.slots[0].options) == 15
    assert row.gold[0].value == "intent_2_4"
    _noul_is_the_last_option_line(row.request)

    oos = ClincRow(utterance="sing me a sea shanty", intent="oos", is_oos=True)
    a = rewrite_clinc_two_stage(oos, family_id=CLINC_WITHIN_DOMAIN_FAMILY, index=1,
                                config=CONFIG, domain_map=m)
    b = rewrite_clinc_two_stage(oos, family_id=CLINC_WITHIN_DOMAIN_FAMILY, index=9,
                                config=CONFIG, domain_map=m)
    assert a.gold[0].is_noul is True
    assert set(a.request.slots[0].options) in [set(v) for v in DOMAINS.values()]
    # The asked domain is keyed on the utterance, never on its read position.
    assert a.request.slots[0].options == b.request.slots[0].options


def test_two_stage_rows_share_the_existing_clinc_split_unit() -> None:
    """One utterance must land in one repo split whichever CLINC family asks about
    it. The keys are built as ``mixture.rewrite_clinc`` builds them."""
    from qd_data.mixture import rewrite_clinc

    vocab = [i for v in DOMAINS.values() for i in v]
    ins = ClincRow(utterance="please move money", intent="intent_2_4", is_oos=False)
    old = rewrite_clinc(ins, family_id="intent.in_scope", index=0, config=CONFIG,
                        intent_vocabulary=vocab)
    new = rewrite_clinc_two_stage(ins, family_id=CLINC_DOMAIN_FAMILY, index=0,
                                  config=CONFIG, domain_map=ClincDomainMap(domains=DOMAINS))
    assert (old.repo_key, old.identity_key) == (new.repo_key, new.identity_key)


def test_an_unplaced_intent_is_a_counted_refusal() -> None:
    ins = ClincRow(utterance="hello", intent="not_in_map", is_oos=False)
    with pytest.raises(RowRefused) as excinfo:
        rewrite_clinc_two_stage(ins, family_id=CLINC_DOMAIN_FAMILY, index=0,
                                config=CONFIG, domain_map=ClincDomainMap(domains=DOMAINS))
    assert excinfo.value.reason_code == "intent_not_in_domain_map"


# -- MMLU split policy (lead decision 2026-09-29) -----------------------------


def test_mmlu_split_policy_is_pinned_and_mmlu_is_not_a_reportable_benchmark() -> None:
    """test and dev are trained on, validation is the family's val. So an MMLU number
    measured on MMLU's own test split is a training-set number and cannot be reported."""
    mmlu = source_by_id("cais/mmlu")
    assert dict(mmlu.pinned_splits) == {"test": "train", "dev": "train", "validation": "val"}
    assert mmlu.benchmark_reportable is False
    assert "NOT A REPORTABLE BENCHMARK FOR LAPPI" in mmlu.evidence
    assert source_by_id("clinc/clinc_oos").pinned_split_of("train") is None


@pytest.mark.parametrize(("upstream", "pinned"), [("test", "train"), ("dev", "train"),
                                                  ("validation", "val")])
def test_an_mmlu_row_carries_its_pinned_split_into_its_split_unit(
    upstream: str, pinned: str
) -> None:
    row = rewrite_mmlu(parse_mmlu(mmlu_raw(2), index=0, split_name=upstream),
                       family_id=MMLU_FAMILY, index=0, config=CONFIG)
    assert row.metadata[PINNED_SPLIT_KEY] == pinned
    assert row.metadata["upstream_split"] == upstream
    assert row.repo_key == f"mmlu-{pinned}:subject_2"


def test_an_mmlu_row_from_an_unlisted_or_refused_split_cannot_exist() -> None:
    with pytest.raises(SourceUnavailableRefusal):
        parse_mmlu(mmlu_raw(2), index=0, split_name="auxiliary_train")
    with pytest.raises(MalformedRowRefusal, match="pinned split policy"):
        parse_mmlu(mmlu_raw(2), index=0, split_name="train")
    with pytest.raises(MalformedRowRefusal):
        MmluRow(subject="s", question="q", choices=("a", "b"), answer_index=0,
                upstream_split="auxiliary_train")


def test_the_same_question_in_train_and_val_splits_is_two_units() -> None:
    a = rewrite_mmlu(parse_mmlu(mmlu_raw(4), index=0, split_name="test"),
                     family_id=MMLU_FAMILY, index=0, config=CONFIG)
    b = rewrite_mmlu(parse_mmlu(mmlu_raw(4), index=0, split_name="validation"),
                     family_id=MMLU_FAMILY, index=0, config=CONFIG)
    assert a.row_id != b.row_id
    assert a.repo_key != b.repo_key


# -- domain map pin ------------------------------------------------------------


def test_the_domain_map_is_pinned_by_sha256(tmp_path: Path) -> None:
    good = tmp_path / "domains.json"
    good.write_text(json.dumps({k: list(v) for k, v in DOMAINS.items()}), encoding="utf-8")
    with pytest.raises(ValueError, match="is not the pinned domain map"):
        load_clinc_domains(good)
    assert len(CLINC_DOMAINS_SHA256) == 64


# -- replay slice --------------------------------------------------------------


def _train_rows(n: int) -> list:
    return [
        rewrite_mmlu(parse_mmlu(mmlu_raw(i), index=i, split_name="test"),
                     family_id=MMLU_FAMILY, index=i, config=CONFIG)
        for i in range(n)
    ] + [
        rewrite_csqa(parse_csqa(csqa_raw(i), index=i, split_name="train"), family_id=CSQA_FAMILY,
                     index=i, config=CONFIG)
        for i in range(n)
    ]


def test_replay_is_a_disjoint_deterministic_fifteen_percent() -> None:
    rows = _train_rows(1000)
    part = partition_replay(rows, seed=CONFIG.seed)
    gold_ids = {r.row_id for r in part.gold_rows}
    replay_ids = {r.row_id for r in part.replay_rows}
    assert not gold_ids & replay_ids
    assert gold_ids | replay_ids == {r.row_id for r in rows}
    assert all(r.metadata[REPLAY_ROLE_KEY] == REPLAY_ONLY for r in part.replay_rows)
    assert all(REPLAY_ROLE_KEY not in r.metadata for r in part.gold_rows)
    share = len(part.replay_rows) / len(rows)
    assert 0.13 < share < 0.17, share
    for fam, c in part.counts().items():
        assert 0.12 < c["replay"] / (c["gold"] + c["replay"]) < 0.18, fam
    again = partition_replay(list(reversed(rows)), seed=CONFIG.seed)
    assert {r.row_id for r in again.replay_rows} == replay_ids
    # Adding rows moves no existing row between roles.
    grown = partition_replay(_train_rows(1200), seed=CONFIG.seed)
    assert replay_ids <= {r.row_id for r in grown.replay_rows}


def test_a_row_can_never_be_both_gold_and_replay() -> None:
    rows = _train_rows(50)
    part = partition_replay(rows, seed=CONFIG.seed)
    assert part.replay_rows
    leaked = part.replay_rows[0]
    with pytest.raises(ValueError, match="both gold-trained and replay-only"):
        ReplayPartition(gold_rows=(*part.gold_rows, leaked), replay_rows=part.replay_rows,
                        fraction=0.15)
    with pytest.raises(ValueError, match="already carries a replay role"):
        partition_replay([leaked], seed=CONFIG.seed)


def test_replay_draws_only_from_general_training_rows() -> None:
    val_row = rewrite_mmlu(parse_mmlu(mmlu_raw(1), index=0, split_name="validation"),
                           family_id=MMLU_FAMILY, index=0, config=CONFIG)
    with pytest.raises(ValueError, match="training rows only"):
        partition_replay([val_row], seed=CONFIG.seed)
    from qd_data.mixture import rewrite_clinc

    clinc = rewrite_clinc(ClincRow(utterance="hi there", intent="a", is_oos=False),
                          family_id="intent.in_scope", index=0, config=CONFIG,
                          intent_vocabulary=["a", "b"])
    with pytest.raises(ValueError, match="not a replay family"):
        partition_replay([clinc], seed=CONFIG.seed)
    with pytest.raises(ValueError, match="fraction"):
        partition_replay([], seed=CONFIG.seed, fraction=0.0)


# -- the real files, where the approved download is on this host --------------

_CACHE = Path.home() / ".cache" / "qd-decision" / "general"
_CLINC = _CACHE / "clinc__clinc_oos" / "155b9c710419136e17307b80d0a13e68cd46b4ec"
_DOMAINS = (
    _CACHE / "clinc__oos-eval" / "828f8093932c8fe6ca7936c3d2e52903b1c523de" / "domains.json"
)


def test_the_real_domain_map_places_every_real_clinc_intent() -> None:
    """Reported SKIPPED, never passed, on a host without the 2026-09-29 download."""
    if not (_DOMAINS.exists() and (_CLINC / "intent_names.json").exists()):
        pytest.skip(f"not run: the approved download is not on this host ({_CACHE})")
    m = load_clinc_domains(_DOMAINS)
    names = json.loads((_CLINC / "intent_names.json").read_text(encoding="utf-8"))
    assert len(m.domains) == 10
    assert all(len(v) == 15 for v in m.domains.values())
    assert len(names) == 151 and names.index("oos") == 42
    assert m.unmapped(names) == ()
    assert {i for v in m.domains.values() for i in v} == set(names) - {"oos"}
    first = json.loads((_CLINC / "train.jsonl").read_text(encoding="utf-8").split("\n", 1)[0])
    assert isinstance(first["intent"], int)
    row = parse_clinc(first, index=0, label_names=names)
    assert m.domain_of(row.intent) is not None


def test_contaminated_replay_rows_are_rebuilt_out_to_neither_side() -> None:
    """Refuse-on-hit, then rebuild: a decontamination hit leaves the replay slice and
    does not fall back into gold training, and nothing else moves."""
    rows = _train_rows(300)
    before = partition_replay(rows, seed=CONFIG.seed)
    hit = before.replay_rows[0].identity_key
    after = partition_replay(rows, seed=CONFIG.seed,
                             contaminated_identity_keys=frozenset({hit}))
    assert [r.identity_key for r in after.excluded_rows] == [hit]
    assert hit not in {r.identity_key for r in after.replay_rows}
    assert hit not in {r.identity_key for r in after.gold_rows}
    assert {r.row_id for r in after.gold_rows} == {r.row_id for r in before.gold_rows}
    assert len(after.replay_rows) == len(before.replay_rows) - 1
    assert sum(c["excluded"] for c in after.counts().values()) == 1
    with pytest.raises(ValueError, match="name no training row here"):
        partition_replay(rows, seed=CONFIG.seed,
                         contaminated_identity_keys=frozenset({"not-a-row::x"}))


# -- lead decisions under the user's 2026-09-29 delegation ---------------------


def test_the_two_stage_clinc_families_are_registered_once() -> None:
    """Registered in ``qd_data.sources`` and read from there by ``qd_data.general``,
    so the registry is the one owner of their definition."""
    for fam in CLINC_TWO_STAGE_FAMILIES:
        assert TASK_FAMILIES[fam.family_id] is fam
        assert fam.source_id == "clinc/clinc_oos"
    admitted = {f.family_id for f in admitted_task_families()}
    assert {CLINC_DOMAIN_FAMILY, CLINC_WITHIN_DOMAIN_FAMILY} <= admitted


def test_csqa_split_policy_is_pinned_and_csqa_is_an_internal_val_only() -> None:
    csqa = source_by_id("tau/commonsense_qa")
    assert dict(csqa.pinned_splits) == {"train": "train", "validation": "val"}
    assert csqa.benchmark_reportable is False
    assert "NOT A REPORTABLE BENCHMARK FOR LAPPI" in csqa.evidence
    row = rewrite_csqa(parse_csqa(csqa_raw(3), index=0, split_name="validation"),
                       family_id=CSQA_FAMILY, index=0, config=CONFIG)
    assert row.metadata[PINNED_SPLIT_KEY] == "val"
    assert row.repo_key == "csqa-val:concept_3"


def test_the_unlabelled_csqa_test_split_is_not_read() -> None:
    with pytest.raises(SourceUnavailableRefusal, match="no answerKey"):
        parse_csqa(csqa_raw(1), index=0, split_name="test")
    with pytest.raises(MalformedRowRefusal, match="pinned split policy"):
        CsqaRow(qid="q", question="Q?", concept="c", labels=("A", "B"),
                texts=("x", "y"), answer_key="A", upstream_split="test")


def test_one_exclusion_rule_covers_gold_and_replay_rows() -> None:
    """GAP-DATA-GENERAL-CE-ROWS-OVERLAP-FAMILY-VAL, closed: a contaminated gold row is
    excluded exactly as a contaminated replay row is -- never moved to the other role."""
    rows = _train_rows(300)
    before = partition_replay(rows, seed=CONFIG.seed)
    gold_hit = before.gold_rows[0].identity_key
    replay_hit = before.replay_rows[0].identity_key
    after = partition_replay(rows, seed=CONFIG.seed,
                             contaminated_identity_keys=frozenset({gold_hit, replay_hit}))
    assert {r.identity_key for r in after.excluded_rows} == {gold_hit, replay_hit}
    assert gold_hit not in {r.identity_key for r in after.replay_rows}
    assert replay_hit not in {r.identity_key for r in after.gold_rows}
    assert len(after.gold_rows) == len(before.gold_rows) - 1
    assert len(after.replay_rows) == len(before.replay_rows) - 1
