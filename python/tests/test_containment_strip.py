"""v5's template strip (Fable's post-F ruling 3(a); ``qd_train.containment_strip``).

Within each (family, slot), a question line or option value byte-identical across every
rendered row of that family-slot, in the union of sources and targets, is removed before
n-gramming, for containment only. The tests drive real ``qd_data`` rows through the real
renderer, the exporter (``tools/containment_scan.scan_sets``) and ``qd-prep containment``, and
the parity oracle (``qd_train.replay.decontaminate``) through the same strip function.

Every row's question line is its family's ``description`` (``qd_data.mixture._request``), so
the strip removes it in every family; the per-row question of MMLU, CSQA and SQuAD is in the
context block, which the strip never touches -- the "kept whole" tests assert on that.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from containment_scan import (  # noqa: E402
    ScanSet,
    export_block,
    run_containment,
    scan_sets,
    scan_specs,
    splitter_checks,
    write_request,
)
from data_fixtures import INTENT_VOCABULARY, squad_row  # noqa: E402
from replay_decontam import row_texts  # noqa: E402
from test_qd_prep_containment_parity import (  # noqa: E402
    FIXTURE,
    PAIRS_NAME,
    _oracle,
    _scan,
)

from qd_data.config import DataConfig  # noqa: E402
from qd_data.defect_class import DEFECT_FAMILY_ID, DefectRow  # noqa: E402
from qd_data.general import (  # noqa: E402
    CLINC_DOMAIN_FAMILY,
    CLINC_WITHIN_DOMAIN_FAMILY,
    CSQA_FAMILY,
    MMLU_FAMILY,
    ClincDomainMap,
    rewrite_clinc_two_stage,
    rewrite_csqa,
    rewrite_mmlu,
)
from qd_data.loaders import ClincRow, CsqaRow, MmluRow  # noqa: E402
from qd_data.mixture import rewrite_clinc, rewrite_defect_class, rewrite_squad  # noqa: E402
from qd_data.render import render  # noqa: E402
from qd_data.rows import DataRow  # noqa: E402
from qd_data.split import HELD_OUT, SplitReport  # noqa: E402
from qd_train.containment_strip import (  # noqa: E402
    MIN_ROWS_FOR_CONSTANT,
    STRIP_VERSION,
    PartsRow,
    SlotParts,
    slot_parts,
    strip_template,
    template_constants,
    too_short,
)
from qd_train.exclusions import (  # noqa: E402
    ATTESTATION_NAME,
    EXCLUSIONS_NAME,
    ExclusionRefusal,
    read_exclusions,
)
from qd_train.replay import decontaminate, word_ngrams  # noqa: E402
from qd_train.tristate import Ran  # noqa: E402

CONFIG = DataConfig()
CORPUS = {"fixture": "template-strip"}
DOMAINS = ClincDomainMap(domains={
    "banking": ("account_balance", "pay_bill", "transfer"),
    "travel": ("book_flight", "car_rental", "exchange_rate"),
    "work": ("meeting_schedule", "pto_request"),
})
IN_SCOPE = "intent.in_scope"

# The GAP-CONTAINMENT-CONSTANT-QUESTION-DRIVES-CLINC-HITS shape at >= 8 words a side, so the
# assertion is about containment and not about a row too short to compare: the two share
# their first four words and nothing else.
VAL_UTTERANCE = "what is the minimum balance for my checking"
TRAIN_SHARES_PREFIX = "what is the minimum payment on my eddie bauer card"
#: A train utterance that really contains the val one: the strip must keep that hit.
TRAIN_CONTAINS_VAL = f"please tell me {VAL_UTTERANCE} account today"
SHORT_VAL_UTTERANCE = "what is spanish for hello"


def _clinc(utterance: str, family: str, index: int, *, intent: str = "transfer") -> DataRow:
    raw = ClincRow(utterance=utterance, intent=intent, is_oos=False)
    if family in (CLINC_DOMAIN_FAMILY, CLINC_WITHIN_DOMAIN_FAMILY):
        return rewrite_clinc_two_stage(raw, family_id=family, index=index, config=CONFIG,
                                       domain_map=DOMAINS)
    return rewrite_clinc(raw, family_id=family, index=index, config=CONFIG,
                         intent_vocabulary=(*INTENT_VOCABULARY, intent))


def _mmlu(question: str, choices: tuple[str, ...], index: int) -> DataRow:
    raw = MmluRow(subject="astronomy", question=question, choices=choices, answer_index=1,
                  upstream_split="test")
    return rewrite_mmlu(raw, family_id=MMLU_FAMILY, index=index, config=CONFIG)


def _csqa(question: str, texts: tuple[str, ...], index: int) -> DataRow:
    raw = CsqaRow(qid=f"cs{index}", question=question, concept=f"concept{index}",
                  labels=("A", "B", "C", "D", "E"), texts=texts, answer_key="A",
                  upstream_split="train")
    return rewrite_csqa(raw, family_id=CSQA_FAMILY, index=index, config=CONFIG)


def _defect(n: int, cls: str) -> DataRow:
    diff = (f"@@ -1,3 +1,3 @@\n def handler_{n}(request, session):\n"
            f"-    return session.lookup(request.key_{n})\n+    return None\n")
    raw = DefectRow(
        example_id=f"c{n}:x.py#0", pool_id=f"c{n}:x.py", repo=f"o/r{n}", path="x.py",
        symbol=f"handler_{n}", arity=2, language="python", mutation_class=cls,
        operator=f"{cls}.op", diff=diff, diff_span=None if cls == "clean" else (3, 3),
        span_refusal=None, licence="mit",
    )
    return rewrite_defect_class(raw, family_id=DEFECT_FAMILY_ID, index=n, config=CONFIG)


def _squad(i: int) -> DataRow:
    return rewrite_squad(squad_row(i, family="qa.answer_span", force_impossible=False),
                         family_id="qa.answer_span", index=i, config=CONFIG)


def _parts(rows: list[DataRow]) -> list[PartsRow]:
    """The oracle side's own render loop: ``render`` at ``seed=None`` and ``slot_parts``."""
    out: list[PartsRow] = []
    for row in rows:
        rendered = render(row.request, seed=None)
        for slot in rendered.slots:
            out.append((f"{row.row_id}#{slot.name}", row.identity_key, row.family_id,
                        slot.name, slot_parts(rendered.prompt_for(slot.name))))
    return out


def _report(train: list[DataRow], val: list[DataRow],
            heldout: list[DataRow] = ()) -> SplitReport:
    ok = Ran(passed=True, detail="test fixture")
    return SplitReport(
        assignments=(), rows_by_split={"train": tuple(train), "val": tuple(val),
                                       HELD_OUT: tuple(heldout)},
        repo_disjoint=ok, identity_disjoint=ok, near_duplicate_disjoint=ok,
        held_out_families_absent_from_training=ok, dedupe_status=ok,
        content_disjoint_families=ok,
    )


def _texts(report: SplitReport, *, strip: bool) -> dict[str, dict[str, str]]:
    sets, _record = scan_sets(report, config=CONFIG, template_strip=strip)
    return {s.name: {key: text for key, _i, _f, text in s.rows} for s in sets}


def _pairs(texts: dict[str, dict[str, str]]) -> set[tuple[str, str]]:
    report = decontaminate(texts["train"], {"val": texts["val"]})
    return {(p.replay_row, p.target_row) for p in report.pairs}


def _clinc_report() -> tuple[SplitReport, dict[str, DataRow]]:
    rows = {
        "val": _clinc(VAL_UTTERANCE, IN_SCOPE, 0),
        "prefix": _clinc(TRAIN_SHARES_PREFIX, IN_SCOPE, 1),
        "contains": _clinc(TRAIN_CONTAINS_VAL, IN_SCOPE, 2),
        "short_val": _clinc(SHORT_VAL_UTTERANCE, IN_SCOPE, 3),
    }
    return _report([rows["prefix"], rows["contains"]], [rows["val"], rows["short_val"]]), rows


def _key(row: DataRow) -> str:
    return f"{row.row_id}#{row.request.slots[0].name}"


# --- CLINC: the constant question no longer pairs; real overlap still does --------------------


def test_clinc_rows_that_share_only_the_constant_question_no_longer_pair() -> None:
    report, rows = _clinc_report()
    before = _texts(report, strip=False)
    after = _texts(report, strip=True)
    prefix, contains, val = _key(rows["prefix"]), _key(rows["contains"]), _key(rows["val"])
    # Characterization of the pre-strip definition: the shared 11-word question carries
    # the pair over the threshold.
    assert (prefix, val) in _pairs(before)
    # The strip: question and the constant yes/no gone, the utterance whole, >= 8 words.
    assert after["val"][val] == VAL_UTTERANCE and not too_short(after["val"][val])
    assert after["train"][prefix] == TRAIN_SHARES_PREFIX
    assert (prefix, val) not in _pairs(after)
    # Not vacuous: an utterance that really contains the val one hits it after the strip.
    assert (contains, val) in _pairs(after)


def test_a_short_clinc_val_row_is_counted_too_short_after_the_strip_not_clean() -> None:
    report, rows = _clinc_report()
    _sets, record = scan_sets(report, config=CONFIG)
    group = record["family_slots"][f"{IN_SCOPE}/in_scope"]
    assert group["by_set"]["val"] == {
        "rows": 2, "too_short_before_strip": 0, "too_short_after_strip": 1,
    }
    assert group["too_short_after_strip"] == 1
    assert group["stripped"]["question"] == rows["val"].request.question
    assert group["stripped"]["options"] == ["no", "yes"]


def test_intent_domain_strips_its_fixed_list_and_within_domain_keeps_per_domain_lists() -> None:
    """Fable's caveat, measured on the real builders: a family-wide rule strips
    intent.domain's fixed domain list; intent.within_domain asks over ONE domain's intents
    (``qd_data.general.rewrite_clinc_two_stage``), so two domains share no option and nothing
    is stripped from either list. (``intent.classification`` samples 16 sorted random
    intents per row instead.)"""
    utterances = [
        ("how much money do i have in my checking account now", "account_balance"),
        ("please send two hundred dollars to my savings account", "transfer"),
        ("i need a flight to boston leaving on friday morning", "book_flight"),
        ("what is the euro to dollar exchange rate this week", "exchange_rate"),
    ]
    train = [_clinc(u, f, i, intent=intent)
             for i, (u, intent) in enumerate(utterances[:2])
             for f in (CLINC_DOMAIN_FAMILY, CLINC_WITHIN_DOMAIN_FAMILY)]
    val = [_clinc(u, f, 10 + i, intent=intent)
           for i, (u, intent) in enumerate(utterances[2:])
           for f in (CLINC_DOMAIN_FAMILY, CLINC_WITHIN_DOMAIN_FAMILY)]
    sets, record = scan_sets(_report(train, val), config=CONFIG)
    domain = record["family_slots"][f"{CLINC_DOMAIN_FAMILY}/domain"]
    within = record["family_slots"][f"{CLINC_WITHIN_DOMAIN_FAMILY}/intent"]
    assert domain["stripped"]["options"] == sorted(DOMAINS.domains)
    assert domain["option_sets"] == {"distinct": 1, "largest_share_rows": 4}
    assert within["stripped"]["options"] == []
    assert within["option_sets"] == {"distinct": 2, "largest_share_rows": 2}
    texts = {key: text for s in sets for key, _i, fam, text in s.rows
             if fam == CLINC_WITHIN_DOMAIN_FAMILY}
    assert all("account_balance" in t or "book_flight" in t for t in texts.values())


# --- families whose per-row question is the content: kept whole --------------------------------


def test_mmlu_csqa_squad_and_defect_class_rows_keep_their_question_and_options_whole() -> None:
    mmlu = [_mmlu("Which planet is the largest in the solar system by mass?",
                  ("Mars", "Jupiter", "Venus", "Earth"), 0),
            _mmlu("Which element has the chemical symbol Fe on the periodic table?",
                  ("Lead", "Iron", "Tin", "Gold"), 1)]
    csqa = [_csqa("Where would you most likely keep a spare tire for a car?",
                  ("trunk", "kitchen", "garden", "office", "attic"), 0),
            _csqa("What do people usually do when they feel very tired at night?",
                  ("sleep", "run", "sing", "cook", "paint"), 1)]
    squad = [_squad(7), _squad(8)]
    defect = [_defect(1, "stub"), _defect(2, "logic")]
    rows = mmlu + csqa + squad + defect
    texts, record = strip_template({"all": _parts(rows)})
    by_key = {key: text for key, _i, _f, text in texts["all"]}
    for row in rows:
        rendered = render(row.request, seed=None)
        for slot in rendered.slots:
            parts = slot_parts(rendered.prompt_for(slot.name))
            got = by_key[f"{row.row_id}#{slot.name}"]
            # The per-row question (the context block) survives byte for byte, and the
            # family's constant question line does not.
            assert parts.context is not None and parts.context in got
            assert row.request.question not in got
            if row.family_id in (MMLU_FAMILY, CSQA_FAMILY):
                assert got == "\n".join((parts.context, *parts.options))
    assert record["family_slots"][f"{DEFECT_FAMILY_ID}/defect_class"]["stripped"]["options"] == [
        "clean", "cosmetic", "logic", "stub",
    ]
    for family in (MMLU_FAMILY, CSQA_FAMILY):
        assert record["family_slots"][f"{family}/answer"]["stripped"]["options"] == []


def test_a_val_mmlu_question_copied_into_train_still_pairs_after_the_strip() -> None:
    question = ("Which of the following best describes the main function of the "
                "mitochondria in eukaryotic cells during aerobic respiration?")
    val = _mmlu(question, ("energy", "storage", "transport", "division"), 0)
    copied = _mmlu(question, ("protein", "lipids", "water", "light"), 1)
    other = _mmlu("Which gas do plants absorb from the air for photosynthesis in daylight?",
                  ("oxygen", "carbon dioxide", "nitrogen", "argon"), 2)
    after = _texts(_report([copied, other], [val]), strip=True)
    assert (_key(copied), _key(val)) in _pairs(after)
    assert (_key(other), _key(val)) not in _pairs(after)


# --- the rule's edges ------------------------------------------------------------------------


def _row(key: str, family: str, parts: SlotParts) -> PartsRow:
    return (key, f"id-{key}", family, "s", parts)


def test_constancy_is_membership_over_every_set_and_never_touches_the_context() -> None:
    a = SlotParts(question="Q same", context="alpha beta yes", options=("yes", "no", "x1"))
    b = SlotParts(question="Q same", context="gamma", options=("no", "yes", "x2", "x2"))
    c = SlotParts(question="Q other", context="delta", options=("yes", "no"))
    texts, record = strip_template({"train": [_row("a", "f", a), _row("b", "f", b)],
                                    "val": [_row("c", "f", c)]})
    got = {k: t for rows in texts.values() for k, _i, _f, t in rows}
    # yes/no in every row of the union, in any order: stripped. The question differs in
    # one row of another set: kept. "yes" inside a context is context, not an option.
    assert got == {"a": "Q same\nalpha beta yes\nx1", "b": "Q same\ngamma\nx2\nx2",
                   "c": "Q other\ndelta"}
    group = record["family_slots"]["f/s"]
    assert group["stripped"] == {"question": None, "options": ["no", "yes"]}
    assert group["n_stripped"] == 2
    raw = json.dumps({"question": None, "options": ["no", "yes"]}, sort_keys=True,
                     ensure_ascii=False).encode()
    assert group["stripped_sha256"] == hashlib.sha256(raw).hexdigest()


def test_a_question_missing_from_one_row_is_not_constant() -> None:
    rows = [_row("a", "f", SlotParts("Q", "one", ())), _row("b", "f", SlotParts(None, "two", ()))]
    assert template_constants(rows)[("f", "s")].question is None


def test_a_group_of_one_row_strips_nothing_and_is_named() -> None:
    lone = _row("a", "solo", SlotParts("Which?", "the only row", ("p", "q")))
    pair = [_row("b", "duo", SlotParts("Q", "one", ("p",))),
            _row("c", "duo", SlotParts("Q", "two", ("p",)))]
    texts, record = strip_template({"train": [lone, *pair]})
    got = {k: t for k, _i, _f, t in texts["train"]}
    assert MIN_ROWS_FOR_CONSTANT == 2
    assert got["a"] == "Which?\nthe only row\np\nq"
    assert got["b"] == "one" and got["c"] == "two"
    assert record["groups_below_min_rows"] == ["solo/s"]


def test_apply_false_is_byte_for_byte_the_pre_strip_export() -> None:
    """Characterization: the unstripped mode is what the exporter sent before this change,
    ``replay_decontam.row_texts`` (render at seed=None, then prompt_content), key for key."""
    report, _rows = _clinc_report()
    rows = [*report.rows_by_split["train"], *report.rows_by_split["val"],
            _mmlu("Which planet is the largest in the solar system by mass?",
                  ("Mars", "Jupiter", "Venus", "Earth"), 5), _defect(3, "stub"), _squad(9)]
    want, refused = row_texts(rows)
    assert not refused
    texts, record = strip_template({"all": _parts(rows)}, apply=False)
    assert {k: t for k, _i, _f, t in texts["all"]} == want
    assert record["applied"] is False
    assert all(g["n_stripped"] == 0 for g in record["family_slots"].values())


def test_too_short_is_exactly_an_empty_word_ngram_set() -> None:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()[1:]
    texts = [json.loads(line)["text"] for line in lines]
    texts += ["", "one two three four five six seven", "one two three four five six seven eight",
              "İstanbul ΣΟΦΟΣ ① ② ③ ④ ⑤ ⑥ ⑦", "a_b c-d e.f g h i j k l"]
    for text in texts:
        assert too_short(text) == (not word_ngrams(text)), text


def test_a_strip_that_is_not_a_containment_test_is_refused() -> None:
    with pytest.raises(ValueError, match="not an n-gram length"):
        strip_template({}, n=0)
    texts, record = strip_template({})
    assert texts == {} and record["family_slots"] == {} and record["applied"] is True


# --- through the exporter and qd-prep: parity, attestation, and the hook ----------------------


def _mixed_report() -> SplitReport:
    report, _rows = _clinc_report()
    mmlu_q = ("Which of the following best describes the main function of the mitochondria "
              "in eukaryotic cells during aerobic respiration?")
    train = [*report.rows_by_split["train"],
             _mmlu(mmlu_q, ("protein", "lipids", "water", "light"), 1),
             _mmlu("Which gas do plants absorb from the air for photosynthesis in daylight?",
                   ("oxygen", "carbon dioxide", "nitrogen", "argon"), 2),
             _defect(1, "stub"), _defect(2, "logic"), _squad(11), _squad(12)]
    val = [*report.rows_by_split["val"], _mmlu(mmlu_q, ("energy", "storage", "x", "y"), 0),
           _defect(3, "cosmetic"), _squad(13)]
    heldout = [_clinc(TRAIN_SHARES_PREFIX + " again", IN_SCOPE, 20), _defect(4, "clean")]
    return _report(train, val, heldout)


def test_the_stripped_pair_list_is_the_oracles_bit_for_bit(qd_prep_bin: Path,
                                                            tmp_path: Path) -> None:
    """Exporter side: ``scan_sets`` (render, ``slot_parts``, ``strip_template``) -> request ->
    ``qd-prep containment``. Oracle side: its own render loop -> the SAME ``strip_template``
    -> ``decontaminate``. The complete pair lists must agree byte for byte."""
    report = _mixed_report()
    sets, _record = scan_sets(report, config=CONFIG)
    scans = scan_specs(sets)
    got = (_scan(qd_prep_bin, tmp_path, sets, scans) / PAIRS_NAME).read_text(encoding="utf-8")
    texts, _ = strip_template({
        "train": _parts(list(report.rows_by_split["train"])),
        "val": _parts(list(report.rows_by_split["val"])),
        HELD_OUT: _parts(list(report.rows_by_split[HELD_OUT])),
    })
    oracle_sets = [ScanSet(name=s.name, rows=tuple(texts.get(s.name, ())), unrenderable={})
                   for s in sets]
    want, _counted = _oracle(oracle_sets, scans)
    assert got == want
    body = [line.split("\t") for line in want.splitlines()[1:]]
    assert any(b[3] == MMLU_FAMILY for b in body), "the copied MMLU question must pair"
    assert any(b[3] == IN_SCOPE for b in body), "the real CLINC containment must pair"
    # And the strip changed the answer: the template-only CLINC pair is in the unstripped
    # scan and in neither stripped list.
    plain, _ = scan_sets(report, config=CONFIG, template_strip=False)
    unstripped = (_scan(qd_prep_bin, tmp_path, plain, scan_specs(plain), name="plain")
                  / PAIRS_NAME).read_text(encoding="utf-8")
    template_pair = {line.split("\t")[1] for line in unstripped.splitlines()[1:]} - {
        line.split("\t")[1] for line in want.splitlines()[1:]
    }
    assert template_pair, "the unstripped scan found no pair the strip removed"


def _full_scan(binary: Path, report: SplitReport, out: Path, *, strip: bool) -> dict:
    sets, record = scan_sets(report, config=CONFIG, template_strip=strip)
    request = out.with_name(out.name + ".request.bin")
    with request.open("xb") as fh:
        write_request(fh, sets, scan_specs(sets), corpus=CORPUS,
                      export=export_block(sets, engine=binary, strip=record),
                      checks=splitter_checks(report))
    run_containment(binary, request, out, threads=2, timeout_s=120.0)
    return json.loads((out / ATTESTATION_NAME).read_text(encoding="utf-8"))


def test_attestation_v2_records_the_strip_per_family_slot(qd_prep_bin: Path,
                                                          tmp_path: Path) -> None:
    att = _full_scan(qd_prep_bin, _mixed_report(), tmp_path / "out", strip=True)
    strip = att["export"]["template_strip"]
    assert strip["applied"] is True and strip["version"] == STRIP_VERSION and strip["n"] == 8
    groups = strip["family_slots"]
    in_scope = groups[f"{IN_SCOPE}/in_scope"]
    assert in_scope["by_set"]["val"]["too_short_after_strip"] == 1
    assert in_scope["by_set"][HELD_OUT]["rows"] == 1
    for name, group in groups.items():
        raw = json.dumps(group["stripped"], sort_keys=True, ensure_ascii=False).encode()
        assert group["stripped_sha256"] == hashlib.sha256(raw).hexdigest(), name
        assert group["n_stripped"] == (len(group["stripped"]["options"])
                                       + (group["stripped"]["question"] is not None)), name
        assert group["rows"] == sum(s["rows"] for s in group["by_set"].values()), name
    # The attestation's own set counts are of the stripped texts: a row the strip left too
    # short is too short in qd-prep's count as well.
    val_short = sum(g["by_set"].get("val", {}).get("too_short_after_strip", 0)
                    for g in groups.values())
    assert att["sets"]["val"]["too_short"] == val_short
    assert att["clean"] is True, att["not_clean_because"]


def test_a_list_made_without_the_strip_is_refused_by_the_hook(qd_prep_bin: Path,
                                                               tmp_path: Path) -> None:
    report = _mixed_report()
    plain = _full_scan(qd_prep_bin, report, tmp_path / "plain", strip=False)
    assert plain["export"]["template_strip"]["applied"] is False
    with pytest.raises(ExclusionRefusal, match="did not apply v5's template strip"):
        read_exclusions(tmp_path / "plain" / EXCLUSIONS_NAME, corpus=CORPUS)
    _full_scan(qd_prep_bin, report, tmp_path / "stripped", strip=True)
    listed = read_exclusions(tmp_path / "stripped" / EXCLUSIONS_NAME, corpus=CORPUS)
    assert listed.keys and listed.keys <= {r.identity_key for r in report.rows_by_split["train"]}


def test_the_strip_moves_no_row_between_sets() -> None:
    """Rule 2: only texts change. Every set holds the same keys, identity keys and families
    with the strip as without it."""
    report = _mixed_report()
    with_strip, _ = scan_sets(report, config=CONFIG)
    without, _ = scan_sets(report, config=CONFIG, template_strip=False)
    assert [s.name for s in with_strip] == [s.name for s in without]
    for a, b in zip(with_strip, without, strict=True):
        assert [r[:3] for r in a.rows] == [r[:3] for r in b.rows], a.name
        assert a.unrenderable == b.unrenderable, a.name
    assert any(a.rows != b.rows for a, b in zip(with_strip, without, strict=True))
