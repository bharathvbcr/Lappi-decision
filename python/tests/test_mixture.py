"""The open-task-mixture rewriter: one format, per-row licensing, counted refusals.

The five claims under test:

* every family renders through the **one** ``Request`` shape, so nothing here can
  drift from the serving format;
* a dataset whose licence is not on the permissive allowlist is refused **at load**
  with a message naming the licence, and a *row* whose licence is not on it is
  refused even when its dataset is ``mit``;
* the free ``noul`` supervision is actually produced -- CLINC's out-of-scope class
  and SQuAD's unanswerable questions -- and is a gold *value*, never an error;
* two rows that render one prompt and demand two golds are refused, and the
  legitimate unanswerable-beside-answerable pair is not;
* every decode channel reports how much abstention supply it actually carries.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from data_fixtures import (
    INTENT_VOCABULARY,
    clinc_row,
    code_body,
    commitpackft_row,
    small_corpus,
    squad_row,
    squad_title_for,
    vendored_pair,
)

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.errors import LicenceRefused
from qd_data.licences import LicenceConfig
from qd_data.loaders import ClincRow, CommitPackFtRow, SourceUnavailableRefusal, SquadRow
from qd_data.manifest import Manifest, build_manifests
from qd_data.mixture import (
    ABSTAINING_FAMILIES,
    CHANGE_SCOPE_BIN_EDGES,
    CONTRADICTORY_PROMPT,
    LANGUAGE_OPTIONS,
    MAX_NAMED_GOLDS,
    MAX_NAMED_ROW_IDS,
    RowRefused,
    build_mixture,
    check_prompt_consistency,
    rewrite_clinc,
    rewrite_commitpackft,
    rewrite_squad,
)
from qd_data.render import DEFAULT_CAPS, render, render_for_serving
from qd_data.rows import DataRow
from qd_data.schema import MAX_CHOICE_OPTIONS, ChoiceSlot, ScoreSlot, SpanSlot
from qd_data.split import split
from qd_train.data_access import open_training_data
from qd_train.tristate import NotRun, Ran


def _by_family(rows: tuple[DataRow, ...], family: str) -> list[DataRow]:
    return [r for r in rows if r.family_id == family]


# -- one format --------------------------------------------------------------


def test_every_rewritten_row_renders_and_round_trips_through_the_wire() -> None:
    """One prompt format for every family, and the serving path reproduces it.

    This is the mixture-level form of the byte-identity claim: it is not enough that
    one hand-built request renders the same both ways; every row the pipeline
    actually produces must.
    """
    mixture = build_mixture(small_corpus(12), config=DataConfig())
    assert mixture.rows
    for row in mixture.rows:
        training = render(row.request, seed=None)
        serving = render_for_serving(row.request.to_wire())
        assert training.prompts() == serving.prompts()


def test_every_family_uses_exactly_one_slot_of_its_declared_kind() -> None:
    # 24 rows: `intent.classification` needs 15 distractor intents in the pull, so a
    # smaller corpus legitimately produces none of that family.
    mixture = build_mixture(small_corpus(24), config=DataConfig())
    expected = {
        "code.commit_intent": ChoiceSlot,
        "code.language_id": ChoiceSlot,
        "code.change_scope": ScoreSlot,
        "intent.classification": ChoiceSlot,
        "intent.in_scope": ChoiceSlot,
        "qa.answerability": ChoiceSlot,
        "qa.answer_span": SpanSlot,
    }
    seen: set[str] = set()
    for row in mixture.rows:
        assert len(row.request.slots) == 1
        assert isinstance(row.request.slots[0], expected[row.family_id])
        seen.add(row.family_id)
    assert seen == set(expected), f"missing families: {sorted(set(expected) - seen)}"


def test_no_choice_slot_exceeds_the_letter_slice() -> None:
    mixture = build_mixture(small_corpus(20), config=DataConfig())
    for row in mixture.rows:
        slot = row.request.slots[0]
        if isinstance(slot, ChoiceSlot):
            assert 1 <= len(slot.options) <= MAX_CHOICE_OPTIONS


# -- licences ----------------------------------------------------------------


def test_a_non_commercial_dataset_is_refused_at_load_naming_its_licence() -> None:
    """``facebook/anli`` is ``cc-by-nc-4.0``: disqualifying, not an edge case."""
    with pytest.raises(LicenceRefused) as excinfo:
        build_mixture({"facebook/anli": []}, config=DataConfig())
    assert excinfo.value.actual == "cc-by-nc-4.0"
    assert "cc-by-nc-4.0" in str(excinfo.value)
    assert excinfo.value.check == "licence_allowlist"


@pytest.mark.parametrize(
    ("source_id", "licence"),
    [
        ("mteb/stsbenchmark-sts", "unknown"),
        ("nyu-mll/glue", "other"),
        ("google/code_x_glue_cc_defect_detection", "c-uda"),
    ],
)
def test_an_unresolved_licence_is_refused_at_load_naming_it(
    source_id: str, licence: str
) -> None:
    with pytest.raises(LicenceRefused) as excinfo:
        build_mixture({source_id: []}, config=DataConfig())
    assert excinfo.value.actual == licence
    assert licence in str(excinfo.value)


@pytest.mark.parametrize(
    "source_id", ["PolyAI/banking77", "AmazonScience/massive", "nuprl/AgentPack"]
)
def test_an_unreachable_source_is_refused_as_unreachable_not_as_unlicensed(
    source_id: str,
) -> None:
    """Gated and script-only need a human action; a bad licence is closed. The two
    must not report the same way."""
    with pytest.raises(SourceUnavailableRefusal) as excinfo:
        build_mixture({source_id: []}, config=DataConfig())
    assert excinfo.value.check == "source_reachable"


def test_an_agpl_row_inside_an_mit_dataset_is_refused_and_counted() -> None:
    """``bigcode/commitpackft`` is ``mit``; three of its 13 per-row values are not
    permissive. A dataset-level check alone puts AGPL code in the pool."""
    config = DataConfig()
    rows = [
        commitpackft_row(0, licence="mit"),
        commitpackft_row(1, licence="agpl-3.0"),
        commitpackft_row(2, licence="unknown"),
        commitpackft_row(3, licence="apache-2.0"),
    ]
    mixture = build_mixture({"bigcode/commitpackft": rows}, config=config)
    assert set(mixture.licence_histogram) == {"mit", "apache-2.0"}
    refused = mixture.refusals["bigcode/commitpackft"]
    assert refused["licence:agpl-3.0"] == 3, "one refusal per family"
    assert refused["licence:unknown"] == 3


def test_a_human_admitted_licence_lets_its_rows_through() -> None:
    """The override exists and is one config change -- with a recorded justification."""
    config = DataConfig(
        licence=LicenceConfig(
            admitted_by_human={"lgpl-2.1": "counsel reviewed 2026-09-19; weak copyleft ok"}
        )
    )
    # Two rows, because `code.commit_intent` needs another row's message as a decoy.
    mixture = build_mixture(
        {
            "bigcode/commitpackft": [
                commitpackft_row(1, licence="lgpl-2.1"),
                commitpackft_row(2, licence="lgpl-2.1"),
            ]
        },
        config=config,
    )
    assert mixture.licence_histogram == {"lgpl-2.1": 6}


def test_a_non_commercial_licence_cannot_be_admitted_by_configuration() -> None:
    with pytest.raises(LicenceRefused):
        LicenceConfig(admitted_by_human={"cc-by-nc-4.0": "we really want it"})


def test_per_row_licence_and_host_reach_the_model_card_fields() -> None:
    mixture = build_mixture(small_corpus(6), config=DataConfig())
    assert mixture.host_histogram == {"huggingface": len(mixture.rows)}
    assert set(mixture.licence_histogram) == {"mit", "cc-by-3.0", "cc-by-sa-4.0"}
    # ShareAlike must survive to the manifest: a model card that omits it is wrong.
    assert "share-alike" in mixture.obligations["cc-by-sa-4.0"]
    assert "attribution" in mixture.obligations["mit"]


# -- the free noul -----------------------------------------------------------


def test_clinc_out_of_scope_becomes_an_abstention_not_a_class() -> None:
    config = DataConfig()
    row = rewrite_clinc(
        clinc_row(0, force_oos=True), family_id="intent.classification", index=0,
        config=config, intent_vocabulary=INTENT_VOCABULARY,
    )
    gold = row.gold[0]
    assert gold.is_noul is True
    assert gold.value is None
    options = row.request.slots[0].options  # type: ignore[union-attr]
    assert len(options) == MAX_CHOICE_OPTIONS
    assert "oos" not in options, "the abstain answer is noul, never a listed option"


def test_squad_unanswerable_becomes_an_abstention_on_the_span_slot() -> None:
    row = rewrite_squad(
        squad_row(0, force_impossible=True), family_id="qa.answer_span", index=0,
        config=DataConfig(),
    )
    assert row.gold[0].is_noul is True
    assert row.gold[0].value is None


def test_an_answerable_span_points_at_the_line_holding_the_answer() -> None:
    """A span label off by one is invisible in class accuracy and teaches the pointer
    head to point one line off, systematically."""
    raw = squad_row(3, force_impossible=False)
    row = rewrite_squad(raw, family_id="qa.answer_span", index=0, config=DataConfig())
    start, end = row.gold[0].value  # type: ignore[misc]
    context_lines = row.request.context.decode("utf-8").split("\n")
    assert start == end
    assert raw.answers[0] in context_lines[start - 1]


def test_a_gold_answer_cannot_be_both_noul_and_a_value() -> None:
    from qd_data.rows import GoldAnswer

    with pytest.raises(ValueError, match="abstention carries no value"):
        GoldAnswer(slot_name="x", value="a", is_noul=True)
    with pytest.raises(ValueError, match="needs a value"):
        GoldAnswer(slot_name="x", value=None, is_noul=False)


# -- refusals are counted, never silent --------------------------------------


def test_a_language_outside_the_closed_option_set_is_refused_with_a_reason_code() -> None:
    with pytest.raises(RowRefused) as excinfo:
        rewrite_commitpackft(
            commitpackft_row(1, lang="Brainfuck"), family_id="code.language_id",
            index=0, config=DataConfig(), decoy_message="other",
        )
    assert excinfo.value.reason_code == "lang_not_in_option_set"


def test_a_no_op_change_is_refused_rather_than_scored() -> None:
    row = commitpackft_row(1)
    unchanged = type(row)(
        commit=row.commit, repos=row.repos, old_file=row.old_file, new_file=row.new_file,
        old_contents=row.new_contents, new_contents=row.new_contents,
        subject=row.subject, message=row.message, lang=row.lang, licence=row.licence,
    )
    with pytest.raises(RowRefused) as excinfo:
        rewrite_commitpackft(
            unchanged, family_id="code.change_scope", index=0, config=DataConfig(),
        )
    assert excinfo.value.reason_code == "no_changed_lines"


def test_commit_intent_without_a_decoy_is_refused_not_labelled_constant() -> None:
    with pytest.raises(RowRefused) as excinfo:
        rewrite_commitpackft(
            commitpackft_row(1), family_id="code.commit_intent", index=0,
            config=DataConfig(), decoy_message=None,
        )
    assert excinfo.value.reason_code == "no_decoy_available"


def test_a_pull_too_small_for_the_intent_option_set_refuses_visibly() -> None:
    """A sharp edge worth a test: ``intent.classification`` asks over 16 intents, so
    a pull with fewer than 16 distinct in-scope intents produces **none** of that
    family. That must appear as a counted refusal, not as a quietly absent family."""
    config = DataConfig()
    mixture = build_mixture(
        {"clinc/clinc_oos": [clinc_row(i) for i in range(6)]}, config=config
    )
    assert not _by_family(mixture.rows, "intent.classification")
    assert mixture.refusals["clinc/clinc_oos"]["intent_vocabulary_too_small"] == 6
    assert _by_family(mixture.rows, "intent.in_scope"), "the other family still builds"


def test_refusal_counts_reach_the_result_rather_than_disappearing() -> None:
    """Counted refusals, and a status that carries both numbers rather than one.

    The fixture used to make *every* row's ``lang`` miss the option set, and then
    asserted the status was ``Ran``. That assertion encoded the wrong expectation:
    wiping out a held-out family is not a clean pass, and it is now ``NotRun`` --
    see ``test_a_family_wiped_out_by_out_of_set_langs_is_not_run_not_a_clean_pass``,
    which owns that case. What this test is *for* -- the reason count surviving into
    the result, and ``n``/``n_total`` carrying built against attempted -- is
    unchanged, and is asserted here on a partial refusal, which is the general case.
    """
    config = DataConfig()
    rows = [commitpackft_row(i) for i in range(5)] + [
        commitpackft_row(50 + i, lang="Brainfuck") for i in range(5)
    ]
    mixture = build_mixture({"bigcode/commitpackft": rows}, config=config)
    assert mixture.refusals["bigcode/commitpackft"]["lang_not_in_option_set"] == 5
    assert len(_by_family(mixture.rows, "code.language_id")) == 5
    status = mixture.status
    assert isinstance(status, Ran)
    assert status.n_total == status.n + 5


def test_commit_intent_labels_are_not_constant() -> None:
    """A yes/no family whose gold is always 'yes' is a degenerate head waiting to
    happen; the decoy construction must produce both labels."""
    mixture = build_mixture(
        {"bigcode/commitpackft": [commitpackft_row(i) for i in range(40)]},
        config=DataConfig(),
    )
    labels = {r.gold[0].value for r in _by_family(mixture.rows, "code.commit_intent")}
    assert labels == {"yes", "no"}


def test_change_scope_bins_are_fixed_not_quantiles_of_the_sample() -> None:
    """Quantile edges would make the same commit carry a different ordinal in two
    runs, and ``data_snapshot_hash`` would move for an unchanged corpus."""
    config = DataConfig()
    small = build_mixture(
        {"bigcode/commitpackft": [commitpackft_row(i) for i in range(4)]}, config=config
    )
    large = build_mixture(
        {"bigcode/commitpackft": [commitpackft_row(i) for i in range(40)]}, config=config
    )
    small_scope = {
        r.identity_key: r.gold[0].value for r in _by_family(small.rows, "code.change_scope")
    }
    large_scope = {
        r.identity_key: r.gold[0].value for r in _by_family(large.rows, "code.change_scope")
    }
    for key, value in small_scope.items():
        assert large_scope[key] == value
    slot = _by_family(small.rows, "code.change_scope")[0].request.slots[0]
    assert isinstance(slot, ScoreSlot)
    assert slot.bins == len(CHANGE_SCOPE_BIN_EDGES) + 1


def test_an_empty_mixture_is_not_run_not_a_clean_pass() -> None:
    mixture = build_mixture({"bigcode/commitpackft": []}, config=DataConfig())
    assert isinstance(mixture.status, NotRun)
    assert "no rows survived" in mixture.status.reason


def test_a_capped_read_makes_the_mixture_not_run() -> None:
    """*Never present a capped sample as complete coverage.*"""
    mixture = build_mixture(
        {"bigcode/commitpackft": [commitpackft_row(i) for i in range(4)]},
        config=DataConfig(),
        capped_sources=["bigcode/commitpackft"],
    )
    assert mixture.rows
    assert isinstance(mixture.status, NotRun)
    assert "hit its row bound" in mixture.status.reason


def test_the_language_option_set_is_closed_and_fits_the_letter_slice() -> None:
    assert len(LANGUAGE_OPTIONS) == MAX_CHOICE_OPTIONS
    assert len(set(LANGUAGE_OPTIONS)) == len(LANGUAGE_OPTIONS)


# -- per-family coverage: GAP-DATA-COMMITPACKFT-LANG-SET-UNVERIFIED ----------
#
# The gap asked what fraction of `bigcode/commitpackft` rows carry a `lang` inside
# LANGUAGE_OPTIONS. That fraction is unmeasurable here -- the dataset is unreachable
# and `datasets` is not installed -- so these test the question it was a proxy for:
# *what does the pipeline do with a row whose `lang` is outside the closed set?*
#
# It refuses the row loudly and counts the reason, which is right. What it did not
# do is notice when the refusals took a whole family with them: `code.language_id`
# is one of the two held-out families, and a pull whose `lang` spelling does not
# match the option set produces **none** of it while `build_mixture` still returned
# `Ran(passed=True)`. Downstream, `_held_out_families_absent` then passes vacuously
# (no rows of the family exist anywhere, so none is in a training shard), the
# manifest records a clean stage, and every report says the holdout was exercised.


def test_a_family_wiped_out_by_out_of_set_langs_is_not_run_not_a_clean_pass() -> None:
    """The defect this lane was pointed at: total refusal of a family read as a pass.

    ``code.language_id`` is held out. If every row's ``lang`` misses the closed set
    the family has no examples at all, so nothing about it was built and nothing
    about it can be checked -- that is ``NotRun``, and the training door refuses a
    ``NotRun`` snapshot by default. ``Ran(passed=False)`` would **not** do:
    ``qd_train.data_access.open_training_data`` branches only on ``NotRun``.
    """
    rows = [commitpackft_row(i, lang="Brainfuck") for i in range(5)]
    mixture = build_mixture({"bigcode/commitpackft": rows}, config=DataConfig())

    assert not _by_family(mixture.rows, "code.language_id")
    assert mixture.rows, "the other two families still build; this is not an empty mixture"

    status = mixture.status
    assert isinstance(status, NotRun), (
        "a held-out family with zero examples must not report as a clean pass"
    )
    assert "code.language_id" in status.reason
    assert "lang_not_in_option_set" in status.reason


def test_per_family_coverage_carries_built_and_attempted_not_just_a_source_total() -> None:
    """The fraction the gap asks for must be recoverable from the result.

    ``refusals`` is keyed by *source*, and ``n_input`` counts raw rows read once --
    but ``build_mixture`` iterates those rows once per family, so a commitpackft
    source with three families makes ``3 * n_input`` attempts. Dividing a reason
    count by ``n_input`` therefore answers a question nobody asked. The per-family
    pair is the honest denominator.
    """
    good = [commitpackft_row(i) for i in range(5)]
    bad = [commitpackft_row(100 + i, lang="Brainfuck") for i in range(3)]
    mixture = build_mixture({"bigcode/commitpackft": good + bad}, config=DataConfig())

    coverage = mixture.family_coverage["code.language_id"]
    assert isinstance(coverage, Ran)
    assert (coverage.n, coverage.n_total) == (5, 8)
    assert not coverage.is_complete_coverage
    assert "lang_not_in_option_set" in coverage.detail

    # The families that were unaffected say so with full coverage, so the two facts
    # are distinguishable rather than averaged into one source-level number.
    scope = mixture.family_coverage["code.change_scope"]
    assert isinstance(scope, Ran)
    assert (scope.n, scope.n_total) == (8, 8)
    assert scope.is_complete_coverage

    assert mixture.to_json()["family_coverage"]["code.language_id"]["n"] == 5


def test_family_coverage_survives_into_the_json_the_manifest_carries() -> None:
    mixture = build_mixture(
        {"bigcode/commitpackft": [commitpackft_row(i) for i in range(4)]},
        config=DataConfig(),
    )
    blob = mixture.to_json()["family_coverage"]
    assert set(blob) == {"code.commit_intent", "code.language_id", "code.change_scope"}
    for entry in blob.values():
        assert entry["state"] == "ran"
        assert entry["n"] == entry["n_total"] == 4


def test_the_lang_check_is_exact_so_a_case_variant_is_refused_not_bucketed() -> None:
    """No normalisation anywhere on this path, and that is the deliberate choice.

    ``qd_data.loaders.parse_commitpackft`` takes ``lang`` verbatim and the membership
    test is exact, so ``"python"`` is refused rather than folded into ``"Python"``.
    Folding would be *silent bucketing*: the gold label would stop being the value
    the upstream row carried, and a pull whose spelling convention differs from this
    16-value set would look like a clean pull with a slightly different mixture.
    Refusing keeps the mismatch on the record where a human has to widen the set or
    accept the thinning.
    """
    for variant in ("python", "PYTHON", "Python ", " Python"):
        with pytest.raises(RowRefused) as excinfo:
            rewrite_commitpackft(
                commitpackft_row(1, lang=variant), family_id="code.language_id",
                index=0, config=DataConfig(),
            )
        assert excinfo.value.reason_code == "lang_not_in_option_set"
        assert excinfo.value.actual == variant

    # ... and the exact spelling is what reaches the gold answer.
    row = rewrite_commitpackft(
        commitpackft_row(1, lang="Python"), family_id="code.language_id",
        index=0, config=DataConfig(),
    )
    assert row.gold[0].value == "Python"


# -- the Trojan Source class: GAP-DATA-RENDER-BIDI-UNESCAPED -----------------


def test_a_bidi_override_in_a_commit_message_is_refused_and_counted() -> None:
    """A row that reads one way and tokenises another never enters the corpus.

    U+202E RIGHT-TO-LEFT OVERRIDE is the Trojan Source character: the reviewer
    reading the rendered prompt, the diff or the model card is shown text in an order
    the model was never given. Every *structural* check in this lane passes on such a
    row -- the option count, the marker count, the letter map and the line count are
    all untouched -- which is exactly why it needs a refusal of its own: the control
    it defeats is a human reading rendered text.
    """
    rlo = "‮"
    row = commitpackft_row(1)
    poisoned = CommitPackFtRow(
        commit=row.commit, repos=row.repos, old_file=row.old_file, new_file=row.new_file,
        old_contents=row.old_contents,
        new_contents=f"def handler():\n    # {rlo}return False;  //\n    return True\n",
        subject=row.subject, message=row.message, lang="Python", licence=row.licence,
    )
    with pytest.raises(RowRefused) as excinfo:
        rewrite_commitpackft(
            poisoned, family_id="code.language_id", index=0, config=DataConfig(),
        )
    assert excinfo.value.reason_code == "invisible_format_characters"
    assert "U+202E" in str(excinfo.value.actual)

    mixture = build_mixture({"bigcode/commitpackft": [poisoned]}, config=DataConfig())
    assert mixture.refusals["bigcode/commitpackft"]["invisible_format_characters"] >= 1
    assert not mixture.rows


@pytest.mark.parametrize(
    ("name", "ch"),
    [
        ("RLO", "‮"),          # the named Trojan Source override
        ("LRI", "⁦"),          # the isolate form of the same attack
        ("RLM", "‏"),          # the implicit mark the named eight leave out
        ("ALM", "؜"),          # ... and its Arabic sibling
        ("ZWSP", "​"),         # displays identically, tokenises differently
        ("ZWJ", "‍"),
        ("SOFT HYPHEN", "­"),
        ("BOM", "﻿"),
        ("WORD JOINER", "⁠"),
    ],
)
def test_the_whole_invisible_class_is_refused_not_only_the_named_overrides(
    name: str, ch: str
) -> None:
    """The gap named eight characters. Eight characters are not a class.

    The same reordering is reachable through the implicit marks and the same
    "displays identically" trick through the zero-width characters, so the closed
    class -- Unicode general category Cf -- is what is refused.
    """
    row = commitpackft_row(2)
    poisoned = CommitPackFtRow(
        commit=row.commit, repos=row.repos, old_file=row.old_file, new_file=row.new_file,
        old_contents=row.old_contents,
        new_contents=f"def handler():\n    value{ch} = 1\n    return value\n",
        subject=row.subject, message=row.message, lang="Python", licence=row.licence,
    )
    with pytest.raises(RowRefused) as excinfo:
        rewrite_commitpackft(
            poisoned, family_id="code.change_scope", index=0, config=DataConfig(),
        )
    assert excinfo.value.reason_code == "invisible_format_characters", name


def test_an_invisible_character_in_an_option_is_refused_at_the_one_funnel() -> None:
    """Options are untrusted too: CLINC's option set is built from the dataset.

    The gold intent is always one of the sixteen options, so poisoning it is the way
    to reach the option arm deterministically -- the other fifteen are a seeded
    sample and asserting on them would be asserting on the shuffle.
    """
    poisoned_intent = "intent_​000"
    poisoned = ClincRow(
        utterance="please handle this request for my account", intent=poisoned_intent,
        is_oos=False,
    )
    with pytest.raises(RowRefused) as excinfo:
        rewrite_clinc(
            poisoned, family_id="intent.classification", index=0, config=DataConfig(),
            intent_vocabulary=(*INTENT_VOCABULARY, poisoned_intent),
        )
    assert excinfo.value.reason_code == "invisible_format_characters"
    assert "option" in str(excinfo.value.actual)


def test_an_invisible_character_in_a_repo_name_is_refused_before_the_manifest() -> None:
    """``canonical_json`` uses ``ensure_ascii=False``, so a format character in a
    split key reaches the manifest raw and a human auditing it reads the wrong name.
    Same class, so the same refusal, at the ``DataRow`` funnel."""
    with pytest.raises(RowRefused) as excinfo:
        rewrite_commitpackft(
            commitpackft_row(3, repo="org/re‭po"), family_id="code.change_scope",
            index=0, config=DataConfig(),
        )
    assert excinfo.value.reason_code == "invisible_format_characters"
    assert "repo_key" in str(excinfo.value.actual)


def test_ordinary_non_ascii_text_is_not_swept_up_by_the_invisible_check() -> None:
    """The refusal is the Cf class, not "non-ASCII". Arabic, Hebrew, CJK and emoji
    carry real glyphs and must still build, or the check is a data filter wearing a
    security argument."""
    row = commitpackft_row(4)
    for body in ("# مرحبا بالعالم\n", "# שלום עולם\n", "# 你好世界\n", "# ok \U0001f600\n"):
        fine = CommitPackFtRow(
            commit=row.commit, repos=row.repos, old_file=row.old_file,
            new_file=row.new_file, old_contents=row.old_contents,
            new_contents=f"def handler():\n{body}    return True\n",
            subject=row.subject, message=row.message, lang="Python", licence=row.licence,
        )
        built = rewrite_commitpackft(
            fine, family_id="code.language_id", index=0, config=DataConfig(),
        )
        assert body.strip() in built.request.context.decode("utf-8")


# -- corpus self-consistency: GAP-DATA-NOTHING-REFUSES-TWO-ROWS-THAT-CONTRADICT
#
# Every test above this line is about one row. These are about two, and about the
# one defect a per-row check cannot see: a corpus of individually valid rows that
# asks the model the same question twice and demands two different answers.
#
# It was not hypothetical. The FT lane's span channel converged on 0.693147 -- ln 2,
# a fair coin -- because all 78 span rows were 39 prompt-identical pairs with
# contradictory golds. Dedupe saw them (``dedupe_text`` was byte-identical) and kept
# them, correctly: its question is leakage, and within one repo there is none. The
# mixture reported ``Ran(passed=True)``, 608 rows in and 608 out.


def _contradictory_squad_pair(title: str) -> list[SquadRow]:
    """Two rows, one question, two golds -- the shape that measured ln 2.

    The qids differ because ``row_id`` is built from them and ``dedupe`` refuses a
    duplicate id; *nothing else* differs, which is the whole point. This is exactly
    what ``tools/real_tokenizer_pipeline.py::span_rows`` used to emit per Markdown
    file, before it was given a different question for the unanswerable row.
    """
    passage = (
        "Rule one is stated on the opening line of the document.\n"
        "Rule two says the ledger row is written before the claim.\n"
        "Rule three is unrelated and concerns formatting."
    )
    question = "Which line states the rule?"
    needle = "Rule two says"
    return [
        SquadRow(
            qid="q-answerable", title=title, context=passage, question=question,
            answers=(needle,), answer_starts=(passage.index(needle),), is_impossible=False,
        ),
        SquadRow(
            qid="q-unanswerable", title=title, context=passage, question=question,
            answers=(), answer_starts=(), is_impossible=True,
        ),
    ]


def _both_contradictory_pairs() -> list[SquadRow]:
    """The pair once per SQuAD family, each under a title routed to that family: SQuAD is
    partitioned by article title (user decision 2026-09-29), so one title cannot feed both."""
    return [
        *_contradictory_squad_pair(squad_title_for("qa.answer_span", stem="Rules")),
        *_contradictory_squad_pair(squad_title_for("qa.answerability", stem="Rules")),
    ]


def test_two_rows_with_one_prompt_and_two_golds_are_both_dropped_and_counted() -> None:
    """A prompt that carries two golds loses every row, counted; no winner is chosen.

    Both rows are individually valid, both render, both encode, both pass every
    per-row check in the lane -- and a causal model conditions on the prompt and
    nothing else, so it cannot answer better than chance on the pair however long it
    is trained. Until 2026-09-29 the mixture kept both rows and refused the whole
    corpus. Retired then: the approved SQuAD v2 cache holds 29 such pairs upstream
    and CLINC 4, and refusing 315k rows over 33 duplicates is the wrong remedy. The
    group is named, so which rows collided is still legible.
    """
    mixture = build_mixture(
        {"rajpurkar/squad_v2": _both_contradictory_pairs()}, config=DataConfig()
    )
    assert mixture.rows == (), "no row of a contradictory prompt survives"
    assert mixture.refusals["rajpurkar/squad_v2"] == {CONTRADICTORY_PROMPT: 4}

    consistency = mixture.prompt_consistency
    assert isinstance(consistency, Ran)
    assert consistency.passed, "the verdict is on the rows emitted, and none contradict"
    # One raw pair, but two families are built from it, so two rendered prompts each
    # carry two golds: the span pair that measured ln 2, and the yes/no
    # answerability pair behind the same words.
    assert consistency.value == 2
    assert (consistency.n, consistency.n_total) == (4, 4)
    assert "all 4 of their rows were dropped" in consistency.detail

    by_family = {c.family_ids: c for c in mixture.contradictions}
    assert set(by_family) == {("qa.answer_span",), ("qa.answerability",)}

    span_group = by_family[("qa.answer_span",)]
    assert set(span_group.row_ids) == {
        "squad:qa.answer_span:q-answerable",
        "squad:qa.answer_span:q-unanswerable",
    }
    assert span_group.n_rows == 2
    assert len(span_group.golds) == 2, "the two golds are named, not merely counted"
    assert any('"is_noul":true' in g for g in span_group.golds)

    # Every row of both families went, so both families produced nothing: the corpus is
    # NotRun for that, not passed -- a family emptied by the drop is still uncovered.
    status = mixture.status
    assert isinstance(status, NotRun)
    assert "the consistency drop (4 row(s) of 2 contradictory prompt(s)" in status.reason


def test_a_three_row_group_loses_all_three_rather_than_keeping_the_majority() -> None:
    """Two rows agreeing is not evidence that the third is wrong; nothing picks a winner."""
    pair = _contradictory_squad_pair(squad_title_for("qa.answer_span", stem="Three"))
    answerable = pair[0]
    third = SquadRow(
        qid="q-answerable-again", title=answerable.title, context=answerable.context,
        question=answerable.question, answers=answerable.answers,
        answer_starts=answerable.answer_starts, is_impossible=False,
    )
    keep = squad_row(7, family="qa.answer_span")
    mixture = build_mixture(
        {"rajpurkar/squad_v2": [*pair, third, keep]}, config=DataConfig(),
        families=["qa.answer_span"],
    )
    assert [r.row_id for r in mixture.rows] == [f"squad:qa.answer_span:{keep.qid}"]
    assert mixture.refusals["rajpurkar/squad_v2"] == {CONTRADICTORY_PROMPT: 3}
    (group,) = mixture.contradictions
    assert (group.n_rows, group.n_golds) == (3, 2)
    coverage = mixture.family_coverage["qa.answer_span"]
    assert isinstance(coverage, Ran) and (coverage.n, coverage.n_total) == (1, 4)
    assert isinstance(mixture.status, Ran) and mixture.status.passed


def test_one_clinc_utterance_under_two_intents_is_dropped() -> None:
    """The CLINC shape measured on the approved cache: "what is on my to do list" is
    filed under two intents. The letter channel loses the pair the way the span one does.

    It collides on ``intent.domain``, whose option list is the fixed domain set: two
    intents in different domains behind one utterance are one prompt with two golds.
    ``intent.classification`` samples its distractors per row, so the same pair renders
    two different option lists there and is correctly not grouped.
    """
    from qd_data.general import CLINC_DOMAIN_FAMILY, ClincDomainMap

    vocab = sorted(set(INTENT_VOCABULARY))
    q = len(vocab) // 4
    domains = {f"d{k}": tuple(vocab[k * q:(k + 1) * q]) for k in range(4)}
    domain_of = {i: d for d, intents in domains.items() for i in intents}
    rows = [r for r in (clinc_row(i) for i in range(1, 25)) if r.intent in domain_of]
    other = next(i for i in domain_of if domain_of[i] != domain_of[rows[0].intent])
    twin = ClincRow(utterance=rows[0].utterance, intent=other, is_oos=False)
    mixture = build_mixture(
        {"clinc/clinc_oos": [*rows, twin]}, config=DataConfig(),
        families=[CLINC_DOMAIN_FAMILY], clinc_domain_map=ClincDomainMap(domains=domains),
    )
    assert len(mixture.rows) == len(rows) + 1 - 2
    assert mixture.refusals["clinc/clinc_oos"] == {CONTRADICTORY_PROMPT: 2}
    (group,) = mixture.contradictions
    assert group.family_ids == (CLINC_DOMAIN_FAMILY,)
    assert (group.n_rows, group.n_golds) == (2, 2)


def test_the_dropped_groups_reach_the_manifest_and_the_rest_reaches_training(
    tmp_path: Path,
) -> None:
    """End to end: the drop has to be legible where a reader finds it.

    Until 2026-09-29 the same corpus was refused at ``open_training_data`` (the whole
    corpus, over one pair). Now the pair is dropped before the split, the corpus opens,
    no split holds either row, and the manifest names the groups and the count.
    """
    config = DataConfig()
    # A corpus rich enough that every *other* stage reports a clean `Ran` -- a
    # vendored pair for the cross-repo dedupe check, all three sources for the split
    # -- so the refusal under test is attributable to this check and not to a
    # snapshot that was `NotRun` for unrelated reasons. Where the bad pair *lands* is
    # deliberately not asserted: the mixture's verdict is over the whole corpus, so
    # it reaches the training manifest whichever side of the split the two rows
    # fall on.
    corpus = small_corpus(24)
    corpus["bigcode/commitpackft"] += list(vendored_pair())
    corpus["rajpurkar/squad_v2"] += _both_contradictory_pairs()
    mixture = build_mixture(corpus, config=config)
    report = dedupe(list(mixture.rows), config=config)
    manifests = build_manifests(
        config=config, mixture=mixture, dedupe_report=report,
        split_report=split(report, config=config),
    )
    path = tmp_path / "train.json"
    manifests["train"].write(path)

    # The rest of the corpus reaches training: the pair no longer vetoes it.
    open_training_data(path, config=config, repo_root=tmp_path)
    bad = {
        "squad:qa.answer_span:q-answerable", "squad:qa.answer_span:q-unanswerable",
        "squad:qa.answerability:q-answerable", "squad:qa.answerability:q-unanswerable",
    }
    for name, manifest in manifests.items():
        assert not bad & {e.row_id for e in manifest.entries}, name

    # The groups survive the round trip through the file, which is where a reader
    # who was not present for the run has to find them.
    written = Manifest.read(path)
    assert written.mixture_json["refusals"]["rajpurkar/squad_v2"][CONTRADICTORY_PROMPT] == 4
    recorded = written.mixture_json["contradictions"]
    assert [c["n_rows"] for c in recorded] == [2, 2]
    assert all(len(c["golds"]) == 2 for c in recorded)
    assert {tuple(c["family_ids"]) for c in recorded} == {
        ("qa.answer_span",), ("qa.answerability",)
    }
    assert {i for c in recorded for i in c["row_ids"]} == {
        "squad:qa.answer_span:q-answerable",
        "squad:qa.answer_span:q-unanswerable",
        "squad:qa.answerability:q-answerable",
        "squad:qa.answerability:q-unanswerable",
    }


def test_an_unanswerable_row_beside_an_answerable_one_is_not_refused() -> None:
    """The legitimate case, which a check that refused it would make worse than none.

    An unanswerable row is *supposed* to exist beside an answerable one -- that is
    what SQuAD 2.0 is and what the abstention gate is measured on. The shape is not
    the defect; the identical prompt is. SQuAD's unanswerable question uses different
    words, so the honest pair renders two prompts and never meets in a group.
    """
    passage = (
        "Rule one is stated on the opening line of the document.\n"
        "Rule two says the ledger row is written before the claim.\n"
        "Rule three is unrelated and concerns formatting."
    )
    # SQuAD is partitioned by article title between its two families (user decision
    # 2026-09-29, qd_data.split.squad_title_family), so a title feeds one family only.
    span_title = squad_title_for("qa.answer_span", stem="Rules")
    needle = "Rule two says"
    rows = [
        SquadRow(
            qid="q-answerable", title=span_title, context=passage,
            question="Which line states the rule?", answers=(needle,),
            answer_starts=(passage.index(needle),), is_impossible=False,
        ),
        SquadRow(
            qid="q-unanswerable", title=span_title, context=passage,
            question="Which line names the author of the rule?", answers=(),
            answer_starts=(), is_impossible=True,
        ),
    ]
    mixture = build_mixture(
        {"rajpurkar/squad_v2": rows}, config=DataConfig(), families=["qa.answer_span"]
    )

    assert not mixture.contradictions
    consistency = mixture.prompt_consistency
    assert isinstance(consistency, Ran) and consistency.passed
    assert isinstance(mixture.status, Ran) and mixture.status.passed

    spans = _by_family(mixture.rows, "qa.answer_span")
    assert [g.is_noul for r in spans for g in r.gold].count(True) == 1, (
        "the abstaining row is still there; it was not refused away to pass the check"
    )


def test_two_identical_rows_with_the_same_gold_are_a_duplicate_not_a_contradiction() -> None:
    """Dedupe's question stays dedupe's.

    Same prompt *and* same gold teaches one thing twice, which is redundancy, not an
    unlearnable pair -- and a check that conflated the two would be one helper
    answering two questions with a flag meaning "do the other thing".
    """
    passage = "Only line one matters here.\nThe second line is filler.\nThe third too."
    needle = "Only line one"
    rows = [
        SquadRow(
            qid=f"q-{i}", title=squad_title_for("qa.answer_span", stem="Dup"),
            context=passage, question="Which line matters?",
            answers=(needle,), answer_starts=(passage.index(needle),), is_impossible=False,
        )
        for i in range(2)
    ]
    mixture = build_mixture(
        {"rajpurkar/squad_v2": rows}, config=DataConfig(), families=["qa.answer_span"]
    )
    assert not mixture.contradictions
    assert isinstance(mixture.status, Ran) and mixture.status.passed


def test_a_contradiction_is_found_on_the_letter_channel_too() -> None:
    """Not a span-only check.

    The measured case was ``qa.answer_span``; the mechanism is the prompt, so two
    rows carrying one language prompt and two language labels are the same defect
    and must be caught by the same pass.
    """
    shared = code_body("identical", lines=6)
    a = commitpackft_row(1, repo="org/one", body=shared, path="lib/a.py", lang="Python")
    b = CommitPackFtRow(
        commit=f"{2:040x}", repos="org/one", old_file="lib/a.py", new_file="lib/a.py",
        old_contents=a.old_contents, new_contents=a.new_contents,
        subject=a.subject, message=a.message, lang="Rust", licence=a.licence,
    )
    mixture = build_mixture(
        {"bigcode/commitpackft": [a, b]}, config=DataConfig(),
        families=["code.language_id"],
    )
    assert mixture.rows == (), "both rows of the language pair are dropped (2026-09-29)"
    assert mixture.refusals["bigcode/commitpackft"] == {CONTRADICTORY_PROMPT: 2}
    consistency = mixture.prompt_consistency
    assert isinstance(consistency, Ran) and consistency.passed and consistency.value == 1
    (group,) = mixture.contradictions
    assert group.family_ids == ("code.language_id",)
    assert {'"value":"Python"' in g for g in group.golds} == {True, False}


def test_the_consistency_pass_is_bounded_and_says_so_rather_than_grinding() -> None:
    """Every fan-out in this lane is bounded.

    Over the bound it is ``NotRun`` with the bound in the reason -- never a pass over
    a subsample, which is this very defect one level up.
    """
    mixture = build_mixture(
        {"rajpurkar/squad_v2": _both_contradictory_pairs()},
        config=DataConfig(), max_consistency_rows=1,
    )
    consistency = mixture.prompt_consistency
    assert isinstance(consistency, NotRun)
    assert "bounded at 1" in consistency.reason
    assert not mixture.contradictions
    assert len(mixture.rows) == 4, "nothing was grouped, so nothing may be dropped"
    assert CONTRADICTORY_PROMPT not in mixture.refusals["rajpurkar/squad_v2"]
    assert isinstance(mixture.status, NotRun), (
        "an unchecked corpus is not a clean one; the training door refuses not_run"
    )
    assert "unchecked for contradictory supervision" in mixture.status.reason


def test_a_row_that_cannot_be_rendered_is_counted_not_silently_skipped() -> None:
    """A context over ``RenderCaps`` cannot be rendered, so it cannot be grouped.

    ``qd_train.shards`` refuses exactly those rows around its own ``render`` call, so
    such a row reaches no shard and can contradict nothing that does -- which is a
    reason to carry both numbers, not a reason to call the pass complete.
    """
    big = "x" * (DEFAULT_CAPS.max_context_bytes + 1)
    rows = [commitpackft_row(0), commitpackft_row(1, body=big)]
    mixture = build_mixture(
        {"bigcode/commitpackft": rows}, config=DataConfig(), families=["code.language_id"],
    )
    assert len(mixture.rows) == 2, "the over-cap row is built; render is what refuses it"

    consistency = mixture.prompt_consistency
    assert isinstance(consistency, Ran) and consistency.passed
    assert (consistency.n, consistency.n_total) == (1, 2)
    assert not consistency.is_complete_coverage, (
        "one of two rows was grouped; reporting that as full coverage is the defect "
        "the tri-state exists to prevent"
    )
    assert "could not be rendered" in consistency.detail


def test_a_consistency_check_over_an_empty_corpus_claims_nothing() -> None:
    verdict, groups = check_prompt_consistency([])
    assert isinstance(verdict, Ran) and verdict.passed
    assert (verdict.n, verdict.n_total) == (0, 0)
    assert not groups


# -- abstention supply: GAP-DATA-NO-LETTER-ROW-EVER-ABSTAINS ------------------


def test_the_letter_channels_report_their_abstention_supply() -> None:
    """A corpus can teach ``noul`` on the span channel and never once on the letter
    channel. Measured on the real corpus: 0 of 582 letter rows carried a ``noul``
    gold, while 45 of 90 span rows did."""
    no_clinc = {
        "bigcode/commitpackft": [commitpackft_row(i) for i in range(6)],
        "rajpurkar/squad_v2": [
            squad_row(i, family="qa.answerability" if i % 2 else "qa.answer_span")
            for i in range(6)
        ],
    }
    mixture = build_mixture(no_clinc, config=DataConfig())
    abstention = mixture.abstention
    assert set(abstention) == {"choice", "score", "span"}

    span = abstention["span"]
    assert isinstance(span, Ran) and span.passed and span.n > 0

    for channel in ("choice", "score"):
        verdict = abstention[channel]
        assert isinstance(verdict, Ran)
        assert not verdict.passed
        assert verdict.n == 0 and verdict.n_total > 0
        assert "trains this channel against the abstain row" in verdict.detail
        assert "composition rather than a rewriter dropping them" in verdict.detail

    assert isinstance(mixture.status, Ran) and mixture.status.passed, (
        "reported, not adopted: whether a corpus with no letter-channel abstention "
        "may be trained on is a kill criterion, and rule 2 puts adopting one out of "
        "an agent's reach"
    )


def test_clinc_is_the_only_letter_family_that_abstains_without_a_separate_corpus() -> None:
    """The structural half of the finding, checked by execution rather than asserted.

    Four of the five letter families assign a value on every branch of their own
    sources. ``intent.classification`` abstains from CLINC's out-of-scope rows -- which
    is why a corpus built without ``clinc/clinc_oos`` has zero abstaining letter rows
    from its own families however large it is. ``code.defect_class`` abstains on its
    class only over the separately loaded noul corpus, which this corpus does not hold.
    """
    mixture = build_mixture(small_corpus(18), config=DataConfig())
    letter_noul = {
        r.family_id
        for r in mixture.rows
        if any(g.is_noul for g in r.gold)
        and isinstance(r.request.slots[0], (ChoiceSlot, ScoreSlot))
    }
    assert letter_noul == {"intent.classification"}
    # code.defect_class joined the roster with its span slot, and since the noul corpus
    # (2026-09-30) abstains on its class too -- but only over that corpus's rows, which
    # this corpus does not hold, so the execution above still finds CLINC alone.
    assert set(ABSTAINING_FAMILIES) == {
        "intent.classification", "qa.answer_span", "code.defect_class"
    }
    assert [f for f, ch in ABSTAINING_FAMILIES.items() if "choice" in ch] == [
        "intent.classification", "code.defect_class"
    ]

    choice = mixture.abstention["choice"]
    assert isinstance(choice, Ran) and choice.passed
    assert "intent.classification" in choice.detail

    score = mixture.abstention["score"]
    assert isinstance(score, Ran) and not score.passed
    assert "able to abstain at all: none" in score.detail


def test_a_named_group_shows_both_sides_of_the_disagreement_and_bounds_its_payload() -> None:
    """Both halves of "carry both numbers", on the one payload written to every file.

    Naming the lowest row ids outright can show a reader several rows that all carry
    the *same* answer and call it the pair, so one id per distinct gold is taken
    first. And the named lists are capped while ``n_rows``/``n_golds`` are not: a
    pathological corpus of identical prompts with distinct golds would otherwise put
    megabytes of worked examples into a manifest whose job is to be read.
    """
    words = ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot")
    passage = "\n".join(f"{w} is on its own line." for w in words)
    question = "Which line matters?"
    wide = squad_title_for("qa.answer_span", stem="Wide")

    def answerable(i: int) -> SquadRow:
        word = words[i % len(words)]
        return SquadRow(
            qid=f"q{i:03d}", title=wide, context=passage, question=question,
            answers=(word,), answer_starts=(passage.index(word),), is_impossible=False,
        )

    rows = [
        SquadRow(
            qid="q000", title=wide, context=passage, question=question,
            answers=(), answer_starts=(), is_impossible=True,
        ),
        *(answerable(i) for i in range(1, 12)),
    ]
    mixture = build_mixture(
        {"rajpurkar/squad_v2": rows}, config=DataConfig(), families=["qa.answer_span"],
    )
    (group,) = mixture.contradictions

    # Six distinct spans plus the abstention, behind one prompt.
    assert group.n_rows == len(rows) == 12
    assert group.n_golds == len(words) + 1 == 7
    assert len(group.row_ids) == MAX_NAMED_ROW_IDS < group.n_rows
    assert len(group.golds) == MAX_NAMED_GOLDS < group.n_golds

    # The witness rule: the named rows span the disagreement rather than one side. The
    # group's rows are dropped, so which one abstains is read off the input.
    assert mixture.rows == ()
    named = set(group.row_ids)
    abstaining = {f"squad:qa.answer_span:{r.qid}" for r in rows if r.is_impossible}
    answering = {f"squad:qa.answer_span:{r.qid}" for r in rows} - abstaining
    assert named & abstaining, "no abstaining row was named"
    assert named & answering, "no answering row was named"
    assert len(named) == len(group.row_ids), "the named ids are distinct"
