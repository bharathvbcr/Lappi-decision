"""The open-task-mixture rewriter: one format, per-row licensing, counted refusals.

The three claims under test:

* every family renders through the **one** ``Request`` shape, so nothing here can
  drift from the serving format;
* a dataset whose licence is not on the permissive allowlist is refused **at load**
  with a message naming the licence, and a *row* whose licence is not on it is
  refused even when its dataset is ``mit``;
* the free ``noul`` supervision is actually produced -- CLINC's out-of-scope class
  and SQuAD's unanswerable questions -- and is a gold *value*, never an error.
"""

from __future__ import annotations

import pytest
from data_fixtures import (
    INTENT_VOCABULARY,
    clinc_row,
    commitpackft_row,
    small_corpus,
    squad_row,
)
from qd_data.config import DataConfig
from qd_data.errors import LicenceRefused
from qd_data.licences import LicenceConfig
from qd_data.loaders import ClincRow, CommitPackFtRow, SourceUnavailableRefusal
from qd_data.mixture import (
    CHANGE_SCOPE_BIN_EDGES,
    LANGUAGE_OPTIONS,
    RowRefused,
    build_mixture,
    rewrite_clinc,
    rewrite_commitpackft,
    rewrite_squad,
)
from qd_data.render import render, render_for_serving
from qd_data.rows import DataRow
from qd_data.schema import MAX_CHOICE_OPTIONS, ChoiceSlot, ScoreSlot, SpanSlot
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
