"""Exact-content dedupe for structured decision rows (Fable's ruling, 2026-10-03).

``AUDIT/finalize-2026-10-03/dedupe-probe/RULING.md``: the v5 decision pool's structured Open-Jev
rows share ~2 KB of rule prose and carry each problem's facts as compact JSON, which whitespace
shingles barely see. MinHash at 0.8 therefore paired *distinct problems* -- 31-77% of the pairs it
proposed had different gold answers -- and the candidate search hit its 5,000,000 bound.

The ruling, as these tests hold it:

* a row whose ``metadata[NEAR_DUPLICATE_POLICY_KEY]`` is ``EXACT_CONTENT`` is deduped by the digest
  of its ``dedupe_text`` **alone** (no repo, no identity key), in any repo;
* it never enters the MinHash candidate search, in dedupe or in the split's re-derivation;
* the near-duplicate search is reported as covering only the rows it searched, with both counts,
  and the scoped families named -- never as a complete pass;
* its leak is the same content digest on two sides of the split, checked independently in split;
* an unknown policy value is refused, never read as "absent".

The corpora are built to fail the old behaviour: shared prose with different facts (which MinHash
merges), identical content in one repo (which MinHash keeps as within-repo redundancy), and enough
shared-prose rows to overflow a small candidate bound.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from qd_data.config import DataConfig
from qd_data.dedupe import EXACT_CONTENT, NEAR_DUPLICATE_POLICY_KEY, dedupe
from qd_data.rows import DataRow, GoldAnswer
from qd_data.schema import ChoiceSlot, Request
from qd_data.sources import PINNED_SPLIT_KEY
from qd_data.split import split
from qd_train.tristate import NotRun, Ran

#: ~120 words of rule prose every row shares, as the Open-Jev policy rows do.
PROSE = " ".join(
    f"Rule {i}: a candidate is eligible only when requirement {i} is met by evidence that is "
    f"recorded, current and not withdrawn."
    for i in range(10)
)


def _facts(i: int) -> str:
    """One problem's facts, as compact JSON with no whitespace: invisible to word shingles."""
    facts = {"amount": i * 7 % 97, "day": i % 28, "posted": i % 2 == 0}
    return json.dumps({"candidate": f"P{i}", "facts": facts}, separators=(",", ":"))


def _row(
    row_id: str,
    *,
    repo: str,
    text: str,
    family: str = "openjev.policy",
    gold: str = "yes",
    policy: str | None = EXACT_CONTENT,
    pinned: str | None = None,
) -> DataRow:
    metadata: dict[str, str] = {}
    if policy is not None:
        metadata[NEAR_DUPLICATE_POLICY_KEY] = policy
    if pinned is not None:
        metadata[PINNED_SPLIT_KEY] = pinned
    return DataRow(
        row_id=row_id,
        source_id="ZefanCai/Open-Jev-v1.1",
        host="huggingface",
        family_id=family,
        repo_key=repo,
        identity_key=f"{repo}::{row_id}",
        licence_id="apache-2.0",
        obligations=("attribution",),
        request=Request(
            task=family,
            context=text.encode("utf-8"),
            question="Is this candidate supported?",
            slots=(ChoiceSlot(name="answer", options=("yes", "no")),),
            example_id=row_id,
        ),
        gold=(GoldAnswer(slot_name="answer", value=gold),),
        dedupe_text=text,
        metadata=metadata,
    )


def _distinct_problems(n: int, *, policy: str | None = EXACT_CONTENT) -> list[DataRow]:
    """``n`` different problems in ``n`` repos: the shared prose, each its own facts."""
    return [
        _row(f"r{i}", repo=f"scene/{i}", text=f"{PROSE} {_facts(i)}", gold="yes" if i % 2 else "no",
             policy=policy)
        for i in range(n)
    ]


# -- characterization: what MinHash does to these rows (passes before and after) ---------------


def test_minhash_merges_distinct_problems_that_share_prose_when_unmarked() -> None:
    """The defect the ruling scopes out, shown on the unmarked path, which stays as it was."""
    report = dedupe(_distinct_problems(6, policy=None), config=DataConfig())
    assert isinstance(report.status, Ran)
    assert len(report.kept) < 6, "MinHash pairs these: the facts are a few shingles of ~100"


# -- the exact-content policy -------------------------------------------------------------------


def test_marked_distinct_problems_all_survive() -> None:
    report = dedupe(_distinct_problems(6), config=DataConfig())
    assert isinstance(report.status, Ran)
    assert len(report.kept) == 6
    assert report.n_dropped_rows == 0


def test_marked_rows_with_one_digest_are_one_problem_in_any_repo() -> None:
    """Same content in one repo (MinHash keeps it as redundancy) and across repos: one row each."""
    text_a, text_b = f"{PROSE} {_facts(1)}", f"{PROSE} {_facts(2)}"
    rows = [
        _row("a1", repo="scene/1", text=text_a),
        _row("a2", repo="scene/1", text=text_a),  # same repo, same content
        _row("b1", repo="scene/2", text=text_b),
        _row("b2", repo="scene/9", text=text_b),  # another repo, same content
    ]
    report = dedupe(rows, config=DataConfig())
    assert isinstance(report.status, Ran)
    assert sorted(r.row_id for r in report.kept) == ["a1", "b1"]
    assert report.dropped_row_ids == frozenset({"a2", "b2"})


def _twins_train_key_first() -> list[DataRow]:
    """The first v5 build's knock-out on this path: a val decision row and a train row of exactly
    its text in another group, the train key sorting first ("-train:" before "-val:")."""
    text = f"{PROSE} {_facts(4)}"
    return [
        _row("v", repo="openjev.policy-val:g1", text=text, pinned="val"),
        _row("t", repo="openjev.policy-train:g0", text=text, pinned="train"),
    ]


def test_the_v5_rule_drops_the_val_copy_of_identical_content_when_train_sorts_first() -> None:
    """Characterization, kept under the default config: v5's recorded knock-out."""
    assert [r.row_id for r in dedupe(_twins_train_key_first(), config=DataConfig()).kept] == ["t"]


def test_the_v6_rule_keeps_the_val_copy_of_identical_content_in_any_order() -> None:
    """GAP-QD-DATA-DEDUPE-KEEP-RULE-LEXICAL-SPLIT-BLIND-2026-10-03 on the exact-content path, where
    v5's knock-out actually happened (test_pre_dedupe_drops)."""
    config = DataConfig().with_v6_dedupe_rules()
    rows = _twins_train_key_first()
    for order in (rows, rows[::-1]):
        report = dedupe(order, config=config)
        assert [r.row_id for r in report.kept] == ["v"]
        [cluster] = report.exact_content_clusters
        assert cluster.kept_unit_key is not None and cluster.kept_unit_key.startswith(
            "openjev.policy-val:"
        )


def test_a_scoped_copy_outranking_the_searched_owner_of_its_text_is_refused_under_v6() -> None:
    """A searched train row owns its text; a scoped val row with the same text would be dropped
    in its favour (v5) or would need the owner dropped after the MinHash path decided it (v6).
    v6 refuses rather than choose either silently."""
    text = f"{PROSE} {_facts(5)}"
    rows = [_row("owner", repo="scene/train", text=text, policy=None, pinned="train"),
            _row("scoped", repo="scene/val", text=text, pinned="val")]
    assert [r.row_id for r in dedupe(rows, config=DataConfig()).kept] == ["owner"]
    with pytest.raises(ValueError, match="already decided"):
        dedupe(rows, config=DataConfig().with_v6_dedupe_rules())
    lower = [dataclasses.replace(rows[0], metadata={PINNED_SPLIT_KEY: "val"}), rows[1]]
    assert [r.row_id for r in dedupe(lower, config=DataConfig().with_v6_dedupe_rules()).kept] == [
        "owner"
    ], "an owner at least as protected keeps its text, as in v5"


def test_which_identical_row_survives_does_not_depend_on_input_order() -> None:
    text = f"{PROSE} {_facts(3)}"
    rows = [_row(f"x{i}", repo=f"scene/{i}", text=text) for i in range(4)]
    kept = [dedupe(list(o), config=DataConfig()).kept for o in (rows, rows[::-1])]
    assert [r.row_id for r in kept[0]] == [r.row_id for r in kept[1]]
    assert len(kept[0]) == 1


def test_marked_rows_never_enter_the_candidate_search() -> None:
    """The bound the pool hit: 40 shared-prose rows overflow a bound of 3 pairs when banded."""
    rows = _distinct_problems(40)
    unmarked = dedupe(
        _distinct_problems(40, policy=None), config=DataConfig(), max_candidate_pairs=3
    )
    assert isinstance(unmarked.status, NotRun), "characterization: banded, they overflow"
    report = dedupe(rows, config=DataConfig(), max_candidate_pairs=3)
    assert isinstance(report.status, Ran)
    assert report.n_candidate_pairs == 0
    assert len(report.kept) == 40


def test_an_unknown_policy_is_refused_naming_the_row() -> None:
    rows = _distinct_problems(2)
    bad = dataclasses.replace(rows[1], metadata={NEAR_DUPLICATE_POLICY_KEY: "fuzzy"})
    with pytest.raises(ValueError, match=r"r1.*fuzzy"):
        dedupe([rows[0], bad], config=DataConfig())


def test_the_report_names_the_scoped_families_with_both_row_counts() -> None:
    marked = _distinct_problems(3)
    plain = [
        _row(f"c{i}", repo=f"repo/{i}", text=f"unrelated text number {i} " * 8, family="code.x",
             policy=None)
        for i in range(2)
    ]
    report = dedupe(marked + plain, config=DataConfig())
    body = report.to_json()
    scope = body["near_duplicate_scope"]
    assert scope["exact_content_rows_by_family"] == {"openjev.policy": 3}
    assert scope["minhash_rows"] == 2
    assert scope["ruling"].endswith("dedupe-probe/RULING.md")
    assert isinstance(report.status, Ran)
    assert "3 rows" in report.status.detail and "openjev.policy" in report.status.detail


def test_a_corpus_with_no_marked_row_reports_exactly_as_before() -> None:
    """No key appears for an unscoped corpus, so its manifests stay byte-identical."""
    body = dedupe(_distinct_problems(4, policy=None), config=DataConfig()).to_json()
    assert "near_duplicate_scope" not in body
    assert "exact_content_clusters" not in body


# -- the split's independent re-derivation ------------------------------------------------------


def _split_rows() -> list[DataRow]:
    """Marked problems on both sides of the split, plus unmarked rows on both sides."""
    rows = [
        _row(f"t{i}", repo=f"scene/t{i}", text=f"{PROSE} {_facts(i)}", pinned="train")
        for i in range(4)
    ] + [
        _row(f"v{i}", repo=f"scene/v{i}", text=f"{PROSE} {_facts(100 + i)}", pinned="val")
        for i in range(2)
    ]
    rows += [
        _row(f"p{i}", repo=f"repo/p{i}", text=f"plain row {i} about something else " * 6,
             family="code.x", policy=None, pinned="train" if i % 2 else "val")
        for i in range(4)
    ]
    return rows


def test_split_near_duplicate_search_covers_only_unmarked_rows_and_says_so() -> None:
    config = DataConfig()
    report = split(dedupe(_split_rows(), config=config), config=config)
    nd = report.near_duplicate_disjoint
    assert isinstance(nd, Ran)
    assert nd.passed, "distinct problems sharing prose across the split are not a leak"
    assert (nd.n, nd.n_total) == (4, 10)
    assert "openjev.policy" in nd.detail
    assert isinstance(report.status, Ran) and report.status.passed


def test_split_content_digest_check_catches_identical_content_across_splits() -> None:
    """Independent of dedupe: a dedupe that let a duplicate through is caught here."""
    config = DataConfig()
    rows = _split_rows()
    leaked = _row("leak", repo="scene/leak", text=rows[0].dedupe_text, pinned="val")
    good = dedupe(rows, config=config)
    broken = dataclasses.replace(good, kept=(*good.kept, leaked))
    report = split(broken, config=config)
    check = report.exact_content_disjoint
    assert isinstance(check, Ran)
    assert not check.passed
    assert "leak" in check.detail or "t0" in check.detail
    assert isinstance(report.status, Ran) and not report.status.passed


def test_split_of_an_unscoped_corpus_reports_exactly_as_before() -> None:
    config = DataConfig()
    rows = [r for r in _split_rows() if NEAR_DUPLICATE_POLICY_KEY not in r.metadata]
    body = split(dedupe(rows, config=config), config=config).to_json()
    assert "exact_content_disjoint" not in body


# -- the candidate bound: one value, from the config, named when raised ---------------------------


def test_dedupe_and_split_read_the_configs_bound() -> None:
    """A build raises the bound once, in its config; neither search keeps a literal of its own."""
    tight = dataclasses.replace(DataConfig(), max_candidate_pairs=3)
    rows = _distinct_problems(40, policy=None)
    assert isinstance(dedupe(rows, config=tight).status, NotRun)
    assert isinstance(dedupe(rows, config=DataConfig()).status, Ran)

    # The split's re-derivation over rows dedupe kept: unmarked, mutually near (J ~0.95), all in
    # one repo so dedupe keeps them as within-repo pairs, pinned to both sides so the split's
    # search has pairs that cross. Only whether that search completed is asserted here.
    near = [
        _row(f"n{i}", repo="side/x", text=f"{PROSE} {_facts(i)}", policy=None,
             pinned="train" if i % 2 else "val")
        for i in range(12)
    ]
    kept = dedupe(near, config=DataConfig())
    assert len(kept.kept) == 12, "one repo: within-repo pairs are kept"
    assert isinstance(split(kept, config=DataConfig()).near_duplicate_disjoint, Ran)
    assert isinstance(split(kept, config=tight).near_duplicate_disjoint, NotRun)


def test_a_raised_bound_is_named_in_both_reports_and_the_default_is_not() -> None:
    raised = dataclasses.replace(DataConfig(), max_candidate_pairs=7_000_000)
    rows = _split_rows()
    d_raised = dedupe(rows, config=raised)
    s_raised = split(d_raised, config=raised)
    assert d_raised.to_json()["max_candidate_pairs"] == 7_000_000
    assert s_raised.to_json()["max_candidate_pairs"] == 7_000_000
    d_default = dedupe(rows, config=DataConfig())
    assert "max_candidate_pairs" not in d_default.to_json()
    assert "max_candidate_pairs" not in split(d_default, config=DataConfig()).to_json()


def test_a_bound_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="max_candidate_pairs"):
        dataclasses.replace(DataConfig(), max_candidate_pairs=0)
    with pytest.raises(ValueError, match="max_candidate_pairs"):
        dedupe(_distinct_problems(2), config=DataConfig(), max_candidate_pairs=0)
