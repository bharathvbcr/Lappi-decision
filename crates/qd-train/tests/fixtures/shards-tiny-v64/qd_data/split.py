"""Repo-level splitting, and the leakage checks that prove it held.

``docs/hardening.md`` section 2 gives the rule and its limit in the same breath:
*"Repo-level split, never row-level"* and *"repo-level is not sufficient by
itself"*. This module implements the first and **refuses to run without** the
second: :func:`split` takes a :class:`~qd_data.dedupe.DedupeReport`, not a bare row
list, so it is not possible to split a corpus that was never deduped. A caller who
wants to skip dedupe has to construct a report saying dedupe did not run, and that
report's ``NotRun`` status propagates into the split's own status and from there
into the manifest.

**There are two holdouts and they are different mechanisms.** Conflating them is
the subtle error this module is written to avoid:

``repo_split`` -- the *natural held-out set*
    A three-way assignment over ``repo_key``. Content the model never saw. This is
    the boundary near-duplicates must not cross, and it is what the shipping gate's
    paired margin is measured on.

``family holdout`` -- the *two held-out task families*
    Rows whose ``family_id`` is held out go to the held-out shard **whatever their
    repo says**. What is withheld here is the *question shape*, not the content:
    ``docs/schema-api.md`` asks that these families go to ``noul``, which tests
    abstention on an unasked question. Their content may legitimately overlap
    training content -- a commitpackft row can feed a training family and a held-out
    family at once -- so applying the near-duplicate check across this boundary
    would flag the design as a defect.

So the near-duplicate and identity checks run over ``repo_split``, and a separate
check asserts no held-out family identifier appears in a training shard.

The repo assignment is a keyed hash, not a shuffle:

* it is a pure function of ``(repo_key, seed)``, so adding a repo to the corpus does
  not move any other repo across a boundary -- a shuffle-and-slice would reshuffle
  everything and silently invalidate every earlier comparison;
* it is stable across processes and platforms, for the reason
  ``qd_data.render.DeterministicRng`` gives.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Final

from qd_train.tristate import NotRun, Ran, TriState, aggregate

from .config import DEFAULT_MAX_CANDIDATE_PAIRS, SPLITS, DataConfig
from .dedupe import (
    EXACT_CONTENT,
    NEAR_DUPLICATE_RULING,
    DedupeReport,
    band_config_for,
    lsh_prefilter_note,
    near_duplicate_policy,
    text_digest,
)
from .minhash import MinHasher, candidate_pairs, exact_jaccard, shingle
from .rows import DataRow
from .sources import PINNED_SPLIT_KEY

__all__ = [
    "CONTENT_DISJOINT_FAMILIES",
    "HELD_OUT",
    "SQUAD_ANSWERABILITY_TITLE_FRACTION",
    "SQUAD_TITLE_FAMILIES",
    "TRAINING_SPLITS",
    "SplitAssignment",
    "SplitReport",
    "assign_repo",
    "content_disjoint_families",
    "planned_split",
    "repo_split_of",
    "split",
    "squad_title_family",
    "squad_title_repo_key",
]

HELD_OUT: Final[str] = "heldout"

#: The splits a training process may read. ``val`` is a training-time split: it
#: selects hyperparameters, so it is not held out in the sense rule 3 means.
TRAINING_SPLITS: Final[frozenset[str]] = frozenset({"train", "val"})


#: SQuAD v2 feeds two families: ``qa.answerability`` (held out) and ``qa.answer_span``
#: (trained). Over the same rows, a span gold is ``noul`` exactly when the question is
#: unanswerable, so training span on a question's context teaches the held-out family's
#: answer for that context (GAP-DATA-SQUAD-SPAN-NOUL-LEAKS-HELD-OUT-ANSWERABILITY). User
#: decision 2026-09-29, option (a): partition SQuAD by ARTICLE TITLE, so each family draws
#: from its own titles and no title, passage, question or identity feeds both.
SQUAD_TITLE_FAMILIES: Final[tuple[str, str]] = ("qa.answerability", "qa.answer_span")

#: Share of SQuAD titles that go to ``qa.answerability``. The held-out family only has to
#: measure abstention, so it takes the smaller share and the trained span family keeps most
#: titles. A choice, not a measurement; moving it re-partitions the held-out set, which
#: rule 2 permits only by a human decision.
SQUAD_ANSWERABILITY_TITLE_FRACTION: Final[float] = 0.2

#: Families that must share no content: no ``repo_key``, ``identity_key`` or context in
#: common. The general rule (module docstring) lets a held-out family overlap training
#: content; these are the exceptions where the overlap carries the held-out label.
CONTENT_DISJOINT_FAMILIES: Final[tuple[frozenset[str], ...]] = (
    frozenset(SQUAD_TITLE_FAMILIES),
)


def _unit_interval(material: str) -> float:
    digest = hashlib.blake2b(material.encode(), digest_size=8).digest()
    # Uniform in [0, 1) with 53 bits of resolution, the mantissa of a float64.
    return (int.from_bytes(digest, "big") >> 11) / float(1 << 53)


def squad_title_family(title: str, *, seed: int) -> str:
    """Which of :data:`SQUAD_TITLE_FAMILIES` may draw from this SQuAD article title.

    A keyed hash of the title, like :func:`assign_repo`: a pure function of
    ``(title, seed)``, so adding articles never moves an existing one between families.
    """
    if not title.strip():
        raise ValueError(
            "empty SQuAD title: the partition unit is the article, and a row without one "
            "cannot be kept out of the other family"
        )
    u = _unit_interval(f"qd_data.split.squad_title_family.v1|{seed}|{title}")
    return SQUAD_TITLE_FAMILIES[0] if u < SQUAD_ANSWERABILITY_TITLE_FRACTION else (
        SQUAD_TITLE_FAMILIES[1]
    )


def squad_title_repo_key(title: str) -> str:
    """The split unit of every row drawn from one SQuAD article: its ``repo_key``.

    The article, not the question: SQuAD asks many questions of one paragraph, and a
    row-level split would put questions about one passage on both sides. One spelling,
    used by ``qd_data.mixture.rewrite_squad`` and by the ``code.defect_class`` noul rows
    built from the same paragraphs, so the two land in one split and dedupe treats them as
    one repository's rows rather than as a cross-repo duplicate to drop.
    """
    return f"squad-title:{title}"


def assign_repo(
    repo_key: str, *, seed: int, train_fraction: float, val_fraction: float
) -> str:
    """Deterministically place one repo in ``train``, ``val`` or ``heldout``."""
    if not repo_key.strip():
        raise ValueError(
            "empty repo_key: a row with no split unit cannot be assigned, and defaulting "
            "it to its own group would put near-identical rows on both sides"
        )
    if not 0.0 < train_fraction < 1.0 or not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"fractions must be in (0, 1), got train={train_fraction}, val={val_fraction}"
        )
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            f"train ({train_fraction}) + val ({val_fraction}) leaves no held-out repos"
        )
    u = _unit_interval(f"qd_data.split.v1|{seed}|{repo_key}")
    if u < train_fraction:
        return "train"
    if u < train_fraction + val_fraction:
        return "val"
    return HELD_OUT


def repo_split_of(row: DataRow, *, config: DataConfig) -> str:
    """``row``'s content-boundary split: its pinned split, else its repo's hash.

    A source that pins splits (``Source.pinned_splits``, e.g. cais/mmlu: test and dev train,
    validation val) decides the content boundary by its upstream split, not by the repo hash --
    hashing MMLU validation subjects into train is the bug this closes. The pinned split is part
    of such a row's repo_key, so repo-disjointness still holds by construction; a value outside
    SPLITS is refused loudly, never defaulted.
    """
    pinned = row.metadata.get(PINNED_SPLIT_KEY)
    if pinned is not None:
        if pinned not in SPLITS:
            raise ValueError(
                f"{row.row_id}: metadata[{PINNED_SPLIT_KEY!r}] is {pinned!r}, which is not one "
                f"of {SPLITS}"
            )
        return pinned
    return assign_repo(
        row.repo_key,
        seed=config.seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
    )


def planned_split(row: DataRow, *, config: DataConfig) -> str:
    """The split :func:`split` gives ``row`` if it survives dedupe: its family's hold-out, else
    :func:`repo_split_of`. The one owner of that rule; ``split`` assigns with it, dedupe's v6
    keep rule ranks with it and ``qd_train.exclusions.split_before_dedupe`` delegates to it."""
    if config.is_held_out_family(row.family_id):
        return HELD_OUT
    return repo_split_of(row, config=config)


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """Where a row landed, and by which of the two mechanisms."""

    row_id: str
    #: The final shard the row is written to.
    split: str
    #: What the repo hash said, before any family override. The leakage checks use
    #: this, because it is the content boundary.
    repo_split: str
    repo_key: str
    family_id: str
    #: ``"repo_hash"`` or ``"family_holdout"``.
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "split": self.split,
            "repo_split": self.repo_split,
            "repo_key": self.repo_key,
            "family_id": self.family_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SplitReport:
    assignments: tuple[SplitAssignment, ...]
    rows_by_split: dict[str, tuple[DataRow, ...]]
    repo_disjoint: TriState
    identity_disjoint: TriState
    near_duplicate_disjoint: TriState
    held_out_families_absent_from_training: TriState
    dedupe_status: TriState
    #: :data:`CONTENT_DISJOINT_FAMILIES` share no repo_key, identity_key or context.
    content_disjoint_families: TriState
    #: No row the ruling scoped out of the near-duplicate search (``EXACT_CONTENT``) shares
    #: its content digest with a row on another side of the split: that ruling's leak
    #: definition. ``None`` when no row was scoped out, and then absent from every report.
    exact_content_disjoint: TriState | None = None
    #: The near-duplicate re-derivation's candidate bound; reported only when it is not
    #: :data:`~qd_data.config.DEFAULT_MAX_CANDIDATE_PAIRS`, as dedupe's is.
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS

    @property
    def status(self) -> TriState:
        """One tri-state over every post-condition, including dedupe's own.

        ``aggregate`` returns ``NotRun`` if any input did not run, so a split built on
        a dedupe pass that hit its bound cannot report clean.
        """
        return aggregate(
            {
                "dedupe": self.dedupe_status,
                "repo_disjoint": self.repo_disjoint,
                "identity_disjoint": self.identity_disjoint,
                "near_duplicate_disjoint": self.near_duplicate_disjoint,
                "held_out_families_absent_from_training": (
                    self.held_out_families_absent_from_training
                ),
                # Only when it ran: its NotRun means no group had two families present,
                # i.e. nothing that could share content -- not a check that failed to run.
                **(
                    {"content_disjoint_families": self.content_disjoint_families}
                    if isinstance(self.content_disjoint_families, Ran)
                    else {}
                ),
                **(
                    {"exact_content_disjoint": self.exact_content_disjoint}
                    if self.exact_content_disjoint is not None
                    else {}
                ),
            },
            name="split",
        )

    def counts(self) -> dict[str, int]:
        return {s: len(self.rows_by_split.get(s, ())) for s in SPLITS}

    def holdout_breakdown(self) -> dict[str, int]:
        """How the held-out shard was populated. Two populations, counted apart."""
        out = {"natural_repo_holdout": 0, "family_holdout": 0}
        for a in self.assignments:
            if a.split != HELD_OUT:
                continue
            key = "family_holdout" if a.reason == "family_holdout" else "natural_repo_holdout"
            out[key] += 1
        return out

    def to_json(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "counts": self.counts(),
            "holdout_breakdown": self.holdout_breakdown(),
            "repo_disjoint": self.repo_disjoint.to_json(),
            "identity_disjoint": self.identity_disjoint.to_json(),
            "near_duplicate_disjoint": self.near_duplicate_disjoint.to_json(),
            "held_out_families_absent_from_training": (
                self.held_out_families_absent_from_training.to_json()
            ),
            "dedupe_status": self.dedupe_status.to_json(),
            "content_disjoint_families": self.content_disjoint_families.to_json(),
        }
        if self.exact_content_disjoint is not None:
            body["exact_content_disjoint"] = self.exact_content_disjoint.to_json()
        if self.max_candidate_pairs != DEFAULT_MAX_CANDIDATE_PAIRS:
            body["max_candidate_pairs"] = self.max_candidate_pairs
        body["status"] = self.status.to_json()
        return body


def _key_disjointness(
    assignments: tuple[SplitAssignment, ...],
    keys: dict[str, str],
    *,
    label: str,
) -> TriState:
    """Assert one key never appears under two ``repo_split`` values."""
    if not assignments:
        return NotRun(reason=f"{label}: no rows were assigned, so disjointness was not tested")
    where: dict[str, set[str]] = {}
    for a in assignments:
        where.setdefault(keys[a.row_id], set()).add(a.repo_split)
    straddling = sorted(k for k, s in where.items() if len(s) > 1)
    return Ran(
        passed=not straddling,
        value=len(straddling),
        n=len(where),
        n_total=len(where),
        detail=(
            ""
            if not straddling
            else f"{label}: {len(straddling)} key(s) appear in more than one repo split, "
            f"first five: {straddling[:5]}"
        ),
    )


def split(
    report: DedupeReport,
    *,
    config: DataConfig,
    max_candidate_pairs: int | None = None,
) -> SplitReport:
    """Assign every surviving row to ``train``, ``val`` or ``heldout``.

    Takes a :class:`~qd_data.dedupe.DedupeReport` rather than rows, so the ordering
    "dedupe across everything, *then* split" is enforced by the signature. The
    near-duplicate re-derivation's candidate bound is ``config.max_candidate_pairs``
    unless ``max_candidate_pairs`` names one -- the same bound dedupe reads, so a build
    that raised it for dedupe cannot truncate here on the old literal.
    """
    bound = config.max_candidate_pairs if max_candidate_pairs is None else max_candidate_pairs
    if bound < 1:
        raise ValueError(f"max_candidate_pairs must be >= 1, got {bound}")
    rows = report.kept
    assignments: list[SplitAssignment] = []
    for row in rows:
        repo_split = repo_split_of(row, config=config)
        held_by_family = config.is_held_out_family(row.family_id)
        assignments.append(
            SplitAssignment(
                row_id=row.row_id,
                split=planned_split(row, config=config),
                repo_split=repo_split,
                repo_key=row.repo_key,
                family_id=row.family_id,
                reason="family_holdout" if held_by_family else "repo_hash",
            )
        )

    by_id = {r.row_id: r for r in rows}
    rows_by_split: dict[str, list[DataRow]] = {s: [] for s in SPLITS}
    for a in assignments:
        rows_by_split[a.split].append(by_id[a.row_id])

    repo_ok = _key_disjointness(
        tuple(assignments), {r.row_id: r.repo_key for r in rows}, label="repo_key"
    )
    identity_ok = _key_disjointness(
        tuple(assignments), {r.row_id: r.identity_key for r in rows}, label="identity_key"
    )
    nd_ok = _cross_split_near_duplicates(
        rows,
        repo_split_of={a.row_id: a.repo_split for a in assignments},
        config=config,
        max_candidate_pairs=bound,
    )
    fam_ok = _held_out_families_absent(assignments, config=config)

    return SplitReport(
        assignments=tuple(assignments),
        rows_by_split={s: tuple(v) for s, v in rows_by_split.items()},
        repo_disjoint=repo_ok,
        identity_disjoint=identity_ok,
        near_duplicate_disjoint=nd_ok,
        held_out_families_absent_from_training=fam_ok,
        dedupe_status=report.status,
        content_disjoint_families=content_disjoint_families(rows),
        exact_content_disjoint=_exact_content_disjoint(
            rows, split_of={a.row_id: a.split for a in assignments}
        ),
        max_candidate_pairs=bound,
    )


def _exact_content_disjoint(
    rows: tuple[DataRow, ...], *, split_of: dict[str, str]
) -> TriState | None:
    """The ruling's leak check for rows it scoped out of the near-duplicate search.

    For every ``EXACT_CONTENT`` row, every row (scoped or not) with the same ``dedupe_text``
    digest must sit on the same side of the split. Independent of dedupe, which should have
    left one row per digest: a dedupe that let a duplicate through is caught here rather than
    trusted. Exhaustive -- a digest map, no bound -- so it either finds a crossing or there is
    none. ``None`` when no row was scoped out.

    It reads each row's **final** split (``SplitAssignment.split``: train, val or heldout,
    family holdout included), where :func:`_cross_split_near_duplicates` reads the
    ``repo_split``. That is deliberate, not a mismatch: the ruling defines this leak as the
    same content on two sides of the split the model is trained and measured on, and a
    family-held-out row is on the held-out side whatever its repo hash says.
    """
    scoped = [r for r in rows if near_duplicate_policy(r) == EXACT_CONTENT]
    if not scoped:
        return None
    wanted = {text_digest(r.dedupe_text) for r in scoped}
    sides: dict[str, dict[str, list[str]]] = {}
    for r in rows:
        d = text_digest(r.dedupe_text)
        if d in wanted:
            sides.setdefault(d, {}).setdefault(split_of[r.row_id], []).append(r.row_id)
    crossing = sorted(
        (d, by_split) for d, by_split in sides.items() if len(by_split) > 1
    )
    return Ran(
        passed=not crossing,
        value=len(crossing),
        n=len(scoped),
        n_total=len(scoped),
        detail=(
            f"{len(scoped)} rows scoped out by {NEAR_DUPLICATE_RULING}; every content digest "
            "among them sits on one side of the split"
            if not crossing
            else f"{len(crossing)} content digest(s) of scoped rows span splits: "
            + "; ".join(
                ", ".join(f"{s}: {sorted(ids)[:3]}" for s, ids in sorted(by_split.items()))
                for _, by_split in crossing[:5]
            )
        ),
    )


def content_disjoint_families(rows: tuple[DataRow, ...] | list[DataRow]) -> TriState:
    """No :data:`CONTENT_DISJOINT_FAMILIES` group shares a repo_key, identity_key or context.

    ``NotRun`` when no group has rows from two or more of its families: disjointness
    between families that are not both present was not tested, and saying it passed would
    be the vacuous kind of pass.
    """
    tested = 0
    shared: list[str] = []
    for group in CONTENT_DISJOINT_FAMILIES:
        members = [r for r in rows if r.family_id in group]
        if len({r.family_id for r in members}) < 2:
            continue
        tested += len(members)
        for label, key in (
            ("repo_key", lambda r: r.repo_key),
            ("identity_key", lambda r: r.identity_key),
            ("context", lambda r: r.request.context),
        ):
            families_of: dict[object, set[str]] = {}
            for r in members:
                families_of.setdefault(key(r), set()).add(r.family_id)
            both = [k for k, f in families_of.items() if len(f) > 1]
            if both:
                shared.append(
                    f"{sorted(group)} share {len(both)} {label}(s), e.g. {str(both[0])[:80]!r}"
                )
    if not tested:
        return NotRun(
            reason=(
                "no content-disjoint family group had rows from two of its families, so "
                "their disjointness was not tested"
            )
        )
    return Ran(
        passed=not shared,
        value=len(shared),
        n=tested,
        n_total=tested,
        detail="; ".join(shared),
    )


def _held_out_families_absent(
    assignments: list[SplitAssignment], *, config: DataConfig
) -> TriState:
    """``docs/hardening.md`` section 2: *a test asserts their identifiers appear in
    no training shard.* Here it is an assertion the pipeline itself makes."""
    if not assignments:
        return NotRun(
            reason="no rows were assigned, so no shard was inspected for held-out families"
        )
    training = [a for a in assignments if a.split in TRAINING_SPLITS]
    if not training:
        return NotRun(
            reason=(
                "no rows landed in a training shard, so the absence of a held-out family "
                "from one was not tested"
            )
        )
    offenders = sorted({a.family_id for a in training if config.is_held_out_family(a.family_id)})
    return Ran(
        passed=not offenders,
        value=len(offenders),
        n=len(training),
        n_total=len(training),
        detail=(
            ""
            if not offenders
            else f"held-out families present in a training shard: {offenders}"
        ),
    )


def _cross_split_near_duplicates(
    rows: tuple[DataRow, ...],
    *,
    repo_split_of: dict[str, str],
    config: DataConfig,
    max_candidate_pairs: int,
) -> TriState:
    """Re-derive near-duplicate pairs and assert none crosses a **repo** boundary.

    An independent second derivation on purpose. Trusting dedupe's own bookkeeping
    would mean a bug in dedupe reports itself as a clean split -- the same argument
    ``docs/hardening.md`` section 1 makes for re-deriving mutation spans from a
    textual diff rather than from the mutator.

    Rows the ruling scoped out (``EXACT_CONTENT``) are not searched here either. The result
    then covers the searched rows of all rows (``n`` of ``n_total``) and names the scoped
    families, so it never reads as a search of the whole split; their leak check is
    :func:`_exact_content_disjoint`.
    """
    all_rows = rows
    scoped = Counter(r.family_id for r in rows if near_duplicate_policy(r) == EXACT_CONTENT)
    rows = tuple(r for r in rows if near_duplicate_policy(r) != EXACT_CONTENT)
    scope_note = (
        ""
        if not scoped
        else (
            f"; {sum(scoped.values())} rows of "
            + ", ".join(f"{fam} ({n})" for fam, n in sorted(scoped.items()))
            + f" were not searched: exact content by {NEAR_DUPLICATE_RULING}, checked by "
            "exact_content_disjoint instead"
        )
    )
    if len(rows) < 2:
        return NotRun(
            reason=(
                f"cross-split near-duplicate check needs at least 2 rows, got {len(rows)}; "
                "with fewer there is no pair to cross a boundary"
            )
        )
    if len({repo_split_of[r.row_id] for r in rows}) < 2:
        return NotRun(
            reason=(
                "every row landed in one repo split, so no pair could cross a boundary; "
                "this is not evidence that the split separates near-duplicates"
            )
        )
    hasher = MinHasher(num_perm=config.num_perm, seed=config.seed)
    band_config = band_config_for(config)
    sh = {r.row_id: shingle(r.dedupe_text, k=config.shingle_size).shingles for r in rows}
    sigs = {rid: hasher.signature(s) for rid, s in sh.items()}
    cands, truncated = candidate_pairs(sigs, config=band_config, max_pairs=max_candidate_pairs)
    if truncated:
        return NotRun(
            reason=(
                f"cross-split near-duplicate search hit its bound of {max_candidate_pairs} "
                "candidate pairs; the absence of a crossing pair was not established"
            )
        )
    crossing = [(a, b) for a, b in sorted(cands) if repo_split_of[a] != repo_split_of[b]]
    offenders = [
        (a, b, j)
        for a, b in crossing
        if (j := exact_jaccard(sh[a], sh[b])) >= config.dedupe_threshold
    ]
    # Coverage is the rows scanned, not the pairs found: reporting n=0/n_total=0 when
    # LSH proposes no candidate would make a complete scan of a clean corpus read
    # like a check that examined nothing.
    return Ran(
        passed=not offenders,
        value=len(offenders),
        n=len(rows),
        n_total=len(all_rows),
        detail=(
            f"scanned {len(rows)} rows; {len(cands)} candidate pairs, {len(crossing)} of "
            "them crossing a repo split, none confirmed at threshold "
            f"{config.dedupe_threshold}"
            if not offenders
            else "near-duplicate pairs span a repo split boundary: "
            + ", ".join(f"{a}~{b} (J={j:.3f})" for a, b, j in offenders[:5])
        )
        + lsh_prefilter_note(band_config)
        + scope_note,
    )
