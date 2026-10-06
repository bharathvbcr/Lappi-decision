"""Near-duplicate removal, run across pool **and** held-out **before** the split.

``docs/hardening.md`` section 2, in full, because both the rule and its stated
reason constrain the implementation:

    **Repo-level split, never row-level.** Keep the repo name on every row from the
    pull onward. **Repo-level is not sufficient by itself.** Vendored trees, forks
    and copied files put the same code in two repos. MinHash dedupe at 0.8 Jaccard
    runs **across pool and held-out before splitting**.

Running it after the split would be a different and much weaker check: it would
find duplicates *within* train and *within* held-out and leave every cross-split
pair -- the only ones that leak -- in place. :func:`dedupe` therefore takes one
undifferentiated collection of rows and has no notion of splits at all; the
splitter runs on its output.

Two things about *what* is compared were wrong in the obvious implementation and
are worth stating, because both were found by running it:

**The unit is a content unit, not an example.** One ``bigcode/commitpackft`` row
produces three examples -- commit intent, language id, change scope -- from the same
file. Comparing examples makes those three exact duplicates of one another, and a
naive dedupe deletes two thirds of the mixture while reporting a clean pass. A
content unit is ``(identity_key, digest(dedupe_text))``; every example sharing one
is one node, and a dropped unit takes all its examples with it.

**Only *cross-repo* duplicates are dropped.** The hazard named above is a file that
appears in two repositories, because that is the one a repo-level split cannot
separate. Two near-identical rows inside one repository always land on the same side
of the split, so dropping them removes data without removing any leak -- and for
``rajpurkar/squad_v2`` it removes almost everything, since several questions share
one passage and sit at Jaccard ~0.85 by construction. Within-repo near-duplicates
are therefore **counted and reported**, never silently deleted; the count is in the
manifest so the redundancy is visible rather than assumed away.

**Which of a cross-repo pair survives is deterministic**, and ``config.dedupe_keep_rule``
says how. v5's rule (``lexical``, the default) keeps the lexicographically smallest unit key
in each connected component, whatever its split -- so a train copy whose key sorts first
removes its val twin, and no later exclusion can bring that twin back
(GAP-QD-DATA-DEDUPE-KEEP-RULE-LEXICAL-SPLIT-BLIND-2026-10-03). v6's rule (``split_priority``)
keeps the unit its split protects most -- held-out, then val, then train -- and only among
equals the smallest key. A unit's split is the most protected split of any of its rows
(``qd_data.split.planned_split``): one file can carry a held-out-family example beside
trained ones, and then the unit is held-out evidence. Either way the survivor is a pure
function of the component; "whichever came first" would depend on iteration order and would
move ``data_snapshot_hash`` between runs over the same corpus.

**Every stage reports a tri-state.** A candidate-pair set that hit its bound is
``NotRun`` with the bound in the reason, never a clean dedupe over a partial pair
list. That is the difference between "no near-duplicates" and "we stopped looking".

**Structured decision rows are deduped by exact content, not by MinHash** (Fable's
ruling, ``AUDIT/finalize-2026-10-03/dedupe-probe/RULING.md``). A row whose
``metadata[NEAR_DUPLICATE_POLICY_KEY]`` is :data:`EXACT_CONTENT` -- the structured Open-Jev
families -- shares ~2 KB of rule prose with its neighbours and carries its own facts as
compact JSON that whitespace shingles barely see, so MinHash at 0.8 paired *distinct
problems* (31-77% of the pairs it proposed had different gold answers) and its candidate
search overflowed its bound. Such a row is a duplicate only of a row with the same
``dedupe_text`` digest, in any repo, and it never enters the candidate search. The
report says so -- the rows searched, the rows scoped out by family, and the ruling -- so
a search over part of the corpus never reads as a search over all of it. A corpus with
no such row takes exactly the path above and reports exactly what it reported before.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

from qd_train.tristate import NotRun, Ran, TriState

from .config import (
    DEDUPE_KEEP_LEXICAL,
    DEDUPE_KEEP_SPLIT_PRIORITY,
    DEFAULT_MAX_CANDIDATE_PAIRS,
    SPLITS,
    DataConfig,
)
from .minhash import (
    BandConfig,
    MinHasher,
    candidate_pairs,
    choose_bands,
    exact_jaccard,
    shingle,
)
from .rows import DataRow

__all__ = [
    "DEFAULT_MAX_CANDIDATE_PAIRS",
    "EXACT_CONTENT",
    "NEAR_DUPLICATE_POLICY_KEY",
    "NEAR_DUPLICATE_RULING",
    "ContentUnit",
    "DedupeReport",
    "DuplicateCluster",
    "ExactContentCluster",
    "band_config_for",
    "content_unit_key",
    "dedupe",
    "lsh_prefilter_note",
    "near_duplicate_policy",
    "text_digest",
    "unit_split_ranks",
]

#: Row metadata naming how a row's duplicates are found. Absent: MinHash at
#: ``config.dedupe_threshold`` (the module docstring). Set by the rewriter of a source whose
#: family the ruling below scoped out, never by a caller after the fact.
NEAR_DUPLICATE_POLICY_KEY: Final[str] = "near_duplicate_policy"
#: The one policy value: the row is a duplicate only of a row with the same ``dedupe_text``
#: digest, in any repo, and it never enters the MinHash candidate search.
EXACT_CONTENT: Final[str] = "exact_content"
#: Why a row may carry :data:`EXACT_CONTENT`. Named in every report that scoped a row, so
#: the part of the corpus the near-duplicate search did not cover carries its reason.
NEAR_DUPLICATE_RULING: Final[str] = "AUDIT/finalize-2026-10-03/dedupe-probe/RULING.md"
_POLICIES: Final[frozenset[str]] = frozenset({EXACT_CONTENT})


def near_duplicate_policy(row: DataRow) -> str | None:
    """``row``'s policy: ``None`` (MinHash) or :data:`EXACT_CONTENT`.

    Any other value is refused: a misspelt policy read as "absent" would put a row the
    ruling scoped out back into the search, and one read as "exact" would scope out a
    row nobody ruled on.
    """
    value = row.metadata.get(NEAR_DUPLICATE_POLICY_KEY)
    if value is not None and value not in _POLICIES:
        raise ValueError(
            f"{row.row_id}: metadata[{NEAR_DUPLICATE_POLICY_KEY!r}] is {value!r}; the only "
            f"policy is {EXACT_CONTENT!r} ({NEAR_DUPLICATE_RULING}), or the key is absent"
        )
    return value


def band_config_for(config: DataConfig) -> BandConfig:
    """The banding dedupe and the split's re-derivation both search with: ``choose_bands`` at
    the config's threshold, carrying its LSH agreement prefilter when one is set."""
    banding = choose_bands(num_perm=config.num_perm, threshold=config.dedupe_threshold)
    if not config.lsh_min_agreement_permille:
        return banding
    return replace(banding, min_agreement_permille=config.lsh_min_agreement_permille)


def lsh_prefilter_note(band_config: BandConfig) -> str:
    """The clause a report adds when the search ran under the prefilter; empty without one, so
    a v5 report reads as it did."""
    if not band_config.min_agreement_permille:
        return ""
    return (
        f"; LSH prefilter: a banded pair was a candidate only when its signatures agreed on >= "
        f"{band_config.min_agreement_permille} per mille of {band_config.num_perm_used} "
        f"positions, which a pair at the threshold fails with probability "
        f"{band_config.prefilter_miss_at(band_config.threshold):.2e}"
    )


def text_digest(text: str) -> str:
    """The digest of compared text that :func:`content_unit_key` and the exact-content
    policy both key on."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def content_unit_key(row: DataRow) -> str:
    """``identity_key`` plus a digest of the compared text.

    Both halves are load-bearing. Without ``identity_key``, two files with identical
    boilerplate in one repo merge. Without the digest, two commits touching the same
    path merge into one unit and one of their texts is silently discarded.
    """
    return f"{row.identity_key}|{text_digest(row.dedupe_text)}"


@dataclass(frozen=True, slots=True)
class ContentUnit:
    """One distinct piece of compared content, and every example built from it."""

    key: str
    repo_key: str
    text: str
    row_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DuplicateCluster:
    """One connected component of the cross-repo near-duplicate graph."""

    kept_unit_key: str
    dropped_unit_keys: tuple[str, ...]
    #: Repos the component spans. Always more than one: same-repo edges are not
    #: unioned, so a single-repo component cannot exist here.
    repo_keys: tuple[str, ...]
    #: The lowest exact Jaccard on any confirmed edge in this component, so a cluster
    #: held together by a chain of 0.81 pairs is visible as such.
    min_edge_jaccard: float
    n_rows_dropped: int

    def to_json(self) -> dict[str, Any]:
        return {
            "kept_unit_key": self.kept_unit_key,
            "dropped_unit_keys": list(self.dropped_unit_keys),
            "repo_keys": list(self.repo_keys),
            "min_edge_jaccard": round(self.min_edge_jaccard, 6),
            "n_rows_dropped": self.n_rows_dropped,
        }


@dataclass(frozen=True, slots=True)
class ExactContentCluster:
    """Units of :data:`EXACT_CONTENT` rows with one ``dedupe_text`` digest: one problem.

    Unlike a :class:`DuplicateCluster` it may sit in one repo -- identical content is one
    problem wherever it sits -- and it may hold a MinHash unit (``minhash_unit_key``): an
    exact row whose text equals a searched row's is dropped in that row's favour, so the
    one owner of a searched text stays the MinHash path.
    """

    digest: str
    kept_unit_key: str | None
    minhash_unit_key: str | None
    dropped_unit_keys: tuple[str, ...]
    repo_keys: tuple[str, ...]
    n_rows_dropped: int

    def to_json(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "kept_unit_key": self.kept_unit_key,
            "minhash_unit_key": self.minhash_unit_key,
            "dropped_unit_keys": list(self.dropped_unit_keys),
            "repo_keys": list(self.repo_keys),
            "n_rows_dropped": self.n_rows_dropped,
        }


@dataclass(frozen=True, slots=True)
class DedupeReport:
    """What dedupe did, in enough detail to be re-derived."""

    kept: tuple[DataRow, ...]
    clusters: tuple[DuplicateCluster, ...]
    n_input_rows: int
    n_input_units: int
    n_dropped_rows: int
    n_dropped_units: int
    #: Confirmed near-duplicate pairs whose two units sit in different repos. This is
    #: the population a repo-level split alone cannot separate.
    n_cross_repo_pairs: int
    #: Confirmed pairs inside one repo. **Reported, not dropped** -- see the module
    #: docstring. A large number here is redundancy, not leakage.
    n_within_repo_pairs: int
    #: Units whose text exceeded the shingling bound and were compared on a prefix.
    n_truncated_for_shingling: int
    band_config: BandConfig
    threshold: float
    n_candidate_pairs: int
    #: ``Ran`` only when the candidate search completed within its bound.
    status: TriState
    #: Exactly which rows were removed. Carried rather than re-derived, so a caller
    #: auditing the drop does not have to reproduce the clustering to find out.
    dropped_row_ids: frozenset[str] = frozenset()
    #: Input rows the ruling scoped out of the MinHash search, by family, sorted. Empty for
    #: a corpus with no :data:`EXACT_CONTENT` row, which then reports exactly as before.
    exact_content_rows_by_family: tuple[tuple[str, int], ...] = ()
    exact_content_clusters: tuple[ExactContentCluster, ...] = ()
    #: The candidate-pair bound this search ran under. Reported only when it is not
    #: :data:`DEFAULT_MAX_CANDIDATE_PAIRS`, so a pass under a raised bound is distinguishable
    #: from a pass under the default and an unraised build's manifest is unchanged.
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS
    #: ``config.dedupe_keep_rule``. Reported only when it is not v5's, as the bound is.
    keep_rule: str = DEDUPE_KEEP_LEXICAL

    @property
    def n_exact_content_rows(self) -> int:
        return sum(n for _, n in self.exact_content_rows_by_family)

    @property
    def n_possible_unit_pairs(self) -> int:
        """``C(n_input_units, 2)`` -- every pair that could have been a duplicate.

        The denominator the candidate count needs. ``n_candidate_pairs`` alone reads
        as a finding; against this it reads as what it is, a filtered subset.
        """
        n = self.n_input_units
        return n * (n - 1) // 2

    @property
    def lsh_recall_at_threshold(self) -> float:
        """The banding's per-pair recall at :attr:`threshold`.

        ``GAP-DATA-LSH-RECALL-BOUND``: the report used to carry ``bands`` and
        ``rows_per_band`` and stop there, which told a reader the cause of the
        recall loss without telling them its size, and left "no near-duplicates
        found" reading as "no near-duplicates exist". Derived from the banding
        rather than stored, so it cannot disagree with the configuration that
        produced it.
        """
        return self.band_config.recall_at_threshold

    def to_json(self) -> dict[str, Any]:
        body = self._base_json()
        if self.max_candidate_pairs != DEFAULT_MAX_CANDIDATE_PAIRS:
            body["max_candidate_pairs"] = self.max_candidate_pairs
        if self.keep_rule != DEDUPE_KEEP_LEXICAL:
            body["dedupe_keep_rule"] = self.keep_rule
        if self.band_config.min_agreement_permille:
            body["lsh_min_agreement_permille"] = self.band_config.min_agreement_permille
        if self.exact_content_rows_by_family:
            # Only when a row was scoped out, so an unscoped corpus's manifest is unchanged.
            body["near_duplicate_scope"] = {
                "ruling": NEAR_DUPLICATE_RULING,
                "minhash_rows": self.n_input_rows - self.n_exact_content_rows,
                "exact_content_rows_by_family": dict(self.exact_content_rows_by_family),
            }
            body["exact_content_clusters"] = [c.to_json() for c in self.exact_content_clusters]
        return body

    def _base_json(self) -> dict[str, Any]:
        return {
            "n_input_rows": self.n_input_rows,
            "n_input_units": self.n_input_units,
            "n_kept_rows": len(self.kept),
            "n_dropped_rows": self.n_dropped_rows,
            "n_dropped_units": self.n_dropped_units,
            "n_cross_repo_pairs": self.n_cross_repo_pairs,
            "n_within_repo_pairs": self.n_within_repo_pairs,
            "n_truncated_for_shingling": self.n_truncated_for_shingling,
            "threshold": self.threshold,
            "bands": self.band_config.bands,
            "rows_per_band": self.band_config.rows,
            "n_candidate_pairs": self.n_candidate_pairs,
            # Both numbers. A candidate count on its own reads as coverage.
            "n_possible_unit_pairs": self.n_possible_unit_pairs,
            "lsh_per_pair_recall_at_threshold": round(self.lsh_recall_at_threshold, 6),
            "clusters": [c.to_json() for c in self.clusters],
            "status": self.status.to_json(),
        }


class _UnionFind:
    __slots__ = ("_parent",)

    def __init__(self, keys: list[str]) -> None:
        self._parent: dict[str, str | None] = dict.fromkeys(keys)

    def find(self, x: str) -> str:
        root = x
        while (p := self._parent[root]) is not None:
            root = p
        while (p := self._parent[x]) is not None:
            self._parent[x] = root
            x = p
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Deterministic: the lexicographically smaller key always becomes the root, so a
        # component's root (and the order clusters are reported in) does not depend on the
        # order edges arrived in. Which member survives is the keep rule's, not the root's.
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        self._parent[hi] = lo


def unit_split_ranks(
    rows: Sequence[DataRow], units: dict[str, ContentUnit], *, config: DataConfig
) -> dict[str, int]:
    """Each unit's split rank for the v6 keep rule: the index in :data:`SPLITS` (train 0, val
    1, heldout 2) of the most protected split any of its rows would be assigned
    (``qd_data.split.planned_split``). Rows of one unit share a repo and an identity, so they
    must share a repo split; a unit whose rows disagree is refused rather than ranked."""
    # qd_data.split imports this module (it splits a DedupeReport), so the rule's one owner is
    # imported where it is used rather than at the top.
    from .split import planned_split, repo_split_of

    by_id = {r.row_id: r for r in rows}
    ranks: dict[str, int] = {}
    for key, unit in units.items():
        members = [by_id[rid] for rid in unit.row_ids]
        repo_splits = {repo_split_of(m, config=config) for m in members}
        if len(repo_splits) > 1:
            raise ValueError(
                f"content unit {key!r} holds rows in repo splits {sorted(repo_splits)}: one "
                "text in one repo is on one side of the split, so a rewriter pinned its rows "
                "inconsistently"
            )
        ranks[key] = max(SPLITS.index(planned_split(m, config=config)) for m in members)
    return ranks


def _keeper(members: list[str], ranks: dict[str, int] | None) -> str:
    """The surviving key of ``members`` (sorted): the smallest (v5), or under the v6 rule the
    most protected split's smallest."""
    if ranks is None:
        return members[0]
    return min(members, key=lambda k: (-ranks[k], k))


def _build_units(rows: tuple[DataRow, ...]) -> dict[str, ContentUnit]:
    grouped: dict[str, list[DataRow]] = {}
    for row in rows:
        grouped.setdefault(content_unit_key(row), []).append(row)
    units: dict[str, ContentUnit] = {}
    for key, members in grouped.items():
        repos = {m.repo_key for m in members}
        if len(repos) > 1:
            raise ValueError(
                f"content unit {key!r} spans repos {sorted(repos)}: the unit key is built "
                "from identity_key, which begins with the repo, so this means a rewriter "
                "emitted an identity_key that does not identify its repo"
            )
        units[key] = ContentUnit(
            key=key,
            repo_key=members[0].repo_key,
            text=members[0].dedupe_text,
            row_ids=tuple(sorted(m.row_id for m in members)),
        )
    return units


def dedupe(
    rows: list[DataRow] | tuple[DataRow, ...],
    *,
    config: DataConfig,
    max_candidate_pairs: int | None = None,
) -> DedupeReport:
    """MinHash-LSH near-duplicate removal at ``config.dedupe_threshold``.

    Takes **all** rows -- pool and held-out together -- and returns the survivors.
    There is deliberately no split argument: see the module docstring. The candidate
    bound is ``config.max_candidate_pairs`` unless ``max_candidate_pairs`` names one.
    """
    bound = config.max_candidate_pairs if max_candidate_pairs is None else max_candidate_pairs
    if bound < 1:
        raise ValueError(f"max_candidate_pairs must be >= 1, got {bound}")
    rows = tuple(rows)
    ids = [r.row_id for r in rows]
    if len(set(ids)) != len(ids):
        dupes = sorted(i for i, c in Counter(ids).items() if c > 1)
        raise ValueError(
            f"duplicate row_id(s) {dupes[:5]} ({len(dupes)} total): row ids key every "
            "downstream map, so a collision would silently drop a row rather than dedupe it"
        )

    band_config = band_config_for(config)

    if not rows:
        return DedupeReport(
            kept=(), clusters=(), n_input_rows=0, n_input_units=0, n_dropped_rows=0,
            n_dropped_units=0, n_cross_repo_pairs=0, n_within_repo_pairs=0,
            n_truncated_for_shingling=0, band_config=band_config,
            threshold=config.dedupe_threshold, n_candidate_pairs=0,
            status=NotRun(
                reason="dedupe received zero rows; nothing was compared, so nothing was cleared"
            ),
            max_candidate_pairs=bound,
            keep_rule=config.dedupe_keep_rule,
        )

    policies = {r.row_id: near_duplicate_policy(r) for r in rows}
    units = _build_units(rows)
    exact_keys: set[str] = set()
    for key, unit in units.items():
        unit_policies = {policies[rid] for rid in unit.row_ids}
        if len(unit_policies) > 1:
            raise ValueError(
                f"content unit {key!r} holds rows with policies "
                f"{sorted(str(p) for p in unit_policies)}: one text is either searched for "
                "near-duplicates or scoped out, never both"
            )
        if unit_policies == {EXACT_CONTENT}:
            exact_keys.add(key)
    # The MinHash path, over the units the ruling did not scope out. With no scoped unit this
    # is every unit, and everything below reads exactly as it did before the ruling.
    searched = {k: u for k, u in units.items() if k not in exact_keys}
    ranks = (
        unit_split_ranks(rows, units, config=config)
        if config.dedupe_keep_rule == DEDUPE_KEEP_SPLIT_PRIORITY
        else None
    )

    hasher = MinHasher(num_perm=config.num_perm, seed=config.seed)
    shingled: dict[str, frozenset[bytes]] = {}
    signatures: dict[str, tuple[int, ...]] = {}
    n_truncated = 0
    for key, unit in searched.items():
        sh = shingle(unit.text, k=config.shingle_size)
        if not sh.shingles:
            # DataRow.__post_init__ refuses empty dedupe_text, so this is unreachable
            # for a validly constructed row. Refuse loudly rather than skip silently.
            raise ValueError(
                f"content unit {key!r} produced no shingles from non-empty text; "
                "it would be invisible to dedupe"
            )
        n_truncated += int(sh.truncated)
        shingled[key] = sh.shingles
        signatures[key] = hasher.signature(sh.shingles)

    cands, truncated = candidate_pairs(signatures, config=band_config, max_pairs=bound)

    uf = _UnionFind(sorted(searched))
    edge_j: dict[tuple[str, str], float] = {}
    n_within = 0
    for a, b in sorted(cands):
        j = exact_jaccard(shingled[a], shingled[b])
        if j < config.dedupe_threshold:
            continue
        if units[a].repo_key == units[b].repo_key:
            # Reported, not merged: same repo means same side of the split, so this
            # pair cannot leak, and deleting it would thin the mixture for nothing.
            n_within += 1
            continue
        edge_j[(a, b)] = j
        uf.union(a, b)

    components: dict[str, list[str]] = {}
    for key in sorted(searched):
        components.setdefault(uf.find(key), []).append(key)

    clusters: list[DuplicateCluster] = []
    dropped_units: set[str] = set()
    dropped_rows: set[str] = set()
    for members in (sorted(v) for _, v in sorted(components.items())):
        if len(members) < 2:
            continue
        keep = _keeper(members, ranks)
        drop = tuple(k for k in members if k != keep)
        member_set = set(members)
        edges = [j for (a, b), j in edge_j.items() if a in member_set and b in member_set]
        rows_dropped = [rid for k in drop for rid in units[k].row_ids]
        clusters.append(
            DuplicateCluster(
                kept_unit_key=keep,
                dropped_unit_keys=drop,
                repo_keys=tuple(sorted({units[k].repo_key for k in members})),
                min_edge_jaccard=min(edges) if edges else 0.0,
                n_rows_dropped=len(rows_dropped),
            )
        )
        dropped_units.update(drop)
        dropped_rows.update(rows_dropped)

    # The MinHash path's own counts, before the exact path adds to the sets: its detail below
    # states what MinHash did, and the scope note states the rest.
    n_minhash_dropped_rows, n_minhash_dropped_units = len(dropped_rows), len(dropped_units)
    exact_clusters, exact_units_dropped, exact_rows_dropped = _exact_content_clusters(
        units, exact_keys, ranks
    )
    dropped_units.update(exact_units_dropped)
    dropped_rows.update(exact_rows_dropped)
    exact_by_family = tuple(
        sorted(Counter(r.family_id for r in rows if policies[r.row_id] == EXACT_CONTENT).items())
    )
    n_exact_rows = sum(n for _, n in exact_by_family)
    scope_note = (
        ""
        if not exact_by_family
        else (
            f"; scoped out by {NEAR_DUPLICATE_RULING}: {n_exact_rows} rows of "
            + ", ".join(f"{fam} ({n})" for fam, n in exact_by_family)
            + f" are deduped by exact content digest ({len(exact_rows_dropped)} dropped) and "
            f"were NOT searched for near-duplicates, so the search covered "
            f"{len(rows) - n_exact_rows} of {len(rows)} rows"
        )
    )
    scope_note += lsh_prefilter_note(band_config)
    if ranks is not None:
        scope_note += (
            "; keep rule split_priority: each component kept its held-out, else val, else "
            "train member, the smallest key among equals"
        )

    kept = tuple(r for r in rows if r.row_id not in dropped_rows)

    if truncated:
        status: TriState = NotRun(
            reason=(
                f"candidate-pair search hit its bound of {bound}; the pair "
                "list is partial, so the surviving rows are not a verified deduplicated set"
                + scope_note
            )
        )
    else:
        # Coverage is carried in the **pair** dimension, not the unit dimension.
        # Every unit was signed and banded, so "n units of n units" is true and is
        # also the wrong number: what a reader of a dedupe report wants to know is
        # how many of the pairs that could be duplicates were actually compared.
        # Banded LSH proposed `len(cands)` of `n_possible` and exact Jaccard
        # confirmed exactly those, so `is_complete_coverage` is False unless the
        # banding happened to propose everything -- which is the honest reading.
        n_units = len(searched)
        n_possible = n_units * (n_units - 1) // 2
        recall = band_config.recall_at_threshold
        status = Ran(
            passed=True,
            value=len(dropped_rows),
            n=min(len(cands), n_possible),
            n_total=n_possible,
            detail=(
                f"{n_minhash_dropped_rows} of {len(rows)} rows dropped across "
                f"{n_minhash_dropped_units} content units at Jaccard >= "
                f"{config.dedupe_threshold}; {len(edge_j)} confirmed pairs crossed a repo "
                f"boundary and would have survived a repo-level split alone; "
                f"{n_within} confirmed pairs were within one repo and were kept. "
                f"LSH is a probabilistic candidate generator, not an exhaustive "
                f"search: banding b={band_config.bands}, r={band_config.rows} proposes "
                f"a pair at J={config.dedupe_threshold} with probability "
                f"{recall:.4f}, so {len(cands)} of {n_possible} possible unit pairs "
                f"were compared exactly and the confirmed-pair counts are a lower "
                f"bound, not a complete enumeration"
                + scope_note
            ),
        )

    return DedupeReport(
        kept=kept,
        clusters=tuple(clusters),
        n_input_rows=len(rows),
        n_input_units=len(units),
        n_dropped_rows=len(dropped_rows),
        n_dropped_units=len(dropped_units),
        n_cross_repo_pairs=len(edge_j),
        n_within_repo_pairs=n_within,
        n_truncated_for_shingling=n_truncated,
        band_config=band_config,
        threshold=config.dedupe_threshold,
        n_candidate_pairs=len(cands),
        status=status,
        dropped_row_ids=frozenset(dropped_rows),
        max_candidate_pairs=bound,
        exact_content_rows_by_family=exact_by_family,
        exact_content_clusters=tuple(exact_clusters),
        keep_rule=config.dedupe_keep_rule,
    )


def _exact_content_clusters(
    units: dict[str, ContentUnit], exact_keys: set[str], ranks: dict[str, int] | None
) -> tuple[list[ExactContentCluster], set[str], set[str]]:
    """The exact-content path: scoped units grouped by the digest of their text alone.

    In each group the keep rule's unit survives (as in the MinHash path, so the survivor does
    not depend on input order) -- unless a searched unit has the same text, in which case every
    scoped unit of the group is dropped in its favour: the MinHash path stays the one owner of
    a text it searched. Under the v6 rule (``ranks``) that owner is the most protected searched
    unit of the text, and a scoped unit that outranks it is refused: keeping it would mean
    dropping a unit the MinHash path already decided, after the fact.
    """
    by_digest: dict[str, list[str]] = {}
    for key in sorted(exact_keys):
        by_digest.setdefault(text_digest(units[key].text), []).append(key)
    searched_by_digest: dict[str, list[str]] = {}
    for key in sorted(units):
        if key not in exact_keys:
            searched_by_digest.setdefault(text_digest(units[key].text), []).append(key)
    searched_owner = {d: _keeper(keys, ranks) for d, keys in searched_by_digest.items()}
    clusters: list[ExactContentCluster] = []
    dropped_units: set[str] = set()
    dropped_rows: set[str] = set()
    for digest, members in sorted(by_digest.items()):
        owner = searched_owner.get(digest)
        if owner is None and len(members) < 2:
            continue
        if owner is not None and ranks is not None:
            outranking = [m for m in members if ranks[m] > ranks[owner]]
            if outranking:
                raise ValueError(
                    f"exact-content unit {outranking[0]!r} (split rank {ranks[outranking[0]]}) "
                    f"has the text of searched unit {owner!r} (split rank {ranks[owner]}): the "
                    "v6 keep rule would keep the scoped unit and drop a unit the MinHash path "
                    "already decided, which this dedupe does not do; refused rather than "
                    "dropping the more protected copy"
                )
        keep = None if owner is not None else _keeper(members, ranks)
        drop = tuple(members) if owner is not None else tuple(m for m in members if m != keep)
        rows_dropped = [rid for k in drop for rid in units[k].row_ids]
        involved = [*members, *([owner] if owner is not None else [])]
        clusters.append(
            ExactContentCluster(
                digest=digest,
                kept_unit_key=keep,
                minhash_unit_key=owner,
                dropped_unit_keys=drop,
                repo_keys=tuple(sorted({units[k].repo_key for k in involved})),
                n_rows_dropped=len(rows_dropped),
            )
        )
        dropped_units.update(drop)
        dropped_rows.update(rows_dropped)
    return clusters, dropped_units, dropped_rows
