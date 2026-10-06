"""The data config, and the assertions it refuses to load without.

One of these checks is the reason this module exists. ``docs/plan-corrections.md``:

    **The two held-out task families must be chosen from what actually survives**
    -- holding out a family that was never going to load is not a holdout.

That failure is completely silent. You name two families in a config, the pipeline
never sees them because their source was refused at load anyway, every downstream
report says "held-out families abstained", and the gate passes without ever having
tested anything. So :class:`DataConfig` refuses to construct unless each held-out
family is one that is *otherwise admitted* -- it has to be a family that would have
been trained on, or holding it out means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final

from .licences import DEFAULT_LICENCE_CONFIG, LicenceConfig
from .sources import TASK_FAMILIES, admitted_task_families

__all__ = [
    "DEDUPE_KEEP_LEXICAL",
    "DEDUPE_KEEP_RULES",
    "DEDUPE_KEEP_SPLIT_PRIORITY",
    "DEFAULT_HELD_OUT_FAMILIES",
    "DEFAULT_MAX_CANDIDATE_PAIRS",
    "N_HELD_OUT_FAMILIES",
    "POOL_MAX_CANDIDATE_PAIRS",
    "SPLITS",
    "V6_LSH_MIN_AGREEMENT_PERMILLE",
    "DataConfig",
    "Split",
]

Split = str
#: In ascending order of protection: the split-aware dedupe keep rule ranks a split by its index.
SPLITS: Final[tuple[str, ...]] = ("train", "val", "heldout")

#: v5's keep rule: the lexicographically smallest unit key of a near-duplicate component
#: survives, whatever its split, so a train copy whose key sorts first removes its val twin
#: (GAP-QD-DATA-DEDUPE-KEEP-RULE-LEXICAL-SPLIT-BLIND-2026-10-03; v5 worked around it with a
#: named pre-dedupe drop list). The default, so a v5 rebuild re-derives v5's rows and hashes.
DEDUPE_KEEP_LEXICAL: Final[str] = "lexical"
#: v6's keep rule: the unit its split protects most survives -- held-out, then val, then train
#: (:data:`SPLITS` order) -- and only among equals the smallest key.
DEDUPE_KEEP_SPLIT_PRIORITY: Final[str] = "split_priority"
DEDUPE_KEEP_RULES: Final[tuple[str, ...]] = (DEDUPE_KEEP_LEXICAL, DEDUPE_KEEP_SPLIT_PRIORITY)
#: v6's LSH prefilter: a banded pair is a candidate only if its banded signatures agree on at
#: least this many positions per mille (estimated Jaccard >= 0.65), so one shared band at J
#: ~0.3-0.6 no longer counts toward the candidate bound
#: (GAP-DEDUPE-LSH-BAND-CANDIDATES-NOT-DUPLICATES-2026-10-03). Per mille, so Python and
#: ``qd-prep lsh`` compare the same integers and cannot round a boundary pair differently.
V6_LSH_MIN_AGREEMENT_PERMILLE: Final[int] = 650

#: The near-duplicate searches' candidate-pair bound (dedupe and the split's re-derivation).
#: At 0.8 Jaccard on a code corpus the candidate set is a small multiple of the unit count, so
#: a set this large meant the corpus was pathological (for example every row identical) and
#: the run must say so rather than grind. Templated short-text decision rows broke that
#: premise: at the banding b=16, r=8 a pair at J=0.6 is a candidate with probability 0.24, and
#: the v5 pool's procedural and command families propose millions of such pairs while holding
#: few duplicates (AUDIT/finalize-2026-10-03/dedupe-probe/RULING.md). A build that needs more
#: sets ``DataConfig.max_candidate_pairs`` to a measured figure; reports name any bound that
#: is not this one.
DEFAULT_MAX_CANDIDATE_PAIRS: Final[int] = 5_000_000
#: The bound for a build that reads the v5 decision pool, measured rather than guessed (Fable,
#: 2026-10-03, RULING.md "(A)"). The pool's MinHash population -- every family but the four
#: structured Open-Jev ones, which the ruling dedupes by exact content -- proposed 5,874,260
#: candidate pairs in dedupe and 2,892,003 in the split's re-derivation (native path, ceiling
#: 50M, peak RSS 8.9 GB, 93 s; AUDIT/finalize-2026-10-03/dedupe-probe/residual-measure.json).
#: Twice the pool's count plus v4's whole-corpus 550,147 (a proxy for v5's non-pool rows, which
#: are unmeasured) is 12.3M; rounded up. A build that still exceeds it reports NotRun and
#: refuses, as at any bound.
POOL_MAX_CANDIDATE_PAIRS: Final[int] = 12_500_000

#: The plan fixes the count at two.
N_HELD_OUT_FAMILIES: Final[int] = 2

#: Both are ``choice`` families whose *source* still contributes another family to
#: training. That is deliberate: the holdout then tests generalisation to an unseen
#: **question shape** rather than to an unseen corpus, which is what
#: ``docs/schema-api.md`` is asking for when it says the held-out families must go
#: to ``noul``. Holding out the only ``score`` family or the only ``span`` family
#: instead would mean the model never saw that slot type at all, and its abstention
#: would prove nothing about the question.
DEFAULT_HELD_OUT_FAMILIES: Final[tuple[str, ...]] = ("code.language_id", "qa.answerability")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Everything the data lane needs, validated at construction."""

    seed: int = 20260919
    licence: LicenceConfig = DEFAULT_LICENCE_CONFIG
    held_out_families: tuple[str, ...] = DEFAULT_HELD_OUT_FAMILIES
    #: Filesystem roots that a training process must never read.
    held_out_roots: tuple[Path, ...] = (Path("data/heldout"),)
    #: Any path segment equal to one of these marks a path held out, wherever it
    #: sits. Convention is not the mechanism -- see qd_train.data_access -- but a
    #: belt alongside the explicit roots costs nothing.
    held_out_path_markers: tuple[str, ...] = ("heldout", "held_out", "held-out")

    dedupe_threshold: float = 0.8
    shingle_size: int = 5
    num_perm: int = 128

    #: Repo-level split fractions. They are stated separately and must sum to 1.0:
    #: a single ``train_fraction`` with "the rest" left implicit is how a three-way
    #: split quietly becomes a two-way one.
    train_fraction: float = 0.9
    val_fraction: float = 0.05

    #: Bounds. Every batch, payload and read is bounded; these are the data lane's.
    max_rows_per_source: int = 2_000_000
    max_row_bytes: int = 1_048_576
    #: The near-duplicate searches' candidate-pair bound (:data:`DEFAULT_MAX_CANDIDATE_PAIRS`).
    #: Not in :meth:`fingerprint`: a bound decides only whether a search completes, and a
    #: search that completes finds the same pairs under any bound, so it does not change the
    #: data. A search that hits it reports ``NotRun``, and the build refuses.
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS
    #: Which unit of a near-duplicate component survives (:data:`DEDUPE_KEEP_RULES`). In
    #: :meth:`fingerprint` only when it is not v5's, so a v5 config hashes as it always did.
    dedupe_keep_rule: str = DEDUPE_KEEP_LEXICAL
    #: The LSH agreement prefilter, per mille of the banded signature; 0 is v5's (none). In
    #: :meth:`fingerprint` only when set: it can drop a confirmed pair whose estimate fell under
    #: it, so unlike the bound it can change the data.
    lsh_min_agreement_permille: int = 0

    metadata: dict[str, str] = field(default_factory=dict)
    #: v6's benchmark re-pin (:meth:`with_v6_benchmark_targets`): the upstream evaluation splits
    #: in ``qd_data.sources.BENCHMARK_TARGET_SPLITS`` are never rows, only decontamination
    #: targets. False is v5's (MMLU test and dev train, CLINC test is split by intent); in
    #: :meth:`fingerprint` only when set, so a v5 config hashes as it always did.
    benchmark_eval_splits_are_targets: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError(f"seed must be int, got {type(self.seed).__name__}")

        fams = tuple(self.held_out_families)
        if len(fams) != N_HELD_OUT_FAMILIES:
            raise ValueError(
                f"the plan fixes the holdout at {N_HELD_OUT_FAMILIES} task families; "
                f"got {len(fams)}: {fams}"
            )
        if len(set(fams)) != len(fams):
            raise ValueError(f"held_out_families contains a duplicate: {fams}")

        unknown = [f for f in fams if f not in TASK_FAMILIES]
        if unknown:
            raise ValueError(
                f"held-out families {unknown} are not registered task families. "
                f"Known: {sorted(TASK_FAMILIES)}"
            )

        # The check this module exists for.
        admitted = {f.family_id for f in admitted_task_families(self.licence)}
        never_admitted = [f for f in fams if f not in admitted]
        if never_admitted:
            raise ValueError(
                f"held-out families {never_admitted} are not otherwise admitted, so holding "
                "them out tests nothing: their source is refused at load anyway (licence or "
                "reachability). A holdout must name a family that would otherwise have been "
                "trained on. Admitted families are: " + ", ".join(sorted(admitted))
            )

        remaining = admitted - set(fams)
        if not remaining:
            raise ValueError(
                f"holding out {fams} leaves no admitted task family to train on"
            )

        if not 0.0 < self.dedupe_threshold <= 1.0:
            raise ValueError(f"dedupe_threshold must be in (0, 1], got {self.dedupe_threshold}")
        if self.shingle_size < 1:
            raise ValueError(f"shingle_size must be >= 1, got {self.shingle_size}")
        if self.num_perm < 16 or self.num_perm % 2:
            raise ValueError(f"num_perm must be an even int >= 16, got {self.num_perm}")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError(f"train_fraction must be in (0, 1), got {self.train_fraction}")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError(
                f"train_fraction ({self.train_fraction}) + val_fraction "
                f"({self.val_fraction}) leaves no repos for the natural held-out set. "
                "A held-out set of size zero passes every leakage check vacuously."
            )
        if self.max_rows_per_source < 1 or self.max_row_bytes < 1:
            raise ValueError("row bounds must be positive")
        if self.max_candidate_pairs < 1:
            raise ValueError(
                f"max_candidate_pairs must be >= 1, got {self.max_candidate_pairs}: a bound "
                "of 0 makes every near-duplicate search report NotRun"
            )
        if self.dedupe_keep_rule not in DEDUPE_KEEP_RULES:
            raise ValueError(
                f"dedupe_keep_rule is {self.dedupe_keep_rule!r}; known: {DEDUPE_KEEP_RULES}"
            )
        permille = self.lsh_min_agreement_permille
        if not isinstance(permille, int) or isinstance(permille, bool) or not 0 <= permille <= 1000:
            raise ValueError(
                f"lsh_min_agreement_permille must be an int in [0, 1000], got {permille!r}"
            )
        if not self.held_out_roots:
            raise ValueError(
                "held_out_roots is empty: the training-time path check would then have "
                "nothing to refuse, which is the check silently disabled"
            )
        if not isinstance(self.benchmark_eval_splits_are_targets, bool):
            raise TypeError(
                "benchmark_eval_splits_are_targets must be bool, got "
                f"{type(self.benchmark_eval_splits_are_targets).__name__}"
            )

    @property
    def training_families(self) -> tuple[str, ...]:
        """Admitted families minus the holdout: exactly what may enter a shard."""
        admitted = {f.family_id for f in admitted_task_families(self.licence)}
        return tuple(sorted(admitted - set(self.held_out_families)))

    def is_held_out_family(self, family_id: str) -> bool:
        return family_id in set(self.held_out_families)

    def with_v6_dedupe_rules(self) -> DataConfig:
        """This config under v6's dedupe rules: the split-aware keep and the LSH prefilter."""
        return replace(
            self,
            dedupe_keep_rule=DEDUPE_KEEP_SPLIT_PRIORITY,
            lsh_min_agreement_permille=V6_LSH_MIN_AGREEMENT_PERMILLE,
        )

    def fingerprint(self) -> dict[str, object]:
        """The config fields that change the data. Feeds ``data_snapshot_hash``.

        The two dedupe rules are named only when they are not v5's, so every config that
        existed before them fingerprints byte for byte as it did."""
        body: dict[str, object] = {
            "seed": self.seed,
            "held_out_families": sorted(self.held_out_families),
            "dedupe_threshold": self.dedupe_threshold,
            "shingle_size": self.shingle_size,
            "num_perm": self.num_perm,
            "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction,
            "admitted_by_human": dict(sorted(self.licence.admitted_by_human.items())),
        }
        if self.dedupe_keep_rule != DEDUPE_KEEP_LEXICAL:
            body["dedupe_keep_rule"] = self.dedupe_keep_rule
        if self.lsh_min_agreement_permille:
            body["lsh_min_agreement_permille"] = self.lsh_min_agreement_permille
        if self.benchmark_eval_splits_are_targets:
            body["benchmark_eval_splits_are_targets"] = True
        return body

    def with_v6_benchmark_targets(self) -> DataConfig:
        """This config under v6's benchmark re-pin (data-clean plan section 3 item 6;
        GAP-DECISION-INDEX-PANEL-CLINC-AND-MMLU-TEST-ITEMS-ARE-LAPPI-TRAINING-DATA-2026-10-03):
        MMLU test and dev and CLINC test are refused as rows (``benchmark_eval_split_is_a_target``)
        and scanned as targets by ``tools/containment_scan.py --v6-benchmark-targets``."""
        return replace(self, benchmark_eval_splits_are_targets=True)
