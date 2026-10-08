"""v5's contrast rows: prose-noul route (ii), derived after the split.

The human's item 5 (ii) (``AUDIT/fable-optimize-2026-10-02/human-decisions.md``): MMLU/CSQA
**train** questions rendered under ``code.defect_class`` with gold noul, beside their own
letter-gold rows. Fable's v5 review section 2.4, in the recommended form the lead took
(``campaign/v5-preregistered.DRAFT.json`` ``data.sources[4].amended``):

* derived at pipeline time, **after** dedupe, split and ``--exclude-identity-keys``, from the
  train twins that survived all three -- so an excluded twin never has a contrast row, with no
  exclusion mapping and no over-draw;
* ``per_family[f]`` rows per family (1,200 MMLU + 800 CSQA), drawn in the keyed blake2b order of
  the twin's identity key (:func:`contrast_order_key`);
* identity ``contrast:<twin identity>``; recorded in the train shard header as
  ``contrast_rows {count, sha256, seed}`` (``qd_train.artifacts.ContrastRows``);
* never a dedupe or split unit: the rows are appended to the train split after it is final.

Every row is built by ``qd_data.defect_class.prose_defect_row`` -- the one "prose text -> Z row"
builder the own-prose route also uses -- and rendered by the one
``qd_data.mixture.rewrite_defect_class``.

**Only on the decontamination rebuild.** The DRAFT's build order derives contrast rows "after
the rebuild's dedupe, split and exclusions" (``build_order[3]``). A first build is the
containment scan's source; a contrast row there could put a ``contrast:`` key into
``exclusions.txt``, which the rebuild's ``apply_exclusions`` -- run before any contrast row
exists -- would refuse as naming no train row. So :func:`apply_contrast` derives nothing when no
exclusion list was applied, and says so as a ``NotRun`` the build records, never as zero rows.
A contrast row's prompt n-grams are a subset of its twin's, so the first scan's pair list
already bounds every contrast row of the rebuild (Fable 2.4): no re-scan is needed.

**One entry point for the build and its rebuild.** ``tools/real_tokenizer_pipeline.py``'s
``run`` and ``tools/real_ft_run.py``'s ``ft_splits`` (the trainer's rebuild of the same rows)
must both call :func:`apply_contrast` at the same point -- after ``apply_exclusions``, before
``split_off_replay`` -- with the same arguments, or the trainer's rebuild disagrees with the
shard set it is checking.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from qd_data.config import DataConfig
from qd_data.defect_class import (
    DEFECT_FAMILY_ID,
    NOUL_ROUTE_CONTRAST,
    NOUL_ROUTE_KEY,
    PROSE_FORM_QUESTION,
    ContrastSpec,
    noul_contrast_spec,
    prose_defect_row,
)
from qd_data.mixture import RowRefused, rewrite_defect_class
from qd_data.rows import DataRow, row_content_hash
from qd_data.split import SplitReport

from .artifacts import ContrastRows
from .tristate import NotRun, Ran, TriState

__all__ = [
    "CONTRAST_PATHS",
    "V6_CONTRAST_MOVES",
    "ContrastRecord",
    "ContrastShortfall",
    "apply_contrast",
    "contrast_order_key",
    "contrast_rows_sha256",
    "derive_contrast_rows",
    "v6_contrast_spec",
]

#: The paths a contrast row's context sits under, one per row by its twin's keyed digest.
#: Documents and code files alike, as v3b's SQuAD prose rows' paths are, so the path does not
#: predict the label -- and never ``notes.txt``, the OOD suite's prose path, which
#: ``prose_defect_row`` refuses.
CONTRAST_PATHS: Final[tuple[str, ...]] = (
    "docs/faq.md",
    "docs/questions.md",
    "QUESTIONS.md",
    "doc/quiz.rst",
    "src/quiz.py",
    "internal/quiz/doc.go",
    "src/quiz.rs",
    "src/quiz.ts",
)


#: v6's contrast re-spec (``AUDIT/v6-rulings-2026-10-08/fable-v6-data-design-ruling.md`` R2): under
#: ``DataConfig.benchmark_eval_splits_are_targets`` MMLU's dev and test are decontamination targets,
#: never rows, so ``knowledge.multiple_choice`` has no train twin and its contrast quota moves to
#: ``commonsense.multiple_choice`` (v5's 1,200 + 800 become 2,000 CSQA). Applied by
#: :func:`v6_contrast_spec`, only under that config, so a v5 build derives v5's rows.
V6_CONTRAST_MOVES: Final[dict[str, str]] = {
    "knowledge.multiple_choice": "commonsense.multiple_choice",
}


class ContrastShortfall(SystemExit):
    """The surviving train twins cannot supply the rows the noul corpus asks for."""


def contrast_order_key(identity_key: str, *, seed: int) -> str:
    """The draw order: blake2b-8 of ``seed`` and the twin's identity key, NUL-separated."""
    material = f"{seed}\x00{identity_key}".encode()
    return hashlib.blake2b(material, digest_size=8).hexdigest()


def contrast_rows_sha256(rows: list[DataRow] | tuple[DataRow, ...]) -> str:
    """sha256 over ``row_id<TAB>content_hash<LF>`` per row, in the order they were appended."""
    h = hashlib.sha256()
    for row in rows:
        h.update(f"{row.row_id}\t{row_content_hash(row)}\n".encode())
    return h.hexdigest()


@dataclass(frozen=True)
class ContrastRecord:
    """What :func:`derive_contrast_rows` added, and every twin it read and passed over."""

    count: int
    sha256: str
    seed: int
    by_family: dict[str, int]
    #: Train twins per family that survived dedupe, split and the exclusion list.
    eligible: dict[str, int]
    #: Twins read in draw order and passed over, by reason (a repeated identity or text, or a
    #: row ``rewrite_defect_class`` refuses).
    skipped: dict[str, int]

    def header(self) -> ContrastRows:
        return ContrastRows(count=self.count, sha256=self.sha256, seed=self.seed)

    def detail(self) -> str:
        return (
            f"{self.count} contrast row(s) {self.by_family} from surviving train twins "
            f"{self.eligible}, passed over {self.skipped or 'none'}; seed {self.seed}, rows "
            f"sha256 {self.sha256}"
        )


def derive_contrast_rows(
    split_report: SplitReport, *, spec: ContrastSpec, config: DataConfig
) -> tuple[SplitReport, ContrastRecord]:
    """``split_report`` with ``spec``'s contrast rows appended to its train split.

    For each family, its train rows are read in :func:`contrast_order_key` order (ties broken by
    identity key, then row id) and the first ``spec.per_family[family]`` usable ones become
    rows. A twin whose identity or text an earlier row already carries is passed over and
    counted, and so is one the rewriter refuses. Fewer usable twins than asked refuses the build
    (:class:`ContrastShortfall`): the count is pre-registered, so a smaller one is not v5's.
    """
    train = tuple(split_report.rows_by_split.get("train", ()))
    already = [r.row_id for r in train if r.metadata.get(NOUL_ROUTE_KEY) == NOUL_ROUTE_CONTRAST]
    if already:
        raise ContrastShortfall(
            f"the train split already holds {len(already)} contrast row(s) (first "
            f"{already[0]!r}); contrast rows are derived once, after the split"
        )
    added: list[DataRow] = []
    by_family: dict[str, int] = {}
    eligible: dict[str, int] = {}
    skipped: Counter[str] = Counter()
    used_identities: set[str] = set()
    used_texts: set[str] = set()
    for family in sorted(spec.per_family):
        want = spec.per_family[family]
        twins = sorted(
            (r for r in train if r.family_id == family),
            key=lambda r: (
                contrast_order_key(r.identity_key, seed=spec.seed),
                r.identity_key,
                r.row_id,
            ),
        )
        eligible[family] = len(twins)
        got = 0
        for twin in twins:
            if got == want:
                break
            text = twin.request.context.decode("utf-8")
            if twin.identity_key in used_identities:
                skipped[f"{family}:repeated_identity"] += 1
                continue
            if text.strip() in used_texts:
                skipped[f"{family}:repeated_text"] += 1
                continue
            digest = contrast_order_key(twin.identity_key, seed=spec.seed)
            defect = prose_defect_row(
                text,
                route=NOUL_ROUTE_CONTRAST,
                form=PROSE_FORM_QUESTION,
                example_id=f"{NOUL_ROUTE_CONTRAST}:{twin.row_id}",
                repo=twin.repo_key,
                path=CONTRAST_PATHS[int(digest, 16) % len(CONTRAST_PATHS)],
                licence=twin.licence_id,
                identity_key=f"{NOUL_ROUTE_CONTRAST}:{twin.identity_key}",
            )
            try:
                row = rewrite_defect_class(
                    defect, family_id=DEFECT_FAMILY_ID, index=0, config=config
                )
            except RowRefused as e:
                skipped[f"{family}:refused:{e.reason_code}"] += 1
                continue
            used_identities.add(twin.identity_key)
            used_texts.add(text.strip())
            added.append(row)
            got += 1
        if got < want:
            raise ContrastShortfall(
                f"{family}: {got} contrast row(s) from {len(twins)} surviving train twin(s), "
                f"{want} asked (passed over {dict(skipped)}); the count is pre-registered and "
                "is never made up from another family"
            )
        by_family[family] = got
    report = dataclasses.replace(
        split_report, rows_by_split={**split_report.rows_by_split, "train": train + tuple(added)}
    )
    record = ContrastRecord(
        count=len(added),
        sha256=contrast_rows_sha256(added),
        seed=spec.seed,
        by_family=by_family,
        eligible=eligible,
        skipped=dict(sorted(skipped.items())),
    )
    return report, record


def v6_contrast_spec(spec: ContrastSpec, *, config: DataConfig) -> ContrastSpec:
    """``spec`` as ``config`` derives it: unchanged under v5's config; under v6's benchmark
    re-pin each family of :data:`V6_CONTRAST_MOVES` hands its quota to its target, the seed
    kept, so the total the noul corpus asked for is still the total derived."""
    if not config.benchmark_eval_splits_are_targets:
        return spec
    per_family: dict[str, int] = {}
    for family, n in spec.per_family.items():
        target = V6_CONTRAST_MOVES.get(family, family)
        per_family[target] = per_family.get(target, 0) + n
    return ContrastSpec(per_family=per_family, seed=spec.seed)


def apply_contrast(
    split_report: SplitReport,
    *,
    defect_noul: Path | None,
    exclusions_applied: bool,
    config: DataConfig,
) -> tuple[SplitReport, ContrastRecord | None, TriState | None]:
    """The one call the build and the trainer's rebuild make, after ``apply_exclusions``.

    ``(report, record, status)``. With no noul corpus, or one that asks for no contrast rows,
    ``split_report`` comes back untouched with ``record`` and ``status`` ``None``: the build is
    the build it was. With a request but no exclusion list applied (the first, scan-source
    build), it comes back untouched with a ``NotRun`` naming why. Otherwise the rows are derived
    and ``status`` is a ``Ran`` carrying both counts.
    """
    spec = noul_contrast_spec(defect_noul) if defect_noul is not None else None
    if spec is None:
        return split_report, None, None
    spec = v6_contrast_spec(spec, config=config)
    if not exclusions_applied:
        return (
            split_report,
            None,
            NotRun(
                reason=(
                    f"{defect_noul} asks for contrast rows {spec.per_family}, and they are "
                    "derived only on the rebuild that applies --exclude-identity-keys (DRAFT "
                    "build_order[3]); this build is the containment scan's source and carries "
                    "none"
                )
            ),
        )
    report, record = derive_contrast_rows(split_report, spec=spec, config=config)
    asked = sum(spec.per_family.values())
    return (
        report,
        record,
        Ran(
            passed=record.count == asked,
            value=record.count,
            n=record.count,
            n_total=asked,
            detail=record.detail(),
        ),
    )
