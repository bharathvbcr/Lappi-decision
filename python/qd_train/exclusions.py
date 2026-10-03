"""Fable's v5 decontamination exclusion, applied once (ruling of 2026-10-02, Q2).

``qd-prep containment`` (``crates/qd-prep/src/containment.rs``, driven by
``tools/containment_scan.py``) writes one directory: the complete pair list, ``exclusions.txt``
-- the identity keys of every train row that contains at least half of some val or
repo-held-out row's word 8-grams -- and attestation v2. This module is the ONE place those
keys leave the train split: :func:`apply_exclusions`, which
``tools/real_tokenizer_pipeline.py --exclude-identity-keys`` and ``tools/real_ft_run.py``'s
``ft_splits`` rebuild both call after ``split`` and before ``split_off_replay``, so the
replay draw sees the decontaminated train split and ``partition_replay`` is handed no
contaminated keys of its own.

Refused, before any row moves: an ``exclusions.txt`` that is not byte-sorted, unique,
non-empty LF lines; an attestation beside it that is missing, not version 2, not CLEAN on
its enforced targets, made under another ``n`` or threshold than the rule's (8, 0.5), made
without v5's template strip (``export.template_strip``: ``qd_train.containment_strip``,
applied, this ``STRIP_VERSION`` and ``STRIP_RULE`` exactly, so a version-1 list is refused),
for another corpus, or naming another file's sha256; and a key that
names no train row. Only
train rows move: val and held-out never do (rule 2), and an identity key a held-out row
shares by design (the two task-holdout families) leaves that row where it is.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

from qd_data.fingerprint import code_fingerprint
from qd_data.rows import DataRow
from qd_data.split import SplitReport

from .containment_strip import STRIP_RULE, STRIP_VERSION
from .replay import DEFAULT_N, DEFAULT_THRESHOLD

__all__ = [
    "ATTESTATION_NAME",
    "ATTESTATION_VERSION",
    "EXCLUSIONS_NAME",
    "SAME_FAMILY_SCOPE_KEY",
    "ExclusionRefusal",
    "Exclusions",
    "apply_exclusions",
    "containment_corpus",
    "read_exclusions",
]

#: The files ``qd-prep containment`` writes beside each other.
EXCLUSIONS_NAME: Final[str] = "exclusions.txt"
ATTESTATION_NAME: Final[str] = "attestation.json"
ATTESTATION_VERSION: Final[int] = 2
#: Bound on the list read: v4's whole train split is 289,142 rows.
MAX_KEYS: Final[int] = 2_000_000
#: The corpus key naming the same-family containment scope: Python's one spelling (the
#: pipeline's ``corpus_identity`` writes it from here), and qd-prep containment's
#: ``SAME_FAMILY_SCOPE_KEY``, which reads it from the request's corpus object.
SAME_FAMILY_SCOPE_KEY: Final[str] = "decisions_pool_same_family_not_enforced"


class ExclusionRefusal(SystemExit):
    """An exclusion list, or its attestation, that does not describe this build."""


@dataclass(frozen=True)
class Exclusions:
    """A verified ``exclusions.txt``: its keys, its sha256 and the attestation that vouches."""

    path: Path
    keys: frozenset[str]
    sha256: str
    attestation: Mapping[str, Any]


def containment_corpus(identity: Mapping[str, object]) -> dict[str, object]:
    """What a containment attestation names as its corpus: ``identity`` (what ``ft_splits``
    was called with, ``real_tokenizer_pipeline.corpus_identity``) plus the sha256 of the
    ``qd_data`` sources that turned it into rows. The second is not optional: a ``qd_data``
    change (v5's CLINC re-keying) moves rows between splits without changing any input the
    first names."""
    fingerprint = json.dumps(code_fingerprint(), sort_keys=True).encode("utf-8")
    return {**identity, "qd_data_fingerprint_sha256": hashlib.sha256(fingerprint).hexdigest()}


def _keys_of(path: Path, raw: bytes) -> list[str]:
    if raw and not raw.endswith(b"\n"):
        raise ExclusionRefusal(f"{path}: the last line has no LF; exclusions.txt ends each key")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExclusionRefusal(f"{path}: not UTF-8 ({exc})") from exc
    keys = text.split("\n")[:-1] if text else []
    if any(not k or "\r" in k or "\t" in k for k in keys):
        raise ExclusionRefusal(f"{path}: an empty key, a CR or a tab; not an exclusion list")
    if len(keys) > MAX_KEYS:
        raise ExclusionRefusal(f"{path}: {len(keys)} keys; the bound is {MAX_KEYS}")
    encoded = [k.encode("utf-8") for k in keys]
    if any(a >= b for a, b in pairwise(encoded)):
        raise ExclusionRefusal(
            f"{path}: keys are not byte-sorted and unique, as qd-prep containment writes them"
        )
    return keys


def read_exclusions(path: Path, *, corpus: Mapping[str, object]) -> Exclusions:
    """``path`` (an ``exclusions.txt``) and the attestation beside it, verified against
    ``corpus`` (:func:`containment_corpus` of this build's inputs), or a refusal."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExclusionRefusal(f"--exclude-identity-keys {path}: unreadable ({exc})") from exc
    keys = _keys_of(path, raw)
    digest = hashlib.sha256(raw).hexdigest()
    att_path = path.parent / ATTESTATION_NAME
    try:
        att = json.loads(att_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExclusionRefusal(
            f"{att_path}: no readable attestation beside {path.name} ({exc}); an exclusion "
            "list is only the decontamination's finding with the attestation that made it"
        ) from exc
    if not isinstance(att, dict) or att.get("version") != ATTESTATION_VERSION:
        raise ExclusionRefusal(f"{att_path}: not a version {ATTESTATION_VERSION} attestation")
    if att.get("tool") != "qd-prep containment":
        raise ExclusionRefusal(f"{att_path}: written by {att.get('tool')!r}, not qd-prep")
    if att.get("n") != DEFAULT_N or att.get("threshold") != DEFAULT_THRESHOLD:
        raise ExclusionRefusal(
            f"{att_path}: n={att.get('n')} threshold={att.get('threshold')}, not the rule's "
            f"{DEFAULT_N} and {DEFAULT_THRESHOLD} (Fable: n and threshold unchanged)"
        )
    export = att.get("export")
    strip = export.get("template_strip") if isinstance(export, dict) else None
    if (
        not isinstance(strip, dict)
        or strip.get("applied") is not True
        or strip.get("rule") != STRIP_RULE
        or strip.get("version") != STRIP_VERSION
    ):
        made = (
            {k: strip.get(k) for k in ("applied", "version", "rule")}
            if isinstance(strip, dict) else None
        )
        raise ExclusionRefusal(
            f"{att_path}: the scan did not apply v5's template strip version {STRIP_VERSION} "
            f"(export.template_strip {made!r}; this build requires applied=True, version "
            f"{STRIP_VERSION} and qd_train.containment_strip.STRIP_RULE exactly). A list over "
            "unstripped prompt_content, or under version 1, measures constant question and "
            "option text, not overlap -- version 1 left intent.within_domain's per-domain "
            "intent lists in and excluded 63-66% of CLINC keys on the subsamples, "
            "HANDOFF/prep2-2026-10-02.md -- and v5's "
            "rule (campaign/v5-preregistered data.decontamination.rule) strips it"
        )
    if att.get("exclusions_sha256") != digest:
        raise ExclusionRefusal(
            f"{att_path} vouches for exclusions sha256 {att.get('exclusions_sha256')!r}; "
            f"{path} is {digest}"
        )
    if att.get("n_exclusions") != len(keys):
        raise ExclusionRefusal(
            f"{att_path} counts {att.get('n_exclusions')} keys; {path} holds {len(keys)}"
        )
    remaining = att.get("remaining_hits")
    if (
        att.get("clean") is not True
        or not isinstance(remaining, dict)
        or not remaining
        or any(v != 0 for v in remaining.values())
        or att.get("excluded_from") != "train"
    ):
        raise ExclusionRefusal(
            f"{att_path} is not CLEAN on its enforced targets (clean={att.get('clean')}, "
            f"remaining_hits={remaining}, excluded_from={att.get('excluded_from')!r}, "
            f"because {att.get('not_clean_because')}): this build is refused"
        )
    if att.get("corpus") != dict(corpus):
        raise ExclusionRefusal(
            f"{att_path} was made for corpus {att.get('corpus')}, and this build is "
            f"{dict(corpus)}: an exclusion list for another corpus excludes the wrong rows"
        )
    # The same-family scope (a corpus that reads a decision pool names one): the attestation
    # must show the scan applied exactly it. A qd-prep that predates the scope ignores the
    # corpus key and writes no block, so its list would carry the template hits the scope
    # leaves out; a block where the corpus names no scope is a list made under another rule.
    scope = corpus.get(SAME_FAMILY_SCOPE_KEY)
    block = att.get("same_family_not_enforced")
    applied = block.get("families") if isinstance(block, dict) else None
    if (scope is None) != (block is None) or (scope is not None and applied != list(scope)):
        raise ExclusionRefusal(
            f"{att_path}: the corpus names same-family scope {scope!r} and the attestation "
            f"shows {applied!r} applied (same_family_not_enforced): the scan did not apply "
            "this build's scope (AUDIT/finalize-2026-10-03/containment-scope-ruling-2026-10-03.md)"
        )
    return Exclusions(path=path, keys=frozenset(keys), sha256=digest, attestation=att)


def apply_exclusions(
    split_report: SplitReport, path: Path | None, *, corpus: Mapping[str, object]
) -> tuple[SplitReport, Exclusions | None, tuple[DataRow, ...]]:
    """``(report, exclusions, excluded_rows)``: ``split_report`` with every train row whose
    identity key ``path`` lists removed. ``path=None`` returns ``split_report`` itself,
    untouched, so a build without the flag is the build it was.

    Called once, after ``split`` and before ``split_off_replay``. Every key must name at
    least one train row: a key naming none is a list for another corpus or split, and
    excluding nothing for it would attest a cleanliness this build does not have.
    """
    if path is None:
        return split_report, None, ()
    exclusions = read_exclusions(path, corpus=corpus)
    train = split_report.rows_by_split.get("train", ())
    named = {r.identity_key for r in train} & exclusions.keys
    unnamed = sorted(exclusions.keys - named)
    if unnamed:
        raise ExclusionRefusal(
            f"{path}: {len(unnamed)} key(s) name no train row of this build, first "
            f"{unnamed[:3]}: the list is not this split's"
        )
    kept = tuple(r for r in train if r.identity_key not in exclusions.keys)
    excluded = tuple(r for r in train if r.identity_key in exclusions.keys)
    report = dataclasses.replace(
        split_report, rows_by_split={**split_report.rows_by_split, "train": kept}
    )
    return report, exclusions, excluded
