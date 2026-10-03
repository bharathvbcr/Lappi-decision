"""Promotion-verdict fixtures more than one test module needs. Not a test module.

Two things a fixture standing for a real scored row has to say, because the verdict reads them:

- The decisions record it is judged under. The repo's ``docs/promotion-decisions.json`` is human
  data that changes when a human rules (all six were decided on 2026-10-03), so a test about
  the verdict's mechanics names the record it means: :func:`as_built_record`, every question
  open, each value the gate or control as the code computes it (main a8c22ae^).
- The metrics tools/real_ft_run.py writes beside the gates: the decode count condition 7 reads
  for the outcome-count gates, and the per-family metrics a decided record re-derives from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from qd_train.ledger import DEFAULT_DECISIONS_PATH, RunRecorder
from qd_train.tristate import Ran

#: Captured at import, so a test that points the verdict's default elsewhere still builds its
#: record from the committed one.
REPO_RECORD: Final[Path] = DEFAULT_DECISIONS_PATH

#: The record as built, before any human ruled (main a8c22ae^).
AS_BUILT: Final[dict[str, dict[str, object]]] = {
    "promotion_population": {
        "status": "open", "value": "pooled: every val family's choice rows, judged together",
        "families": "all"},
    "average_may_promote": {"status": "open", "value": False},
    "ece_population": {
        "status": "open",
        "value": "per slot shape and per language; a letter row with no language leaves the "
                 "gate not_run"},
    "degenerate_head_floor": {
        "status": "open",
        "value": "mean predictive entropy >= 0.15 nats and no predicted class above 0.95, on "
                 "every slot shape"},
    "privileged_hunk_pass_rule": {"status": "open", "value": "not set: the control stays not_run"},
    "transfer_gate_definition": {"status": "open",
                                 "value": "not specified: the control stays not_run"},
}

#: F's score-val decode (f4feac15, b45406b5, c962cdd9): every eligible val row decoded.
COMPLETE_DECODE: Final[tuple[int, int]] = (18223, 18223)


def write_record(directory: Path, *, base: str = "as_built",
                 **changes: dict[str, object]) -> Path:
    """The committed record with some entries changed -- what a human edit would look like.
    ``base`` is ``"as_built"`` (every question open) or ``"repo"`` (the record as committed).
    Each call writes a new file, so one test can hold several records."""
    record = json.loads(REPO_RECORD.read_text(encoding="utf-8"))
    if base == "as_built":
        for name, fields in AS_BUILT.items():
            record["decisions"][name].update(
                {**fields, "decided_by": None, "decided_on": None, "decision_ref": None})
    for name, fields in changes.items():
        record["decisions"][name].update(fields)
    path = directory / f"decisions-{len(list(directory.glob('decisions-*.json')))}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def as_built_record(directory: Path) -> Path:
    return write_record(directory)


def record_decode_coverage(rec: RunRecorder,
                           decoded: tuple[int, int] = COMPLETE_DECODE) -> None:
    """``val_rows_decoded`` as tools/real_ft_run.py _record_score writes it."""
    n, n_total = decoded
    rec.metric("val_rows_decoded", Ran(passed=n == n_total, value=n, n=n, n_total=n_total))


def record_scored_row_metrics(rec: RunRecorder, *, perm_agree: int = 2291,
                              suite: tuple[int, int, int] = (58, 59, 57),
                              in_abstained: int = 13, family_ece: bool = True,
                              share: float | None = 0.27,
                              decoded: tuple[int, int] | None = COMPLETE_DECODE) -> None:
    """The per-family and per-category metrics a scored eval row carries, as F's did, green
    unless told otherwise: what the verdict reads under a decided record, and the decode
    count condition 7 reads for the outcome-count gates."""
    family = "code.defect_class"
    if decoded is not None:
        record_decode_coverage(rec, decoded)
    rec.metric(f"permutation_consistency.family.{family}",
               Ran(passed=True, value=perm_agree / 2304, n=perm_agree, n_total=2304))
    for category, n in zip(("prose", "unseen-language", "scrambled"), suite, strict=True):
        rec.metric(f"ood_abstain.{category}", Ran(passed=True, value=n / 60, n=n, n_total=60))
    rec.metric(f"ood_abstain.in_distribution.family.{family}",
               Ran(passed=True, value=in_abstained / 2304, n=in_abstained, n_total=2304))
    if family_ece:
        rec.metric(f"ece.family.{family}.choice.k4",
                   Ran(passed=True, value=0.0207, n=2304, n_total=2304))
    rec.metric("degenerate_head.choice.k4", Ran(passed=True, value=0.38, n=2304, n_total=2304))
    if share is not None:
        rec.metric("degenerate_head.choice.k4.top_class_share",
                   Ran(passed=share <= 0.95, value=share, n=int(share * 2304), n_total=2304))
