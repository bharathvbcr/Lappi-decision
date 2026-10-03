"""A7: v5's val and held-out rows are v4's, plus CLINC's re-keyed oos utterances, and nothing else.

The gate, verbatim from ``campaign/v5-preregistered.DRAFT.json`` ``amendments_pending[11]``:
per-family val and held-out counts and identity-key sets equal to v4's for every non-CLINC
family; CLINC val 1,572 (within_domain 1,571) and held-out 1,262 per family; any other
difference refuses (rule 2).

A build with the general-decision pool (``--decisions-pool``) adds that pool's families, and
the amendment's A7 is "v4's families byte-equal, new families at the counts stated here"
(``campaign/v5-preregistered.DRAFT.json`` data.sources, 0e8d35c). Pass the same pool with
``--decisions-pool``: each pool family's val rows must be exactly the pool's val rows, by
identity key, as ``qd_data.decisions.rewrite_typed_decision`` builds them (the pool's own
manifest names how many), and it may have no held-out rows. Every v4 family's check is
unchanged, and without the flag a pool family refuses as any unnamed family does.

    python tools/v5_a7_check.py \\
        --v4 /Users/bharath/qd-campaign/phase4-v4-2026-10-01 --v5 <v5 build --out> \\
        [--decisions-pool <the pool the v5 build read>] --report <path.json>

Both builds are read through ``qd_data.manifest.Manifest.read``, which re-derives each file's
``data_snapshot_hash`` and refuses an edited one. The v4 baseline is itself checked against the
gate's numbers first, so pointing ``--v4`` at the wrong build refuses rather than passing a
comparison against the wrong rows. A CLINC family passes only when its in-scope rows are v4's,
row for row by identity key, and the rest are exactly the oos rows the re-key moved in (DRAFT
``data`` CLINC ``effect_at_seed_20260919``: 72 val, 62 held-out). Exit 0 only when every
family of both splits passes; every difference is listed, with the first identity keys that
are missing or extra, which is what names an own-prose or contrast row that knocked a val or
held-out row out in dedupe.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from qd_data.manifest import Manifest
from qd_train.replay import write_text_atomic

#: Where a build writes the two manifests A7 reads (``real_tokenizer_pipeline._write_manifests``).
SPLIT_PATHS: dict[str, str] = {
    "val": "data/pool/val.json",
    "heldout": "data/heldout/heldout.json",
}
#: A CLINC oos utterance's split unit after the re-key (``qd_data.mixture.clinc_keys``).
OOS_REPO_PREFIX = "clinc-oos:"
#: Every CLINC row's identity spelling for an oos utterance (v4's, kept by the re-key).
OOS_IDENTITY_PREFIX = "clinc-intent:oos::"
#: How many identity keys of each kind a difference lists.
SHOW = 5


@dataclass(frozen=True, slots=True)
class A7Gate:
    """The gate's numbers. ``non_clinc`` is v4's count per family (which v5 must equal);
    ``clinc`` is v5's count per family; ``oos`` is how many oos rows each CLINC family gains."""

    non_clinc: dict[str, dict[str, int]]
    clinc: dict[str, dict[str, int]]
    oos: dict[str, int]

    def families(self, split: str) -> set[str]:
        return set(self.non_clinc[split]) | set(self.clinc[split])


_CLINC = ("intent.classification", "intent.domain", "intent.in_scope", "intent.within_domain")

#: The DRAFT's A7 numbers, literally; ``test_v5_a7_check`` re-reads them from the DRAFT.
GATE = A7Gate(
    non_clinc={
        "val": {
            "code.defect_class": 2304,
            "commonsense.multiple_choice": 1197,
            "knowledge.multiple_choice": 1485,
            "qa.answer_span": 5139,
        },
        "heldout": {
            "code.defect_class": 2182,
            "qa.answer_span": 6563,
            "qa.answerability": 24160,
        },
    },
    clinc={
        "val": {f: (1571 if f == "intent.within_domain" else 1572) for f in _CLINC},
        "heldout": dict.fromkeys(_CLINC, 1262),
    },
    oos={"val": 72, "heldout": 62},
)


@dataclass(frozen=True, slots=True)
class PoolExpectation:
    """What a build that read a general-decision pool must hold of that pool's families:
    ``val[family]`` is the identity keys of the pool's val rows; held-out holds none."""

    examples_sha256: str
    val: dict[str, Counter[str]]

    def families(self) -> set[str]:
        return set(self.val)


def pool_expectation(pool_dir: Path) -> PoolExpectation:
    """Every val row of the pool, rewritten as the pipeline rewrites it. The pool is read
    through ``load_decision_pool`` (checked against its manifest's sha256), and the per-family
    counts must equal the manifest's ``val_by_family``, so the numbers checked are the ones the
    pool states."""
    from qd_data.decisions import load_decision_pool, pool_data_config, rewrite_typed_decision

    pool = load_decision_pool(pool_dir)
    config = pool_data_config()
    val: dict[str, Counter[str]] = {}
    for rows in pool.raw.values():
        for i, raw in enumerate(rows):
            if raw.split != "val":
                continue
            row = rewrite_typed_decision(raw, family_id=raw.family_id, index=i, config=config)
            val.setdefault(row.family_id, Counter())[row.identity_key] += 1
    stated = pool.manifest.get("val_by_family")
    got = {f: sum(c.values()) for f, c in val.items()}
    if stated != got:
        raise SystemExit(
            f"{pool_dir}: the pool's val rows per family are {got}, its manifest states {stated}"
        )
    return PoolExpectation(examples_sha256=pool.examples_sha256, val=val)


def _identities(manifest: Manifest) -> dict[str, Counter[str]]:
    out: dict[str, Counter[str]] = {}
    for entry in manifest.entries:
        out.setdefault(entry.family_id, Counter())[entry.identity_key] += 1
    return out


def _diff(want: Counter[str], got: Counter[str]) -> dict[str, Any]:
    missing = want - got
    extra = got - want
    return {
        "missing": sum(missing.values()),
        "extra": sum(extra.values()),
        "first_missing": sorted(missing)[:SHOW],
        "first_extra": sorted(extra)[:SHOW],
    }


def _read(root: Path, split: str) -> Manifest:
    path = root / SPLIT_PATHS[split]
    manifest = Manifest.read(path)
    if manifest.split != split:
        raise SystemExit(f"{path} is the {manifest.split!r} manifest, not {split!r}")
    return manifest


def check_split(
    v4: Manifest, v5: Manifest, split: str, gate: A7Gate, pool: PoolExpectation | None = None
) -> list[dict[str, Any]]:
    """One result per family the gate, v4, v5 or the pool names. Each carries ``passed`` and,
    when it failed, every reason."""
    ids4 = _identities(v4)
    ids5 = _identities(v5)
    oos5: dict[str, Counter[str]] = {}
    for entry in v5.entries:
        if entry.repo_key.startswith(OOS_REPO_PREFIX):
            oos5.setdefault(entry.family_id, Counter())[entry.identity_key] += 1
    oos4 = sum(1 for e in v4.entries if e.repo_key.startswith(OOS_REPO_PREFIX))
    results: list[dict[str, Any]] = []
    pool_families = pool.families() if pool is not None else set()
    for family in sorted(gate.families(split) | set(ids4) | set(ids5) | pool_families):
        want = ids4.get(family, Counter())
        got = ids5.get(family, Counter())
        reasons: list[str] = []
        row: dict[str, Any] = {
            "split": split,
            "family": family,
            "v4": sum(want.values()),
            "v5": sum(got.values()),
        }
        if pool is not None and family in pool_families and family not in gate.families(split):
            row["kind"] = "decision-pool"
            expected = pool.val.get(family, Counter()) if split == "val" else Counter()
            row["expected"] = sum(expected.values())
            if row["v4"]:
                reasons.append(f"baseline: v4 has {row['v4']} {split} rows of a pool family")
            row.update(_diff(expected, got))
            if row["missing"] or row["extra"]:
                reasons.append(
                    f"identity keys differ from the pool's {split} rows: {row['missing']} "
                    f"missing, {row['extra']} extra"
                )
        elif family not in gate.families(split):
            row["kind"] = "not in the gate"
            reasons.append(f"{family} is not a {split} family the gate names")
        elif family in gate.non_clinc[split]:
            row["kind"] = "non-clinc"
            row["expected"] = gate.non_clinc[split][family]
            if row["v4"] != row["expected"]:
                reasons.append(
                    f"baseline: v4 has {row['v4']} rows, the gate says v4 has {row['expected']}"
                )
            row.update(_diff(want, got))
            if row["missing"] or row["extra"]:
                reasons.append(
                    f"identity keys differ from v4's: {row['missing']} missing, "
                    f"{row['extra']} extra"
                )
        else:
            row["kind"] = "clinc"
            row["expected"] = gate.clinc[split][family]
            oos = oos5.get(family, Counter())
            row["oos"] = sum(oos.values())
            if oos4:
                reasons.append(f"baseline: v4's {split} has {oos4} rows on an oos split unit")
            if row["v4"] + gate.oos[split] != row["expected"]:
                reasons.append(
                    f"baseline: v4's {row['v4']} + {gate.oos[split]} oos != the gate's "
                    f"{row['expected']}"
                )
            if row["oos"] != gate.oos[split]:
                reasons.append(f"{row['oos']} oos rows, the gate's re-key moves {gate.oos[split]}")
            wrong = sorted(k for k in oos if not k.startswith(OOS_IDENTITY_PREFIX))
            if wrong:
                reasons.append(
                    f"{len(wrong)} rows on an oos split unit are not oos identities: {wrong[:SHOW]}"
                )
            row.update(_diff(want, got - oos))
            if row["missing"] or row["extra"]:
                reasons.append(
                    f"in-scope identity keys differ from v4's: {row['missing']} missing, "
                    f"{row['extra']} extra"
                )
            if row["v5"] != row["expected"]:
                reasons.append(f"v5 has {row['v5']} rows, the gate says {row['expected']}")
        row["passed"] = not reasons
        row["reasons"] = reasons
        results.append(row)
    return results


def check(
    v4_root: Path, v5_root: Path, *, gate: A7Gate = GATE, decisions_pool: Path | None = None
) -> dict[str, Any]:
    pool = pool_expectation(decisions_pool) if decisions_pool is not None else None
    families: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, str]] = {}
    for split in SPLIT_PATHS:
        v4 = _read(v4_root, split)
        v5 = _read(v5_root, split)
        snapshots[split] = {"v4": v4.snapshot_hash(), "v5": v5.snapshot_hash()}
        families.extend(check_split(v4, v5, split, gate, pool))
    failed = [f"{r['split']}/{r['family']}" for r in families if not r["passed"]]
    return {
        "gate": "A7 (campaign/v5-preregistered.DRAFT.json amendments_pending[11])",
        "v4": str(v4_root),
        "v5": str(v5_root),
        **(
            {} if pool is None else {
                "decisions_pool": str(decisions_pool),
                "decisions_pool_examples_sha256": pool.examples_sha256,
            }
        ),
        "data_snapshot_hash": snapshots,
        "passed": not failed,
        "families_checked": len(families),
        "failed": failed,
        "families": families,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--v4", required=True, type=Path, help="v4 build root (holds data/)")
    p.add_argument("--v5", required=True, type=Path, help="v5 build root (holds data/)")
    p.add_argument("--report", type=Path, help="write the full report here as well")
    p.add_argument(
        "--decisions-pool", type=Path,
        help="the general-decision pool the v5 build read (its --decisions-pool); its families "
        "are checked against the pool's own val rows",
    )
    args = p.parse_args(argv)
    report = check(args.v4, args.v5, decisions_pool=args.decisions_pool)
    text = json.dumps(report, indent=1) + "\n"
    if args.report:
        write_text_atomic(args.report, text)
    for row in report["families"]:
        mark = "ok  " if row["passed"] else "FAIL"
        print(f"{mark} {row['split']:8s} {row['family']:30s} v4 {row['v4']:6d} v5 {row['v5']:6d}")
        for reason in row["reasons"]:
            print(f"       {reason}")
    print(
        f"A7 {'PASS' if report['passed'] else 'REFUSED'}: "
        f"{report['families_checked'] - len(report['failed'])}/{report['families_checked']} "
        "families"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
