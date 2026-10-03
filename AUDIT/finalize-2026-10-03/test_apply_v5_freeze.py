#!/usr/bin/env python3
"""Tests for apply_v5_freeze.py on a small synthetic v5 build (lane L-v5-freeze, 2026-10-03).

The v4 format test (README of the HANDOFF) exercises the parsers on a real build, but v4 has no
A7 report, no exclusions, no decision pool and no near-duplicate scope, so half the script's paths
never run there. This builds a tiny build output with every v5 member the freeze reads, bound to
the real scan, the real decision pool and the real noul parts (by sha256, as a real build is),
and checks the fills, the refusals and that no forbidden list is ever copied.

Standard library only. It reads the machine's real inputs and skips when they are absent:

    python3 AUDIT/finalize-2026-10-03/test_apply_v5_freeze.py

QD_FREEZE_DRAFT, QD_FREEZE_REPO name the DRAFT and the build checkout (defaults: the v5-build
worktree). It writes only under a temporary directory inside build/ of this checkout.
"""

from __future__ import annotations

import array
import contextlib
import copy
import io
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import apply_v5_freeze as fz  # noqa: E402

BUILD_WT = Path("/Users/bharath/Code/research/Lappi-decision/build/v5-build-wt")
DRAFT = Path(
    os.environ.get("QD_FREEZE_DRAFT", BUILD_WT / "campaign" / "v5-preregistered.DRAFT.json")
)
REPO = Path(os.environ.get("QD_FREEZE_REPO", BUILD_WT))
SCAN = Path("/Users/bharath/qd-campaign/v5-containment-v2-2026-10-03")
DEC = Path("/Users/bharath/qd-campaign/v5-decisions-data-2026-10-03")
POOL = DEC / "pool-v5-decisions-v4"
POOL_REPORT = DEC / "patches" / "POOL-RESULT-v4.md"
FETCH = DEC / "fetch-record-decisions-v4-2026-10-03.json"
F_LEDGER = REPO_ROOT / "ledger" / "gh200-p4-v4-2026-10-01.jsonl"
SECRET = "SECRET-HELD-OUT-KEY-NEVER-COPIED"
NEEDED = (
    DRAFT,
    REPO / fz.ACCOUNTING,
    SCAN / "attestation.json",
    POOL / "manifest.json",
    REPO / "data" / "pool" / "own-prose-v1" / "units.jsonl",
    F_LEDGER,
)


def first_id(path: Path) -> str:
    with path.open(encoding="utf-8") as fh:
        return str(json.loads(fh.readline())["id"])


def write_npy(path: Path, values: list[int]) -> None:
    header = ("{'descr': '<i8', 'fortran_order': False, 'shape': (%d,), }" % len(values)).encode()
    pre = b"\x93NUMPY\x01\x00"
    pad = (64 - (len(pre) + 2 + len(header) + 1) % 64) % 64
    header += b" " * pad + b"\n"
    path.write_bytes(
        pre + struct.pack("<H", len(header)) + header + array.array("q", values).tobytes()
    )


def entry(row_id: str, family: str, source: str, repo_key: str, licence: str = "mit") -> dict:
    return {
        "row_id": row_id,
        "content_hash": "0" * 64,
        "split": "x",
        "source_id": source,
        "host": "local",
        "family_id": family,
        "repo_key": repo_key,
        "identity_key": f"{repo_key}::{row_id}",
        "licence_id": licence,
        "obligations": [],
    }


class Fixture:
    """One synthetic v5 build under ``root``; ``spec`` edits apply before writing."""

    def __init__(
        self,
        root: Path,
        *,
        decision_tokens: int = 1000,
        val_prose: bool = False,
        a7_passed: bool = True,
        recipe_exclusions: str | None = None,
    ) -> None:
        self.root = root
        att = json.loads((SCAN / "attestation.json").read_text(encoding="utf-8"))
        excl = fz.sha256_file(SCAN / "exclusions.txt")
        pool_sha = json.loads((POOL / "manifest.json").read_text(encoding="utf-8"))[
            "examples_sha256"
        ]
        parts = REPO / "data" / "pool"
        v3c = json.loads((parts / "defect-noul-v3c" / "manifest.json").read_text(encoding="utf-8"))
        unit = first_id(parts / "own-prose-v1" / "units.jsonl")
        with (parts / "own-prose-v1" / "units.jsonl").open(encoding="utf-8") as fh:
            self.unit_form = json.loads(fh.readline())["form"]
        g6 = first_id(parts / "commitpackft-g6-v1" / "examples.jsonl")
        v3b = first_id(parts / "defect-noul-v3b" / "examples.jsonl")
        d = "qdm:code.defect_class:"
        # (row_id, family, source, repo_key, [sequence lengths and slots])
        rows = [
            (f"{d}r1::a.py::f/1#0", fz.DEFECT, "qd-mutate/commitpackft", "r1", [900, 400]),
            (f"{d}compose:0001", fz.DEFECT, "qd-mutate/commitpackft", "r2", [9500, 9000]),
            (
                f"{d}own-prose:{unit}",
                fz.DEFECT,
                "qd-mutate/commitpackft",
                "own-prose:apps/X",
                [300, 200],
            ),
            (
                f"{d}contrast:mmlu:knowledge.multiple_choice:aux:abc:1",
                fz.DEFECT,
                "qd-mutate/commitpackft",
                "r3",
                [300, 200],
            ),
            (f"{d}{g6}", fz.DEFECT, "qd-mutate/commitpackft", "r4", [500, 200]),
            (f"{d}{v3b}", fz.DEFECT, "qd-mutate/commitpackft", "r5", [400, 200]),
            *[
                (f"clinc:{f}:oos1:1", f, "clinc/clinc_oos", "clinc-oos:0123456789abcdef", [50])
                for f in fz.CLINC_FAMILIES
            ],
            (
                "decision:openjev-policy-1",
                "openjev.policy",
                "ZefanCai/Open-Jev-v1.1",
                "openjev.policy-train:g1",
                [decision_tokens],
            ),
            (
                "decision:vitaminc-1",
                "vitaminc.nli",
                "tals/vitaminc",
                "vitaminc.nli-train:g2",
                [1000],
            ),
        ]
        self.rows = rows
        out = root / "build"
        for sub in ("data/pool", "data/heldout", "shards/train", "shards/val"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        train_entries = [entry(r, f, s, k) for r, f, s, k, _ in rows]
        train = {
            "manifest_format_version": 1,
            "split": "train",
            "data_snapshot_hash": "a" * 64,
            "n_rows": len(train_entries),
            "dedupe": {
                "n_candidate_pairs": 123,
                "max_candidate_pairs": fz.POOL_BOUND,
                "status": {"state": "ran", "passed": True},
                "near_duplicate_scope": {
                    "ruling": "AUDIT/finalize-2026-10-03/dedupe-probe/RULING.md",
                    "minhash_rows": 10,
                    "exact_content_rows_by_family": {"openjev.policy": 5},
                },
                "exact_content_clusters": [
                    {
                        "digest": "x",
                        "kept_unit_key": "k",
                        "minhash_unit_key": None,
                        "dropped_unit_keys": ["u1", "u2"],
                        "repo_keys": ["r"],
                        "n_rows_dropped": 2,
                    }
                ],
            },
            "split_report": {
                "near_duplicate_disjoint": {
                    "state": "ran",
                    "passed": True,
                    "n": 10,
                    "n_total": 10,
                    "detail": "scanned 10 rows; 7 candidate pairs, 0 of them crossing a repo split",
                },
                "exact_content_disjoint": {"state": "ran", "passed": True, "n": 5, "n_total": 5},
            },
            "entries": train_entries,
        }
        (out / "data/pool/train.json").write_text(json.dumps(train, indent=2), encoding="utf-8")
        val_entries = [
            entry(f"{d}v1", fz.DEFECT, "qd-mutate/commitpackft", "rv"),
            entry("decision:openjev-policy-v", "openjev.policy", "ZefanCai/Open-Jev-v1.1", "g"),
            entry("decision:vitaminc-v", "vitaminc.nli", "tals/vitaminc", "g"),
        ]
        if val_prose:
            val_entries.append(
                entry(
                    f"{d}own-prose:{unit}", fz.DEFECT, "qd-mutate/commitpackft", "own-prose:apps/X"
                )
            )
        val = {
            "manifest_format_version": 1,
            "split": "val",
            "data_snapshot_hash": "b" * 64,
            "n_rows": len(val_entries),
            "entries": val_entries,
        }
        (out / "data/pool/val.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
        held = {
            "manifest_format_version": 1,
            "split": "heldout",
            "data_snapshot_hash": "c" * 64,
            "n_rows": 1,
            "entries": [entry(SECRET, fz.DEFECT, "qd-mutate/commitpackft", SECRET)],
        }
        (out / "data/heldout/heldout.json").write_text(json.dumps(held, indent=2), encoding="utf-8")
        seqs, lengths = [], []
        for r, *_rest, ls in rows:
            for j, length in enumerate(ls):
                seqs.append({"row_id": r, "slot_kind": 1 + 2 * j, "slot_name": "s"})
                lengths.append(length)
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        write_npy(out / "shards/train/offsets.npy", offsets)
        (out / "shards/train/sequence_index.json").write_text(
            json.dumps(
                {
                    "excluded": [],
                    "format": "qd-sequence-index/1",
                    "role": "gold",
                    "rows_in": len(rows),
                    "sequences": seqs,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        common = {"remap_hash": "e" * 64, "tokenizer_hash": "f" * 64}
        (out / "shards/train/header.json").write_text(
            json.dumps(
                {
                    **common,
                    "buckets": [64, 512, 1024, 4096, 10240, 40960],
                    "data_snapshot_hash": "a" * 64,
                    "split": "train",
                    "n_sequences": len(seqs),
                    "total_tokens": sum(lengths),
                    "exclusions_sha256": excl,
                    "shard_hash": "1" * 64,
                    "max_seq_len": 10240,
                }
            ),
            encoding="utf-8",
        )
        (out / "shards/train/remap.json").write_text(
            json.dumps({"remap_hash": "e" * 64}), encoding="utf-8"
        )
        (out / "shards/val/header.json").write_text(
            json.dumps({**common, "split": "val", "shard_hash": "2" * 64}), encoding="utf-8"
        )
        vseq = [
            {"row_id": f"{d}v{i}", "slot_kind": 1, "slot_name": "defect_class"} for i in range(2304)
        ]
        vseq += [
            {"row_id": f"{d}v{i}", "slot_kind": 3, "slot_name": "defect_span"} for i in range(2000)
        ]
        (out / "shards/val/sequence_index.json").write_text(
            json.dumps({"sequences": vseq}), encoding="utf-8"
        )
        fam = []
        for split, every in (("val", 1572), ("heldout", 1262)):
            for f in fz.CLINC_FAMILIES:
                v5 = 1571 if (split == "val" and f == "intent.within_domain") else every
                fam.append(
                    {
                        "split": split,
                        "family": f,
                        "kind": "clinc",
                        "v4": v5 - fz.EXPECTED_OOS[split],
                        "v5": v5,
                        "oos": fz.EXPECTED_OOS[split],
                        "passed": True,
                        "reasons": [],
                        "missing": 0,
                        "extra": 0,
                        "first_missing": [SECRET],
                        "first_extra": [],
                    }
                )
        fam.append(
            {
                "split": "heldout",
                "family": fz.DEFECT,
                "kind": "non-clinc",
                "v4": 2182,
                "v5": 2182,
                "passed": True,
                "reasons": [],
                "missing": 0,
                "extra": 0,
                "first_missing": [],
                "first_extra": [SECRET],
            }
        )
        fam.append(
            {
                "split": "val",
                "family": "openjev.policy",
                "kind": "decision-pool",
                "v4": 0,
                "v5": 701,
                "expected": 701,
                "passed": True,
                "reasons": [],
                "missing": 0,
                "extra": 0,
            }
        )
        a7 = {
            "gate": "A7",
            "passed": a7_passed,
            "families_checked": len(fam),
            "failed": [] if a7_passed else ["val/x"],
            "decisions_pool_examples_sha256": pool_sha,
            "data_snapshot_hash": {
                "val": {"v4": "9" * 64, "v5": "b" * 64},
                "heldout": {"v4": "8" * 64, "v5": "c" * 64},
            },
            "families": fam,
        }
        (out / "a7.json").write_text(json.dumps(a7), encoding="utf-8")
        self.build_out = out
        row = {
            "row_id": "build-row",
            "written_at": "2026-10-03T21:00:00+00:00",
            "status": "completed",
            "code_commit": "ab" * 20,
            "protocol": {"data_snapshot_hash": "a" * 64, "tokenizer_hash": "f" * 64},
            "recipe": {
                "rev": "ab" * 20,
                "exclusions_sha256": recipe_exclusions or excl,
                "decisions_pool_examples_sha256": pool_sha,
                "defect_noul_examples_sha256": v3c["examples_sha256"],
                "defect_class_examples_sha256": "6" * 64,
            },
            "metrics": {
                "decontam_exclusion": {
                    "state": "ran",
                    "value": 50000,
                    "n": 50000,
                    "n_total": 600000,
                },
                "over_max_seq_len_composed_train": {
                    "state": "ran",
                    "value": 0,
                    "n": 0,
                    "n_total": 25000,
                },
                "composed_span_survival_by_length": {
                    "state": "ran",
                    "detail": json.dumps({"bins": {"09000-09999": {"rows": 1}}}),
                },
            },
        }
        other = copy.deepcopy(row)
        other["row_id"], other["protocol"]["data_snapshot_hash"] = "other-row", "d" * 64
        self.ledger = root / "mac-v5-shards.jsonl"
        self.ledger.write_text(json.dumps(other) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
        self.rates = root / "rates.json"
        self.rates.write_text(
            json.dumps(
                [
                    {
                        "scan": str(SCAN),
                        "n_exclusions": att["n_exclusions"],
                        "families": {
                            fz.DEFECT: {"train_keys": 100, "excluded": 7},
                            "openjev.policy": {"train_keys": 50, "excluded": 0},
                        },
                        "groups": {"CLINC (intent.*)": {"train_keys": 1044, "excluded": 0}},
                    }
                ]
            ),
            encoding="utf-8",
        )
        self.zero = root / "zero.json"
        z = {
            "key_ii_blind": 5,
            "key_ii_blind_matches_attestation": True,
            "exact_match_in_train": 0,
            "contiguous_word_subsequence_of_a_train_utterance": 0,
            "subsequence_utterances": [SECRET],
        }
        self.zero.write_text(
            json.dumps(
                [
                    {
                        "scan": str(SCAN),
                        "val": z,
                        "heldout": z,
                        "computed_and_agrees_with_attestation": True,
                    }
                ]
            ),
            encoding="utf-8",
        )
        self.build_log = root / "build.log"
        self.build_log.write_text(
            "mac_heavy: ok\n      223.45 real       180.12 user        20.33 sys\n"
            "          2867456000  maximum resident set size\n",
            encoding="utf-8",
        )
        self.scan_log = root / "scan.log"
        self.scan_log.write_text(
            "      1234.00 real       180.12 user        20.33 sys\n"
            "          19000000000  maximum resident set size\n",
            encoding="utf-8",
        )
        self.out = root / "v5-preregistered.json"

    def argv(self, *extra: str, draft: Path = DRAFT) -> list[str]:
        return [
            "--draft",
            str(draft),
            "--repo",
            str(REPO),
            "--f-ledger",
            str(F_LEDGER),
            "--build-out",
            str(self.build_out),
            "--build-ledger",
            str(self.ledger),
            "--scan",
            str(SCAN),
            "--pool",
            str(POOL),
            "--pool-report",
            str(POOL_REPORT),
            "--fetch-record",
            str(FETCH),
            "--family-rates",
            str(self.rates),
            "--zero-checks",
            str(self.zero),
            "--build-log",
            str(self.build_log),
            "--scan-log",
            str(self.scan_log),
            "--gh200-state",
            "GH200 queue at 21:50Z: no tierb or fsucc word",
            "--out",
            str(self.out),
            *extra,
        ]


def run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = fz.main(argv)
    return rc, out.getvalue(), err.getvalue()


@unittest.skipUnless(all(p.exists() for p in NEEDED), f"needs the machine's real inputs: {NEEDED}")
class FreezeTest(unittest.TestCase):
    def setUp(self) -> None:
        (REPO_ROOT / "build").mkdir(exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="freeze-test-", dir=REPO_ROOT / "build"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_a_whole_v5_build_fills_every_item_and_copies_no_list(self) -> None:
        fx = Fixture(self.tmp)
        rc, out, err = run(fx.argv())
        self.assertEqual(rc, 0, err + out)
        before = json.loads(DRAFT.read_text(encoding="utf-8"))
        text = fx.out.read_text(encoding="utf-8")
        after = json.loads(text)
        self.assertNotIn("draft", after)
        self.assertNotIn(SECRET, text)
        old, new = before["amendments_pending"], after["amendments_pending"]
        self.assertEqual(len(new), 29)
        for i, (o, w) in enumerate(zip(old, new)):
            self.assertTrue(w.startswith(o) and len(w) > len(o), i)
        for k in before:
            if k not in ("draft", "amendments_pending", "amendments_applied"):
                self.assertEqual(after[k], before[k], k)
        # The draft key's line is the only line the rename removes from the DRAFT's text.
        self.assertEqual(
            len(text.splitlines()), len(DRAFT.read_text(encoding="utf-8").splitlines()) - 1
        )
        fill = [w[len(o) :] for o, w in zip(old, new)]
        self.assertIn("PASS", fill[11])
        self.assertIn("2.87 GB", fill[9])
        self.assertIn("containment runtime 1,234 s", fill[10])
        self.assertIn("code.defect_class 7 of 100", fill[2])
        self.assertIn("exact 0, contiguous word-subsequence 0", fill[15])
        self.assertIn("openjev.policy 5", fill[28])
        self.assertIn("exact-content drops 2 rows in 1 clusters", fill[28])
        self.assertIn(f"own-prose {fx.unit_form} 1", fill[3])
        self.assertIn("contrast knowledge.multiple_choice 1", fill[3])
        self.assertIn("commitpackft-g6 commit 1", fill[3])
        self.assertIn("in held-out: none", fill[4])
        self.assertIn("2 of 5 with-flag F seeds hold prose", fill[6])
        self.assertIn("GH200 queue at 21:50Z", fill[8])
        self.assertIn("ZefanCai/Open-Jev-v1.1: 1 train rows, 1,000 tokens", fill[24])
        self.assertIn("@10ad6888", fill[24])
        self.assertTrue(fill[13].startswith(" Pending after launch"))
        # The A7 reader's regexes (python/tests/test_v5_a7_check.py) still find the gate's numbers.
        for pat in (
            r"\(val: ([^;]+); held-out: ([^)]+)\)",
            r"CLINC val ([\d,]+) \(within_domain ([\d,]+)\) and held-out ([\d,]+) per family",
        ):
            self.assertEqual(re.search(pat, new[11]).groups(), re.search(pat, old[11]).groups())
        self.assertIn("differences from the DRAFT's stated expectations", out)
        self.assertIn("train oos rows", out)

    def test_a_share_below_two_thirds_of_v4s_refuses_and_names_the_number(self) -> None:
        fx = Fixture(self.tmp, decision_tokens=30000)
        rc, out, err = run(fx.argv())
        self.assertEqual(rc, 2)
        self.assertIn("item 24: code.defect_class's effective token share is", err)
        self.assertIn("55.80", err)
        self.assertFalse(fx.out.exists())

    def test_an_item_no_anchor_claims_refuses(self) -> None:
        fx = Fixture(self.tmp)
        d = json.loads(DRAFT.read_text(encoding="utf-8"))
        d["amendments_pending"].append("a new pending item")
        extra = self.tmp / "draft-30.json"
        extra.write_text(json.dumps(d, indent=2), encoding="utf-8")
        rc, _, err = run(fx.argv(draft=extra))
        self.assertEqual(rc, 2)
        self.assertIn("holds 30 items", err)
        d["amendments_pending"].pop()
        d["amendments_pending"][11] = "A7 reworded " + d["amendments_pending"][11]
        extra.write_text(json.dumps(d, indent=2), encoding="utf-8")
        rc, _, err = run(fx.argv(draft=extra))
        self.assertEqual(rc, 2)
        self.assertIn("anchor 11", err)

    def test_a_suffix_amended_after_this_script_still_anchors(self) -> None:
        fx = Fixture(self.tmp)
        d = json.loads(DRAFT.read_text(encoding="utf-8"))
        d["amendments_pending"][2] += " Amended later: the containment scope."
        later = self.tmp / "draft-later.json"
        later.write_text(json.dumps(d, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        rc, _, err = run(fx.argv(draft=later))
        self.assertEqual(rc, 0, err)
        self.assertIn(
            "Amended later: the containment scope. Filled ",
            json.loads(fx.out.read_text(encoding="utf-8"))["amendments_pending"][2],
        )

    def test_a7_refused_refuses(self) -> None:
        rc, _, err = run(Fixture(self.tmp, a7_passed=False).argv())
        self.assertEqual(rc, 2)
        self.assertIn("A7 REFUSED", err)

    def test_a_scan_the_build_did_not_apply_refuses(self) -> None:
        rc, _, err = run(Fixture(self.tmp, recipe_exclusions="0" * 64).argv())
        self.assertEqual(rc, 2)
        self.assertIn("is not the one this build applied", err)

    def test_an_own_prose_row_in_val_refuses(self) -> None:
        rc, _, err = run(Fixture(self.tmp, val_prose=True).argv())
        self.assertEqual(rc, 2)
        self.assertIn("own-prose or contrast rows are in data/pool/val.json", err)

    def test_an_existing_out_refuses_and_dry_run_writes_nothing(self) -> None:
        fx = Fixture(self.tmp)
        rc, out, err = run(fx.argv("--dry-run"))
        self.assertEqual(rc, 0, err)
        self.assertFalse(fx.out.exists())
        fx.out.write_text("{}", encoding="utf-8")
        rc, _, err = run(fx.argv())
        self.assertEqual(rc, 2)
        self.assertIn("exists", err)


if __name__ == "__main__":
    unittest.main()
