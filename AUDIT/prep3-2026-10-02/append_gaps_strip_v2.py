"""Lane L-prep3's gap records for strip version 2, appended once through qd_train.gaps.append_gap.

Kept as the record of exactly what was appended. Each is an update of an existing id (the
last line per id is current). It refuses to append any of them twice from this lane.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qd_train.gaps import append_gap, read_gaps

LANE = "L-prep3"
EVIDENCE = (
    "AUDIT/prep3-2026-10-02/{pass-check-v2.json, zero-checks-subsample.json, "
    "side-by-side.json, subsample-reproduction.json, oracle-crosscheck-v2.json}; "
    "HANDOFF/prep3-2026-10-02.md"
)

RECORDS = [
    {
        "id": "GAP-CONTAINMENT-KEY-II-BLIND-SHORT-UTTERANCES-2026-10-02",
        "opened": "2026-10-02",
        "lane": LANE,
        "owner": "lead",
        "status": "open",
        "tool": (
            "qd_train.containment_strip.strip_template version 2 (a81d1a7) via "
            "tools/containment_scan.py on the 1/2/5% subsamples (rev 881ab304, L-prep2's corpus "
            "flags); AUDIT/prep3-2026-10-02/key_ii_blind_zero_checks.py over each request.bin"
        ),
        "question": (
            "Under strip version 2, every intent.* slot text is the bare utterance, and CLINC "
            "utterances under 8 words are invisible to key (ii). How many are there, and do the "
            "two zero-checks of Fable's CLINC strip ruling section 2 hold for the val and "
            "held-out ones?"
        ),
        "answer": (
            "Implemented and measured on the subsamples; the full scan is pending (the lead "
            "runs it; HANDOFF/prep3-2026-10-02.md has the command). Attestation v2 now records "
            "export.template_strip.key_ii_blind per set (n, sha256 and the complete sorted "
            "identity-key list) and too_short_after_strip per family-slot and per set. "
            "key_ii_blind (identity keys) train/val/heldout: 1% 102/4/2, 2% 197/9/7, 5% "
            "451/21/34 [V], equal to each intent.* family-slot's too_short_after_strip and to "
            "Fable's oracle; every blind key is a CLINC key. The two zero-checks over the "
            "key-(ii)-blind val and held-out keys: 0 have a lower-cased \\w+ exact match in a "
            "CLINC train utterance and 0 are a contiguous word-subsequence of one, at 1/2/5% "
            "[V; the script's matcher found a planted positive control]. Those keys are "
            "unprotected by key (ii), covered by keys (i) and (iii); never clean. Stays open "
            "until the full scan's key_ii_blind counts and both zero-checks are reported; a "
            "nonzero there is a GAP for the human, not a rule change."
        ),
        "evidence": EVIDENCE,
    },
    {
        "id": "GAP-CONTAINMENT-STRIP-V1-WINDOW-2026-10-02",
        "opened": "2026-10-02",
        "lane": LANE,
        "owner": "lead",
        "status": "resolved",
        "tool": (
            "qd_train.containment_strip (STRIP_VERSION 2, STRIP_RULE naming the amendment) and "
            "qd_train.exclusions.read_exclusions, a81d1a7; "
            "python/tests/test_containment_exclusions.py::"
            "test_a_version_1_strip_attestation_is_refused_by_the_hook"
        ),
        "question": (
            "Until STRIP_VERSION 2 lands, a version-1 full-scan list would pass "
            "qd_train.exclusions.read_exclusions. What keeps a version-1 list out of a build?"
        ),
        "answer": (
            "The code, from a81d1a7: STRIP_VERSION is 2 and STRIP_RULE names Fable's CLINC "
            "strip ruling section 1; read_exclusions compares applied, version and rule exactly "
            "and refuses a version-1 attestation, a version-2 one under the version-1 rule, and "
            "a version-1 one under the new rule. The test fails against 6e62363 (DID NOT RAISE "
            "on the version-1 attestation) and passes at a81d1a7. The window closes on main "
            "when the lead merges this branch; until then main's hook still accepts a version-1 "
            "list, so the procedural rule (no full scan, no list applied to a build) holds up "
            "to that merge. No full scan was run by this lane."
        ),
        "evidence": EVIDENCE,
    },
    {
        "id": "GAP-CONTAINMENT-CONSTANT-QUESTION-DRIVES-CLINC-HITS-2026-10-02",
        "opened": "2026-10-02",
        "lane": LANE,
        "owner": "human",
        "status": "open",
        "tool": (
            "qd_train.containment_strip version 2 (a81d1a7) via tools/containment_scan.py on "
            "the 1/2/5% subsamples (byte-identical to L-prep2's: every count and sha256), rev "
            "881ab304; AUDIT/prep3-2026-10-02/check_pass.py"
        ),
        "question": (
            "Does word 8-gram containment >= 0.5 over rendered prompt_content find real overlap "
            "in every family, or does each family's constant question make short rows hit each "
            "other?"
        ),
        "answer": (
            "Update 2026-10-02, version 2 measured (L-prep3): Fable's CLINC strip ruling "
            "section 4 PASSES at all three sizes [V, check_pass.py exit 0]. CLINC (intent.*) "
            "excluded keys 0/246, 0/464, 0/1089 (version 1: 155, 293, 721; unstripped: 8, 39, "
            "338), with 0 enforced pairs carrying an intent.* row and none in any scan. The "
            "non-intent subset of exclusions.txt is byte-identical to version 1's (8/16/41 "
            "keys, the ruling's three sha256), and so are all non-intent pair rows (32/64/162). "
            "Every intent.* row was checked equal to its context block; too_short_after_strip "
            "per intent.* family-slot and key_ii_blind 102/4/2, 197/9/7, 451/21/34; CLEAN; "
            "identity_disjoint, near_duplicate_disjoint and repo_disjoint ran and passed. "
            "Pairs 32/64/162, identity keys excluded 8/16/41 (all code.defect_class, plus one "
            "MMLU key at 5%). Fable's oracle on the same requests agrees (0 CLINC keys, the same "
            "too-short counts). Stays open: the full v5 scan has not run, and the human "
            "ratifies by the commit that renames the DRAFT."
        ),
        "evidence": EVIDENCE,
    },
]

existing = read_gaps()
for record in RECORDS:
    if any(r.get("id") == record["id"] and r.get("lane") == LANE for r in existing):
        sys.exit(f"{record['id']} already has an {LANE} line in gaps.jsonl; refusing a duplicate")
for record in RECORDS:
    append_gap(record)
    print(f"appended {record['id']}")
