"""The lead's gap records for Fable's CLINC strip ruling (2026-10-02), appended once through
qd_train.gaps.append_gap. Refuses to run twice: it stops if any new id already exists or an
updated id is no longer open.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qd_train.gaps import append_gap, read_gaps  # noqa: E402

RULING = "AUDIT/prep2-2026-10-02/fable-clinc-strip-ruling.md"
current = {}
for r in read_gaps():
    current[r.get("id")] = r

NEW = [
    {
        "id": "GAP-CONTAINMENT-KEY-II-BLIND-SHORT-UTTERANCES-2026-10-02",
        "opened": "2026-10-02",
        "lane": "fable-clinc",
        "owner": "the lane implementing STRIP_VERSION 2",
        "status": "open",
        "tool": f"Fable read-only lane ({RULING}); Fable's Python oracle AUDIT/prep2-2026-10-02/clinc_oracle.py",
        "question": (
            "Under strip version 2, every intent.* slot text is the bare utterance. At 5%, 451/1089 train, "
            "21/69 val and 34/64 held-out CLINC keys fall under 8 words, so key (ii) (8-gram containment) "
            "cannot see them. Are they protected?"
        ),
        "answer": (
            "Covered by keys (i), exact identity/content, and (iii), MinHash over the bare utterance at k = 5 "
            "tokens with Jaccard >= 0.8 across intent splits. Fable's oracle: 0 of the key-(ii)-blind val and "
            "held-out utterances has a lower-cased exact match in train, and 0 is a contiguous "
            "word-subsequence of a train utterance, at 1/2/5% [V per the ruling, Python oracle; qd-prep "
            "parity not shown]. No new key; Fable: 'A key specified after measuring its population at zero is "
            "a story, not protection'. Pending: the full scan repeats both zero-checks and reports "
            "export.template_strip.key_ii_blind per set. Those keys are named 'unprotected by key (ii), "
            "covered by (i) and (iii)', never clean. A nonzero result goes to the human as a GAP, not as a "
            "rule change."
        ),
        "evidence": RULING + " sections 2 and 4",
    },
    {
        "id": "GAP-CONTAINMENT-STRIP-V1-WINDOW-2026-10-02",
        "opened": "2026-10-02",
        "lane": "fable-clinc",
        "owner": "lead",
        "status": "open",
        "tool": f"Fable read-only lane ({RULING} section 5)",
        "question": (
            "L-prep2's version-1 strip is merged on main. Until STRIP_VERSION 2 lands, a version-1 full-scan "
            "list would pass qd_train.exclusions.read_exclusions (exclusions.py:132-137). What keeps a "
            "version-1 list out of a build?"
        ),
        "answer": (
            "Procedure only, until version 2 lands: no full scan and no exclusion list applied to any build "
            "in the window. The implementing lane bumps both STRIP_VERSION (to 2) and STRIP_RULE (to name "
            "the amendment); the hook compares both exactly, which closes the window."
        ),
        "evidence": RULING + " section 5",
    },
    {
        "id": "GAP-FABLE-CLINC-RULING-NAVIGATION-2026-10-02",
        "opened": "2026-10-02",
        "lane": "fable-clinc",
        "owner": "lead",
        "status": "open",
        "tool": f"Fable read-only lane ({RULING})",
        "question": "Was Fable's CLINC strip ruling navigated through DevMap and GitPulse?",
        "answer": (
            "No. Fable read everything directly: DevMap is degraded and GitPulse was not consulted. Its "
            "oracle is Python (AUDIT/prep2-2026-10-02/clinc_oracle.py); qd-prep parity for the oracle's "
            "counts is not shown. The implementing lane's version-2 subsample scans are the qd-prep "
            "measurement."
        ),
        "evidence": RULING,
    },
]
UPDATES = {
    "GAP-CONTAINMENT-CONSTANT-QUESTION-DRIVES-CLINC-HITS-2026-10-02": {
        "status": "open",
        "answer_append": (
            " Update 2026-10-02 (Fable's CLINC strip ruling, " + RULING + "): the version-1 family-wide strip "
            "raised CLINC exclusion to 63.0/63.1/66.2% at 1/2/5% (HANDOFF/prep2-2026-10-02.md), all of it "
            "from intent.within_domain's per-domain option lists. Fable chose (2b): in the four intent.* "
            "families every option value is stripped, because the labels come from CLINC's closed "
            "vocabulary, so an intent.* row is its utterance. Fable's oracle predicts 0/0/0 CLINC keys at "
            "1/2/5%. Version 2 is pending its implementing lane and the section 4 pass condition. The DRAFT's "
            "rule was amended by AUDIT/prep2-2026-10-02/apply_clinc_strip_amendment.py. Stays open; the "
            "human ratifies by the rename commit."
        ),
    },
    "GAP-CONTAINMENT-STRIP-RULE-PREMISES-2026-10-02": {
        "status": "resolved",
        "answer_append": (
            " Resolved 2026-10-02 by Fable's CLINC strip ruling (" + RULING + " section 1). The lane's four "
            "readings are adopted in the amended rule text: the question line is constant in every family; "
            "the per-domain lists are intent.within_domain's, not intent.classification's (Fable corrected "
            "its own caveat); a group of fewer than 2 rows strips nothing and is named; constancy means "
            "membership at seed=None."
        ),
    },
}

for g in NEW:
    if g["id"] in current:
        sys.exit(f"{g['id']} already exists")
for gid, u in UPDATES.items():
    if gid not in current:
        sys.exit(f"{gid} missing")
    if current[gid].get("status") != "open":
        sys.exit(f"{gid} is {current[gid].get('status')}, not open")
for g in NEW:
    append_gap(g)
    print("appended", g["id"])
for gid, u in UPDATES.items():
    rec = dict(current[gid])
    rec["status"] = u["status"]
    rec["answer"] = (rec.get("answer") or "") + u["answer_append"]
    if u["status"] == "resolved":
        rec["resolved"] = "2026-10-02"
    rec["evidence"] = ((rec.get("evidence") or "") + "; " + RULING).lstrip("; ")
    append_gap(rec)
    print("updated", gid, "->", u["status"])
