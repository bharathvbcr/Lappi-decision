"""Append lane L-v5-freeze's three gap records (2026-10-03), once, through qd_train.gaps.

    python3 AUDIT/finalize-2026-10-03/append_gaps_v5_freeze.py

append_gap validates each record and appends one line with O_APPEND and fsync; it never
rewrites the file. A second run refuses before writing, because the ids already exist.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_train.gaps import append_gap, current_records  # noqa: E402

RECORDS = [
    {
        "id": "GAP-V5-FREEZE-PRELUDE-DIGESTS-CANNOT-LIVE-IN-THE-LANE-COMMIT-2026-10-03",
        "opened": "2026-10-03",
        "lane": "L-v5-freeze",
        "owner": "lead",
        "status": "resolved",
        "tool": (
            "read: campaign/post-f-queue/v5_prelude_mac.py and v5_common.sh at v5-build "
            "ca48960"
        ),
        "question": (
            "amendments_pending items 7 (the Mac prelude's batches and width) and 18 "
            "(per-seed corpus.plan_order_digest) ask for values the Mac prelude prints. Can "
            "the freeze write them into campaign/v5-preregistered.json?"
        ),
        "answer": (
            "No. The prelude runs at the lane commit L (V5_LANE_AT), and L must already "
            "carry the renamed file (binds_iff; v5_common.sh "
            "V5_PREREG=$V5_LANE/campaign/v5-preregistered.json), so its output cannot be in "
            "the commit it runs at. Resolved by wording, per Fable's freeze ruling (~20:30Z, "
            "via the lead): those items say the values are recorded in the prelude record "
            "(sha256 pinned at deploy as V5_PRELUDE_SHA256, checked by v5_common.sh "
            "v5_prelude_check against V5_LANE_AT) and copied in by the first post-launch "
            "amendment, before seed 0's ft row. Two parts of the items are not in that "
            "record at all (v5_prelude_mac.py's record dict carries no needle field and no "
            "recipe hash): item 5's needle suite digest and min/median/max "
            "needle_suite_tokens come from seed 0's epoch-score-val row, and the digest is "
            "on no row (GAP-V5-RULES-REBUILT-SUITE-NOT-ON-THE-ROW-2026-10-02); item 7's "
            "recipe hash is seed 0's ft row's protocol.recipe_hash. "
            "AUDIT/finalize-2026-10-03/apply_v5_freeze.py writes those two as pending after "
            "launch."
        ),
    },
    {
        "id": "GAP-L-V5-FREEZE-NAVIGATION-DEVMAP-MALFORMED-NO-LISTAGENTS-2026-10-03",
        "opened": "2026-10-03",
        "lane": "L-v5-freeze",
        "owner": "lead",
        "status": "open",
        "tool": (
            "devmap_status (MCP, repo_path /Users/bharath/Code/research/Lappi-decision, "
            "~20:20Z); gitpulse_insights (this lane's worktree); ToolSearch for ListAgents"
        ),
        "question": (
            "Could lane L-v5-freeze find the DRAFT's readers with DevMap, and check for a "
            "peer in its own worktree with ListAgents, before writing?"
        ),
        "answer": (
            "No. devmap_status returned 'database disk image is malformed', broader than "
            "GAP-DEVMAP-FTS5-CORRUPT-SEARCH-REFUSES-2026-10-03's FTS error. GitPulse "
            "Insights' codeintel facet reported no DevMap database at this worktree's path "
            "(available false); its worktrees, agents, changes and collisions facets were "
            "ok. No ListAgents tool exists in this session. The reader list in "
            "HANDOFF/v5-freeze-2026-10-03.md was made with `git grep -n v5-preregistered "
            "v5-build -- python crates campaign tools`: tracked files only, NOT "
            "graph-confirmed; untracked and gitignored files in build/v5-build-wt were not "
            "searched. The lane wrote only in its own worktree (branch l-v5-freeze) and a "
            "gitignored build/ scratch export of v5-build."
        ),
    },
    {
        "id": "GAP-V5-FREEZE-PER-FAMILY-EXCLUSIONS-AND-RUNTIMES-ON-NO-BUILD-RECORD-2026-10-03",
        "opened": "2026-10-03",
        "lane": "L-v5-freeze",
        "owner": "lead",
        "status": "open",
        "tool": (
            "read: tools/real_tokenizer_pipeline.py and python/qd_train/exclusions.py at "
            "v5-build ca48960; /Users/bharath/qd-campaign/v5-containment-v2-2026-10-03/"
            "attestation.json"
        ),
        "question": (
            "Do the v5 build's own records carry what amendments_pending items 2, 9, 10 and "
            "15 ask for: the excluded-key count per source family, the containment runtime, "
            "the build's peak RSS, the CLINC rate on the full scan and the two "
            "key-(ii)-blind zero-checks?"
        ),
        "answer": (
            "No. The attestation has hits_by_source_family (train rows hit per family and "
            "target), n_exclusions and the key_ii_blind counts, but no excluded-key count "
            "per family and no runtime. The build row has decontam_exclusion (a total) and "
            "wall_clock_s, and no peak RSS. The per-family counts and the CLINC rate come "
            "from AUDIT/prep2-2026-10-02/family_rates.py, and the zero-checks from "
            "AUDIT/prep3-2026-10-02/key_ii_blind_zero_checks.py. Both read the scan's "
            "request.bin, which carries held-out text, so the freeze script does not run "
            "them. The runtime and the peak RSS come only from /usr/bin/time -l logs. "
            "apply_v5_freeze.py takes --family-rates, --zero-checks, --build-log and "
            "--scan-log, and without them writes 'Not filled: <reason>' and lists the item "
            "in its summary."
        ),
    },
]

known = current_records()
clash = [r["id"] for r in RECORDS if r["id"] in known]
if clash:
    sys.exit(f"already recorded: {clash}")
for r in RECORDS:
    append_gap(r)
    print(f"appended {r['id']}")
