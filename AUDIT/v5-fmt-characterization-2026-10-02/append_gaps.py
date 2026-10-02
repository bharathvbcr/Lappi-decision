"""Resolve the two characterization-digest gaps through qd_train.gaps.append_gap.

One O_APPEND + fsync line per record, validated first; never a read-modify-write. The ledger's
last line per id is current, so these lines close the open records without touching them. A
record whose id is already current with the same status is skipped, so a second run appends
nothing.

Run from the worktree root:
    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python AUDIT/v5-fmt-characterization-2026-10-02/append_gaps.py
"""

from qd_train.gaps import append_gap, current_records

RESOLVED = "2026-10-02"
EVIDENCE = (
    "AUDIT/v5-fmt-characterization-2026-10-02/ (decode_diff.py, build_one.py, export_old.sh, "
    "run.sh, decode-and-diff-run.txt)"
)

RECORDS = [
    {
        "id": "GAP-L-V5FMT-CHARACTERIZATION-CRITERION-STALE-2026-10-02",
        "opened": "2026-10-02",
        "resolved": RESOLVED,
        "lane": "lead",
        "owner": "lead",
        "status": "resolved",
        "tool": "build_one.py: the 60-pair build from a git archive of 92e1c74 and from v5-build "
        "f7ae29d, under tools/mac_heavy.sh; decode_diff.py: each sequence decoded through its "
        "shard's remap and the pinned tokenizer",
        "question": "Is GAP-L-V5DATA-CHARACTERIZATION-DIGEST-MOVES-2026-10-02's refresh criterion "
        "still the right one after the prompt-format change?",
        "answer": "Applied as this record's action asked: decode both builds and confirm that the "
        "token delta is exactly the format line plus the moved question line. [V, run] The "
        "92e1c74 build digests to 92299328... and the v5-build one to 3c9ae50a.... Every one of "
        "the 86 train, 86 train-no-decode and 12 val sequences decodes to the old text with "
        "the format-2 line inserted after the begin line and the question line moved after the "
        "context end; there is no other edit. Each is 9 tokens longer. target_index shifts by "
        "exactly that delta. slot_kind, span_target, candidate_offsets and candidate_positions "
        "are identical. remap, sequence_index, coverage, contradictions, span_check and the "
        "pool manifests are byte-identical. Each header differs only in: buckets and "
        "max_seq_len (+9), total_tokens, prompt_format 2, the render.py and defect_class.py "
        "fingerprints, shard_hash and created_at. Unexplained: 0. The pin in "
        "python/tests/test_containment_exclusions.py is now 3c9ae50a....",
        "evidence": EVIDENCE,
    },
    {
        "id": "GAP-L-V5DATA-CHARACTERIZATION-DIGEST-MOVES-2026-10-02",
        "opened": "2026-10-02",
        "resolved": RESOLVED,
        "lane": "lead",
        "owner": "lead",
        "status": "resolved-with-residual",
        "tool": "the decode-and-diff of GAP-L-V5FMT-CHARACTERIZATION-CRITERION-STALE-2026-10-02; "
        "git diff --stat 8d41faa...faaf8d4 (L-v5-queue, not yet merged)",
        "question": "Does python/tests/test_containment_exclusions.py::test_a_build_without_the_"
        "flag_writes_what_it_wrote_before still hold on the v5 build commit?",
        "answer": "Pin refreshed d67c7de0... -> 3c9ae50a..., after both explained steps. "
        "Step 1, L-v5-data -> 92299328.... [V, run] A fresh build at 92e1c74 reproduces this "
        "record's 92299328... exactly. The byte-level account of that step is this record's "
        "own diff against 541d423 (tokens, offsets and supervision identical; only "
        "fingerprints, admitted sources and derived hashes moved), not rerun by the lead. "
        "Step 2, prompt format 2 -> 3c9ae50a...: fully decoded, with 0 unexplained. "
        "Residual: L-v5-queue (faaf8d4) has not merged. Its diff touches only "
        "campaign/post-f-queue, python/tests, AUDIT, HANDOFF and gaps.jsonl, so it cannot move "
        "this build [V, diff --stat]. Any later lane that edits qd_data, qd_train or the "
        "pipeline will move the digest again, and the test fails loud.",
        "evidence": EVIDENCE,
    },
]


def main() -> None:
    current = current_records()
    for rec in RECORDS:
        now = current.get(rec["id"])
        if now is not None and now.get("status") == rec["status"]:
            print(f"skip {rec['id']}: already {rec['status']}")
            continue
        append_gap(rec)
        print(f"appended {rec['id']} ({rec['status']})")


if __name__ == "__main__":
    main()
