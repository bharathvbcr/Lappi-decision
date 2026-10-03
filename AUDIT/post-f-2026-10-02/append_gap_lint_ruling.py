"""Record Fable's ruling on the lint gate's pinned files through qd_train.gaps.append_gap.

One O_APPEND + fsync line, validated first; never a read-modify-write. The gap stays open: the
three AUDIT records are now ignored rule by rule in pyproject.toml, and the two oracle scripts
wait for a qd-train owner. Skipped if the current record already carries this answer.

Run from the worktree root:
    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python \
        AUDIT/post-f-2026-10-02/append_gap_lint_ruling.py
"""

from qd_train.gaps import append_gap, current_records

GID = "GAP-L-LINT-PINNED-FILES-KEEP-THE-GATE-RED-2026-10-02"
ANSWER = (
    "Fable ruled (AUDIT/post-f-2026-10-02/fable-v5-queue-readings-ruling.md, item 1). The "
    "three AUDIT records are evidence, so they are not reformatted: pyproject.toml "
    "[tool.ruff.lint.per-file-ignores] now ignores, rule by rule, only what each already breaks "
    "(recency_diagnostic.py E501/I001, clinc_oracle.py E702/B007, append_gaps_recency.py E501), "
    "with a comment naming this gap. The two oracle scripts "
    "(tools/qd_train_oracle_optimizer_table.py, 6 findings; tools/qd_train_oracle_head_digest.py, "
    "4) are NOT ignored. A qd-train owner reformats them and regenerates their fixtures, and "
    "that holds only if every fixture field except script_sha256 comes out byte-identical. If "
    "anything else moves, stop and record it. That work is off the v5 critical path. [V, run] "
    "ruff 0.16.8 on v5-build after the ignores: 10 findings, all in those two scripts (36 "
    "before: these 31, plus 5 in the lead's new AUDIT/v5-fmt-characterization-2026-10-02 "
    "scripts, now fixed)."
)


def main() -> None:
    now = current_records().get(GID)
    if now is not None and now.get("answer") == ANSWER:
        print(f"skip {GID}: already recorded")
        return
    append_gap(
        {
            "id": GID,
            "opened": "2026-10-02",
            "lane": "lead",
            "owner": "lead",
            "status": "open",
            "tool": "advisor (Fable) consult; ruff 0.16.8 (.venv) over the v5-build worktree",
            "question": now["question"] if now else "The lint gate's pinned files: ignore or fix?",
            "answer": ANSWER,
        }
    )
    print(f"appended {GID} (open)")


if __name__ == "__main__":
    main()
