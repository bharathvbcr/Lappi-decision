"""Record the per-invocation GPU idle of tools/real_ft_run.py's CPU startup through
qd_train.gaps.append_gap: one O_APPEND + fsync line, validated first, never a read-modify-write.
Skipped if the current record already carries this answer.

Run from the repo root:
    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python \
        AUDIT/gpu-idle-preamble-2026-10-03/append_gap.py
"""

from qd_train.gaps import append_gap, current_records

GID = "GAP-BOX-GPU-IDLE-REAL-FT-RUN-PREAMBLE-2026-10-03"
QUESTION = (
    "Where does the box GPU idle between queued jobs, and is it a Python bottleneck worth a "
    "Rust port?"
)
ANSWER = (
    "[V, measured] From 2026-10-01 13:20 to 2026-10-03 00:37 (~35 h), gpu.csv shows 3-10 min "
    "idle gaps (util < 10%) at nearly every tools/real_ft_run.py start. They sum to ~152 min "
    "(~7%, ~$5.80 at $2.29/h; AUDIT/gpu-idle-preamble-2026-10-03/idle-gaps.txt). J7' "
    "decomposed: ~318 s of single-threaded startup before the first GPU sample >= 50%. Of "
    "that, 48.2 s is qd-prep minhash+LSH (Rust); ~4.5 min is Python, not yet profiled "
    "(val-set build, permutation pass, needle/OOD suites, and a train shard-set pass over "
    "289,142 rows / 306.9 M tokens in a run that only scores). The averaging before J7' was "
    "84 s + 140 s of Python plus 30 s of Rust: once per averaging point, no port. [I] The "
    "fix is a skip or a content-keyed cache at the canonical owner if the cost is a redundant "
    "pass, and a Rust port with a benchmark only if it is real tokenization/batching work. "
    "Launched waiters are never edited, so a fix can land only in v5's lane commit "
    "(V5_LANE_AT unset). Needs: a stdlib cProfile of the startup (on the box, a launch: "
    "human yes) and the human's call on whether ~$8-10 over v5 is worth a lane."
)


def main() -> None:
    now = current_records().get(GID)
    if now is not None and now.get("answer") == ANSWER:
        print(f"skip {GID}: already recorded")
        return
    append_gap(
        {
            "id": GID,
            "opened": "2026-10-03",
            "lane": "lead",
            "owner": "lead",
            "status": "open",
            "tool": "box gpu.csv (nvidia-smi, 15 s) and queue logs over read-only ssh; advisor",
            "question": QUESTION,
            "answer": ANSWER,
        }
    )
    print(f"appended {GID} (open)")


if __name__ == "__main__":
    main()
