"""The lead's records for Fable's seed-order ruling (2026-10-02).

AUDIT/post-f-2026-10-02/fable-seed-order-ruling.md section 5 asks that the box's digest read be
recorded in the two order gaps as the confirming evidence; section 3 strengthens the bimodality
statement and names one unverified blind spot (J4's order). Appended once through
qd_train.gaps.append_gap; refuses a second run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qd_train.gaps import append_gap, read_gaps

RULING = "AUDIT/post-f-2026-10-02/fable-seed-order-ruling.md"
DIGEST = "1f38209195866e8e4b60565ae565ae1dfd46637f1d562d23b14c09c0be32af26"
BOX = (
    "the lead's read-only ssh, 2026-10-02 ~17:55 UTC: /home/ubuntu/ckpt/p4-v4/"
    "epoch-seed{0,1,2,3}-cuda.json all carry consumed_digest " + DIGEST + " (seed 3's "
    "optimizer_step 9683), equal to the plan L-recency rebuilt on the Mac"
)
DIGEST_GAP = "GAP-F-FT-ROWS-CARRY-NO-ORDER-SENSITIVE-PLAN-DIGEST-2026-10-02"
ORDER_GAP = "GAP-REAL-FT-RUN-BATCH-ORDER-IGNORES-THE-TRAINING-SEED-2026-10-02"
BIMODAL_GAP = "GAP-OOD-PROSE-ABSTENTION-BIMODAL-ACROSS-SEEDS-2026-10-02"

done = {(r.get("id"), r.get("tool")) for r in read_gaps()}
if (DIGEST_GAP, "ssh read of F's checkpoints; " + RULING) in done:
    sys.exit("already appended")

append_gap(
    {
        "id": DIGEST_GAP,
        "opened": "2026-10-02",
        "resolved": "2026-10-02",
        "lane": "lead",
        "owner": "lead",
        "status": "resolved",
        "tool": "ssh read of F's checkpoints; " + RULING,
        "question": "Can F's batch order be checked against anything F recorded?",
        "answer": (
            "Yes, on the box: " + BOX + ". That confirms the rebuilt order, the one-order reading, "
            "and that the box's numpy permuted as the Mac's 2.5.0 did. Forward fix (Fable section "
            "1): v5 ft rows carry corpus.plan_seed, corpus.plan_order_digest and "
            "train.consumed_digest as metrics (lane L-v5-train)."
        ),
        "evidence": RULING + " section 5; HANDOFF/recency-2026-10-02.md",
    }
)
append_gap(
    {
        "id": ORDER_GAP,
        "opened": "2026-10-02",
        "lane": "lead",
        "owner": "L-v5-train",
        "status": "open",
        "tool": "ssh read of F's checkpoints; " + RULING,
        "question": (
            "Does the training seed set the batch order in real_ft_run's epoch arm, and is one "
            "order for all seeds wanted for v5?"
        ),
        "answer": (
            "Confirmed on the box (" + BOX + "): every F seed trained one order. Fable ruled (b) "
            "for v5: --batch-order seed (recipe key batch_order = 'seed', the constant string; "
            "plan seed = the training seed, only into reader.batches), paired across v5, "
            "seeds 3-4, "
            "the arm and J5'; the DRAFT is amended by "
            "AUDIT/post-f-2026-10-02/apply_seed_order_amendment.py. Open until L-v5-train's flag, "
            "its fails-pre-fix test and the three comment corrections (backbone.py:1064, "
            "trainer.py:692, artifacts.py:604-606) land on v5-build. F's own rows are bound and "
            "unchanged: absent flag = today's behaviour byte for byte."
        ),
        "evidence": RULING + " sections 1 and 4",
    }
)
append_gap(
    {
        "id": BIMODAL_GAP,
        "opened": "2026-10-02",
        "lane": "lead",
        "owner": "lead",
        "status": "open",
        "tool": RULING,
        "question": "Does the one-order finding change the bimodality reading?",
        "answer": (
            "Report-only (Fable section 3): strengthened, not changed -- 'the recipe and the data "
            "order do not determine the outcome': with one order (" + BOX + "), init plus kernel "
            "nondeterminism alone separate seed 0's 54/60 from seeds 1-2's 0/60 and 1/60. Blind "
            "spot, unverified: '2 of 4 with flag, 0 of 3 without' is clean only if J4's three "
            "seeds also shared one order at their commits; a report-only CPU check, not a blocker. "
            "Also report-only: F's seeds 3-4 rule fired at 0.344 with no order variance, and the "
            "fsucc/F' envelope excludes order variance (a tighter, less general test than its text "
            "implies; the room arithmetic stands)."
        ),
        "evidence": RULING + " section 3; HANDOFF/recency-2026-10-02.md",
    }
)
print("appended 3 records")
