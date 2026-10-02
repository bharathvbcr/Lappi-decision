"""Apply Fable's seed-order ruling to the v5 DRAFT (2026-10-02).

The ruling is AUDIT/post-f-2026-10-02/fable-seed-order-ruling.md. F's seeds trained on one batch
order (L-recency, HANDOFF/recency-2026-10-02.md; box consumed_digest 1f382091 on seeds 0-3), and
Fable ruled (b): v5 varies the order with the training seed, paired across v5, the arm and J5'.
The lead ran this once, before the DRAFT binds. Its changes:
- recipe.added gains `--batch-order seed` (section 1: the flag, the constant recipe key, the
  derivation and the three ft-row metrics);
- seeds.seeds_3_4, j5prime.what and readings.R4_lower_layers gain the section 2/3 sentences;
- arm_noul_weight.confounds and arm_noul_weight.identity.ft_rows gain the pairing (section 2);
- amendments_pending gains the two section 5 items;
- amendments_applied is appended.
No threshold moves (section 2). Textual insertion keeps the file's layout; a second run fails,
because the flag would already be in recipe.added.
"""

import json
import sys
from pathlib import Path

DRAFT = Path(__file__).resolve().parents[2] / "campaign" / "v5-preregistered.DRAFT.json"
RULING = "AUDIT/post-f-2026-10-02/fable-seed-order-ruling.md"
text = DRAFT.read_text(encoding="utf-8")
d = json.loads(text)
if any(a.get("flag") == "--batch-order seed" for a in d["recipe"]["added"]):
    sys.exit("already applied")


def q(v):
    return json.dumps(v, ensure_ascii=True)


def replace_once(old, new):
    global text
    if text.count(old) != 1:
        sys.exit(f"anchor not found exactly once: {old[:80]!r}")
    text = text.replace(old, new)


def append_to(value, sentence):
    replace_once(q(value), q(value + sentence))


added = {
    "flag": "--batch-order seed",
    "why": (
        f"Fable's seed-order ruling ({RULING} section 1, option (b)): F's seeds 0-3 trained on one "
        "batch order (the plan is built once at DataConfig().seed; box consumed_digest 1f382091 on "
        "/home/ubuntu/ckpt/p4-v4/epoch-seed{0,1,2,3}-cuda.json; "
        "GAP-REAL-FT-RUN-BATCH-ORDER-IGNORES-THE-TRAINING-SEED-2026-10-02), so a seed envelope "
        "over "
        "one order is a claim about that order; v5's envelope includes order variance"
    ),
    "code": (
        "needed (L-v5-train): absent = today's behaviour byte for byte; recipe key batch_order = "
        "'seed' written by _recipe_pieces only when given, the constant string and never the "
        "numeric seed, so every v5 ft row shares one recipe_hash (one_configuration, "
        "qd_post_f_rules.rs:2869-2906); plan seed = the training seed, passed only to "
        "reader.batches(seed=...), never through config.seed (data_snapshot_hash, suites, "
        "alphabets, inventory); plan_all, labels_by_batch_all, keep, plan_small and C1's "
        "epoch_alphabets move inside the seed loop, one plan at a time; the test fails pre-fix"
    ),
    "metrics": (
        "ft-row metrics, not recipe: corpus.plan_seed, corpus.plan_order_digest (sha256 over the "
        "plan's (bucket, rows) tuples in order) and train.consumed_digest copied from the final "
        "checkpoint (closes GAP-F-FT-ROWS-CARRY-NO-ORDER-SENSITIVE-PLAN-DIGEST-2026-10-02 forward)"
    ),
    "pairing": (
        "every v5 run carries it: v5 seeds 0-2, seeds 3-4, the noul-weight arm and J5'; arm seed s "
        "and J5' seed s train on v5 seed s's order"
    ),
}
replace_once(
    '\n    ],\n    "conditionals": [',
    ",\n      " + q(added) + '\n    ],\n    "conditionals": [',
)

append_to(
    d["seeds"]["seeds_3_4"],
    f" Seeds 3 and 4 run with --batch-order seed, plan seeds 3 and 4 ({RULING} section 2).",
)
append_to(
    d["j5prime"]["what"],
    f" J5' seed s trains on v5 seed s's batch order (--batch-order seed; {RULING} section 2).",
)
append_to(
    d["readings"]["R4_lower_layers"],
    " F seeds 0-4 share one batch order (GAP-REAL-FT-RUN-BATCH-ORDER-IGNORES-THE-TRAINING-SEED-"
    "2026-10-02), so the count measures init and kernel variance at one order "
    f"({RULING} section 3).",
)

arm = d["arm_noul_weight"]
old_conf = arm["confounds"]
marker = "Only the loss weight differs from v5, seed for seed;"
if marker not in old_conf:
    sys.exit("confounds marker not found")
new_conf = old_conf.replace(
    marker,
    "Only the loss weight differs from v5, seed for seed, including the batch order (plan seed s "
    f"on both sides, --batch-order seed; {RULING} section 2);",
)
replace_once(q(old_conf), q(new_conf))

old_ft = arm["identity"]["ft_rows"]
tail = " Any other difference refuses."
if not old_ft.endswith(tail):
    sys.exit("identity.ft_rows tail not found")
new_ft = old_ft[: -len(tail)] + (
    "; corpus.plan_order_digest equal to v5's ft row of the same seed, absent on either side "
    f"refuses (the pairing check, {RULING} section 2)." + tail
)
replace_once(q(old_ft), q(new_ft))

pending = d["amendments_pending"]
last = pending[-1]
replace_once(
    q(last) + '\n  ],\n  "build_order"',
    q(last)
    + ",\n    "
    + q(
        "per-seed corpus.plan_order_digest as the Mac prelude printed them (" + RULING
        + " section 5)"
    )
    + ",\n    "
    + q(
        "the batch_order key present on every v5, seeds 3-4, arm and J5' ft row (" + RULING
        + " section 5)"
    )
    + '\n  ],\n  "build_order"',
)

applied = d["amendments_applied"]
append_to(
    applied,
    " Then on 2026-10-02 (~18:00 UTC), Fable's seed-order ruling (" + RULING + "; option (b), "
    "no threshold moves) was applied by AUDIT/post-f-2026-10-02/apply_seed_order_amendment.py: "
    "recipe.added --batch-order seed, seeds.seeds_3_4, j5prime.what, readings.R4_lower_layers, "
    "arm_noul_weight.confounds and identity.ft_rows, and two amendments_pending items.",
)

out = json.loads(text)
assert any(a.get("flag") == "--batch-order seed" for a in out["recipe"]["added"])
assert out["arm_noul_weight"]["identity"]["ft_rows"].endswith(tail)
assert len(out["amendments_pending"]) == len(pending) + 2
assert all(ord(c) < 128 for c in text)
DRAFT.write_text(text, encoding="utf-8")
print("amended", DRAFT)
