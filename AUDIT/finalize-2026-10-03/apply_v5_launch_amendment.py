"""Apply the human's v5 launch answers and Fable's dedupe rulings to the v5 DRAFT (2026-10-03).

    python3 AUDIT/finalize-2026-10-03/apply_v5_launch_amendment.py <DRAFT path>

<DRAFT path> is a checkout's campaign/v5-preregistered.DRAFT.json, after the 2x H100 amendment
(apply_h100_amendment.py). The lane that owns the v5 queue (L-v5-2gpu) runs this against its own
DRAFT, so the DRAFT and the queue that reads `launch.approved` land together.

Two sources:

(1) The human's answers, ~18:42Z, verbatim "Yes to all, waive R9, approve ~$400"
    (AUDIT/finalize-2026-10-03/human-answers-2026-10-03-v5-launch.md). They settle three
    amendments_pending items: the 2x H100 yes, R9, and C1/C2b/C3. They also bring VitaminC in at
    its cap and add two decontamination targets.
(2) Fable's dedupe rulings (AUDIT/finalize-2026-10-03/dedupe-probe/RULING.md):
    - the four structured Open-Jev families are deduped by exact content and scoped out of the
      near-duplicate search, reported not_run by ruling;
    - the measured candidate bound for builds that read the pool;
    - Open-Jev's val draw keyed on the scene;
    - J5' gated on j5prime.runs_iff verbatim.

All of it is applied before any v5 result exists. No gate, threshold or population moves.

`launch.approved` is new. The queue enforces the human's ceiling, not the projection it used to
read as "approved". The box ceiling is the human's ~$400. The GPU-run ceiling is that, less the
box's non-run hours, which the queue cannot see (it counts GPU steps only):
- setup and preflight, two GPUs for ~2 h: 2 x 2 x $4.19 = $16.76;
- the last round's idle GPU (J5'2 alone, ~6.0 h): $25.14;
- GPU 1 idling while lane 0 probes, up to the probe's 1,800 s cap: $2.10. The probe's own GPU
  is in v5.spend already (GAP-V5-2GPU-PROBE-SPEND-IN-V5-SPEND), so it is not subtracted, and a
  retry's second $2.10 is the box watch's to see.
That leaves $356.00, or 84.96 GPU-h at $4.19. It is ~15% over the $309.7 projection, which is
headroom for an H100 slower than the GH200's cadence. (Fable checked this derivation, ~18:55Z.)

Textual edits keep the file's layout. A second run fails, because launch.approved would already
be there.
"""

import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    sys.exit(__doc__)
DRAFT = Path(sys.argv[1]).resolve()
ANSWERS = "AUDIT/finalize-2026-10-03/human-answers-2026-10-03-v5-launch.md"
RULING = "AUDIT/finalize-2026-10-03/dedupe-probe/RULING.md"
VERBATIM = "Yes to all, waive R9, approve ~$400"
text = DRAFT.read_text(encoding="utf-8")
d = json.loads(text)
if "approved" in d["launch"]:
    sys.exit("already applied")
if "hardware" not in d:
    sys.exit("the 2x H100 amendment is not applied; apply apply_h100_amendment.py first")

RATE = 4.19
BOX_USD = 400.0
OVERHEAD = {"setup": round(2 * 2 * RATE, 2), "last_round_idle": round(6.0 * RATE, 2),
            "probe_idle_gpu": 2.10}
RUNS_USD = round(BOX_USD - sum(OVERHEAD.values()), 2)  # 356.0
RUNS_H = round(RUNS_USD / RATE, 2)  # 84.96
POOL_BOUND = 12_500_000
PENDING_SETTLED = (
    "the human's yes on the 2x H100 amendment.",
    "R9 (readings.R9_pause_after_seed_0), waived or kept",
    "C1, C2b and C3 (C2a is already off",
)
PENDING_DECISION_DATA = "data.sources: decision data"


def q(v):
    return json.dumps(v, ensure_ascii=True)


def replace_once(old, new):
    global text
    if text.count(old) != 1:
        sys.exit(f"anchor not found exactly once ({text.count(old)}): {old[:80]!r}")
    text = text.replace(old, new)


def append_to(value, sentence):
    replace_once(q(value), q(value + sentence))


# --- launch: the human's yes and the ceilings the queue enforces --------------------------------
launch = d["launch"]
new_yes = launch["human_yes"] + (
    f" The human's yes on the 11-run plan at the box's rate, verbatim: '{VERBATIM}' (~18:42Z "
    f"2026-10-03, {ANSWERS}). It is V5_HUMAN_YES, and launch.approved holds the ceilings it sets."
)
approved = {
    "box_usd": BOX_USD,
    "runs_usd": RUNS_USD,
    "runs_gpu_hours": RUNS_H,
    "basis": (
        f"the human's '~$400' is the box ceiling (wall time x $8.38/h). The queue counts GPU steps "
        f"only, so its ceiling is the box's less the non-run hours it cannot see: setup and "
        f"preflight on two GPUs for ~2 h (${OVERHEAD['setup']}), the last round's idle GPU while "
        f"J5' seed 2 runs alone for ~6.0 h (${OVERHEAD['last_round_idle']}), and GPU 1 idling "
        f"while lane 0 probes, up to the probe's cap (${OVERHEAD['probe_idle_gpu']}; the probe's "
        "own GPU is in v5.spend already). That is "
        f"${RUNS_USD} = {RUNS_H} GPU-h at ${RATE}. It is "
        "~15% over projected_cost_usd.total, which stays the estimate. A GPU step that would "
        "cross runs_usd needs V5_OVER_BUDGET_YES, and the lead watches the box's wall-clock spend "
        "against box_usd"
    ),
}
block = json.dumps(approved, indent=2, ensure_ascii=True).replace("\n", "\n    ")
replace_once(q(launch["human_yes"]) + "\n  },", q(new_yes) + f',\n    "approved": {block}\n  }},')

# --- R9 and the conditionals --------------------------------------------------------------------
append_to(
    d["readings"]["R9_pause_after_seed_0"],
    f" Waived on the 2x H100 box by the human ('{VERBATIM}', ~18:42Z 2026-10-03, {ANSWERS}): "
    "no pause after seed 0, V5_R9=waive. v5-pause's reading is still written, report-only.",
)
by_id = {c["id"]: c for c in d["recipe"]["conditionals"]}
for cid, field in (("C1", "on_iff"), ("C2b", "on_iff"), ("C3", "rule")):
    append_to(
        by_id[cid][field],
        f" Taken as off by the human ('{VERBATIM}', ~18:42Z 2026-10-03, {ANSWERS}). Its GH200 "
        "rows are read afterwards, report-only.",
    )

# --- J5' on runs_iff verbatim -------------------------------------------------------------------
lanes = d["hardware"]["lanes"]
old_j5 = (
    "J5' seed s needs v5 seed s's ft and eval rows. box_q_v5j5.sh reads v5's ledger and v5 seed "
    "s's training log, and no checkpoint. So J5' runs ahead of the arm when a lane would "
    "otherwise idle. That never delays the arm, which waits on v5 seeds 0-2's rows either way. "
    "It is a change to the human's order, and is named for the human's yes"
)
if lanes.count(old_j5) != 1:
    sys.exit("hardware.lanes' J5' sentence is not the form this amendment rewrites")
new_lanes = lanes.replace(old_j5, (
    "J5' runs iff j5prime.runs_iff, verbatim (Fable, ~18:30Z 2026-10-03): every J5' seed waits "
    "for v5 seeds 0-2's three completed ft rows, plus v5 seed s's own eval row for its batch "
    "order. The lane job (campaign/post-f-queue/box_q_v5.sh through v5_common.sh) reads v5's "
    "ledger and v5 seed s's training log, and no checkpoint. Under the greedy rule J5' takes a "
    "lane only when no earlier job's inputs are ready, so it never delays the arm, and with "
    "runs_iff verbatim it starts after seeds 0-2 in every branch. J5' ahead of the arm is a "
    f"change to the human's order. The human said yes ('{VERBATIM}', ~18:42Z 2026-10-03, "
    f"{ANSWERS})"
))
replace_once(q(lanes), q(new_lanes))

# --- data: VitaminC and the two new decontamination targets -------------------------------------
pending = d["amendments_pending"]
decision = [p for p in pending if p.startswith(PENDING_DECISION_DATA)]
if len(decision) != 1:
    sys.exit("the decision-data pending item is not there exactly once")
append_to(
    decision[0],
    " VitaminC: every row carries a big_bench_canary field (GUID 26b5c67b-..., [U] BIG-bench's). "
    "It was held at cap 0 until the human's answer, and is in at its cap of 5,000 seeded "
    f"('{VERBATIM}', ~18:42Z 2026-10-03, {ANSWERS}). The field is never read into a row. A "
    "model trained on it is contaminated for any BIG-bench task built from VitaminC.",
)
append_to(
    d["data"]["decontamination"]["refuse"],
    " The decision pool's own decontamination: the pool build (qd-prep decisions --target, "
    "pinned in data/decisions/v5-allocation.json target_sha256) excludes every pool train row "
    "that contains a target, and refuses a --target set other than the pinned one. The targets "
    "are ARC test (ARC-Challenge 1,172 and ARC-Easy 2,376 items, allenai/ai2_arc @210d026f), "
    "BoolQ validation (google/boolq @35b264d0, data/validation-00000-of-00001.parquet, 1,257,630 "
    "B, sha256 52355d11...) and VitaminC test (tals/vitaminc @be6febb7, test.jsonl, 28,717,099 "
    "B, sha256 7ad1808d...), with JevBench and JevJudge as before. The pool is the only source of "
    "ARC, BoolQ and VitaminC rows. The v5 build's containment scan keeps its own targets, the "
    "val and held-out rows above, and does not read these three. BoolQ validation and VitaminC "
    f"test were downloaded with the human's yes ({ANSWERS}) and are recorded in "
    "~/qd-campaign/v5-ccbysa-panel-2026-10-03/fetch-record.jsonl. Their use is decontamination "
    "only.",
)
item6 = "C1, C2a, C2b, C3 and R4 as decided, with the deciding row ids"
if item6 not in pending:
    sys.exit("pending item 'C1, C2a, C2b, C3 and R4 as decided' is not there verbatim")
replace_once(
    q(item6),
    q("R4 as decided, with its deciding row ids. C1, C2a, C2b and C3 are off without deciding "
      "rows: C2a by Fable's ruling (its own item), C1, C2b and C3 by the human's answer "
      f"(~18:42Z 2026-10-03, {ANSWERS}). Their GH200 rows are read report-only"),
)

# --- the dedupe rulings, a new top-level block --------------------------------------------------
dedupe_block = {
    "ruling": RULING,
    "scope": (
        "openjev.policy, openjev.evidence, openjev.routing and openjev.rubric carry "
        "metadata near_duplicate_policy = exact_content, set by their rewriter. They are deduped "
        "by the digest of their dedupe_text alone, in any repo, and never enter the MinHash "
        "candidate search. Their rows share ~2 KB of rule prose and carry their facts as compact "
        "JSON; MinHash at 0.8 paired distinct problems (31-77% of its pairs had different golds)"
    ),
    "reporting": (
        "for those families the near-duplicate search is reported as not run by this ruling, "
        "never as passed: dedupe's detail and near_duplicate_scope name the scoped rows by family "
        "and the rows searched; split's near_duplicate_disjoint reports n of n_total rows"
    ),
    "leak": (
        "for the scoped rows, a leak is the same content digest on two sides of the final split. "
        "split's exact_content_disjoint checks it independently of dedupe, exhaustively, and it "
        "enters the split's aggregate. Different facts are a different problem, whatever prose "
        "they share"
    ),
    "candidate_bound": (
        f"builds that read the decision pool run at POOL_MAX_CANDIDATE_PAIRS = {POOL_BOUND:,} "
        "(qd_data.config; DataConfig.max_candidate_pairs set by pool_data_config). Measured: the "
        "pool's MinHash population proposed 5,874,260 candidates in dedupe and 2,892,003 in the "
        "split. Twice that plus v4's 550,147, rounded up. The default 5,000,000 is unchanged, and "
        "both reports name a bound that is not the default. A build past it reports not_run and "
        "refuses. The structural fix is deferred "
        "(GAP-DEDUPE-LSH-BAND-CANDIDATES-NOT-DUPLICATES-2026-10-03)"
    ),
    "draw_key": (
        "Open-Jev's val draw is keyed on the scene (group_key) alone, not (family, group). That "
        "closes 3 painting-geometry scenes that were train in one family and val in another"
    ),
}
anchor = '\n  "hardware": '
if text.count(anchor) != 1:
    sys.exit("hardware anchor not found exactly once")
dblock = json.dumps(dedupe_block, indent=2, ensure_ascii=True).replace("\n", "\n  ")
text = text.replace(anchor, f'\n  "dedupe": {dblock},{anchor}')

# --- amendments_pending and amendments_applied --------------------------------------------------
settled = [p for p in pending if p.startswith(PENDING_SETTLED)]
if len(settled) != len(PENDING_SETTLED):
    sys.exit(f"expected {len(PENDING_SETTLED)} settled pending items, found {len(settled)}")
for p in settled:
    if pending.index(p) == 0:
        sys.exit("a settled item is first; this amendment removes items after another")
    replace_once(",\n    " + q(p), "")
added_pending = [
    "the final v5 build's dedupe and split under POOL_MAX_CANDIDATE_PAIRS: the candidates in "
    "both searches, the scoped rows by family, the exact-content drops, exact_content_disjoint, "
    "and the rebuilt decision pool's examples sha256 with the too-short-to-scan counts of its "
    "targets (arc-test, boolq-val, vitaminc-test)",
]
remaining = [p for p in pending if p not in settled]
last = remaining[-1]
replace_once(q(last), q(last) + "".join(",\n    " + q(p) for p in added_pending))
append_to(
    d["amendments_applied"],
    " On 2026-10-03 (~18:50Z) the human's launch answers and Fable's dedupe rulings were applied "
    f"(AUDIT/finalize-2026-10-03/apply_v5_launch_amendment.py; {ANSWERS}; {RULING}). The "
    f"changes: launch.human_yes, with the verbatim '{VERBATIM}'; a new launch.approved (box "
    f"${BOX_USD:g}, runs ${RUNS_USD:g} = {RUNS_H:g} GPU-h); R9 waived; C1, C2b and C3 taken as "
    "off; hardware.lanes' J5' text set to runs_iff verbatim; VitaminC in at 5,000; ARC test, "
    "BoolQ validation and VitaminC test as the pool build's enforced targets; a new dedupe block "
    "(the four-family exact-content scope, its reporting and leak definition, the 12,500,000 "
    "bound, the scene draw key). amendments_pending loses its three settled items, rewords the "
    "conditionals item, and gains one. No gate, threshold or population moves.",
)

after = json.loads(text)
assert after["launch"]["approved"]["runs_usd"] == RUNS_USD
assert after["launch"]["approved"]["box_usd"] == BOX_USD
assert after["launch"]["human_yes"].endswith("holds the ceilings it sets.")
assert "box_q_v5j5" not in after["hardware"]["lanes"]
assert after["dedupe"]["ruling"] == RULING
assert len(after["amendments_pending"]) == len(pending) - len(settled) + len(added_pending)
assert not any(p.startswith(PENDING_SETTLED) for p in after["amendments_pending"])
assert after["launch"]["projected_cost_usd"] == d["launch"]["projected_cost_usd"]
assert after["hardware"]["probe"] == d["hardware"]["probe"]
DRAFT.write_text(text, encoding="utf-8")
print(f"applied to {DRAFT}: approved box ${BOX_USD:g}, runs ${RUNS_USD:g} / {RUNS_H:g} GPU-h; "
      f"pending -{len(settled)} +{len(added_pending)}; dedupe block; R9 waived; C1/C2b/C3 off")
