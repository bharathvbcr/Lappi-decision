"""Apply two rulings to the v5 DRAFT (2026-10-03): the 2x H100 box, and seeds 0-4 bound.

    python3 AUDIT/finalize-2026-10-03/apply_h100_amendment.py <DRAFT path>

<DRAFT path> is a checkout's campaign/v5-preregistered.DRAFT.json.

The canonical DRAFT is v5-build's, which carries the 2026-10-03 amendments that main's copy does
not. The lane that rebuilds the v5 queue for the box (L-v5-2gpu) runs this against its own DRAFT
as its first commit, so the DRAFT and the scripts that read it land together.

Ruling 1, the box. The human chose it at ~16:26Z ("Go with 2x H100 (Lambda)"), recorded in
AUDIT/finalize-2026-10-03/report-to-human-2026-10-03-pipeline.md item 1. Fable ruled the same day
on the lanes, the probe and its 12 GiB margin, and the ledger. The memory reading is
AUDIT/finalize-2026-10-03/v5_h100_budget.py. Fable's review asked for four fixes, applied here:
- musl is not bound;
- the probe gets its job fields and its ledger name;
- a precedence sentence covers the GH200 text left in other blocks;
- J5''s inputs are read from box_q_v5j5.sh.

Ruling 2, seeds 0-4. Fable (2026-10-03, under the human's budget lift) ruled that v5's main arm
runs seeds 0-4 unconditionally; the noul-weight arm stays at seeds 0-2. This was in
amendments_pending, and Fable ruled at ~17:25Z that it be bound now, so the queue is built once.
J5' stays x3 on seeds 0-2, because the ruling named only the main arm and the noul arm.

The two are applied together because they meet in the cost block: a state with seeds bound at
the GH200's rate should never exist. What is still the human's call goes into
amendments_pending, decided nowhere here: R9, C1/C2b/C3, the new cost and the setup downloads.

Changes:
- launch.rule, launch.projected_cost_usd, launch.projected_gpu_hours and launch.human_yes.
  Costs are priced at $4.19 per GPU-hour. `v5_seeds_3_4` replaces `if_seeds_3_4`.
- seeds.v5 and seeds.seeds_3_4. seeds.ledger's gh200- names become h100x2- names.
- comparability.same_rows_but_confounded gains the hardware.
- A new top-level `hardware` block, placed before amendments_pending.
- amendments_pending loses the seeds item now bound and gains four items. amendments_applied
  is appended.
No gate, threshold or population moves. The seeds move because Fable ruled it.

Textual insertion keeps the file's layout. A second run fails, because the hardware block would
already be there.
"""

import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    sys.exit(__doc__)
DRAFT = Path(sys.argv[1]).resolve()
REPORT = "AUDIT/finalize-2026-10-03/report-to-human-2026-10-03-pipeline.md"
text = DRAFT.read_text(encoding="utf-8")
d = json.loads(text)
if "hardware" in d:
    sys.exit("already applied")

RATE = 4.19  # per GPU-hour: $8.38/h for the box's two GPUs
SEED_H, J5_H = 7.0, 6.0
SEED_USD, J5_USD = round(SEED_H * RATE, 1), round(J5_H * RATE, 1)  # 29.3, 25.1
CAP_USD = round(32400 / 3600 * RATE, 2)  # 37.71
V5_0_2, V5_3_4 = round(3 * SEED_USD, 1), round(2 * SEED_USD, 1)  # 87.9, 58.6
J5_X3, NW_X3 = round(3 * J5_USD, 1), round(3 * SEED_USD, 1)  # 75.3, 87.9
TOTAL_USD = round(V5_0_2 + V5_3_4 + J5_X3 + NW_X3, 1)  # 309.7
TOTAL_H = 8 * SEED_H + 3 * J5_H  # 74.0: five v5 seeds and three arm seeds at 7.0 h, three J5'
RUNS = 11
SEEDS_PENDING_PREFIX = "seeds: v5's main arm runs seeds 0-4 unconditionally"


def q(v):
    return json.dumps(v, ensure_ascii=True)


def replace_once(old, new):
    global text
    if text.count(old) != 1:
        sys.exit(f"anchor not found exactly once: {old[:80]!r}")
    text = text.replace(old, new)


def append_to(value, sentence):
    replace_once(q(value), q(value + sentence))


launch = d["launch"]
cost = launch["projected_cost_usd"]
hours = launch["projected_gpu_hours"]

append_to(
    launch["rule"],
    f" On the 2x H100 box (the human's choice, ~16:26Z 2026-10-03, {REPORT} item 1) a run's "
    f"cap prices 32,400 s x ${RATE}/h per GPU = ${CAP_USD}, and the new total needs the "
    "human's yes (amendments_pending).",
)

check = cost["check"]
for old, new in ((", $16.0;", f", ${SEED_USD};"), (", $13.7.", f", ${J5_USD}.")):
    if check.count(old) != 1:
        sys.exit(f"cost check anchor {old!r} not found exactly once")
    check = check.replace(old, new)
check += (
    f" Priced at the 2x H100 box's ${RATE} per GPU-hour ($8.38/h for its two GPUs); at the "
    "GH200's $2.29/h these were $16.0 and $13.7. The hours are the GH200 cadence, unmeasured "
    "on an H100 [I: the same GH100 compute; HBM3 at ~3.35 TB/s against the GH200's ~4 TB/s, "
    "so if anything slower]. A v5 seed also no longer runs _train's whole-plan floor "
    "evaluation, about 57 min of each F seed (amendments_pending) [I]. The 7.0 h and 6.0 h "
    "stay as the pre-registered estimate, and the box's measured hours replace them "
    "(amendments_pending). Seeds 3-4 are unconditional (seeds.seeds_3_4), so v5_seeds_3_4 is "
    "part of the total."
)
replace_once(q(cost["check"]), q(check))
for key, old, new in (
    ("v5_seeds_0_2", 48.0, V5_0_2),
    ("j5prime_x3", 41.2, J5_X3),
    ("noul_weight_x3", 48.0, NW_X3),
    ("total", 137.2, TOTAL_USD),
):
    if cost[key] != old:
        sys.exit(f"launch.projected_cost_usd.{key} is {cost[key]!r}, not {old}")
    replace_once(f'"{key}": {q(old)}', f'"{key}": {q(new)}')
if cost["if_seeds_3_4"] != 32.0 or hours["if_seeds_3_4"] != 14.0 or hours["total"] != 60.0:
    sys.exit("the seeds 3-4 cost or hours are not the form this amendment rebinds")
# One name, one quantity: the block for seeds 3-4 is a block like the others now, not an "if".
replace_once('"if_seeds_3_4": 32.0', f'"v5_seeds_3_4": {q(V5_3_4)}')
replace_once('"total": 60.0,\n      "if_seeds_3_4": 14.0,', f'"total": {q(TOTAL_H)},')
replace_once(
    q(cost["if_nomask_enters"]),
    q("C2a is off (amendments_pending): there is no no-mask total"),
)
replace_once(
    q(cost["caps_total"]),
    q(f"${round(RUNS * CAP_USD, 2)} ({RUNS} runs x ${CAP_USD}: 32,400 s x ${RATE}/h per run)"),
)
replace_once(
    q(hours["slot"]),
    q("one Lambda 2x H100 80 GB SXM5 box, two GPU lanes (hardware block); never the GH200. "
      "Order (the human's answer 1 at 0b559bb, with seeds 3-4 bound by Fable): v5 seeds 0-4, "
      "then the noul-weight arm x3 (seeds 0-2) iff v5nw.room is room or V5NW_HUMAN_YES is "
      "pinned, then J5' x3 (seeds 0-2). The jobs run under hardware.lanes' scheduling rule, "
      "and each run holds its lane's lock (gpu0.lock or gpu1.lock). The 11 jobs under that "
      "rule, R9 waived: [s0|s1] [s2|s3] [s4|nw0] [nw1|nw2] [J5'0|J5'1] [J5'2|idle], six rounds. "
      "Wall time after the attestation [I, at the GH200 cadence]: ~41 h with R9 waived "
      "(~$344 for the box at $8.38/h), and ~48 h with R9 kept (~$400; seed 0 runs alone), "
      "plus 1-2 h of setup and the probe"),
)
append_to(
    launch["human_yes"],
    f" Hardware: the human chose the 2x H100 box (~16:26Z 2026-10-03, {REPORT} item 1). The "
    "yes above was on ~$137 at the GH200's rate, for seeds 0-2. The total at the box's rate, "
    f"with seeds 3-4 bound (~${TOTAL_USD} at ${RATE} per GPU-hour, {TOTAL_H:g} GPU-h; "
    "~$344-400 for the box's wall time), needs the human's yes before launch "
    "(amendments_pending).",
)

seeds = d["seeds"]
replace_once(
    q(seeds["v5"]),
    q("0, 1, 2, 3, 4; one ft row and one completed epoch-score-val eval row each, quick false "
      "(five seeds, full schedule, no subsample). Seeds 3 and 4 are unconditional for the main "
      "arm (Fable, 2026-10-03, under the human's budget lift; seeds.seeds_3_4)"),
)
replace_once(
    q(seeds["seeds_3_4"]),
    q("retired for the main arm (Fable, 2026-10-03, under the human's budget lift). Prose "
      "abstention is seed-bimodal on F (54, 0 and 1 of 60 on seeds 0-2), and three seeds leave "
      "promotion and the avg kind to luck. Seeds 3 and 4 are seeds.v5's: unconditional, in "
      "v5's form, with --batch-order seed (plan seeds 3 and 4; "
      "AUDIT/post-f-2026-10-02/fable-seed-order-ruling.md section 2). qd-post-f-rules seeds34 "
      "is not read for the main arm. They are never part of the arm envelope: the noul-weight "
      "arm stays at seeds 0-2, paired with the main arm's seeds 0-2, which is its "
      "pre-registered comparison. J5' stays x3, on seeds 0-2"),
)
new_ledger = seeds["ledger"].replace("/gh200-v5-noulw-", "/h100x2-v5-noulw-").replace(
    "/gh200-v5-", "/h100x2-v5-"
)
if new_ledger == seeds["ledger"] or "gh200" in new_ledger:
    sys.exit(f"seeds.ledger is not the form this amendment rewrites: {seeds['ledger']!r}")
replace_once(q(seeds["ledger"]), q(new_ledger))

append_to(
    d["comparability"]["same_rows_but_confounded"],
    " v5 also runs on an H100 where F ran on the GH200 (hardware block): the same software, "
    "another GPU.",
)

hardware = {
    "box": (
        "one Lambda 2x H100 80 GB SXM5 instance, $8.38/h ($4.19 per GPU-hour), x86_64. The "
        f"human's choice: 'Go with 2x H100 (Lambda)', ~16:26Z 2026-10-03 ({REPORT} item 1)"
    ),
    "every_row": (
        "every v5 row runs on this box, none on the GH200: the probe, seeds 0-4, the "
        "noul-weight arm, J5' and the trajectory rows. F's rows stand as measured on the GH200, "
        "and comparability.rule already keeps v5 from being read against F's envelope"
    ),
    "precedence": (
        "this block governs wherever another block of this file names: the GH200's queue "
        "(build_order step 7, 'after j6g'); one gpu.lock (recipe.added[0].scoring, "
        "j5prime.what); J5' strictly after seeds 3-4 or the arm (arm_noul_weight.what); or "
        "seeds 3-4 as conditional ('if they run'). The box is the 2x H100, the lock is the "
        "lane's, seeds 3-4 are seeds.v5's, and the order is hardware.lanes' rule. The "
        "conditionals' GH200 ledger references stand: those rows are on the GH200"
    ),
    "environment": (
        "the GH200's qd-venv, package for package (71 dist-infos: torch 2.10.0+cu128, "
        "transformers 5.17.0, fla 0.5.2, triton 3.7.1, causal_conv1d 1.7.0 compiled), checked "
        "against the GH200 freeze by the setup script. x86_64 builds of qd-prep and "
        "qd-post-f-rules, sha-pinned at deploy: static musl first (cross-built on the Mac, no "
        "download), glibc instead if the timed split rebuild on the box takes more than ~2x "
        "the GH200's. Before seed 0: a parity run, and that timed split rebuild, against the "
        "GH200's binaries"
    ),
    "lanes": (
        "two GPU lanes, one lock each (gpu0.lock, gpu1.lock), with CUDA_VISIBLE_DEVICES pinned "
        "per lane. The scheduling rule (Fable, 2026-10-03): each lane takes the earliest job in "
        "the human's order whose inputs are ready. The order is answer 1 at 0b559bb, with seeds "
        "3-4 bound: v5 s0-s4, then the arm x3 (seeds 0-2) iff room, then J5' x3 (seeds 0-2). "
        "The inputs: v5 seeds 1-4 need only V5_CONTINUE, and only when R9 is kept; the arm "
        "needs v5nw.room, read from v5 seeds 0-2's eval rows; J5' seed s needs v5 seed s's ft "
        "and eval rows. box_q_v5j5.sh reads v5's ledger and v5 seed s's training log, and no "
        "checkpoint. So J5' runs ahead of the arm when a lane would otherwise idle. That never "
        "delays the arm, which waits on v5 seeds 0-2's rows either way. It is a change to the "
        "human's order, and is named for the human's yes"
    ),
    "ledger": (
        "one v5 ledger file and one arm ledger file, written by both lanes. "
        "qd_train.ledger.Ledger.append holds fcntl.flock across reading the predecessor and "
        "writing, opens O_APPEND, makes one os.write and fsyncs "
        "(test_concurrent_appends_do_not_break_the_chain). The probe's row goes in "
        "/home/ubuntu/ledger/h100x2-v5-probe-<date>.jsonl, never v5's"
    ),
    "memory": (
        "F's torch-allocated peak was 61.6 GiB (F seed 1, train-s1.log), against an H100's ~79 "
        "GiB [I: from memory]. qd_train.memory reproduces F's recorded device_budget estimate "
        "to the byte (68,510,315,980 B) and prices v5's costliest batch shapes at <= 64.46 "
        "GiB. The costliest is the narrowest bucket with the most rows, not the widest "
        "(AUDIT/finalize-2026-10-03/v5_h100_budget.py and .out)"
    ),
    "probe": {
        "what": (
            "tools/real_ft_run.py --probe-shapes 12 on v5's recipe and data argv, run once on "
            "GPU 0 under gpu0.lock before seed 0. It takes one optimizer step on each distinct "
            "(rows, width) batch shape of v5's plan, costliest first, and writes a quick ft row "
            "with the metric memory_probe into /home/ubuntu/ledger/h100x2-v5-probe-<date>.jsonl"
        ),
        "pass": (
            "every shape stepped, and torch.cuda.max_memory_reserved <= the device total - "
            "margin_gib"
        ),
        "margin_gib": 12,
        "margin_basis": (
            "Fable, 2026-10-03. The probe reads reserved after one step per shape, and a full "
            "run's reserved grows from there. F measured how much: process-level 72.1 GiB "
            "against an allocated peak of 61.6 GiB, ~10.5 GiB of cache and fragmentation over "
            "9,683 steps on a card that never pushed back, plus ~1 GiB of CUDA context that "
            "sits outside max_memory_reserved but inside the device total. Rounded up to 12 "
            "GiB: pass iff reserved <= total - 12 GiB (~67 GiB on an H100). memory.py puts "
            "v5's costliest allocated step at <= 64.46 GiB, so a fresh process's reserved "
            "should land at ~63-66 GiB: a pass with a few GiB to spare, and a fail at the first "
            "shape that fragments badly"
        ),
        "job": (
            "cap 1,800 s (about 40 steps plus the load); 1,800 s x $4.19/h = $2.10 at the cap, "
            "one GPU, under rule 4's $20 line; --approved-by names the human's v5 yes"
        ),
        "record": (
            "on the row (memory_probe) and in the next amendment: the device total, the "
            "allocated peak, the reserved peak, the margin, and which attempt passed"
        ),
        "fallback": (
            "on a fail, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True and the probe again, "
            "at the same 12 GiB; that changes no arithmetic and no recipe key. A second fail "
            "stops v5 and asks the human. No recipe lever moves without the human's yes: "
            "--checkpoint-skip-layers is in the backbone fingerprint"
        ),
    },
}
anchor = '\n  "amendments_pending": '
if text.count(anchor) != 1:
    sys.exit("amendments_pending anchor not found exactly once")
block = json.dumps(hardware, indent=2, ensure_ascii=True).replace("\n", "\n  ")
text = text.replace(anchor, f'\n  "hardware": {block},{anchor}')

pending = d["amendments_pending"]
bound = [p for p in pending if p.startswith(SEEDS_PENDING_PREFIX)]
if len(bound) != 1:
    sys.exit(f"the seeds 0-4 pending item is not there exactly once ({len(bound)})")
idx = pending.index(bound[0])
if idx == 0:
    sys.exit("the seeds 0-4 pending item is first; this amendment expects it after another item")
replace_once(",\n    " + q(bound[0]), "")
added_pending = [
    "the human's yes on the 2x H100 amendment. It covers the cost at the box's rate (~$"
    f"{TOTAL_USD} at the per-GPU rate for {RUNS} runs; ~$344 for the box with R9 waived or "
    "~$400 kept, plus setup), the J5'-before-arm interleave of hardware.lanes, and the setup "
    "downloads (the 70 pinned wheels and causal_conv1d 1.7.0's build)",
    "R9 (readings.R9_pause_after_seed_0), waived or kept on the 2x H100 box. Kept, seed 0 runs "
    "alone: ~7 h with GPU 1 idle, ~+7 h of wall time. The human's call",
    "C1, C2b and C3 (C2a is already off: its own item above) are either decided by their "
    "GH200 rows as recipe.conditionals writes, or taken as off with those rows read "
    "report-only afterwards. The lead and Fable recommend off. The human's call",
    "the box's measured per-run hours, from the probe and seed 0's ft row, replace the "
    "GH200-cadence estimates in launch.projected_cost_usd",
]
last = [p for p in pending if p != bound[0]][-1]
replace_once(q(last), q(last) + "".join(",\n    " + q(p) for p in added_pending))

append_to(
    d["amendments_applied"],
    " On 2026-10-03 (~17:40Z) two rulings were applied together "
    "(AUDIT/finalize-2026-10-03/apply_h100_amendment.py). (1) The 2x H100 box: the human's "
    "choice, with Fable's ruling on the lanes, the probe (its margin, 12 GiB, Fable's) and the "
    "ledger. (2) Fable's seeds 0-4 for the main arm, until then in amendments_pending: the "
    "noul-weight arm stays at seeds 0-2 and J5' stays x3 on seeds 0-2. The changes: "
    "launch.rule, launch.projected_cost_usd (v5_seeds_3_4 replaces if_seeds_3_4), "
    "launch.projected_gpu_hours (total 74.0), launch.human_yes, seeds.v5, seeds.seeds_3_4, "
    "seeds.ledger and comparability.same_rows_but_confounded; a new hardware block; and "
    "amendments_pending, which loses the seeds item and gains four. No gate, threshold or "
    "population moves.",
)

after = json.loads(text)
assert after["hardware"]["probe"]["margin_gib"] == 12
assert after["launch"]["projected_cost_usd"]["total"] == TOTAL_USD
assert after["launch"]["projected_cost_usd"]["v5_seeds_3_4"] == V5_3_4
assert "if_seeds_3_4" not in after["launch"]["projected_cost_usd"]
assert "if_seeds_3_4" not in after["launch"]["projected_gpu_hours"]
assert after["launch"]["projected_gpu_hours"]["total"] == TOTAL_H
assert len(after["amendments_pending"]) == len(pending) - 1 + len(added_pending)
assert not any(p.startswith(SEEDS_PENDING_PREFIX) for p in after["amendments_pending"])
DRAFT.write_text(text, encoding="utf-8")
print(f"applied to {DRAFT}: total ${TOTAL_USD} / {TOTAL_H:g} GPU-h over {RUNS} runs; seed "
      f"${SEED_USD}, J5' ${J5_USD}, cap ${CAP_USD}; pending -1 +{len(added_pending)}")
