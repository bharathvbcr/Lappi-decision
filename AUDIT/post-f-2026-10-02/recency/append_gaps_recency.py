"""L-recency's gap records, appended once through qd_train.gaps.append_gap.

Kept as the record of exactly what was appended. Every number is read from
recency-f-2026-10-02.json, never typed here. Running it again refuses: each new id may be
recorded once, and the bimodality gap takes one L-recency line.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "python"))
from qd_train.gaps import append_gap, read_gaps, validate_gap  # noqa: E402

#: --dry-run validates and prints each line instead of appending it.
DRY = sys.argv[1:] == ["--dry-run"]
if sys.argv[1:] not in ([], ["--dry-run"]):
    sys.exit("usage: append_gaps_recency.py [--dry-run]")
RESULT = HERE / "recency-f-2026-10-02.json"
LANE = "L-recency"
BIMODAL = "GAP-OOD-PROSE-ABSTENTION-BIMODAL-ACROSS-SEEDS-2026-10-02"
NAV = "GAP-L-RECENCY-NAVIGATION-2026-10-02"
DIGEST = "GAP-F-FT-ROWS-CARRY-NO-ORDER-SENSITIVE-PLAN-DIGEST-2026-10-02"
ORDER = "GAP-REAL-FT-RUN-BATCH-ORDER-IGNORES-THE-TRAINING-SEED-2026-10-02"

r = json.loads(RESULT.read_text(encoding="utf-8"))
if not all(v["ok"] for v in r["identity"].values()) or "counts" not in r:
    sys.exit("the result's identity checks did not all pass; nothing to append")

gaps = read_gaps()
ids = {g["id"] for g in gaps}
for new in (NAV, DIGEST, ORDER):
    if new in ids:
        sys.exit(f"{new} is already recorded; refusing a duplicate")
bimodal = [g for g in gaps if g["id"] == BIMODAL]
if not bimodal:
    sys.exit(f"{BIMODAL} is not in gaps.jsonl")
if any(g.get("lane") == LANE for g in bimodal):
    sys.exit(f"{BIMODAL} already carries an {LANE} line; refusing a duplicate")


def emit(record: dict) -> None:
    if DRY:
        print(validate_gap(record, known_ids=ids))
    else:
        append_gap(record)


def unit(group: str, name: str) -> str:
    u = r["counts"][group][name]
    return f"{u['late']}/{u['n']} = {u['fraction']:.4f}"


def ref(group: str, name: str) -> str:
    s = r["references"][group][name]["hypergeometric_bucket_stratified"]
    p = r["references"][group][name]["hypergeometric_unstratified"]
    return (f"expected {s['expected']:.1f} stratified / {p['expected']:.1f} unstratified, "
            f"two-sided p {s['p_two_sided']:.3g} / {p['p_two_sided']:.3g}")


seeds = r["per_seed"]
prose = ", ".join(
    f"seed {s} {seeds[s]['ood_abstain.prose']['n']}/{seeds[s]['ood_abstain.prose']['n_total']} "
    f"({seeds[s]['eval_row_id'][:8]})"
    for s in sorted(seeds)
)
win = r["late_window"]["primary"]
plan = r["plan"]
digest = r.get("consumed_digest", {}).get("consumed_digest")
rows_mc = r["references"]["prose"]["rows_any_sequence_late"]["monte_carlo_bucket_stratified"]
rows_mc_b = r["references"]["v3b_all"]["rows_any_sequence_late"]["monte_carlo_bucket_stratified"]
null = r.get("planner_null")

note = (
    "L-recency (report-only; Fable's post-F ruling line 30; "
    "AUDIT/post-f-2026-10-02/recency/recency-f-2026-10-02.json). F's three seeds trained on ONE "
    f"batch order: real_ft_run.main at a502670 builds plan_all once at DataConfig().seed = "
    f"{plan['plan_seed']} (real_ft_run.py:{r['code_anchors']['plan_built_once_at_config_seed']['line']}, "
    f"config at :{r['code_anchors']['config_is_default']['line']}) and hands that list to _train "
    f"for every seed (:{r['code_anchors']['every_seed_trains_plan_all']['line']}). So the recency "
    "hypothesis cannot explain 54/0/1: the rows trained last are the same rows on every seed. "
    f"The rebuilt plan matches all three ft rows (batches {plan['steps']}, width, shard hash, "
    "plan rows, padding n/n_total; one batch per optimizer step, steps_exhausted). Last 10% = "
    f"positions {win['first_position']}-{win['last_position']} ({win['steps']} steps). "
    f"(a) prose (2,010 rows): rows with any sequence late {unit('prose', 'rows_any_sequence_late')} "
    f"(stratified MC expected {rows_mc['expected']:.1f}, two-sided p {rows_mc['p_two_sided']:.3g}); "
    f"choice-slot sequences {unit('prose', 'sequences_choice_slot')} ({ref('prose', 'sequences_choice_slot')}); "
    f"all sequences {unit('prose', 'sequences_all_slots')}. "
    f"(b) v3b (5,004 rows): rows any-late {unit('v3b_all', 'rows_any_sequence_late')} "
    f"(MC expected {rows_mc_b['expected']:.1f}, p {rows_mc_b['p_two_sided']:.3g}); choice-slot "
    f"{unit('v3b_all', 'sequences_choice_slot')} ({ref('v3b_all', 'sequences_choice_slot')}); all "
    f"sequences {unit('v3b_all', 'sequences_all_slots')}. Expected fraction by steps "
    f"{r['expected_fraction_by_steps']:.4f}, by sequences "
    f"{r['references']['prose']['expected_fraction_by_sequences']:.4f}. Same numbers for seeds 0, 1 "
    f"and 2, beside ood_abstain.prose {prose}. "
    + (
        f"Over {null['n_plans']} plans this planner builds at seeds 0-{null['seeds'][1]} "
        "(unconditional on the batch order, where the p values above condition on it): prose "
        f"choice-slot mean {null['groups']['prose']['sequences_choice_slot']['mean']:.1f} "
        f"(range {null['groups']['prose']['sequences_choice_slot']['min']}-"
        f"{null['groups']['prose']['sequences_choice_slot']['max']}, "
        f"{null['groups']['prose']['sequences_choice_slot']['p_le']:.3g} of plans <= F's), v3b "
        f"choice-slot mean {null['groups']['v3b_all']['sequences_choice_slot']['mean']:.1f} "
        f"(range {null['groups']['v3b_all']['sequences_choice_slot']['min']}-"
        f"{null['groups']['v3b_all']['sequences_choice_slot']['max']}, "
        f"{null['groups']['v3b_all']['sequences_choice_slot']['p_le']:.3g} <= F's). "
        if null else ""
    )
    + (f"Consumed digest of the plan {digest}: if the three final checkpoints in "
       "/home/ubuntu/ckpt/p4-v4 all carry it, the one-order reading is confirmed on the box. "
       if digest else "")
    + "The p values are uncorrected across the ~10 group-by-unit counts read. n = 3 seeds, one "
      "plan; no causal claim."
)

emit({**bimodal[-1], "lane": LANE, "note": note,
            "tool": "AUDIT/post-f-2026-10-02/recency/recency_diagnostic.py (a502670's ShardReader._plan on F's v4 train shards)",
            "evidence": "AUDIT/post-f-2026-10-02/recency/recency-f-2026-10-02.json; "
                        "HANDOFF/recency-2026-10-02.md"})
print(f"{'validated' if DRY else 'appended'} an {LANE} line to {BIMODAL}")

emit({
    "id": NAV,
    "opened": "2026-10-02",
    "lane": LANE,
    "owner": "lead",
    "status": "open",
    "tool": "devmap_status, gitpulse_insights; ListAgents",
    "question": (
        "L-recency's navigation: what could DevMap, GitPulse and ListAgents answer for this lane?"
    ),
    "answer": (
        "DevMap has no store in the lane's worktree (.claude/worktrees/agent-a940b295305b35d51 "
        "has no .devmap/codeintel/devmap.sqlite); symbols were found through the main checkout's "
        "index (generation 3078, fresh, HEAD a3c09de = the lane's base) while main carried 15 "
        "dirty files, none of them the files read here. The lane's code reads are of an a502670 "
        "export, which no index covers: every a502670 line cited is a direct read. GitPulse "
        "insights ran; its collisions facet scanned 16 of 58 worktrees (truncated). ListAgents "
        "is not exposed in this session, so peers in the same worktree could not be listed; the "
        "lane wrote only under AUDIT/post-f-2026-10-02/recency/, HANDOFF/recency-2026-10-02.md "
        "and gaps.jsonl."
    ),
})
print(f"{'validated' if DRY else 'appended'} {NAV}")

emit({
    "id": DIGEST,
    "opened": "2026-10-02",
    "lane": LANE,
    "owner": "lead",
    "status": "open",
    "tool": "rg/json over ledger/gh200-p4-v4-2026-10-01.jsonl; AUDIT/post-f-2026-10-02/recency/recency_diagnostic.py",
    "question": (
        "Can F's batch order be checked against anything F recorded, from the Mac?"
    ),
    "answer": (
        "No. F's ft rows (973cd4e3, 95fa4854, 32990e1a) record batches, width, shard_hash, "
        "plan_rows and padding n/n_total, all of which any plan of this shard set at "
        "batch_tokens 35403 matches whatever its seed. The one order-sensitive record is "
        "Checkpoint.consumed_digest, which lives only in the checkpoints on the box. L-recency "
        "rebuilt the order from a502670's code (numpy "
        f"{r['numpy']} on the Mac; stack/train.lock pins numpy 2.5.2, and which numpy the box ran "
        "is unverified) and computed the plan's consumed digest "
        f"{digest or '(not computed)'}. Residual: one read-only box check -- the consumed_digest "
        "of the three final checkpoints in /home/ubuntu/ckpt/p4-v4 -- confirms or refutes both "
        "the rebuilt order and the one-order reading. A future ft row could carry the plan's "
        "order digest as a metric."
    ),
})
print(f"{'validated' if DRY else 'appended'} {DIGEST}")

emit({
    "id": ORDER,
    "opened": "2026-10-02",
    "lane": LANE,
    "owner": "lead",
    "status": "open",
    "tool": "direct reads of tools/real_ft_run.py, python/qd_train/backbone.py and trainer.py at a502670 and at a3c09de",
    "question": (
        "Does the training seed set the batch order in real_ft_run's epoch arm, as the code's own "
        "comments and the pre-registrations assume?"
    ),
    "answer": (
        "No, at a502670 and still at a3c09de: main builds plan_all once at DataConfig().seed "
        "(20260919) and every seed in --seeds trains that list (a502670 real_ft_run.py:8140, "
        ":8466, :8707; a3c09de :10301, :10674, :10976). The protocol seed sets the span head's "
        "init and the training RNG stream (backbone.py torch.manual_seed); with default kernels "
        "nondeterminism adds the rest. backbone.py's comment (a3c09de :1064) says the seed "
        "'determined the BATCH ORDER', trainer.py:692 refuses a resume because 'the batch order "
        "is a function of the seed', and f-v4-preregistered.json's J6(f) says 'same data order "
        "as F seed 0' -- true of every F seed. Consequence: F's seed-to-seed variance excludes "
        "data order, and so will any arm trained through this epoch arm at a3c09de. J4 and "
        "J6(b) ran it at earlier commits this lane did not read [unverified for them]. Whether "
        "one order for all seeds is wanted is the lead's call; report-only here, nothing changed."
    ),
})
print(f"{'validated' if DRY else 'appended'} {ORDER}")
