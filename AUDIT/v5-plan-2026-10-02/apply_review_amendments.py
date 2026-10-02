"""Apply Fable's v5 review (section 5) and the post-F ruling's v5 items to the v5 DRAFT.

The lead ran this once on 2026-10-02. No v5 build, shard, code change or ledger row existed at
that point. The script is kept as the record of exactly what changed. Every edit is a textual
replacement or insertion that must match exactly once, so the file keeps its hand layout:
compact objects inside arrays, ASCII only. After the edits the script checks that the file
still parses as JSON. Running it a second time fails, because the old texts are gone.

Sources:
- AUDIT/v5-plan-2026-10-02/fable-v5-plan-review.md, section 5 (Fable source sha256 04e969b0...).
- AUDIT/post-f-2026-10-02/fable-post-f-ruling.md, sections 1, 2(b), 3(a) and 4 (Fable source
  sha256 c70d2fb7...).
- The human's answers: d24c865 (AUDIT/v5-plan-2026-10-02/human-answers-v5-review.md), the G6
  download at 9f6b1fa (AUDIT/v5-plan-2026-10-02/g6-commitpackft-download.md), and 0b559bb
  (AUDIT/post-f-2026-10-02/human-answers-post-f.md).
"""

import json
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "campaign" / "v5-preregistered.DRAFT.json"
text = PATH.read_text(encoding="utf-8")
orig = json.loads(text)


def q(value):
    return json.dumps(value, ensure_ascii=True)


def sub(old: str, new: str) -> None:
    global text
    n = text.count(old)
    if n != 1:
        sys.exit(f"expected exactly one match, found {n}: {old[:120]!r}")
    text = text.replace(old, new)


def set_value(key: str, old_value, new_value) -> None:
    sub(f"{q(key)}: {q(old_value)}", f"{q(key)}: {q(new_value)}")


def replaced_or_die(value: str, old: str, new: str) -> str:
    if value.count(old) != 1:
        sys.exit(f"expected exactly one {old!r} in {value[:80]!r}")
    return value.replace(old, new)


def raw(key: str, old_raw: str, new_value) -> None:
    sub(f"{q(key)}: {old_raw}", f"{q(key)}: {q(new_value)}")


lc = orig["launch"]["projected_cost_usd"]
lh = orig["launch"]["projected_gpu_hours"]

# --- launch: cost and hours (review section 5, bullets 1-2; post-F section 1 order) ---
set_value(
    "basis",
    lc["basis"],
    "F cadence: ft 973cd4e3 wall_clock_s 16,024 s; score phase 67 min by q-f.log (ft row 22:28:55 "
    "-> 'train+score done' 23:36; the eval row's wall_clock_s 578 s is 'caller' and undercounts "
    "it); needle control 7 min; CPU controls overlap the next seed (Fable's v5 review section 2.1)",
)
set_value(
    "check",
    lc["check"],
    "Fable's v5 review section 2.1 [I]: a v5 seed at 1.161 x F's positions is 5.17 h train + 1.12 "
    "h score + 0.12 h needle control + 0.1 h prelude/checkpoint + 0.4 h trajectory ~ 7.0 h, $16.0; "
    "a J5' seed (score-val ~ 0.75 h of the 67 min) ~ 6.0 h, $13.7. The lane's per-seed $12.93 "
    "(AUDIT/v5-plan-2026-10-02/v5_accounting.json) priced the score phase from the eval row's 578 "
    "s and is superseded (GAP-EVAL-ROW-WALL-CLOCK-UNDERCOUNTS-SCORE-PHASE-2026-10-02).",
)
raw("v5_seeds_0_2", "38.80", 48.0)
raw("j5prime_x3", "36.60", 41.2)
raw("noul_weight_x3", "38.80", 48.0)
raw("total", "114.20", 137.2)
raw("if_seeds_3_4", "25.87", 32.0)
set_value(
    "if_nomask_enters",
    lc["if_nomask_enters"],
    "~ $116 at -20% train, ~ $95 at -40%",
)
set_value(
    "caps_total",
    lc["caps_total"],
    "$219.84; $272.5 with seeds 3-4",
)
raw("total", "49.9", 60.0)
raw("if_seeds_3_4", "11.3", 14.0)
set_value(
    "slot",
    lh["slot"],
    "one GH200 after the post-F chain and the idle queue (j4ens3 -> avgnp -> s34 -> j7p -> "
    "nomaskp2 -> fslice -> j5p -> nomask/j6f/tierb2 -> j6dv4 -> rung0 -> cudadev -> rungd -> fsucc "
    "-> j5pp -> j6a -> j6g); the v5 waiter waits on j5pp.done, then wait_queued j6a, then "
    "wait_queued j6g; appended, never inserted; no launched waiter is edited; STOP ~ 58 h (quiet) "
    "/ 81 h (fires) after f.done [I]. Order inside the v5 block (Fable's post-F ruling section 1; "
    "the human's answer 1 at 0b559bb): v5 seeds 0-2, then seeds 3-4 iff the spread rule fires, "
    "then the noul-weight arm x3 iff v5nw.room is room or V5NW_HUMAN_YES is pinned, then J5' x3; "
    "each run holds /home/ubuntu/queue/gpu.lock",
)
sub(
    "\n    }\n  },\n  \"data\": {",
    "\n    },\n    \"human_yes\": "
    + q(
        "Yes, launch when ready (Recommended): the human answer 1 at d24c865 "
        "(AUDIT/v5-plan-2026-10-02/human-answers-v5-review.md), on ~$137 / ~60 GPU-h (+$32 / +14 h "
        "with seeds 3-4), with R9 seed-0 pause and R7 skip. Each run still carries --approved-by "
        "and its cap."
    )
    + "\n  },\n  \"data\": {",
)

# --- recipe.added[0]: retention (review bullet 3) ---
added0 = orig["recipe"]["added"][0]
set_value(
    "flag",
    added0["flag"],
    "--checkpoint-every 100000 (as F) plus --retain-tower-every 1000: a checkpoint-form snapshot "
    "<tag>-seed<N>-<device>-step<S>.json holding tower and span_head only (bf16, ~ 3.5 GiB) at "
    "every 1,000th optimizer step and the final step",
)
set_value(
    "disk",
    added0["disk"],
    "3.5 TB free on the box root (df 2026-10-02 ~07:00 UTC; /home/ubuntu/ckpt 279 GB); ~ 42 GiB "
    "per seed, ~ 253 GiB for six seeds, ~ 337 GiB for eight; retained until the next re-plan reads "
    "them",
)
set_value(
    "scoring",
    added0["scoring"],
    "a separate waiter per seed (v5traj-s<N>) after its eval row, holding gpu.lock, cap 3,600 s: "
    "the 180-case OOD suite per snapshot, and one last-3 average row iff tools/ckpt_average.py "
    "--from tower gains a same-seed unequal-step allowance (new code; it requires equal "
    "optimizer_step today, :14), rows tagged trajectory-ood with metrics.checkpoint_step; its "
    ".done never gates the next seed; ~ 20-30 min per seed [I]",
)

# --- recipe.conditionals: C2b (review bullet 4), C3 (post-F 2(b) and the human's answer 3) ---
cond = orig["recipe"]["conditionals"]
set_value(
    "on_iff",
    cond[2]["on_iff"],
    "the fused outcome run passes the three-part rule of "
    "AUDIT/tierb-nomask-flash-screen-design-2026-10-01.md:166-169 applied to the fused run "
    "(tools/perf_tierb_fused.sh execs perf_tierb_outcome.sh fused); the rule printed by "
    "`perf_tierb_outcome.sh fused --print` is pinned under amendments_pending. Independent of C2a.",
)
set_value(
    "rule",
    cond[3]["rule"],
    cond[3]["rule"]
    + " If fsucc printed refused with refused_because all (c) and detail.arms.j6dv4.wins true, the "
    "human decides between F'(d) (fsucc's step 3, launched by hand) and v5 taking the J6(d)-v4 "
    "delta directly; the decision and its words are recorded here before v5's prelude (Fable's "
    "post-F ruling 2(b)). The human decided in advance (0b559bb, "
    "AUDIT/post-f-2026-10-02/human-answers-post-f.md answer 3, 'v5 takes the J6(d) change'): on "
    "that refusal with detail.arms.j6dv4.wins true, v5's base takes --lr 3e-5 --beta2 0.95 and no "
    "F' runs (the three v5 seeds are the test, confounded with v5's data and format changes); with "
    "j6dv4.wins false, no F' runs and C3 keeps F's flags subject to R4.",
)

# --- data: provenance, contrast rows, G6, decontamination ---
src = orig["data"]["sources"]
sub(
    f'"split": {q(src[3]["split"])}, "local": true}}',
    f'"split": {q(src[3]["split"])}, "local": true, "provenance": '
    + q(
        "Fable's v5 review section 2.9: own-repo prose is text the user authored or had written in "
        "their own repositories, not text hosted there. (1) Repo admission: a remote naming "
        "bharathvbcr and no upstream remote naming another owner (a fork); no-remote repos out. "
        "(2) Tree pruning: nested .git trees; vendor/ external/ third_party/ node_modules/ "
        "site-packages/; any directory with its own LICENSE/COPYING/NOTICE; any file with a "
        "Copyright/(c)/SPDX-License-Identifier line naming another holder; YAML-front-matter model "
        "cards; any Markdown byte-identical to a file in the seven 'other' repos. (3) Content: "
        "Markdown prose paragraphs, and commit bodies of admitted repos where git is readable, "
        "under the same filters (fenced/indented code, tables, HTML, link-only lines and trailers "
        "stripped), >= 25 words, ASCII ratio >= 0.9, <= 150 rows/repo. (4) Record: "
        "own-prose-v1/files.jsonl (repo, path, sha256, words), sha-pinned in "
        "own-prose-v1/manifest.json, so the human can strike any file. (5) Authorship: "
        "owner-granted (train-plan:229-231) covers only what survives. The human's tick on the "
        "repo list: 'All but Lappi-decision (Recommended)' at d24c865, the 29 non-fork repos of "
        "AUDIT/v5-plan-2026-10-02/own_repo_inventory.json other than Lappi-decision "
        "(devtools/DevPrism is a fork by rule (1): a second remote delibae/claude-prism)."
    )
    + "}",
)
sub(
    "{" + f'"source": {q(src[4]["source"])}',
    "{" + f'"source": {q(src[4]["source"])}, "amended": '
    + q(
        "Fable's v5 review section 2.4, recommended form, taken by the lead: derived at pipeline "
        "time in python/qd_train after dedupe, split and --exclude-identity-keys, from the "
        "surviving train twins: 1,200 MMLU + 800 CSQA in blake2b order of identity key, identity "
        "'contrast:<twin identity>', recorded in the shard header as contrast_rows {count, sha256, "
        "seed}; never a dedupe or split unit; no over-draw. This supersedes 'form' (its over-draw "
        "and exclusion clauses) and 'split' below. One 'prose text -> defect_class Z row' function "
        "serves this route and the own-prose route."
    ),
)
sub(
    "}\n    ],\n    \"clinc_oos_rekey\"",
    "},\n      "
    + "{"
    + ", ".join(
        f"{q(k)}: {q(v)}"
        for k, v in [
            ("source", "unseen-language noul (G6): bigcode/commitpackft, eight languages (new)"),
            (
                "rows_v5_train",
                ">= 1,000 rows over >= 5 languages after the corpus-v3 per-row licence filter and "
                "exclusions (Fable's v5 review section 4 Q2); the per-language cap is set by the "
                "data lane and pinned in the manifest before the build (amendments_pending)",
            ),
            (
                "supply",
                "5,978 rows downloaded 2026-10-02 ~12:25 UTC to /Users/bharath/qd-campaign/"
                "commitpackft-g6-2026-10-02/<lang>/data.jsonl, each file's size and sha256 equal "
                "to the Hub's LFS size and oid: perl 2,288, clojure 2,403, erlang 480, ocaml 333, "
                "julia 180, tcl 103, r 121, fortran 70 "
                "(AUDIT/v5-plan-2026-10-02/g6-commitpackft-download.md). Under the seven licences "
                "in corpus v3's rows (mit, apache-2.0, bsd-3-clause, bsd-2-clause, isc, unlicense, "
                "cc0-1.0) ~3,616 rows pass [I: from the recorded per-row licence counts, assuming "
                "the filter admits exactly those seven; the lane reads the filter and verifies]",
            ),
            (
                "approval",
                "'Fetch the file list (Recommended)' (d24c865), then 'Download all 8 "
                "(Recommended)' (9f6b1fa)",
            ),
            (
                "why",
                "none of these languages is in the pool (go, python, rust, typescript), the OOD "
                "suite (c, java, ruby, haskell, sql, lua), the needle suite (swift) or v3b's "
                "templates (csharp, elixir, kotlin, php, scala, shell); F's unseen-language counts "
                "were 20/0/9 of 60",
            ),
            ("licence", "per row, corpus-v3 filter; dataset card licence mit"),
            ("feeds", "code.defect_class choice gold Z + span abstain (real code files, not "
                      "templates)"),
            ("split", "pinned to train (metadata[PINNED_SPLIT_KEY] = 'train'); never in val or "
                      "held-out"),
            ("local", True),
        ]
    )
    + "}\n    ],\n    \"clinc_oos_rekey\"",
)
dec = orig["data"]["decontamination"]
set_value(
    "rule",
    dec["rule"],
    dec["rule"]
    + " Constant template text is stripped before n-gramming, for containment only: within each "
    "(family, slot), any question line and any option value whose text is byte-identical across "
    "every rendered row of that family-slot in the union of sources and targets is removed from "
    "prompt_content (CLINC's intent.* families carry one fixed question per family and fixed "
    "option lists; MMLU, CSQA, SQuAD and defect_class questions vary per row and are kept whole). "
    "The strip is one function in python/qd_train, applied identically to sources and targets, in "
    "the exporter before the request is written and in the decontaminate oracle, so qd-prep "
    "containment's n-gram core is unchanged and the parity test still compares complete pair "
    "lists. A row left with fewer than 8 words is too_short, as today, and attestation v2 counts "
    "it per family-slot as too_short_after_strip; attestation v2 also records per family-slot the "
    "stripped strings' count and sha256. Keys (i) exact identity/content and (iii) MinHash are "
    "unaffected, and val and held-out rows never move. Before the full list is applied, the 1%, 2% "
    "and 5% subsample scans are re-run with the strip and the per-family exclusion rates reported "
    "beside the unstripped ones (CLINC 8.5/9.1/20.1%, MMLU 5.9-6.9%, code.defect_class 0.8-1.1%); "
    "a family-wide constancy rule strips intent.domain's fixed list but may miss "
    "intent.classification's per-domain option lists, and the post-strip per-family split decides "
    "whether a per-(family, option-set) rule is needed, by amendment here before the full scan "
    "(Fable's post-F ruling 3(a); the human ratifies by the commit that renames this file). "
    "Containment is measured on canonical option order on both sides, which is at least as "
    "inclusive as the shard order the trainer reads (Fable's post-F ruling section 4).",
)
set_value(
    "tool",
    dec["tool"],
    "qd-prep containment + attestation v2 + tools/real_tokenizer_pipeline.py "
    "--exclude-identity-keys, merged on main at 73758bd (lane L-prep); the template strip above is "
    "new code (lane L-prep2).",
)
set_value(
    "refuse",
    dec["refuse"],
    dec["refuse"]
    + "; attestation v2's remaining_hits is stated for the final rebuilt row set (by arithmetic on "
    "the first pair list: a contrast row's n-grams are a subset of its twin's, so a surviving "
    "twin's contrast row cannot hit; own-prose rows are first-build sources). When A7 fails, "
    "exclude the offending own-prose or contrast row and rebuild; never touch a val or "
    "held-out row",
)
na = orig["data"]["not_added"]
set_value(
    "unseen_language",
    na["unseen_language"],
    "superseded: G6 was answered (download yes at 9f6b1fa); v3b's 834 template rows stay and the "
    "commitpackft G6 source is added (data.sources, last entry)",
)

# --- format: eval refusal (review bullet 13), freeze (bullet 14) ---
rb = orig["format"]["refusal_both_ways"]
sub(
    f'"training": {q(rb["training"])}',
    f'"training": {q(rb["training"])},\n      "eval": '
    + q(
        "tools/real_ft_run.py --score-checkpoint and the trajectory scorer refuse a checkpoint "
        "whose recipe prompt_format (absent = 1) differs from render.PROMPT_FORMAT; the test fails "
        "on the pre-change code (GAP-V5-SCORE-CHECKPOINT-LACKS-PROMPT-FORMAT-REFUSAL-2026-10-02)"
    ),
)
set_value(
    "freeze",
    orig["format"]["freeze"],
    "The unpinned v4 reader on main is the uncommitted edits in the main checkout "
    "(crates/qd-metal, crates/qd-runtime serve/service, crates/qd-preflight; another session's, "
    "not lane L-metal's, per the lead's correction in "
    "AUDIT/v5-plan-2026-10-02/fable-v5-plan-review.md); the excluded re-score runs from worktrees "
    "pinned at 0264732, rungd from the overlay at 01b6db2. That session commits its own edits and "
    "runs its GPU checks from a worktree pinned at that commit; main is then free. The build may "
    "run on the v5 branch; the build commit is on main before launch.",
)

# --- seeds (review bullet 15); v5 report-only readouts (post-F section 1) ---
set_value(
    "seeds_3_4",
    orig["seeds"]["seeds_3_4"],
    orig["seeds"]["seeds_3_4"]
    + " seeds34 takes --f-ledger and --ft-row as paths (qd_post_f_rules.rs:243-252); L-v5-rules "
    "confirms its identity checks are not F-specific.",
)
sub(
    f'"ledger": {q(orig["seeds"]["ledger"])}',
    f'"ledger": {q(orig["seeds"]["ledger"])},\n    "report_only": '
    + q(
        [
            "per seed, from the trajectory-ood rows: the first retained step at which "
            "ood_abstain.prose has 2*n >= n_total, and whether it still holds at the final step "
            "(learned-then-lost vs never-learned) (Fable's post-F ruling section 1)",
        ]
    ),
)

# --- arm_noul_weight: what, flag, identity, launch condition, report-only ---
arm = orig["arm_noul_weight"]
set_value(
    "what",
    arm["what"],
    "v5's recipe plus --noul-weight 4, seeds 0, 1, 2, after v5 (and seeds 3-4 if they run) and "
    "before J5' (Fable's post-F ruling section 1; the human's answer 1 at 0b559bb).",
)
set_value(
    "flag",
    arm["flag"],
    "--noul-weight w multiplies the cross-entropy at every supervised letter position whose target "
    "is the noul letter (Z) and whose row's family is code.defect_class (noul_weight_scope = "
    "'code.defect_class'); CLINC's Z-gold rows and every other family are unweighted; the ~15% "
    "rise in letter-loss mass is recorded, not renormalised. The letter loss stays a mean over "
    "supervised positions (sum of w_i * ce_i over the count of supervised positions), so w is a "
    "pure multiplier on a Z-gold defect_class row's pull and every other row's pull is unchanged; "
    "the span channel's abstain row is not weighted. New code in python/qd_train/fused_ce.py (a "
    "per-position weight) and python/qd_train/backbone.py (_letter_loss); recipe keys noul_weight "
    "and noul_weight_scope recorded only when given; refused with w <= 0 or non-finite.",
)
set_value(
    "ft_rows",
    arm["identity"]["ft_rows"],
    replaced_or_die(
        arm["identity"]["ft_rows"],
        "except exactly one added key, noul_weight = 4.0.",
        "except exactly two added keys, noul_weight = 4.0 and noul_weight_scope = "
        "'code.defect_class'.",
    ),
)
set_value(
    "launch_condition",
    arm["launch_condition"],
    "qd-post-f-rules v5-noulw --room prints room or no_room from v5 seeds 0-2's eval rows into "
    "$Q/v5nw.room before the arm's slot; the arm waiter runs iff the word is room, or "
    "V5NW_HUMAN_YES is pinned; otherwise it logs SKIPPED (R7) and J5' takes the slot.",
)
last_ro = arm["report_only"][-1]
sub(
    q(last_ro) + "\n    ]",
    q(last_ro)
    + ",\n      "
    + q(
        "per seed, from the trajectory-ood rows: the first retained step at which "
        "ood_abstain.prose has 2*n >= n_total, and whether it still holds at the final step "
        "(learned-then-lost vs never-learned) (Fable's post-F ruling section 1)"
    )
    + "\n    ]",
)

# --- readings: R4, R5, R9 ---
rd = orig["readings"]
set_value(
    "R4_lower_layers",
    rd["R4_lower_layers"],
    rd["R4_lower_layers"]
    + " And J6(f)'s own verdict under its pre-registration is not refused (Fable's v5 review F12). "
    "After F seeds 0-2 (prose 54/0/1 of 60) the with-flag count is 1 of 3 and, rule (ii) having "
    "fired, is read x/5 over F seeds 0-4; the '2-of-3 vs 0-of-3' rationale is moot and the rule's "
    "text stands (Fable's post-F ruling section 1).",
)
set_value(
    "R5_noul_weight_scope",
    rd["R5_noul_weight_scope"],
    "the weight applies to Z-gold letter positions of code.defect_class rows only: the parity "
    "argument is about that family's prose rows, and weighting CLINC's 3,648 Z rows would confound "
    "the arm with CLINC in-distribution effects (Fable's review amends Fable's 'on Z-gold letter "
    "rows')",
)
sub(
    f'"R8_needle_landing": {q(rd["R8_needle_landing"])}',
    f'"R8_needle_landing": {q(rd["R8_needle_landing"])},\n    "R9_pause_after_seed_0": '
    + q(
        "after v5 seed 0's epoch-score-val row, qd-post-f-rules v5-pause prints continue iff "
        "val_top1.span has 5*n >= 4*n_total and the 8K needle worst bucket has 2*n >= n_total on "
        "the rebuilt suite; otherwise pause: the waiter starts seeds 1-2 only once V5_CONTINUE is "
        "pinned by the human. A spending rule, not a gate (F seed 0/1 span 0.902/0.912, "
        "f4feac15/aeca8d69)."
    ),
)

# --- amendments_pending additions (review bullet 16, post-F, G6) ---
last_ap = orig["amendments_pending"][-1]
sub(
    q(last_ap) + "\n  ]",
    q(last_ap)
    + ",\n    "
    + ",\n    ".join(
        q(s)
        for s in [
            "C2b's printed rule text",
            "the census: 9-10k supply and the build's peak RSS",
            "containment runtime and exclusion count per source family on the final set",
            "(A7) per-family val and held-out counts and identity-key sets equal to v4's for every "
            "non-CLINC family (val: code.defect_class 2,304, commonsense 1,197, knowledge 1,485, "
            "qa.answer_span 5,139; held-out: code.defect_class 2,182, qa.answer_span 6,563, "
            "qa.answerability 24,160); CLINC val 1,572 (within_domain 1,571) and held-out 1,262 "
            "per family; any other difference refuses (rule 2)",
            "own-prose-v1/files.jsonl sha256 and the human's ticked repo list",
            "the first seed's measured trajectory-scoring time",
            "the pin names V5_CONTINUE, V5NW_HUMAN_YES, v5nw.room",
            "the template strip's subsample re-run: per-family exclusion rates with and without "
            "the strip at 1%, 2% and 5%, the stripped strings' count and sha256 per family-slot, "
            "and whether a per-(family, option-set) rule was needed",
            "the G6 source: per-language rows after the licence filter and exclusions, the "
            "per-language cap, and the examples sha256",
            "C3's fsucc reading: the fsucc word, the decision JSON's sha256, "
            "detail.arms.j6dv4.wins, and which branch of the human's advance answer applies",
        ]
    )
    + "\n  ]",
)

# --- build_order: prerequisites, decontam, queue order ---
bo = orig["build_order"]
sub(
    q(bo[0]),
    q(
        "0. Prerequisites: L-prep merged (containment, attestation v2, --exclude-identity-keys; "
        "73758bd); L-j6g's checker subcommand merged (c314398); the template strip and its "
        "subsample re-run (L-prep2) read before the full scan; the last v4 readers on main done "
        "(format.freeze)."
    ),
)
sub(
    q(bo[3]),
    q(
        "3. Decontam: qd-prep containment over the rendered rows (L-prep's request form, with the "
        "template strip applied identically to sources and targets); exclusions.txt; rebuild with "
        "--exclude-identity-keys; contrast rows derived after the rebuild's dedupe, split and "
        "exclusions."
    ),
)
sub(
    q(bo[7]),
    q(
        "7. Box queue after j6g -> STOP, on the human's yes (d24c865): v5 s0-s2 (R9 may pause "
        "after s0) -> seeds 3-4 iff spread -> noul-weight x3 iff v5nw.room is room or "
        "V5NW_HUMAN_YES is pinned -> J5' x3 (order: the human's answer 1 at 0b559bb)."
    ),
)

# --- the record of this pass ---
sub(
    f'"committed_by": {q(orig["committed_by"])}',
    f'"committed_by": {q(orig["committed_by"])},\n  "amendments_applied": '
    + q(
        "Fable's v5 review section 5 (AUDIT/v5-plan-2026-10-02/fable-v5-plan-review.md, Fable "
        "source sha256 04e969b0...) and the v5 items of Fable's post-F ruling, sections 1, 2(b), "
        "3(a) and 4 (AUDIT/post-f-2026-10-02/fable-post-f-ruling.md, source sha256 c70d2fb7...), "
        "with the human's answers at d24c865, 9f6b1fa and 0b559bb, applied by the lead on "
        "2026-10-02 through AUDIT/v5-plan-2026-10-02/apply_review_amendments.py, before any v5 "
        "build, shard, code change or ledger row exists. Contrast rows take the review's "
        "recommended form (section 2.4). The file stays DRAFT until the build-time values in "
        "amendments_pending are filled and it is renamed to campaign/v5-preregistered.json on main "
        "(binds_iff)."
    ),
)

out = json.loads(text)
assert out["launch"]["projected_cost_usd"]["total"] == 137.2
assert out["launch"]["projected_gpu_hours"]["total"] == 60.0
assert len(out["data"]["sources"]) == len(orig["data"]["sources"]) + 1
assert "R9_pause_after_seed_0" in out["readings"]
assert "eval" in out["format"]["refusal_both_ways"]
assert all(ord(c) < 128 for c in text)
PATH.write_text(text, encoding="utf-8")
print("amended", PATH)
