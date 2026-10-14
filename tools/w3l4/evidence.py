# ruff: noqa: E501
"""Wave-3 L4: closures, keeps, re-owns and board annotations with read-only evidence. Dry run unless --apply.

Evidence was gathered by rg/Read on ~/Code/research/ojas and ~/Code/research/tessl (no DevMap index
covers ojas; tessl's is not fresh), by gitpulse_insights/gitpulse_active_changes on both, and by
direct reads in this tree. Every record is appended through qd_train.gaps.append_gap.
"""

import sys

from qd_train.gaps import append_gap, current_records

APPLY = "--apply" in sys.argv
LANE = "wave3-l4-gaps-remainder (ft-5c92df4cc91)"
DAY = "2026-10-13"

OJAS = "ojas HEAD bc498c4 (main; gitpulse_active_changes: 0 dirty files)"
TESSL = (
    "tessl HEAD 2972027 (main; 4 dirty files at read time: CHANGELOG.md, docs/qwen35.md, "
    "src/qwen35_adamw.rs, tests/qwen35_train.rs; none is cited below as evidence)"
)
METHOD = (
    "Method: read-only rg/Read of the working trees; no DevMap index covers ojas and tessl's is "
    "is_fresh=false, so this is rg evidence, not graph-confirmed. The harness refuses git against a "
    "foreign tree, so HEAD shas were read from .git/refs/heads/main (ojas) and .git/logs/HEAD "
    "(tessl). No test or build was run in either tree."
)

cur = current_records()
SEEN: set[str] = set()


def emit(gid: str, status: str, answer: str, **extra: object) -> None:
    rec = cur[gid]
    assert gid not in SEEN, gid
    SEEN.add(gid)
    out: dict[str, object] = {
        "id": gid,
        "question": rec.get("question") or rec.get("finding"),
        "status": status,
        "lane": LANE,
        "checked": DAY,
        "answer": answer,
    }
    if rec.get("opened"):
        out["opened"] = rec["opened"]
    if status != "open":
        out["closed"] = DAY
    out.update(extra)
    if APPLY:
        append_gap(out)
    else:
        print(gid, status)


# ---------------------------------------------------------------- closures with code evidence

emit(
    "GAP-FABLE-CUDA-CUDARC-TWO-PINS-UNVERIFIED-2026-10-01",
    "resolved",
    OJAS + ". There is one cudarc pin workspace-wide: rg over every Cargo.toml in ojas "
    "(excluding target) finds cudarc only in ojas-cuda/Cargo.toml:22-30 "
    "(version =0.19.10, feature cuda-12080), and ojas/Cargo.lock:127-128 has a single "
    'name = "cudarc" at 0.19.10. The sibling crate the record feared (ojas-qwen35-cuda, '
    "cuda-12080 beside ojas-cuda's then cuda-13040) no longer exists as a directory at HEAD "
    "(ojas-cuda absorbed it; ojas/tasks/gp-cuda-backend-provider.md:69). The question, what the "
    "build does with two different cuda-XXXXX selections, is moot: no workspace member selects a "
    "second. " + METHOD,
    evidence_commit="bc498c4",
    board_card="ojas ft-71629756 (L-CUDA rows, per ft-5c92df4c)",
)

emit(
    "GAP-OJAS-ADVICE-MROPE-COLLAPSE-INFERRED-2026-10-01",
    "resolved-with-residual",
    OJAS
    + ". The refusal this record asked for exists on the ojas side: ojas-qwen35/src/config.rs:22-30 "
    "documents that mrope_section is accepted only when its entries are positive and sum to "
    "rotary_dim/2 and that text-only is enforced per sequence (reserved vision token ids refused), "
    "and :545-585 parses and checks the fields; tests/cpu_config.rs:293 "
    "rope_and_mrope_are_read_and_refused_where_they_do_not_collapse pins it. "
    + TESSL
    + ": tessl still does not read the fields (src/qwen35_model.rs:174 comment). The collapse "
    "argument itself (angle = position * inv_freq whichever stream a section assigns) is still an "
    "argument: the falsifier the record named, a real-2B oracle loading the snapshot config, is "
    "ojas-qwen35/tests/gpu_parity.rs:735, an #[ignore] GPU test not run here. " + METHOD,
    evidence_commit="bc498c4",
    residual="The real-2B collapse check (gpu_parity.rs:735) has not run; owner ojas (L-ojas-qwen35 lane).",
)

emit(
    "GAP-L-CUDA-ORACLE-TESSL-PATH-INCLUDE-2026-10-01",
    "resolved",
    OJAS + ". The #[path] into the sibling tessl checkout is gone: "
    "ojas-cuda/tests/reference_gdn_published_vs_tessl.rs:35 now includes "
    "fixtures/tessl/gdn_train_published.rs (an in-crate byte copy pinned by sha256, the module "
    "docs at :14-26), and ojas-cuda/tests/fmt_boundary.rs:127 "
    "no_path_attribute_in_this_crate_leaves_the_crate scans every .rs file for a #[path] that "
    "escapes the crate (its pre-fix-case test is :175). The crate dir ojas-qwen35-cuda named in the "
    "record is absent at HEAD; the file moved to ojas-cuda. Not run here. " + METHOD,
    evidence_commit="bc498c4",
)

emit(
    "GAP-TESSL-GELU-DUP",
    "resolved-with-residual",
    TESSL + ". The two disagreeing GELUs are one: kernels/act.h:24-30 defines "
    "tessl_gelu_pytorch_tanh once (clamped cubic and tanh argument, outer factor the unclamped x, "
    "comment :16-23 says so), and kernels/mlp_gelu_tanh.metal:20,32 and kernels/qwen35_mlp.metal:50-51 "
    "call it; there is no file-local helper left. The test reference is tests/common/mod.rs:472-476 "
    "(gelu_pytorch_tanh, same form) and tests/q4_interleaved.rs:418-420 uses it instead of the "
    "inline clamped-xc expression at the old :448-450. " + METHOD,
    evidence_commit="2972027",
    residual="tessl's tests were not run here; the closure is a code-read of one definition and its callers.",
)

emit(
    "GAP-TESSL-HOST-LEASE-EXCLUSIVE-BLOCKS-PARALLEL-MAPPING-2026-10-03",
    "resolved-with-residual",
    TESSL + ". Read at HEAD, not just relayed: tessl/src/runtime.rs:854-860 acquire_access takes "
    'access_busy with compare_exchange(false, true) and returns "runtime busy: another host '
    'mapping, encoder, or submit is active" to a second caller; runtime.rs:552 and :728 state the '
    "runtime is single-threaded (!Send, !Sync). So one HostMapping at a time is confirmed (the old "
    "citation runtime.rs:606-618 has moved to :854-862). " + METHOD,
    evidence_commit="2972027",
    residual="The ~2-4 ms copy cost at 8K stays inferred; no measurement was made.",
)

for gid, why in (
    (
        "GAP-TESSL-GITPULSE-COLLISIONS-STALE-WORKTREES-2026-10-06",
        "gitpulse_insights on /Users/bharath/Code/research/tessl today (2026-10-13)",
    ),
    (
        "GAP-TESSL-STALE-WORKTREES-BREAK-COLLISION-SCAN-2026-10-06",
        "gitpulse_insights on /Users/bharath/Code/research/tessl today (2026-10-13)",
    ),
):
    emit(
        gid,
        "resolved",
        why + " returns collisions ok=true, failed_worktrees=0, scanned_worktrees=10, "
        "unscanned_worktrees=0, truncated=false; the worktree list has no target/wt_head or "
        "target/wt_norm, and rg finds neither name under tessl/.git/worktrees (nine registrations, "
        "all alive). The prune ft-62f3bba tracked has happened. overlapping_files=0 over a complete "
        "scan now means clean, for the 10 worktrees scanned.",
        board_card="tessl ft-62f3bba (worktree prune), per ft-5c92df4c",
        evidence_commit="gitpulse_insights tessl 2026-10-13",
    )

# ---------------------------------------------------------------- kept with evidence

emit(
    "GAP-TESSL-PROMOTED-44-ASSERT",
    "open",
    TESSL + ". Still open, re-read: tests/promoted_kernels.rs:120-131 "
    "promoted_count_matches_the_kernel_sources still asserts PROMOTED.len() == 44 (the 44 at :126). "
    "Nothing was added to PROMOTED, so K5/K6 are not yet breaking it; the trap is unchanged. "
    + METHOD,
)

emit(
    "GAP-OJAS-QWEN35-ABSENT-KEY-DEFAULTS-INFERRED-2026-10-01",
    "open",
    OJAS + ". Still inference: ojas-qwen35/src/config.rs:286-291 says the evidence for defaulting "
    "absent attn_output_gate and mamba_ssm_dtype is tessl's tiny fixture omitting both and matching "
    "transformers' autograd, and :374-395 implement the defaults; no read of Qwen3_5TextConfig "
    "source is cited. Doc-only evidence either way, so the record stays open. " + METHOD,
)

emit(
    "GAP-L-CUDA-ORACLE-QWEN35-GDN-FIXTURE-RULE-UNNAMED-2026-10-01",
    "open",
    TESSL + ". Unchanged: tests/fixtures/qwen35/ still lists qwen35_gdn_{a,a_log,b,dt_bias,k,q,v,"
    "state0,state_f64,state_hf,y_f64,y_hf}.npy, none carrying the word published, and "
    "scripts/gen_qwen35_fixtures.py has no occurrence of published. Renaming or annotating is "
    "tessl's call (rule 6 keeps Lappi out of it); re-owned to tessl. " + METHOD,
    owner="tessl",
)

emit(
    "GAP-OJAS-QWEN35-2B-STAGING-HEADROOM-2026-10-01",
    "open",
    OJAS + ". The ojas half is still open: ojas-qwen35/src/state.rs:496 staging_tensors still "
    "exists and is called at :536 and :735 (load_state), so the migration this record's "
    "resolution_plan ends with (delete staging_tensors) has not happened. Board: the ojas half is "
    "ojas ft-c57c98e0, per ft-5c92df4c. The 50.20/51.54 GB measurements are not re-run. " + METHOD,
    board_card="ojas ft-c57c98e0",
)

emit(
    "GAP-OJAS-QWEN35-GPU-PARITY-NOT-RUN-2026-10-01",
    "open",
    OJAS + ". Still not run, and a count drift: ojas-qwen35/tests/gpu_parity.rs now holds 10 "
    "#[ignore] items (the record says 7 gpu_* tests), module doc :2 says cargo test never runs one. "
    "No run evidence is in the tree or in ojas; stays open for the Mac GPU window. " + METHOD,
    owner="ojas (L-ojas-qwen35 lane)",
)

emit(
    "GAP-LTRAINER-CLIP-COEF-F32-TOWER-F64-HEAD-2026-10-01",
    "open",
    OJAS + ". Still true at the ojas side: ojas-qwen35/src/step.rs:259 clip_coefficient returns "
    "Result<f32>, taking the tower squared norm in f64. The record's rung (b) measurement was not "
    "run. " + METHOD,
)

DEVICE_IDS = [
    "GAP-L-CUDA-M0-DEVICE-NOT-RUN-2026-10-01",
    "GAP-L-CUDA-M0-NVRTC-PTX-JIT-PATH-2026-10-01",
    "GAP-L-CUDA-M0-LINUX-GLIBC239-NOT-RUN-2026-10-01",
    "GAP-L-CUDA-GDN-DEVICE-NOT-RUN-2026-10-01",
    "GAP-L-CUDA-GDN-BITWISE-MIRROR-ON-DEVICE-2026-10-01",
    "GAP-L-CUDA-GDN-TRIGGER-TIMING-UNMEASURED-2026-10-01",
    "GAP-L-CUDA-GDN-TEST-BINARY-GLIBC239-2026-10-01",
    "GAP-L-CUDA-SMALL-DEVICE-NOT-RUN-2026-10-01",
    "GAP-L-CUDA-SMALL-K6-LIBDEVICE-COSF-SINF-2026-10-01",
    "GAP-L-CUDA-M1-DEVICE-NOT-RUN-2026-10-01",
    "GAP-L-CUDA-M1-GDN-TIMING-BUDGET-ESTIMATE-2026-10-01",
    "GAP-L-CUDA-SMALL-K10-VIEWLESS-GEMM-COPIES-2026-10-01",
    "GAP-L-CUDA-OJAS-LINUX-LINKED-NOT-RUN-2026-10-01",
    "GAP-L-CUDA-GDN-SPEED-INFERRED-2026-10-01",
]
DEVICE_ANSWER = (
    OJAS
    + ". Kept open and re-owned to the ojas board (ojas ft-71629756: K10 bench, GDN timing, the "
    "GH200 run, per ft-5c92df4c). Evidence is DOC-ONLY, so it does not close the record: "
    "ojas-cuda/README.md:39-43 says the GH200 ran rung0 (174/174) and every device-test binary on "
    '2026-10-05, cross-built from the standalone crate, and that "the merged crate has not yet '
    'run on a device"; ojas/tasks/gp-cuda-backend-provider.md:79 and :106 list the GH200 run, the '
    "GDN timing numbers and the sm_90 K10 bench as still open; Lappi's "
    "HANDOFF/lead-pipeline-2026-10-03.md:540-542 reports the 10-05 run from queue markers "
    "(q-rung0.log, q-cudadev.log), which are on the box and not in this tree. Path drift: the "
    "ojas-qwen35-cuda directory these records name no longer exists; its tests live in "
    "ojas-cuda/tests (device_k0, device_k1, device_k3_gates_published, device_k4_conv1d, "
    "device_k6_qk_norm_rope, device_k7_rmsnorm, device_k8, device_k10_ce_rows, device_k11, "
    "device_gdn_published, device_gdn_published_mirror, device_backend; there is no device_k9 file "
    "at HEAD). " + METHOD
)
for gid in DEVICE_IDS:
    emit(
        gid,
        "open",
        DEVICE_ANSWER,
        owner="ojas (board: ojas ft-71629756)",
        board_card="ojas ft-71629756",
    )

emit(
    "GAP-CUDA-EMU-STAGED-IN-LAPPI-BELONGS-IN-OJAS-2026-10-12",
    "open",
    OJAS + ". Re-checked: rg -uu cuda_emu over /Users/bharath/Code/research/ojas (excluding "
    "target*) finds no file, so ojas still has no copy; Lappi's tools/cuda_emu/ (Cargo.toml, "
    "src/) is still here. Moving it is an ojas-session task (CLAUDE.md rule 6); no board card "
    "exists for it among those listed on ft-5c92df4c. " + METHOD,
)

emit(
    "GAP-TESSL-GDN-VALUE-HEADS-UNVERIFIED-FOR-4B-9B-2026-10-06",
    "open",
    "Half answered, re-owned. " + TESSL + ": tessl's training step does support more value heads "
    "than key heads: src/qwen35_train.rs:23-25 (scope note: value heads repeated from key heads "
    "before gdn_train, as transformers does), :693-695 repeat_heads and :724 its gradient; the "
    "(2, 4) test is in tests/qwen35_train.rs, which is dirty and not cited. "
    + OJAS
    + ": the refusal is ojas's: ojas-qwen35/src/config.rs:504-510 refuses linear_num_value_heads != "
    "linear_num_key_heads (module doc :12-16 says the refusal is ojas's Tape GDN path, not "
    "tessl's). So the Rust trainer over ojas-qwen35 still cannot train a 4B/9B; whether "
    "qd-train-metal over tessl can is not established here (no run). Board: tessl "
    "ft-b7376d2e / ft-836fdba, per ft-5c92df4c. " + METHOD,
    owner="tessl ft-b7376d2e / ft-836fdba; ojas Tape GDN repeat",
    board_card="tessl ft-b7376d2e, ft-836fdba",
)

# ---------------------------------------------------------------- board annotations (item 3)

NOTE = (
    "Annotation only: ft-5c92df4c criterion 4 names the owning board card for this record. "
    "Nothing was re-verified for this record in this pass, and nothing was written to the other "
    "repo's board."
)
ANNOTATE: dict[str, str] = {
    "GAP-RSIJEV-RELEASE-TOKENIZER-JSON-IS-NOT-THE-SERVED-TOKENIZER-2026-10-06": "tessl ft-7d20b218 (rsijev-metal tokenizer; until rsijev-metal has a repo)",
    "GAP-RSIJEV-V6-MAIN-EXIT-TEMPERATURE-TWO-VALUES-2026-10-06": "tessl ft-7d20b218 (rsijev-metal main-exit temperature)",
    "GAP-DEVMAP-RSIJEV-REFERENCE-CLONE-NOT-INDEXED-2026-10-06": "tessl ft-7d20b218 (rsijev-metal unindexed clone)",
    "GAP-DEVMAP-METAL-ENTRY-POINTS-ARE-NOT-SYMBOLS-2026-10-06": "DevCouncil ft-72b89073 (Metal entry points)",
    "GAP-DEVMAP-METAL-EDGES": "DevCouncil ft-72b89073 (Metal entry points)",
    "GAP-OJAS-TESSL-DEVMAP-BLIND-SPOTS-2026-10-06": "DevCouncil ft-32522da4 (Rust call resolution, items 1-3 and 5) and ft-72b89073 (Metal entry points, item 4)",
    "GAP-OJAS-QWEN35-DEVMAP-WALK-INCOMPLETE-2026-10-01": "DevCouncil ft-32522da4 (Rust call resolution)",
    "GAP-STRESS-SUBAGENT-NAVIGATED-WITHOUT-DEVMAP-MCP-2026-10-06": "DevCouncil ft-620d41f2 (subagents without DevMap tools)",
    "GAP-DEVMAP-DEAD-SYMBOLS-MISSES-STRUCT-LITERAL-USE-2026-10-06": "DevCouncil ft-400418e8 (struct-literal dead symbols)",
    "GAP-DEVTYPE-LAPPI-LANE-NO-LISTAGENTS-2026-10-06": "GitPulse ft-8e09f101 (same-worktree blindness)",
    "GAP-GITPULSE-LAPPI-CONTEXT-IS-LOSSY-UTF8-2026-10-06": "GitPulse ft-d10620bf (GitPulse caller gaps)",
    "GAP-GITPULSE-AI-TESTS-READ-REAL-TOOLS-JSON-AND-COULD-RECORD-2026-10-06": "GitPulse ft-d10620bf (GitPulse caller gaps)",
    "GAP-GITPULSE-N8N-EMAIL-ENGINE-UNTRUSTED-2026-10-06": "none possible: n8n-email-engine is untrusted in GitPulse, so no card can be filed until it is trusted",
}
for gid, card in ANNOTATE.items():
    status = cur[gid]["status"]
    emit(
        gid,
        status,
        NOTE + " Owning board: " + card + ".",
        board_card=card,
    )

print("records", len(SEEN))
