# HANDOFF L-cuda-oracle (2026-10-01): float64 host references for the CUDA Qwen3.5 kernels

Lane L-cuda-oracle of "train Lappi with ojas". Host-only Rust on the Mac; no GPU, no ledger row
(nothing here is a training run). Every file below is in
`/Users/bharath/Code/research/ojas/ojas-qwen35-cuda/` (paths the brief assigned this lane:
`tests/reference/**`, `tests/reference_*.rs`, `tests/fixtures/**`). `Cargo.toml` was not touched.
Nothing in ojas is committed by this lane (the brief forbids git in ojas); the ojas coordinator owns that.

**Interim state (written after K2 landed, before K3–K11):** this file is rewritten at lane end.

## K2: GDN at the published rule (done)

- Reference: `tests/reference/gdn_published.rs`. A port of tessl `tests/common/gdn_train.rs`
  (sha256 `dc455ac993c29e8b34f4300c427aca55ff33ce437f9b67f78a454930eca98656`) plus the Metal
  kernel's checkpoint seam: the forward returns checkpoints `[B,H,ceil(T/64),Dk,Dv]` (state entering
  token 64c, before its decay; `gdn_train.metal:96-101`), and the backward consumes them and
  recomputes each 64-token chunk (`gdn_train.metal:197-206`).
- Validation (raw output: `AUDIT/ojas-training-2026-10-01/l-cuda-oracle-failfirst-gdn-reverted-green.txt`):
  - golden: tessl's 13-case `gdn_published_*` corpus (copied byte-identical, all 107 files checked
    against tessl's `gdn_published_SHA256SUMS` in-test), worst 2.448e-16 of max|golden| (L8191),
    bound 1e-12 (`tessl/tests/gdn_fixtures.rs:63`).
  - tessl in process: tessl's `gdn_train_f64`/`gdn_train_bwd_f64` compiled in via `#[path]`, at
    T = 1, 63, 64, 65, 130 with tessl's flags (`tessl/tests/gdn_train.rs:365-371`), Dk = 128:
    o, final state, dq, dk, dv, dg, dbeta, ds0 all **bit-identical** (bound 1e-12).
  - derivative: central differences through the checkpointed backward at T = 1, 63, 64, 65, 130,
    worst 9.98e-10 (bound 1e-7, `tessl/tests/gdn_train.rs:140,168`).
  - checkpoint c equals the final state of a forward over the first 64c tokens, bitwise.
  - repo rule diverges from the golden by >1e-3 for T>1 and equals published bitwise at T=1.
- Fail-first: mutation 1 (published read replaced by the undecayed read) and mutation 2 (beta gate
  dropped) each fail the corpus, tessl-in-process and FD tests; reverted, green again.
  Raw: `l-cuda-oracle-failfirst-gdn-mutation1-repo-read.txt`, `...-mutation2-beta-dropped.txt`.

## Gaps so far

`GAP-L-CUDA-ORACLE-GITPULSE-TRUST-2026-10-01`, `GAP-L-CUDA-ORACLE-LISTAGENTS-UNAVAILABLE-2026-10-01`,
`GAP-L-CUDA-ORACLE-QWEN35-GDN-FIXTURE-RULE-UNNAMED-2026-10-01`.

## First command for the next lane

```
cargo test --offline --manifest-path /Users/bharath/Code/research/ojas/ojas-qwen35-cuda/Cargo.toml --test reference_gdn_published --test reference_gdn_published_vs_tessl
```
