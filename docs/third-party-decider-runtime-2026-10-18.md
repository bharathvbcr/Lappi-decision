# A third-party decision model behind `qd.sock`: scope and recommendation

Status: **design only, 2026-10-18.** Nothing here is built. The owner's decisions on §10 are
recorded at the end of §10. The build waits for the Vani eval. This lane downloaded nothing, installed
nothing and committed nothing. Every step that downloads, adds a dependency or writes code is listed
in §9 and needs the owner's yes first.

To close the open gaps, this lane did two things on the installed llama.cpp 0.7.0. First, it ran
throwaway probe servers over the already-downloaded d1 file: Metal and CPU-only, on Unix sockets
under the gitignored `build/d1-probe-2026-10-18/`. All of them are stopped, each by the PID this
lane started. Second, it read the upstream source and the Hugging Face configs. The results are in
§12, and the sections they change are corrected in place.

The question: can a third-party decision model answer the requests Lappi's callers already send?
The candidates are LiquidAI d1-3B, Perplexity pplx-decider-flash-v1.2 and Mapika decider-4b. The
callers are DevCouncil, GitPulse, DevType, and Vani via `~/Code/devtools/decide/lappi`. Each talks
to `$HOME/Library/Caches/qd/qd.sock` under `docs/caller-contract.md`, and none of them should
change.

Evidence labels: **V** means verified this session from the named source. **I** means inferred,
with the reasoning given. **U** means unverified. A number with no label is an estimate.

## 0. Recommendation

**Owner directive, later on 2026-10-18: "use metal and tessl by default".** The served runtime is a
native port on tessl/qd-metal, on Metal (§10, §13). The ranking that follows that directive is
§13.1, and it supersedes the table below. The table is kept because its llama.cpp row is now the
**eval path and parity oracle**, not the served runtime, and its facts still hold.

| Rank (before the directive) | What | Work (estimate, unverified) | Why this rank |
| --- | --- | --- | --- |
| 1 | **d1-3B through upstream `llama-server /v1/systemone`**, behind a Rust adapter that serves `qd.sock` (§3, §5) | Adapter ~6–8 days. Cross-lane contract work for the new refusal kinds in §10 Q4 (regenerating `fixtures/wire/`, teaching `qd_wire` the kinds, moving the counts pinned in `wire_fixtures.rs` / `refusal_is_not_noul.rs`) is **extra**, ~2 days. llama.cpp 0.7.0 is **already installed and loads d1** (owner-approved upgrade, §1). Calibration fit and eval: owner labelling time plus ~1 day | The only candidate with an official GGUF and an upstream Mac runtime. Text-only needs one 2.87 GB file, which is already on disk (§9). Its probabilities are a full per-option vector, so Lappi's own calibration, margin and conformal code apply unchanged |
| 2 | **pplx-decider-flash-v1.2 on tessl / qd-metal natively** (§6.2), behind the same `Answerer` seam | ~2–3 weeks. This revises the earlier 2–4-week audit figure: the bidirectional h256 kernel already exists in tessl, but the snapshot-free backend it forces was not priced in that figure | Higher Decision Index than d1 **on the same edition**. In the Space's own DI 0.3 data (`data/v03.json`), d1-3b scores v 38.90 (public 48.99) and pplx-decider-flash-v1.2 scores v 54.81 (public 60.06) (V, §12). The cards' 48.57 and 58.88 come from different editions and must not be compared. **Upstream llama.cpp is not a faithful route for v1.2**: its `pplx-decider` support appears to run the model causally, while the v1.2 card requires non-causal full attention (§6.1). So the native path is the only owner-controlled faithful one. One question per request (§4.2) |
| 2b | pplx-flash via a **patched llama.cpp** that turns off the causal mask for `pplx-decider` decisions | ~3–5 days plus upstream review time (C++ in someone else's project) | Cheaper than rank 2 if upstream accepts the patch. It is the owner's call whether contributing C++ upstream fits the language policy. It is not new owner code |
| 3 | Mapika decider-4b | Least kernel work of the three. Its answers are tied-`lm_head` letter rows at slot positions with the default causal mask (§6.3), which is the scoring qd-metal already does. So it needs N4 (config, which already passes `config.rs`'s checks), N5 (its own prompt) and P1–P2. There is no upstream llama.cpp type for it (V) | The licence is only in card metadata (no LICENSE file in the repo). Calibration degrades off-task (ECE 0.184 on the JevBench hard tier, by its own card). This Mac has an open MPS failure on its sibling decider-2b (`GAP-DECIDER-2B-MPS-PATH-GIVES-NO-USABLE-ANSWERS-2026-10-06`) |

**Nothing is served until the owner labels data.** Every candidate's probabilities are uncalibrated
for the owner's tasks: the d1 GGUF carries no temperature key (§5.3), and pplx's README disclaims
calibration for your data. Lappi's runtime refuses a slot shape its calibration table has no entry
for (`calibration_entry_missing`, `crates/qd-runtime/src/calibration.rs:18-21`). So the honest
state after rank 1 is built is the same as today: wired, and refusing until a table is fitted on
labelled examples (§8).

**The owner's decisions are recorded in §10** (Q1–Q5 accepted, Q6 deferred, the build waits for
the eval, Metal and tessl by default). The one open approval that changes what happens next is
pplx's weights and venv for a throwaway eval (§13.1).

## 1. What was checked

| Fact | Label | Source |
| --- | --- | --- |
| Homebrew `llama.cpp` 0.6.0_1 is installed (build 11429, commit `d81235049`, upstream "bump version to 0.6.0 (#29997)", 2026-10-05) | V | `llama-server --version`; GitHub commits API |
| That build registers `/v1/systemone` (`libllama-server-impl.dylib` holds the route and `server_decision_context::*` symbols) | V | `rg -a` over `/opt/homebrew/Cellar/llama.cpp/0.6.0_1/lib/libllama-server-impl.dylib` |
| `/v1/systemone` is upstream: a4cb4c6, "add /v1/systemone API (models: laya, julia-1, lev, openjev, kev)" (#29818), 2026-10-02, in `tools/server/server-decision.cpp` | V | ggml-org/llama.cpp history. It is not a Liquid fork |
| d1-3B support (88dcc46, #30110, 2026-10-07) and pplx-decider support (da263e7, #30044, 2026-10-06) are both **after** `d81235049` (ahead by 54 and 18 commits) | V | GitHub compare API |
| So the installed 0.6.0 cannot serve d1-3B or pplx-decider. A d1 request would most likely be refused as not a decision model of a known type | I | The two points above, plus the dylib string "This model is not a decision model" |
| Homebrew's current stable formula is **0.7.0** (tag v0.7.0, commit `620d2225`), 186 commits ahead of 88dcc46, so it contains both PRs | V | formulae.brew.sh `llama.cpp.json`; GitHub compare |
| Upstream `pplx-decider` (#30044) touches only conversion, `common`, `clip.cpp` and the server. It reuses `MODEL_ARCH.QWEN35`, makes **no `src/` or ggml change**, and folds the readout into `lm_head` rows at the label token ids | V | PR diff |
| Upstream therefore runs pplx-flash-v1.2 **causally**, which the v1.2 card says does not reproduce the checkpoint | V by source (§12, S4: the mask reads `cparams.causal_attn`, which defaults true, and nothing sets it false for `pplx-decider`), with the unread-portion caveat in S4. Not run, because no pplx GGUF is approved | No causal flag in the converter. `qwen35.cpp` has no decision or bidirectional reference. `server-context.cpp` never calls `llama_set_causal_attn` (all three V). A third-party GGUF card (Jakevin) reports Q4_K_M agreement of 87.4% causal against 97.2% non-causal (U, a third-party claim) |
| Upstream d1 (#30110) uses the generic label path with `label_groups` (an option's score is the max over its label tokens), `n_options_max = 255`, and `MODEL_ARCH.LFM2` | V | PR diff |
| systemone `probabilities` is a JSON **object keyed by option key**. Options are scored in the request's option order. `decision_sort_keys` reorders only rendered prompt input for LEV and the joint prompt, not options before scoring | V | `server-context.cpp` (`post_systemone`), `server-decision.cpp` |
| No upstream decision type exists for Mapika decider-4b | V | `common/common.cpp` decision-type list |
| **Update, later on 2026-10-18.** The installed 0.6.0 refuses d1: `failed to init decision model: unsupported decision model type: lfm2-d1`, exit 1 | V (the parent session's server log) | Relayed by the parent session |
| With the owner's approval, Homebrew llama.cpp was upgraded to **0.7.0** (build 11669, commit `620d2225c`; ggml 0.26.0 → 0.27.0). It logs `decision model type: lfm2-d1`, `model loaded`, and `/health` answers ok | V (relayed) | The parent session ran it, not this lane |
| The live reply shape is `{"model", "answers": {slot: {"type":"choice","choice","probabilities":{opt:p,…},"confidence"}}}`, and `probabilities` covers every option | V (relayed) | Parent smoke run |
| Smoke run: Vani's four `jd_triage` questions on 3 made-up postings gave 11 of 12 right. Per posting (4 questions in one request): 1,648 ms cold, then 490 and 274 ms warm, over loopback TCP | Smoke only: **not an eval and not a ledger row** (rule 5) | `Vani/build/d1-smoke/smoke.ts` (throwaway, gitignored) |
| `llama-server --host` accepts UNIX socket paths. It has `--api-key` / `--api-key-file` and `-np/--parallel` | V | `llama-server --help` (0.6.0) |
| d1-3B: `lfm2_vl` (LFM2 text model + SigLIP2 vision), 3.12B parameters, 32,768-token context, licence "LFM Open License v1.0" | V | HF `LiquidAI/d1-3B` card and config.json |
| LFM licence: commercial use only below US$10M annual revenue; redistribution with the licence and notices kept | V | HF licence text. Whether the owner's use is under the threshold is the owner's call |
| d1 card: 30 ms per decision on an M5 Pro; Brier 0.14; "calibrated" with no ECE given; Decision Index 48.57 on **DI 0.2.1** | V (card claim) | HF card. **Not measured on this Mac** |
| The d1 Q8_0 GGUF embeds `tokenizer.chat_template.systemone` and `lfm2.decision.type = lfm2-d1`, and has **no** `*.decision.temperature.*` key | V | GGUF header of the on-disk Q8_0 file, read-only (§9 says who downloaded it) |
| pplx-decider-flash-v1.2: Qwen3.5-4B-Base, BF16, ~9.08 GB, Apache-2.0. "Full-attention layers are noncausal over the whole request"; linear-attention layers stay causal; a separate readout head is applied "at each `<\|fim_suffix\|>` marker, one per question"; `MAX_OPTIONS = 255`; reference code needs "Python 3.12+ and a CUDA GPU" | V | HF card, `source/model.py`, `source/protocol.py`, `requirements*.txt` |
| pplx DI **0.3**: 58.88 joint, 60.05 one question per request | V | HF card |
| DI 0.2.1 and DI 0.3 are different editions, so the cards' 48.57 (d1) and 58.88 (pplx) are not directly comparable. On 0.3 alone: d1-3b v 38.90, pplx-flash v 54.81 | V | the cards; `multimodalart/jev-decision-index` `data/methodology.json` and `data/v03.json` (§12) |
| decider-4b v2.1: Qwen3.5-4B-Base, 8.41 GB bf16; Apache-2.0 in card metadata but no LICENSE file; ECE 0.031 in-task, 0.077 held-out, 0.147 on generated held-out families, 0.184 on the JevBench hard tier; needs `flash-linear-attention` for speed | V | HF card and file tree |

## 2. The contract a replacement must keep

These come from `docs/schema-api.md` and `docs/caller-contract.md`, and are binding:

- One JSON line in, one line out, on a `0600` per-user Unix socket, under one deadline. Interactive
  callers allow ≤ 300 ms; batch callers allow ≤ 10 s.
- `status` is one of `ok`, `refused` or `error`. Each `ok` slot carries all five keys
  `{value, conformal_set, score, noul, degraded}`. Unknown keys are refused in both directions.
- `score` is a calibrated margin `p_top − p_second`. `conformal_set` is split-conformal on
  calibrated probabilities. `noul` means **the model abstained**. A refusal never means abstention.
- `choice` takes ≤ 16 options. A caller narrows a larger list itself.
- Callers treat `degraded: true` as `model_abstained` (caller-contract §2).
- The callers check the envelope's `backend` string only against `^[!-~]{1,256}$`
  (`decide/lappi/client.go:326-330`). A new backend name is therefore accepted unchanged.

## 3. Where a third-party model plugs in

### 3.1 Not at `DecisionBackend`

`DecisionBackend` (`crates/qd-runtime/src/backend.rs:211-258`) is Lappi's own model seam:

- `prefill` takes the **prompt-format-2** prefix as a string.
- `decode_slot` takes a rendered suffix and must return exactly `k + 1` letter rows from a
  read-only decode of a snapshot.

A third-party model was trained on its own prompt, which for d1 is the GGUF's `systemone`
template. Serving it Lappi's format 2 is the failure `schema-api.md` exists to prevent: "the model
is served a prompt shape it never saw". Re-parsing format 2 back into structure inside a backend
would be a second renderer, the drift this repo has hit three times.

### 3.2 At the answering source, behind the same `Service`

The socket already has a single entry point: `Service::handle_request(&DecisionRequest) -> Response`
(`crates/qd-runtime/src/service.rs:191`). It calls `Runtime::answer(request, deadline)`
(`crates/qd-runtime/src/runtime.rs:362`). `Runtime::answer_inner` does three things in order:

1. hash pins, then `admission::admit` (`runtime.rs:385-388`);
2. `answer(&ctx, request)`, which renders, prefills, decodes and runs the permuted pass;
3. calibration, through `crate::calibration`.

The proposal:

- Make the `Service` factory produce an `Answerer`, not a concrete `Runtime`. Its one method is
  `answer(&DecisionRequest, Option<Deadline>) -> Response`, plus `identity()`.
- `Runtime` implements it unchanged.
- A new `SystemOneRuntime` implements it too. It reuses unchanged: wire parsing and every refusal;
  `admission`; `calibration::calibrate` (temperature, margin, conformal set, `noul_margin`); the
  derangement seed `render::second_pass_permutation`; and the whole `Service` lifecycle (idle
  eviction, poison and rebuild, `max_in_flight`).

The new parts:

- **Request translation.** `DecisionRequest` becomes systemone JSON.
- **Transport.** HTTP/1.1 over a Unix socket to a supervised `llama-server --host
  <dir>/llama.sock`. `std::os::unix::net::UnixStream` with a hand-written bounded request writer
  and response reader. **No new crate.** `serde_json` and `sha2` are already in `qd-runtime`'s
  dependencies. A TCP port is not used, so no loopback listener is open to other local users.
- **Answer translation.** The returned `probabilities` become `Logits` (`ln p`, with the reserved
  `noul` row at a finite floor, §5.3). They then go through the same calibration code a Lappi
  answer uses.

Language: Rust. The adapter belongs next to the wire, refusal and calibration code it reuses.
Written in Go it would be a second implementation of `wire.rs` and `calibration.rs`, which breaks
the "one canonical owner" rule. llama.cpp stays a separately versioned process with a documented
HTTP contract. That is the house pattern for an engine from another project; gusset is for
in-process Go↔Rust, and does not apply.

Where it lives: a new crate `crates/qd-systemone/` with a `qd-systemone-serve` binary. It depends
only on `qd-runtime`, as `qd-metal-serve` does. `qd-runtime`'s own change is the `Answerer` trait
and the factory type, roughly 60 lines, with `Runtime` as its first implementor.

### 3.3 What `Runtime::answer` does that the adapter must replace or decide

| Runtime step | Adapter behaviour | Decision |
| --- | --- | --- |
| `expect` hash pins (`check_hashes`) | `weight_hash` is the GGUF's sha256. `tokenizer_hash` is the sha256 of the GGUF's tokenizer metadata block, or the GGUF hash if that block cannot be isolated (state which in the identity). `head_hash` is the empty string, and a request pinning one is refused `head_hash_mismatch`. `calibration_hash` is the fitted table's hash, as today | Mechanical |
| `admission::admit` against the release's `trained_families` | A third-party model trained none of Lappi's families. `trained_families` would have to become `admitted_families`: families that **passed the owner's eval gate** (§8), written to the adapter's manifest with the ledger row that admitted them. The refusal kind stays `task_not_trained`, so callers keep matching it, and its message says "not admitted for this backend" | **Owner** (§10, Q2) |
| `code.defect_class` context-shape admission | Kept unchanged (it is a property of the request) | — |
| Rendering and the 17-row letter decode | Replaced by the translation in §4 | — |
| Permuted second pass (derangement) | Kept: §4.3 shows it can be expressed in systemone | — |
| Slot-isolation hash check | **Cannot run**: llama.cpp's KV cache is not host-visible. See §4.2 | **Owner** (§10, Q1) |
| Registered route | Refused `registered_route_unsupported`. A head file bound to Lappi's backbone has no meaning on another model. This needs a new refusal kind, which is a contract change | Mechanical, plus a corpus fixture |

## 4. (a) Field-by-field mapping

### 4.1 Request: `DecisionRequest` → systemone

| Lappi request | d1-3B via `/v1/systemone` | pplx-flash (`source/protocol.py`, rendered by the native backend) | Lost or changed |
| --- | --- | --- | --- |
| `schema_version` | — | — | none: checked by the adapter |
| `task` | — | — | **Not seen by the model.** It is used only for admission and calibration lookup. A format-2 Lappi model reads `task` in its prompt; a third-party model cannot |
| `route` | — | — | `registered` is refused (§3.3) |
| `context_b64` + `context_len` | `state` (a JSON **string**) | `State:\n…` | The bytes must be **valid UTF-8**, because systemone carries the state as JSON text. Lappi carries bytes. A non-UTF-8 context needs a new refusal (`context_not_utf8`), since lossy replacement would change the input silently. Vani's context is UTF-8 by construction: it is cut back to a rune boundary (V, caller-contract §4). DevCouncil's and GitPulse's diffs are raw bytes and can be non-UTF-8 (a Latin-1 source file, say). Those requests are refused, which costs nothing today because neither family is admitted for any third-party model |
| `question` (one per request) | `instructions`, per question | `Question i:\n<instr>` | For an **admitted** family, the instructions come from that family's pinned fixture, as measured. The d1 eval's shadow sends jdtriage's **combined** Prompt to every slot, not the per-family question (campaign pre-registration, amendment 1). So a d1 port admitted on that verdict renders the combined Prompt. The request's `question` is still validated against the family's contract string and refused if it differs, but it is not what d1 reads. For a family with no pinned fixture, the request's `question` is copied into every slot |
| slot `name` | the key in `questions{}` | marker order | none |
| `choice.options` (≤ 16, bare strings) | `criteria: {key: description}` | `A: …` lines | Lappi options carry no descriptions, **and descriptions change d1's answers.** On 3 synthetic postings × 4 Vani slots, sending the fixture's descriptions and sending the bare option string disagreed on **3 of 12** choices (§12, M5). So the adapter takes each **admitted** family's option descriptions from that family's sha256-pinned fixture (for Vani, `fixtures/caller/career-jd-triage-fixture.json`), recorded in the `admitted_families` manifest. The eval and the served adapter must send the **same** prompt variant (§8.2). For a family with no pinned fixture, `description = option` text. The adapter picks the key names itself (§4.3). The 16-option cap is **stricter** than systemone's 255, so nothing is lost |
| `score` (2–16 bins, CORAL ordinal) | `type: score`, `criteria`: an array of **2–10** levels | ordinal | **Bins 11–16 cannot be expressed**: refused `bins_over_backend_cap`, a new refusal and a contract change. No released family trains a score slot (`schema-api.md`:66-67), so nothing breaks today |
| `span` | **no equivalent** | **no equivalent** | **Lost.** Neither model has a line-pointer head. Refused `slot_kind_unsupported` (new kind). The only span caller is DevCouncil's `code.defect_class`, which is already unserved (`calibration_entry_missing`, caller-contract §4) |
| (none) | `type: noul`, a **yes/no question** | `noul` returns P(true) | **The same name with the opposite meaning.** In systemone, `noul` is a boolean question type: the d1 template prints "Reply with yes or no only", and upstream sets `answer["noul"] = P(true)` (V). In Lappi, `noul` is *abstention*. The adapter never sends a systemone `noul` question. A Lappi yes/no is a two-option `choice`. A test should pin that no systemone `noul` request is ever emitted, so the name cannot carry both meanings in one code path. This would be the fourth "one name, two meanings" defect in this repo |
| `expect` | — | — | §3.3 |
| `example_id`, `metadata` | — | — | Kept for the caller record only |
| `images` | (d1 supports it; needs the mmproj) | — | Not in Lappi's contract. Never sent |

### 4.2 Answer: systemone → Lappi `ok` envelope

systemone `choice` answers carry `choice`, `probabilities` and `confidence`. `score` answers carry
`score` (the expected value Σ i·pᵢ), `legend`, `probabilities` and `confidence` (V, upstream
README and `server-decision.cpp`).

| Lappi slot key | Derived from | Notes |
| --- | --- | --- |
| `value` | argmax of the **calibrated** probabilities, mapped back to the caller's option string | Not systemone's `choice`. Computed after Lappi's temperature, so a fitted table can change the top option only through the second-pass agreement rule, as it does today |
| `score` | Lappi margin `p_top − p_second` on calibrated probabilities | **systemone's `confidence` is dropped.** It is a different quantity, and caller-contract forbids a shared confidence number |
| `conformal_set` | `calibration::calibrate` on `ln p` with the fitted quantile for `(choice, k+1)` | Needs a table fitted on labelled data (§8). With no entry, the request is refused `calibration_entry_missing`, as today |
| `noul` | Lappi's rule: margin below `noul_margin`, **or** the two passes disagree | systemone has no abstain row. The reserved row is filled with a finite floor logit (§5.3), so it never wins on its own |
| `degraded` | See below | **Owner decision** |
| `backend` | e.g. `systemone/lfm2-d1/d1-3B-Q8_0/llama.cpp-<build>` | Passes `backend_pattern`. Never just `lappi` |
| score-slot `value` | argmax bin | systemone's expected-value `score` is dropped. Lappi's ordinal answer is a bin, not a mean |

**`degraded`, and the slot-isolation check (§10 Q1).** `schema-api.md` defines exactly three
sources of `degraded`. One of them is "the slot-isolation check could not run". Callers read
`degraded: true` as abstention. An adapter that cannot hash llama.cpp's KV cache therefore has two
honest options:

1. **Report `degraded: true`.** Every answer then reads as abstained in every caller, which makes
   the adapter useless.
2. **Replace the white-box check with a black-box check that runs, and amend the contract.** This
   is the recommended option. At start-up, and then on every N-th request, the adapter asks a
   canary request two ways: each slot alone, and all slots together. It requires the
   probabilities to match within a stated tolerance. A mismatch poisons the runtime and the
   service rebuilds it, exactly as `ReadonlyViolated` does now.

   **The tolerance, from measurement (§12, M3):**
   - On Metal, repeats of one request are bit-identical.
   - Asking a slot alone and asking it jointly differ by at most **4.5e-4** in any probability,
     across `-np 4` and `-np 8`.
   - Replacing the *other* questions with unrelated ones moved an answer by at most 1.4e-4, the
     same size as batch-composition noise.

   So the canary passes at **max |Δp| ≤ 2e-3** (about 4× the worst observed) on Metal, and must
   also return identical bits when the same request is repeated.

   CPU is **not** a valid serving backend for this check. With parallel slots its answers are not
   run-to-run deterministic (4e-3 spread), and alone and joint differ by up to 2.5e-2. If Metal is
   unavailable the adapter answers `unavailable`, never a CPU fallback.

   `schema-api.md` gains a sentence: for an external engine, isolation is checked black-box, and
   `SlotIsolationCheck` gains a `BlackBox { canary_digest }` variant. A check that ran is then
   reported as having run, and nothing that did not run is reported as passed.

**pplx-flash breaks isolation by design in joint mode.** Its full-attention layers are non-causal
"over the whole request" (V). So in a joint request, question 1's options attend to question 2's.
The adapter must send **one question per request** for pplx. That is also its better DI number
(60.05 against 58.88 joint).

d1 joint requests keep questions apart in the sense Lappi means (**V**, two ways):

- **Source.** `server_decision_group_tasks` copies only the shared prefix KV
  (`mem.seq_cp`) into child slots, and each question evaluates its own tail in its own sequence
  (§12, S3).
- **Measurement.** Changing the other questions moves an answer by no more than noise (§12, M3).

### 4.3 The permuted second pass, expressed in systemone

Lappi's step 4 requires a **derangement** of the option order and agreement between the two
passes (`schema-api.md`:388-403). A systemone request carries `criteria` as a JSON object.

- **Upstream keeps the request's order (V).** `common_json` wraps `nlohmann::ordered_json`, so
  object keys keep insertion order. `parse_questions` builds options in that order. The only sort
  (by key) runs when `choice_sorted` is set, which happens for CLEF only (§12, S1).
- **Measured.** A reply lists `probabilities` in the request's key order. Reordering the options
  moves the probabilities: up to 0.36 on one option under a cyclic shift, with 0 argmax
  disagreements in 24 slots (§12, M4). Position bias is real, and that is why the second pass
  exists.

The adapter still never relies on JSON object order. It names option keys `o00`…`o15` and
assigns option texts to them in pass order: identity for pass 1, Sattolo's derangement from
`render::second_pass_permutation` for pass 2. It writes the keys in that same order, which is also
their sorted order, so the order is the one intended whether the parser preserves or sorts keys.
It reads `probabilities` **by key**, never by position.

Both passes go in **one** HTTP request as two questions, so they share the state prefill.

For pplx, the two passes are **two independent sequences** (see the isolation point in §4.2). In
the native backend they are two rows of one batched forward (N2), not one joint prompt. A disagreement
between the passes is `noul`, as today.

### 4.4 Truncation

Upstream's `fill_task_laya` and `fill_task_d1omni` truncate to a token budget (V). **d1 does
not**: its `fill_task` only tokenizes (§12, S2).

- A request over the per-slot context fails loudly. Measured: HTTP 400,
  `{"type":"exceed_context_size_error","n_prompt_tokens":40072,"n_ctx":8192}` (§12, M6).
- So Lappi's no-truncation rule holds on this path. The adapter maps that error to the new refusal
  `context_over_backend_budget`, naming both numbers.
- To refuse **before** asking, it counts tokens with the same server's `/tokenize`.

**The budget is per slot.** `n_ctx_slot = n_ctx / np`, as the server logs (V). With `-c 65536
-np 8`, each question gets 8,192 tokens.

A 24,576-byte probe text at Vani's cap (caller-contract §4) is **4,158 tokens** (V, `/tokenize`,
§12, M6), so it fits with about 2× headroom. The probe text was a synthetic posting followed by one
English sentence repeated, not a real posting. Text with a different token density (CJK,
code) can differ, which is why the adapter counts and does not assume.

## 5. (b) d1-3B: can a Rust or Go adapter get calibrated choice probabilities?

**Yes, through the server; not through the library without reimplementing the server's decision
code.**

### 5.1 Server or library

The decision logic lives in `tools/server/server-decision.cpp`, not in `libllama` (V): the
prompt build (`fill_task_laya` / `fill_task_joint`), option rendering, temperature lookup and
answer formatting. Linking `libllama` from Rust would mean writing `-sys` bindings, a new
dependency, and **reimplementing** that file, which would then drift from upstream.

Supervising `llama-server` as a child process gets the maintained implementation, with no new
crate on the Lappi side. Recommended.

### 5.2 The supervised server

- Started by the adapter with an explicit argv. It is never found via `PATH` guessing: the binary
  path is configured and its sha256 recorded in the adapter's identity.
  `--host <qd cache dir>/llama-d1/llama.sock`, `-m <gguf>`, `-np 8 -c 65536`,
  `--api-key-file <0600 file>`. No `-hf`: the adapter never downloads.
  - `-np` must be at least the number of questions per request. Upstream groups questions in
    chunks of `n_slots` and prefills the shared state once per chunk (§12, S3). With `-np 4`, 8
    questions cost about 2× the time of 4 (§12, M7).
  - `-c` is `np` × the per-question budget (§4.4).
- **The socket's mode is not llama-server's to get right.** Measured: `--host <path>.sock` serves
  `/v1/systemone` over a Unix socket in 0.7.0, but creates the socket **0755**. It also logs "no
  API key is set and CORS allows all origins" (§12, M1). The adapter therefore creates the
  enclosing directory `0700` *before* starting the child, and passes `--api-key-file`.
- **llama-server leaves its socket file behind**, even after a clean SIGTERM (exit 0), and the
  next start on that path fails with "couldn't bind HTTP server socket" (V, two sessions
  independently, §12, M8). So before every child start, the adapter:
  1. checks the socket path is inside its own `0700` directory and is a socket, not a regular
     file or a symlink;
  2. connects to it with a short timeout;
  3. removes it only if the connection is refused (nothing listening), and refuses to start if
     something answers;
  4. removes it again after the child exits.

  A path that is not a socket is a startup error, never deleted.
- Readiness: wait for `/health` under a bounded start-up deadline. A cold start took about 2 s
  through Vani's loopback server (M9). This lane's Metal probe logged "model loaded" 3.8 s after
  launch (`-np 4 -c 32768`). Both are within the first-ask `unavailable` that callers already
  expect. Until then the service answers
  `unavailable`, which callers already handle (the first ask after a cold start is expected to
  come back `unavailable`; caller-contract §1).
- Lifetime: the adapter owns the child's PID and kills only that PID, on idle eviction or
  shutdown. This follows the house rule against kill-by-name.
- **Security impact.**
  - The change adds one more Unix socket inside a `0700` per-user directory. The directory, not
    the socket's own 0755 mode, keeps other users out. The socket is also guarded by
    `--api-key-file` (`0600`).
  - It opens no TCP listener.
  - Nothing outside the user's own processes can reach the model. Access is no wider than
    `qd.sock` today.
  - Context bytes go only to a local process. The model file is read from a configured path, and
    nothing is fetched (`-hf` is never passed).

### 5.3 Calibration

- systemone probabilities are `softmax((logit − max)/T)`. `T` is read from
  `<arch>.decision.temperature.<qtype>[.<bucket>]` and defaults to 1.0 (V, upstream).
- The d1 Q8_0 GGUF has **no** temperature key (V, header), so d1's probabilities are its raw
  softmax.
- The adapter turns them back into logits (`ln p`). Lappi's `calibration_fit` then fits
  temperature, conformal quantile and `noul_margin` per `(choice, k+1)` on the owner's labelled
  examples (§8), exactly as for a Lappi release.
- **The runtime refuses non-finite logits** (`BackendError::NonFiniteLogit` → `non_finite_logit`,
  `crates/qd-runtime/src/refusal.rs:452,479`). `calibrate` assumes every element is finite
  (`calibration.rs`, the comment above its softmax) (V). So neither the abstain row nor an option
  the server reports at exactly `p = 0` may be `ln 0 = −∞`.
- The adapter writes both as one fixed finite floor, **`FLOOR = −1e30`**: finite in f32, which
  `qd-calib-fit`'s `scores` requires.
  - A floor near `ln(f32::MIN_POSITIVE)` ≈ −87 would be **wrong**. The fitter's temperature range
    is `TEMPERATURE_LO = 0.05` to `TEMPERATURE_HI = 20.0` (V, `calibration_fit.rs:49,51`). At
    T = 20, −103/T ≈ −5.2, so the abstain row would keep about 0.5% of the softmax mass.
  - At −1e30, even −1e30/20 = −5e28 underflows `exp` to exactly 0 in f64. The abstain row then
    never wins and never enters a conformal set at any fitted temperature (V, arithmetic). A test
    pins both.
- **Fitting: the exact input (V, `src/bin/qd_calib_fit.rs:243-300, 410-440`).**
  - The fitter reads one JSONL file per eval row. Each letter line is
    `{eval_row_id, row_id, slot_name, kind, correct, rows, noul_row, gold_row, top, row_logits}`,
    with an optional `seed`.
  - It requires `noul_row == rows − 1`, `gold_row < rows`, `top < rows`, and `rows` finite f32
    `row_logits`. It refuses a duplicate `(eval_row_id, row_id, slot_name)`.
  - The adapter's eval mode writes exactly this, with the floor already applied. Every field the
    reader checks is one the adapter has. There is no `qd-calib-fit` change.
- **The ECE gate needs 100 rows per entry** (`ECE_MIN_SAMPLES = 100`; below it, `ece_state` is
  `NotRun`, V). Table entries are per `(kind, rows)`, so Vani's four slots share two entries:
  - `choice:6` is role_fit plus seniority;
  - `choice:5` is sponsorship plus work_mode.

  Each labelled posting therefore adds 2 rows per entry, and the fit split needs **≥ 50 postings**
  for the ECE gate to *run* (§8.2).

### 5.4 Is `/v1/systemone` real?

Yes, upstream, since 2026-10-02 (V). It is not a Liquid fork, and the d1 GGUF README points to
ggml-org/llama.cpp (V).

- 0.6.0 refused the model (V, §1).
- 0.7.0 is now installed with the owner's approval, and it loads and answers d1 (V, relayed, §1).
- **Metal agrees with llama.cpp's own CPU path (V, §12, M2).** Across 97 answers, the worst
  probability difference is **2.4e-2**, with **0 choice flips**. Three repeats each of three
  requests on Metal gave bit-identical probabilities (M3). That is the scope measured, not a
  general guarantee.
- What this does *not* establish is agreement with the unquantized model. The reference for that
  is the BF16/F16 GGUF (5,403,160,160 bytes) on CPU, a download nobody has approved. That check is
  listed in §9 as optional. Accuracy on the owner's task is the eval's job (§8), not a parity
  check's.

### 5.5 Latency

The card claims 30 ms on an M5 Pro. This lane's probe measured warm Metal timings on a synthetic
posting of about 60 words, over a Unix socket, taking the minimum of 5 runs (§12, M7):

| Request | `-np 4 -c 32768` | `-np 8 -c 65536` |
| --- | --- | --- |
| 1 question | 171–176 ms | **82–89 ms** |
| 1 choice slot with its permuted pass (2 questions) | — | **110 ms** (127 ms typical) |
| 4 questions (a Vani posting, no second pass) | 500–712 ms | **249–288 ms** |
| 8 questions (a Vani posting, with second passes) | 1,018–1,083 ms | **402–435 ms** |

**Reading:**
- `-np 8` is the configuration to use.
- A Vani posting with second passes is about 0.4 s against a 10 s batch budget.
- A single interactive choice slot is about 0.11 s against DevType's 300 ms. DevType's family is
  untrained for any model, so this is capacity, not a plan.

These are **probe figures, not ledger rows** (rule 5). Before any number enters a gate, it is
re-measured under `tools/mac_heavy.sh` on real posting lengths and written as a ledger row.

## 6. (c) The Qwen3.5-4B models: a kernel plan

### 6.1 First: does upstream llama.cpp already do it?

**Not faithfully.** Upstream's `pplx-decider` (da263e7, #30044) is a converter plus a server
decision type on the stock `QWEN35` graph. It makes no change to `src/` or ggml (V), and nothing in
it turns the causal mask off (V). So it runs v1.2's full-attention layers causally (V by source,
§12 S4). The v1.2
card says those layers are non-causal over the whole request.

The kernels Metal would need do exist upstream: `GGML_OP_GATED_DELTA_NET` and `GGML_OP_SSM_CONV`
have Metal cases (V). That Qwen3.5 lowers to them is I. So the gap is the mask, not Metal.

There is no official GGUF. A community one exists, `Jakevin/pplx-decider-flash-v1.2-GGUF`
(Apache-2.0, text-only, V). Its own card says its GGUF metadata is causal and that upstream's
template targets the older v1 prompt. Its runner applies the readout outside llama.cpp, in Python,
from the original checkpoint. Neither is a route for shipped code.

That leaves two faithful options:

- **2b, patch upstream.** For `pplx-decider` decisions, set causal attention off for the
  full-attention layers only. The mechanism exists (V, §12 S4):
  - `llama_set_causal_attn` sets `cparams.causal_attn`.
  - The attention mask reads it: `set_input_kq_mask(..., cparams.causal_attn)` in both
    `llm_graph_input_attn_kv` and `llm_graph_input_mem_hybrid`.
  - The recurrent inputs (`llm_graph_input_rs`, `build_rs`) do not reference it.

  So calling it unmasks the full-attention layers and leaves gated-delta-net causal, which is
  exactly v1.2's `decision_config.json` `"attention_mode": "noncausal_full_attention"`. The
  `build_recurrent_attn` body sits in a part of `llama-graph.cpp` that was not read, so the patch's
  first test is the P2 "GDN still causal" test. The template also has to move to v1.2's prompt. This is C++ in
  ggml-org/llama.cpp, with conversion through upstream's Python converter (a one-time build step,
  which the owner should rule on). After that, pplx-flash is the rank-1 adapter with a second model
  file, one question per request (§4.2).
- **Rank 2, native**, below. It is owner-controlled code under the language policy, with no
  dependency on upstream review.

### 6.2 Native in tessl and qd-metal

What exists (V unless noted):

- `qd-metal` runs Qwen3.5 from `config.json`: 18 GDN layers and 6 attention layers for the 2B
  (`crates/qd-metal/src/model.rs:1-21`, `config.rs:329-339`).
- Prefill attention is causal `nn::flash_attn_rows`. Continuations use `qwen35::attn_prefix_rows`
  over a shared snapshot.
- tessl **already has bidirectional attention**: `kernels/encoder_attn.metal` (header: "no causal
  rule: a query sees keys on both sides", per-row live length `lens[b]`, GQA via `Hkv`), with
  `encoder_attn_rows_h256_r16_g32` and bf16 tiled variants. It is exposed as
  `tessl::embedgemma2::encoder_attn{,_bf16}` (`src/embedgemma2.rs:271-318`).
- Qwen3.5 attention is head dim 256 (`tessl/src/qwen35.rs:2651`, and qd-metal refuses anything
  else at `config.rs:207-212`).

So the "non-causal attention kernel" the earlier estimate counted is not missing. What is missing:

| # | Work | Where | Notes |
| --- | --- | --- | --- |
| N1 | A per-layer attention mode in the Qwen3.5 layer loop: `full_attention` layers call `encoder_attn_bf16` (positions 0..T−1, `lens` per row, `window = 0`) in place of `flash_attn_rows`. qk-norm, partial RoPE and the output gate are unchanged | `qd-metal/src/model.rs` (`Model::run`); tessl unchanged, unless `encoder_attn` needs a bf16-output variant at h256 for this path | GDN layers stay causal (V, pplx `model.py` keeps linear-attention causal) |
| N2 | **The snapshot design does not survive.** With non-causal attention, the prefix's K/V depend on the suffix (the question and options), so "prefill once, snapshot, decode N slots read-only" is wrong for this model. Each question is a full forward over `[system, State, Question, Options, marker]`, and N questions are N batch rows of different lengths (`encoder_attn` supports per-row `lens`; GDN with right padding is causal, so the padding after the marker cannot reach it). The permuted pass is one more row | a new `qd-metal` backend kind, not `MetalBackend` | This is the cost the 2–4-week estimate did not price. It is also why this backend implements the §3.2 `Answerer` and not `DecisionBackend`: there is no snapshot to be read-only about |
| N3 | Readout head at marker rows: gather the final-norm hidden states at the `<\|fim_suffix\|>` positions and score them through `readout.safetensors` | `qd-metal` (`qwen35_score_rows` is already generic over head dtype, `model.rs:27-31`) | **V (§12, S5):** `hidden = backbone(...).last_hidden_state` (post final norm), gathered at the marker, then `torch.nn.Linear(2560, 255, bias=False)`. It is loaded with a strict `load_state_dict` from `readout.safetensors`, so the file holds exactly one tensor, `weight`, `[255, 2560]` BF16 (1,305,600 bytes plus header = 1,305,712). There is no bias, no extra norm and no nonlinearity. Logits are trimmed to the question's option count. The checkpoint's own temperature is 1.5539 (`decision_config.json`), which the adapter ignores in favour of Lappi's fitted table |
| N4 | 4B config admission | `qd-metal/src/config.rs` | **The 4B passes every existing check (V, §12, S5):** hidden 2560, 32 layers (24 `linear_attention`, 8 `full_attention`), heads 16/4, `head_dim` 256, GDN key heads 16 / value heads 32 (`v_heads % k_heads == 0`, which `GdnProjLayout` requires, `tessl/src/qwen35.rs:801`), key/value dims 128/128, conv 4, intermediate 9216, vocab 248320, `tie_word_embeddings` true, `attn_output_gate` true, silu, no attention bias, no MLP-only layers. What is new is the **checkpoint layout**: `Qwen3_5Model` (no `lm_head`), plus an MTP layer and a 24-layer vision tower that the loader must skip by name. That is about 1 day, unchanged |
| N5 | Tokenizer and protocol render, a Rust port of `protocol.py`'s layout (system prompt, `State:`, `Question i:`, `Options:`, codes `A`…, marker) | new crate | It must be byte-identical to `protocol.py`. Pin it with a golden-prompt test generated by the Python oracle (P1 below) |

**Parity tests against a Python reference oracle.** These are allowed: Python as an oracle
(language policy item 1), never shipped.

- **P0 (tessl): bidirectional attention at Qwen3.5's shape.** `encoder_attn` at h256, GQA 8:2 (2B)
  and **16:4 (4B)**, against an f64 CPU reference. Add a "future key is visible"
  test, the inverse of `tests/flash_attn_rows_h64.rs::head_dim_64_causal_matches_cpu_and_ignores_a_future_key`:
  a large key at t > t_q must change row t_q.
- **P1 (oracle fixtures).** `perplexity-ai/pplx-decider-flash-v1.2/source/src/pplx_decider/model.py` on **CPU fp32**
  (V: the code falls back to fp32 on CPU, so no CUDA is needed). It runs in a dedicated `uv` venv
  pinned to the repo's `requirements.txt` (torch 2.14.0, transformers 5.17.0). The shared ML venv
  has torch 2.12.1, so a new venv is a new dependency and needs a yes. Per layer, it emits the
  hidden states at a few fixed rows, the marker rows' readout logits, and the rendered token ids,
  for 3 fixed prompts (1, 2 and 16 options, one with a two-pass derangement).
  Any GDN fixture must name `rule="published"` (rule 9).
- **P2 (qd-metal GPU).**
  - Layer-by-layer hidden-state parity against P1 through the existing per-layer probe
    (`model.rs:258`).
  - Readout-logit parity within the bf16 tolerances `parity_row.rs` already uses.
  - Token-id equality with P1, so the render is byte-identical.
  - A "GDN still causal" test: changing a later token leaves earlier GDN-layer outputs unchanged
    while the attention layers' outputs change.
- **P3 (adapter-level).** One-question-alone against one-question-batched-with-others gives
  identical readout logits. This is §4.2's black-box isolation check, run as a test.

Estimate for 6.2: N1 2–3 days, N2 5–7 days, N3 1–2 days, N4 1 day, N5 2 days, P0–P3 3–4 days.
**About 2–3 weeks**, unverified.

### 6.3 decider-4b

Same base model and **identical config values** to pplx-flash (§12, S6), but packaged as
`Qwen3_5ForCausalLM`.

- **How it reads answers (V, `decider/model.py`):** `last_hidden_state` at each slot position,
  times `lm_head.weight[letters]`, then masked to the option count. These are tied-embedding
  letter rows, the scoring qd-metal already does.
- **Attention (I):** nothing in `config.json`, `decider_config.json`, `model.py` or `systemone.py`
  sets non-causal attention, so the default causal mask applies. Its batching, serve and
  shared-prefix files were not read.

So its plan is N4, N5 and P1–P2, with no N1–N3 and no new kernel. It is the cheapest native model,
and it still ranks 3rd for the licence and calibration reasons in §0.

## 7. Callers: what stays unchanged, and the one thing that does not

- DevCouncil, GitPulse, DevType and `decide/lappi`: unchanged wire, unchanged six readings,
  unchanged deadlines. The new backend string passes `backend_pattern` (V).
- **Model identity is lost at the `decide` layer.** `decide/lappi.Backend.Name()` returns the
  constant `"lappi"`, and every `decide.Answer` it builds says `Backend: "lappi"`
  (`decide/lappi/backend.go:126,167,227,254`) (V). Vani's eval keys its cells on that string
  (`Vani/internal/eval/report.go:176`, the shadow at `pipeline/runner.go:304`) (V). Two models
  behind `qd.sock` would therefore both report as `lappi` in Vani's eval.

  The eval plan (§8) avoids changing `decide` by running each model on **its own socket**: the
  adapter on `…/qd-d1.sock`, and Vani's shadow `decide.Backend` pointed at it. Promotion is then a
  socket-path change. If per-model identity is wanted inside Vani's report, the small fix is
  `Name()` returning `lappi:<envelope backend>`. That is a `decide` contract change, and needs the
  owner's yes.

  **For the eval being built now, the gap does not bite (V, graph-confirmed in Vani, §12, S7).**
  Line numbers are as read on 2026-10-18. Another session was editing Vani at the time, and its
  DevMap generation moved from 268 to 271 while this lane read it.
  - The shadow's annotation is written by `pipeline/runner.go::triageAnnotations` through
    `slotBackend`, which takes the `backend=` tag the `decide.Router` puts in `Detail`
    (`r.Shadow.Name()`, `decide/router.go:282`).
  - The parent session's d1 shadow is Vani's `decide.shadow: "local"`, a `/v1/systemone` backend
    (`internal/config/config.go:93,513-571`). It is built on `decide/jev` with a configurable
    `Name` (`jev.go:117,321`).
  - So it reports under its own name, not as `lappi`. The gap applies only to models served behind
    `qd.sock` through `decide/lappi`.

## 8. (d) Eval on the owner's real tasks

The tools exist in Vani (V): `vani label -n N`, `vani eval`, per-slot × backend precision
(correct/answered) and coverage (answered/labelled), the shadow table, and `Promotion.Verdict()`
with `MinPromotionLabels = 30` (`internal/eval/report.go:75-108`). Verdicts are:

- "insufficient labels" below 30;
- "not promotable" when shadow precision is below the primary's, or its coverage is lower;
- otherwise "promotable".

Promotion is a config change the owner makes, never automatic.

### 8.1 Before anything is labelled (§10, Q3)

`caller-contract.md` §5 records the owner's 2026-10-06 ruling: "Synthetic only". No real personal
data goes into training or eval. Whether job postings count as personal data no longer needs
settling: the owner's decision below covers them explicitly. Vani's own rule keeps the owner's name, contact and work authorization out of prompts, logs,
ledger and fixtures (V, `Vani/CLAUDE.md`).

**Decided 2026-10-18 (§10):** labelled real postings may be eval data for third-party deciders,
and never training data. Lappi-side, nothing is copied into `data/`. Labels stay in Vani's store,
where git holds only hashes (V, `evalcmd.go:21-23`).

### 8.2 The plan

**Precondition: met.** An earlier relay said Vani had no postings. It had looked in
`~/Library/Application Support/Vani`, a stray duplicate. Vani's live state is `Vani/.state`,
selected through `VANI_CONFIG` and `VANI_STATE_DIR`. There, `state/ledger/events.jsonl` holds
**3,927 events, 604 of them with `triage.` annotations** (line counts, read on 2026-10-18 at about
21:17 CDT; unique postings are fewer than or equal to that). No `eval/` label directory exists yet,
so labelling (step 1) can start now with `vani label` pointed at that state directory.

1. **Labels.** Vani's verdict needs at least 30 labelled postings per slot. A posting labels all
   four `career.jd_*` slots (role_fit, sponsorship, work_mode, seniority).

   **80 postings are needed**, split by posting hash before anything is read:
   - **50 for the calibration fit.** Each table entry is shared by two slots, so 50 postings give
     the 100 rows per entry that the ECE gate needs to *run* (§5.3). With fewer, the fit still
     works, but its ECE is reported not run.
   - **30 held out for the verdict.** Fitting on the postings that decide promotion would make the
     verdict measure the fit.

   That is four `vani label -n 20` sessions.

   **Same prompt as served.** The shadow must send the prompt variant the adapter would serve,
   meaning the option descriptions from the pinned fixture (§4.1). Descriptions moved 3 of 12
   probe answers (§12, M5). The shadow records the variant in its backend name or config, so a
   verdict is never read across variants.
2. **Fit.** The adapter answers the fit split and writes `--verdicts-out` rows (§5.3).
   `qd-calib-fit` (Lappi's existing fitter) writes a table per `(choice, k+1)`: k = 5, 4, 4 and 5 options, so rows 6, 5, 5 and 6. Its hash goes in
   the adapter's manifest. Until it exists, every request is `calibration_entry_missing`.
3. **Verdict.** Vani runs with Jev as primary and d1 as the shadow over the held-out split. Today
   that shadow is the parent session's `local` systemone shadow; after a build it would be the
   adapter's socket. `vani eval` reports per-slot precision and coverage for both, plus the
   verdict. Each candidate (d1, then pplx if rank 2 lands) is a separate run on its own socket
   (§7).
4. **Ledger.** Each run writes a ledger row: model file sha256, llama.cpp build, table hash, split
   hashes, counts. A run with fewer than 30 held-out labels per slot is `quick` and cannot promote
   (rule 8).
5. **Admission.** A family enters the adapter's `admitted_families` only on a "promotable" verdict
   with its ledger row cited (§3.3).

**Not covered.** DevCouncil, GitPulse and DevType have no labelled sets. Their caller records are
held out and "not eval data" by the owner's ruling (caller-contract §5). Their families are also
untrained for any model ("proposed for v6"). An adapter admits none of them until each has its own
labelled set and verdict. This is a coverage gap, stated rather than filled.

## 9. (e) Downloads needing approval

Sizes from the Hugging Face tree API (V).

| For | File | Bytes | State on this Mac |
| --- | --- | --- | --- |
| Rank 1, d1 text-only | `LiquidAI/d1-3B-GGUF/d1-3B-Q8_0.gguf` | 2,874,781,280 | **Done, owner-approved** (relayed by the parent session). It is on disk at `~/Code/research/models/LiquidAI-d1-3B-GGUF/` with `LICENSE` and `README.md`. The parent session downloaded it, not this lane. sha256 `2f0942d5a5f64cf69c3356d2be439b71644b1f0eb3a580912a5a0179eeaba77d`, which matches the Hub's LFS oid (the parent checked against the Hub; this lane recomputed the same hash locally) |
| Rank 1, alternative | `d1-3B-Q4_K_M.gguf` | 1,674,456,672 | Not needed unless memory or latency forces it. Its calibration would need a separate fit |
| Rank 1, images | `mmproj-d1-3B-Q8_0.gguf` | 583,109,728 | **Not needed** (V): the server loads d1 and answers text questions with no `--mmproj` (§12, M1) |
| Optional, unquantized parity | `d1-3B-BF16.gguf` (or `-F16`) | 5,403,160,160 | Not approved and not requested. The only way to check Q8_0-on-Metal against the unquantized model (§5.4). Worth it only if the eval looks marginal |
| Rank 1, runtime | Homebrew llama.cpp **0.7.0** (contains 88dcc46) | — | **Done, owner-approved**: build 11669, `620d2225c`. ggml moved 0.26.0 → 0.27.0, and whisper-cli still starts (relayed) |
| Rank 2 / 2b, pplx weights | `perplexity-ai/pplx-decider-flash-v1.2`: `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors` | 4,972,946,624 / 4,105,669,472 | The weights the native backend loads, and the oracle's input |
| | `readout.safetensors`, `config.json`, `decision_config.json`, `joint_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `LICENSE`, `NOTICE`, `source/*.py` | 1,305,712; 2,898; 5,435; 248; 19,989,492; 1,268; 7,756; 11,544; 132; ~45 KB | Needed by the native backend, by the P1 oracle, and by any conversion |
| Rank 2, oracle venv | torch 2.14.0, torchvision 0.29.0, transformers 5.17.0, safetensors 0.8.0, huggingface-hub 1.31.0 (pplx `requirements.txt`) | — | New Python dependencies in a dedicated `uv` venv. Oracle only, never shipped |
| Not recommended | `Jakevin/pplx-decider-flash-v1.2-GGUF` (Q8_0 5,157,835,072; Q4_K_M 3,066,386,752) | — | Community-made, causal per its own card |
| Rank 3 | `Mapika/decider-4b/model.safetensors` | 8,411,558,400 | Not recommended now |

## 10. Owner decisions

1. **Q1: `degraded` for an external engine.** Should the black-box isolation canary (§4.2) be
   amended into `schema-api.md` as the check that runs for an external engine? The recommendation
   is yes. The alternative is `degraded: true`, which every caller reads as abstained.
2. **Q2: admission.** Should `admitted_families`, entered only by a promotable verdict, replace
   `trained_families` for third-party models (§3.3)?
3. **Q3: eval data.** Are labelled real job postings allowed as eval data under "synthetic only"
   (§8.1)?
4. **Q4: approvals.** The llama.cpp 0.7.0 upgrade and the d1 download are already approved and
   done. Still open:
   - the `qd-runtime` `Answerer` change and the new `qd-systemone` crate (no new crates). The
     parent session suggested a Go adapter over loopback TCP instead. This doc recommends Rust
     over a Unix socket for two reasons:
     - Go would be a second implementation of `wire.rs`, the refusal vocabulary and
       `calibration.rs`, which are this repo's single owners and have drifted three times when
       copied.
     - A loopback port is reachable by every local user, while a `0600` socket is not.

     Either way the wire contract is the same, so this is the owner's call;
   - the four new refusal kinds: `context_not_utf8`, `bins_over_backend_cap`,
     `slot_kind_unsupported`, `context_over_backend_budget`;
   - `registered_route_unsupported`.

   Each refusal kind is a wire-corpus change under `fixtures/wire/` that `qd_wire` reads.
5. **Q5: the LFM licence's US$10M revenue threshold.** Does it fit the owner's intended use?
6. **Q6: pplx route.** Native tessl (rank 2, ~2–3 weeks, owner code) or an upstream llama.cpp
   patch (2b, C++ in another project, plus a one-time Python conversion)? Neither starts before
   rank 1's eval shows whether a third-party model beats Jev on the owner's labels at all.

### Decisions recorded 2026-10-18

The owner's answers, quoted from their answer card as relayed by the parent session
(`local_c00a8a86…`). This lane recorded them and built nothing.

| Item | Owner's answer (verbatim) | What it means here |
| --- | --- | --- |
| Build | "Wait for the eval first" | Rank 1 (the d1 adapter) is **not built** until d1 has been measured against the owner's labels in Vani and beats Jev (§8). Meanwhile the parent session is wiring d1 into Vani as a **shadow backend over loopback, for that measurement only**. That shadow is eval plumbing, to be deleted if the adapter later replaces it. It is not the `qd.sock` adapter, and it does not decide the Rust-vs-Go adapter question in Q4, which stays open until a build is approved |
| Q1 | "Isolation canary (Q1)" | Accepted. When the adapter is built, the black-box isolation canary (§4.2) is the check that runs for an external engine, and `schema-api.md` is amended to say so. Not applied now |
| Q2 | "admitted_families (Q2)" | Accepted. A third-party model admits a family only after a promotable verdict, citing its ledger row (§3.3). Not applied now |
| Q3 | "Yes, real postings (Recommended)" | Labelled real job postings may be **eval** data for third-party deciders, and **never training** data. This amends the 2026-10-06 "synthetic only" ruling (caller-contract §5) **for eval only**. Labels stay in Vani's store, as in §8.1 |
| Q4 | "Four new refusal kinds (Q4)" | Accepted as listed, together with `registered_route_unsupported`. These take effect **when the adapter is built**, not now. No fixture, corpus or `qd_wire` change is made before then |
| Q5 | Not asked separately | The owner approved the d1 download and its use after being told the LFM 1.0 terms (commercial use only below US$10M annual revenue). Recorded as **accepted for personal use**. Any commercial use would need a fresh check against the threshold |
| Q6 | Deferred | No pplx work, native or upstream, until the rank-1 eval reports |
| pplx eval | "Not approved. Go with d1 by default: evaluate d1 through the Vani shadow, and port d1 natively to tessl if it beats Jev." (said directly in this lane's session, later on 2026-10-18) | **d1 by default.** The pplx weights (9.08 GB) and the oracle venv are **not approved**, so pplx has no eval path and is not ported. The sequence is: label postings, evaluate d1 through the Vani `local` shadow (§8.2), and port d1 natively to tessl/qd-metal (§13.2) **only if** the verdict is "promotable" against Jev. If d1 does not beat Jev, nothing is ported, and pplx needs a fresh approval to be considered at all |
| Open design choices | "Choose best and proceed" (said directly in this lane's session, after the "pplx eval" decision) | **Read as:** make the choices the doc left open and carry on with the decided sequence (label → eval → port if promotable). It is **not** read as lifting the "if it beats Jev" condition. Chosen: **L1 (b)**, a Q8_0 GEMV/GEMM kernel in tessl (exact weights, 2.9 GB resident, the house answer). **L5:** port the `lfm2` pre-tokenizer regex from `llama-vocab.cpp`, gated by `/tokenize` parity, which needs no download, the owner having just declined one. The d1 port estimate is **3–4 weeks**. The eval is pre-registered at `campaign/d1-vani-jdtriage-eval-preregistered.json`. If the owner meant "start the port now", that is a one-line correction here. **Confirmed by the owner, 2026-10-18:** "Wait for the verdict. Keep the port gated on d1 beating Jev in the pre-registered eval." |
| Runtime | "use metal and tessl by default" (said directly in this lane's session, later on 2026-10-18) | **The served runtime for any third-party decider is a native port on tessl/qd-metal, on Metal.** llama.cpp moves from "the runtime" to two narrower roles: the **eval path** (the Vani `local` shadow measuring d1 now) and the **parity oracle** a native port is tested against. It is not the served path. CPU is never a serving fallback. This supersedes the llama.cpp adapter as rank 1 (§0, §13). The Rust-vs-Go adapter question in Q4 is moot for the served path. The `Answerer` seam (§3.2), the refusal kinds (Q4), the isolation canary (Q1) and `admitted_families` (Q2) all carry over unchanged to native backends |

## 11. Gaps

Every U that the first draft carried has been closed in §12, or is listed below with why it
cannot be closed without a decision the owner has not taken.

**Still open, each gated on something the owner has not decided:**

| Gap | Why it stays open | What closes it |
| --- | --- | --- |
| Q8_0-on-Metal against the **unquantized** d1 | Needs the 5.4 GB BF16/F16 GGUF, which is not approved (§9) | That download plus one CPU run of the probe set. Metal-vs-CPU on Q8_0 is closed (§12, M2) |
| d1's **accuracy on the owner's task** | Triaged postings exist (§8.2), but none are labelled | The owner labelling 80 postings, then the eval |
| The **ledger row** for latency | A probe is not a ledger row (rule 5) | A `tools/mac_heavy.sh` run on real posting lengths when the eval runs |
| The body of `build_recurrent_attn` (pplx route 2b) | Q6 is deferred, so no pplx work | The P2 "GDN still causal" test, if 2b is ever chosen |
| decider-4b's batching and shared-prefix code | Not recommended (rank 3) | Reading those files, if it is ever promoted |
| LFM2 RoPE form (half-split vs interleaved) and RMSNorm weight form for the d1 port | Upstream `lfm2.cpp` takes `rope_type` from members not in that file. The half-split is inferred from HF `rotate_half` | The P-L parity test (§13.2). A wrong form fails it on the first prompt |
| Which tensor is gate and which is up in d1's SwiGLU | Not stated in the upstream summary read | The GGUF names `ffn_gate` / `ffn_up` (§12 header dump). P-L confirms |
| Whether d1 beats Jev | No labels yet. pplx is out of scope by the owner's decision (§10, "pplx eval") | The owner labelling 80 postings, then `vani eval`'s verdict on the d1 shadow |
| pplx-flash on DI 0.2.1 | Not needed: both models are compared on DI 0.3 (§12, S8) | — |
| Third-party (Jakevin) agreement numbers | Not used for any decision. Upstream's causal route is rejected on primary source (§12, S4) | — |

**Recorded in `gaps.jsonl`:**

- **`GAP-DECIDE-NO-DEVMAP-STORE-2026-10-18`.** `~/Code/devtools/decide` still has no DevMap store,
  and this lane did not build one. Neither the repo (no `.gitignore`, no `.git/info/exclude` entry) nor the
  global excludes file (`~/.gitignore_global`) ignores `.devmap/` or `.devcouncil/`. So a store
  would show up as untracked files in a repo other sessions use. The owner can build it
  with `devmap` from that repo.
  - Its facts here are direct reads: `rg`/`sed` on `lappi/backend.go`, `lappi/client.go`,
    `router.go` and `jev/jev.go`.
  - The **Vani** facts in §7 are now **graph-confirmed** (Vani has a fresh store, §12, S7).
- **`GAP-DECIDE-LAPPI-BACKEND-NAME-HIDES-MODEL-2026-10-18`.** Narrowed to models behind `qd.sock`
  (§7). The `local` shadow has its own name.
- The tessl DevMap index is stale on one file (`BUILD_REPORT.md`), which is in no kernel path used
  here.

## 12. Gap closure, 2026-10-18

Raw probe data and the throwaway scripts (`gen.mjs`, `run.mjs`, `analyze.mjs`, `leak.mjs`) are in
`build/d1-probe-2026-10-18/` (gitignored). Every probe used the already-downloaded
`d1-3B-Q8_0.gguf` (sha256 `2f0942d5…ba77d`) on Homebrew llama.cpp 0.7.0 (build 11669,
`620d2225c`), with 3 synthetic postings and Vani's four `jd_triage` questions. These are probe
figures, not ledger rows.

**Source reads (S).** Upstream code was read at commit `620d2225cc436ab0b001e3200e590e3b286ef9ab`,
and Hugging Face files on `main`, both by a research agent. Large files were partly scanned, and
each place where that matters is named.

- **S1. Key order.** `common_json` is a pimpl wrapper over `nlohmann::ordered_json`
  (`common/json.cpp`: `using nlohmann::ordered_json;` and a `static_assert` on its size).
  `parse_questions` iterates `criteria.items()` in insertion order. `std::sort` by key runs only
  under `choice_sorted`, which is set in the CLEF branch alone.
- **S2. No truncation for d1.** `fill_task` only tokenizes. Truncation exists only in
  `fill_task_laya` and `fill_task_d1omni`. Overflow is `ERROR_TYPE_EXCEED_CONTEXT_SIZE`, which is
  HTTP 400 `exceed_context_size_error` (`server-context.cpp`, `server-common.cpp`).
- **S3. Prefix sharing.** `server_decision_group_tasks` groups tasks in chunks of `n_slots`, sets
  `n_tokens_shared` to their common prefix, and copies only that KV to child slots
  (`mem.seq_cp`). Each question evaluates its own tail in its own sequence.
- **S4. Causal flag.** `cparams.causal_attn` defaults to the model's hparam (true unless the GGUF
  says otherwise). The KV mask uses it in `llm_graph_input_attn_kv` and
  `llm_graph_input_mem_hybrid`. The recurrent path does not reference it.
  - Nothing for `pplx-decider` sets it false: not the converter, not `qwen35.cpp`, and not
    tools/server or common.
  - Unread: about 160k characters of `server-context.cpp`, 84k of `llama-model.cpp` and 52k of
    `llama-graph.cpp`.
- **S5. pplx-flash v1.2.**
  - The readout is `nn.Linear(2560, 255, bias=False)` on post-norm `last_hidden_state` at the
    `<|fim_suffix|>` marker (`source/src/pplx_decider/model.py`).
  - `decision_config.json` has `"attention_mode": "noncausal_full_attention"`, `"pooling":
    "last"` and temperature 1.5539.
  - The `config.json` values are as in §6.2 N4.
- **S6. decider-4b.**
  - `config.json` matches S5's values, with `Qwen3_5ForCausalLM` and `qwen3_5_text`.
  - `decider/model.py` scores `F.linear(h[slot positions], lm_head.weight[letters])`.
  - No non-causal setting appears in the files read.
- **S7. Vani, through DevMap.** The store at `Vani/.devcouncil/codeintel/` is fresh (generation
  268–271). `devmap_explore AnnTriageSuffixShadowBackend` returns its callers
  `pipeline/runner.go::triageAnnotations` and `eval/report.go::BuildReport`.
  `devmap_search slotBackend` returns the `backend=` parser.
- **S8. Decision Index.**
  - `data/methodology.json`: "0.3 keeps the 0.2.1 public panel and rules". GSM8K is rebuilt,
    ForecastBench retired, Arts fixed at 10%, and private tests are added. The full score is
    0.20 public + 0.50 private same-skill + 0.30 new-domain.
  - `data/v03.json`:

    | Model | v | public | same-skill | new-domain |
    | --- | --- | --- | --- | --- |
    | d1-3b | 38.90 | 48.99 | 41.23 | 27.96 |
    | pplx-decider-flash-v1.2 | 54.81 | 60.06 | 56.16 | 43.17 |
    | pplx-decider-v1.2 (27B) | 65.55 | — | — | — |

**Measurements (M), this lane, Metal unless stated.**

- **M1. Unix socket.** `--host <dir>/d1-metal.sock` logs "listening on unix://…" and `/health`
  returns ok. The socket file is created `srwxr-xr-x` (0755). The server warns that "no API key
  is set and CORS allows all origins". It loaded with no `--mmproj` and answered every text
  question.
- **M2. Metal against CPU** (`-ngl 0 --device none`), same 39 requests: across 97 answers, max
  |Δp| = 2.42e-2 and 0 choice flips. Metal at `-np 4` against `-np 8`: max |Δp| = 4.2e-4, 0 flips.
- **M3. Isolation.**

  | Comparison | Metal | CPU |
  | --- | --- | --- |
  | Spread over 3 repeats | 0 (bit-identical) | up to 4.0e-3 for joint requests |
  | Max \|Δp\|, slot alone vs in a joint request of 4, over 24 slot answers | 3.0e-4 (`-np 4`), 4.5e-4 (`-np 8`) | 2.5e-2 |
  | Real other questions vs 3 unrelated ones | 1.4e-4 | 1.9e-3 |
  | Alone vs with the unrelated ones | 1.0e-4 | 6.0e-3 |

  These are noise-sized on Metal. No content leaks between questions.
- **M4. Permutation.** A cyclic shift of the options (the second pass) gave 0 argmax
  disagreements in 24 slot answers, and max |Δp| = 0.364 on one option (p3 work_mode). Reversing
  the key order of the sponsorship question moved it by up to 0.035. The reply lists keys in
  request order.
- **M5. Descriptions against bare option strings** (`joint4`). 3 of 12 choices differ, on both
  Metal and CPU:

  | Slot | With descriptions | Bare option strings | Truth |
  | --- | --- | --- | --- |
  | p3 sponsorship | not_stated 0.66 | citizenship_or_clearance_required 0.75 | bare |
  | p3 work_mode | not_stated 0.37 | onsite 0.98 | bare |
  | p2 seniority | staff_plus 0.81 | senior 0.31 | descriptions |

  Confidence also moves elsewhere: p1 role_fit is 0.80 with descriptions and 0.65 bare.
- **M6. Context.** A 40,072-token state returns HTTP 400 `exceed_context_size_error` with
  `n_ctx` 8192, the per-slot figure. A 24,576-byte plain-English posting is 4,158 tokens by
  `/tokenize`.
- **M7. Latency.** The table in §5.5. The `-np 8` server also answered the 2-question pair at
  110–138 ms over 6 runs.
- **M8. Stale socket** (V, two sessions).
  - The parent session observed it first. A socket file survives a clean SIGTERM (exit 0), and the
    next start on the same path fails "couldn't bind HTTP server socket".
  - This lane's three probe servers, each stopped with SIGTERM, left `d1-cpu.sock`,
    `d1-metal.sock` and `d1-np8.sock`. `curl --unix-socket` to each got exit 7 (connection
    refused), and only then were they removed.
- **M9. Parent session, relayed** (detail in Vani `docs/FOLLOWUPS.md`, "Verified" and "Known
  limits"):
  - CPU (`-ngl 0`) and Metal gave the same choice on 12 of 12 slots, with max |Δp| about 0.03.
    That agrees with M2.
  - CPU took about 4 s per posting and Metal about 0.3 s.
  - A cold start through Vani's `internal/localserver` (TCP loopback) took about 2 s.
- **Hygiene.** This lane started three servers (PIDs 78320, 1736 and 38574), stopped each by PID,
  and removed their three leftover sockets after confirming nothing listened. No server it
  started is still running.

## 13. Metal and tessl by default: the native plan

### 13.1 Ranking under the directive

**Decided 2026-10-18 (§10): d1 by default.** The owner did not approve pplx's eval downloads.

1. Label postings.
2. Evaluate d1 through the Vani shadow.
3. Port d1 natively (§13.2) only if it beats Jev.

The table below is kept for the reasoning; the pplx rows are inactive until a fresh approval.

| Rank | What | Work (estimate, unverified) | Why |
| --- | --- | --- | --- |
| **Eval first** | Measure d1 and pplx-flash against Jev on the owner's labelled postings **before** porting either | d1: the Vani `local` llama.cpp shadow already being wired, which gives a verdict through Vani's own `Promotion`. pplx: see the next paragraph | The owner said "wait for the eval first". Porting is weeks either way, so build only the model that wins. The DI 0.3 gap (pplx 54.81 against d1 38.90) favours pplx, but DI is not the owner's task |

**How pplx would get a verdict.** Vani's verdict comes from a `decide.Backend` shadow, and a Python
oracle is not one. Two ways, both throwaway and both needing the owner's yes:

1. **Offline replay (recommended).**
   - A throwaway script runs pplx's own `source/src/pplx_decider` on CPU fp32 over the 30
     held-out labelled postings: 4 slots × 2 passes = 240 forwards of a 4B model.
   - The time is **hours** on this Mac, unmeasured.
   - It then computes Vani's `Cell`/`Promotion` arithmetic (precision = correct/answered,
     coverage = answered/labelled, `MinPromotionLabels = 30`) against the same labels and Jev's
     recorded answers.
   - No service is involved; it is analysis that never ships (language policy, item 3).
2. **A Python `/v1/systemone` shim behind Vani's `local` shadow.** This is a Python *service*,
   even if eval-only. It is named so the owner can refuse it.

Both need the pplx weights (9.08 GB) and an oracle venv (§9). **If the owner says no to those,
"eval first" becomes "d1 by default"**: d1 has an eval path today, and pplx would have none until
a native port exists.
| 1 if d1 wins | **d1-3B native on tessl/qd-metal** (§13.2) | ~2.5–3.5 weeks | **d1 is causal** (§13.2). So it fits Lappi's existing answering procedure unchanged: prefill the state once, snapshot, and answer each slot and its permuted pass as read-only continuations. The white-box slot-isolation hash runs as it does for Lappi's own model. The parity oracle (llama.cpp with the same Q8_0 file) is already on disk |
| 1 if pplx wins | **pplx-flash v1.2 native** (§6.2) | ~2–3 weeks | Its non-causal attention forces the snapshot-free backend (N2), with one forward per question and per pass, and the black-box canary (Q1). Its oracle needs the downloads in §9 |
| 3 | decider-4b native (§6.3) | ~1.5 weeks | Cheapest native model: causal Qwen3.5 with tied letter rows, so it is today's qd-metal scoring. Not recommended, for the licence and calibration reasons in §0 |
| — | llama.cpp behind `qd.sock` (`qd-systemone`, §3.2, §5) | — | **Not the served path.** It stays in the plan as the eval shadow and the parity oracle only |

### 13.2 d1-3B on tessl/qd-metal

**The architecture (V, from the GGUF header read locally, plus upstream `src/models/lfm2.cpp`
`graph<>` and HF `config.json`):**

- 30 pre-norm blocks with hidden size 2048: **22 short-conv blocks and 8 GQA attention blocks**.
  The attention blocks are at 0-indexed layers 2, 5, 9, 13, 17, 21, 24 and 27 (the GGUF's
  `head_count_kv`).
- Attention: 32 query heads and 8 KV heads at head dim 64. Q/K RMSNorm runs before RoPE, with
  theta 1e6 and scale 1/√64.
- Short-conv block:
  - `in_proj` from 2048 to 6144, split into B, C and x;
  - `bx = B·x`, then a depthwise causal conv with kernel 3 and no bias over `bx`, with **no
    activation**;
  - `y = C·conv`, then `out_proj`.
  - The carried state is the last 2 columns per layer.
- Feed-forward is SwiGLU with intermediate size 10752.
- The final RMSNorm is the GGUF's `token_embd_norm`, i.e. `output_norm` in the graph (I, from the
  tensor-name mapping).
- The output head is tied to the 128,000-row embedding. ε is 1e-5.

**Causal (V, three ways):**
- The GGUF has no `*.attention.causal` and no `decision.block_count` key. The local header dump
  matched none of `causal|decision.block_count`. So upstream runs the ordinary causal `graph<>`.
  The non-causal `graph_decision` is only for d1-omni-style models that carry a decision head.
- HF `runner.py` says "One-pass typed decisions on a causal LFM backbone", and reads
  `.logits[0, -1]`.
- HF `hybrid.py` uses `is_causal=True` and a left-padded conv.

**The answer read-out (V, `runner.py` and upstream `server-decision.cpp`):**
- Take the last prompt token's logits.
- Each option's score is the **max** over its label-token group: the option code's token, plus
  the single-token form of `" " + code`.
- Softmax over the options follows. llama.cpp's version is equivalent, because a log-softmax over
  the whole vocabulary shifts every group by the same constant.
- No temperature key is set (§5.3).
- d1 runs **one** pass per question. A reversed option order changes the probabilities (§12, M4:
  up to 0.035), so llama.cpp does not average variants for d1.

**So the native port needs only the label rows, not the whole 128k-row head:** gather the
embedding rows of each option's group, score them, and take the max per group. That is Lappi's
letter read-out with a group-max added.

**What tessl already has (V):**

| Need | tessl |
| --- | --- |
| Causal prefill attention at h64 | `nn::flash_attn_rows`, entry `flash_attn_rows_h64_r8_g8` |
| Q/K RMSNorm plus RoPE over the whole head at any head dim, plain weight | `qwen35::attn_qk_norm_rope_packed` (`weight_bias` 0, `rotary_dim = head_dim`, half-split rotation). That LFM2 uses the half-split (NEOX) form is I, from HF's `rotate_half`; P-L2 pins it |
| RMSNorm, SwiGLU, GEMM | `nn::rms_norm_*`, `nn::mlp_silu`, `gemm` / `gemm_epilogue` |
| Snapshot-read conv state | `qwen35::conv1d_silu` with `StateIn::Snapshot`. It fuses SiLU, so it is not directly usable |
| Label-row scoring | `qwen35_score_rows` (generic over head dtype) |

**The work:**

| # | Work | Where | Estimate |
| --- | --- | --- | --- |
| L1 | **Weights.** Read the Q8_0 GGUF already on disk: blocks of 32 int8 values with one f16 scale. That is a small reader with **no new crate** and **no new download**. A Q8_0 value carries up to about 18 significant bits, and bf16 holds 8, so how it is dequantized decides parity. **Open choice, recommendation (b):** | `qd-metal` (a GGUF reader beside `safetensors.rs`), and tessl for (b) | (a) 2 days; (b) 2 days + 4–6 days kernel; (c) 2 days |
| | **(a) To f32 at load.** Exact, so the native port runs the same weights llama.cpp runs, and P-L's 2.4e-2 starting tolerance applies. About 11 GB for the text tower: within memguard's 32 GiB per-process cap, but 4× the file. It uses tessl's `GemmOperands::ExactF32` path, which is slower | | |
| | **(b) A Q8_0 GEMV/GEMM kernel in tessl,** the house answer: a focused kernel over a heavy dependency, with `gemm_q4_mlx` as the pattern. Exact weights, 2.9 GB resident, and the fastest of the three. Ships with an f64 reference test and a committed benchmark | | |
| | **(c) To bf16 at load.** 5.4 GB, but every weight is re-rounded, so P-L must **widen** its tolerance by a weight-rounding term that has not been measured. Not recommended as the parity baseline | | |
| | The BF16 `model.safetensors` (6,247,065,504 bytes, including the vision tower to skip) would be a fourth route through the existing loader. It is a different (unquantized) checkpoint, and a download | | |
| L2 | **Short-conv kernel.** `lfm2_short_conv`: B·x, then the depthwise causal conv (kernel 3) with `StateIn` (fresh, carried or snapshot) and `state_out`, then ·C, fused. With an f64 CPU reference test in the style of `tests/common/qwen35.rs::conv1d_silu_f64`. The alternative is an activation flag on `conv1d_silu` plus two elementwise multiplies, but a fused kernel is the smaller surface | **tessl** (canonical, rule 6) | 2–3 days |
| L3 | **Prefix attention at h64.** `qwen35::attn_prefix_rows` is instantiated only at `PREFIX_ATTN_HEAD_DIM = 256` (`tessl/src/qwen35.rs:2651`). An h64 instantiation is needed for continuations over the shared prefix KV, pinned bit-for-bit to `flash_attn_rows_h64` the way `prefix_attention_is_flash_attn_rows_instantiation` pins h256 | **tessl** | 1–2 days |
| L4 | **The LFM2 layer loop.** A second model kind beside Qwen3.5's `Model::run`, with `PrefixState` = 22 conv states (2 × 2048 each) plus 8 KV caches, snapshot-read continuations, and N questions as N batch rows. It reuses `MetalBackend`'s structure and the snapshot digest, so the **white-box slot-isolation check runs**. It implements `DecisionBackend` (see §13.3 for how that squares with §3.1) | `qd-metal` | 4–5 days |
| L5 | **Tokenizer.** The GGUF carries the vocabulary and merges (128,000 tokens and 293,320 merges), but **not the pre-tokenizer**. `tokenizer.ggml.pre = lfm2` names a split regex that lives in llama.cpp's `llama-vocab.cpp`, not in the file. Two options: port that regex into a `tokenizers` pre-tokenizer built from the GGUF (no download, some risk), or download HF `tokenizer.json` (17,905,750 bytes, which carries the pre-tokenizer; that needs an approval, but the cost is small). Either way, token ids must equal llama.cpp's `/tokenize` on the probe set, including CJK and code | `qd-metal` | 1 day with `tokenizer.json`; 2 days porting the regex |
| L6 | **Prompt render.** Port the 1,365-character `systemone` Jinja template (§12) to Rust, byte-identical to llama.cpp's own rendering. llama.cpp's `/apply-template` route (present in the 0.7.0 route table) is the golden source. The shared prefix runs through `\n\n\nQUESTION:\n`, and each question's suffix is its instructions, options and reply line | new module, beside `qd-runtime/src/render.rs` | 2 days |
| L7 | **Read-out.** Final norm, the label-group rows from the tied embedding, max per group, then Lappi's calibration (§5.3) | `qd-metal` | 1 day |
| P-L | **Parity against the llama.cpp oracle on the same Q8_0 file.** Token ids equal `/tokenize`; the rendered prompt equals `/apply-template`; per-option probabilities on the probe set match `/v1/systemone` within a stated tolerance, with 0 choice flips. The tolerance starts at Metal-vs-CPU's measured 2.4e-2 (§12, M2), because both quantization and kernels differ. A conv state digest proves slot isolation. A per-layer comparison against llama.cpp is not available through the server, so layer-level parity uses an f64 CPU reference of the LFM2 block on random weights, the same pattern as tessl's Qwen3.5 tests | tessl + qd-metal tests | 3 days |

**About 2.5–3.5 weeks** in total with L1 (a), or **3–4 weeks** with the recommended L1 (b) Q8_0
kernel. Unverified.

- It is **causal**, so it needs neither the black-box canary nor the snapshot-free rework that
  pplx needs.
- It needs **no new crate**. The Q8_0 weights are on disk. The only download it may want is
  `tokenizer.json` (17.9 MB, L5), and that is avoidable by porting the `lfm2` pre-tokenizer regex.
- A HF/PyTorch oracle in BF16 (`runner.py`) would be a second, unquantized reference. It needs the
  6.25 GB download and a venv, so it stays optional (§9).

### 13.3 `DecisionBackend` and the renderer: squaring §3.1 with L4

§3.1 says a third-party model does not plug in at `DecisionBackend`. That was written for an
engine whose state Lappi cannot see (llama.cpp, or non-causal pplx). For a **causal native port**
it is half true:

- The trait itself is renderer-agnostic. It takes a prefix string and slot suffixes, and returns
  rows.
- What is format-2-specific is the **renderer** that `answer.rs::answer_generic` calls.

So native d1 is a d1 `DecisionBackend` (prefill, snapshot and read-only decodes with host-visible
state), driven by the existing answering loop with a **d1 renderer**. Concretely:

- `answer.rs` takes a `Renderer` (Lappi format 2 today, `systemone`-for-d1 next). The renderer
  owns the prefix and suffix strings and the rows per slot. For d1 that is one row per option:
  its group-max, plus the reserved abstain row at the floor.
- The loop (prefill → snapshot → two decodes → derangement agreement → calibrate) stays **one
  copy**. The alternative, a second loop inside a d1 `Answerer`, is the near-copy drift this repo
  has paid for three times.
- `Answerer` (§3.2) remains the seam only for backends with no host-visible state, which today
  means pplx-flash's snapshot-free path.

### 13.4 What carries over unchanged

These are designed engine-agnostic and apply to a native backend as written:

- the `Answerer` seam (§3.2);
- `admitted_families` (Q2);
- the four refusal kinds plus `registered_route_unsupported` (Q4);
- descriptions from the pinned fixture (§4.1);
- the abstain-row floor and verdict-file format (§5.3);
- the 80-posting eval (§8.2).

Two things change:
- d1's native backend runs Lappi's **white-box** isolation hash. Q1's black-box canary is needed
  only for pplx (non-causal) and for any engine whose state the host cannot see.
- `context_over_backend_budget` becomes a check against the port's own token cap. There is no
  server to return a 400.
