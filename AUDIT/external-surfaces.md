# External-Surface Verification — qwen-decision Build Plan

**Audit date:** 2026-09-19
**Lane:** external-surface verification only. No code written, no plan files changed.
**Evidence rule:** every row is VERIFIED (primary source URL + observed fact), REFUTED (with evidence), or
UNVERIFIED (could not reach a primary source; the "what I tried" section says exactly what).
Recollection was never used as evidence. Where a claim is arithmetic over verified values, it is labelled
**VERIFIED (arithmetic)** and the arithmetic is shown.

Tools used: HuggingFace public API + `raw/main` file endpoints, GitHub REST API via authenticated `gh`
(account `bharathvbcr`), PyPI JSON API, direct page fetches. Network access worked throughout; nothing in
this report is UNVERIFIED because of a network failure.

---

## 1. Qwen3.5-2B-Base — existence and config

Primary source: `https://huggingface.co/Qwen/Qwen3.5-2B-Base/raw/main/config.json` (fetched 2026-09-19).
The `Qwen/Qwen3.5-2B` and `Qwen/Qwen3.5-2B-Base` configs were fetched separately and are **byte-identical**.

| Claim | Status | Evidence (URL + observed fact) | Impact if wrong |
|---|---|---|---|
| `Qwen/Qwen3.5-2B-Base` exists | **VERIFIED** | `https://huggingface.co/api/models/Qwen/Qwen3.5-2B-Base` returns `id: Qwen/Qwen3.5-2B-Base`, `gated: False`, `private: False`, `downloads: 555637`. Sibling official checkpoints confirmed via `api/models?search=Qwen3.5`: `Qwen3.5-0.8B`, `-2B`, `-4B`, `-9B`, `-27B`, `-35B-A3B`, `-122B-A10B`, `-397B-A17B-FP8`, plus `-2B-Base`, `-4B-Base`, `-9B-Base`, `-35B-A3B-Base`. | Whole plan |
| Apache-2.0 licence | **VERIFIED** | Same API: `cardData.license: apache-2.0`, tag `license:apache-2.0`, `license_link: https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/main/LICENSE`. Not gated. | Publishability / redistribution |
| 262K native context | **VERIFIED** | `config.json` → `text_config.max_position_embeddings = 262144`. Model card states "262,144 natively and extensible up to 1,010,000 tokens." | Long-context claims |
| 24 total layers = 18 Gated DeltaNet + 6 full attention | **VERIFIED** | `num_hidden_layers = 24`. `layer_types` array literally enumerates 24 entries; `full_attention` occurs at indices 3, 7, 11, 15, 19, 23 → **6 full, 18 linear**. Corroborated by `full_attention_interval = 4`. | Layer-surgery / freezing design |
| Vocab size 248,320 | **VERIFIED** | `text_config.vocab_size = 248320`. | Tokenizer / head sizing |
| Embedding ≈ 509M params | **VERIFIED (arithmetic)** | 248320 × 2048 (`hidden_size`) = **508,559,360**. `tie_word_embeddings = true`, and `model.safetensors.index.json` contains **no `lm_head`** key — embedding and output head are one tensor, counted once. | Trainable-param budget |
| GDN state = 16 heads × 128×128 | **VERIFIED** | `linear_num_key_heads = 16`, `linear_key_head_dim = 128`, `linear_value_head_dim = 128`. | State-size / memory math |
| **`linear_num_key_heads` FEWER than `linear_num_value_heads`** | **REFUTED** | Both are **16**. `"linear_num_key_heads": 16, "linear_num_value_heads": 16`. They are **equal**; there is no key/value head asymmetry in the GDN blocks. | **See PLAN-BREAKING #1** |
| Partial rotary, 64 of 256 head dim | **VERIFIED (arithmetic)** | `head_dim = 256`, `rope_parameters.partial_rotary_factor = 0.25` → 0.25 × 256 = **64**. Also `rope_theta = 10000000`, `rope_type: default`, `mrope_interleaved: true`, `mrope_section: [11, 11, 10]`. | RoPE / long-context scaling code |
| QK-norm without V-norm | **VERIFIED** | `src/transformers/models/qwen3_5/modeling_qwen3_5.py` (main) lines 773-774 define `self.q_norm` and `self.k_norm` as `Qwen3_5RMSNorm(self.head_dim, ...)`; applied at lines 792-793. Grep for `v_norm` over the whole file returns **zero hits**. | Norm-placement assumptions |
| Depthwise causal conv, kernel size 4, over concatenated Q\|\|K\|\|V | **VERIFIED** | `linear_conv_kernel_dim = 4`. `modeling_qwen3_5.py` ~line 520: `self.conv_dim = self.key_dim * 2 + self.value_dim`; `nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, groups=conv_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size - 1, bias=False)` → **depthwise** (`groups == channels`) and **causal** (left pad k-1). Single fused projection `self.in_proj_qkv = nn.Linear(hidden_size, key_dim*2 + value_dim)`; conv is applied to `mixed_qkv` and only afterwards split: `torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)` → `query, key, value`. | Conv-state / packing handling |

### Additional config facts the plan does not state (material)

| Fact | Evidence |
|---|---|
| **Qwen3.5-2B-Base is multimodal.** Top-level `architectures: ["Qwen3_5ForConditionalGeneration"]`, a full `vision_config` (depth 24, hidden 1024, patch 16), and `image_token_id`/`video_token_id`/`vision_start_token_id`. HF `pipeline_tag` is **`image-text-to-text`**, not `text-generation`. | `config.json`; `api/models/Qwen/Qwen3.5-2B-Base` |
| **The "Base" checkpoint physically contains vision and MTP weights.** `model.safetensors.index.json`: **632 tensors total**, of which **297** are `model.visual.*` and **15** are `mtp.*` (e.g. `mtp.layers.0.mlp.down_proj.weight`, `mtp.fc.weight`). `metadata.total_size = 4,548,144,832` bytes. | `https://huggingface.co/Qwen/Qwen3.5-2B-Base/raw/main/model.safetensors.index.json` |
| Text stack: `hidden_size 2048`, `intermediate_size 6144`, `num_attention_heads 8`, `num_key_value_heads 2`, `attn_output_gate: true`, `mamba_ssm_dtype: "float32"`, `rms_norm_eps 1e-06`, `mlp_only_layers: []`. Embedding key is `model.language_model.embed_tokens.weight`. | `config.json`, index json |
| At bf16, `total_size` 4.548 GB ⇒ ≈ **2.27B total params including the vision tower and MTP block**. The 509M embedding is therefore ≈ 22% of the shipped checkpoint, not of a 2.0B text-only model. | arithmetic over verified `total_size` |

---

## 2. `transformers` support for Qwen3.5

| Claim | Status | Evidence | Impact if wrong |
|---|---|---|---|
| A `Qwen3_5` model class exists | **VERIFIED** | `gh api repos/huggingface/transformers/contents/src/transformers/models/qwen3_5` lists `__init__.py`, `configuration_qwen3_5.py`, `modeling_qwen3_5.py`, `modular_qwen3_5.py`, `tokenization_qwen3_5.py`. |  |
| `Qwen3_5ForConditionalGeneration` exists | **VERIFIED** | `modeling_qwen3_5.py` line 1756: `class Qwen3_5ForConditionalGeneration(Qwen3_5PreTrainedModel, GenerationMixin)`. |  |
| **"Only on main, needs a pinned SHA"** | **REFUTED** | Qwen3.5 support merged **2026-02-09**, PR [#43830](https://github.com/huggingface/transformers/pull/43830) "Adding Support for Qwen3.5", merge commit `fc9137225880a9d03f130634c20f9dbe36a7b8bf`. The model dir is **present at tag `v5.2.0`** (published 2026-02-16) and absent at `v5.1.0` (404). It is present at every later tag checked (`v5.10.1`, `v5.11.0`, `v5.12.0`, `v5.13.0`, `v5.15.0`, `v5.16.0`, `v5.17.0`). Current stable release is **v5.17.0, published 2026-09-09**. | **See PLAN-BREAKING #4** |
| A concrete main SHA to pin | **VERIFIED (available, but see above)** | `main` HEAD at audit time: **`c587bc884db2c2e31fc2b8102314656b17aa07b1`** (2026-09-18T20:31:36Z, "Fix AXK2 integration test…"). Most recent commit touching `modeling_qwen3_5.py`: `59eed1a6ba3566d5d4e8bec23bc1a21bfbda3e84` (2026-09-17, "Add auto_docstring task model overrides (#47815)"). **Pin `v5.17.0` instead** unless something on main after 2026-09-09 is required. |  |
| `Qwen3_5ForConditionalGeneration` has a vision tower | **VERIFIED** | `modeling_qwen3_5.py` line 1319: `self.visual = AutoModel.from_config(config.vision_config)` inside `Qwen3_5Model`; `get_image_features` / `get_video_features` at 1492-1497. `Qwen3_5ForConditionalGeneration` (1756) builds on it. |  |
| `Qwen3_5ForConditionalGeneration` has an MTP head | **REFUTED (in transformers)** | There is **no MTP module in the transformers implementation**. The only `mtp` references in the entire 1800-line file are `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` (line 925) and `[r"^mtp.*", r"^model.visual.*"]` (line 1670). transformers **discards** the checkpoint's MTP weights on load; it never constructs an MTP head. (The weights do exist in the checkpoint — see §1.) | **See PLAN-BREAKING #4** |
| **Step 1 "drop the vision tower and the MTP head" is needed** | **REFUTED** | `class Qwen3_5ForCausalLM` (line 1664) already declares `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`, has `config: Qwen3_5TextConfig`, and builds only `Qwen3_5TextModel` + `lm_head`. Loading `Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-2B-Base")` drops **both** with no surgery. | **See PLAN-BREAKING #4** |

---

## 3. transformers issue #48718 — fla fallback warning (the S1 gate)

Issue is real: `https://github.com/huggingface/transformers/issues/48718` — **OPEN**, created 2026-09-11 by
`TarzanZhao`, 2 comments, title: *"Qwen3.5-9B fine-tuning with run_clm.py on 8x B200 ran the gated DeltaNet
layers on the fp32 torch chunk loop because flash-linear-attention was not installed"*.

| Claim | Status | Evidence | Impact if wrong |
|---|---|---|---|
| Issue exists, fallback is an fp32 torch chunk loop | **VERIFIED** | Issue body: *"A profile showed the 24 gated DeltaNet layers on the fp32 torch chunk loop in `modeling_qwen3_5.py`, forward and backward, for about half the step, because `flash-linear-attention` was not installed."* Confirmed in source: `torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule` both do `.to(torch.float32, memory_format=torch.contiguous_format)`; config sets `mamba_ssm_dtype: "float32"`. |  |
| **"~43% slower"** | **REFUTED (direction inverted)** | Issue body measures **1.22 s/step without fla → 0.70 s/step with fla**. 43% is the *saving*: (1.22−0.70)/1.22 = **42.6% reduction**. The **fallback is 1.22/0.70 = 1.74× → 74% slower**, not 43%. Worse, the transformers source comment (`hub_kernels.py` lines 974-977) states: *"for `chunk_gated_delta_rule` the gap is more than an order of magnitude on an H100."* The 74% is a whole-step figure where GDN is ~half the step; the kernel-level gap is >10×. The same issue reports a further drop to **0.49 s/step** once fused norms + `causal_conv1d` + Trainer trims land. | **See PLAN-BREAKING #2** |
| "Silent except one startup line" | **REFUTED** | See below — it is not at startup, and it is not reliably emitted at all. | **See PLAN-BREAKING #2** |

### The exact warning text (this is what the S1 gate must match)

Source of truth is **not** the issue — it is `src/transformers/integrations/hub_kernels.py` on `main`, inside
`use_kernel_func_from_hub_with_fallback`, lines 973-983:

```python
if not is_new_implementation and not is_torchdynamo_compiling():
    # These torch paths are readable references, not fast kernels, so their runtimes are
    # significantly slower: for `chunk_gated_delta_rule` the gap is more than an order of
    # magnitude on an H100. Warn the user when they end up on one. The logger is untraceable,
    # hence the guard.
    distribution = _PACKAGE_TO_DISTRIBUTION.get(package, package)
    logger.warning_once(
        f"`{func_name}` is falling back to its reference PyTorch implementation because "
        f"`{distribution}` is not installed. This is correct but much slower; install "
        f"`{distribution}` for the optimized kernel."
    )
```

`_PACKAGE_TO_DISTRIBUTION = {"fla": "flash-linear-attention"}` (line 80). `modeling_qwen3_5.py` registers four
wrapped functions:

- line 300 `@use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")`
- line 437 `@use_kernel_func_from_hub_with_fallback("fused_recurrent_gated_delta_rule", "fla")`
- line 269 `@use_kernel_func_from_hub_with_fallback("causal_conv1d_fn", "causal_conv1d")`
- line 249 `@use_kernel_func_from_hub_with_fallback("causal_conv1d_update", "causal_conv1d")`

So the **rendered** strings the plan would have to match are:

```
`chunk_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed. This is correct but much slower; install `flash-linear-attention` for the optimized kernel.
```
```
`causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed. This is correct but much slower; install `causal_conv1d` for the optimized kernel.
```

Note the causal-conv1d case renders the **import name** `causal_conv1d` (underscore), not the PyPI
distribution name `causal-conv1d` (hyphen), because `causal_conv1d` is absent from `_PACKAGE_TO_DISTRIBUTION`.

### Why the S1 gate as designed is broken

1. **It is not a startup line.** The warning fires inside `wrapped()`, i.e. on the **first call** of the kernel
   function — the first forward pass — not at import. The issue reporter's phrase "a `warning_once` at import"
   is the reporter's description and is **wrong about the mechanism**.
2. **It is suppressed entirely under `torch.compile`.** The guard is
   `not is_new_implementation and not is_torchdynamo_compiling()`. Compile the model and the warning **never
   fires** while the slow path runs at full cost. A log-grep gate then passes on a run that is >10× slow in
   the GDN kernels. This is a **fail-open** gate.
3. **`warning_once` fires once per process.** Rank-0-only log filtering under FSDP/DDP, or any
   `TRANSFORMERS_VERBOSITY` above `warning`, removes it.
4. **Absence of the warning does not prove the fast path.** Points 2-3 make the negative uninformative.

**Recommended replacement gate (positive verification, not log-grep):** assert at runtime that
`importlib.import_module("fla")` succeeds **and** that the resolved implementation is not the torch reference —
`transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule` is decorated, so check the bound
implementation identity, or assert `transformers.utils.import_utils.is_flash_linear_attention_available()` is
`True` on the training host. If a log grep is kept as a secondary signal, anchor on the stable substring
`falling back to its reference PyTorch implementation` (not on `flash-linear-attention`, which also appears in
install docs, and not on `much slower` alone).

---

## 4. transformers issue #46093 — sequence packing on hybrid models

| Claim | Status | Evidence | Impact if wrong |
|---|---|---|---|
| Issue exists | **VERIFIED** | `https://github.com/huggingface/transformers/issues/46093`, opened 2026-05-20 by `liangxuZhang`: *"[Inquiry] What is the current status and roadmap for supporting packed sequences (packing) on Qwen3.5/hybrid linear attention models?"* |  |
| **"Sequence packing unsupported for hybrid models"** | **REFUTED** | Issue is **CLOSED**, `state_reason: completed`, closed **2026-05-26T11:02:44Z**, 8 comments. PR **[#45034](https://github.com/huggingface/transformers/pull/45034) "Pass packed boundary metadata to Qwen3.5 linear-attention fast kernels from data collator"** was **merged 2026-05-13** (merge commit `695fc7e7656dea48bac2787806942b76f0432899`) — it plumbs `cu_seq_lens_q` and `seq_idx` into the Qwen3.5 GDN module. Maintainer `@vasqu`: *"The easiest way is to properly prepare your data for that little extra effort."* Reporter's closing comment: *"I will adjust our own data pipeline to explicitly prepare and inject cu_seq_lens_q and seq_idx into the batch to ensure perfect boundary isolation for Qwen3.5 GDN."* | **See PLAN-BREAKING #3** |
| "No packing → 10-20% padding waste" line item | **REFUTED / STALE** | Packing is available. The line item should be deleted and replaced with the correct-usage requirement below. | **See PLAN-BREAKING #3** |

### The real hazard the plan should carry instead

Packing works but **only if boundary metadata is passed explicitly**. From the issue thread:

- Passing **`position_ids` alone** (the TRL / veRL convention, `SFTTrainer` padding-free mode) **does not
  trigger** varlen handling in the GDN path. The reporter: *"The code runs smoothly, but the FLA layer and
  causal convolutions will completely fail to isolate sample boundaries. This leads to severe cross-sample
  information leakage… without any explicit stack trace."*
- `DataCollatorWithFlattening` defaults `return_flash_attn_kwargs=False` and `return_seq_idx=False`, so the
  native collator **also** silently skips the varlen path unless both are toggled on.
- Maintainer `@vasqu` confirmed there will be **no `position_ids`-only path** for GDN (torch.compile
  data-dependent-branch problems) and that a fail-fast check is hard for the same reason: *"Warning / erroring
  would introduce a dynamic data dependent branch so it is hard to do."*

So the correct plan item is: **enable packing, and add an explicit assertion that `cu_seq_lens_q` and `seq_idx`
are present in the batch dict**, because a misconfiguration corrupts training silently with no error.

---

## 5. MLX PR #4020 — Metal gated-delta-net kernels

| Claim | Status | Evidence | Impact if wrong |
|---|---|---|---|
| PR exists, implements Metal GDN kernels | **VERIFIED** | `gh api repos/ml-explore/mlx/pulls/4020` → title **"Adding metal kernels for the gated delta nets."**, author `tpegolotti`, base `main`. |  |
| **"Still open"** | **REFUTED** | `state: closed`, **`merged: true`**, **`merged_at: 2026-09-15T08:51:28Z`**, merge commit `8f76a0aa2bbf9c29698337078db333c9bea1c1bf`. Merged **four days before this audit**. | **See PLAN-BREAKING #5** |
| "Forward only" | **VERIFIED** | PR description: forward pass only, backward listed as future work. Three kernel variants (sequential recurrence; simdgroup 8×8 chunk-parallel; NAX 16×16 tiles). Validated against Qwen3.5 9B/27B/35B with matching perplexity. Benchmarks on M1 Max / M5 Max: *"1.3–1.45× speedup for simdgroup and 1.6–2.2× for NAX implementations at longer sequence lengths"*. |  |
| **"Nothing in the open currently ships this"** (the publishability argument) | **REFUTED** | Merged to `mlx` `main` on 2026-09-15 and public. **Nuance worth keeping:** the latest tagged MLX release is **v0.32.2, published 2026-08-25** — *before* the merge — so no released MLX wheel carries these kernels yet, and the merged work is **forward-only** (no backward, so no training). A narrowed novelty claim ("no released package ships a *trainable* Metal GDN path") may survive; the claim as written does not. | **See PLAN-BREAKING #5** |

---

## 6. flash-linear-attention and causal-conv1d — install story and Apple MPS

| Claim | Status | Evidence | Impact if wrong |
|---|---|---|---|
| fla is pure-Triton | **VERIFIED** | `https://pypi.org/pypi/fla-core/json`: v0.5.2 (2026-07-27). The `[cuda]` extra requires `torch>=2.7.0, triton>=3.3`; `[cpu]` likewise requires `triton>=3.3`. Kernels are Triton JIT, no compiled extension. |  |
| **"fla has no wheel"** | **REFUTED (but harmless)** | `https://pypi.org/pypi/flash-linear-attention/json`: v0.5.2 (2026-07-27) ships **`flash_linear_attention-0.5.2-py3-none-any.whl`** plus an sdist; `fla-core` ships `fla_core-0.5.2-py3-none-any.whl` (819,225 bytes) plus sdist. A **pure-python wheel exists** — `pip install` needs no compiler. The substance the plan meant (no precompiled binary kernels; Triton compiles at first call) is correct. | Install-time budgeting |
| fla install shape | **VERIFIED** | `flash-linear-attention==0.5.2` requires `fla-core==0.5.2` and `transformers>=4.45.0`. `fla-core` bare install pulls only `einops`; **a backend extra is required** to get torch/triton. Extras offered: **`[cuda]`, `[rocm]`, `[xpu]`, `[npu]`, `[cpu]`**. `python_requires >=3.10`. |  |
| **causal-conv1d needs a source build** | **VERIFIED** | `https://pypi.org/pypi/causal-conv1d/json`: latest **v1.7.0 (2026-08-20)**, distribution files = **`causal_conv1d-1.7.0.tar.gz` (sdist) ONLY — zero wheels**. `requires_dist: torch, packaging, ninja`; `requires_python >=3.9`. Description: *"Causal depthwise conv1d in CUDA with a PyTorch interface. Support fp32, fp16, bf16. Kernel size 2, 3, 4."* Requires nvcc + a CUDA toolchain at install time. | Cluster bring-up time |
| **Neither supports Apple MPS** | **VERIFIED** | Two independent signals. (a) `fla-core` declares backends **cuda / rocm / xpu / npu / cpu only**; the project states implementations are *"platform-agnostic and verified on NVIDIA, AMD, and Intel hardware"*; **no MPS/Metal/Apple Silicon reference anywhere** in the PyPI metadata. (b) transformers' own gates, `src/transformers/utils/import_utils.py`: `is_flash_linear_attention_available()` returns `(is_torch_cuda_available() or is_torch_xpu_available() or is_torch_mlu_available()) and is_available and fla_version >= 0.2.2` — **MPS is not in the disjunction**; and `is_causal_conv1d_available()` is `is_torch_cuda_available() and _is_package_available("causal_conv1d")` — **CUDA-only**. causal-conv1d is a CUDA kernel with no Metal path at all. | **See PLAN-BREAKING #6** |

### Sharp edge for the Mac depth-probe specifically

`modeling_qwen3_5.py` **does not call** `is_flash_linear_attention_available()` or
`is_causal_conv1d_available()`. Dispatch is done by `use_kernel_func_from_hub_with_fallback`, which does a bare
`importlib.import_module(package)` inside a `try/except Exception` and silently falls back to the torch
function on *any* exception. Consequences on Apple Silicon:

- The plan's depth probe **will** run, on the pure-torch reference path: `torch_chunk_gated_delta_rule` and
  `torch_recurrent_gated_delta_rule` are ordinary PyTorch, and `causal_conv1d_fn`/`causal_conv1d_update` fall
  back to `F.conv1d`. Nothing in these paths is CUDA-specific in the source.
- **But the reference path casts to `torch.float32`** (`.to(torch.float32, memory_format=torch.contiguous_format)`
  in both GDN functions) and `config.mamba_ssm_dtype` is `"float32"`. Whether every op in that path has an MPS
  kernel at the installed torch version is **UNVERIFIED** — it can only be settled by running it on this Mac,
  which is outside this lane. The plan is right to flag it.
- If `fla` is installed on the Mac, transformers **will attempt to call it** (the bare-import dispatch, not the
  CUDA-gated helper). Triton has no Metal backend, so this fails at kernel launch rather than falling back
  cleanly. **Do not install `fla` on the Mac.**

---

## 7. Dataset licences and current hosts

Verified live against `huggingface.co/api`, `datasets-server.huggingface.co`, `zenodo.org/api`, `api.github.com`
and `arxiv.org`. Read-only existence checks used the HF token already present at `~/.cache/huggingface/token`;
**no gate agreement was accepted** — that requires the user's explicit action.

| Dataset (requested id) | Status | Evidence (host + licence + gate + size) | Impact if wrong |
|---|---|---|---|
| **`nuprl/AgentPack`** | **REFUTED (ungated assumption)** | Exists. `license: apache-2.0` (agrees across `cardData.license`, tag `license:apache-2.0`, and README YAML). **`"gated": "auto"`** — card renders *"You need to agree to share your contact information to access this dataset"*; an authenticated fetch returns **HTTP 403, `x-error-message: Access to dataset nuprl/AgentPack is restricted and you are not in the authorized list.`** Size via the still-public `tree` API: `train/` = 19 `.jsonl.gz` shards, **62,913,418,803 bytes (62.91 GB compressed)**, plus `canitedit/train.jsonl.gz` 33.4 MB and `octocoder/train.jsonl.gz` 6.0 MB. **Row count is contradictory:** README says *"1.3M commits … April up to mid-August 2025"*; arXiv 2509.21891**v2** (2026-03-27) says *"1.8M code edits … up to early October 2025"*. The repo carries `-202509` and `-202510` shards, so the README is stale — do not use 1.3M. | **See PLAN-BREAKING #8** |
| ↳ AgentPack **per-row source-repo licence field** | **UNVERIFIED — and evidence points to "does not exist"** | Column names could not be obtained; every route is gate-blocked (see "What I could not reach"). **Negative evidence from the authors' own paper (arXiv 2509.21891v2, 48,665 chars): the substring `licen` occurs exactly twice, and neither is a data field** — once in arXiv's own banner (`License: CC BY-SA 4.0`, which is the *paper's* licence, **not** the dataset's Apache-2.0), and once as a *filename* in the Appendix A.1 file-extension table beside `.gitignore` and `.env_template`. The paper describes **no licence-based filtering of source repositories anywhere.** It does confirm repo identity is captured: *"the GH Archive metadata has the commit message, the repository name, and the commit hash, which is all we need to later fetch the diff"* — so a repo identifier is very likely per-row (**inferred**, not verified). | **See PLAN-BREAKING #8** |
| `bigcode/commitpackft` | **VERIFIED** | Exists, `license: mit`, **not gated**, **702,062 rows / 1,545.02 MB** (confirmed twice inside the repo: README split table, and summing `line_count.txt` across 278 language files). **Critically, it HAS the per-row licence field AgentPack lacks.** README documents verbatim: `license` = *"license of the repository the code stems from, one of `['mit', 'artistic-2.0', 'isc', 'cc0-1.0', 'epl-1.0', 'mpl-2.0', 'unlicense', 'unknown', 'apache-2.0', 'bsd-3-clause', 'agpl-3.0', 'lgpl-2.1', 'bsd-2-clause']`"*; `repos` = *"name of the the repository the code stems from"*; and *"Each sample comes from a code repository with a permissive license. The license is provided by the `license` field for each sample."* | Licence-filter design |
| CLINC150 → `clinc/clinc_oos` | **VERIFIED** | Exists (bare `clinc_oos` resolves to the same repo), `license: ['cc-by-3.0']`, not gated, 3 configs; `plus` config = 15,250 / 3,100 / 5,500. | Intent-eval set |
| Banking77 → `PolyAI/banking77` | **VERIFIED (licence) / REFUTED (loadable)** | Exists, `license: ['cc-by-4.0']`, not gated, train 10,003 / test 3,080. **But it is script-only:** tree contains `banking77.py` and **no data files**; dataset-server returns HTTP 500 *"Dataset scripts are no longer supported, but found banking77.py"*. The script fetches CSVs live from `raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets`. Counts came from `dataset_infos.json`, which also states the licence as *"Creative Commons Attribution 4.0 International"*. | **See PLAN-BREAKING #9** |
| MASSIVE → `AmazonScience/massive` | **VERIFIED (licence) / REFUTED (loadable)** | Exists, `license: ['cc-by-4.0']`, not gated, *">1M utterances across 52 languages"*. In-repo `LICENSE` is the CC Attribution 4.0 International text, `Copyright Amazon.com Inc. or its affiliates`. **Also script-only** — `massive.py`, no data files, dataset-server HTTP 500 with the same "scripts are no longer supported" error. | **See PLAN-BREAKING #9** |
| MNLI → `nyu-mll/glue` cfg `mnli` | **VERIFIED, but licence is not usable as a value** | Exists, **`license: ['other']`**, not gated, 431,992 rows (train 392,702). GLUE card is explicit: *"The primary GLUE tasks are built on and derived from existing datasets. We refer users to the original licenses accompanying each dataset."* | Licence-filter design |
| MNLI → `nyu-mll/multi_nli` | **VERIFIED** | Exists, **`license: ['cc-by-3.0','cc-by-sa-3.0','mit','other']`** (four at once, incl. `other`), not gated, 392,702 / 9,815 / 9,832. | Licence-filter design |
| ANLI → `facebook/anli` | **REFUTED (for commercial use)** | Exists, not gated, 169,265 rows / 77.1 MB — but **`license: ['cc-by-nc-4.0']`. NON-COMMERCIAL.** | **See PLAN-BREAKING #9** |
| STS-B → `nyu-mll/glue` cfg `stsb` | **VERIFIED, licence `['other']`** | Exists, not gated, 8,628 rows. Same GLUE "refer to original licenses" caveat. | Licence-filter design |
| STS-B → `mteb/stsbenchmark-sts` | **VERIFIED, licence literally unknown** | Exists, not gated, 8,628 rows (5,749 / 1,500 / 1,379), **`license: 'unknown'`**. | **See PLAN-BREAKING #9** |
| SQuAD 2.0 → `rajpurkar/squad_v2` | **VERIFIED** | Exists, `license: ['cc-by-sa-4.0']`, not gated, 130,319 / 11,873, 128.4 MB. Note **ShareAlike**. | Redistribution terms |
| CodeReviewer → `microsoft/CodeReviewer` | **REFUTED (wrong host)** | **Does not exist as an HF dataset.** Both `microsoft/CodeReviewer` and `microsoft/codereviewer` return HTTP 404 `{"error":"Repository not found"}` **with valid auth** (a deliberate nonsense control id returned the same 404, so the 404 semantics are trustworthy, not an auth artefact). `microsoft/codereviewer` **is a model**, not a dataset (tag `license:apache-2.0`). Real data location: **Zenodo DOI `10.5281/zenodo.6900648`**, *"Automating Code Review Activities by Large-Scale Pre-Training"*, `license.id: cc-by-4.0`, access `open`, 4 files / **4.82 GB** (`Diff_Quality_Estimation.zip` 2.81 GB, `Code_Refinement.zip` 1.17 GB, `Comment_Generation.zip` 847 MB, `CodeReviewer.zip` 0.8 MB). The code repo `microsoft/CodeBERT` is **MIT** (GitHub API `spdx_id`) — **code MIT, data CC-BY-4.0, different terms.** | Data-acquisition step |
| Devign → `google/code_x_glue_cc_defect_detection` | **VERIFIED, licence is non-standard** | Exists, not gated, 27,318 rows / 56.9 MB, splits 21,854 / 2,732 / 2,732. **`license: ['c-uda']`** — card: *"Computational Use of Data Agreement (C-UDA) License"*, computational use only, **not a standard OSS licence**. Columns are `id`, `func`, `target`, `project`, `commit_id` — a `project` source field but **no per-row licence field** here either. | **See PLAN-BREAKING #9** |


---

## 8. Teacher model availability

| Claim | Status | Evidence | Impact if wrong |
|---|---|---|---|
| `Qwen/Qwen3.8-27B` exists | **VERIFIED** | `https://huggingface.co/api/models/Qwen/Qwen3.8-27B` → `gated: False`, `license: apache-2.0`. Config: `model_type qwen3_5`, `architectures ["Qwen3_5ForConditionalGeneration"]`, `num_hidden_layers 64`, `hidden_size 5120`, `intermediate_size 17408`, `vocab_size 248320`, `max_position_embeddings 262144`. No expert fields → **dense**. Same vocab as Qwen3.5-2B (248,320), so logit-level distillation is vocabulary-compatible. | Teacher pipeline |
| A 4-bit MLX build of the Qwen teacher exists | **VERIFIED** | `https://huggingface.co/api/models/mlx-community/Qwen3.8-27B-4bit` → exists, `gated: False`, `license: apache-2.0`, 99,250 downloads. Also `lmstudio-community/Qwen3.8-27B-MLX-4bit`. |  |
| **It is already on this Mac** | **VERIFIED** | `~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit` — **15 GB on disk**, snapshot `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` complete with `model-0000{1,2,3}-of-00003.safetensors`, `model.safetensors.index.json`, tokenizer and processor configs. **No `.incomplete` files.** Also locally cached: `mlx-community/Qwen3.8-27B-MTP-4bit`, `-nvfp4`, `-MTP-nvfp4`, `incoai/Qwen3.8-27B-DFlash2`. |  |
| `Gemma 4 31B` exists | **VERIFIED** | `https://huggingface.co/api/models/google/gemma-4-31B` → exists, **`gated: False`**, **`license: apache-2.0`**, 680,205 downloads. `google/gemma-4-31B-it` likewise `gated: False`, `apache-2.0`. Gemma 4 is **not** under the Gemma Terms of Use and **not** gated — unlike Gemma 2/3. | Teacher licensing |
| **A usable Gemma 4 31B is on this Mac** | **REFUTED** | `~/.cache/huggingface/hub/models--mlx-community--gemma-4-31B-it-assistant-bf16` exists but is **926 MB** — a 31B bf16 checkpoint is ~62 GB. This is a metadata/partial cache, **not a usable model**. Also present: `z-lab/gemma-4-31B-it-DFlash` and `mlx-community/gemma-4-e4b-it-4bit` (a much smaller E4B model). **No 4-bit MLX Gemma-4-31B is cached locally.** | Second-teacher plan |
| **"Qwen3.8 shipped nothing dense under 27B"** | **VERIFIED** | `https://huggingface.co/api/models?author=Qwen&search=Qwen3.8` returns the complete official lineup: **`Qwen3.8-27B`, `Qwen3.8-27B-FP8`, `Qwen3.8-Flash-Next`, `Qwen3.8-Flash-Next-FP8`, `Qwen3.8-2.4T-A95B`, `Qwen3.8-2.4T-A95B-FP8`** — six repos, nothing else. `Qwen3.8-Flash-Next` is **MoE**, not dense: `model_type qwen4_exp`, `architectures ["Qwen4ExpForConditionalGeneration"]`, `num_experts 512`, `moe_intermediate_size 640`, `num_hidden_layers 48`, `hidden_size 2560`. `Qwen3.8-2.4T-A95B` is MoE by name. So the smallest **dense** Qwen3.8 is 27B. (Sub-27B `Qwen3.8-*-Distill` repos on the Hub are third-party — `empero-ai` — not official Qwen releases.) | Teacher-size argument |

---

## 9. Lambda Labs on-demand pricing

Primary source: `https://lambda.ai/pricing`, "Instances pricing" section, fetched 2026-09-19 (response `date:`
header confirmed `Sat, 19 Sep 2026`). Note the rebrand: `lambdalabs.com/` 301s to `lambda.ai/`, but deep paths
were **not** preserved — `lambdalabs.com/pricing` and `lambdalabs.com/service/gpu-cloud` both return **HTTP
404**. `lambda.ai/service/gpu-cloud` redirects to `lambda.ai/instances`.

| Claim | Status | Evidence | Impact if wrong |
|---|---|---|---|
| **The price column is per-GPU, not per-instance** | **VERIFIED** | Column is literally labelled **`PRICE/GPU/HR*`**. Self-proving: **1x H100 SXM = $4.29** while **8x H100 SXM = $3.99**. If those were per-instance, eight GPUs would cost less than one. Rows were bound to their tabs via `aria-controls`/`aria-labelledby`, and vCPU/RAM/storage scale exactly 8:4:2:1 across tabs. | **See PLAN-BREAKING #7** |
| 1x GH200 (96 GB) | **VERIFIED** | **$2.29/hr** (1 GPU, so $2.29/hr instance). "NVIDIA GH200 \| 96 GB \| 64 vCPU \| 432 GiB \| 4 TiB SSD". Offered **only** as 1x — no 2x/4x/8x GH200 anywhere on the page. |  |
| 8x H100 SXM (80 GB) | **VERIFIED** | **$3.99/GPU/hr → $31.92/hr instance**. |  |
| 1x H100 PCIe (80 GB) | **VERIFIED** | **$3.29/hr** (1 GPU). Offered **only** as 1x. |  |
| 8x A100 SXM 80 GB | **VERIFIED** | **$2.79/GPU/hr → $22.32/hr instance**. Careful: the 8x tab lists **two** A100 SXM rows — 80 GB at $2.79 and 40 GB at $1.99. A100 80GB is offered **only** as 8x; the 1x/2x/4x tabs carry 40GB A100s only, so "1x A100 80GB" is not a purchasable shape. |  |

All figures exclude tax — every table is footnoted `* plus applicable sales tax/VAT/GST`.

### Recomputed session costs (instance $/hr × hours)

| Shape | Instance $/hr | 8h | 12h | 24h | 36h | 48h |
|---|---|---|---|---|---|---|
| 1x GH200 | $2.29 | **$18.32** | — | **$54.96** | — | **$109.92** |
| 8x H100 SXM | **$31.92** | — | **$383.04** | **$766.08** | **$1,149.12** | **$1,532.16** |
| 1x H100 PCIe | $3.29 | — | — | **$78.96** | — | **$157.92** |
| 8x A100 80GB | **$22.32** | — | — | **$535.68** | — | **$1,071.36** |

**Against a $1,250 total budget:** a single 36-hour 8xH100 session is **$1,149** — 92% of the entire budget.
A single 48-hour 8xH100 session is **$1,532** and **exceeds the budget outright**. A 48h 8xA100 session is
$1,071, 86% of budget. Only the GH200 and 1xH100-PCIe shapes are cheap enough to run repeatedly.

**On-demand vs reserved:** the page distinguishes them but never prints the literal string "on-demand". The
"Instances pricing" tables (used above) are described as *"self-serve, first-come access"* — these are the
on-demand rates. A separate "1-Click Clusters pricing" section (columns `DURATION`/`GPU COUNT`, values
"2 weeks – 1 year" and "1 year+", action "Talk to our team") is the reserved product, and is *more* expensive
per GPU: H100 at $6.16 / $5.85 / $5.54 per GPU/hr for 16 / 64 / 256 GPUs, since those are interconnected
multi-node clusters. The 1-year+ tier shows `—` rather than a price; no 3-year tier is listed.

---

# PLAN-BREAKING FINDINGS

### 1. GDN key and value head counts are **equal**, not asymmetric — REFUTED
`linear_num_key_heads = 16` and `linear_num_value_heads = 16` in `Qwen/Qwen3.5-2B-Base/config.json`. The plan
asserts key heads are *fewer* than value heads. Any code, memory calculation, state-shape assumption, or
narrative built on a K/V head asymmetry in the GDN blocks is wrong. There is no asymmetry to exploit or work
around. (Note: the *full-attention* layers **are** asymmetric — `num_attention_heads 8`,
`num_key_value_heads 2` — so the claim may be a transposition from the GQA layers onto the GDN layers.)

### 2. The S1 gate is a fail-open log grep, and the "43% slower" figure is inverted — REFUTED
Two independent defects:
- **Direction.** 1.22 s/step → 0.70 s/step means the fallback is **74% slower** (1.74×), and 43% is the
  *saving*. The transformers source itself says the `chunk_gated_delta_rule` gap is *"more than an order of
  magnitude on an H100."* Any throughput or cost model using 43% as the penalty understates it.
- **Mechanism.** The warning is emitted on **first forward**, not at startup, and is **suppressed entirely
  under `torch.compile`** by the `not is_torchdynamo_compiling()` guard. A gate that greps the startup log for
  it passes silently on a compiled run that is on the slow path. Replace with a positive runtime assertion that
  `fla` is importable and the resolved kernel is not the torch reference. If a grep is retained, the stable
  anchor is `falling back to its reference PyTorch implementation`. Exact strings in §3.

### 3. Sequence packing IS supported — the "no packing, 10-20% padding waste" item is stale — REFUTED
Issue #46093 closed `completed` on 2026-05-26; PR #45034 merged 2026-05-13 plumbs `cu_seq_lens_q` and
`seq_idx` into the Qwen3.5 GDN path. **Reclaim the padding waste.** But replace the line item with a hard
requirement, because the failure mode is worse than the one the plan feared: passing **`position_ids` alone**
(the TRL/veRL padding-free convention) runs without error while the FLA layers and causal convolutions fail to
isolate sample boundaries — *silent cross-sample leakage*. `DataCollatorWithFlattening` defaults
(`return_flash_attn_kwargs=False`, `return_seq_idx=False`) have the same silent effect. Maintainers confirmed
no `position_ids`-only path will be added and a fail-fast check is impractical. **Add an explicit assertion
that both kwargs reach the model.**

### 4. Step 1 (drop vision tower + MTP head) is already done for you, and the "main-only" premise is stale — REFUTED
- `Qwen3_5ForCausalLM` already declares `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`
  and constructs only `Qwen3_5TextModel` + `lm_head`. Loading that class drops both with **no surgery**.
- There is **no MTP module in transformers at all** — the checkpoint's 15 `mtp.*` tensors are discarded on
  load. You cannot "drop" a head transformers never builds; conversely, if the plan wanted to *keep* MTP, that
  is a from-scratch implementation, not a deletion.
- Qwen3.5 has been in **tagged releases since v5.2.0 (2026-02-16)**; current stable is **v5.17.0
  (2026-09-09)**. Pinning a raw `main` SHA is unnecessary and gives up release testing. Pin `v5.17.0`. (If a
  main pin is still wanted for some post-v5.17.0 fix: HEAD is
  `c587bc884db2c2e31fc2b8102314656b17aa07b1`, 2026-09-18.)

### 5. MLX PR #4020 is MERGED — the publishability argument is refuted as written — REFUTED
Merged **2026-09-15**, four days before this audit (commit `8f76a0aa2bbf9c29698337078db333c9bea1c1bf`). The
plan's "nothing in the open currently ships this" no longer holds. **Salvageable narrowing:** the merged work
is **forward-only** (no backward pass — explicitly future work), and the latest MLX *release* is **v0.32.2,
2026-08-25**, which predates the merge — so no released MLX wheel carries it. A claim restricted to *trainable*
Metal GDN, or to *released* packages, may still stand. Rewrite the claim before it appears in any writeup;
as stated it is falsifiable in one API call.

### 6. On Apple Silicon there is no fast path at all, by construction — VERIFIED (confirms and sharpens the plan's own flag)
`is_flash_linear_attention_available()` gates on `cuda or xpu or mlu` — **MPS is not in the list**.
`is_causal_conv1d_available()` is **CUDA-only**. `causal-conv1d` 1.7.0 ships **sdist only, no wheels**, and is
a CUDA kernel requiring nvcc. `fla-core` declares backends cuda/rocm/xpu/npu/cpu with **no Apple/Metal/MPS
reference anywhere**. The Mac depth probe therefore runs the **fp32 torch reference path** — which the source
comment rates at **>10× slower than the kernel on an H100** — so Mac timings cannot be extrapolated to cluster
timings at all; treat the probe as a correctness check only. Additional trap: `modeling_qwen3_5.py` dispatches
via a bare `importlib.import_module("fla")`, **not** via the CUDA-gated helper, so if `fla` is installed on the
Mac transformers will try to call Triton kernels that have no Metal backend. **Do not install `fla` on the
Mac.** Whether every op in the fp32 reference path has an MPS kernel at the installed torch version remains
**UNVERIFIED** — only a local run settles it.

### 7. Lambda prices are PER GPU — an 8x budget error is live — REFUTED (if the plan used the headline number)
The column is labelled **`PRICE/GPU/HR*`**. An 8x H100 SXM box is **$31.92/hr**, not $3.99/hr; 8x A100 80GB is
**$22.32/hr**, not $2.79/hr. Self-proving from the page: 1x H100 SXM is **$4.29** while 8x is **$3.99** —
per-instance reading would make eight GPUs cheaper than one. Against a **$1,250** budget: a 36h 8xH100 session
is **$1,149 (92% of budget)** and a 48h session is **$1,532 — over budget on its own**. Re-derive every session
total from the instance rates in §9 before the budget is used for anything. Also structural: **GH200 and H100
PCIe are 1x-only** (no multi-GPU shape exists), and **A100 80GB is 8x-only** (no 1x/2x/4x) — so any plan step
assuming a 4xA100 or a multi-GPU GH200 node cannot be bought at these rates.

### 8. `nuprl/AgentPack` is GATED, and the per-row licence field the filter design needs almost certainly does not exist — REFUTED
Two independent blockers on the plan's largest data dependency:
- **Gated.** `"gated": "auto"`; an authenticated fetch returns **403 `Access to dataset nuprl/AgentPack is
  restricted and you are not in the authorized list.`** A human must accept the terms before any of it —
  including the schema — can be read. Phase 1 cannot "confirm" this unattended.
- **No per-row source-repo licence field, on the best evidence available.** The plan flags this as unknown and
  says it gates the whole licence-filter design. It could not be settled directly (gate-blocked), but the
  authors' own paper contains the substring `licen` **exactly twice in 48,665 characters**, neither a data
  field, and describes **no licence-based filtering of source repos at all**. Do not let the arXiv banner
  `License: CC BY-SA 4.0` be mistaken for the dataset licence — that is the *paper's* licence; the dataset is
  Apache-2.0.
- **Consequence:** a filter that expects AgentPack rows to self-declare their upstream repo's licence has
  nothing to filter on. You would have to join repo name → licence yourself via the GitHub API, at 1.3–1.8M
  rows. **`bigcode/commitpackft` already has exactly the field you want** (`license` per row from a documented
  13-value permissive set, plus `repos`, plus an explicit "each sample comes from a permissive-licensed
  repository" statement) at 702,062 rows, MIT, ungated, and 40× smaller. If the licence filter is load-bearing,
  commitpackft is the design that works today and AgentPack is the one that does not.
- Also correct the size: **62.91 GB compressed**, and the row count is **1.8M** per the paper, not the 1.3M in
  the stale README.

### 9. Four datasets carry licences that break an automated licence filter, and two will not load at all — REFUTED
Licence problems:
- **`facebook/anli` is `cc-by-nc-4.0` — NON-COMMERCIAL.** Hard exclude for any commercial use; this is not a
  filter edge case, it is a disqualification.
- **`google/code_x_glue_cc_defect_detection` (Devign) is `c-uda`** — Computational Use of Data Agreement,
  computational use only, not a standard OSS licence.
- **`mteb/stsbenchmark-sts` is literally `'unknown'`** — cannot be auto-classified by any filter.
- **`nyu-mll/glue` is `'other'` for both MNLI and STS-B**, with the card explicitly deferring: *"We refer users
  to the original licenses accompanying each dataset."* GLUE supplies **no single licence** — each task's terms
  must be resolved upstream. `nyu-mll/multi_nli` reports **four licences simultaneously** including `other`.
- `rajpurkar/squad_v2` is `cc-by-sa-4.0` — **ShareAlike**, which carries redistribution obligations if derived
  data is published.

Loading problems (will fail at runtime, not licence-related):
- **`PolyAI/banking77` and `AmazonScience/massive` are script-only** — loader `.py` and no data files.
  Dataset-server returns HTTP 500 *"Dataset scripts are no longer supported"*. **These fail under
  `datasets>=3.0`.** banking77's script additionally fetches CSVs live from `raw.githubusercontent.com`, so it
  depends on a third-party GitHub path staying up.
- **`microsoft/CodeReviewer` does not exist on HuggingFace** (404 with valid auth, control-tested). The data is
  on **Zenodo DOI `10.5281/zenodo.6900648`**, CC-BY-4.0, 4.82 GB across 4 zips. Any plan step that
  `load_dataset("microsoft/CodeReviewer")` will fail. Note the split terms: the CodeBERT **code** is MIT, the
  **data** is CC-BY-4.0.


---

# WHAT I COULD NOT REACH

| Item | Why UNVERIFIED | What I tried |
|---|---|---|
| Whether the fp32 torch GDN reference path actually executes end-to-end on Apple MPS at the torch version installed on this Mac | Requires running it; this lane writes no code and executes no training. Source inspection shows nothing CUDA-specific in the reference path, but that is an inference, not a verification. | Read `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule` / `causal_conv1d_fn` in `modeling_qwen3_5.py` on `main`; read both availability gates in `import_utils.py`; read `fla-core` and `causal-conv1d` PyPI metadata for any Apple/Metal reference (none). |
| Live Lambda capacity — whether any of the four shapes is currently launchable | The pricing page carries no stock, capacity, or "sold out" indicator. "First-come access" implies capacity can be exhausted. Availability is only in the authenticated dashboard or `cloud.lambda.ai/api/v1/instance-types`. | Fetched `cloud.lambda.ai/api/v1/instance-types` → **401** `{"code":"global/invalid-api-key","message":"No API key was provided."}`. **Did not authenticate** — out of scope and not authorised. |
| Whether Lambda rates vary by region | The pricing page shows no region selector and no per-region pricing, and does not state that rates are uniform either way. | Read the full `lambda.ai/pricing` page and `lambda.ai/instances`. |
| The plan document itself | `/Users/bharath/Code/research/qwen-decision/` did not exist at audit time — only `AUDIT/`, created for this report. Every plan claim in this report is quoted from the audit brief, not from a plan file. Where the brief paraphrased a plan assertion, I verified the paraphrase. | `ls /Users/bharath/Code/research/qwen-decision/` → "No such file or directory"; `ls /Users/bharath/Code/research/` → BINN, Evo2PopgenSBI, GenoThermalTargeting, KneeMRIContactSheets, LocalModelBench, MLSystemsLab, ProprioceptionStudy, StressProject, tessl. |
| Exact provenance of the "43%" figure in the plan | I verified the underlying measurements (1.22 → 0.70 s/step) and showed 43% is the saving, not the penalty. I cannot confirm which the plan author intended without the plan text. | Read issue #48718 body in full via GitHub API. |
| **`nuprl/AgentPack` column names — including whether a per-row source-repo licence field exists** | The dataset is gated (`"gated": "auto"`); every schema route returns 401/403/404. **No gate agreement was accepted** — that is the user's action to take, not mine. Section 7 records strong *negative* evidence from the authors' paper, but the schema itself is unread. | `datasets-server…/first-rows?dataset=nuprl%2FAgentPack&config=default&split=train` → 401 unauthenticated, **404 authenticated**; `/info`, `/splits`, `/size`, `/is-valid` → same; `api/datasets/nuprl/AgentPack/croissant` → 401; `…/parquet/default/train/0.parquet` → listing public but **files 401/403**; `…/resolve/main/train/claude-202504.jsonl.gz` → 403 not-in-authorized-list; `…/resolve/main/canitedit/prepare.py` → 401; `github.com/nuprl/AgentPack` and two name variants → 404, **no public code repo exists**. Fell back to reading the full text of arXiv 2509.21891v2. |
| `nuprl/AgentPack` exact row count | README (1.3M) and paper (1.8M) disagree and the schema is unreadable. The repo's `-202509`/`-202510` shards show the README is stale, so 1.8M is the better figure — but it is **inferred**, not counted. | Compared README YAML, README prose, arXiv v2 abstract, and the `tree` API shard listing. |
| `bigcode/commitpackft` byte size from the official endpoint | The HF size endpoint returned **HTTP 500 `"server is busier than usual"`** on both attempts. The 702,062 rows / 1,545.02 MB figures came from the repo's own README split table and `line_count.txt` files instead — two in-repo sources that agree. | `datasets-server…/size?dataset=bigcode%2Fcommitpackft` ×2 → 500. |
| Original Devign release licence (`sites.google.com/view/devign`) | The landing page returns HTTP 200 but **no licence statement was located on it**. Only the `google/code_x_glue_cc_defect_detection` rehost's `c-uda` licence is verified. | Fetched the Devign landing page. |
