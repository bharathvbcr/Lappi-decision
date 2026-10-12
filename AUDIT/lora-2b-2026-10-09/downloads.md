# LoRA-2B downloads: Qwen3.5-2B-Base and Qwen3.5-0.8B-Base (fetched 2026-10-12)

Approved by the human on 2026-10-09 (`human-decisions.md`, row "D4 + D8"; Fable D4 in `fable-ft-decisions.md`).
Fetched by fix lane `lane4-lora-ckpt` with `huggingface_hub` 1.33.0 already installed in
`/Users/bharath/.venvs/ml` (nothing installed). Labels: **[V]** I ran the command this session and read
its output; **[I]** inferred.

## Repo ids and where they landed

The repo ids are the ones the ruling names: `Qwen/Qwen3.5-2B-Base` (D4: "snapshot `b1485b2f…`") and
`Qwen/Qwen3.5-0.8B-Base` (D4: "the smoke and oracle-cycle model") [V `rg` over this folder]. Both resolved on the
Hub [V `HfApi().model_info`]. The cache is the default HF hub cache, the same convention every other path in this
repo uses (`~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/<sha>`; e.g.
`python/tests/test_backbone.py:SNAPSHOT`, `crates/qd-metal/src/config.rs:282`). Disk before the fetch: 288 GiB free
(`df -h`), above the 30 GB floor [V].

| Model | Hub revision (resolved commit) | Local snapshot |
|---|---|---|
| `Qwen/Qwen3.5-2B-Base` | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| `Qwen/Qwen3.5-0.8B-Base` | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |

The 2B-Base revision equals the snapshot every v5 and v6 number was run on (`HANDOFF/v5-queue-2026-10-02.md:383`
and others) [V]. Before this fetch the local 2B-Base snapshot held only `config.json`, `tokenizer.json` and
`tokenizer_config.json`; the fetch added the weights and the other nine files beside them (12 files each; the Hub
revision is unchanged, so the earlier three files are the same bytes) [V].

## Weight shards

Each model is one shard. `shasum -a 256` run over the file the snapshot symlink resolves to; byte counts equal the
Hub's `files_metadata` sizes [V].

| File | Bytes | sha256 |
|---|---|---|
| `Qwen3.5-2B-Base/.../model.safetensors-00001-of-00001.safetensors` | 4,548,221,488 | `928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d` |
| `Qwen3.5-0.8B-Base/.../model.safetensors-00001-of-00001.safetensors` | 1,746,942,600 | `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c` |

Other files that the loaders read (sha256, [V]):

| File | 2B-Base | 0.8B-Base |
|---|---|---|
| `config.json` | `ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4` | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| `model.safetensors.index.json` | `74d2ddfe79f10f35b27b498632f02b97b60dd9ec39b35d7c5a890c399284e319` | `ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a` |
| `tokenizer.json` | `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927` | same (identical bytes) |
| `tokenizer_config.json` | `e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de` | same (identical bytes) |

The two models share a tokenizer byte for byte, as the 4B-Base's `tokenizer.json` does too [V].

## Stored dtypes (the ruling-2a map)

Read from the safetensors headers (`/Users/bharath/.venvs/ml/bin/python -I`, header JSON only, no tensor data) [V].
Both models have 320 `model.language_model.*` text tensors, 36 of them F32 and 284 BF16. The 36 are exactly
`layers.<i>.linear_attn.A_log` and `layers.<i>.linear_attn.norm.weight` for the 18 GDN layers; every F32 name matches
one of those two suffixes. (2B-Base also stores 312 non-text BF16 tensors, 0.8B-Base 168; they are the vision tower
and are not loaded.) This matches ruling 2a's "36" for the 2B instruct header and now holds for the Base headers.

Digests (formulas, so the pre-registration can pin them as a rule and a later reader can recompute):

- `dtype_map_sha256` = sha256 over `"\n".join(f"{name}\t{dtype}" for name in sorted(text_names))` where `name` has the
  `model.language_model.` prefix stripped: `f03e32c6115fe57e701de501b202232e53b0ec28339cc25aa75d7b6c85c5641a`
  (identical for 2B-Base and 0.8B-Base: same names, same dtypes).
- `f32_names_sha256` = sha256 over `"\n".join(sorted names stored F32)`:
  `e8f4db87650ee41693ee41fbb719ed14782b1d984d3b924b7a06ac99109ab9b1` (identical for both).
- Safetensors header sha256 (`json.dumps(header_without___metadata__, sort_keys=True)`): 2B-Base
  `7a1a78d58dd6cb7195ed93a3dbd5d8c8e917514cc4914a3a442160f4c78f9c9a`, 0.8B-Base
  `44db78c384ece32b3a1fcedd12334a0213489b26edd0b696c94ff5b0a0be0c97`.

`load_text_tower(..., keep_stored_dtype=<those 36 names>)` (commit `3af1ae2`) keeps them f32; the default is the v5/v6 cast.

## Not done here

Nothing was loaded onto a GPU and no model was run. The 0.8B-Base config was not loaded into either Rust stack (Fable
D4 marks "passes both stacks' shape checks" as [I] by inspection; that stays [I]).
