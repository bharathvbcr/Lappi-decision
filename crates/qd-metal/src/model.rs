//! The GPU model: real Qwen3.5-2B weights uploaded once, a prefill that leaves a
//! [`PrefixState`], continuations that read that state without writing it, and answer-row scoring.
//!
//! The layer loop is `tessl/src/bin/bench_qwen35_layers.rs` (commit 3c9b67d) with real weights
//! and state carried in and out:
//!
//! ```text
//! resid = embed(ids)                       host gather of the bf16 table, widened to f32
//! per layer:
//!   xb = rms_norm_bf16(resid, 1 + w_in_norm)
//!   GDN:  proj = xb @ W_in;  qkv = conv1d_silu(proj[:, :conv_dim], conv state)
//!         o = gdn_chunk_forward | gdn_recurrent (qkv, gates(proj), GDN state)
//!         y = gated_rms_norm(o, z(proj), w) -> bf16;  resid += y @ W_out
//!   attn: proj = xb @ W_in
//!         prefill:       attn_qk_norm_rope -> q, K/V cache;
//!                        o = attn_prefill_by_length (attn_prefill at every length)
//!         continuation:  attn_qk_norm_rope_suffix -> q, suffix K/V;
//!                        o = attn_prefix_rows(q, shared prefix K/V, suffix K/V)
//!         y = o * sigmoid(gate) -> bf16;  resid += y @ W_o
//!   xb = rms_norm_bf16(resid, 1 + w_post_norm)
//!   resid += cast_bf16(silu(xb @ W_gate) * (xb @ W_up)) @ W_down
//! logits = score_answer_rows(resid[slot rows], final norm (w_offset 1), embed[answer ids])
//! ```
//!
//! # State, and what "read-only" means here
//!
//! A [`PrefixState`] is every layer's carried state after a prefix: the conv history and the
//! recurrent GDN state for the 18 GDN layers, the K/V cache for the 6 attention layers. A
//! continuation ([`Model::run`] with `start = Some(state)`) reads all of it in place and writes
//! none of it: GDN and conv through `StateIn::Snapshot` (tessl never writes a snapshot and refuses
//! one as a `state_out`), attention through `qwen35::attn_prefix_rows`, which reads the prefix
//! K/V as a [`qwen35::SharedPrefix`] and takes each row's own suffix K/V from a separate cache.
//! The snapshot-hash test (`tests/gpu.rs`) checks the device bytes are unchanged.
//!
//! # Many questions, one prefix
//!
//! N questions over one prefix run as N batch rows, and nothing of the prefix is copied: GDN and
//! conv state are read at batch stride 0 (`StateIn::Snapshot`), and the prefix K/V at stride 0
//! (`attn_prefix_rows`, tessl 7095bec). The only copy is [`concat_kv`], and only when a
//! write-back (batch 1) has to materialise the new state's contiguous `prefix ‖ suffix` cache.

use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;

use tessl::gemm::cast_f32_to_bf16_into;
use tessl::qwen35::{
    self, AttnProjLayout, AttnShape, AttnTargets, Cols, GdnParams, GdnProjLayout, GdnWorkspace,
    LmHead, OutCols, StateIn,
};
use tessl::tensor::gpu_copy;
use tessl::{gemm, gemm_epilogue, nn, DType, Epilogue, GemmBackend, GpuBuffer, GpuRuntime, Tensor};

use crate::config::ModelConfig;
use crate::error::{GpuContext, MetalError, Result};
use crate::safetensors::SafeTensors;
use crate::weights::{self, HostLayer, HostMixer, TensorDigest, TEXT_PREFIX};

const BACKEND: GemmBackend = GemmBackend::TensorOps;

/// Continuations of at most this many tokens use the token-by-token GDN kernel; longer ones use
/// the chunked kernel. tessl's guidance: prefer the chunked path past a few dozen tokens.
pub const RECURRENT_MAX_SEQ: u32 = 16;

/// Layers prepared on the CPU at once during load (bounds the host copies held at a time).
const LOAD_PARALLEL_LAYERS: usize = 6;

/// The v1 prefix-state record's domain tag.
const DIGEST_V1_TAG: &[u8] = b"qd-metal.prefix-state.v1\0";

/// The digest scratch kept between digests is bounded by a state of this many tokens:
/// `qd-metal-serve`'s default `--max-tokens` (bin/serve.rs), so the product's longest default
/// context reuses its scratch (~423 MB on the 2B) instead of re-faulting it three times per
/// decision. A longer context's scratch is released after its digest.
pub const DIGEST_SCRATCH_RETAIN_TOKENS: u32 = 16_384;

/// Device bytes of a batch-1 [`PrefixState`] of `tokens` tokens on `cfg`, the
/// [`PrefixState::nbytes`] of such a state: per GDN layer the conv state `[conv_dim, kernel - 1]`
/// and the recurrent state, per attention layer K and V `[tokens, kv_heads, head_dim]`, all f32.
pub fn prefix_state_nbytes(cfg: &ModelConfig, tokens: u32) -> Result<usize> {
    let gdn = cfg.gdn_layout()?;
    let conv = gdn.conv_dim() as usize * (cfg.conv_kernel as usize - 1);
    let recurrent = gdn.dims(1, 1).state_elems_per_row();
    let per_pos = cfg.kv_heads as usize * cfg.head_dim as usize;
    let elems = cfg
        .n_gdn()
        .checked_mul(conv + recurrent)
        .zip(cfg.n_attention().checked_mul(2 * per_pos).and_then(|n| n.checked_mul(tokens as usize)))
        .and_then(|(fixed, kv)| fixed.checked_add(kv))
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| MetalError::Input(format!("a {tokens}-token prefix state overflows usize bytes")))?;
    Ok(elems)
}

/// The most scratch a thread keeps between digests: the state of [`DIGEST_SCRATCH_RETAIN_TOKENS`].
fn digest_scratch_retain_bytes(cfg: &ModelConfig) -> Result<usize> {
    prefix_state_nbytes(cfg, DIGEST_SCRATCH_RETAIN_TOKENS)
}

/// Release `scratch` if it holds more than `keep` bytes of capacity.
fn release_oversized_scratch(scratch: &mut Vec<u8>, keep: usize) {
    if scratch.capacity() > keep {
        *scratch = Vec::new();
    }
}

thread_local! {
    /// The host copy the parallel digest hashes, reused across digests on this thread. A copy is
    /// needed at all because tessl's host lease is exclusive — one live mapping per runtime
    /// (`runtime.rs` `acquire_access`) — and `GpuRuntime` is `!Send`, so the mapped bytes cannot be
    /// handed to hashing threads; owned bytes can. Copying runs at memory bandwidth, a few ms at
    /// 8K, against hundreds of ms of single-threaded SHA-256.
    static DIGEST_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// The v1 record over per-buffer SHA-256s: the domain tag, the token count, then for each buffer
/// in [`PrefixState`]'s fixed order its tag, its byte length and its SHA-256, all hashed once.
fn digest_v1<'a>(tokens: u32, parts: impl IntoIterator<Item = (&'a str, usize, [u8; 32])>) -> [u8; 32] {
    let mut acc = Vec::with_capacity(DIGEST_V1_TAG.len() + 4 + 48 * (8 + 8 + 32));
    acc.extend_from_slice(DIGEST_V1_TAG);
    acc.extend_from_slice(&tokens.to_le_bytes());
    for (tag, len, hash) in parts {
        acc.extend_from_slice(tag.as_bytes());
        acc.extend_from_slice(&(len as u64).to_le_bytes());
        acc.extend_from_slice(&hash);
    }
    qd_runtime::sha256(&acc)
}

struct GdnWeights {
    w_in: Tensor,
    w_out: Tensor,
    conv_w: GpuBuffer,
    a_log: GpuBuffer,
    dt_bias: GpuBuffer,
    norm_w: GpuBuffer,
}

struct AttnWeights {
    w_in: Tensor,
    w_out: Tensor,
    q_norm: GpuBuffer,
    k_norm: GpuBuffer,
}

enum Mixer {
    /// `slot` indexes the GDN entries of a [`PrefixState`].
    Gdn { w: GdnWeights, slot: usize },
    /// `slot` indexes the attention entries of a [`PrefixState`].
    Attn { w: AttnWeights, slot: usize },
}

struct Layer {
    in_norm: GpuBuffer,
    post_norm: GpuBuffer,
    mixer: Mixer,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
}

/// Every layer's carried state after a prefix of `tokens` tokens (batch 1).
pub struct PrefixState {
    tokens: u32,
    /// Per GDN layer: `[conv_dim, kernel - 1]` f32.
    conv: Vec<GpuBuffer>,
    /// Per GDN layer: `[v_heads, 128, v_dim]` f32.
    gdn: Vec<GpuBuffer>,
    /// Per attention layer: `[1, tokens, kv_heads, head_dim]` f32, capacity exactly `tokens`.
    k: Vec<GpuBuffer>,
    v: Vec<GpuBuffer>,
    /// [`digest_scratch_retain_bytes`] of the model that built this state.
    scratch_retain_bytes: usize,
}

impl PrefixState {
    pub fn tokens(&self) -> u32 {
        self.tokens
    }

    /// Device bytes this state holds.
    pub fn nbytes(&self) -> usize {
        self.buffers().map(|(_, b)| b.nbytes()).sum()
    }

    fn buffers(&self) -> impl Iterator<Item = (&'static str, &GpuBuffer)> {
        self.conv
            .iter()
            .map(|b| ("conv", b))
            .chain(self.gdn.iter().map(|b| ("gdn", b)))
            .chain(self.k.iter().map(|b| ("k", b)))
            .chain(self.v.iter().map(|b| ("v", b)))
    }

    /// SHA-256 over the state **as it is in device memory**: the token count, then each
    /// buffer's tag, length and SHA-256, in a fixed order (the v1 record, [`digest_v1`]). Reading
    /// maps the shared buffers, which first waits for every GPU command already encoded. Each
    /// buffer is one ordinary SHA-256, computed in parallel across buffers
    /// ([`qd_runtime::sha256_slices_parallel`]): the bytes a one-thread loop would give, which
    /// the 2026-10-03 A/B rows checked bit for bit before the one-thread loop was deleted.
    pub fn digest(&self) -> Result<[u8; 32]> {
        DIGEST_SCRATCH.with(|cell| {
            let mut scratch = cell.try_borrow_mut().map_err(|_| {
                MetalError::State("the prefix-state digest re-entered itself on one thread".into())
            })?;
            let out = self.digest_parallel(&mut scratch);
            release_oversized_scratch(&mut scratch, self.scratch_retain_bytes);
            out
        })
    }

    /// Copy every buffer into `scratch`, one host mapping at a time (each dropped before the next
    /// is taken: the lease is exclusive), then hash the copies in parallel.
    fn digest_parallel(&self, scratch: &mut Vec<u8>) -> Result<[u8; 32]> {
        scratch.clear();
        scratch.reserve(self.nbytes());
        let mut spans: Vec<(&'static str, std::ops::Range<usize>)> = Vec::new();
        for (tag, b) in self.buffers() {
            let start = scratch.len();
            {
                let bytes = b.try_contents_u8().gpu("map state for hashing")?;
                scratch.extend_from_slice(&bytes);
            }
            spans.push((tag, start..scratch.len()));
        }
        let slices: Vec<&[u8]> = spans.iter().map(|(_, r)| &scratch[r.clone()]).collect();
        let hashes =
            qd_runtime::sha256_slices_parallel(&slices, qd_runtime::SHA256_PARALLEL_MAX_WORKERS);
        Ok(digest_v1(
            self.tokens,
            spans.iter().zip(hashes).map(|((tag, r), h)| (*tag, r.len(), h)),
        ))
    }
}

/// Called after every layer with the layer index and the residual stream `[rows, hidden]`.
pub type LayerObserver<'a> = dyn FnMut(usize, &[f32]) -> Result<()> + 'a;

/// A forward pass's residual stream and, when asked for, the state after it.
pub struct RunOutput {
    /// `[batch * seq, hidden]` f32, before the final norm.
    pub resid: Tensor,
    pub batch: u32,
    pub seq: u32,
    pub state: Option<PrefixState>,
}

/// Answer-row scores for each slot row: logits and the log-softmax over the answer set.
#[derive(Debug, Clone, PartialEq)]
pub struct Scores {
    pub n_answers: usize,
    /// `[n_slots, n_answers]`
    pub logits: Vec<f32>,
    /// `[n_slots, n_answers]`
    pub logprobs: Vec<f32>,
}

pub struct Model {
    rt: Arc<GpuRuntime>,
    cfg: ModelConfig,
    gdn: GdnProjLayout,
    attn: AttnProjLayout,
    layers: Vec<Layer>,
    final_norm: GpuBuffer,
    /// `[vocab, hidden]` bf16: the embedding and, tied, the LM head.
    embed: Tensor,
    weight_hash: String,
    embed_digest: [u8; 32],
}

fn upload_f32(rt: &Arc<GpuRuntime>, data: &[f32], what: &str) -> Result<GpuBuffer> {
    let b = rt.alloc_buffer_hot(data.len().max(1) * 4).gpu(what)?;
    {
        let mut dst = b.try_contents_f32().gpu(what)?;
        dst[..data.len()].copy_from_slice(data);
    }
    Ok(b)
}

fn upload_u32(rt: &Arc<GpuRuntime>, data: &[u32], what: &str) -> Result<GpuBuffer> {
    let b = rt.alloc_buffer(data.len().max(1) * 4).gpu(what)?;
    {
        let mut dst = b.try_contents_u32().gpu(what)?;
        dst[..data.len()].copy_from_slice(data);
    }
    Ok(b)
}

fn upload_bf16(rt: &Arc<GpuRuntime>, rows: usize, cols: usize, bits: &[u16], what: &str) -> Result<Tensor> {
    if bits.len() != rows * cols {
        return Err(MetalError::Weights(format!(
            "{what}: {} elements for [{rows}, {cols}]",
            bits.len()
        )));
    }
    let t = rt.alloc_tensor_bf16_hot(&[rows, cols]).gpu(what)?;
    {
        let mut dst = t.buffer.try_contents_u16().gpu(what)?;
        dst[..bits.len()].copy_from_slice(bits);
    }
    Ok(t)
}

fn scratch_f32(rt: &Arc<GpuRuntime>, elems: usize, what: &str) -> Result<GpuBuffer> {
    rt.alloc_buffer(elems.max(1) * 4).gpu(what)
}

/// State buffers outlive the call: hot, so they stay resident rather than being recycled.
fn state_f32(rt: &Arc<GpuRuntime>, elems: usize, what: &str) -> Result<GpuBuffer> {
    rt.alloc_buffer_hot(elems.max(1) * 4).gpu(what)
}

fn to_u32(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| MetalError::Input(format!("{what} = {v} exceeds u32")))
}

fn residual_add() -> Epilogue<'static> {
    Epilogue {
        beta: 1.0,
        ..Epilogue::default()
    }
}

/// Attention of a continuation's queries over a shared prefix plus each row's own suffix.
///
/// Every continuation goes through here. One query per row (`tq == 1`) takes tessl's split-KV
/// `qwen35::attn_prefix_decode` (72eb526; bit-identical to `nn::flash_attn_decode` over a copied
/// prefix, its `o` is `[batch, heads, 256]`, the same bytes as `[batch, 1, heads, 256]`); more
/// takes `attn_prefix_rows`, whose one simdgroup per (row, head) walks all `P + S` keys and is
/// the wrong shape for a single query.
///
/// The live suffix length is one device u32 for the whole batch, so every row of a call has
/// the same suffix length. [`Model::run`] guarantees it: it takes `batch` rows of one `seq`, so
/// questions of different lengths are grouped by length by the caller, never padded.
#[allow(clippy::too_many_arguments)]
fn prefix_attention(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    prefix: qwen35::SharedPrefix<'_>,
    suffix_k: &GpuBuffer,
    suffix_v: &GpuBuffer,
    suffix_len: &GpuBuffer,
    q_pos: &GpuBuffer,
    o: &GpuBuffer,
    dims: nn::AttnDims,
) -> Result<()> {
    if dims.tq == 1 {
        qwen35::attn_prefix_decode(rt, q, prefix, suffix_k, suffix_v, suffix_len, q_pos, o, dims, false)
            .gpu("attn_prefix_decode")
    } else {
        qwen35::attn_prefix_rows(rt, q, prefix, suffix_k, suffix_v, suffix_len, q_pos, o, dims, false)
            .gpu("attn_prefix_rows")
    }
}

/// `prefix ‖ suffix` as one contiguous batch-1 cache of `p + s` positions: the attention state a
/// write-back leaves. The only K/V copy on the continuation path.
fn concat_kv(
    rt: &Arc<GpuRuntime>,
    prefix: &GpuBuffer,
    p: usize,
    suffix: &GpuBuffer,
    s: usize,
    per_pos: usize,
) -> Result<GpuBuffer> {
    let out = state_f32(rt, (p + s) * per_pos, "write-back kv")?;
    for (src, off, n) in [(prefix, 0, p), (suffix, p, s)] {
        if n == 0 {
            continue;
        }
        let from = Tensor::from_buffer(rt, src.clone(), &[n * per_pos], DType::F32, 0)
            .gpu("write-back kv source")?;
        let to = Tensor::from_buffer(rt, out.clone(), &[n * per_pos], DType::F32, off * per_pos * 4)
            .gpu("write-back kv destination")?;
        gpu_copy(&from, &to).gpu("write-back kv copy")?;
    }
    Ok(out)
}

/// Every intermediate of one pass, sized for `rows = batch * seq`.
struct Acts {
    rows: usize,
    resid: Tensor,
    xb: Tensor,
    g_proj: Tensor,
    g_qkv: GpuBuffer,
    g_o: GpuBuffer,
    g_y: Tensor,
    g_ws: Option<GdnWorkspace>,
    a_proj: Tensor,
    a_q: GpuBuffer,
    a_o: GpuBuffer,
    a_y: Tensor,
    /// `[seq]`: the live K/V length. A prefill has no prefix, so it is `attn_prefill`'s `tkv`;
    /// a continuation's is its suffix length, as `attn_prefix_rows` takes it.
    kv_len: GpuBuffer,
    q_pos: GpuBuffer,
    kv_pos: GpuBuffer,
    m_gate: Tensor,
    m_up: Tensor,
    m_mid: Tensor,
    m_midb: Tensor,
}

impl Model {
    /// Load the text tower from `snapshot` onto `rt`. `rt` must have TensorOps (bf16 GEMM).
    pub fn load(rt: &Arc<GpuRuntime>, snapshot: &Path) -> Result<Self> {
        if !rt.has_tensorops() {
            return Err(MetalError::Gpu(
                "bf16 GEMMs need the TensorOps backend, which this device lacks".into(),
            ));
        }
        let cfg = ModelConfig::load(snapshot)?;
        let st = SafeTensors::open(&weights_file(snapshot)?)?;
        weights::check_text_tensor_set(&st, &cfg)?;
        let (h, v) = (cfg.hidden, cfg.vocab);

        let embed_name = format!("{TEXT_PREFIX}embed_tokens.weight");
        st.expect(&embed_name, &[v, h], &[crate::safetensors::Dtype::Bf16])?;
        let embed = rt.alloc_tensor_bf16_hot(&[v, h]).gpu("embed")?;
        let embed_digest = {
            let mut dst = embed.buffer.try_contents_u8().gpu("embed")?;
            let n = v * h * 2;
            st.read_into(&embed_name, &mut dst[..n])?;
            qd_runtime::sha256(&dst[..n])
        };
        let mut digests: Vec<TensorDigest> = vec![(embed_name, embed_digest)];

        let norm_name = format!("{TEXT_PREFIX}norm.weight");
        let final_norm_host = {
            let info = st.expect(&norm_name, &[h], &[crate::safetensors::Dtype::Bf16, crate::safetensors::Dtype::F32])?.clone();
            let mut bytes = vec![0u8; info.nbytes()];
            st.read_into(&norm_name, &mut bytes)?;
            digests.push((norm_name, qd_runtime::sha256(&bytes)));
            crate::safetensors::widen(info.dtype, &bytes)
        };
        let final_norm = upload_f32(rt, &final_norm_host, "final norm")?;

        let gdn = cfg.gdn_layout()?;
        let attn = cfg.attn_layout()?;
        let mut layers = Vec::with_capacity(cfg.n_layers());
        let (mut n_gdn, mut n_attn) = (0usize, 0usize);
        let indices: Vec<usize> = (0..cfg.n_layers()).collect();
        for chunk in indices.chunks(LOAD_PARALLEL_LAYERS) {
            let prepared: Vec<Result<HostLayer>> = std::thread::scope(|s| {
                let handles: Vec<_> = chunk
                    .iter()
                    .map(|&i| {
                        let (st, cfg) = (&st, &cfg);
                        s.spawn(move || weights::prepare_layer(st, cfg, i))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|hd| {
                        hd.join().unwrap_or_else(|_| {
                            Err(MetalError::Weights("a layer-preparation thread panicked".into()))
                        })
                    })
                    .collect()
            });
            for host in prepared {
                let host = host?;
                digests.extend(host.digests.iter().cloned());
                let tag = |n: &str| format!("layer {} {n}", host.index);
                let mixer = match &host.mixer {
                    HostMixer::Gdn {
                        w_in,
                        w_out,
                        conv_w,
                        a_log,
                        dt_bias,
                        norm_w,
                    } => {
                        let m = Mixer::Gdn {
                            w: GdnWeights {
                                w_in: upload_bf16(rt, h, gdn.width() as usize, w_in, &tag("gdn w_in"))?,
                                w_out: upload_bf16(rt, gdn.value_dim() as usize, h, w_out, &tag("gdn w_out"))?,
                                conv_w: upload_f32(rt, conv_w, &tag("conv"))?,
                                a_log: upload_f32(rt, a_log, &tag("A_log"))?,
                                dt_bias: upload_f32(rt, dt_bias, &tag("dt_bias"))?,
                                norm_w: upload_f32(rt, norm_w, &tag("gated norm"))?,
                            },
                            slot: n_gdn,
                        };
                        n_gdn += 1;
                        m
                    }
                    HostMixer::Attn {
                        w_in,
                        w_out,
                        q_norm,
                        k_norm,
                    } => {
                        let m = Mixer::Attn {
                            w: AttnWeights {
                                w_in: upload_bf16(rt, h, attn.width() as usize, w_in, &tag("attn w_in"))?,
                                w_out: upload_bf16(rt, (cfg.q_heads * cfg.head_dim) as usize, h, w_out, &tag("attn w_out"))?,
                                q_norm: upload_f32(rt, q_norm, &tag("q_norm"))?,
                                k_norm: upload_f32(rt, k_norm, &tag("k_norm"))?,
                            },
                            slot: n_attn,
                        };
                        n_attn += 1;
                        m
                    }
                };
                layers.push(Layer {
                    in_norm: upload_f32(rt, &host.in_norm, &tag("input norm"))?,
                    post_norm: upload_f32(rt, &host.post_norm, &tag("post norm"))?,
                    mixer,
                    w_gate: upload_bf16(rt, h, cfg.intermediate, &host.w_gate, &tag("gate"))?,
                    w_up: upload_bf16(rt, h, cfg.intermediate, &host.w_up, &tag("up"))?,
                    w_down: upload_bf16(rt, cfg.intermediate, h, &host.w_down, &tag("down"))?,
                });
            }
        }
        rt.synchronize().gpu("weight upload")?;
        let weight_hash = weights::weight_hash(&mut digests);
        Ok(Self {
            rt: Arc::clone(rt),
            cfg,
            gdn,
            attn,
            layers,
            final_norm,
            embed,
            weight_hash,
            embed_digest,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn runtime(&self) -> &Arc<GpuRuntime> {
        &self.rt
    }

    /// SHA-256 over every tensor this model read (name + bytes digest), hex.
    pub fn weight_hash(&self) -> &str {
        &self.weight_hash
    }

    /// SHA-256 of the tied embedding / LM head bytes.
    pub fn embed_digest(&self) -> [u8; 32] {
        self.embed_digest
    }

    /// Gather the bf16 embedding rows of `ids` into `resid` as f32, on the host: the first write
    /// of every [`Model::run`], whose caller checked `ids` against the vocabulary. Mapping the
    /// table and the residual commits and waits for every GPU command already encoded. A one-
    /// dispatch GPU gather (`tessl::qwen35::embed_rows`) gave the same bits and measured no faster
    /// (ledger mac-qd-metal-2026-10-02, total min 409.7 vs 410.5 ms at T=512, 4,275.7 vs 4,281.1
    /// at T=8192), so it was deleted.
    fn embed_rows(&self, ids: &[u32], resid: &Tensor) -> Result<()> {
        let h = self.cfg.hidden;
        let mut rows = vec![0f32; ids.len() * h];
        {
            let table = self.embed.buffer.try_contents_u16().gpu("map embedding")?;
            for (i, &id) in ids.iter().enumerate() {
                let id = id as usize;
                let src = &table[id * h..(id + 1) * h];
                for (d, &s) in rows[i * h..(i + 1) * h].iter_mut().zip(src) {
                    *d = tessl::tensor::bf16_bits_to_f32(s);
                }
            }
        }
        let mut dst = resid.buffer.try_contents_f32().gpu("map residual")?;
        dst[..rows.len()].copy_from_slice(&rows);
        Ok(())
    }

    fn acts(&self, batch: u32, seq: u32, prefix: u32, chunked: bool) -> Result<Acts> {
        let rt = &self.rt;
        let rows = (batch as usize) * (seq as usize);
        let (h, inter) = (self.cfg.hidden, self.cfg.intermediate);
        let qd = (self.cfg.q_heads * self.cfg.head_dim) as usize;
        let vd = self.gdn.value_dim() as usize;
        // Every position of prefix + continuation must fit the kernels' u32 indexing.
        prefix
            .checked_add(seq)
            .ok_or_else(|| MetalError::Input("prefix + continuation exceeds u32 positions".into()))?;
        // The MLP's element count is passed to the kernels as u32; refuse it here, before any
        // allocation or dispatch, rather than in the layer loop after earlier layers are encoded.
        // Saturating: a product past usize is past u32 too, and `to_u32` names it.
        to_u32(rows.saturating_mul(inter), "mlp elements")?;
        let g_proj = rt.alloc_tensor_f32(&[rows, self.gdn.width() as usize]).gpu("gdn proj")?;
        let a_proj = rt.alloc_tensor_f32(&[rows, self.attn.width() as usize]).gpu("attn proj")?;
        let resid = rt.alloc_tensor_f32(&[rows, h]).gpu("resid")?;
        for t in [&g_proj, &a_proj, &resid] {
            if t.byte_offset() != 0 {
                return Err(MetalError::Gpu(
                    "the qwen35 kernels and the embedding gather address the projections and the \
                     residual from their buffer's start"
                        .into(),
                ));
            }
        }
        Ok(Acts {
            rows,
            resid,
            xb: rt.alloc_tensor_bf16(&[rows, h]).gpu("xb")?,
            g_proj,
            g_qkv: scratch_f32(rt, rows * self.gdn.conv_dim() as usize, "gdn qkv")?,
            g_o: scratch_f32(rt, rows * vd, "gdn out")?,
            g_y: rt.alloc_tensor_bf16(&[rows, vd]).gpu("gdn y")?,
            g_ws: if chunked {
                Some(GdnWorkspace::new(rt, &self.gdn.dims(batch, seq)).gpu("gdn workspace")?)
            } else {
                None
            },
            a_proj,
            a_q: scratch_f32(rt, rows * qd, "attn q")?,
            a_o: scratch_f32(rt, rows * qd, "attn out")?,
            a_y: rt.alloc_tensor_bf16(&[rows, qd]).gpu("attn y")?,
            kv_len: upload_u32(rt, &[seq], "kv len")?,
            q_pos: upload_u32(rt, &[prefix], "q pos")?,
            kv_pos: upload_u32(rt, &[0], "kv pos")?,
            m_gate: rt.alloc_tensor_f32(&[rows, inter]).gpu("mlp gate")?,
            m_up: rt.alloc_tensor_f32(&[rows, inter]).gpu("mlp up")?,
            m_mid: rt.alloc_tensor_f32(&[rows, inter]).gpu("mlp mid")?,
            m_midb: rt.alloc_tensor_bf16(&[rows, inter]).gpu("mlp mid bf16")?,
        })
    }

    /// One forward pass over `batch` rows of `seq` tokens (`ids` row-major, `batch * seq`).
    ///
    /// * `start = None` is a prefill from nothing; `Some(state)` continues every row from that
    ///   state, which is read and never written.
    /// * `keep_state` (batch 1 only) returns the state after this pass: for a prefill, the
    ///   prefix state; for a continuation, the state after prefix + these tokens.
    /// * `observe`, when given, is called after every layer with the residual stream
    ///   (`[batch * seq, hidden]`). It synchronizes per layer, so it is for parity only.
    pub fn run(
        &self,
        ids: &[u32],
        batch: u32,
        seq: u32,
        start: Option<&PrefixState>,
        keep_state: bool,
        mut observe: Option<&mut LayerObserver<'_>>,
    ) -> Result<RunOutput> {
        let rt = &self.rt;
        let cfg = &self.cfg;
        if batch == 0 || seq == 0 {
            return Err(MetalError::Input(format!("batch {batch} x seq {seq} is empty")));
        }
        let rows = (batch as usize)
            .checked_mul(seq as usize)
            .ok_or_else(|| MetalError::Input("batch x seq overflows".into()))?;
        if ids.len() != rows {
            return Err(MetalError::Input(format!(
                "{} ids for batch {batch} x seq {seq}",
                ids.len()
            )));
        }
        if let Some(bad) = ids.iter().find(|&&id| id as usize >= cfg.vocab) {
            return Err(MetalError::Input(format!("token id {bad} is outside the {}-row vocabulary", cfg.vocab)));
        }
        if keep_state && batch != 1 {
            return Err(MetalError::Input("keep_state is for batch 1".into()));
        }
        if let Some(s) = start
            && (s.conv.len() != cfg.n_gdn() || s.k.len() != cfg.n_attention())
        {
            return Err(MetalError::State("prefix state does not match this model's layers".into()));
        }
        let prefix = start.map_or(0, |s| s.tokens);
        let recurrent = start.is_some() && seq <= RECURRENT_MAX_SEQ;
        let a = self.acts(batch, seq, prefix, !recurrent)?;
        self.embed_rows(ids, &a.resid)?;

        let (h, eps) = (cfg.hidden, cfg.rms_eps);
        let rows_u = to_u32(rows, "rows")?;
        let hidden_u = to_u32(h, "hidden")?;
        let kw = cfg.conv_kernel;
        let conv_dim = self.gdn.conv_dim();
        let vd = self.gdn.value_dim();
        let qd = cfg.q_heads * cfg.head_dim;
        let per_pos = (cfg.kv_heads * cfg.head_dim) as usize;

        let mut state_conv = Vec::new();
        let mut state_gdn = Vec::new();
        let mut state_k = Vec::new();
        let mut state_v = Vec::new();

        for (li, layer) in self.layers.iter().enumerate() {
            nn::rms_norm_bf16(rt, &a.resid.buffer, &layer.in_norm, &a.xb.buffer, rows_u, hidden_u, eps)
                .gpu("input norm")?;
            match &layer.mixer {
                Mixer::Gdn { w, slot } => {
                    let proj = &a.g_proj.buffer;
                    qwen35::fused_projection(&a.xb, &w.w_in, &a.g_proj, BACKEND).gpu("gdn in-proj")?;
                    let conv_in = match start {
                        Some(s) => StateIn::Snapshot(&s.conv[*slot]),
                        None => StateIn::Zero,
                    };
                    let conv_out = if keep_state {
                        Some(state_f32(rt, batch as usize * conv_dim as usize * (kw as usize - 1), "conv state")?)
                    } else {
                        None
                    };
                    qwen35::conv1d_silu(
                        rt,
                        Cols::dense(proj, self.gdn.width()),
                        &w.conv_w,
                        kw,
                        conv_in,
                        &a.g_qkv,
                        conv_out.as_ref(),
                        batch,
                        seq,
                        conv_dim,
                    )
                    .gpu("conv1d_silu")?;
                    let dims = self.gdn.dims(batch, seq);
                    let gdn_in = match start {
                        Some(s) => StateIn::Snapshot(&s.gdn[*slot]),
                        None => StateIn::Zero,
                    };
                    let gdn_out = if keep_state {
                        Some(state_f32(rt, batch as usize * dims.state_elems_per_row(), "gdn state")?)
                    } else {
                        None
                    };
                    let params = GdnParams {
                        a_log: &w.a_log,
                        dt_bias: &w.dt_bias,
                    };
                    match &a.g_ws {
                        Some(ws) => qwen35::gdn_chunk_forward(
                            rt,
                            &dims,
                            &self.gdn.conv_qkv(&a.g_qkv),
                            &self.gdn.gates(proj),
                            &params,
                            gdn_in,
                            ws,
                            Cols::dense(&a.g_o, vd),
                            gdn_out.as_ref(),
                        )
                        .gpu("gdn_chunk_forward")?,
                        None => qwen35::gdn_recurrent(
                            rt,
                            &dims,
                            &self.gdn.conv_qkv(&a.g_qkv),
                            &self.gdn.gates(proj),
                            &params,
                            gdn_in,
                            Cols::dense(&a.g_o, vd),
                            gdn_out.as_ref(),
                        )
                        .gpu("gdn_recurrent")?,
                    }
                    qwen35::gated_rms_norm(
                        rt,
                        Cols::dense(&a.g_o, vd),
                        self.gdn.z(proj),
                        &w.norm_w,
                        OutCols {
                            cols: Cols::dense(&a.g_y.buffer, vd),
                            dtype: DType::BF16,
                        },
                        rows_u,
                        cfg.gdn_v_heads,
                        cfg.gdn_v_dim,
                        eps,
                    )
                    .gpu("gated_rms_norm")?;
                    qwen35::project_residual(&a.g_y, &w.w_out, &a.resid, BACKEND).gpu("gdn out-proj")?;
                    if let (Some(c), Some(g)) = (conv_out, gdn_out) {
                        state_conv.push(c);
                        state_gdn.push(g);
                    }
                }
                Mixer::Attn { w, slot } => {
                    let pc = Cols::dense(&a.a_proj.buffer, self.attn.width());
                    qwen35::fused_projection(&a.xb, &w.w_in, &a.a_proj, BACKEND).gpu("attn in-proj")?;
                    let shape = AttnShape {
                        batch,
                        seq,
                        q_heads: cfg.q_heads,
                        kv_heads: cfg.kv_heads,
                        head_dim: cfg.head_dim,
                        rotary_dim: cfg.rotary_dim,
                    };
                    let dims = nn::AttnDims {
                        batch,
                        tq: seq,
                        heads: cfg.q_heads,
                        heads_kv: cfg.kv_heads,
                        window: 0,
                        scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                    };
                    // This pass's own K/V, [batch, seq, kv_heads, head_dim]: the whole cache for
                    // a prefill, each row's suffix cache for a continuation.
                    let own = batch as usize * seq as usize * per_pos;
                    let alloc = if keep_state && start.is_none() { state_f32 } else { scratch_f32 };
                    let (kc, vc) = (alloc(rt, own, "k cache")?, alloc(rt, own, "v cache")?);
                    let targets = AttnTargets {
                        q_out: &a.a_q,
                        k_cache: &kc,
                        v_cache: &vc,
                    };
                    match start {
                        None => {
                            qwen35::attn_qk_norm_rope(
                                rt, &shape, pc, &w.q_norm, &w.k_norm, &targets, 0, cfg.rope_theta, eps,
                            )
                            .gpu("attn_qk_norm_rope")?;
                            // `qwen35::prefill_attn_kernel` is `attn_prefill` at every
                            // length. Same buffers, causal mask and f32 accumulate.
                            // Head dim is the h256 kernel (`config` refuses any other).
                            qwen35::attn_prefill_by_length(
                                rt, &a.a_q, &kc, &vc, &a.a_o, &a.kv_len, &a.q_pos, &a.kv_pos, dims,
                                false,
                            )
                            .gpu("attn_prefill_by_length")?;
                        }
                        Some(s) => {
                            qwen35::attn_qk_norm_rope_suffix(
                                rt, &shape, pc, &w.q_norm, &w.k_norm, &targets, prefix, 0,
                                cfg.rope_theta, eps,
                            )
                            .gpu("attn_qk_norm_rope_suffix")?;
                            prefix_attention(
                                rt,
                                &a.a_q,
                                qwen35::SharedPrefix {
                                    k: &s.k[*slot],
                                    v: &s.v[*slot],
                                    len: prefix,
                                },
                                &kc,
                                &vc,
                                &a.kv_len,
                                &a.q_pos,
                                &a.a_o,
                                dims,
                            )?;
                        }
                    }
                    qwen35::attn_output_gate(
                        rt,
                        &a.a_o,
                        pc,
                        OutCols {
                            cols: Cols::dense(&a.a_y.buffer, qd),
                            dtype: DType::BF16,
                        },
                        rows_u,
                        cfg.q_heads,
                        cfg.head_dim,
                    )
                    .gpu("attn_output_gate")?;
                    gemm_epilogue(&a.a_y, &w.w_out, &a.resid, BACKEND, residual_add()).gpu("attn o-proj")?;
                    if keep_state {
                        let (k, v) = match start {
                            None => (kc, vc),
                            Some(s) => (
                                concat_kv(rt, &s.k[*slot], prefix as usize, &kc, seq as usize, per_pos)?,
                                concat_kv(rt, &s.v[*slot], prefix as usize, &vc, seq as usize, per_pos)?,
                            ),
                        };
                        state_k.push(k);
                        state_v.push(v);
                    }
                }
            }
            // MLP
            nn::rms_norm_bf16(rt, &a.resid.buffer, &layer.post_norm, &a.xb.buffer, rows_u, hidden_u, eps)
                .gpu("post norm")?;
            gemm(&a.xb, &layer.w_gate, &a.m_gate, BACKEND).gpu("mlp gate")?;
            gemm(&a.xb, &layer.w_up, &a.m_up, BACKEND).gpu("mlp up")?;
            let n_mid = to_u32(rows * cfg.intermediate, "mlp elements")?;
            nn::mlp_silu(rt, &a.m_gate.buffer, &a.m_up.buffer, &a.m_mid.buffer, n_mid).gpu("mlp_silu")?;
            cast_f32_to_bf16_into(&a.m_mid, &a.m_midb).gpu("mlp cast")?;
            gemm_epilogue(&a.m_midb, &layer.w_down, &a.resid, BACKEND, residual_add()).gpu("mlp down")?;

            if let Some(f) = observe.as_deref_mut() {
                rt.synchronize().gpu("observe sync")?;
                let resid = a.resid.buffer.try_contents_f32().gpu("observe map")?;
                f(li, &resid[..a.rows * h])?;
            }
        }

        let state = if keep_state {
            Some(PrefixState {
                tokens: prefix + seq,
                conv: state_conv,
                gdn: state_gdn,
                k: state_k,
                v: state_v,
                scratch_retain_bytes: digest_scratch_retain_bytes(cfg)?,
            })
        } else {
            None
        };
        Ok(RunOutput {
            resid: a.resid,
            batch,
            seq,
            state,
        })
    }

    /// Score `answers` at the given rows of `out`'s residual stream (final norm + tied LM head,
    /// restricted to the answer rows). Waits for the GPU.
    pub fn score(&self, out: &RunOutput, slot_rows: &[u32], answers: &[u32]) -> Result<Scores> {
        let rt = &self.rt;
        // `RunOutput`'s fields are public: a hand-built one must not wrap past the row check.
        let rows = out.batch.checked_mul(out.seq).ok_or_else(|| {
            MetalError::Input(format!("{} x {} rows overflow u32", out.batch, out.seq))
        })?;
        if slot_rows.is_empty() || answers.is_empty() {
            return Err(MetalError::Input("scoring needs at least one slot row and one answer".into()));
        }
        if let Some(bad) = slot_rows.iter().find(|&&r| r >= rows) {
            return Err(MetalError::Input(format!("slot row {bad} is outside the {rows} rows run")));
        }
        if let Some(bad) = answers.iter().find(|&&id| id as usize >= self.cfg.vocab) {
            return Err(MetalError::Input(format!("answer id {bad} is outside the vocabulary")));
        }
        let n_slots = to_u32(slot_rows.len(), "slots")?;
        let n_answers = to_u32(answers.len(), "answers")?;
        let slots = upload_u32(rt, slot_rows, "slots")?;
        let ans = upload_u32(rt, answers, "answers")?;
        let n_out = slot_rows.len() * answers.len();
        let logits = scratch_f32(rt, n_out, "logits")?;
        let logprobs = scratch_f32(rt, n_out, "logprobs")?;
        qwen35::score_answer_rows(
            rt,
            &out.resid.buffer,
            rows,
            to_u32(self.cfg.hidden, "hidden")?,
            &slots,
            n_slots,
            &self.final_norm,
            1.0,
            self.cfg.rms_eps,
            LmHead {
                weight: &self.embed.buffer,
                dtype: DType::BF16,
                vocab: to_u32(self.cfg.vocab, "vocab")?,
            },
            &ans,
            n_answers,
            &logits,
            &logprobs,
        )
        .gpu("score_answer_rows")?;
        let read = |b: &GpuBuffer| -> Result<Vec<f32>> {
            let m = b.try_contents_f32().gpu("read scores")?;
            Ok(m[..n_out].to_vec())
        };
        Ok(Scores {
            n_answers: answers.len(),
            logits: read(&logits)?,
            logprobs: read(&logprobs)?,
        })
    }

    /// Prefill `ids` from nothing and keep the state.
    pub fn prefill(&self, ids: &[u32]) -> Result<(RunOutput, PrefixState)> {
        let seq = to_u32(ids.len(), "prefill tokens")?;
        let mut out = self.run(ids, 1, seq, None, true, None)?;
        let state = out
            .state
            .take()
            .ok_or_else(|| MetalError::State("prefill kept no state".into()))?;
        Ok((out, state))
    }
}

/// The single safetensors file holding the text tower. A multi-file checkpoint is refused: this
/// loader reads one file, and a text tower split across several is not what it was built for.
pub fn weights_file(snapshot: &Path) -> Result<std::path::PathBuf> {
    let index = snapshot.join("model.safetensors.index.json");
    if index.is_file() {
        let text = std::fs::read_to_string(&index)
            .map_err(|e| MetalError::Weights(format!("{}: {e}", index.display())))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| MetalError::Weights(format!("{}: {e}", index.display())))?;
        let map = v
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| MetalError::Weights(format!("{}: no weight_map", index.display())))?;
        let mut files: Vec<&str> = map
            .iter()
            .filter(|(k, _)| k.starts_with(TEXT_PREFIX))
            .filter_map(|(_, f)| f.as_str())
            .collect();
        files.sort_unstable();
        files.dedup();
        return match files.as_slice() {
            [one] => Ok(snapshot.join(one)),
            other => Err(MetalError::Weights(format!(
                "the text tower spans {} files ({other:?}); this loader reads exactly one",
                other.len()
            ))),
        };
    }
    let single = snapshot.join("model.safetensors");
    if single.is_file() {
        return Ok(single);
    }
    Err(MetalError::Weights(format!("{}: no safetensors index or model.safetensors", snapshot.display())))
}

#[cfg(test)]
mod digest_record_tests {
    use super::{
        digest_scratch_retain_bytes, digest_v1, prefix_state_nbytes, release_oversized_scratch,
        DIGEST_SCRATCH_RETAIN_TOKENS,
    };
    use crate::config::tests_support::QWEN35_2B;
    use crate::config::ModelConfig;

    /// The v1 record against an independent oracle: Python's `hashlib` over the same layout
    /// (`AUDIT/qdm-digest-2026-10-03/digest_v1_oracle.py`). It pins the record layout the GPU
    /// pins rely on without a GPU, and with qd-runtime's `sha256_parallel_tests` (each parallel
    /// hash equals the one-thread `sha256`) it is the tessl-independent guard on the digest code.
    #[test]
    fn the_v1_record_matches_an_independent_oracle() {
        let k: Vec<u8> = (0..=255u8).cycle().take(768).collect();
        let parts: [(&str, &[u8]); 4] = [("conv", b"abc"), ("gdn", b""), ("k", &k), ("v", b"\x00")];
        let got = digest_v1(7, parts.iter().map(|(t, b)| (*t, b.len(), qd_runtime::sha256(b))));
        assert_eq!(
            qd_runtime::hex(&got),
            "a1b6d7a836b2b373bcc9b3f2da2e3195ae7ae503a68e71934cabd39945d44079"
        );
    }

    /// The size formula against states measured on the GPU: `state_mb` of the 2026-10-03 A/B
    /// rows (147da0cc, 28505f4c) at prefixes of 115, 399, 762, 2033 and 8185 tokens.
    #[test]
    fn prefix_state_bytes_match_the_measured_states() {
        let cfg = ModelConfig::from_json(QWEN35_2B).unwrap();
        for (tokens, bytes) in [
            (115, 23_027_712),
            (399, 30_007_296),
            (762, 38_928_384),
            (2033, 70_164_480),
            (8185, 221_356_032),
        ] {
            assert_eq!(prefix_state_nbytes(&cfg, tokens).unwrap(), bytes, "{tokens} tokens");
        }
    }

    /// Fail-first (Fable ruling 2, #1): the retain cap was a fixed 256 MiB, below the ~423 MB
    /// state of the 16,384 tokens `qd-metal-serve` admits by default, so that scratch was released
    /// after every digest and re-faulted on the next. The capacity is reserved, never touched.
    #[test]
    fn the_scratch_of_a_default_max_tokens_state_is_kept() {
        let cfg = ModelConfig::from_json(QWEN35_2B).unwrap();
        let state = prefix_state_nbytes(&cfg, DIGEST_SCRATCH_RETAIN_TOKENS).unwrap();
        assert!(state > 256 << 20, "the case this test is about: {state} bytes");
        let keep = digest_scratch_retain_bytes(&cfg).unwrap();
        let mut scratch: Vec<u8> = Vec::with_capacity(state);
        release_oversized_scratch(&mut scratch, keep);
        assert!(scratch.capacity() >= state, "a {state}-byte scratch was released (keep {keep})");
        // One token past the bound is released: the cap still bounds.
        let mut longer: Vec<u8> = Vec::with_capacity(keep + 1);
        release_oversized_scratch(&mut longer, keep);
        assert_eq!(longer.capacity(), 0);
    }
}
