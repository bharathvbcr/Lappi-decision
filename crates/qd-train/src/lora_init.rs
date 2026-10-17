//! The LoRA-A init rule, recomputed without torch.
//!
//! `python/qd_train/lora.py::attach_lora` draws every adapter's `A` (`[r, in]`) from one CPU
//! `torch.Generator` seeded with the run seed, visiting the adapted modules in
//! `model.named_modules()` order and calling `nn.init.kaiming_uniform_(a=sqrt(5))` on each. The
//! pre-registration (`campaign/lora-2b-preregistered.json`, `init.lora_a`) pins that as a *rule*:
//! the A digest is a function of the seed, recorded in the row, and checked by recomputing it.
//! This module is the Rust side of that recomputation.
//!
//! * The generator is torch's CPU `mt19937`: `manual_seed(s)` is the reference `init_genrand(s)`,
//!   and a float32 `uniform_(from, to)` consumes one 32-bit draw per element, takes its low 24
//!   bits, scales by `2^-24` and maps that onto `[from, to)` in f32
//!   (`at::uniform_real_distribution<float>`, serial on CPU).
//! * `kaiming_uniform_(a=sqrt(5))` with `fan_in = in` bounds that draw by
//!   `sqrt(3) * gain / sqrt(in)`, `gain = sqrt(2 / (1 + a^2))`, computed in f64 as Python does
//!   and handed to `uniform_` as `(-bound, bound)`.
//!
//! `tests/lora_init.rs` holds both to `tests/fixtures/lora-plan-oracle.json`, which
//! `tools/qd_train_oracle_lora_plan.py` wrote by running `attach_lora` itself.
//!
//! The digest is `run_control._sidecar_digest` over one float32 `TensorRef` per adapter named
//! `<module>.lora_A.weight`, the construction the span head's content digest already uses.

use crate::run_control::{RunControlError, sidecar_digest, tensor_ref_digest};

const N: usize = 624;
const M: usize = 397;

/// torch's `at::mt19937` (the reference `genrand_int32`).
struct Mt19937 {
    state: [u32; N],
    next: usize,
}

impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, next: N }
    }

    fn twist(&mut self) {
        for i in 0..N {
            let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % N] & 0x7fff_ffff);
            let mut v = self.state[(i + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                v ^= 0x9908_b0df;
            }
            self.state[i] = v;
        }
        self.next = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.next >= N {
            self.twist();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `uniform_real_distribution<float>(from, to)`: 24 mantissa bits of one 32-bit draw.
    fn uniform_f32(&mut self, from: f32, to: f32) -> f32 {
        let x = (self.next_u32() & 0x00ff_ffff) as f32 * (1.0 / 16_777_216.0f32);
        x * (to - from) + from
    }
}

/// Why the rule could not be applied or did not reproduce the claimed digest.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LoraInitError {
    #[error(
        "seed {0} does not fit torch's 32-bit mt19937 seeding; torch would truncate it silently, \
         so it is refused"
    )]
    SeedTooLarge(u64),
    #[error("no adapted module: an adapter over nothing has no A to recompute")]
    NoModules,
    #[error("{module}: rank {rank} and in_features {in_features} must both be positive")]
    BadShape {
        module: String,
        rank: usize,
        in_features: usize,
    },
    #[error("{0} is listed twice; attach_lora wraps each module once")]
    DuplicateModule(String),
    #[error(transparent)]
    Digest(#[from] RunControlError),
    #[error(
        "the LoRA-A digest {claimed} is not what attach_lora's rule gives at seed {seed} over \
         these {modules} module(s): {recomputed}. A mismatch voids the row (campaign/\
         lora-2b-preregistered.json init.lora_a)"
    )]
    Mismatch {
        seed: u64,
        modules: usize,
        claimed: String,
        recomputed: String,
    },
}

/// `attach_lora`'s A matrices at `seed`: `(module name, [r, in] row-major values)` in the order
/// given, which must be `model.named_modules()` order. B is zero and is not drawn.
pub fn recompute_lora_a(
    seed: u64,
    rank: usize,
    modules: &[(String, usize)],
) -> Result<Vec<(String, Vec<f32>)>, LoraInitError> {
    let seed32 = u32::try_from(seed).map_err(|_| LoraInitError::SeedTooLarge(seed))?;
    if modules.is_empty() {
        return Err(LoraInitError::NoModules);
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut rng = Mt19937::new(seed32);
    let mut out = Vec::with_capacity(modules.len());
    for (name, in_features) in modules {
        if rank == 0 || *in_features == 0 {
            return Err(LoraInitError::BadShape {
                module: name.clone(),
                rank,
                in_features: *in_features,
            });
        }
        if !seen.insert(name.as_str()) {
            return Err(LoraInitError::DuplicateModule(name.clone()));
        }
        // nn.init._calculate_gain / kaiming_uniform_, in f64 as Python evaluates them.
        let a = 5f64.sqrt();
        let gain = (2.0 / (1.0 + a * a)).sqrt();
        let std = gain / (*in_features as f64).sqrt();
        let bound = 3f64.sqrt() * std;
        let (from, to) = (-bound as f32, bound as f32);
        let n = rank * in_features;
        out.push((
            name.clone(),
            (0..n).map(|_| rng.uniform_f32(from, to)).collect(),
        ));
    }
    Ok(out)
}

/// The LoRA-A digest of `(module name, [r, in] values)` tensors: `_sidecar_digest` over float32
/// `TensorRef`s named `<module>.lora_A.weight`. `in_features` is each tensor's length over
/// `rank`.
pub fn lora_a_digest(
    rank: usize,
    tensors: &[(String, Vec<f32>)],
) -> Result<String, LoraInitError> {
    let mut refs = Vec::with_capacity(tensors.len());
    for (name, values) in tensors {
        if rank == 0 || values.is_empty() || values.len() % rank != 0 {
            return Err(LoraInitError::BadShape {
                module: name.clone(),
                rank,
                in_features: values.len().checked_div(rank).unwrap_or(0),
            });
        }
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let digest = tensor_ref_digest("float32", &[rank, values.len() / rank], 4, &bytes)?;
        refs.push((format!("{name}.lora_A.weight"), digest));
    }
    Ok(sidecar_digest(&refs)?)
}

/// Recompute the digest from `seed` by the rule and refuse unless it is `claimed`. Returns the
/// recomputed digest. This is the check; a digest only copied from a checkpoint never passes
/// through it.
pub fn verify_lora_a_digest(
    seed: u64,
    rank: usize,
    modules: &[(String, usize)],
    claimed: &str,
) -> Result<String, LoraInitError> {
    let recomputed = lora_a_digest(rank, &recompute_lora_a(seed, rank, modules)?)?;
    if recomputed != claimed {
        return Err(LoraInitError::Mismatch {
            seed,
            modules: modules.len(),
            claimed: claimed.to_owned(),
            recomputed,
        });
    }
    Ok(recomputed)
}

/// The D17 init digest: `TensorRef.digest` of a float32 zero tensor `[rows, hidden]`
/// (`campaign/lora-2b-preregistered.json` `init.d17`).
pub fn d17_zero_init_digest(rows: usize, hidden: usize) -> Result<String, LoraInitError> {
    let n = rows
        .checked_mul(hidden)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| RunControlError::Invalid(format!("[{rows}, {hidden}] overflows")))?;
    Ok(tensor_ref_digest("float32", &[rows, hidden], 4, &vec![0u8; n])?)
}
