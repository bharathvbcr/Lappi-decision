//! N towers behind one [`DecisionBackend`]: the same request through each, the per-row letter
//! log-probabilities averaged.
//!
//! # The decode
//!
//! For every slot query, each member returns its `rows` logits. Each member's are validated
//! ([`validate_logits`]) and turned into log-probabilities over those rows (a log-softmax); the
//! ensemble returns the **arithmetic mean** of the members' log-probabilities, row by row, in
//! member order, computed in `f64`. The runtime then calibrates and applies its abstention rule
//! to that mean exactly as it does to one tower's logits ([`crate::calibration::calibrate`]:
//! temperature, softmax, margin, conformal set, `noul`). Softmax over the rows of the mean
//! log-probabilities is the normalised geometric mean of the members' row distributions, and it
//! equals softmax of the mean raw logits, because each member's log-normaliser is one constant
//! across its rows, and a temperature divides that constant too (`tests/ensemble_backend.rs`,
//! `the_mean_log_softmax_and_the_mean_logit_give_one_distribution`).
//! The string a manifest records for this is [`crate::release::ENSEMBLE_DECODE`].
//!
//! # Fail closed
//!
//! * fewer than 2 or more than [`MAX_MEMBERS`] members, members that disagree on tokenizer,
//!   letter ids (`label_set_hash`) or declared calibration table, or two members that are one
//!   tower: refused at construction;
//! * any member's error fails the request, carrying which member when the error has a detail;
//!   a member's invalid logits fail it as [`BackendError::DecodeFailed`] naming the member. The
//!   mean of N-1 members is never returned.
//!
//! # State
//!
//! The ensemble's state buffer frames every member's (backend, prompt digest, token count,
//! state) in member order. The runtime hashes it either side of a read-only decode, so a member
//! that wrote its state moves the ensemble's hash and the slot-isolation check catches it.
//! Members are asked in turn, not in parallel: N calls, N bounded by [`MAX_MEMBERS`].

use std::sync::Arc;

use crate::backend::{
    BackendIdentity, DecisionBackend, DecodeMode, Logits, PrefillHandle, SlotQuery, StateBuffer,
    StateSnapshot, validate_logits,
};
use crate::refusal::BackendError;

/// Most members an ensemble may hold. Three is the release candidate; the bound is on device
/// memory and latency, which grow linearly in N.
pub const MAX_MEMBERS: usize = 8;

const STATE_MAGIC: &[u8; 4] = b"QDE1";

/// SHA-256 over `parts` under a domain, so an ensemble's hash cannot equal a single tower's.
fn combined_hash(domain: &str, parts: &[&str]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"qd-ensemble.");
    h.update(domain.as_bytes());
    h.update([0u8]);
    for part in parts {
        h.update(part.as_bytes());
        h.update([0u8]);
    }
    crate::hex(&h.finalize())
}

/// The ensemble's weight hash, from its members' in member order. What `expect.weight_hash`
/// pins for an ensemble, and what `ensemble_manifest.json` records.
pub fn ensemble_weight_hash(member_weight_hashes: &[&str]) -> String {
    combined_hash("weights.v1", member_weight_hashes)
}

/// A member that says its state is host-visible must hand some over. Framed into the ensemble's
/// state, an empty part is still hashed either side of a decode (the framing is never empty), so
/// the runtime would record a slot-isolation check that ran for a member it compared nothing of.
fn require_member_state(
    i: usize,
    member: &dyn DecisionBackend,
    state: &StateBuffer,
) -> Result<(), BackendError> {
    let id = member.identity();
    if id.state_host_visible && state.is_empty() {
        return Err(BackendError::PrefillFailed {
            detail: format!(
                "ensemble member {i} (`{}`) claims host-visible state but handed none, so the \
                 slot-isolation check would compare nothing of it",
                id.name
            ),
        });
    }
    Ok(())
}

/// One member's part of the framed state.
struct Part {
    backend: String,
    prompt_digest: [u8; 32],
    token_count: u64,
    state: Vec<u8>,
}

fn encode_parts(parts: &[Part]) -> Result<Vec<u8>, String> {
    let n = u32::try_from(parts.len()).map_err(|e| format!("member count: {e}"))?;
    let mut out = Vec::new();
    out.extend_from_slice(STATE_MAGIC);
    out.extend_from_slice(&n.to_le_bytes());
    for part in parts {
        let name_len =
            u32::try_from(part.backend.len()).map_err(|e| format!("backend name length: {e}"))?;
        let state_len =
            u64::try_from(part.state.len()).map_err(|e| format!("state length: {e}"))?;
        out.extend_from_slice(&name_len.to_le_bytes());
        out.extend_from_slice(part.backend.as_bytes());
        out.extend_from_slice(&part.prompt_digest);
        out.extend_from_slice(&part.token_count.to_le_bytes());
        out.extend_from_slice(&state_len.to_le_bytes());
        out.extend_from_slice(&part.state);
    }
    Ok(out)
}

/// A strict reader over the framed state: every length is bounded by what is left, and a
/// trailing byte is refused.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], String> {
        if n > self.bytes.len() {
            return Err(format!(
                "the ensemble state ends inside {what}: {n} bytes wanted, {} left",
                self.bytes.len()
            ));
        }
        let (head, rest) = self.bytes.split_at(n);
        self.bytes = rest;
        Ok(head)
    }

    fn array<const N: usize>(&mut self, what: &str) -> Result<[u8; N], String> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N, what)?);
        Ok(out)
    }

    fn len(&mut self, what: &str) -> Result<usize, String> {
        let n = u64::from_le_bytes(self.array::<8>(what)?);
        usize::try_from(n).map_err(|e| format!("{what} {n}: {e}"))
    }
}

fn decode_parts(bytes: &[u8], members: usize) -> Result<Vec<Part>, String> {
    let mut r = Reader { bytes };
    if r.take(4, "the magic")? != STATE_MAGIC {
        return Err("the state is not an ensemble state (bad magic)".to_string());
    }
    let n = u32::from_le_bytes(r.array::<4>("the member count")?);
    if usize::try_from(n).ok() != Some(members) {
        return Err(format!(
            "the state frames {n} members; this ensemble has {members}"
        ));
    }
    let mut parts = Vec::with_capacity(members);
    for i in 0..members {
        let name_len = u32::from_le_bytes(r.array::<4>("a name length")?);
        let name_len = usize::try_from(name_len).map_err(|e| format!("name length: {e}"))?;
        let backend = std::str::from_utf8(r.take(name_len, "a backend name")?)
            .map_err(|e| format!("member {i}'s backend name: {e}"))?
            .to_string();
        let prompt_digest = r.array::<32>("a prompt digest")?;
        let token_count = u64::from_le_bytes(r.array::<8>("a token count")?);
        let state_len = r.len("a state length")?;
        let state = r.take(state_len, "a member state")?.to_vec();
        parts.push(Part {
            backend,
            prompt_digest,
            token_count,
            state,
        });
    }
    if !r.bytes.is_empty() {
        return Err(format!(
            "{} bytes follow the last member's state",
            r.bytes.len()
        ));
    }
    Ok(parts)
}

fn token_count(tokens: u64, i: usize) -> Result<usize, BackendError> {
    usize::try_from(tokens).map_err(|e| BackendError::DecodeFailed {
        detail: format!("ensemble member {i}'s token count: {e}"),
    })
}

/// Name the member in an error that carries a detail. Errors without one (a deadline, an
/// overload, a logit fault already naming its slot) pass through unchanged: their kind is what
/// a caller acts on.
fn from_member(i: usize, name: &str, error: BackendError) -> BackendError {
    let tag = |detail: String| format!("ensemble member {i} (`{name}`): {detail}");
    match error {
        BackendError::Unavailable { detail } => BackendError::Unavailable {
            detail: tag(detail),
        },
        BackendError::Poisoned { reason } => BackendError::Poisoned {
            reason: tag(reason),
        },
        BackendError::RebuildFailed { detail } => BackendError::RebuildFailed {
            detail: tag(detail),
        },
        BackendError::PrefillFailed { detail } => BackendError::PrefillFailed {
            detail: tag(detail),
        },
        BackendError::DecodeFailed { detail } => BackendError::DecodeFailed {
            detail: tag(detail),
        },
        other @ (BackendError::ReferenceBackendNotEnabled { .. }
        | BackendError::ReadonlyViolated { .. }
        | BackendError::LogitShapeMismatch { .. }
        | BackendError::NonFiniteLogit { .. }
        | BackendError::LogitKindMismatch { .. }
        | BackendError::DeadlineExceeded { .. }
        | BackendError::Overloaded { .. }) => other,
    }
}

/// Log-softmax over one member's rows, in `f64`. The rows were validated finite and non-empty.
fn log_softmax(values: &[f32]) -> Vec<f64> {
    let max = values
        .iter()
        .map(|v| f64::from(*v))
        .fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = values.iter().map(|v| (f64::from(*v) - max).exp()).sum();
    let log_norm = max + sum.ln();
    values.iter().map(|v| f64::from(*v) - log_norm).collect()
}

/// N members behind one backend.
pub struct EnsembleBackend {
    members: Vec<Arc<dyn DecisionBackend>>,
    identity: BackendIdentity,
}

impl EnsembleBackend {
    /// Build from members already loaded. Refuses anything it could not average honestly.
    pub fn new(members: Vec<Arc<dyn DecisionBackend>>) -> Result<Self, BackendError> {
        if !(2..=MAX_MEMBERS).contains(&members.len()) {
            return Err(BackendError::Unavailable {
                detail: format!(
                    "an ensemble has 2 to {MAX_MEMBERS} members, not {}",
                    members.len()
                ),
            });
        }
        let first = members[0].identity();
        for (i, member) in members.iter().enumerate().skip(1) {
            let id = member.identity();
            for (what, mine, theirs) in [
                ("tokenizer_hash", &id.tokenizer_hash, &first.tokenizer_hash),
                ("label_set_hash", &id.label_set_hash, &first.label_set_hash),
                (
                    "calibration_hash",
                    &id.calibration_hash,
                    &first.calibration_hash,
                ),
            ] {
                if mine != theirs {
                    return Err(BackendError::Unavailable {
                        detail: format!(
                            "ensemble member {i} (`{}`) reports {what} {mine}; member 0 (`{}`) \
                             reports {theirs}. Members that disagree on the tokenizer, the \
                             letter ids or the table do not score the same rows",
                            id.name, first.name
                        ),
                    });
                }
            }
            if let Some(j) = members[..i]
                .iter()
                .position(|other| other.identity().weight_hash == id.weight_hash)
            {
                return Err(BackendError::Unavailable {
                    detail: format!(
                        "ensemble members {j} and {i} report one weight_hash ({}); one tower \
                         averaged twice is not two",
                        id.weight_hash
                    ),
                });
            }
        }
        let weights: Vec<&str> = members
            .iter()
            .map(|m| m.identity().weight_hash.as_str())
            .collect();
        let heads: Vec<&str> = members
            .iter()
            .map(|m| m.identity().head_hash.as_str())
            .collect();
        let names: Vec<&str> = members.iter().map(|m| m.identity().name.as_str()).collect();
        let identity = BackendIdentity {
            name: format!("ensemble/{}[{}]", members.len(), names.join(",")),
            tokenizer_hash: first.tokenizer_hash.clone(),
            weight_hash: ensemble_weight_hash(&weights),
            head_hash: combined_hash("head.v1", &heads),
            label_set_hash: first.label_set_hash.clone(),
            calibration_hash: first.calibration_hash.clone(),
            is_model: members.iter().all(|m| m.identity().is_model),
            state_host_visible: members.iter().all(|m| m.identity().state_host_visible),
        };
        Ok(Self { members, identity })
    }

    fn parts(&self, state: &StateBuffer) -> Result<Vec<Part>, String> {
        decode_parts(state.as_bytes(), self.members.len())
    }
}

impl DecisionBackend for EnsembleBackend {
    fn identity(&self) -> &BackendIdentity {
        &self.identity
    }

    fn prefill(&self, prefix: &str) -> Result<PrefillHandle, BackendError> {
        let mut parts = Vec::with_capacity(self.members.len());
        let mut tokens: Option<usize> = None;
        for (i, member) in self.members.iter().enumerate() {
            let name = member.identity().name.as_str();
            let handle = member
                .prefill(prefix)
                .map_err(|e| from_member(i, name, e))?;
            require_member_state(i, member.as_ref(), &handle.state)?;
            match tokens {
                None => tokens = Some(handle.token_count),
                Some(t) if t == handle.token_count => {}
                Some(t) => {
                    return Err(BackendError::PrefillFailed {
                        detail: format!(
                            "ensemble member {i} (`{name}`) read the prefix as {} tokens, \
                             member 0 as {t}; one tokenizer cannot do that",
                            handle.token_count
                        ),
                    });
                }
            }
            parts.push(Part {
                backend: handle.backend,
                prompt_digest: handle.prompt_digest,
                token_count: u64::try_from(handle.token_count).map_err(|e| {
                    BackendError::PrefillFailed {
                        detail: format!("member {i}'s token count: {e}"),
                    }
                })?,
                state: handle.state.as_bytes().to_vec(),
            });
        }
        let state =
            encode_parts(&parts).map_err(|detail| BackendError::PrefillFailed { detail })?;
        Ok(PrefillHandle {
            backend: self.identity.name.clone(),
            prompt_digest: crate::sha256(prefix.as_bytes()),
            token_count: tokens.unwrap_or(0),
            state: StateBuffer::from_bytes(state),
        })
    }

    fn snapshot(&self, handle: &PrefillHandle) -> Result<StateSnapshot, BackendError> {
        let parts = self
            .parts(&handle.state)
            .map_err(|detail| BackendError::PrefillFailed { detail })?;
        let mut out = Vec::with_capacity(parts.len());
        for (i, (member, part)) in self.members.iter().zip(parts).enumerate() {
            let name = member.identity().name.as_str();
            let member_handle = PrefillHandle {
                backend: part.backend,
                prompt_digest: part.prompt_digest,
                token_count: token_count(part.token_count, i)?,
                state: StateBuffer::from_bytes(part.state),
            };
            let snap = member
                .snapshot(&member_handle)
                .map_err(|e| from_member(i, name, e))?;
            require_member_state(i, member.as_ref(), &snap.state)?;
            out.push(Part {
                backend: snap.backend,
                prompt_digest: snap.prompt_digest,
                token_count: part.token_count,
                state: snap.state.as_bytes().to_vec(),
            });
        }
        let state = encode_parts(&out).map_err(|detail| BackendError::PrefillFailed { detail })?;
        Ok(StateSnapshot {
            backend: self.identity.name.clone(),
            prompt_digest: handle.prompt_digest,
            token_count: handle.token_count,
            state: StateBuffer::from_bytes(state),
        })
    }

    fn decode_slot(
        &self,
        snapshot: &mut StateSnapshot,
        query: &SlotQuery<'_>,
        mode: DecodeMode,
    ) -> Result<Logits, BackendError> {
        let mut parts = self
            .parts(&snapshot.state)
            .map_err(|detail| BackendError::DecodeFailed { detail })?;
        let mut sum = vec![0f64; query.rows];
        for (i, (member, part)) in self.members.iter().zip(parts.iter_mut()).enumerate() {
            let name = member.identity().name.as_str();
            let mut member_snapshot = StateSnapshot {
                backend: part.backend.clone(),
                prompt_digest: part.prompt_digest,
                token_count: token_count(part.token_count, i)?,
                state: StateBuffer::from_bytes(std::mem::take(&mut part.state)),
            };
            let logits = member
                .decode_slot(&mut member_snapshot, query, mode)
                .map_err(|e| from_member(i, name, e))?;
            validate_logits(&logits, query).map_err(|e| BackendError::DecodeFailed {
                detail: format!("ensemble member {i} (`{name}`): {e}"),
            })?;
            for (acc, logp) in sum.iter_mut().zip(log_softmax(&logits.values)) {
                *acc += logp;
            }
            part.state = member_snapshot.state.as_bytes().to_vec();
        }
        snapshot.state = StateBuffer::from_bytes(
            encode_parts(&parts).map_err(|detail| BackendError::DecodeFailed { detail })?,
        );
        let n = f64::from(u32::try_from(self.members.len()).map_err(|e| {
            BackendError::DecodeFailed {
                detail: format!("member count: {e}"),
            }
        })?);
        Ok(Logits {
            kind: query.kind,
            // The trait carries f32. The mean is formed in f64 and narrowed once, here.
            values: sum.iter().map(|s| (s / n) as f32).collect(),
        })
    }

    fn pooled_features(&self, _snapshot: &StateSnapshot) -> Result<Vec<f32>, BackendError> {
        Err(BackendError::Unavailable {
            detail: "pooled features are not defined for an ensemble: a registered head reads \
                     one tower's features, and which tower's, or what mean of them, was never \
                     decided"
                .to_string(),
        })
    }
}
