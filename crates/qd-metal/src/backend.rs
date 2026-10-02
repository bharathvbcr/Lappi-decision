//! [`MetalBackend`]: the [`DecisionBackend`] over [`crate::model::Model`].
//!
//! # One GPU-owning thread, a bounded queue
//!
//! tessl's `GpuRuntime` is deliberately `!Send` (Metal encoders are thread-affine; see the
//! `compile_fail` doctest in `tessl/src/runtime.rs`), while `DecisionBackend: Send + Sync`. The
//! only way to meet both without an `unsafe impl` is the shape plan Phase 4 asks for anyway: one
//! worker thread creates the runtime, loads the weights and owns every GPU object; the backend
//! handle holds a **bounded** `sync_channel` to it. A full queue is
//! [`BackendError::Overloaded`], never an unbounded wait; a job that outlives the deadline is
//! [`BackendError::DeadlineExceeded`].
//!
//! # The state buffer the runtime hashes
//!
//! `qd_runtime::answer` hashes `StateSnapshot::state` before and after every read-only decode.
//! The real state is in device memory (tens to hundreds of MB), so the host mirror is a 52-byte
//! record: magic, entry id, token count, and the SHA-256 of the state **as read back from the
//! device** ([`crate::model::PrefixState::digest`]). [`MetalBackend::decode_slot`] recomputes that
//! digest from device memory after every decode and writes it into the snapshot, so a decode
//! that wrote any byte of the state moves the runtime's hash. That is why
//! `state_host_visible` is `true`: the check the runtime runs is a check of the device state,
//! not of a host copy that could disagree with it.
//!
//! # Snapshots and the prefix cache
//!
//! A prefill is cached by `prompt_digest` (SHA-256 of the prefix): a second request over the
//! same prefix skips the prefill (plan Lane B: *"the biggest win is not a kernel — it is prefix
//! reuse"*). A snapshot shares the prefill's device state; a [`DecodeMode::WriteBack`] decode
//! replaces the snapshot's state with the continuation's end state (new buffers), so it never
//! changes the prefill or another snapshot. The trait has no release call, so entries are held
//! in a bounded LRU; decoding an evicted snapshot is a [`BackendError::DecodeFailed`] that says
//! so — never a silent re-prefill.

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::thread::JoinHandle;
use std::time::Duration;

use qd_runtime::backend::{
    BackendIdentity, DecisionBackend, DecodeMode, Logits, PrefillHandle, QueryKind, SlotQuery,
    StateBuffer, StateSnapshot,
};
use qd_runtime::BackendError;

use crate::error::{MetalError, Result};
use crate::model::{Model, PrefixState};
use crate::tokenizer::QwenTokenizer;

/// The name in every answer this backend produces.
pub const BACKEND_NAME: &str = "qd-metal/qwen3.5-2b-base/tessl";

const STATE_MAGIC: &[u8; 4] = b"QDM1";
/// Bytes of the host-side state record.
pub const STATE_RECORD_BYTES: usize = 4 + 8 + 8 + 32;

/// How the backend is built. Every field is stated by the caller; nothing is defaulted silently.
#[derive(Debug, Clone)]
pub struct MetalConfig {
    /// The Qwen3.5-2B-Base snapshot directory; `None` resolves the one in the HF cache.
    pub snapshot: Option<PathBuf>,
    /// Hash of the calibration table the runtime is built with. The backend reports it; it does
    /// not own the table.
    pub calibration_hash: String,
    /// Jobs that may wait for the GPU thread. Beyond this, requests are refused `Overloaded`.
    pub queue_capacity: usize,
    /// Prefills and snapshots held at once (LRU beyond this).
    pub max_entries: usize,
    /// Longest prefix + continuation, in tokens. Bounds every state and workspace.
    pub max_tokens: usize,
    /// How long a caller waits for one job.
    pub job_timeout: Duration,
}

/// The 52-byte host record for a device state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateRecord {
    pub entry: u64,
    pub tokens: u64,
    pub digest: [u8; 32],
}

impl StateRecord {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(STATE_RECORD_BYTES);
        b.extend_from_slice(STATE_MAGIC);
        b.extend_from_slice(&self.entry.to_le_bytes());
        b.extend_from_slice(&self.tokens.to_le_bytes());
        b.extend_from_slice(&self.digest);
        b
    }

    pub fn parse(bytes: &[u8]) -> std::result::Result<Self, String> {
        if bytes.len() != STATE_RECORD_BYTES {
            return Err(format!(
                "state record is {} bytes, not {STATE_RECORD_BYTES}; it was not produced by {BACKEND_NAME}",
                bytes.len()
            ));
        }
        if &bytes[..4] != STATE_MAGIC {
            return Err(format!("state record magic is not {STATE_MAGIC:?}"));
        }
        let u64_at = |at: usize| {
            let mut w = [0u8; 8];
            w.copy_from_slice(&bytes[at..at + 8]);
            u64::from_le_bytes(w)
        };
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&bytes[20..52]);
        Ok(Self {
            entry: u64_at(4),
            tokens: u64_at(12),
            digest,
        })
    }
}

enum Job {
    Prefill {
        prefix: String,
        reply: SyncSender<std::result::Result<PrefillHandle, BackendError>>,
    },
    Snapshot {
        handle: PrefillHandle,
        reply: SyncSender<std::result::Result<StateSnapshot, BackendError>>,
    },
    Decode {
        snapshot: StateSnapshot,
        suffix: String,
        slot: String,
        rows: usize,
        kind: QueryKind,
        mode: DecodeMode,
        reply: SyncSender<std::result::Result<(Logits, StateSnapshot), BackendError>>,
    },
}

pub struct MetalBackend {
    identity: BackendIdentity,
    /// `None` only inside `drop`, which closes the channel before joining the worker.
    tx: Option<SyncSender<Job>>,
    queue_capacity: usize,
    job_timeout: Duration,
    worker: Option<JoinHandle<()>>,
}

impl MetalBackend {
    /// Start the GPU thread, load the model and tokenizer on it, and wait until it is ready.
    pub fn start(config: MetalConfig) -> Result<Self> {
        if config.queue_capacity == 0 || config.max_entries == 0 || config.max_tokens == 0 {
            return Err(MetalError::Config(
                "queue_capacity, max_entries and max_tokens must all be non-zero".into(),
            ));
        }
        if config.calibration_hash.trim().is_empty() {
            return Err(MetalError::Config("calibration_hash is empty".into()));
        }
        let snapshot = crate::config::resolve_snapshot(config.snapshot.as_deref())?;
        let (tx, rx) = mpsc::sync_channel::<Job>(config.queue_capacity);
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<BackendIdentity>>(1);
        let worker_cfg = config.clone();
        let handle = std::thread::Builder::new()
            .name("qd-metal-gpu".into())
            .spawn(move || {
                let worker = Worker::new(&snapshot, &worker_cfg);
                match worker {
                    Ok(w) => {
                        let id = w.identity(&worker_cfg.calibration_hash);
                        if ready_tx.send(Ok(id)).is_ok() {
                            w.serve(rx);
                        }
                    }
                    Err(e) => {
                        // The starter is waiting on this; if it is gone there is no one to tell.
                        let _gone = ready_tx.send(Err(e));
                    }
                }
            })
            .map_err(|e| MetalError::Gpu(format!("spawn GPU thread: {e}")))?;
        let identity = ready_rx
            .recv()
            .map_err(|_| MetalError::Gpu("the GPU thread exited before reporting ready".into()))??;
        Ok(Self {
            identity,
            tx: Some(tx),
            queue_capacity: config.queue_capacity,
            job_timeout: config.job_timeout,
            worker: Some(handle),
        })
    }

    fn submit<T>(
        &self,
        make: impl FnOnce(SyncSender<std::result::Result<T, BackendError>>) -> Job,
    ) -> std::result::Result<T, BackendError> {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        let tx = self.tx.as_ref().ok_or_else(|| BackendError::Unavailable {
            detail: "the qd-metal backend is shutting down".into(),
        })?;
        match tx.try_send(make(reply_tx)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                return Err(BackendError::Overloaded {
                    limit: self.queue_capacity,
                })
            }
            Err(TrySendError::Disconnected(_)) => {
                return Err(BackendError::Unavailable {
                    detail: "the qd-metal GPU thread has exited".into(),
                })
            }
        }
        match reply_rx.recv_timeout(self.job_timeout) {
            Ok(r) => r,
            Err(RecvTimeoutError::Timeout) => Err(BackendError::DeadlineExceeded {
                limit_ms: u64::try_from(self.job_timeout.as_millis()).unwrap_or(u64::MAX),
            }),
            Err(RecvTimeoutError::Disconnected) => Err(BackendError::Unavailable {
                detail: "the qd-metal GPU thread dropped the job without answering".into(),
            }),
        }
    }
}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        // Close the channel first: the worker finishes what is queued, then `recv` fails and it
        // exits. (Sending a shutdown message instead could fail on a full queue and leave the
        // join waiting on a worker blocked in `recv` forever.)
        drop(self.tx.take());
        if let Some(h) = self.worker.take() {
            // A panicked worker has already failed every job it held; nothing to add.
            let _joined = h.join();
        }
    }
}

impl DecisionBackend for MetalBackend {
    fn identity(&self) -> &BackendIdentity {
        &self.identity
    }

    fn prefill(&self, prefix: &str) -> std::result::Result<PrefillHandle, BackendError> {
        let prefix = prefix.to_string();
        self.submit(|reply| Job::Prefill { prefix, reply })
    }

    fn snapshot(&self, handle: &PrefillHandle) -> std::result::Result<StateSnapshot, BackendError> {
        let handle = handle.clone();
        self.submit(|reply| Job::Snapshot { handle, reply })
    }

    fn decode_slot(
        &self,
        snapshot: &mut StateSnapshot,
        query: &SlotQuery<'_>,
        mode: DecodeMode,
    ) -> std::result::Result<Logits, BackendError> {
        let job_snapshot = snapshot.clone();
        let suffix = query.suffix.to_string();
        let slot = query.slot_name.to_string();
        let (rows, kind) = (query.rows, query.kind);
        let (logits, updated) = self.submit(|reply| Job::Decode {
            snapshot: job_snapshot,
            suffix,
            slot,
            rows,
            kind,
            mode,
            reply,
        })?;
        // The worker re-read the device state after the decode; the runtime hashes this.
        *snapshot = updated;
        Ok(logits)
    }

    fn pooled_features(&self, _snapshot: &StateSnapshot) -> std::result::Result<Vec<f32>, BackendError> {
        Err(BackendError::Unavailable {
            detail: "pooled features are not defined for the base weights: docs/schema-api.md \
                     names a 'single GEMV on pooled features' but not the pooling, and a head \
                     fitted against an invented one would be bound to a guess. The registered \
                     route is not served by qd-metal until that is decided."
                .into(),
        })
    }
}

// ------------------------------------------------------------------ worker ---

struct Entry {
    state: Rc<PrefixState>,
    /// The text `state` covers: the prefill's prefix, extended by any write-back.
    prefix: Rc<str>,
    /// `prefix`'s tokens; always `state.tokens()` long.
    prefix_ids: Rc<[u32]>,
    /// Set for a cached prefill, which decodes never touch (they go through a snapshot).
    is_prefill: bool,
    /// The original prefill's `prompt_digest`, which every handle and snapshot carries.
    prompt_digest: [u8; 32],
    digest: [u8; 32],
    last_used: u64,
}

struct Worker {
    model: Model,
    tok: QwenTokenizer,
    entries: HashMap<u64, Entry>,
    /// prompt_digest -> the prefill entry for it (never written back to).
    prefills: HashMap<[u8; 32], u64>,
    next_id: u64,
    clock: u64,
    max_entries: usize,
    max_tokens: usize,
    poisoned: Option<String>,
}

fn prefill_failed(e: MetalError) -> BackendError {
    BackendError::PrefillFailed { detail: e.to_string() }
}

fn decode_failed(e: impl std::fmt::Display) -> BackendError {
    BackendError::DecodeFailed { detail: e.to_string() }
}

impl Worker {
    fn new(snapshot: &std::path::Path, cfg: &MetalConfig) -> Result<Self> {
        let rt = tessl::GpuRuntime::new().map_err(MetalError::Gpu)?;
        rt.set_async_encode(true).map_err(MetalError::Gpu)?;
        let model = Model::load(&rt, snapshot)?;
        let tok = QwenTokenizer::load(&snapshot.join("tokenizer.json"))?;
        Ok(Self {
            model,
            tok,
            entries: HashMap::new(),
            prefills: HashMap::new(),
            next_id: 1,
            clock: 0,
            max_entries: cfg.max_entries,
            max_tokens: cfg.max_tokens,
            poisoned: None,
        })
    }

    fn identity(&self, calibration_hash: &str) -> BackendIdentity {
        let letters: Vec<u8> = self
            .tok
            .letter_ids()
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect();
        let mut head = b"qd-metal.head.tied-embedding-letter-rows.v1\0".to_vec();
        head.extend_from_slice(&self.model.embed_digest());
        head.extend_from_slice(&letters);
        let mut labels = b"qd-metal.label-set.v1\0ABCDEFGHIJKLMNOPZ\0".to_vec();
        labels.extend_from_slice(&letters);
        BackendIdentity {
            name: BACKEND_NAME.to_string(),
            tokenizer_hash: self.tok.hash().to_string(),
            weight_hash: self.model.weight_hash().to_string(),
            head_hash: qd_runtime::hex(&qd_runtime::sha256(&head)),
            label_set_hash: qd_runtime::hex(&qd_runtime::sha256(&labels)),
            calibration_hash: calibration_hash.to_string(),
            is_model: true,
            state_host_visible: true,
        }
    }

    fn serve(mut self, rx: Receiver<Job>) {
        // Ends when the backend drops its sender and the queue is drained.
        while let Ok(job) = rx.recv() {
            // A caller that timed out has dropped its receiver; a failed send has no one to tell.
            match job {
                Job::Prefill { prefix, reply } => {
                    let r = self.guard().and_then(|()| self.prefill(&prefix));
                    let _gone = reply.send(r);
                }
                Job::Snapshot { handle, reply } => {
                    let r = self.guard().and_then(|()| self.snapshot(&handle));
                    let _gone = reply.send(r);
                }
                Job::Decode {
                    snapshot,
                    suffix,
                    slot,
                    rows,
                    kind,
                    mode,
                    reply,
                } => {
                    let r = self
                        .guard()
                        .and_then(|()| self.decode(snapshot, &suffix, &slot, rows, kind, mode));
                    let _gone = reply.send(r);
                }
            }
        }
    }

    fn guard(&self) -> std::result::Result<(), BackendError> {
        match &self.poisoned {
            Some(reason) => Err(BackendError::Poisoned {
                reason: reason.clone(),
            }),
            None => Ok(()),
        }
    }

    /// A GPU failure poisons the worker: tessl marks its runtime unusable after an encode or
    /// submit failure, and state written by a half-run command buffer cannot be trusted.
    fn note(&mut self, e: &MetalError) {
        if matches!(e, MetalError::Gpu(_)) {
            self.poisoned = Some(e.to_string());
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn insert(&mut self, entry: Entry) -> u64 {
        while self.entries.len() >= self.max_entries {
            let Some((&oldest, _)) = self.entries.iter().min_by_key(|(_, e)| e.last_used) else {
                break;
            };
            self.entries.remove(&oldest);
            self.prefills.retain(|_, id| *id != oldest);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(id, entry);
        id
    }

    fn record(&self, id: u64) -> std::result::Result<StateRecord, String> {
        let e = self.entries.get(&id).ok_or_else(|| format!("entry {id} is gone"))?;
        Ok(StateRecord {
            entry: id,
            tokens: u64::from(e.state.tokens()),
            digest: e.digest,
        })
    }

    fn prefill(&mut self, prefix: &str) -> std::result::Result<PrefillHandle, BackendError> {
        let prompt_digest = qd_runtime::sha256(prefix.as_bytes());
        let now = self.tick();
        if let Some(&id) = self.prefills.get(&prompt_digest)
            && let Some(e) = self.entries.get_mut(&id)
            && &*e.prefix == prefix
        {
            e.last_used = now;
            let tokens = e.state.tokens() as usize;
            let rec = self.record(id).map_err(|m| BackendError::PrefillFailed { detail: m })?;
            return Ok(PrefillHandle {
                backend: BACKEND_NAME.to_string(),
                prompt_digest,
                token_count: tokens,
                state: StateBuffer::from_bytes(rec.to_bytes()),
            });
        }
        let ids = self.tok.encode(prefix).map_err(prefill_failed)?;
        if ids.is_empty() {
            return Err(prefill_failed(MetalError::Input("the prefix encodes to no tokens".into())));
        }
        if ids.len() > self.max_tokens {
            return Err(prefill_failed(MetalError::Input(format!(
                "the prefix is {} tokens; this backend serves at most {}",
                ids.len(),
                self.max_tokens
            ))));
        }
        let result = self.model.prefill(&ids).and_then(|(_, state)| {
            let digest = state.digest()?;
            Ok((state, digest))
        });
        let (state, digest) = match result {
            Ok(v) => v,
            Err(e) => {
                self.note(&e);
                return Err(prefill_failed(e));
            }
        };
        let tokens = state.tokens() as usize;
        let id = self.insert(Entry {
            state: Rc::new(state),
            prefix: Rc::from(prefix),
            prefix_ids: Rc::from(ids),
            is_prefill: true,
            prompt_digest,
            digest,
            last_used: now,
        });
        self.prefills.insert(prompt_digest, id);
        let rec = self.record(id).map_err(|m| BackendError::PrefillFailed { detail: m })?;
        Ok(PrefillHandle {
            backend: BACKEND_NAME.to_string(),
            prompt_digest,
            token_count: tokens,
            state: StateBuffer::from_bytes(rec.to_bytes()),
        })
    }

    fn lookup(&self, backend: &str, state: &StateBuffer, prompt_digest: &[u8; 32]) -> std::result::Result<u64, String> {
        if backend != BACKEND_NAME {
            return Err(format!("state belongs to backend {backend:?}, not {BACKEND_NAME}"));
        }
        let rec = StateRecord::parse(state.as_bytes())?;
        let e = self.entries.get(&rec.entry).ok_or_else(|| {
            format!(
                "state entry {} is no longer held (evicted after {} newer entries, or from an \
                 earlier process); prefill again",
                rec.entry, self.max_entries
            )
        })?;
        if &e.prompt_digest != prompt_digest || u64::from(e.state.tokens()) != rec.tokens {
            return Err(format!("state entry {} does not match the handle it came with", rec.entry));
        }
        Ok(rec.entry)
    }

    fn snapshot(&mut self, handle: &PrefillHandle) -> std::result::Result<StateSnapshot, BackendError> {
        let id = self
            .lookup(&handle.backend, &handle.state, &handle.prompt_digest)
            .map_err(|detail| BackendError::PrefillFailed { detail })?;
        let now = self.tick();
        let src = &self.entries[&id];
        let entry = Entry {
            state: Rc::clone(&src.state),
            prefix: Rc::clone(&src.prefix),
            prefix_ids: Rc::clone(&src.prefix_ids),
            is_prefill: false,
            prompt_digest: src.prompt_digest,
            digest: src.digest,
            last_used: now,
        };
        let tokens = entry.state.tokens() as usize;
        let new_id = self.insert(entry);
        let rec = self.record(new_id).map_err(|detail| BackendError::PrefillFailed { detail })?;
        Ok(StateSnapshot {
            backend: BACKEND_NAME.to_string(),
            prompt_digest: handle.prompt_digest,
            token_count: tokens,
            state: StateBuffer::from_bytes(rec.to_bytes()),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &mut self,
        mut snapshot: StateSnapshot,
        suffix: &str,
        slot: &str,
        rows: usize,
        kind: QueryKind,
        mode: DecodeMode,
    ) -> std::result::Result<(Logits, StateSnapshot), BackendError> {
        if kind != QueryKind::Letters {
            return Err(decode_failed(format!(
                "slot `{slot}` asks for the {} head; the base weights have no pointer head, only \
                 the tied LM head's letter rows",
                kind.as_str()
            )));
        }
        let id = self
            .lookup(&snapshot.backend, &snapshot.state, &snapshot.prompt_digest)
            .map_err(decode_failed)?;
        let answers = self.tok.answer_ids(rows).map_err(decode_failed)?;
        let now = self.tick();
        let (state, prefix, prefix_ids) = {
            let e = self
                .entries
                .get_mut(&id)
                .ok_or_else(|| decode_failed("snapshot entry vanished"))?;
            if e.is_prefill {
                return Err(decode_failed(
                    "decode was handed a prefill handle's state, not a snapshot's; the cached \
                     prefill is shared by later requests and is never decoded from directly",
                ));
            }
            e.last_used = now;
            (Rc::clone(&e.state), Rc::clone(&e.prefix), Rc::clone(&e.prefix_ids))
        };
        // The continuation's tokens are those of prefix + suffix after the prefix's: the model
        // must see exactly what one pass over the whole prompt would. A tokenization that merges
        // across the boundary is refused, not approximated.
        let mut whole = String::with_capacity(prefix.len() + suffix.len());
        whole.push_str(&prefix);
        whole.push_str(suffix);
        let all = self.tok.encode(&whole).map_err(decode_failed)?;
        if all.len() <= prefix_ids.len() || all[..prefix_ids.len()] != prefix_ids[..] {
            return Err(decode_failed(format!(
                "slot `{slot}`: prefix + suffix does not tokenize as the prefix's tokens followed \
                 by the suffix's ({} vs {} prefix tokens); the continuation would not be the prompt",
                all.len(),
                prefix_ids.len()
            )));
        }
        // `prefix` is the text the state covers (a write-back extends it), so this holds always.
        if state.tokens() as usize != prefix_ids.len() {
            return Err(decode_failed(format!(
                "entry {id}: state holds {} tokens but its text is {}",
                state.tokens(),
                prefix_ids.len()
            )));
        }
        let cont: Vec<u32> = all[prefix_ids.len()..].to_vec();
        if cont.is_empty() {
            return Err(decode_failed(format!("slot `{slot}`: the suffix adds no tokens")));
        }
        if state.tokens() as usize + cont.len() > self.max_tokens {
            return Err(decode_failed(format!(
                "prefix + suffix is {} tokens; this backend serves at most {}",
                state.tokens() as usize + cont.len(),
                self.max_tokens
            )));
        }
        let seq = u32::try_from(cont.len()).map_err(decode_failed)?;
        let keep = mode == DecodeMode::WriteBack;
        let result = (|| -> Result<(Vec<f32>, Option<PrefixState>, [u8; 32])> {
            let mut out = self.model.run(&cont, 1, seq, Some(&state), keep, None)?;
            let scores = self.model.score(&out, &[seq - 1], &answers)?;
            let new_state = out.state.take();
            // The digest is re-read from the device after the decode ran.
            let digest = match &new_state {
                Some(s) => s.digest()?,
                None => state.digest()?,
            };
            Ok((scores.logits, new_state, digest))
        })();
        let (values, new_state, digest) = match result {
            Ok(v) => v,
            Err(e) => {
                self.note(&e);
                return Err(decode_failed(e));
            }
        };
        let e = self
            .entries
            .get_mut(&id)
            .ok_or_else(|| decode_failed("snapshot entry vanished during the decode"))?;
        if let Some(s) = new_state {
            // Write-back: the state now covers prefix + suffix, and so does the entry's text.
            e.state = Rc::new(s);
            e.prefix = Rc::from(whole.as_str());
            e.prefix_ids = Rc::from(all);
        }
        e.digest = digest;
        let rec = StateRecord {
            entry: id,
            tokens: u64::from(e.state.tokens()),
            digest,
        };
        snapshot.token_count = e.state.tokens() as usize;
        snapshot.state = StateBuffer::from_bytes(rec.to_bytes());
        Ok((Logits { kind, values }, snapshot))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_record_round_trips_and_refuses_foreign_bytes() {
        let r = StateRecord {
            entry: 7,
            tokens: 1234,
            digest: [9; 32],
        };
        let b = r.to_bytes();
        assert_eq!(b.len(), STATE_RECORD_BYTES);
        assert_eq!(StateRecord::parse(&b).unwrap(), r);
        assert!(StateRecord::parse(&b[..51]).unwrap_err().contains("52"));
        let mut bad = b.clone();
        bad[0] = b'X';
        assert!(StateRecord::parse(&bad).unwrap_err().contains("magic"));
        // The reference backend's 512-byte state is not one of ours.
        assert!(StateRecord::parse(&[0u8; 512]).is_err());
    }

    #[test]
    fn a_changed_digest_changes_the_bytes_the_runtime_hashes() {
        let a = StateRecord { entry: 1, tokens: 5, digest: [0; 32] };
        let b = StateRecord { digest: [1; 32], ..a };
        assert_ne!(
            StateBuffer::from_bytes(a.to_bytes()).digest(),
            StateBuffer::from_bytes(b.to_bytes()).digest()
        );
    }

    #[test]
    fn the_backend_handle_is_send_and_sync() {
        fn require<T: Send + Sync>() {}
        require::<MetalBackend>();
    }
}
