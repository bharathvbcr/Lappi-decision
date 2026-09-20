//! **The one contract both modes share.**
//!
//! `qd serve` and `qd oneshot` differ only in where the bytes come from. Both call
//! [`Service::handle_line`] with one line of JSON and write back what it returns. Nothing else in
//! either mode looks at a request or builds a response, so "the two modes behave the same" is a
//! property of the call graph rather than a promise in a comment —
//! `tests/two_modes_one_contract.rs` drives the identical bytes through a live Unix socket and
//! through the in-process path and asserts the replies are byte-identical.
//!
//! That matters because the two are used by different callers for different reasons: DevType talks
//! to the socket, and dcverify tries the socket and falls back to the sidecar, because a verify run
//! has to work with nothing else running. If the fallback answered differently from the socket, a
//! verify run would be non-reproducible in a way that depended on whether an unrelated agent
//! happened to be warm.
//!
//! # Lifecycle, and why it is all here
//!
//! [`crate::runtime::Runtime`] is immutable and shared behind an `Arc`. Everything that changes —
//! the warm slot, eviction, poisoning, rebuilds, counters — is in this module behind one mutex, and
//! **that mutex is never held across a decode**. A request locks to take an `Arc` and unlocks
//! before it asks the backend anything, which is what stops one slow request from wedging the
//! agent. `tests/lifecycle.rs` asserts it with a backend that blocks.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::answer::Deadline;
use crate::backend::BackendIdentity;
use crate::refusal::BackendError;
use crate::runtime::{Runtime, RuntimeConfig};
use crate::schema::{DecisionRequest, Response};
use crate::wire::{parse_line, ControlOp, Incoming};

/// How a `Service` obtains a runtime. Injectable so a test can supply a backend that poisons, that
/// blocks, or that writes back under `readonly`.
pub type RuntimeFactory = Arc<dyn Fn() -> Result<Runtime, BackendError> + Send + Sync>;

/// Bounds. Every one of them is a bound, not a hint.
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    pub runtime: RuntimeConfig,
    /// `docs/hardening.md` §6: *"Idle eviction at 10 minutes... a launchd agent that never evicts
    /// keeps 1.1 GB wired forever; one that evicts too eagerly makes every DevType call cold."*
    pub idle_timeout: Duration,
    /// Wall clock for one request, end to end.
    pub request_timeout: Duration,
    /// Concurrent requests. Beyond this the service says [`BackendError::Overloaded`] rather than
    /// queueing without bound.
    pub max_in_flight: usize,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            runtime: RuntimeConfig::default(),
            idle_timeout: Duration::from_secs(600),
            request_timeout: Duration::from_secs(30),
            max_in_flight: 8,
        }
    }
}

#[derive(Debug)]
struct Lifecycle {
    warm: Option<Arc<Runtime>>,
    in_flight: usize,
    started: Instant,
    last_used: Instant,
    requests_served: u64,
    refusals: u64,
    backend_errors: u64,
    builds: u64,
    build_failures: u64,
    rebuilds_after_poison: u64,
    idle_evictions: u64,
    last_cold_start: Option<Duration>,
    /// Set by a poison, consumed by the next build, which is then born `degraded`.
    next_build_degraded: bool,
    shutdown_requested: bool,
}

/// The warm runtime, its bounds and its counters.
pub struct Service {
    cfg: ServiceConfig,
    factory: RuntimeFactory,
    state: Mutex<Lifecycle>,
    wake: Condvar,
}

impl Service {
    /// A service that builds from configuration.
    pub fn new(cfg: ServiceConfig) -> Self {
        let runtime_cfg = cfg.runtime.clone();
        let factory: RuntimeFactory = Arc::new(move || Runtime::build(&runtime_cfg));
        Self::with_factory(cfg, factory)
    }

    pub fn with_factory(cfg: ServiceConfig, factory: RuntimeFactory) -> Self {
        let now = Instant::now();
        Self {
            cfg,
            factory,
            state: Mutex::new(Lifecycle {
                warm: None,
                in_flight: 0,
                started: now,
                last_used: now,
                requests_served: 0,
                refusals: 0,
                backend_errors: 0,
                builds: 0,
                build_failures: 0,
                rebuilds_after_poison: 0,
                idle_evictions: 0,
                last_cold_start: None,
                next_build_degraded: false,
                shutdown_requested: false,
            }),
            wake: Condvar::new(),
        }
    }

    pub fn config(&self) -> &ServiceConfig {
        &self.cfg
    }

    /// A poisoned mutex here is recovered rather than propagated. The data behind it is counters
    /// and an `Option<Arc>`; a panicking request cannot leave them in a state that makes the next
    /// request wrong, and killing the agent because one connection thread panicked is a worse
    /// failure than continuing.
    fn lock(&self) -> MutexGuard<'_, Lifecycle> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    // -- the shared contract -------------------------------------------------------------------

    /// Handle one protocol line and return one reply, **without** a trailing newline.
    ///
    /// The caller frames it: the socket writes `reply + b"\n"`, and so does `qd oneshot`. Framing
    /// outside this function is what makes the two modes' bytes comparable.
    pub fn handle_line(&self, line: &[u8]) -> Vec<u8> {
        match parse_line(line) {
            Ok(Incoming::Op(op)) => self.handle_op(op),
            Ok(Incoming::Request(request)) => {
                let response = self.handle_request(&request);
                encode(&response)
            }
            Err(refusal) => {
                self.lock().refusals += 1;
                encode(&Response::refused(refusal))
            }
        }
    }

    fn handle_op(&self, op: ControlOp) -> Vec<u8> {
        match op {
            ControlOp::Status => encode(&OpResponse::Status(Box::new(self.status()))),
            ControlOp::Ping => encode(&OpResponse::Ack {
                op: op.as_str().to_string(),
                acted: true,
            }),
            ControlOp::Evict => {
                let acted = self.evict();
                encode(&OpResponse::Ack {
                    op: op.as_str().to_string(),
                    acted,
                })
            }
            ControlOp::Shutdown => {
                self.request_shutdown();
                encode(&OpResponse::Ack {
                    op: op.as_str().to_string(),
                    acted: true,
                })
            }
        }
    }

    /// Answer one validated request, handling poison-and-rebuild.
    pub fn handle_request(&self, request: &DecisionRequest) -> Response {
        let response = match self.acquire() {
            Ok(acquired) => {
                let deadline = Some(Deadline::after(self.cfg.request_timeout));
                let first = acquired.runtime.answer(request, deadline);
                if is_poison(&first) {
                    // Release the in-flight slot before rebuilding: the rebuild takes the same
                    // lock, and a rebuild that waited on its own request would deadlock the agent
                    // exactly when it is already in trouble.
                    drop(acquired);
                    self.rebuild_after_poison(request)
                } else {
                    first
                }
            }
            Err(error) => Response::failed(error),
        };
        self.record(&response);
        response
    }

    fn rebuild_after_poison(&self, request: &DecisionRequest) -> Response {
        {
            let mut st = self.lock();
            st.warm = None;
            st.next_build_degraded = true;
            st.rebuilds_after_poison += 1;
        }
        match self.acquire() {
            Ok(acquired) => {
                let deadline = Some(Deadline::after(self.cfg.request_timeout));
                let second = acquired.runtime.answer(request, deadline);
                if is_poison(&second) {
                    // Poisoned again immediately after a rebuild: this is not a transient fault and
                    // retrying forever would turn one bad request into a hot loop.
                    Response::failed(BackendError::RebuildFailed {
                        detail: "the rebuilt backend poisoned again on the same request; refusing \
                                 to retry a second time"
                            .to_string(),
                    })
                } else {
                    second
                }
            }
            Err(error) => Response::failed(BackendError::RebuildFailed {
                detail: error.to_string(),
            }),
        }
    }

    fn acquire(&self) -> Result<Acquired<'_>, BackendError> {
        {
            let mut st = self.lock();
            if st.in_flight >= self.cfg.max_in_flight {
                return Err(BackendError::Overloaded {
                    limit: self.cfg.max_in_flight,
                });
            }
            st.in_flight += 1;
        }
        let guard = InFlight { service: self };

        let runtime = {
            let mut st = self.lock();
            if st.warm.is_none() {
                let degraded = st.next_build_degraded;
                let started = Instant::now();
                match (self.factory)() {
                    Ok(mut runtime) => {
                        if degraded {
                            runtime.mark_degraded();
                        }
                        st.last_cold_start = Some(started.elapsed());
                        st.builds += 1;
                        st.next_build_degraded = false;
                        st.warm = Some(Arc::new(runtime));
                    }
                    Err(error) => {
                        st.build_failures += 1;
                        let error = if degraded {
                            BackendError::RebuildFailed {
                                detail: error.to_string(),
                            }
                        } else {
                            error
                        };
                        drop(st);
                        return Err(error);
                    }
                }
            }
            st.last_used = Instant::now();
            st.warm.clone()
        };

        match runtime {
            Some(runtime) => Ok(Acquired {
                runtime,
                _guard: guard,
            }),
            // The block above either populated `warm` or returned; this arm exists so the happy
            // path does not unwrap.
            None => Err(BackendError::Unavailable {
                detail: "the warm runtime slot was empty immediately after a successful build"
                    .to_string(),
            }),
        }
    }

    fn record(&self, response: &Response) {
        let mut st = self.lock();
        match response {
            Response::Ok(_) => st.requests_served += 1,
            Response::Refused(_) => st.refusals += 1,
            Response::Error(_) => st.backend_errors += 1,
        }
    }

    // -- lifecycle -----------------------------------------------------------------------------

    /// Drop the warm runtime now. In-flight requests keep their own `Arc` and finish normally.
    /// Returns whether there was anything to drop.
    pub fn evict(&self) -> bool {
        let mut st = self.lock();
        st.warm.take().is_some()
    }

    /// Drop the warm runtime **if** it has been idle for longer than the configured timeout and
    /// nothing is in flight. Returns whether it evicted.
    pub fn reap_idle(&self) -> bool {
        let mut st = self.lock();
        if st.in_flight > 0 || st.warm.is_none() {
            return false;
        }
        if st.last_used.elapsed() < self.cfg.idle_timeout {
            return false;
        }
        st.warm = None;
        st.idle_evictions += 1;
        true
    }

    pub fn request_shutdown(&self) {
        let mut st = self.lock();
        st.shutdown_requested = true;
        drop(st);
        self.wake.notify_all();
    }

    pub fn shutdown_requested(&self) -> bool {
        self.lock().shutdown_requested
    }

    pub fn is_warm(&self) -> bool {
        self.lock().warm.is_some()
    }

    /// Sleep until `tick` elapses or someone calls [`Service::request_shutdown`].
    fn wait_tick(&self, tick: Duration) {
        let st = self.lock();
        let (_guard, _timeout) = self
            .wake
            .wait_timeout_while(st, tick, |s| !s.shutdown_requested)
            .unwrap_or_else(|e| e.into_inner());
    }

    /// The idle-eviction loop. Runs until `stop` is set or a `shutdown` op arrives.
    ///
    /// The tick is a quarter of the idle timeout, clamped, so eviction happens within 25% of the
    /// configured window without a thread that wakes up pointlessly on a 10-minute timeout.
    pub fn run_idle_reaper(service: Arc<Service>, stop: Arc<AtomicBool>) {
        let tick = service
            .cfg
            .idle_timeout
            .div_f64(4.0)
            .clamp(Duration::from_millis(5), Duration::from_secs(60));
        loop {
            if stop.load(Ordering::SeqCst) || service.shutdown_requested() {
                return;
            }
            service.wait_tick(tick);
            if stop.load(Ordering::SeqCst) || service.shutdown_requested() {
                return;
            }
            service.reap_idle();
        }
    }

    /// Everything a caller can ask about the agent without asking it a question.
    pub fn status(&self) -> StatusReport {
        let st = self.lock();
        StatusReport {
            runtime_version: crate::RUNTIME_VERSION.to_string(),
            warm: st.warm.is_some(),
            degraded: st.warm.as_ref().is_some_and(|r| r.is_degraded()),
            backend: st.warm.as_ref().map(|r| r.identity().clone()),
            in_flight: st.in_flight,
            max_in_flight: self.cfg.max_in_flight,
            requests_served: st.requests_served,
            refusals: st.refusals,
            backend_errors: st.backend_errors,
            builds: st.builds,
            build_failures: st.build_failures,
            rebuilds_after_poison: st.rebuilds_after_poison,
            idle_evictions: st.idle_evictions,
            last_cold_start_ms: st.last_cold_start.map(as_millis),
            idle_timeout_ms: as_millis(self.cfg.idle_timeout),
            request_timeout_ms: as_millis(self.cfg.request_timeout),
            idle_ms: as_millis(st.last_used.elapsed()),
            uptime_ms: as_millis(st.started.elapsed()),
            shutdown_requested: st.shutdown_requested,
        }
    }
}

fn as_millis(d: Duration) -> u64 {
    d.as_millis().min(u64::MAX as u128) as u64
}

fn is_poison(response: &Response) -> bool {
    matches!(
        response,
        Response::Error(env) if matches!(env.error, BackendError::Poisoned { .. })
    )
}

/// Decrements the in-flight counter and stamps `last_used` however the request ended, including a
/// panic unwinding out of a connection thread.
struct InFlight<'a> {
    service: &'a Service,
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        let mut st = self.service.lock();
        st.in_flight = st.in_flight.saturating_sub(1);
        st.last_used = Instant::now();
        drop(st);
        self.service.wake.notify_all();
    }
}

struct Acquired<'a> {
    runtime: Arc<Runtime>,
    _guard: InFlight<'a>,
}

/// The reply to a control op. Tagged on `status` like [`Response`] is, with values that cannot
/// collide with `ok` / `refused` / `error`, so one reader handles every line the protocol emits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum OpResponse {
    Status(Box<StatusReport>),
    Ack {
        op: String,
        /// Whether the op changed anything. `evict` on an already-cold agent is `acted: false`, not
        /// a failure — and not a success that implies it dropped something.
        acted: bool,
    },
}

/// Lifecycle and identity, including the **measured** cold start.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatusReport {
    pub runtime_version: String,
    pub warm: bool,
    pub degraded: bool,
    /// `None` when nothing is warm. Absent is not the same as "a clean build with empty hashes".
    pub backend: Option<BackendIdentity>,
    pub in_flight: usize,
    pub max_in_flight: usize,
    pub requests_served: u64,
    pub refusals: u64,
    pub backend_errors: u64,
    pub builds: u64,
    pub build_failures: u64,
    pub rebuilds_after_poison: u64,
    pub idle_evictions: u64,
    /// Measured around the last successful build. `None` means no build has succeeded yet — never
    /// a zero, which would read as an instant cold start.
    pub last_cold_start_ms: Option<u64>,
    pub idle_timeout_ms: u64,
    pub request_timeout_ms: u64,
    pub idle_ms: u64,
    pub uptime_ms: u64,
    pub shutdown_requested: bool,
}

/// Serialize a reply, without a path that can fail silently.
fn encode<T: Serialize>(value: &T) -> Vec<u8> {
    match serde_json::to_vec(value) {
        Ok(bytes) => bytes,
        Err(error) => {
            let fallback = Response::failed(BackendError::DecodeFailed {
                detail: format!("the reply could not be serialized: {error}"),
            });
            serde_json::to_vec(&fallback).unwrap_or_else(|_| {
                br#"{"status":"error","error":{"kind":"decode_failed","detail":"the reply could not be serialized"},"message":"the reply could not be serialized"}"#
                    .to_vec()
            })
        }
    }
}
