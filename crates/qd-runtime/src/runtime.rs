//! One warm runtime: hash binding, the answering procedure, and the `degraded` flag.
//!
//! A `Runtime` is immutable once built and is shared across concurrent requests behind an `Arc`.
//! Everything that changes over time — eviction, poisoning, rebuilds, counters — lives in
//! [`crate::service::Service`], which holds the `Arc`. That split is what lets the socket serve
//! several requests at once without any of them holding a lock across a decode.
//!
//! # Hash binding
//!
//! `docs/schema-api.md` refuses when the *"tokenizer / weight / head / label-set hash != build"*,
//! because *"a swapped tokenizer maps wrong ids silently. Answering would be confidently wrong."*
//! [`Runtime::check_hashes`] compares every hash the caller pinned against the loaded build and
//! returns [`Refusal::HashMismatch`] carrying **both** values and which one failed.
//!
//! A caller that pins nothing still gets the construction-time binding: the backend's identity is
//! what it is, and the registered route additionally checks that the head file names the backbone
//! that is actually loaded.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::admission::TrainedFamilies;
use crate::answer::{answer, AnswerContext, Deadline};
use crate::backend::{BackendIdentity, DecisionBackend};
use crate::calibration::CalibrationTable;
use crate::ensemble::EnsembleBackend;
use crate::reference::ReferenceBackend;
use crate::refusal::{BackendError, HashKind, QdError, Refusal};
use crate::registry::HeadRegistry;
use crate::release::{Ensemble, Release};
use crate::render::RenderCaps;
use crate::schema::{DecisionRequest, HashExpectation, Response, Route};

/// What a build of the runtime is made of.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub caps: RenderCaps,
    /// The deterministic reference backend is **off** unless this says otherwise. There is no
    /// environment default, no "if nothing else is configured" path, and no way to reach a letter
    /// without setting it.
    pub enable_reference_backend: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            caps: RenderCaps::DEFAULT,
            enable_reference_backend: false,
        }
    }
}

/// A built, warm runtime.
pub struct Runtime {
    backend: Arc<dyn DecisionBackend>,
    calibration: CalibrationTable,
    registry: HeadRegistry,
    caps: RenderCaps,
    /// What admission checks a request's task against (`crate::admission`).
    trained: TrainedFamilies,
    degraded: bool,
    cold_start: Duration,
}

impl std::fmt::Debug for Runtime {
    /// Written by hand because the backend is a trait object. It prints the identity and the
    /// degraded flag — what a log needs — and never the weights or the state.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let identity = self.backend.identity();
        f.debug_struct("Runtime")
            .field("backend", &identity.name)
            .field("is_model", &identity.is_model)
            .field("degraded", &self.degraded)
            .field("cold_start_ms", &self.cold_start.as_millis())
            .field("registered_tasks", &self.registry.tasks())
            .field("trained_families", &self.trained)
            .finish()
    }
}

impl Runtime {
    /// Build from configuration.
    ///
    /// With no backend enabled this is [`BackendError::Unavailable`] — a typed error naming what is
    /// absent and why. **It is not an empty answer and it is not a `noul`.** K1-K7 do not exist
    /// yet; a build that pretended otherwise would be the worst bug this system can have.
    ///
    /// The reference backend comes from no release, so admission's task check does not run
    /// ([`TrainedFamilies::NoRelease`]); every answer it gives is `degraded`.
    pub fn build(cfg: &RuntimeConfig) -> Result<Self, BackendError> {
        let started = Instant::now();
        if !cfg.enable_reference_backend {
            return Err(BackendError::Unavailable {
                detail: "this build carries no model backend. The Metal backend (K1-K7) starts \
                         only after the shipping gate, so there is nothing to answer with. The \
                         deterministic reference backend can be switched on explicitly for testing \
                         the plumbing (`qd serve --reference-backend`); it is not a model, it \
                         reports every answer as degraded, and its letters mean nothing"
                    .to_string(),
            });
        }
        let backend: Arc<dyn DecisionBackend> = Arc::new(ReferenceBackend::new(true));
        let calibration = CalibrationTable::reference();
        Self::assemble(
            backend,
            calibration,
            HeadRegistry::new(),
            cfg.caps,
            TrainedFamilies::NoRelease,
            started.elapsed(),
        )
    }

    /// Build around a backend that loaded from `release`, with the release's calibration table.
    ///
    /// There is no calibration parameter: the table is the one [`Release::open`] verified against
    /// the manifest, so the table that calibrates is the one `expect.calibration_hash` can pin.
    /// The backend must report the tower, tokenizer and calibration hash the release binds
    /// ([`Release::check_backend`]). A mismatch is [`BackendError::Unavailable`] carrying the
    /// release refusal: a runtime that cannot be built has nothing to answer with, and the model
    /// is never asked.
    ///
    /// Admission checks every request's task against the release's `trained_families`; a
    /// release that does not record them admits no task ([`crate::admission`]).
    pub fn from_release(
        release: &Release,
        backend: Arc<dyn DecisionBackend>,
        registry: HeadRegistry,
        caps: RenderCaps,
    ) -> Result<Self, BackendError> {
        release
            .check_backend(backend.identity())
            .map_err(|refusal| BackendError::Unavailable {
                detail: refusal.to_string(),
            })?;
        Self::assemble(
            backend,
            release.calibration().clone(),
            registry,
            caps,
            TrainedFamilies::of_release(release.trained_families()),
            Duration::ZERO,
        )
    }

    /// Build around N member backends, each loaded from the matching tower of `ensemble`, with
    /// the ensemble's calibration table.
    ///
    /// Refused, as [`BackendError::Unavailable`] carrying why, unless there is exactly one
    /// member per tower, in order, and each member reports its tower's weight and tokenizer
    /// hashes ([`crate::release::Tower::check_backend`]) and declares the ensemble's table.
    /// [`EnsembleBackend::new`] then refuses members that disagree on tokenizer, letter ids or
    /// table. N-1 members are never built into a runtime. Admission checks every request's task
    /// against the members' `trained_families`, which [`Ensemble::open`] requires to agree.
    pub fn from_ensemble(
        ensemble: &Ensemble,
        members: Vec<Arc<dyn DecisionBackend>>,
        registry: HeadRegistry,
        caps: RenderCaps,
    ) -> Result<Self, BackendError> {
        let towers = ensemble.towers();
        if members.len() != towers.len() {
            return Err(BackendError::Unavailable {
                detail: format!(
                    "ensemble {} names {} towers and {} members were loaded; a partial ensemble \
                     is not served",
                    ensemble.dir().display(),
                    towers.len(),
                    members.len()
                ),
            });
        }
        let table_hash = ensemble.calibration().hash();
        for (i, (tower, member)) in towers.iter().zip(&members).enumerate() {
            let identity = member.identity();
            tower
                .check_backend(identity)
                .map_err(|refusal| BackendError::Unavailable {
                    detail: format!("ensemble member {i}: {refusal}"),
                })?;
            if identity.calibration_hash != table_hash {
                return Err(BackendError::Unavailable {
                    detail: format!(
                        "ensemble member {i} (`{}`) declares calibration_hash {}; the ensemble's \
                         table is {table_hash}",
                        identity.name, identity.calibration_hash
                    ),
                });
            }
        }
        let backend = EnsembleBackend::new(members)?;
        Self::assemble(
            Arc::new(backend),
            ensemble.calibration().clone(),
            registry,
            caps,
            TrainedFamilies::of_release(ensemble.trained_families()),
            Duration::ZERO,
        )
    }

    /// Build around a caller-supplied backend, with no release. This is the seam the tests drive.
    ///
    /// There is no release, so there is no record of what the backend was trained on, and
    /// admission's task check does not run ([`TrainedFamilies::NoRelease`];
    /// [`Runtime::trained_families`] says so). Every model path in the product is built from a
    /// release with [`Runtime::from_release`] (`crates/qd-metal/src/serve.rs`), which does check
    /// it. Gating this seam too is open: its callers include test files another owner holds
    /// uncommitted edits to (GAP-RUNTIME-WITH-BACKEND-RUNS-NO-TRAINED-FAMILY-CHECK-2026-10-03).
    pub fn with_backend(
        backend: Arc<dyn DecisionBackend>,
        calibration: CalibrationTable,
        registry: HeadRegistry,
        caps: RenderCaps,
    ) -> Result<Self, BackendError> {
        Self::assemble(
            backend,
            calibration,
            registry,
            caps,
            TrainedFamilies::NoRelease,
            Duration::ZERO,
        )
    }

    fn assemble(
        backend: Arc<dyn DecisionBackend>,
        calibration: CalibrationTable,
        registry: HeadRegistry,
        caps: RenderCaps,
        trained: TrainedFamilies,
        cold_start: Duration,
    ) -> Result<Self, BackendError> {
        caps.validate().map_err(|e| BackendError::Unavailable {
            detail: format!("render caps rejected: {e}"),
        })?;
        calibration
            .validate()
            .map_err(|e| BackendError::Unavailable {
                detail: format!("calibration table rejected: {e}"),
            })?;
        let degraded = !backend.identity().is_model;
        Ok(Self {
            backend,
            calibration,
            registry,
            caps,
            trained,
            degraded,
            cold_start,
        })
    }

    /// Mark this runtime as one that was rebuilt after a poison.
    ///
    /// `docs/schema-api.md`: *"`degraded` is true when the runtime answered from a
    /// rebuilt-after-poison `GpuRuntime`."* It stays true for the life of this runtime; the flag
    /// clears when the runtime is itself replaced by a clean build, which is what idle eviction
    /// eventually does.
    pub fn mark_degraded(&mut self) {
        self.degraded = true;
    }

    pub fn is_degraded(&self) -> bool {
        self.degraded
    }

    pub fn identity(&self) -> &BackendIdentity {
        self.backend.identity()
    }

    pub fn cold_start(&self) -> Duration {
        self.cold_start
    }

    pub fn caps(&self) -> &RenderCaps {
        &self.caps
    }

    pub fn registry(&self) -> &HeadRegistry {
        &self.registry
    }

    pub fn calibration(&self) -> &CalibrationTable {
        &self.calibration
    }

    /// What a request's task is checked against: the release's record, a release without one,
    /// or no release at all (the check not run).
    pub fn trained_families(&self) -> &TrainedFamilies {
        &self.trained
    }

    /// Compare every hash the caller pinned against the loaded build.
    ///
    /// On the registered route `head_hash` is the **head file's** hash, because that is the thing a
    /// registered caller ships and therefore the thing it can pin. On the generic route it is the
    /// backend's `lm_head`. The registered head's binding to the backbone is a separate check and
    /// runs whether or not the caller pinned anything.
    ///
    /// `calibration_hash` is compared with the table this runtime **calibrates with**, not with
    /// the hash the backend declares: the backend does not own the table, and a pin that passed
    /// for a table that is not serving would be approval of something unexamined
    /// (`GAP-RT-CALIBRATION-HASH-PIN-BINDS-THE-BACKEND-NOT-THE-LOADED-TABLE`).
    pub fn check_hashes(
        &self,
        expect: &HashExpectation,
        head_hash: &str,
    ) -> Result<(), Refusal> {
        let identity = self.identity();
        let calibration_hash = self.calibration.hash();
        let pairs: [(HashKind, &Option<String>, &str); 5] = [
            (
                HashKind::Tokenizer,
                &expect.tokenizer_hash,
                &identity.tokenizer_hash,
            ),
            (HashKind::Weights, &expect.weight_hash, &identity.weight_hash),
            (HashKind::Head, &expect.head_hash, head_hash),
            (
                HashKind::LabelSet,
                &expect.label_set_hash,
                &identity.label_set_hash,
            ),
            (
                HashKind::Calibration,
                &expect.calibration_hash,
                &calibration_hash,
            ),
        ];
        for (which, pinned, actual) in pairs {
            if let Some(pinned) = pinned
                && pinned != actual
            {
                return Err(Refusal::HashMismatch {
                    which,
                    expected: actual.to_string(),
                    actual: pinned.clone(),
                });
            }
        }
        Ok(())
    }

    /// Answer one validated request.
    ///
    /// Always returns a [`Response`] — the three outcomes are exhaustive and a caller must match
    /// them. A refusal and a backend failure reach the caller in **different variants**, and
    /// neither can reach the caller as a slot value.
    pub fn answer(&self, request: &DecisionRequest, deadline: Option<Deadline>) -> Response {
        match self.answer_inner(request, deadline) {
            Ok(envelope) => Response::Ok(envelope),
            Err(QdError::Refused(refusal)) => Response::refused(refusal),
            Err(QdError::Backend(error)) => Response::failed(error),
        }
    }

    fn answer_inner(
        &self,
        request: &DecisionRequest,
        deadline: Option<Deadline>,
    ) -> Result<crate::schema::AnswerEnvelope, QdError> {
        // Resolve the head first on the registered route: the hash the caller can pin is the head
        // file's, so it has to exist and be bound before there is anything to compare against.
        let head_hash = match request.route {
            Route::Generic => self.identity().head_hash.clone(),
            Route::Registered => {
                self.registry
                    .resolve(&request.task, self.identity())?
                    .head_hash
                    .clone()
            }
        };
        self.check_hashes(&request.expect, &head_hash)?;
        // Before the model: a task the release did not train, or a context its task was never
        // trained on, is refused, not answered.
        crate::admission::admit(request, &self.trained)?;

        let ctx = AnswerContext {
            backend: self.backend.as_ref(),
            calibration: &self.calibration,
            registry: &self.registry,
            caps: &self.caps,
            deadline,
            degraded: self.degraded,
        };
        answer(&ctx, request)
    }
}
