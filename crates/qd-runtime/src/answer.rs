//! The answering procedure of `docs/schema-api.md`.
//!
//! > 1. Prefill the context **once**.
//! > 2. **Snapshot** the recurrent GDN state and the attention KV.
//! > 3. Answer each slot as a 1-token query **from the snapshot**. Never write back.
//! > 4. For `choice`: run a **second pass with the options permuted** and require agreement.
//! >    Disagreement is `noul`.
//! > 5. Return per-slot `{value, conformal_set, score, noul, degraded}`.
//!
//! # The invariant this module enforces rather than documents
//!
//! Step 3 is the whole snapshot design. If a decode writes back, slot 2's answer depends on slot
//! 1's and the prefix cache is not a cache. So every decode here goes through [`readonly_decode`],
//! which hashes the state buffer either side of the call and raises
//! [`BackendError::ReadonlyViolated`] — naming the slot index and both hashes — if it moved. Where
//! the state is not host-visible the check is [`SlotIsolationCheck::NotRun`] and the answer is
//! degraded; it never silently reads as a check that passed.
//!
//! # Where refusals are decided
//!
//! Everything that can refuse — rendering, calibration lookup, the registered head and its binding
//! — is resolved **before** the first backend call. A refusal must not depend on whether a backend
//! happened to be warm, and a caller must never see a backend error for a request that was not
//! answerable in the first place.
//!
//! # What step 4 does not cover
//!
//! * `score` slots are **not** permuted. The bins are ordered and the loss is cumulative
//!   (CORAL-style); permuting them would destroy the ordering the loss depends on, which is why
//!   `python/qd_data/render.py` refuses to shuffle them either. The contract says step 4 is "for
//!   `choice`", and it is.
//! * The **registered route** has no second pass. There is no option text in the prompt to permute
//!   — the head's rows *are* the option order — and permuting the rows of a GEMV is exactly
//!   equivariant, so the check would agree with itself by construction. Recorded as an
//!   under-specification of `docs/schema-api.md`, which says "same abstain rule" without saying
//!   what the permutation half of it means for a route with no options in the prompt. The margin
//!   half of the abstain rule applies unchanged.

use std::collections::BTreeMap;
use std::time::Instant;

use crate::backend::{
    validate_logits, DecisionBackend, DecodeMode, Logits, QueryKind, SlotIsolationCheck, SlotQuery,
    StateSnapshot,
};
use crate::calibration::{calibrate, CalibrationEntry, CalibrationTable};
use crate::context::Context;
use crate::refusal::{BackendError, QdError, Refusal};
use crate::registry::{gemv, HeadRegistry, RegisteredHead};
use crate::render::{
    permuted_slot_suffix, render, second_pass_permutation, RenderCaps, RowLabel, SlotRender,
};
use crate::schema::{
    AnswerEnvelope, ConformalSet, DecisionRequest, Route, SlotAnswer, SlotKind, SlotSpec, SlotValue,
    SpanValue, RESERVED_NOUL_ROWS,
};

/// How many rows a `span` slot's pointer head ranges over: one per line start in *this* context,
/// plus the reserved `noul` row.
///
/// This is the definition [`crate::schema::SlotSpec::answer_rows`] defers to. It is context-
/// dependent, which is exactly why a fixed-shape registered head cannot answer a `span`.
///
/// The reserved row is **last**, at index `context.line_count()`, exactly as it is last for
/// `choice` and `score`. A pointer head trained with the abstain column first, served by this
/// runtime, would put every "no evidence" answer on the context's final line and no loss curve
/// would show it. `python/qd_train/heads.py` places the abstain column at index `n_candidates`,
/// which agrees. Asserted in `tests/answer_envelope_contract.rs`.
pub fn span_rows(context: &Context) -> usize {
    context.line_count() + RESERVED_NOUL_ROWS
}

/// Which halves of the contract's abstain rule apply to one (route, slot kind) pair.
///
/// `docs/schema-api.md` says of the registered route: *"Same prefill, same calibration table, same
/// abstain rule."* **"Same abstain rule" cannot be literally true**, and this enum is where that is
/// stated instead of left for a reader to discover. The rule has two halves — the calibrated-margin
/// threshold, and step 4's permuted second pass — and the permutation half has no meaning on a
/// route with no option text in the prompt: the head's rows *are* the option order, and permuting
/// the rows of a GEMV is exactly equivariant, so the check would agree with itself by construction
/// and abstain on nothing.
///
/// `GAP-RT-SPEC-REGISTERED-ROUTE`. The consequence is stated rather than hidden: the registered
/// route has one fewer abstention mechanism than the generic route, so its `noul` rate will not
/// match the generic route's even on a perfectly fitted head. The contract's generic-vs-registered
/// agreement test must account for that, or it reports a route difference as a head bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbstainRule {
    /// The margin threshold **and** step 4's permuted second pass. Generic-route `choice` only.
    MarginAndPermutedSecondPass,
    /// The margin threshold, plus the span's own ordering check: an end before its start is not a
    /// low-confidence span, it is not a span, and it abstains rather than being reordered.
    MarginAndSpanOrdering,
    /// The margin threshold alone.
    MarginOnly,
}

impl AbstainRule {
    /// Whether step 4's second pass runs under this rule.
    pub fn runs_permuted_second_pass(self) -> bool {
        matches!(self, AbstainRule::MarginAndPermutedSecondPass)
    }
}

/// The whole table, in one place. `None` means the pair is not served at all.
///
/// The single `None` is a registered-route `span`: its pointer head ranges over *this* context's
/// line starts, so its row count changes per request, and a head file has a fixed row count. That
/// is [`Refusal::RegisteredRouteSpanUnsupported`] rather than a quiet fall back to the generic
/// route, which would make `route` advisory and make the contract's generic-vs-registered
/// agreement test compare the generic route with itself.
///
/// `score` is margin-only on **both** routes: its bins are ordered and its loss is cumulative
/// (CORAL-style), so permuting them would destroy what the loss depends on. The contract scopes
/// step 4 to `choice`, and this is that scoping made total.
pub fn abstain_rule(route: Route, kind: SlotKind) -> Option<AbstainRule> {
    match (route, kind) {
        (Route::Generic, SlotKind::Choice) => Some(AbstainRule::MarginAndPermutedSecondPass),
        (Route::Generic, SlotKind::Score) => Some(AbstainRule::MarginOnly),
        (Route::Generic, SlotKind::Span) => Some(AbstainRule::MarginAndSpanOrdering),
        (Route::Registered, SlotKind::Choice | SlotKind::Score) => Some(AbstainRule::MarginOnly),
        (Route::Registered, SlotKind::Span) => None,
    }
}

/// Everything the procedure needs, with no lifecycle attached.
///
/// Separated from [`crate::runtime::Runtime`] so the procedure can be driven directly by a test
/// against a hand-built backend, which is how the slot-isolation and readonly-violation tests are
/// written.
pub struct AnswerContext<'a> {
    pub backend: &'a dyn DecisionBackend,
    pub calibration: &'a CalibrationTable,
    pub registry: &'a HeadRegistry,
    pub caps: &'a RenderCaps,
    /// Bounded, always. `None` only in a test that is asserting something other than the deadline.
    pub deadline: Option<Deadline>,
    /// True when the runtime itself is degraded — a rebuilt-after-poison backend, or one that is
    /// not a model. ORed with anything the procedure discovers.
    pub degraded: bool,
}

/// A wall-clock bound, carrying the limit it was built from so
/// [`BackendError::DeadlineExceeded`] can name a number the caller configured rather than the zero
/// that is left when it fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Deadline {
    pub at: Instant,
    pub limit_ms: u64,
}

impl Deadline {
    pub fn after(limit: std::time::Duration) -> Self {
        Self {
            at: Instant::now() + limit,
            limit_ms: limit.as_millis().min(u64::MAX as u128) as u64,
        }
    }
}

/// Per-slot plan, resolved before the backend is touched.
struct SlotPlan<'a> {
    spec: &'a SlotSpec,
    render: &'a SlotRender,
    rows: usize,
    entry: CalibrationEntry,
}

/// Run the procedure.
pub fn answer(ctx: &AnswerContext<'_>, request: &DecisionRequest) -> Result<AnswerEnvelope, QdError> {
    let prompt = render(request, ctx.caps)?;
    let identity = ctx.backend.identity();

    // -- resolve every refusal before the first backend call --------------------------------
    let head: Option<&RegisteredHead> = match request.route {
        Route::Generic => None,
        Route::Registered => Some(ctx.registry.resolve(&request.task, identity)?),
    };

    let mut plans: Vec<SlotPlan<'_>> = Vec::with_capacity(request.slots.len());
    for (index, spec) in request.slots.iter().enumerate() {
        // `render` emits one `SlotRender` per request slot, in request order, so the index is the
        // pairing. Looking it up by name would re-derive a mapping that is already a fact of the
        // renderer, and would need a refusal for a case the renderer cannot produce.
        let slot_render = prompt
            .slots
            .get(index)
            .ok_or(Refusal::EmptySlotName { index })?;
        let rows = match spec {
            SlotSpec::Choice { .. } | SlotSpec::Score { .. } => {
                spec.answer_rows().unwrap_or(RESERVED_NOUL_ROWS)
            }
            SlotSpec::Span { .. } => span_rows(&request.context),
        };
        // The one place `abstain_rule` decides something at runtime: a pair with no rule is a pair
        // this runtime does not serve, and the table is the single owner of which those are. The
        // rest of what the table states — which halves of the rule each served pair runs under —
        // is asserted against observed behaviour in `tests/answer_envelope_contract.rs` rather
        // than restated here, because restating it is how two copies come to disagree.
        if abstain_rule(request.route, spec.kind()).is_none() {
            return Err(Refusal::RegisteredRouteSpanUnsupported {
                slot: spec.name().to_string(),
                rows,
            }
            .into());
        }
        if let Some(head) = head
            && !head.slots.contains_key(spec.name())
        {
            return Err(Refusal::RegisteredHeadSlotMissing {
                task: request.task.clone(),
                slot: spec.name().to_string(),
                available: head.slots.keys().cloned().collect(),
            }
            .into());
        }
        let entry = *ctx
            .calibration
            .lookup(spec.name(), spec.kind(), rows)?;
        plans.push(SlotPlan {
            spec,
            render: slot_render,
            rows,
            entry,
        });
    }

    // -- 1. prefill once, 2. snapshot -------------------------------------------------------
    check_deadline(ctx.deadline)?;
    let handle = ctx.backend.prefill(&prompt.prefix)?;
    check_deadline(ctx.deadline)?;
    let mut snapshot = ctx.backend.snapshot(&handle)?;

    let pooled = match head {
        Some(_) => {
            check_deadline(ctx.deadline)?;
            Some(ctx.backend.pooled_features(&snapshot)?)
        }
        None => None,
    };

    // -- 3-5. one slot at a time, always from the snapshot ----------------------------------
    let mut degraded = ctx.degraded || !identity.is_model;
    let mut slots: BTreeMap<String, SlotAnswer> = BTreeMap::new();

    for (slot_index, plan) in plans.iter().enumerate() {
        check_deadline(ctx.deadline)?;
        let (answer, isolation) = match (head, pooled.as_deref()) {
            (Some(head), Some(features)) => {
                let registered = answer_registered(plan, head, features)?;
                (registered, None)
            }
            _ => {
                let (a, checks) = answer_generic(ctx, request, plan, slot_index, &mut snapshot)?;
                (a, Some(checks))
            }
        };
        if let Some(checks) = isolation
            && checks.iter().any(|c| !c.ran())
        {
            degraded = true;
        }
        slots.insert(plan.spec.name().to_string(), answer);
    }

    if degraded {
        for slot in slots.values_mut() {
            slot.degraded = true;
        }
    }

    Ok(AnswerEnvelope {
        // The request's version, not a constant: it was validated against
        // `SUPPORTED_SCHEMA_VERSIONS` before this function ran, and answering in the version that
        // was asked is what lets a caller pin one.
        schema_version: request.schema_version,
        backend: identity.name.clone(),
        degraded,
        slots,
    })
}

/// A slot on the generic route: one or two 1-token queries against the snapshot.
fn answer_generic(
    ctx: &AnswerContext<'_>,
    request: &DecisionRequest,
    plan: &SlotPlan<'_>,
    slot_index: usize,
    snapshot: &mut StateSnapshot,
) -> Result<(SlotAnswer, Vec<SlotIsolationCheck>), QdError> {
    let name = plan.spec.name();
    match plan.spec {
        SlotSpec::Choice { options, .. } => {
            let query = SlotQuery {
                slot_name: name,
                suffix: &plan.render.suffix,
                rows: plan.rows,
                kind: QueryKind::Letters,
            };
            let (first, check_a) =
                readonly_decode(ctx.backend, snapshot, &query, slot_index)?;
            let cal = calibrate(name, &first.values, &plan.entry)?;
            let noul_row = plan.rows - RESERVED_NOUL_ROWS;

            // Step 4: the second pass, with the options permuted.
            let digest = request.digest();
            let perm = second_pass_permutation(&digest, name, options.len());
            let permuted = permuted_slot_suffix(plan.spec, ctx.caps, &perm)?;
            let second_query = SlotQuery {
                slot_name: name,
                suffix: &permuted.suffix,
                rows: plan.rows,
                kind: QueryKind::Letters,
            };
            let (second, check_b) =
                readonly_decode(ctx.backend, snapshot, &second_query, slot_index)?;
            let cal2 = calibrate(name, &second.values, &plan.entry)?;
            let checks = vec![check_a, check_b];

            if cal.top == noul_row || cal2.top == noul_row {
                return Ok((SlotAnswer::abstained(cal.margin, false), checks));
            }
            // Map both winners back to the option they name. The second pass rendered option
            // `perm[row]` at row `row`, so agreement is about the *value*, not the letter — which
            // is the whole point: letter-position bias shows up here as disagreement.
            let first_value = options.get(cal.top);
            let second_value = perm.get(cal2.top).and_then(|i| options.get(*i));
            let (Some(first_value), Some(second_value)) = (first_value, second_value) else {
                return Err(BackendError::LogitShapeMismatch {
                    slot: name.to_string(),
                    expected: plan.rows,
                    actual: plan.rows,
                }
                .into());
            };
            if first_value != second_value || cal.margin < plan.entry.noul_margin {
                return Ok((SlotAnswer::abstained(cal.margin, false), checks));
            }

            let set: Vec<String> = cal
                .set
                .iter()
                .filter(|row| **row != noul_row)
                .filter_map(|row| options.get(*row).cloned())
                .collect();
            Ok((
                SlotAnswer::answered(
                    SlotValue::Choice(first_value.clone()),
                    Some(ConformalSet::Choices(set)),
                    cal.margin,
                ),
                checks,
            ))
        }

        SlotSpec::Score { bins, .. } => {
            let query = SlotQuery {
                slot_name: name,
                suffix: &plan.render.suffix,
                rows: plan.rows,
                kind: QueryKind::Letters,
            };
            let (logits, check) = readonly_decode(ctx.backend, snapshot, &query, slot_index)?;
            let cal = calibrate(name, &logits.values, &plan.entry)?;
            let checks = vec![check];
            let noul_row = plan.rows - RESERVED_NOUL_ROWS;
            if cal.top == noul_row || cal.margin < plan.entry.noul_margin {
                return Ok((SlotAnswer::abstained(cal.margin, false), checks));
            }
            let bin = bin_for_row(plan.render, cal.top).ok_or_else(|| {
                BackendError::LogitShapeMismatch {
                    slot: name.to_string(),
                    expected: *bins as usize + RESERVED_NOUL_ROWS,
                    actual: plan.rows,
                }
            })?;
            let set: Vec<u32> = cal
                .set
                .iter()
                .filter(|row| **row != noul_row)
                .filter_map(|row| bin_for_row(plan.render, *row))
                .collect();
            Ok((
                SlotAnswer::answered(
                    SlotValue::Score(bin),
                    Some(ConformalSet::Scores(set)),
                    cal.margin,
                ),
                checks,
            ))
        }

        SlotSpec::Span { .. } => {
            // Two pointer reads from the same snapshot, both read-only. `docs/schema-api.md`
            // specifies "a pointer head over line-start tokens" and an answer of
            // `{start_line, end_line}` without saying whether that is one decode or two. Two is the
            // reading that keeps every decode a 1-token query; recorded as an under-specification.
            let noul_row = plan.rows - RESERVED_NOUL_ROWS;
            let start_query = SlotQuery {
                slot_name: name,
                suffix: &plan.render.suffix,
                rows: plan.rows,
                kind: QueryKind::PointerStart,
            };
            let (start_logits, check_a) =
                readonly_decode(ctx.backend, snapshot, &start_query, slot_index)?;
            let end_query = SlotQuery {
                kind: QueryKind::PointerEnd,
                ..start_query
            };
            let (end_logits, check_b) =
                readonly_decode(ctx.backend, snapshot, &end_query, slot_index)?;
            let checks = vec![check_a, check_b];

            let start = calibrate(name, &start_logits.values, &plan.entry)?;
            let end = calibrate(name, &end_logits.values, &plan.entry)?;
            // The weaker pointer bounds the span's confidence: a span is only as good as its less
            // certain end.
            let score = start.margin.min(end.margin);
            if start.top == noul_row || end.top == noul_row || end.top < start.top {
                // A span that runs backwards is not a low-confidence span, it is not a span. It
                // abstains rather than being silently reordered into a plausible-looking answer.
                return Ok((SlotAnswer::abstained(score, false), checks));
            }
            if score < plan.entry.noul_margin {
                return Ok((SlotAnswer::abstained(score, false), checks));
            }
            Ok((
                SlotAnswer::answered(
                    SlotValue::Span(SpanValue {
                        start_line: start.top + 1,
                        end_line: end.top + 1,
                    }),
                    // `null` for `span`, per the contract's example.
                    None,
                    score,
                ),
                checks,
            ))
        }
    }
}

/// A slot on the registered route: one GEMV on the pooled features. Same snapshot, same calibration
/// table, same margin-based abstain rule.
fn answer_registered(
    plan: &SlotPlan<'_>,
    head: &RegisteredHead,
    features: &[f32],
) -> Result<SlotAnswer, QdError> {
    let name = plan.spec.name();
    let matrix = head
        .slots
        .get(name)
        .ok_or_else(|| Refusal::RegisteredHeadSlotMissing {
            task: head.task.clone(),
            slot: name.to_string(),
            available: head.slots.keys().cloned().collect(),
        })?;
    if head.feature_dim != features.len() {
        return Err(BackendError::LogitShapeMismatch {
            slot: format!("{name} (pooled features)"),
            expected: head.feature_dim,
            actual: features.len(),
        }
        .into());
    }
    let values = gemv(name, matrix, features, plan.rows)?;
    let logits = Logits {
        kind: QueryKind::Letters,
        values,
    };
    let query = SlotQuery {
        slot_name: name,
        suffix: &plan.render.suffix,
        rows: plan.rows,
        kind: QueryKind::Letters,
    };
    validate_logits(&logits, &query)?;

    let cal = calibrate(name, &logits.values, &plan.entry)?;
    let noul_row = plan.rows - RESERVED_NOUL_ROWS;
    if cal.top == noul_row || cal.margin < plan.entry.noul_margin {
        return Ok(SlotAnswer::abstained(cal.margin, false));
    }
    match plan.spec {
        SlotSpec::Choice { options, .. } => {
            let value = options
                .get(cal.top)
                .ok_or_else(|| BackendError::LogitShapeMismatch {
                    slot: name.to_string(),
                    expected: plan.rows,
                    actual: cal.top,
                })?;
            let set: Vec<String> = cal
                .set
                .iter()
                .filter(|row| **row != noul_row)
                .filter_map(|row| options.get(*row).cloned())
                .collect();
            Ok(SlotAnswer::answered(
                SlotValue::Choice(value.clone()),
                Some(ConformalSet::Choices(set)),
                cal.margin,
            ))
        }
        SlotSpec::Score { .. } => {
            let bin =
                bin_for_row(plan.render, cal.top).ok_or_else(|| BackendError::LogitShapeMismatch {
                    slot: name.to_string(),
                    expected: plan.rows,
                    actual: cal.top,
                })?;
            let set: Vec<u32> = cal
                .set
                .iter()
                .filter(|row| **row != noul_row)
                .filter_map(|row| bin_for_row(plan.render, *row))
                .collect();
            Ok(SlotAnswer::answered(
                SlotValue::Score(bin),
                Some(ConformalSet::Scores(set)),
                cal.margin,
            ))
        }
        // Refused in `answer` before any backend call; unreachable here, and refused rather than
        // panicked if the two ever disagree.
        SlotSpec::Span { name } => Err(Refusal::RegisteredRouteSpanUnsupported {
            slot: name.clone(),
            rows: plan.rows,
        }
        .into()),
    }
}

fn bin_for_row(render: &SlotRender, row: usize) -> Option<u32> {
    match render.rows.get(row) {
        Some(RowLabel::Score(bin)) => Some(*bin),
        _ => None,
    }
}

/// Decode one query from the snapshot with the `readonly` flag, and **verify** it.
///
/// `docs/hardening.md` §6: *"`readonly` decode leaves the state buffer hash-identical. This is the
/// slot-isolation guarantee; if it fails, slot 2's answer depends on slot 1's, and the whole
/// snapshot design is void."*
pub fn readonly_decode(
    backend: &dyn DecisionBackend,
    snapshot: &mut StateSnapshot,
    query: &SlotQuery<'_>,
    slot_index: usize,
) -> Result<(Logits, SlotIsolationCheck), BackendError> {
    let host_visible = backend.identity().state_host_visible;
    let before = if host_visible {
        Some(snapshot.state_digest())
    } else {
        None
    };

    let logits = backend.decode_slot(snapshot, query, DecodeMode::ReadOnly)?;

    let check = match before {
        Some(before) => {
            let after = snapshot.state_digest();
            if after != before {
                return Err(BackendError::ReadonlyViolated {
                    slot_index,
                    before: crate::hex(&before),
                    after: crate::hex(&after),
                });
            }
            SlotIsolationCheck::Ran {
                state_hash: crate::hex(&before),
            }
        }
        None => SlotIsolationCheck::NotRun {
            reason: format!(
                "backend `{}` reports state_host_visible = false, so the state buffer could not be \
                 hashed either side of the decode; slot isolation is unverified for this answer",
                backend.identity().name
            ),
        },
    };

    validate_logits(&logits, query)?;
    Ok((logits, check))
}

fn check_deadline(deadline: Option<Deadline>) -> Result<(), BackendError> {
    match deadline {
        Some(d) if Instant::now() >= d.at => Err(BackendError::DeadlineExceeded {
            limit_ms: d.limit_ms,
        }),
        _ => Ok(()),
    }
}
