//! `docs/hardening.md` §6, the lifecycle half.
//!
//! > Poisoned `GpuRuntime` returns DEGRADED and rebuilds; the rebuild is tested, not assumed.
//! > Idle eviction at 10 minutes, and a measured cold start — a launchd agent that never evicts
//! > keeps 1.1 GB wired forever; one that evicts too eagerly makes every DevType call cold.
//! > Concurrent requests on the socket; a slow request must not wedge the agent.
//!
//! The timeouts here are milliseconds rather than minutes because a test that waits ten minutes is
//! a test nobody runs. The **value** is configuration; the **behaviour** is what is under test, and
//! `ServiceConfig::default()` is asserted to carry the plan's ten minutes so the shipped number is
//! covered too.

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{Behaviour, Wrapped};
use qd_runtime::refusal::BackendError;
use qd_runtime::runtime::{Runtime, RuntimeConfig};
use qd_runtime::schema::{CallerReading, Response};
use qd_runtime::service::{RuntimeFactory, Service, ServiceConfig};
use serde_json::json;

fn config(idle: Duration, request: Duration, max_in_flight: usize) -> ServiceConfig {
    ServiceConfig {
        runtime: RuntimeConfig {
            caps: qd_runtime::render::RenderCaps::DEFAULT,
            enable_reference_backend: true,
        },
        idle_timeout: idle,
        request_timeout: request,
        max_in_flight,
    }
}

/// Spin until `predicate` holds or the budget runs out. Returns whether it held.
fn wait_for(budget: Duration, mut predicate: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + budget;
    while Instant::now() < deadline {
        if predicate() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    predicate()
}

// -- the shipped defaults ----------------------------------------------------------------------

#[test]
fn the_shipped_idle_timeout_is_the_plans_ten_minutes() {
    let defaults = ServiceConfig::default();
    assert_eq!(defaults.idle_timeout, Duration::from_secs(600));
    assert!(
        defaults.request_timeout > Duration::ZERO,
        "every request is bounded"
    );
    assert!(defaults.max_in_flight >= 1, "every fan-out is bounded");
    assert!(
        !defaults.runtime.enable_reference_backend,
        "the reference backend must never be on by default"
    );
}

// -- cold start and eviction --------------------------------------------------------------------

#[test]
fn nothing_is_built_until_a_request_needs_it() {
    let service = common::reference_service();
    assert!(!service.is_warm());
    assert_eq!(service.status().builds, 0);
    assert_eq!(
        service.status().last_cold_start_ms, None,
        "no build has happened, which is not the same fact as a zero-millisecond one"
    );
    // A control op must not warm the runtime.
    let _ = service.handle_line(br#"{"op":"ping"}"#);
    assert!(!service.is_warm());
}

#[test]
fn the_cold_start_is_measured_not_assumed() {
    let factory: RuntimeFactory = Arc::new(|| {
        std::thread::sleep(Duration::from_millis(25));
        Runtime::with_backend(
            Arc::new(qd_runtime::reference::ReferenceBackend::new(true)),
            qd_runtime::calibration::CalibrationTable::reference(),
            qd_runtime::registry::HeadRegistry::new(),
            qd_runtime::render::RenderCaps::DEFAULT,
        )
    });
    let service = Service::with_factory(
        config(Duration::from_secs(600), Duration::from_secs(10), 4),
        factory,
    );

    let reply = service.handle_line(&common::line(&common::sample_request()));
    assert_eq!(common::status_of(&reply), "ok");

    let status = service.status();
    assert_eq!(status.builds, 1);
    let measured = status
        .last_cold_start_ms
        .expect("a successful build records its cold start");
    assert!(
        measured >= 20,
        "the cold start must be measured around the real build, got {measured} ms"
    );
    assert!(status.warm);
    assert!(
        status.backend.is_some(),
        "a warm agent reports which backend is loaded"
    );
}

#[test]
fn an_idle_runtime_is_evicted_and_the_next_request_rebuilds() {
    let service = Service::new(config(
        Duration::from_millis(60),
        Duration::from_secs(10),
        4,
    ));
    let payload = common::line(&common::sample_request());

    assert_eq!(common::status_of(&service.handle_line(&payload)), "ok");
    assert!(service.is_warm());
    assert!(
        !service.reap_idle(),
        "a runtime used a moment ago is not idle"
    );

    assert!(
        wait_for(Duration::from_secs(5), || service.reap_idle()),
        "the runtime was never evicted"
    );
    assert!(!service.is_warm());
    assert_eq!(service.status().idle_evictions, 1);

    assert_eq!(common::status_of(&service.handle_line(&payload)), "ok");
    assert_eq!(
        service.status().builds,
        2,
        "the request after an eviction pays a cold start"
    );
}

#[test]
fn the_reaper_thread_evicts_on_its_own() {
    let service = Arc::new(Service::new(config(
        Duration::from_millis(40),
        Duration::from_secs(10),
        4,
    )));
    assert_eq!(
        common::status_of(&service.handle_line(&common::line(&common::sample_request()))),
        "ok"
    );
    assert!(service.is_warm());

    let stop = Arc::new(AtomicBool::new(false));
    let reaper = {
        let service = Arc::clone(&service);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || Service::run_idle_reaper(service, stop))
    };

    let evicted = wait_for(Duration::from_secs(5), || !service.is_warm());
    stop.store(true, Ordering::SeqCst);
    service.request_shutdown();
    let _ = reaper.join();

    assert!(evicted, "the reaper thread never evicted an idle runtime");
    assert_eq!(service.status().idle_evictions, 1);
}

#[test]
fn the_evict_op_drops_the_runtime_and_says_whether_it_did() {
    let service = common::reference_service();
    let first = service.handle_line(br#"{"op":"evict"}"#);
    assert_eq!(common::status_of(&first), "ack");
    let parsed: serde_json::Value = serde_json::from_slice(&first).expect("parses");
    assert_eq!(
        parsed.get("acted").and_then(serde_json::Value::as_bool),
        Some(false),
        "evicting a cold agent acted on nothing, and must not claim otherwise"
    );

    assert_eq!(
        common::status_of(&service.handle_line(&common::line(&common::sample_request()))),
        "ok"
    );
    let second = service.handle_line(br#"{"op":"evict"}"#);
    let parsed: serde_json::Value = serde_json::from_slice(&second).expect("parses");
    assert_eq!(
        parsed.get("acted").and_then(serde_json::Value::as_bool),
        Some(true)
    );
    assert!(!service.is_warm());
}

// -- poison and rebuild ---------------------------------------------------------------------------

/// A factory whose first `n` builds poison and whose later builds are clean models.
fn poison_then_clean(poison_builds: usize) -> (RuntimeFactory, Arc<AtomicUsize>) {
    let count = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&count);
    let factory: RuntimeFactory = Arc::new(move || {
        let index = counter.fetch_add(1, Ordering::SeqCst);
        if index < poison_builds {
            Ok(Wrapped::runtime(Behaviour::PoisonsOnPrefill))
        } else {
            Ok(Wrapped::runtime(Behaviour::ClaimsToBeAModel))
        }
    });
    (factory, count)
}

#[test]
fn a_poisoned_runtime_rebuilds_and_the_answer_says_degraded() {
    // Control first: a clean model answers *not* degraded, so `degraded` below is attributable to
    // the rebuild and to nothing else.
    let clean = Service::with_factory(
        config(Duration::from_secs(600), Duration::from_secs(10), 4),
        Arc::new(|| Ok(Wrapped::runtime(Behaviour::ClaimsToBeAModel))),
    );
    let control = clean.handle_line(&common::line(&common::sample_request()));
    let control: serde_json::Value = serde_json::from_slice(&control).expect("parses");
    assert_eq!(
        control.get("degraded").and_then(serde_json::Value::as_bool),
        Some(false),
        "the control must be clean"
    );

    let (factory, builds) = poison_then_clean(1);
    let service = Service::with_factory(
        config(Duration::from_secs(600), Duration::from_secs(10), 4),
        factory,
    );
    let reply = service.handle_line(&common::line(&common::sample_request()));
    assert_eq!(
        common::status_of(&reply),
        "ok",
        "a poisoned runtime rebuilds and answers: {}",
        String::from_utf8_lossy(&reply)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&reply).expect("parses");
    assert_eq!(
        parsed.get("degraded").and_then(serde_json::Value::as_bool),
        Some(true),
        "an answer from a rebuilt-after-poison runtime is degraded"
    );
    assert!(
        parsed
            .pointer("/slots/verdict/degraded")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        "every slot of a degraded answer carries the flag"
    );

    let status = service.status();
    assert_eq!(status.rebuilds_after_poison, 1);
    assert_eq!(builds.load(Ordering::SeqCst), 2, "exactly one rebuild");
    assert!(status.degraded, "the warm runtime is still the rebuilt one");
}

#[test]
fn a_runtime_that_poisons_again_after_a_rebuild_stops_rather_than_looping() {
    let (factory, builds) = poison_then_clean(usize::MAX);
    let service = Service::with_factory(
        config(Duration::from_secs(600), Duration::from_secs(10), 4),
        factory,
    );
    let reply = service.handle_line(&common::line(&common::sample_request()));
    assert_eq!(common::status_of(&reply), "error");
    let parsed: serde_json::Value = serde_json::from_slice(&reply).expect("parses");
    assert_eq!(
        parsed.pointer("/error/kind").and_then(serde_json::Value::as_str),
        Some("rebuild_failed")
    );
    assert_eq!(
        builds.load(Ordering::SeqCst),
        2,
        "one rebuild attempt, then stop: a hot loop is not a retry policy"
    );
}

#[test]
fn a_rebuild_that_cannot_build_is_reported_as_a_rebuild_failure() {
    let count = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&count);
    let factory: RuntimeFactory = Arc::new(move || {
        let index = counter.fetch_add(1, Ordering::SeqCst);
        if index == 0 {
            Ok(Wrapped::runtime(Behaviour::PoisonsOnPrefill))
        } else {
            Err(BackendError::Unavailable {
                detail: "the device is gone".to_string(),
            })
        }
    });
    let service = Service::with_factory(
        config(Duration::from_secs(600), Duration::from_secs(10), 4),
        factory,
    );
    let reply = service.handle_line(&common::line(&common::sample_request()));
    let parsed: serde_json::Value = serde_json::from_slice(&reply).expect("parses");
    assert_eq!(common::status_of(&reply), "error");
    assert_eq!(
        parsed.pointer("/error/kind").and_then(serde_json::Value::as_str),
        Some("rebuild_failed"),
        "a failed rebuild is its own error, not the original build's"
    );
    assert!(service.status().build_failures >= 1);
}

// -- concurrency ------------------------------------------------------------------------------------

#[test]
fn a_slow_request_does_not_wedge_the_agent() {
    const SLOW: Duration = Duration::from_millis(600);
    let service = Arc::new(Service::with_factory(
        config(Duration::from_secs(600), Duration::from_secs(30), 8),
        Arc::new(|| Ok(Wrapped::runtime(Behaviour::SlowForTask { delay: SLOW }))),
    ));

    // Warm it first, so the slow request's time is its own and not a shared cold start.
    assert_eq!(
        common::status_of(&service.handle_line(&common::line(&common::sample_request()))),
        "ok"
    );

    let mut slow_payload = common::sample_request();
    slow_payload["task"] = json!("slow-task");
    let slow_payload = common::line(&slow_payload);

    let slow_service = Arc::clone(&service);
    let slow = std::thread::spawn(move || slow_service.handle_line(&slow_payload));

    assert!(
        wait_for(Duration::from_secs(5), || service.status().in_flight == 1),
        "the slow request never reached the runtime"
    );

    let fast_payload = common::line(&common::sample_request());
    let started = Instant::now();
    for _ in 0..4 {
        assert_eq!(common::status_of(&service.handle_line(&fast_payload)), "ok");
    }
    let fast_elapsed = started.elapsed();

    let slow_reply = slow.join().expect("the slow request finished");
    assert_eq!(common::status_of(&slow_reply), "ok");
    assert!(
        fast_elapsed < SLOW / 2,
        "four fast requests took {fast_elapsed:?} while one slow request was in flight; the agent \
         is serialising on something"
    );
    assert_eq!(
        service.status().in_flight,
        0,
        "the in-flight counter must return to zero"
    );
}

#[test]
fn requests_beyond_the_concurrency_cap_are_told_so() {
    const SLOW: Duration = Duration::from_millis(500);
    let service = Arc::new(Service::with_factory(
        config(Duration::from_secs(600), Duration::from_secs(30), 1),
        Arc::new(|| Ok(Wrapped::runtime(Behaviour::SlowForTask { delay: SLOW }))),
    ));
    assert_eq!(
        common::status_of(&service.handle_line(&common::line(&common::sample_request()))),
        "ok"
    );

    let mut slow_payload = common::sample_request();
    slow_payload["task"] = json!("slow-task");
    let slow_payload = common::line(&slow_payload);
    let slow_service = Arc::clone(&service);
    let slow = std::thread::spawn(move || slow_service.handle_line(&slow_payload));

    assert!(
        wait_for(Duration::from_secs(5), || service.status().in_flight == 1),
        "the slow request never reached the runtime"
    );

    let reply = service.handle_line(&common::line(&common::sample_request()));
    let parsed: serde_json::Value = serde_json::from_slice(&reply).expect("parses");
    assert_eq!(common::status_of(&reply), "error");
    assert_eq!(
        parsed.pointer("/error/kind").and_then(serde_json::Value::as_str),
        Some("overloaded"),
        "over the cap the agent says so rather than queueing without bound"
    );

    assert_eq!(common::status_of(&slow.join().expect("joins")), "ok");
}

// -- deadlines ---------------------------------------------------------------------------------------

#[test]
fn a_request_that_outruns_its_deadline_is_stopped_and_names_the_limit() {
    const LIMIT: Duration = Duration::from_millis(50);
    let service = Service::with_factory(
        config(Duration::from_secs(600), LIMIT, 4),
        Arc::new(|| {
            Ok(Wrapped::runtime(Behaviour::SlowForTask {
                delay: Duration::from_millis(400),
            }))
        }),
    );
    // Warm first: a cold start is not the request's own time.
    assert_eq!(
        common::status_of(&service.handle_line(&common::line(&common::sample_request()))),
        "ok"
    );

    let mut payload = common::sample_request();
    payload["task"] = json!("slow-task");
    let reply = service.handle_line(&common::line(&payload));
    let parsed: serde_json::Value = serde_json::from_slice(&reply).expect("parses");
    assert_eq!(common::status_of(&reply), "error");
    assert_eq!(
        parsed.pointer("/error/kind").and_then(serde_json::Value::as_str),
        Some("deadline_exceeded")
    );
    assert_eq!(
        parsed
            .pointer("/error/limit_ms")
            .and_then(serde_json::Value::as_u64),
        Some(LIMIT.as_millis() as u64),
        "the error must name the configured limit, not the zero left when it fires"
    );
}

// -- backend failures that must not read as answers -----------------------------------------------

#[test]
fn a_backend_returning_the_wrong_shape_fails_rather_than_answering() {
    for (behaviour, expected) in [
        (Behaviour::WrongLogitShape, "logit_shape_mismatch"),
        (Behaviour::NonFiniteLogit, "non_finite_logit"),
    ] {
        let runtime = Wrapped::runtime(behaviour);
        let request = common::validated(&common::sample_request());
        let response = runtime.answer(&request, None);
        assert_eq!(response.caller_reading(), CallerReading::BackendFailed);
        match response {
            Response::Error(envelope) => assert_eq!(envelope.error.kind(), expected),
            other => panic!("expected {expected}, got {other:?}"),
        }
    }
}

#[test]
fn the_reference_backend_refuses_to_serve_unless_it_is_switched_on() {
    let disabled = qd_runtime::reference::ReferenceBackend::new(false);
    use qd_runtime::backend::DecisionBackend;
    match disabled.prefill("anything") {
        Err(BackendError::ReferenceBackendNotEnabled { name }) => {
            assert_eq!(name, qd_runtime::reference::REFERENCE_BACKEND_NAME);
        }
        other => panic!("a disabled reference backend must refuse, got {other:?}"),
    }
}

// -- counters ---------------------------------------------------------------------------------------

#[test]
fn the_status_counters_separate_answers_refusals_and_failures() {
    let service = common::reference_service();

    assert_eq!(
        common::status_of(&service.handle_line(&common::line(&common::sample_request()))),
        "ok"
    );
    let mut bad = common::sample_request();
    bad["schema_version"] = json!(99);
    assert_eq!(
        common::status_of(&service.handle_line(&common::line(&bad))),
        "refused"
    );
    assert_eq!(
        common::status_of(&common::backendless_service().handle_line(&common::line(
            &common::sample_request()
        ))),
        "error"
    );

    let status = service.status();
    assert_eq!(status.requests_served, 1);
    assert_eq!(status.refusals, 1);
    assert_eq!(status.backend_errors, 0);
}
