//! **`qd serve` and `qd oneshot` share one contract.**
//!
//! Not "should", and not a comment above the two entry points: this file sends the identical bytes
//! down a live Unix socket and through the in-process sidecar, and compares the replies byte for
//! byte — including the framing newline, because a caller reading lines sees that too.
//!
//! It matters because dcverify uses **both**: it tries the socket and falls back to the sidecar,
//! since a verify run has to work with nothing else running. If the two paths could differ, a
//! verify result would depend on whether an unrelated agent happened to be warm.
//!
//! The three outcomes are covered separately — an answer, a refusal and a backend failure — because
//! it is entirely possible to share a code path for the happy case and diverge on the errors, which
//! is the divergence that matters most.

mod common;

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use qd_runtime::oneshot::{ask, ask_over_socket, ServedBy};
use qd_runtime::serve::{ServeOptions, Server};
use qd_runtime::service::{Service, ServiceConfig};
use qd_runtime::runtime::RuntimeConfig;
use qd_runtime::render::RenderCaps;

fn config(with_backend: bool) -> ServiceConfig {
    ServiceConfig {
        runtime: RuntimeConfig {
            caps: RenderCaps::DEFAULT,
            enable_reference_backend: with_backend,
        },
        idle_timeout: Duration::from_secs(600),
        request_timeout: Duration::from_secs(10),
        max_in_flight: 8,
    }
}

/// Run a server on a fresh socket for the duration of `body`, then stop it and join the thread.
fn with_server<R>(tag: &str, cfg: ServiceConfig, body: impl FnOnce(&std::path::Path) -> R) -> R {
    let path = common::temp_socket_path(tag);
    let service = Arc::new(Service::new(cfg));
    let server = Server::bind(Arc::clone(&service), ServeOptions::new(path.clone()))
        .expect("the test socket binds");
    let shutdown = server.shutdown_handle();
    let handle = std::thread::spawn(move || server.run());

    let result = body(&path);

    shutdown.stop();
    let joined = handle.join();
    assert!(joined.is_ok(), "the server thread panicked");
    let _ = std::fs::remove_file(&path);
    result
}

/// What a caller sees from each mode, framed exactly as it arrives on the wire.
fn both_modes(tag: &str, cfg: ServiceConfig, payload: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let socket_reply = with_server(tag, cfg.clone(), |path| {
        let mut reply = ask_over_socket(path, payload, Duration::from_secs(10))
            .expect("the socket answers");
        reply.push(b'\n');
        reply
    });

    let sidecar = Service::new(cfg);
    let mut sidecar_reply = sidecar.handle_line(payload);
    sidecar_reply.push(b'\n');

    (socket_reply, sidecar_reply)
}

fn assert_identical(tag: &str, cfg: ServiceConfig, payload: &[u8], expected_status: &str) {
    let (socket, sidecar) = both_modes(tag, cfg, payload);
    assert_eq!(
        String::from_utf8_lossy(&socket),
        String::from_utf8_lossy(&sidecar),
        "the two modes disagreed on {}",
        String::from_utf8_lossy(&payload[..payload.len().min(80)])
    );
    assert_eq!(common::status_of(&socket[..socket.len() - 1]), expected_status);
    assert_eq!(
        socket.last(),
        Some(&b'\n'),
        "both modes frame a reply with exactly one newline"
    );
}

#[test]
fn an_answer_is_byte_identical_in_both_modes() {
    assert_identical(
        "ok",
        config(true),
        &common::line(&common::sample_request()),
        "ok",
    );
}

#[test]
fn a_refusal_is_byte_identical_in_both_modes() {
    let mut bad = common::sample_request();
    bad["schema_version"] = serde_json::json!(99);
    assert_identical("refused", config(true), &common::line(&bad), "refused");
}

#[test]
fn a_backend_failure_is_byte_identical_in_both_modes() {
    // No backend enabled: the shipping default. Both modes must say `unavailable`, not `noul`.
    assert_identical(
        "error",
        config(false),
        &common::line(&common::sample_request()),
        "error",
    );
}

#[test]
fn a_malformed_line_is_byte_identical_in_both_modes() {
    assert_identical("malformed", config(true), b"{not json", "refused");
}

#[test]
fn an_unknown_control_op_is_byte_identical_in_both_modes() {
    assert_identical("unknown-op", config(true), br#"{"op":"restart"}"#, "refused");
}

#[test]
fn a_line_carrying_both_op_and_schema_version_is_refused_in_both_modes() {
    assert_identical(
        "ambiguous",
        config(true),
        br#"{"op":"status","schema_version":1}"#,
        "refused",
    );
}

/// The dcverify path end to end: with no agent running, `ask` falls back and answers the same.
#[test]
fn the_dcverify_fallback_answers_exactly_what_the_socket_would_have() {
    let payload = common::line(&common::sample_request());

    let over_socket = with_server("fallback-live", config(true), |path| {
        ask_over_socket(path, &payload, Duration::from_secs(10)).expect("the socket answers")
    });

    // Now with nothing listening: the same path, the same request.
    let absent = common::temp_socket_path("fallback-absent");
    let sidecar = Service::new(config(true));
    let (reply, served_by) = ask(
        &sidecar,
        Some(absent.as_path()),
        &payload,
        Duration::from_millis(500),
    );
    assert_eq!(served_by, ServedBy::Sidecar, "nothing was listening");
    assert_eq!(
        String::from_utf8_lossy(&reply),
        String::from_utf8_lossy(&over_socket),
        "the fallback answered differently from the agent it replaced"
    );
}

#[test]
fn the_socket_is_preferred_when_one_is_listening() {
    let payload = common::line(&common::sample_request());
    with_server("prefer-socket", config(true), |path| {
        let sidecar = Service::new(config(true));
        let (_, served_by) = ask(&sidecar, Some(path), &payload, Duration::from_secs(10));
        assert_eq!(served_by, ServedBy::Socket);
    });
}

// -- the socket's own bounds -----------------------------------------------------------------

#[test]
fn a_line_over_the_payload_cap_is_refused_and_the_connection_closed() {
    with_server("over-cap", config(true), |path| {
        let stream = std::os::unix::net::UnixStream::connect(path).expect("connects");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("sets a read timeout");
        let mut writer = stream.try_clone().expect("clones");
        // One line, comfortably over the cap, written in chunks so nothing here allocates it twice.
        let chunk = vec![b'x'; 64 * 1024];
        let mut written = 0usize;
        writer.write_all(br#"{"schema_version":1,"task":""#).expect("writes");
        while written < qd_runtime::wire::MAX_PAYLOAD_BYTES + 128 * 1024 {
            writer.write_all(&chunk).expect("writes");
            written += chunk.len();
        }
        writer.write_all(b"\"}\n").expect("writes");
        writer.flush().expect("flushes");

        let reply = qd_runtime::oneshot::read_one_line(
            std::io::BufReader::new(stream),
            qd_runtime::wire::MAX_PAYLOAD_BYTES,
        )
        .expect("reads a reply");
        assert_eq!(common::status_of(&reply), "refused");
        assert_eq!(common::refusal_kind(&reply), "payload_over_cap");
    });
}

#[test]
fn several_requests_travel_down_one_connection() {
    with_server("pipelined", config(true), |path| {
        let stream = std::os::unix::net::UnixStream::connect(path).expect("connects");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("sets a read timeout");
        let mut writer = stream.try_clone().expect("clones");
        let payload = common::line(&common::sample_request());
        for _ in 0..3 {
            writer.write_all(&payload).expect("writes");
            writer.write_all(b"\n").expect("writes");
        }
        writer.flush().expect("flushes");

        let mut reader = std::io::BufReader::new(stream);
        let mut seen = Vec::new();
        for _ in 0..3 {
            let reply = qd_runtime::oneshot::read_one_line(
                &mut reader,
                qd_runtime::wire::MAX_PAYLOAD_BYTES,
            )
            .expect("reads a reply");
            assert_eq!(common::status_of(&reply), "ok");
            seen.push(reply);
        }
        assert_eq!(seen[0], seen[1], "the same request answered differently");
        assert_eq!(seen[1], seen[2]);
    });
}

#[test]
fn binding_over_a_live_socket_is_refused() {
    with_server("double-bind", config(true), |path| {
        let second = Service::new(config(true));
        let error = Server::bind(Arc::new(second), ServeOptions::new(path))
            .expect_err("binding over a live agent must fail");
        assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);
    });
}

#[test]
fn a_stale_socket_file_from_a_killed_agent_is_replaced() {
    let path = common::temp_socket_path("stale");
    {
        // Bind and drop without running: the file is left behind, exactly as a killed agent does.
        let listener = std::os::unix::net::UnixListener::bind(&path).expect("binds");
        drop(listener);
    }
    assert!(path.exists(), "the fixture must leave a socket file behind");
    let service = Arc::new(Service::new(config(true)));
    let server = Server::bind(service, ServeOptions::new(path.clone()))
        .expect("a dead socket file is cleared, not treated as a live agent");
    drop(server);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_shutdown_op_stops_the_agent() {
    let path = common::temp_socket_path("shutdown-op");
    let service = Arc::new(Service::new(config(true)));
    let server =
        Server::bind(Arc::clone(&service), ServeOptions::new(path.clone())).expect("binds");
    let handle = std::thread::spawn(move || server.run());

    let reply = ask_over_socket(&path, br#"{"op":"shutdown"}"#, Duration::from_secs(10))
        .expect("the agent acknowledges");
    assert_eq!(common::status_of(&reply), "ack");

    // Nothing else connects from here on. `accept` has no timeout, so an agent that only sets a
    // flag would sit blocked until some unrelated client happened to dial it — which is not
    // "shut down", and is exactly what this test missed until a live run caught it.
    let deadline = Instant::now() + Duration::from_secs(10);
    let finished = Arc::new(AtomicBool::new(false));
    while Instant::now() < deadline && !finished.load(Ordering::SeqCst) {
        if handle.is_finished() {
            finished.store(true, Ordering::SeqCst);
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        finished.load(Ordering::SeqCst),
        "a shutdown op must stop the accept loop"
    );
    let _ = handle.join();
    let _ = std::fs::remove_file(&path);
}
