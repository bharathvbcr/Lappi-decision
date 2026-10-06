//! **`qd serve` under load and under hostile clients.**
//!
//! Every calling app (`docs/caller-contract.md`) shares one agent, so one misbehaving caller must
//! not be able to take it from the others. These tests drive a real `Server` on a real socket with
//! the reference backend and assert on what every *other* client sees.
//!
//! * many concurrent clients through a small connection cap: each gets a whole, parseable reply
//!   (an answer, or `overloaded`), none hangs, and the agent still answers afterwards;
//! * a connection that drips a request a byte at a time loses its slot once its line has taken
//!   longer than `read_timeout`, measured from the line's first byte, not from the last byte;
//! * connections that connect and send nothing fill the cap, the next client is told
//!   `overloaded` at once, and the agent recovers when they go;
//! * torn lines, garbage and rapid connect/close churn leave the agent serving.

mod common;

use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use qd_runtime::oneshot::ask_over_socket;
use qd_runtime::render::RenderCaps;
use qd_runtime::runtime::RuntimeConfig;
use qd_runtime::serve::{ServeOptions, Server};
use qd_runtime::service::{Service, ServiceConfig};
use serde_json::Value;

fn config() -> ServiceConfig {
    ServiceConfig {
        runtime: RuntimeConfig { caps: RenderCaps::DEFAULT, enable_reference_backend: true },
        idle_timeout: Duration::from_secs(600),
        request_timeout: Duration::from_secs(10),
        max_in_flight: 8,
    }
}

fn with_server<R>(
    tag: &str,
    max_connections: usize,
    read_timeout: Duration,
    body: impl FnOnce(&Path) -> R,
) -> R {
    with_service_server(
        tag,
        Service::new(config()),
        max_connections,
        read_timeout,
        Duration::from_secs(5),
        body,
    )
}

fn with_service_server<R>(
    tag: &str,
    service: Service,
    max_connections: usize,
    read_timeout: Duration,
    write_timeout: Duration,
    body: impl FnOnce(&Path) -> R,
) -> R {
    let path = common::temp_socket_path(tag);
    let service = Arc::new(service);
    let mut options = ServeOptions::new(path.clone());
    options.max_connections = max_connections;
    options.read_timeout = read_timeout;
    options.write_timeout = write_timeout;
    let server = Server::bind(Arc::clone(&service), options).expect("the test socket binds");
    let shutdown = server.shutdown_handle();
    let handle = std::thread::spawn(move || server.run());
    let result = body(&path);
    shutdown.stop();
    assert!(handle.join().is_ok(), "the server thread panicked");
    let _ = std::fs::remove_file(&path);
    result
}

fn ping_ok(path: &Path) -> bool {
    match ask_over_socket(path, br#"{"op":"ping"}"#, Duration::from_secs(5)) {
        Ok(reply) => serde_json::from_slice::<Value>(&reply).is_ok(),
        Err(_) => false,
    }
}

/// A reply is one of the contract's three envelopes, or a ping's ack; nothing else is acceptable.
fn classify(reply: &[u8]) -> String {
    let v: Value = serde_json::from_slice(reply).unwrap_or_else(|e| {
        panic!("a reply that is not JSON: {e}: {}", String::from_utf8_lossy(reply))
    });
    match v.get("status").and_then(Value::as_str) {
        Some("ok") => "ok".into(),
        Some("refused") => format!("refused:{}", v.pointer("/refusal/kind").and_then(Value::as_str).unwrap_or("?")),
        Some("error") => format!("error:{}", v.pointer("/error/kind").and_then(Value::as_str).unwrap_or("?")),
        other => panic!("a reply with status {other:?}: {v}"),
    }
}

#[test]
fn a_reply_slower_than_the_write_timeout_to_compute_is_still_delivered() {
    // `write_timeout` bounds *writing* a reply, not the model's time to produce one; that is
    // `request_timeout`'s job. A write deadline started when the request line arrived would expire
    // during a long decode and drop a reply the model finished in time (a 128 KiB context, say).
    // Here the backend takes 1 s and the write timeout is 300 ms: the reply must still arrive.
    let service = Service::with_factory(
        config(),
        Arc::new(|| Ok(common::Wrapped::runtime(common::Behaviour::SlowForTask { delay: Duration::from_secs(1) }))),
    );
    with_service_server(
        "stress-slowreply",
        service,
        4,
        Duration::from_secs(5),
        Duration::from_millis(300),
        |path| {
            let mut slow = common::sample_request();
            slow["task"] = serde_json::json!("slow-task");
            let started = Instant::now();
            let reply = ask_over_socket(path, &common::line(&slow), Duration::from_secs(10))
                .unwrap_or_else(|e| panic!("the slow reply never arrived ({:?} in): {e}", started.elapsed()));
            assert_eq!(classify(&reply), "ok", "{}", String::from_utf8_lossy(&reply));
            assert!(started.elapsed() >= Duration::from_secs(1), "the backend's delay was not exercised");
            // And the connection is still usable for the next request.
            assert!(ping_ok(path));
        },
    );
}

#[test]
fn many_concurrent_clients_through_a_small_cap_each_get_a_whole_reply() {
    const CLIENTS: usize = 96;
    const PER_CLIENT: usize = 5;
    let request = common::line(&common::sample_request());
    with_server("stress-many", 8, Duration::from_secs(10), |path| {
        let started = Instant::now();
        let outcomes: Vec<Vec<String>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..CLIENTS)
                .map(|_| {
                    let request = request.clone();
                    scope.spawn(move || {
                        (0..PER_CLIENT)
                            .map(|_| match ask_over_socket(path, &request, Duration::from_secs(20)) {
                                Ok(reply) => classify(&reply),
                                Err(e) => format!("io:{:?}", e.kind()),
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("client thread")).collect()
        });
        let elapsed = started.elapsed();
        let flat: Vec<&String> = outcomes.iter().flatten().collect();
        assert_eq!(flat.len(), CLIENTS * PER_CLIENT);
        // Three acceptable outcomes: an answer, `overloaded` (over the connection or request cap),
        // or a connect refused by a full listen backlog (macOS refuses at once; a caller reads it
        // as `unavailable`). A timeout, a torn reply or any other error is a failure.
        let count = |want: &str| flat.iter().filter(|s| s.as_str() == want).count();
        let ok = count("ok");
        let overloaded = count("error:overloaded");
        let refused = count("io:ConnectionRefused");
        let other: Vec<_> = flat
            .iter()
            .filter(|s| !matches!(s.as_str(), "ok" | "error:overloaded" | "io:ConnectionRefused"))
            .collect();
        assert!(
            other.is_empty(),
            "saw {other:?} (ok {ok}, overloaded {overloaded}, refused {refused})"
        );
        eprintln!("burst: ok {ok}, overloaded {overloaded}, connect refused {refused}");
        assert!(ok > 0, "some requests were answered (ok {ok}, overloaded {overloaded})");
        assert!(elapsed < Duration::from_secs(60), "the burst took {elapsed:?}");
        assert!(ping_ok(path), "the agent still answers after the burst");
    });
}

#[test]
fn a_dripping_request_loses_its_slot_once_the_line_outlasts_the_read_timeout() {
    // One connection slot. A client drips its request a byte every 200 ms, each gap well inside
    // the 600 ms read timeout. Measured per read, it would hold the only slot forever; measured
    // from the line's first byte, the agent drops it after ~600 ms and serves the next caller.
    with_server("stress-drip", 1, Duration::from_millis(600), |path| {
        let mut dripper = UnixStream::connect(path).expect("connects");
        let started = Instant::now();
        let mut closed_at = None;
        for byte in br#"{"op":"ping"   ...this never ends"#.iter().cycle().take(40) {
            if dripper.write_all(std::slice::from_ref(byte)).is_err() {
                closed_at = Some(started.elapsed());
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
            // A dropped connection shows as EOF on read, an error on the next write, or (macOS)
            // EINVAL from setsockopt on a socket the peer has already shut down.
            if dripper.set_read_timeout(Some(Duration::from_millis(1))).is_err() {
                closed_at = Some(started.elapsed());
                break;
            }
            let mut probe = [0u8; 256];
            match dripper.read(&mut probe) {
                Ok(0) => {
                    closed_at = Some(started.elapsed());
                    break;
                }
                Err(e)
                    if e.kind() != std::io::ErrorKind::WouldBlock
                        && e.kind() != std::io::ErrorKind::TimedOut =>
                {
                    closed_at = Some(started.elapsed());
                    break;
                }
                Ok(_) | Err(_) => {}
            }
        }
        let closed_at = closed_at.expect("the agent never dropped a request that dripped for 8 s");
        assert!(
            closed_at < Duration::from_millis(2000),
            "the agent held a dripping line for {closed_at:?}; the bound is the 600 ms read timeout from the line's first byte"
        );
        assert!(ping_ok(path), "the slot was released and the next caller is served");
    });
}

#[test]
fn silent_connections_fill_the_cap_and_the_next_caller_is_told_at_once() {
    with_server("stress-silent", 4, Duration::from_millis(800), |path| {
        let silent: Vec<UnixStream> =
            (0..4).map(|_| UnixStream::connect(path).expect("connects")).collect();
        // Let the accept loop register all four.
        std::thread::sleep(Duration::from_millis(100));
        let started = Instant::now();
        let reply = ask_over_socket(path, br#"{"op":"ping"}"#, Duration::from_secs(5)).expect("a reply");
        assert_eq!(classify(&reply), "error:overloaded");
        assert!(started.elapsed() < Duration::from_millis(500), "told in {:?}", started.elapsed());
        drop(silent);
        // The silent four also time out on their own; either way the cap frees.
        let deadline = Instant::now() + Duration::from_secs(5);
        while !ping_ok(path) {
            assert!(Instant::now() < deadline, "the agent did not recover after the silent connections left");
            std::thread::sleep(Duration::from_millis(50));
        }
    });
}

#[test]
fn torn_lines_garbage_and_churn_leave_the_agent_serving() {
    with_server("stress-churn", 16, Duration::from_millis(500), |path| {
        // Half a request, then hang up; then connect and close at once, 200 times. A burst can fill
        // the listen backlog, and on macOS a full backlog refuses the connect immediately
        // (ECONNREFUSED, measured 2026-10-06; GAP-UNIX-CONNECT-HAS-NO-TIMEOUT-2026-10-06). Callers
        // read that as `unavailable` / `connect_refused`, so it is counted here, not failed on:
        // the property under test is that the agent is still serving afterwards.
        let mut refused = 0usize;
        for i in 0..250 {
            match UnixStream::connect(path) {
                Ok(mut s) if i < 50 => {
                    let _ = s.write_all(br#"{"schema_version":1,"task":"#);
                }
                Ok(s) => drop(s),
                Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => refused += 1,
                Err(e) => panic!("connect #{i} failed with something other than a refusal: {e}"),
            }
        }
        assert!(refused < 250, "every churn connect was refused; the agent never accepted");
        let deadline = Instant::now() + Duration::from_secs(5);
        while !ping_ok(path) {
            assert!(Instant::now() < deadline, "the agent did not recover after the churn ({refused} refused)");
            std::thread::sleep(Duration::from_millis(50));
        }
        // Garbage lines: each gets a typed refusal, never a hang or a crash.
        let garbage: [&[u8]; 6] = [
            b"\x00\x01\x02",
            b"\xff\xfe\xfd",
            b"[]",
            b"null",
            b"{\"op\":\"ping\",\"schema_version\":1}",
            b"{\"schema_version\":1}",
        ];
        for g in garbage {
            let reply = ask_over_socket(path, g, Duration::from_secs(5)).expect("a reply to garbage");
            let kind = classify(&reply);
            assert!(kind.starts_with("refused:"), "{:?} -> {kind}", String::from_utf8_lossy(g));
        }
        // Several requests down one connection still work after all that.
        let s = UnixStream::connect(path).expect("connects");
        s.set_read_timeout(Some(Duration::from_secs(5))).expect("timeout");
        let mut w = s.try_clone().expect("clone");
        let mut r = BufReader::new(s);
        for _ in 0..10 {
            w.write_all(b"{\"op\":\"ping\"}\n").expect("writes");
            let mut line = String::new();
            r.read_line(&mut line).expect("reads");
            assert!(line.ends_with('\n') && serde_json::from_str::<Value>(line.trim()).is_ok(), "{line:?}");
        }
        assert!(ping_ok(path));
    });
}
