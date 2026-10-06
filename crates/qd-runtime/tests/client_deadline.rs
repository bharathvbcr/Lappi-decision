//! **A socket timeout bounds the whole exchange, not each read.**
//!
//! `set_read_timeout` is a per-`read(2)` bound. A peer that sends one byte just inside it, over and
//! over, resets it every time, so a caller that asked for "at most 500 ms" could wait for as long as
//! the peer liked. For DevType that is a palette keystroke hung behind a stuck agent; for dcverify it
//! is a verify run that never reaches its fallback. Both callers pass one number and read it as the
//! bound on the whole exchange, so the client has to make it one.
//!
//! The fake agent here is a plain `UnixListener` that never runs the service: what is under test is
//! the client's framing and deadline, not anything the runtime answers.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use qd_runtime::oneshot::ask_over_socket;

fn scratch_socket(tag: &str) -> PathBuf {
    // Short: sun_path is 104 bytes on macOS, and a temp dir path can already use most of it.
    let dir = std::env::temp_dir().join(format!("qdcd-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir.join("s.sock")
}

/// Accept one connection, read the request line, then send `reply` one byte at a time with
/// `gap` between bytes. Returns when the reply is sent or the client hangs up.
fn dripping_agent(path: &PathBuf, reply: &'static [u8], gap: Duration) -> thread::JoinHandle<()> {
    let listener = UnixListener::bind(path).expect("binds");
    thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accepts");
        let mut reader = BufReader::new(stream.try_clone().expect("clones"));
        let mut line = Vec::new();
        let _ = reader.read_until(b'\n', &mut line);
        let mut writer = stream;
        for byte in reply {
            thread::sleep(gap);
            if writer.write_all(std::slice::from_ref(byte)).is_err() {
                return;
            }
            let _ = writer.flush();
        }
    })
}

#[test]
fn a_peer_that_drips_bytes_cannot_hold_the_caller_past_its_timeout() {
    let path = scratch_socket("drip");
    // 40 bytes at 150 ms each is 6 s of reply, every gap well inside a 500 ms per-read timeout.
    let reply: &'static [u8] = b"{\"status\":\"error\",\"schema_version\":1}\n";
    let agent = dripping_agent(&path, reply, Duration::from_millis(150));

    let timeout = Duration::from_millis(500);
    let started = Instant::now();
    let outcome = ask_over_socket(&path, br#"{"op":"ping"}"#, timeout);
    let elapsed = started.elapsed();

    assert!(
        outcome.is_err(),
        "a reply that cannot finish inside the timeout is a failed exchange, got {outcome:?}"
    );
    assert!(
        elapsed < Duration::from_millis(1500),
        "the 500 ms timeout must bound the whole exchange; the caller was held {elapsed:?}"
    );
    let err = outcome.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::TimedOut, "{err}");
    let _ = agent.join();
}

#[test]
fn a_prompt_reply_inside_the_deadline_is_still_read_whole() {
    let path = scratch_socket("prompt");
    let reply: &'static [u8] = b"{\"status\":\"error\"}\n";
    let agent = dripping_agent(&path, reply, Duration::from_millis(1));
    let got = ask_over_socket(&path, br#"{"op":"ping"}"#, Duration::from_secs(5)).expect("answers");
    assert_eq!(got, b"{\"status\":\"error\"}");
    agent.join().expect("agent thread");
}

#[test]
fn a_reply_written_before_the_agent_closed_is_read_even_if_the_request_write_fails() {
    // `serve::refuse_connection`'s shape: write `overloaded` at accept, close, never read. Repeat
    // so the race (request written before or after the close) is hit both ways; every round must
    // come back with the reply, never an I/O error.
    let reply: &'static [u8] = b"{\"status\":\"error\",\"schema_version\":1,\"error\":{\"kind\":\"overloaded\"}}\n";
    for round in 0..50 {
        let path = scratch_socket(&format!("early{round}"));
        let listener = UnixListener::bind(&path).expect("binds");
        let agent = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accepts");
            stream.write_all(reply).expect("writes");
            let _ = stream.shutdown(std::net::Shutdown::Both);
        });
        // A request big enough that it cannot all fit in the socket buffer before the close.
        let request = vec![b' '; 512 * 1024];
        let got = ask_over_socket(&path, &request, Duration::from_secs(5));
        agent.join().expect("agent thread");
        let got = got.unwrap_or_else(|e| panic!("round {round}: the agent's reply was lost: {e}"));
        assert_eq!(got, &reply[..reply.len() - 1], "round {round}");
    }
}

#[test]
fn an_agent_that_accepts_and_never_writes_times_out_at_the_deadline() {
    let path = scratch_socket("silent");
    let listener = UnixListener::bind(&path).expect("binds");
    let holder = thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accepts");
        thread::sleep(Duration::from_secs(3));
        drop(stream);
    });
    let started = Instant::now();
    let outcome = ask_over_socket(&path, br#"{"op":"ping"}"#, Duration::from_millis(300));
    let elapsed = started.elapsed();
    assert!(outcome.is_err());
    assert!(elapsed < Duration::from_millis(1200), "held {elapsed:?}");
    holder.join().expect("holder thread");
}
