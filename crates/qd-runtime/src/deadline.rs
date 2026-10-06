//! One deadline for a whole exchange, on a `UnixStream`.
//!
//! `set_read_timeout` / `set_write_timeout` bound a single `read(2)` / `write(2)`. A peer that moves
//! one byte just inside the bound, over and over, resets it each time, so "at most N ms" becomes "as
//! long as the peer likes". Both ends of the socket had that hole: the client
//! (`oneshot::ask_over_socket`, `tests/client_deadline.rs`) and the agent's line reader
//! (`serve::serve_connection`, `tests/serve_stress.rs`). This type is the one fix for both: each
//! operation is given only what is left of the deadline, and an operation that would start with
//! nothing left fails `TimedOut` without touching the socket.

use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

pub(crate) struct DeadlineStream {
    stream: UnixStream,
    deadline: Instant,
}

impl DeadlineStream {
    pub(crate) fn new(stream: UnixStream, deadline: Instant) -> Self {
        Self { stream, deadline }
    }

    /// Start a new exchange on the same connection (the agent does this once per line).
    pub(crate) fn set_deadline(&mut self, deadline: Instant) {
        self.deadline = deadline;
    }

    fn remaining(&self) -> io::Result<Duration> {
        let left = self.deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(deadline_passed());
        }
        Ok(left)
    }
}

fn deadline_passed() -> io::Error {
    io::Error::new(io::ErrorKind::TimedOut, "the socket exchange ran past its deadline")
}

/// macOS reports an expired socket timeout as `EAGAIN` (`WouldBlock`), Linux as `TimedOut`. A
/// caller asks one question — did the deadline pass? — so both read as `TimedOut`.
fn timed_out(error: io::Error) -> io::Error {
    if error.kind() == io::ErrorKind::WouldBlock {
        deadline_passed()
    } else {
        error
    }
}

/// Apply a timeout, tolerating the one failure that is not a failure.
///
/// macOS refuses `SO_RCVTIMEO` / `SO_SNDTIMEO` with `EINVAL` (`InvalidInput`) on a socket the peer
/// has already shut down. That is exactly the state in which the agent has written its reply (an
/// `overloaded` refusal, say) and closed: failing here would throw away a reply sitting in the
/// receive buffer (`tests/serve_stress.rs`, the burst test, saw it as `io:InvalidInput`). On such a
/// socket a read returns the buffered bytes or EOF at once and a write fails at once, so carrying
/// on cannot block; and the timeout already on the socket, set by an earlier operation of this
/// exchange, is never longer than the deadline was then.
fn apply(result: io::Result<()>) -> io::Result<()> {
    match result {
        Err(e) if e.kind() == io::ErrorKind::InvalidInput => Ok(()),
        other => other,
    }
}

impl Read for DeadlineStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let left = self.remaining()?;
        apply(self.stream.set_read_timeout(Some(left)))?;
        self.stream.read(buf).map_err(timed_out)
    }
}

impl Write for DeadlineStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let left = self.remaining()?;
        apply(self.stream.set_write_timeout(Some(left)))?;
        self.stream.write(buf).map_err(timed_out)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stream.flush()
    }
}
