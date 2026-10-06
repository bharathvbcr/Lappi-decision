//! `qd oneshot` — one request on stdin, one reply on stdout, exit.
//!
//! Two callers, two reasons:
//!
//! * **DevType** talks to the socket. It is interactive and the agent is already warm.
//! * **dcverify** tries the socket and falls back to this sidecar, because a verify run has to work
//!   with nothing else running. A verify that could only run when an unrelated agent happened to be
//!   up would be a verify whose result depended on the machine's mood.
//!
//! The fallback is only safe because both paths end in the same [`Service::handle_line`]. Nothing
//! in this module inspects a request or builds a reply.

use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::deadline::DeadlineStream;
use crate::service::Service;
use crate::wire::MAX_PAYLOAD_BYTES;

/// Which path answered. Reported on stderr so a verify log records it, and returned so a caller can
/// assert on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServedBy {
    /// A running `qd serve` answered.
    Socket,
    /// This process answered in-process.
    Sidecar,
}

impl ServedBy {
    pub fn as_str(self) -> &'static str {
        match self {
            ServedBy::Socket => "socket",
            ServedBy::Sidecar => "sidecar",
        }
    }
}

/// Read one line, bounded.
///
/// Stops at the first newline, or at EOF, or one byte past `cap` — whichever comes first, so a
/// caller that pipes a 2 GB file at `qd oneshot` gets an error, not an allocation. An empty vector
/// means EOF with nothing read.
///
/// Takes a [`BufRead`] rather than a [`Read`] on purpose: a reply arrives on a connection that
/// stays open, so reading to EOF would block until the timeout, and a fresh `BufReader` per call
/// would swallow whatever the previous call had buffered past its newline. Both matter — the second
/// is what makes several requests on one connection work.
///
/// Over the cap this reports `InvalidData` without measuring how far over. The socket server
/// measures it and answers [`crate::refusal::Refusal::PayloadOverCap`] with a real number; this
/// path is `qd oneshot` reading its own stdin, where the caller already knows what it sent.
pub fn read_one_line<R: BufRead>(reader: R, cap: usize) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut limited = reader.take(cap as u64 + 1);
    let read = limited.read_until(b'\n', &mut buf)?;
    if read == 0 {
        return Ok(buf);
    }
    if buf.last() == Some(&b'\n') {
        buf.pop();
    }
    if buf.len() > cap {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("the line is longer than the {cap}-byte payload cap"),
        ));
    }
    Ok(buf)
}

/// Send one line to a running agent and read one reply.
///
/// `timeout` bounds the **whole exchange** — connect, write and the complete reply — not each
/// read, so an agent that drips its reply cannot hold the caller past it.
///
/// Every error here is an `io::Error`, and every one of them means "fall back to the sidecar". A
/// **refusal or a backend error inside the reply is not an error here** — it is the agent's answer,
/// and retrying it in-process would be asking a second opinion of the same code.
pub fn ask_over_socket(
    socket: &Path,
    line: &[u8],
    timeout: Duration,
) -> io::Result<Vec<u8>> {
    if line.len() > MAX_PAYLOAD_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "request is over the payload cap",
        ));
    }
    // std refuses a zero read/write timeout, but only after the connect: the agent would see a
    // connection that sends nothing, and the caller a fallback blamed on an agent "not answering".
    if timeout.is_zero() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "a zero socket timeout cannot bound the exchange; refused before connecting",
        ));
    }
    let deadline = Instant::now() + timeout;
    let stream = UnixStream::connect(socket)?;
    let mut stream = DeadlineStream::new(stream, deadline);

    // One write of the framed line: a request and its newline are one message, and a write split
    // across a deadline would leave the agent holding half a line.
    let mut framed = Vec::with_capacity(line.len() + 1);
    framed.extend_from_slice(line);
    framed.push(b'\n');
    // A failed write is not yet the outcome. An agent over its connection cap writes `overloaded`
    // the moment it accepts and closes (`serve::refuse_connection`), so the request can meet a
    // closed socket while that reply sits unread in the receive buffer. Read it; only a socket
    // with nothing to read reports the write's error.
    let written = stream.write_all(&framed).and_then(|()| stream.flush());

    let mut reader = BufReader::new(stream);
    let reply = match read_one_line(&mut reader, MAX_PAYLOAD_BYTES) {
        Ok(reply) => reply,
        Err(read_error) => return Err(written.err().unwrap_or(read_error)),
    };
    if reply.is_empty() {
        return Err(written.err().unwrap_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the agent closed the connection without replying",
            )
        }));
    }
    Ok(reply)
}

/// Answer one line: the socket when there is one, otherwise in-process.
///
/// `socket` is `None` for a pure sidecar run. When it is `Some` and the socket cannot be reached,
/// the reason is written to stderr — a silent fallback would hide an agent that has been down for a
/// week behind results that still look fine.
pub fn ask(
    service: &Service,
    socket: Option<&Path>,
    line: &[u8],
    timeout: Duration,
) -> (Vec<u8>, ServedBy) {
    if let Some(path) = socket {
        match ask_over_socket(path, line, timeout) {
            Ok(reply) => return (reply, ServedBy::Socket),
            Err(error) => {
                eprintln!(
                    "qd oneshot: {} is not answering ({error}); falling back to the in-process \
                     sidecar",
                    path.display()
                );
            }
        }
    }
    (service.handle_line(line), ServedBy::Sidecar)
}

/// Write a reply as one JSON line. The framing matches `qd serve`'s, so the two modes' bytes on the
/// wire are identical and not merely equivalent.
pub fn write_reply<W: Write>(writer: &mut W, reply: &[u8]) -> io::Result<()> {
    writer.write_all(reply)?;
    writer.write_all(b"\n")?;
    writer.flush()
}
