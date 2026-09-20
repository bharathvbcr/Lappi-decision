//! `qd serve` — one warm runtime behind a Unix socket, JSON lines.
//!
//! This is the launchd agent DevType talks to. It owns no request logic: every line goes to
//! [`Service::handle_line`], the same call `qd oneshot` makes.
//!
//! Everything here is `std`. `UnixListener`, `set_read_timeout` and `Condvar::wait_timeout` cover
//! the whole server, so there is no async runtime, no socket crate and no timer crate in the
//! dependency list — `crates/qd-runtime/Cargo.toml` says so and this module is why it can.
//!
//! # Bounds
//!
//! * one line is at most [`crate::wire::MAX_PAYLOAD_BYTES`], and a longer one is refused **and the
//!   connection closed**, because a half-read line would resynchronise the protocol onto the middle
//!   of a payload;
//! * concurrent connections are capped, and a connection over the cap is told so and closed;
//! * concurrent *requests* are capped by [`Service`], which answers [`BackendError::Overloaded`];
//! * every read and every write has a timeout, so a client that connects and stops does not hold a
//!   thread forever.

use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::refusal::{BackendError, Refusal};
use crate::schema::Response;
use crate::service::Service;
use crate::wire::MAX_PAYLOAD_BYTES;

/// How far past the cap the server will read while measuring an over-cap line before it gives up
/// and reports a lower bound. Memory stays O(1) either way; this only bounds the time spent
/// counting.
const OVERSIZE_SCAN_CEILING: usize = MAX_PAYLOAD_BYTES * 4;

#[derive(Debug, Clone)]
pub struct ServeOptions {
    pub socket_path: PathBuf,
    pub max_connections: usize,
    pub read_timeout: Duration,
    pub write_timeout: Duration,
}

impl ServeOptions {
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.into(),
            max_connections: 64,
            read_timeout: Duration::from_secs(120),
            write_timeout: Duration::from_secs(30),
        }
    }
}

/// A bound listener. Split from [`Server::run`] so a caller — and a test — knows the socket exists
/// before anything tries to connect to it.
pub struct Server {
    listener: UnixListener,
    service: Arc<Service>,
    options: ServeOptions,
    stop: Arc<AtomicBool>,
    live_connections: Arc<AtomicUsize>,
}

/// Stops a running [`Server`] from another thread.
#[derive(Clone)]
pub struct ShutdownHandle {
    stop: Arc<AtomicBool>,
    socket_path: PathBuf,
    service: Arc<Service>,
}

impl ShutdownHandle {
    /// Set the stop flag and wake the accept loop.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.service.request_shutdown();
        wake_accept_loop(&self.socket_path);
    }
}

/// Unblock a thread sitting in `UnixListener::accept`.
///
/// `accept` has no timeout, so setting a flag does not stop the agent: it would sit blocked until
/// some unrelated client happened to dial it, which could be never. Dialling our own socket is what
/// wakes it — the accept loop checks the flag *before* it does anything with the connection, so the
/// wake connection is accepted, ignored and dropped, and the loop exits.
///
/// Both ways out of the loop need this: [`ShutdownHandle::stop`] from another thread, and a
/// `shutdown` control op arriving on a connection. The op is the one that is easy to miss, because
/// the connection carrying it closes correctly whether or not the listener ever notices, so a test
/// that dials the socket again afterwards passes against an agent that never shuts down.
///
/// Best effort: if the listener is already gone the connect fails, which is the state this was
/// trying to reach anyway.
fn wake_accept_loop(path: &Path) {
    let _ = UnixStream::connect(path);
}

impl std::fmt::Debug for Server {
    /// By hand because `Service` holds a trait object behind a mutex. Prints the bounds and the
    /// path, which is what a failure message needs.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Server")
            .field("socket_path", &self.options.socket_path)
            .field("max_connections", &self.options.max_connections)
            .field("live_connections", &self.live_connections)
            .field("stopped", &self.stop.load(Ordering::SeqCst))
            .finish()
    }
}

impl Server {
    /// Bind the socket.
    ///
    /// A leftover socket file from a killed agent is removed, but only after proving nobody is
    /// listening on it: if a connect succeeds, another agent owns this path and binding over it
    /// would silently steal DevType's traffic.
    pub fn bind(service: Arc<Service>, options: ServeOptions) -> io::Result<Self> {
        clear_stale_socket(&options.socket_path)?;
        if let Some(parent) = options.socket_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let listener = UnixListener::bind(&options.socket_path)?;
        // A per-user agent socket. Other local users have no business talking to it.
        std::fs::set_permissions(
            &options.socket_path,
            std::fs::Permissions::from_mode(0o600),
        )?;
        Ok(Self {
            listener,
            service,
            options,
            stop: Arc::new(AtomicBool::new(false)),
            live_connections: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn socket_path(&self) -> &Path {
        &self.options.socket_path
    }

    pub fn shutdown_handle(&self) -> ShutdownHandle {
        ShutdownHandle {
            stop: Arc::clone(&self.stop),
            socket_path: self.options.socket_path.clone(),
            service: Arc::clone(&self.service),
        }
    }

    /// Accept until stopped. Starts the idle-eviction reaper and joins it on the way out.
    pub fn run(self) -> io::Result<()> {
        let reaper_stop = Arc::clone(&self.stop);
        let reaper_service = Arc::clone(&self.service);
        let reaper = std::thread::Builder::new()
            .name("qd-idle-reaper".to_string())
            .spawn(move || Service::run_idle_reaper(reaper_service, reaper_stop))?;

        let mut connections: Vec<JoinHandle<()>> = Vec::new();
        for incoming in self.listener.incoming() {
            if self.stop.load(Ordering::SeqCst) || self.service.shutdown_requested() {
                break;
            }
            let stream = match incoming {
                Ok(stream) => stream,
                // One bad accept is not a reason to take the agent down; a broken listener will
                // keep returning and the loop exits on the stop flag.
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    eprintln!("qd serve: accept failed: {error}");
                    continue;
                }
            };

            connections.retain(|h| !h.is_finished());
            if self.live_connections.load(Ordering::SeqCst) >= self.options.max_connections {
                refuse_connection(stream, self.options.max_connections);
                continue;
            }

            let service = Arc::clone(&self.service);
            let options = self.options.clone();
            let live = Arc::clone(&self.live_connections);
            live.fetch_add(1, Ordering::SeqCst);
            match std::thread::Builder::new()
                .name("qd-conn".to_string())
                .spawn(move || {
                    serve_connection(service, stream, &options);
                    live.fetch_sub(1, Ordering::SeqCst);
                }) {
                Ok(handle) => connections.push(handle),
                Err(error) => {
                    self.live_connections.fetch_sub(1, Ordering::SeqCst);
                    eprintln!("qd serve: could not spawn a connection thread: {error}");
                }
            }
        }

        self.stop.store(true, Ordering::SeqCst);
        self.service.request_shutdown();
        for handle in connections {
            // A connection thread that panicked has already been isolated by its own unwinding;
            // the agent still needs to shut down cleanly.
            let _ = handle.join();
        }
        let _ = reaper.join();
        let _ = std::fs::remove_file(&self.options.socket_path);
        Ok(())
    }
}

fn clear_stale_socket(path: &Path) -> io::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    match UnixStream::connect(path) {
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::AddrInUse,
            format!(
                "{} is a live socket: another qd agent is already listening there. Refusing to \
                 bind over it, because that would silently take over its callers",
                path.display()
            ),
        )),
        Err(_) => std::fs::remove_file(path),
    }
}

fn refuse_connection(stream: UnixStream, limit: usize) {
    let reply = match serde_json::to_vec(&Response::failed(BackendError::Overloaded { limit })) {
        Ok(bytes) => bytes,
        Err(_) => return,
    };
    let mut stream = stream;
    let _ = stream.write_all(&reply);
    let _ = stream.write_all(b"\n");
    let _ = stream.flush();
}

fn serve_connection(service: Arc<Service>, stream: UnixStream, options: &ServeOptions) {
    if stream.set_read_timeout(Some(options.read_timeout)).is_err()
        || stream
            .set_write_timeout(Some(options.write_timeout))
            .is_err()
    {
        return;
    }
    let read_half = match stream.try_clone() {
        Ok(half) => half,
        Err(_) => return,
    };
    let mut reader = BufReader::new(read_half);
    let mut writer = BufWriter::new(stream);

    loop {
        match read_line_bounded(&mut reader, MAX_PAYLOAD_BYTES) {
            Ok(LineRead::Eof) => break,
            Ok(LineRead::Line(line)) => {
                if line.iter().all(u8::is_ascii_whitespace) {
                    continue;
                }
                let reply = service.handle_line(&line);
                if write_line(&mut writer, &reply).is_err() {
                    break;
                }
                if service.shutdown_requested() {
                    break;
                }
            }
            Ok(LineRead::OverCap { measured }) => {
                let reply = match serde_json::to_vec(&Response::refused(Refusal::PayloadOverCap {
                    cap: MAX_PAYLOAD_BYTES,
                    actual: measured,
                })) {
                    Ok(bytes) => bytes,
                    Err(_) => break,
                };
                let _ = write_line(&mut writer, &reply);
                // The rest of that line is not a line. Closing is the only way to avoid parsing
                // the middle of a payload as the next request.
                break;
            }
            Err(_) => break,
        }
    }
    let _ = writer.flush();
    drop(writer);
    if service.shutdown_requested() {
        wake_accept_loop(&options.socket_path);
    }
}

fn write_line(writer: &mut BufWriter<UnixStream>, reply: &[u8]) -> io::Result<()> {
    writer.write_all(reply)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

enum LineRead {
    Line(Vec<u8>),
    Eof,
    OverCap { measured: usize },
}

/// Read one newline-terminated line, holding at most `cap` bytes in memory.
///
/// On an over-cap line the remainder is drained with a fixed scratch buffer so the caller can be
/// told how big it actually was. If the line runs past [`OVERSIZE_SCAN_CEILING`] the count returned
/// is a lower bound; either way the connection is closed by the caller, so a partially consumed
/// line can never be parsed as the next request.
fn read_line_bounded<R: BufRead>(reader: &mut R, cap: usize) -> io::Result<LineRead> {
    let mut buf: Vec<u8> = Vec::new();
    let read = reader
        .by_ref()
        .take(cap as u64 + 1)
        .read_until(b'\n', &mut buf)?;
    if read == 0 {
        return Ok(LineRead::Eof);
    }
    if buf.last() == Some(&b'\n') {
        buf.pop();
        if buf.len() > cap {
            return Ok(LineRead::OverCap { measured: buf.len() });
        }
        return Ok(LineRead::Line(buf));
    }
    if buf.len() <= cap {
        // EOF without a trailing newline: a complete final line.
        return Ok(LineRead::Line(buf));
    }

    let mut measured = buf.len();
    let mut scratch = [0u8; 8192];
    while measured < OVERSIZE_SCAN_CEILING {
        let n = reader.read(&mut scratch)?;
        if n == 0 {
            break;
        }
        match scratch[..n].iter().position(|b| *b == b'\n') {
            Some(index) => {
                measured += index;
                break;
            }
            None => measured += n,
        }
    }
    Ok(LineRead::OverCap { measured })
}
