//! The one seam through which the door and the shard reader touch file *contents*.
//!
//! Every byte the reader takes from a manifest or a shard set comes through a
//! [`ShardFiles`], so a test can wrap [`OsFiles`] and count what was read from each file.
//! That is how "refused **before** any token byte is read" is a measured statement rather
//! than an argument: the fail-first tests in `tests/door.rs` assert zero bytes read from
//! `tokens.u32` (and zero bytes from any shard file for the path-level refusals).
//!
//! Path *resolution* for the held-out check does not go through here: it consults the real
//! filesystem's symlinks, because a held-out tree reached through a link is exactly what
//! layer 1 exists to catch.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Content access for manifests and shard sets.
pub trait ShardFiles: Send + Sync {
    /// The whole file, refused (`InvalidData`) if it is longer than `max_bytes`.
    fn read(&self, path: &Path, max_bytes: u64) -> io::Result<Vec<u8>>;
    /// The file's length in bytes, without reading it.
    fn len(&self, path: &Path) -> io::Result<u64>;
    /// Whether a regular file exists at `path`.
    fn is_file(&self, path: &Path) -> bool;
    /// A handle for positional reads (`tokens.u32`).
    fn open_ranged(&self, path: &Path) -> io::Result<Box<dyn RangedRead>>;
}

/// Positional reads from one open file.
pub trait RangedRead: Send + Sync {
    /// Fill `buf` from `offset`, or fail; a short read is an error.
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;
}

/// Open `path` and prove the handle is the regular file now at `path`, not something a link
/// was swapped in for.
///
/// Rule 3's per-entry sweep checks a path and the open comes later, so a member swapped for a
/// symlink in between was read unchecked (a check-then-open race). Closed here at the open:
/// after opening, `symlink_metadata(path)` must show a non-link whose `(dev, ino)` equals the
/// opened handle's. A link at the path (whatever it pointed at when it was opened) or a path
/// that changed hands since the open is refused as `PermissionDenied`, naming rule 3. No
/// dependency: std's `symlink_metadata` stands in for `O_NOFOLLOW`, whose value is per-platform
/// and would need `libc`.
fn open_regular(path: &Path) -> io::Result<File> {
    let file = File::open(path)?;
    let opened = file.metadata()?;
    let on_disk = std::fs::symlink_metadata(path)?;
    let same = on_disk.file_type().is_file() && opened.is_file() && same_file(&opened, &on_disk);
    if !same {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "{}: not the regular file the rule-3 sweep checked (a symlink, or replaced \
                 between the check and the open); CLAUDE.md rule 3",
                path.display()
            ),
        ));
    }
    Ok(file)
}

#[cfg(unix)]
fn same_file(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    a.dev() == b.dev() && a.ino() == b.ino()
}

#[cfg(not(unix))]
fn same_file(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    a.len() == b.len()
}

/// The real filesystem. Every file it opens must be a regular file, not a link
/// ([`open_regular`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct OsFiles;

impl ShardFiles for OsFiles {
    fn read(&self, path: &Path, max_bytes: u64) -> io::Result<Vec<u8>> {
        let file = open_regular(path)?;
        let len = file.metadata()?.len();
        if len > max_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} is {len} bytes, over the {max_bytes}-byte bound",
                    path.display()
                ),
            ));
        }
        let mut buf = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
        file.take(max_bytes.saturating_add(1))
            .read_to_end(&mut buf)?;
        if buf.len() as u64 > max_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} grew past the {max_bytes}-byte bound while read",
                    path.display()
                ),
            ));
        }
        Ok(buf)
    }

    fn len(&self, path: &Path) -> io::Result<u64> {
        Ok(open_regular(path)?.metadata()?.len())
    }

    fn is_file(&self, path: &Path) -> bool {
        path.is_file()
    }

    fn open_ranged(&self, path: &Path) -> io::Result<Box<dyn RangedRead>> {
        Ok(Box::new(OsRanged::open(path)?))
    }
}

#[cfg(unix)]
struct OsRanged(File);

#[cfg(unix)]
impl OsRanged {
    fn open(path: &Path) -> io::Result<Self> {
        Ok(Self(open_regular(path)?))
    }
}

#[cfg(unix)]
impl RangedRead for OsRanged {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(&self.0, buf, offset)
    }
}

#[cfg(not(unix))]
struct OsRanged(std::sync::Mutex<File>);

#[cfg(not(unix))]
impl OsRanged {
    fn open(path: &Path) -> io::Result<Self> {
        Ok(Self(std::sync::Mutex::new(open_regular(path)?)))
    }
}

#[cfg(not(unix))]
impl RangedRead for OsRanged {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        use std::io::{Seek, SeekFrom};
        let mut file = self
            .0
            .lock()
            .map_err(|_| io::Error::other("token file lock poisoned"))?;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(buf)
    }
}
