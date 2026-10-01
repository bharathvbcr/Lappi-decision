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

/// The real filesystem.
#[derive(Clone, Copy, Debug, Default)]
pub struct OsFiles;

impl ShardFiles for OsFiles {
    fn read(&self, path: &Path, max_bytes: u64) -> io::Result<Vec<u8>> {
        let file = File::open(path)?;
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
        Ok(std::fs::metadata(path)?.len())
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
        Ok(Self(File::open(path)?))
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
        Ok(Self(std::sync::Mutex::new(File::open(path)?)))
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
