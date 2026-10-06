//! Reading an input file whole, bounded. The one owner of "how big may this file be": every
//! whole-file read in this crate goes through [`read_bounded`].
//!
//! A size check on `metadata(path)` followed by `read(path)` is not a bound: a symlink to
//! `/dev/zero` reports length 0 and then reads without end, and a file that grows between the
//! two calls is read past the bound. So the file is opened once, the handle must be a regular
//! file, and the read itself stops one byte past the bound.

use std::io::Read;
use std::path::Path;

/// A config, fetch record, view record, survey manifest or pinned data file: small JSON. The
/// largest in use is the v6 fetch record at about 93 KB.
pub const MAX_RECORD_BYTES: u64 = 64 << 20;

/// The open handle of `path`, refused unless it is a regular file. `what` names the input.
///
/// The type is checked twice: on the path before opening, because opening a FIFO blocks until
/// a writer appears, and on the open handle, because the path can be swapped in between.
pub fn open_regular(path: &Path, what: &str) -> Result<(std::fs::File, u64), String> {
    let not_regular = |meta: &std::fs::Metadata| {
        format!(
            "{what} {}: not a regular file ({:?}); a directory, device or pipe is refused",
            path.display(),
            meta.file_type()
        )
    };
    let before = std::fs::metadata(path).map_err(|e| format!("{what} {}: {e}", path.display()))?;
    if !before.is_file() {
        return Err(not_regular(&before));
    }
    let file = std::fs::File::open(path).map_err(|e| format!("{what} {}: {e}", path.display()))?;
    let meta = file
        .metadata()
        .map_err(|e| format!("{what} {}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(not_regular(&meta));
    }
    Ok((file, meta.len()))
}

/// `PATH.partial` beside `PATH`, where an output is written before it is renamed into place.
/// Appended, never `with_extension`, which would turn `pool.v5` and `pool.v6` into one
/// `pool.partial` and let two outputs share it (`containment::write_dir` learned this first).
pub fn partial_path(path: &Path) -> Result<std::path::PathBuf, String> {
    let mut name = path
        .file_name()
        .ok_or_else(|| format!("{} names no file or directory", path.display()))?
        .to_os_string();
    name.push(".partial");
    Ok(path.with_file_name(name))
}

/// Create `path` exclusively (it must not exist), write `bytes`, and sync. For a file inside a
/// `.partial` directory that is renamed into place whole; [`write_new_file`] for a file alone.
pub fn write_synced_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    f.write_all(bytes)
        .and_then(|()| f.sync_all())
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// `bytes` at `path`, whole or not at all: written to `PATH.partial` (created exclusively, so a
/// concurrent writer's partial is never touched), synced, renamed. Refused if `path` exists.
/// A failure after this call created the partial removes it.
pub fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if path.exists() {
        return Err(format!(
            "{} exists; refusing to overwrite it",
            path.display()
        ));
    }
    let partial = partial_path(path)?;
    write_synced_new(&partial, bytes)?;
    std::fs::rename(&partial, path).map_err(|e| {
        let e = format!("{}: {e}", path.display());
        match std::fs::remove_file(&partial) {
            Ok(()) => e,
            Err(r) => format!("{e}; and {} was left: {r}", partial.display()),
        }
    })
}

/// A new directory `out_dir`, whole or not at all: `fill` writes into `DIR.partial`, which is
/// then renamed into place. Refused before anything is written if `out_dir` or the partial
/// exists, or the parent is missing. A failure after this call created the partial, in `fill`
/// or in the rename, removes it, so no half directory is left; a partial this call did not
/// create (a concurrent run's) is never removed.
pub fn write_new_dir(
    out_dir: &Path,
    fill: impl FnOnce(&Path) -> Result<(), String>,
) -> Result<(), String> {
    let partial = partial_path(out_dir)?;
    for p in [out_dir, partial.as_path()] {
        if p.exists() {
            return Err(format!("{} exists; refusing to overwrite it", p.display()));
        }
    }
    std::fs::create_dir(&partial).map_err(|e| format!("{}: {e}", partial.display()))?;
    let done = fill(partial.as_path()).and_then(|()| {
        std::fs::rename(&partial, out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))
    });
    done.map_err(|e| match std::fs::remove_dir_all(&partial) {
        Ok(()) => e,
        Err(r) => format!("{e}; and {} was left: {r}", partial.display()),
    })
}

/// The bytes of `path`, refused past `max` bytes, whatever its metadata said.
pub fn read_bounded(path: &Path, max: u64, what: &str) -> Result<Vec<u8>, String> {
    let (file, len) = open_regular(path, what)?;
    let over = |n: u64| format!("{what} {}: {n} bytes; the bound is {max}", path.display());
    if len > max {
        return Err(over(len));
    }
    // `len` is at most `max` here, and `max` is a compile-time bound well inside usize.
    let mut bytes = Vec::with_capacity(usize::try_from(len).map_err(|e| e.to_string())?);
    file.take(max.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|e| format!("{what} {}: {e}", path.display()))?;
    let got = bytes.len() as u64;
    if got > max {
        return Err(format!("{} (it grew while being read)", over(got)));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "qd-prep-files-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_file_at_the_bound_is_read_and_one_byte_over_is_refused() {
        let dir = scratch("bound");
        let at = dir.join("at");
        std::fs::write(&at, vec![b'x'; 10]).unwrap();
        assert_eq!(read_bounded(&at, 10, "input").unwrap().len(), 10);
        let err = read_bounded(&at, 9, "input").unwrap_err();
        assert!(err.contains("10 bytes; the bound is 9"), "{err}");
    }

    #[test]
    fn a_symlink_to_an_endless_device_is_refused_without_reading_it() {
        let dir = scratch("zero");
        let link = dir.join("catalog.json");
        std::os::unix::fs::symlink("/dev/zero", &link).unwrap();
        let err = read_bounded(&link, 1 << 20, "catalog").unwrap_err();
        assert!(err.contains("not a regular file"), "{err}");
    }

    /// Before `write_new_dir`, `decisions::write_out` and its copies left `DIR.partial` behind
    /// on any failure after creating it, and the next run refused to start on it.
    #[test]
    fn a_failed_directory_write_leaves_nothing_and_a_concurrent_partial_is_kept() {
        let dir = scratch("newdir");
        let out = dir.join("pool");
        let err = write_new_dir(&out, |p| {
            write_synced_new(&p.join("a.jsonl"), b"{}\n")?;
            Err("the fill failed".to_owned())
        })
        .unwrap_err();
        assert_eq!(err, "the fill failed");
        assert!(!out.exists() && !partial_path(&out).unwrap().exists());

        // The rename fails: a dangling symlink passes `exists()` and blocks the rename.
        let blocked = dir.join("blocked");
        std::os::unix::fs::symlink(dir.join("nowhere"), &blocked).unwrap();
        let err = write_new_dir(&blocked, |p| write_synced_new(&p.join("a"), b"x")).unwrap_err();
        assert!(err.contains("blocked"), "{err}");
        assert!(!partial_path(&blocked).unwrap().exists());

        // Another run's partial: refused, and left exactly as it was.
        let theirs = dir.join("theirs");
        let their_partial = partial_path(&theirs).unwrap();
        std::fs::create_dir(&their_partial).unwrap();
        std::fs::write(their_partial.join("a"), b"x").unwrap();
        let err = write_new_dir(&theirs, |_| Ok(())).unwrap_err();
        assert!(err.contains("exists; refusing"), "{err}");
        assert!(their_partial.join("a").exists());

        // And the whole write lands whole.
        let ok = dir.join("ok");
        write_new_dir(&ok, |p| write_synced_new(&p.join("a"), b"x")).unwrap();
        assert_eq!(std::fs::read(ok.join("a")).unwrap(), b"x");
        assert!(!partial_path(&ok).unwrap().exists());
    }

    #[test]
    fn a_new_file_is_written_whole_and_never_over_an_existing_one() {
        let dir = scratch("newfile");
        let f = dir.join("m.json");
        write_new_file(&f, b"{}\n").unwrap();
        assert_eq!(std::fs::read(&f).unwrap(), b"{}\n");
        assert!(write_new_file(&f, b"[]\n").unwrap_err().contains("exists"));
        assert_eq!(std::fs::read(&f).unwrap(), b"{}\n");
        assert!(!partial_path(&f).unwrap().exists());
    }

    #[test]
    fn two_outputs_differing_only_after_a_dot_get_two_partials() {
        let a = partial_path(Path::new("/p/pool.v5")).unwrap();
        let b = partial_path(Path::new("/p/pool.v6")).unwrap();
        assert_eq!(a, Path::new("/p/pool.v5.partial"));
        assert_ne!(a, b);
        assert_eq!(
            partial_path(Path::new("/p/synth-email-v1")).unwrap(),
            Path::new("/p/synth-email-v1.partial")
        );
        assert!(partial_path(Path::new("/")).is_err());
    }

    #[test]
    fn a_directory_and_a_missing_path_are_refused_by_name() {
        let dir = scratch("dir");
        let err = read_bounded(&dir, 1 << 20, "--config").unwrap_err();
        assert!(
            err.contains("--config") && err.contains("not a regular file"),
            "{err}"
        );
        let err = read_bounded(&dir.join("absent.json"), 1 << 20, "--config").unwrap_err();
        assert!(err.contains("absent.json"), "{err}");
    }
}
