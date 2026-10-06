//! Rule 3's path marker, with one owner in this crate. A held-out set lives under a path segment
//! named for it (`crates/qd-train/src/held_out.rs::DEFAULT_HELD_OUT_PATH_MARKERS`, the trainer's
//! door). An evaluation set's output must carry one ([`check_held_out_path`]), and a producer of
//! training rows refuses to read an input that carries one ([`refuse_training_input`]).
//!
//! Decontamination targets are not training inputs. A producer scans its rows against held-out
//! sets so that none of them leaks into training, so `--target` files may sit under a marker.
//!
//! Segments match ASCII case-insensitively, as `natural-bugs` always has. The trainer's door also
//! folds the few non-ASCII code points whose case folding is ASCII; this crate does not depend on
//! `qd-train` (it would pull the serving stack into a data tool), so that door stays the last
//! check before any training read.

use std::path::{Component, Path};

/// `crates/qd-train/src/held_out.rs::DEFAULT_HELD_OUT_PATH_MARKERS`.
pub const HELD_OUT_PATH_MARKERS: [&str; 3] = ["heldout", "held_out", "held-out"];

/// The first segment of `path` that equals a held-out marker, ASCII case-insensitively.
fn marker_segment(path: &Path) -> Option<String> {
    path.components().find_map(|c| match c {
        Component::Normal(seg) => {
            let s = seg.to_string_lossy();
            HELD_OUT_PATH_MARKERS
                .iter()
                .any(|m| s.eq_ignore_ascii_case(m))
                .then(|| s.into_owned())
        }
        _ => None,
    })
}

/// Refuse an output path that `qd-train`'s path check would not refuse: no `..`, and some
/// segment equal (ASCII case-insensitively) to a held-out marker.
pub fn check_held_out_path(path: &Path) -> Result<(), String> {
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(format!(
            "{} has a '..' segment; a held-out marker before it would not mark where the \
             files land",
            path.display()
        ));
    }
    if marker_segment(path).is_some() {
        Ok(())
    } else {
        Err(format!(
            "{} has no path segment in {HELD_OUT_PATH_MARKERS:?}. This set must never be \
             trained on, and a path without a marker is one qd-train's path check would let \
             a training process open (CLAUDE.md rule 3).",
            path.display()
        ))
    }
}

/// Refuse to read `path` as an input to training rows when it, or the file it resolves to,
/// sits under a held-out marker. `what` names the input in the error (`--config`, `catalog`).
///
/// Both spellings are checked: the path as given, and its canonical form when it exists, so a
/// symlink from an unmarked directory into a held-out one is refused too. A path that does not
/// exist is refused by the read that follows, which names it.
pub fn refuse_training_input(path: &Path, what: &str) -> Result<(), String> {
    let refuse = |seen: &Path, seg: String| {
        Err(format!(
            "{what} {} is held-out data (path segment {seg:?} at {}): a producer of training \
             rows never reads it (CLAUDE.md rule 3). Held-out sets enter only as `--target` \
             decontamination sets.",
            path.display(),
            seen.display()
        ))
    };
    if let Some(seg) = marker_segment(path) {
        return refuse(path, seg);
    }
    match std::fs::canonicalize(path) {
        Ok(real) => match marker_segment(&real) {
            Some(seg) => refuse(&real, seg),
            None => Ok(()),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("{what} {}: {e}", path.display())),
    }
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
            "qd-prep-heldout-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn an_output_path_needs_a_held_out_segment_and_no_dot_dot() {
        let ok = |p: &str| check_held_out_path(Path::new(p)).is_ok();
        assert!(ok("/x/v6/heldout/natural-bugs"));
        assert!(ok("/x/HeldOut/natural-bugs"));
        assert!(ok("data/held-out/nb"));
        assert!(ok("data/held_out/nb"));
        assert!(!ok("/x/v6/natural-bugs"));
        assert!(!ok("/x/heldoutset/nb"), "a marker is a whole segment");
        assert!(!ok("/x/heldout/../train/nb"), "'..' escapes the marker");
    }

    #[test]
    fn a_training_input_under_a_marker_is_refused_by_any_spelling() {
        for p in [
            "/d/heldout/tssb.zip",
            "/d/HELDOUT/x.jsonl",
            "rel/held-out/x.json",
            "/d/Held_Out/x",
        ] {
            let err = refuse_training_input(Path::new(p), "--config").unwrap_err();
            assert!(
                err.contains("held-out data") && err.contains("--config"),
                "{p}: {err}"
            );
        }
        assert!(refuse_training_input(Path::new("/d/heldoutset/x"), "c").is_ok());
        assert!(refuse_training_input(Path::new("/d/missing/x.json"), "c").is_ok());
    }

    #[test]
    fn a_symlink_into_a_held_out_directory_is_refused() {
        let dir = scratch("symlink");
        let held = dir.join("heldout");
        std::fs::create_dir_all(&held).unwrap();
        std::fs::write(held.join("rows.jsonl"), b"{}\n").unwrap();
        let link = dir.join("innocent.jsonl");
        std::os::unix::fs::symlink(held.join("rows.jsonl"), &link).unwrap();
        let err = refuse_training_input(&link, "catalog").unwrap_err();
        assert!(
            err.contains("innocent.jsonl") && err.contains("catalog"),
            "{err}"
        );
        let plain = dir.join("plain.jsonl");
        std::fs::write(&plain, b"{}\n").unwrap();
        assert!(refuse_training_input(&plain, "catalog").is_ok());
    }
}
