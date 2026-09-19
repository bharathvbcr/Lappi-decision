//! Artifact integrity: do the files on this box match the ones the build was pinned to?
//!
//! This is the half of the preflight that **does** work on any host, CUDA or not, and
//! it is the half the Python checker cannot do safely — verifying the environment from
//! inside the Python environment being verified means a broken install fails the check
//! for the wrong reason, and an import-time side effect can change what is measured.
//!
//! The runtime already refuses a tokenizer/weight/head/label-set hash that does not
//! match the build (`docs/schema-api.md`). This runs the same check *before* a 34-hour
//! block starts, rather than at the first request.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::tristate::TriState;

/// One pinned artifact: a path and the digest it is expected to have.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnedArtifact {
    pub name: String,
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub artifacts: Vec<PinnedArtifact>,
}

/// Stream a file through SHA-256. Streamed rather than read whole: a weights shard is
/// gigabytes, and a preflight that needs the file in RAM would OOM on exactly the box
/// where it matters most.
pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Verify every artifact in the manifest.
///
/// A missing manifest is `NotRun`: nothing was pinned, so nothing was checked. It is
/// emphatically **not** a pass — "no manifest" and "all artifacts match" must not
/// produce the same result, which is the whole tri-state discipline in one case.
pub fn check_artifacts(manifest_path: Option<&Path>, root: &Path) -> TriState {
    let Some(manifest_path) = manifest_path else {
        return TriState::not_run(
            "no artifact manifest supplied (--artifacts), so no artifact was verified",
        );
    };
    if !manifest_path.exists() {
        return TriState::not_run(format!(
            "artifact manifest {} does not exist; nothing was verified",
            manifest_path.display()
        ));
    }

    let raw = match std::fs::read_to_string(manifest_path) {
        Ok(r) => r,
        Err(e) => {
            return TriState::not_run(format!("could not read {}: {e}", manifest_path.display()))
        }
    };
    let manifest: ArtifactManifest = match serde_json::from_str(&raw) {
        Ok(m) => m,
        Err(e) => return TriState::fail(format!("artifact manifest is malformed: {e}")),
    };
    if manifest.artifacts.is_empty() {
        return TriState::not_run(format!(
            "artifact manifest {} pins zero artifacts; an empty manifest verifies nothing",
            manifest_path.display()
        ));
    }

    let mut mismatched = Vec::new();
    let mut missing = Vec::new();
    let mut ok = 0usize;

    for artifact in &manifest.artifacts {
        let full = if artifact.path.is_absolute() {
            artifact.path.clone()
        } else {
            root.join(&artifact.path)
        };
        match sha256_file(&full) {
            Err(e) => missing.push(format!("{} ({e})", artifact.name)),
            Ok(actual) if actual != artifact.sha256 => mismatched.push(format!(
                "{}: expected {}, found {}",
                artifact.name,
                &artifact.sha256[..artifact.sha256.len().min(12)],
                &actual[..actual.len().min(12)]
            )),
            Ok(_) => ok += 1,
        }
    }

    let total = manifest.artifacts.len();
    if missing.is_empty() && mismatched.is_empty() {
        return TriState::Ran {
            passed: true,
            value: None,
            n: Some(ok as u64),
            n_total: Some(total as u64),
            detail: format!("all {total} pinned artifacts match"),
        };
    }

    let mut parts = Vec::new();
    if !missing.is_empty() {
        parts.push(format!("unreadable: {}", missing.join("; ")));
    }
    if !mismatched.is_empty() {
        parts.push(format!("DIGEST MISMATCH: {}", mismatched.join("; ")));
    }
    TriState::Ran {
        passed: false,
        value: None,
        n: Some(ok as u64),
        n_total: Some(total as u64),
        detail: format!("{ok}/{total} artifacts verified. {}", parts.join(" | ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmpdir() -> PathBuf {
        let d = std::env::temp_dir().join(format!("qdpf-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&d);
        d
    }

    fn write(dir: &Path, name: &str, body: &[u8]) -> PathBuf {
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(body).unwrap();
        p
    }

    #[test]
    fn hashing_matches_a_known_vector() {
        let d = tmpdir();
        let p = write(&d, "abc.txt", b"abc");
        // SHA-256("abc"), the standard published vector.
        assert_eq!(
            sha256_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn a_missing_manifest_is_not_run_not_a_pass() {
        let s = check_artifacts(None, Path::new("/"));
        assert!(!s.did_run());
        assert!(!s.is_pass());
    }

    #[test]
    fn an_empty_manifest_verifies_nothing() {
        let d = tmpdir();
        let m = write(&d, "empty.json", br#"{"schema_version":1,"artifacts":[]}"#);
        let s = check_artifacts(Some(&m), &d);
        assert!(!s.did_run(), "an empty manifest must not read as a clean one");
    }

    #[test]
    fn matching_digests_pass() {
        let d = tmpdir();
        write(&d, "w.bin", b"weights");
        let digest = sha256_file(&d.join("w.bin")).unwrap();
        let body = format!(
            r#"{{"schema_version":1,"artifacts":[{{"name":"weights","path":"w.bin","sha256":"{digest}"}}]}}"#
        );
        let m = write(&d, "ok.json", body.as_bytes());
        assert!(check_artifacts(Some(&m), &d).is_pass());
    }

    #[test]
    fn a_changed_file_fails_with_both_digests() {
        let d = tmpdir();
        write(&d, "t.bin", b"tokenizer-v1");
        let digest = sha256_file(&d.join("t.bin")).unwrap();
        let body = format!(
            r#"{{"schema_version":1,"artifacts":[{{"name":"tokenizer","path":"t.bin","sha256":"{digest}"}}]}}"#
        );
        let m = write(&d, "drift.json", body.as_bytes());
        write(&d, "t.bin", b"tokenizer-v2-swapped-underneath");

        let s = check_artifacts(Some(&m), &d);
        assert!(s.did_run() && !s.is_pass());
        match s {
            TriState::Ran { detail, .. } => assert!(detail.contains("DIGEST MISMATCH")),
            _ => panic!("expected Ran"),
        }
    }

    #[test]
    fn an_unreadable_artifact_fails_rather_than_being_skipped() {
        let d = tmpdir();
        let m = write(
            &d,
            "absent.json",
            br#"{"schema_version":1,"artifacts":[{"name":"head","path":"not-here.bin","sha256":"00"}]}"#,
        );
        let s = check_artifacts(Some(&m), &d);
        assert!(s.did_run() && !s.is_pass());
    }

    #[test]
    fn a_malformed_manifest_fails_rather_than_being_ignored() {
        let d = tmpdir();
        let m = write(&d, "bad.json", b"{not json");
        assert!(!check_artifacts(Some(&m), &d).is_pass());
    }
}
