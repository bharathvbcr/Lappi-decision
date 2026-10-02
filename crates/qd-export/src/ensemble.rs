//! The ensemble export: N released towers and one calibration table -> one directory the runtime
//! opens with `qd_runtime::release::Ensemble::open`.
//!
//! ```text
//! <out>/tower-0/  ... <out>/tower-{N-1}/   each a member release, copied and re-verified
//! <out>/calibration.json                   the ensemble's table (fitted on the mean decode)
//! <out>/ensemble_manifest.json             qd-ensemble.v1
//! ```
//!
//! Every member is opened by the runtime's own reader (`Tower::open`: manifest and config
//! binding) and every file its manifest lists is copied with its sha256 checked against that
//! record, so the ensemble directory carries no byte its members' manifests do not vouch for.
//! The members must agree on `config.json`, tokenizer and trained width and be N different
//! towers; the trained width is not recorded anywhere upstream, so the operator states it per
//! member and cites where it came from (`trained_width_source`), and the manifest records both.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use qd_runtime::ensemble::{MAX_MEMBERS, ensemble_weight_hash};
use qd_runtime::hex;
use qd_runtime::release::{
    CALIBRATION_FILE, ENSEMBLE_DECODE, ENSEMBLE_FORMAT, ENSEMBLE_MANIFEST_FILE, MANIFEST_FILE,
    Tower,
};

use crate::export::{
    Staging, is_hex64, len64, read_calibration, read_small, refuse, write_verified,
};
use crate::refusal::{Refusal, RefusalKind, Result};
use crate::safetensors::CHUNK_BYTES;

/// Files one member release may list. A release is ten; a manifest listing far more is not one.
const MAX_MEMBER_FILES: usize = 32;

#[derive(Debug, Clone)]
pub struct EnsembleRequest {
    /// Member release directories (`qd-export --out`), in the order their log-probabilities are
    /// averaged.
    pub members: Vec<PathBuf>,
    /// Each member's trained width in tokens, in member order.
    pub trained_widths: Vec<u64>,
    /// Where the widths come from (ledger row ids, a handoff), recorded verbatim.
    pub trained_width_source: String,
    /// The ensemble's calibration table (`qd_runtime` `CalibrationTable` JSON).
    pub calibration: PathBuf,
    /// The ensemble directory; must not exist.
    pub out: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnsembleSummary {
    pub out: PathBuf,
    /// `qd_runtime::ensemble::ensemble_weight_hash` over the members' weight hashes.
    pub weight_hash: String,
    pub member_weight_hashes: Vec<String>,
    pub calibration_hash: String,
    pub trained_width: u64,
}

fn ensemble_refusal(msg: impl Into<String>) -> Refusal {
    refuse(RefusalKind::Ensemble, msg)
}

/// A member: its opened tower and the files its manifest lists, with their recorded sha256.
struct Member {
    source: PathBuf,
    tower: Tower,
    files: BTreeMap<String, String>,
}

fn plain_file_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', '\0'])
}

fn read_member(i: usize, dir: &Path) -> Result<Member> {
    let tower = Tower::open(dir).map_err(|e| {
        ensemble_refusal(format!(
            "member {i} ({}) does not open as a release: {e}",
            dir.display()
        ))
    })?;
    let manifest_path = dir.join(MANIFEST_FILE);
    let bytes = read_small(&manifest_path, RefusalKind::Ensemble)?;
    if hex(&crate::export::sha256(&bytes)) != tower.manifest_sha256() {
        return Err(ensemble_refusal(format!(
            "{} changed while it was being read",
            manifest_path.display()
        )));
    }
    let doc: Value = serde_json::from_slice(&bytes)
        .map_err(|e| ensemble_refusal(format!("{}: {e}", manifest_path.display())))?;
    let listed = doc
        .get("files")
        .and_then(Value::as_object)
        .ok_or_else(|| ensemble_refusal(format!("{}: no files object", manifest_path.display())))?;
    if listed.len() > MAX_MEMBER_FILES {
        return Err(ensemble_refusal(format!(
            "{} lists {} files, over {MAX_MEMBER_FILES}",
            manifest_path.display(),
            listed.len()
        )));
    }
    let mut files = BTreeMap::new();
    for (name, entry) in listed {
        if !plain_file_name(name) || name == MANIFEST_FILE {
            return Err(ensemble_refusal(format!(
                "{} lists {name:?}, which is not a plain file name of the release",
                manifest_path.display()
            )));
        }
        let sha = entry
            .get("sha256")
            .and_then(Value::as_str)
            .filter(|s| is_hex64(s))
            .ok_or_else(|| {
                ensemble_refusal(format!(
                    "{}: files.{name}.sha256 is not 64 hex digits",
                    manifest_path.display()
                ))
            })?;
        files.insert(name.clone(), sha.to_ascii_lowercase());
    }
    Ok(Member {
        source: dir.to_path_buf(),
        tower,
        files,
    })
}

/// Copy `src` to a new `dst`, hashing as it goes; refuse unless the copy's sha256 is `want`.
fn copy_verified(src: &Path, dst: &Path, want: &str) -> Result<()> {
    let mut input = File::open(src).map_err(|e| Refusal::io(src, e))?;
    let mut output = File::options()
        .write(true)
        .create_new(true)
        .open(dst)
        .map_err(|e| Refusal::io(dst, e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK_BYTES];
    loop {
        let n = input.read(&mut buf).map_err(|e| Refusal::io(src, e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        output
            .write_all(&buf[..n])
            .map_err(|e| Refusal::io(dst, e))?;
    }
    output.sync_all().map_err(|e| Refusal::io(dst, e))?;
    let got = hex(&hasher.finalize());
    if got != want {
        return Err(ensemble_refusal(format!(
            "{} has sha256 {got}; its release manifest records {want}",
            src.display()
        )));
    }
    Ok(())
}

/// Refuse members that could not be averaged honestly, before anything is written.
fn check_members(members: &[Member], widths: &[u64]) -> Result<()> {
    let first = &members[0].tower;
    for (i, member) in members.iter().enumerate().skip(1) {
        let tower = &member.tower;
        for (what, mine, theirs) in [
            (
                "config_sha256",
                tower.config_sha256(),
                first.config_sha256(),
            ),
            (
                "tokenizer_hash",
                tower.tokenizer_hash(),
                first.tokenizer_hash(),
            ),
        ] {
            if mine != theirs {
                return Err(ensemble_refusal(format!(
                    "member {i} has {what} {mine}, member 0 has {theirs}"
                )));
            }
        }
        if widths[i] != widths[0] {
            return Err(ensemble_refusal(format!(
                "member {i} was trained at width {}, member 0 at {}",
                widths[i], widths[0]
            )));
        }
        if let Some(j) = members[..i]
            .iter()
            .position(|other| other.tower.weight_hash() == tower.weight_hash())
        {
            return Err(ensemble_refusal(format!(
                "members {j} and {i} are one tower (weight_hash {})",
                tower.weight_hash()
            )));
        }
    }
    Ok(())
}

fn write_ensemble(
    req: &EnsembleRequest,
    members: &[Member],
    calibration: &(Vec<u8>, qd_runtime::calibration::CalibrationTable),
    dir: &Path,
) -> Result<EnsembleSummary> {
    let mut entries = Vec::with_capacity(members.len());
    for (i, member) in members.iter().enumerate() {
        let name = format!("tower-{i}");
        let target = dir.join(&name);
        std::fs::create_dir(&target).map_err(|e| Refusal::io(&target, e))?;
        for (file, sha) in &member.files {
            copy_verified(&member.source.join(file), &target.join(file), sha)?;
        }
        copy_verified(
            &member.source.join(MANIFEST_FILE),
            &target.join(MANIFEST_FILE),
            member.tower.manifest_sha256(),
        )?;
        let tower = &member.tower;
        entries.push(json!({
            "dir": name,
            "source": member.source.display().to_string(),
            "release_manifest_sha256": tower.manifest_sha256(),
            "weights_sha256": tower.weights_sha256(),
            "weight_hash": tower.weight_hash(),
            "config_sha256": tower.config_sha256(),
            "tokenizer_hash": tower.tokenizer_hash(),
            "trained_width": req.trained_widths[i],
        }));
    }

    let (table_bytes, table) = calibration;
    let table_sha = write_verified(
        &dir.join(CALIBRATION_FILE),
        table_bytes,
        RefusalKind::Calibration,
    )?;
    let table_hash = table.hash();
    let member_hashes: Vec<&str> = members.iter().map(|m| m.tower.weight_hash()).collect();
    let weight_hash = ensemble_weight_hash(&member_hashes);
    let first = &members[0].tower;
    let manifest = json!({
        "format": ENSEMBLE_FORMAT,
        "tool": "qd-export-ensemble",
        "tool_version": env!("CARGO_PKG_VERSION"),
        "decode": ENSEMBLE_DECODE,
        "members": entries,
        "trained_width_source": req.trained_width_source,
        "calibration": {
            "file": CALIBRATION_FILE,
            "source": req.calibration.display().to_string(),
            "file_sha256": table_sha,
            "name": table.name,
            "table_hash": table_hash,
        },
        "expected_identity": {
            "weight_hash": weight_hash,
            "member_weight_hashes": member_hashes,
            "config_sha256": first.config_sha256(),
            "tokenizer_hash": first.tokenizer_hash(),
            "trained_width": req.trained_widths[0],
            "calibration_hash": table_hash,
            // The members' own, as `Tower::open` read and checked it from each member's
            // manifest; never this binary's constant.
            "prompt_format": first.prompt_format(),
        },
        "files": {
            CALIBRATION_FILE: {"sha256": table_sha, "bytes": len64(table_bytes)?},
        },
    });
    let text = serde_json::to_string_pretty(&manifest)
        .map_err(|e| refuse(RefusalKind::Io, e.to_string()))?
        + "\n";
    write_verified(
        &dir.join(ENSEMBLE_MANIFEST_FILE),
        text.as_bytes(),
        RefusalKind::Io,
    )?;

    Ok(EnsembleSummary {
        out: req.out.clone(),
        weight_hash,
        member_weight_hashes: member_hashes.iter().map(|h| h.to_string()).collect(),
        calibration_hash: table_hash,
        trained_width: req.trained_widths[0],
    })
}

/// Write the ensemble. Nothing is written unless every member opens, agrees and is distinct and
/// the table is one the runtime reads back unchanged; a failure after staging began removes the
/// staging directory.
pub fn export_ensemble(req: &EnsembleRequest) -> Result<EnsembleSummary> {
    if !(2..=MAX_MEMBERS).contains(&req.members.len()) {
        return Err(ensemble_refusal(format!(
            "{} members; an ensemble has 2 to {MAX_MEMBERS}",
            req.members.len()
        )));
    }
    if req.trained_widths.len() != req.members.len() {
        return Err(ensemble_refusal(format!(
            "{} members and {} trained widths; each member states its own",
            req.members.len(),
            req.trained_widths.len()
        )));
    }
    if req.trained_widths.contains(&0) {
        return Err(ensemble_refusal(
            "a trained width of 0 tokens is not a width",
        ));
    }
    if qd_runtime::is_blank(&req.trained_width_source) {
        return Err(ensemble_refusal(
            "the trained widths need a source (ledger row ids, a handoff): nothing upstream \
             records them, so the manifest records who said so",
        ));
    }
    let staging = Staging::plan(&req.out)?;
    let members = req
        .members
        .iter()
        .enumerate()
        .map(|(i, dir)| read_member(i, dir))
        .collect::<Result<Vec<_>>>()?;
    check_members(&members, &req.trained_widths)?;
    let calibration = read_calibration(&req.calibration)?;
    staging.publish(|dir| write_ensemble(req, &members, &calibration, dir))
}
