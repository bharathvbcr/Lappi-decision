//! The vocabulary remap, which this trainer can only take as the identity.
//!
//! Lappi's PyTorch trainer slices the tower's tied embedding to the shard set's remap before it
//! trains: `python/qd_train/backbone.py` `remap_text_tower` (through
//! `qd_train.remap.apply_remap_to_model`) keeps row `new_to_old[j]` of the original as row `j`.
//! ojas-qwen35 loads a snapshot as it is and has no remap. So this trainer runs only on a shard
//! set whose remap is the identity over the snapshot's vocabulary. That is the v4 set's: its
//! `remap.npz` maps each of the 248,320 ids to itself. Any other remap, and a set with no remap
//! artifact, is refused by name. Training on renumbered ids against an un-renumbered embedding
//! would train every token as a different token.
//!
//! The reader has already re-read `remap.npz`, re-hashed it and matched it to the header's
//! `remap_hash` (`shard_remap_matches_header`). This module requires that check to have run and
//! passed, then reads the same file's tables and checks that they are the identity.

use std::path::Path;

use qd_train::files::ShardFiles;
use qd_train::npz::StoredZip;
use qd_train::shards::{ShardReader, REMAP_NAME};

/// The reader's check that the remap beside the shards is the one the header pins.
pub const REMAP_CHECK: &str = "shard_remap_matches_header";

/// The reader's own bound on `remap.npz`.
const MAX_REMAP_BYTES: u64 = 256 << 20;

/// Whether `old_to_new` and `new_to_old` are the identity: the same length, and every id maps
/// to itself in both. Returns the vocabulary size.
pub fn check_identity(old_to_new: &[i32], new_to_old: &[i32]) -> Result<u64, String> {
    if new_to_old.is_empty() {
        return Err("the remap keeps no tokens".into());
    }
    if old_to_new.len() != new_to_old.len() {
        return Err(format!(
            "the remap keeps {} of {} tokens: the tower's embedding would have to be sliced \
             (backbone.py remap_text_tower), and ojas-qwen35 has no remap",
            new_to_old.len(),
            old_to_new.len()
        ));
    }
    for (table, values) in [("new_to_old", new_to_old), ("old_to_new", old_to_new)] {
        if let Some((i, v)) = values.iter().enumerate().find(|(i, v)| i32::try_from(*i).ok() != Some(**v)) {
            return Err(format!(
                "{table}[{i}] = {v}: the remap renumbers ids, so the shards' ids are not the \
                 snapshot's and its embedding rows would be trained as the wrong tokens; \
                 ojas-qwen35 has no remap"
            ));
        }
    }
    Ok(new_to_old.len() as u64)
}

fn table(zip: &StoredZip, npz: &Path, key: &str) -> Result<Vec<i32>, String> {
    let a = zip.array(key).map_err(|e| format!("{}: {e}", npz.display()))?;
    if a.shape.len() != 1 {
        return Err(format!("{}: {key} is not 1-D", npz.display()));
    }
    a.cast::<i32>(key).map_err(|e| format!("{}: {e}", npz.display()))
}

/// Refuse unless the reader's remap check passed, the remap beside the shards is the identity,
/// and its vocabulary is both the header's and the snapshot's. Returns that vocabulary.
pub fn require_identity_remap(reader: &ShardReader, files: &dyn ShardFiles, snapshot_vocab: u64) -> Result<u64, String> {
    match reader.check(REMAP_CHECK) {
        Some(t) if t.is_pass() => {}
        Some(t) => {
            return Err(format!(
                "{REMAP_CHECK} is {t:?}: without a remap that was re-read and matched to the \
                 header, the ids' vocabulary is unknown"
            ));
        }
        None => return Err(format!("the reader recorded no {REMAP_CHECK} check")),
    }
    let npz = reader.root().join(format!("{REMAP_NAME}.npz"));
    let bytes = files.read(&npz, MAX_REMAP_BYTES).map_err(|e| format!("{}: {e}", npz.display()))?;
    let zip = StoredZip::parse(&bytes).map_err(|e| format!("{}: {e}", npz.display()))?;
    let vocab = check_identity(&table(&zip, &npz, "old_to_new")?, &table(&zip, &npz, "new_to_old")?)
        .map_err(|e| format!("{}: {e}", npz.display()))?;
    let header = reader.header().vocab_size;
    if vocab != header || vocab != snapshot_vocab {
        return Err(format!(
            "the identity remap covers {vocab} ids, the shard header declares {header} and the \
             snapshot's vocabulary is {snapshot_vocab}: all three must agree"
        ));
    }
    Ok(vocab)
}
