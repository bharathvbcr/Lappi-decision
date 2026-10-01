//! A hand-rolled reader for `np.savez` archives: a ZIP whose members are **stored**.
//!
//! `np.savez` writes every member with `ZIP_STORED` (`np.savez_compressed` is the deflating
//! one, and nothing in the shard contract calls it), and opens each member with
//! `force_zip64=True`, so local headers carry a ZIP64 extra field while the central directory
//! uses ZIP64 only for a member past 4 GiB. Sizes and offsets are therefore taken from the
//! central directory (resolving ZIP64 extras where a field is `0xFFFFFFFF`), and the local
//! header is read only to find where the data starts and to check it names the same member.
//!
//! Refused by name, never decoded halfway: a compressed member (**loudly**: a deflated
//! `supervision.npz` is a set some other writer produced, and guessing at it would be a
//! dependency this crate does not take), an encrypted member, a multi-disk archive, a CRC32
//! mismatch, duplicate member names, and any offset or length that runs outside the file.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::npy::{NpyArray, NpyError};

const EOCD_SIG: u32 = 0x0605_4b50;
const ZIP64_LOCATOR_SIG: u32 = 0x0706_4b50;
const ZIP64_EOCD_SIG: u32 = 0x0606_4b50;
const CENTRAL_SIG: u32 = 0x0201_4b50;
const LOCAL_SIG: u32 = 0x0403_4b50;
const ZIP64_EXTRA_ID: u16 = 0x0001;
/// The EOCD record is 22 bytes plus a comment of at most 65,535.
const EOCD_SEARCH: usize = 22 + 0xffff;
/// Ceiling on members; `supervision.npz` has five and `remap.npz` two.
const MAX_MEMBERS: usize = 256;

/// An archive this reader refuses.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NpzError {
    /// Structurally not a ZIP this reader can trust.
    #[error("malformed .npz: {0}")]
    Malformed(String),
    /// A member is compressed. `np.savez` never writes one.
    #[error(
        "member {name:?} uses compression method {method}; np.savez writes members STORED \
         (method 0). A deflated .npz was written by something other than this shard \
         contract's writer, and this reader refuses it rather than guessing"
    )]
    Compressed {
        /// The member.
        name: String,
        /// The ZIP compression method.
        method: u16,
    },
    /// A member's bytes do not hash to the CRC32 the archive records.
    #[error("member {name:?}: CRC32 {found:08x} but the archive records {recorded:08x}")]
    Crc {
        /// The member.
        name: String,
        /// Computed.
        found: u32,
        /// Recorded.
        recorded: u32,
    },
    /// A requested member is absent.
    #[error("no member {0:?}")]
    Missing(String),
    /// A member is not a readable `.npy`.
    #[error("member {name:?}: {source}")]
    Npy {
        /// The member.
        name: String,
        /// Why.
        source: NpyError,
    },
}

/// The members of one stored archive, by name, with their bytes.
#[derive(Debug)]
pub struct StoredZip {
    members: BTreeMap<String, Vec<u8>>,
}

impl StoredZip {
    /// Parse a whole archive.
    pub fn parse(bytes: &[u8]) -> Result<Self, NpzError> {
        let eocd = find_eocd(bytes)?;
        let disk = u16_at(bytes, eocd + 4)?;
        let cd_disk = u16_at(bytes, eocd + 6)?;
        let mut entries = u64::from(u16_at(bytes, eocd + 10)?);
        let mut cd_size = u64::from(u32_at(bytes, eocd + 12)?);
        let mut cd_offset = u64::from(u32_at(bytes, eocd + 16)?);
        if (disk != 0 && disk != 0xffff) || (cd_disk != 0 && cd_disk != 0xffff) {
            return Err(NpzError::Malformed("multi-disk archive".to_owned()));
        }
        if entries == 0xffff || cd_size == 0xffff_ffff || cd_offset == 0xffff_ffff {
            let locator = eocd
                .checked_sub(20)
                .ok_or_else(|| NpzError::Malformed("ZIP64 locator before byte 0".to_owned()))?;
            if u32_at(bytes, locator)? != ZIP64_LOCATOR_SIG {
                return Err(NpzError::Malformed(
                    "ZIP64 fields without a ZIP64 locator".to_owned(),
                ));
            }
            let z64 = to_usize(u64_at(bytes, locator + 8)?)?;
            if u32_at(bytes, z64)? != ZIP64_EOCD_SIG {
                return Err(NpzError::Malformed(
                    "ZIP64 locator points at no ZIP64 EOCD".to_owned(),
                ));
            }
            entries = u64_at(bytes, z64 + 32)?;
            cd_size = u64_at(bytes, z64 + 40)?;
            cd_offset = u64_at(bytes, z64 + 48)?;
        }
        if entries > MAX_MEMBERS as u64 {
            return Err(NpzError::Malformed(format!(
                "{entries} members, over {MAX_MEMBERS}"
            )));
        }
        let cd_start = to_usize(cd_offset)?;
        let cd_end = cd_start
            .checked_add(to_usize(cd_size)?)
            .filter(|e| *e <= bytes.len())
            .ok_or_else(|| NpzError::Malformed("central directory runs past the end".to_owned()))?;

        let mut members = BTreeMap::new();
        let mut at = cd_start;
        for _ in 0..entries {
            if u32_at(bytes, at)? != CENTRAL_SIG {
                return Err(NpzError::Malformed(format!(
                    "no central header at byte {at}"
                )));
            }
            let flags = u16_at(bytes, at + 8)?;
            let method = u16_at(bytes, at + 10)?;
            let crc = u32_at(bytes, at + 16)?;
            let mut comp = u64::from(u32_at(bytes, at + 20)?);
            let mut uncomp = u64::from(u32_at(bytes, at + 24)?);
            let name_len = usize::from(u16_at(bytes, at + 28)?);
            let extra_len = usize::from(u16_at(bytes, at + 30)?);
            let comment_len = usize::from(u16_at(bytes, at + 32)?);
            let start_disk = u16_at(bytes, at + 34)?;
            let mut local = u64::from(u32_at(bytes, at + 42)?);
            let name_bytes = slice(bytes, at + 46, name_len)?;
            let name = std::str::from_utf8(name_bytes)
                .map_err(|_| NpzError::Malformed("a member name is not UTF-8".to_owned()))?
                .to_owned();
            let extra = slice(bytes, at + 46 + name_len, extra_len)?;
            if uncomp == 0xffff_ffff || comp == 0xffff_ffff || local == 0xffff_ffff {
                let mut z = zip64_extra(extra)
                    .ok_or_else(|| NpzError::Malformed(format!("{name}: ZIP64 sizes, no extra")))?;
                if uncomp == 0xffff_ffff {
                    uncomp = take_u64(&mut z, &name)?;
                }
                if comp == 0xffff_ffff {
                    comp = take_u64(&mut z, &name)?;
                }
                if local == 0xffff_ffff {
                    local = take_u64(&mut z, &name)?;
                }
            }
            if start_disk != 0 && start_disk != 0xffff {
                return Err(NpzError::Malformed(format!("{name}: on disk {start_disk}")));
            }
            if flags & 0x0001 != 0 {
                return Err(NpzError::Malformed(format!("{name}: encrypted member")));
            }
            if method != 0 {
                return Err(NpzError::Compressed { name, method });
            }
            if comp != uncomp {
                return Err(NpzError::Malformed(format!(
                    "{name}: stored member with {comp} compressed and {uncomp} uncompressed bytes"
                )));
            }
            let data = member_data(bytes, to_usize(local)?, &name, to_usize(comp)?)?;
            let found = crc32(data);
            if found != crc {
                return Err(NpzError::Crc {
                    name,
                    found,
                    recorded: crc,
                });
            }
            if members.insert(name.clone(), data.to_vec()).is_some() {
                return Err(NpzError::Malformed(format!("duplicate member {name:?}")));
            }
            at = at + 46 + name_len + extra_len + comment_len;
            if at > cd_end {
                return Err(NpzError::Malformed(
                    "central directory overruns its size".to_owned(),
                ));
            }
        }
        Ok(Self { members })
    }

    /// Member names, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.members.keys().map(String::as_str)
    }

    /// The array stored as `<key>.npy` -- `np.load(path)[key]`.
    pub fn array(&self, key: &str) -> Result<NpyArray, NpzError> {
        let name = format!("{key}.npy");
        let bytes = self
            .members
            .get(&name)
            .ok_or_else(|| NpzError::Missing(name.clone()))?;
        NpyArray::parse(bytes).map_err(|source| NpzError::Npy { name, source })
    }
}

fn member_data<'a>(
    bytes: &'a [u8],
    local: usize,
    name: &str,
    len: usize,
) -> Result<&'a [u8], NpzError> {
    if u32_at(bytes, local)? != LOCAL_SIG {
        return Err(NpzError::Malformed(format!(
            "{name}: no local header at byte {local}"
        )));
    }
    let local_method = u16_at(bytes, local + 8)?;
    if local_method != 0 {
        return Err(NpzError::Compressed {
            name: name.to_owned(),
            method: local_method,
        });
    }
    let name_len = usize::from(u16_at(bytes, local + 26)?);
    let extra_len = usize::from(u16_at(bytes, local + 28)?);
    let local_name = slice(bytes, local + 30, name_len)?;
    if local_name != name.as_bytes() {
        return Err(NpzError::Malformed(format!(
            "{name}: the local header names {:?}",
            String::from_utf8_lossy(local_name)
        )));
    }
    slice(bytes, local + 30 + name_len + extra_len, len)
}

fn find_eocd(bytes: &[u8]) -> Result<usize, NpzError> {
    if bytes.len() < 22 {
        return Err(NpzError::Malformed(
            "shorter than an end-of-central-directory".to_owned(),
        ));
    }
    let lowest = bytes.len().saturating_sub(EOCD_SEARCH);
    let mut at = bytes.len() - 22;
    loop {
        if u32_at(bytes, at)? == EOCD_SIG {
            let comment = usize::from(u16_at(bytes, at + 20)?);
            if at + 22 + comment == bytes.len() {
                return Ok(at);
            }
        }
        if at == lowest {
            return Err(NpzError::Malformed(
                "no end-of-central-directory record".to_owned(),
            ));
        }
        at -= 1;
    }
}

/// The ZIP64 extended-information extra field's payload, if present.
fn zip64_extra(extra: &[u8]) -> Option<&[u8]> {
    let mut i = 0;
    while i + 4 <= extra.len() {
        let id = u16::from_le_bytes([extra[i], extra[i + 1]]);
        let len = usize::from(u16::from_le_bytes([extra[i + 2], extra[i + 3]]));
        let body = extra.get(i + 4..i + 4 + len)?;
        if id == ZIP64_EXTRA_ID {
            return Some(body);
        }
        i += 4 + len;
    }
    None
}

fn take_u64(z: &mut &[u8], name: &str) -> Result<u64, NpzError> {
    let (head, tail) = z
        .split_first_chunk::<8>()
        .ok_or_else(|| NpzError::Malformed(format!("{name}: ZIP64 extra too short")))?;
    *z = tail;
    Ok(u64::from_le_bytes(*head))
}

fn slice(bytes: &[u8], at: usize, len: usize) -> Result<&[u8], NpzError> {
    at.checked_add(len)
        .and_then(|end| bytes.get(at..end))
        .ok_or_else(|| NpzError::Malformed(format!("{len} bytes at {at} run past the end")))
}

fn u16_at(bytes: &[u8], at: usize) -> Result<u16, NpzError> {
    let s = slice(bytes, at, 2)?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn u32_at(bytes: &[u8], at: usize) -> Result<u32, NpzError> {
    let s = slice(bytes, at, 4)?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64_at(bytes: &[u8], at: usize) -> Result<u64, NpzError> {
    let s = slice(bytes, at, 8)?;
    Ok(u64::from_le_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}

fn to_usize(v: u64) -> Result<usize, NpzError> {
    usize::try_from(v).map_err(|_| NpzError::Malformed(format!("offset {v} exceeds usize")))
}

const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xedb8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

/// The ZIP CRC-32 (reflected, polynomial 0xEDB88320).
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xffff_ffffu32;
    for &b in data {
        c = CRC_TABLE[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8);
    }
    c ^ 0xffff_ffff
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(crc32(b""), 0);
    }

    /// A one-member stored archive, built the way `zipfile` lays one out.
    fn archive(name: &str, data: &[u8], method: u16) -> Vec<u8> {
        let mut out = Vec::new();
        let crc = crc32(data);
        let n = u16::try_from(name.len()).unwrap();
        let size = u32::try_from(data.len()).unwrap();
        out.extend_from_slice(&LOCAL_SIG.to_le_bytes());
        out.extend_from_slice(&[20, 0, 0, 0]);
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);
        let cd = out.len();
        out.extend_from_slice(&CENTRAL_SIG.to_le_bytes());
        out.extend_from_slice(&[20, 0, 20, 0, 0, 0]);
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&[0; 12]);
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        let cd_size = u32::try_from(out.len() - cd).unwrap();
        out.extend_from_slice(&EOCD_SIG.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0, 1, 0, 1, 0]);
        out.extend_from_slice(&cd_size.to_le_bytes());
        out.extend_from_slice(&u32::try_from(cd).unwrap().to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    #[test]
    fn a_stored_member_reads_back_and_a_deflated_one_is_refused_by_name() {
        let ok = StoredZip::parse(&archive("x.npy", b"hello", 0)).unwrap();
        assert_eq!(ok.names().collect::<Vec<_>>(), vec!["x.npy"]);
        match StoredZip::parse(&archive("x.npy", b"hello", 8)) {
            Err(NpzError::Compressed { name, method: 8 }) => assert_eq!(name, "x.npy"),
            other => panic!("a deflated member must be refused loudly, got {other:?}"),
        }
    }

    #[test]
    fn a_flipped_byte_fails_the_crc() {
        let mut bytes = archive("x.npy", b"hello", 0);
        let at = bytes.windows(5).position(|w| w == b"hello").unwrap();
        bytes[at] = b'j';
        assert!(matches!(
            StoredZip::parse(&bytes),
            Err(NpzError::Crc { .. })
        ));
    }

    #[test]
    fn truncation_is_refused_not_panicked_on() {
        let bytes = archive("x.npy", b"hello", 0);
        for cut in [0, 10, 30, bytes.len() - 1] {
            assert!(StoredZip::parse(&bytes[..cut]).is_err());
        }
    }
}
