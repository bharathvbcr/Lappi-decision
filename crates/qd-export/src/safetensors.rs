//! Safetensors, read and written by hand: an 8-byte little-endian header length, a JSON header
//! of `{name: {dtype, shape, data_offsets}}`, then the raw little-endian data.
//!
//! **Why a reader here as well as in qd-metal.** `qd_metal::safetensors` is the serving loader's
//! reader, and this crate cannot depend on it: qd-metal links tessl, and tessl links Metal
//! unconditionally, so the dependency would make this binary unbuildable for the Linux box the
//! averaged checkpoint lives on. The two readers are kept honest against each other rather than
//! trusted: `tests/roundtrip.rs` parses every file this crate writes with qd-metal's reader and
//! requires the same `(name, dtype, shape, offsets)` set. Extracting one tessl-free reader both
//! crates use is recorded as `GAP-J7-EXPORT-SAFETENSORS-READER-DUPLICATED`.
//!
//! This reader knows every dtype the format names, so a tensor of a dtype the export refuses is
//! refused *as a dtype*, by name, rather than failing as an unparseable header.
//!
//! **The writer** lays the tensors out back to back, in name order, with no gap: the Python
//! `safetensors` package refuses a data section with holes or trailing bytes. The only padding is
//! the format's own: the header is padded with spaces so the data section starts on an 8-byte
//! boundary. The header string is built here, in a fixed order, rather than by serialising a
//! JSON map whose key order depends on a feature another workspace crate could turn on: the same
//! inputs give the same bytes, so a release's hashes are reproducible.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::refusal::{Refusal, RefusalKind, Result};

/// The format's own limit on the header (safetensors spec: 100 MB), the same bound qd-metal
/// applies.
pub const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// Bytes read per positioned read when streaming a file or a tensor. A multiple of every
/// element size, so a chunk never splits an element.
pub const CHUNK_BYTES: usize = 16 * 1024 * 1024;

/// Every dtype the safetensors format names with a whole-byte element size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dtype {
    Bool,
    U8,
    I8,
    F8E5M2,
    F8E4M3,
    I16,
    U16,
    F16,
    Bf16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
}

impl Dtype {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "BOOL" => Dtype::Bool,
            "U8" => Dtype::U8,
            "I8" => Dtype::I8,
            "F8_E5M2" => Dtype::F8E5M2,
            "F8_E4M3" => Dtype::F8E4M3,
            "I16" => Dtype::I16,
            "U16" => Dtype::U16,
            "F16" => Dtype::F16,
            "BF16" => Dtype::Bf16,
            "I32" => Dtype::I32,
            "U32" => Dtype::U32,
            "F32" => Dtype::F32,
            "F64" => Dtype::F64,
            "I64" => Dtype::I64,
            "U64" => Dtype::U64,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Dtype::Bool => "BOOL",
            Dtype::U8 => "U8",
            Dtype::I8 => "I8",
            Dtype::F8E5M2 => "F8_E5M2",
            Dtype::F8E4M3 => "F8_E4M3",
            Dtype::I16 => "I16",
            Dtype::U16 => "U16",
            Dtype::F16 => "F16",
            Dtype::Bf16 => "BF16",
            Dtype::I32 => "I32",
            Dtype::U32 => "U32",
            Dtype::F32 => "F32",
            Dtype::F64 => "F64",
            Dtype::I64 => "I64",
            Dtype::U64 => "U64",
        }
    }

    pub fn size(self) -> usize {
        match self {
            Dtype::Bool | Dtype::U8 | Dtype::I8 | Dtype::F8E5M2 | Dtype::F8E4M3 => 1,
            Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::Bf16 => 2,
            Dtype::I32 | Dtype::U32 | Dtype::F32 => 4,
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Absolute byte range in the file.
    pub start: u64,
    pub end: u64,
}

impl TensorInfo {
    pub fn nbytes(&self) -> u64 {
        self.end - self.start
    }
}

/// An open safetensors file whose header has been checked in full.
pub struct SafeTensorsFile {
    path: PathBuf,
    file: File,
    len: u64,
    tensors: BTreeMap<String, TensorInfo>,
}

fn bad(path: &Path, msg: impl std::fmt::Display) -> Refusal {
    Refusal::new(RefusalKind::SourceFormat, format!("{}: {msg}", path.display()))
}

impl SafeTensorsFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Refusal::io(path, e))?;
        let len = file.metadata().map_err(|e| Refusal::io(path, e))?.len();
        if len < 8 {
            return Err(bad(path, format!("{len} bytes is too short for a header length")));
        }
        let mut len_bytes = [0u8; 8];
        file.read_exact_at(&mut len_bytes, 0).map_err(|e| Refusal::io(path, e))?;
        let header_len = u64::from_le_bytes(len_bytes);
        if header_len == 0 || header_len > MAX_HEADER_BYTES {
            return Err(bad(path, format!("header length {header_len} is outside (0, {MAX_HEADER_BYTES}]")));
        }
        let data_start = 8 + header_len;
        if data_start > len {
            return Err(bad(path, format!("header length {header_len} runs past the end of a {len}-byte file")));
        }
        let mut header = vec![0u8; usize::try_from(header_len).map_err(|e| bad(path, e))?];
        file.read_exact_at(&mut header, 8).map_err(|e| Refusal::io(path, e))?;
        let tensors = parse_header(&header, data_start, len).map_err(|m| bad(path, m))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            len,
            tensors,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }

    /// `dst.len()` bytes of `info`'s data, starting `offset` bytes into the tensor.
    pub fn read_tensor_range(&self, info: &TensorInfo, offset: u64, dst: &mut [u8]) -> Result<()> {
        let n = u64::try_from(dst.len()).map_err(|e| bad(&self.path, e))?;
        let end = offset.checked_add(n).ok_or_else(|| bad(&self.path, "read range overflows"))?;
        if end > info.nbytes() {
            return Err(bad(&self.path, format!("read of [{offset}, {end}) runs past a {}-byte tensor", info.nbytes())));
        }
        self.file
            .read_exact_at(dst, info.start + offset)
            .map_err(|e| Refusal::io(&self.path, e))
    }

    /// SHA-256 of the whole file, streamed.
    pub fn sha256(&self) -> Result<[u8; 32]> {
        sha256_file(&self.path)
    }
}

/// SHA-256 of a file, streamed in [`CHUNK_BYTES`] reads.
pub fn sha256_file(path: &Path) -> Result<[u8; 32]> {
    let mut f = File::open(path).map_err(|e| Refusal::io(path, e))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; CHUNK_BYTES];
    loop {
        let n = f.read(&mut buf).map_err(|e| Refusal::io(path, e))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().into())
}

fn parse_header(header: &[u8], data_start: u64, file_len: u64) -> std::result::Result<BTreeMap<String, TensorInfo>, String> {
    let root: Value = serde_json::from_slice(header).map_err(|e| format!("header is not JSON: {e}"))?;
    let obj = root.as_object().ok_or("header is not a JSON object")?;
    let data_len = file_len - data_start;
    let mut tensors = BTreeMap::new();
    for (name, entry) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype_s = entry
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{name}: no dtype"))?;
        let dtype = Dtype::parse(dtype_s).ok_or_else(|| format!("{name}: dtype {dtype_s:?} is not a safetensors dtype this reader knows"))?;
        let shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{name}: no shape"))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .and_then(|d| usize::try_from(d).ok())
                    .ok_or_else(|| format!("{name}: shape entry {d} is not a usize"))
            })
            .collect::<std::result::Result<Vec<usize>, String>>()?;
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{name}: no data_offsets"))?;
        let [b, e] = offsets.as_slice() else {
            return Err(format!("{name}: data_offsets has {} entries, not 2", offsets.len()));
        };
        let (b, e) = (
            b.as_u64().ok_or_else(|| format!("{name}: data_offsets[0] is not a u64"))?,
            e.as_u64().ok_or_else(|| format!("{name}: data_offsets[1] is not a u64"))?,
        );
        if e < b || e > data_len {
            return Err(format!("{name}: data_offsets [{b}, {e}) do not fit the {data_len}-byte data section"));
        }
        let numel = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| format!("{name}: shape {shape:?} overflows"))?;
        let want = numel
            .checked_mul(dtype.size())
            .ok_or_else(|| format!("{name}: byte size overflows"))?;
        if u64::try_from(want).map_err(|e| e.to_string())? != e - b {
            return Err(format!(
                "{name}: {shape:?} x {} bytes = {want}, but data_offsets span {}",
                dtype.size(),
                e - b
            ));
        }
        tensors.insert(
            name.clone(),
            TensorInfo {
                dtype,
                shape,
                start: data_start + b,
                end: data_start + e,
            },
        );
    }
    let mut ranges: Vec<(&str, u64, u64)> = tensors.iter().map(|(n, t)| (n.as_str(), t.start, t.end)).collect();
    ranges.sort_by_key(|r| (r.1, r.2));
    for w in ranges.windows(2) {
        if w[1].1 < w[0].2 {
            return Err(format!("tensors {} and {} overlap", w[0].0, w[1].0));
        }
    }
    Ok(tensors)
}

// ------------------------------------------------------------------ writer ---

/// One tensor the writer will hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl PlannedTensor {
    fn nbytes(&self) -> Option<u64> {
        let numel = self.shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))?;
        u64::try_from(numel.checked_mul(self.dtype.size())?).ok()
    }
}

/// What a finished writer produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenFile {
    pub sha256: [u8; 32],
    pub bytes: u64,
    /// `(name, sha256 of the tensor's bytes)`, in name order.
    pub tensor_digests: Vec<(String, [u8; 32])>,
}

/// `(start, end)` of each tensor within the data section.
pub type DataOffsets = Vec<(u64, u64)>;

/// The header bytes, padded, for tensors laid out back to back in the given order.
///
/// Returned with the [`DataOffsets`] of each tensor, in the same order.
pub fn header_bytes(plan: &[PlannedTensor]) -> std::result::Result<(Vec<u8>, DataOffsets), String> {
    let mut s = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
    let mut offsets = Vec::with_capacity(plan.len());
    let mut at = 0u64;
    for t in plan {
        let n = t.nbytes().ok_or_else(|| format!("{}: byte size overflows", t.name))?;
        let end = at.checked_add(n).ok_or("data section overflows u64")?;
        let name = serde_json::to_string(&t.name).map_err(|e| format!("{}: {e}", t.name))?;
        let shape: Vec<String> = t.shape.iter().map(usize::to_string).collect();
        s.push(',');
        s.push_str(&name);
        s.push_str(&format!(
            ":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{at},{end}]}}",
            t.dtype.as_str(),
            shape.join(",")
        ));
        offsets.push((at, end));
        at = end;
    }
    s.push('}');
    // The data section starts at 8 + header length; pad with spaces to an 8-byte boundary, which
    // is what the reference implementation does and what aligned readers rely on.
    while (8 + s.len()) % 8 != 0 {
        s.push(' ');
    }
    let len = u64::try_from(s.len()).map_err(|e| e.to_string())?;
    if len > MAX_HEADER_BYTES {
        return Err(format!("header of {len} bytes exceeds {MAX_HEADER_BYTES}"));
    }
    let mut out = Vec::with_capacity(8 + s.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(s.as_bytes());
    Ok((out, offsets))
}

/// Streams tensors into a new safetensors file, in name order, each exactly its planned size.
pub struct Writer {
    path: PathBuf,
    out: BufWriter<File>,
    file_hash: Sha256,
    bytes: u64,
    plan: Vec<(PlannedTensor, u64)>,
    /// Index of the tensor being written, or of the next one when `open` is false.
    cur: usize,
    open: bool,
    cur_written: u64,
    cur_hash: Sha256,
    digests: Vec<(String, [u8; 32])>,
}

fn werr(path: &Path, msg: impl std::fmt::Display) -> Refusal {
    Refusal::new(RefusalKind::Io, format!("{}: {msg}", path.display()))
}

impl Writer {
    /// Create `path` (which must not exist) and write the header for `tensors`.
    pub fn create(path: &Path, mut tensors: Vec<PlannedTensor>) -> Result<Self> {
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        for w in tensors.windows(2) {
            if w[0].name == w[1].name {
                return Err(werr(path, format!("tensor {} planned twice", w[0].name)));
            }
        }
        let (header, offsets) = header_bytes(&tensors).map_err(|m| werr(path, m))?;
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| Refusal::io(path, e))?;
        let mut w = Self {
            path: path.to_path_buf(),
            out: BufWriter::with_capacity(CHUNK_BYTES, file),
            file_hash: Sha256::new(),
            bytes: 0,
            plan: tensors
                .into_iter()
                .zip(offsets)
                .map(|(t, (s, e))| (t, e - s))
                .collect(),
            cur: 0,
            open: false,
            cur_written: 0,
            cur_hash: Sha256::new(),
            digests: Vec::new(),
        };
        w.emit(&header)?;
        Ok(w)
    }

    /// The tensors in the order they must be written.
    pub fn order(&self) -> Vec<PlannedTensor> {
        self.plan.iter().map(|(t, _)| t.clone()).collect()
    }

    fn emit(&mut self, bytes: &[u8]) -> Result<()> {
        self.out.write_all(bytes).map_err(|e| Refusal::io(&self.path, e))?;
        self.file_hash.update(bytes);
        self.bytes += u64::try_from(bytes.len()).map_err(|e| werr(&self.path, e))?;
        Ok(())
    }

    pub fn begin(&mut self, name: &str) -> Result<()> {
        if self.open {
            return Err(werr(&self.path, format!("begin({name}) while {} is still open", self.plan[self.cur].0.name)));
        }
        let Some((next, _)) = self.plan.get(self.cur) else {
            return Err(werr(&self.path, format!("begin({name}) after every planned tensor was written")));
        };
        if next.name != name {
            return Err(werr(&self.path, format!("begin({name}) but the next planned tensor is {}", next.name)));
        }
        self.open = true;
        self.cur_written = 0;
        self.cur_hash = Sha256::new();
        Ok(())
    }

    pub fn write(&mut self, bytes: &[u8]) -> Result<()> {
        if !self.open {
            return Err(werr(&self.path, "write with no tensor open"));
        }
        let n = u64::try_from(bytes.len()).map_err(|e| werr(&self.path, e))?;
        let (t, want) = &self.plan[self.cur];
        if self.cur_written + n > *want {
            return Err(werr(&self.path, format!("{}: {} bytes written past its {want}", t.name, self.cur_written + n)));
        }
        self.cur_hash.update(bytes);
        self.cur_written += n;
        self.emit(bytes)
    }

    /// Close the open tensor, which must be complete. Returns the SHA-256 of its bytes.
    pub fn end(&mut self) -> Result<[u8; 32]> {
        if !self.open {
            return Err(werr(&self.path, "end with no tensor open"));
        }
        let (t, want) = &self.plan[self.cur];
        if self.cur_written != *want {
            return Err(werr(&self.path, format!("{}: {} of {want} bytes written", t.name, self.cur_written)));
        }
        let digest: [u8; 32] = std::mem::take(&mut self.cur_hash).finalize().into();
        self.digests.push((t.name.clone(), digest));
        self.open = false;
        self.cur += 1;
        Ok(digest)
    }

    /// Flush and fsync. Every planned tensor must have been written.
    pub fn finish(self) -> Result<WrittenFile> {
        if self.open || self.cur != self.plan.len() {
            return Err(werr(
                &self.path,
                format!("finished with {} of {} tensors written", self.digests.len(), self.plan.len()),
            ));
        }
        let path = self.path;
        let file = self.out.into_inner().map_err(|e| Refusal::io(&path, e.error()))?;
        file.sync_all().map_err(|e| Refusal::io(&path, e))?;
        Ok(WrittenFile {
            sha256: self.file_hash.finalize().into(),
            bytes: self.bytes,
            tensor_digests: self.digests,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A path in the temp dir, removed on drop.
    pub(crate) struct TempPath(pub PathBuf);
    impl TempPath {
        pub(crate) fn new(tag: &str) -> Self {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            Self(std::env::temp_dir().join(format!("qd-export-{tag}-{}-{n}", std::process::id())))
        }
    }
    impl Drop for TempPath {
        fn drop(&mut self) {
            // A leftover temp file only costs disk; the test already has its answer.
            let _ignored = std::fs::remove_file(&self.0).or_else(|_| std::fs::remove_dir_all(&self.0));
        }
    }

    pub(crate) fn write_raw(header: &str, data: &[u8]) -> TempPath {
        let p = TempPath::new("raw");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        std::fs::write(&p.0, bytes).unwrap();
        p
    }

    #[test]
    fn header_is_padded_to_eight_and_tensors_are_contiguous_in_name_order() {
        let plan = vec![
            PlannedTensor { name: "a".into(), dtype: Dtype::Bf16, shape: vec![3] },
            PlannedTensor { name: "b".into(), dtype: Dtype::F32, shape: vec![2, 1] },
        ];
        let (h, offs) = header_bytes(&plan).unwrap();
        assert_eq!(h.len() % 8, 0, "8 + header length must be a multiple of 8");
        let n = u64::from_le_bytes(h[..8].try_into().unwrap());
        assert_eq!(n as usize, h.len() - 8);
        assert_eq!(offs, [(0, 6), (6, 14)]);
        let json: Value = serde_json::from_slice(&h[8..]).unwrap();
        assert_eq!(json["__metadata__"]["format"], "pt");
        assert_eq!(json["b"]["data_offsets"], serde_json::json!([6, 14]));
        assert_eq!(json["a"]["dtype"], "BF16");
    }

    #[test]
    fn writer_round_trips_through_the_reader_and_enforces_the_plan() {
        let p = TempPath::new("w");
        let plan = vec![
            PlannedTensor { name: "z".into(), dtype: Dtype::F32, shape: vec![2] },
            PlannedTensor { name: "a".into(), dtype: Dtype::Bf16, shape: vec![1, 3] },
        ];
        let mut w = Writer::create(&p.0, plan).unwrap();
        assert_eq!(w.order()[0].name, "a", "written in name order");
        assert!(w.begin("z").is_err(), "out of order");
        w.begin("a").unwrap();
        w.write(&[1, 2, 3, 4]).unwrap();
        assert!(w.end().is_err(), "incomplete");
        w.write(&[5, 6]).unwrap();
        assert!(w.write(&[7]).is_err(), "past the planned size");
        w.end().unwrap();
        w.begin("z").unwrap();
        w.write(&1.5f32.to_le_bytes()).unwrap();
        w.write(&(-2.0f32).to_le_bytes()).unwrap();
        w.end().unwrap();
        let out = w.finish().unwrap();
        assert_eq!(out.sha256, sha256_file(&p.0).unwrap());
        assert_eq!(out.bytes, std::fs::metadata(&p.0).unwrap().len());
        assert_eq!(out.tensor_digests.len(), 2);

        let st = SafeTensorsFile::open(&p.0).unwrap();
        let a = &st.tensors()["a"];
        assert_eq!((a.dtype, a.shape.clone(), a.nbytes()), (Dtype::Bf16, vec![1, 3], 6));
        let mut buf = [0u8; 6];
        st.read_tensor_range(a, 0, &mut buf).unwrap();
        assert_eq!(buf, [1, 2, 3, 4, 5, 6]);
        let z = &st.tensors()["z"];
        let mut zb = [0u8; 4];
        st.read_tensor_range(z, 4, &mut zb).unwrap();
        assert_eq!(f32::from_le_bytes(zb), -2.0);
        assert!(st.read_tensor_range(z, 6, &mut zb).is_err());

        // An existing file is never overwritten.
        assert!(Writer::create(&p.0, vec![]).is_err());
    }

    #[test]
    fn a_writer_that_is_not_finished_says_so() {
        let p = TempPath::new("unfinished");
        let plan = vec![PlannedTensor { name: "a".into(), dtype: Dtype::U8, shape: vec![1] }];
        let w = Writer::create(&p.0, plan).unwrap();
        assert!(w.finish().unwrap_err().detail.contains("0 of 1"));
    }

    #[test]
    fn reader_refuses_malformed_headers() {
        let cases: [(&str, usize, &str); 7] = [
            ("not json", 0, "not JSON"),
            ("[1]", 0, "not a JSON object"),
            (r#"{"a":{"dtype":"Q7","shape":[1],"data_offsets":[0,1]}}"#, 1, "dtype \"Q7\""),
            (r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#, 8, "span 4"),
            (r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#, 4, "do not fit"),
            (
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}"#,
                8,
                "overlap",
            ),
            (r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#, 4, "not 2"),
        ];
        for (header, data_len, needle) in cases {
            let f = write_raw(header, &vec![0u8; data_len]);
            let err = SafeTensorsFile::open(&f.0).err().expect(needle);
            assert_eq!(err.kind, RefusalKind::SourceFormat);
            assert!(err.detail.contains(needle), "{needle:?} not in {err:?}");
        }
        // Every dtype the format names parses, so an unwanted one is refused as a dtype later.
        let f = write_raw(r#"{"a":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}"#, &[0u8; 8]);
        assert_eq!(SafeTensorsFile::open(&f.0).unwrap().tensors()["a"].dtype, Dtype::I64);
    }
}
